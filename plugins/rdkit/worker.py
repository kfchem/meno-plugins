"""The RDKit plugin's worker: RDKit, answering a fixed set of requests - the
roles it fills for Meno (its manifest; docs/PLUGINS.md).

One JSON object per line on stdin, one per line back on stdout:

    {"id": 7, "op": "to_smiles", "molblock": "..."}
    {"id": 7, "ok": true, "result": {"smiles": "CCO"}}
    {"id": 7, "ok": false, "error": "..."}

It runs no code it is sent - only the operations below - and needs no
network. When RDKit has been imported it says so:

    {"event": "ready", "version": "2026.03.6"}

Structures come in as MOL blocks, V2000 or V3000, and go back as V3000 ones
with their atoms in the order they came in. (Clean-up is Meno's own layout
engine's, in the app: src/lib/layout.) Their 3D structures go back as
atoms, bonds and coordinates: the atoms in the order they came in, the
hydrogens made for them after.

It also fills kinds of a workflow's step (docs/WORKFLOWS.md, its
manifest's `steps`), given molecules in 3D as Meno gives a writer one -
each atom's element, place, charge and radical, the bonds by index:

- *Conformers*, as a job: `prepare` says what each job is - this folder's
  conformers_job.py, run by its environment's Python on the molecule
  written into the job's folder - and `collect` reads back what it wrote;
- *Duplicates*, at once (`run`): those alike within an RMSD over their
  heavy atoms, at their best over the molecule's symmetries, set aside.
"""

import json
import math
import re
import os
import sys
import time

from rdkit import Chem, RDLogger, rdBase
from rdkit.Geometry import Point3D
from rdkit.Chem import (
    Crippen,
    Descriptors,
    rdChemReactions,
    rdCIPLabeler,
    rdDepictor,
    rdDistGeom,
    rdForceFieldHelpers,
    rdMolDescriptors,
)
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    GetStereoisomerCount,
    StereoEnumerationOptions,
)

# RDKit's own warnings go to stderr, not into the answers.
RDLogger.DisableLog("rdApp.*")
rdDepictor.SetPreferCoordGen(True)

# Meno's rule for which of RDKit's two depictors lays a drawing out: CoordGen,
# except for a ring of LARGEST_RING atoms or more, or MOST_RINGS rings or
# more, where RDKit's own does - by how long CoordGen takes. Measured with
# Compute2DCoords, RDKit 2026.03.6, on an Apple M4, 2026-11-01: CoordGen's
# time grows unevenly past some 25 rings or a ring of some 30 atoms - a chain
# of para-linked benzene rings 0.00 s at 25 rings, 1.1 s at 26, 0.01 s at 31,
# 2.1 s at 32, 9.0 s at 35; a ring of carbons 0.04 s at 30 atoms, 1.3 s at
# 33, 1.2 s at 34, 18.9 s at 37 - where RDKit's own took 0.01 s at most. The
# bounds are those set on 2026-10-11, when a ring of 34 carbons took 2.5 s
# and of 37 over 20 s, a chain of 32 rings 2.2 s and of 35 over 8 s. Meno
# lays both of the worker's drawings out again with its own engine
# (from_smiles, drawing_of); only their stereochemistry and scale come from
# here.
LARGEST_RING = 34
MOST_RINGS = 25


def lay_out_2d(mol):
    """A molecule's 2D coordinates: CoordGen's, but for a ring of
    LARGEST_RING atoms or more, or MOST_RINGS rings or more, RDKit's own
    depictor's."""
    rings = [len(r) for r in Chem.GetSymmSSSR(mol)]
    plain = len(rings) >= MOST_RINGS or max(rings, default=0) >= LARGEST_RING
    rdDepictor.Compute2DCoords(mol, forceRDKit=plain)


def acceptors_as_drawn(mol):
    """A molecule's coordination bonds (RDKit's dative, read from a MOL
    block's type 9 or a plugin entry's `coordination`) taking no hydrogen
    from either end, as Meno counts them (src/lib/io/unplaced.ts): RDKit
    counts a dative bond in its acceptor's valence, and gives a plain B
    acceptor two hydrogens where it is drawn with three. Each acceptor not a
    metal, its hydrogens not fixed, is given those it has with its dative
    bonds set aside - fixed. A metal is given none either way. Changed in
    place; unsanitised in, unsanitised out."""
    datives = [b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DATIVE]
    acceptors = {b.GetEndAtomIdx() for b in datives if not is_metal(b.GetEndAtom()) and not b.GetEndAtom().GetNoImplicit()}
    if not acceptors:
        return mol
    aside = Chem.RWMol(mol)
    for b in datives:
        aside.GetBondWithIdx(b.GetIdx()).SetBondType(Chem.BondType.ZERO)
    aside.UpdatePropertyCache(strict=False)
    for i in acceptors:
        atom = mol.GetAtomWithIdx(i)
        atom.SetNumExplicitHs(atom.GetNumExplicitHs() + aside.GetAtomWithIdx(i).GetNumImplicitHs())
        atom.SetNoImplicit(True)
    return mol


#: Acceptors of group 13, neutral with three bonds besides, whose coordination bond RDKit cannot hold as a dative
#: bond - it counts it in their valence - and so holds as the bond's dipolar form inside it (`dipolar_adducts`).
DIPOLAR_ACCEPTORS = {"B", "Al", "Ga", "In", "Tl"}
#: Donors that can carry a +1 charge with one bond more, as that form needs.
DIPOLAR_DONORS = {"N", "P", "As", "O", "S", "Se", "F", "Cl", "Br", "I"}


def adduct_why(mol, b):
    """Why a coordination bond to a full main-group acceptor cannot be held
    in its dipolar form, in words; None where it can."""
    donor, acceptor = b.GetBeginAtom(), b.GetEndAtom()
    if acceptor.GetSymbol() not in DIPOLAR_ACCEPTORS:
        return f"{acceptor.GetSymbol()} is not one of B, Al, Ga, In and Tl"
    if acceptor.GetFormalCharge():
        return f"{acceptor.GetSymbol()} is drawn charged"
    if donor.GetFormalCharge():
        return f"{donor.GetSymbol()} is drawn charged"
    if donor.GetSymbol() not in DIPOLAR_DONORS:
        return f"{donor.GetSymbol()} cannot carry the charge"
    datives = [x for x in acceptor.GetBonds() if x.GetBondType() == Chem.BondType.DATIVE]
    if len(datives) > 1:
        return f"{acceptor.GetSymbol()} takes {len(datives)} of them"
    besides = acceptor.GetNumExplicitHs() + sum(x.GetBondTypeAsDouble() for x in acceptor.GetBonds() if x.GetBondType() != Chem.BondType.DATIVE)
    if besides != 3:
        return f"{acceptor.GetSymbol()} has {besides:g} bonds besides, not three"
    return None


def dipolar_adducts(mol):
    """Each coordination bond to a group-13 acceptor, neutral with three
    bonds besides, from a donor that can carry a charge, held as its
    dipolar form: a single bond, the donor +1 and the acceptor -1. Inside
    RDKit only, which cannot hold it as a dative bond: the charges as drawn
    are kept on each atom (`DRAWN_CHARGE`) and are what is reported back, and
    the donor names its acceptor (`LENDS_TO`). The same charge and electrons
    as drawn. Changed in place; unsanitised in, unsanitised out. Those it
    held so, as (donor, acceptor)."""
    held = []
    for b in list(mol.GetBonds()):
        if b.GetBondType() != Chem.BondType.DATIVE or is_metal(b.GetEndAtom()) or adduct_why(mol, b):
            continue
        donor, acceptor = b.GetBeginAtom(), b.GetEndAtom()
        for atom in (donor, acceptor):
            atom.SetIntProp(DRAWN_CHARGE, atom.GetFormalCharge())
        b.SetBondType(Chem.BondType.SINGLE)
        donor.SetFormalCharge(donor.GetFormalCharge() + 1)
        acceptor.SetFormalCharge(acceptor.GetFormalCharge() - 1)
        donor.SetIntProp(LENDS_TO, acceptor.GetIdx())
        held.append((donor.GetIdx(), acceptor.GetIdx()))
    return held


#: An atom's charge as drawn, kept where RDKit holds it otherwise (`dipolar_adducts`), and its acceptor's index on a donor.
DRAWN_CHARGE = "meno_drawn_charge"
LENDS_TO = "meno_lends_to"


def drawn_charge(atom):
    """An atom's charge as drawn: kept where RDKit holds a dipolar form."""
    return atom.GetIntProp(DRAWN_CHARGE) if atom.HasProp(DRAWN_CHARGE) else atom.GetFormalCharge()


def dipolar_words(mol):
    """The coordination bonds held in their dipolar form, as a row of *How it
    was made* says what was computed - only what differs from the drawing,
    whose charges the structure shows: "N\u2192B computed as a single bond,
    N+ and B\u2212". None where there are none."""
    pairs = sorted({(a.GetSymbol(), mol.GetAtomWithIdx(a.GetIntProp(LENDS_TO)).GetSymbol()) for a in mol.GetAtoms() if a.HasProp(LENDS_TO)})
    if not pairs:
        return None
    if len(pairs) == 1:
        d, x = pairs[0]
        return f"{d}\u2192{x} computed as a single bond, {d}+ and {x}\u2212"
    return ", ".join(f"{d}\u2192{x}" for d, x in pairs) + " computed as single bonds, each donor + and acceptor \u2212"


def as_drawn(mol):
    """A copy of `mol` with each coordination bond held in its dipolar form
    (`dipolar_adducts`) as drawn again: a dative bond from its donor to its
    acceptor, and each atom's charge as drawn - for a drawing written back,
    which shows what the chemist drew, not how RDKit holds it. Where the
    atom a donor names as its acceptor is no longer that acceptor - not
    bonded to it, or not a group-13 acceptor held so (its charge as drawn
    kept, its charge here -1): the molecule renumbered since - why, in
    words: never a drawing written without its arrow, nor with it on
    another bond."""
    out = Chem.RWMol(mol)
    for a in mol.GetAtoms():
        if a.HasProp(LENDS_TO):
            to = a.GetIntProp(LENDS_TO)
            acceptor = mol.GetAtomWithIdx(to) if 0 <= to < mol.GetNumAtoms() else None
            bond = out.GetBondBetweenAtoms(a.GetIdx(), to) if acceptor is not None else None
            if (bond is None or not acceptor.HasProp(DRAWN_CHARGE) or acceptor.GetSymbol() not in DIPOLAR_ACCEPTORS
                    or acceptor.GetFormalCharge() != -1):
                raise ValueError(f"its coordination bond from {a.GetSymbol()} could not be written back as drawn")
            bond.SetBondType(Chem.BondType.DATIVE)
            if bond.GetBeginAtomIdx() != a.GetIdx():
                bond.SetBeginAtomIdx(a.GetIdx())
                bond.SetEndAtomIdx(a.GetIntProp(LENDS_TO))
        if a.HasProp(DRAWN_CHARGE):
            out.GetAtomWithIdx(a.GetIdx()).SetFormalCharge(a.GetIntProp(DRAWN_CHARGE))
    out = out.GetMol()
    out.UpdatePropertyCache(strict=False)
    return out


def held_dipolar(mol):
    """Whether `mol` holds a coordination bond in its dipolar form (`dipolar_adducts`)."""
    return any(a.HasProp(LENDS_TO) for a in mol.GetAtoms())


def sanitized(mol):
    """`mol` sanitised - a coordination bond to a full group-13 acceptor held
    in its dipolar form first (`dipolar_adducts`) - or why not in words: a
    coordination bond RDKit cannot hold otherwise, named, with why; any other
    problem as RDKit gives it."""
    dipolar_adducts(mol)
    problem = Chem.SanitizeMol(mol, catchErrors=True)
    if problem == Chem.SanitizeFlags.SANITIZE_NONE:
        return mol
    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.DATIVE or is_metal(b.GetEndAtom()):
            continue
        donor, acceptor = b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()
        why = adduct_why(mol, b) or f"{acceptor} takes no further bond there"
        raise ValueError(f"RDKit cannot hold the coordination bond from {donor} to {acceptor} as drawn: {why}.")
    raise ValueError(f"cannot make sense of the structure ({problem})")


def read(molblock, sanitize=True):
    """A molecule from a MOL block, kept as drawn: no hydrogens removed, and
    none taken by a coordination bond (`acceptors_as_drawn`); one to a full
    group-13 acceptor held in its dipolar form (`dipolar_adducts`), whether
    sanitised or not, so that every op reads an adduct the same way."""
    mol = Chem.MolFromMolBlock(molblock, sanitize=False, removeHs=False)
    if mol is None:
        raise ValueError("This structure could not be read.")
    acceptors_as_drawn(mol)
    dipolar_adducts(mol)
    if sanitize:
        sanitized(mol)
        Chem.AssignChiralTypesFromBondDirs(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol


def op_rxn_to_smiles(m):
    """A reaction as Meno writes it to an RXN file - its reactants, its reagents (agents) and its products, each
    molecule as its MOL block holds it, wedges and atom map numbers too - as a reaction SMILES: "reactants>agents>
    products", each molecule's canonical SMILES, its stereo kept, an H drawn to carry a wedge not written."""
    rxn = rdChemReactions.ReactionFromRxnBlock(m["rxnblock"], sanitize=False, removeHs=False)
    if rxn is None:
        raise ValueError("This reaction could not be read.")
    out = rdChemReactions.ChemicalReaction()
    for templates, add in ((rxn.GetReactants(), out.AddReactantTemplate), (rxn.GetAgents(), out.AddAgentTemplate), (rxn.GetProducts(), out.AddProductTemplate)):
        for mol in templates:
            mol = Chem.Mol(mol)
            problem = Chem.SanitizeMol(mol, catchErrors=True)
            if problem != Chem.SanitizeFlags.SANITIZE_NONE:
                raise ValueError(f"cannot make sense of a molecule of the reaction ({problem})")
            Chem.AssignChiralTypesFromBondDirs(mol)
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            # (an H drawn to carry a wedge not written, as `to_smiles` writes none: the reaction's atoms are read as
            # queries, which RDKit keeps their hydrogens on unless asked)
            ps = Chem.RemoveHsParameters()
            ps.removeWithQuery = True
            add(Chem.RemoveHs(mol, ps))
    return {"smiles": rdChemReactions.ReactionToSmiles(out)}


def op_to_smiles(m):
    # an H drawn to carry a wedge says its centre's configuration, which
    # the SMILES says without it
    return {"smiles": Chem.MolToSmiles(Chem.RemoveHs(read(m["molblock"])))}


def op_from_smiles(m):
    mol = Chem.MolFromSmiles(m["smiles"])
    if mol is None:
        # (why, in words, is Meno's to say - from the SMILES itself, and what RDKit made of it: `smiles_read`)
        raise ValueError("This SMILES could not be read.")
    return {"molblock": drawing_block(mol)}


def op_smiles_read(m):
    """What RDKit makes of a SMILES, as facts: whether it reads (`MolFromSmiles`); whether it parses at all,
    unsanitised; and, where it does, the problems RDKit's own check finds (`DetectChemistryProblems`), each its kind as
    RDKit names it with the atoms it names - for one atom, its symbol, its charge, and its bonds (their orders and its
    explicit hydrogens summed, as written). Saying why in words is Meno's (chem/smilesWhy)."""
    s = str(m.get("smiles") or "")
    reads = Chem.MolFromSmiles(s) is not None
    mol = Chem.MolFromSmiles(s, sanitize=False)
    if mol is None:
        return {"reads": reads, "parsed": False, "problems": []}
    problems = []
    for problem in Chem.DetectChemistryProblems(mol):
        kind = problem.GetType()
        if kind in ("AtomValenceException", "AtomKekulizeException"):
            k = problem.GetAtomIdx()
            atom = mol.GetAtomWithIdx(k)
            bonds = int(round(sum(b.GetBondTypeAsDouble() for b in atom.GetBonds()) + atom.GetNumExplicitHs()))
            problems.append({"kind": kind, "atom": k, "symbol": atom.GetSymbol(), "charge": atom.GetFormalCharge(), "bonds": bonds})
        elif kind == "KekulizeException":
            problems.append({"kind": kind, "atoms": sorted(problem.GetAtomIndices())})
        else:
            problems.append({"kind": kind})
    return {"reads": reads, "parsed": True, "problems": problems}


def drawing_block(mol):
    """A molecule as a drawing, a V3000 MOL block: laid out in 2D (for
    Meno's engine to lay out again), its stereo wedged as the molecule has
    it."""
    lay_out_2d(mol)
    # (an adduct held in its dipolar form written as drawn: its arrow, and its charges as drawn - `as_drawn`)
    written = lambda: Chem.MolToV3KMolBlock(as_drawn(mol) if held_dipolar(mol) else mol)
    block = written()
    # where what is written does not read back as the molecule, RDKit's own
    # depictor lays it out instead: (E)-cyclooctene laid out by CoordGen
    # reads back (Z), its double bond drawn cis, and by RDKit's own (E)
    # (RDKit 2026.03.6, 2026-11-01)
    if not reads_as(block, mol):
        rdDepictor.Compute2DCoords(mol, forceRDKit=True)
        block = written()
    return block


def reads_as(molblock, mol):
    """Whether a MOL block reads back as the molecule it was written from,
    stereochemistry and all."""
    if held_dipolar(mol):
        # (read as the worker reads a drawing, its adduct held in its dipolar form again: RDKit's own reading refuses it)
        try:
            back = read(molblock)
        except ValueError:
            return False
        return Chem.MolToSmiles(back) == Chem.MolToSmiles(mol)
    back = Chem.MolFromMolBlock(molblock)
    return back is not None and Chem.MolToSmiles(back) == Chem.MolToSmiles(mol)


def too_many(atom):
    """An atom with more bonds than it can have: its valence, and the most
    it could be - by the element's own valences, or, when it carries a
    charge, by those of the element it then has the electrons of."""
    table = Chem.GetPeriodicTable()
    z = atom.GetAtomicNum() - atom.GetFormalCharge()
    allowed = [v for v in table.GetValenceList(z) if v >= 0] if 0 < z <= 118 else []
    entry = {"valence": atom.GetValence(Chem.ValenceType.EXPLICIT)}
    if allowed:
        entry["most"] = max(allowed)
    return entry


def op_analyse(m):
    """What a chemist checks by eye: hydrogens, valence, aromatic rings,
    stereo labels - per atom and bond, in the order they came in. Each
    fragment is made sense of on its own, so that a carbon with five bonds
    in one does not keep another's stereocentres from being labelled.

    Where nothing is wrong anywhere, the whole is made sense of at once.
    Nothing worked out here reaches past a fragment, so it comes out as each
    fragment would on its own - and a drawing of a thousand fragments is not
    taken apart into them: RDKit takes a fragment out by removing every
    other atom, one at a time, which for a thousand records ran on for most
    of an hour."""
    mol = read(m["molblock"], sanitize=False)
    mol.UpdatePropertyCache(strict=False)
    atoms = [{"index": a.GetIdx(), "hydrogens": a.GetTotalNumHs()} for a in mol.GetAtoms()]
    bonds = [{"index": b.GetIdx()} for b in mol.GetBonds()]
    for p in Chem.DetectChemistryProblems(mol):
        if p.GetType() == "AtomValenceException":
            atoms[p.GetAtomIdx()]["valenceError"] = too_many(mol.GetAtomWithIdx(p.GetAtomIdx()))

    whole = Chem.Mol(mol)
    if Chem.SanitizeMol(whole, catchErrors=True) == Chem.SanitizeFlags.SANITIZE_NONE:
        parts = [(whole, range(whole.GetNumAtoms()))]
    else:
        mapping = []
        frags = Chem.GetMolFrags(
            mol, asMols=True, sanitizeFrags=False, fragsMolAtomMapping=mapping
        )
        parts = [
            (frag, index)
            for frag, index in zip(frags, mapping)
            if Chem.SanitizeMol(frag, catchErrors=True) == Chem.SanitizeFlags.SANITIZE_NONE
        ]
    for frag, index in parts:
        Chem.AssignChiralTypesFromBondDirs(frag)
        Chem.AssignStereochemistry(frag, cleanIt=True, force=True)
        rdCIPLabeler.AssignCIPLabels(frag)
        for a in frag.GetAtoms():
            entry = atoms[index[a.GetIdx()]]
            entry["hydrogens"] = a.GetTotalNumHs()
            if a.HasProp("_CIPCode"):
                entry["cip"] = a.GetProp("_CIPCode")
            if a.GetIsAromatic():
                entry["aromatic"] = True
        for b in frag.GetBonds():
            drawn = mol.GetBondBetweenAtoms(
                index[b.GetBeginAtomIdx()], index[b.GetEndAtomIdx()]
            )
            entry = bonds[drawn.GetIdx()]
            if b.GetIsAromatic():
                entry["aromatic"] = True
            if b.HasProp("_CIPCode"):
                entry["cip"] = b.GetProp("_CIPCode")
    return {"atoms": atoms, "bonds": bonds}


# --- 3D structures -----------------------------------------------------------

#: The most conformers a search may be asked to try: the most Meno's *How many* gives (lib/chem/conformersSought
#: `MOST_CONFORMERS`, where its basis is stated).
MOST_CONFORMERS = 5000
# conformers' random seed, and how many steps an optimisation may take, unless asked
SEED = 0x4D45
STEPS = 2000
# the force fields conformers may be optimised in
FIELDS = ("MMFF94", "MMFF94s", "UFF")


def whole_in(value, label, low, high):
    """`value` as a whole number from `low` to `high` - refused, saying so,
    where it is none (a fraction, text) or lies outside them: never cut to
    one, nor moved into range (the manifest marks each such option `whole`,
    and says its range)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value != int(value):
        raise ValueError(f"{label} is to be a whole number: {value!r} is not one")
    if not low <= value <= high:
        raise ValueError(f"{label} is to be from {low} to {high}: {int(value)} is not")
    return int(value)


def options_of(m):
    """How conformers are to be made: how many to try, as Meno sends it
    (`count`: the chemist's number, or as many as the molecule needs, which
    Meno works out - lib/chem/conformersSought), from 1 to MOST_CONFORMERS;
    and the options the manifest declares for the role (roleOptions), each
    as it may be. A number that is none of these is refused (`whole_in`)."""
    o = m.get("options") or {}

    def number(key, label, default, low, high):
        return whole_in(o[key], label, low, high) if o.get(key) is not None else default

    count = whole_in(m["count"], "How many conformers to try", 1, MOST_CONFORMERS) if m.get("count") is not None else None
    return {
        "count": count,
        "seed": number("seed", "Random seed", SEED, 0, 2**31 - 1),
        "field": o.get("field") if o.get("field") in FIELDS else "MMFF94",
        "iters": number("iters", "Optimisation steps, at most", STEPS, 10, 100000),
    }
#: Kilocalories per mole in a hartree: energies go back in hartrees. The hartree (4.3597447222071e-18 J,
#: CODATA 2018) times the Avogadro constant (6.02214076e23 per mole, exact, SI 2019), over the
#: thermochemical calorie (4.184 J, exact).
KCAL_PER_HARTREE = 627.509474


#: How FindPotentialStereo says an element's configuration is drawn, as Meno's requests name it: given, not given, or
#: drawn as not known - a wavy bond, a double bond of either configuration (CFG=2, as Meno writes it).
STATES = {
    Chem.StereoSpecified.Specified: "specified",
    Chem.StereoSpecified.Unspecified: "unspecified",
    Chem.StereoSpecified.Unknown: "unknown",
}


def stereo_read(blocks):
    """A structure drawn, its parts read as one (`drawing_of_parts`) and
    sanitised, its stereo assigned from its wedges - or what kept it from
    being: None where it cannot be read at all, else RDKit's sanitisation
    problem by name."""
    mol = drawing_of_parts(blocks)
    if mol is None:
        return None, None
    problem = Chem.SanitizeMol(mol, catchErrors=True)
    if problem != Chem.SanitizeFlags.SANITIZE_NONE:
        return None, str(problem).split(".")[-1]
    Chem.AssignChiralTypesFromBondDirs(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol, None


def blocks_of(m):
    """The parts a request gives: `molblocks`, or one `molblock`."""
    return m.get("molblocks") or [m["molblock"]]


def op_stereo_elements(m):
    """Every element of a structure that could be stereogenic, as RDKit's
    FindPotentialStereo finds it: its kind (`STEREO_KINDS`), the atom or
    bond it is centred on, by index - its parts' atoms and bonds in their
    order, one after another - and how its configuration is drawn
    (`STATES`). None left out: which are left open, and which never
    chosen, are Meno's (lib/chem/stereoOpen). One that cannot be read:
    `read` false; one RDKit cannot make sense of: its `problem`."""
    mol, problem = stereo_read(blocks_of(m))
    if mol is None:
        return {"read": False} if problem is None else {"read": True, "problem": problem}
    found = Chem.FindPotentialStereo(Chem.Mol(mol), cleanIt=True)
    return {
        "read": True,
        "elements": [
            {"kind": STEREO_KINDS.get(str(si.type).split(".")[-1], "other"), "at": si.centeredOn, "state": STATES.get(si.specified, "unknown")}
            for si in found
        ],
    }


def neighbours_of(atom):
    """An atom's neighbours, by index, in the order of its bonds: the order RDKit's chiral tags are read in."""
    return [b.GetOtherAtomIdx(atom.GetIdx()) for b in atom.GetBonds()]


def parity(named, order):
    """+1 where `named` is an even permutation of `order`, -1 where odd."""
    named, sign = list(named), 1
    for i, want in enumerate(order):
        j = named.index(want)
        if j != i:
            named[i], named[j] = named[j], named[i]
            sign = -sign
    return sign


def configured(mol, given):
    """`mol` with the configurations Meno gives (lib/chem/stereoOpen
    `Configuration`) as RDKit's tags. A centre's: the sign of the volume its
    first three neighbours span, in the order Meno names them - RDKit's
    CHI_TETRAHEDRAL_CCW where that is positive for its neighbours in the
    order of its bonds (measured on RDKit 2026.03.6, 2026-11-04, and held by
    scripts/calc/test_plugin_rdkit.py), turned by the parity of Meno's order
    against that one. A double bond's: cis or trans of the two atoms named."""
    out = Chem.Mol(mol)
    for c in (given or {}).get("centres") or []:
        atom = out.GetAtomWithIdx(int(c["atom"]))
        order, named = neighbours_of(atom), [int(n) for n in c["neighbours"]]
        if sorted(order) != sorted(named):
            raise ValueError(f"atom {c['atom']}'s neighbours are not {named}")
        sense = int(c["sense"]) * parity(named, order)
        atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW if sense > 0 else Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    for d in (given or {}).get("doubles") or []:
        bond = out.GetBondWithIdx(int(d["bond"]))
        bond.SetStereoAtoms(int(d["ends"][0]), int(d["ends"][1]))
        bond.SetStereo(Chem.BondStereo.STEREOCIS if d["cis"] else Chem.BondStereo.STEREOTRANS)
    if given and (given.get("centres") or given.get("doubles")):
        # (written as RDKit writes the stereoisomer it is: the double bonds' cis and trans as their neighbours'
        #  directions, which its perception reads, and then a tag on a centre the rest leave no stereocentre - a
        #  ribitol's middle carbon, its ends alike - taken off, and a ring's as its canonical form has it. Each of
        #  the 250 stereoisomers of a 70-structure set made again so is the one enumerated: 2026-11-04)
        Chem.SetDoubleBondNeighborDirections(out)
        Chem.AssignStereochemistry(out, cleanIt=True, force=True)
    return out


def configuration_of_isomer(mol, atoms, bonds):
    """What `configured` gives, read back from a stereoisomer: the
    configuration of each of the centres `atoms` and double bonds `bonds`
    (by index) it has one for, in Meno's terms."""
    centres, doubles = [], []
    for i in atoms:
        atom = mol.GetAtomWithIdx(i)
        tag = atom.GetChiralTag()
        if tag in (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
            centres.append({"atom": i, "neighbours": neighbours_of(atom), "sense": 1 if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW else -1})
    for i in bonds:
        bond = mol.GetBondWithIdx(i)
        stereo = bond.GetStereo()
        if stereo not in (Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS, Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ):
            continue
        # (E and Z as RDKit's enumeration sets them: of the two stereo atoms it names, trans and cis)
        ends = list(bond.GetStereoAtoms())
        if len(ends) != 2:
            continue
        doubles.append({"bond": i, "ends": ends, "cis": stereo in (Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOZ)})
    return {"centres": centres, "doubles": doubles}


def held_one_way(mol, moving, cis):
    """`mol` with each C=N in `moving` given a configuration - all cis, or all trans - so that an enumeration of
    what is left open passes them by; where an end of one has nothing else bonded to it, none."""
    out = Chem.Mol(mol)
    for i in moving:
        bond = out.GetBondWithIdx(i)
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        na = next((x.GetIdx() for x in a.GetNeighbors() if x.GetIdx() != b.GetIdx()), None)
        nb = next((x.GetIdx() for x in b.GetNeighbors() if x.GetIdx() != a.GetIdx()), None)
        if na is None or nb is None:
            continue
        bond.SetStereoAtoms(na, nb)
        bond.SetStereo(Chem.BondStereo.STEREOCIS if cis else Chem.BondStereo.STEREOTRANS)
    return out


def let_go(mol, moving):
    """`mol` with each C=N in `moving` left without a configuration again: of either, as Meno draws it."""
    for i in moving:
        mol.GetBondWithIdx(i).SetStereo(Chem.BondStereo.STEREOANY)
    return mol


class TooLong(Exception):
    """A structure's stereoisomers not all tried by the time given (`embeddable`'s `deadline`)."""


def embeddable(mol, moving, most, in_3d=True, deadline=None):
    """Each stereoisomer the stereo `mol` leaves unassigned gives - the ones
    it gives kept as they are - at most `most`, and, `in_3d`, only those
    that can be made in 3D (RDKit's embedding tried, seeded by the isomer:
    the same answer every time): a bridgehead left open gives only the one
    its rings allow. The C=N bonds `moving` are held one way while the rest
    are given (`held_one_way`) - the other way where none of them embeds so
    - and let go again in each: never enumerated, never chosen.

    Worked out once for each structure as it is asked about - its atoms,
    bonds and stereo as given, in their order, and how (`EMBEDDABLE`): a
    page asks what to make and then makes it, and trying to embed each
    stereoisomer is slow (a camphor some 10 s). The tries themselves are
    never cut short; but where a `deadline` is given (time.monotonic()),
    TooLong is raised once it has passed, looked at as each stereoisomer
    comes back - between RDKit's tries, never within one - and nothing is
    kept of it. Each caller is given copies of its own."""
    key = (Chem.MolToMolBlock(mol, includeStereo=True, kekulize=False), tuple(moving), most, in_3d)
    made = EMBEDDABLE.get(key)
    if made is None:
        options = StereoEnumerationOptions(onlyUnassigned=True, unique=True, maxIsomers=most, tryEmbedding=in_3d)
        made = []
        for cis in ([False, True] if moving else [None]):
            for isomer in EnumerateStereoisomers(held_one_way(mol, moving, cis) if moving else mol, options=options):
                if deadline is not None and time.monotonic() > deadline:
                    raise TooLong()
                made.append(let_go(isomer, moving))
            if made:
                break
        if deadline is not None and time.monotonic() > deadline:
            raise TooLong()
        if moving:
            unique = {}
            for isomer in made:
                unique.setdefault(Chem.MolToSmiles(isomer), isomer)
            made = list(unique.values())
        if len(EMBEDDABLE) >= EMBEDDABLE_KEPT:
            EMBEDDABLE.pop(next(iter(EMBEDDABLE)))
        EMBEDDABLE[key] = made
    return [Chem.Mol(m) for m in made]


#: What `embeddable` has worked out, by structure and how it was asked, the oldest first - as many as
#: `EMBEDDABLE_KEPT`.
EMBEDDABLE = {}
EMBEDDABLE_KEPT = 64


def op_stereoisomers(m):
    """A structure's stereoisomers, as Meno asks for them (Workspace/chem/
    openStereo): the configurations `configured` gives made first, then
    each stereoisomer the stereo still unassigned gives (`embeddable`) -
    tried in 3D where `inThreeD` - the C=N bonds named in `notEnumerated`
    passed by. At most `most`; where `whole`, one with more than that is
    answered `tooMany`, none made - counted only as far as one past it,
    never all (twenty double bonds left open give a million), with RDKit's
    bound on how many there could be. Not done within `withinMs`, where it
    is given: `tooLong`. Else each made, in the order RDKit gives them:
    its configuration of the open elements Meno names (`open`, its centres
    and double bonds), its SMILES and, with `drawings`, itself as a drawing
    (`drawing_block`) - or why it cannot be one; and `free` where none was
    made: none can be made in 3D with its stereo held."""
    started = time.monotonic()
    deadline = started + m["withinMs"] / 1000 if m.get("withinMs") else None
    mol, problem = stereo_read(blocks_of(m))
    if mol is None:
        raise ValueError(f"cannot make sense of the structure ({problem})" if problem else "This structure could not be read.")
    mol = configured(mol, m.get("configured"))
    moving = sorted(int(i) for i in m.get("notEnumerated") or [])
    most = int(m["most"])
    if m.get("whole"):
        # (how many the open elements could give at most, before any is made: over the most, counted only to one past
        #  it, as unique - one past the most is reached at once, but where many of the bound are alike, a symmetric
        #  structure with many left open: rare, and looked at no longer than any other. The C=N bonds passed by held
        #  one way: not counted)
        options = StereoEnumerationOptions(onlyUnassigned=True, unique=True, maxIsomers=0)
        counted_of = held_one_way(mol, moving, False)
        bound = GetStereoisomerCount(counted_of, options=options)
        if bound > most:
            for counted, _ in enumerate(EnumerateStereoisomers(counted_of, options=options), 1):
                if counted > most:
                    return {"tooMany": {"counted": counted, "bound": bound}}
                if deadline is not None and time.monotonic() > deadline:
                    return {"tooLong": True}
    try:
        made = embeddable(mol, moving, most, bool(m.get("inThreeD", True)), deadline)
    except TooLong:
        return {"tooLong": True}
    atoms = [int(i) for i in (m.get("open") or {}).get("atoms") or []]
    bonds = [int(i) for i in (m.get("open") or {}).get("bonds") or []]
    isomers = []
    for iso in made:
        one = {"configured": configuration_of_isomer(iso, atoms, bonds), "smiles": Chem.MolToSmiles(iso)}
        if m.get("drawings"):
            try:
                one["molblock"] = drawing_block(Chem.Mol(iso))
            except ValueError as e:
                # (an adduct's drawing that cannot be written back as drawn - `as_drawn`)
                one["error"] = str(e)
        isomers.append(one)
    return {"isomers": isomers, "free": not made}


# What FindPotentialStereo calls each kind of element, as Meno's configurations name them.
STEREO_KINDS = {
    "Atom_Tetrahedral": "centre",
    "Bond_Double": "double",
    "Bond_Atropisomer": "axis",
    "Bond_Cumulene_Even": "cumulene",
}


def configuration_of(molblock):
    """Which stereoisomer a molecule in 3D is, read from its own geometry:
    each element that could be stereogenic - a centre, a double bond, an
    axis - its atoms, and whether its configuration is given (from 3D,
    every one that the geometry fixes); and a key that is the same for two
    molecules exactly when they are the same stereoisomer (RDKit's
    canonical SMILES, its hydrogens taken away after the geometry was read).

    The elements are FindPotentialStereo's, which tells apart what a test of
    neighbours' ranks alone would not - cis and trans in a 1,4-disubstituted
    cyclohexane, whose ring carbons each carry the configuration. RDKit
    marks the function experimental: the tests in scripts/calc hold its
    answers on a set of known cases, run whenever the plugin's RDKit
    changes (docs/PLUGINS.md)."""
    mol = read(molblock)
    conf = mol.GetConformer()
    conf.Set3D(True)
    Chem.AssignStereochemistryFrom3D(mol, confId=conf.GetId(), replaceExistingTags=True)
    elements = []
    for s in Chem.FindPotentialStereo(mol, cleanIt=False, flagPossible=True):
        name = str(s.type).split(".")[-1]
        kind = STEREO_KINDS.get(name, "other")
        # (an atom's element is the atom; a bond's, its two atoms)
        if name.startswith("Atom_"):
            atoms = [s.centeredOn]
        else:
            b = mol.GetBondWithIdx(s.centeredOn)
            atoms = sorted([b.GetBeginAtomIdx(), b.GetEndAtomIdx()])
        elements.append({"kind": kind, "atoms": atoms, "specified": s.specified == Chem.StereoSpecified.Specified})
    return {"elements": elements, "key": Chem.MolToSmiles(Chem.RemoveHs(mol))}


def op_configurations(m):
    """Each molecule's configuration (`configuration_of`), in the order given
    - or, for one that cannot be read, why, so that the rest are still
    answered."""
    out = []
    for block in m["molblocks"]:
        try:
            out.append(configuration_of(block))
        except Exception as e:  # noqa: BLE001 - one molecule's failure is said, not raised
            out.append({"error": str(e) or type(e).__name__})
    return {"molecules": out}


def kekule_order(bond):
    # (aromatic rings as alternating bonds: a 3D structure's bonds are drawn
    # as single, double and triple)
    return {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3}.get(bond.GetBondType(), 1)


def mirror_of(mol):
    """A stereoisomer's mirror image, its atoms and bonds as they are - or
    None when it has stereo that is not known to turn in a mirror (only
    stereocentres and axes do; double bonds stay as they are)."""
    m = Chem.Mol(mol)
    swap = {
        Chem.ChiralType.CHI_TETRAHEDRAL_CW: Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW: Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    }
    for a in m.GetAtoms():
        tag = a.GetChiralTag()
        if tag in swap:
            a.SetChiralTag(swap[tag])
        elif tag != Chem.ChiralType.CHI_UNSPECIFIED:
            return None
    axes = {
        Chem.BondStereo.STEREOATROPCW: Chem.BondStereo.STEREOATROPCCW,
        Chem.BondStereo.STEREOATROPCCW: Chem.BondStereo.STEREOATROPCW,
    }
    for b in m.GetBonds():
        if b.GetStereo() in axes:
            b.SetStereo(axes[b.GetStereo()])
    return m


def smiles_from_3d(mol, frame):
    """The SMILES a structure's 3D coordinates (its atoms', hydrogens and
    all, as a flat list) say, stereo and all."""
    m = Chem.AddHs(mol)
    conf = Chem.Conformer(m.GetNumAtoms())
    for i in range(m.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(frame[3 * i], frame[3 * i + 1], frame[3 * i + 2]))
    m.RemoveAllConformers()
    m.AddConformer(conf, assignId=True)
    Chem.AssignStereochemistryFrom3D(m)
    return smiles_of(m)


def smiles_of(mol):
    """A stereoisomer's SMILES, any hydrogens drawn left out."""
    return Chem.MolToSmiles(Chem.RemoveHs(mol))


def reflected(entry):
    """Conformers in a mirror: every x turned about the plane x = 0."""
    return {
        **entry,
        "frames": [[-v if k % 3 == 0 else v for k, v in enumerate(f)] for f in entry["frames"]],
    }


# --- metals ------------------------------------------------------------------

# A metal's bonds, as ETKDG and UFF are given them. ETKDG's bounds for a bond
# to a metal come from radii that put Pd-Cl near 2.7 A, and its ligands may
# meet; UFF types a metal from its hybridization, which RDKit does not perceive
# for one (it said "Unrecognized hybridization" for Pd, and with no terms for
# its bonds pushed it 8 to 25 A away or crashed). So a metal's bonds are given
# their covalent lengths, its ligands kept apart, and it is typed for UFF as a
# geometry UFF has for it; where UFF has none, nothing is optimised, and the
# chip says so.

#: Pyykko's single-, double- and triple-bond covalent radii, in picometres:
#: the same as src/lib/chem/elements.ts (whose comment gives the sources), held
#: so by test_plugin_rdkit.py.
RADII = {
    "H": (32, None, None), "He": (46, None, None), "Li": (133, 124, None), "Be": (102, 90, 85),
    "B": (85, 78, 73), "C": (75, 67, 60), "N": (71, 60, 54), "O": (63, 57, 53), "F": (64, 59, 53),
    "Ne": (67, 96, None), "Na": (155, 160, None), "Mg": (139, 132, 127), "Al": (126, 113, 111),
    "Si": (116, 107, 102), "P": (111, 102, 94), "S": (103, 94, 95), "Cl": (99, 95, 93),
    "Ar": (96, 107, 96), "K": (196, 193, None), "Ca": (171, 147, 133), "Sc": (148, 116, 114),
    "Ti": (136, 117, 108), "V": (134, 112, 106), "Cr": (122, 111, 103), "Mn": (119, 105, 103),
    "Fe": (116, 109, 102), "Co": (111, 103, 96), "Ni": (110, 101, 101), "Cu": (112, 115, 120),
    "Zn": (118, 120, None), "Ga": (124, 117, 121), "Ge": (121, 111, 114), "As": (121, 114, 106),
    "Se": (116, 107, 107), "Br": (114, 109, 110), "Kr": (117, 121, 108), "Rb": (210, 202, None),
    "Sr": (185, 157, 139), "Y": (163, 130, 124), "Zr": (154, 127, 121), "Nb": (147, 125, 116),
    "Mo": (138, 121, 113), "Tc": (128, 120, 110), "Ru": (125, 114, 103), "Rh": (125, 110, 106),
    "Pd": (120, 117, 112), "Ag": (128, 139, 137), "Cd": (136, 144, None), "In": (142, 136, 146),
    "Sn": (140, 130, 132), "Sb": (140, 133, 127), "Te": (136, 128, 121), "I": (133, 129, 125),
    "Xe": (131, 135, 122), "Cs": (232, 209, None), "Ba": (196, 161, 149), "La": (180, 139, 139),
    "Ce": (163, 137, 131), "Pr": (176, 138, 128), "Nd": (174, 137, None), "Pm": (173, 135, None),
    "Sm": (172, 134, None), "Eu": (168, 134, None), "Gd": (169, 135, 132), "Tb": (168, 135, None),
    "Dy": (167, 133, None), "Ho": (166, 133, None), "Er": (165, 133, None), "Tm": (164, 131, None),
    "Yb": (170, 129, None), "Lu": (162, 131, 131), "Hf": (152, 128, 122), "Ta": (146, 126, 119),
    "W": (137, 120, 115), "Re": (131, 119, 110), "Os": (129, 116, 109), "Ir": (122, 115, 107),
    "Pt": (123, 112, 110), "Au": (124, 121, 123), "Hg": (133, 142, None), "Tl": (144, 142, 150),
    "Pb": (144, 135, 137), "Bi": (151, 141, 135), "Po": (145, 135, 129), "At": (147, 138, 138),
    "Rn": (142, 145, 133), "Fr": (223, 218, None), "Ra": (201, 173, 159), "Ac": (186, 153, 140),
    "Th": (175, 143, 136), "Pa": (169, 138, 129), "U": (170, 134, 118), "Np": (171, 136, 116),
    "Pu": (172, 135, None), "Am": (166, 135, None), "Cm": (166, 136, None), "Bk": (168, 139, None),
    "Cf": (168, 140, None), "Es": (165, 140, None), "Fm": (167, None, None),
    "Md": (173, 139, None), "No": (176, 159, None), "Lr": (161, 141, None), "Rf": (157, 140, 131),
    "Db": (149, 136, 126), "Sg": (143, 128, 121), "Bh": (141, 128, 119), "Hs": (134, 125, 118),
    "Mt": (129, 125, 113), "Ds": (128, 116, 112), "Rg": (121, 116, 118), "Cn": (122, 137, 130),
    "Nh": (136, None, None), "Fl": (143, None, None), "Mc": (162, None, None),
    "Lv": (175, None, None), "Ts": (165, None, None), "Og": (157, None, None),
}

#: How far a metal's bond may be from its covalent length while embedding (A).
METAL_BOND_SLACK = 0.05
#: The narrowest angle two atoms bonded to one metal, of different ligands, are let make while embedding and
#: optimising (degrees), so that ligands do not pile up - ETKDG knows no angle at a metal, and without one the
#: seven fluorides of [ZrF7]3- came out 33 degrees apart - by how many atoms are bonded to the metal (FLOORS):
#: - at most FLOOR_MOST_BONDED: NARROWEST_AT_METAL, below the narrowest angle of the ideal polyhedra of four,
#:   five and six points, a trigonal prism with equal edges, 2 asin(sqrt(3/7)) = 81.8 degrees (the
#:   octahedron, square plane, trigonal bipyramid and square pyramid: 90);
#: - at most FLOOR_MOST_CROWDED: NARROWEST_CROWDED, below the narrowest angle of the ideal polyhedra of seven to
#:   twelve points that were worked out: the pentagonal bipyramid, 72 degrees (360/5), and the icosahedron,
#:   63.4 (arctan 2). No common polyhedron of ten or eleven points was: there the margin rests on the
#:   icosahedron alone, 3.4 degrees;
#: - more: none.
#: Only between point donors (floor_pairs): none at a metal with a ligand bound through atoms side by side
#: (eta), whose atoms spread round where it sits - Cp2TiCl2's Cp carbon and Cl come near 76 degrees, a
#: Zeise-type alkene's carbon and its cis Cl near 71; never within one ligand, whose own bonds set its bite (en
#: on a lanthanide bites about 65); nor for a hydride, which may sit closer to another (cis H-M-H).
NARROWEST_AT_METAL = 75.0
#: The most atoms bonded to a metal at which NARROWEST_AT_METAL holds.
FLOOR_MOST_BONDED = 6
#: The floor where more are bonded (degrees), and the most at which it holds.
NARROWEST_CROWDED = 60.0
FLOOR_MOST_CROWDED = 12
#: The floor at a metal by how many atoms are bonded to it: (at most this many, the angle), in order.
FLOORS = ((FLOOR_MOST_BONDED, NARROWEST_AT_METAL), (FLOOR_MOST_CROWDED, NARROWEST_CROWDED))
#: How far under its floor a conformer may come out and be kept (degrees). The floor is held by bounds while
#: embedding and a restraint while optimising, neither of them exact: measured here at three seeds each,
#: [ZrF7]3-, [NbF7]2- and OsCl4Br4, pressed against the 60 degree floor, came out up to 2.0 degrees under it;
#: over 1230 embeddings of [OsCl6]2-, PdCl2(PPh3)2 and OsCl4(PEt3)2 none came within 4 degrees of the 75
#: degree floor but the one stuck in a well 13 under it. Five degrees keeps the first and drops the last.
FLOOR_SLACK = 5.0


def floor_for(bonded):
    """The floor at a metal with `bonded` atoms bonded to it (degrees), or None where none holds."""
    return next((angle for most, angle in FLOORS if bonded <= most), None)


#: UFF's geometries for a metal, in the digit its atom types use, and the
#: hybridization RDKit types each from: tetrahedral, square planar, octahedral,
#: then the rest - tried in this order for up to four bonds, octahedral first
#: for more.
UFF_GEOMETRY = {
    "3": Chem.HybridizationType.SP3,
    "4": Chem.HybridizationType.SP2D,
    "6": Chem.HybridizationType.SP3D2,
    "5": Chem.HybridizationType.SP3D,
    "1": Chem.HybridizationType.SP,
    "2": Chem.HybridizationType.SP2,
}
#: What each geometry is called on the chip.
GEOMETRY_WORDS = {"1": "linear", "2": "trigonal", "3": "tetrahedral", "4": "square planar", "5": "trigonal bipyramidal", "6": "octahedral"}


def is_metal(atom):
    """A d- or f-block atom: one ETKDG and UFF need told about."""
    z = atom.GetAtomicNum()
    return 21 <= z <= 30 or 39 <= z <= 48 or 57 <= z <= 80 or 89 <= z <= 112


def covalent_length(bond):
    """A bond's length from its atoms' covalent radii, for its order where
    both have a radius for it, else as single (A)."""
    a, b = bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()
    k = {Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2}.get(bond.GetBondType(), 0)
    ra, rb = RADII.get(a, (None,) * 3), RADII.get(b, (None,) * 3)
    if ra[k] is None or rb[k] is None:
        k = 0
    if ra[0] is None or rb[0] is None:
        return None
    return (ra[k] + rb[k]) / 100.0


#: How much longer than its covalent length a metal's bond to each atom of a
#: ring or chain bound through all its atoms (eta) is taken to be (A): Meno's
#: rule, which gives ferrocene's Fe-C as about 2.06, ruthenocene's Ru-C 2.15,
#: bis(benzene)chromium's Cr-C 2.12 and zirconocene dichloride's Zr-C 2.44,
#: against about 2.05, 2.19, 2.14 and 2.50 measured.
ETA_EXTRA = 0.15
#: How near an atom of an eta system and another ligand's donor of the same
#: metal may come (A).
ETA_CLEAR = 2.6
#: How near two atoms of two eta systems of one metal may come (A): about
#: the nearest two carbons of two rings come in a bent metallocene, so that
#: the rings do not overlap - the one restraint between them, the force field
#: counting nothing between fragments; their angle at the metal is UFF's.
ETA_APART = 2.9


def eta_systems(mol):
    """Each ring or chain bound to a metal through all its atoms, as the
    CTfile writes one - a bond from the metal to a star at its centre, its
    ENDPTS the atoms, ATTACH=ALL (RDKit keeps the bond as a dative one to a
    dummy atom, the ENDPTS as its property): [(metal, star, atoms)]."""
    found = []
    for bond in mol.GetBonds():
        if not bond.HasProp("_MolFileBondEndPts"):
            continue
        a, z = bond.GetBeginAtom(), bond.GetEndAtom()
        star, metal = (z, a) if z.GetAtomicNum() == 0 else (a, z)
        if star.GetAtomicNum() != 0 or not is_metal(metal):
            continue
        words = bond.GetProp("_MolFileBondEndPts").strip("()").split()
        atoms = [int(w) - 1 for w in words[1:]]
        if len(atoms) >= 2 and all(0 <= i < mol.GetNumAtoms() for i in atoms):
            found.append((metal.GetIdx(), star.GetIdx(), atoms))
    return found


def as_ligand_anions(mol, eta):
    """A ring of five or a chain of three carbons bound through all its
    atoms and drawn neutral - one carbon not double-bonded within it, with
    two hydrogens or an unpaired electron - taken as the cyclopentadienide
    or allyl anion it is bound as, the metal one charge higher, as a
    complex's formula balances it (src/lib/chem/ligands.ts): not a ring with
    a CH2 in it, nor radical carbons. Whether any was."""
    changed = False
    for metal, _, atoms in eta:
        inside = set(atoms)
        system = [mol.GetAtomWithIdx(i) for i in atoms]
        if len(atoms) not in (3, 5) or any(a.GetSymbol() != "C" or a.GetFormalCharge() for a in system):
            continue
        lone = [
            a for a in system
            if not any(b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtomIdx(a.GetIdx()) in inside for b in a.GetBonds())
        ]
        if len(lone) != 1:
            continue
        c = lone[0]
        held = sum(b.GetBondTypeAsDouble() for b in c.GetBonds() if b.GetBondType() != Chem.BondType.DATIVE)
        c.SetFormalCharge(-1)
        c.SetNumRadicalElectrons(0)
        c.SetNoImplicit(True)
        c.SetNumExplicitHs(max(0, 3 - int(round(held))))
        m = mol.GetAtomWithIdx(metal)
        m.SetFormalCharge(m.GetFormalCharge() + 1)
        changed = True
    if changed:
        mol.UpdatePropertyCache(strict=False)
        problem = Chem.SanitizeMol(mol, catchErrors=True)
        if problem != Chem.SanitizeFlags.SANITIZE_NONE:
            raise ValueError(f"cannot make sense of the structure ({problem})")
    return changed


def eta_apart(mol, eta):
    """Two eta systems of one metal kept from overlapping: each pair of their
    heavy atoms at least ETA_APART apart, [(i, j, low)]."""
    by_metal = {}
    for metal, _, atoms in eta:
        by_metal.setdefault(metal, []).append([c for c in atoms if mol.GetAtomWithIdx(c).GetAtomicNum() > 1])
    out = []
    for systems in by_metal.values():
        for x, a in enumerate(systems):
            for b in systems[x + 1:]:
                out += [(i, j, ETA_APART) for i in a for j in b]
    return out


def eta_length(mol, metal, atom):
    """A metal's distance to an atom of an eta system it binds (A), or None."""
    rm = RADII.get(mol.GetAtomWithIdx(metal).GetSymbol(), (None,))[0]
    ra = RADII.get(mol.GetAtomWithIdx(atom).GetSymbol(), (None,))[0]
    return None if rm is None or ra is None else (rm + ra) / 100.0 + ETA_EXTRA


def metal_bounds(mol, eta=()):
    """ETKDG's bounds for a molecule with metals: each metal's bonds at their
    covalent lengths, and its ligands at least a floor's angle apart as
    seen from it where one holds (floor_pairs); None where it has no metal
    bonded, or the bounds cannot be made to agree."""
    from rdkit import DistanceGeometry
    metals = [a for a in mol.GetAtoms() if is_metal(a) and a.GetDegree()]
    if not metals:
        return None
    stars = {star for _, star, _ in eta}
    # (the molecule with the metals' bonds left out)
    apart = Chem.RWMol(mol)
    for m in sorted((m.GetIdx() for m in metals), reverse=True):
        for n in [n.GetIdx() for n in apart.GetAtomWithIdx(m).GetNeighbors()]:
            apart.RemoveBond(m, n)
    try:
        bm = rdDistGeom.GetMoleculeBoundsMatrix(mol)
    except RuntimeError:
        # (RDKit's own bounds can fail for a ring closed only through metals - two metals bridged by
        # two halides: then those of the molecule without its metals' bonds, which the rest sets)
        free = apart.GetMol()
        free.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(free)
        bm = rdDistGeom.GetMoleculeBoundsMatrix(free)

    def put(i, j, low, high):
        i, j = min(i, j), max(i, j)
        bm[i][j] = high  # (upper bounds above the diagonal, lower below)
        bm[j][i] = low

    floor = {}
    for m, j, k, angle in floor_pairs(mol):
        floor.setdefault(m, []).append((j, k, angle))
    for m in metals:
        i = m.GetIdx()
        floor.setdefault(i, [])
        length = {}
        for bond in m.GetBonds():
            d = covalent_length(bond)
            if d is None:
                continue
            j = bond.GetOtherAtomIdx(i)
            length[j] = d
            put(i, j, d - METAL_BOND_SLACK, d + METAL_BOND_SLACK)
        for j, k, angle in floor[i]:
            if j not in length or k not in length:
                continue
            cos = math.cos(math.radians(angle))
            dj, dk = length[j], length[k]
            low = math.sqrt(dj * dj + dk * dk - 2 * dj * dk * cos)
            lo, hi = min(j, k), max(j, k)
            put(j, k, max(bm[hi][lo], low), max(dj + dk + 2 * METAL_BOND_SLACK, low))
    # an eta system over its metal: each of its atoms at its eta length, its star anywhere (set at its
    # centre after), the atoms on it let come near the metal, and other ligands' donors kept off it
    n = mol.GetNumAtoms()
    for metal, star, atoms in eta:
        for k in range(n):
            if k != star:
                put(k, star, 0.0, 100.0)
        donors = [x.GetIdx() for x in mol.GetAtomWithIdx(metal).GetNeighbors() if x.GetIdx() not in stars]
        for c in atoms:
            d = eta_length(mol, metal, c)
            if d is not None:
                put(metal, c, d - METAL_BOND_SLACK, d + METAL_BOND_SLACK)
            for x in mol.GetAtomWithIdx(c).GetNeighbors():
                k = x.GetIdx()
                if k not in atoms and k != metal and k != star:
                    lo, hi = min(k, metal), max(k, metal)
                    bm[hi][lo] = min(bm[hi][lo], 2.5)
            for k in donors:
                lo, hi = min(k, c), max(k, c)
                bm[hi][lo] = max(bm[hi][lo], ETA_CLEAR)
                bm[lo][hi] = max(bm[lo][hi], ETA_CLEAR)
    # (two eta systems of one metal kept from overlapping: eta_apart)
    for i, j, low in eta_apart(mol, eta):
        lo, hi = min(i, j), max(i, j)
        bm[hi][lo] = max(bm[hi][lo], low)
        bm[lo][hi] = max(bm[lo][hi], low)
    if not DistanceGeometry.DoTriangleSmoothing(bm):
        return None
    return bm


#: A square-planar metal's angles while UFF optimises (degrees): RDKit types
#: Pd, Pt, Ni and Au as square planar (Pd4+2) only from SP2D, for which its
#: angle terms have their one minimum at 90 degrees, and four ligands then
#: settled near tetrahedral. So two trans and four cis angles are held.
TRANS_AT_METAL = (172.0, 180.0)
CIS_AT_METAL = (85.0, 95.0)


def planar_angles(mol, cid):
    """The angles to hold at each square-planar metal with four bonds, in one
    conformer: its two trans pairs those of its ligands that are furthest
    apart there as embedded, the rest cis - so that cis and trans isomers
    both come out of a search, as embedding left them. [(i, metal, j, (low, high))]"""
    pos = mol.GetConformer(cid).GetPositions()
    held = []
    for atom in mol.GetAtoms():
        if not is_metal(atom) or atom.GetDegree() != 4 or atom.GetHybridization() != Chem.HybridizationType.SP2D:
            continue
        c = atom.GetIdx()
        nb = [n.GetIdx() for n in atom.GetNeighbors()]

        def angle(i, j):
            u, v = pos[i] - pos[c], pos[j] - pos[c]
            cos = float((u * v).sum() / math.sqrt(float((u * u).sum() * (v * v).sum())))
            return math.degrees(math.acos(max(-1.0, min(1.0, cos))))

        pairings = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
        trans = max(pairings, key=lambda pr: sum(angle(nb[a], nb[b]) for a, b in pr))
        for x in range(4):
            for y in range(x + 1, 4):
                held.append((nb[x], c, nb[y], TRANS_AT_METAL if (x, y) in trans else CIS_AT_METAL))
    return held


#: A metal UFF has no type for, and the lighter one of its group that stands
#: in for it while UFF optimises: the stand-in's angle terms shape it, and its
#: bonds are held at its own covalent lengths. Its energies are not the
#: metal's, so none are given.
STAND_IN = {"Ru": "Fe", "Os": "Fe", "Rh": "Co", "Ir": "Co", "W": "Mo", "Zr": "Ti", "Hf": "Ti", "Y": "Sc"}
#: How strongly a stood-in metal's bonds, and any metal's ligands' distance
#: apart, are held (kcal/mol/A^2): UFF's octahedral and square-planar angle
#: terms have a minimum at 0 degrees too, and two ligands of a metal are
#: never kept apart by their van der Waals terms (they are 1-3), so without
#: this two ligands could settle 30 degrees apart.
HOLD_BOND = 500.0
HOLD_APART = 100.0


def ligands_apart(mol):
    """Each metal's ligands held at least a floor's angle apart as seen
    from it, where one holds (floor_pairs), as metal_bounds keeps them
    while embedding: [(j, k, low, high)]."""
    held = []
    for m, j, k, angle in floor_pairs(mol):
        cos = math.cos(math.radians(angle))
        dj = covalent_length(mol.GetBondBetweenAtoms(m, j))
        dk = covalent_length(mol.GetBondBetweenAtoms(m, k))
        if dj is not None and dk is not None:
            held.append((j, k, math.sqrt(dj * dj + dk * dk - 2 * dj * dk * cos), dj + dk + 1.0))
    return held


def floor_pairs(mol):
    """The atoms a floor holds apart, as seen from their metal, and the
    angle: [(metal, j, k, angle)] - at a metal with as many atoms bonded to
    it as a floor holds at (floor_for) and no eta ligand on it (one through a star, or through atoms
    bonded to each other), each two of different ligands - joined by no
    path but through a metal - neither a hydride."""
    import itertools

    metals = [a for a in mol.GetAtoms() if is_metal(a) and a.GetDegree()]
    if not metals:
        return []
    apart = Chem.RWMol(mol)
    for m in sorted((a.GetIdx() for a in metals), reverse=True):
        for n in [n.GetIdx() for n in apart.GetAtomWithIdx(m).GetNeighbors()]:
            apart.RemoveBond(m, n)
    hops = Chem.GetDistanceMatrix(apart, force=True)  # (not the copy's cached one, with the metal's bonds)
    whole = mol.GetNumAtoms()

    def hydride(atom):
        return atom.GetAtomicNum() == 1 and atom.GetDegree() == 1

    pairs = []
    for metal in metals:
        near = sorted(metal.GetNeighbors(), key=lambda x: x.GetIdx())
        angle = floor_for(len(near))
        if len(near) < 2 or angle is None or any(x.GetAtomicNum() == 0 for x in near):
            continue
        if any(mol.GetBondBetweenAtoms(j.GetIdx(), k.GetIdx()) for j, k in itertools.combinations(near, 2)):
            continue
        for j, k in itertools.combinations(near, 2):
            if hops[j.GetIdx()][k.GetIdx()] <= whole or hydride(j) or hydride(k):
                continue
            pairs.append((metal.GetIdx(), j.GetIdx(), k.GetIdx(), angle))
    return pairs


def angle_at(mol, cid, m, j, k):
    """The angle j-m-k in conformer `cid` (degrees)."""
    p = mol.GetConformer(cid).GetPositions()
    u, v = p[j] - p[m], p[k] - p[m]
    cos = float((u * v).sum() / math.sqrt(float((u * u).sum() * (v * v).sum())))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def floor_words(mol, symbol, floor):
    """What the floor did at the metals of `symbol`, for its row: the angle held, or why none was."""
    at = {m: angle for m, _, _, angle in floor}
    metals = [a for a in mol.GetAtoms() if a.GetSymbol() == symbol and a.GetDegree() > 1]
    if not metals:
        return ""
    held = sorted({at[a.GetIdx()] for a in metals if a.GetIdx() in at})
    if held:
        return ", ligands at least " + " or ".join(f"{x:g}\u00b0" for x in held) + " apart"
    a = metals[0]
    if floor_for(a.GetDegree()) is None:
        return f", no angle held between ligands: {a.GetDegree()} atoms bonded to it"
    near = list(a.GetNeighbors())
    if any(x.GetAtomicNum() == 0 for x in near) or any(
        mol.GetBondBetweenAtoms(x.GetIdx(), y.GetIdx()) for i, x in enumerate(near) for y in near[i + 1:]
    ):
        return ", no angle held between ligands: a ligand bound through atoms side by side"
    return ""


#: Halogens bridging two metals (mu-halides, as in [Pd(allyl)Cl]2): UFF's
#: types for them have a natural angle of 180 degrees, and RDKit's angle term
#: for an atom it does not take as linear divides by its sine - so such an
#: atom's angles blow up, pulling the molecule apart. While UFF optimises it
#: is given the linear form, which stays finite, its bonds to the metals are
#: held at their lengths, and its angle between them as embedded,
#: BRIDGE_SLACK degrees either way. The energies given are UFF's with that
#: form.
HALOGENS = {"F", "Cl", "Br", "I"}
BRIDGE_SLACK = 10.0


def bridges(mol):
    """Each halogen bound to two metals or more, and each pair of them, as
    angles at it: [(metal, halogen, metal)]."""
    out = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in HALOGENS:
            metals = [n.GetIdx() for n in atom.GetNeighbors() if is_metal(n)]
            out += [(i, atom.GetIdx(), j) for x, i in enumerate(metals) for j in metals[x + 1:]]
    return out


def bridge_angles(mol, cid, bridged):
    """The angles to hold at each bridging halogen in one conformer: as
    embedded there, BRIDGE_SLACK either way. [(i, halogen, j, (low, high))]"""
    pos = mol.GetConformer(cid).GetPositions()
    held = []
    for i, c, j in bridged:
        u, v = pos[i] - pos[c], pos[j] - pos[c]
        cos = float((u * v).sum() / math.sqrt(float((u * u).sum() * (v * v).sum())))
        a = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
        held.append((i, c, j, (max(0.0, a - BRIDGE_SLACK), min(180.0, a + BRIDGE_SLACK))))
    return held


def uff_with_metals(mol, ids, iters, ff_mol=None, bonds=(), apart=(), interfrag=False):
    """UFF's optimisation of each conformer of a molecule with metals: its
    metals' ligands kept apart, its square-planar metals' angles held
    (planar_angles), `bonds` held at their lengths and `apart` [(i, j, low,
    high)] kept so - done on `ff_mol`, the molecule with stand-ins, where one
    is given, and the places taken back; with `interfrag`, nothing between
    its fragments counted but these (an eta system's metal and ring are).
    As UFFOptimizeMoleculeConfs answers: (not converged, energy) each, the
    energy UFF's own, without the holding terms."""
    ff_mol = ff_mol or mol
    apart = ligands_apart(mol) + list(apart)
    bridged = bridges(ff_mol)
    if bridged:
        if ff_mol is mol:
            ff_mol = Chem.Mol(mol)
        for _, x, _ in bridged:
            ff_mol.GetAtomWithIdx(x).SetHybridization(Chem.HybridizationType.SP)
        # (and its bonds to the metals held at their lengths, as a stood-in metal's are)
        held = {(min(i, j), max(i, j)) for i, j, _ in bonds}
        bonds = list(bonds)
        for i, x, j in bridged:
            for m in (i, j):
                bond = mol.GetBondBetweenAtoms(m, x)
                d = covalent_length(bond)
                if d is not None and (min(m, x), max(m, x)) not in held:
                    held.add((min(m, x), max(m, x)))
                    bonds.append((m, x, d))

    def uff(cid):
        return rdForceFieldHelpers.UFFGetMoleculeForceField(ff_mol, confId=cid, ignoreInterfragInteractions=interfrag)

    def held_by(cid):
        ff = uff(cid)
        for i, j, d in bonds:
            ff.UFFAddDistanceConstraint(i, j, False, d, d, HOLD_BOND)
        for j, k, low, high in apart:
            ff.UFFAddDistanceConstraint(j, k, False, low, high, HOLD_APART)
        return ff

    angles = {cid: planar_angles(ff_mol, cid) + bridge_angles(ff_mol, cid, bridged) for cid in ids}
    more = {}
    if not any(angles.values()):
        # (the same terms for every conformer: optimised at once, on every core)
        done = rdForceFieldHelpers.OptimizeMoleculeConfs(ff_mol, held_by(ids[0]), numThreads=0, maxIters=iters)
        more = {cid: m for cid, (m, _) in zip(ids, done)}
    else:
        for cid in ids:
            ff = held_by(cid)
            for i, c, j, (low, high) in angles[cid]:
                ff.UFFAddAngleConstraint(i, c, j, False, low, high, 200.0)
            ff.Initialize()
            more[cid] = ff.Minimize(maxIts=iters)
    if ff_mol is not mol:
        for cid in ids:
            conf = mol.GetConformer(cid)
            for k, p in enumerate(ff_mol.GetConformer(cid).GetPositions()):
                conf.SetAtomPosition(k, Point3D(float(p[0]), float(p[1]), float(p[2])))
    return [(more[cid], uff(cid).CalcEnergy()) for cid in ids]


def uff_stood_in(mol, ids, iters, eta=()):
    """Each conformer optimised by UFF with STAND_IN's metals standing in for
    those it has no type for, their bonds held at their own covalent lengths;
    and each eta system's star taken out of it (a lone hydrogen with no
    interactions: the force field has no star), the system held at its eta lengths over
    its metal, the metal typed for its sites - an eta system one - and
    nothing between fragments counted but what is held. As
    UFFOptimizeMoleculeConfs answers, with the stand-ins' energies; None
    where a metal has no stand-in, or UFF still lacks a parameter."""
    rw = Chem.RWMol(mol)
    sites = {}
    for metal, star, _ in eta:
        rw.RemoveBond(metal, star)
        sites[metal] = sites.get(metal, 0) + 1
    for metal in sites:
        sites[metal] += rw.GetAtomWithIdx(metal).GetDegree()
    for _, star, _ in eta:
        a = rw.GetAtomWithIdx(star)
        a.SetAtomicNum(1)
        a.SetNoImplicit(True)
    ff_mol = rw.GetMol()
    ff_mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(ff_mol)  # (its rings found again, its bonds to the stars gone)
    for atom in ff_mol.GetAtoms():
        if atom.GetSymbol() in STAND_IN and sites.get(atom.GetIdx(), atom.GetDegree()) <= 6:
            atom.SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(STAND_IN[atom.GetSymbol()]))
    typed = typed_for_uff(ff_mol, sites)
    if any(digit is None for _, digit in typed) or not rdForceFieldHelpers.UFFHasAllMoleculeParams(ff_mol):
        return None
    stars = {star for _, star, _ in eta}
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), covalent_length(b)) for b in mol.GetBonds()
             if (b.GetBeginAtom().GetSymbol() in STAND_IN or b.GetEndAtom().GetSymbol() in STAND_IN)
             and covalent_length(b) is not None]
    apart = []
    for metal, star, atoms in eta:
        for c in atoms:
            d = eta_length(mol, metal, c)
            if d is not None:
                bonds.append((metal, c, d))
            for k in (x.GetIdx() for x in mol.GetAtomWithIdx(metal).GetNeighbors()):
                if k not in stars:
                    apart.append((c, k, ETA_CLEAR, 100.0))
    # (and its systems kept from overlapping, as while embedding: the force field counts nothing between them)
    apart += [(i, j, low, 100.0) for i, j, low in eta_apart(mol, eta)]
    return uff_with_metals(mol, ids, iters, ff_mol=ff_mol, bonds=bonds, apart=apart, interfrag=bool(eta))


def typed_for_uff(mol, sites=None):
    """Each metal given the hybridization UFF types it from, as the geometry
    UFF has for it that suits its number of bonds - or of `sites`, by atom,
    where given (an eta system one site); and what each was taken as, for
    the chip: [(symbol, geometry digit or None)]."""
    said = []
    for atom in mol.GetAtoms():
        bound = (sites or {}).get(atom.GetIdx(), atom.GetDegree())
        if not is_metal(atom) or not bound:
            continue
        if bound > 6:
            # (a ring bound through each of its atoms: no geometry of UFF's fits it)
            said.append((atom.GetSymbol(), None))
            continue
        order = "3465" + "12" if bound <= 4 else "6534" + "12"
        took = None
        for digit in order:
            trial = Chem.Mol(mol)
            trial.GetAtomWithIdx(atom.GetIdx()).SetHybridization(UFF_GEOMETRY[digit])
            if rdForceFieldHelpers.GetUFFVdWParams(trial, atom.GetIdx(), atom.GetIdx()) is not None:
                took = digit
                break
        if took:
            atom.SetHybridization(UFF_GEOMETRY[took])
        said.append((atom.GetSymbol(), took))
    return said


def method_of(mol, ids, field, stood_in, metals, eta):
    """How a structure's conformers were made, in one line: embedding, the
    force field, and each thing held while it optimised that is no term of
    the force field's - so that a shape is read as what made it, a starting
    geometry, not taken for one found."""
    made = "ETKDG v3 + " + (field or ("UFF" if stood_in else "no force field"))
    held = []
    # (a metal's own bonds - not to an eta system's star, which is no atom)
    if any(is_metal(a) and any(x.GetAtomicNum() > 0 for x in a.GetNeighbors()) for a in mol.GetAtoms()):
        held.append("metal bonds at covalent lengths")
    if eta:
        held.append("η rings: metal–ring atom distances, rings ≥ %g Å apart" % ETA_APART)
    if metals and any(planar_angles(mol, cid) for cid in ids[:1]):
        held.append("square-planar angles")
    if bridges(mol):
        held.append("bridging halides' angles as embedded")
    return made + ("; restrained: " + ", ".join(held) if held else "")


def conformers_of(iso, how, made, page=False):
    """A stereoisomer's conformers. On Meno's own page (`page`), of two
    enantiomers the one whose SMILES comes first is made, and the other is
    its mirror image: the same shapes and energies, whichever is asked for
    and in whichever request. A workflow's step makes each as RDKit makes
    it. `made` keeps what has been made, by SMILES, for the rest of the
    request."""
    smiles = smiles_of(iso)
    if not page:
        if smiles not in made:
            made[smiles] = conformers(iso, **how)
        return made[smiles]
    mirror = mirror_of(iso)
    twin = smiles_of(mirror) if mirror is not None else smiles
    if twin >= smiles:
        if smiles not in made:
            made[smiles] = conformers(iso, **how)
        return made[smiles]
    if twin not in made:
        made[twin] = conformers(mirror, **how)
    entry = reflected(made[twin])
    # (and if the mirror has not given this one after all, it is made itself)
    if smiles_from_3d(iso, entry["frames"][0]) != smiles:
        if smiles not in made:
            made[smiles] = conformers(iso, **how)
        return made[smiles]
    return entry


def conformers(mol, count=None, seed=SEED, field="MMFF94", iters=STEPS, rows=()):
    """A stereoisomer's conformers, embedded (ETKDG) and optimised (MMFF94,
    MMFF94s or UFF, as asked - UFF where MMFF has no parameters) - every one
    RDKit made, as it made it, in the order it embedded them, with their
    energies; and how they were made, said as rows to show (`how`). How many
    are tried is `count`, as Meno sends it. A metal's bonds are embedded at
    their covalent lengths, its ligands at least a floor's angle apart where
    one holds (FLOORS), and a conformer that comes out narrower is dropped
    and counted. Where no force field has parameters for a metal, a lighter
    one of its group stands in (STAND_IN): its energies are given as the model's
    (`modelEnergies`), none as the molecule's; where none may, the embedded
    shapes are kept as they are, with no energies, and the rows say why. An
    eta system is held over its metal (eta_systems), its star set at its
    centre, which `stars` names; it gives the model's energies alone."""
    if not count:
        raise ValueError("How many conformers to try was not given.")
    eta = eta_systems(mol)
    mol = Chem.AddHs(mol)
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = seed
    # (nothing pruned while embedding: ETKDG's raw coordinates are not yet chairs, and two embeddings that would
    # relax into different chairs can sit close to each other there - one was thrown away before the force
    # field could tell them apart, and which of a ring's two chairs survived was arbitrary. Every one optimised
    # is given back, as it came out.)
    params.pruneRmsThresh = -1.0
    params.numThreads = 0  # (every core)
    bounds = metal_bounds(mol, eta)
    # (each distance held, embedded at its value: an acid held at its base is a part of its own, placed there, not
    #  wherever ETKDG lays a part - laid apart, a weak hold never brought them together, pyridine's BH3 17 A off)
    if rows:
        # (refused before anything is embedded: none fixed - a search moves every atom - and only where it has no
        #  metal, whose own holds are not yet combined with them)
        if any(r["kind"] == "fixed" for r in rows):
            raise ValueError("A conformer search moves every atom, so it holds no atoms fixed: take them out of its restraints, or minimise with them instead")
        if eta or any(is_metal(a) and a.GetDegree() for a in mol.GetAtoms()):
            raise ValueError("A conformer search holds restraints only where the molecule has no metal: search without them, then minimise with them")
    held = [r for r in rows if r["kind"] == "distance"]
    if held:
        bounds = held_bounds(mol, held, bounds)
        params.embedFragmentsSeparately = False
    if bounds is not None:
        params.SetBoundsMat(bounds)
    if eta:
        # (an eta system is a fragment of its own, placed by the bounds to its metal: embedded together, from
        # random coordinates, which a star at no fixed place needs)
        params.embedFragmentsSeparately = False
        params.useRandomCoords = True
    ids = list(rdDistGeom.EmbedMultipleConfs(mol, numConfs=count, params=params))
    if not ids:
        # (a crowded cage embeds from random coordinates where it does not otherwise)
        params.useRandomCoords = True
        ids = list(rdDistGeom.EmbedMultipleConfs(mol, numConfs=count, params=params))
    if not ids:
        raise ValueError("no 3D structure could be made of it")
    # (from random coordinates, where it was: said on its row)
    randomly = params.useRandomCoords
    asked = field
    metals = typed_for_uff(mol)
    eta_metals = {mol.GetAtomWithIdx(metal).GetSymbol() for metal, _, _ in eta}
    stood = None
    untyped = sorted({el for el, digit in metals if digit is None})
    if rows:
        # (its restraints held while each conformer is optimised, as Minimise with restraints holds them)
        field, make = restrained_field(mol, rows, field)
        stood_in, results = False, []
        for cid in ids:
            ff = make(cid, rows)
            more = ff.Minimize(maxIts=iters)
            places = list(ff.Positions())
            conf = mol.GetConformer(cid)
            for i in range(mol.GetNumAtoms()):
                conf.SetAtomPosition(i, Point3D(places[3 * i], places[3 * i + 1], places[3 * i + 2]))
            # (its energy the force field's own: the restraints' not counted in)
            results.append((more, make(cid, ()).CalcEnergy(places)))
    elif eta:
        # (the stars and the systems' bonds are not UFF's: held, as uff_stood_in holds them, with no energies)
        field, results = None, None
        stood = uff_stood_in(mol, ids, iters, eta)
        stood_in = stood is not None
        untyped = sorted({el for el, digit in metals if digit is None and el in STAND_IN})
    elif field in ("MMFF94", "MMFF94s") and rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol):
        stood_in = False
        results = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol, numThreads=0, maxIters=iters, mmffVariant=field)
    elif not untyped and rdForceFieldHelpers.UFFHasAllMoleculeParams(mol):
        field, stood_in = "UFF", False
        results = uff_with_metals(mol, ids, iters) if metals else rdForceFieldHelpers.UFFOptimizeMoleculeConfs(mol, numThreads=0, maxIters=iters)
    else:
        # (UFF with no terms for an atom pushed it far off, or stopped on a missing parameter: it is not used
        # as it is - a lighter metal of the group stands in where one may, and no energies are given)
        field, results = None, None
        stood = uff_stood_in(mol, ids, iters)
        stood_in = stood is not None
    energy = {cid: e for cid, (_, e) in zip(ids, results)} if results else None
    # (the energies of the model they were optimised in, where that is not the molecule - a lighter metal of the
    # group standing in, eta rings held: not the molecule's, so not given as its energies, but given as the
    # model's, and the model said in the line that says how they were made)
    model = {cid: e for cid, (_, e) in zip(ids, stood)} if stood else None
    method = method_of(mol, ids, field, stood_in, metals, eta)
    if stood:
        method += "; modelled: " + ", ".join([f"{STAND_IN[el]} for {el}" for el in untyped] + (["the \u03b7 rings held"] if eta else []))
    for _, star, atoms in eta:
        for cid in ids:
            conf = mol.GetConformer(cid)
            c = conf.GetPositions()[atoms].mean(axis=0)
            conf.SetAtomPosition(star, Point3D(float(c[0]), float(c[1]), float(c[2])))
    # (a conformer whose ligands came out closer at a metal than the floor they were held to while embedding
    # and optimising is no shape the method allows: it sat in a well of the force field the hold could not
    # pull it out of - [OsCl6]2- with Fe standing in, 87.7 kcal/mol above the octahedron, two Cl 62 degrees
    # apart. Dropped, and counted on the rows)
    floor = floor_pairs(mol)
    dropped = {}
    for cid in ids:
        under = min(((angle_at(mol, cid, m, j, k) - angle, m, angle) for m, j, k, angle in floor), default=None)
        if under and under[0] < -FLOOR_SLACK:
            dropped[cid] = under
    left = [cid for cid in ids if cid not in dropped]
    if not left:
        m, angle = next((m, angle) for _, m, angle in dropped.values())
        raise ValueError(f"every structure made of it came out with two ligands at {mol.GetAtomWithIdx(m).GetSymbol()} "
                         f"closer than the {angle:g}\u00b0 they were held apart")
    # (every conformer as RDKit made it, in the order it embedded them - none taken for another, moved, renumbered
    #  or ranked)
    kept = left
    # (how far the conformers kept missed what their restraints hold, the largest of each kind over them all)
    misses = [restraints_missed([v for p in mol.GetConformer(cid).GetPositions() for v in p], rows) for cid in kept] if rows else []
    missed = [max((m[k] for m in misses if m[k] is not None), default=None) for k in (0, 1)]
    shown = Chem.Mol(mol)
    Chem.Kekulize(shown, clearAromaticFlags=True)
    return {
        # (each charge as drawn, where RDKit held a coordination bond in its dipolar form)
        "atoms": [{"el": a.GetSymbol(), "charge": drawn_charge(a)} for a in mol.GetAtoms()],
        "bonds": [
            {"a1": b.GetBeginAtomIdx(), "a2": b.GetEndAtomIdx(), "order": kekule_order(b)}
            for b in shown.GetBonds()
        ],
        "frames": [
            [round(v, 4) for p in mol.GetConformer(cid).GetPositions() for v in p] for cid in kept
        ],
        **({"energies": [energy[cid] / KCAL_PER_HARTREE for cid in kept]} if energy else {}),
        **({"modelEnergies": [model[cid] / KCAL_PER_HARTREE for cid in kept]} if model and not energy else {}),
        **({"stars": sorted(star for _, star, _ in eta)} if eta else {}),
        "field": field or ("UFF" if stood_in else "none"),
        # (how they were made, in one line - and what was held that the force field does not say: shown with them)
        "method": method,
        # (short rows: the chip says them in a narrow card)
        "how": [
            {"label": "Embedded", "text": "ETKDG v3" + (", from random coordinates" if randomly else "")},
            {"label": "Random seed", "text": f"{seed}"},
            *([{"label": "Optimised", "text": f"{field}, at most {iters} steps"}] if field else [
                {"label": "Optimised", "text": f"UFF, at most {iters} steps"
                 + "".join(f", {STAND_IN[el]} standing in for {el}" for el in untyped)} if stood_in else
                {"label": "Optimised", "text": "not: no force field here has parameters for " + " or ".join(untyped or ["it"])},
                {"label": "Energies", "text": "none: \u03b7-bound rings are held, not modelled" if eta
                 else "none: UFF has no parameters for " + " or ".join(untyped or ["it"])},
            ]),
            *([{"label": "Asked for", "text": f"{asked}, which has no parameters for this molecule"}] if field and field != asked else []),
            *([{"label": "Held", "text": "; ".join(w for w in (restraints_words(rows), missed_words(missed)) if w)}] if rows else []),
            *([{"label": "Coordination bonds", "text": dipolar_words(mol)}] if dipolar_words(mol) else []),
            *[{"label": el, "text": ("bonds at covalent lengths, \u03b7 rings held over it" if el in eta_metals
               else f"bonds at covalent lengths, {GEOMETRY_WORDS[d]} in UFF" if d and field == "UFF" else "bonds at covalent lengths")
               + floor_words(mol, el, floor)}
              for el, d in sorted(set(metals), key=lambda x: (x[0], x[1] or ""))],
            *([{"label": "Dropped", "text": f"{len(dropped)}: " + "; ".join(
                f"two ligands at {mol.GetAtomWithIdx(m).GetSymbol()} closer than the {angle:g}\u00b0 floor held"
                f" (narrowest {angle + min(u for u, mm, _ in dropped.values() if mm == m):.0f}\u00b0)"
                for m, angle in sorted({(m, angle) for _, m, angle in dropped.values()}))}] if dropped else []),
            {"label": "Kept", "text": f"{len(kept)} of {len(ids)} embedded ({count} sought)"},
            {"label": "Made by", "text": f"RDKit {rdBase.rdkitVersion}"},
        ],
    }


def op_conformers(m):
    """Conformers of a structure drawn in 2D: of each stereoisomer `isomers`
    names, in its order - its configuration of what the drawing leaves
    open, in Meno's terms (`configured`), which Meno chose (Workspace/chem/
    openStereo) - or, none named, of the structure as drawn. Each goes back
    with the CIP label of every stereocentre and double bond (`cip`), atom
    by atom and bond by bond, and its SMILES. Each is every conformer RDKit
    made, in the order it embedded them (`conformers`). `onPage`, which
    Meno's own page sends and a workflow's step never does, has an
    enantiomer made as the mirror image of its twin (`conformers_of`); else
    each is made itself."""
    drawn = read(m["molblock"])
    as_ligand_anions(drawn, eta_systems(drawn))
    made, cache = [], {}
    how = options_of(m)
    for given in m.get("isomers") or [None]:
        iso = configured(drawn, given)
        entry = dict(conformers_of(iso, how, cache, bool(m.get("onPage"))))
        rdCIPLabeler.AssignCIPLabels(iso)
        entry["cip"] = {
            "atoms": {str(a.GetIdx()): a.GetProp("_CIPCode") for a in iso.GetAtoms() if a.HasProp("_CIPCode")},
            "bonds": {str(b.GetIdx()): b.GetProp("_CIPCode") for b in iso.GetBonds() if b.HasProp("_CIPCode")},
        }
        entry["smiles"] = Chem.MolToSmiles(iso)
        made.append(entry)
    return {"isomers": made}


def op_drawing_of(m):
    """A molecule in 3D as a formula to draw: its heavy atoms, in their
    order, laid out in 2D, wedged as its 3D structure says (for Meno's engine
    to draw afresh). Where its bonds' orders are not known - all single, as a
    file of coordinates alone gives them - they are found from where its
    atoms are, if they can be."""
    mol = Chem.MolFromMolBlock(m["molblock"], sanitize=False, removeHs=False)
    if mol is None:
        raise ValueError("This structure could not be read.")
    # (whether its bonds' orders were found: DetermineBondOrders works from
    # each atom's electrons as what is bonded to it gives them, hydrogens
    # and all, so they are sought only where the molecule has its hydrogens)
    perceived = False
    if m.get("perceive") and any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        from rdkit.Chem import rdDetermineBonds

        try:
            found = Chem.Mol(mol)
            rdDetermineBonds.DetermineBondOrders(found, charge=int(m.get("charge", 0)))
            mol = found
            perceived = True
        except Exception:
            pass  # (as the file has them, then)
    problem = Chem.SanitizeMol(mol, catchErrors=True)
    if problem != Chem.SanitizeFlags.SANITIZE_NONE:
        raise ValueError(f"cannot make sense of the structure ({problem})")
    Chem.AssignStereochemistryFrom3D(mol)
    heavy = Chem.RemoveHs(mol)
    heavy.RemoveAllConformers()
    lay_out_2d(heavy)
    return {"molblock": Chem.MolToV3KMolBlock(heavy), "perceived": perceived}


def op_orders_from_smiles(m):
    """A molecule in 3D whose bonds' orders its file does not give - a PDB
    entry's ligand, its hydrogens left out - given them from a SMILES of the
    same molecule: RDKit's AssignBondOrdersFromTemplate, matching the
    SMILES's heavy atoms to its own. Back with its atoms in their order, its
    bonds' orders (kekulized) and charges as the SMILES has them."""
    from rdkit.Chem import AllChem

    mol = Chem.MolFromMolBlock(m["molblock"], sanitize=False, removeHs=False)
    if mol is None:
        raise ValueError("This structure could not be read.")
    template = Chem.MolFromSmiles(m["smiles"])
    if template is None:
        raise ValueError("This SMILES could not be read.")
    # (the whole of it, not a part: as many heavy atoms either way)
    heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() != 1)
    if template.GetNumHeavyAtoms() != heavy:
        raise ValueError("the SMILES is not this molecule's: its atoms and bonds do not match")
    try:
        given = AllChem.AssignBondOrdersFromTemplate(template, mol)
    except Exception as e:
        raise ValueError("the SMILES is not this molecule's: its atoms and bonds do not match") from e
    implicit(given)
    Chem.Kekulize(given, clearAromaticFlags=True)
    return {"molblock": Chem.MolToV3KMolBlock(given, kekulize=True)}


def implicit(mol):
    """Each atom left to carry the hydrogens its valence leaves it, none
    taken as an unpaired electron: a file of coordinates that leaves its
    hydrogens out says nothing either way."""
    for a in mol.GetAtoms():
        a.SetNoImplicit(False)
        a.SetNumRadicalElectrons(0)
    problem = Chem.SanitizeMol(mol, catchErrors=True)
    if problem != Chem.SanitizeFlags.SANITIZE_NONE:
        raise ValueError(f"cannot make sense of the structure ({problem})")


def op_add_hydrogens(m):
    """A molecule in 3D with its bonds' orders known, given the hydrogens its
    file leaves out: RDKit's AddHs, each placed where its atom's geometry
    puts it (addCoords). How many each atom carries is `hydrogens`, by atom,
    where it is given - as the chemist chose them, a bond to a metal in
    place of a hydrogen or not - and its unpaired electrons are the file's;
    else as many as its valence leaves room for. Back with its atoms in
    their order and the hydrogens after them."""
    mol = Chem.MolFromMolBlock(m["molblock"], sanitize=False, removeHs=False)
    if mol is None:
        raise ValueError("This structure could not be read.")
    counts = m.get("hydrogens")
    if counts is None:
        implicit(mol)
    else:
        if len(counts) != mol.GetNumAtoms():
            raise ValueError("a count of hydrogens for each atom is wanted")
        for a, h in zip(mol.GetAtoms(), counts):
            a.SetNoImplicit(True)
            a.SetNumExplicitHs(int(h))
        # (each atom's valence not checked: its hydrogens are the chemist's, a dative bond to a metal among its bonds)
        problem = Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True)
        if problem != Chem.SanitizeFlags.SANITIZE_NONE:
            raise ValueError(f"cannot make sense of the structure ({problem})")
    with_h = Chem.AddHs(mol, addCoords=True)
    return {"molblock": Chem.MolToV3KMolBlock(with_h, kekulize=True)}


# --- a workflow's steps --------------------------------------------------------

#: The script a conformer search runs as a job, beside this one.
CONFORMERS_JOB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conformers_job.py")
#: What a conformer search's job writes, and is read back from.
CONFORMERS_OUT = "conformers.json"

BOND_TYPES = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC}
RADICALS = {"doublet": 1, "triplet": 2, "singlet": 2}


def held_apart(entry, rows):
    """The coordination bonds of an entry a distance restraint holds, as
    pairs of atoms: held by that restraint alone - no bond of the force
    field's - as case F's protocol holds an acid at a base (docs/PLUGINS.md)."""
    pairs = {frozenset(r["atoms"]) for r in rows if r["kind"] == "distance"}
    return {frozenset((int(b["a1"]), int(b["a2"]))) for b in entry.get("bonds") or [] if b.get("coordination") and frozenset((int(b["a1"]), int(b["a2"]))) in pairs}


def mol_of(entry, apart=()):
    """A molecule in 3D as Meno gives one: its atoms - element, place in
    angstroms, charge, radical - and its bonds by index with their orders;
    its stereo read from where its atoms are. A coordination bond in
    `apart` (`held_apart`) is left out: a restraint holds it."""
    atoms = entry.get("atoms") or []
    if not atoms:
        raise ValueError("a molecule has no atoms")
    rw = Chem.RWMol()
    for a in atoms:
        atom = Chem.Atom(str(a.get("el", "")))
        atom.SetFormalCharge(int(a.get("charge") or 0))
        if a.get("radical") in RADICALS:
            atom.SetNumRadicalElectrons(RADICALS[a["radical"]])
            atom.SetNoImplicit(True)
        rw.AddAtom(atom)
    for b in entry.get("bonds") or []:
        if b.get("coordination") and frozenset((int(b["a1"]), int(b["a2"]))) in apart:
            continue
        # (a coordination bond as RDKit's dative, from the atom that lends its pair - never a covalent single)
        kind = Chem.BondType.DATIVE if b.get("coordination") else BOND_TYPES.get(int(b.get("order", 1)), Chem.BondType.SINGLE)
        rw.AddBond(int(b["a1"]), int(b["a2"]), kind)
    mol = acceptors_as_drawn(rw.GetMol())
    conf = Chem.Conformer(len(atoms))
    for i, a in enumerate(atoms):
        conf.SetAtomPosition(i, Point3D(float(a["x"]), float(a["y"]), float(a["z"])))
    mol.AddConformer(conf, assignId=True)
    sanitized(mol)
    Chem.AssignStereochemistryFrom3D(mol)
    return mol


MINIMISE_JOB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "minimise_job.py")
#: What a restrained minimisation's job writes, for `collect` to read back.
MINIMISE_OUT = "minimised.json"

#: RDKit's torsion constraint holds by k (the angle off, in degrees) squared; the entry record's dihedral by
#: k (the angle off, in radians) squared, halved (src/lib/entry.ts `Restraint`): the record's k, times this, is RDKit's.
TORSION_K = 0.5 * (math.pi / 180.0) ** 2


def restraint_rows(rows, n):
    """An entry's restraints (the entry record's `Restraint`, docs/PLUGINS.md)
    as they are held here, each checked - or why one cannot be, in words."""
    held = []
    for k, r in enumerate(rows or []):
        kind, atoms = r.get("kind"), list(r.get("atoms") or [])
        want = {"distance": 2, "dihedral": 4}.get(kind)
        if kind not in ("distance", "dihedral", "fixed") or (want and len(atoms) != want) or not atoms:
            raise ValueError(f"Restraint {k + 1} is no distance, dihedral or atoms fixed that can be held")
        if any(not isinstance(a, int) or a < 0 or a >= n for a in atoms):
            raise ValueError(f"Restraint {k + 1} names an atom the molecule does not have")
        if kind != "fixed":
            value, strength = r.get("value"), r.get("strength")
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not isinstance(strength, (int, float)) or not strength > 0:
                raise ValueError(f"Restraint {k + 1}, a {kind}, has no value or strength to hold it by")
        held.append({"kind": kind, "atoms": atoms, "value": r.get("value"), "strength": r.get("strength")})
    return held


def minimised(mol, rows, field="MMFF94", iters=STEPS, start=None):
    """`mol` minimised in a force field with its restraints held (Minimise
    with restraints): a distance by RDKit's distance constraint, k d^2 / 2 as
    the record has it (kcal mol-1 A-2, the same k); a dihedral by RDKit's
    torsion constraint, its k the record's times TORSION_K; atoms fixed as
    fixed points. MMFF94 or MMFF94s as asked, UFF where MMFF has no
    parameters; none where UFF has none either - said. From `start`, the
    places the restraints were first held at, where given; else from where
    its atoms are. Nothing between its parts is left out. Its energy is the
    force field's, without the restraints' own. Whether it converged within
    `iters` steps is said, and how far it missed what its restraints hold
    (`restraints_missed`)."""
    conf = mol.GetConformer()
    if start is not None:
        if len(start) != 3 * mol.GetNumAtoms():
            raise ValueError("Where its restraints were first held is not the molecule's atoms")
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, Point3D(float(start[3 * i]), float(start[3 * i + 1]), float(start[3 * i + 2])))
    field, make = restrained_field(mol, rows, field)
    ff = make(-1, rows)
    converged = ff.Minimize(maxIts=iters) == 0
    places = list(ff.Positions())
    # (its energy the force field's own, at the places reached: the restraints' not counted in)
    energy = make(-1, ()).CalcEnergy(places)
    return {"frame": [round(v, 5) for v in places], "energy": energy / KCAL_PER_HARTREE, "field": field, "converged": converged, "iters": iters,
            "missed": restraints_missed(places, rows)}


def held_bounds(mol, held, bounds=None):
    """A molecule's bounds for embedding (its own, or `bounds`, a metal's
    already set) with each distance held at its value, upper and lower, and
    smoothed; or why the distances held cannot all be met together."""
    from rdkit import DistanceGeometry

    bounds = rdDistGeom.GetMoleculeBoundsMatrix(mol) if bounds is None else bounds.copy()
    for r in held:
        i, j = sorted(r["atoms"])
        bounds[i][j] = bounds[j][i] = float(r["value"])
    if not DistanceGeometry.DoTriangleSmoothing(bounds):
        raise ValueError("The distances held cannot all be met together in one structure")
    return bounds


def restraints_words(rows):
    """What restraints were held, in a row's words: "2 distances, 1 torsion"."""
    count = lambda kind: sum(1 for r in rows if r["kind"] == kind)
    said = [f"{n} {word}{'' if n == 1 else 's'}" for n, word in ((count("distance"), "distance"), (count("dihedral"), "torsion")) if n]
    fixed = sum(len(r["atoms"]) for r in rows if r["kind"] == "fixed")
    if fixed:
        said.append(f"{fixed} atom{'' if fixed == 1 else 's'} fixed")
    return ", ".join(said) + " held, the energies without them"


def dihedral_deg(places, a, b, c, d):
    """The dihedral of four atoms at `places` (x, y and z by atom in turn), in degrees from -180 to 180."""
    P = [places[3 * i:3 * i + 3] for i in (a, b, c, d)]
    sub = lambda u, v: [u[0] - v[0], u[1] - v[1], u[2] - v[2]]
    cross = lambda u, v: [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
    dot = lambda u, v: u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
    b1, b2, b3 = sub(P[1], P[0]), sub(P[2], P[1]), sub(P[3], P[2])
    n1, n2 = cross(b1, b2), cross(b2, b3)
    return math.degrees(math.atan2(math.sqrt(dot(b2, b2)) * dot(b1, n2), dot(n1, n2)))


def restraints_missed(places, rows):
    """How far `places` came from what `rows` hold: the largest miss of a
    distance, in A, and of a dihedral, in degrees the shorter way round -
    each None where none of its kind is held. Atoms fixed do not move."""
    at = lambda i: places[3 * i:3 * i + 3]
    distances = [abs(math.dist(at(r["atoms"][0]), at(r["atoms"][1])) - r["value"]) for r in rows if r["kind"] == "distance"]
    torsions = [abs((dihedral_deg(places, *r["atoms"]) - r["value"] + 180.0) % 360.0 - 180.0) for r in rows if r["kind"] == "dihedral"]
    return [max(distances) if distances else None, max(torsions) if torsions else None]


def missed_words(missed):
    """The largest misses, in a row's words: "missed by at most 0.004 Å and 1.2°"; none where nothing was measured."""
    said = [f"{missed[0]:.3f} \u00c5" if missed[0] is not None else "", f"{missed[1]:.1f}\u00b0" if missed[1] is not None else ""]
    said = [w for w in said if w]
    return "missed by at most " + " and ".join(said) if said else ""


def restrained_field(mol, rows, field="MMFF94"):
    """The force field a molecule is minimised in with restraints held - MMFF94
    or MMFF94s as asked, UFF where MMFF has no parameters, neither said where
    UFF has none - and how to make it for a conformer with restraints held
    (`make(cid, rows)`, initialised), each as the entry record states it: a
    distance by RDKit's distance constraint (k d^2 / 2, the same k), a
    dihedral by its torsion constraint (the record's k times TORSION_K),
    atoms fixed as fixed points. Nothing between the molecule's parts is
    left out."""
    if field in ("MMFF94", "MMFF94s") and rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol):
        props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol, mmffVariant=field)
        base = lambda cid: rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, props, confId=cid, ignoreInterfragInteractions=False)
        distance, torsion = "MMFFAddDistanceConstraint", "MMFFAddTorsionConstraint"
    elif rdForceFieldHelpers.UFFHasAllMoleculeParams(mol):
        field = "UFF"
        base = lambda cid: rdForceFieldHelpers.UFFGetMoleculeForceField(mol, confId=cid, ignoreInterfragInteractions=False)
        distance, torsion = "UFFAddDistanceConstraint", "UFFAddTorsionConstraint"
    else:
        raise ValueError("No force field here has parameters for it: neither MMFF94 nor UFF")

    def make(cid, held):
        ff = base(cid)
        for r in held:
            a = r["atoms"]
            if r["kind"] == "distance":
                getattr(ff, distance)(a[0], a[1], False, float(r["value"]), float(r["value"]), float(r["strength"]))
            elif r["kind"] == "dihedral":
                getattr(ff, torsion)(a[0], a[1], a[2], a[3], False, float(r["value"]), float(r["value"]), float(r["strength"]) * TORSION_K)
            else:
                for i in a:
                    ff.AddFixedPoint(i)
        ff.Initialize()
        return ff

    return field, make


def op_prepare(m):
    """A conformer search's jobs, or a restrained minimisation's: one for
    each molecule, its molecule - its restraints with it, and where they
    were first held - and the options written into its folder, run by this
    environment's Python."""
    step = m.get("step")
    if step not in ("conformers", "minimise"):
        raise ValueError(f"There is no step {step!r} to run as a job.")
    options = m.get("options") or {}
    jobs = []
    for i, entry in enumerate(m.get("entries") or []):
        rows = restraint_rows(entry.get("restraints"), len(entry.get("atoms") or []))
        mol_of(entry, held_apart(entry, rows))  # (one RDKit cannot make sense of, said now - not in its job)
        given = {"atoms": entry["atoms"], "bonds": entry.get("bonds") or [], "options": options, "count": entry.get("count")}
        if rows:
            given["restraints"] = rows
        if step == "minimise" and entry.get("start") is not None:
            given["start"] = entry["start"]
        jobs.append({
            "entries": [i],
            "program": "python",
            "args": ["-u", MINIMISE_JOB if step == "minimise" else CONFORMERS_JOB, "molecule.json"],
            "files": [{"name": "molecule.json", "text": json.dumps(given)}],
            "reads": [MINIMISE_OUT if step == "minimise" else CONFORMERS_OUT],
        })
    if not jobs:
        raise ValueError("nothing came in")
    return {"jobs": jobs}


def missed_results(missed):
    """A restrained minimisation's largest misses, as results on its molecule (the general form, docs/PLUGINS.md)."""
    d, t = missed or [None, None]
    return [
        *([{"id": "restraint-distance-missed", "on": "molecule", "group": "Restraints", "label": "Largest distance miss", "value": round(d, 4), "quantity": "length"}] if d is not None else []),
        *([{"id": "restraint-torsion-missed", "on": "molecule", "group": "Restraints", "label": "Largest torsion miss", "value": round(t, 2), "quantity": "angle"}] if t is not None else []),
    ]


def op_collect(m):
    """What a conformer search's job wrote, as Meno's output form: each
    conformer a geometry, in the order RDKit embedded them, with its
    energy. Or a restrained minimisation's: the geometry it reached, its
    energy, and whether it converged."""
    if m.get("step") not in ("conformers", "minimise"):
        raise ValueError(f"There is no step {m.get('step')!r} to run as a job.")
    if m.get("ended") != "done":
        lines = [line.strip() for line in (m.get("log") or "").splitlines() if line.strip()]
        return {"why": lines[-1] if lines else "it said nothing"}
    if m.get("step") == "minimise":
        text = (m.get("files") or {}).get(MINIMISE_OUT)
        if not text:
            raise ValueError("the minimisation wrote nothing")
        made = json.loads(text)
        return {"outputs": [{
            "schema": 1,
            "program": "RDKit",
            "version": made.get("version"),
            "method": f"{made['field']}, restrained",
            "atoms": [a["el"] for a in made["atoms"]],
            "frames": [made["frame"]],
            "energies": [made["energy"]],
            "optimised": bool(made.get("converged")),
            # (how far it missed what its restraints hold: a weak strength set by hand shows here, not silently)
            "results": missed_results(made.get("missed")),
        } for _ in m.get("entries") or []]}
    text = (m.get("files") or {}).get(CONFORMERS_OUT)
    if not text:
        raise ValueError("the search wrote no conformers")
    made = json.loads(text)
    outputs = []
    for entry in m.get("entries") or []:
        outputs.append({
            "schema": 1,
            "program": "RDKit",
            "version": made.get("version"),
            "method": f"ETKDG v3, {made.get('field', 'MMFF94')}" if made.get("field") != "none" else "ETKDG v3, not optimised",
            "atoms": [a["el"] for a in made["atoms"]],
            "frames": made["frames"],
            "energies": made.get("energies"),
            "results": [],
        })
    return {"outputs": outputs}


def op_run(m):
    """A step done at once: one of the steps on structures drawn (`STRUCTURE_STEPS`)."""
    if m.get("step") in STRUCTURE_STEPS:
        return STRUCTURE_STEPS[m["step"]](m.get("entries") or [], m.get("options") or {})
    raise ValueError(f"There is no step {m.get('step')!r} to run at once.")


# --- Steps on structures drawn ------------------------------------------------
#
# Each is given the structures of a set, each as its MOL blocks - one for
# each of its parts - and answers for each, in their order: kept as drawn
# ({"as": "drawn"}), made anew ({"made": [block, ...]}: its stereoisomers,
# drawings for Meno's engine to lay out again), or set aside, saying why
# and what kind of reason it is ({"aside": why, "kind": kind}). Nothing is
# changed in a structure kept, and nothing is chosen for one: what cannot
# be decided from the drawing is set aside.


def drawing_of_parts(blocks):
    """A structure drawn, its parts read as one molecule - unsanitised, as
    drawn, but read as every op of the worker reads a drawing (`read`): a
    coordination bond takes no hydrogen (`acceptors_as_drawn`), and one to
    a full group-13 acceptor is held in its dipolar form (`dipolar_adducts`)
    - or None where it cannot be read at all."""
    mols = [Chem.MolFromMolBlock(b, sanitize=False, removeHs=False) for b in blocks]
    if not mols or any(x is None for x in mols):
        return None
    mol = mols[0]
    for more in mols[1:]:
        mol = Chem.CombineMols(mol, more)
    acceptors_as_drawn(mol)
    dipolar_adducts(mol)
    return mol


def op_structure_facts(m):
    """What RDKit makes of each structure drawn, its parts read as one
    (`drawing_of_parts`), as facts for Meno's *Check structures* to decide
    on - none of it judged here: each atom's element, atomic number and
    charge, the valence RDKit gives it and the most it could be
    (`too_many`), its unpaired electrons, and which part of the structure it
    is in; and where RDKit cannot make sense of the whole, what its
    sanitisation said (`problem`). One it cannot read at all: `read` false.
    A lone H+ written as V2000 is not read (Meno writes V3000, which it
    is)."""
    out = []
    for e in m.get("structures") or []:
        mol = drawing_of_parts(e.get("molblocks") or [])
        if mol is None:
            out.append({"read": False})
            continue
        mol.UpdatePropertyCache(strict=False)
        part = {i: k for k, frag in enumerate(Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)) for i in frag}
        atoms = [
            {"el": a.GetSymbol(), "z": a.GetAtomicNum(), "charge": a.GetFormalCharge(), "radicals": a.GetNumRadicalElectrons(), "part": part[a.GetIdx()], **too_many(a)}
            for a in mol.GetAtoms()
        ]
        facts = {"read": True, "atoms": atoms}
        problem = Chem.SanitizeMol(Chem.Mol(mol), catchErrors=True)
        if problem != Chem.SanitizeFlags.SANITIZE_NONE:
            facts["problem"] = str(problem).split(".")[-1]
        out.append(facts)
    return {"structures": out}


#: The descriptors *Descriptors* gives each structure: its name, unit (Meno's units table, or "none"), how
#: RDKit works it out, and the function. Each of the molecule as drawn, its hydrogens counted as its atoms have
#: them.
DESCRIPTORS = [
    ("mw", "g/mol", "average molecular weight (Descriptors.MolWt)", lambda m: Descriptors.MolWt(m)),
    ("logp", "none", "Wildman-Crippen logP (Crippen.MolLogP)", lambda m: Crippen.MolLogP(m)),
    ("tpsa", "Å2", "topological polar surface area, Ertl (rdMolDescriptors.CalcTPSA)", lambda m: rdMolDescriptors.CalcTPSA(m)),
    ("hbd", "none", "hydrogen-bond donors (rdMolDescriptors.CalcNumHBD)", lambda m: rdMolDescriptors.CalcNumHBD(m)),
    ("hba", "none", "hydrogen-bond acceptors (rdMolDescriptors.CalcNumHBA)", lambda m: rdMolDescriptors.CalcNumHBA(m)),
    ("rotatable", "none", "rotatable bonds, strict (rdMolDescriptors.CalcNumRotatableBonds)", lambda m: rdMolDescriptors.CalcNumRotatableBonds(m)),
]


def step_descriptors(entries, options):
    """*Descriptors*: each structure kept as drawn, with RDKit's descriptors
    of it (`DESCRIPTORS`), each its value, unit and how it was worked out;
    set aside, saying why, where RDKit cannot make sense of it - nothing
    worked out of a structure it does not read as drawn."""
    out = []
    for e in entries:
        mol = drawing_of_parts(e.get("molblocks") or [])
        if mol is None:
            out.append({"aside": "it could not be read", "kind": "unreadable"})
            continue
        problem = Chem.SanitizeMol(mol, catchErrors=True)
        if problem != Chem.SanitizeFlags.SANITIZE_NONE:
            out.append({"aside": f"RDKit cannot make sense of it ({str(problem).split('.')[-1]})", "kind": "valence"})
            continue
        # (a lone atom or ion - Na+, Cl-, a lone metal: one atom, no bond, no hydrogen - has no logP of its own: Crippen's
        #  is a sum over atoms typed by what they are bonded to, and gives every lone ion one type's value. Left out, and
        #  said so; its weight and the rest, as for any)
        lone = mol.GetNumAtoms() == 1 and mol.GetAtomWithIdx(0).GetTotalNumHs() == 0
        given = [d for d in DESCRIPTORS if not (lone and d[0] in UNDEFINED_ALONE)]
        properties = {name: {"value": round(float(f(mol)), 4) if unit != "none" or name == "logp" else int(f(mol)), "unit": unit, "method": how} for name, unit, how, f in given}
        if lone:
            properties["not_given"] = {"value": f"{', '.join(UNDEFINED_ALONE)}: not defined for a lone ion", "unit": "none", "method": "what Descriptors leaves out of a lone atom or ion"}
        out.append({"as": "drawn", "properties": properties})
    return {"structures": out}


#: The descriptors not defined for a lone atom or ion: Wildman-Crippen logP, a sum over atoms typed by their bonds.
UNDEFINED_ALONE = ("logp",)


#: The steps on structures drawn, by kind.
STRUCTURE_STEPS = {"descriptors": step_descriptors}


OPS = {
    "prepare": op_prepare,
    "collect": op_collect,
    "run": op_run,
    "to_smiles": op_to_smiles,
    "rxn_to_smiles": op_rxn_to_smiles,
    "from_smiles": op_from_smiles,
    "smiles_read": op_smiles_read,
    "analyse": op_analyse,
    "structure_facts": op_structure_facts,
    "stereo_elements": op_stereo_elements,
    "stereoisomers": op_stereoisomers,
    "configurations": op_configurations,
    "conformers": op_conformers,
    "drawing_of": op_drawing_of,
    "orders_from_smiles": op_orders_from_smiles,
    "add_hydrogens": op_add_hydrogens,
}


def answer(line):
    try:
        m = json.loads(line)
    except ValueError:
        return {"ok": False, "error": "not JSON"}
    rid = m.get("id")
    op = OPS.get(m.get("op"))
    if op is None:
        return {"id": rid, "ok": False, "error": f"no such request: {m.get('op')}"}
    try:
        return {"id": rid, "ok": True, "result": op(m)}
    except Exception as e:  # an answer, never a crash
        return {"id": rid, "ok": False, "error": str(e) or type(e).__name__}


def main():
    print(json.dumps({"event": "ready", "version": rdBase.rdkitVersion}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if line:
            print(json.dumps(answer(line)), flush=True)


if __name__ == "__main__":
    main()
