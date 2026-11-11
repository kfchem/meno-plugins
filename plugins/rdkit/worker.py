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
# how far apart two draws' seeds lie: more than the most conformers tried, so that no draw's embeddings are
# seeded as another's are, whichever way RDKit seeds each conformer from the seed
DRAW_STRIDE = 1_000_003
STEPS = 2000
#: How many threads a conformer search may use, unless asked: 0, as RDKit takes it, is every core. At most MOST_THREADS.
THREADS = 0
MOST_THREADS = 1024
# the force fields conformers may be optimised in
FIELDS = ("MMFF94", "MMFF94s", "UFF")
#: What is done where MMFF94 or MMFF94s is asked and has no parameters for the molecule (a boron, a metal): UFF in
#: its place, said in how they were made - or the molecule refused, saying why. The chemist's choice (`fallback`),
#: UFF unless chosen otherwise.
FALLBACKS = ("UFF", "refuse")


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
    the options the manifest declares for the role (roleOptions), each as it
    may be; and the draw, where Meno asks in rounds (`draws`): its seed the
    role's, moved on DRAW_STRIDE for each draw - another set of tries for
    each, the same set for the same; and `threads`, how many threads the
    embedding and the optimising may use (0, unsaid: every core, as RDKit
    takes it - the same seed gave the same conformers and energies whichever
    it was, in the tests `HowManyTried` holds). A number that is none of these is refused
    (`whole_in`)."""
    o = m.get("options") or {}

    def number(key, label, default, low, high):
        return whole_in(o[key], label, low, high) if o.get(key) is not None else default

    count = whole_in(m["count"], "How many conformers to try", 1, MOST_CONFORMERS) if m.get("count") is not None else None
    # (the draw Meno asks for, a round of a search in rounds: its seed the role's, moved on by the draw - one that is no
    #  whole number from 0 refused, never run as another)
    draw = whole_in(m["draw"], "The draw asked for", 0, 2**31 - 1) if m.get("draw") is not None else 0
    return {
        "count": count,
        "seed": (number("seed", "Random seed", SEED, 0, 2**31 - 1) + draw * DRAW_STRIDE) % 2**31,
        "field": o.get("field") if o.get("field") in FIELDS else "MMFF94",
        "iters": number("iters", "Optimisation steps, at most", STEPS, 10, 100000),
        "threads": number("threads", "Threads, at most", THREADS, 0, MOST_THREADS),
        "fallback": o.get("fallback") if o.get("fallback") in FALLBACKS else "UFF",
    }


def refused_without_mmff(mol, field, fallback):
    """Raised where MMFF94 or MMFF94s is asked, has no parameters for `mol` - a metal, or an eta system's star,
    which MMFF types as nothing - and UFF in its place was not chosen (`fallback`, FALLBACKS)."""
    if fallback == "refuse" and field in ("MMFF94", "MMFF94s") and not rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol):
        raise ValueError(f"{field} has no parameters for it, and UFF in its place was not chosen: choose UFF, or let UFF stand in")
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


def smiles_of(mol):
    """A stereoisomer's SMILES, any hydrogens drawn left out."""
    return Chem.MolToSmiles(Chem.RemoveHs(mol))


# --- metals ------------------------------------------------------------------

# A metal's bonds, as ETKDG and UFF are given them. ETKDG's bounds for a bond
# to a metal come from radii that put Pd-Cl near 2.7 A, and its ligands may
# meet; UFF types a metal from its hybridization, which RDKit does not perceive
# for one (it said "Unrecognized hybridization" for Pd, and with no terms for
# its bonds pushed it 8 to 25 A away or crashed). So what Meno sends to hold
# (`holds`: lib/chem/metalHolds - a metal's bonds' lengths, how far apart its
# ligands are kept, its eta rings, which metals are square planar, the
# stand-ins it allows) is made RDKit's bounds and UFF's restraints here, and a
# metal typed for UFF as a geometry UFF has for it; where UFF has none,
# nothing is optimised, and the chip says so.


def holds_of(m):
    """What Meno sends to hold a structure with metals (lib/chem/metalHolds
    `MetalHolds`), as the worker uses it - or None, where it sends none."""
    h = m.get("holds")
    if not h:
        return None
    pair = lambda i, j: (min(int(i), int(j)), max(int(i), int(j)))
    geometry = {int(g["metal"]): g["as"] for g in h.get("geometry") or []}
    return {
        "slack": float(h["slack"]),
        "length": {pair(b["i"], b["j"]): float(b["length"]) for b in h.get("bonds") or []},
        "floors": [(int(f["metal"]), int(f["j"]), int(f["k"]), float(f["angle"])) for f in h.get("floors") or []],
        "floor_rule": [(int(most), float(angle)) for most, angle in h.get("floorRule") or []],
        "eta": [(int(e["metal"]), int(e["star"]), [int(c) for c in e["atoms"]]) for e in h.get("eta") or []],
        "eta_length": {(int(e["metal"]), int(c)): float(d) for e in h.get("eta") or [] for c, d in zip(e["atoms"], e["lengths"]) if d is not None},
        "eta_clear": float(h.get("etaClear", 0)),
        "eta_apart": float(h.get("etaApart", 0)),
        "eta_near": float(h.get("etaNear", 0)),
        "anions": [(int(a["atom"]), int(a["hydrogens"]), int(a["metal"])) for a in h.get("anions") or []],
        "geometry": geometry,
        "trans": tuple(h["planar"]["trans"]) if h.get("planar") else (172.0, 180.0),
        "cis": tuple(h["planar"]["cis"]) if h.get("planar") else (85.0, 95.0),
        "hydrides": {int(x["metal"]): float(x["length"]) for x in h.get("hydrides") or []},
        "bridges": [(int(b["metals"][0]), int(b["halogen"]), int(b["metals"][1])) for b in h.get("bridges") or []],
        "bridge_slack": float(h.get("bridgeSlack", 0)),
        "stand_in": {s["el"]: s["as"] for s in h.get("standIn") or []},
    }


def with_hydrides(mol, holds):
    """`holds` with each hydrogen RDKit put on a metal of `mol` (one its file gives it that is not drawn as an atom:
    a hydride) given the length Meno gives that metal's hydrides."""
    for metal, d in (holds or {}).get("hydrides", {}).items():
        for x in mol.GetAtomWithIdx(metal).GetNeighbors():
            key = (min(metal, x.GetIdx()), max(metal, x.GetIdx()))
            if x.GetAtomicNum() == 1 and key not in holds["length"]:
                holds["length"][key] = d
    return holds


def bond_length(holds, i, j):
    """The length Meno gives a metal's bond between atoms `i` and `j` (A), or None."""
    return (holds or {}).get("length", {}).get((min(i, j), max(i, j)))


#: UFF's geometries for a metal, in the digit its atom types use, and the
#: hybridization RDKit types each from: tetrahedral, square planar, octahedral,
#: then the rest - tried in this order for up to four bonds, octahedral first
#: for more; a geometry Meno gives the metal first.
UFF_GEOMETRY = {
    "3": Chem.HybridizationType.SP3,
    "4": Chem.HybridizationType.SP2D,
    "6": Chem.HybridizationType.SP3D2,
    "5": Chem.HybridizationType.SP3D,
    "1": Chem.HybridizationType.SP,
    "2": Chem.HybridizationType.SP2,
}
#: What each geometry is called on the chip, and the digit of each geometry Meno names.
GEOMETRY_WORDS = {"1": "linear", "2": "trigonal", "3": "tetrahedral", "4": "square planar", "5": "trigonal bipyramidal", "6": "octahedral"}
GEOMETRY_DIGITS = {words: digit for digit, words in GEOMETRY_WORDS.items()}


def is_metal(atom):
    """A d- or f-block atom: one ETKDG and UFF need told about."""
    z = atom.GetAtomicNum()
    return 21 <= z <= 30 or 39 <= z <= 48 or 57 <= z <= 80 or 89 <= z <= 112


def as_ligand_anions(mol, holds):
    """Each ring or chain Meno takes as the anion it is bound as (`anions`: a
    cyclopentadienide's or allyl's carbon drawn neutral, given a charge of -1
    and its hydrogens, its metal one charge higher) made so. Whether any was."""
    anions = (holds or {}).get("anions") or []
    for atom, hydrogens, metal in anions:
        c = mol.GetAtomWithIdx(atom)
        c.SetFormalCharge(-1)
        c.SetNumRadicalElectrons(0)
        c.SetNoImplicit(True)
        c.SetNumExplicitHs(hydrogens)
        m = mol.GetAtomWithIdx(metal)
        m.SetFormalCharge(m.GetFormalCharge() + 1)
    if anions:
        mol.UpdatePropertyCache(strict=False)
        problem = Chem.SanitizeMol(mol, catchErrors=True)
        if problem != Chem.SanitizeFlags.SANITIZE_NONE:
            raise ValueError(f"cannot make sense of the structure ({problem})")
    return bool(anions)


def eta_apart(holds):
    """Two eta systems of one metal kept from overlapping, as Meno keeps them: each pair of their heavy atoms at
    least `etaApart` apart, [(i, j, low)]."""
    by_metal = {}
    for metal, _, atoms in holds["eta"]:
        by_metal.setdefault(metal, []).append(atoms)
    out = []
    for systems in by_metal.values():
        for x, a in enumerate(systems):
            for b in systems[x + 1:]:
                out += [(i, j, holds["eta_apart"]) for i in a for j in b]
    return out


def metal_bounds(mol, holds):
    """ETKDG's bounds for a molecule with metals, as Meno holds them: each
    metal's bonds at the lengths it gives, within its `slack`, and the
    ligands its floors name at least their angle apart as seen from their
    metal; each eta system's atoms at their lengths from the metal, its star
    anywhere, the atoms on it let come within `etaNear` of the metal, other
    donors `etaClear` off it, two systems `etaApart`. None where it has no
    metal bonded, or the bounds cannot be made to agree."""
    from rdkit import DistanceGeometry
    metals = [a for a in mol.GetAtoms() if is_metal(a) and a.GetDegree()]
    if not metals or holds is None:
        return None
    slack = holds["slack"]
    eta = holds["eta"]
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
    for m, j, k, angle in holds["floors"]:
        floor.setdefault(m, []).append((j, k, angle))
    for m in metals:
        i = m.GetIdx()
        floor.setdefault(i, [])
        length = {}
        for bond in m.GetBonds():
            j = bond.GetOtherAtomIdx(i)
            d = bond_length(holds, i, j)
            if d is None:
                continue
            length[j] = d
            put(i, j, d - slack, d + slack)
        for j, k, angle in floor[i]:
            if j not in length or k not in length:
                continue
            cos = math.cos(math.radians(angle))
            dj, dk = length[j], length[k]
            low = math.sqrt(dj * dj + dk * dk - 2 * dj * dk * cos)
            lo, hi = min(j, k), max(j, k)
            put(j, k, max(bm[hi][lo], low), max(dj + dk + 2 * slack, low))
    # an eta system over its metal: each of its atoms at its eta length, its star anywhere (set at its
    # centre after), the atoms on it let come near the metal, and other ligands' donors kept off it
    n = mol.GetNumAtoms()
    for metal, star, atoms in eta:
        for k in range(n):
            if k != star:
                put(k, star, 0.0, 100.0)
        donors = [x.GetIdx() for x in mol.GetAtomWithIdx(metal).GetNeighbors() if x.GetIdx() not in stars]
        for c in atoms:
            d = holds["eta_length"].get((metal, c))
            if d is not None:
                put(metal, c, d - slack, d + slack)
            for x in mol.GetAtomWithIdx(c).GetNeighbors():
                k = x.GetIdx()
                if k not in atoms and k != metal and k != star:
                    lo, hi = min(k, metal), max(k, metal)
                    bm[hi][lo] = min(bm[hi][lo], holds["eta_near"])
            for k in donors:
                lo, hi = min(k, c), max(k, c)
                bm[hi][lo] = max(bm[hi][lo], holds["eta_clear"])
                bm[lo][hi] = max(bm[lo][hi], holds["eta_clear"])
    # (two eta systems of one metal kept from overlapping: eta_apart)
    for i, j, low in eta_apart(holds):
        lo, hi = min(i, j), max(i, j)
        bm[hi][lo] = max(bm[hi][lo], low)
        bm[lo][hi] = max(bm[lo][hi], low)
    if not DistanceGeometry.DoTriangleSmoothing(bm):
        return None
    return bm


def planar_angles(mol, cid, holds):
    """The angles to hold at each square-planar metal with four bonds, in one
    conformer, as Meno holds them: its two trans pairs those of its ligands
    that are furthest apart there as embedded (`trans`), the rest `cis` -
    so that cis and trans isomers both come out of a search, as embedding
    left them. A metal UFF types square planar whose four Meno says lie
    tetrahedral (d10) is held so by nothing. [(i, metal, j, (low, high))]"""
    pos = mol.GetConformer(cid).GetPositions()
    held = []
    for atom in mol.GetAtoms():
        if not is_metal(atom) or atom.GetDegree() != 4 or atom.GetHybridization() != Chem.HybridizationType.SP2D:
            continue
        c = atom.GetIdx()
        if holds["geometry"].get(c) == "tetrahedral":
            continue
        nb = [n.GetIdx() for n in atom.GetNeighbors()]

        def angle(i, j):
            u, v = pos[i] - pos[c], pos[j] - pos[c]
            cos = float((u * v).sum() / math.sqrt(float((u * u).sum() * (v * v).sum())))
            return math.degrees(math.acos(max(-1.0, min(1.0, cos))))

        pairings = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
        trans = max(pairings, key=lambda pr: sum(angle(nb[a], nb[b]) for a, b in pr))
        for x in range(4):
            for y in range(x + 1, 4):
                held.append((nb[x], c, nb[y], holds["trans"] if (x, y) in trans else holds["cis"]))
    return held


#: How strongly a stood-in metal's bonds, and any metal's ligands' distance
#: apart, are held (kcal/mol/A^2): UFF's octahedral and square-planar angle
#: terms have a minimum at 0 degrees too, and two ligands of a metal are
#: never kept apart by their van der Waals terms (they are 1-3), so without
#: this two ligands could settle 30 degrees apart.
HOLD_BOND = 500.0
HOLD_APART = 100.0


def ligands_apart(holds):
    """Each metal's ligands held at least their floor's angle apart as seen
    from it (`floors`), as metal_bounds keeps them while embedding: [(j, k,
    low, high)]."""
    held = []
    for m, j, k, angle in holds["floors"]:
        cos = math.cos(math.radians(angle))
        dj, dk = bond_length(holds, m, j), bond_length(holds, m, k)
        if dj is not None and dk is not None:
            held.append((j, k, math.sqrt(dj * dj + dk * dk - 2 * dj * dk * cos), dj + dk + 1.0))
    return held


def angle_at(mol, cid, m, j, k):
    """The angle j-m-k in conformer `cid` (degrees)."""
    p = mol.GetConformer(cid).GetPositions()
    u, v = p[j] - p[m], p[k] - p[m]
    cos = float((u * v).sum() / math.sqrt(float((u * u).sum() * (v * v).sum())))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def floor_words(mol, symbol, holds):
    """What the floor did at the metals of `symbol`, for its row: the angle held, or why none was."""
    at = {m: angle for m, _, _, angle in holds["floors"]}
    metals = [a for a in mol.GetAtoms() if a.GetSymbol() == symbol and a.GetDegree() > 1]
    if not metals:
        return ""
    held = sorted({at[a.GetIdx()] for a in metals if a.GetIdx() in at})
    if held:
        return ", ligands at least " + " or ".join(f"{x:g}°" for x in held) + " apart"
    a = metals[0]
    most = max((m for m, _ in holds["floor_rule"]), default=None)
    if most is not None and a.GetDegree() > most:
        return f", no angle held between ligands: {a.GetDegree()} atoms bonded to it"
    near = list(a.GetNeighbors())
    if any(x.GetAtomicNum() == 0 for x in near) or any(
        mol.GetBondBetweenAtoms(x.GetIdx(), y.GetIdx()) for i, x in enumerate(near) for y in near[i + 1:]
    ):
        return ", no angle held between ligands: a ligand bound through atoms side by side"
    return ""


def bridge_angles(mol, cid, holds):
    """The angles to hold at each bridging halogen in one conformer, as Meno
    holds them: as embedded there, `bridgeSlack` either way. [(i, halogen,
    j, (low, high))]"""
    pos = mol.GetConformer(cid).GetPositions()
    held = []
    slack = holds["bridge_slack"]
    for i, c, j in holds["bridges"]:
        u, v = pos[i] - pos[c], pos[j] - pos[c]
        cos = float((u * v).sum() / math.sqrt(float((u * u).sum() * (v * v).sum())))
        a = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
        held.append((i, c, j, (max(0.0, a - slack), min(180.0, a + slack))))
    return held


def uff_with_metals(mol, ids, holds, iters, ff_mol=None, bonds=(), apart=(), interfrag=False, threads=THREADS):
    """UFF's optimisation of each conformer of a molecule with metals: its
    metals' ligands kept apart, its square-planar metals' angles held
    (planar_angles), a halogen bridging two metals given UFF's linear form
    (whose angle term stays finite there: its natural angle is 180 degrees,
    and RDKit's term for an atom it does not take as linear divides by its
    sine) with its bonds held and its angle as embedded (bridge_angles),
    `bonds` held at their lengths and `apart` [(i, j, low, high)] kept so -
    done on `ff_mol`, the molecule with stand-ins, where one is given, and
    the places taken back; with `interfrag`, nothing between its fragments
    counted but these (an eta system's metal and ring are). As
    UFFOptimizeMoleculeConfs answers: (not converged, energy) each, the
    energy UFF's own, without the holding terms."""
    ff_mol = ff_mol or mol
    apart = ligands_apart(holds) + list(apart)
    bridged = holds["bridges"]
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
                d = bond_length(holds, m, x)
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

    angles = {cid: planar_angles(ff_mol, cid, holds) + bridge_angles(ff_mol, cid, holds) for cid in ids}
    more = {}
    if not any(angles.values()):
        # (the same terms for every conformer: optimised at once, on every core)
        done = rdForceFieldHelpers.OptimizeMoleculeConfs(ff_mol, held_by(ids[0]), numThreads=threads, maxIters=iters)
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


def uff_stood_in(mol, ids, holds, iters, threads=THREADS):
    """Each conformer optimised by UFF with the metals Meno lets stand in
    (`standIn`: a lighter one of the group) for those it has no type for,
    their bonds held at their own lengths; and each eta system's star taken
    out of it (a lone hydrogen with no interactions: the force field has no
    star), the system held at its eta lengths over its metal, the metal
    typed for its sites - an eta system one - and nothing between fragments
    counted but what is held. As UFFOptimizeMoleculeConfs answers, with the
    stand-ins' energies; None where a metal has no stand-in, or UFF still
    lacks a parameter."""
    eta = holds["eta"]
    stand_in = holds["stand_in"]
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
        if atom.GetSymbol() in stand_in and sites.get(atom.GetIdx(), atom.GetDegree()) <= 6:
            atom.SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(stand_in[atom.GetSymbol()]))
    typed = typed_for_uff(ff_mol, holds, sites)
    if any(digit is None for _, digit in typed) or not rdForceFieldHelpers.UFFHasAllMoleculeParams(ff_mol):
        return None
    stars = {star for _, star, _ in eta}
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), bond_length(holds, b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()
             if (b.GetBeginAtom().GetSymbol() in stand_in or b.GetEndAtom().GetSymbol() in stand_in)
             and bond_length(holds, b.GetBeginAtomIdx(), b.GetEndAtomIdx()) is not None]
    apart = []
    for metal, star, atoms in eta:
        for c in atoms:
            d = holds["eta_length"].get((metal, c))
            if d is not None:
                bonds.append((metal, c, d))
            for k in (x.GetIdx() for x in mol.GetAtomWithIdx(metal).GetNeighbors()):
                if k not in stars:
                    apart.append((c, k, holds["eta_clear"], 100.0))
    # (and its systems kept from overlapping, as while embedding: the force field counts nothing between them)
    apart += [(i, j, low, 100.0) for i, j, low in eta_apart(holds)]
    return uff_with_metals(mol, ids, holds, iters, ff_mol=ff_mol, bonds=bonds, apart=apart, interfrag=bool(eta), threads=threads)


def typed_for_uff(mol, holds, sites=None):
    """Each metal given the hybridization UFF types it from, as the geometry
    UFF has for it that suits its number of bonds - or of `sites`, by atom,
    where given (an eta system one site) - the one Meno gives it first; and
    what each was taken as, for the chip: [(symbol, geometry digit or None)]."""
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
        given = GEOMETRY_DIGITS.get(((holds or {}).get("geometry") or {}).get(atom.GetIdx(), ""))
        if given:
            order = given + order.replace(given, "")
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


def method_of(mol, ids, field, stood_in, metals, holds):
    """How a structure's conformers were made, in one line: embedding, the
    force field, and each thing held while it optimised that is no term of
    the force field's - so that a shape is read as what made it, a starting
    geometry, not taken for one found."""
    made = "ETKDG v3 + " + (field or ("UFF" if stood_in else "no force field"))
    held = []
    # (a metal's own bonds - not to an eta system's star, which is no atom)
    if any(is_metal(a) and any(x.GetAtomicNum() > 0 for x in a.GetNeighbors()) for a in mol.GetAtoms()):
        held.append("metal bonds at covalent lengths")
    if holds and holds["eta"]:
        held.append("η rings: metal–ring atom distances, rings ≥ %g Å apart" % holds["eta_apart"])
    if holds and metals and any(planar_angles(mol, cid, holds) for cid in ids[:1]):
        held.append("square-planar angles")
    if holds and holds["bridges"]:
        held.append("bridging halides' angles as embedded")
    return made + ("; restrained: " + ", ".join(held) if held else "")


def conformers_of(iso, how, made):
    """A stereoisomer's conformers, as RDKit makes them (`conformers`): each
    stereoisomer made itself, an enantiomer as much as any. `made` keeps
    what has been made, by SMILES, for the rest of the request."""
    smiles = smiles_of(iso)
    if smiles not in made:
        made[smiles] = conformers(iso, **how)
    return made[smiles]


def conformers(mol, count=None, seed=SEED, field="MMFF94", iters=STEPS, rows=(), holds=None, fallback="UFF", threads=THREADS):
    """A stereoisomer's conformers, embedded (ETKDG) and optimised (MMFF94,
    MMFF94s or UFF, as asked - UFF where MMFF has no parameters, or the
    molecule refused before anything is embedded, as `fallback` says) - every one
    RDKit made, as it made it, in the order it embedded them, with their
    energies; and how they were made, said as rows to show (`how`). How many
    are tried is `count`, as Meno sends it. A molecule with metals is held as
    Meno's `holds` say (holds_of; lib/chem/metalHolds): its metals' bonds at
    the lengths given, its ligands a floor's angle apart, its eta systems over
    their metals, each star set at its system's centre (which `stars` names);
    where no force field has parameters for a metal, a lighter one of its
    group stands in where Meno allows it: its energies are given as the
    model's (`modelEnergies`), none as the molecule's; where none may, the
    embedded shapes are kept as they are, with no energies, and the rows say
    why. Nothing made is dropped: which are kept is Meno's."""
    if not count:
        raise ValueError("How many conformers to try was not given.")
    eta = holds["eta"] if holds else []
    mol = Chem.AddHs(mol)
    holds = with_hydrides(mol, holds)
    refused_without_mmff(mol, field, fallback)
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = seed
    # (nothing pruned while embedding: ETKDG's raw coordinates are not yet chairs, and two embeddings that would
    # relax into different chairs can sit close to each other there - one was thrown away before the force
    # field could tell them apart, and which of a ring's two chairs survived was arbitrary. Every one optimised
    # is given back, as it came out.)
    params.pruneRmsThresh = -1.0
    params.numThreads = threads  # (0: every core)
    bounds = metal_bounds(mol, holds)
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
    metals = typed_for_uff(mol, holds)
    stand_in = holds["stand_in"] if holds else {}
    eta_metals = {mol.GetAtomWithIdx(metal).GetSymbol() for metal, _, _ in eta}
    stood = None
    untyped = sorted({el for el, digit in metals if digit is None})
    if rows:
        # (its restraints held while each conformer is optimised, as Minimise with restraints holds them)
        field, make = restrained_field(mol, rows, field, fallback)
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
        stood = uff_stood_in(mol, ids, holds, iters, threads)
        stood_in = stood is not None
        untyped = sorted({el for el, digit in metals if digit is None and el in stand_in})
    elif field in ("MMFF94", "MMFF94s") and rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol):
        stood_in = False
        results = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol, numThreads=threads, maxIters=iters, mmffVariant=field)
    elif not untyped and rdForceFieldHelpers.UFFHasAllMoleculeParams(mol):
        field, stood_in = "UFF", False
        results = uff_with_metals(mol, ids, holds, iters, threads=threads) if metals and holds else rdForceFieldHelpers.UFFOptimizeMoleculeConfs(mol, numThreads=threads, maxIters=iters)
    else:
        # (UFF with no terms for an atom pushed it far off, or stopped on a missing parameter: it is not used
        # as it is - a lighter metal of the group stands in where one may, and no energies are given)
        field, results = None, None
        stood = uff_stood_in(mol, ids, holds, iters, threads) if holds else None
        stood_in = stood is not None
    energy = {cid: e for cid, (_, e) in zip(ids, results)} if results else None
    # (the energies of the model they were optimised in, where that is not the molecule - a lighter metal of the
    # group standing in, eta rings held: not the molecule's, so not given as its energies, but given as the
    # model's, and the model said in the line that says how they were made)
    model = {cid: e for cid, (_, e) in zip(ids, stood)} if stood else None
    method = method_of(mol, ids, field, stood_in, metals, holds)
    if stood:
        method += "; modelled: " + ", ".join([f"{stand_in[el]} for {el}" for el in untyped] + (["the \u03b7 rings held"] if eta else []))
    for _, star, atoms in eta:
        for cid in ids:
            conf = mol.GetConformer(cid)
            c = conf.GetPositions()[atoms].mean(axis=0)
            conf.SetAtomPosition(star, Point3D(float(c[0]), float(c[1]), float(c[2])))
    # (every conformer as RDKit made it, in the order it embedded them - none taken for another, moved, renumbered
    #  or ranked)
    kept = ids
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
                 + "".join(f", {stand_in[el]} standing in for {el}" for el in untyped)} if stood_in else
                {"label": "Optimised", "text": "not: no force field here has parameters for " + " or ".join(untyped or ["it"])},
                {"label": "Energies", "text": "none: \u03b7-bound rings are held, not modelled" if eta
                 else "none: UFF has no parameters for " + " or ".join(untyped or ["it"])},
            ]),
            *([{"label": "Asked for", "text": f"{asked}, which has no parameters for this molecule"}] if field and field != asked else []),
            *([{"label": "Held", "text": "; ".join(w for w in (restraints_words(rows), missed_words(missed)) if w)}] if rows else []),
            *([{"label": "Coordination bonds", "text": dipolar_words(mol)}] if dipolar_words(mol) else []),
            *[{"label": el, "text": ("bonds at covalent lengths, \u03b7 rings held over it" if el in eta_metals
               else f"bonds at covalent lengths, {GEOMETRY_WORDS[d]} in UFF" if d and field == "UFF" else "bonds at covalent lengths")
               + (floor_words(mol, el, holds) if holds else "")}
              for el, d in sorted(set(metals), key=lambda x: (x[0], x[1] or ""))],
            # (all of them, as made: which are kept is Meno's - lib/chem/metalHolds `keptMade` - who says so here)
            {"label": "Kept", "text": f"{len(ids)} of {len(ids)} embedded ({count} sought)"},
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
    made, in the order it embedded them (`conformers`), each made itself."""
    drawn = read(m["molblock"])
    holds = holds_of(m)
    as_ligand_anions(drawn, holds)
    made, cache = [], {}
    how = {**options_of(m), "holds": holds}
    for given in m.get("isomers") or [None]:
        iso = configured(drawn, given)
        entry = dict(conformers_of(iso, how, cache))
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
        # (held exactly, by a program that constrains: never held here by some strength in its place)
        if kind != "fixed" and r.get("exact"):
            raise ValueError(f"Restraint {k + 1}, a {kind}, is to be held exactly, and this plugin holds one only by a strength")
        if kind != "fixed":
            value, strength = r.get("value"), r.get("strength")
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not isinstance(strength, (int, float)) or not strength > 0:
                raise ValueError(f"Restraint {k + 1}, a {kind}, has no value or strength to hold it by")
        held.append({"kind": kind, "atoms": atoms, "value": r.get("value"), "strength": r.get("strength")})
    return held


def minimised(mol, rows, field="MMFF94", iters=STEPS, start=None, fallback="UFF"):
    """`mol` minimised in a force field with its restraints held (Minimise
    with restraints): a distance by RDKit's distance constraint, k d^2 / 2 as
    the record has it (kcal mol-1 A-2, the same k); a dihedral by RDKit's
    torsion constraint, its k the record's times TORSION_K; atoms fixed as
    fixed points. MMFF94 or MMFF94s as asked, UFF where MMFF has no
    parameters (or refused, as `fallback` says); none where UFF has none
    either - said. From `start`, the
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
    field, make = restrained_field(mol, rows, field, fallback)
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


def restrained_field(mol, rows, field="MMFF94", fallback="UFF"):
    """The force field a molecule is minimised in with restraints held - MMFF94
    or MMFF94s as asked, UFF where MMFF has no parameters (or refused, as
    `fallback` says: FALLBACKS), neither said where UFF has none - and how to
    make it for a conformer with restraints held
    (`make(cid, rows)`, initialised), each as the entry record states it: a
    distance by RDKit's distance constraint (k d^2 / 2, the same k), a
    dihedral by its torsion constraint (the record's k times TORSION_K),
    atoms fixed as fixed points. Nothing between the molecule's parts is
    left out."""
    refused_without_mmff(mol, field, fallback)
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
        if entry.get("holds"):
            given["holds"] = entry["holds"]
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
