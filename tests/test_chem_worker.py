"""Tests for the RDKit plugin's worker (plugins/rdkit/worker.py).

They need RDKit, so they run in an environment built from the chem lock:

    uv venv .venv-chem --python 3.12
    uv pip install --python .venv-chem/bin/python --require-hashes --no-deps \
        -r plugins/rdkit/requirements.lock
    .venv-chem/bin/python -m unittest tests/test_chem_worker.py
"""

import importlib.util
import json
import pathlib
import time
import unittest

from rdkit import Chem

WORKER = pathlib.Path(__file__).resolve().parents[1] / "plugins/rdkit/worker.py"
spec = importlib.util.spec_from_file_location("chem_worker", WORKER)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)

# 4-chlorophenylalanine as the editor writes it: the amino group on a wedge
# from the alpha carbon (atom 8), a ring of alternating bonds.
PCPA = """
  Meno

 13 13  0  0  0  0  0  0  0  0999 V2000
    0.0000    1.5000    0.0000 C   0  0
   -1.2990    0.7500    0.0000 C   0  0
   -1.2990   -0.7500    0.0000 C   0  0
    0.0000   -1.5000    0.0000 C   0  0
    1.2990   -0.7500    0.0000 C   0  0
    1.2990    0.7500    0.0000 C   0  0
    2.5981    1.5000    0.0000 C   0  0
    3.8971    0.7500    0.0000 C   0  0
    5.1962    1.5000    0.0000 C   0  0
    5.1962    3.0000    0.0000 O   0  0
    6.4952    0.7500    0.0000 O   0  0
    3.8971   -0.7500    0.0000 N   0  0
   -2.5981   -1.5000    0.0000 Cl  0  0
  1  2  2  0
  2  3  1  0
  3  4  2  0
  4  5  1  0
  5  6  2  0
  6  1  1  0
  6  7  1  0
  7  8  1  0
  8  9  1  0
  9 10  2  0
  9 11  1  0
  8 12  1  1
  3 13  1  0
M  END
"""


def ask(op, **args):
    return worker.answer(json.dumps({"id": 1, "op": op, **args}))


def v3000(atoms, bonds):
    """A V3000 block, for more atoms than V2000 counts: as v2000 takes them."""
    stereo = {None: 0, "up": 1, "down": 3}
    lines = ["", "  test", "", "  0  0  0     0  0            999 V3000",
             "M  V30 BEGIN CTAB", f"M  V30 COUNTS {len(atoms)} {len(bonds)} 0 0 0", "M  V30 BEGIN ATOM"]
    for i, (el, x, y) in enumerate(atoms):
        lines.append(f"M  V30 {i + 1} {el} {x:.4f} {y:.4f} 0 0")
    lines.append("M  V30 END ATOM")
    lines.append("M  V30 BEGIN BOND")
    for i, (a, b, order, s) in enumerate(bonds):
        lines.append(f"M  V30 {i + 1} {order} {a + 1} {b + 1}" + (f" CFG={stereo[s]}" if s else ""))
    lines += ["M  V30 END BOND", "M  V30 END CTAB", "M  END"]
    return "\n".join(lines) + "\n"


def v3000_3d(atoms, bonds):
    """A V3000 block in 3D, as Meno writes one past 999 atoms (molWriter
    writeMolfile3d): atoms as (element, x, y, z), bonds as (first, second,
    order), 0-based."""
    lines = ["", "  Meno              3D", "", "  0  0  0     0  0  0  0  0  0999 V3000",
             "M  V30 BEGIN CTAB", f"M  V30 COUNTS {len(atoms)} {len(bonds)} 0 0 0", "M  V30 BEGIN ATOM"]
    for i, (el, x, y, z) in enumerate(atoms):
        lines.append(f"M  V30 {i + 1} {el} {x:.4f} {y:.4f} {z:.4f} 0")
    lines += ["M  V30 END ATOM", "M  V30 BEGIN BOND"]
    for i, (a, b, order) in enumerate(bonds):
        lines.append(f"M  V30 {i + 1} {order} {a + 1} {b + 1}")
    lines += ["M  V30 END BOND", "M  V30 END CTAB", "M  END"]
    return "\n".join(lines) + "\n"


def v2000(atoms, bonds):
    """A V2000 block: atoms as (element, x, y), bonds as (first, second,
    order, "up" / "down" / None), 0-based, a wedge narrow at `first`."""
    lines = ["", "  test", "", f"{len(atoms):3d}{len(bonds):3d}  0  0  0  0  0  0  0  0999 V2000"]
    for el, x, y in atoms:
        lines.append(f"{x:10.4f}{y:10.4f}{0:10.4f} {el:<3} 0  0")
    stereo = {None: 0, "up": 1, "down": 6}
    for a, b, order, s in bonds:
        lines.append(f"{a + 1:3d}{b + 1:3d}{order:3d}{stereo[s]:3d}")
    return "\n".join(lines + ["M  END"]) + "\n"


def smiles_from_3d(mol, frame):
    """The SMILES a structure's 3D coordinates (its atoms', hydrogens and all, as a flat list) say, stereo and all."""
    from rdkit.Geometry import Point3D

    m = Chem.AddHs(mol)
    conf = Chem.Conformer(m.GetNumAtoms())
    for i in range(m.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(frame[3 * i], frame[3 * i + 1], frame[3 * i + 2]))
    m.RemoveAllConformers()
    m.AddConformer(conf, assignId=True)
    Chem.AssignStereochemistryFrom3D(m)
    return Chem.MolToSmiles(Chem.RemoveHs(m))


class ChemWorkerTest(unittest.TestCase):
    def test_writes_a_canonical_smiles_with_its_stereo(self):
        r = ask("to_smiles", molblock=PCPA)
        self.assertTrue(r["ok"])
        self.assertEqual(r["result"]["smiles"], "N[C@@H](Cc1ccc(Cl)cc1)C(=O)O")

    def test_writes_a_reaction_smiles_reactants_agents_products(self):
        # (an RXN file as Meno writes one - "rrrpppaaa", each molecule a $MOL - acetic acid and ethanol to ethyl
        # acetate and water over sulfuric acid; the (R)-CHFClBr wedge as a reactant keeps its centre)
        def rxn(reactants, products, agents):
            head = ["$RXN", "", "", "", f"{len(reactants):3d}{len(products):3d}{len(agents):3d}"]
            return "\n".join(head) + "\n" + "".join("$MOL\n" + b for b in reactants + products + agents)
        acid = v2000([("C", 0, 0), ("C", 1.3, 0.75), ("O", 2.6, 0), ("O", 1.3, 2.25)], [(0, 1, 1, None), (1, 2, 1, None), (1, 3, 2, None)])
        ethanol = v2000([("C", 0, 0), ("C", 1.3, 0.75), ("O", 2.6, 0)], [(0, 1, 1, None), (1, 2, 1, None)])
        ester = v2000([("C", 0, 0), ("C", 1.3, 0.75), ("O", 1.3, 2.25), ("O", 2.6, 0), ("C", 3.9, 0.75), ("C", 5.2, 0)], [(0, 1, 1, None), (1, 2, 2, None), (1, 3, 1, None), (3, 4, 1, None), (4, 5, 1, None)])
        water = v2000([("O", 0, 0)], [])
        acid_s = v2000([("S", 0, 0), ("O", 1.4, 0), ("O", -1.4, 0), ("O", 0, 1.4), ("O", 0, -1.4)], [(0, 1, 1, None), (0, 2, 1, None), (0, 3, 2, None), (0, 4, 2, None)])
        r = ask("rxn_to_smiles", rxnblock=rxn([acid, ethanol], [ester, water], [acid_s]))
        self.assertTrue(r["ok"])
        self.assertEqual(r["result"]["smiles"], "CC(=O)O.CCO>O=S(=O)(O)O>CCOC(C)=O.O")
        chfclbr = v2000(
            [("C", 0, 0), ("F", 0, 1.5), ("Cl", 1.3, -0.75), ("Br", -1.3, -0.75), ("H", 0.4, -1.4)],
            [(0, 1, 1, None), (0, 2, 1, None), (0, 3, 1, None), (0, 4, 1, "up")],
        )
        smiles = ask("rxn_to_smiles", rxnblock=rxn([chfclbr], [water], []))["result"]["smiles"]
        self.assertEqual(smiles.split(">>")[0], ask("to_smiles", molblock=chfclbr)["result"]["smiles"])
        self.assertIn("@", smiles)
        # (H2 as a reactant - its two hydrogens drawn, a molecule of its own, not taken away as hydrogens are -
        # and a lone metal as the agent: ethene hydrogenated over palladium)
        ethene = v2000([("C", 0, 0), ("C", 1.3, 0)], [(0, 1, 2, None)])
        h2 = v2000([("H", 0, 0), ("H", 0.74, 0)], [(0, 1, 1, None)])
        ethane = v2000([("C", 0, 0), ("C", 1.5, 0)], [(0, 1, 1, None)])
        palladium = v2000([("Pd", 0, 0)], [])
        self.assertEqual(ask("rxn_to_smiles", rxnblock=rxn([ethene, h2], [ethane], [palladium]))["result"]["smiles"], "C=C.[H][H]>[Pd]>CC")

    def test_writes_a_smiles_without_an_h_drawn_to_carry_a_wedge(self):
        # (R)-CHFClBr's H drawn, on a wedge
        block = v2000(
            [("C", 0, 0), ("F", 0, 1.5), ("Cl", 1.3, -0.75), ("Br", -1.3, -0.75), ("H", 0.4, -1.4)],
            [(0, 1, 1, None), (0, 2, 1, None), (0, 3, 1, None), (0, 4, 1, "up")],
        )
        smiles = ask("to_smiles", molblock=block)["result"]["smiles"]
        self.assertNotIn("[H]", smiles)
        self.assertIn("@", smiles)

    def test_reads_a_smiles_into_a_v3000_block_with_2d_coordinates(self):
        r = ask("from_smiles", smiles="C[C@H](N)C(=O)O")
        block = r["result"]["molblock"]
        self.assertIn("V3000", block)
        # and back to the same molecule
        self.assertEqual(ask("to_smiles", molblock=block)["result"]["smiles"], "C[C@H](N)C(=O)O")

    def test_analyses_hydrogens_aromaticity_and_stereo(self):
        r = ask("analyse", molblock=PCPA)["result"]
        atoms = r["atoms"]
        self.assertEqual(atoms[7].get("cip"), "S")  # L-amino acid
        self.assertEqual(atoms[11]["hydrogens"], 2)  # NH2
        self.assertTrue(all(atoms[i].get("aromatic") for i in range(6)))
        self.assertNotIn("valenceError", atoms[8])

    def test_names_the_atoms_whose_valence_is_wrong(self):
        bad = PCPA.replace("  9 10  2  0", "  9 10  3  0")  # C#O on a carboxyl carbon
        r = ask("analyse", molblock=bad)["result"]
        self.assertEqual(r["atoms"][8]["valenceError"], {"valence": 5, "most": 4})

    def test_labels_one_fragment_when_another_makes_no_sense(self):
        # (S)-CHFClBr beside a carbon with five bonds
        block = v2000(
            [("C", 0, 0), ("F", 0, 1.5), ("Cl", 1.3, -0.75), ("Br", -1.3, -0.75),
             ("C", 10, 0), ("C", 11.5, 0), ("C", 8.5, 0), ("C", 10, 1.5), ("C", 10, -1.5), ("C", 11, 1)],
            [(0, 1, 1, "up"), (0, 2, 1, None), (0, 3, 1, None)]
            + [(4, j, 1, None) for j in range(5, 10)],
        )
        r = ask("analyse", molblock=block)["result"]
        self.assertEqual(r["atoms"][0].get("cip"), "S")
        self.assertIn("valenceError", r["atoms"][4])

    def test_analyses_a_page_of_many_records_as_each_alone_and_quickly(self):
        # PCPA and trans-but-2-ene, 400 of each side by side - and the same
        # page with a carbon of five bonds on it, made sense of fragment by
        # fragment: every record labelled as it is alone, either way
        trans = (
            [("C", 0, 0), ("C", 1.3, 0.75), ("C", 2.6, 0), ("C", 3.9, 0.75)],
            [(0, 1, 1, None), (1, 2, 2, None), (2, 3, 1, None)],
        )
        pcpa = worker.read(PCPA, sanitize=False)
        records = [
            ([(a.GetSymbol(), pcpa.GetConformer().GetAtomPosition(a.GetIdx()).x,
               pcpa.GetConformer().GetAtomPosition(a.GetIdx()).y) for a in pcpa.GetAtoms()],
             [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), int(b.GetBondTypeAsDouble()),
               "up" if b.GetBondDir() == worker.Chem.BondDir.BEGINWEDGE else None) for b in pcpa.GetBonds()]),
            trans,
        ]
        alone = [ask("analyse", molblock=v2000(*r))["result"] for r in records]
        five = ([("C", 0, 0), ("C", 1.5, 0), ("C", -1.5, 0), ("C", 0, 1.5), ("C", 0, -1.5), ("C", 1, 1)],
                [(0, j, 1, None) for j in range(1, 6)])

        def page(copies, extra=None):
            atoms, bonds = [], []
            for k in range(copies):
                for r in records + ([extra] if extra and k == 0 else []):
                    first = len(atoms)
                    atoms += [(el, x + 20 * k, y) for el, x, y in r[0]]
                    bonds += [(a + first, b + first, o, s) for a, b, o, s in r[1]]
            return v2000(atoms, bonds) if len(atoms) < 1000 else v3000(atoms, bonds)

        for extra in (None, five):
            started = time.time()
            r = ask("analyse", molblock=page(400, extra))["result"]
            if extra is None:
                self.assertLess(time.time() - started, 5)
            at = 0
            for k in range(400):
                for one in alone:
                    n = len(one["atoms"])
                    got = [{**a, "index": a["index"] - at} for a in r["atoms"][at:at + n]]
                    self.assertEqual(got, one["atoms"])
                    at += n
                if extra and k == 0:
                    self.assertIn("valenceError", r["atoms"][at])
                    at += len(five[0])
            self.assertEqual(at, len(r["atoms"]))

    def test_labels_double_bonds_e_and_z(self):
        trans = v2000(
            [("C", 0, 0), ("C", 1.3, 0.75), ("C", 2.6, 0), ("C", 3.9, 0.75)],
            [(0, 1, 1, None), (1, 2, 2, None), (2, 3, 1, None)],
        )
        self.assertEqual(ask("analyse", molblock=trans)["result"]["bonds"][1]["cip"], "E")

    def test_reads_an_unknown_atom_as_any_atom(self):
        # what the editor writes for a label that is not an element
        block = v2000([("C", 0, 0), ("*", 1.5, 0)], [(0, 1, 1, None)])
        self.assertEqual(ask("to_smiles", molblock=block)["result"]["smiles"], "*C")

    def test_gives_the_elements_that_could_be_stereogenic_and_how_each_is_drawn(self):
        # PCPA's centre is drawn on a wedge: given
        self.assertTrue(all(e["state"] == "specified" for e in ask("stereo_elements", molblock=PCPA)["result"]["elements"]))
        # butan-2-ol drawn flat: its carbinol carbon not given - which is open, Meno says (lib/chem/stereoOpen)
        flat = v2000(
            [("C", 0, 0), ("C", 1.3, 0.75), ("C", 2.6, 0), ("C", 3.9, 0.75), ("O", 1.3, 2.25)],
            [(0, 1, 1, None), (1, 2, 1, None), (2, 3, 1, None), (1, 4, 1, None)],
        )
        self.assertEqual(ask("stereo_elements", molblock=flat)["result"], {"read": True, "elements": [{"kind": "centre", "at": 1, "state": "unspecified"}]})
        made = ask("stereoisomers", molblock=flat, open={"atoms": [1], "bonds": []}, most=32, inThreeD=True)["result"]
        self.assertEqual((len(made["isomers"]), made["free"]), (2, False))

    def test_makes_conformers_as_its_options_say_and_says_how(self):
        # (every one RDKit made: which are the same is Meno's to say)
        made = ask("conformers", molblock=PCPA, count=5, options={"field": "MMFF94s", "iters": 300, "seed": 7})["result"]["isomers"][0]
        self.assertEqual(made["field"], "MMFF94s")
        how = {r["label"]: r["text"] for r in made["how"]}
        self.assertEqual(how["Embedded"], "ETKDG v3")
        self.assertEqual(how["Random seed"], "7")
        self.assertEqual(how["Optimised"], "MMFF94s, at most 300 steps")
        self.assertEqual(how["Kept"], "5 of 5 embedded (5 sought)")
        self.assertTrue(how["Made by"].startswith("RDKit "))
        self.assertEqual(len(made["frames"]), 5)
        raw = ask("conformers", molblock=PCPA, count=5, options={"field": "MMFF94s", "iters": 300, "seed": 7})["result"]["isomers"][0]
        self.assertEqual(raw["frames"], made["frames"])
        # (a force field it does not know: the default; a number out of its range: refused, saying so, never moved into it)
        odd = ask("conformers", molblock=PCPA, count=2, options={"field": "GAFF"})["result"]["isomers"][0]
        self.assertEqual(odd["field"], "MMFF94")
        refused = ask("conformers", molblock=PCPA, count=2, options={"iters": 1})
        self.assertEqual((refused["ok"], refused["error"]), (False, "Optimisation steps, at most is to be from 10 to 100000: 1 is not"))

    def test_makes_conformers_keeping_the_atoms_and_the_drawn_wedge(self):
        made = ask("conformers", molblock=PCPA, count=8)["result"]["isomers"]
        self.assertEqual(len(made), 1)
        mol = made[0]
        # the drawn atoms first, in their order, the hydrogens made for them after
        self.assertEqual([a["el"] for a in mol["atoms"][:13]], ["C"] * 9 + ["O", "O", "N", "Cl"])
        self.assertTrue(all(a["el"] == "H" for a in mol["atoms"][13:]))
        self.assertEqual(len(mol["atoms"]), 13 + 10)
        # every frame with every atom, each its energy
        self.assertTrue(all(len(f) == 3 * len(mol["atoms"]) for f in mol["frames"]))
        self.assertEqual(len(mol["frames"]), len(mol["energies"]))
        # the wedge's configuration, read back from the 3D structure
        from rdkit import Chem

        built = Chem.RWMol()
        for a in mol["atoms"]:
            built.AddAtom(Chem.Atom(a["el"]))
        for b in mol["bonds"]:
            built.AddBond(b["a1"], b["a2"], {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE}[b["order"]])
        conf = Chem.Conformer(len(mol["atoms"]))
        xyz = mol["frames"][0]
        for i in range(len(mol["atoms"])):
            conf.SetAtomPosition(i, (xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]))
        built = built.GetMol()
        Chem.SanitizeMol(built)
        built.AddConformer(conf)
        Chem.AssignStereochemistryFrom3D(built)
        drawn = ask("analyse", molblock=PCPA)["result"]["atoms"][7]["cip"]
        self.assertEqual(Chem.FindMolChiralCenters(built)[0], (7, drawn))

    def test_makes_each_stereoisomer_it_is_given(self):
        # 3-aminobutan-2-ol drawn flat: two open centres, four isomers
        flat = v2000(
            [("C", 0, 0), ("C", 1.3, 0.75), ("C", 2.6, 0), ("C", 3.9, 0.75), ("O", 1.3, 2.25), ("N", 2.6, -1.5)],
            [(0, 1, 1, None), (1, 2, 1, None), (2, 3, 1, None), (1, 4, 1, None), (2, 5, 1, None)],
        )
        isomers = [i["configured"] for i in ask("stereoisomers", molblock=flat, open={"atoms": [1, 2], "bonds": []}, most=32, inThreeD=True)["result"]["isomers"]]
        one = ask("conformers", molblock=flat, count=4)["result"]["isomers"]
        every = ask("conformers", molblock=flat, count=4, isomers=isomers)["result"]["isomers"]
        self.assertEqual(len(one), 1)
        self.assertEqual(len(every), 4)
        labels = {tuple(sorted(i["cip"]["atoms"].items())) for i in every}
        self.assertEqual(len(labels), 4)
        self.assertTrue(all(set(i["cip"]["atoms"]) == {"1", "2"} for i in every))
        drawn = ask("conformers", molblock=PCPA, count=2)["result"]["isomers"][0]
        self.assertEqual(drawn["cip"]["atoms"], {"7": ask("analyse", molblock=PCPA)["result"]["atoms"][7]["cip"]})

    def test_makes_each_stereoisomer_itself_an_enantiomer_as_much_as_any(self):
        # 3-aminobutan-2-ol's four isomers - two pairs of enantiomers - each made itself, its conformers in its own
        # configuration (a mirror image, where one is wanted, is Meno's to make)
        flat = v2000(
            [("C", 0, 0), ("C", 1.3, 0.75), ("C", 2.6, 0), ("C", 3.9, 0.75), ("O", 1.3, 2.25), ("N", 2.6, -1.5)],
            [(0, 1, 1, None), (1, 2, 1, None), (2, 3, 1, None), (1, 4, 1, None), (2, 5, 1, None)],
        )
        isomers = [i["configured"] for i in ask("stereoisomers", molblock=flat, open={"atoms": [1, 2], "bonds": []}, most=32, inThreeD=True)["result"]["isomers"]]
        every = ask("conformers", molblock=flat, count=6, isomers=isomers)["result"]["isomers"]
        self.assertEqual(len({tuple(sorted(i["cip"]["atoms"].items())) for i in every}), 4)
        for iso in every:
            # (its first conformer read as it lies, its atoms and bonds as they came back)
            built = Chem.RWMol()
            for a in iso["atoms"]:
                built.AddAtom(Chem.Atom(a["el"]))
            for b in iso["bonds"]:
                built.AddBond(b["a1"], b["a2"], Chem.BondType.SINGLE)
            built = built.GetMol()
            Chem.SanitizeMol(built)
            conf = Chem.Conformer(built.GetNumAtoms())
            for i in range(built.GetNumAtoms()):
                conf.SetAtomPosition(i, tuple(iso["frames"][0][3 * i:3 * i + 3]))
            built.AddConformer(conf)
            Chem.AssignStereochemistryFrom3D(built)
            self.assertEqual(Chem.MolToSmiles(Chem.RemoveHs(built)), Chem.MolToSmiles(Chem.MolFromSmiles(iso["smiles"])))

    def test_makes_a_centre_as_its_configuration_says_read_from_a_molecule_made_before(self):
        # butan-2-ol drawn flat, made in 3D as each isomer; its centre's configuration read from those places, as Meno
        # reads it (lib/chem/stereoOpen `configuredFrom`: its neighbours in its bonds' order, the sign of their
        # volume), made again - from the same drawing, or from one with a chlorine added - as it was
        flat = v2000(
            [("C", 0, 0), ("C", 1.3, 0.75), ("C", 2.6, 0), ("C", 3.9, 0.75), ("O", 1.3, 2.25)],
            [(0, 1, 1, None), (1, 2, 1, None), (2, 3, 1, None), (1, 4, 1, None)],
        )
        grown = v2000(
            [("C", 0, 0), ("C", 1.3, 0.75), ("C", 2.6, 0), ("C", 3.9, 0.75), ("O", 1.3, 2.25), ("Cl", 5.2, 0)],
            [(0, 1, 1, None), (1, 2, 1, None), (2, 3, 1, None), (1, 4, 1, None), (3, 5, 1, None)],
        )

        def volume(frame, centre, a, b, c):
            # (the sign says which way round a, b and c go about the centre)
            p = lambda i: frame[3 * i : 3 * i + 3]
            o = p(centre)
            u, v, w = ([p(i)[k] - o[k] for k in range(3)] for i in (a, b, c))
            return u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0]) + u[2] * (v[0] * w[1] - v[1] * w[0])

        isomers = [i["configured"] for i in ask("stereoisomers", molblock=flat, open={"atoms": [1], "bonds": []}, most=32, inThreeD=True)["result"]["isomers"]]
        for iso in ask("conformers", molblock=flat, count=4, isomers=isomers)["result"]["isomers"]:
            f = iso["frames"][0]
            given = {"centres": [{"atom": 1, "neighbours": [0, 2, 4], "sense": 1 if volume(f, 1, 0, 2, 4) > 0 else -1}], "doubles": []}
            (again,) = ask("conformers", molblock=flat, count=4, isomers=[given])["result"]["isomers"]
            self.assertEqual(again["cip"], iso["cip"])
            (more,) = ask("conformers", molblock=grown, count=4, isomers=[given])["result"]["isomers"]
            self.assertGreater(volume(f, 1, 0, 2, 4) * volume(more["frames"][0], 1, 0, 2, 4), 0)
        # (its neighbours named in another order: the same centre, the same configuration)
        f = ask("conformers", molblock=flat, count=2, isomers=isomers[:1])["result"]["isomers"][0]
        turned = {"centres": [{"atom": 1, "neighbours": [2, 0, 4], "sense": 1 if volume(f["frames"][0], 1, 2, 0, 4) > 0 else -1}], "doubles": []}
        self.assertEqual(ask("conformers", molblock=flat, count=2, isomers=[turned])["result"]["isomers"][0]["cip"], f["cip"])

    def test_makes_a_double_bond_cis_or_trans_as_its_configuration_says(self):
        import numpy as np
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdCIPLabeler

        open_ = Chem.MolFromSmiles("CC=CCC")
        for smiles, label in (("C/C=C/CC", "E"), ("C/C=C\\CC", "Z")):
            made = Chem.AddHs(Chem.MolFromSmiles(smiles))
            AllChem.EmbedMolecule(made, randomSeed=7)
            xyz = made.GetConformer().GetPositions()
            axis = xyz[2] - xyz[1]
            cis = float(np.dot(np.cross(axis, xyz[0] - xyz[1]), np.cross(axis, xyz[3] - xyz[2]))) > 0
            out = worker.configured(open_, {"doubles": [{"bond": 1, "ends": [0, 3], "cis": cis}]})
            rdCIPLabeler.AssignCIPLabels(out)
            self.assertEqual(out.GetBondWithIdx(1).GetProp("_CIPCode"), label)
            # (and made so in 3D)
            self.assertEqual(smiles_from_3d(out, worker.conformers(out, 2)["frames"][0]), Chem.MolToSmiles(Chem.MolFromSmiles(smiles)))

    def test_draws_a_molecule_in_3d_as_a_formula_keeping_its_configuration(self):
        from rdkit import Chem

        made = ask("conformers", molblock=PCPA, count=2)["result"]["isomers"][0]
        lines = ["", "  test", "", f"{len(made['atoms']):3d}{len(made['bonds']):3d}  0  0  0  0  0  0  0  0999 V2000"]
        xyz = made["frames"][0]
        for i, a in enumerate(made["atoms"]):
            lines.append(f"{xyz[3*i]:10.4f}{xyz[3*i+1]:10.4f}{xyz[3*i+2]:10.4f} {a['el']:<3} 0  0")
        for b in made["bonds"]:
            # (all single, as coordinates alone give them: the orders are found again)
            lines.append(f"{b['a1'] + 1:3d}{b['a2'] + 1:3d}  1  0")
        block = "\n".join(lines + ["M  END"]) + "\n"
        drawn = ask("drawing_of", molblock=block, perceive=True)["result"]["molblock"]
        mol = Chem.MolFromMolBlock(drawn)
        # its heavy atoms, in their order, and the same molecule, configuration and all
        self.assertEqual([a.GetSymbol() for a in mol.GetAtoms()], ["C"] * 9 + ["O", "O", "N", "Cl"])
        self.assertEqual(Chem.MolToSmiles(mol), ask("to_smiles", molblock=PCPA)["result"]["smiles"])

    def test_draws_a_molecule_in_3d_of_more_atoms_than_v2000_counts_as_a_formula(self):
        # an all-trans alkane, C333H668 - 1001 atoms - its bonds all single,
        # as a file of coordinates gives them, in a V3000 block
        n = 333
        atoms = [("C", 1.258 * i, 0.444 if i % 2 else -0.444, 0.0) for i in range(n)]
        bonds = [(i, i + 1, 1) for i in range(n - 1)]
        for i in range(n):
            side = 1 if i % 2 else -1
            for z in (0.89, -0.89):
                atoms.append(("H", atoms[i][1], atoms[i][2] + side * 0.63, z))
                bonds.append((i, len(atoms) - 1, 1))
        for i, d in ((0, -1), (n - 1, 1)):
            atoms.append(("H", atoms[i][1] + d * 1.03, atoms[i][2] + (1 if i % 2 else -1) * 0.36, 0.0))
            bonds.append((i, len(atoms) - 1, 1))
        self.assertEqual(len(atoms), 1001)
        found = ask("drawing_of", molblock=v3000_3d(atoms, bonds), perceive=True)["result"]
        self.assertTrue(found["perceived"])
        drawn = Chem.MolFromMolBlock(found["molblock"])
        # its heavy atoms, in their order: a chain of 333 carbons
        self.assertEqual(drawn.GetNumAtoms(), n)
        self.assertEqual(Chem.MolToSmiles(drawn), "C" * n)

    def test_says_whether_it_found_a_molecule_in_3d_s_bond_orders(self):
        # benzoic acid in 3D, its bonds all single as a file of coordinates gives them
        from rdkit.Chem import AllChem
        mol = Chem.AddHs(Chem.MolFromSmiles("OC(=O)c1ccccc1"))
        AllChem.EmbedMolecule(mol, randomSeed=11)
        flat = coordinates_only(mol)
        found = ask("drawing_of", molblock=Chem.MolToMolBlock(flat, kekulize=False), perceive=True)["result"]
        self.assertTrue(found["perceived"])
        self.assertEqual(Chem.MolToSmiles(Chem.MolFromMolBlock(found["molblock"])), "O=C(O)c1ccccc1")
        # its hydrogens left out, they cannot be found: it says so, and keeps the file's
        heavy = coordinates_only(Chem.RemoveHs(mol))
        lost = ask("drawing_of", molblock=Chem.MolToMolBlock(heavy, kekulize=False), perceive=True)
        self.assertFalse(lost["ok"] and lost["result"]["perceived"])

    def test_lays_out_a_long_chain_of_rings_and_a_large_ring_in_seconds(self):
        # (a chain of 40 para-linked rings, a ring of 60 carbons: past the sizes CoordGen is used for)
        t = time.time()
        r = ask("from_smiles", smiles="-".join(["c1ccc(cc1)"] * 40))
        self.assertTrue(r["ok"])
        r = ask("from_smiles", smiles="C1" + "C" * 59 + "1")
        self.assertTrue(r["ok"])
        # (its places flat, from RDKit's own depictor: an embedding in 3D takes long, and the bond orders go by bonds alone)
        from rdkit.Chem import rdDepictor
        mol = Chem.AddHs(Chem.MolFromSmiles("-".join(["c1ccc(cc1)"] * 40)))
        rdDepictor.Compute2DCoords(mol, forceRDKit=True)
        r = ask("drawing_of", molblock=Chem.MolToMolBlock(coordinates_only(mol), kekulize=False), perceive=True)
        self.assertTrue(r["ok"])
        self.assertTrue(r["result"]["perceived"])
        self.assertLess(time.time() - t, 10)

    def test_gives_a_ligand_its_bond_orders_from_a_smiles_and_then_its_hydrogens(self):
        from rdkit.Chem import AllChem
        mol = Chem.AddHs(Chem.MolFromSmiles("OC(=O)c1ccccc1"))
        AllChem.EmbedMolecule(mol, randomSeed=11)
        heavy = coordinates_only(Chem.RemoveHs(mol))
        block = Chem.MolToMolBlock(heavy, kekulize=False)
        given = ask("orders_from_smiles", molblock=block, smiles="OC(=O)c1ccccc1")["result"]["molblock"]
        got = Chem.MolFromMolBlock(given, removeHs=False)
        # its atoms in their order, the bonds as the SMILES has them
        self.assertEqual([a.GetSymbol() for a in got.GetAtoms()], [a.GetSymbol() for a in heavy.GetAtoms()])
        self.assertEqual(Chem.MolToSmiles(got), "O=C(O)c1ccccc1")
        # then its hydrogens, each where its atom puts it, after its atoms
        with_h = Chem.MolFromMolBlock(ask("add_hydrogens", molblock=given)["result"]["molblock"], removeHs=False)
        self.assertEqual(with_h.GetNumAtoms(), heavy.GetNumAtoms() + 6)
        self.assertEqual([a.GetSymbol() for a in with_h.GetAtoms()][: heavy.GetNumAtoms()], [a.GetSymbol() for a in heavy.GetAtoms()])
        conf = with_h.GetConformer()
        for a in with_h.GetAtoms():
            if a.GetSymbol() == "H":
                n = a.GetNeighbors()[0].GetIdx()
                self.assertLess((conf.GetAtomPosition(a.GetIdx()) - conf.GetAtomPosition(n)).Length(), 1.2)
        # a SMILES of another molecule: said, not forced
        self.assertFalse(ask("orders_from_smiles", molblock=block, smiles="c1ccccc1")["ok"])

    def test_answers_what_it_cannot_do_with_an_error_not_a_crash(self):
        self.assertEqual(ask("exec", code="1")["error"], "no such request: exec")
        self.assertFalse(ask("from_smiles", smiles="C1CC")["ok"])
        self.assertFalse(worker.answer("not json")["ok"])

    def test_says_what_it_could_not_read_without_naming_a_program(self):
        # (the words are shown as they are, in Quick Add and elsewhere: the maintainer, 2026-10-12 - naming no program. Why
        # a SMILES cannot be read is Meno's to say, from what `smiles_read` gives: src/lib/chem/smilesWhy)
        said = ask("from_smiles", smiles="C1CC")["error"]
        self.assertEqual(said, "This SMILES could not be read.")
        self.assertNotIn("RDKit", said)
        self.assertEqual(ask("to_smiles", molblock="no structure")["error"], "This structure could not be read.")


def coordinates_only(mol):
    """A molecule as a file of coordinates gives it: its atoms where they
    stand, every bond single, nothing else known."""
    flat = Chem.RWMol(mol)
    for b in flat.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
    for a in flat.GetAtoms():
        a.SetIsAromatic(False)
        a.SetFormalCharge(0)
    return flat.GetMol()


if __name__ == "__main__":
    unittest.main()
