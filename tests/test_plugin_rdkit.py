"""Tests for the RDKit plugin's worker (plugins/rdkit/worker.py).

The worker needs RDKit: these run in the plugin's own environment, installed
from its lock - in CI's job "plugins" on every push, and by hand with

    <the RDKit plugin's Python> -m unittest tests/test_plugin_rdkit.py

(The app's environment for it, under its data folder at uv/plugin-rdkit/venv,
will do.)
"""

import importlib.util
import math
import os
import pathlib
import time
import unittest

try:
    from rdkit import Chem
    from rdkit.Chem import rdDistGeom
except ImportError:  # (CI's Python)
    Chem = None

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = ROOT / "plugins/rdkit/worker.py"
#: What Meno holds each structure with metals these tests make to, by its connection table (`table_of`): Meno's own
#: answer (its lib/chem/metalHolds `metalHoldsOf`), written by Meno from the blocks these tests record into the file
#: MENO_RECORD_BLOCKS names - again whenever Meno's holds change.
METAL_HOLDS = ROOT / "tests/metal-holds.json"


def table_of(molblock):
    """A MOL block's connection table - its atoms, bonds and properties, in their order - with its header, its atoms'
    coordinates and its bonds' wedges left out: all Meno's holds are worked out from (lib/chem/metalHolds
    `holdsGraphOf`), and the same on every platform, where the coordinates RDKit lays a structure out at - and so the
    wedges it chooses - differ from one build to another."""
    lines = molblock.splitlines()[3:]
    if not lines:
        return ""
    if "V3000" not in lines[0]:
        n, m = int(lines[0][0:3]), int(lines[0][3:6])
        atoms = [line[30:] for line in lines[1 : 1 + n]]
        bonds = [line[:9] + line[12:] for line in lines[1 + n : 1 + n + m]]
        return "\n".join([lines[0], *atoms, *bonds, *lines[1 + n + m :]])
    out, block = [], ""
    for line in lines:
        if line.startswith("M  V30 BEGIN ") or line.startswith("M  V30 END "):
            block = line.split()[3] if line.startswith("M  V30 BEGIN ") else ""
        elif block == "ATOM" and line.startswith("M  V30 "):
            t = line.split()
            line = " ".join(t[:4] + t[7:])
        elif block == "BOND" and line.startswith("M  V30 "):
            line = " ".join(t for t in line.split() if not t.startswith("CFG="))
        out.append(line)
    return "\n".join(out)


def held(molblock):
    """A request's structure, its MOL block, with what Meno holds it to where it has metals (METAL_HOLDS) - found by its
    connection table (`table_of`). A structure with a metal bonded to anything that the file has no holds for fails the
    test that sends it, saying so: sent unheld, it would be embedded as the plugin alone would, and pass or fail for no
    reason of its own."""
    import json
    import os

    record = os.environ.get("MENO_RECORD_BLOCKS")
    if record:
        with open(record, "a", encoding="utf-8") as f:
            f.write(json.dumps(molblock) + "\n")
    if not hasattr(held, "known"):
        held.known = {table_of(c["molblock"]): c["holds"] for c in json.loads(METAL_HOLDS.read_text())} if METAL_HOLDS.exists() else {}
    holds = held.known.get(table_of(molblock))
    if holds is None and not record and metal_bonded(molblock):
        raise AssertionError(f"{METAL_HOLDS.name} has no holds for this structure with a metal: write it again from Meno")
    return {"molblock": molblock, **({"holds": holds} if holds else {})}


def metal_bonded(molblock):
    """Whether a MOL block has a d- or f-block metal bonded to anything: one Meno sends holds for."""
    mol = Chem.MolFromMolBlock(molblock, sanitize=False, removeHs=False)
    is_metal = lambda z: 21 <= z <= 30 or 39 <= z <= 48 or 57 <= z <= 80 or 89 <= z <= 112
    return mol is not None and any(is_metal(a.GetAtomicNum()) and a.GetDegree() > 0 for a in mol.GetAtoms())


@unittest.skipUnless(Chem, "RDKit is not installed here")
class HeldAsMenoSays(unittest.TestCase):
    """The holds these tests send are found by a structure's connection table alone (`table_of`), so that the same
    structure laid out at other coordinates - by another build of RDKit, on another platform - finds them; and a
    structure with a metal the file has none for fails, never sent unheld."""

    def test_finds_the_holds_of_a_structure_laid_out_elsewhere(self):
        import json
        cases = json.loads(METAL_HOLDS.read_text())
        self.assertEqual(len({table_of(c["molblock"]) for c in cases}), len(cases))
        for c in cases:
            # (its atoms' coordinates, and nothing else, moved in their last digits - as another build lays it out)
            lines = c["molblock"].splitlines()
            if "V3000" in lines[3]:
                inside = False
                for k, line in enumerate(lines):
                    if line.startswith("M  V30 BEGIN ATOM") or line.startswith("M  V30 END ATOM"):
                        inside = line.startswith("M  V30 BEGIN")
                    elif inside:
                        t = line.split(" ")
                        lines[k] = line.replace(f" {t[5]} {t[6]} ", f" {float(t[5]) + 0.0001:.4f} {float(t[6]) - 0.0001:.4f} ", 1)
            else:
                n, m = int(lines[3][0:3]), int(lines[3][3:6])
                for k in range(4, 4 + n):
                    x, y = float(lines[k][0:10]), float(lines[k][10:20])
                    lines[k] = f"{x + 0.0001:10.4f}{y - 0.0001:10.4f}" + lines[k][20:]
                # (and a wedge chosen on another bond, as one laid out elsewhere may be)
                for k in range(4 + n, 4 + n + m):
                    lines[k] = lines[k][:9] + ("  1" if lines[k][9:12] == "  0" else "  0") + lines[k][12:]
            moved = "\n".join(lines) + "\n"
            self.assertNotEqual(moved, c["molblock"])
            self.assertEqual(held(moved).get("holds"), c["holds"])

    def test_fails_a_structure_with_a_metal_it_has_no_holds_for(self):
        mol = Chem.MolFromSmiles("[Cu]C#N", sanitize=False)
        with self.assertRaisesRegex(AssertionError, "has no holds for this structure with a metal"):
            held(Chem.MolToMolBlock(mol))
        # (one with no metal, or a metal bonded to nothing: none to hold, none needed)
        self.assertNotIn("holds", held(Chem.MolToMolBlock(Chem.MolFromSmiles("CCO"))))
        self.assertNotIn("holds", held(Chem.MolToMolBlock(Chem.MolFromSmiles("[Cu+2]", sanitize=False))))


@unittest.skipUnless(Chem, "RDKit is not installed here")
class AddHydrogens(unittest.TestCase):
    """A molecule's hydrogens added, as many on each atom as the chemist chose."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("plugin_rdkit", WORKER)
        cls.worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.worker)

    @staticmethod
    def block(atoms, bonds):
        rw = Chem.RWMol()
        conf = Chem.Conformer(len(atoms))
        for k, (el, charge, at) in enumerate(atoms):
            a = Chem.Atom(el)
            a.SetFormalCharge(charge)
            rw.AddAtom(a)
            conf.SetAtomPosition(k, at)
        for a1, a2, order in bonds:
            rw.AddBond(a1, a2, {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE}[order])
        m = rw.GetMol()
        m.AddConformer(conf)
        return Chem.MolToMolBlock(m, kekulize=False)

    @staticmethod
    def hydrogens(block, n):
        m = Chem.MolFromMolBlock(block, sanitize=False, removeHs=False)
        return [sum(1 for x in m.GetAtomWithIdx(i).GetNeighbors() if x.GetSymbol() == "H") for i in range(n)]

    def test_an_amine_on_a_metal_gets_an_amines_hydrogens_where_they_are_given(self):
        # (zinc's bond to the nitrogen is the zinc's: NH2, as chosen - by valence alone, an amido's one)
        block = self.block([("Zn", 0, (0, 0, 0)), ("N", 0, (2.1, 0, 0)), ("C", 0, (3.5, 0.3, 0))], [(0, 1, 1), (1, 2, 1)])
        self.assertEqual(self.hydrogens(self.worker.op_add_hydrogens({"molblock": block, "hydrogens": [0, 2, 3]})["molblock"], 3), [0, 2, 3])
        self.assertEqual(self.hydrogens(self.worker.op_add_hydrogens({"molblock": block})["molblock"], 3), [0, 1, 3])

    def test_glycines_zwitterion_and_a_count_for_each_atom(self):
        block = self.block(
            [("N", 1, (0, 0, 0)), ("C", 0, (1.47, 0, 0)), ("C", 0, (2, 1.4, 0)), ("O", 0, (1.3, 2.4, 0)), ("O", -1, (3.3, 1.5, 0))],
            [(0, 1, 1), (1, 2, 1), (2, 3, 2), (2, 4, 1)],
        )
        self.assertEqual(self.hydrogens(self.worker.op_add_hydrogens({"molblock": block, "hydrogens": [3, 2, 0, 0, 0]})["molblock"], 5), [3, 2, 0, 0, 0])
        with self.assertRaises(ValueError):
            self.worker.op_add_hydrogens({"molblock": block, "hydrogens": [3, 2]})


@unittest.skipUnless(Chem, "RDKit is not installed here")
class FromSmiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("plugin_rdkit", WORKER)
        cls.worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.worker)

    def drawn(self, smiles):
        """What the MOL block a SMILES is drawn as reads back as."""
        block = self.worker.op_from_smiles({"smiles": smiles})["molblock"]
        return Chem.MolToSmiles(Chem.MolFromMolBlock(block))

    def test_keeps_a_trans_double_bond_in_a_ring_of_eight(self):
        # (CoordGen drew it cis: the block said (Z)-cyclooctene)
        given = "C1CC/C=C/CCC1"
        self.assertEqual(self.drawn(given), Chem.CanonSmiles(given))

    def test_keeps_what_coordgen_draws_well(self):
        for given in ["C1CCCC/C=C/CCC1", "C[C@H](N)C(=O)O", "C[S@@](=O)c1ccc(C)cc1", "C/C=C\\C=C\\C(=O)O"]:
            self.assertEqual(self.drawn(given), Chem.CanonSmiles(given), given)




@unittest.skipUnless(Chem, "RDKit is not installed here")
class SmilesNotRead(unittest.TestCase):
    """A SMILES that cannot be read: said so, and what RDKit made of it given as facts (`smiles_read`). Why, in words,
    is Meno's (src/lib/chem/smilesWhy, with its own tests)."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("plugin_rdkit", WORKER)
        cls.worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.worker)

    def test_says_it_could_not_be_read_and_no_more(self):
        for smiles in ("C1CCCC", "C(C)(C)(C)(C)C", "SnCl4", "hello", "c1cccc1"):
            with self.assertRaises(ValueError) as caught:
                self.worker.op_from_smiles({"smiles": smiles})
            self.assertEqual(str(caught.exception), "This SMILES could not be read.")

    def test_leaves_what_reads_alone(self):
        for smiles in ("Sc1ccccc1", "CCCl", "c1ccccc1Br", "[Cl-].[Na+]", "C[Hg]C"):
            self.assertIn("molblock", self.worker.op_from_smiles({"smiles": smiles}))

    def test_gives_rdkits_problems_as_facts(self):
        read = lambda smiles: self.worker.op_smiles_read({"smiles": smiles})
        self.assertEqual(read("C(C)(C)(C)(C)C"), {"reads": False, "parsed": True, "problems": [
            {"kind": "AtomValenceException", "atom": 0, "symbol": "C", "charge": 0, "bonds": 5}]})
        self.assertEqual(read("C[O-](C)C")["problems"], [{"kind": "AtomValenceException", "atom": 1, "symbol": "O", "charge": -1, "bonds": 3}])
        self.assertEqual(read("c1cccc1")["problems"], [{"kind": "KekulizeException", "atoms": [0, 1, 2, 3, 4]}])
        self.assertEqual(read("Co")["problems"], [{"kind": "AtomKekulizeException", "atom": 1, "symbol": "O", "charge": 0, "bonds": 1}])
        # (one that reads, one that does not parse at all)
        self.assertEqual(read("Cl[Sn](Cl)(Cl)Cl"), {"reads": True, "parsed": True, "problems": []})
        self.assertEqual(read("C1CC("), {"reads": False, "parsed": False, "problems": []})
        # (and the words are not the plugin's: none of Meno's reasons is said here)
        self.assertFalse(hasattr(self.worker, "why_unreadable"))


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(Chem, "RDKit is not installed here")
class AWorkflowsSearch(unittest.TestCase):
    """A conformer search gives RDKit's own conformers: every one embedded and optimised, as RDKit made it, in the
    order it embedded them - none taken for another, laid over another, renumbered or ranked; each stereoisomer
    made itself, an enantiomer as much as any (a mirror image, where one is wanted, is Meno's to make)."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("plugin_rdkit", WORKER)
        cls.worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.worker)

    def test_gives_rdkits_own_conformers_frame_for_frame(self):
        # (the known answer, RDKit's own: ETKDG v3 at the worker's seed, MMFF94 optimisation, in the order embedded)
        from rdkit.Chem import rdForceFieldHelpers
        mol = Chem.MolFromSmiles("C[C@H](O)CC")
        made = self.worker.conformers(mol, count=12)
        raw = Chem.AddHs(mol)
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = self.worker.SEED
        params.pruneRmsThresh = -1.0
        params.numThreads = 0
        ids = list(rdDistGeom.EmbedMultipleConfs(raw, numConfs=12, params=params))
        energy = {cid: e for cid, (_, e) in zip(ids, rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(raw, numThreads=0, maxIters=self.worker.STEPS, mmffVariant="MMFF94"))}
        self.assertEqual(made["frames"], [[round(v, 4) for p in raw.GetConformer(c).GetPositions() for v in p] for c in ids])
        for got, cid in zip(made["energies"], ids):
            self.assertAlmostEqual(got, energy[cid] / self.worker.KCAL_PER_HARTREE, places=9)
        rows = {r["label"]: r["text"] for r in made["how"]}
        self.assertEqual(rows["Kept"], "12 of 12 embedded (12 sought)")

    def test_makes_each_enantiomer_itself_whatever_else_it_is_sent(self):
        # (butan-2-ol's two enantiomers: what the page once sent to have one made as the other's mirror image, in the
        #  request or in its options, changes nothing - and never reaches the options)
        for smiles in ("C[C@@H](O)CC", "C[C@H](O)CC"):
            mb = self.worker.op_from_smiles({"smiles": smiles})["molblock"]
            plain = self.worker.op_conformers({"molblock": mb, "count": 4})
            self.assertEqual(self.worker.op_conformers({"molblock": mb, "count": 4, "onPage": True}), plain)
            self.assertEqual(self.worker.op_conformers({"molblock": mb, "count": 4, "options": {"onPage": True, "page": True}}), plain)
        self.assertNotIn("page", self.worker.options_of({"options": {"onPage": True, "page": True}}))
        self.assertFalse(hasattr(self.worker, "mirror_of"))

    def test_says_when_it_embedded_from_random_coordinates(self):
        rows = lambda e: {r["label"]: r["text"] for r in e["how"]}
        self.assertEqual(rows(self.worker.conformers(Chem.MolFromSmiles("CCCC"), count=4))["Embedded"], "ETKDG v3")
        # (an eta complex is embedded from random coordinates, its star at no fixed place)
        fe = self.worker.op_conformers({**held(EtaBound.ferrocene()), "count": 4})["isomers"][0]
        self.assertEqual(rows(fe)["Embedded"], "ETKDG v3, from random coordinates")


@unittest.skipUnless(Chem, "RDKit is not installed here")
class Metals(unittest.TestCase):
    """A metal's bonds come out at their lengths, and its shape as UFF has it - or, where UFF has no type for it, as
    a lighter metal of its group shapes it, with no energies; never pushed off, as UFF with no terms for it did."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("plugin_rdkit", WORKER)
        cls.worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.worker)

    def made(self, smiles, count=10):
        return self.worker.op_conformers({**held(self.worker.op_from_smiles({"smiles": smiles})["molblock"]), "count": count})["isomers"][0]

    def lengths_and_angles(self, entry, frame, symbol):
        import math
        import numpy as np
        P = np.array(entry["frames"][frame]).reshape(-1, 3)
        m = next(i for i, a in enumerate(entry["atoms"]) if a["el"] == symbol)
        near = {b["a2"] if b["a1"] == m else b["a1"]: b for b in entry["bonds"] if m in (b["a1"], b["a2"])}
        lengths = {j: float(np.linalg.norm(P[j] - P[m])) for j in near}
        ids = sorted(near)
        angles = sorted(
            math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(P[i] - P[m], P[j] - P[m]) / lengths[i] / lengths[j])))))
            for x, i in enumerate(ids) for j in ids[x + 1:]
        )
        return {entry["atoms"][j]["el"]: round(d, 2) for j, d in lengths.items()}, angles, lengths

    def test_closes_a_palladacycle_through_the_metal(self):
        # (the maintainer's: a ring of fifteen closed through Pd - UFF without a type for Pd put it 8 to 25 A away)
        entry = self.made("C1CCCCCCCCCCCC(C)C[Pd]1")
        for f in range(len(entry["frames"])):
            _, angles, lengths = self.lengths_and_angles(entry, f, "Pd")
            for d in lengths.values():
                self.assertAlmostEqual(d, 2.08, delta=0.1)
        self.assertEqual(entry["field"], "UFF")
        self.assertTrue(entry["energies"])

    def test_makes_a_square_planar_palladium(self):
        # (its lowest: the conformers come in the order embedded)
        entry = self.made("Cl[Pd](Cl)(P(C)(C)C)P(C)(C)C")
        lowest = min(range(len(entry["frames"])), key=lambda k: entry["energies"][k])
        lengths, angles, _ = self.lengths_and_angles(entry, lowest, "Pd")
        self.assertAlmostEqual(lengths["Cl"], 2.32, delta=0.06)
        self.assertAlmostEqual(lengths["P"], 2.41, delta=0.06)
        self.assertGreater(angles[-1], 165)
        self.assertGreater(angles[-2], 165)
        self.assertTrue(all(80 < a < 100 for a in angles[:4]), angles)

    def test_makes_an_octahedral_iron(self):
        entry = self.made("Cl[Fe-3](Cl)(Cl)(Cl)(Cl)Cl", count=3)
        _, angles, _ = self.lengths_and_angles(entry, 0, "Fe")
        self.assertTrue(all(85 < a < 95 for a in angles[:12]), angles)
        self.assertTrue(all(a > 170 for a in angles[12:]), angles)

    def test_shapes_a_ruthenium_carbene_with_iron_standing_in_and_gives_no_energies(self):
        # (a Grubbs-type catalyst: UFF has no type RDKit can give Ru - it stopped on a missing parameter)
        entry = self.made("Cl[Ru](Cl)(=Cc1ccccc1)(P(C)(C)C)P(C)(C)C", count=5)
        self.assertNotIn("energies", entry)
        lengths, angles, _ = self.lengths_and_angles(entry, 0, "Ru")
        self.assertAlmostEqual(lengths["Cl"], 2.24, delta=0.08)
        self.assertAlmostEqual(lengths["P"], 2.36, delta=0.1)
        self.assertGreater(angles[-1], 160)
        rows = {r["label"]: r["text"] for r in entry["how"]}
        self.assertIn("Fe standing in for Ru", rows["Optimised"])
        self.assertIn("no parameters for Ru", rows["Energies"])

    def test_says_where_nothing_could_be_optimised(self):
        # (osmium with eight bonds: no geometry of UFF's fits it, so none stands in)
        mol = Chem.MolFromSmiles("Cl[Os](Cl)(Cl)(Cl)(Cl)(Cl)(Cl)Cl")
        if mol is None:
            self.skipTest("RDKit read no eight-bonded osmium here")
        entry = self.worker.conformers(mol, count=2)
        self.assertNotIn("energies", entry)
        rows = {r["label"]: r["text"] for r in entry["how"]}
        self.assertTrue(rows["Optimised"].startswith("not: no force field here has parameters for Os"), rows)


@unittest.skipUnless(Chem, "RDKit is not installed here")
class RingChairs(unittest.TestCase):
    """A ring's two chairs are both conformers of it: pruning while embedding, on coordinates no force field had
    touched yet, kept whichever chair came first and threw the other away."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("plugin_rdkit", WORKER)
        cls.worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.worker)

    def chairs(self, smiles):
        """Each conformer of `smiles` as (kcal/mol above the lowest, how far its substituent leans off the ring's
        normal): a chair's axial bond lies along the normal, its equatorial bond across it. Of a cyclohexane
        with one substituent on it, whatever that substituent is."""
        import numpy as np
        mb = self.worker.op_from_smiles({"smiles": smiles})["molblock"]
        # (as many as Meno sends a ring that can flip: lib/chem/conformersSought)
        entry = self.worker.op_conformers({**held(mb), "count": 200})["isomers"][0]
        els = [a["el"] for a in entry["atoms"]]
        nbr, single = {}, {}
        for b in entry["bonds"]:
            nbr.setdefault(b["a1"], []).append(b["a2"])
            nbr.setdefault(b["a2"], []).append(b["a1"])
            if b["order"] == 1:
                single.setdefault(b["a1"], []).append(b["a2"])
                single.setdefault(b["a2"], []).append(b["a1"])

        def walk(at, path):
            """Six carbons joined by single bonds, back where it began: the saturated ring, not a benzene."""
            if len(path) == 6:
                return path if path[0] in single.get(at, []) else None
            for n in single.get(at, []):
                if els[n] == "C" and n not in path:
                    got = walk(n, path + [n])
                    if got:
                        return got
            return None

        ring = next(filter(None, (walk(i, [i]) for i, el in enumerate(els) if el == "C")), None)
        self.assertEqual(len(ring or []), 6, smiles)
        # (the ring carbon that carries the substituent, and the substituent's first atom)
        c1, sub = next((i, n) for i in ring for n in nbr[i] if els[n] != "H" and n not in ring)
        lowest = min(entry["energies"])
        out = []
        for k, frame in enumerate(entry["frames"]):
            P = np.array(frame).reshape(-1, 3)
            R = P[ring]
            normal = np.linalg.svd(R - R.mean(axis=0))[2][2]
            v = P[sub] - P[c1]
            out.append(((entry["energies"][k] - lowest) * 627.5094740631, abs(float(v @ normal)) / float(np.linalg.norm(v))))
        return out

    def test_gives_both_chairs_of_methoxycyclohexane_the_equatorial_one_lowest(self):
        # (the axial chair is a real minimum about half a kcal/mol above the equatorial one: a set without it is
        # wrong for anyone reasoning about a ring's substituents)
        chairs = [(kcal, cos) for kcal, cos in self.chairs("COC1CCCCC1") if kcal < 3.0]
        axial = [kcal for kcal, cos in chairs if cos > 0.6]
        equatorial = [kcal for kcal, cos in chairs if cos < 0.5]
        self.assertTrue(equatorial, chairs)
        self.assertTrue(axial, chairs)
        # (the equatorial chair is the lowest, and the axial one is near it, not a ring flip away in energy)
        self.assertEqual(min(equatorial), 0.0)
        self.assertLess(min(axial), 2.0)

    def test_gives_both_chairs_of_phenylcyclohexane_whose_axial_one_is_rarely_embedded(self):
        # (an embedding falls into phenylcyclohexane's axial chair about once in twenty-five tries - measured over
        # 8000 embeddings, 4.3% - so thirty tries missed it altogether: it is the molecule that says how many are
        # needed of a ring that can flip, not a fixed number)
        chairs = self.chairs("c1ccccc1C1CCCCC1")
        axial = [kcal for kcal, cos in chairs if cos > 0.6]
        equatorial = [kcal for kcal, cos in chairs if cos < 0.5]
        self.assertTrue(equatorial, chairs)
        self.assertTrue(axial, chairs)
        self.assertEqual(min(equatorial), 0.0)
        # (the axial chair of a phenyl is some 3 kcal/mol up, and a real minimum: in a set of its conformers)
        self.assertLess(min(axial), 4.0)


@unittest.skipUnless(Chem, "RDKit is not installed here")
class StructureFacts(unittest.TestCase):
    """What RDKit makes of a structure drawn, as facts for Meno's *Check structures* to decide on: each atom's element,
    atomic number, charge, valence and the most it could be, unpaired electrons and part; what it could not make
    sense of; nothing judged."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("plugin_rdkit", WORKER)
        cls.worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.worker)

    def facts(self, *smiles):
        blocks = []
        for s in smiles:
            m = Chem.MolFromSmiles(s) or Chem.MolFromSmiles(s, sanitize=False)
            m.UpdatePropertyCache(strict=False)
            blocks.append(Chem.MolToV3KMolBlock(m, kekulize=False))
        return self.worker.op_structure_facts({"structures": [{"molblocks": blocks}]})["structures"][0]

    def test_gives_each_atom_as_rdkit_makes_it(self):
        salt = self.facts("CC(=O)[O-]", "[Na+]")
        self.assertEqual([(a["el"], a["z"], a["charge"], a["part"]) for a in salt["atoms"]], [("C", 6, 0, 0), ("C", 6, 0, 0), ("O", 8, 0, 0), ("O", 8, -1, 0), ("Na", 11, 1, 1)])
        self.assertEqual([a["radicals"] for a in self.facts("[CH3]")["atoms"]], [1])
        five = self.facts("C(C)(C)(C)(C)C")["atoms"][0]
        self.assertEqual((five["valence"], five["most"]), (5, 4))

    def test_says_what_it_could_not_make_sense_of_and_what_it_could_not_read(self):
        self.assertEqual(self.facts("c1cccc1")["problem"], "SANITIZE_KEKULIZE")
        self.assertNotIn("problem", self.facts("c1ccccc1"))
        self.assertEqual(self.worker.op_structure_facts({"structures": [{"molblocks": ["not a block"]}]})["structures"], [{"read": False}])

    def test_is_asked_and_judges_nothing_itself(self):
        # (Check structures is Meno's: the plugin fills no such step, and answers the facts)
        import json
        manifest = json.loads((WORKER.parent / "manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(any("check-structures" in s["kinds"] for s in manifest["steps"]))
        self.assertNotIn("check-structures", self.worker.STRUCTURE_STEPS)


@unittest.skipUnless(Chem, "RDKit is not installed here")
class HowManyTried(unittest.TestCase):
    """How many conformers are tried is Meno's to say (lib/chem/conformersSought): the search tries the number it
    is sent, never one of its own."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("plugin_rdkit", WORKER)
        cls.worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.worker)

    def test_tries_the_number_it_is_sent(self):
        self.assertEqual(self.worker.options_of({"count": 12})["count"], 12)
        # (a number in the role's options is none of its: Meno sends it beside them)
        self.assertIsNone(self.worker.options_of({"options": {"count": 12}})["count"])
        self.assertIsNone(self.worker.options_of({})["count"])

    def test_gives_another_set_of_tries_for_each_draw_and_the_same_for_the_same(self):
        seed = self.worker.options_of({"count": 5})["seed"]
        self.assertEqual(self.worker.options_of({"count": 5, "draw": 0})["seed"], seed)
        self.assertEqual(self.worker.options_of({"count": 5, "draw": 2})["seed"], (seed + 2 * self.worker.DRAW_STRIDE) % 2**31)
        self.assertGreater(self.worker.DRAW_STRIDE, self.worker.MOST_CONFORMERS)
        mb = self.worker.op_from_smiles({"smiles": "CCCCO"})["molblock"]
        frames = lambda draw: self.worker.op_conformers({"molblock": mb, "count": 5, "draw": draw})["isomers"][0]["frames"]
        self.assertEqual(frames(1), frames(1))
        self.assertNotEqual(frames(0), frames(1))
        for bad, said in ((-1, "from 0 to"), (1.5, "a whole number"), ("1", "a whole number"), (True, "a whole number")):
            with self.assertRaisesRegex(ValueError, f"The draw asked for is to be {said}"):
                self.worker.options_of({"count": 5, "draw": bad})

    def test_makes_none_where_it_is_sent_no_number(self):
        with self.assertRaisesRegex(ValueError, "How many conformers to try was not given"):
            self.worker.op_conformers({**held(self.worker.op_from_smiles({"smiles": "CCO"})["molblock"])})

    def test_says_its_search_takes_a_number_of_tries_and_asks_none_itself(self):
        import json
        manifest = json.loads((WORKER.parent / "manifest.json").read_text(encoding="utf-8"))
        [role] = [r for r in manifest["roles"] if r.get("role") == "conformers"]
        self.assertIs(role["tries"], True)
        self.assertIs(role["draws"], True)
        self.assertFalse({"sought", "count"} & {o["id"] for o in role["options"]})

    def test_a_workflows_search_tries_the_number_sent_with_each_molecule(self):
        # (a Conformers step's job: the count Meno sends with the molecule written beside it, and tried)
        import json, subprocess, sys, tempfile
        mol = Chem.AddHs(Chem.MolFromSmiles("CCCO"))
        rdDistGeom.EmbedMolecule(mol, randomSeed=1)
        xyz = mol.GetConformer().GetPositions()
        entry = {
            "atoms": [{"el": a.GetSymbol(), "x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for a, p in zip(mol.GetAtoms(), xyz)],
            "bonds": [{"a1": b.GetBeginAtomIdx(), "a2": b.GetEndAtomIdx(), "order": 1} for b in mol.GetBonds()],
            "count": 3,
        }
        [job] = self.worker.op_prepare({"step": "conformers", "entries": [entry], "options": {"iters": 50}})["jobs"]
        given = json.loads(job["files"][0]["text"])
        self.assertEqual((given["count"], given["options"]), (3, {"iters": 50}))
        with tempfile.TemporaryDirectory() as here:
            with open(f"{here}/molecule.json", "w", encoding="utf-8") as f:
                f.write(job["files"][0]["text"])
            ran = subprocess.run([sys.executable, *job["args"]], cwd=here, capture_output=True, text=True)
            self.assertEqual(ran.returncode, 0, ran.stderr)
            self.assertIn("up to 3 conformers", ran.stdout)
            with open(f"{here}/conformers.json", encoding="utf-8") as f:
                made = json.load(f)
        self.assertEqual({r["label"]: r["text"] for r in made["how"]}["Kept"], "3 of 3 embedded (3 sought)")

    def test_takes_as_many_as_meno_lets_a_chemist_ask_for(self):
        # (Meno's *How many* goes to 5000: lib/chem/conformersSought MOST_CONFORMERS)
        self.assertEqual(self.worker.MOST_CONFORMERS, 5000)
        for n in (501, 1000, 5000):
            self.assertEqual(self.worker.options_of({"count": n})["count"], n)


@unittest.skipUnless(Chem, "RDKit is not installed here")
class WholeNumbers(unittest.TestCase):
    """A count, a seed and a number of steps are whole numbers, each in its range: the manifest marks them `whole`,
    and one that is not - a fraction, text, one out of range - is refused, saying so, never cut to one or moved into
    range."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("plugin_rdkit", WORKER)
        cls.worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.worker)

    def test_marks_every_number_it_takes_whole(self):
        import json
        manifest = json.loads((WORKER.parent / "manifest.json").read_text(encoding="utf-8"))
        # (its roles' options, and its steps', by the role or the steps' kinds)
        numbers = {(r["role"], o["id"]): o.get("whole") for r in manifest["roles"] for o in r.get("options") or [] if o["type"] == "number"}
        numbers |= {("/".join(s["kinds"]), o["id"]): o.get("whole") for s in manifest["steps"] for o in s.get("options") or [] if o["type"] == "number"}
        self.assertEqual(numbers, {("conformers", "iters"): True, ("conformers", "seed"): True, ("minimise", "steps"): True})

    def test_takes_a_whole_number_in_its_range_as_it_is(self):
        o = self.worker.options_of({"count": 12.0, "options": {"seed": 7, "iters": 10}})
        self.assertEqual((o["count"], o["seed"], o["iters"]), (12, 7, 10))
        self.assertEqual(self.worker.options_of({"count": 5000, "options": {"seed": 2**31 - 1, "iters": 100000}})["seed"], 2**31 - 1)
        # (none given: the plugin's own, as before)
        self.assertEqual((self.worker.options_of({})["seed"], self.worker.options_of({})["iters"]), (self.worker.SEED, self.worker.STEPS))

    def test_refuses_a_fraction_text_or_one_out_of_range_saying_so(self):
        for asked, said in (
            ({"count": 2.5}, "How many conformers to try is to be a whole number: 2.5 is not one"),
            ({"count": 0}, "How many conformers to try is to be from 1 to 5000: 0 is not"),
            ({"count": 5001}, "How many conformers to try is to be from 1 to 5000: 5001 is not"),
            ({"options": {"seed": 7.5}}, "Random seed is to be a whole number: 7.5 is not one"),
            ({"options": {"seed": -1}}, "Random seed is to be from 0 to 2147483647: -1 is not"),
            ({"options": {"seed": "7"}}, "Random seed is to be a whole number: '7' is not one"),
            ({"options": {"iters": 9}}, "Optimisation steps, at most is to be from 10 to 100000: 9 is not"),
            ({"options": {"iters": True}}, "Optimisation steps, at most is to be a whole number: True is not one"),
        ):
            with self.assertRaises(ValueError) as e:
                self.worker.options_of(asked)
            self.assertEqual(str(e.exception), said)

    def test_a_minimise_job_refuses_steps_that_are_no_whole_number(self):
        import json, subprocess, sys, tempfile
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        rdDistGeom.EmbedMolecule(mol, randomSeed=1)
        entry = {
            "atoms": [{"el": a.GetSymbol(), "x": p[0], "y": p[1], "z": p[2]} for a, p in zip(mol.GetAtoms(), mol.GetConformer().GetPositions())],
            "bonds": [{"a1": b.GetBeginAtomIdx(), "a2": b.GetEndAtomIdx(), "order": 1} for b in mol.GetBonds()],
        }
        for steps, ok in ((50, True), (50.5, False), (0, False)):
            [job] = self.worker.op_prepare({"step": "minimise", "entries": [entry], "options": {"field": "MMFF94", "steps": steps}})["jobs"]
            with tempfile.TemporaryDirectory() as here:
                with open(f"{here}/{job['files'][0]['name']}", "w", encoding="utf-8") as f:
                    f.write(job["files"][0]["text"])
                ran = subprocess.run([sys.executable, *job["args"]], cwd=here, capture_output=True, text=True)
            self.assertEqual(ran.returncode == 0, ok, ran.stderr)
            if not ok:
                self.assertIn("Steps at most is to be", ran.stderr)


@unittest.skipUnless(Chem, "RDKit is not installed here")
class MetalFloor(unittest.TestCase):
    """Two atoms bonded to a metal, of different ligands, held at least NARROWEST_AT_METAL apart - only where the
    ideal polyhedra of four, five and six points are the shapes there may be - and a conformer that came out
    under it dropped, and counted. Where a lighter metal stands in, its energies order the conformers."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("plugin_rdkit", WORKER)
        cls.worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.worker)

    def mol(self, smiles):
        m = Chem.MolFromSmiles(smiles, sanitize=False)
        m.UpdatePropertyCache(strict=False)
        return m

    def made(self, smiles, count=10):
        # (as many as Meno sends where nothing turns - lib/chem/conformersSought - or, given, as it sends that molecule;
        #  and what it holds it to, `held`)
        mb = Chem.MolToMolBlock(self.mol(smiles), kekulize=False)
        return self.worker.op_conformers({**held(mb), "count": count})["isomers"][0]

    def angles(self, entry, frame, symbol):
        import numpy as np
        P = np.array(entry["frames"][frame]).reshape(-1, 3)
        m = next(i for i, a in enumerate(entry["atoms"]) if a["el"] == symbol)
        near = sorted(b["a2"] if b["a1"] == m else b["a1"] for b in entry["bonds"] if m in (b["a1"], b["a2"]))
        out = []
        for x, i in enumerate(near):
            for j in near[x + 1:]:
                u, v = P[i] - P[m], P[j] - P[m]
                out.append(math.degrees(math.acos(max(-1.0, min(1.0, float(u @ v / np.linalg.norm(u) / np.linalg.norm(v)))))))
        return sorted(out), {entry["atoms"][i]["el"]: i for i in near}, P, m

    def test_gives_hexachloroosmate_as_the_octahedron_again_and_again(self):
        # (the known answer: one conformer, Oh. Fe standing in for Os, one embedding stopped in a well 87.7
        #  kcal/mol above the octahedron, two Cl 62 degrees apart - under the floor, which Meno sets aside and counts
        #  (lib/chem/metalHolds `keptMade`): every one given here; the other nine are the octahedron, relabelled)
        entry = self.made("Cl[Os-2](Cl)(Cl)(Cl)(Cl)Cl")
        self.assertEqual(len(entry["frames"]), 10)
        shapes = [self.angles(entry, k, "Os")[0] for k in range(10)]
        under = [k for k, angles in enumerate(shapes) if angles[0] < 75 - 5]
        self.assertEqual(len(under), 1)
        self.assertEqual(round(shapes[under[0]][0]), 62)
        for k, angles in enumerate(shapes):
            if k not in under:
                self.assertTrue(all(abs(a - 90) < 1 for a in angles[:12]), angles)
                self.assertTrue(all(a > 179 for a in angles[12:]), angles)
        rows = {r["label"]: r["text"] for r in entry["how"]}
        self.assertNotIn("Dropped", rows)
        self.assertEqual(rows["Kept"], "10 of 10 embedded (10 sought)")
        # (the model said once, in the line that says how they were made; its energies given as the model's, none
        #  as the molecule's)
        self.assertTrue(entry["method"].endswith("; modelled: Fe for Os"), entry["method"])
        self.assertNotIn("energies", entry)
        self.assertEqual(len(entry["modelEnergies"]), 10)

    def test_gives_the_trans_isomer_of_an_osmium_bisphosphine_the_lowest_energy_of_its_model(self):
        # (trans-OsCl4(PEt3)2, the one known: the lowest by the stand-in's energies, wherever it was embedded)
        entry = self.made("CCP(CC)(CC)[Os](Cl)(Cl)(Cl)(Cl)P(CC)(CC)CC", count=200)
        lowest = min(range(len(entry["frames"])), key=lambda k: entry["modelEnergies"][k])
        _, at, P, m = self.angles(entry, lowest, "Os")
        ps = [i for i, a in enumerate(entry["atoms"]) if a["el"] == "P"]
        import numpy as np
        u, v = P[ps[0]] - P[m], P[ps[1]] - P[m]
        self.assertGreater(math.degrees(math.acos(float(u @ v / np.linalg.norm(u) / np.linalg.norm(v)))), 170)

    def test_holds_seven_and_eight_bonded_atoms_at_the_crowded_floor(self):
        # (without a floor, ETKDG knowing no angle at a metal, [ZrF7]3-'s fluorides came out 33 degrees apart and
        #  OsCl4Br4's halides 25)
        for smiles, symbol, n in (("F[Zr-3](F)(F)(F)(F)(F)F", "Zr", 7), ("Cl[Os](Cl)(Cl)(Cl)(Br)(Br)(Br)Br", "Os", 8)):
            entry = self.made(smiles)
            for f in range(len(entry["frames"])):
                angles, _, _, _ = self.angles(entry, f, symbol)
                self.assertGreater(angles[0], 60 - 5, (smiles, angles[:3]))
            rows = {r["label"]: r["text"] for r in entry["how"]}
            self.assertIn("ligands at least 60\u00b0 apart", rows[symbol])
            # (pressed against the floor they come out up to two degrees under it: none is dropped for that)
            self.assertNotIn("Dropped", rows)

    def test_holds_nothing_at_a_bent_metallocene_whose_chlorides_stay_apart(self):
        # (Cp2ZrCl2: no floor at a metal with eta rings - their own holds keep the Cl off them - and its Cl-Zr-Cl
        #  still comes out as a bent metallocene's, about 97 degrees in crystal structures)
        import numpy as np
        block = EtaBound.zirconocene_dichloride()
        entry = self.worker.op_conformers({**held(block), "count": 10})["isomers"][0]
        for f in range(len(entry["frames"])):
            P = np.array(entry["frames"][f]).reshape(-1, 3)
            u, v = P[3] - P[0], P[4] - P[0]
            self.assertTrue(80 < math.degrees(math.acos(float(u @ v / np.linalg.norm(u) / np.linalg.norm(v)))) < 115)
        rows = {r["label"]: r["text"] for r in entry["how"]}
        self.assertNotIn("Dropped", rows)

    def test_gives_an_eta_complex_the_energies_of_the_model_it_was_optimised_in(self):
        # (Cp2ZrCl2, Ti standing in for Zr, its rings held; ferrocene, its rings held: each conformer that model's
        #  energy, in the order embedded - none the molecule's)
        energies = []
        optimised = self.worker.uff_with_metals

        def optimise(mol, ids, holds, iters, **kw):
            out = optimised(mol, ids, holds, iters, **kw)
            energies.append([e for _, e in out])
            return out

        self.worker.uff_with_metals = optimise
        try:
            zr = self.worker.op_conformers({**held(EtaBound.zirconocene_dichloride()), "count": 10})["isomers"][0]
            fe = self.worker.op_conformers({**held(EtaBound.ferrocene()), "count": 10})["isomers"][0]
        finally:
            self.worker.uff_with_metals = optimised
        self.assertTrue(zr["method"].endswith("; modelled: Ti for Zr, the \u03b7 rings held"), zr["method"])
        self.assertTrue(fe["method"].endswith("; modelled: the \u03b7 rings held"), fe["method"])
        for entry, model in zip((zr, fe), energies):
            self.assertNotIn("energies", entry)
            self.assertGreater(len(entry["frames"]), 1)
            self.assertEqual(entry["modelEnergies"], [e / self.worker.KCAL_PER_HARTREE for e in model])

    def test_holds_nothing_at_a_metal_with_an_eta2_ligand(self):
        # (Zeise's anion: the alkene's carbon and the cis Cl come near 71 degrees in real structures)
        entry = self.made("Cl[Pt-](Cl)(Cl)1CC1")
        rows = {r["label"]: r["text"] for r in entry["how"]}
        self.assertIn("no angle held between ligands: a ligand bound through atoms side by side", rows["Pt"])

@unittest.skipUnless(Chem, "RDKit is not installed here")
class EtaBound(unittest.TestCase):
    """A ring bound to a metal through all its atoms, written as the CTfile's multi-centre bond (a bond to a star
    at its centre, ENDPTS its atoms, ATTACH=ALL): held over its metal, its star at its centre, its cyclopentadienide
    counted as one - and no energies, the holds being no force field's."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("plugin_rdkit", WORKER)
        cls.worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.worker)

    @staticmethod
    def ferrocene(radical=False):
        """Ferrocene as Meno writes it from a drawing: Fe 1, two Kekule rings of five carbons drawn neutral
        (2-6, 7-11), a star at each ring's centre (12, 13) - each ring's carbon with no double bond in it a CH2,
        or, from a Cp label, a CH with an unpaired electron."""
        import math
        rows, bonds = ["M  V30 1 Fe 0 0 0 0"], []
        for k, (base, y) in enumerate(((2, 1.5), (7, -1.5))):
            for i in range(5):
                rad = " RAD=2" if radical and i == 4 else ""
                rows.append(f"M  V30 {base + i} C {math.cos(2 * math.pi * i / 5):.4f} {y + 0.3 * math.sin(2 * math.pi * i / 5):.4f} 0 0{rad}")
            for i in range(5):
                bonds.append((2 if i in (0, 2) else 1, base + i, base + (i + 1) % 5))
        rows += ["M  V30 12 * 0 1.5 0 0", "M  V30 13 * 0 -1.5 0 0"]
        lines = [f"M  V30 {n + 1} {t} {a} {b}" for n, (t, a, b) in enumerate(bonds)]
        lines += ["M  V30 11 9 1 12 ENDPTS=(5 2 3 4 5 6) ATTACH=ALL", "M  V30 12 9 1 13 ENDPTS=(5 7 8 9 10 11) ATTACH=ALL"]
        return "\n".join(["", "  Meno", "", "  0  0  0     0  0            999 V3000", "M  V30 BEGIN CTAB", "M  V30 COUNTS 13 12 0 0 0",
                          "M  V30 BEGIN ATOM", *rows, "M  V30 END ATOM", "M  V30 BEGIN BOND", *lines, "M  V30 END BOND",
                          "M  V30 END CTAB", "M  END"])

    @staticmethod
    def zirconocene_dichloride():
        """Cp2ZrCl2 as Meno writes it from Zr with two Cp labels and two Cl: Zr 1, two stars (2, 3), two Cl (4, 5),
        each Cp's ring of five carbons (6-10, 11-15), drawn neutral with one carbon's electron unpaired."""
        import math
        rows = ["M  V30 1 Zr 0 0 0 0", "M  V30 2 * 0 2.47 0 0", "M  V30 3 * 0 -2.47 0 0", "M  V30 4 Cl 1.25 0.25 0 0", "M  V30 5 Cl 1.25 -0.25 0 0"]
        bonds = []
        for base, y in ((6, 2.5), (11, -2.5)):
            for i in range(5):
                rad = " RAD=2" if i == 0 else ""
                rows.append(f"M  V30 {base + i} C {math.cos(2 * math.pi * i / 5):.4f} {y + 0.5 * math.sin(2 * math.pi * i / 5):.4f} 0 0{rad}")
            for i in range(5):
                bonds.append((2 if i in (1, 3) else 1, base + i, base + (i + 1) % 5))
        lines = ["M  V30 1 9 1 2 ENDPTS=(5 6 7 8 9 10) ATTACH=ALL", "M  V30 2 9 1 3 ENDPTS=(5 11 12 13 14 15) ATTACH=ALL",
                 "M  V30 3 1 1 4", "M  V30 4 1 1 5"]
        lines += [f"M  V30 {n + 5} {t} {a} {b}" for n, (t, a, b) in enumerate(bonds)]
        return "\n".join(["", "  Meno", "", "  0  0  0     0  0            999 V3000", "M  V30 BEGIN CTAB", "M  V30 COUNTS 15 14 0 0 0",
                          "M  V30 BEGIN ATOM", *rows, "M  V30 END ATOM", "M  V30 BEGIN BOND", *lines, "M  V30 END BOND",
                          "M  V30 END CTAB", "M  END"])

    def made(self, block):
        # (as many as Meno sends a metallocene: nothing in it turns, and its rings are held)
        return self.worker.op_conformers({**held(block), "count": 10})["isomers"][0]

    def test_says_how_it_was_made_and_what_was_held(self):
        # (one line, with every structure: embedding, the force field, and what was held that is no term of it)
        entry = self.made(self.ferrocene())
        self.assertTrue(entry["method"].startswith("ETKDG v3 + UFF; restrained: "), entry["method"])
        self.assertIn("\u03b7 rings: metal\u2013ring atom distances", entry["method"])
        self.assertIn("rings \u2265 2.9 \u00c5 apart", entry["method"])
        # (a molecule of no metal: nothing held)
        plain = self.worker.op_conformers({**held(self.worker.op_from_smiles({"smiles": "CCO"})["molblock"]), "count": 10})["isomers"][0]
        self.assertEqual(plain["method"], "ETKDG v3 + MMFF94")

    def test_keeps_a_bent_metallocenes_rings_apart(self):
        # (Cp2ZrCl2's rings were let fold onto each other, 41 degrees apart at the metal: nothing kept them apart)
        import numpy as np
        entry = self.made(self.zirconocene_dichloride())
        self.assertEqual(entry["stars"], [1, 2])
        for frame in entry["frames"]:
            P = np.array(frame).reshape(-1, 3)
            near = min(float(np.linalg.norm(P[i] - P[j])) for i in range(5, 10) for j in range(10, 15))
            self.assertGreater(near, 2.8)
            a, b = P[1] - P[0], P[2] - P[0]
            self.assertGreater(float(np.degrees(np.arccos(a @ b / np.linalg.norm(a) / np.linalg.norm(b)))), 110.0)

    def test_holds_each_ring_over_its_metal_with_its_star_at_its_centre(self):
        import numpy as np
        for radical in (False, True):
            entry = self.made(self.ferrocene(radical))
            self.assertEqual(entry["stars"], [11, 12])
            for frame in entry["frames"]:
                P = np.array(frame).reshape(-1, 3)
                for star, ring in ((11, range(1, 6)), (12, range(6, 11))):
                    centre = P[list(ring)].mean(axis=0)
                    self.assertLess(float(np.linalg.norm(P[star] - centre)), 1e-3)
                    for c in ring:
                        self.assertAlmostEqual(float(np.linalg.norm(P[c] - P[0])), 2.06, delta=0.06)

    def test_counts_each_ring_as_a_cyclopentadienide_not_a_ch2_nor_a_radical(self):
        for radical in (False, True):
            entry = self.made(self.ferrocene(radical))
            els = [a["el"] for a in entry["atoms"]]
            self.assertEqual(els.count("H"), 10, radical)
            self.assertEqual(entry["atoms"][0]["charge"], 2)
            self.assertEqual(sum(a["charge"] for a in entry["atoms"][1:11]), -2)

    def test_gives_no_energies_and_says_why(self):
        entry = self.made(self.ferrocene())
        self.assertNotIn("energies", entry)
        rows = {r["label"]: r["text"] for r in entry["how"]}
        self.assertEqual(rows["Optimised"], "UFF, at most 2000 steps")
        self.assertIn("held, not modelled", rows["Energies"])


@unittest.skipUnless(Chem, "RDKit is not installed here")
class HalideBridged(unittest.TestCase):
    """Two metals bridged by two chlorides, as in [Pd(allyl)Cl]2: its Pd2Cl2 ring held together - UFF's angle at a
    chloride bound twice once pulled it 8 to 12 A apart - and made at all where RDKit's own bounds could not be."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("plugin_rdkit", WORKER)
        cls.worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.worker)

    @staticmethod
    def block(atoms, bonds):
        rows = [f"M  V30 {n + 1} {el} {x:.4f} {y:.4f} 0 0{extra}" for n, (el, x, y, extra) in enumerate(atoms)]
        lines = [f"M  V30 {n + 1} {b}" for n, b in enumerate(bonds)]
        return "\n".join(["", "  Meno", "", "  0  0  0     0  0            999 V3000", "M  V30 BEGIN CTAB",
                          f"M  V30 COUNTS {len(atoms)} {len(bonds)} 0 0 0", "M  V30 BEGIN ATOM", *rows, "M  V30 END ATOM",
                          "M  V30 BEGIN BOND", *lines, "M  V30 END BOND", "M  V30 END CTAB", "M  END"])

    def drawn(self):
        """As drawn by hand: each allyl's three carbons bound to its Pd by a bond each."""
        atoms = [("Pd", -1.25, 0, ""), ("Pd", 1.25, 0, ""), ("Cl", 0, 1.125, ""), ("Cl", 0, -1.125, ""),
                 ("C", -2.625, 1.125, ""), ("C", -3.375, 0, ""), ("C", -2.625, -1.125, ""),
                 ("C", 2.625, 1.125, ""), ("C", 3.375, 0, ""), ("C", 2.625, -1.125, "")]
        bonds = ["1 1 3", "1 1 4", "1 2 3", "1 2 4", "2 5 6", "1 6 7", "1 1 5", "1 1 6", "1 1 7",
                 "2 8 9", "1 9 10", "1 2 8", "1 2 9", "1 2 10"]
        return self.block(atoms, bonds)

    def from_label(self):
        """As Meno writes it from the label [Pd(allyl)Cl]2: each allyl bound by one multi-centre bond, the
        chlorides' second bonds dative."""
        atoms = [("Pd", -1.9, 0.85, ""), ("C", -4.15, -0.45, ""), ("C", -5.65, -0.45, ""), ("C", -6.4, -1.75, " RAD=2"),
                 ("*", -5.4, -0.88, ""), ("Cl", 0, 1.75, ""), ("Pd", 1.9, 0.85, ""), ("C", 4.15, -0.45, ""),
                 ("C", 5.65, -0.45, ""), ("C", 6.4, -1.75, " RAD=2"), ("*", 5.4, -0.88, ""), ("Cl", 0, -0.05, "")]
        bonds = ["2 2 3", "1 3 4", "9 1 5 ENDPTS=(3 2 3 4) ATTACH=ALL", "1 1 6", "2 8 9", "1 9 10",
                 "9 7 11 ENDPTS=(3 8 9 10) ATTACH=ALL", "1 7 12", "9 6 7", "9 12 1"]
        return self.block(atoms, bonds)

    def check(self, entry, slack=0.1):
        import numpy as np
        els = [a["el"] for a in entry["atoms"]]
        pd = [i for i, e in enumerate(els) if e == "Pd"]
        cl = [i for i, e in enumerate(els) if e == "Cl"]
        heavy = [i for i, e in enumerate(els) if e not in ("H", "*")]
        for frame in entry["frames"]:
            P = np.array(frame).reshape(-1, 3)
            for m in pd:
                for x in cl:
                    self.assertAlmostEqual(float(np.linalg.norm(P[m] - P[x])), 2.27, delta=slack)
            self.assertLess(float(np.linalg.norm(P[pd[0]] - P[pd[1]])), 4.0)
            span = max(float(np.linalg.norm(P[i] - P[j])) for i in heavy for j in heavy)
            self.assertLess(span, 9.0)

    def test_keeps_the_dimer_drawn_by_hand_together(self):
        # (each allyl bound by three bonds to its Pd, five in all: UFF has no shape for that, and a conformer may come
        # out uneven - but held together, where every one had Pd-Cl 1.89 and 2.76 A)
        self.check(self.worker.op_conformers({**held(self.drawn()), "count": 4})["isomers"][0], slack=0.4)

    def test_makes_and_keeps_the_dimer_from_its_label_together(self):
        entry = self.worker.op_conformers({**held(self.from_label()), "count": 4})["isomers"][0]
        self.check(entry)
        import numpy as np
        P = np.array(entry["frames"][0]).reshape(-1, 3)
        for m, ring in ((0, (1, 2, 3)), (6, (7, 8, 9))):
            for c in ring:
                self.assertAlmostEqual(float(np.linalg.norm(P[m] - P[c])), 2.1, delta=0.15)


@unittest.skipUnless(Chem, "RDKit is not installed here")
class Configurations(unittest.TestCase):
    """Which stereoisomer each molecule in 3D is, read from its own geometry - held on cases whose answers are
    known, since RDKit marks FindPotentialStereo experimental: a change in a new RDKit is caught here, not
    inherited (docs/PLUGINS.md, *Configurations*)."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("plugin_rdkit", WORKER)
        cls.worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.worker)

    def made(self, smiles, seed=11):
        """A MOL block of `smiles` in 3D, its configuration as the SMILES gives it."""
        from rdkit.Chem import AllChem

        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        self.assertEqual(AllChem.EmbedMolecule(mol, params), 0)
        return Chem.MolToMolBlock(mol)

    def configured(self, smiles):
        return self.worker.op_configurations({"molblocks": [self.made(smiles)]})["molecules"][0]

    def test_finds_the_elements_that_can_be_stereogenic_and_no_others(self):
        # (each case's answer by hand: how many centres and double bonds could be stereogenic)
        known = {
            "C[C@H]1CC[C@@H](C)CC1": ("trans-1,4-dimethylcyclohexane: each ring carbon carries cis/trans", 2, 0),
            "C[C@H]1CC[C@H](C)CC1": ("cis-1,4-dimethylcyclohexane", 2, 0),
            "C[C@H]1CCC[C@@H](C)C1": ("cis-1,3-dimethylcyclohexane", 2, 0),
            "CC1CCCCC1": ("methylcyclohexane: C1's two ring paths alike", 0, 0),
            "CC1(C)CCCCC1": ("1,1-dimethylcyclohexane", 0, 0),
            "CC=C": ("propene", 0, 0),
            "Clc1ccc(Cl)cc1": ("1,4-dichlorobenzene", 0, 0),
            "C[C@@H](O)[C@H](C)O": ("meso-2,3-butanediol: two centres, the molecule achiral", 2, 0),
            "C/C=C/C": ("(E)-2-butene", 0, 1),
        }
        for smiles, (what, centres, doubles) in known.items():
            got = self.configured(smiles)
            kinds = [e["kind"] for e in got["elements"]]
            self.assertEqual((kinds.count("centre"), kinds.count("double")), (centres, doubles), what)
            # (from 3D, every element the geometry fixes is given)
            self.assertTrue(all(e["specified"] for e in got["elements"]), what)

    def test_keys_two_molecules_alike_exactly_when_they_are_one_stereoisomer(self):
        def key(smiles, seed=11):
            return self.worker.op_configurations({"molblocks": [self.made(smiles, seed)]})["molecules"][0]["key"]

        # (one stereoisomer, two different geometries: one key)
        self.assertEqual(key("C[C@H]1CC[C@@H](C)CC1", 11), key("C[C@H]1CC[C@@H](C)CC1", 23))
        # (cis and trans; R and S; E and Z; meso and R,R: each pair two keys)
        for a, b in [
            ("C[C@H]1CC[C@H](C)CC1", "C[C@H]1CC[C@@H](C)CC1"),
            ("C[C@@H](O)CC", "C[C@H](O)CC"),
            ("C/C=C/C", "C/C=C\\C"),
            ("C[C@@H](O)[C@H](C)O", "C[C@@H](O)[C@@H](C)O"),
        ]:
            self.assertNotEqual(key(a), key(b), (a, b))

    def test_says_why_for_a_molecule_it_cannot_read_and_answers_the_rest(self):
        good = self.made("CC(O)CC")
        out = self.worker.op_configurations({"molblocks": ["not a MOL block", good]})["molecules"]
        self.assertIn("error", out[0])
        self.assertIn("key", out[1])


@unittest.skipUnless(Chem, "RDKit is not installed here")
class StereoRequests(unittest.TestCase):
    """What the plugin gives of a structure's stereo, and makes of it as asked (`stereo_elements`, `stereoisomers`,
    `conformers`): every element RDKit perceives and how it is drawn, none left out; the stereoisomers of what is
    unassigned, those that can be made in 3D - a bridgehead its rings fix gives one - the C=N bonds named passed by,
    at most as many as asked, refused whole or cut short as asked; and each configuration Meno gives, made RDKit's
    tags and back. What is left open, and what is done with it, are Meno's (lib/chem/stereoOpen)."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("plugin_rdkit", WORKER)
        cls.worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.worker)

    def block(self, smiles, hydrogens=False, either=False):
        """`smiles` as a drawing: laid out in 2D, wedged where the SMILES gives a configuration - each double bond it
        gives none for drawn of either configuration, with `either`, as Meno writes one Generate made (CFG=2)."""
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if hydrogens:
            mol = Chem.AddHs(mol)
        AllChem.Compute2DCoords(mol)
        if either:
            for b in mol.GetBonds():
                if b.GetBondType() == Chem.BondType.DOUBLE and b.GetStereo() == Chem.BondStereo.STEREONONE and not b.IsInRing():
                    b.SetStereo(Chem.BondStereo.STEREOANY)
            return Chem.MolToV3KMolBlock(mol)
        return Chem.MolToMolBlock(mol)

    def open_of(self, block):
        """The centres and double bonds drawn without a configuration, or with one drawn as not known, by index."""
        found = self.worker.op_stereo_elements({"molblock": block})["elements"]
        left = [e for e in found if e["state"] != "specified"]
        return {"atoms": [e["at"] for e in left if e["kind"] == "centre"], "bonds": [e["at"] for e in left if e["kind"] == "double"]}

    def isomers(self, block, most=32, **asked):
        return self.worker.op_stereoisomers({"molblock": block, "open": self.open_of(block), "most": most, "inThreeD": True, **asked})

    def test_gives_every_element_and_how_it_is_drawn(self):
        # (butan-2-ol flat: its carbinol carbon not given; wedged: given; a wavy bond on it: drawn as not known)
        self.assertEqual(self.worker.op_stereo_elements({"molblock": self.block("CC(O)CC")}), {"read": True, "elements": [{"kind": "centre", "at": 1, "state": "unspecified"}]})
        self.assertEqual(self.worker.op_stereo_elements({"molblock": self.block("C[C@H](O)CC")})["elements"][0]["state"], "specified")
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("CC(O)CC")
        AllChem.Compute2DCoords(mol)
        mol.GetBondBetweenAtoms(1, 2).SetBondDir(Chem.BondDir.UNKNOWN)
        self.assertEqual(self.worker.op_stereo_elements({"molblock": Chem.MolToMolBlock(mol)})["elements"][0]["state"], "unknown")
        # (a double bond of either configuration, as Meno writes one; a C=N whose E/Z is none of its own among them - which
        #  is Meno's to say)
        got = self.worker.op_stereo_elements({"molblock": self.block("CN=C(N)NCC(O)C", either=True)})["elements"]
        self.assertEqual(sorted((e["kind"], e["at"], e["state"]) for e in got), [("centre", 6, "unspecified"), ("double", 1, "unknown")])
        # (its parts, one after another)
        two = self.worker.op_stereo_elements({"molblocks": [self.block("CCO"), self.block("CC(O)CC")]})["elements"]
        self.assertEqual([e["at"] for e in two], [4])

    def test_says_what_it_cannot_read_or_make_sense_of(self):
        self.assertEqual(self.worker.op_stereo_elements({"molblocks": ["not a MOL block"]}), {"read": False})
        five = Chem.RWMol()
        n = five.AddAtom(Chem.Atom(7))
        for _ in range(5):
            five.AddBond(n, five.AddAtom(Chem.Atom(6)), Chem.BondType.SINGLE)
        mol = five.GetMol()
        mol.UpdatePropertyCache(strict=False)
        self.assertEqual(self.worker.op_stereo_elements({"molblock": Chem.MolToMolBlock(mol)}), {"read": True, "problem": "SANITIZE_PROPERTIES"})

    def test_reads_a_centre_as_rdkit_does_from_3d(self):
        # (the tag `configured` gives a centre from the sign of its neighbours' volume is the one RDKit reads from 3D:
        #  each centre of each structure, embedded three ways, its neighbours named in their bonds' order and turned)
        import numpy as np
        from rdkit.Chem import AllChem

        for smiles in ["C[C@H](O)CC", "C[C@@H](O)CC", "F[C@](Cl)(Br)I", "N[C@@H](C)C(=O)O", "C1CC[C@H](C)[C@@H](O)C1", "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O"]:
            for seed in range(3):
                made = Chem.AddHs(Chem.MolFromSmiles(smiles))
                AllChem.EmbedMolecule(made, randomSeed=seed)
                mol = Chem.RemoveHs(made)
                Chem.AssignStereochemistryFrom3D(mol, replaceExistingTags=True)
                pos = mol.GetConformer().GetPositions()
                flat = Chem.Mol(mol)
                for a in flat.GetAtoms():
                    a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
                for a in mol.GetAtoms():
                    if a.GetChiralTag() not in (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
                        continue
                    ns = self.worker.neighbours_of(a)
                    named = ns[1:] + ns[:1] if seed == 1 else ns[::-1] if seed == 2 else ns
                    u, v, w = (pos[n] - pos[a.GetIdx()] for n in named[:3])
                    sense = 1 if u.dot(np.cross(v, w)) > 0 else -1
                    got = self.worker.configured(flat, {"centres": [{"atom": a.GetIdx(), "neighbours": named, "sense": sense}]})
                    self.assertEqual(got.GetAtomWithIdx(a.GetIdx()).GetChiralTag(), a.GetChiralTag(), (smiles, seed, a.GetIdx()))

    def test_gives_each_stereoisomer_its_configuration_which_makes_it_again(self):
        block = self.block("OC(C=CC)C=CC")
        got = self.isomers(block)
        self.assertEqual(got["free"], False)
        self.assertEqual(len(got["isomers"]), 4)
        base = self.worker.read(block)
        for one in got["isomers"]:
            self.assertEqual(Chem.MolToSmiles(self.worker.configured(base, one["configured"])), one["smiles"])
        self.assertEqual(len({i["smiles"] for i in got["isomers"]}), 4)

    def test_a_cage_or_bridged_ring_its_rings_fix_is_one_way_only(self):
        for name, smiles, centres in [
            ("adamantane", "C1C2CC3CC1CC(C2)C3", 4),
            ("cubane", "C12C3C4C1C5C2C3C45", 8),
            ("norbornane", "C1CC2CCC1C2", 2),
            ("bicyclo[1.1.0]butane", "C1C2C1C2", 2),
        ]:
            block = self.block(smiles)
            # (its bridgeheads are open, as drawn - and give one stereoisomer that can be made, so there is nothing to choose)
            self.assertEqual(len(self.open_of(block)["atoms"]), centres, name)
            got = self.isomers(block)
            self.assertEqual((len(got["isomers"]), got["free"]), (1, False), name)

    def test_what_could_go_more_than_one_way_is_counted_so(self):
        # (camphor unwedged: its two enantiomers; 5-methylnorbornene, the methyl alone wedged: exo or endo, through the bridgeheads)
        for name, smiles in [("camphor", "CC1(C)C2CCC1(C)C(=O)C2"), ("5-methylnorbornene", "C[C@H]1CC2C=CC1C2"), ("2-butanol", "CC(O)CC")]:
            self.assertEqual(len(self.isomers(self.block(smiles))["isomers"]), 2, name)
        # (the same, every centre wedged: nothing open)
        self.assertEqual(self.open_of(self.block("C[C@@H]1C[C@@H]2C=C[C@H]1C2")), {"atoms": [], "bonds": []})

    def test_an_allenes_double_bonds_are_given_as_double_bonds_and_not_enumerated(self):
        # (an allene's axis is not one RDKit's enumeration gives: one, its two double bonds open - which is why Meno says
        #  the cumulated ones)
        block = self.block("CC=C=CC")
        self.assertEqual(self.open_of(block), {"atoms": [], "bonds": [1, 2]})
        self.assertEqual(len(self.isomers(block)["isomers"]), 1)

    def test_a_cage_no_stereoisomer_of_which_embeds_is_answered_free(self):
        # (2-methylbicyclo[1.1.1]pentane: its bridgeheads and the methyl's carbon open, and no stereoisomer RDKit gives of
        #  them embeds with its stereo held - made all the same, as the embedding places it)
        for hydrogens in (False, True):
            got = self.isomers(self.block("C1C2CC1C2C", hydrogens))
            self.assertEqual((got["isomers"], got["free"]), ([], True), hydrogens)
        made = self.worker.op_conformers({**held(self.block("C1C2CC1C2C")), "count": 1})["isomers"]
        self.assertEqual(len(made), 1)
        self.assertTrue(made[0]["frames"])
        # (prismane: free too)
        self.assertTrue(self.isomers(self.block("C12C3C4C1C3C24"))["free"])

    def test_works_out_what_embeds_once_for_a_structure_asked_about_again(self):
        # (asked twice, as the page asks what to make and then makes it: tried in 3D once - the same answer, and copies)
        block = self.block("CC1=CCC2CC1C2(C)C")
        calls = []
        real = self.worker.EnumerateStereoisomers
        self.worker.EMBEDDABLE.clear()
        self.worker.EnumerateStereoisomers = lambda *a, **k: calls.append(1) or real(*a, **k)
        try:
            first, again = self.isomers(block), self.isomers(block)
        finally:
            self.worker.EnumerateStereoisomers = real
        self.assertEqual(len(calls), 1)
        self.assertEqual(first, again)
        mol = self.worker.read(block)
        self.worker.embeddable(mol, [], 32)[0].SetProp("asked", "by one")
        self.assertFalse(self.worker.embeddable(mol, [], 32)[0].HasProp("asked"))

    def test_a_c_n_named_is_passed_by_and_left_as_it_came(self):
        # (the guanidine a Generate made: 2 for its stereocentre, not 4 - its C=N, named, held while the rest are given)
        block = self.block("CN=C(N)NCC(O)c1ccc(OC(F)F)cc1", either=True)
        got = self.worker.op_stereoisomers({"molblock": block, "open": {"atoms": [6], "bonds": []}, "notEnumerated": [1], "most": 32, "inThreeD": True})
        self.assertEqual(sorted(i["smiles"] for i in got["isomers"]), ["CN=C(N)NC[C@@H](O)c1ccc(OC(F)F)cc1", "CN=C(N)NC[C@H](O)c1ccc(OC(F)F)cc1"])
        # (not named: given each way)
        self.assertEqual(len(self.worker.op_stereoisomers({"molblock": block, "open": {"atoms": [6], "bonds": [1]}, "most": 32, "inThreeD": True})["isomers"]), 4)
        # (and made in 3D with it as the embedding places it)
        made = self.worker.op_conformers({**held(block), "isomers": [i["configured"] for i in got["isomers"]], "count": 1})["isomers"]
        self.assertEqual(len(made), 2)

    def test_refuses_whole_past_the_most_without_making_them_all(self):
        got = self.isomers(self.block("CC(O)C(O)C(O)C(O)C(O)C(O)C"), whole=True)
        self.assertEqual(got, {"tooMany": {"counted": 33, "bound": 64}})
        # (not whole: the first so many)
        self.assertEqual(len(self.isomers(self.block("CC(O)C(O)C(O)C(O)C(O)C(O)C"), most=8)["isomers"]), 8)
        # (twenty stereocentres left open: a million stereoisomers, never each made to be counted - a Generate's
        #  glycolipid, twenty left open, timed the step out counting them all for some 165 s)
        started = time.monotonic()
        got = self.isomers(self.block("C" + "CC(O)" * 20 + "CN"), whole=True)
        self.assertLess(time.monotonic() - started, 20)
        self.assertEqual(got, {"tooMany": {"counted": 33, "bound": 1048576}})

    def test_answers_too_long_past_the_time_given_and_keeps_nothing_of_it(self):
        # (stereoisomers slow to try in 3D - made so here, a fifth of a second each - past the time given)
        worker = self.worker
        enumerate_as_is = worker.EnumerateStereoisomers

        def slow(mol, options=None):
            for isomer in enumerate_as_is(mol, options=options):
                time.sleep(0.2)
                yield isomer

        worker.EnumerateStereoisomers = slow
        try:
            got = self.isomers(self.block("CC(O)C(N)CC(F)CC"), withinMs=300)
        finally:
            worker.EnumerateStereoisomers = enumerate_as_is
        self.assertEqual(got, {"tooLong": True})
        # (not kept as worked out: given time, it is made whole)
        self.assertEqual(len(self.isomers(self.block("CC(O)C(N)CC(F)CC"), withinMs=60000)["isomers"]), 8)

    def test_gives_each_as_a_drawing_where_asked(self):
        got = self.isomers(self.block("CC(O)CC"), drawings=True)
        for one in got["isomers"]:
            self.assertEqual(Chem.MolToSmiles(Chem.MolFromMolBlock(one["molblock"])), one["smiles"])
        self.assertNotIn("molblock", self.isomers(self.block("CC(O)CC"))["isomers"][0])

    def test_makes_the_conformers_of_each_configuration_given(self):
        block = self.block("CC(O)C(N)C")
        got = self.isomers(block)
        made = self.worker.op_conformers({**held(block), "isomers": [i["configured"] for i in got["isomers"]], "count": 2})["isomers"]
        self.assertEqual([m["smiles"] for m in made], [i["smiles"] for i in got["isomers"]])
        self.assertEqual(len({tuple(sorted(m["cip"]["atoms"].items())) for m in made}), 4)
        # (none given: the structure as drawn)
        self.assertEqual(len(self.worker.op_conformers({**held(block), "count": 2})["isomers"]), 1)


@unittest.skipUnless(Chem, "RDKit is not installed here")
class StructureSteps(unittest.TestCase):
    """Steps on structures drawn (`op_run`), and the facts *Check structures* decides on: each structure kept as
    drawn, with what is worked out of it, or set aside saying why - never one changed."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("plugin_rdkit", WORKER)
        cls.worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.worker)

    def entry(self, smiles, radical_on=None):
        """A structure as Meno gives one: a MOL block for each of its parts, laid out in 2D, wedged as the SMILES says
        - an atom given an unpaired electron where `radical_on` says."""
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if radical_on is not None:
            mol = Chem.RWMol(mol)
            atom = mol.GetAtomWithIdx(radical_on)
            atom.SetNumRadicalElectrons(1)
            atom.SetNumExplicitHs(max(atom.GetTotalNumHs() - 1, 0))
            atom.SetNoImplicit(True)
            mol = mol.GetMol()
            Chem.SanitizeMol(mol)
        blocks = []
        for part in Chem.GetMolFrags(mol, asMols=True):
            AllChem.Compute2DCoords(part)
            # (RDKit lays a lone atom out nowhere - NaN - where Meno draws it somewhere: at the origin, here)
            conf = part.GetConformer()
            for i in range(part.GetNumAtoms()):
                if math.isnan(conf.GetAtomPosition(i).x):
                    conf.SetAtomPosition(i, (0.0, 0.0, 0.0))
            blocks.append(Chem.MolToMolBlock(part))
        return {"molblocks": blocks}

    def run_step(self, step, entries, options=None):
        return self.worker.op_run({"step": step, "entries": entries, "options": options or {}, "holds": "structures"})["structures"]

    def facts(self, entries):
        return self.worker.op_structure_facts({"structures": entries})["structures"]

    def test_gives_check_structures_the_facts_it_decides_on(self):
        # (Check structures is Meno's - workflow/checkStructures - on these: parts, an atom's charge and atomic number, unpaired electrons)
        ok, salt, proton, radical = self.facts([self.entry("CCO"), self.entry("C[NH3+].[Cl-]"), self.entry("[H+]"), self.entry("CC", radical_on=1)])
        self.assertEqual({a["part"] for a in ok["atoms"]}, {0})
        self.assertEqual(sorted({a["part"] for a in salt["atoms"]}), [0, 1])
        self.assertEqual([(a["z"], a["charge"]) for a in proton["atoms"]], [(1, 1)])
        self.assertEqual([a["radicals"] for a in radical["atoms"]], [0, 1])
        self.assertFalse(any(a["radicals"] for a in ok["atoms"]))

    def test_gives_an_atom_with_more_bonds_than_it_can_have_its_valence_and_most(self):
        five = Chem.RWMol()
        n = five.AddAtom(Chem.Atom(7))
        for _ in range(5):
            c = five.AddAtom(Chem.Atom(6))
            five.AddBond(n, c, Chem.BondType.SINGLE)
        from rdkit.Chem import AllChem

        mol = five.GetMol()
        mol.UpdatePropertyCache(strict=False)
        AllChem.Compute2DCoords(mol)
        [got] = self.facts([{"molblocks": [Chem.MolToMolBlock(mol)]}])
        self.assertEqual((got["atoms"][0]["el"], got["atoms"][0]["valence"], got["atoms"][0]["most"]), ("N", 5, 3))

    def test_descriptors_of_each_as_drawn_with_unit_and_how(self):
        got = self.run_step("descriptors", [self.entry("CC(=O)Nc1ccc(O)cc1"), self.entry("[H+]")])
        props = got[0]["properties"]
        # (paracetamol's, as RDKit 2026.03.6 works them out - its weight and polar surface area the published ones)
        self.assertEqual({k: (v["value"], v["unit"]) for k, v in props.items()}, {
            "mw": (151.165, "g/mol"), "logp": (1.3506, "none"), "tpsa": (49.33, "Å2"), "hbd": (2, "none"), "hba": (2, "none"), "rotatable": (1, "none"),
        })
        self.assertTrue(all(v["method"] for v in props.values()))
        self.assertEqual(got[0]["as"], "drawn")
        # (a lone proton, read as drawn: its weight and the rest, none chosen for it)
        self.assertEqual(got[1]["properties"]["mw"]["value"], 1.008)

    def test_descriptors_of_a_lone_ion_leave_out_its_logp_saying_why(self):
        # (a lone atom or ion as Meno gives one - its valence written as none: Na+, Cl-, a sodium atom - has no logP of
        #  its own, Crippen's typing giving every one the same; its weight as any. Methane, a lone C with its H, has one)
        def lone(el, charge):
            chg = f" CHG={charge}" if charge else ""
            return {"molblocks": [
                "\n  Meno\n\n  0  0  0     0  0            999 V3000\nM  V30 BEGIN CTAB\nM  V30 COUNTS 1 0 0 0 0\nM  V30 BEGIN ATOM\n"
                f"M  V30 1 {el} 0.0000 0.0000 0 0{chg} VAL=-1\nM  V30 END ATOM\nM  V30 END CTAB\nM  END\n"
            ]}
        got = self.run_step("descriptors", [lone("Na", 1), lone("Cl", -1), lone("Na", 0), self.entry("C")])
        for g, mw in zip(got[:3], [22.99, 35.453, 22.99]):
            props = g["properties"]
            self.assertNotIn("logp", props)
            self.assertEqual(props["not_given"]["value"], "logp: not defined for a lone ion")
            self.assertEqual(props["mw"]["value"], mw)
            self.assertEqual((props["tpsa"]["value"], props["hbd"]["value"]), (0.0, 0))
        self.assertIn("logp", got[3]["properties"])
        self.assertNotIn("not_given", got[3]["properties"])

@unittest.skipUnless(Chem, "RDKit is not installed here")
class SmilesStereoOpen(unittest.TestCase):
    """A SMILES read (`from_smiles`), and the elements its drawing gives no configuration (`stereo_elements`): what a
    paste of it is told is Meno's rule on these (lib/chem/stereoOpen), as a step refusing a structure for it has it."""

    #: Paclitaxel, every one of its eleven stereocentres given (as PubChem's isomeric SMILES of CID 36314 gives them).
    PACLITAXEL = (
        "CC1=C2[C@H](C(=O)[C@@]3([C@H](C[C@@H]4[C@]([C@H]3[C@@H]([C@@](C2(C)C)(C[C@@H]1OC(=O)[C@@H]([C@H](C5=CC=CC=C5)"
        "NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C"
    )

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("plugin_rdkit", WORKER)
        cls.worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.worker)

    def left(self, smiles):
        block = self.worker.op_from_smiles({"smiles": smiles})["molblock"]
        got = [e for e in self.worker.op_stereo_elements({"molblock": block})["elements"] if e["state"] != "specified"]
        return sum(e["kind"] == "centre" for e in got), sum(e["kind"] == "double" for e in got)

    def test_paclitaxel_given_whole_leaves_none_open_and_one_mark_short_leaves_one(self):
        self.assertEqual(self.left(self.PACLITAXEL), (0, 0))
        # (the side chain's phenyl-bearing centre given no configuration: ten of eleven, as a stereo-incomplete record has it)
        self.assertEqual(self.left(self.PACLITAXEL.replace("[C@H](C5=CC=CC=C5)", "C(C5=CC=CC=C5)")), (1, 0))

    def test_a_double_bond_given_no_configuration_is_one_left_open_and_nothing_stereogenic_none(self):
        self.assertEqual(self.left("CC=CC(=O)O"), (0, 1))
        self.assertEqual(self.left("C/C=C/C(=O)O"), (0, 0))
        self.assertEqual(self.left("CCO"), (0, 0))

@unittest.skipUnless(Chem, "RDKit is not installed here")
class CoordinationBonds(unittest.TestCase):
    """A coordination bond - a MOL block's type 9, a plugin entry's `coordination` - is RDKit's dative bond, taking no
    hydrogen from either end as Meno counts them; where RDKit cannot hold it, refused in words - never a hydrogen
    short, never a covalent single."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("plugin_rdkit", WORKER)
        cls.worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.worker)

    @staticmethod
    def block(atoms, bonds):
        """A V3000 MOL block as Meno writes one: atoms (element, x, y, and a charge or none), bonds (type, from 1, to 1)."""
        lines = ["", "  Meno", "", "  0  0  0     0  0            999 V3000", "M  V30 BEGIN CTAB", f"M  V30 COUNTS {len(atoms)} {len(bonds)} 0 0 0", "M  V30 BEGIN ATOM"]
        lines += [f"M  V30 {i + 1} {el} {x} {y} 0 0{f' CHG={c}' if c else ''}" for i, (el, x, y, c) in enumerate(atoms)]
        lines += ["M  V30 END ATOM", "M  V30 BEGIN BOND"]
        lines += [f"M  V30 {i + 1} {t} {a} {b}" for i, (t, a, b) in enumerate(bonds)]
        lines += ["M  V30 END BOND", "M  V30 END CTAB", "M  END", ""]
        return "\n".join(lines)

    def formula(self, molblock):
        iso = self.worker.op_conformers({**held(molblock), "count": 2})["isomers"][0]
        return "".join(sorted(a["el"] for a in iso["atoms"]))

    def made(self, molblock):
        return self.worker.op_conformers({**held(molblock), "count": 2})["isomers"][0]

    def test_a_group_13_adduct_drawn_with_an_arrow_is_held_in_its_dipolar_form_its_charges_reported_as_drawn(self):
        # (H3N->BH3 as plain atoms: before 0.1.12 RDKit gave the B two hydrogens - H3N->BH2 - and 0.1.12 refused it)
        iso = self.made(self.block([("N", 0, 0, 0), ("B", 1.6, 0, 0)], [(9, 1, 2)]))
        self.assertEqual("".join(sorted(a["el"] for a in iso["atoms"])), "BHHHHHHN")
        self.assertEqual([a["charge"] for a in iso["atoms"]], [0] * 8)
        self.assertIn({"label": "Coordination bonds", "text": "N\u2192B computed as a single bond, N+ and B\u2212"}, iso["how"])
        n, b = iso["frames"][0][0:3], iso["frames"][0][3:6]
        self.assertTrue(1.5 < math.dist(n, b) < 1.8, math.dist(n, b))
        # (an alane, an ether's and a ketone's BF3, and Al2Cl6 drawn with its two bridges as arrows: each made, all its atoms)
        for name, atoms, bonds, formula in [
            ("Me3N->AlCl3", [("N", 0, 0, 0), ("Al", 2, 0, 0), ("Cl", 3, 1, 0), ("Cl", 3, -1, 0), ("Cl", 2, 2, 0), ("C", -1, 1, 0), ("C", -1, -1, 0), ("C", 0, 1.5, 0)],
             [(9, 1, 2), (1, 2, 3), (1, 2, 4), (1, 2, 5), (1, 1, 6), (1, 1, 7), (1, 1, 8)], "AlCCCClClClHHHHHHHHHN"),
            ("Me2O->BF3", [("O", 0, 0, 0), ("B", 1.6, 0, 0), ("F", 2.5, 1, 0), ("F", 2.5, -1, 0), ("F", 1.6, 1.5, 0), ("C", -1, 1, 0), ("C", -1, -1, 0)],
             [(9, 1, 2), (1, 2, 3), (1, 2, 4), (1, 2, 5), (1, 1, 6), (1, 1, 7)], "BCCFFFHHHHHHO"),
            ("Me2C=O->BF3", [("O", 0, 0, 0), ("B", 1.6, 0, 0), ("F", 2.5, 1, 0), ("F", 2.5, -1, 0), ("F", 1.6, 1.5, 0), ("C", -1.2, 0, 0), ("C", -2, 1, 0), ("C", -2, -1, 0)],
             [(9, 1, 2), (1, 2, 3), (1, 2, 4), (1, 2, 5), (2, 1, 6), (1, 6, 7), (1, 6, 8)], "BCCCFFFHHHHHHO"),
            ("Al2Cl6", [("Al", 0, 0, 0), ("Al", 3, 0, 0), ("Cl", 1.5, 1, 0), ("Cl", 1.5, -1, 0), ("Cl", -1, 1, 0), ("Cl", -1, -1, 0), ("Cl", 4, 1, 0), ("Cl", 4, -1, 0)],
             [(1, 1, 3), (9, 3, 2), (1, 2, 4), (9, 4, 1), (1, 1, 5), (1, 1, 6), (1, 2, 7), (1, 2, 8)], "AlAlClClClClClCl"),
        ]:
            iso = self.made(self.block(atoms, bonds))
            self.assertEqual("".join(sorted(a["el"] for a in iso["atoms"])), formula, name)
            self.assertTrue(all(a["charge"] == 0 for a in iso["atoms"]), name)
        # (two pairs, said together: Al2Cl6's two Cl->Al bridges are one pair of elements, so one; an ammine and an ether onto two B, two)
        self.assertIn({"label": "Coordination bonds", "text": "Cl\u2192Al computed as a single bond, Cl+ and Al\u2212"}, iso["how"])
        two = self.made(self.block([("N", 0, 0, 0), ("B", 1.6, 0, 0), ("O", 6, 0, 0), ("B", 7.6, 0, 0), ("C", -1, 1, 0), ("C", 5, 1, 0), ("C", 5, -1, 0)],
                                   [(9, 1, 2), (9, 3, 4), (1, 1, 5), (1, 3, 6), (1, 3, 7)]))
        self.assertIn({"label": "Coordination bonds", "text": "N\u2192B, O\u2192B computed as single bonds, each donor + and acceptor \u2212"}, two["how"])

    def test_every_op_that_reads_a_drawing_reads_an_adduct_the_same_way(self):
        # (H3N->BH3 drawn with an arrow: the facts Check structures decides on give no atom more bonds than it can have,
        #  nothing RDKit cannot make sense of - Descriptors works it out, and the canvas's analysis marks no valence, as
        #  Conformers makes it)
        adduct = self.block([("N", 0, 0, 0), ("B", 1.6, 0, 0)], [(9, 1, 2)])
        [facts] = self.worker.op_structure_facts({"structures": [{"molblocks": [adduct]}]})["structures"]
        self.assertTrue(facts["read"])
        self.assertNotIn("problem", facts)
        self.assertFalse(any(a["valence"] > a.get("most", a["valence"]) for a in facts["atoms"]))
        described = self.worker.step_descriptors([{"molblocks": [adduct]}], {})["structures"][0]
        self.assertEqual(described["as"], "drawn")
        # (its weight: B 10.81, N 14.007, six H 1.008 each - the same in either form)
        self.assertAlmostEqual(described["properties"]["mw"]["value"], 10.81 + 14.007 + 6 * 1.008, delta=0.01)
        seen = self.worker.op_analyse({"molblock": adduct})
        self.assertFalse(any("valenceError" in a for a in seen["atoms"]))
        self.assertEqual([a["hydrogens"] for a in seen["atoms"]], [3, 3])
        # (one it cannot hold gives Check structures an atom with more bonds than it can have, as Conformers refuses it)
        silicon = self.block([("N", 0, 0, 0), ("Si", 2, 0, 0), ("F", 3, 1, 0), ("F", 3, -1, 0), ("F", 2, 2, 0), ("F", 2, -2, 0)], [(9, 1, 2), (1, 2, 3), (1, 2, 4), (1, 2, 5), (1, 2, 6)])
        [facts] = self.worker.op_structure_facts({"structures": [{"molblocks": [silicon]}]})["structures"]
        self.assertTrue(any(a["valence"] > a["most"] for a in facts["atoms"] if "most" in a) or "problem" in facts)

    def isomers_drawn(self, block):
        return self.worker.op_stereoisomers({"molblock": block, "open": {"atoms": [0], "bonds": []}, "most": 32, "inThreeD": True, "drawings": True})["isomers"]

    def test_stereoisomers_of_an_adduct_are_drawn_with_its_arrow_and_its_charges_as_drawn(self):
        # (1-fluoroethylamine->BH3, its carbon a stereocentre left open: two stereoisomers, each written back as drawn - the
        #  N->B arrow, N and B uncharged - and each reading back as the one its SMILES, of the dipolar form, says)
        adduct = self.block([("C", 0, 0, 0), ("F", 0, 1.5, 0), ("C", -1.3, -0.75, 0), ("N", 1.3, -0.75, 0), ("B", 2.6, 0, 0)], [(1, 1, 2), (1, 1, 3), (1, 1, 4), (9, 4, 5)])
        made = self.isomers_drawn(adduct)
        self.assertEqual(len(made), 2)
        self.assertEqual(len({i["smiles"] for i in made}), 2)
        for block, smiles in ((i["molblock"], i["smiles"]) for i in made):
            drawn = Chem.MolFromMolBlock(block, sanitize=False, removeHs=False)
            arrows = [b for b in drawn.GetBonds() if b.GetBondType() == Chem.BondType.DATIVE]
            self.assertEqual([(b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()) for b in arrows], [("N", "B")])
            self.assertTrue(all(a.GetFormalCharge() == 0 for a in drawn.GetAtoms()))
            self.assertRegex(smiles, r"\[NH2\+\]")
            self.assertRegex(smiles, r"\[BH3-\]")
            self.assertEqual(Chem.MolToSmiles(self.worker.read(block)), smiles)

    def test_an_adduct_renumbered_since_is_never_written_without_its_arrow(self):
        # (H3N->BH3 held in its dipolar form, then renumbered - its B and N swapped - so that the index its N keeps of
        #  its acceptor no longer names one bonded to it: refused in words - each stereoisomer said so, which Stereoisomers sets aside)
        mol = self.worker.read(self.block([("N", 0, 0, 0), ("B", 1.6, 0, 0)], [(9, 1, 2)]))
        renumbered = Chem.RenumberAtoms(mol, [1, 0] + list(range(2, mol.GetNumAtoms())))
        self.assertEqual(renumbered.GetAtomWithIdx(1).GetIntProp(self.worker.LENDS_TO), 1)
        with self.assertRaisesRegex(ValueError, r"^its coordination bond from N could not be written back as drawn$"):
            self.worker.as_drawn(renumbered)
        # (and an index past its atoms)
        renumbered.GetAtomWithIdx(1).SetIntProp(self.worker.LENDS_TO, 99)
        with self.assertRaisesRegex(ValueError, r"^its coordination bond from N could not be written back as drawn$"):
            self.worker.as_drawn(renumbered)
        # (and one naming an atom bonded to the donor that is not its acceptor - one of the N's own hydrogens: never that
        #  N-H written as the arrow)
        withH = Chem.AddHs(self.worker.read(self.block([("N", 0, 0, 0), ("B", 1.6, 0, 0)], [(9, 1, 2)])))
        n = next(x for x in withH.GetAtoms() if x.GetSymbol() == "N")
        h = next(x.GetIdx() for x in n.GetNeighbors() if x.GetSymbol() == "H")
        arrows = lambda m: [(x.GetBeginAtom().GetSymbol(), x.GetEndAtom().GetSymbol()) for x in m.GetBonds() if x.GetBondType() == Chem.BondType.DATIVE]
        self.assertEqual(arrows(self.worker.as_drawn(withH)), [("N", "B")])
        n.SetIntProp(self.worker.LENDS_TO, h)
        with self.assertRaisesRegex(ValueError, r"^its coordination bond from N could not be written back as drawn$"):
            self.worker.as_drawn(withH)
        from unittest import mock
        adduct = self.block([("C", 0, 0, 0), ("F", 0, 1.5, 0), ("C", -1.3, -0.75, 0), ("N", 1.3, -0.75, 0), ("B", 2.6, 0, 0)], [(1, 1, 2), (1, 1, 3), (1, 1, 4), (9, 4, 5)])
        with mock.patch.object(self.worker, "as_drawn", side_effect=ValueError("its coordination bond from N could not be written back as drawn")):
            self.assertEqual({i.get("error") for i in self.isomers_drawn(adduct)}, {"its coordination bond from N could not be written back as drawn"})

    def test_its_smiles_is_of_the_dipolar_form_the_same_molecule(self):
        adduct = self.block([("N", 0, 0, 0), ("B", 1.6, 0, 0)], [(9, 1, 2)])
        self.assertEqual(self.worker.op_to_smiles({"molblock": adduct})["smiles"], Chem.MolToSmiles(Chem.MolFromSmiles("[NH3+][BH3-]")))

    def test_a_main_group_adduct_it_cannot_hold_is_refused_in_words_never_a_hydrogen_short(self):
        for atoms, bonds, why in [
            # (Si, not of group 13; a charged acceptor; a charged donor; a donor that cannot carry the charge; two arrows onto one B)
            ([("N", 0, 0, 0), ("Si", 2, 0, 0), ("F", 3, 1, 0), ("F", 3, -1, 0), ("F", 2, 2, 0), ("F", 2, -2, 0)], [(9, 1, 2), (1, 2, 3), (1, 2, 4), (1, 2, 5), (1, 2, 6)], "Si is not one of B, Al, Ga, In and Tl"),
            ([("N", 0, 0, 0), ("B", 1.6, 0, -1)], [(9, 1, 2)], "B is drawn charged"),
            ([("N", 0, 0, 1), ("B", 1.6, 0, 0)], [(9, 1, 2)], "N is drawn charged"),
            ([("C", 0, 0, 0), ("B", 1.6, 0, 0)], [(9, 1, 2)], "C cannot carry the charge"),
            ([("N", 0, 0, 0), ("B", 1.6, 0, 0), ("N", 3.2, 0, 0)], [(9, 1, 2), (9, 3, 2)], "B takes 2 of them"),
        ]:
            with self.assertRaisesRegex(ValueError, "^RDKit cannot hold the coordination bond from [A-Z][a-z]? to [A-Z][a-z]? as drawn: " + why + r"\.$"):
                self.made(self.block(atoms, bonds))

    def test_a_metal_acceptor_keeps_its_ligands_hydrogens(self):
        # (cis-[PtCl2(NH3)2], each ammine's bond to Pt an arrow: six hydrogens, none taken)
        cisplatin = self.block([("Pt", 0, 0, 0), ("Cl", 2, 0, 0), ("Cl", 0, 2, 0), ("N", -2, 0, 0), ("N", 0, -2, 0)], [(1, 1, 2), (1, 1, 3), (9, 4, 1), (9, 5, 1)])
        self.assertEqual(self.formula(cisplatin), "ClClHHHHHHNNPt")

    def test_a_plugin_entrys_coordination_bond_is_dative_never_a_covalent_single(self):
        # (H3N->BH3 with all its hydrogens, its B-N bond marked: its dipolar form, the charges as drawn kept; a Pt ammine's N->Pt: dative)
        h = lambda x, k: {"el": "H", "x": x, "y": 0.9 * (k - 1), "z": 0.5}
        borane = {
            "atoms": [{"el": "N", "x": 0, "y": 0, "z": 0}, {"el": "B", "x": 1.66, "y": 0, "z": 0}] + [h(-0.4, k) for k in range(3)] + [h(2.1, k) for k in range(3)],
            "bonds": [{"a1": 0, "a2": 1, "order": 1, "coordination": True}] + [{"a1": 0, "a2": 2 + k, "order": 1} for k in range(3)] + [{"a1": 1, "a2": 5 + k, "order": 1} for k in range(3)],
        }
        held = self.worker.mol_of(borane)
        bond = held.GetBondBetweenAtoms(0, 1)
        self.assertEqual((bond.GetBondType(), held.GetAtomWithIdx(0).GetFormalCharge(), held.GetAtomWithIdx(1).GetFormalCharge()), (Chem.BondType.SINGLE, 1, -1))
        self.assertEqual([self.worker.drawn_charge(a) for a in held.GetAtoms()][:2], [0, 0])
        ammine = {
            "atoms": [{"el": "Pt", "x": 0, "y": 0, "z": 0}, {"el": "N", "x": 2.05, "y": 0, "z": 0}] + [h(2.4, k) for k in range(3)] + [{"el": "Cl", "x": -2.3, "y": 0, "z": 0}],
            "bonds": [{"a1": 1, "a2": 0, "order": 1, "coordination": True}] + [{"a1": 1, "a2": 2 + k, "order": 1} for k in range(3)] + [{"a1": 0, "a2": 5, "order": 1}],
        }
        mol = self.worker.mol_of(ammine)
        bond = mol.GetBondBetweenAtoms(0, 1)
        self.assertEqual((bond.GetBondType(), bond.GetBeginAtomIdx()), (Chem.BondType.DATIVE, 1))
        self.assertEqual(mol.GetAtomWithIdx(1).GetTotalNumHs(includeNeighbors=True), 3)


@unittest.skipUnless(Chem, "RDKit is not installed here")
class WhereMmffHasNoParameters(unittest.TestCase):
    """Where MMFF94 or MMFF94s is asked and has none for the molecule, UFF stands in, said in how it was made - or,
    where the chemist chose it (`fallback`: refuse), the molecule is refused, saying why, before anything is made.
    Trimethylborane: MMFF94 has no type for its boron, UFF has one (both as RDKit's own checks say)."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("plugin_rdkit", WORKER)
        cls.worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.worker)

    def borane(self):
        from rdkit.Chem import AllChem, rdForceFieldHelpers
        mol = Chem.AddHs(Chem.MolFromSmiles("CB(C)C"))
        self.assertFalse(rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol))
        self.assertTrue(rdForceFieldHelpers.UFFHasAllMoleculeParams(mol))
        AllChem.EmbedMolecule(mol, randomSeed=11)
        return mol

    def test_takes_uff_in_its_place_unless_told_to_refuse(self):
        made = self.worker.conformers(Chem.MolFromSmiles("CB(C)C"), count=3)
        self.assertEqual(made["field"], "UFF")
        self.assertIn({"label": "Optimised", "text": "UFF, at most 2000 steps"}, made["how"])
        self.assertEqual(self.worker.minimised(self.borane(), [])["field"], "UFF")

    def test_refuses_where_told_to_saying_why(self):
        with self.assertRaisesRegex(ValueError, "MMFF94 has no parameters for it, and UFF in its place was not chosen"):
            self.worker.conformers(Chem.MolFromSmiles("CB(C)C"), count=3, fallback="refuse")
        with self.assertRaisesRegex(ValueError, "MMFF94s has no parameters for it"):
            self.worker.minimised(self.borane(), [], field="MMFF94s", fallback="refuse")
        # (a molecule MMFF has parameters for, and UFF asked for itself: as before, whatever the choice)
        self.assertEqual(self.worker.conformers(Chem.MolFromSmiles("CCO"), count=2, fallback="refuse")["field"], "MMFF94")
        self.assertEqual(self.worker.conformers(Chem.MolFromSmiles("CB(C)C"), count=2, field="UFF", fallback="refuse")["field"], "UFF")

    def test_refuses_an_eta_complex_where_told_to_its_star_no_atom_mmff_types(self):
        with self.assertRaisesRegex(ValueError, "MMFF94 has no parameters for it"):
            self.worker.op_conformers({**held(EtaBound.ferrocene()), "count": 2, "options": {"fallback": "refuse"}})

    def test_is_a_choice_of_the_search_and_of_minimise_uff_unless_chosen_otherwise(self):
        import json
        manifest = json.loads((WORKER.parent / "manifest.json").read_text(encoding="utf-8"))
        given = [o for r in manifest["roles"] + manifest["steps"] for o in r.get("options") or [] if o["id"] == "fallback"]
        self.assertEqual(len(given), 2)
        for o in given:
            self.assertEqual((o["type"], [c["value"] for c in o["choices"]], o["default"]), ("choice", list(self.worker.FALLBACKS), "UFF"))
        [minimise] = [s for s in manifest["steps"] if s["kinds"] == ["minimise"]]
        self.assertIn("fallback", [o["id"] for o in minimise["options"]])

    def test_a_minimise_job_refuses_where_told_to(self):
        import json, subprocess, sys, tempfile
        mol = self.borane()
        entry = {
            "atoms": [{"el": a.GetSymbol(), "x": p[0], "y": p[1], "z": p[2]} for a, p in zip(mol.GetAtoms(), mol.GetConformer().GetPositions())],
            "bonds": [{"a1": b.GetBeginAtomIdx(), "a2": b.GetEndAtomIdx(), "order": 1} for b in mol.GetBonds()],
        }
        for fallback, ok in (("UFF", True), ("refuse", False)):
            [job] = self.worker.op_prepare({"step": "minimise", "entries": [entry], "options": {"field": "MMFF94", "steps": 50, "fallback": fallback}})["jobs"]
            with tempfile.TemporaryDirectory() as here:
                with open(f"{here}/{job['files'][0]['name']}", "w", encoding="utf-8") as f:
                    f.write(job["files"][0]["text"])
                ran = subprocess.run([sys.executable, *job["args"]], cwd=here, capture_output=True, text=True)
            self.assertEqual(ran.returncode == 0, ok, ran.stderr)
            self.assertIn("UFF:" if ok else "MMFF94 has no parameters for it", ran.stdout if ok else ran.stderr)

    def test_reads_the_choice_from_the_options_uff_unless_refuse(self):
        self.assertEqual(self.worker.options_of({"options": {"fallback": "refuse"}})["fallback"], "refuse")
        self.assertEqual(self.worker.options_of({"options": {"fallback": "something"}})["fallback"], "UFF")
        self.assertEqual(self.worker.options_of({})["fallback"], "UFF")


@unittest.skipUnless(Chem, "RDKit is not installed here")
class MinimiseWithRestraints(unittest.TestCase):
    """A molecule minimised with its restraints held, each by the energy the entry record states (src/lib/entry.ts):
    a distance by k d^2 / 2 (kcal mol-1 A-2), a dihedral by k phi^2 / 2 (phi in radians, kcal mol-1 rad-2), atoms
    fixed where they are - known answers on butane, its torsion and its ends' distance set by hand."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("plugin_rdkit", WORKER)
        cls.worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.worker)

    def butane(self, dihedral):
        from rdkit.Chem import AllChem, rdMolTransforms
        mol = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
        AllChem.EmbedMolecule(mol, randomSeed=11)
        rdMolTransforms.SetDihedralDeg(mol.GetConformer(), 0, 1, 2, 3, dihedral)
        return mol

    def dihedral(self, frame):
        import numpy as np
        P = np.array(frame).reshape(-1, 3)[:4]
        b1, b2, b3 = P[1] - P[0], P[2] - P[1], P[3] - P[2]
        n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
        return math.degrees(math.atan2(np.linalg.norm(b2) * np.dot(b1, n2), np.dot(n1, n2)))

    def test_holds_each_restraint_by_the_energy_the_record_states(self):
        # (known answers, by hand: at a geometry set off from what is held, the force field with the restraint less
        #  the force field without it is k (d - value)^2 / 2 for a distance, k (phi - value)^2 / 2 for a dihedral, phi
        #  in radians the shorter way round - across +-180 too - in MMFF94 and in UFF alike)
        from rdkit.Chem import rdMolTransforms
        mol = self.butane(-170.0)
        conf = mol.GetConformer()
        places = [v for p in conf.GetPositions() for v in p]
        ends = rdMolTransforms.GetBondLength(conf, 0, 3)
        for field in ("MMFF94", "UFF"):
            used, make = self.worker.restrained_field(mol, [], field)
            self.assertEqual(used, field)
            free = make(-1, ()).CalcEnergy(places)
            held = lambda rows: make(-1, rows).CalcEnergy(places) - free
            self.assertAlmostEqual(held([{"kind": "distance", "atoms": [0, 3], "value": ends + 0.3, "strength": 500.0}]), 500.0 * 0.3**2 / 2, delta=1e-6)
            self.assertAlmostEqual(held([{"kind": "distance", "atoms": [0, 3], "value": ends - 0.2, "strength": 37.0}]), 37.0 * 0.2**2 / 2, delta=1e-6)
            # (-170 against 175: 15 degrees the shorter way, across +-180, not 345)
            for value, off in ((175.0, 15.0), (180.0, 10.0), (-150.0, 20.0)):
                self.assertAlmostEqual(held([{"kind": "dihedral", "atoms": [0, 1, 2, 3], "value": value, "strength": 50.0}]), 50.0 * math.radians(off) ** 2 / 2, delta=1e-6)
            # (both at once: their energies add)
            both = [{"kind": "distance", "atoms": [0, 3], "value": ends + 0.3, "strength": 500.0}, {"kind": "dihedral", "atoms": [0, 1, 2, 3], "value": 175.0, "strength": 50.0}]
            self.assertAlmostEqual(held(both), 500.0 * 0.3**2 / 2 + 50.0 * math.radians(15.0) ** 2 / 2, delta=1e-6)

    def test_says_how_far_it_missed_what_its_restraints_hold(self):
        # (a torsion held weakly misses: the largest miss is given back, measured on what it reached, and as results)
        made = self.worker.minimised(self.butane(60.0), [{"kind": "dihedral", "atoms": [0, 1, 2, 3], "value": 180.0, "strength": 2.0}])
        miss = 180 - abs(self.dihedral(made["frame"]))
        self.assertIsNone(made["missed"][0])
        self.assertAlmostEqual(made["missed"][1], miss, delta=1e-3)
        self.assertGreater(miss, 1.0)
        self.assertEqual(self.worker.missed_results(made["missed"]), [{"id": "restraint-torsion-missed", "on": "molecule", "group": "Restraints", "label": "Largest torsion miss", "value": round(miss, 2), "quantity": "angle"}])
        self.assertEqual(self.worker.missed_words([0.0123, 1.26]), "missed by at most 0.012 \u00c5 and 1.3\u00b0")
        self.assertEqual(self.worker.missed_words([None, None]), "")

    def test_holds_a_torsion_at_180_where_it_would_stay_gauche(self):
        free = self.worker.minimised(self.butane(60.0), [])
        self.assertLess(abs(self.dihedral(free["frame"])), 90)
        held = self.worker.minimised(self.butane(60.0), [{"kind": "dihedral", "atoms": [0, 1, 2, 3], "value": 180.0, "strength": 100.0}])
        self.assertGreater(abs(self.dihedral(held["frame"])), 175)
        self.assertTrue(held["converged"])
        self.assertEqual(held["field"], "MMFF94")

    def test_holds_a_distance_and_fixes_atoms_and_gives_the_force_fields_own_energy(self):
        from rdkit.Chem import rdForceFieldHelpers
        mol = self.butane(180.0)
        before = mol.GetConformer().GetPositions()
        made = self.worker.minimised(mol, [{"kind": "distance", "atoms": [0, 3], "value": 3.2, "strength": 500.0}, {"kind": "fixed", "atoms": [0]}])
        P = [made["frame"][3 * i : 3 * i + 3] for i in range(mol.GetNumAtoms())]
        self.assertAlmostEqual(math.dist(P[0], P[3]), 3.2, delta=0.05)
        # (where it stood, to the 1e-5 A the frame is written to)
        self.assertLess(math.dist(P[0], list(before[0])), 1e-4)
        # (its energy the force field's own at the places reached, the restraints' not in it)
        plain = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, rdForceFieldHelpers.MMFFGetMoleculeProperties(mol), ignoreInterfragInteractions=False)
        self.assertAlmostEqual(made["energy"], plain.CalcEnergy(made["frame"]) / self.worker.KCAL_PER_HARTREE, places=6)

    def test_starts_from_where_its_restraints_were_first_held_where_given(self):
        gauche, anti = self.butane(60.0), self.butane(180.0)
        start = [round(v, 6) for p in gauche.GetConformer().GetPositions() for v in p]
        made = self.worker.minimised(anti, [], start=start)
        self.assertLess(abs(self.dihedral(made["frame"])), 90)

    def test_refuses_in_words_a_restraint_it_cannot_hold(self):
        with self.assertRaisesRegex(ValueError, r"^Restraint 1 names an atom the molecule does not have$"):
            self.worker.restraint_rows([{"kind": "distance", "atoms": [0, 99], "value": 2.0, "strength": 1.0}], 14)
        with self.assertRaisesRegex(ValueError, r"^Restraint 2, a dihedral, has no value or strength to hold it by$"):
            self.worker.restraint_rows([{"kind": "fixed", "atoms": [0]}, {"kind": "dihedral", "atoms": [0, 1, 2, 3], "value": 180.0}], 14)

    def test_a_conformer_search_holds_its_restraints_where_there_is_no_metal(self):
        rows = [{"kind": "dihedral", "atoms": [0, 1, 2, 3], "value": 180.0, "strength": 100.0}]
        made = self.worker.conformers(Chem.MolFromSmiles("CCCC"), count=10, rows=rows)
        self.assertTrue(all(abs(self.dihedral(f)) > 175 for f in made["frames"]))
        # (the row says how far the conformers kept missed it at most: the largest of theirs, measured here)
        worst = max(180 - abs(self.dihedral(f)) for f in made["frames"])
        self.assertIn({"label": "Held", "text": f"1 torsion held, the energies without them; missed by at most {worst:.1f}\u00b0"}, made["how"])
        # (refused before anything is embedded: they turn on the restraints and the molecule alone)
        from unittest import mock
        with mock.patch.object(self.worker.rdDistGeom, "EmbedMultipleConfs", side_effect=AssertionError("embedded")):
            with self.assertRaisesRegex(ValueError, r"^A conformer search moves every atom, so it holds no atoms fixed"):
                self.worker.conformers(Chem.MolFromSmiles("CCCC"), count=2, rows=[{"kind": "fixed", "atoms": [0]}])
            with self.assertRaisesRegex(ValueError, r"^A conformer search holds restraints only where the molecule has no metal"):
                self.worker.conformers(Chem.MolFromSmiles("N[Pt](N)(Cl)Cl"), count=2, rows=rows)

    def test_a_search_embeds_a_distance_held_at_its_value(self):
        # (pyridine's N and BH3's B, two parts, held at 0.9 of Alvarez's radii together, 3.21 A: laid where ETKDG
        #  lays parts, a hold never brought them together - 17 A off at any strength)
        mol = Chem.AddHs(Chem.MolFromSmiles("c1ccncc1.B"))
        n = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N")
        b = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "B")
        made = self.worker.conformers(mol, count=5, rows=[{"kind": "distance", "atoms": [n, b], "value": 3.213, "strength": 500.0}])
        for f in made["frames"]:
            self.assertAlmostEqual(math.dist(f[3 * n : 3 * n + 3], f[3 * b : 3 * b + 3]), 3.213, delta=0.01)
        # (a distance no structure can have with the rest - a bond held at 5 A - refused in words)
        with self.assertRaisesRegex(ValueError, r"^The distances held cannot all be met together in one structure$"):
            self.worker.conformers(Chem.MolFromSmiles("CCCC"), count=2, rows=[{"kind": "distance", "atoms": [0, 1], "value": 5.0, "strength": 500.0}])

    def test_an_acid_held_at_its_base_by_its_restraint_alone(self):
        # (H3N->BH3, its coordination bond held at 0.9 of Alvarez's radii together, 3.21 A: no bond of the force
        #  field's between them - the restraint holds them - so B and N are each as their hydrogens make them)
        h = lambda x, k, s: {"el": "H", "x": x, "y": 0.9 * (k - 1), "z": 0.5 * s}
        entry = {
            "atoms": [{"el": "N", "x": 0, "y": 0, "z": 0}, {"el": "B", "x": 1.66, "y": 0, "z": 0}] + [h(-0.4, k, 1) for k in range(3)] + [h(2.1, k, -1) for k in range(3)],
            "bonds": [{"a1": 0, "a2": 1, "order": 1, "coordination": True}] + [{"a1": 0, "a2": 2 + k, "order": 1} for k in range(3)] + [{"a1": 1, "a2": 5 + k, "order": 1} for k in range(3)],
        }
        rows = [{"kind": "distance", "atoms": [0, 1], "value": 0.9 * (1.66 + 1.91), "strength": 500.0}]
        self.assertEqual(self.worker.held_apart(entry, rows), {frozenset((0, 1))})
        mol = self.worker.mol_of(entry, self.worker.held_apart(entry, rows))
        self.assertIsNone(mol.GetBondBetweenAtoms(0, 1))
        made = self.worker.minimised(mol, rows)
        self.assertAlmostEqual(math.dist(made["frame"][0:3], made["frame"][3:6]), 3.213, delta=0.05)

    def test_prepares_a_job_with_the_entrys_restraints_and_start(self):
        import json as _json
        mol = self.butane(60.0)
        entry = {
            "atoms": [{"el": a.GetSymbol(), "x": p[0], "y": p[1], "z": p[2]} for a, p in zip(mol.GetAtoms(), mol.GetConformer().GetPositions())],
            "bonds": [{"a1": b.GetBeginAtomIdx(), "a2": b.GetEndAtomIdx(), "order": 1} for b in mol.GetBonds()],
            "restraints": [{"kind": "dihedral", "atoms": [0, 1, 2, 3], "value": 180.0, "strength": 50.0}],
            "start": [0.0] * (3 * mol.GetNumAtoms()),
        }
        job = self.worker.op_prepare({"step": "minimise", "entries": [entry], "options": {"field": "MMFF94", "steps": 2000, "start": "first"}})["jobs"][0]
        given = _json.loads(job["files"][0]["text"])
        self.assertEqual((job["args"][1].endswith("minimise_job.py"), job["reads"]), (True, ["minimised.json"]))
        self.assertEqual(given["restraints"][0]["strength"], 50.0)
        self.assertEqual(len(given["start"]), 3 * mol.GetNumAtoms())
