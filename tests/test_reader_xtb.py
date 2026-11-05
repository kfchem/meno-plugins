"""Tests for the xTB plugin's reader (plugins/xtb/reader.py).

The reader needs Python alone, so these run anywhere; CI runs them:

    python3 -m unittest tests/test_reader_xtb.py

What is read is as xtb's documentation (xtb-docs.readthedocs.io) and its
real outputs have it. The texts below are printed forms written after xtb
6.7.1's, their numbers made up. Whole outputs of real runs are not kept in
the repository: made by running xtb on the inputs in `INPUTS` and put in
calc-samples/xTB-xtb/<run>/out.log (its own stream) and err.log (its
other), they are read too, and skipped where absent.
"""

import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
READER = ROOT / "plugins/xtb/reader.py"
spec = importlib.util.spec_from_file_location("plugin_xtb_reader", READER)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)

SAMPLES = ROOT / "calc-samples/xTB-xtb"
# each run: its input (Xmol, angstroms), and xtb's arguments after it - run as `xtb input.xyz <args> -P 1 > out.log 2> err.log`
FORMALDEHYDE = "4\nformaldehyde\nC 0.000 0.000 -0.530\nO 0.000 0.000 0.680\nH 0.000 0.940 -1.110\nH 0.000 -0.940 -1.110\n"
ACETATE = (
    "7\nacetate\nC -1.050 0.000 0.000\nC 0.480 0.000 0.000\nO 1.080 1.100 0.000\nO 1.080 -1.100 0.000\n"
    "H -1.420 1.030 0.000\nH -1.420 -0.515 0.892\nH -1.420 -0.515 -0.892\n"
)
INPUTS = {
    "opt-gfn2": (FORMALDEHYDE, "--opt normal --gfn 2"),
    "opt-notconv": (FORMALDEHYDE, "--opt normal --gfn 2 --cycles 2"),
    "opt-anion": (ACETATE, "--opt --gfn 2 --chrg -1"),
    "hess-gfnff": (FORMALDEHYDE, "--hess --gfnff"),
    "sp-gfn1": ("3\nwater\nO 0.000 0.000 0.119\nH 0.000 0.757 -0.477\nH 0.000 -0.757 -0.477\n", "--gfn 1"),
}

# the table of elements a run begins with, and its setup box (GFN2-xTB): a charge, its unpaired electrons, no solvation
BEGUN = """
   * xtb version 6.7.1 (abc1234) compiled by 'someone' on 2026-01-01
 * started run on 2026/01/01 at 12:00:00.000
   ID    Z sym.   atoms
    1    6 C      1, 2
    2    8 O      3, 4
    3    1 H      5-7

          program call               : xtb input.xyz --gfn 2 --chrg -1 -P 4
          ...................................................
          :                      SETUP                      :
          :.................................................:
          :  # electrons                        24          :
          :  Hamiltonian                  GFN2-xTB          :
          :  GBSA solvation                  false          :
          :  net charge                         -1          :
          :  unpaired electrons                  0          :
          ...................................................
"""
ENDED = """
         :::::::::::::::::::::::::::::::::::::::::::::::::::::
         ::                     SUMMARY                     ::
         :::::::::::::::::::::::::::::::::::::::::::::::::::::
         :: total energy             -14.100000000000 Eh    ::
         :: HOMO-LUMO gap              5.300000000000 eV    ::
         :::::::::::::::::::::::::::::::::::::::::::::::::::::
           -------------------------------------------------
          | TOTAL ENERGY              -14.123456789012 Eh   |
          | GRADIENT NORM               0.000100000000 Eh/α |
          | HOMO-LUMO GAP               5.300000000000 eV   |
           -------------------------------------------------
------------------------------------------------------------------------
 * finished run on 2026/01/01 at 12:00:01.000
------------------------------------------------------------------------
"""
# an optimisation's end, and the final structure it prints as it was given it (Xmol)
OPTIMISED = """
   *** GEOMETRY OPTIMIZATION CONVERGED AFTER 5 ITERATIONS ***
 final structure:
================
7
 xtb: 6.7.1 (abc1234)
C           -1.10000000000000        0.01000000000000        0.00000000000000
C            0.50000000000000        0.00000000000000        0.00000000000000
O            1.10000000000000        1.10000000000000        0.00000000000000
O            1.10000000000000       -1.10000000000000        0.00000000000000
H           -1.40000000000000        1.00000000000000        0.00000000000000
H           -1.40000000000000       -0.50000000000000        0.90000000000000
H           -1.40000000000000       -0.50000000000000       -0.90000000000000

 Bond Distances (Angstroems)
"""
# GFN2-xTB's property printout: its orbitals about the frontier, each atom's q, and its dipole
PRINTOUT = """
    * Orbital Energies and Occupations

         #    Occupation            Energy/Eh            Energy/eV
      -------------------------------------------------------------
         1        2.0000           -0.6000000             -16.3000
       ...           ...                  ...                  ...
        11        2.0000           -0.1600000              -4.3000
        12        2.0000           -0.1500000              -4.1000 (HOMO)
        13                          0.0400000               1.0000 (LUMO)
        14                          0.3000000               8.2000
      -------------------------------------------------------------
                  HL-Gap            0.1900000 Eh            5.1000 eV

     #   Z          covCN         q      C6AA      α(0)
     1   6 C        3.700    -0.100    23.000     6.800
     2   6 C        2.700     0.400    20.000     7.400
     3   8 O        0.900    -0.600    26.000     6.900
     4   8 O        0.900    -0.600    26.000     6.900
     5   1 H        0.900    -0.030     3.200     2.800
     6   1 H        0.900    -0.035     3.200     2.800
     7   1 H        0.900    -0.035     3.200     2.800

molecular dipole:
                 x           y           z       tot (Debye)
 q only:       -1.700       0.010       0.000
   full:       -2.000      -0.000       0.000       5.084
molecular quadrupole (traceless):
                xx          xy          yy          xz          yz          zz
 q only:       -0.700       0.000       0.900       0.000      -0.000      -0.100
   full:       -1.200      -0.000       1.300       0.000      -0.000      -0.100
"""


class Run(unittest.TestCase):
    def test_what_it_was(self):
        out = reader.read("run.out", BEGUN + PRINTOUT + ENDED)
        self.assertEqual((out["program"], out["version"], out["method"], out["basis"]), ("xtb", "6.7.1", "GFN2-xTB", None))
        self.assertEqual((out["charge"], out["multiplicity"]), (-1, 1))
        self.assertEqual(out["atoms"], ["C", "C", "O", "O", "H", "H", "H"])
        # (its setup says none is used)
        self.assertEqual(out["solvation"], "none")

    def test_a_single_point_prints_no_geometry(self):
        out = reader.read("run.out", BEGUN + PRINTOUT + ENDED)
        self.assertEqual(out["frames"], [])
        self.assertIsNone(out["energies"])
        self.assertIsNone(out["optimised"])
        self.assertAlmostEqual(reader.total_energy(BEGUN + ENDED), -14.123456789012)

    def test_an_optimisation_s_final_structure_its_energy_and_its_outcome(self):
        out = reader.read("run.out", BEGUN + OPTIMISED + PRINTOUT + ENDED)
        self.assertEqual(len(out["frames"]), 1)
        self.assertEqual(out["frames"][0][:3], [-1.1, 0.01, 0.0])
        self.assertEqual(out["energies"], [-14.123456789012])
        self.assertTrue(out["optimised"])
        failed = OPTIMISED.replace("GEOMETRY OPTIMIZATION CONVERGED AFTER 5 ITERATIONS", "FAILED TO CONVERGE GEOMETRY OPTIMIZATION IN 2 ITERATIONS")
        self.assertFalse(reader.read("run.out", BEGUN + failed + PRINTOUT + ENDED)["optimised"])

    def test_a_final_structure_given_as_a_turbomole_coord_in_bohr(self):
        coord = " final structure:\n================\n$coord\n    0.00000000000000    0.00000000000000    1.88972612462577      o\n$end\n"
        water = "   ID    Z sym.   atoms\n    1    8 O      1\n"
        self.assertEqual(reader.final_structure(water + coord, ["O"]), [0.0, 0.0, 1.88972612462577 * reader.BOHR])

    def test_the_atoms_table_must_give_each_place_once(self):
        self.assertEqual(reader.atoms_of("   ID    Z sym.   atoms\n    1    8 O      1\n    2    1 H      2, 3\n"), ["O", "H", "H"])
        self.assertIsNone(reader.atoms_of("   ID    Z sym.   atoms\n    1    8 O      1\n    2    1 H      1, 3\n"))
        self.assertIsNone(reader.atoms_of("no table"))
        with self.assertRaises(ValueError):
            reader.read("run.out", "nothing of xtb's")


class HowItEnded(unittest.TestCase):
    def test_its_own_stream_says_it_came_to_its_end(self):
        # (an output of xtb's own stream alone: "normal termination of xtb" is on its other)
        self.assertEqual(reader.run_of(BEGUN + ENDED)[:2], ("normal", None))
        self.assertEqual(reader.run_of("normal termination of xtb\n")[:2], ("normal", None))
        self.assertNotIn("run", {r["id"] for r in reader.read("run.out", BEGUN + PRINTOUT + ENDED)["results"]})

    def test_cut_short(self):
        out = reader.read("run.out", BEGUN + OPTIMISED + PRINTOUT)
        self.assertEqual(reader.run_of(BEGUN)[0], "cut")
        self.assertEqual(next(r for r in out["results"] if r["id"] == "run")["value"], "did not come to its end")
        self.assertEqual((out["converged"], out["unmarked"]), ([False], "cut"))

    def test_stopped_on_an_error_says_why(self):
        error = (
            "########################################################################\n[ERROR] Program stopped due to fatal error\n"
            "-2- Geometry optimization failed\n-1- xtb_geoopt: Geometry optimization did not converge\n"
            "########################################################################\n"
        )
        self.assertEqual(reader.run_of(BEGUN + error), ("error", "Geometry optimization failed: xtb_geoopt: Geometry optimization did not converge", []))
        results = {r["id"]: r["value"] for r in reader.read("run.out", BEGUN + error)["results"]}
        self.assertEqual(results["run"], "ended with an error")
        # (before any atom was read: no molecule, and why)
        unread = "[ERROR] Program stopped due to fatal error\n-2- reading geometry input 'input.xyz' failed\n-1- Error: Cannot map symbol to atomic number\n"
        with self.assertRaisesRegex(ValueError, "reading geometry input 'input.xyz' failed: Cannot map symbol to atomic number"):
            reader.read("run.out", unread)

    def test_warnings_in_xtb_s_words(self):
        warned = "[WARNING] Runtime exception occurred\n-1- hessian_numhess: Hessian on incompletely optimized geometry!\n"
        self.assertEqual(reader.warnings_of(warned), [{"kind": "hessian-not-stationary", "text": "hessian_numhess: Hessian on incompletely optimized geometry!"}])
        self.assertEqual(reader.warnings_of("[ERROR] Program stopped due to fatal error\n-1- no\n"), [])


class Results(unittest.TestCase):
    def results(self, text):
        return {r["id"]: r for r in reader.read("run.out", BEGUN + text + ENDED)["results"]}

    def test_the_orbitals_about_the_frontier_as_data_with_their_numbers(self):
        # (those xtb printed alone, in its order, each by its number - the ones it left out between them none made up;
        #  Meno names them, lists them and works out the gap: src/lib/calc/orbitals.ts)
        out = reader.read("run.out", BEGUN + PRINTOUT + ENDED)
        self.assertEqual(out["orbitals"], [
            {"spin": "both", "energy": -0.6, "occupation": 2.0, "number": 1},
            {"spin": "both", "energy": -0.16, "occupation": 2.0, "number": 11},
            {"spin": "both", "energy": -0.15, "occupation": 2.0, "number": 12},
            {"spin": "both", "energy": 0.04, "occupation": 0.0, "number": 13},
            {"spin": "both", "energy": 0.3, "occupation": 0.0, "number": 14},
        ])
        ids = {r["id"] for r in out["results"]}
        self.assertFalse(ids & {"orbitals", "homo-lumo-gap"})
        # (none printed: none given)
        self.assertNotIn("orbitals", reader.read("run.out", BEGUN + ENDED))

    def test_each_atom_s_charge_as_printed(self):
        self.assertEqual(self.results(PRINTOUT)["charges"]["values"], [-0.1, 0.4, -0.6, -0.6, -0.03, -0.035, -0.035])
        gfn1 = "  Mulliken/CM5 charges         n(s)   n(p)   n(d)\n     1O    -0.66000 -0.99000   1.696  4.970  0.000\n     2H     0.33000  0.49500   0.667  0.000  0.000\n     3H     0.33000  0.49500   0.667  0.000  0.000\n"
        r = {x["id"]: x["values"] for x in reader.charges_of(gfn1, 3)}
        self.assertEqual(r, {"charges.mulliken": [-0.66, 0.33, 0.33], "charges.cm5": [-0.99, 0.495, 0.495]})
        self.assertEqual(reader.charges_of(gfn1, 4), [])

    def test_the_dipole_as_printed_and_the_point_it_is_taken_about(self):
        d = self.results(PRINTOUT)["dipole"]
        self.assertEqual(d["value"], 5.084)
        self.assertAlmostEqual(d["vector"][0], -2.0 * reader.DEBYE_PER_AU)
        self.assertEqual(d["origin"], [0.0, 0.0, 0.0])
        self.assertEqual(d["label"], "Dipole moment")
        self.assertNotIn("centre", d)
        # (after an optimisation, its Geometry Summary says where that geometry's centre of mass is: given with it)
        summary = "      molecular mass/u    :       59.0440000\n   center of mass at/Å    :        0.3912345       0.0012345       0.0000000\n"
        self.assertEqual(self.results(PRINTOUT + summary)["dipole"]["centre"], [0.3912345, 0.0012345, 0.0])
        # (one printed before the dipole is of another geometry: not given)
        self.assertNotIn("centre", self.results(summary + PRINTOUT)["dipole"])
        gfn1 = " dipole moment from electron density (au)\n     X       Y       Z   \n   0.0000   0.0000  -1.1000  total (Debye):    2.796\n"
        self.assertEqual(reader.dipole_of(gfn1)[0]["value"], 2.796)
        # (GFN-FF's, of its charges: a "q only" row with its total, and no "full" row)
        gfnff = "molecular dipole:\n                 x           y           z       tot (Debye)\n q only:        0.000       0.000      -0.700       1.779\n\n"
        self.assertEqual(reader.dipole_of(gfnff)[0]["value"], 1.779)

    def test_vibrations_and_thermochemistry(self):
        freq = """
 projected vibrational frequencies (cm⁻¹)
eigval :       -0.00    -0.00    -0.00     0.00     0.00     0.00
eigval :     -177.20  1100.00  1200.00
 reduced masses (amu)
   1:  6.00   2: 11.00   3: 12.00   4: 14.00   5: 12.00   6:  5.00   7:  1.10   8:  4.40
   9:  1.90
 IR intensities (km·mol⁻¹)
   1:  9.00   2:  6.00   3:  2.00   4:  8.00   5:  2.00   6:  7.00   7:  0.10   8:136.20
   9: 21.40
 Raman intensities (Ä⁴*amu⁻¹)
   1:  0.00
found            1  significant imaginary frequency
   temp. (K)  partition function   enthalpy   heat capacity  entropy
                                   cal/mol     cal/K/mol   cal/K/mol   J/K/mol
 298.15  VIB   1.00                   28.000      0.500      0.100
         TOT                        2400.0000     8.5000    52.0000   217.5000
       T/K    H(0)-H(T)+PV         H(T)/Eh          T*S/Eh         G(T)/Eh
   ------------------------------------------------------------------------
    298.15    0.380000E-02    0.290000E-01    0.250000E-01    0.400000E-02
   ------------------------------------------------------------------------
         :: total free energy          -7.170000000000 Eh   ::
         :: zero point energy           0.025000000000 Eh   ::
         :: G(RRHO) contrib.            0.004000000000 Eh   ::
          | TOTAL ENTHALPY             -7.145000000000 Eh   |
"""
        out = reader.read("run.out", BEGUN + freq + ENDED)
        r = {x["id"]: x for x in out["results"]}
        # (the translations and rotations at naught left out; a wide IR intensity run into its number read)
        self.assertEqual([row["cells"] for row in r["vibrations"]["rows"]], [[1, -177.2, 1.1, 0.1], [2, 1100.0, 4.4, 136.2], [3, 1200.0, 1.9, 21.4]])
        self.assertEqual(out["imaginary"], [-177.2])
        self.assertEqual(r["thermo.entropy"]["value"], 52.0)
        self.assertEqual(r["thermo.free-energy"]["value"], -7.17)
        self.assertEqual(r["thermo.enthalpy"]["value"], -7.145)
        self.assertEqual(r["thermo.temperature"]["value"], 298.15)
        # (no frequencies: no imaginary ones judged)
        self.assertNotIn("imaginary", reader.read("run.out", BEGUN + ENDED))

    def test_a_linear_molecule_s_frequencies_and_point_group(self):
        # (xtb heads a linear molecule's "vibrational frequencies", unprojected; an intensity too wide for its field it
        # prints as asterisks - not known; and the point group it spells "din" or "cin", in Schoenflies' notation)
        linear = """
 vibrational frequencies (cm⁻¹)
eigval :        0.00     0.00     0.00     0.00     0.00   600.00
eigval :      600.00  1400.00  2600.00
 reduced masses (amu)
   1: 14.00   2: 14.00   3: 14.00   4: 16.00   5: 16.00   6: 13.00   7: 13.00   8: 16.00
   9: 13.00
 IR intensities (km·mol⁻¹)
   1:  0.00   2:  0.00   3:  0.00   4:  0.00   5:  0.00   6: 68.00   7: 68.00   8:  0.00
   9:******
          :  symmetry                              din      :
          :  rotational number                       2      :
         :: total free energy          -10.300000000000 Eh   ::
"""
        r = self.results(linear)
        self.assertEqual([row["cells"] for row in r["vibrations"]["rows"]], [[1, 600.0, 13.0, 68.0], [2, 600.0, 13.0, 68.0], [3, 1400.0, 16.0, 0.0], [4, 2600.0, 13.0, None]])
        self.assertEqual(r["thermo.point-group"]["value"], "D\u221eh")
        self.assertEqual(self.results(linear.replace("din", "cin"))["thermo.point-group"]["value"], "C\u221ev")
        self.assertEqual(self.results(linear.replace("din", "c2v"))["thermo.point-group"]["value"], "c2v")

    def test_the_solvation_as_its_setup_says_it(self):
        alpb = "      * Solvation model:               ALPB\n        Solvent                        water\n        Reference state                gsolv [1 M gas/solution]\n"
        self.assertEqual(reader.solvation_of(alpb), "ALPB(water), reference state gsolv [1 M gas/solution]")
        self.assertEqual(reader.solvation_of(BEGUN), "none")
        # (GFN-FF prints no such box: its program call asked for none)
        self.assertEqual(reader.solvation_of("          program call               : xtb input.xyz --gfnff -P 4\n"), "none")
        self.assertIsNone(reader.solvation_of("          program call               : xtb input.xyz --gfnff --alpb water -P 4\n"))
        self.assertIsNone(reader.solvation_of(""))


class GaussianForm(unittest.TestCase):
    """The frequencies xtb writes in Gaussian's form for MOLDEN (g98.out), read on their own."""

    G98 = """ Entering Gaussian System
 *********************************************
 Gaussian 98:
 frequency output generated by the xtb code
 *********************************************
                         Standard orientation:
 --------------------------------------------------------------------
  Center     Atomic     Atomic              Coordinates (Angstroms)
  Number     Number      Type              X           Y           Z
 --------------------------------------------------------------------
    1          6             0        0.000000    0.000000    0.000000
    2          8             0        0.000000    0.000000    1.200000
    3          8             0        0.000000    0.000000   -1.200000
 --------------------------------------------------------------------
                      1                      2                      3
                        a                      a                      a 
 Frequencies --     0.0000               600.0000              1400.0000
 Red. masses --     1.0000                13.0000                16.0000
 IR Inten    --     0.0000                68.0000                 0.0000
 Atom AN      X      Y      Z        X      Y      Z        X      Y      Z
   1   6     0.00   0.00   0.00    -0.85   0.00   0.00     0.00   0.00   0.00
   2   8     0.00   0.00   0.00     0.37   0.00   0.00     0.00   0.00   0.71
   3   8     0.00   0.00   0.00     0.37   0.00   0.00     0.00   0.00  -0.71
                      4
                        a 
 Frequencies --  2600.0000
 Red. masses --    13.0000
 IR Inten    --  1046.0000
 Atom AN      X      Y      Z        X      Y      Z        X      Y      Z
   1   6    -0.00  -0.00  -0.85
   2   8     0.00   0.00   0.37
   3   8     0.00   0.00   0.37
"""

    def test_its_geometry_and_its_vibrations_with_their_motion(self):
        out = reader.read_jobs("g98.out", self.G98, "xtb-g98")
        self.assertEqual((out["program"], out["atoms"], out["frames"]), ("xtb", ["C", "O", "O"], [[0.0, 0.0, 0.0, 0.0, 0.0, 1.2, 0.0, 0.0, -1.2]]))
        rows = out["results"][0]["rows"]
        # (one at naught left out; a block of one mode under a heading that names three)
        self.assertEqual([r["cells"] for r in rows], [[1, 600.0, 13.0, 68.0], [2, 1400.0, 16.0, 0.0], [3, 2600.0, 13.0, 1046.0]])
        self.assertEqual(rows[2]["move"], [-0.0, -0.0, -0.85, 0.0, 0.0, 0.37, 0.0, 0.0, 0.37])
        # (told by what it holds too, its kind unsaid)
        self.assertEqual(reader.read_jobs("g98.out", self.G98)["atoms"], ["C", "O", "O"])

    def test_a_gaussian_output_is_not_this(self):
        with self.assertRaises(ValueError):
            reader.read_g98("run.log", " Entering Gaussian System, Link 0=g16\n")


class Whole(unittest.TestCase):
    """Outputs of real runs, where they are in calc-samples/ (`INPUTS`)."""

    def output(self, run, both=False):
        out = SAMPLES / run / "out.log"
        if not out.exists():
            raise unittest.SkipTest(f"no xTB-xtb/{run} in calc-samples/")
        text = out.read_text(encoding="utf-8")
        return reader.read("out.log", text + ((SAMPLES / run / "err.log").read_text(encoding="utf-8") if both else ""))

    def test_an_optimisation_read_from_its_own_stream_alone_or_with_its_other(self):
        for both in (False, True):
            out = self.output("opt-gfn2", both)
            self.assertEqual((out["atoms"], len(out["frames"]), out["optimised"]), (["C", "O", "H", "H"], 1, True))
            self.assertNotIn("run", {r["id"] for r in out["results"]})

    def test_an_optimisation_that_did_not_converge_says_so(self):
        self.assertFalse(self.output("opt-notconv")["optimised"])

    def test_an_ion_s_charge_and_dipole(self):
        out = self.output("opt-anion")
        self.assertEqual(out["charge"], -1)
        self.assertEqual(next(r for r in out["results"] if r["id"] == "dipole")["origin"], [0.0, 0.0, 0.0])

    def test_gfn_ff_frequencies(self):
        out = self.output("hess-gfnff")
        self.assertEqual(out["method"], "GFN-FF")
        self.assertEqual(len(next(r for r in out["results"] if r["id"] == "vibrations")["rows"]), 6)

    def test_gfn1_charges_by_both_schemes(self):
        ids = {r["id"] for r in self.output("sp-gfn1")["results"]}
        self.assertTrue({"charges.mulliken", "charges.cm5"} <= ids)


if __name__ == "__main__":
    unittest.main()
