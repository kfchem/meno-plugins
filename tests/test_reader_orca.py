"""The ORCA plugin's reading of ORCA's output (plugins/orca/reader.py).

Python alone, as the reader is - CI runs them:

    python3 -m unittest tests/test_reader_orca.py

Every test is written by hand: lines in ORCA's printed form, with made-up numbers.
"""

import importlib.util
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins/orca"


def load(path, name):
    """A module of a plugin's, loaded as its process loads it: its folder on the path - and the reader let go of after,
    so that another plugin's loaded in the same tests finds its own."""
    sys.path.insert(0, str(path.parent))
    sys.modules.pop("reader", None)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(path.parent))
        sys.modules.pop("reader", None)


reader = load(PLUGIN / "reader.py", "orca_reader")
worker = load(PLUGIN / "worker.py", "orca_worker_for_reading")

def read(name, text):
    """What the worker answers for an output, as Meno asks it."""
    answer = worker.answer(json.dumps({"id": 1, "op": "read", "kind": "orca", "name": name, "text": text}))
    if not answer["ok"]:
        raise ValueError(answer["error"])
    return answer["result"]


def result(r, id_):
    return next((x for x in r["results"] if x["id"] == id_), None)


def echo(*lines):
    """An input as ORCA's output echoes it, to its end."""
    return "".join(f"|{k + 1:3d}> {line}\n" for k, line in enumerate(lines)) + f"|{len(lines) + 1:3d}>                          ****END OF INPUT****\n"


GEOMETRY = "---------------------------------\nCARTESIAN COORDINATES (ANGSTROEM)\n---------------------------------\n  O      0.000000    0.000000    0.117000\n  H      0.000000    0.757000   -0.469000\n  H      0.000000   -0.757000   -0.469000\n\n"


def final(value):
    return f"FINAL SINGLE POINT ENERGY       {value}\n"


class Run(unittest.TestCase):
    def test_ended_normally_with_an_error_or_cut_short(self):
        self.assertEqual(reader.run_of("...\n                             ****ORCA TERMINATED NORMALLY****\n")[:2], ("normal", None))
        error = "[file made_up/made_up.cpp, line 1]: Error (MADE_UP): made-up words, the SCF has not converged.\n\nORCA finished by error termination in LEANSCF\n"
        self.assertEqual(reader.run_of(error)[:2], ("error", "made-up words, the SCF has not converged."))
        self.assertEqual(reader.run_of("ORCA finished by error termination in Startup\n")[:2], ("error", "ORCA finished by error termination in Startup"))
        self.assertEqual(reader.run_of("still going\n")[:2], ("cut", None))
        # (a job another follows says its end only at the file's end)
        self.assertEqual(reader.run_of("still going\n", followed=True)[0], "normal")

    def test_each_scf_converged_or_not(self):
        text = "*           SCF CONVERGED AFTER   9 CYCLES          *\n*        SCF NOT CONVERGED AFTER   2 CYCLES         *\n"
        self.assertEqual(reader.run_of(text)[2], [True, False])


class Level(unittest.TestCase):
    BASIS = "----- Orbital basis set information -----\nYour calculation utilizes the basis: {}\n"

    def test_a_functional_orca_names_its_own_way_and_its_basis(self):
        text = echo("! wB97X-D3 6-31G(d) Opt Freq") + self.BASIS.format("6-31G(d)") + "The omegaB97X-D3 functional is recognized. Fit by Chai et al.\n"
        self.assertEqual(reader.level_of(text), ("wB97X-D3", "6-31G(d)"))

    def test_a_functional_the_manual_lists_where_orca_says_it_recognised_none(self):
        dft = " Density Functional     Method          .... DFT(GTOs)\n"
        self.assertEqual(reader.level_of(echo("! B3LYP def2-SVP Opt Freq") + self.BASIS.format("def2-SVP") + dft), ("B3LYP", "def2-SVP"))
        self.assertEqual(reader.level_of(echo("! pbe0 def2-TZVP") + self.BASIS.format("def2-TZVP") + dft), ("pbe0", "def2-TZVP"))
        self.assertEqual(reader.level_of(echo("! M062X 6-31G(d)") + self.BASIS.format("6-31G(d)") + dft), ("M062X", "6-31G(d)"))
        # (a composite method: no basis of the input's)
        self.assertEqual(reader.level_of(echo("! PBEh-3c") + self.BASIS.format("def2-mSVP") + dft), ("PBEh-3c", None))
        # (whole keywords only: none that merely holds one's name, and none where ORCA ran no DFT)
        self.assertEqual(reader.level_of(echo("! B3LYPX def2-SVP") + self.BASIS.format("def2-SVP") + dft)[0], None)
        self.assertEqual(reader.level_of(echo("! B3LYP def2-SVP") + self.BASIS.format("def2-SVP"))[0], None)

    def test_a_composite_method_and_no_basis(self):
        text = echo("! r2SCAN-3c Opt") + self.BASIS.format("def2-mTZVPP") + "The R2SCAN3C composite method is recognized\n"
        self.assertEqual(reader.level_of(text), ("r2SCAN-3c", None))

    def test_a_correlated_method_its_final_energy_is_of(self):
        hf = " Ab initio Hamiltonian  Method          .... Hartree-Fock(GTOs)\n"
        self.assertEqual(reader.level_of(echo("! RI-MP2 def2-SVP def2-SVP/C") + self.BASIS.format("def2-SVP") + hf), ("RI-MP2", "def2-SVP"))
        self.assertEqual(reader.level_of(echo("! DLPNO-CCSD(T) def2-TZVP def2-TZVP/C") + self.BASIS.format("def2-TZVP") + hf), ("DLPNO-CCSD(T)", "def2-TZVP"))
        # (Hartree-Fock, as the input names it - or names nothing at all)
        self.assertEqual(reader.level_of(echo("! hf def2-SVP") + self.BASIS.format("def2-SVP") + hf), ("hf", "def2-SVP"))

    def test_a_tight_binding_method_by_the_xtb_program_s_name(self):
        self.assertEqual(reader.level_of(echo("! NATIVE-XTB2 Opt")), ("GFN2-xTB", None))
        self.assertEqual(reader.level_of(echo("! XTB2 Opt")), ("GFN2-xTB", None))

    def test_solvation_as_the_input_says_it_or_none_where_it_asks_none(self):
        self.assertEqual(reader.solvation_of(echo("! B3LYP def2-SVP CPCM(water)")), "CPCM(water)")
        self.assertEqual(reader.solvation_of(echo("! B3LYP def2-SVP SMD(chloroform)")), "SMD(chloroform)")
        self.assertEqual(reader.solvation_of(echo("! B3LYP def2-SVP CPCM")), "CPCM(water)")
        self.assertEqual(reader.solvation_of(echo("! B3LYP def2-SVP", "%cpcm smd true", 'SMDsolvent "toluene"', "end")), "SMD(toluene)")
        self.assertEqual(reader.solvation_of(echo("! B3LYP def2-SVP", "%cpcm epsilon 80.4 refrac 1.33 end")), "CPCM(epsilon 80.4, refrac 1.33)")
        self.assertEqual(reader.solvation_of(echo("! NATIVE-XTB2 ALPB(water)")), "ALPB(water)")
        # (none asked: the gas phase - said only where the input is echoed whole)
        self.assertEqual(reader.solvation_of(echo("! B3LYP def2-SVP")), "none")
        self.assertIsNone(reader.solvation_of("|  1> ! B3LYP def2-SVP CPCM(water)\n"))


class Geometries(unittest.TestCase):
    def test_each_geometry_its_energy_after_it(self):
        text = GEOMETRY + final("-76.31234567") + GEOMETRY + final("-76.31239876")
        geos = reader.geometries(text)
        self.assertEqual([[el for el, _ in g["atoms"]] for g in geos], [["O", "H", "H"]] * 2)
        self.assertEqual(reader.energies_of(text, geos), ([-76.31234567, -76.31239876], None))

    def test_an_energy_stated_again_unchanged_is_the_same(self):
        text = GEOMETRY + final("-11.398765") + "...\n" + final("-11.398765")
        self.assertEqual(reader.energies_of(text, reader.geometries(text)), ([-11.398765], None))

    def test_no_energy_where_they_are_not_one_to_each_geometry(self):
        geos = reader.geometries(GEOMETRY + GEOMETRY)
        self.assertEqual(reader.energies_of(GEOMETRY + GEOMETRY + final("-1.5"), geos)[1], "a geometry has no final single-point energy of its own")
        text = GEOMETRY + final("-1.5") + final("-1.6")
        self.assertEqual(reader.energies_of(text, reader.geometries(text))[1], "a final single-point energy stands at no geometry of its own")
        neb = echo("! NATIVE-XTB2 NEB-TS") + GEOMETRY + final("-4.4")
        self.assertEqual(reader.energies_of(neb, reader.geometries(neb))[1], "a nudged elastic band's images are not one to each geometry")

    def test_ghost_atoms_and_point_charges_are_no_atoms(self):
        text = (
            "---------------------------------\nCARTESIAN COORDINATES (ANGSTROEM)\n---------------------------------\n"
            "  O      0.000000    0.000000    0.117000\n  H      0.000000    0.757000   -0.469000\n  H      0.000000    0.000000    2.500000\n"
            "  Q      0.000000    0.000000   -3.000000\n\n----------------------------\nCARTESIAN COORDINATES (A.U.)\n----------------------------\n"
            "  NO LB      ZA    FRAG     MASS         X           Y           Z\n"
            "   0 O     8.0000    0    15.999    0.000000    0.000000    0.223456\n   1 H     1.0000    0     1.008    0.000000    1.412345   -0.891234\n"
            "   2 H     0.0000    0     0.000    0.000000    0.000000    4.712345\n   3 Q     0.5000    0     0.000    0.000000    0.000000   -5.654321\n\n"
        )
        [g] = reader.geometries(text)
        self.assertEqual(([el for el, _ in g["atoms"]], g["ghosts"], g["masses"]), (["O", "H", "H"], {2}, [15.999, 1.008, 0.0]))

    def test_a_number_too_wide_for_its_field_is_not_known(self):
        self.assertIsNone(reader.number("**********"))
        text = GEOMETRY.replace("0.117000", "*********")
        self.assertEqual(reader.geometries(text), [])

    def test_which_geometry_an_optimisation_converged_on(self):
        text = "GEOMETRY OPTIMIZATION CYCLE   1\n" + GEOMETRY + final("-1.5") + "***        THE OPTIMIZATION HAS CONVERGED     ***\n*** FINAL ENERGY EVALUATION AT THE STATIONARY POINT ***\n" + GEOMETRY + final("-1.6")
        geos = reader.geometries(text)
        self.assertEqual(reader.marks_of(text, geos, "normal"), ([False, True], None))
        self.assertTrue(reader.optimised_of(text))
        # (one that came to its end without converging: each marked not, none found; one stopped, why)
        unmet = "GEOMETRY OPTIMIZATION CYCLE   1\n" + GEOMETRY + final("-1.5") + "GEOMETRY OPTIMIZATION CYCLE   2\n" + GEOMETRY + final("-1.6")
        self.assertEqual(reader.marks_of(unmet, reader.geometries(unmet), "normal"), ([False, False], None))
        self.assertFalse(reader.optimised_of(unmet))
        self.assertEqual(reader.marks_of(unmet, reader.geometries(unmet), "error"), ([False, False], "error"))
        # (no optimisation: none marked, one geometry of a run that ended well is the one)
        self.assertEqual(reader.marks_of(GEOMETRY, reader.geometries(GEOMETRY), "normal"), (None, None))


class Thermochemistry(unittest.TestCase):
    TEXT = (
        "--------------------------\nTHERMOCHEMISTRY AT 298.15K\n--------------------------\n\n"
        "Temperature         ...   298.15 K\nPressure            ...     1.00 atm\nTotal Mass          ...    18.02 AMU\nQuasi RRHO          ...    False\n\n"
        "Zero point energy                ...      0.02130000 Eh      13.37 kcal/mol\n"
        "Total Enthalpy                    ...    -76.31230000 Eh\n"
        "Point Group:  C2v, Symmetry Number:   2  \n"
        "Final entropy term                ...      0.02190000 Eh     13.74 kcal/mol\n"
        "Final Gibbs free energy         ...    -76.33460000 Eh\n"
        "G-E(el)                           ...      0.00410000 Eh      2.57 kcal/mol\n"
    )

    def test_each_value_as_printed(self):
        got = {r["id"]: r["value"] for r in reader.thermo_of(self.TEXT)}
        self.assertEqual(got["thermo.free-energy"], -76.3346)
        self.assertEqual(got["thermo.enthalpy"], -76.3123)
        self.assertEqual(got["thermo.zpve"], 0.0213)
        self.assertEqual(got["thermo.g-correction"], 0.0041)
        self.assertAlmostEqual(got["thermo.entropy"], 0.0219 / 298.15 * 627509.474, 9)
        self.assertEqual((got["thermo.temperature"], got["thermo.pressure"]), (298.15, 1.0))
        self.assertEqual((got["thermo.point-group"], got["thermo.symmetry-number"], got["thermo.treatment"]), ("C2v", 2, "RRHO"))

    def test_a_linear_molecule_s_point_group_in_standard_notation(self):
        linear = self.TEXT.replace("Point Group:  C2v, Symmetry Number:   2", "Point Group:  C(inf)v, Symmetry Number:   1")
        self.assertEqual(next(r["value"] for r in reader.thermo_of(linear) if r["id"] == "thermo.point-group"), "C\u221ev")
        self.assertEqual(next(r["value"] for r in reader.thermo_of(linear.replace("C(inf)v", "D(inf)h")) if r["id"] == "thermo.point-group"), "D\u221eh")

    def test_quasi_rrho_and_its_frequency(self):
        grimme = self.TEXT.replace("Quasi RRHO          ...    False\n", "Quasi RRHO          ...     True\n") + "Vibrational entropy computed according to the QRRHO of S. Grimme\nChem.Eur.J. 2012 18 9955 using a reference frequency of 100.0 cm-1\n"
        self.assertEqual(next(r["value"] for r in reader.thermo_of(grimme) if r["id"] == "thermo.treatment"), "quasi-RRHO (Grimme), 100 cm⁻¹")


class Vibrations(unittest.TestCase):
    TEXT = (
        "-----------------------\nVIBRATIONAL FREQUENCIES\n-----------------------\n\n"
        "     0:       0.00 cm**-1\n     1:       0.00 cm**-1\n     2:       0.00 cm**-1\n     3:       0.00 cm**-1\n     4:       0.00 cm**-1\n     5:       0.00 cm**-1\n"
        "     6:    -500.00 cm**-1  ***imaginary mode***\n     7:    1700.00 cm**-1\n     8:    3800.00 cm**-1\n\n"
        "------------\nNORMAL MODES\n------------\n\n"
        "                  0          1          2          3          4          5    \n"
        + "".join(f"      {i}       0.000000   0.000000   0.000000   0.000000   0.000000   0.000000\n" for i in range(9))
        + "                  6          7          8    \n"
        + "".join(f"      {i}       0.{i}00000   0.100000   0.200000\n" for i in range(9))
        + "\n\n-----------\nIR SPECTRUM\n-----------\n\n Mode   freq       eps      Int      T**2         TX        TY        TZ\n"
        "       cm**-1   L/(mol*cm) km/mol    a.u.\n----------------------------------------------------------------------------\n"
        "  7:   1700.00   0.017000   90.00  0.003000  (-0.000000  0.000000 -0.050000)\n"
        "  8:   3800.00   0.006000   30.00  0.000500  (-0.000000  0.020000 -0.000000)\n\n"
    )

    def test_each_mode_its_frequency_intensity_and_motion_and_the_imaginary_ones(self):
        [v] = reader.vibrations_of(self.TEXT, 3, set())
        self.assertEqual([c["label"] for c in v["columns"]], ["Mode", "Frequency", "IR intensity"])
        # (an imaginary one ORCA leaves out of its spectrum: its intensity not known)
        self.assertEqual([r["cells"] for r in v["rows"]], [[1, -500.0, None], [2, 1700.0, 90.0], [3, 3800.0, 30.0]])
        self.assertEqual(v["rows"][0]["move"], [i / 10 for i in range(9)])
        self.assertEqual(reader.imaginary_of(self.TEXT), [-500.0])
        self.assertIsNone(reader.imaginary_of("no frequencies here"))


class Properties(unittest.TestCase):
    DIPOLE = (
        "Position of electric origin ... (  0.0000,  0.0000,  0.1000) \n...\n"
        "-------------\nDIPOLE MOMENT\n-------------\n\nMethod             : SCF\n"
        "                                X                 Y                 Z\n"
        "Total Dipole Moment    :      0.000000000       0.000000000      -0.800000000\n"
        "                        -----------------------------------------\n"
        "Magnitude (a.u.)       :      0.800000000\nMagnitude (Debye)      :      2.033400000\n"
    )

    def test_the_dipole_as_printed_its_components_and_the_point_it_was_taken_about(self):
        [d] = reader.dipole_of(self.DIPOLE)
        self.assertEqual(d["value"], 2.0334)
        self.assertAlmostEqual(d["vector"][2], -2.0334, 12)
        self.assertAlmostEqual(d["origin"][2], 0.1 * reader.BOHR, 12)
        # (its other printed form: three numbers, no brackets)
        plain = self.DIPOLE.replace("(  0.0000,  0.0000,  0.1000) ", "    0.000000    0.000000    0.100000")
        self.assertAlmostEqual(reader.dipole_of(plain)[0]["origin"][2], 0.1 * reader.BOHR, 12)

    def test_partial_charges_by_scheme_and_spin_populations(self):
        text = (
            "-------------------------------------------\nMULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS\n-------------------------------------------\n"
            "   0 C :   -0.150000    2.080000\n   1 H :    0.075000   -0.040000\n   2 H :    0.075000   -0.040000\nSum of atomic charges         :    0.0000000\n\n"
            "HIRSHFELD ANALYSIS\n------------------\n\n  ATOM     CHARGE      SPIN    \n   0 C   -0.030000    1.900000\n   1 H    0.015000    0.050000\n   2 H    0.015000    0.050000\n\n  TOTAL   0.000000    2.000000\n"
        )
        got = {r["id"]: r["values"] for r in reader.charges_of(text, 3, set())}
        self.assertEqual(got["charges.mulliken"], [-0.15, 0.075, 0.075])
        self.assertEqual(got["spins.mulliken"], [2.08, -0.04, -0.04])
        self.assertEqual(got["charges.hirshfeld"], [-0.03, 0.015, 0.015])
        self.assertEqual(got["spins.hirshfeld"], [1.9, 0.05, 0.05])

    def test_the_orbitals_all_printed_in_meno_s_form_each_its_energy_in_hartrees(self):
        text = (
            "----------------\nORBITAL ENERGIES\n----------------\n\n  NO   OCC          E(Eh)            E(eV) \n"
            "   0   2.0000     -19.200000      -522.4500 \n   1   2.0000      -0.400000       -10.8800 \n"
            "   2   0.0000       0.170000         4.6300 \n   3   0.0000       0.250000         6.8000 \n"
        )
        # (in ORCA's order, as printed: Meno names and lists them)
        self.assertEqual(reader.orbitals_of(text), [
            {"spin": "both", "energy": -19.2, "occupation": 2.0},
            {"spin": "both", "energy": -0.4, "occupation": 2.0},
            {"spin": "both", "energy": 0.17, "occupation": 0.0},
            {"spin": "both", "energy": 0.25, "occupation": 0.0},
        ])
        # (an open shell's: alpha, then beta; with symmetry where ORCA printed it)
        uks = (
            "----------------\nORBITAL ENERGIES\n----------------\n                 SPIN UP ORBITALS\n  NO   OCC          E(Eh)            E(eV)    Irrep \n"
            "   0   1.0000     -10.234567      -278.4985    1-A1\n   1   0.0000       0.084567         2.3012    1-B1\n\n"
            "                 SPIN DOWN ORBITALS\n  NO   OCC          E(Eh)            E(eV)    Irrep \n"
            "   0   1.0000     -10.190000      -277.2840    1-A1\n   1   0.0000       0.100000         2.7211    1-B1\n"
        )
        self.assertEqual([(o["spin"], o["occupation"], o.get("symmetry")) for o in reader.orbitals_of(uks)], [("alpha", 1.0, "A1"), ("alpha", 0.0, "B1"), ("beta", 1.0, "A1"), ("beta", 0.0, "B1")])
        self.assertIsNone(reader.orbitals_of("no orbitals here"))


class Jobs(unittest.TestCase):
    def test_each_job_of_a_file_of_several_is_its_own(self):
        text = echo("! r2SCAN-3c", "* xyz 0 1", "*", "$new_job", "! B97-3c", "* xyz 0 1", "*") + (
            "                 $$$$$$$$$$$$$$$$  JOB NUMBER  1 $$$$$$$$$$$$$$\n" + GEOMETRY + final("-76.4")
            + "                 $$$$$$$$$$$$$$$$  JOB NUMBER  2 $$$$$$$$$$$$$$\n" + GEOMETRY + final("-76.3")
            + "                             ****ORCA TERMINATED NORMALLY****\n"
        )
        r = read("two.out", text)
        self.assertEqual([j["job"] for j in r["jobs"]], [{"number": 1, "of": 2, "kind": "Single point"}, {"number": 2, "of": 2, "kind": "Single point"}])
        self.assertEqual([j["energies"] for j in r["jobs"]], [[-76.4], [-76.3]])
        self.assertIsNone(result(r["jobs"][0], "run"))


class Bands(unittest.TestCase):
    def test_a_band_s_output_gives_the_geometry_it_started_from_alone(self):
        # (however many geometries it wrote after - its refinement's steps: no path's points)
        text = echo("! NATIVE-XTB2 NEB-TS") + GEOMETRY.replace("0.117000", "0.100000") + final("-4.40") + GEOMETRY + final("-4.41") + GEOMETRY + final("-4.42") + "****ORCA TERMINATED NORMALLY****\n"
        r = read("band.out", text)
        self.assertEqual((len(r["frames"]), r["frames"][0][2], r["unmarked"]), (1, 0.1, "path"))


if __name__ == "__main__":
    unittest.main()
