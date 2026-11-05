"""Tests for the Gaussian plugin's reader
(plugins/gaussian/reader.py).

The reader needs Python alone, so these run anywhere; CI runs them:

    python3 -m unittest tests/test_reader_gaussian.py

No output of a real run is kept here (docs/WORKSPACE.md, stage 3): the
lines below are written as Gaussian 16 prints them, laid out as its own
reference (gaussian.com: the keyword pages, "Molecule Specification",
"About Gaussian 16 Input", "Vibrational Analysis in Gaussian",
"Thermochemistry in Gaussian") lays them out, with made-up numbers:
hand-written fragments only.
"""

import importlib.util
import json
import math
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins/gaussian"
# (the worker loaded with its own folder on the path, as Meno runs it there - its reader the one beside it, never another
#  plugin's taken from those already imported - and that reader read)
sys.path.insert(0, str(PLUGIN))
sys.modules.pop("reader", None)
try:
    spec = importlib.util.spec_from_file_location("plugin_gaussian", PLUGIN / "worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
finally:
    sys.path.remove(str(PLUGIN))
    sys.modules.pop("reader", None)
reader = worker.reader

RULE = " " + "-" * 69


def route(text):
    """A route section as Gaussian echoes it: between lines of dashes, wrapped at 70 characters - a word cut where a
    line is full - each line begun by a space."""
    lines = [text[i : i + 70] for i in range(0, len(text), 70)]
    return [" " + "-" * 40] + [" " + line for line in lines] + [" " + "-" * 40]


def orientation(xyz, title="Standard orientation:"):
    """A block of coordinates: atomic numbers and x, y, z."""
    out = [f"                         {title}                         ", RULE,
           " Center     Atomic      Atomic             Coordinates (Angstroms)",
           " Number     Number       Type             X           Y           Z", RULE]
    out += [f"{i + 1:7d}{z:11d}{0:12d}    {x:12.6f}{y:12.6f}{w:12.6f}" for i, (z, x, y, w) in enumerate(xyz)]
    return out + [RULE]


WATER = [(8, 0.0, 0.0, 0.119), (1, 0.0, 0.763, -0.477), (1, 0.0, -0.763, -0.477)]
WATER_2 = [(8, 0.0, 0.0, 0.121), (1, 0.0, 0.760, -0.480), (1, 0.0, -0.760, -0.480)]


def scf(e, method="RB3LYP"):
    return [f" SCF Done:  E({method}) =  {e:.9f}     A.U. after    9 cycles"]


def forces(values):
    out = [" -------------------------------------------------------------------",
           " Center     Atomic                   Forces (Hartrees/Bohr)",
           " Number     Number              X              Y              Z",
           " -------------------------------------------------------------------"]
    out += [f"{i + 1:7d}{z:9d}     {x:15.9f}{y:15.9f}{w:15.9f}" for i, (z, x, y, w) in enumerate(values)]
    return out + [" -------------------------------------------------------------------"]


def head(route_text, charge=0, multiplicity=1):
    return [
        " Entering Gaussian System, Link 0=g16",
        " Initial command:",
        " /g16/l1.exe \"/tmp/Gau-1.inp\" -scrdir=\"/tmp/\"",
        " Entering Link 1 = /g16/l1.exe PID=      9552.",
        " ******************************************",
        " Gaussian 16:  ES64L-G16RevZ.01   1-Jan-2020",
        "                 1-Nov-2026 ",
        " ******************************************",
        *route(route_text),
        " -------",
        " water",
        " -------",
        " NAtoms=      3 NQM=        3 NQMF=       0 NMMI=      0 NMMIF=      0",
        " IAtWgt=          16           1           1",
        " AtmWgt=  15.9949146   1.0078250   1.0078250",
        " NucSpn=           0           1           1",
        " Symbolic Z-matrix:",
        f" Charge = {charge:2d} Multiplicity = {multiplicity}",
        " O                     0.        0.        0.119 ",
        " H                     0.        0.763    -0.477 ",
        " H                     0.        0.763    -0.477 ",
        " ",
    ]


def population(mulliken=(-0.8, 0.4, 0.4), dipole=(0.0, 0.0, -2.1), density="SCF"):
    return [
        f"          Population analysis using the {density} density.",
        " **********************************************************************",
        " Orbital symmetries:",
        "       Occupied  (A1) (A1) (B2) (A1) (B1)",
        "       Virtual   (A1) (B2)",
        " The electronic state is 1-A1.",
        " Alpha  occ. eigenvalues --  -19.13000  -0.99000  -0.52000  -0.36000  -0.29000",
        " Alpha virt. eigenvalues --    0.06000   0.15000",
        "          Condensed to atoms (all electrons):",
        " Mulliken charges:",
        "               1",
        f"     1  O   {mulliken[0]:9.6f}",
        f"     2  H   {mulliken[1]:9.6f}",
        f"     3  H   {mulliken[2]:9.6f}",
        " Sum of Mulliken charges =   0.00000",
        " Dipole moment (field-independent basis, Debye):",
        f"    X=         {dipole[0]:11.4f}    Y=         {dipole[1]:11.4f}    Z=         {dipole[2]:11.4f}  Tot=         {sum(v * v for v in dipole) ** 0.5:11.4f}",
    ]


def frequencies():
    """Water's three vibrations, as Freq prints them, three to a block."""
    return [
        " Harmonic frequencies (cm**-1), IR intensities (KM/Mole), Raman scattering",
        " activities (A**4/AMU), depolarization ratios for plane and unpolarized",
        " incident light, reduced masses (AMU), force constants (mDyne/A),",
        " and normal coordinates:",
        "                      1                      2                      3",
        "                     A1                     A1                     B2",
        " Frequencies --   1700.1234              3700.5678              3800.9012",
        " Red. masses --      1.1000                 1.0500                 1.1000",
        " Frc consts  --      1.8000                 8.4000                 9.2000",
        " IR Inten    --     70.1000                 4.2000                45.3000",
        "  Atom  AN      X      Y      Z        X      Y      Z        X      Y      Z",
        "     1   8     0.00   0.00   0.07     0.00   0.00  -0.05     0.00   0.07   0.00",
        "     2   1     0.00  -0.43  -0.56     0.00   0.58   0.40     0.00  -0.56   0.43",
        "     3   1     0.00   0.43  -0.56     0.00  -0.58   0.40     0.00  -0.56  -0.43",
        "",
        " -------------------",
        " - Thermochemistry -",
        " -------------------",
        " Temperature   298.150 Kelvin.  Pressure   1.00000 Atm.",
        " Full point group                 C2V     NOp   4",
        " Rotational symmetry number  2.",
        " Zero-point correction=                           0.021347 (Hartree/Particle)",
        " Thermal correction to Energy=                    0.024113",
        " Thermal correction to Enthalpy=                  0.025057",
        " Thermal correction to Gibbs Free Energy=         0.007129",
        " Sum of electronic and zero-point Energies=            -12.389000",
        " Sum of electronic and thermal Energies=               -12.386000",
        " Sum of electronic and thermal Enthalpies=             -12.380000",
        " Sum of electronic and thermal Free Energies=          -12.400000",
        "",
        "                     E (Thermal)             CV                S",
        "                      KCal/Mol        Cal/Mol-Kelvin    Cal/Mol-Kelvin",
        " Total                   15.100              6.000             45.100",
        " Electronic               0.000              0.000              0.000",
    ]


def archive(nimag=0, sep="\\"):
    """An archive entry, wrapped at 70 characters, each line begun by a space."""
    body = sep.join(["1", "1", "GINC-HOST", "Freq", "RB3LYP", "6-31G(d)", "H2O1", "USER", "1-Nov-2026", "0", "", "#p b3lyp/6-31G(d) freq",
                     "", "water", "", "0,1", "O,0.,0.,0.119", "", "Version=ES64L-G16RevZ.01", "State=1-A1", f"HF=-12.41", f"NImag={nimag}", "", "@"])
    return [" " + body[i : i + 70] for i in range(0, len(body), 70)]


def opt_freq():
    """A water Opt Freq: two optimisation steps, its converged geometry printed once more, then its Freq step at that
    geometry, the same SCF energy; all its parts as Gaussian 16 prints them."""
    return "\n".join(
        head("#p B3LYP/6-31G(d) Opt Freq")
        + [" Standard basis: 6-31G(d) (6D, 7F)"]
        + orientation(WATER) + scf(-12.40) + forces([(8, 0.0, 0.0, 0.03), (1, 0.0, 0.01, -0.015), (1, 0.0, -0.01, -0.015)])
        + orientation(WATER_2) + scf(-12.41) + forces([(8, 0.0, 0.0, 0.0), (1, 0.0, 0.0, 0.0), (1, 0.0, 0.0, 0.0)])
        + [" Optimization completed.", "    -- Stationary point found."]
        + orientation(WATER_2) + population()
        + [" Normal termination of Gaussian 16 at Sun Nov  1 10:00:00 2026.", " Link1:  Proceeding to internal job step number  2."]
        + route("#P Geom=AllCheck Guess=TCheck SCRF=Check GenChk RB3LYP/6-31G(d) Freq")
        + orientation(WATER_2) + scf(-12.41) + forces([(8, 0.0, 0.0, 0.0), (1, 0.0, 0.0, 0.0), (1, 0.0, 0.0, 0.0)])
        + population(dipole=(0.0, 0.0, -2.0)) + frequencies()
        + [" Optimization completed.", "    -- Stationary point found."]
        + archive()
        + [" Normal termination of Gaussian 16 at Sun Nov  1 10:01:00 2026.", ""]
    )


def results(out):
    return {r["id"]: r for r in out["results"]}


class TheRoute(unittest.TestCase):
    def test_a_route_wrapped_over_lines_read_whole(self):
        lines = route("#p b3lyp/6-31G(d) opt=(calcfc,tight) freq=noraman scrf=(smd,solvent=dichloromethane) pop=nbo")
        (_, text), = reader.routes([""] + lines)
        self.assertEqual(text, "#p b3lyp/6-31G(d) opt=(calcfc,tight) freq=noraman scrf=(smd,solvent=dichloromethane) pop=nbo")
        self.assertEqual(reader.keyword(text, "SCRF"), "scrf=(smd,solvent=dichloromethane)")
        self.assertEqual(reader.options_of(reader.keyword(text, "opt")), ["calcfc", "tight"])
        self.assertIsNone(reader.keyword(text, "irc"))

    def test_the_method_as_the_scf_names_it_the_basis_as_standard_basis_prints_it(self):
        lines = head("#p b3lyp/6-31g(d) opt") + [" Standard basis: 6-31G(d) (6D, 7F)"] + scf(-12.4)
        self.assertEqual(reader.level_of(reader.routes(lines), lines), ("B3LYP", "6-31G(d)"))
        lines = head("#p wb97xd/6-31G(d)") + scf(-12.4, "RwB97XD")
        self.assertEqual(reader.level_of(reader.routes(lines), lines), ("wB97XD", "6-31G(d)"))

    def test_a_post_scf_method_as_the_route_names_it_in_capitals(self):
        lines = head("#p mp2/sto-3g") + [" Standard basis: STO-3G (5D, 7F)"] + scf(-74.96, "RHF")
        self.assertEqual(reader.level_of(reader.routes(lines), lines), ("MP2", "STO-3G"))

    def test_solvation_as_the_route_says_it(self):
        gas = head("#p b3lyp/6-31G(d) opt")
        self.assertEqual(reader.solvation_of(reader.routes(gas), gas), "none")
        smd = head("#p b3lyp/6-31G(d) scrf=(smd,solvent=water) opt") + [" Solvent              : Water, Eps=  78.400000 Eps(inf)=   1.800000"]
        self.assertEqual(reader.solvation_of(reader.routes(smd), smd), "SMD(Water)")
        # (a model named alone takes the manual's default solvent, water; none named, IEFPCM)
        cpcm = head("#p b3lyp/6-31G(d) scrf=cpcm")
        self.assertEqual(reader.solvation_of(reader.routes(cpcm), cpcm), "CPCM(Water)")
        pcm = head("#p b3lyp/6-31G(d) scrf=(solvent=toluene)")
        self.assertEqual(reader.solvation_of(reader.routes(pcm), pcm), "IEFPCM(toluene)")
        # (a step that takes its solvation from the checkpoint says what the step before it said)
        self.assertEqual(reader.read("water.log", opt_freq())["solvation"], "none")

    def test_unsaid_where_no_route_was_read(self):
        self.assertIsNone(reader.solvation_of([], []))


class TheRun(unittest.TestCase):
    def test_normal_error_and_cut(self):
        normal = " Entering Gaussian System\n SCF Done:  E(RHF) =  -75.98\n Normal termination of Gaussian 16 at Tue Oct 13.\n"
        self.assertEqual(reader.run_of(normal), ("normal", None, [True]))
        error = (" Entering Gaussian System\n A made-up reason the input cannot be run.\n"
                 " Error termination via Lnk1e in /g16/l301.exe at Tue Oct 13.\n")
        self.assertEqual(reader.run_of(error)[:2], ("error", "A made-up reason the input cannot be run."))
        self.assertEqual(reader.run_of(" Entering Gaussian System\n SCF Done:  E(RHF) =  -75.98\n")[0], "cut")

    def test_an_scf_that_did_not_converge(self):
        unmet = " >>>>>>>>>> Convergence criterion not met.\n SCF Done:  E(RHF) =  -75.98\n SCF Done:  E(RHF) =  -75.99\n"
        self.assertEqual(reader.run_of(unmet)[2], [False, True])

    def test_a_job_that_ended_normally_with_its_scf_not_converged_says_so_and_gives_no_energy(self):
        text = "\n".join(head("#p ub3lyp/6-31G(d) scf=(maxcycle=3)", 0, 3) + orientation(WATER) + [" >>>>>>>>>> Convergence criterion not met."]
                         + scf(-39.15, "UB3LYP") + [" Normal termination of Gaussian 16 at Sun Nov  1 10:00:00 2026."])
        o = reader.read("w.log", text)
        self.assertIsNone(o["energies"])
        self.assertEqual((results(o)["run"]["label"], results(o)["run"]["value"]), ("SCF", "not converged"))

    def test_an_optimisation_out_of_steps_says_so_not_the_quotation_before_its_end(self):
        text = ("\n Optimization stopped.\n    -- a made-up reason, in its words\n    -- more made-up words\n"
                " SOME WORDS OF WISDOM.\n\n                       -- AN AUTHOR\n Error termination request processed by link 9999.\n"
                " Error termination via Lnk1e in /g16/l9999.exe at Sun Nov  1 10:00:00 2026.\n Job cpu time:       0 days\n")
        self.assertEqual(reader.run_of(text)[:2], ("error", "Optimization stopped. -- a made-up reason, in its words"))

    def test_an_opt_freq_cut_in_its_frequencies_did_not_come_to_its_end(self):
        text = opt_freq()
        cut = text[: text.index(" Harmonic frequencies")]
        self.assertEqual(reader.run_of(cut)[0], "cut")
        out = reader.read("water.log", cut)
        self.assertEqual(results(out)["run"]["value"], "did not come to its end")

    def test_said_first_in_the_chip_s_line_with_why(self):
        r = reader.run_results("error", "Convergence failure -- run terminated.", [False])
        self.assertEqual([(x["id"], x["value"]) for x in r], [("run", "ended with an error"), ("run.why", "Convergence failure -- run terminated.")])
        self.assertEqual(reader.run_results("normal", None, [True, False])[0]["label"], "SCF")
        self.assertEqual(reader.run_results("normal", None, [True]), [])


class AnOptFreq(unittest.TestCase):
    def setUp(self):
        self.out = reader.read("water.log", opt_freq())

    def test_what_the_calculation_was(self):
        o = self.out
        self.assertEqual((o["program"], o["version"], o["method"], o["basis"]), ("Gaussian", "16, Revision Z.01", "B3LYP", "6-31G(d)"))
        self.assertEqual((o["charge"], o["multiplicity"]), (0, 1))
        self.assertEqual(o["atoms"], ["O", "H", "H"])
        self.assertEqual(o["masses"], [15.9949146, 1.007825, 1.007825])

    def test_each_geometry_with_its_energy_the_reprint_not_again(self):
        o = self.out
        # (two steps, then the Freq step's: the converged one printed once more, with no SCF of its own, is no geometry)
        self.assertEqual(len(o["frames"]), 3)
        self.assertEqual(o["frames"][0][:3], [0.0, 0.0, 0.119])
        self.assertEqual(o["energies"], [-12.40, -12.41, -12.41])

    def test_the_freq_step_s_point_is_the_one_entry(self):
        self.assertTrue(self.out["optimised"])
        self.assertEqual(self.out["converged"], [False, False, True])

    def test_its_rms_gradient_at_each_geometry(self):
        g = results(self.out)["rms-gradient"]
        self.assertAlmostEqual(g["values"][0], ((0.03 ** 2 + 2 * 0.01 ** 2 + 2 * 0.015 ** 2) / 9) ** 0.5)
        self.assertEqual(g["values"][1:], [0.0, 0.0])

    def test_its_thermochemistry_as_printed(self):
        r = results(self.out)
        self.assertEqual(r["thermo.free-energy"]["value"], -12.4)
        self.assertEqual(r["thermo.free-energy"]["rank"], 1)
        self.assertEqual(r["thermo.enthalpy"]["value"], -12.38)
        self.assertEqual(r["thermo.zpve"]["value"], 0.021347)
        self.assertEqual(r["thermo.g-correction"]["value"], 0.007129)
        self.assertEqual(r["thermo.entropy"]["value"], 45.1)
        self.assertEqual((r["thermo.temperature"]["value"], r["thermo.pressure"]["value"]), (298.15, 1.0))
        self.assertEqual((r["thermo.point-group"]["value"], r["thermo.symmetry-number"]["value"]), ("C2V", 2))

    def test_its_vibrations_each_moving_the_molecule(self):
        v = results(self.out)["vibrations"]
        self.assertEqual([c["label"] for c in v["columns"]], ["Mode", "Symmetry", "Frequency", "Reduced mass", "IR intensity"])
        self.assertEqual(v["columns"][1]["quantity"], "symmetry")
        self.assertEqual(v["rows"][0]["cells"], [1, "A1", 1700.1234, 1.1, 70.1])
        self.assertEqual(v["rows"][2]["move"], [0.0, 0.07, 0.0, 0.0, -0.56, 0.43, 0.0, -0.56, -0.43])
        self.assertEqual(self.out["imaginary"], [])

    def test_the_last_population_analysis_s(self):
        r = results(self.out)
        self.assertEqual(r["charges.mulliken"]["values"], [-0.8, 0.4, 0.4])
        d = r["dipole"]
        self.assertEqual((d["value"], d["vector"], d["origin"]), (2.0, [0.0, 0.0, -2.0], [0.0, 0.0, 0.0]))

    def test_every_orbital_in_meno_s_form_each_its_energy_in_hartrees(self):
        # (Meno names them and lists them: no list or gap of the reader's own)
        r = results(self.out)
        self.assertNotIn("orbitals", r)
        self.assertNotIn("homo-lumo-gap", r)
        o = self.out["orbitals"]
        self.assertEqual([(x["spin"], x["occupation"]) for x in o], [("both", 2.0)] * 5 + [("both", 0.0)] * 2)
        self.assertEqual([x["energy"] for x in o[3:6]], [-0.36, -0.29, 0.06])
        self.assertEqual(o[5]["symmetry"], "A1")


class Geometries(unittest.TestCase):
    def test_the_input_orientation_where_no_standard_one_is_printed(self):
        text = "\n".join(head("#p hf/3-21g opt nosymm") + orientation(WATER, "Input orientation:") + scf(-75.5) + [" Normal termination of Gaussian 16"])
        self.assertEqual(reader.read("w.log", text)["frames"][0][:3], [0.0, 0.0, 0.119])

    def test_the_standard_one_where_both_are(self):
        moved = [(z, x + 1.0, y, w) for z, x, y, w in WATER]
        text = "\n".join(head("#p hf/3-21g") + orientation(moved, "Input orientation:") + orientation(WATER) + scf(-75.5) + [" Normal termination of Gaussian 16"])
        self.assertEqual(reader.read("w.log", text)["frames"], [[v for _, x, y, w in WATER for v in (x, y, w)]])

    def test_a_run_stopped_after_its_last_energy_keeps_the_geometries_it_has_one_for(self):
        text = "\n".join(head("#p hf/3-21g opt") + orientation(WATER) + scf(-75.5) + orientation(WATER_2) + [" Error termination via Lnk1e in /g16/l502.exe"])
        o = reader.read("w.log", text)
        self.assertEqual((len(o["frames"]), o["energies"]), (1, [-75.5]))
        self.assertEqual((o["optimised"], o["converged"]), (False, [False]))
        self.assertEqual(results(o)["run"]["value"], "ended with an error")

    def test_ghost_atoms_left_out(self):
        lines = head("#p b3lyp/6-31G(d)")
        k = lines.index(" Symbolic Z-matrix:")
        lines[k + 4] = " He-Bq                 0.        0.        3.0 "
        text = "\n".join(lines + orientation(WATER[:2] + [(2, 0.0, 0.0, 3.0)]) + scf(-12.0)
                         + [" Mulliken charges:", "               1", "     1  O   -0.4", "     2  H    0.4", "     3  He   0.0",
                            " Sum of Mulliken charges =   0.00000", " Normal termination of Gaussian 16"])
        o = reader.read("w.log", text)
        self.assertEqual(o["atoms"], ["O", "H"])
        self.assertEqual(len(o["frames"][0]), 6)
        self.assertEqual(results(o)["charges.mulliken"]["values"], [-0.4, 0.4])


class NotGaussian(unittest.TestCase):
    def test_another_program_s_output_in_gaussian_s_layout_is_refused(self):
        # (xtb's frequencies written for viewers in Gaussian 98's layout: its first line Gaussian's banner without the
        #  program Gaussian ran, then whose it is - that program's output, read by its own plugin)
        text = "\n".join([" Entering Gaussian System", " *********************************************", " Gaussian 98:",
                          " frequency output generated by the xtb code", " *********************************************"]
                         + orientation(WATER) + frequencies())
        with self.assertRaisesRegex(ValueError, "does not begin as Gaussian's own output does"):
            reader.read("g98.out", text)
        r = worker.answer(json.dumps({"id": 8, "op": "read", "name": "g98.out", "text": text}))
        self.assertFalse(r["ok"])


class CutShort(unittest.TestCase):
    def test_a_scan_cut_after_its_last_scf_its_last_geometry_without_an_rms_gradient_the_others_theirs(self):
        # (a relaxed scan cut short: its last geometry's SCF printed, its forces not)
        zero = [(8, 0.0, 0.0, 0.0), (1, 0.0, 0.0, 0.0), (1, 0.0, 0.0, 0.0)]
        moved = [(z, x, y + 0.1, w) for z, x, y, w in WATER]
        text = "\n".join(head("#p hf/3-21g opt=modredundant") + orientation(WATER) + scf(-75.5) + forces([(8, 0.0, 0.0, 0.03)] + zero[1:])
                         + [" Optimization completed.", "    -- Stationary point found."]
                         + orientation(WATER_2) + scf(-75.6) + forces(zero) + orientation(moved) + scf(-75.7))
        o = reader.read("w.log", text)
        self.assertEqual(results(o)["run"]["value"], "did not come to its end")
        self.assertEqual(o["energies"], [-75.5, -75.6, -75.7])
        g = results(o)["rms-gradient"]["values"]
        self.assertEqual((round(g[0], 6), g[1], g[2]), (0.01, 0.0, None))


class Energies(unittest.TestCase):
    def test_the_most_correlated_each_geometry_has_as_printed(self):
        # (made-up numbers in the printed layout: an SCF of -74.96, its MP2 correction -0.04, a CCSD(T) a little lower)
        text = "\n".join(head("#p mp2/sto-3g") + orientation(WATER) + scf(-74.96, "RHF")
                         + [" E2 =    -0.4000000000D-01 EUMP2 =    -0.75000000000000D+02"] + [" Normal termination of Gaussian 16"])
        self.assertEqual(reader.read("w.log", text)["energies"], [-75.0])
        text = text.replace(" Normal", " CCSD(T)= -0.75010000000D+02\n Normal")
        self.assertEqual(reader.read("w.log", text)["energies"], [-75.01])

    def test_a_field_too_narrow_for_its_number_is_not_known(self):
        self.assertIsNone(reader.number("**********"))
        self.assertEqual(reader.numbers("-10.10000-10.05000 *******"), [-10.1, -10.05, None])


class OpenShell(unittest.TestCase):
    def test_spin_populations_and_each_spin_s_orbitals(self):
        text = "\n".join(head("#p ub3lyp/6-31G(d)", 0, 3) + orientation(WATER) + scf(-39.0, "UB3LYP") + [
            " Alpha  occ. eigenvalues --  -10.20000  -0.70000  -0.30000  -0.20000",
            " Alpha virt. eigenvalues --    0.05000",
            "  Beta  occ. eigenvalues --  -10.10000  -0.60000",
            "  Beta virt. eigenvalues --   -0.10000   0.07000",
            " Mulliken charges and spin densities:",
            "               1          2",
            "     1  O   -0.300000   1.900000",
            "     2  H    0.150000   0.050000",
            "     3  H    0.150000   0.050000",
            " Sum of Mulliken charges =   0.00000   2.00000",
            " Normal termination of Gaussian 16",
        ])
        r = results(reader.read("w.log", text))
        self.assertEqual(r["spins.mulliken"]["values"], [1.9, 0.05, 0.05])
        self.assertEqual(r["charges.mulliken"]["values"], [-0.3, 0.15, 0.15])
        o = reader.read("w.log", text)["orbitals"]
        self.assertEqual([(x["spin"], x["occupation"]) for x in o], [("alpha", 1.0)] * 4 + [("alpha", 0.0)] + [("beta", 1.0)] * 2 + [("beta", 0.0)] * 2)
        self.assertEqual(o[-2]["energy"], -0.1)
        self.assertNotIn("orbitals.a", r)


class OtherCharges(unittest.TestCase):
    def lines(self, block, pop="nbo"):
        return "\n".join(head(f"#p b3lyp/6-31G(d) pop={pop}") + orientation(WATER) + scf(-12.4) + block + [" Normal termination of Gaussian 16"])

    def test_hirshfeld_and_cm5(self):
        text = self.lines([
            " Hirshfeld charges, spin densities, dipoles, and CM5 charges using IRadAn=      4:",
            "                Q-H        S-H        Dx         Dy         Dz        Q-CM5",
            "     1  O   -0.330000   0.000000   0.000000   0.000000  -0.200000  -0.650000",
            "     2  H    0.165000   0.000000   0.000000   0.100000   0.050000   0.325000",
            "     3  H    0.165000   0.000000   0.000000  -0.100000   0.050000   0.325000",
            "       Tot   0.000000   0.000000   0.000000   0.000000  -0.100000   0.000000",
        ], "hirshfeld")
        r = results(reader.read("w.log", text))
        self.assertEqual(r["charges.hirshfeld"]["values"], [-0.33, 0.165, 0.165])
        self.assertEqual(r["charges.cm5"]["values"], [-0.65, 0.325, 0.325])
        self.assertNotIn("spins.hirshfeld", r)

    def test_an_electrostatic_fit_named_by_the_route(self):
        text = self.lines([" ESP charges:", "               1", "     1  O   -0.700000", "     2  H    0.350000", "     3  H    0.350000",
                           " Sum of ESP charges =   0.00000"], "chelpg")
        self.assertEqual(results(reader.read("w.log", text))["charges.chelpg"]["values"], [-0.7, 0.35, 0.35])

    def test_natural_charges(self):
        text = self.lines([
            " Summary of Natural Population Analysis:",
            "",
            "                                       Natural Population",
            "                Natural  -----------------------------------------------",
            "    Atom  No    Charge         Core      Valence    Rydberg      Total",
            " -----------------------------------------------------------------------",
            "      O    1   -0.90000      2.00000     6.87609    0.02391     8.90000",
            "      H    2    0.45000      0.00000     0.51483    0.03517     0.55000",
            "      H    3    0.45000      0.00000     0.51483    0.03517     0.55000",
            " =======================================================================",
        ])
        self.assertEqual(results(reader.read("w.log", text))["charges.natural"]["values"], [-0.9, 0.45, 0.45])


class Paths(unittest.TestCase):
    def test_a_reaction_path_s_points_in_the_order_of_its_coordinate(self):
        # (each point's geometry, then its summary; the start's, where the job gives it, before its own)
        def point(xyz, e, n, path):
            return orientation(xyz, "Input orientation:") + scf(e) + [f" Point Number:{n:4d}          Path Number:{path:4d}"]
        moved = [(z, x, y + 0.1, w) for z, x, y, w in WATER]
        text = "\n".join(head("#p b3lyp/6-31G(d) irc=(calcfc,maxpoints=2)") + point(WATER, -12.40, 0, 1) + point(WATER_2, -12.41, 1, 1)
                         + point(moved, -12.42, 2, 1) + point(WATER_2, -12.43, 1, 2) + [" Normal termination of Gaussian 16"])
        o = reader.read("w.log", text)
        self.assertEqual(o["energies"], [-12.43, -12.40, -12.41, -12.42])
        self.assertEqual((o["unmarked"], results(o)["run.path"]["value"]), ("path", "4 points"))
        # (a path started from the checkpoint: its start's geometry not among them)
        text = "\n".join(head("#p b3lyp/6-31G(d) irc=(rcfc,maxpoints=2) geom=check") + [" Point Number:   0          Path Number:   1"]
                         + point(WATER_2, -12.41, 1, 1) + point(moved, -12.42, 2, 1) + point(WATER, -12.43, 1, 2) + [" Normal termination of Gaussian 16"])
        self.assertEqual(reader.read("w.log", text)["energies"], [-12.43, -12.41, -12.42])

    def test_a_rigid_scan_s_points_are_a_path_not_an_optimisation(self):
        text = "\n".join(head("#p hf/3-21g scan") + orientation(WATER) + scf(-75.5) + orientation(WATER_2) + scf(-75.6)
                         + [" Normal termination of Gaussian 16"])
        o = reader.read("w.log", text)
        self.assertIsNone(o["optimised"])
        self.assertEqual((o["unmarked"], results(o)["run.path"]["value"]), ("path", "2 points"))

    def test_an_excited_state_s_geometries_have_their_energies_unread(self):
        text = "\n".join(head("#p b3lyp/6-31G(d) td opt") + orientation(WATER) + scf(-12.4)
                         + [" Excited State   1:      Singlet-B1     7.1000 eV  174.63 nm  f=0.0137  <S**2>=0.000"]
                         + orientation(WATER_2) + scf(-12.39) + population(density="CI") + [" Normal termination of Gaussian 16"])
        o = reader.read("w.log", text)
        self.assertIsNone(o["energies"])
        r = results(o)
        self.assertEqual(r["run.energies"]["value"], "energies not read")
        # (its charges and dipole of a density the output does not say is the ground state's: left out)
        self.assertNotIn("charges.mulliken", r)
        self.assertNotIn("dipole", r)
        # (its orbitals, the SCF's, given as any job's are)
        self.assertTrue(o["orbitals"])


class SymmetryLabels(unittest.TestCase):
    def test_a_linear_molecule_s_species_and_point_group_in_standard_notation(self):
        self.assertEqual([reader.species(x) for x in ("SGG", "SGU", "PIU", "PIG", "DLTG", "PHIU", "SG", "PI")], ["\u03a3g", "\u03a3u", "\u03a0u", "\u03a0g", "\u0394g", "\u03a6u", "\u03a3", "\u03a0"])
        self.assertEqual((reader.point_group("D*H"), reader.point_group("C*V")), ("D\u221eh", "C\u221ev"))

    def test_others_as_printed(self):
        self.assertEqual([reader.species(x) for x in ("BU", "A'", 'A"', "?A")], ["BU", "A'", 'A"', "?A"])
        self.assertEqual(reader.point_group("C2V"), "C2V")

    def test_as_read_from_a_linear_molecule_s_lines(self):
        lines = [" Orbital symmetries:", "       Occupied  (SGG) (SGU) (SGG) (SGU) (SGG) (PIU) (PIU)", "                 (PIG) (PIG)",
                 "       Virtual   (PIU) (PIU) (SGG)", " The electronic state is 1-SGG."]
        self.assertEqual(reader.orbital_symmetries(lines)[0][:6], ["\u03a3g", "\u03a3u", "\u03a3g", "\u03a3u", "\u03a3g", "\u03a0u"])


class Archive(unittest.TestCase):
    def test_nimag_even_where_a_line_break_splits_it(self):
        lines = archive(1)
        text = "\n".join(lines)
        self.assertEqual(reader.nimag_of(text), 1)
        self.assertEqual(reader.nimag_of("\n".join(archive(2, "|"))), 2)
        self.assertIsNone(reader.nimag_of(" SCF Done"))


class Jobs(unittest.TestCase):
    def test_link1_jobs_each_after_the_file_s_first_line(self):
        one = opt_freq()
        two = "\n".join(head("#p geom=allcheck b3lyp/6-311+G(d,p)")[1:] + orientation(WATER_2) + scf(-12.45) + [" Normal termination of Gaussian 16", ""])
        texts = reader.jobs(one + "\n" + two)
        self.assertEqual(len(texts), 2)
        self.assertTrue(texts[1].startswith(" Entering Gaussian System"))
        out = reader.read_jobs("water.log", one + "\n" + two)
        self.assertEqual([j["job"] for j in out["jobs"]], [{"number": 1, "of": 2, "kind": "Optimisation and frequencies"}, {"number": 2, "of": 2, "kind": "Single point"}])
        self.assertEqual(out["energies"], [-12.45])

    def test_each_begun_where_it_enters_its_first_link_with_no_command_said_before_it(self):
        # (Gaussian 16 on Windows says the command it was started by before its first job alone)
        two = "\n".join([line for line in head("#p geom=allcheck b3lyp/6-311+G(d,p)")[1:] if "Initial command" not in line and "Gau-1.inp" not in line]
                        + orientation(WATER_2) + scf(-12.45) + [" Normal termination of Gaussian 16", ""])
        self.assertEqual(len(reader.jobs(opt_freq() + "\n" + two)), 2)

    def test_a_file_of_one_job_read_whole(self):
        self.assertNotIn("jobs", reader.read_jobs("water.log", opt_freq()))


FCHK = """water
SP        RB3LYP                        6-31G(d)
Number of atoms                            I                3
Route                                      C   N=           2
#P B3LYP/6-31G(d) Opt Fr eq
Charge                                     I                0
Multiplicity                               I                1
Number of alpha electrons                  I                5
Number of beta electrons                   I                5
Atomic numbers                             I   N=           3
           8           1           1
Current cartesian coordinates              R   N=           9
  0.00000000E+00  0.00000000E+00  2.25000000E-01  0.00000000E+00  1.44000000E+00
 -9.00000000E-01  0.00000000E+00 -1.44000000E+00 -9.00000000E-01
Total Energy                               R     -1.24100000000000E+01
Alpha Orbital Energies                     R   N=           7
 -1.91300000E+01 -9.90000000E-01 -5.20000000E-01 -3.60000000E-01 -2.90000000E-01
  6.00000000E-02  1.50000000E-01
Mulliken Charges                           R   N=           3
 -8.00000000E-01  4.00000000E-01  4.00000000E-01
Dipole Moment                              R   N=           3
  0.00000000E+00  0.00000000E+00 -8.00000000E-01
Gaussian Version                           C   N=           2
ES64L-G16RevZ.01
"""


class AFormattedCheckpoint(unittest.TestCase):
    """As "Interfacing to Gaussian 16" lays one out: each item its label (A40), type and value, or "N=" and its values
    after, so many to a line - a text list's lines read as text, never as an item."""

    def setUp(self):
        self.out = reader.read_jobs("water.fchk", FCHK)

    def test_told_from_a_log(self):
        self.assertTrue(reader.is_fchk(FCHK))
        self.assertFalse(reader.is_fchk(opt_freq()))

    def test_its_molecule_in_angstroms_its_energy_and_level(self):
        o = self.out
        self.assertEqual(o["atoms"], ["O", "H", "H"])
        self.assertAlmostEqual(o["frames"][0][4], 1.44 * reader.BOHR)
        self.assertEqual((o["energies"], o["method"], o["basis"], o["version"]), ([-12.41], "B3LYP", "6-31G(d)", "16, Revision Z.01"))
        self.assertEqual((o["charge"], o["multiplicity"]), (0, 1))

    def test_its_charges_dipole_in_debye_and_orbitals(self):
        r = results(self.out)
        self.assertEqual(r["charges.mulliken"]["values"], [-0.8, 0.4, 0.4])
        self.assertAlmostEqual(r["dipole"]["value"], 0.8 * reader.DEBYE_PER_E_BOHR)
        self.assertEqual(r["dipole"]["origin"], [0.0, 0.0, 0.0])
        # (its orbitals in Meno's form: as many occupied, of both spins, as its electrons of each)
        o = self.out["orbitals"]
        self.assertEqual({x["spin"] for x in o}, {"both"})
        self.assertEqual(sum(x["occupation"] for x in o), 10)
        self.assertNotIn("orbitals", r)


def fchk_item(label, kind, values):
    """An item as "Interfacing to Gaussian 16" lays one out: its label (A40), its type, and "N=" and its values after,
    six integers (I12) or five reals (E16.8) to a line."""
    per, form = (6, "{:12d}") if kind == "I" else (5, "{:16.8E}")
    lines = [f"{label:<40}  {kind}   N={len(values):12d}"]
    lines += ["".join(form.format(v) for v in values[i : i + per]) for i in range(0, len(values), per)]
    return lines


# (made up: two hydrogens 1.4 bohr apart; on the first an s shell and an sp shell of two primitives each - the sp
#  shell's p coefficients not its s ones - and a pure d shell of one, on the second an s shell - 1 + 4 + 5 + 1
#  functions - and two orbitals of each spin)
BASIS = {
    "Shell types": ("I", [0, -1, -2, 0]),
    "Number of primitives per shell": ("I", [2, 2, 1, 1]),
    "Shell to atom map": ("I", [1, 1, 1, 2]),
    "Primitive exponents": ("R", [3.4, 0.6, 0.8, 0.25, 1.1, 0.6]),
    "Contraction coefficients": ("R", [0.15, 0.9, 0.3, 0.8, 1.0, 1.0]),
    "P(S=P) Contraction coefficients": ("R", [0.0, 0.0, 0.6, 0.5, 0.0, 0.0]),
}


def basis_fchk(beta=False, **changed):
    items = {
        "Atomic numbers": ("I", [1, 1]),
        "Current cartesian coordinates": ("R", [0.0, 0.0, 0.0, 0.0, 0.0, 1.4]),
        **BASIS,
        "Alpha Orbital Energies": ("R", [-0.5, 0.2]),
        "Alpha MO coefficients": ("R", [float(i) for i in range(22)]),
        **({"Beta Orbital Energies": ("R", [-0.4, 0.3]), "Beta MO coefficients": ("R", [float(-i) for i in range(22)])} if beta else {}),
        **changed,
    }
    lines = ["h2", "SP        UHF                           made-up", "Number of atoms                            I                2",
             "Number of basis functions                  I               11",
             f"Number of alpha electrons                  I                1", f"Number of beta electrons                   I                {0 if beta else 1}"]
    for label, item in items.items():
        if item is not None:
            lines += fchk_item(label, *item)
    return "\n".join(lines) + "\n"


def self_overlap(e, f, g=None):
    """Two functions' overlap from their primitives, all on one centre: x^p e^(-a r^2) and x^q e^(-b r^2) overlap as the
    product, over x, y and z, of the integral of x^(p + q) e^(-(a + b) x^2)."""
    g = f if g is None else g
    return sum(c * d * math.prod(reader.moment(e["powers"][i][k] + e["powers"][j][k], e["exponents"][i] + e["exponents"][j]) for k in range(3))
               for i, c in zip(f["primitives"], f["coefficients"]) for j, d in zip(g["primitives"], g["coefficients"]))


class AFormattedCheckpointSWavefunction(unittest.TestCase):
    """Its basis and orbitals given whole, in Meno's general form (src/lib/calc/expansion.ts): Meno draws the orbitals,
    checking them orthonormal first."""

    def test_its_primitives_functions_and_orbitals(self):
        e = reader.fchk_expansion(basis_fchk())
        self.assertEqual(e["nuclei"], [0.0, 0.0, 0.0, 0.0, 0.0, 1.4])
        # (the s shell's two primitives; the sp shell's s of each of its exponents, then its p; the d shell's six Cartesian ones)
        self.assertEqual(e["centres"], [0] * 16 + [1])
        self.assertEqual(e["powers"][:10], [[0, 0, 0]] * 4 + [[1, 0, 0], [0, 1, 0], [0, 0, 1]] * 2)
        self.assertEqual(e["exponents"], [3.4, 0.6, 0.8, 0.25] + [0.8] * 3 + [0.25] * 3 + [1.1] * 6 + [0.6])
        self.assertEqual(len(e["functions"]), 11)
        self.assertEqual(e["functions"][0]["primitives"], [0, 1])
        self.assertEqual([f["primitives"] for f in e["functions"][1:5]], [[2, 3], [4, 7], [5, 8], [6, 9]])
        # (orbital by orbital, each its coefficient of every function)
        self.assertEqual(e["orbitals"], [[float(i) for i in range(11)], [float(i) for i in range(11, 22)]])

    def test_each_primitive_normalised_times_its_coefficient_the_sp_shell_s_p_its_own(self):
        e = reader.fchk_expansion(basis_fchk())
        norm_s = lambda a: (2 * a / math.pi) ** 0.75
        norm_p = lambda a: norm_s(a) * 2 * math.sqrt(a)
        ratio = lambda f: f["coefficients"][0] / f["coefficients"][1]
        # (the s shell: d1 N(a1) to d2 N(a2), whatever the function's own norm)
        self.assertAlmostEqual(ratio(e["functions"][0]), 0.15 * norm_s(3.4) / (0.9 * norm_s(0.6)), places=12)
        # (the sp shell: its s from the contraction coefficients, its p from the P(S=P) ones)
        self.assertAlmostEqual(ratio(e["functions"][1]), 0.3 * norm_s(0.8) / (0.8 * norm_s(0.25)), places=12)
        for f in e["functions"][2:5]:
            self.assertAlmostEqual(ratio(f), 0.6 * norm_p(0.8) / (0.5 * norm_p(0.25)), places=12)

    def test_each_function_normalised_and_a_shell_s_pure_functions_orthogonal(self):
        e = reader.fchk_expansion(basis_fchk())
        for f in e["functions"]:
            self.assertAlmostEqual(self_overlap(e, f), 1.0, places=12)
        d = e["functions"][5:10]
        for i in range(5):
            for j in range(i):
                self.assertAlmostEqual(self_overlap(e, d[i], d[j]), 0.0, places=12)

    def test_pure_functions_are_the_reference_s_polynomials_in_its_order(self):
        def by(l, m, term):
            """The solid harmonic's coefficients over its coefficient of `term`, which is positive."""
            p = reader.solid_harmonic(l, m)
            self.assertGreater(p[term], 0)
            return {k: round(c / p[term], 12) for k, c in p.items()}
        # (d: 3ZZ-RR, XZ, YZ, XX-YY, XY)
        self.assertEqual(by(2, 0, (0, 0, 2)), {(0, 0, 2): 1.0, (2, 0, 0): -0.5, (0, 2, 0): -0.5})
        self.assertEqual([reader.solid_harmonic(2, m).keys() for m in (1, -1, 2, -2)], [{(1, 0, 1)}, {(0, 1, 1)}, {(2, 0, 0), (0, 2, 0)}, {(1, 1, 0)}])
        self.assertEqual(by(2, 2, (2, 0, 0)), {(2, 0, 0): 1.0, (0, 2, 0): -1.0})
        # (f: ZZZ-ZRR, XZZ-XRR, YZZ-YRR, XXZ-YYZ, XYZ, XXX-XYY, XXY-YYY - each its first term positive)
        self.assertEqual(by(3, 0, (0, 0, 3)), {(0, 0, 3): 1.0, (2, 0, 1): -1.5, (0, 2, 1): -1.5})
        self.assertEqual(by(3, 1, (1, 0, 2)), {(1, 0, 2): 1.0, (3, 0, 0): -0.25, (1, 2, 0): -0.25})
        self.assertEqual(by(3, -1, (0, 1, 2)), {(0, 1, 2): 1.0, (2, 1, 0): -0.25, (0, 3, 0): -0.25})
        self.assertEqual(by(3, 2, (2, 0, 1)), {(2, 0, 1): 1.0, (0, 2, 1): -1.0})
        self.assertEqual(by(3, -2, (1, 1, 1)), {(1, 1, 1): 1.0})
        self.assertEqual(by(3, 3, (3, 0, 0)), {(3, 0, 0): 1.0, (1, 2, 0): -3.0})
        self.assertEqual(by(3, -3, (2, 1, 0)), {(2, 1, 0): 1.0, (0, 3, 0): round(-1 / 3, 12)})
        # (g's last two: x^4 - 6x^2y^2 + y^4 and x^3y - xy^3, its labels' +4 and -4)
        self.assertEqual(by(4, 4, (4, 0, 0)), {(4, 0, 0): 1.0, (2, 2, 0): -6.0, (0, 4, 0): 1.0})
        self.assertEqual(by(4, -4, (3, 1, 0)), {(3, 1, 0): 1.0, (1, 3, 0): -1.0})

    def test_a_pure_shell_s_functions_laid_out_in_the_format_s_order(self):
        def shell_functions(t):
            """The functions of one pure shell of type `t` on a lone atom, as the polynomial each is over the primitives'
            powers: each coefficient over its largest."""
            n = {-2: 5, -3: 7, -4: 9}[t]
            text = "\n".join(["one shell", "SP        RHF                           made-up", "Number of atoms                            I                1",
                              f"Number of basis functions                  I   {n:14d}", "Number of alpha electrons                  I                1",
                              "Number of beta electrons                   I                1"]
                             + fchk_item("Atomic numbers", "I", [1]) + fchk_item("Current cartesian coordinates", "R", [0.0, 0.0, 0.0])
                             + fchk_item("Shell types", "I", [t]) + fchk_item("Number of primitives per shell", "I", [1])
                             + fchk_item("Shell to atom map", "I", [1]) + fchk_item("Primitive exponents", "R", [0.7])
                             + fchk_item("Contraction coefficients", "R", [1.0]) + fchk_item("Alpha Orbital Energies", "R", [0.0] * n)
                             + fchk_item("Alpha MO coefficients", "R", [0.0] * n * n)) + "\n"
            e = reader.fchk_expansion(text)
            out = []
            for f in e["functions"]:
                terms = {tuple(e["powers"][i]): c for i, c in zip(f["primitives"], f["coefficients"])}
                top = max(terms.values(), key=abs)
                out.append({k: round(c / top, 9) for k, c in terms.items()})
            return out
        # (d: 3ZZ-RR, XZ, YZ, XX-YY, XY)
        self.assertEqual(shell_functions(-2), [
            {(0, 0, 2): 1.0, (2, 0, 0): -0.5, (0, 2, 0): -0.5}, {(1, 0, 1): 1.0}, {(0, 1, 1): 1.0}, {(2, 0, 0): 1.0, (0, 2, 0): -1.0}, {(1, 1, 0): 1.0}])
        # (f: ZZZ-ZRR, XZZ-XRR, YZZ-YRR, XXZ-YYZ, XYZ, XXX-XYY, XXY-YYY)
        f = shell_functions(-3)
        self.assertEqual([sorted(x) for x in f], [
            [(0, 0, 3), (0, 2, 1), (2, 0, 1)], [(1, 0, 2), (1, 2, 0), (3, 0, 0)], [(0, 1, 2), (0, 3, 0), (2, 1, 0)],
            [(0, 2, 1), (2, 0, 1)], [(1, 1, 1)], [(1, 2, 0), (3, 0, 0)], [(0, 3, 0), (2, 1, 0)]])
        # (each over its largest coefficient, its sign with it: x^3 - 3xy^2 over its -3)
        self.assertEqual((f[3][(2, 0, 1)], f[5][(3, 0, 0)], f[6][(2, 1, 0)]), (1.0, round(-1 / 3, 9), 1.0))
        # (g: m = 0, +1, -1, ... +4, -4 - their powers of z falling, a cosine-like then a sine-like one each |m|)
        g = shell_functions(-4)
        self.assertEqual([max(k[2] for k in x) for x in g], [4, 3, 3, 2, 2, 1, 1, 0, 0])
        self.assertEqual([all(k[1] % 2 == (i % 2 == 0 and i > 0) for k in x) for i, x in enumerate(g)], [True] * 9)
        # (each over its largest coefficient, its sign with it: x^4 - 6x^2y^2 + y^4 over its -6)
        self.assertEqual(g[7], {(4, 0, 0): round(-1 / 6, 9), (2, 2, 0): 1.0, (0, 4, 0): round(-1 / 6, 9)})
        self.assertEqual(g[8], {(3, 1, 0): 1.0, (1, 3, 0): -1.0})

    def test_cartesian_functions_in_gaussian_s_order(self):
        names = lambda L: [("X" * p[0] + "Y" * p[1] + "Z" * p[2]) or "S" for p in reader.CARTESIAN[L]]
        self.assertEqual(names(2), ["XX", "YY", "ZZ", "XY", "XZ", "YZ"])
        self.assertEqual(names(3), ["XXX", "YYY", "ZZZ", "XYY", "XXY", "XXZ", "XZZ", "YZZ", "YYZ", "XYZ"])
        self.assertEqual(names(4)[:3] + names(4)[-2:], ["ZZZZ", "YZZZ", "YYZZ", "XXXY", "XXXX"])

    def test_its_orbitals_promise_their_surfaces_alpha_then_beta(self):
        o = reader.read_jobs("h2.fchk", basis_fchk(beta=True))["orbitals"]
        self.assertEqual([(x["spin"], x["surface"]["ask"]) for x in o], [("alpha", "expansion:0"), ("alpha", "expansion:1"), ("beta", "expansion:2"), ("beta", "expansion:3")])
        e = reader.fchk_expansion(basis_fchk(beta=True))
        self.assertEqual(e["orbitals"][2], [float(-i) for i in range(11)])
        # (a restricted calculation's, open shell or not: one set, of both spins)
        self.assertEqual(len(reader.fchk_expansion(basis_fchk())["orbitals"]), 2)

    def test_none_given_where_its_basis_is_not_all_there_or_past_g(self):
        for changed in ({"Shell types": ("I", [0, -1, 5, 0])}, {"Alpha MO coefficients": ("R", [0.0] * 21)}, {"Primitive exponents": None},
                        {"P(S=P) Contraction coefficients": None}, {"Shell to atom map": ("I", [1, 1, 1, 3])}):
            text = basis_fchk(**changed)
            with self.assertRaises(ValueError):
                reader.fchk_expansion(text)
            self.assertTrue(all("surface" not in x for x in reader.read_jobs("h2.fchk", text)["orbitals"]))
        with self.assertRaisesRegex(ValueError, "l = 5"):
            reader.fchk_expansion(basis_fchk(**{"Shell types": ("I", [0, -1, 5, 0])}))


class TheWorker(unittest.TestCase):
    def test_read_answers_meno_s_form(self):
        r = worker.answer(json.dumps({"id": 6, "op": "read", "name": "water.log", "text": opt_freq()}))
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["result"]["schema"], reader.SCHEMA)
        self.assertEqual(r["result"]["atoms"], ["O", "H", "H"])

    def test_read_refuses_in_words_what_holds_no_molecule(self):
        r = worker.answer(json.dumps({"id": 7, "op": "read", "name": "notes.log", "text": " Entering Gaussian System, Link 0=g16\n"}))
        self.assertEqual(r, {"id": 7, "ok": False, "error": "It could not be read as Gaussian's output: it holds no molecule."})

    def test_ask_gives_a_formatted_checkpoint_s_wavefunction_and_nothing_else(self):
        r = worker.answer(json.dumps({"id": 8, "op": "ask", "kind": "gaussian-fchk", "key": "expansion", "name": "h2.fchk", "text": basis_fchk()}))
        self.assertTrue(r["ok"], r)
        self.assertEqual(len(r["result"]["expansion"]["functions"]), 11)
        r = worker.answer(json.dumps({"id": 9, "op": "ask", "kind": "gaussian-fchk", "key": "orbital:0", "name": "h2.fchk", "text": basis_fchk()}))
        self.assertEqual(r, {"id": 9, "ok": False, "error": "Its wavefunction could not be given: it made no promise 'orbital:0'."})
        r = worker.answer(json.dumps({"id": 10, "op": "ask", "kind": "gaussian", "key": "expansion", "name": "water.log", "text": opt_freq()}))
        self.assertEqual(r["error"], "Its wavefunction could not be given: only a formatted checkpoint holds its wavefunction.")

    def test_collect_judges_a_run_as_read_does(self):
        cut = opt_freq()[: opt_freq().index(" Harmonic frequencies")]
        self.assertFalse(worker.ended_well(cut))
        self.assertTrue(worker.ended_well(opt_freq()))


if __name__ == "__main__":
    unittest.main()
