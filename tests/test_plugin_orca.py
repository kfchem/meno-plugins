"""Tests for the ORCA plugin's worker (plugins/orca/worker.py).

The worker needs Python alone - ORCA is run by Meno, not by it - so these
run anywhere; CI runs them:

    python3 -m unittest tests/test_plugin_orca.py

What ORCA is given is checked against ORCA's own manual (ORCA 6.1 Manual,
faccts.de: "General Structure of the Input File", "Parallel and
Multi-Process Modules", "Implicit Solvation Models"); no output of a real
run is kept.
"""

import importlib.util
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = ROOT / "plugins/orca/worker.py"
MANIFEST = ROOT / "plugins/orca/manifest.json"



def load_worker(path, name):
    """A plugin's worker, loaded as its process runs it: its folder on the path, for the reader beside it - and that
    reader let go of after, so that another plugin's worker loaded in the same tests finds its own."""
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


worker = load_worker(WORKER, "plugin_orca")
# (MPI's launcher taken to be there, whatever this computer has: each test that says otherwise says so)
worker.which = lambda name, path=None: f"/usr/local/bin/{name}"

WATER = {
    "name": "water",
    "atoms": [
        {"el": "O", "x": 0.0, "y": 0.0, "z": 0.11926},
        {"el": "H", "x": 0.0, "y": 0.76324, "z": -0.47704},
        {"el": "H", "x": 0.0, "y": -0.76324, "z": -0.47704},
    ],
    "bonds": [{"a1": 0, "a2": 1, "order": 1}, {"a1": 0, "a2": 2, "order": 1}],
    "charge": 0,
    "multiplicity": 1,
}


def ask(op, **m):
    said = worker.answer(json.dumps({"id": 1, "op": op, **m}))
    if not said["ok"]:
        raise AssertionError(said["error"])
    return said["result"]


def defaults(kind):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    step = next(s for s in manifest["steps"] if kind in s["kinds"])
    return {o["id"]: o["default"] for o in step["options"]}


class Prepare(unittest.TestCase):
    def test_a_job_for_each_entry_its_input_as_the_manual_lays_it_out(self):
        jobs = ask("prepare", step="optimise", entries=[WATER, {**WATER, "charge": -1, "multiplicity": 2}], options=defaults("optimise"), cores=4)["jobs"]
        self.assertEqual(len(jobs), 2)
        job = jobs[0]
        self.assertEqual((job["entries"], job["program"], job["args"], job["reads"]), ([0], "orca", ["input.inp"], ["input.wfx"]))
        # (then orca_2aim, beside ORCA, writing its orbitals as a WFX file, as the manual's "orca_2aim BaseName")
        self.assertEqual(job["then"], [{"program": "orca_2aim", "args": ["input"]}])
        self.assertEqual(job["files"][0]["name"], "input.inp")
        lines = job["files"][0]["text"].splitlines()
        # a simple input line; the processes; the coordinates between * xyz charge multiplicity and *
        self.assertEqual(lines[0], "! B3LYP def2-SVP D3BJ OPT")
        self.assertEqual(lines[1], "%pal nprocs 4 end")
        self.assertEqual(lines[2], "* xyz 0 1")
        self.assertEqual(lines[3].split(), ["O", "0.0000000000", "0.0000000000", "0.1192600000"])
        self.assertEqual(lines[-1], "*")
        self.assertEqual(jobs[1]["files"][0]["text"].splitlines()[2], "* xyz -1 2")

    def test_an_optimisation_holds_the_atoms_fixed_meno_asks_it_to_from_0_as_orca_numbers_them(self):
        fixed = {**WATER, "restraints": [{"kind": "fixed", "atoms": [2, 0]}, {"kind": "fixed", "atoms": [0]}]}
        lines = ask("prepare", step="optimise", entries=[fixed], options={**defaults("optimise"), "hold": "fixed"})["jobs"][0]["files"][0]["text"].splitlines()
        self.assertIn("%geom Constraints {C 0 C} {C 2 C} end end", lines)
        # (a saddle point's: one %geom block, its first Hessian and the constraints together)
        saddle = ask("prepare", step="saddle-point", entries=[fixed], options={**defaults("saddle-point"), "method": "B3LYP", "hold": "fixed"})["jobs"][0]["files"][0]["text"].splitlines()
        self.assertEqual([l for l in saddle if l.startswith("%geom")], ["%geom Calc_Hess true Constraints {C 0 C} {C 2 C} end end"])
        # (not asked to - Meno's default - nothing fixed, or a single point: no constraint)
        for step, options, entry in [("optimise", defaults("optimise"), fixed), ("optimise", {**defaults("optimise"), "hold": "fixed"}, WATER), ("energy", {**defaults("energy"), "hold": "fixed"}, fixed)]:
            text = ask("prepare", step=step, entries=[entry], options=options)["jobs"][0]["files"][0]["text"]
            self.assertNotIn("Constraints", text)

    def test_an_optimisation_asked_to_hold_a_torsion_refuses_that_entry_alone_naming_it(self):
        both = {**WATER, "restraints": [{"kind": "fixed", "atoms": [0]}, {"kind": "dihedral", "atoms": [0, 1, 2, 0], "value": 180, "strength": 1}]}
        out = ask("prepare", step="optimise", entries=[both, WATER], options={**defaults("optimise"), "hold": "fixed"})
        self.assertEqual([j["entries"] for j in out["jobs"]], [[1]])
        self.assertEqual(out["refused"], [{"entry": 0, "why": "ORCA: its restraints hold dihedral, which it holds none of: only atoms fixed.", "kind": "refused"}])

    def test_each_kind_its_run_solvation_memory_and_one_process_unsaid(self):
        energy = ask("prepare", step="energy", entries=[WATER], options={**defaults("energy"), "solvation": "CPCM", "solvent": "tetrahydrofuran", "maxcore": 3000})["jobs"][0]
        lines = energy["files"][0]["text"].splitlines()
        self.assertEqual(lines[0], "! B3LYP def2-SVP D3BJ SP CPCM(tetrahydrofuran)")
        self.assertEqual(lines[1], "%maxcore 3000")
        # (the memory it asks of ORCA, declared for the queue to count: %maxcore for each of its processes)
        self.assertEqual(energy["memory"], 3000)
        four = ask("prepare", step="energy", entries=[WATER], options={**defaults("energy"), "maxcore": 3000}, cores=4)["jobs"][0]
        self.assertEqual(four["memory"], 12000)
        self.assertNotIn("memory", ask("prepare", step="energy", entries=[WATER], options=defaults("energy"))["jobs"][0])
        freq = ask("prepare", step="frequencies", entries=[WATER], options={**defaults("frequencies"), "dispersion": "D4", "solvation": "SMD", "solvent": "water", "keywords": "TightSCF"}, cores=1)["jobs"][0]
        lines = freq["files"][0]["text"].splitlines()
        self.assertEqual(lines[0], "! B3LYP def2-SVP D4 FREQ SMD(water) TightSCF")
        self.assertEqual(lines[1], "* xyz 0 1")

    def test_with_no_mpi_to_run_its_processes_each_job_runs_on_one_core_said_once_never_failed_at_its_start(self):
        from unittest import mock
        found = mock.patch.object(worker, "which", lambda name, path=None: f"/usr/local/bin/{name}")
        missing = mock.patch.object(worker, "which", lambda name, path=None: None)
        opts = {**defaults("energy"), "maxcore": 2000}
        # (MPI there: its processes, as before - no fewer cores asked, nothing said)
        with found:
            there = ask("prepare", step="energy", entries=[WATER, WATER], options=opts, cores=4)
        self.assertIn("%pal nprocs 4 end", there["jobs"][0]["files"][0]["text"])
        self.assertNotIn("cores", there["jobs"][0])
        self.assertNotIn("note", there)
        self.assertEqual(there["jobs"][0]["memory"], 8000)
        # (none: one core, each job - no %pal, one core asked for the queue to count, its memory for one - said once, on the step)
        with missing:
            alone = ask("prepare", step="energy", entries=[WATER, WATER], options=opts, cores=4)
        for job in alone["jobs"]:
            self.assertNotIn("%pal", job["files"][0]["text"])
            self.assertEqual(job["cores"], 1)
            self.assertEqual(job["memory"], 2000)
        self.assertEqual(alone["note"], "run on one core: no MPI on this computer")
        # (one core given - Settings set so, or a computer of one - nothing to say, nothing changed)
        with missing:
            one = ask("prepare", step="energy", entries=[WATER], options=opts, cores=1)
        self.assertNotIn("note", one)
        self.assertNotIn("cores", one["jobs"][0])
        # (said beside what else the step says: frequencies at ORCA's own xTB, computed numerically)
        with missing:
            numeric = ask("prepare", step="frequencies", entries=[WATER], options={**defaults("frequencies"), "method": "NATIVE-XTB2", "basis": ""}, cores=4)
        self.assertEqual(numeric["note"], "frequencies computed numerically; run on one core: no MPI on this computer")
        # (a path search too)
        with missing:
            path = ask("prepare", step="path-search", entries=[{**H2, "end": H2_END}], options=defaults("path-search"), cores=4)
        self.assertNotIn("%pal", path["jobs"][0]["files"][0]["text"])
        self.assertEqual(path["jobs"][0]["cores"], 1)

    @unittest.skipIf(sys.platform == "win32", "a launcher made runnable as macOS and Linux run one")
    def test_the_launcher_looked_for_on_the_path_its_jobs_are_given_not_this_workers(self):
        import os
        import shutil
        import stat
        import tempfile
        from unittest import mock
        base = tempfile.mkdtemp()
        try:
            def program(folder, name):
                os.makedirs(folder, exist_ok=True)
                path = os.path.join(folder, name)
                with open(path, "w") as f:
                    f.write("#!/bin/sh\n")
                os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
                return path
            orca = program(os.path.join(base, "orca_6"), "orca")
            own = os.path.join(base, "env")
            program(os.path.join(own, "bin"), worker.LAUNCHER)
            plain = os.path.join(base, "usr", "bin")
            os.makedirs(plain)
            with mock.patch.object(worker, "which", shutil.which), mock.patch.object(worker.sys, "prefix", own), \
                    mock.patch.dict(os.environ, {"PATH": os.pathsep.join([os.path.join(own, "bin"), plain])}):
                # (in this worker's own environment alone: no job is given it - not found)
                self.assertFalse(worker.launcher_found(programs={"orca": orca}))
                # (beside ORCA, where it was found: its jobs' PATH begins there - found)
                program(os.path.join(base, "orca_6"), worker.LAUNCHER)
                self.assertTrue(worker.launcher_found(programs={"orca": orca}))
                # (on the PATH Meno was started with: found, ORCA's folder given or not)
                os.remove(os.path.join(base, "orca_6", worker.LAUNCHER))
                program(plain, worker.LAUNCHER)
                self.assertTrue(worker.launcher_found(programs=None))
        finally:
            shutil.rmtree(base, ignore_errors=True)

    def test_the_mpi_launcher_looked_for_is_each_systems(self):
        import importlib.util
        from unittest import mock
        for platform, name in (("darwin", "mpirun"), ("linux", "mpirun"), ("win32", "mpiexec")):
            with mock.patch.object(sys, "platform", platform):
                fresh = load_worker(WORKER, f"orca_worker_{platform}")
            self.assertEqual(fresh.LAUNCHER, name)
            asked = []
            self.assertFalse(fresh.launcher_found(lambda n, path=None: asked.append(n)))
            self.assertEqual(asked, [name])

    def test_frequencies_treat_low_vibrations_as_orca_does_unless_asked_harmonic(self):
        # (ORCA's own, its default: nothing written)
        own = ask("prepare", step="frequencies", entries=[WATER], options=defaults("frequencies"), cores=1)["jobs"][0]["files"][0]["text"]
        self.assertNotIn("%freq", own)
        self.assertEqual(defaults("frequencies")["vibrations"], "none")
        # (harmonic: the rigid-rotor harmonic oscillator alone, ORCA's %freq QuasiRRHO false)
        rrho = ask("prepare", step="frequencies", entries=[WATER], options={**defaults("frequencies"), "vibrations": "rrho", "maxcore": 2000}, cores=1)["jobs"][0]["files"][0]["text"].splitlines()
        self.assertEqual(rrho[0], "! B3LYP def2-SVP D3BJ FREQ")
        self.assertEqual(rrho[1:3], ["%maxcore 2000", "%freq QuasiRRHO false end"])
        # (a frequency job's alone: an optimisation has no such option, and writes nothing of it)
        self.assertNotIn("%freq", ask("prepare", step="optimise", entries=[WATER], options={**defaults("optimise"), "vibrations": "rrho"}, cores=1)["jobs"][0]["files"][0]["text"])
        # (a treatment it does not know: refused)
        with self.assertRaises(Exception):
            ask("prepare", step="frequencies", entries=[WATER], options={**defaults("frequencies"), "vibrations": "mrrho"})

    def test_frequencies_at_orcas_own_xtb_are_asked_for_numerically_and_said_to_be(self):
        # (at ORCA's own xTB, its frequencies asked for numerically: NUMFREQ)
        native = {**defaults("frequencies"), "method": "NATIVE-XTB2", "basis": "", "dispersion": "none"}
        prepared = ask("prepare", step="frequencies", entries=[WATER], options=native)
        self.assertEqual(prepared["jobs"][0]["files"][0]["text"].splitlines()[0], "! NATIVE-XTB2 NUMFREQ")
        self.assertEqual(prepared["note"], "frequencies computed numerically")
        # (in any case, as typed)
        lower = ask("prepare", step="frequencies", entries=[WATER], options={**native, "method": "native-gfn1-xtb"})
        self.assertEqual(lower["jobs"][0]["files"][0]["text"].splitlines()[0], "! native-gfn1-xtb NUMFREQ")
        # (at any other method, or in any other kind of step, as before, and nothing said)
        for step, options, first in [
            ("frequencies", {**native, "method": "XTB2"}, "! XTB2 FREQ"),
            ("frequencies", defaults("frequencies"), "! B3LYP def2-SVP D3BJ FREQ"),
            ("optimise", {**native}, "! NATIVE-XTB2 OPT"),
        ]:
            other = ask("prepare", step=step, entries=[WATER], options=options)
            self.assertEqual(other["jobs"][0]["files"][0]["text"].splitlines()[0], first)
            self.assertNotIn("note", other)

    def test_a_lone_atom_asked_to_be_optimised_is_run_as_sp(self):
        # (ORCA, given OPT for one atom, warns and stops with no energy)
        sodium = {"name": "sodium", "atoms": [{"el": "Na", "x": 0.0, "y": 0.0, "z": 0.0}], "bonds": [], "charge": 1, "multiplicity": 1}
        lines = ask("prepare", step="optimise", entries=[sodium], options=defaults("optimise"))["jobs"][0]["files"][0]["text"].splitlines()
        self.assertEqual(lines[0], "! B3LYP def2-SVP D3BJ SP")
        self.assertEqual(lines[1], "* xyz 1 1")
        # (frequencies are asked of it as of any molecule)
        freq = ask("prepare", step="frequencies", entries=[sodium], options=defaults("frequencies"))["jobs"][0]["files"][0]["text"]
        self.assertIn(" FREQ", freq.splitlines()[0])

    def test_keeps_the_wavefunction_only_where_asked(self):
        job = ask("prepare", step="energy", entries=[WATER], options={**defaults("energy"), "wavefunction": False})["jobs"][0]
        self.assertNotIn("then", job)
        self.assertEqual(job["reads"], [])

    def test_text_typed_into_the_simple_input_line_names_no_program_and_no_file(self):
        # (the manual names them only in a block, begun by "%", in quotes: %method ProgExt "path", %MOInp "file")
        for options in ({"keywords": 'ExtOpt %method ProgExt "/bin/sh" end'}, {"keywords": 'MORead %moinp "/etc/hosts"'}, {"basis": "def2-SVP\n%method ProgExt x end"}):
            said = worker.answer(json.dumps({"id": 1, "op": "prepare", "step": "energy", "entries": [WATER], "options": {**defaults("energy"), **options}}))
            self.assertFalse(said["ok"], options)

    def test_an_isotope_is_given_its_mass_after_its_coordinates(self):
        # (heavy water: each H a deuterium, its mass as Meno gives it - the manual's "M =" after the coordinates)
        d2o = {**WATER, "atoms": [WATER["atoms"][0]] + [{**a, "isotope": 2, "mass": 2.01410177812} for a in WATER["atoms"][1:]]}
        text = ask("prepare", step="frequencies", entries=[d2o], options=defaults("frequencies"))["jobs"][0]["files"][0]["text"]
        atoms = [l for l in text.splitlines() if l[:1] in ("O", "H")]
        self.assertFalse(atoms[0].endswith("M = 2.01410177812"))
        self.assertTrue(all(l.endswith("  M = 2.01410177812") for l in atoms[1:]), atoms)
        # (no isotope: as before)
        plain = ask("prepare", step="frequencies", entries=[WATER], options=defaults("frequencies"))["jobs"][0]["files"][0]["text"]
        self.assertNotIn("M =", plain)
        # (an isotope Meno knows no mass for: refused, naming it - never given as the natural element)
        said = worker.answer(json.dumps({"id": 1, "op": "prepare", "step": "energy", "entries": [{**WATER, "atoms": [{**WATER["atoms"][0], "isotope": 99}] + WATER["atoms"][1:]}], "options": defaults("energy")}))
        self.assertFalse(said["ok"])
        self.assertIn("99O", said["error"])

    def test_a_job_running_an_xtb_method_through_the_xtb_program_asks_for_the_xtb_plugins_and_no_other_does(self):
        # (each keyword ORCA runs through the xtb program, in the method or the keywords, any case; its native ones not)
        for method, keywords in [("XTB2", ""), ("gfn2-xtb", ""), ("XTB0", ""), ("XTB1", ""), ("GFN1-XTB", ""), ("XTB", ""), ("GFN0-XTB", ""), ("B3LYP", "GFN-FF"), ("XTBFF", "")]:
            job = ask("prepare", step="optimise", entries=[WATER], options={**defaults("optimise"), "method": method, "basis": "", "dispersion": "none", "keywords": keywords})["jobs"][0]
            self.assertEqual(job.get("needs"), ["xtb"], (method, keywords))
        for method in ["B3LYP", "NATIVE-GFN2-XTB", "NATIVE-XTB2", "NATIVE-XTB", "NATIVE-SPGFN2-XTB", "XTB2PLUS"]:
            job = ask("prepare", step="optimise", entries=[WATER], options={**defaults("optimise"), "method": method})["jobs"][0]
            self.assertNotIn("needs", job, method)
        # (what it asks for, its manifest declares)
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["depends"], [{"plugin": "xtb", "versions": ">=0.1.4 <0.2.0", "optional": True, "programs": [{"program": "xtb", "env": "XTBEXE"}]}])

    def test_refuses_what_it_does_not_know(self):
        for options in ({"wavefunction": "yes"}, {"dispersion": "D9"}, {"solvation": "COSMO"}, {"solvation": "SMD", "solvent": "lava"}, {"method": ""}, {"keywords": "TightSCF # x"}, {"method": "B3LYP\n%pal nprocs 64 end"}, {"maxcore": -1}):
            said = worker.answer(json.dumps({"id": 1, "op": "prepare", "step": "energy", "entries": [WATER], "options": {**defaults("energy"), **options}}))
            self.assertFalse(said["ok"], options)
        said = worker.answer(json.dumps({"id": 1, "op": "prepare", "step": "conformers", "entries": [WATER], "options": {}}))
        self.assertFalse(said["ok"])
        said = worker.answer(json.dumps({"id": 1, "op": "prepare", "step": "energy", "entries": [{**WATER, "charge": 0.5}], "options": defaults("energy")}))
        self.assertFalse(said["ok"])

    def test_every_option_offered_is_one_it_takes(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for step in manifest["steps"]:
            options = {o["id"]: o for o in step["options"]}
            self.assertEqual(step["programs"], ["orca"])
            # (memory 0 leaves it to ORCA: unset, so a step's line never shows a bare 0)
            self.assertEqual(options["maxcore"].get("unset"), 0, step["kinds"])
            # (a path search's: its method, its run, its images, the steps refining its highest point, keywords and memory - each read by its input)
            if step["kinds"] == ["path-search"]:
                self.assertEqual(set(options), {"method", "basis", "run", "images", "maxiter", "keywords", "maxcore"})
                self.assertEqual({c["value"] for c in options["run"]["choices"]}, set(worker.PATH_RUNS))
                self.assertEqual((options["images"]["default"], options["maxiter"].get("unset")), (8, 0))
                continue
            self.assertEqual({c["value"] for c in options["solvent"]["choices"]}, worker.SOLVENTS)
            self.assertEqual({c["value"] for c in options["dispersion"]["choices"]}, set(worker.DISPERSIONS))
            self.assertEqual({c["value"] for c in options["solvation"]["choices"]}, worker.SOLVATIONS)
            self.assertEqual(step["programs"], ["orca"])
        self.assertEqual(manifest["installed"][0]["name"], "orca")


# (what ORCA prints last where it ended well)
ENDED = "ORCA output\n                             ****ORCA TERMINATED NORMALLY****\n"


# (a path search's files, in ORCA's form, made up for these tests: three images of a diatomic and the point it
#  refined between the second and the third, and a log whose path summary and frequencies say so)
def xyz(frames):
    return "".join(f"2\nCoordinates from ORCA-job x E {e:.12f}\n  H 0.0 0.0 0.0\n  H {d:.6f} 0.0 0.0\n" for d, e in frames)


PATH_FILES = {"input_MEP_trj.xyz": xyz([(0.74, -1.17), (0.90, -1.10), (1.10, -1.12)]), "input_NEB-TS_converged.xyz": xyz([(0.97, -1.09)])}
PATH_LOG = """
---------------------------------------------------------------
                      PATH SUMMARY FOR NEB-TS
---------------------------------------------------------------
Image     E(Eh)   dE(kcal/mol)  max(|Fp|)  RMS(Fp)
  0      -1.17000     0.00       0.00100   0.00050
  1      -1.10000    43.93       0.00100   0.00050 <= CI
 TS      -1.09000    50.20       0.00001   0.00000 <= TS
  2      -1.12000    31.37       0.00100   0.00050

VIBRATIONAL FREQUENCIES
-----------------------
     0:       0.00 cm**-1
     5:    -812.50 cm**-1  ***imaginary mode***
     6:    4100.25 cm**-1

NORMAL MODES

Number of iterations:
     NEB                               ... 5  55.6%
     TS optimization                   ... 4  44.4%
Number of SCF / gradient calculations:

                             ****ORCA TERMINATED NORMALLY****
"""
# (a band that did not converge, as its marker words say it: made up around them)
NOT_CONVERGED_LOG = """
NEB not converged. (made-up words after the marker)
Number of iterations:
     NEB                               ... 3  100.0%
Number of SCF / gradient calculations:
[file orca_main/run.cpp, line 1557]: ORCA finished with error return - aborting the run
"""
# (and where one image's SCF did not converge, ORCA carrying on with the band)
SCF_FAILED = "[file made_up/made_up.cpp, line 1]: Error (MADE_UP): made-up words, the SCF has not converged, more made-up words \n"
def irc_log(capped=()):
    """ORCA's log of a reaction path, in its form: each side, then the path summary - the point it started from marked."""
    side = lambda name: f"""
         *                          {name} IRC                      *
    0        -1.0950   -1.00   0.01  0.005
""" + ("""         *  MAXIMUM NUMBER OF ITERATIONS REACHED - STOPPING IRC RUN  *
""" if name.lower() in capped else """                      ***            THE IRC HAS CONVERGED          ***
""")
    return side("FORWARD") + side("BACKWARD") + """
                       IRC PATH SUMMARY
Step        E(Eh)      dE(kcal/mol)  max(|G|)   RMS(G)
   1        -1.170000   -50.20    0.000800  0.000400
   2        -1.100000    -6.27    0.010000  0.005000
   3        -1.090000     0.00    0.000016  0.000008 <= TS
   4        -1.120000   -18.83    0.010000  0.005000

                             ****ORCA TERMINATED NORMALLY****
"""


IRC_FILES = {"input_IRC_Full_trj.xyz": xyz([(0.74, -1.17), (0.90, -1.10), (0.97, -1.09), (1.10, -1.12)])}


class ReactionPath(unittest.TestCase):
    def test_a_job_for_each_point_irc_both_ways_its_steps_where_said(self):
        options = defaults("reaction-path")
        job = ask("prepare", step="reaction-path", entries=[H2], options={**options, "method": "NATIVE-XTB2", "basis": "", "dispersion": "none"})["jobs"][0]
        lines = job["files"][0]["text"].splitlines()
        self.assertEqual(lines[0], "! NATIVE-XTB2 IRC")
        self.assertFalse(any(l.startswith("%irc") for l in lines))
        self.assertEqual(job["reads"], ["input_IRC_Full_trj.xyz"])
        more = ask("prepare", step="reaction-path", entries=[H2], options={**options, "method": "XTB2", "maxiter": 256})["jobs"][0]
        self.assertIn("%irc maxiter 256 end", more["files"][0]["text"].splitlines())
        # (no level of its own unless set: the chemist sets the start's)
        self.assertEqual((options["method"], options["basis"], options["dispersion"]), ("", "", "none"))
        with self.assertRaisesRegex(AssertionError, "there is no method"):
            ask("prepare", step="reaction-path", entries=[H2], options=options)

    def test_reads_the_path_end_to_end_its_start_marked_each_side_said(self):
        options = {**defaults("reaction-path"), "method": "NATIVE-XTB2", "basis": ""}
        out = ask("collect", step="reaction-path", entries=[H2], options=options, files=IRC_FILES, log=irc_log(), ended="done")["outputs"][0]
        self.assertEqual(out["energies"], [-1.17, -1.10, -1.09, -1.12])
        self.assertEqual(out["converged"], [False, False, True, False])
        self.assertEqual(out["method"], "GFN2-xTB")
        self.assertEqual(out["results"][0]["values"], ["backward, step 2 of 2", "backward, step 1 of 2", "the point it started from", "forward, step 1 of 1"])
        self.assertNotIn("optimised", out)
        # (a side stopped at ORCA's cap: the path as far as it came, said - no converged one)
        capped = ask("collect", step="reaction-path", entries=[H2], options=options, files=IRC_FILES, log=irc_log(("forward",)), ended="done")["outputs"][0]
        self.assertEqual(capped["optimised"], False)
        self.assertIn("The forward side stopped at ORCA's iteration cap", capped["warnings"][0]["text"])
        # (ORCA said a side stopped, but not where: a side, never read as complete)
        unsaid = irc_log().replace("THE IRC HAS CONVERGED", "").replace("FORWARD IRC", "").replace("BACKWARD IRC", "") + "MAXIMUM NUMBER OF ITERATIONS REACHED - STOPPING IRC RUN\n"
        unsaid = unsaid.replace("****ORCA TERMINATED NORMALLY****", "") + "****ORCA TERMINATED NORMALLY****\n"
        one = ask("collect", step="reaction-path", entries=[H2], options=options, files=IRC_FILES, log=unsaid, ended="done")["outputs"][0]
        self.assertEqual(one["optimised"], False)
        self.assertTrue(one["warnings"][0]["text"].startswith("A side stopped at ORCA's iteration cap"))
        # (a summary that does not list the frames written, or marks no start: not read)
        bad = ask("collect", step="reaction-path", entries=[H2], options=options, files={"input_IRC_Full_trj.xyz": xyz([(0.74, -1.17)])}, log=irc_log(), ended="done")["outputs"][0]
        self.assertEqual(bad["kind"], "no-geometry")


class SaddlePoint(unittest.TestCase):
    def test_a_guess_optimised_to_a_saddle_point_its_first_hessian_computed_unless_asked_not_no_sides(self):
        options = defaults("saddle-point")
        nh3 = {"name": "nh3", "atoms": [{"el": "N", "x": 0, "y": 0, "z": 0.1}, {"el": "H", "x": 1.01, "y": 0, "z": 0}, {"el": "H", "x": -0.505, "y": 0.8747, "z": 0}, {"el": "H", "x": -0.505, "y": -0.8747, "z": 0}], "bonds": [], "charge": 0, "multiplicity": 1}
        said = ask("prepare", step="saddle-point", entries=[nh3], options={**options, "method": "NATIVE-XTB2", "basis": "", "dispersion": "none"})
        lines = said["jobs"][0]["files"][0]["text"].splitlines()
        self.assertEqual(lines[0], "! NATIVE-XTB2 OPTTS")
        self.assertIn("%geom Calc_Hess true end", lines)
        # (at ORCA's own xTB its first Hessian is a numerical one: said on the step's line; no wavefunction kept)
        self.assertEqual(said.get("note"), "its first Hessian computed numerically")
        self.assertEqual(said["jobs"][0]["reads"], [])
        # (no level of its own unless set: the chemist sets it)
        self.assertEqual((options["method"], options["basis"], options["dispersion"]), ("", "", "none"))
        with self.assertRaisesRegex(AssertionError, "there is no method"):
            ask("prepare", step="saddle-point", entries=[nh3], options=options)
        dft = ask("prepare", step="saddle-point", entries=[nh3], options={**options, "method": "B3LYP", "basis": "def2-SVP", "dispersion": "D3BJ", "hessian": "model"})
        self.assertEqual(dft["jobs"][0]["files"][0]["text"].splitlines()[0], "! B3LYP def2-SVP D3BJ OPTTS")
        self.assertFalse(any("Calc_Hess" in l for l in dft["jobs"][0]["files"][0]["text"].splitlines()))
        self.assertNotIn("note", dft)
        # (sides wired beside it: ORCA's OptTS takes none - refused, never dropped)
        with self.assertRaisesRegex(AssertionError, "takes no sides"):
            ask("prepare", step="saddle-point", entries=[{**nh3, "sides": {"reactant": nh3, "product": nh3}}], options=options)
        with self.assertRaises(AssertionError):
            ask("prepare", step="saddle-point", entries=[nh3], options={**options, "hessian": "guessed"})


H2 = {"name": "h2", "atoms": [{"el": "H", "x": 0, "y": 0, "z": 0}, {"el": "H", "x": 0.74, "y": 0, "z": 0}], "bonds": [], "charge": 0, "multiplicity": 1}
H2_END = {**H2, "atoms": [{"el": "H", "x": 0, "y": 0, "z": 0}, {"el": "H", "x": 1.10, "y": 0, "z": 0}]}


class PathSearch(unittest.TestCase):
    def test_a_job_for_each_pair_its_input_from_the_start_to_its_end_neb_ts_alone(self):
        options = defaults("path-search")
        job = ask("prepare", step="path-search", entries=[{**H2, "end": H2_END}], options=options)["jobs"][0]
        self.assertEqual((job["program"], job["args"], job["reads"]), ("orca", ["input.inp"], ["input_MEP_trj.xyz", "input_NEB-TS_converged.xyz"]))
        text = next(f["text"] for f in job["files"] if f["name"] == "input.inp")
        lines = text.splitlines()
        # (its method - ORCA's own GFN2-xTB, by default - and NEB-TS alone: no frequencies, a Frequencies step's; the end named for %neb)
        self.assertEqual(lines[0], "! NATIVE-XTB2 NEB-TS")
        self.assertIn('  neb_end_xyzfile "end.xyz"', lines)
        # (its images, ORCA's 8 unless said; no %geom block where its refining steps are left to ORCA)
        self.assertIn("  nimages 8", lines)
        self.assertFalse(any(l.startswith("%geom") for l in lines))
        # (a finer band about its highest point, more images, more steps refining it - as asked)
        zoom = ask("prepare", step="path-search", entries=[{**H2, "end": H2_END}], options={**options, "run": "ZOOM-NEB-TS", "images": 6, "maxiter": 999})["jobs"][0]
        zlines = zoom["files"][0]["text"].splitlines()
        self.assertEqual(zlines[0], "! NATIVE-XTB2 ZOOM-NEB-TS")
        self.assertIn("%geom maxiter 999 end", zlines)
        self.assertIn("  nimages 6", zlines)
        for bad in ({"run": "NEB-CI"}, {"images": 0}, {"images": 2.5}, {"maxiter": -1}):
            with self.assertRaises(AssertionError, msg=bad):
                ask("prepare", step="path-search", entries=[{**H2, "end": H2_END}], options={**options, **bad})
        end = next(f["text"] for f in job["files"] if f["name"] == "end.xyz").splitlines()
        self.assertEqual(end[0], "2")
        self.assertEqual(end[3].split()[:2], ["H", "1.1000000000"])
        self.assertNotIn("needs", job)
        # (at GFN2-xTB through the xtb program, the xTB plugin's xtb asked for; a method and its basis set as given)
        xtb = ask("prepare", step="path-search", entries=[{**H2, "end": H2_END}], options={**options, "method": "XTB2"})["jobs"][0]
        self.assertEqual((xtb["files"][0]["text"].splitlines()[0], xtb["needs"]), ("! XTB2 NEB-TS", ["xtb"]))
        kept = ask("prepare", step="path-search", entries=[{**H2, "end": H2_END}], options={**options, "method": "gfn2-xtb", "keywords": "TightSCF"})["jobs"][0]
        self.assertEqual(kept["files"][0]["text"].splitlines()[0], "! gfn2-xtb NEB-TS TightSCF")
        # (for now at a tight-binding level alone, with no dispersion correction or solvation added: its ends' level says
        #  neither, so a path given one could be at a level they were not - ends optimised with D3BJ, a path without it)
        for method, basis, keywords, why in [
            ("B3LYP", "def2-SVP", "", "tight-binding level only"),
            ("B3LYP", "def2-SVP", "D3BJ", "tight-binding level only"),
            ("NATIVE-XTB2", "", "CPCM(water)", "no dispersion correction or solvation"),
            ("NATIVE-XTB2", "", "TightSCF smd(water)", "no dispersion correction or solvation"),
            ("XTB2", "", "ALPB(water)", "no dispersion correction or solvation"),
            ("NATIVE-XTB2", "", "D4", "no dispersion correction or solvation"),
            ("XTB2 GBSA(water)", "", "", "tight-binding level only"),
        ]:
            with self.assertRaisesRegex(AssertionError, why, msg=(method, keywords)):
                ask("prepare", step="path-search", entries=[{**H2, "end": H2_END}], options={**options, "method": method, "basis": basis, "keywords": keywords})
        # (no end, or an end of other atoms: refused)
        with self.assertRaises(AssertionError):
            ask("prepare", step="path-search", entries=[H2], options=options)
        with self.assertRaises(AssertionError):
            ask("prepare", step="path-search", entries=[{**H2, "end": {**H2_END, "atoms": H2_END["atoms"][:1]}}], options=options)

    def test_reads_the_path_in_its_summarys_order_the_point_refined_marked_with_its_frequencies(self):
        out = ask("collect", step="path-search", entries=[{**H2, "end": H2_END}], options=defaults("path-search"), files=PATH_FILES, log=PATH_LOG, ended="done")["outputs"][0]
        # (the images, and the point refined where the summary lists it: after the second image)
        self.assertEqual(out["energies"], [-1.17, -1.10, -1.09, -1.12])
        self.assertEqual(out["converged"], [False, False, True, False])
        self.assertEqual([f[3] for f in out["frames"]], [0.74, 0.90, 0.97, 1.10])
        self.assertEqual(out["results"][0]["values"], ["NEB image 1 of 3", "NEB image 2 of 3", "highest point refined", "NEB image 3 of 3"])
        self.assertEqual(out["imaginary"], [-812.5])
        # (its level, as the step gave it: a tight-binding method by its name, whichever runs it, with no basis set)
        self.assertEqual((out["method"], out.get("basis")), ("GFN2-xTB", None))
        dft = ask("collect", step="path-search", entries=[{**H2, "end": H2_END}], options={"method": "B3LYP", "basis": "def2-SVP"}, files=PATH_FILES, log=PATH_LOG, ended="done")["outputs"][0]
        self.assertEqual((dft["method"], dft["basis"]), ("B3LYP", "def2-SVP"))
        # (how far each part went, as ORCA counts it)
        self.assertEqual(next(r for r in out["results"] if r["id"] == "path.iterations")["value"], "NEB 5 \u00b7 TS optimization 4")
        self.assertNotIn("warnings", out)
        # (a band that did not converge: said plainly, never in the words of ORCA's error return)
        stopped = ask("collect", step="path-search", entries=[{**H2, "end": H2_END}], options={}, files={}, log=NOT_CONVERGED_LOG, ended="failed")
        self.assertEqual(stopped, {"why": 'Its band did not converge in 3 iterations: ORCA stopped before refining a highest point ("NEB not converged")', "kind": "not-converged"})
        # (its band converged, refining its highest point not, in the steps ORCA allows - ORCA then going on to its
        #  properties, there ending by an error that says nothing of it, or ending normally: said plainly, nothing read -
        #  not even the band; no point refined, no entry)
        refining = PATH_LOG.replace("     TS optimization                   ... 4  44.4%", "     TS optimization                   ... 1001  91.2%")
        refining = ("       Made-up words: did not converge but reached the maximum number of\n       optimization cycles. More made-up words.\n" + refining)
        said = ('Its band converged in 5 iterations, but refining its highest point did not converge in 1001 steps ("did not '
                'converge but reached the maximum number of optimization cycles"): raise Steps for refining its highest point, '
                'or give ends nearer the path')
        props = refining.replace("****ORCA TERMINATED NORMALLY****", "ERROR (MADE-UP): a made-up failure\nORCA finished by error termination in PROPERTIES")
        for log, ended in ((props, "done"), (props, "failed"), (refining, "done")):
            self.assertEqual(ask("collect", step="path-search", entries=[{**H2, "end": H2_END}], options={}, files=PATH_FILES, log=log, ended=ended),
                             {"why": said, "kind": "not-converged"})
        # (an image's SCF that did not converge, ORCA carrying on: a warning, in its words - the path read all the same)
        warned = ask("collect", step="path-search", entries=[{**H2, "end": H2_END}], options=defaults("path-search"), files=PATH_FILES, log=SCF_FAILED + PATH_LOG, ended="done")["outputs"][0]
        self.assertEqual(warned["energies"], [-1.17, -1.10, -1.09, -1.12])
        self.assertEqual(warned["warnings"][0]["kind"], "other")
        self.assertTrue(warned["warnings"][0]["text"].startswith("1 of the band's SCF calculations did not converge, ORCA carrying on: [file made_up"))
        # (ended - exit 0 - where ORCA's log does not say it ended normally: not read)
        cut = ask("collect", step="path-search", entries=[{**H2, "end": H2_END}], options={}, files=PATH_FILES, log=PATH_LOG.replace("ORCA TERMINATED NORMALLY", ""), ended="done")
        self.assertIn("does not say it ended normally", cut["why"])
        # (its vibrations, its zeros aside: from which Meno counts the imaginary ones - one here)
        self.assertEqual([r["cells"] for r in next(x for x in out["results"] if x["id"] == "vibrations")["rows"]], [[-812.5], [4100.25]])
        # (no frequencies worked out: none said, never none found)
        quiet = PATH_LOG[: PATH_LOG.index("VIBRATIONAL")] + ENDED
        self.assertNotIn("imaginary", ask("collect", step="path-search", entries=[{**H2, "end": H2_END}], options={}, files=PATH_FILES, log=quiet, ended="done")["outputs"][0])
        # (no path written, or a job that did not end well: why, in ORCA's words where it said)
        self.assertEqual(ask("collect", step="path-search", entries=[H2], options={}, files={}, log=PATH_LOG, ended="done")["outputs"][0]["kind"], "no-geometry")
        self.assertIn("error", ask("collect", step="path-search", entries=[H2], options={}, files={}, log="ORCA finished by error termination in NUMCALC", ended="failed")["why"])


class Collect(unittest.TestCase):
    def test_what_orca_printed_is_its_output_for_meno_s_readers(self):
        said = ask("collect", step="optimise", entries=[WATER, {**WATER, "name": ""}], options={}, files={}, log=ENDED, ended="done")
        self.assertEqual(said, {"read": [{"kind": "orca", "log": True, "name": "water.out"}, {"kind": "orca", "log": True, "name": "orca.out"}]})
        # (with the wavefunction orca_2aim wrote, kept with the molecule)
        said = ask("collect", step="optimise", entries=[WATER], options={}, files={"input.wfx": "<Title>\n water\n</Title>"}, log=ENDED, ended="done")
        self.assertEqual(said["read"][0]["wavefunction"], {"kind": "wfx", "file": "input.wfx", "name": "water.wfx"})

    def test_says_why_a_job_failed(self):
        log = "\n ... \nORCA finished by error termination in SCF\n[a made-up line]\n  .... aborting the run\n"
        self.assertEqual(ask("collect", step="energy", entries=[WATER], options={}, files={}, log=log, ended="failed"), {"why": "ORCA finished by error termination in SCF"})
        self.assertEqual(ask("collect", step="energy", entries=[WATER], options={}, files={}, log="", ended="stopped"), {"why": "ORCA said nothing"})
        # (one brought down under way says no error: not its settings' "Startup error" line, but where it got to)
        down = "   Startup error        MadeUpSetting   ....  1.000000\n ... \n               ***  a made-up step under way  ***\n"
        self.assertEqual(ask("collect", step="energy", entries=[WATER], options={}, files={}, log=down, ended="failed"), {"why": "ORCA stopped without saying why, after: ***  a made-up step under way  ***"})
        # (its error's own message, the words after "Error (...):", as its output is read: reader `why_of`)
        self.assertEqual(ask("collect", step="energy", entries=[WATER], options={}, files={}, log=SCF_FAILED, ended="failed")["why"], SCF_FAILED.split("): ", 1)[1].strip())
        # (a line that begins ERROR - or FATAL ERROR - is its own words, before any other line that says error)
        unitary = "   Startup error        MadeUpSetting   ....  1.000000\nERROR IN A MADE-UP CHECK\n[a made-up line]\n  .... aborting the run\n"
        self.assertEqual(ask("collect", step="energy", entries=[WATER], options={}, files={}, log=unitary, ended="failed")["why"], "ERROR IN A MADE-UP CHECK")
        fatal = "   Startup error        MadeUpSetting   ....  1.000000\nFATAL ERROR in the integrals\n  .... aborting the run\n"
        self.assertEqual(ask("collect", step="energy", entries=[WATER], options={}, files={}, log=fatal, ended="failed")["why"], "FATAL ERROR in the integrals")

    def test_an_optimisation_that_stopped_under_way_is_read_for_where_it_got_to(self):
        cycle = "GEOMETRY OPTIMIZATION CYCLE   3\n ... \nORCA finished by error termination in SCF\n"
        stopped = ask("collect", step="optimise", entries=[WATER], options={}, files={}, log=cycle, ended="failed")
        self.assertEqual(stopped, {"why": "ORCA finished by error termination in SCF", "partial": [{"kind": "orca", "log": True, "name": "water.out"}]})
        # (not before it was under way, nor for a single point)
        self.assertNotIn("partial", ask("collect", step="optimise", entries=[WATER], options={}, files={}, log="ORCA finished by error termination in SCF\n", ended="failed"))
        self.assertNotIn("partial", ask("collect", step="energy", entries=[WATER], options={}, files={}, log=cycle, ended="failed"))

    def test_an_optimisation_that_took_the_most_cycles_orca_allows_is_said_not_converged_however_orca_ended_after(self):
        # (the marker words, over two lines, made up around them - then an error that says nothing of it, or a normal end)
        warned = ("GEOMETRY OPTIMIZATION CYCLE  49\n ... \nGEOMETRY OPTIMIZATION CYCLE  50\n ... \n"
                  "       Made-up words: did not converge but reached the maximum number of\n"
                  "       optimization cycles. More made-up words.\n")
        props = warned + "ERROR (MADE-UP): a made-up failure\n\nORCA finished by error termination in PROPERTIES\n  .... aborting the run\n"
        normal = warned + "FINAL SINGLE POINT ENERGY -76.4\n                             ****ORCA TERMINATED NORMALLY****\n"
        said = ('The optimisation did not converge in 50 cycles, the most it was allowed ("did not converge but '
                'reached the maximum number of optimization cycles"): run it again to go on from the geometry it reached')
        for log, ended in ((props, "done"), (props, "failed"), (normal, "done")):
            # (never read as a result: the geometry it reached read for a run again alone, as one that stopped is)
            self.assertEqual(ask("collect", step="optimise", entries=[WATER], options={}, files={}, log=log, ended=ended),
                             {"why": said, "kind": "not-converged", "partial": [{"kind": "orca", "log": True, "name": "water.out"}]})
            # (a saddle point's, so too)
            self.assertEqual(ask("collect", step="saddle-point", entries=[WATER], options={}, files={}, log=log, ended=ended),
                             {"why": said, "kind": "not-converged", "partial": [{"kind": "orca", "log": True, "name": "water.out"}]})
        # (a single point or frequencies never optimise: the words are no sign of anything there)
        self.assertEqual(ask("collect", step="energy", entries=[WATER], options={}, files={}, log=props, ended="done"), {"why": "ORCA finished by error termination in PROPERTIES"})

    def test_a_job_that_exited_0_but_ended_by_error_termination_failed_and_only_one_that_terminated_normally_is_read(self):
        # (a parallel run whose MPI is not there, its exit code 0 all the same: read as done)
        startup = "%pal nprocs 4 end\nsh: mpirun: command not found\nORCA finished by error termination in Startup\n[a made-up line]\n  .... aborting the run\n"
        # (said plainly, with what to do: Open MPI is what ORCA runs in parallel with, its manual's 2.5.2.1)
        self.assertEqual(ask("collect", step="optimise", entries=[WATER], options={}, files={}, log=startup, ended="done"), {"why": worker.NO_MPI_SAID})
        self.assertTrue(worker.NO_MPI_SAID.startswith("ORCA runs on more than one core only with Open MPI"))
        # (a start-up error of another kind: in ORCA's words, as before)
        other = startup.replace("sh: mpirun: command not found\n", "")
        self.assertEqual(ask("collect", step="optimise", entries=[WATER], options={}, files={}, log=other, ended="done"), {"why": "ORCA finished by error termination in Startup"})
        # (nor one that never says it ended - cut off, though its runner saw it end)
        self.assertEqual(ask("collect", step="optimise", entries=[WATER], options={}, files={}, log="FINAL SINGLE POINT ENERGY -76.4\n", ended="done"), {"why": "ORCA's log does not say it ended normally; it last said: FINAL SINGLE POINT ENERGY -76.4"})
        self.assertEqual(ask("collect", step="optimise", entries=[WATER], options={}, files={}, log="", ended="done"), {"why": "ORCA's log does not say it ended normally"})
        # (one that terminated normally: read by Meno's readers, as before)
        normal = "FINAL SINGLE POINT ENERGY -76.4\n                             ****ORCA TERMINATED NORMALLY****\nTOTAL RUN TIME: 0 days 0 hours 0 minutes 1 seconds\n"
        self.assertIn("read", ask("collect", step="optimise", entries=[WATER], options={}, files={}, log=normal, ended="done"))


if __name__ == "__main__":
    unittest.main()
