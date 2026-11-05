"""Tests for the Gaussian plugin's worker
(plugins/gaussian/worker.py).

The worker needs Python alone - Gaussian is run by Meno, not by it - so
these run anywhere; CI runs them:

    python3 -m unittest tests/test_plugin_gaussian.py

What a Gaussian input holds, and how it is laid out, is checked against
Gaussian's own reference (gaussian.com, "About Gaussian 16 Input", "Link 0
Commands", "Molecule Specifications", "SCRF", "Running Gaussian"), not
against any other program; no output of a real run is kept.
"""

import importlib.util
import json
import pathlib
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = ROOT / "plugins/gaussian/worker.py"
MANIFEST = ROOT / "plugins/gaussian/manifest.json"
# (the worker's folder on the path while it is loaded, as Meno runs it there: its own reader beside it, never another
#  plugin's taken from those already imported)
sys.path.insert(0, str(WORKER.parent))
sys.modules.pop("reader", None)
try:
    spec = importlib.util.spec_from_file_location("plugin_gaussian", WORKER)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
finally:
    sys.path.remove(str(WORKER.parent))
    sys.modules.pop("reader", None)

WATER = {
    "name": "water",
    "atoms": [
        {"el": "O", "x": 0.0, "y": 0.0, "z": 0.11926},
        {"el": "H", "x": 0.0, "y": 0.76324, "z": -0.47704},
        {"el": "H", "x": 0.0, "y": -0.76324, "z": -0.47704},
    ],
    "bonds": [{"a1": 0, "a2": 1, "order": 1}, {"a1": 0, "a2": 2, "order": 1}],
}


def defaults():
    """The options' defaults, as the manifest declares them - as Meno would send them, unchanged."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {o["id"]: o["default"] for o in manifest["writes"][0]["options"]}


def write(molecule=WATER, name="water.gjf", **options):
    line = json.dumps({"id": 1, "op": "write", "kind": "gaussian-input", "name": name, "molecules": [molecule], "options": {**defaults(), **options}})
    return worker.answer(line)


def text(**kw):
    r = write(**kw)
    assert r["ok"], r
    return r["result"]["text"]


class Sections(unittest.TestCase):
    def test_the_sections_in_order_each_ended_as_the_reference_says(self):
        lines = text().split("\n")
        # Link 0 (no blank line after it), the route section, a blank line,
        # the title, a blank line, the molecule specification, a blank line
        self.assertEqual(lines[0], "%Chk=water")
        self.assertEqual(lines[1], "# B3LYP/6-31G(d) EmpiricalDispersion=GD3BJ Opt Freq")
        self.assertEqual(lines[2], "")
        self.assertEqual(lines[3], "water")
        self.assertEqual(lines[4], "")
        self.assertEqual(lines[5], "0 1")
        self.assertEqual(lines[6].split(), ["O", "0.00000000", "0.00000000", "0.11926000"])
        self.assertEqual(len(lines[6:9]), 3)
        self.assertEqual(lines[9:], ["", ""])

    def test_the_route_section_from_the_options(self):
        route = lambda **o: text(**o).split("\n")[1]  # noqa: E731
        self.assertEqual(route(job="sp", dispersion="none"), "# B3LYP/6-31G(d) SP")
        self.assertEqual(route(job="opt", method="M062X", basis="def2-TZVP", dispersion="GD3", keywords="SCRF=(Solvent=Water)"),
                         "# M062X/def2-TZVP EmpiricalDispersion=GD3 Opt SCRF=(Solvent=Water)")
        # (a line break in a keyword would end the section: it is a space)
        self.assertEqual(route(job="freq", dispersion="none", keywords="Int=UltraFine\nNoSymm"), "# B3LYP/6-31G(d) Freq Int=UltraFine NoSymm")

    def test_link_0_commands_only_where_asked(self):
        lines = text(checkpoint=False, processors=8, memory="16GB").split("\n")
        self.assertEqual(lines[:2], ["%NProcShared=8", "%Mem=16GB"])
        self.assertTrue(text(checkpoint=False).startswith("# "))
        # (the checkpoint named after the input's file, its spaces made "_")
        self.assertEqual(text(name="/runs/my run.com").split("\n")[0], "%Chk=my_run")

    def test_the_title_holds_none_of_what_the_reference_says_to_avoid(self):
        title = lambda t: text(title=t).split("\n")[3]  # noqa: E731
        self.assertEqual(title("cis_2-butene #1 @ 298 K!"), "cis 2-butene 1 298 K")
        self.assertEqual(title("back\\slash – dash"), "back slash dash")
        # (no title given: the molecule's name; none of that either: still one)
        self.assertEqual(title(""), "water")
        self.assertEqual(text(title="", molecule={**WATER, "name": "___"}, name="").split("\n")[3], "Molecule")

    def test_charge_multiplicity_and_isotopes(self):
        anion = {**WATER, "atoms": WATER["atoms"][:2]}
        self.assertEqual(text(molecule=anion, charge=-1, multiplicity=1).split("\n")[5], "-1 1")
        heavy = {**WATER, "atoms": [{**WATER["atoms"][0]}, {**WATER["atoms"][1], "isotope": 2}, WATER["atoms"][2]]}
        self.assertTrue(text(molecule=heavy).split("\n")[7].startswith("H(Iso=2) "))


class Refused(unittest.TestCase):
    def refused(self, why, **kw):
        r = write(**kw)
        self.assertFalse(r["ok"])
        self.assertIn(why, r["error"])

    def test_a_multiplicity_its_electrons_cannot_have(self):
        # water has 10 electrons: a doublet cannot be; its cation, 9, can
        self.refused("would have 10 electrons", multiplicity=2)
        self.assertTrue(write(charge=1, multiplicity=2)["ok"])
        self.refused("would have 9 electrons", charge=1, multiplicity=1)

    def test_what_is_not_a_whole_number_or_not_an_element(self):
        self.refused("whole number", charge=0.5)
        self.refused("at least 1", multiplicity=0)
        self.refused("is no element", molecule={**WATER, "atoms": [{"el": "Ph", "x": 0, "y": 0, "z": 0}]})
        self.refused("memory must be", memory="lots")
        self.refused("there are no atoms", molecule={"atoms": [], "bonds": []})

    def test_one_molecule_of_its_own_kind(self):
        r = worker.answer(json.dumps({"id": 2, "op": "write", "kind": "orca-input", "molecules": [WATER], "options": {}}))
        self.assertFalse(r["ok"])
        r = worker.answer(json.dumps({"id": 3, "op": "write", "kind": "gaussian-input", "molecules": [WATER, WATER], "options": {}}))
        self.assertIn("one molecule", r["error"])
        self.assertEqual(worker.answer('{"id": 4, "op": "run"}')["ok"], False)


def ask(op, **m):
    said = worker.answer(json.dumps({"id": 1, "op": op, **m}))
    if not said["ok"]:
        raise AssertionError(said["error"])
    return said["result"]


def step_defaults(kind):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    step = next(s for s in manifest["steps"] if kind in s["kinds"])
    return {o["id"]: o["default"] for o in step["options"]}


ENTRY = {**WATER, "charge": 0, "multiplicity": 1}

# (after "Running Gaussian": what the output ends with when a run ends in error, the message before it)
FAILED = """ A made-up reason the run stopped.
 Error termination via Lnk1e in /Applications/g16/l202.exe at Thu Oct  8 12:00:00 2026.
"""
# (a run whose link died: its output stops where it was, with no termination line)
DIED = """ Harmonic frequencies (cm**-1), IR intensities (KM/Mole)

 Berny optimization.
"""


class Steps(unittest.TestCase):
    def setUp(self):
        # (as on macOS and Linux, wherever these run: Windows' own form is Windows' tests')
        patcher = mock.patch.object(worker, "windows", return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_job_for_each_entry_run_as_g16_with_its_input_as_standard_input(self):
        jobs = ask("prepare", step="optimise", entries=[ENTRY, {**ENTRY, "name": "water 2"}], options=step_defaults("optimise"), cores=4)["jobs"]
        self.assertEqual(len(jobs), 2)
        job = jobs[0]
        # ("g16 <input-file >output-file": its output what it prints, the job's log)
        self.assertEqual((job["entries"], job["program"], job["args"], job["stdin"], job["reads"]), ([0], "g16", [], "input.gjf", ["wavefunction.wfx"]))
        text = job["files"][0]["text"]
        self.assertEqual(job["files"][0]["name"], "input.gjf")
        lines = text.splitlines()
        # Link 0: the processors from Meno's cores, no checkpoint; the route: the method, dispersion and the job
        self.assertEqual(lines[0], "%NProcShared=4")
        self.assertNotIn("%Chk", text)
        self.assertEqual(lines[1], "# B3LYP/6-31G(d) EmpiricalDispersion=GD3BJ Opt Output=WFX")
        self.assertEqual(lines[3], "water")
        self.assertEqual(lines[5], "0 1")
        # (the wavefunction's file, "read from the input stream, on a separate line": after the molecule, ended by a blank line)
        self.assertEqual(text.split("\n\n")[3:], ["wavefunction.wfx", ""])

    def test_declares_the_memory_it_asks_of_gaussian_for_the_queue_to_count(self):
        # (%Mem in its units - bytes by 1024s, or 8-byte words; bare, words - as MiB, rounded up; unset, none)
        mem = lambda memory: ask("prepare", step="energy", entries=[ENTRY], options={**step_defaults("energy"), "memory": memory})["jobs"][0].get("memory")
        self.assertEqual([mem("16GB"), mem("500MB"), mem("1500kb"), mem("2GW"), mem("1000000")], [16384, 500, 2, 16384, 8])
        self.assertIsNone(mem(""))

    def test_a_lone_atom_asked_to_be_optimised_is_run_as_sp(self):
        # (Gaussian, given Opt for one atom, stops in its optimiser)
        sodium = {"name": "sodium", "atoms": [{"el": "Na", "x": 0.0, "y": 0.0, "z": 0.0}], "bonds": [], "charge": 1, "multiplicity": 1}
        lines = ask("prepare", step="optimise", entries=[sodium], options=step_defaults("optimise"))["jobs"][0]["files"][0]["text"].splitlines()
        route = next(line for line in lines if line.startswith("#"))
        self.assertEqual(route, "# B3LYP/6-31G(d) EmpiricalDispersion=GD3BJ SP Output=WFX")

    def test_keeps_the_wavefunction_only_where_asked(self):
        job = ask("prepare", step="energy", entries=[ENTRY], options={**step_defaults("energy"), "wavefunction": False})["jobs"][0]
        self.assertNotIn("Output=WFX", job["files"][0]["text"])
        self.assertNotIn("wavefunction.wfx", job["files"][0]["text"])
        self.assertEqual(job["reads"], [])
        said = worker.answer(json.dumps({"id": 1, "op": "prepare", "step": "energy", "entries": [ENTRY], "options": {**step_defaults("energy"), "wavefunction": "yes"}}))
        self.assertFalse(said["ok"])

    def test_each_kind_its_job_and_solvation_as_scrf_says(self):
        energy = ask("prepare", step="energy", entries=[ENTRY], options={**step_defaults("energy"), "solvation": "SMD", "solvent": "TetraHydroFuran"})["jobs"][0]
        self.assertIn("SP SCRF=(SMD,Solvent=TetraHydroFuran)", energy["files"][0]["text"].splitlines()[0])
        freq = ask("prepare", step="frequencies", entries=[ENTRY], options={**step_defaults("frequencies"), "solvation": "PCM", "solvent": "Water", "keywords": "Int=UltraFine"})["jobs"][0]
        self.assertTrue(freq["files"][0]["text"].splitlines()[0].endswith("Freq SCRF=(PCM,Solvent=Water) Int=UltraFine Output=WFX"))

    def test_refuses_what_it_does_not_know(self):
        for options in ({"solvation": "COSMO"}, {"solvation": "PCM", "solvent": "Lava"}, {"dispersion": "D9"}):
            said = worker.answer(json.dumps({"id": 1, "op": "prepare", "step": "energy", "entries": [ENTRY], "options": {**step_defaults("energy"), **options}}))
            self.assertFalse(said["ok"], options)
        said = worker.answer(json.dumps({"id": 1, "op": "prepare", "step": "conformers", "entries": [ENTRY], "options": {}}))
        self.assertFalse(said["ok"])

    def test_text_typed_into_the_route_names_no_program_and_no_file(self):
        # (a workspace's steps may come from someone else, and run where it is opened: External runs a program the
        # input names - gaussian.com, "External" - "@" places a file in the input, "%" begins a Link 0 command)
        refused = [
            {"keywords": 'External="RunTink Amber"'},
            {"keywords": "external=MyScript"},
            {"keywords": "Opt Exter=MyScript"},
            {"method": 'ONIOM(B3LYP/6-31G(d):AM1:External="RunTink Amber")'},
            {"method": "ONIOM(External=RunCC:B3LYP)"},
            {"basis": "@/etc/hosts"},
            {"keywords": "Opt @basis.gbs/N"},
            {"keywords": "Opt %Subst L103 /tmp"},
            {"keywords": "Opt #P"},
            {"keywords": 'Geom=Check "x"'},
            {"keywords": "Opt ! a comment hiding what follows"},
        ]
        for options in refused:
            said = worker.answer(json.dumps({"id": 1, "op": "prepare", "step": "energy", "entries": [ENTRY], "options": {**step_defaults("energy"), **options}}))
            self.assertFalse(said["ok"], options)
            self.assertFalse(write(**options)["ok"], options)
        # (what is typed there in the ordinary way still goes in as it was typed)
        kept = {
            "method": "CAM-B3LYP",
            "basis": "6-311++G(2d',p')",
            "keywords": "Opt=(CalcFC,MaxCycles=100) Int=UltraFine SCF=XQC Pop=NBO ExtraBasis",
        }
        line = text(**kept).split("\n")[1]
        self.assertEqual(line, "# CAM-B3LYP/6-311++G(2d',p') EmpiricalDispersion=GD3BJ Opt Freq Opt=(CalcFC,MaxCycles=100) Int=UltraFine SCF=XQC Pop=NBO ExtraBasis")
        oniom = text(method="ONIOM(B3LYP/6-31G(d):UFF)", basis="").split("\n")[1]
        self.assertTrue(oniom.startswith("# ONIOM(B3LYP/6-31G(d):UFF) "), oniom)

    def test_every_option_offered_is_one_it_takes(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for step in manifest["steps"]:
            solvents = next(o for o in step["options"] if o["id"] == "solvent")["choices"]
            self.assertEqual({c["value"] for c in solvents}, worker.SOLVENTS)
            self.assertEqual(step["programs"], ["g16"])
        self.assertEqual(manifest["installed"][0]["name"], "g16")

    def test_its_output_is_read_by_meno_s_readers_and_a_failure_says_why(self):
        read = ask("collect", step="optimise", entries=[ENTRY], options={}, files={}, log=" Normal termination of Gaussian 16", ended="done")
        self.assertEqual(read, {"read": [{"kind": "gaussian", "log": True, "name": "water.log"}]})
        # (with the wavefunction it wrote, kept with the molecule)
        read = ask("collect", step="optimise", entries=[ENTRY], options={}, files={"wavefunction.wfx": "<Title>"}, log=" Normal termination of Gaussian 16", ended="done")
        self.assertEqual(read["read"][0]["wavefunction"], {"kind": "wfx", "file": "wavefunction.wfx", "name": "water.wfx"})
        why = ask("collect", step="optimise", entries=[ENTRY], options={}, files={}, log=FAILED, ended="failed")
        self.assertEqual(why, {"why": "A made-up reason the run stopped."})
        # (an optimisation that stopped once under way: its output read too, for the geometry it reached - Runs again)
        stopped = ask("collect", step="optimise", entries=[ENTRY], options={}, files={}, log=" Berny optimization.\n" + FAILED, ended="failed")
        self.assertEqual(stopped, {"why": "A made-up reason the run stopped.", "partial": [{"kind": "gaussian", "log": True, "name": "water.log"}]})
        # (not for a single point, nor where it wrote nothing)
        self.assertEqual(ask("collect", step="energy", entries=[ENTRY], options={}, files={}, log=FAILED, ended="failed"), {"why": "A made-up reason the run stopped."})
        # (a link that dies says nothing of why: what it printed last is said)
        died = ask("collect", step="frequencies", entries=[ENTRY], options={}, files={}, log=DIED, ended="failed")
        self.assertEqual(died, {"why": "Gaussian stopped without saying why, after: Berny optimization."})
        self.assertEqual(ask("collect", step="energy", entries=[ENTRY], options={}, files={}, log="", ended="failed"), {"why": "Gaussian said nothing"})

    def test_a_job_its_runner_saw_end_well_but_whose_output_does_not_say_normal_termination_failed(self):
        # (an error its exit code did not tell: failed, in Gaussian's words)
        why = ask("collect", step="optimise", entries=[ENTRY], options={}, files={}, log=FAILED, ended="done")
        self.assertEqual(why, {"why": "A made-up reason the run stopped."})
        # (cut off, saying nothing of how it ended; a normal termination then an error one)
        self.assertEqual(ask("collect", step="frequencies", entries=[ENTRY], options={}, files={}, log=DIED, ended="done"), {"why": "Gaussian stopped without saying why, after: Berny optimization."})
        both = " Normal termination of Gaussian 16\n" + FAILED
        self.assertIn("why", ask("collect", step="optimise", entries=[ENTRY], options={}, files={}, log=both, ended="done"))


class OnWindows(unittest.TestCase):
    """Gaussian 16W given its input and its output file by name, its output read back from that file."""

    def setUp(self):
        patcher = mock.patch.object(worker, "windows", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_job_names_its_input_and_output_and_reads_the_output_back(self):
        [job] = ask("prepare", step="optimise", entries=[ENTRY], options=step_defaults("optimise"), cores=2)["jobs"]
        # (`g16 input.gjf input.log`: no standard input; its output a file of the job's, read back with its wavefunction)
        self.assertEqual((job["program"], job["args"], job["reads"]), ("g16", ["input.gjf", "input.log"], ["wavefunction.wfx", "input.log"]))
        self.assertNotIn("stdin", job)
        self.assertEqual(job["files"][0]["name"], "input.gjf")
        # (the same input as elsewhere: only how it is run differs)
        with mock.patch.object(worker, "windows", return_value=False):
            [elsewhere] = ask("prepare", step="optimise", entries=[ENTRY], options=step_defaults("optimise"), cores=2)["jobs"]
        self.assertEqual(job["files"], elsewhere["files"])
        [energy] = ask("prepare", step="energy", entries=[ENTRY], options={**step_defaults("energy"), "wavefunction": False}, cores=2)["jobs"]
        self.assertEqual(energy["reads"], ["input.log"])

    def test_its_output_is_the_file_it_wrote_never_the_log(self):
        done = " Normal termination of Gaussian 16"
        read = ask("collect", step="optimise", entries=[ENTRY], options={}, files={"input.log": done}, log="", ended="done")
        self.assertEqual(read, {"read": [{"kind": "gaussian", "file": "input.log", "name": "water.log"}]})
        read = ask("collect", step="optimise", entries=[ENTRY], options={}, files={"input.log": done, "wavefunction.wfx": "<Title>"}, log="", ended="done")
        self.assertEqual(read["read"][0]["wavefunction"]["file"], "wavefunction.wfx")
        # (what the log says is not Gaussian's output there: a failure is the file's to say)
        self.assertEqual(ask("collect", step="optimise", entries=[ENTRY], options={}, files={"input.log": FAILED}, log=done, ended="done"), {"why": "A made-up reason the run stopped."})
        stopped = ask("collect", step="optimise", entries=[ENTRY], options={}, files={"input.log": " Berny optimization.\n" + FAILED}, log="", ended="failed")
        self.assertEqual(stopped["partial"], [{"kind": "gaussian", "file": "input.log", "name": "water.log"}])
        # (no file written: nothing said)
        self.assertEqual(ask("collect", step="energy", entries=[ENTRY], options={}, files={}, log="", ended="failed"), {"why": "Gaussian said nothing"})

    def test_the_manifest_names_its_program_on_windows(self):
        installed = json.loads(MANIFEST.read_text(encoding="utf-8"))["installed"][0]
        self.assertEqual(installed["files"]["windows"], "g16.exe")
        self.assertEqual(installed["env"]["GAUSS_EXEDIR"], "{folder}")


class SystemForm(unittest.TestCase):
    """Which form a job takes, as the system the worker runs on says it."""

    def test_the_form_follows_the_system_the_worker_runs_on(self):
        # (windows() itself, unpatched: Windows by sys.platform's "win32" alone - macOS and Linux the standard input's form)
        for system, named in [("win32", True), ("darwin", False), ("linux", False)]:
            with mock.patch.object(worker.sys, "platform", system):
                self.assertEqual(worker.windows(), named, system)
                [job] = ask("prepare", step="energy", entries=[ENTRY], options={**step_defaults("energy"), "wavefunction": False}, cores=1)["jobs"]
                self.assertEqual((job["args"], job.get("stdin"), job["reads"]), (["input.gjf", "input.log"], None, ["input.log"]) if named else ([], "input.gjf", []), system)


if __name__ == "__main__":
    unittest.main()
