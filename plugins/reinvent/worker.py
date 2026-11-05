"""The REINVENT 4 plugin's worker: new molecules, for Meno's step
*Generate* (its manifest; docs/WORKFLOWS.md, *Kinds of step*), sampled with
REINVENT 4 (Apache-2.0; Loeffler et al., J. Cheminform. 2024, 16, 20) from
its prior trained on PubChem.

One JSON object per line on stdin, one per line back on stdout:

    {"id": 3, "op": "run", "step": "generate", "entries": [], "options": {"count": 20, "seed": 7}}
    {"id": 3, "ok": true, "result": {"outputs": [...], "said": "...", "by": {...}}}
    {"id": 3, "ok": false, "error": "..."}

It runs no code it is sent - only the operation below - and reaches no
network: REINVENT 4's prior is downloaded by Meno, with the user's leave,
into the folder given as this script's one argument.

    {"event": "ready", "version": "4.8.24"}

REINVENT 4 samples as many SMILES as asked from the prior, on the CPU with
the seed set, so that the same asking gives the same SMILES (its `-s`,
README *Running*; measured here, the same seed gave the same SMILES twice
over). The seed is a whole number from 1: REINVENT 4 sets the seed it is
given where it is not 0 (reinvent/Reinvent.py line 120 at 4.8.24+80a8d21,
the release the lock pins: `seed = args.seed or getattr(val_config,
"seed", None)`), so a seed of 0 would be a run that does not sample the
same way again. What it gives is handed on as it is
(docs/PLUGINS.md, *Generate*): each SMILES as REINVENT 4 writes it, its
stereo as written or left out, nothing made of it in 3D, nothing refused -
every one it samples, those it marks a duplicate or could not read among
them (`unique_molecules = false`: REINVENT 4's own filter left off, so
that which are the same, and which are kept, are Meno's own steps' to
say, *Duplicates* and *Filter*) - with the model's own
values beside it, named: its NLL under the prior (the negative natural
logarithm of the probability the prior gave each token of it, summed:
reinvent/models/reinvent/models/model.py, `sample`, at the release pinned)
and its SMILES_state as REINVENT 4 writes it. Meno reads each SMILES
itself, as it reads one pasted.
"""

import csv
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import tempfile

# The folder Meno keeps the plugin's files in (the manifest's `downloads`).
FILES = sys.argv[1] if len(sys.argv) > 1 else None
#: The prior, as the manifest downloads it, and what it is: REINVENT 4's priors (v2) on Zenodo, trained on
#: PubChem as of 2024-06-03 (the file's own metadata, `origina_data_source`).
PRIOR = "reinvent_pubchem.prior"
PRIOR_SAID = "reinvent_pubchem.prior (REINVENT 4 priors v2, Zenodo 10.5281/zenodo.20701824; PubChem 2024-06-03)"
#: The most SMILES one run samples. Each is a number of SMILES, never a time: what one asking gives is the same
#: anywhere, and a run too slow fails whole, never cut short.
MOST = 100
#: The seed where none is asked for.
SEED = 1

VERSION = importlib.metadata.version("reinvent")


def sampled(count, seed, folder):
    """The SMILES REINVENT 4 samples from the prior: `count` of them with
    the random seed `seed`, on the CPU, working in `folder`; each with its
    SMILES_state and NLL, [(smiles, state, nll)], in the order it wrote
    them - duplicates kept, so that how many came is how many were asked
    for."""
    toml = os.path.join(folder, "sampling.toml")
    out = os.path.join(folder, "sampled.csv")
    log = os.path.join(folder, "reinvent.log")
    with open(toml, "w", encoding="utf-8") as f:
        f.write(
            'run_type = "sampling"\n'
            'device = "cpu"\n'
            "[parameters]\n"
            f"model_file = {json.dumps(os.path.join(FILES or '', PRIOR))}\n"
            f"output_file = {json.dumps(out)}\n"
            f"num_smiles = {int(count)}\n"
            # (every one sampled, duplicates and those it could not read among them: which are kept is Meno's to say)
            "unique_molecules = false\n"
        )
    reinvent = os.path.join(os.path.dirname(sys.executable), "reinvent")
    done = subprocess.run(
        [reinvent, "-s", str(int(seed)), "-l", log, toml],
        cwd=folder, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if done.returncode != 0 or not os.path.exists(out):
        said = []
        if os.path.exists(log):
            with open(log, encoding="utf-8", errors="replace") as f:
                said = [line.strip() for line in f if line.strip()]
        raise ValueError("REINVENT 4 stopped: " + (said[-1] if said else f"exit {done.returncode}"))
    with open(out, encoding="utf-8") as f:
        return [(row["SMILES"], int(row["SMILES_state"]) if row.get("SMILES_state", "").strip().lstrip("-").isdigit() else None, float(row["NLL"])) for row in csv.DictReader(f)]


#: Each number option as the step's card names it (the manifest's labels).
LABELS = {"count": "How many", "seed": "Random seed"}


def number(options, key, default, low, high):
    """An option as a whole number, as it was set - or its default, where it was not. Refused, naming it and what it
    may be, where it is not a whole number from `low` to `high`: never run with another than the one set."""
    v = options.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != int(v) or not low <= v <= high:
        raise ValueError(f"{LABELS[key]} is {v}: it takes a whole number from {low} to {high}")
    return int(v)


#: REINVENT 4's SMILES_state, as its sampler names each (reinvent/models/model_factory/sample_batch.py `SmilesState`).
STATES = {0: "invalid", 1: "valid", 2: "duplicate"}


def states_said(states):
    """How many of each SMILES_state REINVENT 4 gave, in a few words: ": 18 valid, 2 invalid" - those it gave none of left out."""
    counts = [(STATES.get(k, f"state {k}"), sum(1 for s in states if s == k)) for k in sorted({s for s in states if s is not None})]
    return (": " + ", ".join(f"{n} {name}" for name, n in counts)) if counts else ""


def op_run(m):
    """*Generate*: `count` SMILES sampled with the seed `seed`, each an
    output as REINVENT 4 gave it - the SMILES, and its NLL and SMILES_state
    named - what made them (`by`), and the run in a line (`said`)."""
    if m.get("step") != "generate":
        raise ValueError(f"There is no step {m.get('step')!r} to run at once.")
    if not FILES or not os.path.exists(os.path.join(FILES, PRIOR)):
        raise ValueError("REINVENT 4's prior is not here: add the plugin again in Settings, Plugins")
    o = m.get("options") or {}
    count, seed = number(o, "count", 20, 1, MOST), number(o, "seed", SEED, 1, 2**31 - 1)
    with tempfile.TemporaryDirectory(prefix="meno-reinvent-") as folder:
        rows = sampled(count, seed, folder)
    # (an NLL that is no number - NaN, infinite - written as none, which JSON can hold, and counted)
    no_nll = sum(1 for _, _, nll in rows if not math.isfinite(nll))
    outputs = [
        {
            "smiles": smiles,
            "properties": {
                "nll": {"value": nll if math.isfinite(nll) else None, "unit": "none"},
                **({"smiles_state": {"value": state, "unit": "none"}} if state is not None else {}),
            },
        }
        for smiles, state, nll in rows
    ]
    return {
        "outputs": outputs,
        "said": f"{len(rows)} sampled from {PRIOR}, seed {seed}, CPU" + states_said([state for _, state, _ in rows]) + (f", {no_nll} with no NLL (not a number)" if no_nll else ""),
        "by": {"program": "REINVENT 4", "version": VERSION, "model": PRIOR_SAID, "method": f"sampled, seed {seed}, on the CPU"},
    }


OPS = {"run": op_run}


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
    print(json.dumps({"event": "ready", "version": VERSION}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if line:
            print(json.dumps(answer(line)), flush=True)


if __name__ == "__main__":
    main()
