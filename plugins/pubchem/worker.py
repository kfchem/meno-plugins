"""The PubChem plugin's worker: a structure found by its name at PubChem, by
its PUG REST service (Meno's role *A structure found by its name, in a
database outside Meno*: Meno's docs/PLUGINS.md, *Roles*).

One JSON object per line on stdin, one per line back on stdout:

    {"id": 4, "op": "find_name", "name": "aspirin", "most": 8}
    {"id": 4, "ok": true, "result": {"found": [{"id": "CID 2244", "page": "https://pubchem.ncbi.nlm.nih.gov/compound/2244", "title": "...", "smiles": "...", "formula": "...", "charge": 0, "centres": {"defined": 0, "undefined": 0}, "doubleBonds": {"defined": 0, "undefined": 0}, "parts": 1}], "more": 0}}
    {"id": 4, "ok": true, "result": {"none": true}}
    {"id": 4, "ok": true, "result": {"error": "PubChem could not be reached"}}
    {"id": 4, "ok": false, "error": "..."}

    {"event": "ready", "version": "0.1.0"}

What goes out is the name looked up, and nothing else: in the body of one
request (never in its address, which is what logs keep), to
pubchem.ncbi.nlm.nih.gov alone - the one host of the manifest - through
Meno's network gate (the proxy Meno points it at), once the person has
allowed it. No header says anything of the person or the computer: the
plugin's own name and version, what it takes back, and how the body is
written.

The plugin does no chemistry: it maps the fields PubChem gives to Meno's,
and refuses an answer that does not fit them - in words, never guessed at.
It reads no SMILES, counts nothing and chooses nothing: the compounds come
in PubChem's own order, and Meno checks each before it is drawn.

PubChem's pages each limit is read from (cited by their titles here, and in
docs/pubchem.md with the sections):
- PUG REST, https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
- PUG REST Tutorial, https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest-tutorial
- Programmatic Access, https://pubchem.ncbi.nlm.nih.gov/docs/programmatic-access
- Dynamic Request Throttling, https://pubchem.ncbi.nlm.nih.gov/docs/dynamic-request-throttling
- the schema PUG REST links, https://pubchem.ncbi.nlm.nih.gov/pug_rest/pug_rest.xsd
"""

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

VERSION = "0.1.0"

#: The service's one place, as the plugin's manifest names it. A test alone gives another (`--base`), to a stand-in
#: server on its own computer; what ships is this.
HOST = "pubchem.ncbi.nlm.nih.gov"
BASE_URL = f"https://{HOST}"

#: Name to ids: the name in the POST body, as `name=...`, the argument named as the path's (PUG REST Tutorial, *How To
#: Use HTTP POST* and *Special Characters in the URL*; PUG REST, *Request (POST) Body*). The whole name must match by
#: default (PUG REST Tutorial, *By Name*): `name_type` is not sent.
NAME_PATH = "/rest/pug/compound/name/cids/JSON"
#: Ids to records: the properties, each as PUG REST, *Compound Property Tables*, says it - `SMILES` with configurations
#: and isotopes, never `ConnectivitySMILES` (which has neither).
PROPERTIES = ("Title", "MolecularFormula", "SMILES", "Charge", "AtomStereoCount", "DefinedAtomStereoCount", "UndefinedAtomStereoCount",
              "BondStereoCount", "DefinedBondStereoCount", "UndefinedBondStereoCount", "CovalentUnitCount")

#: At most 5 requests a second (Programmatic Access, its usage policy; also at the head of PUG REST): one at a time,
#: and at least this long, in seconds, from one request's start to the next's.
SPACING = 0.2
#: One request is given at most 30 s - "the default timeout on PubChem servers" (Programmatic Access; PUG REST
#: Tutorial, *Error Handling*).
TIMEOUT = 30.0
#: A whole lookup is given at most this long, in seconds: inside Meno's own wait for the answer (90 s), so that the
#: plugin's words, not Meno's, say that PubChem took too long.
DEADLINE = 80.0
#: Where the `X-Throttling-Control` header says Red or Black of any of its statuses (Dynamic Request Throttling), the
#: next request waits this long, in seconds. The page asks that requests be moderated by that header and gives no
#: number: this one is the plugin's own choice, with no measured basis.
THROTTLED_WAIT = 1.0
#: A 503 - "Too many requests or server is busy, retry later" (PUG REST, *Status Codes*) - is asked once more, after
#: this long, in seconds: the plugin's own choice, as the page gives no number.
BUSY_WAIT = 2.0
#: The most of an answer read, in bytes: a larger one is refused, not read.
MOST_READ = 1 << 20

#: Meno's own bounds on what it takes back (docs/PLUGINS.md, *Roles*): an answer past one is refused here, in words.
NAME_MOST = 300
TITLE_MOST = 300
SMILES_MOST = 4000
FORMULA_MOST = 300
MOST_ASKED = 100

#: The headers sent, all of them beside those HTTP itself needs (Host, Content-Length, Connection,
#: Accept-Encoding: identity): none says who asks, or from where.
HEADERS = {"User-Agent": f"meno-plugin-pubchem/{VERSION}", "Accept": "application/json"}

#: What the plugin says, in words.
NOT_REACHED = "PubChem could not be reached"
NOT_READ_NAME = "PubChem could not read this name"
BUSY = "PubChem is busy; try again in a moment"
LATE = "PubChem did not answer in time"


def not_read(status):
    return f"PubChem answered what this plugin does not read (HTTP {status})"


class Said(Exception):
    """A lookup that ends in words: what the person is told."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    # (an answer elsewhere is no answer of the one host: a redirect is not followed, but taken as it comes)
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Service:
    """PubChem, as this worker reaches it: one request at a time, spaced, throttled where it says so, each within
    its time and its size."""

    def __init__(self, base=BASE_URL, clock=time.monotonic, sleep=time.sleep):
        self.base = base.rstrip("/")
        self.clock = clock
        self.sleep = sleep
        self.next_at = 0.0
        # (the default handlers: the proxy Meno points the worker at is honoured, from its environment)
        self.opener = urllib.request.build_opener(_NoRedirect)
        self.opener.addheaders = []

    def _wait_turn(self, until):
        now = self.clock()
        if now < self.next_at:
            if self.next_at > until:
                raise Said(LATE)
            self.sleep(self.next_at - now)

    def _once(self, method, path, body, until):
        """One request: its status, its headers and its body (at most MOST_READ + 1 bytes), or words."""
        self._wait_turn(until)
        started = self.clock()
        self.next_at = started + SPACING
        left = min(TIMEOUT, until - started)
        if left <= 0:
            raise Said(LATE)
        headers = dict(HEADERS)
        if body is not None:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        req = urllib.request.Request(self.base + path, data=body, headers=headers, method=method)
        try:
            try:
                answer = self.opener.open(req, timeout=left)
            except urllib.error.HTTPError as e:
                answer = e
            with answer:
                status = answer.getcode()
                got = answer.headers
                chunks, size = [], 0
                while size <= MOST_READ:
                    if self.clock() - started > left:
                        raise Said(LATE)
                    chunk = answer.read(min(65536, MOST_READ + 1 - size))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    size += len(chunk)
        except Said:
            raise
        except TimeoutError:
            raise Said(LATE) from None
        except urllib.error.URLError as e:
            raise Said(LATE if isinstance(e.reason, TimeoutError) else NOT_REACHED) from None
        except OSError:
            raise Said(NOT_REACHED) from None
        throttle = got.get("X-Throttling-Control", "") if got else ""
        if "Red" in throttle or "Black" in throttle:
            self.next_at = max(self.next_at, self.clock() + THROTTLED_WAIT)
        return status, got, b"".join(chunks)

    def ask(self, method, path, body, until):
        """A request's JSON answer (a 404's too, for its fault's code), its status said in words else."""
        status, got, raw = self._once(method, path, body, until)
        if status == 503:
            if self.clock() + BUSY_WAIT > until:
                raise Said(BUSY)
            self.sleep(BUSY_WAIT)
            status, got, raw = self._once(method, path, body, until)
            if status == 503:
                raise Said(BUSY)
        if status == 504:
            raise Said(LATE)
        if status == 400:
            raise Said(NOT_READ_NAME)
        if status not in (200, 404):
            raise Said(not_read(status))
        # (an answer over the size read is refused; one not said to be JSON - an HTML page, say - is never read as JSON)
        kind = (got.get("Content-Type", "") if got else "").split(";")[0].strip().lower()
        if len(raw) > MOST_READ or kind != "application/json":
            raise Said(not_read(status))
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise Said(not_read(status)) from None
        return status, data


def _record(v):
    return isinstance(v, dict)


def _whole(v):
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _words(v, most):
    return isinstance(v, str) and v.strip() != "" and len(v) <= most


def ids_of(status, data):
    """The ids PubChem holds under the name, in its order - or None where it has none."""
    if status == 404:
        fault = data.get("Fault") if _record(data) else None
        if _record(fault) and fault.get("Code") == "PUGREST.NotFound":
            return None
        raise Said(not_read(404))
    found = data.get("IdentifierList") if _record(data) else None
    cids = found.get("CID") if _record(found) else None
    if not isinstance(cids, list) or not all(_whole(c) and c > 0 for c in cids):
        raise Said(not_read(200))
    seen = []
    for c in cids:
        if c not in seen:
            seen.append(c)
    return seen or None


def named_of(p):
    """One record, as Meno takes it - or None where anything of it does not fit."""
    if not _record(p):
        return None
    counts = ("Charge", "AtomStereoCount", "DefinedAtomStereoCount", "UndefinedAtomStereoCount", "BondStereoCount",
              "DefinedBondStereoCount", "UndefinedBondStereoCount", "CovalentUnitCount")
    charge = p.get("Charge")
    if not (isinstance(charge, int) and not isinstance(charge, bool)):
        return None
    if not all(_whole(p.get(k)) for k in counts[1:]) or p["CovalentUnitCount"] < 1:
        return None
    if not (_words(p.get("Title"), TITLE_MOST) and _words(p.get("SMILES"), SMILES_MOST) and _words(p.get("MolecularFormula"), FORMULA_MOST)):
        return None
    # (the counts as the record gives them: defined and undefined, each a part of its total - else they do not fit)
    if p["DefinedAtomStereoCount"] + p["UndefinedAtomStereoCount"] != p["AtomStereoCount"]:
        return None
    if p["DefinedBondStereoCount"] + p["UndefinedBondStereoCount"] != p["BondStereoCount"]:
        return None
    cid = p.get("CID")
    return {
        "id": f"CID {cid}",
        "page": f"https://{HOST}/compound/{cid}",
        "title": p["Title"],
        "smiles": p["SMILES"],
        "formula": p["MolecularFormula"],
        "charge": charge,
        "centres": {"defined": p["DefinedAtomStereoCount"], "undefined": p["UndefinedAtomStereoCount"]},
        "doubleBonds": {"defined": p["DefinedBondStereoCount"], "undefined": p["UndefinedBondStereoCount"]},
        "parts": p["CovalentUnitCount"],
    }


def records_of(data, cids):
    """The records of `cids`, in that order, each whole - else words."""
    table = data.get("PropertyTable") if _record(data) else None
    rows = table.get("Properties") if _record(table) else None
    if not isinstance(rows, list):
        raise Said(not_read(200))
    by_cid = {}
    for p in rows:
        cid = p.get("CID") if _record(p) else None
        if not (_whole(cid) and cid in cids) or cid in by_cid:
            raise Said(not_read(200))
        by_cid[cid] = p
    if set(by_cid) != set(cids):
        raise Said(not_read(200))
    found = [named_of(by_cid[c]) for c in cids]
    if any(f is None for f in found):
        raise Said(not_read(200))
    return found


def find_name(req, service):
    """The compounds PubChem holds under exactly this name, the first `most` in its order, and how many more."""
    name, most = req.get("name"), req.get("most")
    if not isinstance(name, str) or not name.strip() or len(name) > NAME_MOST:
        raise ValueError(f"A name of 1 to {NAME_MOST} characters is looked up")
    if not (_whole(most) and 1 <= most <= MOST_ASKED):
        raise ValueError(f"Between 1 and {MOST_ASKED} compounds are asked for")
    until = service.clock() + DEADLINE
    try:
        status, data = service.ask("POST", NAME_PATH, urllib.parse.urlencode({"name": name}).encode("ascii"), until)
        cids = ids_of(status, data)
        if cids is None:
            return {"none": True}
        first = cids[:most]
        path = f"/rest/pug/compound/cid/{','.join(str(c) for c in first)}/property/{','.join(PROPERTIES)}/JSON"
        status, data = service.ask("GET", path, None, until)
        if status != 200:
            raise Said(not_read(status))
        return {"found": records_of(data, first), "more": len(cids) - len(first)}
    except Said as e:
        return {"error": str(e)}


def answer(line, service):
    """One line asked, its answer."""
    try:
        req = json.loads(line)
    except ValueError:
        return {"ok": False, "error": "Not a request"}
    rid = req.get("id") if isinstance(req, dict) else None
    try:
        if not isinstance(req, dict) or req.get("op") != "find_name":
            raise ValueError(f"Not something this plugin does: {req.get('op') if isinstance(req, dict) else req}")
        return {"id": rid, "ok": True, "result": find_name(req, service)}
    except Exception as e:  # (said in words: what fails here is the request's, never the worker's end)
        return {"id": rid, "ok": False, "error": str(e) or "It failed"}


def main(argv):
    base = BASE_URL
    # (a test's stand-in server, on its own computer: never given by Meno, which passes no arguments)
    if len(argv) == 2 and argv[0] == "--base":
        base = argv[1]
    service = Service(base)
    print(json.dumps({"event": "ready", "version": VERSION}), flush=True)
    for line in sys.stdin:
        if line.strip():
            print(json.dumps(answer(line, service), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
