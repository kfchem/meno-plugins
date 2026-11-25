"""The PubChem plugin's worker (plugins/pubchem/worker.py), against a stand-in for PubChem that each test starts on this
computer: what it sends for a name - method, path, body and headers, byte for byte - how it reads each answer, what it
says of each failure, and how it spaces its requests. Python alone; nothing here reaches PubChem.

Every answer here is HAND-WRITTEN in the shape PubChem's pages give (PUG REST and the schema it links), with
hand-set values - ids, titles, SMILES and counts chosen for the test, not fetched: no answer of the service is part of
this repository."""

import http.server
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = ROOT / "plugins/pubchem/worker.py"
spec = importlib.util.spec_from_file_location("pubchem_worker", WORKER)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)

JSON = "application/json"
PROPS = "Title,MolecularFormula,SMILES,Charge,AtomStereoCount,DefinedAtomStereoCount,UndefinedAtomStereoCount,BondStereoCount,DefinedBondStereoCount,UndefinedBondStereoCount,CovalentUnitCount"


def ids(*cids):
    """A hand-written name-to-ids answer (the schema's IdentifierList)."""
    return {"IdentifierList": {"CID": list(cids)}}


def record(cid, title, formula, smiles, charge=0, centres=(0, 0), bonds=(0, 0), parts=1):
    """A hand-written record, its values set by hand (the schema's PropertyTable/Properties)."""
    return {"CID": cid, "Title": title, "MolecularFormula": formula, "SMILES": smiles, "Charge": charge,
            "AtomStereoCount": sum(centres), "DefinedAtomStereoCount": centres[0], "UndefinedAtomStereoCount": centres[1],
            "BondStereoCount": sum(bonds), "DefinedBondStereoCount": bonds[0], "UndefinedBondStereoCount": bonds[1],
            "CovalentUnitCount": parts}


def table(*records):
    return {"PropertyTable": {"Properties": list(records)}}


NOT_FOUND = {"Fault": {"Code": "PUGREST.NotFound", "Message": "No CID found", "Details": ["(hand-written)"]}}
BAD = {"Fault": {"Code": "PUGREST.BadRequest", "Message": "(hand-written)"}}
#: (hand-set: aspirin's structure written by hand, with an id chosen for the test)
ASPIRIN = record(101, "aspirin (stand-in)", "C9H8O4", "CC(=O)OC1=CC=CC=C1C(=O)O")
#: (hand-set: one centre, defined, written in the SMILES with '@@')
CARVONE = record(102, "a carvone (stand-in)", "C10H14O", "CC1=CC[C@@H](CC1=O)C(=C)C", centres=(1, 0))
#: (hand-set: one centre, undefined - no '@')
OPEN = record(103, "an open centre (stand-in)", "C10H14O", "CC1=CCC(CC1=O)C(=C)C", centres=(0, 1))
#: (hand-set: two parts, the charges cancel)
SALT = record(104, "a salt (stand-in)", "C2H3NaO2", "CC(=O)[O-].[Na+]", parts=2)


class StandIn:
    """A stand-in for PubChem on 127.0.0.1: answers given in turn, each (status, content type, body, headers), and
    every request it is sent, as sent."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.seen = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _any(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                outer.seen.append({"at": time.monotonic(), "line": self.requestline, "method": self.command, "path": self.path,
                                   "headers": self.headers.as_string(), "body": body})
                status, kind, data, more = outer.answers.pop(0) if outer.answers else (500, JSON, b"{}", {})
                if callable(data):
                    data(self)
                    return
                raw = data if isinstance(data, bytes) else json.dumps(data).encode("utf-8")
                self.send_response(status)
                if kind:
                    self.send_header("Content-Type", kind)
                for k, v in (more or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(raw)

            do_GET = do_POST = _any

            def log_message(self, *a):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def answer(status, data, kind=JSON, headers=None):
    return (status, kind, data, headers or {})


class Base(unittest.TestCase):
    def setUp(self):
        # (the stand-in is on this computer: no proxy of this one's environment between)
        for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
            self._restore_env(k)
        self._restore_env("no_proxy", "*")
        self.slept = []

    def _restore_env(self, k, v=None):
        old = os.environ.get(k)
        self.addCleanup(lambda: os.environ.__setitem__(k, old) if old is not None else os.environ.pop(k, None))
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

    def stand_in(self, *answers):
        s = StandIn(*answers)
        self.addCleanup(s.close)
        return s

    def look(self, s, name="aspirin", most=8, sleep=None):
        service = worker.Service(s.base, sleep=sleep or (lambda t: (self.slept.append(t), time.sleep(t))))
        return worker.find_name({"name": name, "most": most}, service)


class Request(Base):
    def test_the_name_goes_in_the_body_alone_and_no_header_says_who_asks(self):
        s = self.stand_in(answer(200, ids(101)), answer(200, table(ASPIRIN)))
        self.look(s, name="N,N-diméthyl/x (stand-in)")
        port = s.base.rsplit(":", 1)[1]
        first, second = s.seen
        self.assertEqual(first["line"], "POST /rest/pug/compound/name/cids/JSON HTTP/1.1")
        self.assertEqual(first["body"], b"name=N%2CN-dim%C3%A9thyl%2Fx+%28stand-in%29")
        self.assertEqual(first["headers"], "Accept-Encoding: identity\nContent-Length: 43\n"
                                           f"Host: 127.0.0.1:{port}\nUser-Agent: meno-plugin-pubchem/0.1.0\nAccept: application/json\n"
                                           "Content-Type: application/x-www-form-urlencoded\nConnection: close\n\n")
        self.assertEqual(second["line"], f"GET /rest/pug/compound/cid/101/property/{PROPS}/JSON HTTP/1.1")
        self.assertEqual(second["body"], b"")
        self.assertEqual(second["headers"], "Accept-Encoding: identity\n"
                                            f"Host: 127.0.0.1:{port}\nUser-Agent: meno-plugin-pubchem/0.1.0\nAccept: application/json\nConnection: close\n\n")

    def test_a_plus_in_a_name_is_sent_as_itself_not_as_a_space(self):
        s = self.stand_in(answer(404, NOT_FOUND))
        self.look(s, name="a+b stand-in")
        self.assertEqual(s.seen[0]["body"], b"name=a%2Bb+stand-in")

    def test_smiles_is_asked_for_never_connectivity_smiles_and_a_centre_comes_back_with_it(self):
        s = self.stand_in(answer(200, ids(102)), answer(200, table(CARVONE)))
        got = self.look(s)
        self.assertIn("/property/Title,MolecularFormula,SMILES,", s.seen[1]["path"])
        self.assertNotIn("ConnectivitySMILES", s.seen[1]["path"])
        self.assertEqual(got["found"][0]["smiles"], "CC1=CC[C@@H](CC1=O)C(=C)C")
        self.assertEqual(got["found"][0]["centres"], {"defined": 1, "undefined": 0})

    def test_the_page_is_on_the_manifest_s_host_whatever_the_stand_in_is(self):
        manifest = json.loads((ROOT / "plugins/pubchem/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["roles"], [{"role": "names", "hosts": [worker.HOST]}])
        self.assertEqual(worker.BASE_URL, "https://pubchem.ncbi.nlm.nih.gov")
        s = self.stand_in(answer(200, ids(101)), answer(200, table(ASPIRIN)))
        self.assertEqual(self.look(s)["found"][0]["page"], "https://pubchem.ncbi.nlm.nih.gov/compound/101")


class Found(Base):
    def test_one_record_mapped_field_by_field(self):
        s = self.stand_in(answer(200, ids(101)), answer(200, table(ASPIRIN)))
        self.assertEqual(self.look(s), {"found": [{
            "id": "CID 101", "page": "https://pubchem.ncbi.nlm.nih.gov/compound/101", "title": "aspirin (stand-in)",
            "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "formula": "C9H8O4", "charge": 0,
            "centres": {"defined": 0, "undefined": 0}, "doubleBonds": {"defined": 0, "undefined": 0}, "parts": 1}], "more": 0})

    def test_several_in_pubchem_s_order_the_first_most_and_how_many_more(self):
        s = self.stand_in(answer(200, ids(104, 103, 102, 101, 105)), answer(200, table(CARVONE, SALT, OPEN)))
        got = self.look(s, most=3)
        self.assertIn("/cid/104,103,102/property/", s.seen[1]["path"])
        self.assertEqual([f["id"] for f in got["found"]], ["CID 104", "CID 103", "CID 102"])
        self.assertEqual(got["more"], 2)
        self.assertEqual(got["found"][0]["parts"], 2)
        self.assertEqual(got["found"][1]["centres"], {"defined": 0, "undefined": 1})
        self.assertEqual(got["found"][1]["smiles"], "CC1=CCC(CC1=O)C(=C)C")

    def test_a_charge_as_the_record_gives_it(self):
        ion = record(106, "an anion (stand-in)", "C2H3O2-", "CC(=O)[O-]", charge=-1)
        s = self.stand_in(answer(200, ids(106)), answer(200, table(ion)))
        self.assertEqual(self.look(s)["found"][0]["charge"], -1)

    def test_none_by_the_name(self):
        s = self.stand_in(answer(404, NOT_FOUND))
        self.assertEqual(self.look(s), {"none": True})
        self.assertEqual(len(s.seen), 1)


class Failures(Base):
    def said(self, *answers, **kw):
        return self.look(self.stand_in(*answers), **kw)

    def test_each_status_in_words(self):
        self.assertEqual(self.said(answer(400, BAD)), {"error": "PubChem could not read this name"})
        self.assertEqual(self.said(answer(504, {"Fault": {"Code": "PUGREST.Timeout", "Message": "x"}})), {"error": "PubChem did not answer in time"})
        self.assertEqual(self.said(answer(500, {"Fault": {"Code": "PUGREST.ServerError", "Message": "x"}})), {"error": "PubChem answered what this plugin does not read (HTTP 500)"})
        self.assertEqual(self.said(answer(302, b"", kind=None, headers={"Location": "https://elsewhere.example/"})), {"error": "PubChem answered what this plugin does not read (HTTP 302)"})

    def test_a_404_that_is_not_pubchem_s_not_found_is_not_none(self):
        self.assertEqual(self.said(answer(404, b"<html>gone</html>", kind="text/html")), {"error": "PubChem answered what this plugin does not read (HTTP 404)"})

    def test_busy_once_then_answered(self):
        s = self.stand_in(answer(503, b"<html>busy</html>", kind="text/html"), answer(200, ids(101)), answer(200, table(ASPIRIN)))
        got = self.look(s, sleep=lambda t: self.slept.append(t))
        self.assertEqual(got["found"][0]["id"], "CID 101")
        self.assertIn(worker.BUSY_WAIT, self.slept)
        self.assertEqual([r["method"] for r in s.seen], ["POST", "POST", "GET"])

    def test_busy_twice_is_said(self):
        s = self.stand_in(answer(503, b"<html>busy</html>", kind="text/html"), answer(503, b"<html>busy</html>", kind="text/html"))
        self.assertEqual(self.look(s, sleep=lambda t: None), {"error": "PubChem is busy; try again in a moment"})
        self.assertEqual(len(s.seen), 2)

    def test_an_html_page_is_never_read_as_json(self):
        # (its body is JSON-like text, said to be HTML: still not read)
        self.assertEqual(self.said(answer(200, json.dumps(ids(101)).encode(), kind="text/html")), {"error": "PubChem answered what this plugin does not read (HTTP 200)"})

    def test_an_answer_over_one_mebibyte_is_refused(self):
        big = json.dumps({"IdentifierList": {"CID": [101], "pad": "x" * (1 << 20)}}).encode()
        self.assertEqual(self.said(answer(200, big)), {"error": "PubChem answered what this plugin does not read (HTTP 200)"})

    def test_json_of_another_shape_is_refused(self):
        self.assertEqual(self.said(answer(200, {"Something": []})), {"error": "PubChem answered what this plugin does not read (HTTP 200)"})
        self.assertEqual(self.said(answer(200, ids(101)), answer(200, table())), {"error": "PubChem answered what this plugin does not read (HTTP 200)"})
        self.assertEqual(self.said(answer(200, b"{not json", kind=JSON)), {"error": "PubChem answered what this plugin does not read (HTTP 200)"})

    def test_a_record_that_does_not_fit_is_refused_never_filled_in(self):
        missing = {k: v for k, v in ASPIRIN.items() if k != "SMILES"}
        other = dict(ASPIRIN, CID=999)
        long_smiles = dict(ASPIRIN, SMILES="C" * 4001)
        counts_apart = dict(CARVONE, AtomStereoCount=2)
        for bad in (missing, other, long_smiles, counts_apart, dict(ASPIRIN, Charge="0"), dict(ASPIRIN, CovalentUnitCount=0)):
            with self.subTest(bad=sorted(set(bad.items()) ^ set(ASPIRIN.items()), key=str)[:2]):
                self.assertEqual(self.said(answer(200, ids(101)), answer(200, table(bad))), {"error": "PubChem answered what this plugin does not read (HTTP 200)"})

    def test_no_answer_in_time(self):
        def silent(handler):
            time.sleep(1.5)
        old = worker.TIMEOUT
        worker.TIMEOUT = 0.5
        self.addCleanup(setattr, worker, "TIMEOUT", old)
        self.assertEqual(self.said((200, JSON, silent, {})), {"error": "PubChem did not answer in time"})

    def test_a_refused_connection(self):
        s = self.stand_in()
        base = s.base
        s.close()
        service = worker.Service(base)
        self.assertEqual(worker.find_name({"name": "aspirin", "most": 8}, service), {"error": "PubChem could not be reached"})

    def test_what_is_asked_is_checked_before_anything_goes_out(self):
        s = self.stand_in()
        service = worker.Service(s.base)
        for req in ({"name": "", "most": 8}, {"name": "x" * 301, "most": 8}, {"name": "aspirin", "most": 0}, {"name": 3, "most": 8}):
            with self.subTest(req=str(req)[:40]):
                with self.assertRaises(ValueError):
                    worker.find_name(req, service)
        self.assertEqual(s.seen, [])


class Pace(Base):
    def test_two_requests_are_never_closer_than_a_fifth_of_a_second(self):
        s = self.stand_in(answer(200, ids(101)), answer(200, table(ASPIRIN)), answer(200, ids(101)), answer(200, table(ASPIRIN)))
        service = worker.Service(s.base)
        worker.find_name({"name": "aspirin", "most": 8}, service)
        worker.find_name({"name": "aspirin", "most": 8}, service)
        gaps = [b["at"] - a["at"] for a, b in zip(s.seen, s.seen[1:])]
        self.assertEqual(len(gaps), 3)
        self.assertTrue(all(g >= 0.19 for g in gaps), gaps)

    def test_a_red_or_black_status_delays_the_next_request(self):
        for colour in ("Red", "Black"):
            with self.subTest(colour):
                said = f"Request Count status: Green (10%), Request Time status: {colour} (80%), Service status: Green (20%)"
                s = self.stand_in(answer(200, ids(101), headers={"X-Throttling-Control": said}), answer(200, table(ASPIRIN)))
                self.look(s)
                self.assertGreaterEqual(s.seen[1]["at"] - s.seen[0]["at"], worker.THROTTLED_WAIT - 0.01)

    def test_green_and_yellow_do_not(self):
        said = "Request Count status: Yellow (60%), Request Time status: Green (0%), Service status: Green (20%)"
        s = self.stand_in(answer(200, ids(101), headers={"X-Throttling-Control": said}), answer(200, table(ASPIRIN)))
        self.look(s)
        self.assertLess(s.seen[1]["at"] - s.seen[0]["at"], worker.THROTTLED_WAIT)


class Lines(Base):
    def test_the_worker_says_it_is_ready_and_answers_a_line(self):
        s = self.stand_in(answer(200, ids(101)), answer(200, table(ASPIRIN)))
        lines = '{"id": 1, "op": "find_name", "name": "aspirin", "most": 8}\n{"id": 2, "op": "converse"}\nnot json\n'
        done = subprocess.run([sys.executable, "-I", str(WORKER), "--base", s.base], input=lines, capture_output=True, text=True, timeout=30, env=dict(os.environ))
        out = [json.loads(l) for l in done.stdout.splitlines()]
        self.assertEqual(out[0], {"event": "ready", "version": "0.1.0"})
        self.assertEqual((out[1]["id"], out[1]["ok"], out[1]["result"]["found"][0]["id"]), (1, True, "CID 101"))
        self.assertEqual(out[2], {"id": 2, "ok": False, "error": "Not something this plugin does: converse"})
        self.assertEqual(out[3], {"ok": False, "error": "Not a request"})

    def test_the_version_is_the_manifest_s(self):
        manifest = json.loads((ROOT / "plugins/pubchem/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(worker.VERSION, manifest["version"])


if __name__ == "__main__":
    unittest.main()
