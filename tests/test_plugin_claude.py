"""Tests for the Claude plugin's worker (plugins/claude/worker.py).

The worker needs Anthropic's SDK: these run in the plugin's own environment,
installed from its lock (CI's job "locked" does so on every push):

    <the Claude plugin's Python> -m unittest tests/test_plugin_claude.py

No live service is asked: a fake one, in the test's own transport, answers
as the service streams an answer, and is shown each request whole - where
the key goes, and what is sent. The key here is made up.
"""

import importlib.util
import json
import os
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = ROOT / "plugins/claude/worker.py"

try:
    import anthropic
    import httpx2

    THERE = True
except ImportError:  # (CI's Python, or another plugin's)
    THERE = False

KEY = "sk-ant-test-0123456789abcdefghij"


def load():
    # (nothing written beside the worker: its folder is the plugin as packed - tests/test_layout.py)
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("claude_worker", WORKER)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker


def sse(events):
    """A streamed answer, as the service writes one: each event, its name and its data."""
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def message(blocks, stop="end_turn", model="claude-opus-5-5"):
    """The events of a message of `blocks` - text, or a tool's call - ending for `stop`."""
    events = [{"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant", "content": [], "model": model, "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 1}}}]
    for i, b in enumerate(blocks):
        if b["type"] == "text":
            events += [
                {"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": b["text"]}},
            ]
        elif b["type"] == "thinking":
            events += [
                {"type": "content_block_start", "index": i, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
                {"type": "content_block_delta", "index": i, "delta": {"type": "signature_delta", "signature": b["signature"]}},
            ]
        else:
            events += [
                {"type": "content_block_start", "index": i, "content_block": {"type": "tool_use", "id": b["id"], "name": b["name"], "input": {}}},
                {"type": "content_block_delta", "index": i, "delta": {"type": "input_json_delta", "partial_json": json.dumps(b["input"])}},
            ]
        events.append({"type": "content_block_stop", "index": i})
    events += [{"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None}, "usage": {"output_tokens": 5}}, {"type": "message_stop"}]
    return sse(events)


class Service:
    """The fake service: what it was asked, and what it answers - a streamed message, or an error."""

    def __init__(self, answer=None, status=200, error=None):
        self.asked = []
        self.answer = answer if answer is not None else message([{"type": "text", "text": "It is ethanol."}])
        self.status = status
        self.error = error

    def handle(self, request):
        self.asked.append(request)
        if self.status != 200:
            body = json.dumps({"type": "error", "error": {"type": self.error or "api_error", "message": "no"}}).encode()
            return httpx2.Response(self.status, content=body, headers={"content-type": "application/json"})
        return httpx2.Response(200, content=self.answer, headers={"content-type": "text/event-stream"})

    def make(self, worker):
        return lambda key: worker.client(key, httpx2.Client(transport=httpx2.MockTransport(self.handle)))

    def body(self, n=0):
        return json.loads(self.asked[n].content)


SAID = [
    {"by": "person", "text": "What is in the frame?", "at": 1},
    {"by": "model", "text": "", "calls": [{"tool": "list"}], "keep": {"content": [{"type": "thinking", "thinking": "", "signature": "sig-1"}, {"type": "tool_use", "id": "toolu_1", "name": "list", "input": {}}]}, "at": 2},
    {"by": "tool", "tool": "list", "ok": True, "answer": {"things": [{"id": "structure:1", "name": "<b>ethanol</b>"}]}, "at": 3},
    {"by": "model", "text": "Two things.", "calls": [{"tool": "read", "args": {"id": "structure:1"}}, {"tool": "read", "args": {"id": "x"}}], "at": 4},
    {"by": "tool", "tool": "read", "ok": False, "answer": "There is no such object", "at": 5},
    {"by": "meno", "text": "Stopped at the limit of 24 tool calls for one message.", "at": 6},
]

TOOLS = [{"name": "list", "does": "Lists what is in the frame"}, {"name": "read", "does": "Reads one thing"}]


def asked(**over):
    return {"id": 7, "op": "converse", "told": "You are helping a chemist inside Meno.", "said": SAID, "tools": TOOLS, "options": {}, "key": KEY, **over}


@unittest.skipUnless(THERE, "needs the plugin's environment (its lock)")
class Converse(unittest.TestCase):
    def setUp(self):
        self.worker = load()

    def test_what_is_said_goes_as_the_service_takes_it_and_comes_back_as_meno_reads_it(self):
        service = Service(message([{"type": "thinking", "signature": "sig-2"}, {"type": "text", "text": "Let me read it."}, {"type": "tool_use", "id": "toolu_9", "name": "read", "input": {"id": "structure:1"}}], stop="tool_use"))
        got = self.worker.answer(json.dumps(asked()), service.make(self.worker))
        self.assertTrue(got["ok"], got)
        result = got["result"]
        self.assertEqual(result["text"], "Let me read it.")
        self.assertEqual(result["calls"], [{"tool": "read", "args": {"id": "structure:1"}}])
        # (what is kept of the reply: its content as the service gave it - its thinking's signature, its call's id)
        self.assertEqual([b["type"] for b in result["keep"]["content"]], ["thinking", "text", "tool_use"])
        self.assertEqual(result["keep"]["content"][0]["signature"], "sig-2")
        request = service.asked[0]
        self.assertEqual(str(request.url), "https://api.anthropic.com/v1/messages?beta=true")
        body = service.body()
        self.assertEqual(body["system"], "You are helping a chemist inside Meno.")
        self.assertEqual((body["model"], body["max_tokens"], body["output_config"]), ("claude-opus-5-5", 16000, {"effort": "high"}))
        self.assertEqual(body["thinking"], {"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "drop_block"}})
        self.assertEqual(body["fallbacks"], "default")
        self.assertIn("thinking-binding-controls-2026-08-01", request.headers["anthropic-beta"])
        self.assertIn("server-side-fallback-2026-07-01", request.headers["anthropic-beta"])
        self.assertEqual(body["tools"], [{"name": "list", "description": "Lists what is in the frame", "input_schema": {"type": "object", "additionalProperties": True}}, {"name": "read", "description": "Reads one thing", "input_schema": {"type": "object", "additionalProperties": True}}])
        m = body["messages"]
        self.assertEqual([x["role"] for x in m], ["user", "assistant", "user", "assistant", "user"])
        self.assertEqual(m[0]["content"], [{"type": "text", "text": "What is in the frame?"}])
        # (the reply Meno kept, given back unchanged - its thinking with it)
        self.assertEqual(m[1]["content"], SAID[1]["keep"]["content"])
        # (a tool's answer as data, matched to the call it answers by that call's id)
        self.assertEqual(m[2]["content"], [{"type": "tool_result", "tool_use_id": "toolu_1", "content": json.dumps(SAID[2]["answer"], ensure_ascii=False)}])
        # (a reply Meno kept nothing of: its words and calls, the calls given ids of Meno's)
        self.assertEqual(m[3]["content"], [{"type": "text", "text": "Two things."}, {"type": "tool_use", "id": "meno_3_0", "name": "read", "input": {"id": "structure:1"}}, {"type": "tool_use", "id": "meno_3_1", "name": "read", "input": {"id": "x"}}])
        # (an answer that failed, as an error; a call the turn stopped before, answered as not run; Meno's note after them)
        self.assertEqual(m[4]["content"], [
            {"type": "tool_result", "tool_use_id": "meno_3_0", "content": "There is no such object", "is_error": True},
            {"type": "tool_result", "tool_use_id": "meno_3_1", "content": "Not run: Meno stopped the turn.", "is_error": True},
            {"type": "text", "text": "(Meno: Stopped at the limit of 24 tool calls for one message.)"},
        ])

    def test_the_key_goes_in_the_services_header_alone_and_in_nothing_the_worker_says(self):
        service = Service(status=401, error="authentication_error")
        got = self.worker.answer(json.dumps(asked()), service.make(self.worker))
        self.assertEqual(got, {"id": 7, "ok": False, "error": "The service did not take the key: check it in Settings, Conversations"})
        request = service.asked[0]
        # (as the SDK sends it: the one header, and no other)
        self.assertEqual(request.headers["authorization"], f"Bearer {KEY}")
        self.assertEqual([k for k, v in request.headers.items() if KEY in v], ["authorization"])
        self.assertNotIn(KEY, request.content.decode())
        self.assertNotIn(KEY, str(request.url))
        self.assertNotIn(KEY, json.dumps(got))
        # (nor in what the worker answers when it works)
        ok = self.worker.answer(json.dumps(asked()), Service().make(self.worker))
        self.assertNotIn(KEY, json.dumps(ok))
        # (an error that would have it, said without it)
        def failing(key):
            raise ValueError(f"something said {key} back")
        self.assertEqual(self.worker.answer(json.dumps(asked()), failing)["error"], "something said [key] back")

    def test_nothing_in_its_environment_sends_it_elsewhere_or_with_other_credentials(self):
        for name in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"):
            os.environ[name] = "https://elsewhere.example" if name.endswith("URL") else "other-token"
        service = Service()
        self.worker.answer(json.dumps(asked()), service.make(self.worker))
        request = service.asked[0]
        self.assertEqual(request.url.host, "api.anthropic.com")
        self.assertEqual(request.headers["authorization"], f"Bearer {KEY}")
        self.assertNotIn("other-token", json.dumps(dict(request.headers)))
        self.assertFalse([n for n in self.worker.FOREIGN if n in os.environ])

    def test_a_reply_of_html_or_a_pictures_markdown_comes_back_as_its_words_verbatim(self):
        words = '<img src=x onerror=alert(1)> ![a](http://example.com/y.png) <script>1</script>'
        got = self.worker.answer(json.dumps(asked()), Service(message([{"type": "text", "text": words}])).make(self.worker))
        self.assertEqual(got["result"]["text"], words)
        self.assertNotIn("calls", got["result"])

    def test_options_are_the_manifests_and_one_outside_them_refused_never_run_as_another(self):
        service = Service(message([{"type": "text", "text": "ok"}], model="claude-haiku-5-5"))
        got = self.worker.answer(json.dumps(asked(options={"model": "claude-haiku-5-5", "effort": "low", "reply-tokens": 4000})), service.make(self.worker))
        self.assertTrue(got["ok"], got)
        body = service.body()
        self.assertEqual((body["model"], body["max_tokens"], body["output_config"]), ("claude-haiku-5-5", 4000, {"effort": "low"}))
        # (Haiku has no fallback: none asked)
        self.assertNotIn("fallbacks", body)
        self.assertNotIn("server-side-fallback", service.asked[0].headers["anthropic-beta"])
        for bad in ({"model": "gpt"}, {"effort": "huge"}, {"reply-tokens": 10}, {"reply-tokens": 64001}, {"reply-tokens": True}):
            service = Service()
            got = self.worker.answer(json.dumps(asked(options=bad)), service.make(self.worker))
            self.assertFalse(got["ok"], bad)
            self.assertEqual(service.asked, [])

    def test_a_reply_declined_cut_short_or_failing_is_said_in_words(self):
        declined = self.worker.answer(json.dumps(asked()), Service(message([], stop="refusal")).make(self.worker))
        self.assertEqual(declined["error"], "Claude declined to answer this")
        cut = self.worker.answer(json.dumps(asked()), Service(message([{"type": "text", "text": "Half"}], stop="max_tokens")).make(self.worker))
        self.assertTrue(cut["result"]["text"].startswith("Half\n(Cut short"))
        for status, error, said in ((429, "rate_limit_error", "asked too much"), (500, "api_error", "failed (500)"), (404, "not_found_error", "does not know this model")):
            got = self.worker.answer(json.dumps(asked()), Service(status=status, error=error).make(self.worker))
            self.assertIn(said, got["error"])
        self.assertEqual(self.worker.answer(json.dumps(asked(key="")), Service().make(self.worker))["error"], "No key was given: add it in Settings, Conversations")
        self.assertEqual(self.worker.answer(json.dumps({"id": 1, "op": "read"}))["error"], "Not something this plugin does: read")


if __name__ == "__main__":
    unittest.main()
