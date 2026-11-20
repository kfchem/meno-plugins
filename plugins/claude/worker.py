"""The Claude plugin's worker: a conversation on Meno's page talked through
Claude, Anthropic's model, by its Messages API (Meno's role *Talking with a
model, in a conversation*: docs/CONVERSATIONS.md, *Talking through a plugin*).

One JSON object per line on stdin, one per line back on stdout:

    {"id": 3, "op": "converse", "told": "...", "said": [...], "tools": [{"name": "list", "does": "..."}], "options": {...}, "stream": true, "key": "..."}
    {"id": 3, "partial": "..."}
    {"id": 3, "ok": true, "result": {"text": "...", "calls": [{"tool": "list", "args": {}}], "keep": {...}}}
    {"id": 3, "ok": false, "error": "..."}

Asked with `stream` (a Meno that shows a reply as it comes), Claude's words are
sent as they come, each part a line of its own (`partial`: the words it says on
from the last), before the answer, which holds them all; while it thinks and says
nothing for a minute, an empty part says it is still coming. Asked without, the
reply is answered whole, as an earlier Meno waits for it.

    {"event": "ready", "version": "2.1.0"}

Meno's backend adds the person's key to each `converse` as it passes it on,
from the system's keychain: the key is in no argument, environment or file of
this worker's, and in nothing it writes - its answers and errors are made
without it. It goes to the service in the one header the SDK sends it in. It reaches api.anthropic.com alone, through Meno's network gate
(the proxy Meno points it at), under the purpose the person allowed; nothing
in its environment can send it elsewhere (`client`).

What Meno tells the model (`told`) is its system prompt; what has been said
(`said`) becomes the conversation, each thing in turn:
- the person's words, and Meno's own notes, as the person's turn;
- the model's reply, as Claude gave it where Meno kept it (`keep`: the
  reply's own content - its reasoning, its words, its tool calls with their
  ids - given back unchanged, as the service asks), else its words and calls;
- each tool's answer, as data - never as words of the person's, nor as
  instructions - matched to the call it answers.
Meno's tools are offered by their names and what each does; what each is
given is Meno's to check, as it checks every call.

What comes back: Claude's words, plain text; the tools it calls, each by name
with its arguments; and what Meno keeps of the reply (`keep`). Where Claude
declines, or the service fails, why, in words - never the key, never what was
sent.
"""

import importlib.metadata
import json
import os
import sys
import time

import anthropic

VERSION = importlib.metadata.version("anthropic")

#: The service's one place: api.anthropic.com, as the plugin's manifest names it - never one the environment names.
BASE_URL = "https://api.anthropic.com"

#: What the SDK reads of the environment begins so: let go, every one - none sends it elsewhere, with other
#: credentials, or writes what it sends where Meno shows it (`ANTHROPIC_LOG`).
FOREIGN = "ANTHROPIC_"

#: How long one request answered whole may take, in all, in seconds - from asking to the reply's end: under Meno's own
#: wait for an answer (10 minutes), a stalled read taken into account, so that a stalled service holds no later request
#: behind it. A reply sent as it comes has none: Meno waits from each part of it.
DEADLINE = 560.0
#: How long the service may be silent, in seconds, before the read is given up.
SILENT = 30.0
#: How long a reply sent as it comes may say nothing - while Claude thinks - before an empty part says it is still
#: coming, in seconds: well within Meno's wait from each part (10 minutes).
STILL = 60.0

#: The models offered, and the default (the manifest's choices).
MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5")
MODEL = "claude-opus-5-5"
#: How hard it thinks: the service's levels, and the default.
EFFORTS = ("low", "medium", "high", "xhigh", "max")
EFFORT = "medium"
#: How long one reply may be, in tokens: the default, and the most - and the most of one answered whole, which an
#: earlier Meno waits for whole, 10 minutes at most.
REPLY_TOKENS = 16000
MOST_REPLY_TOKENS = 64000
MOST_WHOLE_TOKENS = 16000

#: The service's betas this asks for: a thinking block whose conversation was edited since (Meno withholds a tool's
#: answer once what it gave leaves the frame) is dropped, not refused; and, where the service declines, another of its
#: models takes the request - its own choice, by why it declined (not for Haiku, which has none).
BINDING = "thinking-binding-controls-2026-08-01"
FALLBACK = "server-side-fallback-2026-07-01"


def client(key, transport=None):
    """The service, as this worker reaches it: by its one address, with the person's key - and, in a test, through
    the transport given."""
    for name in [n for n in os.environ if n.startswith(FOREIGN)]:
        os.environ.pop(name, None)
    more = {"http_client": transport} if transport is not None else {}
    return anthropic.Anthropic(api_key=key, base_url=BASE_URL, max_retries=1, timeout=anthropic.Timeout(SILENT, connect=10.0), **more)


def options_of(options, streamed=True):
    """The options, as the manifest offers them - an option outside what it may be refused, never run as another; and a
    reply longer than one answered whole may be, where it is not sent as it comes."""
    o = options or {}
    model = o.get("model", MODEL)
    if model not in MODELS:
        raise ValueError(f"Not a model this plugin offers: {model}")
    effort = o.get("effort", EFFORT)
    if effort not in EFFORTS:
        raise ValueError(f"Not how hard it may think: {effort}")
    tokens = o.get("reply-tokens", REPLY_TOKENS)
    if isinstance(tokens, bool) or not isinstance(tokens, int) or not 1024 <= tokens <= MOST_REPLY_TOKENS:
        raise ValueError(f"A reply's length is a whole number from 1024 to {MOST_REPLY_TOKENS}")
    if not streamed and tokens > MOST_WHOLE_TOKENS:
        raise ValueError(f"A reply longer than {MOST_WHOLE_TOKENS} tokens is shown as it comes, which this Meno does not do: update Meno, or set a shorter one in Settings, Conversations")
    return model, effort, tokens


def tool_of(t):
    """One of Meno's tools, as the service takes it: by name and what it does - its arguments Meno's to check."""
    return {"name": t["name"], "description": t.get("does", ""), "input_schema": {"type": "object", "additionalProperties": True}}


def answer_of(answer):
    """A tool's answer, as data: JSON text."""
    return answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)


def messages_of(said):
    """What has been said, as the service's conversation: the person's turns (the person's words, Meno's notes, the
    tools' answers) and Claude's - each tool's answer matched, in turn, to the call it answers; a call left without one
    (the turn stopped) answered so, as an error."""
    out = []

    def put(role, blocks):
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": list(blocks)})

    pending = []  # (the calls of Claude's last reply not yet answered, by their ids, in turn)
    for k, s in enumerate(said):
        by = s.get("by")
        if by == "tool":
            if pending:
                use_id = pending.pop(0)
                result = {"type": "tool_result", "tool_use_id": use_id, "content": answer_of(s.get("answer"))}
                if not s.get("ok", True):
                    result["is_error"] = True
                put("user", [result])
            continue
        # (a reply's calls left unanswered: answered as not run, before anything else is said)
        for use_id in pending:
            put("user", [{"type": "tool_result", "tool_use_id": use_id, "content": "Not run: Meno stopped the turn.", "is_error": True}])
        pending = []
        if by == "person":
            put("user", [{"type": "text", "text": s.get("text", "")}])
        elif by == "meno":
            put("user", [{"type": "text", "text": f"(Meno: {s.get('text', '')})"}])
        elif by == "model":
            kept = (s.get("keep") or {}).get("content") if isinstance(s.get("keep"), dict) else None
            if isinstance(kept, list) and kept:
                blocks = kept
            else:
                blocks = [{"type": "text", "text": s["text"]}] if s.get("text") else []
                for j, call in enumerate(s.get("calls") or []):
                    blocks.append({"type": "tool_use", "id": f"meno_{k}_{j}", "name": call.get("tool"), "input": call.get("args") or {}})
            if not blocks:
                blocks = [{"type": "text", "text": "(nothing)"}]
            put("assistant", blocks)
            pending = [b["id"] for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"]
    for use_id in pending:
        put("user", [{"type": "tool_result", "tool_use_id": use_id, "content": "Not run: Meno stopped the turn.", "is_error": True}])
    return out


def words(e):
    """What went wrong with the service, in words: never what was sent, never the key."""
    if isinstance(e, anthropic.AuthenticationError):
        return "The service did not take the key: check it in Settings, Conversations"
    if isinstance(e, anthropic.PermissionDeniedError):
        return "The key may not use this model"
    if isinstance(e, anthropic.NotFoundError):
        return "The service does not know this model"
    if isinstance(e, anthropic.RateLimitError):
        return "The service is asked too much just now: try again in a minute"
    if isinstance(e, anthropic.BadRequestError):
        return f"The service refused the request ({e.status_code})"
    if isinstance(e, anthropic.APIStatusError):
        return f"The service failed ({e.status_code}): try again in a minute"
    if isinstance(e, anthropic.APITimeoutError):
        return "The service took too long to answer: try again in a minute"
    if isinstance(e, anthropic.APIConnectionError):
        return "The service could not be reached: Meno may be offline, or talking to it not allowed"
    return "The service could not be talked to"


def converse(req, make=client, say=None):
    """One request of a conversation, sent; Claude's reply as Meno reads it - or why not, in words. Asked with `stream`,
    each part of its words is given to `say` as it comes - an empty one where it has said nothing for STILL seconds."""
    key = req.get("key")
    if not isinstance(key, str) or not key:
        raise ValueError("No key was given: add it in Settings, Conversations")
    streamed = req.get("stream") is True and say is not None
    model, effort, tokens = options_of(req.get("options"), streamed)
    betas = [BINDING] + ([FALLBACK] if not model.startswith("claude-haiku") else [])
    more = {"fallbacks": "default"} if FALLBACK in betas else {}
    begun = time.monotonic()
    try:
        with make(key).beta.messages.stream(
            model=model,
            max_tokens=tokens,
            system=req.get("told", ""),
            messages=messages_of(req.get("said") or []),
            tools=[tool_of(t) for t in req.get("tools") or []],
            thinking={"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "drop_block"}},
            output_config={"effort": effort},
            betas=betas,
            **more,
        ) as stream:
            # (read to its end - its words said as they come, where asked; or, answered whole, given up once it has taken
            #  too long in all)
            last = begun
            for event in stream:
                now = time.monotonic()
                if streamed:
                    delta = getattr(event, "delta", None) if event.type == "content_block_delta" else None
                    if getattr(delta, "type", None) == "text_delta" and delta.text:
                        say(delta.text)
                        last = now
                    elif now - last > STILL:
                        say("")
                        last = now
                elif now - begun > DEADLINE:
                    raise RuntimeError("The service took too long to answer: try again in a minute")
            reply = stream.get_final_message()
    except anthropic.APIError as e:
        raise RuntimeError(words(e)) from None
    if reply.stop_reason == "refusal":
        raise RuntimeError("Claude declined to answer this")
    content = [b.model_dump(mode="json", exclude_none=True) for b in reply.content]
    text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
    if reply.stop_reason == "max_tokens":
        text += "\n(Cut short: the reply reached its length - Settings, Conversations.)"
    calls = [{"tool": b["name"], "args": b.get("input") or {}} for b in content if b.get("type") == "tool_use"]
    out = {"text": text, "keep": {"content": content}}
    if calls:
        out["calls"] = calls
    return out


def put(m):
    """A line written to Meno."""
    print(json.dumps(m, ensure_ascii=False), flush=True)


def answer(line, make=client, out=put):
    """One line asked, its answer - and, before it, each part of its words as it comes, written by `out`: the key never
    in any of them."""
    try:
        req = json.loads(line)
    except ValueError:
        return {"ok": False, "error": "Not a request"}
    rid = req.get("id")
    key = req.get("key") if isinstance(req.get("key"), str) else None

    def say(text):
        out({"id": rid, "partial": text.replace(key, "[key]") if key else text})

    try:
        if req.get("op") != "converse":
            raise ValueError(f"Not something this plugin does: {req.get('op')}")
        return {"id": rid, "ok": True, "result": converse(req, make, say)}
    except Exception as e:  # (said in words: what fails here is the request's, never the worker's end)
        said = str(e) or "It failed"
        if key:
            said = said.replace(key, "[key]")
        return {"id": rid, "ok": False, "error": said}


def main():
    put({"event": "ready", "version": VERSION})
    for line in sys.stdin:
        if line.strip():
            put(answer(line))


if __name__ == "__main__":
    main()
