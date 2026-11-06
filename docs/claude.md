# Claude: what the plugin relies on of Anthropic's service

The Claude plugin talks with Claude for a conversation on Meno's page
(Meno's role *Talking with a model, in a conversation*). Every fact about
Anthropic's service it relies on is listed here with the primary source it
was checked against, on 2026-11-06. A fact changed at the source is a
change to `plugins/claude/worker.py`, its manifest and this page together.
Its tests (`tests/test_plugin_claude.py`) pin what the worker sends; they
reach no live service.

| What the worker relies on | As it stands | Checked against |
| --- | --- | --- |
| The SDK, and when it was released | `anthropic` 2.1.0, uploaded 2026-10-21 (the newest at least two weeks old when locked) | PyPI, `https://pypi.org/pypi/anthropic/json`, `releases["2.1.0"][0].upload_time` |
| How the key is sent | `Authorization: Bearer <key>`; the older `x-api-key` header is also taken | Authentication: `https://platform.claude.com/docs/en/manage-claude/authentication` (*Create and use a key*); SDK 2.1.0 `anthropic/_client.py` `_api_key_auth` sends `Authorization: Bearer` |
| The API version | `anthropic-version: 2026-10-21`, which the SDK sends | SDK 2.1.0 `anthropic/_constants.py` `ANTHROPIC_API_VERSION`; Authentication (its request example) |
| The models offered | `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5` | Models overview: `https://platform.claude.com/docs/en/about-claude/models/overview` (*Claude API ID*) |
| A reply's length | at most 128K tokens on each, synchronously; the plugin offers 1,024 to 64,000, 16,000 by default, and streams each reply | Models overview (*Max output*) |
| Thinking | adaptive: `thinking: {"type": "adaptive"}`; always on for Opus 5.5 | Models overview (*Thinking*); Steering thinking: `https://platform.claude.com/docs/en/build-with-claude/thinking-steering-and-cost` |
| How hard it thinks | `output_config.effort`: `low`, `medium`, `high`, `xhigh`, `max` - all five on each of the three; the service's default is `medium` on Opus 5.5 and Haiku 5.5, `high` on Sonnet 5.5; the plugin sends `high` unless the person chooses another | Effort: `https://platform.claude.com/docs/en/build-with-claude/effort` (*Effort levels* table, and each model's paragraph) |
| A reply's thinking, given back | the assistant turn echoed exactly as received, its thinking blocks with its tool calls, when a tool's answer is returned | Thinking in tool and multi-turn workflows: `https://platform.claude.com/docs/en/build-with-claude/thinking-tool-workflows` |
| An edited conversation | `thinking.block_binding.prefix_mismatch_behavior: "drop_block"` under the beta `thinking-binding-controls-2026-08-01`: a thinking block whose conversation changed since is dropped, with every one after it, and the request succeeds; taken with adaptive thinking on all three models | Preserved thinking: `https://platform.claude.com/docs/en/build-with-claude/preserved-thinking` |
| A request declined | `stop_reason: "refusal"`; `fallbacks: "default"` under the beta `server-side-fallback-2026-07-01` retries it on the model Anthropic names for why; Haiku 5.5 has no server-side fallback, so the plugin sends none for it | Refusals and fallback: `https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback` |

## Why `drop_block`

Anthropic's page calls `drop_block` a stopgap while the edit that causes a
mismatch is fixed, and notes its cost: the model answers that turn without
the dropped reasoning, and the prompt cache restarts at the edit. Meno makes
one edit on purpose. A tool's answer that gave things now outside the
conversation's frame is withheld from what is sent again, because the frame
is what the conversation may see (Meno's docs/CONVERSATIONS.md, *The frame
is its scope*). That edit is the frame working, not one to fix, so the
plugin keeps `drop_block`: an account Anthropic enforces the check on gets
the turn answered, not refused.

## How long it waits

Meno waits 3 minutes for a worker's answer, and a worker answers one
request at a time. So that a service that stalls holds no later request
behind it, the plugin gives one request 140 seconds in all, from asking to
the reply's end, and the service 30 seconds of silence (10 to connect);
past either it says, in words, that the service took too long. A request
that fails to connect, or that the service is too busy for, is asked again
once. With a silence at its very end, a request ends inside Meno's wait.

