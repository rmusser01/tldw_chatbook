# Google/Gemini native tool-calls — live gate (task-266)

**Date:** 2026-07-17 · **Branch:** `claude/google-native-266` · **Model:** gemini-flash-latest (Gemini 3 family, real generativelanguage.googleapis.com, streaming ON)
**Harness:** `google_gate.py` (this directory) — the real Console reply engine end-to-end, run BEFORE the set flip via an in-process `NATIVE_TOOLS_PROVIDERS` override (AC #2 ordering: the flip commit exists only because this gate passed). API key from a git-excluded local file; never logged/echoed/committed.

## The gate EARNED its keep: two real failures before PASS

1. **`gemini-2.5-flash` 404** — listed by `/v1beta/models` but "no longer available to new users". Switched to `gemini-flash-latest`.
2. **Gemini 3 thought-signature 400** (`gate-A-B` first runs): current models REQUIRE the response `functionCall` part's `thoughtSignature` to be echoed back verbatim on the follow-up request — our conversion dropped it. Fixed by carrying it opaquely through the OpenAI shapes (`google_thought_signature` on the tool_calls entry; non-streaming parser + streaming fragment emission + request converter re-attach + gateway accumulator extra-key preservation), pinned by three tests. Without a live gate this would have shipped working-on-paper and broken on every current Gemini model.

## GATE: PASS — both cases (`gate-A-B-2026-07-17.txt`)

- **Case A — single tool round-trip: PASS (1.8s)** — `tools=` (functionDeclarations) accepted, no fence protocol, streamed functionCall → calculator → functionResponse follow-up turn accepted, answer `18018`, status done.
- **Case B — parallel multi-tool in ONE turn: PASS (2.4s)** — both tools (`get_current_datetime` + `calculator`) called in a single reply, dispatched as one engine batch, both functionResponse results (coalesced one user turn, positional pairing) accepted with signatures intact; correct combined answer.

## What this proves per AC
- **AC #1** (request/response/streaming conversion end-to-end): both cases streaming ON; the id→name/positional result pairing and the signature round-trip proven against the live API.
- **AC #2** (flip only after a real round-trip): satisfied — flip commit follows this gate.
- **AC #3** (no partial states): cohere untouched, still fence-only (task-267); google was fence-only until the flip landed with its passing gate.

## Suites at gate HEAD
Tests/Agents 137 (incl. the google service-level native test); google-native file 19; google+gateway 84.
