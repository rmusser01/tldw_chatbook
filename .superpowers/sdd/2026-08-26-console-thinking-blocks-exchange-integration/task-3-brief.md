# Task 3 brief — Joined lifecycle, backend refusal, and documentation

## Outcome

Prove the complete Console thinking feature across the real local lifecycle: actual
adapter events become durable assistant-owned evidence, live disclosure expands then
collapses once, history restores collapsed, compatible replay follows the resolved
conversation policy exactly once, importable exchange restores the same state, and
edit/delete/stop/failure/no-evidence paths remain honest. A persistent backend that
cannot round-trip the resolved thinking disposition refuses before provider contact.
Document the behavior without promising hidden chain-of-thought.

## Scope and known seams

- Prefer one joined integration test in
  `Tests/Integration/test_console_thinking_end_to_end.py`, composed from the real
  controller/capture/persistence/history/sync/exchange/UI projection seams already
  implemented by TASK-18932.1–.3 and Tasks 1–2 here.
- The repository has no production remote-server persistence adapter for this
  feature. Exercise the established persistent-backend compatibility contract with a
  persistent unsupported fake and a provider spy; do not invent a new server layer.
- Backend refusal must occur on the real pre-provider send path. Unsupported/old
  round-trip version plus an adapter-owned displayable or proprietary disposition
  yields upgrade-oriented, content-free copy; provider is untouched, no assistant
  evidence is fabricated or persisted, and the user's draft/send remains recoverable.
  V1-compatible, ephemeral, and ignored-disposition controls dispatch.
- Join existing importable selected-conversation JSON and Chatbook V2 behavior. Do
  not add thinking to human-readable or diagnostic formats.
- Opaque future envelopes survive unrelated feedback/title/content writes byte-for-
  byte, stay unrendered/unreplayed, and block operations that replace generation
  ownership. Whole-record sync conflict must not splice answer and thinking from
  different sides.
- Keep production edits evidence-driven. If the joined test already passes a seam,
  record the verification instead of refactoring it.
- Root owns isolated live Console verification after implementation and independent
  reviews. Do not run a full repository suite.

## Required joined evidence

Use distinct visible-answer, displayable-thinking, raw ADR-063 continuation, exact
application-notice, and no-evidence canaries. Decode real owners and outputs rather
than relying on aggregate string searches.

1. Create a real temporary conversation and stream a displayable thinking event plus
   answer. Prove actual-turn evidence only, expanded-live state, one terminal
   auto-collapse, and no row from capability alone.
2. Hydrate/restart from the durable message and prove historical collapsed restore.
3. Resolve Auto/Include/Exclude/Required for compatible local llama.cpp/vLLM-style
   replay. Include injects the exact supported envelope once in counted and dispatched
   payloads; Exclude omits it; Required cannot be overridden. A thinking and a
   non-thinking model may share the same local backend.
4. Sync/export/import through selected-conversation JSON and Chatbook V2 into a second
   database, then hydrate the restored assistant evidence and conversation policy.
5. Edit/delete/generation replacement clears or removes assistant-owned thinking and
   continuation atomically. Stop and handled failure preserve only actual received
   evidence; no-event turns fabricate no row. Proprietary evidence renders only
   `Proprietary thinking obfuscated - not available` and never persists that notice.
6. Refuse unsupported persistent backends before provider contact and prove compatible
   and non-applicable controls still dispatch.
7. Cover opaque-version and whole-record conflict invariants at the nearest existing
   persistence/sync tests if they cannot be expressed cleanly in the joined test.

## User documentation

Update only the relevant sections of:

- `Docs/User_Guide/console/chat-basics.md`
- `Docs/User_Guide/console/context-and-rag.md`
- `Docs/User_Guide/console/agent-runs-and-tools.md`
- `Docs/User_Guide/settings.md`
- `tldw_chatbook/Chatbooks/CHATBOOKS_GUIDE.md`

Explain actual adapter-reported evidence; displayable blocks versus the exact
`Thinking · unavailable` disclosure/notice; expanded-live then one-time collapse;
manual expansion and the default-on presentation-only setting; conversation-level
Auto/Include/Exclude plus effective Required and “Save as default for new
conversations”; local llama.cpp/vLLM model-specific compatibility; persistent-backend
upgrade refusal; sensitive importable exchange versus ordinary human-readable
omission; and Planning as safe session-only activity distinct from model thinking.
Do not imply capability alone means reasoning was produced or promise access to
hidden chain-of-thought.

## Required verification

First run the new joined test red, make the smallest evidence-driven correction, then
run:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest -q \
  Tests/DB/test_chachanotes_console_thinking_migration.py \
  Tests/Chat/test_thinking_blocks.py \
  Tests/Chat/test_console_thinking_persistence.py \
  Tests/Sync_Interop/test_console_thinking_sync.py \
  Tests/Chat/test_llamacpp_think_splitter.py \
  Tests/Chat/test_console_provider_gateway.py \
  Tests/Chat/test_console_thinking_capture.py \
  Tests/Chat/test_console_thinking_history.py \
  Tests/Chat/test_console_prepared_request.py \
  Tests/UI/test_console_thinking_disclosures.py \
  Tests/UI/test_settings_context_memory_controls.py \
  Tests/UI/test_console_context_controls.py \
  Tests/Chat/test_thinking_conversation_exchange.py \
  Tests/Chatbooks/test_chatbook_thinking_round_trip.py \
  Tests/Chat/test_thinking_privacy_surfaces.py \
  Tests/Integration/test_console_thinking_end_to_end.py
```

Also run the CSS bundle-sync check, persistent diagnostic inventory check if the
script exists, scoped Ruff format/check over touched production/tests, relevant
`py_compile`, and `git diff --check`. If a documented command/path is stale, locate
the actual repository script and report the discrepancy; do not create a substitute
inventory system.

Update `progress.md`, create `task-3-report.md`, self-review against ADR-090 and the
accepted design, commit, and return the hash plus RED/GREEN/static evidence. Do not
mark the Backlog child or parent Done; root closes them only after independent reviews
and isolated live verification.
