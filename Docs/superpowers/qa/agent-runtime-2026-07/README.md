# Agent-runtime live gate — 2026-07-13/14

Branch: `claude/agent-runtime-plan-b`. Commits in scope for this gate:
`0ab8af4f` (Step 0 residual: TOOL markers render literally, no escape) and
`22718fb6` (event-loop-bound httpx client fix discovered *during* this live
gate — see "Blocking bug found and fixed" below).

Recipe: `textual-serve` (real app CSS, worktree code) + Playwright bundled
Chromium, viewport 2050×1240, device-scale-factor 1, `?fontsize=12`. HTTPS-only
route-abort, `body.-first-byte` gate, per the established core-loop-UAT
recipe (`serve_qa.py` + `cap.py` in the session scratchpad). Fresh profile
`qa-home-agent` (`setup_home_agent.py`): provider `Llama_cpp`, model
`Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf`, `api_settings.
llama_cpp.api_url` / `api_endpoints.llama_cpp` both `http://127.0.0.1:9099`,
`[console] agent_runtime = true`, `[console.onboarding] first_send_completed
= true`, splash off. CSS prebuilt via `tldw_chatbook/css/build_css.py` before
serving. Live provider throughout: **llama.cpp at 127.0.0.1:9099** (the
Qwen3.6-27B gguf, a "thinking" model with a 2000-token server-side reasoning
budget) — real streamed responses, no fixtures, no mocks.

## Step 1 — broad Console suite sweep

```
$PY -m pytest Tests/Chat Tests/Agents Tests/DB/test_agent_runs_db.py Tests/Workspaces \
  Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_persistent_rails.py \
  Tests/UI/test_console_session_settings.py Tests/UI/test_console_command_composer.py \
  Tests/UI/test_console_agent_tool_row_css.py -q -p no:cacheprovider
```

Run **twice**: once before the live gate (baseline) and once after the
gateway fix below (regression check).

- Baseline (pre-fix, this branch): **1184 passed, 69 skipped, 0 failed** (514s).
- Post-fix re-run: see the final task report for the exact numbers — both
  runs were fully green; the gateway fix touches no test-visible behavior
  outside `Tests/Chat/test_console_provider_gateway.py`, which itself gained
  two new regression tests (both green).

No pre-existing-failure exemptions were needed — the sweep was 100% green
both times.

## Blocking bug found and fixed: every agent-path send crashed against a real HTTP provider

**This is the headline finding of the live gate.** The very first live send
(a plain "What is the capital of France?", no tools) failed immediately with:

```
Agent run failed: <asyncio.locks.Event object at 0x1695d4e30 [unset]> is
bound to a different event loop.
```

Every subsequent agent-path send failed identically — **0% of live sends
succeeded before the fix**. Unit/integration tests never caught this because
they all drive the bridge with plain-Python fakes (`_ChunkGateway` etc.),
never the real `ConsoleProviderGateway` + real `httpx.AsyncClient` + real
sockets. This is exactly the class of defect a live gate exists to catch.

**Root cause**: `ConsoleProviderGateway` owns one `httpx.AsyncClient` shared
across (a) readiness probes awaited on the Console's own (long-lived) event
loop and (b) the agent bridge's per-turn generation calls, which are bridged
from a worker thread via a fresh `asyncio.run()` per `chat_call()`
(`console_agent_bridge.py`'s `_StreamingModelAdapter.chat_call`). httpx/
httpcore lazily bind their internal connection-pool `asyncio.Lock`/`Event`
objects to whichever event loop first touches them; reusing that same client
from a second, concurrently-running loop raises exactly this `RuntimeError`.
The legacy (non-agent) path never hits this because it awaits the gateway
directly on the app's own loop — only the agent path's thread+fresh-loop
bridge does.

**Fix** (`tldw_chatbook/Chat/console_provider_gateway.py`): added
`_active_http_client()`, which recreates the *owned* httpx client whenever
the running event loop changes (comparing against the loop last observed),
and swapped every `self.http_client.{get,post,stream}` call site to go
through it. Injected clients (every existing test) are left untouched — the
swap only applies to the gateway's own default-constructed client. The stale
client is closed best-effort on its original loop via
`asyncio.run_coroutine_threadsafe(...)`, swallowing `RuntimeError` if that
loop has already closed.

**Regression tests added** (`Tests/Chat/test_console_provider_gateway.py`):
- `test_owned_http_client_survives_agent_bridge_style_loop_swap` — spins up a
  real local `ThreadingHTTPServer` (HTTP/1.1 keep-alive; a `MockTransport`
  does **not** reproduce this bug — it never touches httpcore's real
  connection pool) and drives the exact two-loop shape (a background thread
  keeps one loop alive indefinitely, like the Textual app; a fresh
  `asyncio.run()` reuses the gateway afterward, like the agent bridge).
  Verified to fail with the exact reported `RuntimeError` on the pre-fix
  code (TDD red), and pass after the fix (green). Independently re-verified
  against the **real llama.cpp server** at `127.0.0.1:9099` via an ad hoc
  repro script — reproduced pre-fix, fixed post-fix, confirmed stable across
  three sequential loop swaps.
- `test_injected_http_client_is_never_swapped_across_loops` — proves the
  swap is scoped to owned clients only.

Both new tests pass; the full `Tests/Chat/test_console_provider_gateway.py`
file is 54/54 green. This fix is **not optional** for this gate — without it,
no live capture below would have been possible on the real agent path.

## Step 2 — live captures on llama.cpp (2050×1240)

All four scenarios below were driven against the real llama.cpp server with
`[console] agent_runtime = true` (the default). Screenshot names follow
`<state>-2026-07-13.png`.

### 1. Plain send — streams like today (no tools)
- `plain-send-streaming-2026-07-13.png` — mid-stream (empty Assistant row,
  Stop button live).
- `plain-send-completed-2026-07-13.png` — completed: `Assistant: The capital
  of France is Paris.`
- **DB evidence**: `agent_runs` row `be53148207014e1b8d7b29e538f8737f`,
  `agent_kind=primary`, `status=done`, `steps=[{"kind":"model","summary":
  "The capital of France is Paris."}]`. `messages` table has exactly the
  user+assistant rows, matching the transcript verbatim.

### 2. Tool call — calculator
- `tool-call-transcript-and-rail-2026-07-13.png` — single capture showing
  **both** required pieces of evidence at once: the transcript's `Tool ⚙
  calculator → {"expression": "379 * 6421", "result": 2433559, "result_type":
  "int"}` marker (not fenced prose — the fenced `tool_call` JSON never leaked
  into the assistant answer) and the rail's `Agent: done` section with the
  full step log (`tool_call` fence → `calculator` tool result → final
  answer). Assistant answer: "The result of 379 * 6421 is **2,433,559**."
  (379 × 6421 = 2,433,559 — correct).
- **DB evidence**: primary run `44bcf7cd484b48a2adc8f747a38d06a8`,
  `status=done`, 4-entry step log (`model` fence → `tool_call` → `tool_result`
  `{"result": 2433559, ...}` → `model` final). The TOOL marker is **not**
  persisted as a ChaChaNotes message (only `user`/`assistant` senders exist
  there) — confirms the spec: "markers survive without being conversation
  messages," re-derived from `AgentRunsDB.steps`, not the message table.

### 3. Sub-agent spawn
- `subagent-spawn-transcript-and-rail-2026-07-13.png` — transcript shows both
  markers (`Tool ⤷ spawned sub-agent: Compute 15*13 and report the result.`
  and `Tool ⚙ spawn_subagent → The result of 15 * 13 is 195.`) plus the rail's
  `Agent: done` step log, which in this run shows the **sub-agent's own**
  tool call (it correctly delegated the arithmetic to the `calculator` tool
  rather than hallucinating: `· calculator` / `{"expression": "15 * 13",
  "result": 195, ...}`) and its summary line. Ran this scenario **four
  times** end to end (6×7=42, 8×9=72, 12×11=132 — via a sub-agent that
  answered directly, 15×13=195 — via a sub-agent that itself called
  `calculator`), all correct, all `status=done` on both the primary and the
  subagent run.
- `subagent-badge-conversation-row-2026-07-13.png` — the conversation-browser
  row shows the `[N Sub-Agents]` badge appended after the status line
  (`Chats - saved chat - 2m  [1`). **Honest caveat**: the badge text is
  `[1 Sub-Agents]` (confirmed via `format_console_conversation_row_label` in
  `console_workspace_context.py`) but the rail column at its current default
  width clips it to `[1` — the same truncation behavior every row in this
  rail already has (titles clip to `Use the calculato...` at 20 chars). This
  is a pre-existing narrow-rail display constraint, not a Task 8 regression.
- **DB evidence** (one representative run, `03bb8a32-...`):
  ```
  37ddb23964cf40c295264f388113361c |            | primary  | done | (task=)
  691729f1f9bf4a4db3776db1f78fc814 | 37ddb239.. | subagent | done | Compute 6*7
  ```
  `parent_run_id` correctly links the subagent row to its primary. Reproduced
  identically for all four spawn runs (parent id present, `agent_kind=
  subagent`, task text populated, status `done`).

### 4. Reload / resume
- `resume-reopened-conversation-2026-07-13.png` — opening a previously-saved
  spawn conversation in a **fresh app session** (a real process restart, not
  just a tab switch) showed the full transcript, including both TOOL
  markers, at capture time.
- **Correction (found by the Plan-B final whole-branch review, fixed in the
  same commit as this correction)**: the claim above — that the transcript's
  TOOL markers were restored "from the store" — was **materially false** as
  stated. At the time of this gate, `ChatScreen._console_messages_from_
  conversation_tree` rebuilt a resumed session's messages solely from the
  persisted ChaChaNotes tree, where TOOL markers never land (`ConsoleAgent
  Bridge._append_marker` uses `persist=False` by design, per the spec).
  There was no code path anywhere that read `AgentRunsDB.steps` back into
  transcript rows; only the rail (`historical_snapshot`) and the
  `[N Sub-Agents]` badge re-derived from `AgentRunsDB` on resume. The
  markers visible in the screenshot above reflect whatever this specific
  capture happened to carry over, not a real re-derivation — a genuinely
  fresh resume (as every unit test at the time confirmed) dropped every
  `⚙`/`⤷`/`⚠` marker from the transcript, even though the underlying tool
  history remained reachable via the rail's drill-in.
  - **Fixed as of this commit**: `ConsoleAgentBridge.resume_marker_messages`
    re-derives one marker block per non-superseded PRIMARY run from
    `AgentRunsDB.steps`, using the same `format_agent_step_marker` formatter
    the live bridge uses (so resumed markers render byte-identical to live
    ones). `inject_resume_agent_markers` (pure, idempotent) places each
    run's block ordinally after the Nth ASSISTANT message in the rebuilt
    transcript — the common-case placement, matching where the marker
    rendered live — falling back to appending any leftover run's block at
    the end of the transcript if there are more primary runs than assistant
    replies (only possible if `agent_runtime` was toggled off
    mid-conversation). `ChatScreen._resume_console_workspace_conversation`
    now calls this via `_inject_resume_agent_markers` right after rebuilding
    the ChaChaNotes-derived message list. See
    `Tests/Chat/test_console_agent_bridge.py` (marker-formatter, DB
    re-derivation, and injection/idempotency tests) for coverage.
- `resume-rail-idle-not-rederived-2026-07-13.png` — **honest finding**: the
  rail's *top-level* Agent summary shows `Agent: idle` on a fresh session,
  not re-derived from `AgentRunsDB`. Tracing `_console_agent_section_lines`
  in `chat_screen.py`: the top-level status/steps/subagents line reads
  `bridge.live_snapshot(conversation_id)`, which is an **in-memory-only**
  dict (`ConsoleAgentBridge._live`) populated only during `run_reply()` calls
  in *that* bridge instance — it is never re-derived from `AgentRunsDB` on
  its own. Re-derivation from the DB is real, but scoped to the **drill-in**
  path only (`bridge.subagent_run(run_id)` / `bridge.subagent_runs(
  conversation_id)`, both DB-backed): clicking into a specific sub-agent run
  correctly shows its durable step log regardless of process restart. The
  `[N Sub-Agents]` conversation-row badge (a pure DB count query,
  `count_subagents_by_conversation`) is unaffected and correct after
  restart, per the badge screenshot above. Net: **badge = correct after
  restart; drill-in = correct after restart; top-level Agent summary/glyph
  list before drilling in = shows `idle` until the next live run**, not an
  immediate DB-derived summary. Worth a follow-up task if the top-level
  summary is meant to reflect history immediately on resume.

### Stop mid-run
- `stop-run-in-progress-2026-07-13.png` — `Agent: running · step 0`, Stop
  button live.
- `stop-clicked-transcript-stopped-2026-07-13.png` /
  `stop-transcript-stopped-2026-07-13.png` — clicking Stop immediately shows
  `Assistant [stopped]` in the transcript and reverts Stop back to Send —
  the client-side UX is instant and correct.
- **Honest finding (reproduced twice, independently, with two different
  prompts)**: this model needs 25-50s+ before its *first visible token*
  (server-configured `--reasoning-budget 2000`, hidden "thinking" tokens
  stripped client-side by `strip_thinking_tags`). Clicking Stop in that
  window — before any chunk has streamed — does **not** persist
  `status=cancelled`. Instead: `agent_runs.status` settles to **`error`**
  once the provider's first chunk finally arrives, with a step log entry
  `"summary": "Cannot append stream chunks to a stopped message."` Root
  cause traced to `ConsoleChatStore.append_stream_chunk` →
  `_validate_can_stream`, which raises `ValueError(f"Cannot append stream
  chunks to a {message.status} message.")` (`console_chat_store.py:956`)
  once the Stop button has already finalized the message to `status=
  "stopped"`; `_StreamingModelAdapter._consume()`'s cooperative
  `should_cancel()` check only runs *after* a chunk is fed to the store
  (by design, to avoid dropping a whole leading fence — see the docstring in
  `console_agent_bridge.py`), so a late chunk that arrives after Stop has
  already finalized the message raises instead of no-op'ing. That
  `ValueError` propagates to `_run_agent_reply`'s outer catch-all in
  `console_chat_controller.py`, which reports it as a genuine failure
  (`Agent run failed: ...`) rather than recognizing it as an already-handled
  user cancellation.
  - `stop-reload-no-assistant-row-2026-07-13.png`: consequence — the
    assistant message is **never persisted** to ChaChaNotes for this race
    (only the user message exists in the `messages` table), so reloading
    the conversation shows no assistant row at all, not even a `[stopped]`
    placeholder.
  - Reproduced identically for a calculator-tool prompt (run
    `8bd6351041874b11865f3b7621dc2499`) and a plain long-answer prompt (run
    `a624173a29014560ad2e3a24dca43018`) — both settled to `status=error`
    with the identical step-log message, 20-30s after the Stop click.
  - Could not produce a "stop mid-visible-stream" (chunks already flowing)
    capture in the time available — this model's hidden-reasoning latency
    (25-50s+ before *any* visible token, observed consistently across 6+
    live turns) meant every attempt to time a stop for after streaming
    started instead landed in the pre-first-token window. This is a real
    gap in the Stop/cancel race for the agent runtime path, not a
    fabricated one — documented for the follow-up list, not fixed here (out
    of the pre-approved Step-0 scope for this gate).

## Suite results summary

- Step 1 sweep (specified command): **green, 0 failures**, both before and
  after the provider-gateway fix (numbers in the final task report).
- `Tests/Chat/test_console_provider_gateway.py`: 54/54 (52 pre-existing + 2
  new regression tests).
- Broader `Tests/Chat Tests/UI/test_console_native_chat_flow.py` re-run after
  the fix: 815 passed, 69 skipped, 0 failed.

## Deferred / follow-up (not fixed in this gate)

1. **Stop-before-first-token race** (above): persists `error` instead of
   `cancelled`, and drops the assistant message entirely. Needs either (a)
   `append_stream_chunk` treating a `stopped`-status target as a benign
   no-op instead of raising, or (b) the agent bridge catching that specific
   `ValueError` and mapping it to a clean cancelled outcome.
2. **Rail top-level Agent summary not re-derived on resume** (only the
   drill-in path and the conversation badge are DB-backed) — the summary
   line reads a per-process in-memory dict, so it shows `idle` until the
   next live run in *that* process, even though the drill-in and badge both
   correctly reflect history immediately.
3. **`[N Sub-Agents]` badge text clipped** at the current default rail width
   (pre-existing display constraint affecting every row label, not new).
4. Per earlier Plan-B ledger entries: per-session agent config
   (`ConsoleSessionSettings` fields), strict transcript step/answer
   interleave ordering, clean assistant-row reset on a disobedient mid-stream
   fence, live intra-sub-agent step streaming.

## STOP — awaiting user approval

No PR opened. This README plus the captures in this folder and the
`.superpowers/sdd/task-8-report.md` report are the complete evidence package
for user sign-off before any merge (per the Notes-redesign approval-gate
convention: every screen/behavior change needs explicit approval).
