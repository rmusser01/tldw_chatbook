# Live Per-Run Usage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show an honest, bounded current-call output-token scalar for the primary Console run and every observable child run without changing final accounting.

**Architecture:** `AgentService` scopes every model loop with its real run identity. The shared streaming adapter captures that identity, emits bounded scalar events, and the bridge publishes one immutable per-run value at most once per second. Existing primary and survivor timers render the value.

**Tech Stack:** Python 3.11, Textual 8, dataclasses, threading/contextlib, pytest/pytest-asyncio

**Spec:** `Docs/superpowers/specs/2026-09-12-live-per-run-usage-design.md`

## Global Constraints

- Preserve final `RunOutcome.total_tokens`, fleet `total_tokens`, provider accounting, persistence, automatic-work budgets, and pricing behavior byte-for-byte.
- Retain only one accumulator per active run; never retain chunks or per-chunk usage samples.
- Provider provenance requires a recognized explicit output/completion field with `type(value) is int and value >= 0`; zero is authoritative, while bool/float/string/negative/missing values are invalid.
- Publish the first observable scalar immediately and subsequent changes at most once per second. Never send a Textual event or repaint directly from a stream callback.
- Keep the existing 0.2-second primary poll and 1-second survivor tick. Idle and settled surfaces must own no new timers.
- No schema, migration, configuration, tokenizer dependency, user-config import, provider-specific branching, cost estimate, or historical backfill.
- Work only in /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/agent-orchestration-pr, branch codex/agent-orchestration-remaining; use .superpowers/sdd/2026-09-11-agent-orchestration-pr-integration/venv/bin/python under pytest isolation. Root owns commits and Backlog status. Leave edits unstaged, run targeted tests only, preserve unrelated static debt, and dispatch no subagents.

## ADR check

ADR required: yes  
ADR path: `backlog/decisions/156-live-per-run-stream-usage-attribution.md`  
Reason: the optional `run_model_scope` establishes a new cross-module `AgentService`/Console-adapter attribution and lifecycle contract. Existing ADRs are immutable and do not define it.

---

### Task 1: Add exact attribution and the bounded scalar bridge

**Files:**

- Read: `backlog/decisions/156-live-per-run-stream-usage-attribution.md`
- Read: `Docs/superpowers/specs/2026-09-12-live-per-run-usage-design.md`
- Modify: `tldw_chatbook/Agents/agent_service.py`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Modify if required for existing emission provenance: `tldw_chatbook/Chat/console_provider_gateway.py`
- Test if touched: targeted `Tests/Chat` gateway provenance/stream tests identified from the actual gateway seam
- Test: `Tests/Agents/test_agent_service.py`
- Test: `Tests/Chat/test_console_agent_bridge.py`

**Interfaces:**

- Consumes: the run ID and agent kind already created inside `AgentService._run_one`; actual textual deltas and partial usage snapshots already observed by `_StreamingModelAdapter`; the bridge clock.
- Produces: optional `AgentService.run_model_scope(run_id, agent_kind)`; typed `AgentLiveUsageEvent`; immutable `AgentLiveTurnUsage`; `AgentLiveSnapshot.turn_usage`; sequence-safe `live_snapshot` and `live_run_snapshot` results.

- [x] Verify root adopted ADR-156 and the design spec and linked both from the In Progress TASK-18923 before code.
- [ ] Add failing `AgentService` tests proving primary, inline child, and fleet child loops enter the callback with their exact `(run_id, agent_kind)`, exit it on success/error/cancel, nest LIFO, and use `nullcontext()` when omitted.
- [ ] Run `pytest -q Tests/Agents/test_agent_service.py -k "model_scope or run_scope"` and confirm the new tests fail because the constructor contract does not exist.
- [x] Add `run_model_scope: Callable[[str, str], AbstractContextManager[None]] | None = None` to `AgentService.__init__`, store it, and enter it immediately around `run_agent_loop` inside the existing `use_run_actor` block.
- [x] Run the same targeted command and confirm it passes.
- [ ] Add failing bridge tests for: cumulative UTF-8 bytes across differently chunked text; `ceil(bytes / 4)` local values; provider precedence; explicit provider zero; rejection of bool/float/string/negative/missing/aggregate-only counts; no local fallback after valid provider observation; and no retained text.
- [ ] Add failing lifecycle/concurrency tests for: monotonic sequence assignment per run without an unpruned finished-run ID map; a new call replacing its predecessor; late old-sequence events ignored; overlapping primary/sibling attribution; first publication immediate; later publication no faster than one second; adapter callback exceptions contained; finish/error/cancel/prune/shutdown cleanup; map cardinality no greater than active runs; final `RunOutcome.total_tokens` unchanged.
- [ ] Run `pytest -q Tests/Chat/test_console_agent_bridge.py -k "live_usage or run_scope"` and confirm the failures describe the missing scalar bridge.
- [x] Implement `AgentLiveTurnUsage`, `AgentLiveUsageEvent`, `_LiveTurnUsageAccumulator`, and the bridge's locked active-run map. Count only cumulative actual provider text bytes, store only integers/timestamps/provenance, arbitrate provider values exactly as specified, and remove state only on a matching sequence.
- [x] Preserve the gateway's existing per-emission synthetic flag through a minimal backward-compatible adapter-facing seam; exclude fallback copy while counting later real provider text. Preserve ordinary gateway consumers and final accounting. Do not infer live output from budget-normalized totals or add provider-specific branches. Exercise actual gateway/adapter emissions for synthetic+real text and explicit raw terminal usage.
- [x] Implement `_StreamingModelAdapter.run_scope`, capture `(run_id, agent_kind)` before submitting async work, emit start/text/provider/finish events, and contain sink errors so telemetry cannot affect model execution. Bound any diagnostic per run/call and use fixed content-free text; no delta, payload or arbitrary exception repr. Cover the late-start-after-terminal case at the real adapter/service boundary so callbacks cannot resurrect an ended run slot.
- [x] Wire the adapter scope into the Console-created `AgentService` and the adapter sink into the bridge. Preserve `None` behavior for every other service construction.
- [x] Prove actual primary and child counts are observable before either run emits its first step: the current child `live_run_snapshot` returns None before on_step, and the primary uses a separate per-turn live key. Establish the exact run-to-current-snapshot attribution through the new scope without inventing AgentSteps or advancing the primary pointer from a child. Preserve per-conversation ownership and reject late started events after scope exit/terminal cleanup without a finished-run history map.
- [x] Run `pytest -q Tests/Agents/test_agent_service.py Tests/Chat/test_console_agent_bridge.py -k "model_scope or run_scope or live_usage"` and confirm all focused tests pass.
- [x] Run scoped changed-line Ruff/format and whitespace checks, write the task report with exact tests/results/warnings/paths, and leave edits unstaged for root commit and independent review.

Task 1 completed through `16c10faca7`, `2f39215549`, and `ca9418b33a`; independent review and two scoped fix rounds are clean. Process deviation: constructor/event contract RED was captured before initial implementation; most broader behavioral tests were added afterward as sensitivity coverage, so their requested pre-implementation RED steps above remain unchecked. Later lifecycle, gateway and terminal-containment fixes have separate actual RED/GREEN evidence. The task report records exact commands and qualified overlapping results. The unchanged per-chunk signals snapshot exception boundary is carried to final branch review. Task 2 and the Backlog task remain open.

### Task 2: Render through the existing cadence and document semantics

**Files:**

- Modify: `tldw_chatbook/UI/Console_Modules/agent.py`
- Modify: `Tests/UI/test_console_turn_activity_line.py`
- Modify: `Tests/UI/test_console_fleet_panel.py`
- Modify: `Tests/UI/test_console_fleet_survivor_tick.py`
- Modify: `Tests/UI/test_console_fleet_lifecycle_controller.py`
- Modify: `Docs/User_Guide/console/agent-runs-and-tools.md`
- Modify: `backlog/tasks/task-18923 - Agent-rail-live-per-run-status-line-elapsed-and-streaming-tokens.md`

**Interfaces:**

- Consumes: `AgentLiveSnapshot.turn_usage`, existing `console_turn_activity_text(snapshot, now, children, pending_approval, pending_copy)`, attached `FleetHandle.run_id`, bridge `live_run_snapshot`, primary transcript poll, and survivor tick.
- Produces: pure `_live_usage_label(usage) -> str`; `_fleet_row_from_handle(handle, *, now, live_snapshot=None)`; primary and child secondary labels with explicit provenance; unchanged timer ownership and terminal budget labels.

- [ ] Add failing pure-render tests for `N provider output tok`, `~N local output tok`, provider zero omission, generating/thinking elapsed from `turn_usage.started_at`, and absence during setup/tool/approval/terminal/idle states.
- [ ] Add failing fleet tests showing each attached child receives only its own live snapshot, row order and identity stay stable, unavailable usage adds nothing, live usage does not replace task/result/error text, and terminal rows retain only existing `budget tok` accounting.
- [ ] Add failing cadence/lifecycle tests showing many chunk events cause at most one changed render per publication window, the existing 0.2-second primary poll observes published changes, the existing 1-second survivor tick advances child counts, final settle performs its current repaint, and idle/settled screens own no survivor or usage timer.
- [ ] Run `pytest -q Tests/UI/test_console_turn_activity_line.py Tests/UI/test_console_fleet_panel.py Tests/UI/test_console_fleet_survivor_tick.py Tests/UI/test_console_fleet_lifecycle_controller.py -k "usage or token or survivor"` and confirm the new assertions fail.
- [ ] Implement `_live_usage_label`; append it only in active generating/thinking branches; use `turn_usage.started_at` as the elapsed base when no primary step exists.
- [ ] Extend `_fleet_row_from_handle` with the optional live snapshot, resolve it by the handle's attached `run_id`, and append the live label without changing status glyphs, elapsed, ordering, cancellation, steering, errors, results, or terminal budget segments.
- [ ] Load /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.agents/skills/impeccable/reference/craft-floor.md before UI editing; follow DESIGN.md and inspect wide/narrow painted rows in one batch, fixing observed defects and confirming once.
- [ ] Keep both existing timers unchanged and rely on snapshot/payload equality to avoid unchanged DOM writes. Add no stream-to-UI callback.
- [ ] Document provider-versus-local labels, current-call reset, zero/unavailable behavior, lack of live prices, and the distinction from terminal budget tokens in the Console user guide.
- [ ] Run the four targeted UI files above. Preserve Task 1 evidence and run only additional service/bridge selectors needed to resolve new integration concerns; do not repeat whole service/bridge modules merely to regenerate unchanged evidence.
- [ ] Run `ruff check tldw_chatbook/Agents/agent_service.py tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/UI/Console_Modules/agent.py Tests/Agents/test_agent_service.py Tests/Chat/test_console_agent_bridge.py Tests/UI/test_console_turn_activity_line.py Tests/UI/test_console_fleet_panel.py Tests/UI/test_console_fleet_survivor_tick.py Tests/UI/test_console_fleet_lifecycle_controller.py` and `git diff --check`.
- [ ] Update TASK-18923 implementation notes after targeted tests and scoped static checks pass; root checks acceptance criteria and marks Done only after independent review. Include ADR-156, modified files, provenance semantics, timer reuse, and the explicit statement that final accounting did not change.
- [ ] Run scoped changed-line Ruff/format and whitespace checks, write the task report with exact tests/results/warnings/paths, and leave edits unstaged for root commit and independent review.

## Final review

- [ ] Read ADR-156, the spec, TASK-18923, and the final diff together; verify every acceptance criterion has direct automated evidence and no unrelated behavior changed.
- [ ] Search changed artifacts for `TBD|TODO|FIXME|similar to|as appropriate` and resolve every planning or implementation placeholder.
- [ ] Verify type names and labels are identical across service, adapter, bridge, UI, tests, spec, ADR, task notes, and guide.
- [ ] Confirm the focused suite did not import user configuration, contact a provider, add a timer, retain stream text, alter a schema, or calculate cost.

