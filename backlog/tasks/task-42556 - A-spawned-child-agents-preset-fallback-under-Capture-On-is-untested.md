---
id: TASK-42556
title: "A spawned child agent's preset fallback under Capture On is untested"
status: Done
assignee:
- '@claude'
created_date: 2026-11-26 04:50
updated_date: 2026-11-26 08:39
labels:
- follow-up
- console
- agents
- trace
priority: low
references:
- tldw_chatbook/Agents/agent_runtime.py
- backlog/decisions/200-preset-pre-tool-fallback-targets.md
- backlog/decisions/337-console-reference-backed-semantic-trace-ledger.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in TASK-42525 (round 26, 2026-11-25). That task made a primary Console agent's provider fallback work under Capture On: the fallback starts a new trace chain. A spawned child's preset fallback was left as it was: children keep one chain and one segment. No test drives a child whose preset target fails into its fallback with Capture On, so it is not known whether that fallback is sent and traced, refused with a clear reason, or fails in a way the user cannot read.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A test through the real controller, bridge, gateway and trace factory (only the provider handlers faked) drives a spawned child's preset fallback with Capture On, and the notes say what happens
- [x] #2 If the fallback is sent, it is traced and the parent's next message is sent too; if it is refused, the refusal happens before any provider call or reservation and the user sees why
- [x] #3 Any product change is shown red before and green after; the test runs on a PR lane
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extend `Tests/Chat/test_console_spawn_subagent_trace_capture.py::_spawn_turn` (real controller, store, bridge, gateway, trace factory; only the HTTP socket faked) with a `preset_fallback` option: the parent spawns the `worker` preset, whose own model answers 404 `model_not_found` (a typed unavailable-model failure, ADR-200) and whose `fallback_models` names a second model.
2. Record what happens with Capture On and Off on dev 6e41919880.
3. Apply AC#2. ADR-337's fallback amendment lists "a spawned child's preset fallback" as still refused, so making it work needs a new rule: report that as an owner question. Check the refusal: before any provider call or reservation, and visible. Fix whichever fails, without a new trace rule.
4. A new test file on a PR lane (recovered-suites), red on dev and green after; User Guide; the brief's checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Status:** the child's fallback is refused, as ADR-337 records ("Still refused: a spawned child's preset fallback", in the TASK-42525 fallback amendment). On dev the refusal came after a reservation, and the user was not told why. Both are fixed. Making the fallback run under capture needs a new ADR-337 rule; that is an owner question (below) and is not implemented.

**Test** (`Tests/Chat/test_console_child_preset_fallback_trace.py`). It runs through the real `ConsoleChatController`, `ConsoleChatStore`, `ConsoleAgentBridge`, `ConsoleProviderGateway` and `ConsoleTraceBoundaryFactory` over real SQLite, via `_spawn_turn` in `Tests/Chat/test_console_spawn_subagent_trace_capture.py`. Only the HTTP socket is faked. The new `preset_fallback` option sets up the case:
- the parent spawns the `worker` preset (`custom-ep:spawntrace`/`synthetic-qwen`);
- the preset's `fallback_models` names `synthetic-qwen-fallback`;
- the preset's own model answers 404 `model_not_found`, a typed unavailable-model failure that ADR-200 lets switch at once.

**What happened on dev 6e41919880.**
- *Capture Off:* the child falls back (`synthetic-qwen` then `synthetic-qwen-fallback`) and completes.
- *Capture On:*
  1. The runtime selected the fallback ("Provider fallback: ... -> custom-ep:spawntrace/synthetic-qwen-fallback (after ChatModelUnavailableError)"), which saves it as the run's active target.
  2. The trace reserved that call, then refused it at dispatch binding: the call is compared with the failed one as a retry, and the model differs.
  3. The reservation stayed `reserved` (route identity empty); the trace rows read `agent_first error`, then `None reserved`.
  4. The fallback model was never asked.
  5. The child ended `error` on "trace capture could not be saved, so this request was not sent", which does not say why.
- In both modes the parent's follow-up after the spawn result was sent (`parent final`).

**Fix.** `AgentService` takes `preset_fallback_refusal` (`Agents/agent_service.py`). Under Capture On, `Chat/console_agent_preparation.py` passes `PRESET_FALLBACK_NOT_CAPTURED`, and every preset alternate becomes an ADR-200 visible skip (`ready=False`, with that reason). As a result:
- the runtime never selects the alternate (no active-target write), builds no call and reserves nothing;
- the child's steps show "Provider fallback skipped: custom-ep:spawntrace/synthetic-qwen-fallback (Capture is on, and a sub-agent's fallback cannot be traced; turn Capture off for this chat to let sub-agents fall back)";
- the child ends on its own model's error ("provider returned HTTP 404 (The requested model is unavailable)").

This covers every preset alternate a captured run's children carry, including code-review children, which take their slot's fallbacks through the same path. Capture Off is unchanged. Provider-only `fallback_providers` (ADR-110) and primaries are untouched.

**Red/green.** With the source fix reverted, `test_a_spawned_childs_preset_fallback[capture_on]` fails at `assert switched == []` (the child switched); `[capture_off]` passes. Both pass with the fix.

**Review fixes and follow-up (2026-11-26).** The owner answered the question below ("Yes, new chain (Recommended)"); TASK-42564 builds it. Under Capture On the child's fallback is now sent and traced, and the visible skip stays only for an alternate that keeps the child's target. The review's minors were applied there: the skip no longer overwrites an unready alternate's own reason, and the copy is shortened to fit the 200-character step summary ("same target, not traceable while Capture is on; turn Capture off to allow it").

**Owner question (answered, see above).** May a spawned child's preset fallback under Capture On start a new chain in the child's own segment, as a primary's provider fallback does since TASK-42525? Recommendation: yes, mirroring amendment 1 for children:
- the child keeps its actor and segment;
- only the child's adapter mints the chain, for the next frozen preset target;
- the chain's first call is a TOOL_LOOP at sequence 0, linked to the failed child call.

Because ADR-200 allows a switch only before any tool batch, the child's history at that point is its system prompt and task, so no history rewrite (amendment 2) is needed. Until then the visible skip stands.

**Lane.** The new file is appended to recovered-suites ("Run the recovered suites"; bootstrap_profile, so after the sandboxed suites), with the workflow comment and `RECOVERED_SUITES_TARGETS` in `Tests/CI/test_ci_queue_pressure_contract.py`. With the step's flags (`-p no:xdist --storage-lease-guard --timeout=180`) at load 40-44 it takes 14.3 s: 2 passed, calls of 7.3 s and 4.5 s.

**Runs** (tree with the fix, load 28-60 from other rounds' agents):
- Soak: the new file and `test_console_spawn_subagent_trace_capture.py` (whose harness changed), 20 rounds at `-n 4`: 22 passed in every round, at load 32.8-60.0.
- Covering files (`test_preset_fallback.py`, `test_fallback_chain.py`, `test_agent_presets.py`, the spawn-trace file, the new file, `test_console_agent_bridge.py`, the code-review tool-provider and dispatch files, `test_console_workflow_scheduling.py`, `test_console_agent_call_thinking.py`, `test_console_worktree_merge_confirm.py`, `test_agent_service_review_containment.py`) at `-n 4`: 610 passed, 27 failed. All 27 fail the same way on dev 6e41919880, run node by node in a scratch worktree: 26 in `test_console_workflow_scheduling.py`, 1 in `test_agent_service_review_containment.py`.
- `Tests/CI` at `-n 4`: 2009 passed, 2 skipped.
- `Tests/Architecture` serially: 1507 passed, 5 skipped.
- `./scripts/preflight.sh` passed. `ruff check tldw_chatbook --select F` is clean, and `Tests/lint/ruff_gate.py` exits 0.
- The recovered-suites step via `r19-step.sh` (`-p no:xdist --storage-lease-guard --timeout=180`): 2024 passed, 1 skipped, 1 deselected, in 2740 s at load 28-45.

**Files.**
- `Agents/agent_service.py`: size row 7698 to 7705, +7, dated.
- `Chat/console_agent_preparation.py`.
- `Tests/Chat/test_console_spawn_subagent_trace_capture.py`: the harness option, `facts.models` and `facts.child_steps`.
- The new test file.
- `Docs/User_Guide/console/semantic-trace-capture.md` (Sub-agent calls).
- `Docs/User_Guide/console/agent-runs-and-tools.md` (Preset fallback).
- `.github/workflows/derived-artifacts.yml` and its lane contract.
<!-- SECTION:NOTES:END -->
