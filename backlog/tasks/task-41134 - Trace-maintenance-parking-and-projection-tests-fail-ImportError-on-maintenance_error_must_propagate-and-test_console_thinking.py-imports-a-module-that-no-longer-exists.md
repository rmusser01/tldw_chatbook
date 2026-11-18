---
id: TASK-41134
title: "Trace maintenance parking and projection tests fail ImportError on maintenance_error_must_propagate, and test_console_thinking.py imports a module that no longer exists"
status: Done
created_date: 2026-11-15 02:20
updated_date: 2026-11-17 02:26
assignee:
- '@claude'
labels:
- follow-up
- console
- trace
- tests
- dev-red
priority: medium
references:
- Tests/Chat/test_console_trace_maintenance_parking.py
- Tests/Chat/test_console_trace_projection.py
- Tests/Chat/test_console_thinking.py
- tldw_chatbook/Chat/console_trace_maintenance_runtime.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** The legacy trace maintenance loop's parking and projection runtime have no green tests, and a whole test file cannot be collected.

**Evidence.** Through run.sh at `-n 4`: `test_console_trace_maintenance_parking.py`, `test_console_trace_projection.py`, `test_trace_event_projection.py` and `test_console_thinking.py` give 11 failed, 74 passed, 1 error.
- The failures log `ImportError: cannot import name 'maintenance_error_must_propagate' from 'tldw_chatbook.Chat.console_trace_maintenance' (unknown location)` from `console_trace_maintenance_runtime.py:17`, inside `ConsoleRuntime._schedule_legacy_trace_maintenance`. The function exists (`console_trace_maintenance.py:2186`); "(unknown location)" suggests the tests replace that module with a stub that lacks it. TASK-40303's notes recorded the same 12 at base 26b2e677dd.
- The error is collection: `test_console_thinking.py:5` imports `tldw_chatbook.Chat.console_thinking`, which is not in the tree.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each failing node passes or is retired with the commit that changed what it pinned named; the notes say whether the import failure can happen outside tests
- [x] #2 test_console_thinking.py is retired or pointed at the module that now owns thinking display
- [x] #3 The fixed files run on a PR lane
<!-- AC:END -->

## Implementation Plan

1. Reproduce at base cc6006fcf5 through run.sh at `-n 4`.
2. Find in dev history what moved `maintenance_error_must_propagate`, `call_owned_maintenance` and `runtime_provider_active` into `console_trace_maintenance` (the runtime loop imports them from there), and give the tests' stand-in module the real helpers, keeping the narrow worker doubles.
3. Trace `tldw_chatbook.Chat.console_thinking` in history; repoint or retire `test_console_thinking.py`, naming what now owns each behaviour.
4. Put the repaired files on a PR lane (the recovered-suites job and its Tests/CI contract), with their serial time.

## Implementation Notes

**Maintenance loop tests (AC#1).** 6125fbc5b8 ("Integrate native Console phases with current dev") moved three closures out of `run_legacy_trace_maintenance` into `console_trace_maintenance.py` as `call_owned_maintenance`, `runtime_provider_active` and `maintenance_error_must_propagate`, and the loop now imports them from that module. The tests replace that module in `sys.modules` with a bare `ModuleType` holding only worker doubles, so the loop's first import raised `ImportError ... (unknown location)` (the stub has no `__file__`) and the loop never ran. The tests now build their stand-in with `fake_maintenance_module(**doubles)` (in `test_console_trace_maintenance_parking.py`, imported by `test_console_trace_projection.py`): the real module's namespace without its three workers, plus the doubles passed. A worker not passed stays absent, so the loop's narrow-double path (`except ImportError`) still runs. All 11 failing nodes pass; none was retired.

The import failure cannot happen outside tests: the same commit defined the three helpers at module level in `console_trace_maintenance.py`, which every production import loads, so only a stand-in module without them can raise it.

**`test_console_thinking.py` (AC#2): retired.** 3a6263c8a7 (2026-10-12, "fix(ci): complete the pooled-seam sweep ...") added it, with siblings, from a Codex snapshot. Its module `tldw_chatbook.Chat.console_thinking` exists only in that snapshot, bf364a4bf7, which is not an ancestor of dev; no dev commit ever held it. With the import stubbed, all 31 nodes still failed, each on an API that never landed: `thinking_display`/`publish_thinking`, `ConsoleProviderStreamSignals(thinking_callback=)`, `StartAnchoredThinkFilter(on_thinking=)`, `ConsoleChatStore.append_thinking_chunk`, `MessageMetadata.thinking_text`, `ConsoleThinkingBlock`, `LocalReasoningReplay`. No module owns thinking display under that contract. Dev captures thinking as typed `ProviderThinkingDelta` stream items (`console_thinking_capture.py`, `Provider_Contract/thinking_blocks.py`, `console_thinking_history.py`). Repointing the file would also invert two dev decisions: human exporters are answer-only (`test_thinking_privacy_surfaces.py::test_human_exporters_search_and_trajectory_are_answer_only_by_construction`), where the file wanted thinking in the text export; and Z.ai reasoning is proprietary (`Tests/LLM_Calls/test_zai.py::test_zai_declares_reasoning_as_proprietary`), where the file wanted it displayed. The file never collected on dev, so deleting it removes no running coverage. Where dev covers the same behaviour:
- Reasoning kept apart from the answer: `test_console_chat_controller.py::test_direct_stream_pairs_typed_thinking_with_visible_answer`, `test_console_thinking_capture.py`, `test_local_reasoning_gateway.py`.
- Literal think tags after the answer: `test_llamacpp_think_splitter.py::test_visible_answer_makes_all_later_tags_literal`.
- Thinking across a tool round: `test_console_thinking_capture.py::test_tool_boundary_advances_round_and_preserves_order`.
- Retry: `test_console_chat_controller.py::test_retry_with_only_thinking_evidence_reaches_failed_terminal` and `::test_answer_only_retry_does_not_reuse_prior_failed_thinking`.
- Variants, reopen and malformed stored data: `test_console_thinking_persistence.py` (`test_persisted_variant_swipe_uses_current_row_version_for_every_selection`, `test_restore_hydrates_supported_thinking`), `test_thinking_blocks.py::test_malformed_supported_durable_data_is_discarded_without_content`.
- Chatbook round trip: `test_thinking_privacy_surfaces.py::test_chatbook_archive_and_import_preserve_only_approved_sensitive_fields`.
- The request-scoped display sink and its Kimi filter have no dev equivalent, because that design never landed.

**Lane (AC#3).** Both repaired files run whole in the recovered-suites job (`.github/workflows/derived-artifacts.yml`), in its sandboxed block after TASK-40303's node, and are pinned in `RECOVERED_SUITES_TARGETS` (`Tests/CI/test_ci_queue_pressure_contract.py`). Serial with the lane's flags (`-p no:xdist --storage-lease-guard --timeout=180`): 85 passed in 42.3 s at load 51-56.

**Evidence.** Red at base cc6006fcf5 (run.sh, `-n 4`, load 26) over the parking, projection, event-projection and thinking files: 11 failed, 97 passed, 1 error, all 11 on the ImportError. After: the first three files give 108 passed (load 55); `Tests/CI` gives 1640 passed, 1 skipped.

**Files.** `Tests/Chat/test_console_trace_maintenance_parking.py`, `Tests/Chat/test_console_trace_projection.py`, `Tests/Chat/test_console_thinking.py` (deleted), `tldw_chatbook/Chat/console_trace_maintenance_runtime.py`, `Tests/Architecture/test_module_size_ratchet.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`.

**Review r15tr-rev1.**
- **The swallow is removed.** `run_legacy_trace_maintenance` used to catch `ImportError` around the physical-maintenance imports ("Narrow test doubles may provide only the legacy worker", b4e6570dde), which would have hidden a renamed GC worker in production for the life of the process. The loop now imports `PhysicalTraceCompactor` and `TraceGarbageCollector` with its other names at start, and that `except ImportError` is gone. Nothing else needed it: the bare stubs it served are now built with `fake_maintenance_module`.
- **The new test is red first.** `test_a_missing_physical_worker_fails_the_loop_before_any_pass` deletes `TraceGarbageCollector` from the stand-in module. It requires the task to end with that ImportError before any batch runs, and `dispose()` to re-raise it. Before the runtime change it was red ("the loop kept running without its GC worker").
- **`fake_maintenance_module` changed.**
  - It asserts that each double names a worker the real module still has.
  - Instead of leaving GC workers out, it now defaults them to inert doubles: a collector with nothing to collect and a compactor with nothing to compact.
  - Leaving the real GC classes against fake databases instead made two fixed-window projection tests flakier. An interleaved A/B gave 2 of 5 runs red on the new code against 1 of 5 on the old.
- **Two fixed sleeps are now polls.** `test_runtime_starts_legacy_maintenance_only_after_ui_ready` slept 70 ms and `test_runtime_retries_legacy_maintenance_after_unexpected_failure` slept 80 ms; both now poll a 5 s deadline (`_until`). The old code failed one of them too, at load 70. After: 6 of 6 runs green, 96 passed each, at load 53-73.
- `console_trace_maintenance_runtime.py`'s ratchet pin is 185, its measured size.

**Outside this task.** 3a6263c8a7 also added `Tests/UI/test_settings_console_thinking.py`, which is red on dev on the same missing APIs (5 failed). Its two Tests/Chat siblings are already in TASK-41031.

Review-round evidence (run.sh): with the lane's flags, the recovered-suites slice (both files plus TASK-41133's nodes) gives 310 passed in 146 s serial at load 57-68. `Tests/CI` gives 1640 passed, 1 skipped; the module ratchet 207 passed; the ruff gate is OK.
