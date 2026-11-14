---
id: TASK-14877
title: 'Dev baseline: 9 failing tests in Tests/Chat mask real regressions'
status: Done
assignee: [rmusser01]
created_date: '2026-08-10 22:46'
updated_date: '2026-10-02 12:30'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
origin/dev at 59cf35d6e fails 9 tests in Tests/Chat with zero related changes in flight. Verified twice on a pristine detached checkout of origin/dev: Tests/Chat + Tests/MCP gives 9 failed / 4331 passed / 62 skipped, and the same 9 names fail when their files are run in isolation. Groups: (a) 5x Tests/Chat/test_console_agent_swap.py::test_mcp_tool_call_* (executes_end_to_end_when_state_allows, ask_state_routes_through_review_hook_and_approves, session_approval_suppresses_card_on_next_turn, ask_state_times_out_denies, gates_subagent_call_same_as_primary); (b) 1x Tests/Chat/test_console_ephemeral.py::test_promotion_restores_ephemeral_flag_if_persist_returns_none_unexpectedly; (c) 3x Tests/Chat/test_tool_output_disclosure.py (full_tool_output_is_reachable_from_the_mounted_transcript, pressing_o_expands_the_selected_marker, two_calls_in_one_turn_expand_independently). Cost already incurred: supervisor-fleet PR 2a (#1477) rebased onto dev and its battery went from 0 failures to 9, which had to be individually traced against a pristine dev checkout to prove none belonged to the branch — roughly 30 minutes of comparison runs that a green baseline would have made unnecessary. The MCP group is the most dangerous: it covers the tool-call permission path, so a genuine permission regression landing there would be indistinguishable from this noise.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Root cause identified for each of the three groups (they may have distinct causes)
- [x] #2 All 9 tests pass on a clean origin/dev checkout, or any that encode obsolete behavior are deliberately rewritten with the change documented
- [ ] #3 Tests/Chat is green on dev across two consecutive runs — NOT MET AS WRITTEN, deliberately left unticked: at the re-triage base (e92b01515f) Tests/Chat carries large red masses owned by tracked classes beyond this task (hook-consent send-gate + config-admission drift, TASK-33621.36 and the session's enrollment notes), plus two stop-path defects this task surfaced and does not own (see Implementation Notes). The three files this task touched were verified green across consecutive runs modulo those two documented, pre-existing reds. Whole-directory green now requires those owner tasks, not this one.
- [x] #4 Tests/UI/test_css_class_coverage_contract.py::test_registry_entries_are_still_composed also fails on dev (flags console_transcript.py) — verified reproducing on merge-base 762596846, unrelated to any fleet work; include it in the sweep
<!-- AC:END -->

## Implementation Plan

1. Re-verify the premise at the current dev base (e92b01515f): run the 9
   originally-named nodes plus the AC#4 CSS node, capture each failure
   signature, and re-triage every item as (a) fixed since filing, (b) live
   defect to fix red→green, or (c) member of a broader already-tracked red
   class to map, not re-fix.
2. For (b) items whose signature is the ADR-126 config-participant admission
   (`RecoveryRequired: raw_source_selection_changed` under the per-test
   sandbox), apply the documented remedy from TASK-32873/ADR-179 and the
   in-base precedent 72a80e0b64: `bootstrap_profile` markers for suites that
   drive real controller sends / mount real transcript widgets.
3. Targeted verification: the touched files green twice consecutively under
   the plain pytest command; a reverse check that the fix does not mask the
   guarded behavior (the failure signature each test had before the marker is
   gone because admission passes, not because an assertion was deleted).
4. Close with the re-triage table in Implementation Notes.

ADR required: no
Reason: test-only repair following the existing ADR-126/ADR-179 enrollment
pattern already established in Tests/conftest.py (TASK-32873) — no new
storage, sync, interface, or product-boundary decision is being made.

## Implementation Notes

Re-triage base: origin/dev tip `e92b01515f` (the task was filed 2026-08-10
against dev `59cf35d6e`; the landscape has changed massively since — premise
re-verified before touching anything).

### Per-test re-triage

| Original test | Verdict | Evidence |
|---|---|---|
| `test_console_agent_swap.py::test_mcp_tool_call_executes_end_to_end_when_state_allows` | (b) live defect → FIXED red→green | Base: send refused, `result.accepted False`, visible copy "Hooks unavailable; review or disable hooks before sending." Probed cause: `_hook_admission_reason` → `read_hooks_config_snapshot()` → `RecoveryRequired("raw_source_selection_changed")` (ADR-126 admission under the per-test sandbox), swallowed by the gate's fail-closed `except` (console_chat_controller.py:6850-6866; gate landed aed1b13501). Remedy: module `pytestmark = pytest.mark.bootstrap_profile`, the in-base precedent for the identical signature (72a80e0b64, test_console_first_send_atomicity.py). Post-fix: 5 passed. Mutation control: changing the expected `execute_calls` effective-state to a sentinel FAILED the test, proving the permission-path assertion genuinely executes under the marker. |
| `test_console_agent_swap.py::test_mcp_tool_call_ask_state_routes_through_review_hook_and_approves` | (b) same signature → FIXED | green post-fix |
| `test_console_agent_swap.py::test_mcp_tool_call_session_approval_suppresses_card_on_next_turn` | (b) same → FIXED | green post-fix |
| `test_console_agent_swap.py::test_mcp_tool_call_ask_state_times_out_denies` | (b) same → FIXED | green post-fix |
| `test_console_agent_swap.py::test_mcp_tool_call_gates_subagent_call_same_as_primary` | (b) same → FIXED | green post-fix |
| `test_console_ephemeral.py::test_promotion_restores_ephemeral_flag_if_persist_returns_none_unexpectedly` | (a) fixed upstream, then superseded | Its original defect (stub missing the `strict_roleplay_context` kwarg → TypeError one frame before the covered branch) was repaired in `2b38e7533f` (2026-08-11, one day after filing). The test itself was later DELETED by `25ea9a1d83` (2026-08-22, "require atomic temporary promotion"), which reworked the promotion rollback these tests stubbed. File today: `python -m pytest Tests/Chat/test_console_ephemeral.py` → 18 passed. |
| `test_tool_output_disclosure.py::test_full_tool_output_is_reachable_from_the_mounted_transcript` | (a)+(b): original defect fixed upstream; red AGAIN at base with a NEW cause → FIXED | The trio's 2026-08 defect (ConsoleTranscriptMessage became a Vertical; `row.render()` Blank) was repaired in `2b38e7533f`. At this base they are red again because `ConsoleTranscript.compose` → `_turn_file_cards_enabled()` → `get_cli_setting("console", ...)` (console_transcript.py:4889) trips the same ADR-126 admission during the mounted harness's compose. Remedy: per-node `@pytest.mark.bootstrap_profile` on the three mounted tests only (TASK-32873 per-node pattern for mixed suites; the file's seven unit tests stay sandboxed). Post-fix: 10 passed. Mutation control: an impossible expansion expectation FAILED the test — assertions execute. |
| `test_tool_output_disclosure.py::test_pressing_o_expands_the_selected_marker` | (a)+(b) same → FIXED | green post-fix |
| `test_tool_output_disclosure.py::test_two_calls_in_one_turn_expand_independently` | (a)+(b) same → FIXED | green post-fix |

### AC#4 sweep (css_class_coverage_contract)

- The named node's 2026-08 assertion failure (stale registry flagging
  console_transcript.py) was already repaired in `2b38e7533f`.
- At this base the whole file ERRORs at setup instead:
  `Tests/UI/conftest.py`'s autouse `_disable_model_catalog_refresh` is the
  first import of `tldw_chatbook.app`, whose module body runs
  `load_settings()` under the per-test profile → the same
  `RecoveryRequired` admission class ("any Tests/UI module that doesn't
  import the app", TASK-33621.36). Fixed with the documented collection-time
  import remedy (`import tldw_chatbook.app` at module scope, as in
  test_install_command_clipboard.py / the TASK-32954 lesson).
- Surfaced and fixed: `test_registry_entries_are_still_unstyled` was red for
  ONE stale entry — `console-inspector-outer-scroll-hint` gained a real
  `.class` rule (css/features/_console_panels.tcss:1642) — deleted per the
  contract's own designed remedy.
- REMAINING red, NOT fixed here (needs its own ADR-150 task):
  `test_every_composed_class_is_styled_or_registered` fails with 27
  unstyled composed tokens (hook-review-* from the expanded-hooks merge at
  this base, library-details-*, console-voice-preview-*, console-settings-*,
  cancel/confirm-button, ...). Registering or styling 27 tokens requires
  per-token design decisions this baseline task must not make blind.

### Surfed defects NOT owned by this task (follow-ups for the owner)

1. `test_console_agent_swap.py::test_stop_cancels_tree_and_persists_cancelled`
   and `::test_stop_before_first_token_persists_cancelled_no_agent_run_failed`
   fail DETERMINISTICALLY (twice each, plus under the plain command) with
   `primary == []` / no run rows. Probed cause: both tests simulate Stop by
   calling `controller._signal_stop(...)` from inside the worker context
   (run_reply / gateway stream); since `63c9d906bc` the prompt-queue
   registry enforces owner-thread access (`QueueThreadViolation`,
   console_prompt_queue.py:363), and `_signal_stop` touches
   `prompt_queue_coordinator.pause_for_stop/interrupt_parent/cancel_pending_stop`,
   so the violation is raised and swallowed into "Agent run failed:
   unexpected provider error" with zero AgentRunsDB rows. Identical failure
   with and without this task's marker (A/B verified via
   `git checkout HEAD --` swap) — pre-existing, unmasked red, untracked
   anywhere in backlog. Fix needs a deliberate choice: re-simulate via the
   run's own `_active_cancel_events[session_id].set()`, or marshal queue
   access inside `_signal_stop`.
2. The 27-token CSS drift above.

### Verification (exact commands and results)

- Premise (base, no fixes): `python -m pytest <the 8 still-existing named
  nodes> -p no:cacheprovider -q -p no:xdist` → **8 failed** (5x "Hooks
  unavailable" send refusals, 3x compose-time RecoveryRequired/NoMatches);
  the 9th node no longer exists (deleted by 25ea9a1d83).
- Whole-file baseline: `python -m pytest Tests/Chat/test_console_agent_swap.py
  -p no:xdist` → **39 failed / 7 passed** (224.75s).
- Post-fix, plain command with default addopts, two consecutive runs of
  `python -m pytest Tests/Chat/test_console_agent_swap.py
  Tests/Chat/test_tool_output_disclosure.py` → **54 passed / 2 failed** both
  times (only the two documented stop-path defects above; also 4x faster —
  55-64s vs 225s — because sandbox DB churn is gone).
- `python -m pytest Tests/UI/test_css_class_coverage_contract.py` →
  **3 passed / 1 failed** (only the 27-token drift guard).
- `python -m pytest Tests/Chat/test_console_ephemeral.py` → 18 passed.

### Files changed

- `Tests/Chat/test_console_agent_swap.py` — module
  `pytestmark = pytest.mark.bootstrap_profile` with rationale comment.
- `Tests/Chat/test_tool_output_disclosure.py` — per-node
  `bootstrap_profile` markers on the three mounted tests, with rationale
  comment; unit tests untouched.
- `Tests/UI/test_css_class_coverage_contract.py` — collection-time
  `import tldw_chatbook.app`; one stale KNOWN_UNSTYLED entry deleted
  (`console-inspector-outer-scroll-hint`) with an inline dated note.
- This task file.

ADR required: no
Reason: test-only repair following the existing ADR-126/ADR-179 enrollment
pattern already established in Tests/conftest.py (TASK-32873) and the
in-base precedent 72a80e0b64; no product code was changed and no new
architectural decision was made.
