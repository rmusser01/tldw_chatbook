---
id: TASK-41947
title: "`request_chat_create_confirm` waits in its own loop, outside the no-running-app round guard"
status: Done
assignee: []
created_date: '2026-11-19 07:30'
updated_date: '2026-11-25 14:05'
labels:
- follow-up
- console
- bug-suspected
priority: medium
references:
- tldw_chatbook/Chat/console_interrupt_rounds.py
dependencies:
- TASK-41637
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41630..41655 follow-up burn-down (2026-11-18/19). Checked on dev d0486978dd unless stated; paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-41637 made an approval round fail closed when no app is running and no view can answer it (`run_round`). `request_chat_create_confirm` has its own wait loop outside `run_round`, so a chat-create confirmation with no running app may still wait forever.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A chat-create confirmation requested with no running app and no answerable view settles as refused promptly (test, red first)
- [x] #2 A chat-create confirmation with a running app still waits for its answer
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (applies ADR-314's existing TASK-41637/41948 rule to the one round kind that waits outside `run_round`; the ADR gets a one-paragraph amendment naming it)

1. Move `run_round`'s inline no-running-app predicate into one module helper in `Chat/console_interrupt_rounds.py` and use it from `run_round` unchanged.
2. `request_chat_create_confirm`: read the same predicate at arm time; when the App cannot answer, show no card or park toast, settle refused at the first poll, and skip the teardown remount.
3. Tests in `Tests/Chat/test_console_chat_create_confirm.py`: never-run and exited App, parked and mounted rounds, settle refused promptly (red on dev); a running App's round still waits until answered.
4. Rewrite AC #2 (stale since TASK-41948), amend ADR-314, keep the module under its size-ratchet row.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Fix.** `request_chat_create_confirm` now applies `run_round`'s no-running-app rule. The predicate moved out of `run_round` into one helper, `_app_cannot_answer` (`not getattr(app, "is_running", True)`, the same semantics as the old `try`/`except AttributeError`), which both use. When it holds at arm time, the chat-create round shows no card and no park toast, settles refused at its first poll and skips the teardown remount. A host without `is_running` (mobile context, `_FakeApp` doubles) still waits. On dev 2959b48314 a parked round with no running App waited forever (ADR-314 arms no deadline), and a mounted one raised `RuntimeError` out of `call_from_thread` ("App is not running" / "Event loop is closed").

**AC #2 rewritten.** It read "One with an answerable view still waits for the answer". TASK-41948 (b4afcd9493) dropped the view condition from `run_round`: only a never-run or exited App fails a round closed, whatever the view says. The criterion now reads "a chat-create confirmation with a running app still waits for its answer", the rule this round kind now shares.

**Tests** (`Tests/Chat/test_console_chat_create_confirm.py`):
- `test_chat_create_confirm_without_a_running_app_is_refused_promptly[never-run|exited App x parked|mounted]`: settles `{"allow": False, "remember": False}` within the file's 5 s bound, no card marshalled, no round left.
- `test_chat_create_confirm_with_a_running_app_waits_for_its_answer`: a running Textual App's round is still waiting after 2.5 polls and returns the user's Allow.

**Evidence.** Red on dev source: the 4 no-app nodes failed (parked: still waiting; mounted: `KeyError` after the worker raised). The running-app node passed on dev, as intended. Negative controls on the final source, one at a time: dropping `app_gone or` from the loop failed all 4 no-app nodes; mounting the card despite `app_gone` failed all 4; a helper that always says "gone" failed the running-app node.

**Runs** (load 30-55 on 18 CPUs). The test file whole: 26 passed serially and at `-n 4` (21 on dev before). Related suites at `-n 4`, run on the first revision of this change (the final one only compacts it for the size row; the file above and the projection node were rerun on the final source) (`test_console_runtime_lifetime.py`, `test_console_chat_create_integration.py`, `test_console_chat_start.py`, `test_console_pending_interrupt_projection.py`, `test_console_skill_script_confirm.py`): 370 passed, 5 failed. The 4 `test_console_chat_start.py` nodes (`test_native_start_child_and_wake_share_original_allowance[*]`, `test_native_configured_v2_initialization_and_stop_proposal[*]`, `OriginalSourceChanged`) fail identically on dev's source. `test_session_owned_pending_projection_journeys` failed once (a skill-install focus assertion) and passed twice serially on this branch.

**Size.** `console_interrupt_rounds.py` measures 6,298, under its 6,301 ratchet row (net -2).

Files: `tldw_chatbook/Chat/console_interrupt_rounds.py`, `Tests/Chat/test_console_chat_create_confirm.py`, `backlog/decisions/314-indefinite-human-approval-waits.md` (one amendment paragraph).
<!-- SECTION:NOTES:END -->
