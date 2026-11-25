---
id: TASK-41947
title: "`request_chat_create_confirm` waits in its own loop, outside the no-running-app round guard"
status: Done
assignee: []
created_date: '2026-11-19 07:30'
updated_date: '2026-11-25 15:40'
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
2. `request_chat_create_confirm`: when the same predicate says the App cannot answer, refuse in the existing "no wired UI" early return, before any round, card, park toast or badge (review round 1 replaced per-branch guards with this).
3. Tests in `Tests/Chat/test_console_chat_create_confirm.py`: never-run and exited App, parked, active-session and legacy rounds, settle refused promptly without touching the App, the park toast or the badge (red on dev); a running App's round still waits until answered.
4. Rewrite AC #2 (stale since TASK-41948), amend ADR-314, keep the module under its size-ratchet row.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Fix.** `request_chat_create_confirm` now applies `run_round`'s no-running-app rule.
- The predicate moved out of `run_round` into one helper, `_app_cannot_answer`, with the same `try: not app.is_running / except AttributeError: False` body. It deliberately avoids a literal `getattr(app, "is_running", ...)`, which `test_app_string_lookup_census` counts (TASK-41637 removed one for that census).
- The chat-create bridge calls the helper in its existing "no wired UI" early return. With no running App it is refused before a round is armed: no card, park toast, badge or poll. It returns `{"allow": False, "remember": False}`, and a prepared creation token is closed.
- A remembered session grant still applies, since it needs no round. A host without `is_running` (mobile context, `_FakeApp` doubles) still waits.
- On dev 2959b48314 such a round waited forever (ADR-314 arms no deadline) or raised out of `call_from_thread`.

**Outcome wording.** Unlike `run_round`, the chat-create bridge has no unresolved outcome. `build_chat_create_tool_closures` reports every non-Allow result, including timeout and cancel, as "The user declined. Do not retry this turn." and counts a denial. The ADR-314 amendment says exactly that (review round 1).

**AC #2 rewritten.** It read "One with an answerable view still waits for the answer". TASK-41948 (b4afcd9493) dropped the view condition from `run_round`: only a never-run or exited App fails a round closed. The criterion now reads "a chat-create confirmation with a running app still waits for its answer".

**Tests** (`Tests/Chat/test_console_chat_create_confirm.py`):
- `test_chat_create_confirm_without_a_running_app_is_refused_promptly`, over a never-run and an exited App, each with a parked, an active-session and a legacy `session_id=None` round. Each settles refused within 5 s, leaves no round, and marshals nothing to the App. The App is a `_RecordingApp` that records and drops `call_from_thread`, so a stray marshal fails an assertion instead of raising inside Textual. Nothing is passed to the park toast (wired to a recorder) or to `add_pending_round`.
- `test_chat_create_confirm_with_a_running_app_waits_for_its_answer`: a registered round on a running Textual App is still waiting and returns the user's Allow. Registration happens only after the check, so the test needs no sleep.

**Evidence.**
- Red on dev source: all 6 no-app nodes still waiting at 5 s.
- Negative controls on the final source:
  - early return removed: the same 6 still waiting;
  - early return removed but the loop breaks at the first poll (the previous revision's shape without its UI guards): all 6 fail on the recorded marshals or badge;
  - helper always reports "gone": the running-App node fails.

**Runs** (after rebasing onto dev aecdc35670; load 23-55 on 18 CPUs).
- The test file whole: 28 passed serially and at `-n 4` (21 on dev).
- `Tests/Architecture` serially: 1507 passed, 5 skipped. This includes `test_app_string_lookup_census`, which counted the first revision's `getattr(app, "is_running", True)` lookup (174 against a pin of 173), and the size ratchet. `console_interrupt_rounds.py` measures 6,300 against its 6,301 row.
- `ruff check tldw_chatbook --select F` is clean, `Tests/lint/ruff_gate.py` exits 0, and `./scripts/preflight.sh` passed under a scratch HOME.
- First revision, at `-n 4`: the related suites (`test_console_runtime_lifetime.py`, `test_console_chat_create_integration.py`, `test_console_chat_start.py`, `test_console_pending_interrupt_projection.py`, `test_console_skill_script_confirm.py`) gave 370 passed, 5 failed.
  - The 4 `test_console_chat_start.py` nodes (`test_native_start_child_and_wake_share_original_allowance[*]`, `test_native_configured_v2_initialization_and_stop_proposal[*]`, `OriginalSourceChanged`) fail identically on dev's source.
  - `test_session_owned_pending_projection_journeys` (a skill-install focus assertion) failed once under load and passed twice serially.

Files: `tldw_chatbook/Chat/console_interrupt_rounds.py`, `Tests/Chat/test_console_chat_create_confirm.py`, `backlog/decisions/314-indefinite-human-approval-waits.md` (one amendment paragraph).
<!-- SECTION:NOTES:END -->
