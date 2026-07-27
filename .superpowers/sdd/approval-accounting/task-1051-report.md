# TASK-1051: `TaskResumeState.from_dict` drops `pending_skill_script`

## The question

`from_dict` (`tldw_chatbook/UI/Screens/chat_screen_state.py`) restores
`pending_skill_install` from a persisted Console snapshot but hardcodes
`pending_skill_script=None`. Two AC branches were allowed: restore both
symmetrically, or document the asymmetry as deliberate. The brief required
investigating which is correct via the call chain, not picking blindly.

## Call-chain investigation

**1. The round-trip boundary is a tab switch, never an app restart.**
`TaskResumeState.to_dict()`/`from_dict()` are only exercised through
`ChatScreen.save_state`/`restore_state` (chat_screen.py:10778-10793), which
`app.py`'s `handle_screen_navigation` calls on every `NavigateToScreen`
(app.py:6243-6291) — i.e. every tab switch. The backing store,
`UI/Navigation/screen_state_store.py`'s `ScreenStateStore`, documents itself
as "memory-only ownership for cross-visit screen snapshots" and never
touches disk. So the description's hedge ("does a script-confirm round
legitimately survive a restart/restore") was answered: there is no
restart/restore path here at all, only cross-navigation.

**2. Even a tab switch does not let a round survive.**
`ChatScreen._create_navigation_screen`'s own docstring (chat_screen.py:
6119-6141) states screens are "never cached and re-mounted" — every
navigation builds a **brand-new** `ChatScreen` instance. Its
`_console_chat_controller` starts as `None` (chat_screen.py:2124) and is
lazily rebuilt from scratch by `_ensure_console_chat_controller`
(chat_screen.py:3611-3677), constructing a **new** `ConsoleChatController`
whose `_pending_skill_install_rounds`/`_pending_skill_script_rounds` dicts
start empty (console_chat_controller.py:860, 888). A confirm round is
*only* an entry in one of those dicts on the OLD controller instance,
guarding a worker thread blocked on a `threading.Event`
(`request_skill_script_confirm`/`request_skill_install_confirm`,
console_chat_controller.py:2619-2853). That entry — and the blocked thread
— is gone the instant the screen is recreated. There is no reconnection
path, in-process or otherwise.

**3. What a symmetric restore would actually do.**
If `from_dict` restored `pending_skill_script`, `ChatTaskCards.sync_state`
(chat_task_cards.py:54-55) would mount a fully rendered, apparently-live
`SkillScriptConfirmCard`. A user's Allow/Deny click reaches
`ConsoleChatController.resolve_pending_skill_script`
(console_chat_controller.py:3022+), which strict-matches `request_id`
against the *currently-armed* round and **silently drops** any resolve that
doesn't match (console_chat_controller.py:3054-3070 area, fail-closed by
design). A restored round can never match a live round (there is none), so
the card would render as a normal, clickable confirm whose buttons do
nothing, forever, with no error. That is a **worse** failure mode than
today's — today the card simply never mounts, which is honest.

**4. `pending_skill_install` has the identical hazard, but is out of scope.**
`_pending_skill_install_rounds` and `resolve_pending_skill_install`
(console_chat_controller.py:2809-2852) have byte-identical architecture to
the script bridge, and are exposed to the exact same dead-card risk on
restore. It IS restored by `from_dict` today, and that round-trip is pinned
by an existing test from the TASK-910 lineage
(`Tests/UI/test_console_skill_install_confirm.py::
test_task_resume_state_pending_skill_install_roundtrip`). This is a real,
pre-existing asymmetry — but TASK-910 added install's round-keyed restore
*before* this dead-UI hazard was identified for script. The script side's
current drop-on-restore behavior was added deliberately, WITH a regression
test already proving the "no dead card" contract
(`Tests/UI/test_skill_script_confirm_card.py::
test_restored_state_drops_the_pending_script_so_no_dead_card_appears`,
committed 2026-07-25, predates this task). TASK-1051's AC only concerns
`pending_skill_install`/`pending_skill_script` symmetry — changing
`pending_skill_install`'s own, separately-tested contract would exceed
scope and break an unrelated passing test. Flagged as a follow-up
candidate, not touched.

## Decision: document the asymmetry as deliberate (AC option 2)

The dataclass field itself is **not** dead: `pending_skill_script` is the
live carrier `ChatScreen._set_console_pending_skill_script`
(chat_screen.py:15940-15953) mutates directly whenever a real, in-session
round is armed, and `ChatTaskCards.sync_state` reads it to render the card
during that live window — that path never goes through `from_dict`. Only
the *restore* direction is dead, and that was already correctly hardcoded
to `None`, with a regression test pinning it. Interpreting the AC's "field
removed from the dataclass/serialization entirely" literally (deleting the
field) would delete real, load-bearing, live functionality — not a
defensible reading once the live-carrier role is accounted for.

**What changed** (`tldw_chatbook/UI/Screens/chat_screen_state.py`, docs
only, no behavior change — `to_dict`/`from_dict` outputs are byte-identical
before and after):
- A field-level comment on `pending_skill_script` pointing at `from_dict`'s
  docstring.
- `from_dict`'s docstring expanded from one line to the full call-chain
  argument above: why the round can't survive (tab-switch → fresh
  screen/controller → empty round dicts → strict `request_id` match), why
  restoring it would be strictly worse than not, and why
  `pending_skill_install`'s identical hazard is a known, separately-scoped,
  already-tested asymmetry rather than an oversight fixed here.

## AC #2 — regression test

Already satisfied by pre-existing coverage; no new test needed since no
behavior changed. `Tests/UI/test_skill_script_confirm_card.py::
test_restored_state_drops_the_pending_script_so_no_dead_card_appears`
round-trips a `TaskResumeState.to_dict()` snapshot carrying a populated
`pending_skill_script` payload through `from_dict` and asserts it comes
back `None`, while a sibling `pending_skill_install` payload and `summary`
survive. `Tests/ProductionApp/test_chat_root_state_removal.py::
test_task_resume_state_rejects_malformed_snapshot_fields` additionally
covers a malformed/well-formed-looking `pending_skill_script` dict
(`{"request_id": "cannot-resume"}`) also coming back `None`.

## Test results

Gate: `.venv/bin/python -m pytest Tests/UI/test_console_workbench_contract.py
Tests/UI/test_skill_script_confirm_card.py
Tests/UI/test_console_skill_install_confirm.py
Tests/ProductionApp/test_chat_root_state_removal.py
Tests/UI/test_console_mcp_approval.py` → **133 passed, 4 failed**.

Two failures match the task's documented pre-existing baseline exactly:
`test_batch_row_widgets_have_nonzero_geometry_and_do_not_overlap_under_bundled_css`
(CSS-geometry batch-row) and
`test_request_mcp_approvals_cancellation_records_denied_decision_to_execution_log`
(MCP cancellation execution-log).

Two more were *not* pre-flagged, so independently verified pre-existing by
`git stash`-ing this task's edit and re-running both against HEAD
`f77132c87` before any change: `test_console_empty_transcript_choose_model_
opens_settings` (`InvalidSelectValueError` in a Settings `Select` widget,
unrelated) and `test_set_console_pending_skill_script_preserves_other_
resume_fields` (`AttributeError: 'ChatScreen' object has no attribute
'chat_state'` — a stale attribute that does not exist anywhere in
`chat_screen.py`; the screen exposes `_task_resume_state` directly, not a
`.chat_state` wrapper). Both reproduce identically with or without this
task's diff — confirmed unrelated to `TaskResumeState`/`from_dict`.

## Files changed

- `tldw_chatbook/UI/Screens/chat_screen_state.py` — comment on
  `pending_skill_script` field + expanded `from_dict` docstring. No
  functional change.
- `backlog/tasks/task-1051 - ....md` — AC#1/AC#2 checked, Implementation
  Plan + Notes added, status → Done.
