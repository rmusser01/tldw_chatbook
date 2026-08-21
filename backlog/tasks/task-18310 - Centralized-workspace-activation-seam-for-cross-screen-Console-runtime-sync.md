---
id: TASK-18310
title: Centralized workspace activation seam for cross-screen Console runtime sync
status: Done
assignee: []
created_date: '2026-08-18 15:30'
updated_date: '2026-08-21 05:22'
labels:
  - workspaces
  - console
dependencies: []
priority: medium
---

## Description (the why)

Activating a workspace from OUTSIDE Console (Library's "Create local workspace"
— shipped long before the create modal — and now Settings via the shared modal,
PR #1809) updates only the registry (`set_active_workspace`) and toasts
"Console now targets it". Console's deeper runtime state (chat store context,
active session, native UI) is only synchronized by Console-internal paths
(`_activate_console_session_for_workspace` at switch/create/archive). The rail
reflects the registry on the next Console visit (live-verified in TASK-18704),
but no resume-time seam activates the matching session the way an in-Console
switch does. Qodo flagged this on PR #1809 (finding 5); it is a pre-existing
design gap shared by the Library path, not a regression of the modal work —
resolving it properly means one centralized activation seam that any screen can
invoke (or a Console resume-time reconcile), instead of duplicating Console's
fragile sequence per caller.

## Acceptance Criteria (the what)

- [x] Activating a workspace from Settings or Library leaves Console — on next visit — in the same runtime state as an in-Console switch to that workspace (session activated or created, store context set, rail synced)
- [x] The mechanism is a single shared seam (or resume-time reconcile), not per-surface copies of Console's activation sequence
- [x] In-Console switch/create behavior is unchanged (order-pinned tests stay green)
- [x] Covered by a test that activates from a non-Console surface and asserts the Console-side session state after resume

## Implementation Plan (the how)

1. Verify the controller ruling's facts: every IN-Console workspace-activation
   path (`_open_console_workspace_switcher`'s `_switch_to`,
   `_handle_workspace_create_result`, and the conversation-browser row-open
   path) calls `set_active_workspace` on the registry AND
   `_activate_console_session_for_workspace` on `ConsoleWorkspaceController`
   together — so registry/session drift can only originate from a
   cross-screen change (Settings' create-modal `_done`, Library's
   `create_local_workspace` `_done`, Settings' "Set active" button).
2. Add `ConsoleWorkspaceController._reconcile_console_session_with_registry`
   in `tldw_chatbook/UI/Console_Modules/workspace.py`: read the registry's
   active workspace (guarded), read the store's active session (guarded),
   cheap-exit when aligned or when either side is `None`, otherwise re-run
   the create-handler's own sequence (`_sync_console_chat_core_state` →
   `_activate_console_session_for_workspace` → `_sync_console_workspace_
   context` → `run_worker(_sync_native_console_chat_ui, ...)`) minus the
   registry write (already done) and the toast (the originating surface
   already announced).
3. Wire it into `ChatScreen.on_screen_resume`
   (`tldw_chatbook/UI/Screens/chat_screen.py`), wrapped in try/except with a
   debug log, placed after the mount-token consumption but NOT gated by
   `mount_already_refreshed` — the reconcile is O(1) when aligned, and the
   mount path itself never reconciles against the registry.
4. TDD: write `Tests/Workspaces/test_console_workspace_reconcile.py` first
   (RED — `AttributeError`), following the house stub pattern in
   `test_console_workspace_create_handler.py` (unbound method invoked on a
   `_Stub`); cover the cheap-aligned-exit, the cross-screen four-step
   sequence (order-pinned), no-active-session, no-registry/raising-registry,
   no-active-workspace, and an end-to-end-ish test against a real
   `ConsoleChatStore` + real registry that runs the real unbound
   `_activate_console_session_for_workspace` body. Implement, confirm GREEN.
5. Run the regression gate (`Tests/Workspaces/`, the three named
   `Tests/UI/` files, a full-suite `--collect-only`) plus an extra sweep of
   every `on_screen_resume`-touching UI test file for safety.
6. Update `Docs/User_Guide/console/sessions-tabs-workspaces.md` and
   `Docs/User_Guide/settings.md` with a one-sentence note on the new
   pickup-on-resume behavior, and refresh both pages' "Verified against"
   stamps.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the resume-time reconcile per the controller ruling, not a
push-based cross-screen seam. The invariant that makes this safe: every
IN-Console workspace-activation path (`_open_console_workspace_switcher`'s
`_switch_to`, `_handle_workspace_create_result`, and the
conversation-browser row-open path) already calls `set_active_workspace`
on the registry AND `_activate_console_session_for_workspace` on
`ConsoleWorkspaceController` together, in the same call — so those paths
can never leave the registry and the Console chat store's active session
disagreeing. Only the three cross-screen callers (Settings' create-modal
`_done`, Library's `create_local_workspace` `_done`, Settings' "Set
active" button — the Qodo finding 5 gap on PR #1809) touch only the
registry. A resume-time reconcile therefore fires exactly when needed
(after a cross-screen change) and is a true no-op otherwise, which is
both simpler and safer than adding a new push-based call from three
separate non-Console surfaces into Console's fragile activate sequence
(more call sites duplicating that sequence, exactly what the AC #2 "not
per-surface copies" line rules out).

Added `ConsoleWorkspaceController._reconcile_console_session_with_registry`
(`tldw_chatbook/UI/Console_Modules/workspace.py`): reads the registry's
active workspace and the store's active session, both guarded against
`None`/exceptions (must never break screen resume), cheap-exits when the
active session's `workspace_id` already matches, and otherwise re-runs the
create handler's own four-step sequence (`_sync_console_chat_core_state`
→ `_activate_console_session_for_workspace` → `_sync_console_workspace_
context` → `run_worker(_sync_native_console_chat_ui, ...)`) minus the
registry write (already done by the other surface) and the toast (that
surface already announced the switch). Wired into `ChatScreen.
on_screen_resume` (`tldw_chatbook/UI/Screens/chat_screen.py`), wrapped in
try/except with a debug log, placed right after the mount-token
consumption but deliberately NOT gated by `mount_already_refreshed` — the
reconcile is O(1) once aligned, and the mount path itself never
reconciles against the registry, so skipping it on the mount's own resume
would leave a store session that predates the first mount (the store is
app-level) permanently unreconciled.

TDD: `Tests/Workspaces/test_console_workspace_reconcile.py` was written
first and confirmed RED (`AttributeError: type object
'ConsoleWorkspaceController' has no attribute
'_reconcile_console_session_with_registry'`) before implementing, then
GREEN after. Covers: the cheap aligned no-op, the cross-screen four-step
sequence in ORDER (list-equality, matching the sibling create-handler
test's style), no-active-session, no-registry-service,
registry-raises-on-read, no-active-workspace, an AC#4 end-to-end-ish test
that runs the REAL unbound `_activate_console_session_for_workspace` body
against a real `ConsoleChatStore` + real `LocalWorkspaceRegistryService`
(a plain registry `set_active_workspace` call simulating a non-Console
surface, then the reconcile invoked as resume would) asserting the
store's active session now belongs to the new workspace, and a regression
test (below) — 8 tests total.

**Bug caught by the extra sweep, not the specified gate.** The first pass
compared `active_session.workspace_id == active.workspace_id` raw. That
misses an established invariant already encoded elsewhere in this same
file: a session's default `workspace_id` (unset, or the explicit
`CONSOLE_GLOBAL_WORKSPACE_ID` sentinel `"global"`) and the registry's
built-in Default workspace row (`DEFAULT_WORKSPACE_ID`,
`"workspace-default"`) are THE SAME state on two layers (task-15120 owner
ruling, see `_set_active_workspace_for_console_session`), not two
different workspaces. The raw comparison read every ordinary
global/unset-workspace mounted session as diverged the instant the
registry's active workspace was its normal resting Default row, tearing
the session down and rebuilding a fresh "Default Chat" session on every
resume. Two tests outside the specified gate caught it going from GREEN
to RED (`Tests/UI/test_console_session_settings.py::
test_mounted_first_chat_ack_exception_during_resume_restores_ui` and
`::test_mounted_console_unmount_times_out_hung_refresh_and_repairs_on_resume`,
both of which call `on_screen_resume` on a mounted session with the
default sentinel workspace_id). Fixed with a module-level
`_normalized_console_workspace_id` helper that folds both sentinels onto
`DEFAULT_WORKSPACE_ID` before comparing, and pinned with a new regression
test, `test_global_session_aligned_with_registry_default_is_a_noop`
(8th test in the file) — mutation-style: reverting the normalization
turns it red again, confirmed manually. This is exactly the kind of trap
`backlog/docs/lessons-testing-evidence.md` warns about (a guard that
looks right until it meets a live fixture) — the specified gate
(`Tests/Workspaces/`, the 3 named `Tests/UI/` files) never exercised a
mounted session with an unset workspace_id, only ones with explicit
non-default ids; the broader sweep is what surfaced it.

**Pre-existing, unrelated flake found during triage.**
`test_mounted_console_unmount_times_out_hung_refresh_and_repairs_on_resume`
still fails intermittently after the fix, but on a DIFFERENT assertion
each run (once a `elapsed < 0.5` timing check, once a
`weakref` GC-liveness check) — and reproduces identically, byte-for-byte,
against a throwaway worktree of unmodified `origin/dev` (no TASK-18310
code present at all). Confirmed NOT a regression from this change; left
untouched per the ACs (fixing it is out of scope here) and not filed as a
new task since it needs its own investigation session, not a drive-by
note.

Regression gate, all green: `Tests/Workspaces/ -q` (273 passed, includes
the order-pinned create-handler tests for AC#3 and the new regression
test); `Tests/UI/test_console_workspace_lifecycle.py
Tests/UI/test_console_workspace_controller.py
Tests/UI/test_console_new_workspace.py -q` (35 passed);
`Tests/ --collect-only -q` (52236 tests collected, no collection errors).
Additionally swept every UI test file that exercises `on_screen_resume`
directly (`test_console_visit_dispatch_dedupe.py`,
`test_settings_panel_scoped_updates.py`, `test_console_workbench_
contract.py`, `test_console_session_settings.py`,
`test_settings_workspaces_category.py`,
`test_settings_configuration_hub.py`, `test_console_command_composer.py`,
`test_console_native_chat_flow.py`, `test_console_internals_
decomposition.py`) as extra insurance beyond the specified gate, since the
new call sits directly in that method — this second pass is what caught
the normalization bug above; after the fix only the one pre-existing,
dev-reproducible flake remains.

Files touched: `tldw_chatbook/UI/Console_Modules/workspace.py`,
`tldw_chatbook/UI/Screens/chat_screen.py`,
`Tests/Workspaces/test_console_workspace_reconcile.py` (new),
`Docs/User_Guide/console/sessions-tabs-workspaces.md`,
`Docs/User_Guide/settings.md`.
<!-- SECTION:NOTES:END -->
