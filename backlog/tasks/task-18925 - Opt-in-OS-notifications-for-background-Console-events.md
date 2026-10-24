---
id: TASK-18925
title: Opt-in OS notifications for background Console events
status: Done
assignee:
- '@Robert'
created_date: 2026-08-19 09:55
updated_date: 2026-10-18 16:04
labels:
- console
- agents
- notifications
- ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Port of hermes-agent's native OS notifications with per-type toggles (2026-08-19 hermes-release review). Console already toasts in-app for background completions, failures, parked approvals, and auto-wakes — but if the terminal window is buried, the user misses them. Fire a desktop notification (macOS via osascript / Linux via notify-send; Windows evaluated and honestly reported) for "run finished", "run failed", and "needs approval", gated by a new opt-in setting (default OFF) with per-type toggles. Notifications carry the conversation/tab name and outcome category only — never message content, never credentials. Cadence must match the existing one-toast-per-event dedupe semantics so this adds a channel, not spam.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 New setting (default OFF) with per-type toggles for run finished / run failed / needs approval; off-by-default is pinned in tests
- [x] #2 macOS and Linux implementations work; unsupported platforms or missing binaries fail silently with a debug log and never crash or block the app
- [x] #3 Notification body includes only conversation/tab name + outcome category; no message text, no tool output, no secrets — verified by tests over the constructed command
- [x] #4 Notification cadence dedupes identically to the in-app toasts (one per background completion / approval park; multi-prompt drain reports once)
- [x] #5 The notification subprocess invocation is time-bounded and failure-tolerant, and never runs on the UI/event-loop thread in a blocking way
- [x] #6 Tests cover setting gating, per-type toggles, command construction (no content leakage), and failure tolerance; user guide documents the setting
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Utils/os_notifications.py dispatcher: pure build_notification_argv(platform, kind, title, body) (fixed argv, no shell) + send_os_notification with scrubbed env, 3s timeout, start_new_session, all failures -> debug log; macOS osascript, Linux notify-send, Windows honest skip
2. Config: nested [console] os_notifications.* (enabled=false master default OFF pinned in tests; run_finished/run_failed/needs_approval=true per-type), modeled on background_effects; dotted reads via get_cli_setting
3. Hook the three once-guarded sites in chat_screen.py: _notify_console_run_outcome, _notify_console_run_failure (enrich title via _console_session_title_and_workspace_name), _park_console_approval; fire off the UI thread; body = session/tab name + outcome category only
4. Settings Console Behavior: OS notifications group (master + 3 StateCheckbox rows) via background_effects dotted-draft staging; update CONSOLE_BEHAVIOR_* frozensets/save order/ownership/search index/field rows; grammar-test compliance
5. Tests: argv construction + no-content-leak + absent-binary tolerance + default-off; hook gating + dedupe via existing seams; settings round-trip + grammar
6. Docs: Docs/User_Guide/settings.md Console Behavior + console.md
<!-- SECTION:PLAN:END -->
## Deferral note (2026-10-11, dev `9f8533686c`)

DEFERRED from the t2-hermes porting slice for capacity + collision risk:
the settings-surface half lands in `settings_screen.py`'s Console Behavior
category (key frozensets `CONSOLE_BEHAVIOR_*`, save order, grammar tests in
`Tests/UI/test_settings_console_behavior_grammar.py`), and the model-config
programme (PR #3057, open) is actively touching the same settings test
family.

Seam map gathered while scoping (all verified in code at dev `9f8533686c`):

- **Hook sites (all three, with their dedupe already once-guarded)**:
  `chat_screen.py::_notify_console_run_outcome` (run finished/failed for a
  NON-viewed session — the copy already carries session title + workspace
  name + verb, exactly the AC#3 content budget); `chat_screen.py::
  _notify_console_run_failure` (viewed-session failure toast);
  `_park_console_approval` (approval park). Hooking these three adds the OS
  channel with the in-app toasts' dedupe semantics for free — they are the
  once-guarded bridges `ConsoleChatController.notify_run_outcome` /
  `notify_run_failure` already call on the main loop thread.
- **Dispatcher**: new module (e.g. `Utils/os_notifications.py`) — macOS
  `osascript -e display notification ...`, Linux `notify-send`; fixed argv,
  scrubbed env, subprocess with a hard timeout, never awaited on the UI
  thread (the plan's security review items). Windows: evaluate and report
  honestly per the plan.
- **Setting**: `[notifications]`-style config section default OFF with
  per-type toggles (run finished / run failed / needs approval), read via
  `config.get_cli_setting`; off-by-default pinned in tests per AC#1.
- **Tests**: command construction (no message content) can pin the exact
  argv without spawning (AC#3); failure tolerance by faking the binary
  absent (AC#2); the hook sites can be driven with the existing
  `_notify_console_run_outcome` seams.

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented on branch feat/18925-os-notifications (worktree .worktrees/tasks-18925-28-34, base dev b07fc04252).

Approach: pure dispatcher + config + three hook sites + Settings group, exactly per plan. Key decisions:
- Utils/os_notifications.py: build_notification_argv (fixed argv, no shell; macOS osascript with AppleScript escaping, Linux notify-send with -- guard; win32/other -> None = honest skip + debug log) + send_os_notification (settings gate, scrubbed minimal env = PATH + DBUS/XDG on linux only, 3s subprocess timeout, start_new_session, capture_output, swallows OSError/SubprocessError -> debug log). Category copy comes only from OS_NOTIFICATION_KIND_COPY — nothing else can enter the body.
- Config: nested [console] os_notifications table (master enabled=false default OFF pinned in Tests/test_config_console_defaults.py; run_finished/run_failed/needs_approval=true per-type). normalize_console_os_notifications mirrors Utils/console_background_effects.py (coerce_bool_flag).
- Hooks: fired from inside _notify_console_run_outcome / _notify_console_run_failure / _park_console_approval AFTER their existing once-guards, so OS cadence == toast cadence by construction (controller transition guard + queue-chain deferral + park round-id memory). Viewed-failure hook resolves the session title via _console_session_title_and_workspace_name instead of reusing visible_copy (error text never enters the OS body). Spawn = daemon one-shot thread (bounded by the 3s subprocess timeout; no Textual worker lifecycle churn) — never blocks the UI loop.
- Settings: Console Behavior 'OS notifications' group (4 StateCheckboxes) following the background_effects dotted-draft staging pattern; CONSOLE_OS_NOTIFICATION_KEYS/SAVE_ORDER + splice into CONSOLE_BEHAVIOR_SAVE_ORDER (required: dirty_values filters through it), post-mount sync with _syncing guard, save merges the nested table through normalize.
- Windows: evaluated -> not implemented, honest skip with debug log per plan.

Files: tldw_chatbook/Utils/os_notifications.py (new); config.py ([console.os_notifications] template); UI/Screens/chat_screen.py (3 hooks + _fire/_display seam + _spawn_os_notification_thread); UI/Screens/settings_screen.py (4 rows, staging/sync/save); Docs/User_Guide/settings.md (group row). Tests: Tests/Utils/test_os_notifications.py (22), Tests/UI/test_console_os_notifications.py (6, private-profile), Tests/UI/test_settings_console_os_notifications_journeys.py (1 journey), +2 defaults tests in Tests/test_config_console_defaults.py.

Verification: all new tests green; grammar test green; ruff clean on new files and no new findings on modified files. Pre-existing local-env failures (RecoveryRequired / network-dependent hub tests / background-effects painting tests) verified identical on pristine dev via stash. Docs: settings.md Console Behavior table. ADR: not required (per plan section).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
