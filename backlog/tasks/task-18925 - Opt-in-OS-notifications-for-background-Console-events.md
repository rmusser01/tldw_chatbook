---
id: TASK-18925
title: Opt-in OS notifications for background Console events
status: In Progress
assignee:
- '@Robert'
created_date: 2026-08-19 09:55
updated_date: 2026-10-31 17:37
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
1. Utils/os_notifications.py dispatcher: pure build_notification_argv(platform, kind, session_display) (fixed argv, no shell) + send_os_notification with scrubbed env, 3s timeout, start_new_session, all failures -> debug log; macOS osascript, Linux notify-send, Windows honest skip
2. Config: nested [console] os_notifications.* (enabled=false master default OFF pinned in tests; run_finished/run_failed/needs_approval=true per-type), modeled on background_effects; dotted reads via get_cli_setting
3. Hook the three once-guarded sites in chat_screen.py: _notify_console_run_outcome, _notify_console_run_failure (enrich title via _console_session_title_and_workspace_name), _park_console_approval; fire off the UI thread; body = session/tab name + outcome category only
4. Settings Console Behavior: OS notifications group (master + 3 StateCheckbox rows) via background_effects dotted-draft staging; update CONSOLE_BEHAVIOR_* frozensets/save order/ownership/search index/field rows; grammar-test compliance
5. Tests: argv construction + no-content-leak + absent-binary tolerance + default-off; hook gating + dedupe via existing seams; settings round-trip + grammar
6. Docs: Docs/User_Guide/settings.md Console Behavior + Docs/Architecture/notifications.md

2026-10-31 current-base review repair:
1. Add failing regressions for Linux literal markup, private nonzero/exception diagnostics, and a real slow child during orderly interpreter shutdown.
2. Keep the existing fixed-argv/once-guarded dispatch; use a non-daemon one-shot, escape the Linux body, and log category plus return code or exception type only.
3. Run targeted notifier/Console/Settings/config tests; review diagnostics, preserve the UI census prefix, and measure/update size pins under the user's explicit waiver.
4. Record verification and native-platform limits, then commit for an independent final review.

ADR required: yes
ADR path: backlog/decisions/282-console-os-notification-dispatch.md
Reason: Records the external platform dispatch, privacy, inbox separation, and subprocess lifetime boundaries.
Plan: [2026-10-31 review repair](../../Docs/superpowers/plans/2026-10-31-task-18925-os-notifications-review-repair.md).
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
Implemented an optional default-off OS channel at the Console's existing once-guarded completion/failure/approval bridges, with canonical Settings master/per-type controls and nested config persistence. macOS uses fixed-argv osascript; Linux uses notify-send; Windows/other unsupported platforms skip.

2026-10-31 review repairs retain a non-daemon one-shot timeout owner through orderly interpreter exit, escape Linux body XML while keeping macOS quoting, and return False/log metadata only for nonzero exits and exceptions. Notifications carry bounded conversation/workspace identity and fixed category, never error copy or tool output. Exit may wait for an admitted notifier's remaining timeout; no UI-loop wait or new service was added.

ADR required: yes. [ADR-282](../decisions/282-console-os-notification-dispatch.md) records the platform, privacy, lifetime, and separate-inbox decision; it supersedes the original no-ADR disposition. Architecture docs, the ADR index, and the reproduced timeout-owner lesson are updated.

Verification: five intended RED regressions, then 28 dispatcher tests green; Settings persistence and both defaults tests passed; 8 Settings grammar tests and 6 exact size-pin checks passed. New modules pass Ruff/format checks; changed legacy files add no Ruff findings. The user's explicit size waiver pins ChatScreen at 24424 lines/739 methods after the thread-start follow-up, Settings at 35758 lines, and config at 12245 lines. The initial diagnostic review/regeneration added only the notifier's three metadata-only debug statements, with no other inventory or sink changes.

Independent review follow-up at dbbfcac reproduced thread-start RuntimeError escaping before the viewed-session error toast. The spawn helper now catches only RuntimeError and logs fixed failure metadata, with no synchronous fallback. The new unmounted regression calls the real viewed-failure bridge, gate, display sanitizer and spawn helper: RED failed at thread startup (1 failed, 2.98 s), then the full dispatcher file passed (29 passed, 11.21 s), including the real slow-child shutdown case. The sender never runs on startup failure and sensitive exception, session/workspace, and error text stay out of logs. No third mounted Console run was performed; task stays In Progress. The one new ChatScreen diagnostic is reviewed and the exact size pin remeasured under the same waiver.

Open qualification: the combined run finished 35 passed/2 failed and a serial Console-file replay 4 passed/2 failed, in different nodes. Each failing child passed its notification assertions and errored during native-app-worker/config retirement teardown; all those tests replace the production spawn helper and sender. Cause is not independently established. The task remains In Progress for independent review. The census preserves its 1173-entry prefix and adds only the green Settings journey (1174 total); Console remains explicitly unqualified, not removed from an existing contract. No third suite, app-cleanup changes, full sweep, CI run, or native macOS/Linux desktop verification was performed.

Files: Utils/os_notifications.py; UI/Screens/chat_screen.py and canonical Settings/search; config defaults; notifier/Console/Settings/default tests; size pins; diagnostic inventory; census; architecture/ADR/lesson docs. Full receipts, node IDs, limitations and commands: [review evidence](../../Docs/superpowers/qa/2026-10-31-task-18925-os-notifications-review.md).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
