---
id: TASK-21118
title: >-
  Console keystroke path does synchronous Workspace-DB reads - measured ~1.25 per key - memoize the workspace context
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - console
  - database
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21118).

Live counter: 20 printable keys in the configured composer -> 25 x `ensure_default_workspace` +
25 x `get_active_workspace` (LocalWorkspaceRegistryService), i.e. ~1.25 synchronous SQLite
round-trips per keystroke on the UI thread (chain: DraftChanged ->
`_build_console_control_state` -> `_current_console_workspace_context`). 62 us/call measured on
a warm fast SSD - the risk cases are cold page cache, slow disks, and the repair branch's
DELETE write. During staged live-work launches, `EvidenceBundle.from_payload` is additionally
re-parsed >=2x per keystroke.

## Acceptance Criteria

- [ ] The workspace context is memoized on the screen and invalidated by workspace-change events (activation, registry mutation); the keystroke path performs zero DB round-trips (counter-probe verified)
- [ ] ensure_default_workspace's repair side-effects move to session-start/workspace-switch; the keystroke path is read-only
- [ ] The staged-launch evidence bundle is parsed once per launch and cached on the launch object
