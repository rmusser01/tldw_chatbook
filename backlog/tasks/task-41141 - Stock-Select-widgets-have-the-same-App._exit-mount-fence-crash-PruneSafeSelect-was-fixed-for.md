---
id: TASK-41141
title: "Stock Select widgets have the same App._exit mount-fence crash PruneSafeSelect was fixed for"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- ui
- textual
- crash
priority: low
references:
- tldw_chatbook/Widgets/prune_safe_select.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** TASK-40323's review found a crash: once `App._exit` is set (`App.exit`, and first thing in `TldwCli._shutdown`), `Widget.mount_all` mounts nothing, so a `Select` composed then has no `SelectOverlay`, but `Select._on_mount` and `set_options` still rebuild it: `NoMatches: 'SelectOverlay'`. It fixed `PruneSafeSelect` only (`prune_safe_select.py:130-170`, `_retired`).

**Evidence.** About 70-110 modules under `tldw_chatbook/` import or construct a stock Textual `Select` (72 files both import `Select` and call `Select(`). Any of them that composes or updates a Select during shutdown can raise the same error.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A test composes a stock Select after App._exit is set and records whether it raises
- [ ] #2 If it does, the app's Selects get the same exit gate (for example by using PruneSafeSelect or one shared subclass), with the test red before the change
- [ ] #3 The notes list which Selects can compose or update during shutdown
<!-- AC:END -->
