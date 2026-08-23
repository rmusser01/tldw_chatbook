---
id: TASK-21115
title: >-
  CSS parse-cache headroom - bundle-ride the 34 new DEFAULT_CSS classes and add a static allowlist ratchet
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - css
  - console
  - library
priority: high
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21115).

34 new `DEFAULT_CSS` declarations across 29 files landed since the task-15450 consolidation
(Console modals/inspector rail/turn-file card, Library dialogs, trajectory, speech; full list in
the evidence doc). Live tour on the pin measured 47 sources (empty transcript, no modals)
against the LRUCache(64) cliff and the 56 soft guard limit; adding conversation-row classes and
~10 distinct modal opens crosses 64 today, at which point every later first-mount of any unseen
widget class re-pays a full cold parse (~150-450 ms fast HW, x3-5 constrained) for the rest of
the session. Accretion is ~+8 classes/3 days while the tour guard is red (TASK-21106) and CI
does not run. All 34 are plain string blocks that can ride the sanctioned
BUNDLED_CSS/BUNDLED_SCREEN_CSS + build_css.py mechanism; `UI/SiteConfigSettings.py:41` is also
the last class-level `CSS` remaining.

## Acceptance Criteria

- [ ] The 34 new DEFAULT_CSS blocks (and SiteConfigSettings' class CSS) ride the bundle; harness parse-standalone requirements still hold
- [ ] A STATIC allowlist ratchet test (AST walk, no app boot) fails on any DEFAULT_CSS/CSS declaration outside the allowlist, so the invariant no longer depends on the slow integration tour running
- [ ] A post-change tour + 12-modal probe stays comfortably under the 56 soft limit; measured count recorded in the task
