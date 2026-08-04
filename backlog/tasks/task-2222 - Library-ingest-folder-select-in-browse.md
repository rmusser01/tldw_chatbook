---
id: TASK-2222
title: >-
  Library ingest: "Select this folder" action in the Browse dialog
status: Done
assignee: []
created_date: '2026-08-04 05:00'
labels:
  - library
  - ingest
  - ux
priority: medium
dependencies: []
---

## Description (the why)

Owner ruling (2026-08-04): folder import must be pickable, not
type-only. In the Browse dialog, "Open" on a directory descends into it
(correct), and a dedicated "Select current folder" action returns the
directory being viewed as the ingest source (the vendored fspicker's
SelectDirectory variant is the reference).

## Acceptance Criteria (the what)

- [x] The ingest Browse dialog offers a visible "Select current folder"
      action (button and/or binding) that returns the directory being
      viewed; Open keeps descending.
- [x] Choosing it fills the path field with the directory and triggers
      pre-flight, exactly like typing the path.
- [x] File selection behavior is unchanged.

## Implementation Plan (the how)

Add an OPT-IN "Select folder" affordance to the shared picker base
(`FileOpen(offer_select_folder=True)`), dismissing with the directory
currently being viewed; the Library ingest Browse opts in.

## Implementation Notes

- `base_dialog`: the input bar renders a `#select-current-folder`
  Button only when the screen sets `_offer_select_folder`, so every
  other FileOpen caller's bar is byte-identical. The handler dismisses
  with `DirectoryNavigation.location` — "Open" keeps descending, per the
  ruling. A `ctrl+s` binding gives the keyboard route (guarded by the
  same flag).
- `file_open`: `offer_select_folder: bool = False` constructor flag,
  documented; the ingest Browse passes True.
- The existing callback already fills the path field, remembers the
  location, and triggers pre-flight — a picked folder therefore behaves
  exactly like a typed one.

**Verification.** 252 core/picker + 54 shell-subset green; collect
clean. Live: Browse shows "Select folder"; clicking it fills the path
with the viewed directory and the forecast reads
"2 will import · 2 will skip".
