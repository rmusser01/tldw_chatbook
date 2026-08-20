---
id: TASK-19196
title: Harden the stts view-cycling test's five one-shot children[0] asserts (19047 family)
status: To Do
assignee: []
created_date: '2026-08-20'
labels:
  - test-health
  - flake
  - stts
dependencies:
  - TASK-19047
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Conditional hardening candidate, filed from the third-wave close-out. At dev
`7877defba`,
`Tests/UI/test_stts_profile_library.py::test_voice_profiles_view_mounts_focused_library_without_hiding_other_views`
(:1085, body :1104-1139) carries five one-shot
`app.query_one(".stts-content").children[0]` isinstance asserts (playground →
settings → audiobook → dictation → playground) — the exact
raising-predicate/empty-window class TASK-19047 fixed elsewhere in the same
file: `STTSWindow.watch_current_view` swaps the body in a `speech-view-mount`
worker, and `.children[0]` sampled between `remove_children()` and `mount()`
raises `IndexError` (see the 2026-08-20 "settle whose predicate can RAISE"
entry in `backlog/docs/lessons-testing-evidence.md`, and the
`_stts_content_first_child` helper 19047 added at :1053, used with
`_wait_until` at e.g. :2910).

Important honesty note: these five asserts NEVER fired across all of 19047's
and its reviewer's CPU-burner load runs — this is a structural sibling of a
proven flake class, not an observed flake. Hence the conditional shape: either
prove it can fire (load reproduction) or convert it mechanically to the
already-shipped helpers and re-prove the whole file under load. Do not
half-convert.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A load-reproduction attempt (19047's CPU-burner loop methodology) against the current asserts is run FIRST and its outcome recorded — either a reproduced failure justifying the fix, or a bounded negative result; OR the five asserts are mechanically converted to `_wait_until` + `_stts_content_first_child` settles matching the file's existing pattern.
- [ ] #2 If converted: the full file's load-loop evidence is re-run and recorded (not just the touched test), per 19047's catalogue-shapes-by-re-running lesson.
- [ ] #3 The test's contract is unchanged: it still pins that each view switch mounts the expected pane type as the content's first child and that cycling back to playground restores `#tts-generate-btn`.
<!-- AC:END -->
