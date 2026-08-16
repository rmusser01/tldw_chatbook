---
id: TASK-16851
title: 'Console transcript: head-pinned selection disables the prune while tailward hydration reveals'
status: To Do
assignee: []
created_date: '2026-08-16'
labels:
  - bug
  - console
  - design
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the TASK-15777 round-3 review (PR #1733, merged `ee741cf10`), verified against the
exact merged commit (`5e1e9e9ac`) and explicitly deferred out of that merge gate as
pre-existing (it reproduces identically at the round-2 commit; round 3's End-drain fix
only unmasked it):

**A far jump both selects its target and lands it at the top of the window.** The prune
protects `selected_message_id` and stops at the first protected group
(`console_transcript.py`, `_compute_prunable_prefix`), so with the target selected at
the *head*, the prune can never trim anything — while the tailward hydration chain keeps
revealing. Probe on the headline flow (far jump, then scroll down): mounted rows
24 → **490**, virtual height **1966 against a high watermark of 900** (2.18x), stable
after 120 idle frames; clearing the selection with Esc collapses it to 150/603. Bounded
only by session length — a 10k-message session would mount all of it. This is the mirror
image of the disclosed tail-pinned trade-off (which stays bounded because the prune still
trims the head); the head-pinned case is genuinely unbounded. Diagnosable (the prune logs
its blocked walk) but invisible to the user; the only recovery is Esc.

Suggested fixes from the review: refuse `_hydrate_tailward` while
`virtual_size.height >= high_mark` and the prune is blocked (the 15455 loop-breaker
applied to the other boundary), or stop protecting the selection when it is the first
mounted group. Include the review's second residual while in the file: a **one-frame
End-during-prune race** — an End pressed between prune entry and `_restore_scroll` has
its anchor cancelled by `_release_anchor_quietly` (entry state wins), so the drain may
stop after one chunk; pill is up and a second End resumes, but a settle-varied pin would
document the bound.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 With a head-pinned selection, a downward walk keeps total mounted height bounded near the high watermark (probe or test on the far-jump-then-scroll-down flow as evidence)
- [ ] #2 The selection and its action row survive whatever bounding mechanism ships (no teleport, no selection loss)
- [ ] #3 The 15777 two-sided suite (13), protected pruning suite, and End-drain pins stay green
- [ ] #4 The one-frame End-during-prune residual is either closed or pinned-and-documented with its recovery
<!-- AC:END -->
