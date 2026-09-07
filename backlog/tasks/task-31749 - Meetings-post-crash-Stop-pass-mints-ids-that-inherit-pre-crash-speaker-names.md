---
id: TASK-31749
title: 'Meetings: post-crash Stop pass mints ids that inherit pre-crash speaker names'
status: Done
assignee: []
created_date: '2026-09-06 14:53'
labels:
  - audio
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After the diarizer worker crashes and is restarted once (the backend then serves coarse labels only), the replacement worker holds no live centroids, so the authoritative Stop pass cannot reconcile and mints fresh ids S1..Sn from scratch. Those ids collide with entries in the meeting's speaker_names map that the user assigned before the crash, so the finished transcript can show a pre-crash name on the wrong voice (previously everything collapsed to one bogus S1; after the independent batch count it can be up to max_speakers of them). Options: carry the pre-crash live centroids across the restart, or namespace post-crash minted ids (or clear the name map for them) so no pre-crash name is applied. Found in the PR #2456 fix-wave re-review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After a worker crash and restart, no pre-crash speaker name is applied to a post-crash minted cluster id
<!-- AC:END -->

## Implementation Notes

After a worker crash the restarted worker is spawned with `--start-id <max id seen>` so post-crash ids are minted past the pre-crash range (the Stop-pass batch mint honours the same floor), the backend records `crashed_at_seq`, and `MeetingSession.stop()` runs the batch pass only over the post-crash span and overlays only those segments, so pre-crash segments keep their ids and names. When no post-crash segment exists the pass is skipped rather than run full-span. Landed in PR #2471 (commit 656ec4236).
