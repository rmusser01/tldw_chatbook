---
id: TASK-41112
title: "test_actor_pack_export.py's golden-bytes cases [character] and [persona] are red on dev and run on no lane"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- actor-packs
- tests
- dev-red
priority: low
references:
- Tests/Actor_Packs/test_actor_pack_export.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** The test pins exported Actor Pack archives byte for byte against independent golden bytes. Red, it no longer tells anyone whether an export change was intended.

**Evidence.** Through run.sh at the stack top: `test_archive_matches_independent_golden_bytes[character]` and `[persona]` (`:580`) fail with `assert output.getvalue() == golden`, first difference at byte 14 (the first local header's CRC-32, so the first member's content changed). TASK-40304 recorded both red on its base too. No workflow names the file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The commit that changed the first member's bytes is named, and the case is classified as intended format drift or an export defect
- [ ] #2 Drift: the golden bytes are regenerated from an independent writer with the change explained; defect: the export is fixed with the test red first
- [ ] #3 The file runs on a PR lane
<!-- AC:END -->
