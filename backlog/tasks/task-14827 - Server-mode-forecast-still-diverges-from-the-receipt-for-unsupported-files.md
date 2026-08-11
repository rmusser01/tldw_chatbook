---
id: TASK-14827
title: >-
  Server-mode forecast still diverges from the receipt for unsupported files
status: To Do
assignee: []
created_date: '2026-08-10 22:30'
labels:
  - library
  - ingest
  - server
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while fixing the xhigh review of the forecast arc (tasks 14820-14826), flagged by the implementing agent as out of scope for that round.

In server mode an unsupported file is forecast as `will skip`, but `build_server_ingest_kwargs` raises `ServerIngestUnsupported` and the job actually **fails**. So the forecast and the receipt disagree on the server path — the same class of defect task-14820 existed to eliminate on the local path.

This matters because it is the second server-path divergence found in one review round: the first (local tooling gaps subtracted from a server-bound forecast, making every server import read as a certain failure) was a regression the arc itself introduced and is now fixed. Both hid in the same blind spot — the governance test `test_forecast_counts_equal_the_real_receipt_for_a_mixed_folder` drives the LOCAL submit path only, so nothing asserts forecast==receipt for server mode at all.

Related, same surface: the canvas still renders local tooling warnings during a server run. Post-fix the folded summary reads "no staged file needs them", which is at least true, but the warning wall is still describing a machine that isn't doing the work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 In server mode, a file the server path will refuse is forecast as a failure, not a skip
- [ ] #2 A governance test asserts forecast counts equal the real receipt for a SERVER submission, mirroring the local one (the absence of this test is why two server-path divergences shipped)
- [ ] #3 Local tooling warnings are not presented as blocking facts during a server-targeted import
<!-- AC:END -->
