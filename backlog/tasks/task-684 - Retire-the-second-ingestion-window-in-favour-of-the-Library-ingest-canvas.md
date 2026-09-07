---
id: TASK-684
title: Retire the second ingestion window in favour of the Library ingest canvas
status: Done
assignee: []
created_date: '2026-07-26 03:27'
updated_date: '2026-07-26 17:54'
labels:
  - ingest
  - cleanup
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The app ships two ingestion interfaces. Only the second one's Local Files tab duplicates the Library ingest canvas; its Server Sources, Server Jobs and Web Clipper tabs are server-backed capabilities the canvas has no equivalent for, and the canvas states outright that ingest runs on Local. Retiring the window outright would therefore delete working features, so the three capabilities are ported into the canvas first and the window is deleted last. Tracked as an umbrella over its subtasks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Import sources opens the Library ingest canvas
- [ ] #2 No route or button reaches the retired window
- [ ] #3 Any capability the retired window had that the canvas lacked is available in the canvas
- [ ] #4 The retired window and its now-unused event handlers are deleted
- [ ] #5 The full test suite passes with the window removed
- [ ] #6 Server-backed ingestion is available from the Library ingest canvas,Remote ingest job status is visible alongside local jobs,Web clipping is available from the Library ingest canvas,Import sources opens the Library ingest canvas,No route or button reaches the retired window,The retired window and its now-unused event handlers are deleted,The full test suite passes with the window removed
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Complete. The second ingestion window is retired; importing lives entirely in Library's Import media canvas.

Delivered in two PRs. #923 brought the three capabilities the window uniquely owned into the canvas -- server-backed file ingest (684.1), remote jobs sharing the local queue (684.2), and web clipping (684.3) -- so retiring the window would not delete working features. #930 then deleted the window, its panels, its two event-handler modules and 13 test files: ~6,780 lines.

The sequencing mattered. As first specified this was a straight deletion; scouting found the window had four tabs and only one duplicated the canvas, so three working features would have gone with it. Porting first was the user's call and the right one.

Live contact with a real server, not the test suite, found what was actually broken: the runtime-policy gate that made server ingest unusable as first built (the switch was offered, every submission failed), a media_type the server rejects, a keyword-only call made positionally, a result field typed as a different domain's model that made every COMPLETED job unparseable, dead pagination, and a missing auth header that broke every web clip. Seven defects, none visible to unit tests.

The recurring cause is worth stating once: a fake written to match my own call site validates the mistake. The durable fix was contract tests against real signatures and verbatim captured payloads, each mutation-checked.

Follow-ups left open: 697 (pre-flight 403 veto), 698 (form reset), 699 (Library shell test nondeterminism), 700 (open a server-ingested item), 701 (TLDW_CONFIG_PATH runtime-policy isolation), 702 (YouTube URL grouping) -- 697 and 702 were fixed in #923; 745 (ingest stylesheets still in the bundle).
<!-- SECTION:NOTES:END -->
