---
id: TASK-702
title: Library ingest reports a YouTube URL as an unsupported file
status: Done
assignee: []
created_date: '2026-07-26 14:12'
updated_date: '2026-07-26 23:45'
labels:
  - library
  - ingest
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pasting a video URL into the Library ingest canvas reports it as an unsupported file, even though the ingest pipeline classifies the same URL as video and can ingest it. The canvas decides a source's type by file extension alone, so any URL without a recognisable extension falls through to the unsupported bucket. Video links are one of the most common things a user wants to import, so this makes the canvas look broken for a mainstream case.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A video-hosting URL is recognised as audio/video by the ingest canvas,A URL without a file extension is not reported as an unsupported file when the pipeline can ingest it,The canvas's classification of a source agrees with what the ingest pipeline will actually do
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in PR #923 as a prerequisite for bringing web clipping into the canvas.

get_type_group decided by file extension alone via detect_file_type, so any extension-less URL fell to the unsupported bucket -- a YouTube link pre-flighted as an unsupported FILE while the pipeline's own classify_ingest_source called the same URL video and would have ingested it. URLs now defer to classify_ingest_source rather than re-deriving the rules, with a test comparing the two classifiers directly so a future divergence fails in CI rather than on a user's screen. An extension still wins where there is one, so a link to a .pdf or .epub is parsed as that rather than scraped as HTML.

Verified live: youtube.com/watch -> audio_video, 1 file, no errors.

The status was never updated when it shipped; the code has been on dev since #923.
<!-- SECTION:NOTES:END -->
