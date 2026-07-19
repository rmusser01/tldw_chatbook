---
id: TASK-224
title: Placeholder for images omitted from provider history
status: Done
assignee:
  - '@claude'
created_date: '2026-07-13 11:15'
updated_date: '2026-07-16 20:31'
labels:
  - console
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
In the Console provider payload builder (PR #621), an image-only user message that falls outside the max_images cap — or any image-only message under a non-vision model — has empty text and a disallowed image, so the entire turn is silently skipped, distorting the conversation shape the model sees. Emit a text placeholder (e.g. "[image omitted]") for such turns so the model still sees that a message existed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Over-cap and non-vision image-only turns appear in provider payloads as a text placeholder instead of vanishing
- [x] #2 Captioned image messages keep their existing text-fallback behavior
- [x] #3 Controller payload tests cover both placeholder paths
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
In _provider_message_payloads' no-text branch: USER messages with usable-but-unbudgeted attachments emit '[image omitted]' (or '[N images omitted]') instead of being skipped. Covers both paths: non-vision model (budget 0) and over-cap (older turn loses the newest-first reservation). Captioned messages keep the existing text fallback; assistant messages unchanged. Tests drive the real submit path via RecordingStreamingGateway.
<!-- SECTION:NOTES:END -->
