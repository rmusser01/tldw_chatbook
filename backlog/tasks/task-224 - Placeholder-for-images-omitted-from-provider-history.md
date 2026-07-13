---
id: TASK-224
title: Placeholder for images omitted from provider history
status: To Do
assignee: []
created_date: '2026-07-13 11:15'
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
- [ ] #1 Over-cap and non-vision image-only turns appear in provider payloads as a text placeholder instead of vanishing
- [ ] #2 Captioned image messages keep their existing text-fallback behavior
- [ ] #3 Controller payload tests cover both placeholder paths
<!-- AC:END -->
