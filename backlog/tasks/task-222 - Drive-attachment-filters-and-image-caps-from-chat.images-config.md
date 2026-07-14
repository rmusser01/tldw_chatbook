---
id: TASK-222
title: Drive attachment filters and image caps from chat.images config
status: In Progress
assignee: ['@claude']
created_date: '2026-07-13 11:15'
labels:
  - chat
  - config
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #621's spec claims the attachment format allowlist and caps read from [chat.images] so the picker and pipeline cannot drift — but ATTACHMENT_FILTER_SPECS (attachment_core.py) and ChatImageHandler constants (SUPPORTED_FORMATS, 10 MB cap, 2048 px resize) are hardcoded, ignoring the existing supported_formats/max_size_mb/resize_max_dimension config keys. Visible drift exists today: the picker's Image Files filter offers .tiff/.tif/.svg which the pipeline rejects with an error toast (behavior inherited from legacy, no regression). Wire both consumers to the config keys so the no-drift-by-construction claim becomes true.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Picker image filter and ChatImageHandler format allowlist derive from [chat.images].supported_formats (no tiff/svg mismatch)
- [ ] #2 Image size cap and resize dimension honor max_size_mb / resize_max_dimension
- [ ] #3 Legacy regression gate stays green (defaults must reproduce current behavior; zero edits to existing gate tests)
- [ ] #4 .tiff/.tif attach end-to-end: decoded and delivered to providers as a payload-safe format (png/jpeg/webp/gif) with mime matching the actual bytes
- [ ] #5 .svg attach end-to-end via cairosvg rasterization when available; .svg absent from picker, routing, and allowlist when cairosvg is unavailable (capability gate)
- [ ] #6 Payload images always carry provider-safe formats with mime matching bytes (repairs latent bmp passthrough and resized-gif mime mismatch)
<!-- AC:END -->
