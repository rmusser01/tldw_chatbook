---
id: TASK-3402
title: H3 static image edit through Image_Generation
status: To Do
assignee: []
created_date: '2026-08-09 04:39'
labels:
  - image
  - generation
  - comfyui
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Package a sanitized H3 static image-edit workflow inside the existing Image_Generation validation, attachment-storage, and metadata boundary. The sanitized copy removes nodes 154 and 166, and node 165 is the canonical edited-image output. No prompt text from the supplied workflow source, raw export artifact, or source identity is recorded. The user's own edit instruction remains part of the existing generation-metadata contract.
<!-- SECTION:DESCRIPTION:END -->

## Design

- [Approved design specification](../../Docs/superpowers/specs/2026-08-10-comfyui-h3-image-edit-design.md)
- [ADR-052 — ComfyUI H3 image edits stay inside the Image Generation provider boundary](../decisions/052-comfyui-h3-image-edit-provider-boundary.md)

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A renamed sanitized API workflow excludes nodes 154 and 166 and selects node 165 as the sole canonical edited-image output.
- [ ] #2 Exactly one staged image and one unstyled instruction produce exactly one PNG at the source dimensions; seed, steps, and sampler are applied, while batch counts other than one, style tokens, size, negative-prompt, CFG, model, alternate-format, and other unsupported overrides are rejected before image upload or prompt submission.
- [ ] #3 Successful results use the existing Image_Generation attachment storage and generation-metadata contract, with no Video_Generation storage path; only the exact initiating attachment is consumed after durable persistence, and H3 Regenerate requires restaging the source.
- [ ] #4 Image Generation owns independent ComfyUI settings and a strict packaged-workflow adapter; all API traffic and output retrieval remain on the configured trusted origin, and settings disclose operator-managed server retention.
- [ ] #5 Repository artifacts and history contain no prompt text from the supplied workflow source, raw export artifact, source identity, media bytes, server descriptors, or sensitive UAT evidence.
- [ ] #6 Focused workflow, distribution, adapter, validation, configuration, Console-storage, privacy, and live-UAT checks pass without running unrelated or full-repository suites.
<!-- AC:END -->
