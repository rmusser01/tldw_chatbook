---
id: TASK-3402
title: H3 static image edit through Image_Generation
status: In Progress
assignee: []
created_date: '2026-08-09 04:39'
updated_date: '2026-08-11 02:34'
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

## Implementation Plan

- [Detailed implementation plan](../../Docs/superpowers/plans/2026-08-10-comfyui-h3-image-edit.md)
- ADR required: yes
- ADR path: `backlog/decisions/052-comfyui-h3-image-edit-provider-boundary.md`
- Reason: the task introduces an image-provider runtime, exact-origin trust, cancellation/lifecycle semantics, and a cross-module persistence contract; ADR-052 already records the approved boundary.

1. Package the privately sanitized exact H3 graph and prove source/wheel/sdist distribution without recording the external source.
2. Extend the existing Image Generation request/result/reference/cancellation contracts at the single validation choke point.
3. Add independent explicit-opt-in ComfyUI Image Generation config, registry, listing, and canonical F9 settings.
4. Implement and mutation-test the strict packaged-workflow adapter, bounded exact-origin transport, node-165 PNG output, and prompt-scoped cancellation.
5. Add exact pending-attachment identity, cancellation propagation, and allowlisted effective metadata to the existing image persistence boundary.
6. Implement the app-owned Console operation/remount lifecycle, exact-once durable success, identity-gated cleanup, and H3 Regenerate refusal.
7. Run only the authorized focused gates, complete sanitized live UAT, perform whole-task review, and close the task only after all evidence passes.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A renamed sanitized API workflow excludes nodes 154 and 166 and selects node 165 as the sole canonical edited-image output.
- [ ] #2 Exactly one staged image and one unstyled instruction produce exactly one PNG at the source dimensions; seed, steps, and sampler are applied, while batch counts other than one, style tokens, size, negative-prompt, CFG, model, alternate-format, and other unsupported overrides are rejected before image upload or prompt submission.
- [ ] #3 Successful results use the existing Image_Generation attachment storage and generation-metadata contract, with no Video_Generation storage path; typed Stop/unmount cancellation and app-owned draining cannot orphan a success, permit a duplicate after remount, or update stale UI; only the exact initiating attachment and unchanged draft are consumed after durable persistence, and H3 Regenerate requires restaging the source.
- [ ] #4 Image Generation owns explicit-opt-in independent ComfyUI settings and a strict packaged-workflow adapter; all API traffic and bounded JSON/output retrieval remain on the configured trusted origin, and settings disclose operator-managed server retention.
- [ ] #5 Repository artifacts and history contain no prompt text from the supplied workflow source, raw export artifact, source identity, media bytes, server descriptors, or sensitive UAT evidence.
- [ ] #6 Focused workflow, distribution, adapter, validation, configuration, Console-storage, privacy, and live-UAT checks pass without running unrelated or full-repository suites.
<!-- AC:END -->
