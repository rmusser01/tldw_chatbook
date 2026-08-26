---
id: TASK-3402
title: H3 static image edit through Image_Generation
status: Done
assignee: []
created_date: '2026-08-09 04:39'
updated_date: '2026-08-12 00:11'
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
- [x] #1 A renamed sanitized API workflow excludes nodes 154 and 166 and selects node 165 as the sole canonical edited-image output.
- [x] #2 Exactly one staged image and one unstyled instruction produce exactly one PNG at the source dimensions; seed, steps, and sampler are applied, while batch counts other than one, style tokens, size, negative-prompt, CFG, model, alternate-format, and other unsupported overrides are rejected before image upload or prompt submission.
- [x] #3 Successful results use the existing Image_Generation attachment storage and generation-metadata contract, with no Video_Generation storage path; typed Stop/unmount cancellation and app-owned draining cannot orphan a success, permit a duplicate after remount, or update stale UI; only the exact initiating attachment and unchanged draft are consumed after durable persistence, and H3 Regenerate requires restaging the source.
- [x] #4 Image Generation owns explicit-opt-in independent ComfyUI settings and a strict packaged-workflow adapter; all API traffic and bounded JSON/output retrieval remain on the configured trusted origin, and settings disclose operator-managed server retention.
- [x] #5 Repository artifacts and history contain no prompt text from the supplied workflow source, raw export artifact, source identity, media bytes, server descriptors, or sensitive UAT evidence.
- [x] #6 Focused workflow, distribution, adapter, validation, configuration, Console-storage, privacy, and live-UAT checks pass without running unrelated or full-repository suites.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
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
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the sanitized single-workflow ComfyUI H3 image-edit provider entirely inside the Image Generation boundary. The packaged graph exposes only canonical node 165; strict local and remote schema validation, exact-origin bounded transport, one PNG result, effective metadata, and typed cancellation all fail closed before unsafe progression. Independent F9 Image Generation settings and runtime snapshots keep ComfyUI isolated from Video Generation and future image providers. Console integration uses exact pending-attachment identity, app-owned operation ownership, shielded runner drain, durable success/failure reconciliation across remounts, stale-screen/session guards, and explicit restaging for Regenerate. ComfyUI server retention remains operator-managed.

Modified groups: workflow packaging and distribution; Image_Generation contracts, config, registry, validation, listing, and adapter; canonical Settings UI; Console attachment/store/generation/operation/app lifecycle; focused tests; ADR/design/plan/UAT and the incident-based detached-task cancellation lesson.

ADR: backlog/decisions/052-comfyui-h3-image-edit-provider-boundary.md. No duplicate ADR was created.

Verification: final authorized 18-file gate 624 passed with 2 existing environment warnings; isolated wheel/sdist install proof 1 passed; full Ruff passed on 32 changed small modules/tests; fatal-only Ruff E9,F63,F7,F82 passed on app.py, chat_screen.py, settings_screen.py, and root config.py; 36 changed Python paths compiled to temporary output; diff, privacy, provenance, provider-boundary, and residue checks passed. Per AC 6 and user direction, no full-repository, RuntimePolicy, or unrelated Video suite was run.

Live UAT: one synthetic 512x512 source passed the real worker/adapter class-schema preflight, returned exactly one node-165 PNG at preserved dimensions with the required effective metadata keys, and durably persisted/rehydrated through the normal Image Generation boundary. The post-review live rerun also passed. Synthetic source/local outputs were removed; server-side retention is operator-managed.

Implementation range: 9ad2f91d56db6fb659046649fdd350d358478c7c..dcd423361. Task heads/review fixes: b6362ad16, 0d7ff152d, 1f5947383, cb7a1b56e, b755eaa0a, 09bf2efaf, dcd423361. Final whole-task review approved with no remaining Critical, Important, or Minor findings.
<!-- SECTION:NOTES:END -->
