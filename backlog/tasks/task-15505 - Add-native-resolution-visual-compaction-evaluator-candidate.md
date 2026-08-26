---
id: TASK-15505
title: Add native-resolution visual-compaction evaluator candidate
status: Done
assignee:
  - '@codex'
created_date: '2026-08-11 22:04'
updated_date: '2026-08-11 22:14'
labels:
  - console
  - context
  - evals
dependencies: []
references:
  - backlog/tasks/task-15482 - Run-evaluator-v3-Terra-raw-context-evidence.md
  - backlog/decisions/054-deterministic-visual-transcript-compaction.md
  - backlog/decisions/056-context-use-visual-compaction-evaluation.md
documentation:
  - Docs/superpowers/qa/visual-compaction-model-evaluation/README.md
modified_files:
  - tldw_chatbook/Chat/console_visual_transcript.py
  - tldw_chatbook/Chat/console_visual_evaluation.py
  - scripts/evaluate_visual_compaction.py
  - Tests/Chat/test_console_visual_transcript.py
  - Tests/Chat/test_console_visual_evaluation.py
  - Docs/superpowers/qa/visual-compaction-model-evaluation/README.md
  - backlog/docs/lessons-live-verification.md
priority: high
type: enhancement
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Introduce a byte-stable native-resolution renderer candidate so visual compaction can reduce patch-based image input cost without replacing the production renderer before measured downstream quality evidence exists.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Evaluator can render the same transcript with the production profile and a native 512x512 candidate
- [x] #2 The candidate removes the mechanical 2x upscale while preserving transcript content pagination provenance and deterministic bytes
- [x] #3 Evaluator reports content-free geometry and patch-count evidence while clearly labeling provider-token savings as unmeasured
- [x] #4 Production visual compaction continues using the existing renderer until a separately authorized live evaluation passes ADR-056 gates
- [x] #5 Focused tests cover profile validation determinism payload invariants backward compatibility and recommendation gating
- [x] #6 Documentation records the current OpenAI image-accounting rationale and the no-billable-call boundary
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Characterize the production v1 renderer and evaluator serialization against current official image-input accounting. 2. Add explicit immutable renderer profiles while leaving the production default pinned to v1. 3. Add a native 512x512 evaluator-only candidate that preserves the logical canvas content pagination source provenance and byte determinism. 4. Extend schema-v3 evidence with content-free renderer geometry and raw 32px patch counts while treating provider token savings as unmeasured until a live run. 5. Add compatibility payload-invariant recommendation-gate and mutation-focused tests. 6. Update the QA guide and complete static analysis focused tests and self-review. ADR required: no. ADR path: backlog/decisions/054-deterministic-visual-transcript-compaction.md and backlog/decisions/056-context-use-visual-compaction-evaluation.md. Reason: the candidate stays evaluator-only and implements the existing versioned-renderer and measured-evidence gates without changing persistence provider ownership or default policy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added closed deterministic renderer profiles while preserving the shipped production_1024 v1 bytes and default. Added an evaluator-only native_512_candidate that removes the mechanical 2x enlargement but keeps the same logical content pagination source provenance and deterministic PNGs. Added content-free geometry evidence: the current pages fall from 1024 to 256 raw 32px patches each while provider savings remain explicitly unmeasured. The live evaluator CLI can select the candidate only through a closed profile choice; no provider calls were made. The checked-in Terra page hashes prove production byte compatibility and support-matrix v1/v2/v3 round trips remain exact. Verification: 42 focused tests passed; the reachable suite had 67 passes and 12 setup/teardown errors from the existing Windows Proactor socketpair/network-guard conflict before affected test bodies ran. Ruff check and format check passed; compileall passed. Mutations that ignored candidate selection or changed the evaluator default were both caught. Qodo follow-up added complete Google-style API docstrings, a canonical-registry parity test for the inert CLI profile choices, and a post-guard defensive profile resolution before any provider call. Updated the QA guide and live-verification lesson. ADR required: no; ADR-054 and ADR-056 govern the change.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a native-resolution visual-compaction evaluation candidate with honest patch-geometry evidence while keeping production byte-identical and making no billable calls.
<!-- SECTION:FINAL_SUMMARY:END -->
