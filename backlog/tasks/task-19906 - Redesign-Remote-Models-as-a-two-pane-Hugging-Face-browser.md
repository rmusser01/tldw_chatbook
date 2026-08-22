---
id: TASK-19906
title: Redesign Remote Models as a two-pane Hugging Face browser
status: Done
assignee:
  - '@codex'
created_date: '2026-08-22 07:31'
updated_date: '2026-08-22'
labels:
  - models
  - ui
dependencies:
  - TASK-596.1
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the efficient browse-and-inspect workflow of the original Hugging Face model manager while retaining the verified managed-install boundary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Remote Models keeps search results visible in a left pane while the selected repository is inspected in a right pane.
- [x] #2 Search results show available repository metadata including downloads, likes, update date, access, and gated status without adding network requests.
- [x] #3 The detail pane exposes pinned provenance, license, compatibility warning, selectable GGUF candidates with human-readable sizes, and one contextual Review and install action.
- [x] #4 Install progress and outcomes remain visible in the selected model detail pane, and the existing managed preflight, consent, verification, and installation flow is unchanged.
- [x] #5 The mounted production-stylesheet screen remains usable at the supported narrow terminal width with keyboard-reachable controls and truthful states.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
Reason: ADR-025 already governs managed remote acquisition, provenance, consent, and downloader retirement; this task changes only the existing Remote view presentation.

1. Add focused mounted UI tests for the persistent two-pane browser, repository metadata, candidate selection, and contextual install progress.
2. Recompose RemoteView around persistent results and a selected-model detail pane while preserving its existing workers and host-screen install messages.
3. Add production-stylesheet narrow-width and compositor evidence, then run focused tests, static checks, and self-review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Rebuilt `RemoteView` as a persistent two-pane Hugging Face browser: dense repository metadata remains on the left while pinned provenance, license, compatibility guidance, GGUF variants, selection, progress, and the single contextual install action live on the right.
- Kept candidate selection separate from acquisition and preserved the ADR-025 boundary: `RemoteView` posts only a frozen install intent; `LLMScreen` still owns managed preflight, consent, verification, and provisioning.
- Added measured responsive behavior for the real Models/Lab body at 80 columns, normalized untrusted update timestamps, preserved keyboard focus across in-place pane updates, and restored validated selected-model context throughout preflight, consent, progress, and terminal outcomes after screen recomposition.
- Hardened install startup so managed-service or credential-resolver construction failures remain sanitized and retryable instead of stranding disabled controls.
- Centralized retained terminal-action values as named constants and added a static regression preventing repeated magic action literals.
- Rebased onto the latest `dev`, audited every diagnostic-inventory delta before refreshing the required generated pin, and confirmed this slice's new diagnostic records only an exception class; inherited ordinary-log path content remains owned by TASK-19864 and is not admitted to the persistent sink. The later Notes merge removed legacy payload-bearing diagnostics and added a payload-free `0600` lock-file sink beneath a verified `0700` application directory.
- Regenerated the consolidated widget stylesheet and added mounted component, real-host compositor, failure-state, focus, consent, and install-lifecycle coverage.
- Verification: the final Remote-focused sweep passed 140 tests across the complete Remote view, Hugging Face adapter, and every Remote-named Models/Lab adoption scenario, including live preflight → consent → install phase-copy transitions and the terminal-action constant guard; all 11 CSS build-integrity tests and all 9 UI latency guardrails also passed. Ruff, formatter, Python compilation, generated-CSS reproducibility, persistent-diagnostic inventory reproduction, profile-owned-path census, `git diff --check`, the 2,406-file duplicate-task-ID guard, and the Impeccable detector (zero findings) passed.
- A follow-up filtered broad sweep reached 5,505 passed and 85 skipped before stopping at five unrelated worktree failures (Console size/binding drift and persona schema census/version drift). The unfiltered sweep also exposed architecture inventories descending into the ignored nested `tldw_chatbook/.venv`; that recurring test-inventory incident is recorded in `backlog/docs/lessons-testing-evidence.md`. No Models/Remote test failed in either broad run.
- ADR required: no. Existing ADR: `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md`.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

- Previous ID: `TASK-19602`.
- Renumbered to `TASK-19906` on 2026-08-22 after the older Library task reached `dev` with the original ID; references owned by this Remote Models slice moved with it.
