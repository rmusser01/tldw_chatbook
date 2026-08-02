---
id: TASK-596.1
title: Add bounded Hugging Face GGUF discovery to the managed model browser
status: Done
assignee:
  - '@codex'
created_date: '2026-08-01 21:51'
updated_date: '2026-08-02 01:47'
labels:
  - stt
  - artifacts
  - ui
  - security
dependencies:
  - TASK-595
references:
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
documentation:
  - Docs/superpowers/specs/2026-08-01-task-596-model-artifact-browser-design.md
  - >-
    Docs/superpowers/specs/2026-08-01-task-596-1-remote-model-discovery-design.md
  - Docs/superpowers/plans/2026-08-01-task-596-1-remote-model-discovery.md
parent_task_id: TASK-596
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users explicitly find and download remote GGUF models through the shared managed-model flow without implying that arbitrary models are runtime-compatible or independently verified.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Opening Remote performs no network request; explicit search or exact repository submission runs off the Textual event loop with bounded, generation-fenced results.
- [x] #2 A selected repository resolves to an immutable commit and offers only LFS-backed single GGUF files or complete bounded GGUF shard sets with recorded sizes and SHA-256 digests.
- [x] #3 A selected candidate reaches the existing managed preflight, consent, download, verification, and installation flow; configured Hugging Face credentials support gated or private repositories without being persisted or forwarded across origins.
- [x] #4 Known license metadata is shown; missing license metadata is recorded as NOASSERTION with a pinned source-review page and requires explicit acknowledgment before download.
- [x] #5 Focused adapter, GGUF grouping, Textual, redirect-security, and managed-acquisition tests cover the flow without adding native or platform-specific dependencies; Windows and Linux gates remain required when runners are available.
- [x] #6 Remote installation labels the model Local integrity recorded and does not activate it; its descriptor uses consumer=unassigned, Installed offers no activation action for that consumer, and no UI presents it as runtime-compatible, transcription-ready, or eligible for automatic routing.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Rebase onto the latest origin/dev, confirm no TASK-596.1 documentation collision, and rerun the affected baseline.
2. Add the bounded Hugging Face metadata adapter, exact repository resolution with blobs=true, GGUF grouping, and managed artifact mapping using test-first steps.
3. Add install-without-activation and reject HTTPS-to-HTTP download redirects while preserving existing defaults.
4. Update shared inventory, activation, plan, and consent widgets to represent unassigned downloads honestly.
5. Add the explicit, lazy Remote view with generation fencing, credential-safe metadata requests, and the existing managed preflight/provision flow.
6. Run focused regression/static gates, collect mocked-payload macOS evidence, request code review, and close TASK-596.1 only after all acceptance criteria pass.

ADR required: no
ADR path: backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
Reason: ADR-025 already owns remote artifact provenance, managed acquisition, activation, and runtime boundaries; this slice is additive within that boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented bounded Hugging Face GGUF discovery through the shared managed-artifact flow.

- Added explicit fixed-origin search and exact-repository resolution with streamed metadata limits, immutable commit pinning, LFS size/SHA-256 validation, complete bounded shard grouping, deterministic artifact identity, pinned source maps, license handling, and sanitized recovery errors.
- Added additive install-without-activation support and HTTPS-to-HTTP redirect rejection while preserving existing callers and credential stripping.
- Added a lazy Remote Textual view with generation and identity fencing, worker-local configured credentials, frozen preflight/consent/provision state, unknown-license acknowledgment, exact selected-file integrity/source review, and Installed refresh. Remote artifacts remain LOCAL_INTEGRITY_RECORDED, consumer=unassigned, inactive, and explicitly compatibility-unverified.
- Updated shared inventory, activation, plan, and consent controls through backward-compatible optional/default inputs; Delete remains available and Activate is absent for unassigned models.
- Added focused adapter/grouping/security/UI tests plus a real mocked-payload resolve-to-managed-install integration test.
- Final whole-branch review and scoped fix re-review are clean. After rebasing onto current dev, 576 affected tests passed with one existing Requests dependency warning; 12 credential-boundary and 8 deterministic macOS evidence tests passed. Branch-edited scope passes Ruff, mypy, py_compile, and diff checks. The known fetch.py F401 and acquisition.py:1823 mypy mismatch were verified on dev and remain out of scope. Windows/Linux gates remain required when runners are available.

ADR required: no. ADR-025 remains authoritative for managed acquisition, provenance, activation, and runtime boundaries. No new dependency, provider framework, cache, compatibility detector, or alternate downloader was introduced.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Users can explicitly discover and securely download pinned Hugging Face GGUF artifacts through the managed model browser without automatic activation or unsupported runtime-compatibility claims.
<!-- SECTION:FINAL_SUMMARY:END -->
