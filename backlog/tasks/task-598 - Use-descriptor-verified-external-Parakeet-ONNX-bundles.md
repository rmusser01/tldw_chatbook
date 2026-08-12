---
id: TASK-598
title: Use descriptor-verified external Parakeet ONNX bundles
status: Done
assignee:
  - '@codex'
created_date: '2026-07-24 01:03'
updated_date: '2026-08-12 03:05'
labels:
  - stt
  - artifacts
  - import
  - onnx
dependencies:
  - TASK-594
  - TASK-596
references:
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
  - backlog/decisions/041-direct-local-gguf-before-managed-acquisition.md
  - backlog/decisions/050-external-parakeet-roots-with-managed-vad.md
documentation:
  - Docs/superpowers/specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md
  - >-
    Docs/superpowers/specs/2026-08-09-task-598-external-parakeet-bundles-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users select known Parakeet ONNX model directories and transcribe from those directories without copying the model into Chatbook's managed store. Chatbook remains responsible for the verified Silero VAD dependency, while an optional managed-copy action remains available for users who want immutable store ownership.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 First-run setup and Lab Models let a user select and persist an external Parakeet v2 or v3 directory for an exact INT8 or F32 catalog descriptor, while Library retains a job-scoped directory override that does not change global source configuration; using either path does not copy, modify, delete, or label the external directory as managed or installed.
- [x] #2 Before use or persistent selection, every descriptor-required model and external-data file is contained within the selected directory, is a materialized regular non-symlink file, and matches the exact catalog byte size and SHA-256; unknown, missing, modified, irregular, or changed bundles fail with stable path-safe errors without parsing ONNX graphs in the UI or resident worker.
- [x] #3 Chatbook reuses or explicitly offers to download only the pinned managed Silero VAD dependency; the user never supplies a VAD path, transcription never initiates a download, and the resident worker holds the exact VAD lease for its full residency interval. Persistent selection commits only after root verification and VAD readiness succeed, so offline operation, failure, or cancellation leaves the prior source unchanged.
- [x] #4 Resolution is deterministic: an explicit per-job directory wins, then the configured preferred source for the exact descriptor (external or managed), then a matching active managed model only when no preference exists, then the existing verified legacy fallback only when neither exists. A remembered non-preferred external path is not a candidate; an invalid explicit or preferred source fails clearly rather than silently switching providers or model sources, and the legacy singular v2 INT8 path migrates safely.
- [x] #5 External-source identity participates in resident reuse/recycle without entering generic logs or transcript provenance as a path or fabricated managed root. Results retain provider/model/precision/language/device fields, use a null artifact root, and record the exact managed VAD dependency identity; UI labels external sources as user-owned rather than installed or managed.
- [x] #6 After validation, an optional managed-copy action copies only the Parakeet root through existing staging, revalidates it, and reuses the managed VAD without writing readiness, changing the active selector, or changing source preference. A valid installed root without readiness is labeled activation-required and can be explicitly activated; activation verifies the complete closure, writes readiness last, switches the selector, and only then prefers managed. Cancellation or failure leaves the external source active and unchanged, and dependency-only rows never offer Activate.
- [x] #7 Focused tests cover v2/v3, INT8/F32, external data, corruption and containment, cancellable/coalesced verification, batch-lifetime reuse and cleanup for identical per-job overrides, source mutation, configuration precedence and migration, atomic managed-VAD consent/config commit, VAD lease/deletion behavior, mixed-source runtime provenance, first-run/Models/Library wiring, optional-copy rollback and later activation, and one real macOS external-mode smoke. The task remains In Progress until the parent design's Linux x86_64, Linux aarch64, Windows x86_64, macOS arm64, and macOS x86_64 wheel-supported evidence gates pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/050-external-parakeet-roots-with-managed-vad.md
Reason: implements the accepted mixed external-root/managed-VAD ownership and runtime boundary.

Detailed plan: Docs/superpowers/plans/2026-08-09-task-598-external-parakeet-bundles.md

1. Add descriptor-backed external-root verification with cancellable, coalesced, process-lifetime retention.
2. Add exact source records, authoritative preference resolution, atomic config handoff, Library scopes, and VAD-only acquisition helpers.
3. Add dependency-only managed leases without fake root readiness.
4. Carry external root plus managed VAD through the shared executor with path-private provenance.
5. Make one app-owned source service authoritative for Library and Console.
6. Correct managed inventory semantics and add install-only optional copy.
7. Add the user-owned external-source section and selection flow to Lab Models.
8. Add atomic external selection to First Run.
9. Validate Library per-job overrides and VAD readiness before enqueue.
10. Run only affected focused gates and record honest isolated macOS evidence while leaving unavailable platform gates open.
11. Complete correctness and complexity review before integration.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented descriptor-verified, user-owned external Parakeet v2/v3 INT8/F32 selection across First Run, Lab Models, and Library job overrides under ADR-050. External roots are verified and used in place; Chatbook owns only the exact managed Silero VAD, deterministic source resolution, path-private null-root provenance, resident identity/revalidation, optional install-only managed copy, and explicit later activation.

The isolated macOS arm64 production-path smoke is recorded in Docs/STT_Evaluation/task-598/macos-evidence.json. GitHub Actions run 31553729188 tested commit 9e006d3e6 on Linux x86_64, Linux aarch64, Windows x86_64, and macOS x86_64. Every lane passed v2/v3 INT8 descriptor verification, optional managed-copy deletion, CPU inference, exact-VAD/null-root provenance, external/cache/store/preference invariants, VAD-only final-store checks, and clean shutdown. Normalized evidence is in platform-evidence.json; all five wheel-supported platform gates pass.

Correctness and Ponytail review added managed-VAD revalidation on resident reuse, off-loop managed-copy planning, shared path-private recovery copy, and lean callback forwarding. Focused production/UI suites and mutation checks pass. The final 25-file changed-test union at 37dfd74c9 reached 1361 passed, 2 failed, 1 skipped, and 10 warnings in 364.05 seconds; both failures were sandbox PermissionError denials while binding ephemeral localhost artifact fixtures. Those exact two nodes passed 2/2 in 1.31 seconds when localhost binding was permitted. TASK-15531 separately resolved the pre-existing MCP workspace-save tooltip audit, and both route aliases pass in the final union. Scoped lint, compile, JSON, privacy, detector, and diff checks pass; retained Ruff/format findings are proven origin/dev debt. AC1-7 and the ADR-050 evidence gates are complete.
<!-- SECTION:NOTES:END -->
