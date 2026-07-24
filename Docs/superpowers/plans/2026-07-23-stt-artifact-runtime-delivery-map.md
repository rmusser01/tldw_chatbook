# STT Artifact and Runtime Delivery Map

**Goal:** Deliver the approved shared model-artifact core, Parakeet ONNX
defaults, optional transcribe.cpp provider, and legacy-provider removal as
independently reviewable Backlog tasks.

**Governing design:** [Cross-Platform STT Runtimes and Shared Model Artifacts](../specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md)

**ADR required:** yes

**ADR path:** [ADR-025](../../../backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md)

**Reason:** ADR-025 governs model storage, dependency readiness, interprocess
leases, STT provider boundaries, process ownership, persistence, dependency
profiles, routing, migration, and legacy removal.

## Planning policy

- TASK-404 is the only active delivery slice and has a complete executable
  implementation plan.
- TASK-405 through TASK-417 remain `To Do`. Each receives its own executable
  implementation plan only after it moves to `In Progress`, before production
  code changes begin.
- A task may be split further before implementation if its file map or
  acceptance criteria cannot remain one independently testable pull request.
- No task may weaken ADR-025 or silently cross an explicit dependency boundary.
- TASK-417 may not remove a legacy provider until all release gates pass for
  the same release candidate.

## Dependency map

| Task | Delivery slice | Depends on |
| --- | --- | --- |
| [TASK-404](../../../backlog/tasks/task-404%20-%20Prove-cross-platform-model-artifact-operation-leases.md) | Cross-platform shared/exclusive artifact leases | — |
| [TASK-405](../../../backlog/tasks/task-405%20-%20Qualify-Parakeet-v2-and-v3-INT8-artifacts.md) | INT8/F32/faster-whisper qualification | — |
| [TASK-406](../../../backlog/tasks/task-406%20-%20Build-shared-model-artifact-descriptors-and-lifecycle.md) | Artifact descriptors, immutable lifecycle, readiness, inventory, deletion | TASK-404 |
| [TASK-407](../../../backlog/tasks/task-407%20-%20Add-verified-managed-model-downloads-and-recovery.md) | Verified managed downloads, resume, staging, recovery | TASK-406 |
| [TASK-408](../../../backlog/tasks/task-408%20-%20Renovate-the-local-model-artifact-browser.md) | Curated/remote/installed artifact browser | TASK-407 |
| [TASK-409](../../../backlog/tasks/task-409%20-%20Add-bounded-local-GGUF-artifact-import.md) | Bounded GGUF local import | TASK-406, TASK-408 |
| [TASK-410](../../../backlog/tasks/task-410%20-%20Add-descriptor-backed-local-ONNX-bundle-import.md) | Descriptor-backed ONNX bundle import | TASK-406, TASK-408 |
| [TASK-411](../../../backlog/tasks/task-411%20-%20Introduce-provider-neutral-STT-contracts-and-coordinator.md) | STT contracts, registry, routing, coordinator, compatibility facade | — |
| [TASK-412](../../../backlog/tasks/task-412%20-%20Persist-STT-provenance-and-retry-lineage.md) | Transcript provenance and durable retry lineage | TASK-411 |
| [TASK-413](../../../backlog/tasks/task-413%20-%20Add-generation-fenced-local-STT-executor.md) | App-owned heavy executor, residency, generation fencing, child cleanup | TASK-404, TASK-406, TASK-411 |
| [TASK-414](../../../backlog/tasks/task-414%20-%20Integrate-Parakeet-ONNX-batch-routing.md) | Parakeet ONNX provider and batch routing | TASK-405, TASK-407, TASK-411, TASK-412, TASK-413 |
| [TASK-415](../../../backlog/tasks/task-415%20-%20Restore-bounded-Parakeet-ONNX-dictation-buffers.md) | Bounded dictation-buffer compatibility | TASK-414 |
| [TASK-416](../../../backlog/tasks/task-416%20-%20Add-curated-optional-transcribe.cpp-STT-provider.md) | Optional curated transcribe.cpp provider | TASK-407, TASK-411, TASK-412, TASK-413 |
| [TASK-417](../../../backlog/tasks/task-417%20-%20Promote-Parakeet-ONNX-defaults-and-remove-legacy-providers.md) | Default promotion, config migration, legacy removal | TASK-408, TASK-409, TASK-410, TASK-414, TASK-415, TASK-416 |

The two independent starting tracks are TASK-404 and TASK-405. TASK-411 can
also proceed without waiting for artifact storage, but Parakeet and
transcribe.cpp integration remain blocked on their artifact and process
dependencies.

## Planned file ownership

The detailed plan for each task may add focused tests or documentation, but
these module boundaries are authoritative unless a reviewed plan amendment
changes them.

| Task | Primary production ownership | Primary tests and evidence |
| --- | --- | --- |
| TASK-404 | `tldw_chatbook/Model_Artifacts/leases.py` | `Tests/Model_Artifacts/test_operation_leases*.py`, native CI matrix, `backlog/docs/model-artifact-operation-leases.md` |
| TASK-405 | `scripts/stt_eval/`, immutable corpus/result manifests | `Tests/STT_Eval/`, versioned benchmark reports |
| TASK-406 | `tldw_chatbook/Model_Artifacts/descriptors.py`, `store.py`, `service.py` | `Tests/Model_Artifacts/test_descriptors.py`, `test_store.py`, `test_service.py` |
| TASK-407 | `tldw_chatbook/Model_Artifacts/downloads.py`, `recovery.py` | Local HTTP-fixture download and crash-recovery tests |
| TASK-408 | `tldw_chatbook/UI/Screens/artifacts_screen.py`, artifact-browser view models | `Tests/UI/test_artifacts_screen.py`, service fakes |
| TASK-409 | `tldw_chatbook/Model_Artifacts/gguf_import.py` | Bounded parser, containment, TOCTOU, and cleanup tests |
| TASK-410 | `tldw_chatbook/Model_Artifacts/onnx_import.py` | Descriptor-backed bundle, corruption, containment, and rollback tests |
| TASK-411 | `tldw_chatbook/STT/contracts.py`, `registry.py`, `routing.py`, `coordinator.py`, `legacy_bridge.py` | `Tests/STT/test_contracts.py`, `test_registry.py`, `test_routing.py`, `test_coordinator.py` |
| TASK-412 | `tldw_chatbook/STT/provenance.py`, media and ingest-job migrations, export/import schemas | Database, pruning, transaction, export/import, and API contract tests |
| TASK-413 | `tldw_chatbook/STT/executor.py`, spawn-safe worker and child-process utilities | Process-generation, model-residency, lease, crash, cancellation, and child-tree tests |
| TASK-414 | `tldw_chatbook/STT/providers/parakeet_onnx.py`, catalog entries, ingestion integration, package extras | Routing, VAD, batch, package-resolution, provider-contract, and platform smoke tests |
| TASK-415 | Dictation controller integration over `LocalSTTExecutor` | Buffer, coalescing, backpressure, latency, cancellation, and coexistence tests |
| TASK-416 | `tldw_chatbook/STT/providers/transcribe_cpp.py`, curated GGUF catalog, optional extra | Per-family capability, artifact, buffer/file, crash, and platform tests |
| TASK-417 | Config migration, provider registration removal, dependency cleanup, release docs | Migration fixtures, stale-ID scans, full release-gate evidence |

## Cross-task invariants

- Omitted language resolves to `en`.
- Semantic default routing is explicit `en` to Parakeet v2, validated explicit
  non-English to Parakeet v3, and `auto`, unsupported languages, or translation
  to faster-whisper.
- Parakeet v3 language is routing-only under `onnx-asr==0.12.0`; it records
  `effective_language=auto` and `requested_language_not_enforced`.
- Parakeet INT8 is the default candidate; F32 is explicit. TASK-405 can block
  promotion without silently substituting F32.
- Cross-engine fallback is never silent. Eligible failures offer a new,
  provenance-linked **Retry with faster-whisper** request.
- The root artifact and every loaded dependency remain leased for the complete
  resident-model lifetime.
- Managed local ONNX import is descriptor-backed. Arbitrary graph parsing is
  outside these tasks.
- Managed v1 ONNX Runtime packaging is CPU-only; accelerator packages are not
  layered over CPU or `all-tools`.
- Production providers never download artifacts or write media database rows.
- Only the existing parent-side writer transaction persists parsed media.

## Active executable plan

- TASK-404:
  [Cross-Platform Model Artifact Operation Leases Implementation Plan](2026-07-23-model-artifact-operation-leases.md)
