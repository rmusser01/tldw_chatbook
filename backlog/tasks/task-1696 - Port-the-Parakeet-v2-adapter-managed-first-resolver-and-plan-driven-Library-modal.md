---
id: TASK-1696
title: >-
  Port the Parakeet v2 adapter, managed-first resolver, and plan-driven Library
  modal
status: Done
assignee:
  - '@claude'
created_date: '2026-08-01 07:02'
updated_date: '2026-08-01 10:41'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reconciliation item 3: the merged TASK-595 downloader has zero production consumers — no descriptor exists, so nothing in the app can download anything. Port from codex/task-595-managed-downloads-v2's design: the Parakeet v2 module becomes a thin adapter supplying the first exact descriptor (pinned repo/revision/license/files/sizes/digests) plus its source map; a model-directory resolver prefers the active managed artifact, then a verified legacy .tldw-verified.json bundle, with explicitly configured directories highest priority and never described as integrity verified; the existing Library install modal renders values from the immutable plan rather than hard-coded constants, keeping its current controls and post-install batch selection. Most of what TASK-1301 needs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Parakeet v2 INT8 installs end-to-end through the shared downloader from the existing Library action
- [x] #2 Console dictation resolves configured dir, then active managed artifact, then verified legacy bundle
- [x] #3 The Library modal's content derives from the preflight plan
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read design spec (codexclone/task-595-v2) sections on default store/STT adapter, download flow, Library/Console behavior, and the reconciliation review item 3.
2. Map existing code: parakeet_v2_installer.py pinned constants, library_screen.py ParakeetV2InstallModal + worker, console_dictation._resolve_model_dir, transcription_service._load_parakeet_onnx_model, Model_Artifacts service.py/acquisition.py APIs.
3. Build tldw_chatbook/Local_Ingestion/parakeet_v2_artifact.py: ArtifactDescriptor + credential-free source map built from the installer's pinned constants (single source of truth), a minimal ArtifactCatalog, a shared managed store root sibling to legacy models/stt/..., an active-managed-artifact resolver using only Model_Artifacts.service, and preflight/provision orchestration helpers that import Model_Artifacts.acquisition only locally (never at module scope).
4. Wire console_dictation._resolve_model_dir and transcription_service._load_parakeet_onnx_model to the new managed-first resolver (configured dir -> active managed -> verified legacy -> existing error), keeping the console/transcription worker import boundary (never import acquisition/fetch).
5. Rewire the Library modal + handlers to run preflight in a background worker, render ParakeetV2InstallModal from the PreflightReport, and grant+provision on confirm, keeping modal ids/worker names/post-install contract intact.
6. Write/extend tests: descriptor/source-map/catalog unit tests, an end-to-end localhost-fixture-server preflight->grant->provision test, modal report-rendering tests, resolver-order tests for both console and batch paths, and boundary-test extension.
7. Run the full explicit gate and adjacent suites; verify no regressions.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Ported the merged managed-download layer's first production consumer.

New: tldw_chatbook/Local_Ingestion/parakeet_v2_artifact.py -- builds the exact ArtifactDescriptor and credential-free per-file source map from parakeet_v2_installer's existing pinned constants (imported, never copied); a minimal ParakeetV2Catalog; the shared managed-store root (get_user_data_dir()/models/managed, sibling of legacy models/stt/...); active_managed_parakeet_v2_dir() using only Model_Artifacts.service (list_installed()); and run_parakeet_v2_preflight/run_parakeet_v2_provision orchestration helpers that import Model_Artifacts.acquisition LOCALLY (inside their own function bodies) so the module stays import-safe for worker-side callers. Provenance is (CHATBOOK_CURATED, LOCAL_INTEGRITY_RECORDED) -- see review fix round below.

Changed: Audio/console_dictation.py and Local_Ingestion/transcription_service.py (:806-830 batch path) now resolve, in order: explicitly configured dir (unchanged validation) -> active managed artifact (via the new resolver) -> verified legacy .tldw-verified.json bundle -> existing error text. Both stay off the async acquisition/HTTP import graph (pinned by the extended Tests/Model_Artifacts/test_credentials_and_boundaries.py boundary test, now also covering parakeet_v2_artifact and console_dictation).

Changed: UI/Screens/library_screen.py -- handle_parakeet_v2_install_requested now runs preflight in a background worker (group 'library_parakeet_v2_preflight') before showing ParakeetV2InstallModal, which renders repository/revision/license/precision/per-artifact+total bytes/destination/free-space verdict/gating_errors entirely from the injected PreflightReport (no hard-coded constants). Confirm grants consent and provisions in the existing 'library_parakeet_v2_install' worker (kept zero-argument per the pinned UI test), then sets transcription_provider/transcription_model_dir to the ACTIVE MANAGED directory exactly as before. Modal ids and the post-install batch-selection contract are unchanged.

Kept: install_verified_parakeet_v2 and verify_parakeet_v2_bundle (legacy verifier) untouched, still used for migration/fallback reads.

Tests: new Tests/Local_Ingestion/test_parakeet_v2_artifact.py (descriptor-matches-installer, source-map, catalog, managed-root-sibling, resolver-without-acquisition, and an end-to-end preflight->grant->provision run against the Tests/Model_Artifacts localhost fixture server with tiny monkeypatched files). Extended Tests/UI/test_parakeet_v2_install_ui.py, Tests/Audio/test_console_dictation.py, Tests/Transcription/test_parakeet_onnx_vertical_slice.py, Tests/Model_Artifacts/test_credentials_and_boundaries.py.

Gate green (commit ae1a23fba): 461 passed on the pinned gate; test_parakeet_v2_artifact.py (10) and the vertical-slice suite (25) verified green separately (numpy absent from shared venv).

REVIEW FIX ROUND 1 (commit ec276b553): reviewer checked HuggingFace's tree API for the pinned revision and found only 2 of 4 declared files (the LFS-tracked ONNX weights) carry a repository-supplied SHA256; config.json/vocab.txt are plain git blobs (git SHA1 oid only), so those two pinned digests were necessarily computed locally. Per ADR-025 the honest per-artifact label for a mixed artifact is the weaker one -- changed provenance from (CHATBOOK_CURATED, INTEGRITY_VERIFIED) to (CHATBOOK_CURATED, LOCAL_INTEGRITY_RECORDED); verification behavior unchanged (every file still checked against its pinned digest). Added a test pinning the exact tuple with a comment explaining why. Also disabled ParakeetV2InstallModal's Install button when the report is ungrantable (gating errors or insufficient space) instead of letting the user confirm a plan that would immediately fail -- same button id, reason already shown inline. Gate re-run green: 471 passed.
<!-- SECTION:NOTES:END -->
