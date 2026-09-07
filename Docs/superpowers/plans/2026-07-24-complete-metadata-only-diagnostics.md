# TASK-494: Complete metadata-only diagnostics implementation plan

> **Execution:** follow the repository TDD and verification workflows. Treat
> the TASK-492 inventory as the checked source of truth and regenerate it only
> after reviewing every owner/topology change.

**Goal:** Complete ADR-029 for all production diagnostic domains outside the
high-risk provider/tool work, including alternate sinks and third-party
records.

**Architecture:** The primary private rotating file sink admits only strict
Chatbook metadata records. Non-persistent UI and terminal handlers remain
descriptive. Unused alternate Loguru file-sink setup is fail-closed instead of
bypassing the application boundary. The checked inventory fingerprints all
production logging owners and persistent topology. A remaining-domain sentinel
matrix validates the boundary against representative owners and real rotation.

**Tech stack:** Python 3.11, stdlib logging, Loguru, pytest.

## ADR check

- ADR required: no
- ADR path: `backlog/decisions/029-local-private-data-boundary.md`
- Reason: TASK-494 completes the already accepted metadata-only persistence
  decision without introducing a new diagnostic or storage architecture.

## Task 1: Close alternate and third-party sink bypasses

**Files:**

- Modify: `tldw_chatbook/Utils/persistent_diagnostics.py`
- Modify: `tldw_chatbook/Metrics/logger_config.py`
- Test: `Tests/Metrics/test_logger_config_privacy.py`
- Test: `Tests/test_remaining_diagnostic_sentinel_matrix.py`

1. Write failing tests showing a third-party sentinel can reach the primary
   file sink and the legacy metrics setup can create unfiltered Loguru files.
2. Reject all non-Chatbook records at the persistent file handler while
   leaving UI/terminal handlers unchanged.
3. Disable the unreferenced direct Loguru file-sink parameters; retain console
   setup and an explicit warning instead of silently creating an unsafe sink.
4. Verify that strict approved metadata still persists through the primary
   application handler.

## Task 2: Verify every remaining production domain

**Files:**

- Test: `Tests/test_remaining_diagnostic_sentinel_matrix.py`

1. Parameterize representative RAG/search, ingestion, media/database,
   Notes/sync, subscription/web, and UI/application-orchestration owners.
2. Exercise standard logging and Loguru normal, debug, and error records with
   query/content/path/config/credential/response/exception sentinels.
3. Force rotation and assert no active or rotated generation contains a
   sentinel while operation/status/count/duration metadata remains.
4. Prove a non-persistent collecting handler still receives third-party
   diagnostic text rejected by the file handler.

## Task 3: Reconcile the checked inventory and task

**Files:**

- Regenerate after review:
  `Docs/security/production-diagnostic-inventory.json`
- Modify:
  `backlog/tasks/task-494 - Complete-metadata-only-boundary-across-remaining-production-diagnostics.md`

1. Run the source guard before regeneration and inspect every changed owner or
   sink entry.
2. Regenerate the inventory only after confirming the metrics bypass has
   disappeared and the primary topology remains guarded.
3. Run focused tests, relevant domain suites, lint/static checks, and
   `git diff --check`.
4. Check every acceptance criterion, add implementation notes linking ADR-029
   and this plan, and set TASK-494 to Done only after all evidence is green.
