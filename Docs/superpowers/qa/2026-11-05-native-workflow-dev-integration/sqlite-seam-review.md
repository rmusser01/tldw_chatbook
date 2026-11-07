# Narrow workflow catalog / private-SQLite seam review

Date: 2026-11-05. Read-only source and saved CI evidence review. No tests, product edits, subagents, Git mutations, profile operations, or CI actions. This supplements the earlier integration reports and does not close their findings.

**Mapping assessment:** the approved Ruling 15 composition is appropriate, and the current export's two production-file changes implement that narrow mapping without an additional source defect identified here. **Final QUALITY remains HOLD:** the writer has not frozen the complete delta or supplied its final scoped verification receipts. This is not a final-head approval or an all-green verdict.

## Introduced sites and current mapping

At `f4e72e54078413c35e01ea52607afe76283cc972`, four functions in `tldw_chatbook/DB/recovery_workflow_schema.py` directly opened `sqlite3.connect(":memory:")`: `workflow_admission_catalog`, `workflow_checkpoint_catalog`, `workflow_window_catalog`, and `workflow_repair_input_catalog`. The saved Windows CI raw-connection census explicitly reports these four additional sites. They are introduced workflow obligations; the unchanged normalizer does not excuse them.

The current export inspected during this review replaces exactly these four opens with `closing(connect_private_sqlite("recovery.workflow_catalog", ":memory:"))`, adding only the necessary imports in that module. At the time of inspection, their line numbers are 388, 414, 439, and 545. The private-SQLite production delta adds one policy at line 137: module `tldw_chatbook/DB/recovery_workflow_schema`, `_MEMORY` only, with default-false centralized backup, recovery capture, foreign read-only source, and read-only preservation flags. The inventory's C99–C102 rows match the four symbols, module, owner ID, and memory classification.

This is a proper use of the existing registry rather than a raw-open exemption. The existing seam validates the registered owner and classifies the exact `:memory:` target. Other target kinds fail before private-directory preparation; URI inputs are already rejected by the seam. The memory branch avoids path preparation and helper work. The existing static guard verifies literal IDs and module ownership. No change to that guard is required for these four sites.

`recovery_operations.py:1772–1801` composes these functions at import time from installed `_AGENT_RUNS_SCHEMA`, legacy catalog constants, and prior installed successor catalogs. Their inputs at the connected production call site are installed SQL, not imported candidate SQL or captured paths. Each function executes the installed catalog and migration, materializes a tuple of `sqlite_master` SQL, and returns it. Switching from the SQLite transaction context manager to `closing()` preserves the returned catalog and explicitly releases the connection on success or error. The prior implicit commit is unnecessary for a memory database whose only externally returned value is that fully materialized tuple. Do not replace `closing()` with a bare connection `with`, which would leave physical close implicit again.

No file/backup/capture authority, recovery validator bypass, schema contents, or captured-path execution was added by the inspected production delta. This review does not establish that the mutable export will remain identical to that delta at freeze.

## CI and upstream evidence, with limits

`ci-windows-plugin-f4e.txt` records **3 failed, 306 passed, 468 skipped in 402.76 seconds**. The failures are:

1. Raw connection census: the four workflow catalog sites plus `Backup_Recovery/detached_layout.py::_normalize_sqlite`.
2. Literal module-owned private seam IDs: only the detached normalizer's unqualified dynamic recovery validation call.
3. Direct backup census: only the detached normalizer's `connection.backup`.

Thus removing the four workflow sites is expected to leave the raw census red for the unchanged normalizer, as well as the other two guard failures. This is a source-based expectation, not an observed post-fix result. The 468 skipped cases remain unqualified.

I independently compared Git object bytes at pinned dev `3c39215749a96d5c1b8c05972fdd2aeb7588b002` and CI head `f4e72e54078413c35e01ea52607afe76283cc972`. All four files named in `upstream-sqlite-inventory-source-proof.json` are identical at those revisions, with the same hashes recorded there: the detached normalizer, private-SQLite inventory guard, private-SQLite registry/seam, and owner inventory. Reading the normalizer confirms the cited raw destination open, dynamic installed-owner validation call, and direct backup are present in those unchanged bytes.

This supports describing those normalizer violations as pre-existing in the pinned source, and keeping an unrelated normalizer repair outside this mapping. It does **not** establish that a pinned-dev test run passed, that a baseline run was performed, that every CI difference has been explained, or that the candidate CI is green. Existing protected-check requirements remain whatever the actual final CI reports.

## Remaining verification

At final freeze, recheck that the scoped delta remains the four replacements, one memory-only registry entry, and matching inventory/documentation. Review the writer's bounded catalog/recovery and inventory receipts, including explicit residual failure attribution. A focused policy check should establish file-target refusal without creation and absence of backup/capture authority; catalog checks should retain exact successor outputs and connection cleanup. No broad suite or unrelated normalizer change is required by this review. The earlier checkpoint and plugin/reference fixes require their separate frozen-delta review and evidence.
