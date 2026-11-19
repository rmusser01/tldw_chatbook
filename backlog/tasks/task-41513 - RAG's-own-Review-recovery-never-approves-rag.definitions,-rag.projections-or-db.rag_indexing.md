---
id: TASK-41513
title: "RAG's own Review recovery never approves rag.definitions, rag.projections or db.rag_indexing"
status: Done
assignee: ['@claude']
created_date: 2026-11-17 05:27
updated_date: 2026-11-18 04:03
labels:
- follow-up
- backup
- rag
priority: high
references:
- tldw_chatbook/Backup_Recovery/recovery_service.py
- Tests/Backup_Recovery/test_rag_model_recovery_controls.py
dependencies:
- TASK-41301
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the bkreview fix round (round 15) once the test's line-121 failure was fixed (TASK-41301). The `[rebuild]` node of `test_rag_model_recovery_controls.py` shows that RAG's own Review recovery never approves `rag.definitions`, `rag.projections` or `db.rag_indexing`; it was red at base, hidden behind the earlier failure. A user who follows the RAG recovery path cannot finish it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Following RAG's own Review recovery approves the owners it needs and the RAG index becomes usable
- [x] #2 The `[rebuild]` node passes, red before the fix
- [x] #3 The review copy names what was approved
<!-- AC:END -->

## Implementation Plan

1. Reproduce the `[rebuild]` node at base, with the child budget it needs, and find why the RAG owners are not approved when the test checks them.
2. If the product never approves, fix the approval; if the test checks before the approval worker has finished, make it wait for the approval's outcome.
3. Name what was approved: the review dialog's owner list and the result notification use the owners' installed labels.
4. Docs: the RAG recovery steps in `Docs/Backup-and-Recovery.md` and the Settings ▸ RAG User Guide page.

## Implementation Notes

**Summary.** RAG's own Review recovery does approve `rag.definitions`, `rag.projections` and `db.rag_indexing`, and the index then becomes usable. The `[rebuild]` node was red for two other reasons: its check raced the approval worker, and its child budget is too short at load. The test now waits for the approval's outcome, and the review copy names what it approves, in the dialog and in the message after it.

**Why the node was red (base 45b1949ca3).**
- **Budget.** `test_settings_separate_rag_review_and_explicit_rebuild` failed at load 31-33 with the isolated child's `subprocess.run(..., timeout=25)`: `timed out after 25 seconds`. With that budget raised for the local run only, it passed 1 of 1 (47.7 s, load 33) and 3 of 3 concurrent runs (45-46 s, load 39-41), including the search over the rebuilt index.
- **The race (the filed finding).** The test pressed the dialog's confirm and checked the approvals after `await host.workers.wait_for_complete(); await pilot.pause()`. The approval worker is started by the dialog's dismiss callback, after the press is handled, so `wait_for_complete()` had no worker to wait for, and `pilot.pause()` returns once the process is idle (at most 1 s), which a descheduled worker thread does not prevent. Under a temporary 3 s `time.sleep` in `approve_recovery_review`, the base test failed: `AssertionError: (['Local model review recorded. ...'], [('rag.definitions', False), ('rag.projections', False), ('db.rag_indexing', False)])`, the filed symptom. With a 10 s wait added before the check, the same run passed: the approval lands, just later.

**Changes.**
- `RAG_Search/activation.py`: `RAGRecoveryReview.owner_names`, the reviewed owners' installed labels (`recovery_service.owner_label`, the names Backup & Restore shows).
- `UI/Screens/settings_screen.py`: the review dialog lists `Owners: Retrieval indexing records, Retrieval definitions, Retrieval indexes` instead of the owner ids, and the message after approval names what `approve_recovery_review` returned: "Approved RAG owners: <names>. No index was rebuilt; local model setup remains separate. Use Reconcile / rebuild when prerequisites are ready." (`markup=False`, as it interpolates). +2 lines; 34,978 against the 34,993 ceiling.
- `Tests/Backup_Recovery/test_rag_model_recovery_controls.py`: the rebuild route asserts the dialog's owner line, waits (up to 120 s) for the approval's outcome message, and asserts it names the three owners before checking the store. `_run_ui` gives the mounted Settings child 200 s instead of the isolated child's 25 s: under `_run`'s 240 s backstop, and `_run`'s 60 s idle check still catches a hang. That budget covers this file's other UI nodes too; they share the harness.
- Docs: `Docs/Backup-and-Recovery.md` (RAG step 2 names the owners and the message) and `Docs/User_Guide/settings/rag.md` (a short "After a restore" section for the three recovery buttons, which the page did not cover).
- `backlog/docs/lessons-testing-evidence.md`: the race, which was filed as a product bug.

**Red/green.**
- Copy (AC#3), red on the base screen and model: `test_settings_separate_rag_review_and_explicit_rebuild` failed at the new dialog assertion (`'Owners: Retrieval indexing records, ...' in host.screen.message`; the dialog still listed owner ids). Green after.
- Race (AC#2): with the same temporary 3 s sleep in `approve_recovery_review`, the fixed test passed 1 of 1 (40 s, load 35-41), where the base test failed.
- `[rebuild]` and the whole file: 11 passed at `-n 4` (174 s, load 56-98): approve, the eight boundaries, rebuild (the index is searchable afterwards) and ordinary.
- `test_activation_rag.py -k review`: 4 passed (the review dialog built from a real `RAGRecoveryReview`).

**Lane.** `test_rag_model_recovery_controls.py` stays on no PR lane (nightly-deep only), by the coordinator's decision: its children import torch, transformers and chromadb, and the PR lanes install no extras.

### Review r16rs-rev1, fix round 1

- **Minor 1.** `Docs/User_Guide/settings/rag.md`'s "After a restore" no longer says each of the three buttons only reviews: the first two review or approve, and **Reconcile / rebuild** loads the model and rebuilds.
- **Minor 2.** `_run_ui` asserts both strings it replaces are in `_ISOLATED` before replacing them (control: with `timeout=26)` as the budget string, both nodes failed at that assert), and its comment says the 240 s backstop counts from the outer child's start, so the outer setup also comes out of it. `-k "rebuild or cancel"`: 3 passed.
