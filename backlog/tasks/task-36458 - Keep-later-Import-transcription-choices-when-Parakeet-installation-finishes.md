---
id: TASK-36458
title: Keep later Import transcription choices when Parakeet installation finishes
status: In Progress
created_date: 2026-10-23 16:40
labels:
- textual
- book-audit
priority: medium
references:
- backlog/tasks/task-931 - Install-verified-Parakeet-v2-INT8-from-Library.md
- backlog/tasks/task-1696 - Port-the-Parakeet-v2-adapter-managed-first-resolver-and-plan-driven-Library-modal.md
- backlog/tasks/task-32755 - Match-audio-video-import-warnings-to-the-selected-STT-backend.md
- backlog/tasks/task-32755.1 - Validate-selected-transcription-packages-and-models-before-import.md
- backlog/tasks/task-32755.3 - Review-all-severities-and-qualify-transcription-readiness-for-merge.md
- backlog/tasks/task-32664 - Review-Parakeet-import-directory-controls-and-picker-continuity.md
- backlog/tasks/task-598 - Use-descriptor-verified-external-Parakeet-ONNX-bundles.md
- backlog/tasks/task-602 - Integrate-Parakeet-ONNX-batch-routing.md
- backlog/tasks/task-602.1 - Stage-gated-Parakeet-v3-batch-routing.md
- backlog/tasks/task-34000.33 - Library-import-real-document-titles-clean-local-HTML-cancellable-batches-disclosed-model.md
- https://github.com/rmusser01/tldw_chatbook/blob/6dd14ec8683de2baeee205cbf8a016e5a7b6fe1e/backlog/tasks/task-32755.4%20-%20Use-base-for-fresh-Auto-imports-and-preview-the-effective-transcription-model.md
- https://github.com/rmusser01/tldw_chatbook/blob/6dd14ec8683de2baeee205cbf8a016e5a7b6fe1e/backlog/tasks/task-32755.5%20-%20Keep-Auto-import-execution-bound-to-its-displayed-model.md
- tldw_chatbook/UI/Screens/library_screen.py
- tldw_chatbook/Widgets/Library/library_ingest_canvas.py
documentation:
- backlog/decisions/014-library-ingest-service-authority-and-recovery.md
- backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
- backlog/decisions/050-external-parakeet-roots-with-managed-vad.md
- backlog/decisions/033-application-session-state-ownership.md
- backlog/decisions/011-chatbook-workbench-ui-system.md
- Docs/superpowers/qa/2026-10-23-textual-book-after-pr3219.md
assignee:
- '@codex'
updated_date: 2026-10-23 19:31
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A managed Parakeet installation admitted for an earlier audio batch unconditionally updates whichever Import form is current when it finishes. A later batch explicitly set to Faster Whisper is changed to Parakeet INT8, and the success message claims it selected the current batch. Keep installation completion from replacing newer current-batch choices.

Priority: Medium (P2), rank 5 of five in this audit. See the report for impact and ordering.

Observed evidence: Actual install button -> consent modal -> thread worker -> original call_from_thread completion. Batch B remains staged and actual Pilot Select chooses faster-whisper before release; completion changes current provider to parakeet-onnx and says selected for this batch. Ordinary A control passes. Preflight/provision/prefer_managed are qualified stubs; no actual artifact installation, persistent preference, inference or Start-import proof. Precision was already int8 and no folder override existed in the recorded before-state; folder/precision changes remain future regression controls, not observed losses.

Existing scope: TASK-931/TASK-1696 own ordinary post-install selection. TASK-32755 and its readiness children own selected-provider admission fences, not the managed-install callback. TASK-32755.4 / PR #3221 changes the fresh Auto default and effective-model preview, not installation completion; its full ACs and Library patch were inspected. TASK-32664 owns folder-picker continuity; TASK-598/TASK-602 own artifact/runtime policy, which remains unchanged. TASK-34000.33 owns other import defects. TASK-32755.5 / PR #3221 binds Auto readiness and queued execution to the displayed model; its full ACs and current code preserve all install callback ASTs and do not cover this obsolete-completion defect.

Related tasks: [TASK-931](task-931%20-%20Install-verified-Parakeet-v2-INT8-from-Library.md), [TASK-1696](task-1696%20-%20Port-the-Parakeet-v2-adapter-managed-first-resolver-and-plan-driven-Library-modal.md), [TASK-32755](task-32755%20-%20Match-audio-video-import-warnings-to-the-selected-STT-backend.md), [TASK-32755.1](task-32755.1%20-%20Validate-selected-transcription-packages-and-models-before-import.md), [TASK-32755.3](task-32755.3%20-%20Review-all-severities-and-qualify-transcription-readiness-for-merge.md), [TASK-32664](task-32664%20-%20Review-Parakeet-import-directory-controls-and-picker-continuity.md), [TASK-598](task-598%20-%20Use-descriptor-verified-external-Parakeet-ONNX-bundles.md), [TASK-602](task-602%20-%20Integrate-Parakeet-ONNX-batch-routing.md), [TASK-602.1](task-602.1%20-%20Stage-gated-Parakeet-v3-batch-routing.md), [TASK-34000.33](task-34000.33%20-%20Library-import-real-document-titles-clean-local-HTML-cancellable-batches-disclosed-model.md), [TASK-32755.4](https://github.com/rmusser01/tldw_chatbook/blob/6dd14ec8683de2baeee205cbf8a016e5a7b6fe1e/backlog/tasks/task-32755.4%20-%20Use-base-for-fresh-Auto-imports-and-preview-the-effective-transcription-model.md), [TASK-32755.5](https://github.com/rmusser01/tldw_chatbook/blob/6dd14ec8683de2baeee205cbf8a016e5a7b6fe1e/backlog/tasks/task-32755.5%20-%20Keep-Auto-import-execution-bound-to-its-displayed-model.md).

ADR check: no new ADR expected for a bounded repair within existing owners and contracts. Applicable decisions: [014-library-ingest-service-authority-and-recovery](../decisions/014-library-ingest-service-authority-and-recovery.md), [025-shared-stt-artifacts-and-runtime-routing](../decisions/025-shared-stt-artifacts-and-runtime-routing.md), [050-external-parakeet-roots-with-managed-vad](../decisions/050-external-parakeet-roots-with-managed-vad.md), [033-application-session-state-ownership](../decisions/033-application-session-state-ownership.md), [011-chatbook-workbench-ui-system](../decisions/011-chatbook-workbench-ui-system.md). Re-evaluate if implementation changes a boundary.

[Textual book audit: evidence, controls, limits and overlap review](../../Docs/superpowers/qa/2026-10-23-textual-book-after-pr3219.md). This task does not claim the repair is implemented.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Completion of an earlier managed Parakeet install cannot replace later accepted provider, model-folder or precision choices, including choices for a changed source/batch.
- [x] #2 An unchanged current request retains successful Parakeet INT8 automatic selection. Installation and consented artifact activation/preference policy can finish even when its original form selection is obsolete.
- [x] #3 Current form values, visible controls and success/status copy agree; an obsolete completion never claims to have selected the current batch when that batch choice was retained.
- [x] #4 Bounded mounted controls retain the actual install button, consent modal, background worker and original UI publication owner, covering ordinary and later-choice cases while stubbing only external acquisition boundaries without downloads or inference.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the existing managed-install request, consent, worker and UI publication against real source and option change ownership.
2. Add mounted regression controls for unchanged success and later accepted provider, precision, model-folder and source/batch choices, including queued Changed delivery; stub only external acquisition boundaries.
3. Capture the original selection context at the install request and fence only automatic form selection when newer intent supersedes it; allow consented install and managed preference to finish with honest status copy.
4. Run focused RED/GREEN controls, relevant existing Parakeet/ingest tests and static checks; preserve receipts and record implementation evidence for root integration.
ADR required: no
ADR path: backlog/decisions/014-library-ingest-service-authority-and-recovery.md; backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md; backlog/decisions/050-external-parakeet-roots-with-managed-vad.md; backlog/decisions/033-application-session-state-ownership.md; backlog/decisions/011-chatbook-workbench-ui-system.md
Reason: Bounded callback ownership repair within existing screen, consent, artifact and preference contracts; no storage, routing or service-boundary change.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Managed Parakeet completion now captures its requesting form identity, accepted-edit generation, source path, transcription choices and live controls. It selects Parakeet INT8 only while that context remains current; obsolete completion retains current editors and gives honest retained-choice copy while consented installation/preference can finish. The original UI-thread callback and zero-argument installer seam remain. Existing completion body moved to UI/Library_Modules/parakeet_install_selection.py to reduce LibraryScreen size; no new service/storage/runtime boundary.

Added nine mounted regression controls covering ordinary success, unchanged prior F32/folder override, later provider, precision, folder, source/batch, provider ABA and genuine queued Changed delivery. Final targeted verification: 34 passed (33 combined including 8 new +25 existing, plus the ninth unchanged-override case separately), 0 failures/errors/skips. Qualified RED independently measured provider overwrite, f32-to-int8 overwrite and folder removal. Actual install button, consent modal, background worker, real ingest preflight, event handlers and original call_from_thread publication retained. External plan/provision/preference and optional runtime availability are qualified seams; no downloads, package installation, inference, Start-import or real persistent preference proof claimed.

Ruff format/check for new files and py_compile for modified Python files pass; tracked diff whitespace check passes. Existing legacy test-file unused App import is pre-existing F401; scoped lint ignoring F401 passes. Full suite not run. Detailed root cause, exact selectors, raw receipts, setup failures and limits: .test-tmp/task-36454-36458/parakeet-report.md. Root owns shared architecture docs, diagnostic inventory/census/CI integration, final review and Done status.

ADR required: no. Existing ADRs 014, 025, 050, 033 and 011 apply (see Implementation Plan for paths). The change is a bounded callback ownership repair and preserves artifact policy, model routing and source preference contracts.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
