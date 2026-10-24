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
updated_date: 2026-10-24 06:55
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A managed Parakeet installation admitted for an earlier audio batch unconditionally updates whichever Import form is current when it finishes. A later batch explicitly set to Faster Whisper is changed to Parakeet INT8, and the success message claims it selected the current batch. Keep installation completion from replacing newer current-batch choices.

Priority: Medium (P2), rank 5 of five in this audit. See the report for impact and ordering.

Observed evidence: Actual install button -> consent modal -> thread worker -> original call_from_thread completion. Batch B remains staged and actual Pilot Select chooses faster-whisper before release; completion changes current provider to parakeet-onnx and says selected for this batch. Ordinary A control passes. Preflight/provision/prefer_managed are qualified stubs; no actual artifact installation, persistent preference, inference or Start-import proof. Precision was already int8 and no folder override existed in the recorded before-state; folder/precision changes remain future regression controls, not observed losses.

Existing scope: TASK-931/TASK-1696 own ordinary post-install selection. TASK-32755 and its readiness children own selected-provider admission fences, not the managed-install callback. TASK-32755.4 / PR #3221 changes the fresh Auto default and effective-model preview, not installation completion; its full ACs and Library patch were inspected. TASK-32664 owns folder-picker continuity; TASK-598/TASK-602 own artifact/runtime policy, which remains unchanged. TASK-34000.33 owns other import defects. TASK-32755.5 / PR #3221 binds Auto readiness and queued execution to the displayed model; its full ACs and current code preserve all install callback ASTs and do not cover this obsolete-completion defect.

Related tasks: [TASK-931](task-931%20-%20Install-verified-Parakeet-v2-INT8-from-Library.md), [TASK-1696](task-1696%20-%20Port-the-Parakeet-v2-adapter-managed-first-resolver-and-plan-driven-Library-modal.md), [TASK-32755](task-32755%20-%20Match-audio-video-import-warnings-to-the-selected-STT-backend.md), [TASK-32755.1](task-32755.1%20-%20Validate-selected-transcription-packages-and-models-before-import.md), [TASK-32755.3](task-32755.3%20-%20Review-all-severities-and-qualify-transcription-readiness-for-merge.md), [TASK-32664](task-32664%20-%20Review-Parakeet-import-directory-controls-and-picker-continuity.md), [TASK-598](task-598%20-%20Use-descriptor-verified-external-Parakeet-ONNX-bundles.md), [TASK-602](task-602%20-%20Integrate-Parakeet-ONNX-batch-routing.md), [TASK-602.1](task-602.1%20-%20Stage-gated-Parakeet-v3-batch-routing.md), [TASK-34000.33](task-34000.33%20-%20Library-import-real-document-titles-clean-local-HTML-cancellable-batches-disclosed-model.md), [TASK-32755.4](https://github.com/rmusser01/tldw_chatbook/blob/6dd14ec8683de2baeee205cbf8a016e5a7b6fe1e/backlog/tasks/task-32755.4%20-%20Use-base-for-fresh-Auto-imports-and-preview-the-effective-transcription-model.md), [TASK-32755.5](https://github.com/rmusser01/tldw_chatbook/blob/6dd14ec8683de2baeee205cbf8a016e5a7b6fe1e/backlog/tasks/task-32755.5%20-%20Keep-Auto-import-execution-bound-to-its-displayed-model.md).

ADR check: no new ADR expected for a bounded repair within existing owners and contracts. Applicable decisions: [014-library-ingest-service-authority-and-recovery](../decisions/014-library-ingest-service-authority-and-recovery.md), [025-shared-stt-artifacts-and-runtime-routing](../decisions/025-shared-stt-artifacts-and-runtime-routing.md), [050-external-parakeet-roots-with-managed-vad](../decisions/050-external-parakeet-roots-with-managed-vad.md), [033-application-session-state-ownership](../decisions/033-application-session-state-ownership.md), [011-chatbook-workbench-ui-system](../decisions/011-chatbook-workbench-ui-system.md). Re-evaluate if implementation changes a boundary.

[Textual book audit: evidence, controls, limits and overlap review](../../Docs/superpowers/qa/2026-10-23-textual-book-after-pr3219.md). The audit records the pre-fix behavior; implementation evidence is recorded below.
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
CI aggregate follow-up: the relocated install error notification interpolates runtime text with Rich markup enabled. Add a real notification-rendering RED control, render failure text literally without changing consent/activation/preferences/selection, remove only the obsolete old-wrapper unsafe census row, append the control to the required census, and run markup/compatibility/boot checks plus fresh Superpowers review. Do not add an unsafe allowlist row. ADR required: no; existing ADR-025/033/011/097 apply. See the latest 2026-10-23 pending-draft plan section.
<!-- SECTION:PLAN:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented and independently reviewed in PR #3228; targeted acceptance evidence complete. Final hosted CI is followed in the PR. Existing task/ADR links retained; no new ADR.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Managed Parakeet completion retains its requesting form identity, accepted-edit generation, source and raw transcription controls. Automatic INT8 selection applies only while that request context is current. Obsolete completion keeps later choices and reports that honestly, while consented installation/preference can finish. The original button, consent, worker and UI-thread callback remain. The narrow parakeet_install_selection helper loads only on request/completion; LibraryScreen stays at 37,525 lines with its lowered ceiling unchanged.

All nine new selection cases and 25 existing contracts pass across the recorded targeted runs. After import deferral, ordinary/provider-change mounted controls and both exact size guards pass (4/4). Perf Guard exposed 559/558 preload modules; the existing native guard reproduced it, then scoped imports restored 558/558 without raising a limit. Rebased preimport/warmup controls pass (2/2,558/558 and93/94). External acquisition/preference seams are bounded; no download, package install or inference proof is claimed. See the [verification report](../../Docs/superpowers/qa/2026-10-23-textual-pending-draft-followups.md) for evidence and limits. Existing ADR-014/025/050/033 application state/011 govern selection and [ADR-097](../decisions/097-boot-budget-ratchets.md) governs import timing; no new ADR. Independent Superpowers review approved the implementation.

Implementation acceptance and independent review are complete in [PR #3228](https://github.com/rmusser01/tldw_chatbook/pull/3228). All changed production files are identical to the reviewed rebased head whose hosted [Perf Guard](https://github.com/rmusser01/tldw_chatbook/actions/runs/40020015425) passed; the subsequent Dictionary fixture repair passed four native controls and scoped review. Final task/documentation metadata is bundled before the final CI run. Hosted checks remain a delivery gate and their final exact-head results are recorded in the PR check rollup and description. Targeted local verification only; no full local suite.
Current-dev integration: after rebasing onto dev d0d926e11b (related PR #3221), the combined LibraryScreen exceeded the existing 37,525 ceiling at 37,551 lines. The complete existing pure ingest-options projection now lives beside LibraryIngestFormState in its already imported state module; the screen retains its delegate. Auto displayed-model/submission, preference and Parakeet consent/activation/selection bodies are unchanged. LibraryScreen is now 37,493 lines/1,354 methods with its line ceiling lowered to 37,493; prior 37,525 statements describe the earlier checkpoint. Final native combined-source selection passes 80 cases (101.251s JUnit), including Auto/readiness, snapshot/submission, mounted ordinary/provider-change Parakeet and boot/size guards. Preimport 558/558, warmup 93/94; no new lint diagnostics or raised budget. ADR-025/033/097 and the linked QA report govern this mechanical extraction; no new ADR. Fresh scoped review and current-head hosted results are recorded in PR #3228.
Required-aggregate follow-up: install-error notifications render literal text with markup disabled, preserving the original message/severity and consented artifact/preference/selection flow. A real App.notify/Toast control reproduces MarkupError for markup-shaped text before the repair and renders it literally afterward. The obsolete old-wrapper markup pin is removed; no unsafe replacement pin is added. The complete targeted selection passes213 cases, including ordinary/provider-change mounted install controls, boot and size guards; the exact native markup guard passes. Existing ADR-025/033/011/097 and the linked QA report apply. Fresh scoped Superpowers review and final-head hosted CI remain the delivery gate.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->