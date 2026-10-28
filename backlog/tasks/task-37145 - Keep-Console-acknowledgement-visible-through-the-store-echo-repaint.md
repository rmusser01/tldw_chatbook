---
id: TASK-37145
title: Keep Console acknowledgement visible through the store echo repaint
status: Done
created_date: 2026-10-28 15:49
assignee:
- '@codex'
labels:
- ci
- textual
- bug
priority: high
references:
- TASK-33620.5
- TASK-33620.15
- TASK-34982
- TASK-15453
- TASK-32522
- Tests/UI/test_console_send_acknowledgement.py
- tldw_chatbook/Widgets/Console/console_transcript.py
updated_date: 2026-10-28 16:24
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR3283 UI shard1 on cc907deb18 recorded four missing-user compositor samples during the acknowledgement-to-store-echo replacement and blocked the required aggregate. Completed TASK-33620.5 already documents a remove/mount residue but its observed one-frame maximum is not a native guarantee. Close that residual continuity gap while preserving acknowledgement authority and existing cancellation-safe reconciliation. This is not a repeat of the original pre-admission feedback or detached-send work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A controlled native mounted replacement keeps the acknowledged USER message in the last committed frame until the exact store echo can appear once; the regression fails before the fix with no unrelated setup errors or skips.
- [x] #2 Cancellation, unmount, reorder and session-switch behavior preserve existing reconciliation ownership and removal completion, and repainting resumes after a cancelled replacement.
- [x] #3 The original end-to-end acknowledgement gate and focused transcript controls pass without raising its frame allowance, adding sleeps/retries, changing message identities, or changing guard caps, dependencies, polling or timeout policy.
- [x] #4 Focused Superpowers review and local verification qualify the repair; relevant documentation records the actual compositor evidence and existing ADR ownership. Actual successor-head PR CI remains a separate delivery gate.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the pinned CI gap review, existing acknowledgement/removal tasks, ADR-094/230, design language and relevant native testing lessons. 2. Establish a deterministic real Transcript/acknowledgement/echo regression by holding a native Unmount during interval-driven reconciliation; inspect actual last compositor commits, not re-rendered suspended DOM. Preserve all release/cleanup in finally and prove product RED. 3. Apply the smallest native repaint transaction around reconciliation while preserving refresh lock, finish_removal, row identity/reuse/order and ownership checks. 4. Run the new witness, cancellation/repaint control, original failing acknowledgement node and targeted existing timer-removal/reorder/session controls. Keep caps/allowances unchanged. 5. Update relevant docs/incident lesson, task notes and acquire fresh scoped Superpowers review. Root owns current-dev integration, CI enrollment/balancing and successor-head CI. ADR required: no. ADR path: N/A. Reason: routine rendering transaction preserving existing runtime/store/view boundaries under existing ADR-094 and accepted ADR-230; no polling redesign or new durable authority.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The transcript now wraps awaited row reconciliation in Textual App.batch_update inside the existing refresh lock. Its last committed compositor frame survives native remove-before-mount replacement; cancellation unwinds the repaint batch while the unchanged finish_removal continues native teardown. Row identities/reuse/order and retirement checks remain unchanged. The source stays 8,652 lines (2 additions/2 deletions); no guard caps, frame allowances, timeout policy, dependencies or polling changes.

TDD: on cc907deb187f4e3de99e9f1bab2f73983ac1870a, ack-37145-red03 drove a real ConsoleSendAcknowledgement into the exact USER store-echo projection with production CSS, set_interval/set_messages/refresh_messages and a held real native Unmount. It failed on 8 missing active-screen compositor commits (1 failure, 0 errors/skips). Earlier red01/02 used widget-consolidated CSS only; red03 is the authoritative full-CSS witness. All holds release/drain in finally. The original core recorder now uses compositor-commit wording and emits time/update/ack/IDs/row geometry context on continuity failure; its original len(gaps) <= 1 assertion is intact.

Final focused verification, prepared external native-profile launcher, ack-37145-final02: 12 passed, 0 errors/skips, 90.04s suite wall. This includes both new echo/cancel regression nodes, the exact original stopped-paint-timer native node, and all 9 original acknowledgement-core nodes. Cancellation proves a separate control commits a repaint before the held native removal is released, followed by exactly one echo and no stale acknowledgement. Seven unchanged original removal/steady-state/reorder/session/branch control bodies also passed through disposable private-profile wrappers (ack-37145-controls01, 165.06s); those are wrapper entries, not unchanged-original-node passes. Wrapper source is preserved with that receipt and the temporary Tests/UI entry file is removed. Initial green01 was 3 native passes plus 3 pre-setup RecoveryRequired(raw_source_selection_changed) errors from plain helper nodes, not a green 6-case run. No ACL, marker, config-admission or private-path guards were weakened.

Scoped AST and diff-check pass. Both touched tests pass Ruff lint/format checks. Transcript retains exactly the baseline 3 E721 findings and formatter debt; its existing size pin is 8,353 against 8,652 baseline/current lines (299 over before this task), unchanged. Static baseline/current source hashes and actual parent/child JUnit path/hash/count/timing receipts are saved under .test-tmp/textual-37031-37034/ack-37145-{static,verified-receipts}.json. The CI four-sample failure remains valid; corrected local baseline03's 45.52s pass did not waive it.

Updated Docs/Architecture/console.md and the actual TASK-37145 compositor incident in backlog/docs/lessons-testing-evidence.md. Existing ownership: TASK-33620.5 Known residue; TASK-34982 cancellation-safe removal. ADR required: no; routine rendering transaction under [ADR-094](../decisions/094-console-turn-lifetime-and-navigation-boundary.md) and [accepted ADR-230](../decisions/230-console-send-preparation-and-io-ownership.md). Broader Proposed ADR-231 polling redesign is excluded. No live terminal-write, elapsed blank-duration, natural race-frequency, provider-latency or whole-send certification is claimed; no full suite was run.

Implementation/self-review complete; task stays In Progress with AC4 open for the root's fresh scoped Superpowers review, current-dev integration and actual successor-head PR CI/delivery closure. Full report: .superpowers/sdd/2026-10-27-textual-37031-37034/ci-ack-37145-report.md.
Fresh scoped Superpowers review of f9d4ad7a29 approves spec/code quality with zero Critical/Important findings. Reviewer independently verifies24XML files and reconstructs historical full-CSS RED03 tool-source provenance; root independently parses actual RED/final/wrapper receipts. Final12nodes pass0F0E0S;7temporary-wrapper control bodies pass with their exact qualification. Native batch repair is locally complete. Root owns latestDev798 integration/enrollment and actual successor-head hosted CI as a separate pending PR delivery gate. Portable report/evidence: Docs/superpowers/qa/2026-10-27-textual-followups-implementation.md. Inherited warnings,3E721/format debt and299-line baseline pin overage remain explicitly qualified; no cap or original frame allowance changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
