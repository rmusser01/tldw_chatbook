---
id: TASK-34806
title: Expose learned response rules in composer and scoped settings
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-04 05:55'
updated_date: '2026-10-12 23:51'
labels: []
dependencies:
  - TASK-34805
references:
  - Docs/superpowers/plans/2026-10-03-console-response-rules.md
documentation:
  - backlog/decisions/228-console-learned-response-rules.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enable the approved Console response-rule behavior for plan task 8. Preserve existing work and make failure states honest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Composer commands dispatch natively and refusals preserve the draft.
- [ ] #2 Chat, Workspace and global settings reach one scoped manager with explicit promotion and tested revisions.
- [ ] #3 Stop remains reachable during checking and token-backed controls render at 80x24 and 120x35.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/228-console-learned-response-rules.md. Reason: native commands and scoped local rule manager. Task 8 RED-GREEN: real composer command dispatch and draft preservation, shared Chat/Workspace/profile manager, public inactive Test then exact CAS Save, binding changes and promotion preview, Stop/status/focus, token-backed painted terminal layouts and targeted governance.

Current-dev PR3010 integration: follow Docs/superpowers/plans/2026-10-11-pr3010-current-dev-integration.md on the actual post-prior-merge base. ADR required: yes. ADR path: backlog/decisions/228-console-learned-response-rules.md. Reason: preserve the approved local storage/recovery, helper custody and shared native/hook acceptance; source ADR219 remapped after the older claimant. Reproduce reviewed failures before fixes; qualify actual next schema, six voice residues, exact private/index/table catalogs, preparation-through-transport custody, lazy first-use, current Goal/Personal Context owners and separately labelled real-provider/native UI evidence. Historical source qualification above is retained; reopened acceptance awaits current integrated verification and independent review.

2026-10-12 released Task1 on private START13b204cb638ca891175610f5e687a80def71c21e. ADR required: yes (existing). ADR path: backlog/decisions/228-console-learned-response-rules.md; preserves ADR227 Goal and current Personal Context/native ownership. Implement complete91path source workflow by current symbol. Qualify actual schema80-to81/four catalog variants/all predecessors, six voice residues, seven tables/three query/index pins/private import purpose, preparation-through-transport physical custody and original-owner usage, current Goal/hook/queue atomic receipts, lazy first-send/global-before-Chat, manager/draft/Stop/currentness and finite native worker retirement. Preserve all bounds and source history. Use targeted RED/GREEN; measure bounded critical cases for existing required job capacity, preserving inherited targets/order/caps/processes with executable membership negatives. Separate actual local-model/external-hook/native paint from scripted evidence. Root owns actual-base reconciliation, independent reviews, hosted proof and task closure; all AC remain open.

2026-10-12 scoped independent-review fix wave from05d975173621e116b6d92398344e92d3709b3238: F1 reproduce real SQLite writer contention on activation and sibling draft/assessment/binding mutations with actual synchronous Stop and source revocation, borrowed-handle survival and no stale publication. Acquire potentially blocking write admission before store/currentness locks, then recheck current source/binding CAS inside the existing short commit fence; retain locks through commit/rollback and preserve temporary publication/adoption. Audit every sibling lock/transaction caller; no timeout, native owner, cancellation capacity, schema or permission change. F2 reproduce public edit-Test-Save-Promote and edit-Test-Save-Disable-Enable, then align selected revision/row/details/test state with the committed revision. Preserve dirty edits on failed Save and scope/CAS checks. Run affected owner/cleanup and invalidated critical/membership proof on frozen source, scoped lint/format/diff, normal commit and one scoped re-review. ADR required: no new ADR; this directly enforces existing ADR228, current SQLite ownership and CAS contracts. All external/model/native/Goal/inherited/actual-base/hosted limits remain open. Original05 external report/handoff/164 sealed assets remain immutable; new fix artifacts are separate.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented native /omfg and /rules commands, shared Chat/Workspace/current-profile manager, exact tested CAS Save, private example inspection, promotion preview, exclusion and Stop/status/focus. Thin UI adapters retain app-owned runtime and token/component patterns; generated CSS rebuilt through its owner. Public targeted run: 55 passes plus the corrected active-run/Workspace witnesses (2 passes); real HTTP literal-send and saved-chat journeys pass. Static checks pass. Required full owner/governance and composer attempts retain independently verified baseline exclusions, documented in Docs/superpowers/reviews/2026-10-03-console-response-rules.md. ADR-228 governs storage, scope and admission; independent whole-branch review belongs to qualification task 34807.

2026-10-12 current integration (historical source evidence above remains unchanged): Composer commands, scoped manager, Console settings and canonical F4 Settings share one runtime. Token-backed UI preserves edits, dirty-cancel confirmation, disabled/tested pins, promotion privacy and focus. All four mounted learn/repair/reopen/exclude journeys passed in phase42 after current public workspace switching and actual button readiness fixes. Pilot is wiring evidence; native terminal paint is still pending because Computer Use explicitly denied Terminal. Existing ADR228 governs, with ADR097 live-revision GC and ADR227 Goal constraints applied. Current report: E/pr3010-current-integration/task-1-report.md. All tasks stay In Progress and acceptance remains open pending independent review, actual-base/hosted qualification and unmet external/model checks.

2026-10-12 final review F1/F2 fix wave: all six store mutation owners now obtain native SQLite writer admission before store/runtime revocation locks, then recheck source and binding CAS under a fence retained through commit/rollback. Temporary binding publication remains postcommit, and adoption/caller-owned transactions and physical DB cleanup owners are unchanged. Save reloads the exact committed revision, row/details and tested definition, retaining it for Promote and Disable/Enable. Genuine pre-fix RED: twelve contention cases fail (eight measured 0.516–0.614s stalls, two early invalidations causing Stop=false, two deferred write-upgrade failures); four public manager sequences repin revision1 after Save2. Mixed first receipt with teardown typo is retained separately. GREEN: original16 regressions16.20s; store/adoption/CAS/capacity/settlement40pass21.07s; manager/runtime/source/cleanup30pass47.91s; added critical and executable/negative membership226pass30.79s. Existing required P0 admission-sensitive command appends only the contention file and exact manager selector: 20 cases sum 22.341s including per-case setup/teardown in this run, within the measured ~7min headroom of the unchanged20min job (parent13m3 plus earlier rules~86s). No new process/cap/allowance or original13-selector omission/reordering. Existing ADR228 applies; no new ADR. Original05 sealed assets remain immutable; new final-fix receipts/report/handoff carry this wave. Final committed-head binding follows; model learning/semantic/tool quality, native paint/Checking-Stop, settled-Goal repair refusal, inherited759/756 guard and actual-base/hosted/re-review gates remain explicitly open. Tasks remain In Progress.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

Source TASK-34361 at codex/wave2-source-pr-3010 (73823fde3a4e) was created on 2026-10-04. The older current-dev task keeps 34361; this preserved younger response-rules task is TASK-34806. Original implementation history remains above; references and dependency identities follow this remap.
