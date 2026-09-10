---
id: TASK-32000
title: Plan and stage both restore destinations with explicit dependency mapping
status: Done
assignee: []
created_date: 2026-09-07 23:57
labels:
- backup-recovery
dependencies:
- task-31978
- task-31995
- task-31996
- task-31997
- task-31999
updated_date: 2026-09-10 20:23
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Both destination modes produce immutable explicit mappings with no writes to live sources or targets during planning/staging.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both destination modes produce immutable explicit mappings with no writes to live sources or targets during planning/staging.
- [x] #2 Replacement preserves unknown data until reviewed and includes managed retirement/rollback scope.
- [x] #3 Isolated restore works independently of damaged current config and rejects source aliases or untrusted destination authority.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute original component04 Task17 and ADR126 only. Validate immutable archive first; build explicit isolated/replacement restore/retire/preserve mappings from caller-supplied local destinations and independently discovered target inventory. Refuse partial replacement, unknown ownership, aliases/collisions/shared effects and changed target fingerprints. Stage validated dependency groups privately on target volumes with supported metadata, schema/domain/assets/credential policy and space checks, without writing live targets. Use existing archive/owner/SQLite/native types and helpers. TDD the original target-unverified regression and named focused filesystem scenarios; no publication, activation, UI, full suite or new feature scope. Record concrete limitations, review and verification before Done.
<!-- SECTION:PLAN:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-04-restore-recovery.md#task-17)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task17 implementation in progress in restore_plan.py/staging.py/test_restore_plan.py under ADR126. Original target_unverified regression recorded import and behavioral red then green. Latest focused cohort31passed2.45s, full new-scope Ruff clean, preliminary production Bandit0. Explicit local usernames are now required for selected configs, exact typed selector mappings verified against installed inert owner discovery. Source-derived synthetic roots, optional omissions, physical shared file candidates, SQLite WAL/SHM retirement scope, metadata disclosures, partial/dependency selection and target fingerprints are covered. Root-owned upstream producer metadata and restore-specific asset validation seams are coordinated separately; full asset-group integration and final report/checks pending. No publication/activation/UI or commits; do not mark Done yet.
Original Task17 asset dependency prerequisite: explicit restore-only logical/relative topology validation added to model recipes, persona/visual roots and recovered-media catalog/payload roles. Old capture validation remains unchanged. Model byte/dependency edges are checked, omitted bytes remain inert recipes; catalog raw payloads are not treated as SQLite. Agent realdata red3fail→green11, additional undeclaredmodel-edge red→fixed; final agent16newpass and26affectedold/newpass. Root independently reran16new:16passed18.95s (/private/tmp/chatbook-root-restore-assets-green.log), production fatalRuffclean/Bandit0. Added exact C82 read-only SQLite callsite inventory; literal-owner/policy checks2pass, roworder initiallyfailed then corrected and rerun. Report /private/tmp/chatbook-restore-asset-dependencies-report.md; ADR-126. Planner/stager integration and review corrections still in progress, no Task17 completion claim.
C82 inventory qualification finalized: documented newsite now follows C81 in stable order; exact expected connection-site range deliberately extended to82 with site rationale. Final stable-ID guard1passed1.81s (/private/tmp/chatbook-restore-assets-sqlite-ids-final.log); prior literal-owner and policy-link guards2passed. No guard bypass or broad connection allowance.
Task17 bounded implementation now in restore_plan.py, staging.py, and test_restore_plan.py; source frozen for root review. Final owned cohort62 passed6.42s (private basetemp, existing venv), full Ruff/format/diff clean; productionBandit0, testBandit73 expectedB101 only. Actual model/persona+core/recovered bundles pass both modes; selected-only nonemptycredential scopes and exact static config selectors covered. Explicit local missing-parent containers and optional durable journal.record_candidate handoff implemented. Known fixed/shared/package owners retained at new inert roots with owner_setup_required; activation/binding not claimed. Report /private/tmp/chatbook-task17-restore-plan-report.md contains exact contracts, evidence and multi-volume/custom-limit/downstream credential limitations. Remains InProgress; no AC checked, no commits/finalization.
Final review correction: untrusted synthetic=True could skip semantic Persona/visual root dependency validation. Exact installed _Assets now rejects a synthetic topology root before skip, including descendant items when container root is trimmed. Actual Persona/core fixture archives wrong referenced bytes with matching archive SHA: forged cases red2 failures (4 controls passed), then targeted6 passed60 deselected6.16s after fix. Full Ruff/format/diff clean; production Bandit0, test Bandit74 expectedB101 only. Report appended at /private/tmp/chatbook-task17-restore-plan-report.md. Parent must restage staging.py/test_restore_plan.py over prior index. No commit, AC change or task finalization.
Final Task17 review approved after correcting imported synthetic-root flag suppression of Persona asset dependency checks; regression red2 then green6. Root independently reran final complete named cohort:66passed9.80s (/private/tmp/chatbook-task17-final-green.log). Prior exact staged-index source-symbol architecture guard11passed12.15s; final correction adds no producer calls. Ruff/format/diff clean and productionBandit0. Both restore destinations produce immutable explicit local mappings and private staging, preserve independently classified unrelated data, refuse unknown ownership/partial replacement/aliases, include retirement coverage and validate real SQLite/assets/selected credential groups. Exact fixed/package/shared owner bindings remain explicit owner_setup_required for downstream Task20/21; no live publication or active relocation claimed for those owners. Actual second-volume qualification remains release evidence, not inferred from same-volume tests. ADR126; reports /private/tmp/chatbook-task17-restore-plan-report.md and /private/tmp/chatbook-task17-planner-review.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented and verified original Task17 immutable restore planning and private staging for isolated/replacement destinations. Explicit producer dependency/shared scope, independent target fingerprints, source preservation, container/metadata handling, owner schema/assets and selected credential validation are covered by66 focused passing cases and approved review. Activation, publication and final owner binding remain their original downstream tasks.
<!-- SECTION:FINAL_SUMMARY:END -->
