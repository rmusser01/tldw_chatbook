---
id: TASK-27016
title: Publish approved Personal Context documentation design
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-01 15:07'
updated_date: '2026-09-02 06:38'
labels: []
dependencies: []
references:
  - Docs/superpowers/specs/2026-08-31-personal-context-documentation-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Publish and maintain an accurate Personal Context documentation design on Chatbook `dev` so both repositories can use one stable implementation reference. Correct the merged reference when implementation audit evidence shows that it overstates shipped behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The specification distinguishes reviewed first-link publication from the absent ongoing Personal Context sync caller and labels protocol or future behavior accordingly.
- [x] #2 The specification accurately documents adaptive-interview egress and disclosure timing, fixed-mode no-provider-call behavior, interview-to-record materialization, transport/TLS behavior, bootstrap disclosure, local removal/recovery limits, conflict surfaces, and incomplete purge distribution.
- [x] #3 Both products can reuse one exact four-bullet shared contract that states first-link convergence, queued later mutations, peer-local state, and the separate Shared Core/Sync V2 boundaries without implying ongoing convergence.
- [x] #4 Diff, scope, semantic, link, and duplicate-ID verification passes for the specification and this task record without application changes.
- [x] #5 The task records the existing-ADR disposition, exact verification evidence, implementation notes, and In Progress review handoff before a PR is opened.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-read the merged documentation design, ADR-102, and verified shipped Personal Context behavior.
2. Correct the specification so every lifecycle, interview, transport, deletion, recovery, conflict, and purge claim distinguishes shipped behavior from protocol capability or future intent.
3. Replace the exact shared four-bullet contract with first-link-only and current-limitation wording suitable for verbatim reuse in both products.
4. Run fail-closed diff, semantic, link, scope, and task-ID checks; record exact evidence and implementation notes.
5. Keep TASK-27016 In Progress until independent review and eventual PR closeout.

ADR required: no new ADR required; existing ADR applies.
ADR path: backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md
Reason: this correction changes documentation claims only; it does not change the architecture governed by ADR-102.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Published the reviewed Personal Context documentation design by applying commits 76dfa83343, 58afb082ac, 9ea1b2f134, and dd2d64bdf5. Normalized the spec metadata into semantic lists so the required whitespace gate preserves rendering.

ADR required: no new ADR required; existing ADR applies.
ADR path: `backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md`.
Reason: the accepted Personal Context ADR governs the implemented authority, sync, and encryption architecture; this task adds no architectural decision.

The required branch update exposed an older TASK-26836 on dev. Under the repository younger-task-renumbers rule, this publication record moved from TASK-26836 to the globally unused TASK-27016; the Renumbering provenance section records the timestamps, reason, and updated inbound references.

Verification evidence at 5fcc8243cc on origin/dev b17946c57a:
- backlog task 27016 --plain resolved the exact TASK-27016 file, status Done, and all three checked acceptance criteria.
- git diff --check origin/dev...HEAD exited 0.
- Exact-scope comparison exited 0 with only Docs/superpowers/specs/2026-08-31-personal-context-documentation-design.md and backlog/tasks/task-27016 - Publish-approved-Personal-Context-documentation-design.md.
- The all-ref sweep found TASK-27016 only on refs/heads/codex/personal-context-docs-spec at the renamed task path; the all-worktree sweep found it only in this isolated publication worktree. No distinct TASK-27016 claimant exists.
- Using the repository Python 3.12 environment, python -m pytest Tests/CI/test_backlog_task_id_uniqueness.py -q passed 3 tests; the system Python 3.9 invocation was discarded because this repository requires Python 3.11+.
- Both files are tracked, the task references the published spec path, and the spec has no task-ID reference requiring an update.
- No application test sweep was run because this is documentation-only.

Final publication verification at `145ac07d527aab6a75e6ffdb406d42b06a7c12f4`: the GitHub `No duplicate backlog task IDs` check passed in 16s, `PR Fast Lane` passed in 8m34s, and `Derived artifacts reproduce from their sources` passed in 5m47s. PR #2292 then merged to `dev` as `0b17f7f73cad28cdb5089aa5fff437b072e640c8`; GitHub Contents API returned the published spec blob `95ebb836330792afe8bf9b15c8eca074cb5294a9` and TASK-27016 blob `41fc737f284441491510bb4160c7687f80d1c30b` from `dev`.

Follow-up correction completed after post-merge quality review: the stale pre-merge sentence now records the completed final result, and every ADR disposition names the exact canonical Personal Context ADR path.

Follow-up verification before closeout on origin/dev `0b17f7f73cad28cdb5089aa5fff437b072e640c8`:
- backlog task 27016 --plain resolved the exact task file in In Progress with all three acceptance criteria still checked and the appended correction plan visible.
- git diff --check exited 0, and the exact-scope assertion found only the specification and TASK-27016 record.
- backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md exists and declares Status: Accepted; the obsolete disposition and pre-merge sentence are absent from the corrected files.
- The repository Python 3.12 environment passed all 3 targeted backlog task-ID uniqueness tests.
- The all-ref and all-worktree sweeps found only the same TASK-27016 filename and identity.
- No application test sweep was run because this follow-up is documentation-only.

Shipped-behavior truth correction prepared from current `origin/dev` `e167d0be2ec254595ecaa100c550d30930e645e7`. The specification now separates ADR-102's intended ongoing-sync architecture from the shipped first-link-only lifecycle. It records the absent ongoing Personal Context caller; Notes/Chat-only Manual Sync; adaptive-interview request contents and delayed provider/model disclosure; fixed-mode no-provider-call behavior; HTTP/TLS and Test Connection behavior; pre-approval bootstrap metadata; local removal, recovery-import, and key-cleanup limits; absent status/conflict UI; and incomplete purge distribution. It also supplies the exact four-bullet statement that both products must reuse.

ADR required: no new ADR required; existing ADR applies.
ADR path: `backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md`.
Reason: this correction changes documentation claims only and records verified implementation gaps; it introduces no storage, authority, synchronization, security, or runtime decision beyond ADR-102.

Verification evidence before independent review:
- A fail-closed semantic contract check passed all 19 required shipped-boundary claims, confirmed exactly four shared-contract bullets, resolved the three local reference paths, and rejected stale ongoing-sync wording.
- `git diff --check` exited 0, and the exact-scope assertion found only the specification and TASK-27016 record.
- `python -m pytest Tests/CI/test_backlog_task_id_uniqueness.py -q` passed all 3 tests under the repository Python 3.12 environment; pytest emitted sandbox cleanup warnings after the successful run.
- The all-ref and all-worktree sweeps found TASK-27016 only at the same canonical task filename and identity; no distinct claimant path exists.
- No application test sweep was run because the correction changes documentation and task metadata only.
- TASK-27016 remains In Progress for independent specification and quality review. No PR was opened.

Critical spec-review correction: the earlier publication matrix conflated peer-local interview draft/transcript objects with answer content later materialized into canonical profile records. `ProfileInterviewCoordinator._change_for_answer()` places the answer in an ordinary payload and defaults the proposed controls to `syncable` and `agent_visible`; saving reviewed changes commits those records through the normal Personal Context service. The specification now states that draft/transcript objects are not Sync payloads as such, while approved answer content follows normal record visibility and syncability: eligible content can publish in the reviewed first-link snapshot, queue in the encrypted outbox after linking, or remain local when device-only.

Review-fix evidence:
- Before the wording change, the negative semantic control failed on the matrix's blanket exclusion of both draft objects and raw answer text from profile Sync.
- After the change, the same control rejects the former blanket draft/answer and whole-mode-local claims while requiring the draft/transcript object boundary, record materialization, first-link publication, later outbox, and device-only outcomes.
- TASK-27016 remains In Progress for re-review. No PR was opened.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

- Previous ID: TASK-26836
- Current ID: TASK-27016
- Reason: current dev contains the older `task-26836 - Console-tray-recomposes-for-state-fields-its-content-mode-never-renders.md` record (created 2026-09-01 14:51); this publication record was created at 2026-09-01 15:07 and therefore moved under the younger-task-renumbers rule.
- Inbound references: the specification contains no task-ID reference; the filename, frontmatter ID, task verification commands, and exact-scope evidence were updated to TASK-27016.
