---
id: TASK-41564
title: Bound dev CI with required smoke, nightly regression and release qualification
  tiers
status: In Progress
created_date: 2026-11-13 21:54
assignee:
- '@codex'
updated_date: 2026-11-17 09:45
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Shorten normal development feedback without losing nightly regression coverage or allowing unqualified release candidates. Use hosted runners independently of the canceled self-hosted CI project.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The existing dev required context always reports and fails closed on preflight, smoke and selected affected checks; high-risk or unclassified changes retain broad regression coverage.
- [ ] #2 One immutable changed dev SHA receives nightly broad regression coverage; unchanged successful SHAs do not repeat, failures retry, and scheduled execution is installed on the default branch before lighter gating is enabled.
- [ ] #3 Promotion to main and publication require full-suite and packaging evidence for the exact candidate; failures, cancellations and skipped expected checks cannot qualify.
- [x] #4 Focused policy tests cover event routing, changed-path selection, unknown/deleted files, failed or missing evidence and rollout disabled behavior; hosted runner routing and queue semantics are preserved.
- [x] #5 Document the tier policy, coverage tradeoffs, rollout and rollback; keep opt-in disabled until live qualification and default-branch schedule activation succeed.
- [ ] #6 A second live check confirms required status association, conservative routing, fail-closed results, nightly and release evidence, branch protections and hosted runner selection before activation; mapped-source PR and post-merge smoke timings are then measured against the successful full baseline.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/360-tiered-development-and-release-ci.md
Reason: Changes required-check selection, nightly scheduling and release qualification boundaries; amends ADR-103; the owner canceled self-hosted CI on 2026-11-17.
1. Add a conservative changed-path selector and explicit reviewed source-to-test map; retain broad coverage for risk-sensitive, deleted or unknown inputs and keep smoke/artifact checks unconditional.
2. Wire selection into existing required aggregation behind CI_TIERED_ENABLED, preserving all existing broad test commands, queue events and runner routes.
3. Add changed-SHA nightly orchestration and exact-candidate promotion/release qualification using reusable existing suites; make publication depend on the full gate.
4. Add focused behavioral and workflow contracts, run targeted checks, review the diff and prepare a separate draft PR without competing with #3529's baseline.
5. Document rollout requiring an active default-branch schedule and live evidence before enabling the lighter policy; record remaining live qualification explicitly.
6. After #3529 merges, rebase only this policy commit onto actual dev and renumber this policy task without changing the already-merged Node TASK-40727. Preserve #3532's controlled benchmark window. Independent integration review found that the new smoke job also needs the baseline's explicit Node 24 setup; extend its existing executable-contract test before adding that step, then re-run focused workflow and package checks.
7. Obtain current-head Qodo review, resolved threads and a successful normal full PR gate before queue integration. Install only the nightly dispatcher on default main through a separate reviewed bootstrap PR; qualify real immutable-SHA nightly coverage, actual scheduled operation, unchanged-success deduplication and failure retry.
8. Qualify comprehensive tests and package/install checks on one exact release candidate through a safe PR or reusable workflow without publishing to PyPI/TestPyPI or merging a product release merely to test. Verify required-check association and strict branch protection/admin enforcement while preserving unrelated rules.
9. Repeat the live prerequisite checks before enabling the opt-in, then measure normal mapped-source PR selection, broad fallback and post-merge smoke. Clear the opt-in if activation misbehaves; do not mark Done before live acceptance is complete.
10. Owner direction on 2026-11-17 supersedes the migration/benchmark waits in steps 4 and 6: finish light CI on GitHub-hosted runners. Rebase only policy commits onto current dev, preserve its new native recovery and Backup_Recovery coverage in the tier routes, reallocate the colliding policy ID, and obtain a new hosted full baseline. Provide safe exact-candidate release qualification through a non-publishing dispatch. Continue nightly, protection, second-live-check and activation acceptance as planned.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the opt-in three-tier workflow policy under ADR-360 (amending ADR-103, preserving ADR-359). The stable required context aggregates preflight, representative subsystem smoke or the existing broad lanes, and exact-candidate release qualification. Preflight now requires the same pinned Ruff F gate and debt ratchet. Broad test commands, matrices, deadlines and package build steps were compared against the parent and preserved exactly. Package build is reusable, and both publish paths require comprehensive-test success. The nightly dispatcher selects changed dev SHAs, retries failures, and qualifies every smoke selection before opt-in; Library shell coverage moves from every dev push to nightly only when enabled.

Validation: exact combined light-job shell invocation with CI contracts and all selected smoke targets: 1974 passed, 1 intentional child-test skip in 156.48s locally. Earlier isolated smoke union: 291 passed in 68.98s. CI plus package metadata contracts: 1706 passed, 1 intentional skip. Final shallow-checkout/selector and derived-workflow contracts: 118 passed in 24.33s. Actionlint, Ruff for new Python, source F gate/debt ratchet, whitespace, task/ADR validators pass. Self-review checked fail-closed missing/skipped results, fork routing, nightly identity/retries, reusable-workflow concurrency and exact event-SHA checkouts. No application full sweep or publication run.

Selection is representative subsystem coverage, not inferred dependency closure; high-risk names, shared inputs, new/deleted sources and unknown subsystems retain broad coverage. Exploratory whole legacy chat/navigation modules exposed existing profile-admission fixture failures and missing mock fixtures; product tests were not altered or removed. The selected maintained seams passed, while comprehensive qualification still owns the whole tree. This is a deliberate coverage/cadence tradeoff documented in scripts/ci/README.md.

Pending live acceptance after #3529 merged: normal full gate on this policy PR; install and qualify regression-nightly.yml on default main; safe exact release candidate/package qualification without publication; strict required CI on main (currently only deletion/force-push rules); enable CI_TIERED_ENABLED only afterwards. Opt-in is confirmed unset. No branch protection or server configuration was changed. The original policy monitor was removed on 2026-11-17. Task remains In Progress; Qodo exception for #3529 is not inherited.


2026-11-14 integration: #3529 merged as d46f753d8287cffd48706f912bfd838dc8756edb after normal full run 43612634465 succeeded (51 successful jobs, one expected skip). Required aggregate 132525514742 completed 17:22:07Z: 1h28m09s from workflow creation, 1h26m50s from first job execution; 626.43 runner-minutes. Initial queue wait 79s; individual job creation-to-start and pre-creation dependency scheduling intervals are retained separately. Last job 17:23:40Z is distinct from required completion and workflow updated 17:23:41Z. All-paginated evidence: /tmp/tldw-ci-policy-monitor-first-success-{run,jobs,job-pages,job-timings,summary}.json. This thread did not modify the owner's #3529 work or baseline closeout.

Replayed only original policy commit d2e63a950e onto that actual dev merge as f0281a6a2344808da6a1e41b0f95273c314bbb82; no conflicts. Independent tiered_rebase_review confirmed all 12 existing broad/Library job bodies unchanged except conditions, unchanged comprehensive job bodies except the duplicate manual-suite condition, and identical extracted package build. It found the new dev-light job omitted the Node 24 prerequisite already required by the baseline. The existing provisioning contract now covers both broad and light jobs: before the workflow correction, only light failed (1 failed/1 passed in 0.54s); after the same setup-node@v4/Node 24 step,431 routing/policy cases passed in 2.62s. Scoped review cleared the correction. No production, deadline, isolation, coverage or runner-resource change.

Rebase validation: 757 focused workflow/package contracts passed in 31.70s; final exact light-job shell with all mapped selections passed 1,977 tests with one intentional child-test skip in 184.02s locally. The earlier run retained separately failed two ID checks before renumbering completed and a localhost bind refused by the sandbox (1974 passed/3 failed/1 skip in 195.89s). The final run used the completed tree and authorized loopback access, with unchanged assertions. Six changed workflows passed actionlint with the existing custom-label config; new/touched Python passed Ruff and format; source F gate and debt ratchet passed (1089 findings against 1,105 baseline, no baseline change). 6,578 task files are unique/readable; 318 ADR files unique. Exact commands and receipt paths: /tmp/tldw-ci-policy-rebase-validation-commands.txt; final smoke /tmp/tldw-ci-policy-rebase-smoke-final.log. No full local application sweep or release publication was run. These are local timings, not the future light-CI server result.

The policy remains unpushed while #3532's separately owned controlled benchmark window is pending. Recheck both owning chats, actual dev and MERGE_QUEUE before publication. #3533 is still draft/unarmed on its old published head; no Qodo review or review threads exist there. The current-head Qodo exception applies only to #3529. Safe release qualification, default-main schedule operation, nightly all-job evidence/deduplication/retry, protections and the second live check remain mandatory before activation.

2026-11-17 resumption: Ruling: self-hosted CI and its benchmark are canceled by the owner; light CI continues independently on hosted runners. The current task plan and ADR-360 supersede the historical waits above. Rebased only the two policy commits from d46f753d onto actual dev 5e3c8b2bd4f70e420147cccce53efc918b715c8e. Preserved the newly merged required macOS plugin-recovery lane and Backup_Recovery dev suites; broad/nightly routing includes both, and lighter protected dev pushes defer the owner suites to nightly. No test body, matrix or deadline was removed. Initial focused integration contracts passed 686 cases in 64.94s. The light job now selects GitHub-hosted ubuntu-latest directly. Added a non-publishing release=true dispatch to qualify comprehensive tests and package/install checks at one immutable event SHA; the required aggregate rejects unsuccessful qualification. The new contract failed before that dispatch existed and passed after it was added. All remaining live ACs stay open.
Final hosted integration review found two Important issues: the release input was missing from two existing workflow-contract predicates, and nightly dispatches shared the merge-queue kick concurrency group. The contracts now match the release route; nightly and release dispatches have their own group suffixes. The new concurrency contract failed before the workflow fix, and the corrected tiered/response-rules/preflight set passed 113 cases in 16.12s. Independent follow-up verified three focused cases in 0.86s and cleared both findings, with no remaining code blockers. The first exact combined smoke attempt retained 1,992 passes, five old-predicate contract failures and one intentional child-test skip in 432.26s; all five failing contracts plus the three new hosted/release/concurrency contracts passed together after correction (8 passed in 1.38s). A corrected exact combined invocation is running separately. Actionlint, new Python Ruff/format, zero-F source gate, debt ratchet (1,090 <=1,105), task uniqueness/readability (6,696 files), ADR uniqueness (318), and whitespace checks passed. The minimal default-main dispatcher is independently published as PR #3578 at 891245ec572cc2b549765cf6d56451be516a5636, unarmed pending policy integration. Qodo responded there that workspace reviews are paused for exhausted credits; its current-head rule is still applicable, with no exception inherited from canceled work.

2026-11-17 live release correction: the owner explicitly approved independent review in place of credit-blocked Qodo for #3533 and nightly bootstrap #3578. The corrected exact light shell passed 1,998 cases with one intentional child-test skip in 402.12s locally. #3533 normal full gate 44142808884 passed preflight and remained healthy. Qualification-only draft #3579, run 44145393952 at e41bbd501244121aa9461b5cb749e43dee708253 (the same reviewed source tree), exposed a Windows checkout failure before lease tests: job134262613338 could not create a long committed QA evidence filename. Reuse the existing Windows pwsh `git config --global core.longpaths true` step before artifact-lease-spike checkout, matching the already-qualified plugin and backup workflows. The new order/condition regression failed before the fix (1 failed0.88s), then both affected workflow contract modules passed70 cases5.96s and actionlint passed. Independent scoped review cleared the fix and independently passed its regression0.43s; native Windows success remains pending. Matrix, test commands/assertions and deadlines are unchanged. Disabled #3533 auto-merge and pulled/rebased its exact remote head before editing, per queue rules. Prior policy and release attempts are retained separately before the head update; neither is a successful full baseline.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Renumbering provenance

Renumbered this unmerged policy task from TASK-40727 to TASK-40998 on 2026-11-14 at the user's explicit direction to preserve the Node task already merged through #3529. The recorded creation dates are policy 2026-11-13 21:54 and Node 2026-11-13 22:08; this follows that explicit direction rather than claiming the Node frontmatter is older. The allocation scanned 742 local/remote refs (195 remote refs), 61 worktrees and 11,122 distinct task blobs, including filename and frontmatter claims across task buckets. Maximum observed stem 40968; the repository helper applied its 30-ID buffer to allocate 40998. ADR-360 and scripts/ci/README.md references move with this policy; Node references remain TASK-40727.

2026-11-17: Renumbered the unpublished policy again, from TASK-40998 to TASK-41564, because the separately published migration task also claimed 40998. That migration is now canceled; its published ID remains reserved. Refreshed scan covered 1,114 refs (206 remote refs), 893 distinct backlog trees, 74 worktrees and 11,929 task blobs, checking filename stems and frontmatter across task buckets. Global maximum was 41534. The existing helper, with the buffer adjusted for the higher global maximum, allocated 41564, 30 above that maximum. Evidence: /tmp/tldw-ci-policy-hosted-id-sweep.json. Active policy references in ADR-360 and scripts/ci/README.md now use TASK-41564.

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
