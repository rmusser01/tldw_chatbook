---
id: TASK-33163
title: Add Console hook settings and consent review
status: Done
assignee:
  - codex
created_date: '2026-09-28 03:03'
updated_date: '2026-09-30 05:37'
labels:
  - hooks
  - console
  - settings
dependencies:
  - TASK-33151
references:
  - backlog/decisions/197-console-hook-configuration-review.md
documentation:
  - Docs/superpowers/specs/2026-09-27-console-hook-settings-and-review-design.md
  - Docs/superpowers/plans/2026-09-27-console-hook-settings-and-review.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let Console users inspect and approve hook commands before execution, and manage hook configuration in canonical Settings, following the approved native design and ADR-197.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A persistent Console Hooks action displays enabled pending and error states and opens exact hook details and current permissions with mouse and keyboard access at supported terminal widths.
- [x] #2 Existing, new, and changed enabled hooks require persisted exact-definition review on the next Send; cancelling or navigating preserves drafts and stale callbacks cannot send twice or into another chat.
- [x] #3 Shared foreground, queued, durable, recovered, and viewless admission and subprocess launches enforce consent, including revocation races, failed persistence, legacy duplicates, and notification target ownership.
- [x] #4 Canonical Settings provides staged hook add/edit/enable/disable/remove and Save/Revert, validation and impact copy, permission review and deep links, preserving concurrent edits and unknown configuration fields.
- [x] #5 Targeted runtime and UI checks, private storage and sensitive-path checks, portable-backup exclusion of hook consent, design-token and generated-style checks, and isolated live verification pass with documented limits and updated user guidance.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/197-console-hook-configuration-review.md
Reason: implement accepted ADR-197 and the reviewed spec; preserve ADR-148, ADR-033, and ADR-150.
1. Finish and self-review Docs/superpowers/plans/2026-09-27-console-hook-settings-and-review.md, grounding each interface and check in the actual repository.
2. At execution, reuse a suitable managed worktree or create one from the committed baseline; preserve the shared checkout and recheck caller seams before edits.
3. Implement lossless hook identity and config transactions, persistent consent and serialized launch authority, and shared Send/queue admission with targeted regressions.
4. Implement the native review modal, persistent Console action, and staged canonical Hooks Settings category using existing UI and token patterns.
5. Run the plan-scoped checks and isolated live verification; update user guidance, review the implementation, and record evidence before marking acceptance criteria complete.
Renumbering provenance: CLI assigned TASK-33152; the live all-ref/history and 87-worktree sweep found TASK-33162 as maximum, so the new task became TASK-33163 before cross-references were added.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the persistent Console Hooks action, expandable exact-command review, next-Send consent gate, and canonical Settings → Expert → Hooks staged editor. Shared foreground, queued, recovered, durable, and viewless admission plus subprocess creation enforce persisted exact-definition consent; cancellation preserves drafts. Guarded saves preserve unknown fields and concurrent edits. Consent uses the existing narrow private-file owner and is excluded from portable backup/restore. ADR-197, ADR-126, ADR-148, ADR-150, and ADR-097 apply; no new ADR, dependency, or DB migration.

Integrated the four implementation commits onto dev 857b3dd7d0 as codex/hook-review-current. Qodo fixes cover stale revoke/disable validation, invalid-master target publication, clean Settings reloads, central path validation, shared strict Pydantic execution/metadata validation, and API docs/typing. Raw profile selection is resolved under the config lease; stale bound-cache grants and same-file profile changes cannot authorize launches or Settings writes. Failed Disable remains fenced per ADR-197; explicit current review restores permission and is covered by regression. Post-await dismissal/shutdown guards and lazy hook imports/styles preserve original first-paint budgets.

Current targeted evidence: 194 runtime, 25 config/mounted UI, and 12 import/first-paint passes. Earlier complete 21-case boot and 37-case token/CSS groups cover unchanged portions; overlapping cohorts are separately named in QA. Authored-line Ruff/format is clean across 51 changed Python files, all three QA scripts pass full checks, generated CSS reproduces, and profile-path/diagnostic/worker/task governance passes. Real TldwCli passed all six private-profile groups at actual 80x24 and 120x40; four captures were visually inspected. Two broader Settings search assertions reproduce on current dev. Self-review and documentation are complete; no full suite or real provider generation ran. Named evidence, historical qualification, and limits are in Docs/superpowers/reviews/2026-09-27-console-hook-settings-and-review.md. PR #2922 review replies, hosted checks, latest-dev confirmation, and authorized merge remain pending.

Latest-dev qualification: clean rebase onto 2a74675eea preserved all eight patches exactly. Inspected upstream provider/config/Console seams. All 245 distinct targeted cases pass (195 runtime/config, 16 mounted UI, 21 complete boot group, 13 latency/stall-persistence). Authored Ruff/format across 51 changed Python files, full QA-script checks, and derived diagnostic/profile-path/CSS checks pass. Previous broader baseline controls and native captures retain 857b3dd7d0 provenance. Qodo reports zero active findings on the prior patch-equivalent published head; all review threads have evidence-backed replies. Feature implementation, self-review and documentation are complete. PR #2922 final integration remains gated on the rebased head hosted checks; no merge claimed.

Hosted PR Fast Lane passed the 1171-case main contract but its admission step has four ownership-harness failures: three constructor-bypassing screen fakes omit the new Hooks controller, and one resume backoff assertion counts the additional hook indicator callback. Reproduce the four cases, update test seams without weakening detach/backoff assertions, then rerun the exact admission group and static checks before publishing.

Hosted CI follow-up: main PR contract passed 1171 cases; UI Fast Lane and Perf Guard passed. Reproduced all four admission ownership-harness failures locally, supplied the new Hooks child to constructor-bypassing teardown fakes and isolated the reconciliation callback count. Original detach, settings-claim and bounded-backoff assertions remain. Four regressions and the exact 123-pass CI admission group pass; its one pre-existing xfail is retained, with no new skips. Combined distinct current-base local passes: 350. Production is unchanged; authored Ruff/format is clean across 52 changed Python files. Added the evidence-backed shared lifecycle testing lesson and updated named QA/report/plan results. Auto-merge is held until the new head passes hosted gates.

Latest integration base is now 5980da9c12 after dev advanced with TASK-33106 per-session MCP character-write refusal. Read its task/provider/Console seams; preserve ADR-183 alongside ADR-197/126. The clean rebase leaves all ten patches unchanged. This base passes 417 distinct named cases: 195 hook runtime/config, 123 exact repaired CI admission (one existing xfail), 96 MCP/character composition, and 21 boot-budget cases; overlapping viewless cases counted once. Authored Ruff/format across 52 changed Python files, QA scripts and derived diagnostic/profile/CSS artifacts pass. Earlier mounted UI/latency and native evidence retain recorded base provenance. Actual dev ref is checked because PR baseRefOid can lag. Feature remains complete; the rebased head hosted gates and authorized merge remain pending.
<!-- SECTION:NOTES:END -->
