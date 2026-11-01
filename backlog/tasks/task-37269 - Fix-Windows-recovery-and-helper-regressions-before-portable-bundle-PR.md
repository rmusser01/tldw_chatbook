---
id: TASK-37269
title: Fix Windows recovery and helper regressions before portable bundle PR
status: Done
assignee:
  - '@codex'
created_date: '2026-10-28T22:53:00Z'
updated_date: '2026-10-29T00:18:00Z'
labels:
  - windows
  - recovery
  - testing
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the remaining Windows recovery and package-helper failures so the portable bundle can be reviewed with reproducible targeted evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Previously failing encrypted-helper, archive and restore cases pass under the Windows private test environment without bypassing runtime safety checks.
- [x] #2 Native alias and private-metadata regressions exercise real Windows filesystem behavior without requiring administrator-only symlink privileges.
- [x] #3 A fresh isolated launch reopens real persisted data with correct profile/bootstrap selection and unchanged original files.
- [x] #4 Affected recovery and portable checks, static checks and review evidence are recorded before opening the portable bundle PR.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR paths: existing ADR-125, ADR-126, ADR-127 and ADR-256
Reason: correct demonstrated bugs/test assumptions within existing recovery and helper contracts.

1. Reproduce helper qualification and encrypted caller failures in the isolated Windows test runtime.
2. Correct native metadata/alias and installed-schema fixture assumptions without weakening safety assertions.
3. Trace and correct fresh-child home/bootstrap selectors; exercise actual isolated launch and data-alias refusal.
4. Run affected recovery files plus the portable cohort and static checks; requalify the artifact if runtime bytes change.
5. Review, record exact results and open the portable bundle PR on a codex/ branch.

Detailed plan: Docs/superpowers/plans/2026-10-28-windows-recovery-test-fixes.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Resolved the recorded Windows/helper regression failures without changing production recovery or crypto behavior. The isolated helper dependency was missing in the earlier environment: a disposable interpreter reproduced unisolated PYTHONPATH success and isolated -I transform_failed, while the fully provisioned private runtime qualifies the unchanged hash-anchored worker. Full crypto tests now use native Windows junction/named-pipe/sharing-denial fixtures, CRLF diagnostics, native ACL metadata and recursive child RSS. Explicit streaming-budget cases replace a vacuous default-limit test; bypassing the real pipe budgets makes both fail.

Archive/planner tests now inspect installed native metadata, exercise actual unprivileged NTFS junction refusal with unchanged target bytes, and compare the untouched historical schema-76 fixture's migrated candidate to installed schema84. Six Persona cases use the existing native seed helper rather than an intentionally POSIX-only publisher. Recovery tests inherit owner-private temporary storage. Fresh launch binds HOME and USERPROFILE consistently, asserts native reparse refusal, and avoids bytecode writes in its disposable package copy using -B without increasing the original timeouts or bypassing authority.

Verification: frozen combined affected recovery + portable cohort361 passed, one intentional duplicate-ZIP warning,1210.67s (output/windows-portable-verified-final.log). Packaging/native-builder18 passed plus5 subtests. Full-file controls: archive48, planner110, crypto44; fresh CLI/data alias2 passed. Strict lint on new runtime/package and corrected tests, fatal lint on legacy fixtures, format and whitespace checks pass. Legacy archive style diagnostics predate this change; planner import diagnostics were removed. Fresh independent whole-branch review found no Critical,Important or Minor findings.

The first combined attempt was interrupted when a sandbox write grant made a worktree ancestor shared-writable:325 passed then one refusal/35 setup errors. Guards correctly refused; it is not counted green. Subsequent collection also exposed sandbox-owned source files lacking workspace-owner access after narrowing inheritance. Only this task-owned tree was corrected and source readability/native ancestors were checked before the successful frozen rerun; primary checkout and runtime guard policy remained unchanged.

ADR required:no new decision. Existing ADR-125/126/127 and accepted portable ADR-283/255/256 govern the behavior. Provisional portable ADR IDs252/253/254 were renumbered because current dev already uses252/253. Decision content and approval are unchanged; native patch bytes/provenance stay cd9196d6b7df2e620d906d9e748654ddef604b28672f5f72fcb7b4fc5112669b, now protected by LF checkout policy.

Plan/evidence/rulings: Docs/superpowers/plans/2026-10-28-windows-recovery-test-fixes.md. Existing exact-ZIP/native GUI qualification remains historical to798f8fca75+portable source; these fixture fixes require no new binary. Final dev integration, fresh startup recheck and PR publication are in progress. No full sweep was run under repository opt-in policy.

Final integration: rebased onto dev3efe8476ce and preserved both lesson entries in the one documentation conflict. Recovery/DB/config/dependency/packaging/test bytes were unchanged across the rebase. Packaging18 tests+5 subtests and native patch provenance passed again.

Post-rebase cold startup exposed a remaining functional-fixture timing assumption. The test now performs one actual CLI launch, then the public profile-selection/admission path for data assertions instead of importing the full TUI twice. Bounds reflect actual native work (restore120s, cold CLI90s, read process150s); this is not a startup performance SLA. The complete isolated-restore/model-import run passed20 cases in178.14s. An explicit child observer captures selectors before argument parsing, asserts the exact selected profile/control pair at exit, and rejects an unselected child even after argv consumption (old observer RED rc0, fixed GREEN rc74). The final real CLI/data/client-identity case passed in64.13s. Existing privacy, source-byte, authority and activation assertions are preserved.

All requested failure groups are resolved. Combined cohort361 passed with one intentional duplicate-ZIP warning; final affected-file and packaging checks, strict/fatal lint scopes, format/whitespace and independent review are recorded in the plan. No product crypto/recovery guard changed, no safety cases skipped and no full sweep run. Canonical portable ADRs283/255/256 supersede their provisional numbers. The historical171MB ZIP remains the qualified798f8fca75+portable artifact; a newly rebuilt dev-tip GUI or clean-machine release is not claimed. Ready for portable PR publication on codex/windows-portable-wezterm-bundle.
<!-- SECTION:NOTES:END -->

### 2026-10-31 review-repair plan (current dev integration)

ADR required: no
ADR paths: existing portable ADR-283 (renumbered from portable254), ADR-255, ADR-256; existing isolated recovery ADR-126.
Reason: repair existing interruption and complete installed-owner contracts; preserve current authority and shared-home pet policy.

1. Rebase onto verified dev 04f6680dc210ac341d72c294cbd68123d4700c20, retaining both lesson/census histories and current schemas.
2. Add RED interruption regressions around the real executor's first candidate receipt and outer marker; move outer publication intent to the durable candidate boundary, preserving malformed/ambiguous-state refusal.
3. Add real persisted pet import/selection coverage; declare only the confined installed pet file/backup family and preserve private portable startup selection without host writes.
4. Renumber the colliding portable decision consistently, verify current schema/owner integration and measured pins, then run targeted recovery/packaging and static checks only.
5. Record exact RED/GREEN receipts, source hashes, self-review and limits. Historical ZIP/361-test evidence remains tied to its original heads; no release rebuild or publication.

### 2026-10-31 review-repair implementation notes

Moved the outer publication marker to the executor's durable candidate receipt boundary before namespace registration/publication. Pre-receipt and marker-write failures leave inert retained attempts; complete markers resume the same operation. Old missing/malformed/multiple-operation states still refuse. Added public-coordinator interruption/retry coverage and an old-order negative control that reproduces the original stranded pending state.

Declared only the installed Windows pet canonical file and exact dated backups, preserved installed owner validation and unknown-content refusal, kept private pet publication separate from bootstrap, confined staged resume to the exact pet family, derived portable child selectors after committed authority validation, and acquired selected pets on subsequent moves. Real ConfigFileStorage bytes, backups, pending activation, fresh reads and a second physical move with the old bundle unavailable pass. Ordinary shared-home recovery policy is unchanged.

Renumbered only portable ADR-254 to283; dev's CLI ADR-254 and50 pre-existing unrelated collision groups are preserved. Current ChaChaNotes85, Workspaces10 and AgentRuns catalog through25 are retained. Appended7 cardinality-bound owner-path inventory rules for9 original portable occurrences; all49 base rules are unchanged. UI gate census1197 and persistent diagnostic inventory712 owners/16 sinks pass. No size-budget rows were changed; touched runtime modules have no applicable ratchet rows and remain below the4000-line row threshold.

Targeted evidence: recovery boundaries10passed; final repair set15passed with1 new negative-fixture setup error, followed by corrected negative1passed; actual persisted-pet/repeated-move case passed66.29s. Final-base interface controls130passed with3 Windows pytest setup/teardown errors caused by an oversized autogenerated bytes parameter ID; concise IDs preserve identical inputs/assertions, and precisely the5 affected deflate/receipt cases pass (JUnit retained). Standalone native builder3tests/5subtests pass. Strict touched-code lint with existing exact-type E721 checks retained, legacy archive fatal lint, formatter and whitespace checks pass. No deadlines, native guards, assertions, encryption policy or payload sizes were relaxed. No full sweep or release rebuild was run.

Full hashes, exact commands, failed and successful receipts, the9 inventory reasons and self-review: .superpowers/sdd/.tmp-pr3287-review-repairs/task-1-report.md. Historical361-test/ZIP/native-terminal claims remain tied to their original recorded source and artifact hashes; current source is qualified by focused native recovery/fresh CLI checks, not a rebuilt GUI ZIP. Local repair is ready for root-owned independent review and publication.
