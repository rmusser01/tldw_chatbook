# Task11 — pinned-dev integration and selected owner qualification

Status: **qualified at pinned dev; clean source handoff for root review**. Controller/ChatScreen structure, newer-dev integration, final startup/public navigation and external gates remain open.

BASE `8d1088a5078337534bb700ce16aecd4d65399660`; old dev `5e0341d1ec701865e019eb2fd8a5e2028ab2d474`; pinned dev `f800952214844c3d41ddbe4b6522c93d4be25b7c`; rebase result `25c521593602d41cba5a6816ad6aa21f6f448a55`; final source `084b5618ded0f144ae01e643a224d09c554abe3d`. The single rebase replayed50 feature commits without conflicts. Recovery ref `refs/recovery/pr2995-task11-base` and mode0600 bundle are preserved; the bundle is excluded from safe evidence. Source commits are `8b367d565ad6b87a6de41c4feeae481ff92ab6c7` (one cancellation fixture) and `084b5618ded0f144ae01e643a224d09c554abe3d` (approved remaining fixture batch). Root metadata checkpoints changed no production/test bytes. No fetch, second rebase, source design change, cap change or publication was performed.

ADR required: no new ADR. Integration applies the existing Task11 ADRs052,054,089,096,031,097 and219. Fixture amendments are test-only canonical private-profile admission, approved in the committed Task11 plan and Backlog before implementation.

## Source, function and QA union

`final-exact-blob-union.json` maps29,797 tracked BASE/dev paths to rebased and final blobs. All production and tooling source remains exactly the automatic rebase result. All five overlapping Python files independently reconstruct byte-for-byte with `git merge-file(BASE,olddev,dev)`; no conflict-stage hand resolution occurred. Per-stage AST/source hashes and combined-body maps are retained.

The controller's four combined methods preserve both owners: `__init__` retains feature/native initialization plus upstream explicit staged-evidence collaborators and hold state; `_submit_draft_body` keeps native request/acceptance/fences, admitted-empty evidence freezing, early compaction assessment before effects, late capacity assessment before commit and one-shot bypass; `resume_durable_postcommit` retains machine acceptance and forwards the consumed compaction answer; `_apply_conversation_memory_preflight` retains feature behavior with upstream assessment/bypass/cannot-fit handling. Upstream runtime held-send behavior, staged-evidence lease/capture/release and UI actions remain intact. Other changed functions retain their exact upstream or BASE ASTs.

All PR3003 commits and pinned PR2987 commits are ancestors. Worker analyzer/test source and its five added census rows are exact upstream. Capacity/copy tests and their implementation bodies carry by exact upstream identity. Task10 store/fork/test blobs are exact carries. All **12,337 historical QA paths**, including the original11,792, retain exact blobs. Original failures, interrupted diagnostics and all completed passing receipts are preserved. No feature/provider/schema/encryption/scanner suite was replayed.

The only source differences after rebase are the three approved test files. The initial cancellation repair and remaining batch total11 canonical test prologues and10 positive async witnesses, one in the round4 shared helper. `final-fixture-source-union.json` reverses only these decorators, request arguments, imports and witnesses to restore every original module AST. The chained text proofs restore original file bytes. Every original body statement/assertion, parameter and decorator remains; other helpers are unchanged. `_accepted_evidence_recovery` has exactly the two selected callers.

## Qualification

| Receipt | Result |
| --- | --- |
| Original targeted Chat batch |22passed; one inherited setup timeout; interrupted exit2 |
| New compaction/staged-dispatch owners and concrete capacity/preparation matrix |21passed |
| Incoming mounted hold flow, three trace-card nodes and six-button CSS guard |7passed |
| One native prepared-start control[hello] and external writer/rollback census |2passed |
| Approved cancellation fixture |1passed |
| Remaining selected evidence/recovery batch, including unchanged bootstrap durable owner |12passed |

The final evidence covers **65 distinct passing cases**, with no skipped/xfail cases in those results. Original failed receipts remain failures. The cancellation GREEN is carried through its signature-only AST-identical reflow; the remaining batch source was stable throughout its run. No passing cohort was rerun. All node selections and reasons were recorded before execution. All actual command arguments/cwd/exits and source hashes are in the named receipts and `command-index.json`; `final-receipt-carry-map.json` binds earlier source hashes to final exact source or the approved reversible test-only admission overlay.

The actual merged worker analyzer, reconstructed diagnostic inventory and UI census pass. Diagnostic reconstruction reports650 owners,1437 TASK-492 calls,56 TASK-31551 calls,7614 TASK-494 calls and16 sink files. UI census preserves its133 floor. These were executed on the merged source and carried exactly through test-only repairs. The large scanner suite was not replayed.

## Inherited setup attribution and bounded fixture repairs

The original cancellation test waits for a capture event that is never reached after its real hook-admission snapshot refuses. Its original300-second timeout and22 preceding passes are retained. Immutable f800 observation-only diagnostics first reproduce the unavailable-hooks refusal, then directly record `RecoveryRequired` with equality to the fixed reason `raw_source_selection_changed`. Wrappers preserve results/exceptions; the parent promptly interrupts each diagnostic before another impossible wait. These are setup-attribution receipts, **not functional cancellation RED**.

The approved single-function repair imports/decorates with existing `private_profile_test`, adds `request`, and asserts `await controller.hook_admission_reason() is None` immediately after constructing the real controller. Original cancellation/capture/staged-launch checks then pass.

The next existing preaccept test fails `DID NOT RAISE RuntimeError` before its injected dictionary failure. Immutable dev repeats that failure and the fixed snapshot exception. Observation-only setup checks confirm the same mismatch for the remaining directly affected group; the durable checkpoint owner already admits under its existing module bootstrap profile and stays unchanged. Root then approved ten further prologues, eight direct-controller witnesses and one shared round4-helper witness. All12 selected cases pass with original assertions intact. No production gate/config/permission/conftest change, marker/skip/xfail addition, timeout change or assertion weakening occurred.

## Static provenance and explicit limits

Fatal Ruff passes. Final fixture formatter and whitespace pass. The touched-source formatter check retains exit1: all97 hunks are exact BASE/dev inheritance. Existing source-whitespace exit2 names upstream EOF blank lines in controller and incoming UI test. These were not normalized. Unrestricted touched lint retains583 inherited findings; source-line/code/message attribution resolves aggregate-count and shifted-line false positives. The three final fixture files retain the same6 baseline unrestricted findings; no new finding. All raw logs remain.

Measured existing rows remain: controller33,695lines vs29,367; store22,338 vs22,344; ChatScreen25,239lines/761methods vs25,218/759. These are measurements, not cap passes. Root owns the structural work. Eager import/source owner maps carry prior loading evidence, including the old58 results, without claiming fresh boot or public-navigation qualification.

Root later observed dev `9878fd251a01b9b65138316088af7bcee8c987b6` with modal changes. That pin is **not integrated here**; root requested a separate bounded next-dev scope after this clean handoff. Task11 claims ancestry only to its pinned f800.

## Evidence hygiene and self-review

All actual test runs use this worktree PYTHONPATH and canonical repository private-profile mechanisms. Unsuppressed cleanup warnings about existing pytest garbage directories and other emitted warnings remain in logs; `unsuppressed-warning-inventory.json` records their occurrence. No unrelated temporary directories were removed. The original slow metadata mapper was stopped and replaced with equivalent line-slice hashing; no source changed. One initial lessons read was output-truncated. The first dev diagnosis accidentally used an imported helper that hashed the managed worktree; `immutable-dev-attribution-source-map.json` independently verifies all8,096 extracted immutable-dev files and the receipt/report disclose the correction. Later diagnostics directly hash their actual source tree.

The safe manifest names only JSON/log/XML under this SDD, plus the report/freeze map hashes. It excludes profiles, configuration bodies, databases, caches and the recovery bundle. Self-review checked complete source union, every approved fixture reversal, positive real async admission, original assertions, exact QA/tooling carries, warning/static provenance, stable receipts and clean Git state. Root owns independent review and all later structural/loading/current-head/publication decisions.
