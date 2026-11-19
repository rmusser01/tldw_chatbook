# Task 6 integration report

## Checkpoint and remaining work

Current HEAD: `4a769ebb5353a68dc69c42d9d16042e2afa58833`. The committed source repairs are ready for the root metadata checkpoint and independent integration review. Task 6 remains open: later dev integration, final combined startup qualification, current-head external CI and publication are pending. No push or merge was performed.

Reviewed BASE: `b265d5bd2c2f5a28e4f56c2115f6247988d29ac2`; reviewed source: `e7cc5337617781e9a4b08b9e324297b537fcad07`; old dev: `81c7c94f486491d6fde09584710dc3a3172225a0`. The resumed existing rebase completed all 21 commits onto pinned `ca2992cb10b24307fbae050643472ffb0a4388e7`, producing `52a3805727ed496723c8c2b18fb1c79e34cfbee4`. Recovery ref/bundle were retained. No rebase restart, abort, skip, or blanket conflict selection occurred. Five approved initial repair paths were committed as `cddb4b1c53f6fc71e95eec0b3fce240f12dfaf69`.

The receipt `dev` field records the observed shared `origin/dev`; it is not an ancestry claim. That ref advanced through ffb038, 83c264 and 67fc531 during local qualification. All receipts remain unchanged. The tested heads above integrate ca2992 only. Later upstream integration must separately prove ancestry and any carried source identity.

ADR required: no new decision. Existing ADR-092 governs fork mutation boundaries; ADR-069 project-control initialization; ADR-094 runtime/view lifetime; ADR-097 startup budgets; our former ADR-211, mechanically renumbered ADR-219, governs bounded starts. ADR-212 belongs to the later upstream shared-pane integration and must be retained there. No new admission, charging, routing, storage, schema or durable-write policy was added.

Runtime/census repair commit: `0b95d7aedb`; mechanical ADR commit: `4a769ebb5353a68dc69c42d9d16042e2afa58833`. Only root metadata remains unstaged. The first source commit emitted existing gc.log/unreachable-object warnings; no Git cleanup was performed.

## Preservation and conflict decisions

- [Initial per-path and 21-commit mapping proof](task-6-preservation.json): 1,838 upstream non-overlap paths initially exact; 161 owned Python paths, 157 strict AST equal and four canonical upstream deltas.
- The four semantic integrations retain immediate ChaCha CRUD transactions, removed dead queue recovery UI, upstream queue tests, and upstream egress tests. Canonical merge artifacts preserve literal values, annotations and decorators.
- [Current source and all-owned AST proof](task-6-boundary-preservation.json) enumerates the additional deliberately changed functions and tested hashes. All original assertions in the three provider-test functions remain AST-identical in [provider proof](task-6-provider-assertion-proof.json).
- Historical QA evidence is unchanged: initial inventory contains 11,572 blobs, including all 1,878 owned PR artifacts exact. The 27 upstream-only QA scripts intentionally reformatted on dev retain upstream blobs; these are not rewritten PR evidence. Current source repair diff contains no QA changes.
- The rebase conflict helper normalized each stage with strict AST equality before clean merge-file resolution. Existing conflict-records and continue logs retain original exits/proofs. No private profiles, config files or databases were copied into reports.
- [ADR renumber proof](task-6-adr-renumber-proof.json): only our active canonical ADR, index, spec and source comments change 211→219. Four Python files remain strict AST equal; migration SQL non-comment lines remain exact. Root owns active plan/Backlog reference changes. Historical revision/number claims remain intact.

## Corrective changes and evidence

1. Initial formatting corrections were strict AST equal. Six existing fixed-phase/exception-class diagnostics were byte-identical to reviewed BASE; root reviewed them before the inventory refresh. Original failing pin output and statement proof remain. Final diagnostic check passes with no additional refresh.
2. Five provider-persistence cases failed identically on pinned dev because they rebound config after private-profile admission. Three test functions now use the existing helper and child-admitted config path. All security/routing assertions remain unchanged. Five original cases plus five adjacent credential controls pass; 315 unaffected provider/Anthropic passes and ten passing Anthropic UI cases are retained rather than repeated.
3. Four already fenced public route names were missing from the bidirectional census. Adding them exposed two pre-existing owner failures. Immutable dev and reviewed-BASE diagnosis is retained; no safe-route exemption was added.
4. Six public mutation owners now hold the existing fork boundary: first persistence, explicit rebind, settings plus context commit, display-name preparation, persona seed, and assistant-name mutation. Validation precedes admission. Settings commit retains preparation→promotion order and both original nested admissions balance. The display-name token begins before live mutation and survives detached persistence until exact acceptance or abandon; no-op/stale/exception paths do not leak leases.
5. Settings UI retains explicit custody of the display task. A coordinator finally drains that exact task before idempotent captured-plan abandon. This covers cancellation before child start and sibling failure while physical persistence continues. Repeated cancellation does not release early or replace the primary sibling exception. No new persistence operation or retry was introduced.
6. Fresh new-chat project controls enter the existing create_session constructor through typed `initial_project_instruction_state`. Default restoration still reads durable/legacy controls; invalid values fail before publication. The external live assignment is removed, with no UI-thread durable write.
7. Census modeling recognizes typed drain DTO provenance and newly constructed voice messages only before escape/publication. Alias/live rebinding, branch rebinding and publication adversarial cases still fail closed. The exact display-name delegate is pinned. The rollback raw count is replaced by three exact owning submission branches and original arguments, with owner/guard/argument adversarial controls.

### RED provenance and corrections

The first race run was invalid evidence for 12 cases: the independent-session fixture had no settings and failed before mutation. The next run exposed four routes while durable-identity checks masked two. Both outputs remain unchanged, including misleading `green` labels with nonzero exits. The corrected isolated git-archive cddb source plus test overlay produces 12/12 failures precisely at the expected fork-transition refusal after actual field assignment; actual eligibility remained true. Source/test overlay hashes and equality are in the preservation proof. Three initial lease REDs were valid. Constructor explicit initialization and both pre-start/sibling custody REDs are separately retained.

An added lock-order control initially expected one admission, while the original nested settings context correctly performs two. The corrected test asserts preparation ownership and exact nesting depths 0/1. One new rollback adversarial source edit initially produced invalid Python; it was narrowed to the call argument. One added speech-owner assertion exceeded the existing scanner model; it was restricted to the newly modeled exact display-name delegate. None of these diagnostic runs is reported as passing.

## Qualification

Completed original owners: native creation/commit 5; immediate CRUD 3; queue/recovery 75; egress 112; provider/Anthropic 315 plus five subsequently repaired failures; profile 59 with two existing platform skips; startup 25 with three budget warnings; provider repair/neighbor controls 10. These results are bound to their recorded source and are not claims about later dev ancestry.

Current corrective source: 81 focused tests passed (public boundaries, caller lifetime, full census and exact neighboring settings/roleplay owners). Five additional validation/lock-order controls passed. The final strengthened full census passed 43 tests. Source formatting changes are AST-equal. Fatal Ruff, five-file formatter, whitespace, diagnostic inventory and affected worker/profile/UI census contracts have exact receipts below.

Startup constants remain unchanged from ca2992. Recorded warnings: imports681/686; UI ready1033/1033; preimport556/556 modules,414787/425347 LOC and126980/135111 largest route. Profile skips were existing os.setxattr/os.removexattr platform limits. Later upstream ADR-097's approved557-module exception is not yet integrated. Final combined startup and external current-head CI remain required; local native success does not replace CI for the earlier native timeout.

No full suite, installation, guard waiver, new skip/xfail, timeout/budget increase, private-profile copy, or historical evidence rewrite occurred.

### Exact command receipts

Each linked JSON records argv, cwd, observed refs, source hashes, exit and output log; pytest receipts link corresponding XML by filename. A nonzero exit remains a failure regardless of its label.

| Receipt | Exit | Result / diagnostic | Seconds |
|---|---:|---|---:|
| [task-6-auth-dev-baseline](task-6-auth-dev-baseline.json) | 1 | 5 failed, 315 deselected | 44.910667999996804 |
| [task-6-auth](task-6-auth.json) | 1 | 5 failed, 315 passed | 114.58 |
| [task-6-boundaries-constructor-caller-green](task-6-boundaries-constructor-caller-green.json) | 0 | 25 passed | 5.21 |
| [task-6-boundary-diagnostics](task-6-boundary-diagnostics.json) | 0 | passed | 55.03 |
| [task-6-boundary-fatal](task-6-boundary-fatal.json) | 0 | passed | 0.31 |
| [task-6-boundary-format-proof](task-6-boundary-format-proof.json) | 0 | passed |  |
| [task-6-boundary-format](task-6-boundary-format.json) | 0 | passed | 0.06 |
| [task-6-boundary-profile-inventory](task-6-boundary-profile-inventory.json) | 0 | passed | 30.41 |
| [task-6-boundary-ui-census](task-6-boundary-ui-census.json) | 0 | passed | 0.04 |
| [task-6-boundary-whitespace](task-6-boundary-whitespace.json) | 0 | passed | 0.17 |
| [task-6-boundary-worker](task-6-boundary-worker.json) | 0 | passed | 23.34 |
| [task-6-caller-custody-red](task-6-caller-custody-red.json) | 1 | 2 failed, 25 deselected | 3.68 |
| [task-6-census-final-static](task-6-census-final-static.json) | 0 | passed | 0.15 |
| [task-6-census-format](task-6-census-format.json) | 0 | passed | 0.15 |
| [task-6-constructor-red](task-6-constructor-red.json) | 1 | 1 failed, 2 passed, 22 deselected | 3.58 |
| [task-6-crud](task-6-crud.json) | 0 | 3 passed | 9.71 |
| [task-6-custody-and-census-controls](task-6-custody-and-census-controls.json) | 1 | 1 failed, 61 passed | 16.85 |
| [task-6-derived-check_backlog_task_files](task-6-derived-check_backlog_task_files.json) | 0 | passed | 0.39 |
| [task-6-derived-check_backlog_task_ids](task-6-derived-check_backlog_task_ids.json) | 0 | passed | 2.2 |
| [task-6-derived-check_bundle_sync](task-6-derived-check_bundle_sync.json) | 0 | passed | 3.24 |
| [task-6-derived-check_canvas_mermaid_assets](task-6-derived-check_canvas_mermaid_assets.json) | 0 | passed | 4.27 |
| [task-6-derived-check_index_plan_pins](task-6-derived-check_index_plan_pins.json) | 0 | passed | 2.19 |
| [task-6-derived-check_persistent_diagnostic_inventory](task-6-derived-check_persistent_diagnostic_inventory.json) | 1 | failed; see retained output | 64.79 |
| [task-6-derived-check_profile_owned_path_inventory](task-6-derived-check_profile_owned_path_inventory.json) | 0 | passed | 31.89 |
| [task-6-derived-check_schema_table_allowlist](task-6-derived-check_schema_table_allowlist.json) | 0 | passed | 0.44 |
| [task-6-derived-check_textual_worker_contract](task-6-derived-check_textual_worker_contract.json) | 0 | passed | 23.06 |
| [task-6-derived-check_timestamp_writers](task-6-derived-check_timestamp_writers.json) | 0 | passed | 14.19 |
| [task-6-derived-check_ui_pr_gate_census](task-6-derived-check_ui_pr_gate_census.json) | 0 | passed | 0.04 |
| [task-6-diagnostic-refresh](task-6-diagnostic-refresh.json) | 0 | passed | 79.43 |
| [task-6-egress](task-6-egress.json) | 0 | 112 passed | 7.6 |
| [task-6-final-boundary-controls](task-6-final-boundary-controls.json) | 0 | 5 passed, 70 deselected | 2.68 |
| [task-6-final-census-typed](task-6-final-census-typed.json) | 0 | 43 passed | 14.84 |
| [task-6-final-fatal](task-6-final-fatal.json) | 0 | passed | 0.15 |
| [task-6-final-guards](task-6-final-guards.json) | 1 | 1 failed, 6 passed | 94.4 |
| [task-6-final-overlap-ratchet](task-6-final-overlap-ratchet.json) | 0 | passed | 5.89 |
| [task-6-final-source-whitespace](task-6-final-source-whitespace.json) | 0 | passed | 0.11 |
| [task-6-fork-census-green](task-6-fork-census-green.json) | 1 | 2 failed, 25 passed | 18.69 |
| [task-6-fork-dev-baseline](task-6-fork-dev-baseline.json) | 1 | 1 failed | 8.192300833994523 |
| [task-6-fork-extra-dev-baseline](task-6-fork-extra-dev-baseline.json) | 1 | 2 failed | 7.322087125037797 |
| [task-6-format-check](task-6-format-check.json) | 1 | 10 warnings | 1.34 |
| [task-6-native](task-6-native.json) | 0 | 5 passed | 21.25 |
| [task-6-overlap-ratchet-snapshot](task-6-overlap-ratchet-snapshot.json) | 0 | passed | 2.68 |
| [task-6-overlap-ratchet-verify](task-6-overlap-ratchet-verify.json) | 2 | failed; see retained output | 4.78 |
| [task-6-profile](task-6-profile.json) | 0 | 59 passed, 2 skipped | 36.63 |
| [task-6-provider-format-snapshot](task-6-provider-format-snapshot.json) | 0 | passed | 0.47 |
| [task-6-provider-format-verify](task-6-provider-format-verify.json) | 0 | passed | 0.67 |
| [task-6-provider-green](task-6-provider-green.json) | 0 | 10 passed | 25.34 |
| [task-6-public-boundaries-all-routes-red](task-6-public-boundaries-all-routes-red.json) | 1 | 12 failed, 13 deselected | 4.15 |
| [task-6-public-boundaries-genuine-red](task-6-public-boundaries-genuine-red.json) | 1 | 11 failed, 7 passed | 8.53 |
| [task-6-public-boundaries-green](task-6-public-boundaries-green.json) | 1 | 12 failed, 6 passed | 4.5 |
| [task-6-public-boundaries-red](task-6-public-boundaries-red.json) | 1 | 15 failed, 3 passed | 5.4 |
| [task-6-public-census-final](task-6-public-census-final.json) | 1 | 2 failed, 67 passed | 22.58 |
| [task-6-public-census-qualified](task-6-public-census-qualified.json) | 0 | 81 passed | 20.11 |
| [task-6-queue](task-6-queue.json) | 0 | 75 passed | 299.27 |
| [task-6-ratchet-snapshot](task-6-ratchet-snapshot.json) | 2 | failed; see retained output | 0.92 |
| [task-6-ruff-fatal](task-6-ruff-fatal.json) | 0 | passed | 0.15 |
| [task-6-startup](task-6-startup.json) | 0 | 25 passed, 3 warnings | 66.87 |
| [task-6-validation-lock-order](task-6-validation-lock-order.json) | 1 | 1 failed, 4 passed, 27 deselected | 2.24 |
| [task-6-whitespace](task-6-whitespace.json) | 2 | 10 warnings | 0.6 |

## Self-review and proposed independent review surface

Review only the documented canonical integration deltas, provider assertion/setup proof, six public boundaries, constructor initialization, exact task custody/drain, strengthened census controls, and ADR mechanical proof. Check validation precedence, preparation/promotion nesting, token ownership on no-op/stale/error/cancellation, exact same-session refusal with other-session eligibility, and scanner rejection after publication/rebinding. Current source has no new charging/routing/authority policy. All production diagnostic statements remain accepted by the unchanged pin.

Retain the initial source reviews with their explicit hash/AST mappings. Root owns the clean metadata checkpoint, latest-dev integration decisions, immutable scoped independent spec/quality review package, publication and merge. The source changes are not yet a claim that the entire Task 6 is finished.
