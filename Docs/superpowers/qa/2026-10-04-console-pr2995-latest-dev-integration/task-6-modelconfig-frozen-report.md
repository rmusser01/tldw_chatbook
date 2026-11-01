# Task 6 — pinned df2 model-settings integration

## Frozen result

Source: **9ed370f5668d82d9e6a0108835aa87136a57ce74**. Pinned dev: **df2ba424de63576d36d9c3e38387c84285303f1c**, proved ancestor. Clean BASE **786c1771cdd4e1311f54c4b26473808ea1cb4c31** replayed33 commits without conflicts to **6323ff208b523496dd29341fccb49e023a2ca1b5**; one two-test repair commit followed. Root plan/Backlog changes remain unstaged. No production correction was made in this phase.

**One unresolved guard:** ChatScreen exceeds its unchanged line/method caps. The exact upstream failure is retained; root is scoping a separate extraction. This report does not claim all qualification is green.

The schema/startup review remains bound to [the immutable dace report](task-6-schema-frozen-report-dace3505ab10.md). [Recovery](task-6-modelconfig-recovery.json) records the verified0600 bundle/ref. [Final preservation](task-6-modelconfig-final-preservation.json) maps all33 commits,283 exact incoming paths, five explained overlap paths,171 owned Python sources and **11,572 exact historical QA blobs**. Incoming model-config QA is also preserved. Every feature/upstream function in the four merged Python files was exact after replay; none was changed by both branches. Both final test repairs have separate AST proofs.

## Qualification receipts

Each JSON links exact argv, logs, exit and source hashes. Observed `dev` in receipts means the shared ref observed during that run; ancestry is established separately above.

| Receipt | Result |
| --- | --- |
| [Contracts](task-6-modelconfig-contracts.json) | 74 passed: None sampler, summary, default pair/native destination, fork settings boundaries, typed navigation/state. |
| [UI seams RED](task-6-modelconfig-ui-seams.json) | 4 passed,1 failed before mount: guarded config source changed. Passing cases cover hidden TopP Apply, no-live-commit picker, created endpoint pair and actual Console→Library→Console. |
| [Immutable df2 RED](task-6-modelconfig-settings-return-upstream-red.json) | Same original Settings return node fails with `raw_source_selection_changed`; exact source unchanged. |
| [Settings return GREEN](task-6-modelconfig-settings-return-green.json) | 1 passed through actual Configure→Settings→Return router journey. |
| [Final startup/caps](task-6-modelconfig-final-startup.json) | 32 passed,1 ChatScreen cap failure,3 headroom warnings. All26 startup checks plus lazy settings closure and app/modal caps pass. |
| [Immutable cap RED](task-6-modelconfig-screen-cap-upstream-red.json) | Upstream25391 lines>25218;764 methods>759 independently measured. Final25448/764. |
| [Schema provenance closure](task-6-modelconfig-schema-provenance-final.json) | 42 passed: exactly20 staged +22 runtime cases. [Explicit before/after test/runtime/SQL hashes](task-6-modelconfig-schema-provenance-hashes.json) equal dace and final commit. |
| [Initial static](task-6-modelconfig-static-initial.json) | Diagnostic, worker, UI census, fatal lint and source whitespace pass.94-file formatter check reports26;24 are exact historical source, two repaired below. |
| [Final touched static](task-6-modelconfig-static-final.json) | Fatal lint, formatter and whitespace pass on both repaired test files. |
| [Derived CSS](task-6-modelconfig-derived-css.json) | All12 independently rebuilt sheets exactly match checked-in source; none published. |

Startup:681/686 own imports;1033/1033 UI-ready;557/557 preimport;415298/425347 preimport LOC;127527/135111 Library route LOC. All thresholds unchanged. Diagnostic inventory:643 owners,1434 TASK492 calls,56 TASK31551 calls,7610 TASK494 calls,16 sinks, no drift. Worker census unchanged324 lookups/151 functions and68 waits/26 entrypoints. UI census133 with upstream floor130.

## Exact corrections and remaining guard

- Only `test_conversation_settings_return_real_navigation_restores_fresh_console_modal` gains existing `private_profile_test`+`request`, with original body/assertions unchanged. [Complete-file reversal proof](task-6-modelconfig-settings-fixture-proof.json) removes exactly that import/decorator/argument and recovers original AST.
- `test_console_provider_gateway.py` and `test_console_native_chat_flow.py` use current formatting. [Immutable BASE/dev failures](task-6-modelconfig-format-baseline.json) and [strict AST/literal/annotation proof](task-6-modelconfig-format-proof.json) separate formatting from the helper delta.24 unchanged historical formatter findings remain explicit; no unrelated formatting sweep occurred.
- The previous schema receipts omitted the then-untracked/newly renamed test modules. Later formatting hashes did not prove contemporaneous test identity. The fresh42-case receipt closes that gap with nine explicit before/after file hashes and final-commit equality. Original failed/passing receipts remain unchanged.
- [Cap history](task-6-modelconfig-screen-cap-history.json) identifies last compliant74965f694f435c590366ec66d671d3506d440ebc at25218/759. [Exact additions](task-6-modelconfig-screen-cap-additions.json) show eight added and three removed names before057: net+5 methods and+174 lines. df2 removes one line; the prior feature adds57 lines and no methods. No threshold was refreshed. A separate existing-controller extraction is proposed to root, pending scope.

## Self-review and next review surface

Review the four clean merged Python files through the function map, then the two-test repair diff `6323ff208b523496dd29341fccb49e023a2ca1b5..9ed370f5668d82d9e6a0108835aa87136a57ce74`. Incoming picker/modal/field-row/saved-default/CSS source is exact; runtime/schema/encryption/fork census owners carry by identity. Existing ADR117/148 and DESIGN§7 guide the proposed cap extraction; this completed phase adds no policy or authority decision. Schema review stays bound to ADR126/158/208/219 and immutable dace evidence.

All task tests have stopped. [Safe evidence manifest](task-6-modelconfig-safe-evidence-manifest.json) copies only top-level receipts/proofs plus exact pytest child logs/XML; no profile/config/database/cache/app-log copy. Prior FD growth warning and historical native timeout remain honestly recorded in earlier reports; current-head external CI remains root's merge gate. Git commit emitted existing automatic-GC/unreachable-object warnings; no cleanup was attempted. No push/merge/install/skip/suppression or budget increase occurred.
