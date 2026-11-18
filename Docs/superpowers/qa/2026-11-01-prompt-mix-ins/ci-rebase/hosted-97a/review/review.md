# Final archive confirmation lifecycle review

## Scope and identity

Read-only review of the final four-file repair over published `97a53c67aefeb64fab42eca1592dc2cf7d6bff46`, with dev context `2dcf038cb8be0af991f92c43b0c262f02471c51d`. Exact working-tree hashes are in `scope.json`; `reviewed.patch` contains the reviewed delta. This supersedes the earlier three-file assessment. It does not certify a future commit, hosted CI or the merge of PR3346.

Requirements read: repository AGENTS.md, requesting-code-review template, Task38544 AC30 and its final pre-edit refinement, ADR166 and the 2026-09-17 archive authority design, relevant Console architecture and lifecycle evidence guidance. Previous feature audits were outside this bounded review.

## Strengths

- The setter adopts the latest model/display and exact session/request ownership before consulting controls. All six fixed controls must have the expected type, be mounted and be outside native pruning/closing before any projection occurs. This covers pre-compose publication, latched parent mounting after child removal, and a Select whose internal children have retired while all six outer nodes remain.
- Existing on_mount replays the latest model after native child mounting. The implementation uses supported Textual8.2.8 query(type).filter(selector). There is no new polling, deferred callback, remount, broad exception catch or helper layer.
- The boolean reports completed projection, including a rendered clear. The sole view caller refuses deferred publication. Its public return for None remains false, and adopting an unrendered replacement cannot acknowledge answerable consent.
- Exact stamped request/session events, visible allowed scopes, expiry, one-shot consumption, coordinator custody and source authority remain intact. Native partial-control tests require false acknowledgement and refusal of the previous stamped event.
- The two original source-remount wait predicates now tolerate query absence before evaluating their original display/request conditions. The original condition AST, all assertions, expiry/detach behavior and deadlines are preserved. Original card tests and scope/typed-decision handlers remain byte-identical; the five-variant resize test file is entirely unchanged.

## Issues

### Critical

None in the final reviewed repair.

### Important

None open. The actual native Select-Unmount hole found during review is closed: the mounted-only draft could call set_options after SelectOverlay had retired; the final same-loop _pruning/_closing check refuses that projection. A permanent real Unmount regression failed before this guard and passed in the final frozen cohort. The earlier clear verdict was explicitly withdrawn when this evidence appeared.

### Minor

None in the final reviewed repair.

## Evidence

The owner retained the initial8-case RED run against old product behavior (1.91s), the first-draft unsupported DOMQuery.filter(type) failure, the corrected draft GREEN and the subsequent real Select-Unmount probe/RED (1.44s). These are separate phases, not final successes. The extra native source-remount run exposed the test's query_one assumption during a legitimate remove/mount gap; the final test-only presence guards preserve its original assertions and timeout contract.

Final frozen native cohort:34passed50.80s, process exit0 (54.78s receipt). It includes the complete26-case card module, all five unchanged original partial-resize variants, and original source-remount/expiry/detach, fresh Restore-as confirmation and compact keyboard-focus flows. Reviewer parsed the final JUnit:34cases, no failures/errors/skips, including the exact Select-Unmount case. Current hashes match the owner's identical before/after frozen hashes. All37 explicitly inventoried owner artifacts match their recorded byte counts and SHA256 values. All four files have empty Ruff diagnostics against the empty97a baseline and report already formatted; read-only diff whitespace checks pass.

Reviewer read source, callers, native Textual8.2.8 lifecycle/API code and owner evidence, and independently checked source/evidence hashes and preservation metadata. Reviewer did not execute app, pytest, native or performance runs. The earlier original local resize-band observation remains a separate unconfirmed precondition; the final five variants pass without a causal claim for that observation.

## Assessment

Ready to integrate this bounded repair: yes. No Critical, Important or Minor source finding remains. The existing ADR166 authority contract applies; this routine lifecycle repair requires no new ADR. The parent retains ownership of unchanged startup/size checks, exact-head publication, protected hosted CI and the actual authorized merge.

Skipped: full-suite execution, duplicate native/performance runs, external actions and unrelated feature re-audit. Risk: final local evidence does not certify future hosted CI or a future head.
