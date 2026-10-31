# TASK-35582 final scoped fix review

Reviewed 2026-10-26. Base `e03c177f91`; head `874d8adb8ceea0f53ec54f08081f0e4d1d2148e1`.

**Verdict: scoped fixes approved. Open findings: 0 Critical, 0 Important, 0 Minor.** I1–I4 and M1–M2 from `final-review.md` are resolved. No new actionable breakage found in the fix diff. This verdict does not close AC23/AC24 or authorize a Done/merge claim.

## Scope and method

Read `final-fix-report.md`, the bounded review package and actual production/test diff, connected admission/settlement and source-read code, the amended ADR-240/ruling 40, current task/QA qualifications, and the recorded test/build evidence. This is the single scoped follow-up to the whole-branch review, not another whole-branch audit. No tests, builds, live calls or subagents were run in this re-review. HEAD was verified; the checkout was clean before writing this report. Independently checked all 12 source hashes and all 9 durable evidence hashes in `final-fix/manifest.json`; every hash matched.

## Strengths

The fixes keep the existing authority owners: source policy stays outside the publication lock, skill consent comes from the real trust owner and original USER arguments, and native process acceptance/cleanup stays with actual runtime callbacks. The new tests exercise the reported bypasses and negative completion cases rather than only asserting new helpers. Packaging evidence checks actual wheel/sdist/installed bytes, and the QA record retains the resource warning and live-qualification limits.

## Finding dispositions

| Finding | Result | Reviewed implementation and evidence |
| --- | --- | --- |
| I1: model report reads bypass current source authority | Resolved | `tldw_chatbook/Agents/code_review_service.py:1135` snapshots captured paths, checks their current read permission outside the publication lock, then revalidates context/binding/revision/path set and exclusions inside `_source_context` before `_read_report`, including cached raw replay. The 12-case report/raw/index/summary matrix exercises guard, exclusion and permission revocation; the callback explicitly proves another thread can acquire the store lock. |
| I2: close waits behind a stalled publication/native admission | Resolved under ruling 40 | `tldw_chatbook/Agents/code_review_store.py:1455` sets the close event and revokes the exact scratch generation before a nonblocking attempt at publication bookkeeping. `_live` at line 439 retires result-only reservations on subsequent locked access. `tldw_chatbook/Agents/code_review_checks.py:131` checks liveness on entry and retains in-flight native ownership; `accepted`/`settled` remain actual callback boundaries. The unchanged stalled-writer regression plus close-before-admission, close-during-commit and uncertain-commit cases are green in the recorded final run. |
| I3: required natural/skill entry requests fail to compose review | Resolved | `tldw_chatbook/Chat/console_code_review.py:54` recognizes the literal spec and conversational requests, and only strips a leading skill alias when the real trusted binding is present. `tldw_chatbook/Chat/console_chat_controller.py:27094` supplies that binding; the parser still uses original accepted USER text. Native Console journeys cover both phrasings, trusted review/preauthorized verification, qualified verification requiring later acceptance, untrusted refusal, and trusted body without an original review request. |
| I4: successful verifier cannot seal its report | Resolved | `tldw_chatbook/Agents/code_review_store.py:928` requires the verifier's complete delivered/disposed ledger with no request gaps before accepting completion intent; line 1121 records that intent. `tldw_chatbook/Agents/code_review_service.py:2088` applies verifier ledger validity instead of reviewer coverage while preserving actual successful terminal/capture receipts and rejection gates. Complete and reasoned-unresolved units seal; premature/missing work, missing intent/capture, error and cancellation do not fabricate completeness. Native journeys assert V1 as well as reviewer seals. |
| M1: saved-team assertion runs only after restoration | Resolved | `Tests/Chat/test_console_code_review_continuation.py:166` deep-copies the configured fixture team; line 598 compares it after runtime settlement and before `after_schedule` or the `finally` restoration. The outer restoration assertion remains a separate cleanup check. |
| M2: guide uses the wrong wait selector | Resolved | `tldw_chatbook/Agents/code_review_guides/review.md:76` uses `wait_agents({"ids":["HANDLE"]})`. The schema-backed guide check and actual fleet-handle selection in the Console harness agree with the native API. Installed artifact evidence includes the corrected guide bytes. |

## Ruling 40 assessment

The revised contract is coherent and implemented by the existing ownership boundaries. Close immediately fences new admission and revokes source/scratch access; it does not pretend it can undo a final admission already in progress. The during-commit test waits for close to finish while still inside that native admission, verifies that no acceptance receipt exists before `commit_launch`, then observes acceptance from the actual callback. The close-before-entry case launches nothing. The uncertain case retains the disposable boundary and charged reservation until the external owner proves deletion. These assertions establish the distinction that the old blocking-close oracle missed.

This deliberately permits an already-admitted native commit to finish after close. It provides neither instantaneous process cancellation nor a refund for uncertain execution, and no such guarantee is claimed. Accepted/executed evidence still requires the actual native callbacks. The amendment records this cost rather than treating the ruling alone as proof.

## Evidence and requirements verdict

The final recorded targeted run is **42 passed, 34 deselected, 1 warning in 52.72s**. The warning is **FD 14 → 289 (+275)**. The connected run was **122 passed, 1 stale-oracle failure**, with the unchanged terminal-verifier refusal reproduced against exact-base methods; the corrected node is in the final passing set. The replacement oracle preserves refusal and checks its actual earlier reason. Recorded static comparison reports zero introduced diagnostics, not a lint-clean repository.

The installed probe loads guide/model modules from the fresh install target, excludes checkout imports, compares guide bytes across wheel/sdist/install/source, and compares the five changed implementation files across those artifacts. Review/verify guide envelopes remain 8,162/8,615 bytes. The manifest hashes match the reviewed head and durable evidence. This is adequate evidence for the scoped packaging change; it is not live provider qualification.

The six review requirements are satisfied. The fix diff preserves the consent, source, native receipt and finite ownership contracts reviewed here. No new schema, authority owner or workflow subsystem was introduced.

## Untouched observations — nonblocking for this scoped fix verdict

AC23 remains open for the previously recorded required regression and resource/Workspace cross-thread retirement qualifications. The new +275 FD warning is not erased by the targeted passes. AC24 remains open: no configured saved live team/presets and zero live provider calls/children. Stock execution still cannot enforce the required aggregate quota and must report unavailable before copy/launch; executed-protocol evidence uses the finite fault-controlled external executor behind the native runtime. Historical reports and the original 39 rulings remain historical evidence; ruling 40 adds the explicit race interpretation. These are retained branch-level limits, not newly introduced fix findings.

## Assessment

**Ready to merge? No, under the task's current completion gates.** The scoped code fixes are approved with no open review findings, but TASK-35582 correctly remains In Progress with AC23 and AC24 unchecked. Do not translate this review into an all-tests-pass, live-qualified, production-ready or Done claim. No further broad review, routine rerun, push or merge was performed.
