# K4 fix round 1 re-review

## Scoped spec-compliance verdict

**Approved: I1, I2 and I3 are ADDRESSED.** Reviewed only the fixes and their regression surface in `8ea34f516fd5e8a9f1cca3d3863cc463f8cc14e2` -> `4f744897ff6a34898b9590bc85fd34591482d079`. No new blocking defect was identified in the supplied six-file diff. This closes the prior K4 Important findings; it is not a final whole-branch review.

## Per-finding assessment

### I1 — ADDRESSED: conditional approval provenance survives floor relaxation

- `tldw_chatbook/Chat/console_interrupt_rounds.py:5484` excludes conditional fs_search_index rows from both ordinary name-stamp aggregation and its answer-provenance projection. Raw forged broad decisions from those rows can no longer create the ordinary name-based grant that caused the original defect.
- `tldw_chatbook/Agents/local_tool_provider.py:1276` consumes the exact run/call/arguments decision by replacing it with a timeout marker. `:3653` checks for that provenance before the ordinary Allow/name-stamp paths even when fs_read has become Allowed; the explicit current Deny check remains first. This preserves one-call behavior and refuses replay/unanswered exact IDs across the reviewed floor transition.
- `tldw_chatbook/Agents/local_tool_provider.py:3727` redirects a newly conditional generic fallback row into the same nonpersistent review path and attaches the current call ID only for that row. Ordinary fallback continues using its original empty-call-ID/tool-name response convention.
- `Tests/Agents/test_project_index_tool.py:737` covers Ask -> Allowed with approve_once, forged approve_session and forged always_allow, then checks absent ordinary stamps, absent persistence, replay, unanswered IDs and sibling IDs. `Tests/Chat/test_console_local_review_hook.py:2180` covers the newly conditional fallback; `:2224` protects ordinary fallback keying and persistence. The final affected receipt includes these cases and the existing native/synthesized runtime and exact-call tests.

The fix is narrow: it preserves ordinary local-tool review semantics while keeping conditional read consent call-specific. The previously reported raw-name-stamp/persistence route is closed.

### I2 — ADDRESSED: writer and query operations share aggregate read admission

- `tldw_chatbook/Workspaces/project_index/service.py:106`, `:198`, `:568` use the same two-permit semaphore for query and writer operations. Writer acquisition is nonblocking before pins, registry fences and service locks. Busy refusal creates no job or deferred start, matching the accepted brief refinement.
- `tldw_chatbook/Workspaces/project_index/service.py:570`, `:657` transfer the writer permit from a local ExitStack into the existing run owner before dispatch. A pre-dispatch failure releases it after cleanup (`:690`); successful dispatch or failed supervisor launch retains ownership through the existing physical retirement path. Normal retirement releases it only after native store close, lease close and operation-pin close (`:825`, `:832`). No permit is released merely because the writer coroutine/caller stopped waiting.
- Query permits remain retained with the typed unproven-retirement record (`tldw_chatbook/Workspaces/project_index/service.py:221`). `_start_pinned` refuses same-key writer work while that failure exists (`:585`). Closure still forbids dispatch after admission.
- `Tests/Workspaces/test_project_index_service.py:1238` holds two query readers, proves Refresh refusal without scope/job/state mutation, retries after return, and checks the closed-service refusal. The measured peak is explicitly the service operation boundary, not an OS descriptor count. This is sufficient alongside the unchanged serial scanner contract: at most two admitted operations, each with at most one reader.
- `Tests/Workspaces/test_project_index_service.py:294` augments actual held-native-DB-close/repeated-cancellation proof with permit retention/release assertions. The failed-writer-dispatch test checks permit recovery (`:1066`). `Tests/Agents/test_project_index_tool.py:567` identifies each synthetic operation pin independently and verifies the failed query retains its own pin and permit while subsequent rejected operations close only their own pins.

The whole-writer permit is a conservative implementation of the accepted application-owned budget. It avoids new worker protocol or process coordination and reduces query concurrency during a build as documented.

### I3 — ADDRESSED: final stale-result wrapper remains within 16 KiB

- `tldw_chatbook/Workspaces/project_index/service.py:351` constructs the final DTO after the final authority/scope/manifest fence checks. `:357` measures its actual compact UTF-8 JSON representation and removes only complete trailing hits until the output is within 16,384 bytes. It preserves valid JSON, the partial status, and refresh_required.
- `Tests/Agents/test_project_index_tool.py:779` recreates the exact 16,384-byte Ready payload with 12,000/4,130-character excerpts followed by a stale third candidate. It asserts all three candidates were visited, valid nonempty JSON remains within the limit, and Partial/Refresh-required survives.

The final check closes the reported one-byte expansion without changing ranking, cached content, path authority or excerpts.

## New-breakage assessment

**No new Critical or Important findings.** Reviewed semaphore ownership on pre-dispatch errors, failed supervisor launch and native retirement; conditional review consumption and fallback routing; and final serialization trimming. The changes are confined to the three findings and their tests. The ordinary-fallback compatibility correction is supported by a specific regression and final affected receipt.

No new Minor product finding. Prior M1 inherited warning noise remains an evidence qualification, not a reason to expand this fix round or suppress warnings.

## Evidence assessed

Read the current brief including both read-budget rulings, the appended fix report, prior review, and the supplied 38,698-byte fix diff once in consecutive portions. Did not rederive a Git diff or inspect unrelated plan/governance changes.

Read retained receipts and the final six-file lint comparison:

- `k4-fix1-covering.log`: **122 passed, 2 failed, 1 warning**. The failures are retained honestly: the synthetic pin-accounting fixture and the unchanged WinError1314 symlink setup limitation.
- `k4-fix1-final-affected.log`: **9 passed, 1 failed, 81 deselected, 1 warning**. All approval/fallback/native/synthesized runtime cases passed after the final product adjustment. The remaining failure was the synthetic pin index bookkeeping.
- `k4-fix1-retirement-final.log`: **1 passed, 29 deselected, 1 warning**, after the test-only pin-index correction. The final diff captures the query token immediately before search and distinguishes its admission from prior status and later startup admissions; it does not weaken the retained-pin assertion.
- `k4-fix1-lint-comparison.json`: six paths, **28 inherited findings, zero added findings**, with final source hashes.

The complete covering run is not all-green. The final source is qualified by its 122 passing cases, the nine affected approval/runtime passes after the last product amendment, and the corrected retirement-only pass. This matches the authorized affected-only verification sequence; repeating the suites would add no answer to a concrete remaining doubt.

No tests or suites were rerun in this review. Bounded source continuations were read only where the diff truncated functions needed for two named checks: writer permit ownership through exception/retirement paths, and exact-call fallback decision handling. No production/index/HEAD mutations, network calls or subagents were used.

## Out-of-scope observations and cannot-verify gates

- The selected-root symlink-swap assertion remains unexecuted on this account because WinError1314 occurs during fixture creation. This inherited platform qualification remains for a privilege-capable environment; it is not passing security evidence.
- Aggregate overlap evidence is service admission plus actual native writer retirement evidence, not a new OS-level simultaneous-read measurement. Independent application service instances retain independent budgets, as the accepted brief specifies.
- POSIX, GUI approval interaction, active-query restore, and prior cross-task storage/owner guarantees were not reopened. The prior approved cache-only lexical benchmark and unchanged Library/DB evidence remain qualified at their original levels.
- Root-owned metadata and the final whole-branch integration review remain outside this task gate.

## Scoped task-quality verdict

**Approved.** All three prior Important findings are addressed with focused, maintainable changes and regressions that exercise their original failure conditions. The remaining limitations are explicitly recorded platform/cross-task qualifications, with no unresolved fix-round blocker identified.