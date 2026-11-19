# Fresh premerge review - PR 3204

Base: `2a5afa03b31f60e00bdcd10a161e3e351fad023b` (fetched dev). Head: `e16100e2b3aea1e1da3d6d73ecbcbefebb135a80`. Review scope includes both previously unmerged workspace guidance and optional folder-bound keyword indexing. This is the new user-authorized integration review, not a repetition of the closed delivery's review cap.

**Source verdict: With fixes. Findings: 0 Critical, 1 Important, 0 Minor.** Complete source findings are below. Final evidence and merge readiness remain pending the already running 22-case integration selection, current-source query-only qualification and external gates. A pending result is not a pass.

## Method and identity

Applied the user-invoked `requesting-code-review/SKILL.md` and `code-reviewer.md` template. Independently verified HEAD and the prepared full/source package hashes: full 18,408,721 bytes, SHA256 `dfe870dff58b607dc09b6d72d283465d6b775197a46f52c39a98a4e7051af29e`; source 1,051,460 bytes, SHA256 `fc64346454dcb1b73b19d69b225e9c7f30eee7ba21acb9a82e5ee88daee90249`. Reviewed production changes in passes: guidance/citations and provider permissions; models/store/publication; scanner wire/native capture/lifetime; service/runtime/cleanup/shutdown; Settings/Console and latest-dev composition. Read the approved design, guidance/keyword plans and amendments, ADR-241, architecture references, relevant lessons and current rebase/fix/sourceguard/test receipts. Inspected targeted tests and historical qualification reports for the behaviors discussed below. Large historical drawing/binary screenshot payloads were treated as artifacts, not executable source; I did not decode all PNGs or claim a fresh visual journey.

Review was read-only on product source, index and HEAD. No helper, suite, benchmark, provider call, GitHub action, dependency installation or branch mutation was performed. This report is the sole authored file.

## Strengths

- Guidance is rendered after final plugin and Goal tool narrowing, from run-admitted aliases rather than global workspace selection. Library-only formatting remains separate; canonical staged capture now receives conditional mixed-source citation instructions outside its exact source block, while optional missing markers do not weaken actual-marker validation.
- Index keys combine admitted data generation, workspace/binding identity and full pinned locator/root identity. Saved policy/scope/job comparisons happen before management effects. Registry revisions/tombstones and the registry-then-cache fence cover other processes; app-local callbacks are only hints.
- FTS candidates contain locations and identity, not model-visible cached text. Run/current exclusions filter before ranking; live Capture validates identity and content hash and final manifest/authority checks precede disclosure. The effective `fs_read` Ask/Blocked floor remains separate from search grants, including exact call-ID one-call approvals.
- Scanner units have bounded inspections, entries, pending work, text and wire frames. The durable attempt-revision frontier avoids a whole-tree attempted set. File reads reject unsafe metadata and changing/non-UTF8/oversized content; native process/tree/pipe closure and inherited lifetime ownership are explicit.
- Cleanup-only detached receipts are bounded, private, handle-based metadata without source locators or query authority. Retry works without the old row and retains all generation lifetimes; missing/unproven ownership refuses. Disable retention and explicit Clear are accurately distinguished.
- The current M1 fix preserves physically live supervisors beyond the diagnostic tail. The actual-first-supervisor regression holds after native retirement, completes twenty later jobs and verifies repeated-cancel close waits for physical exit. The focused final receipt is 10 passed/1 warning in 363.34s, with both this regression and the normal bounded-history control. Earlier meaningful RED and child-pass/outer-timeout evidence remain separate.
- Latest-dev dispatch initialization/retry authority, Settings Tools/MCP ownership and shutdown sequences are retained in the actual composed source. Structural patch equivalence is appropriately supplemented by the pending runtime selection rather than called integration proof. ADR-241 active links retain landed MCP ADR-240 ownership; historical source-era labels remain qualified.

## Issues

### Critical (must fix)

None established.

### Important (should fix)

1. **Failed start cleanup iterates mutable run history without its lock and can lose an unrecorded storage pin.**
   - **Location:** `tldw_chatbook/Workspaces/project_index/service.py:930-934`; concurrent writers are `_start_pinned` at `:1018` and the history collector at `:1171-1184`.
   - **Trigger:** Refresh/Resume has retained a fresh `_DataAuthorityPin`, but `_start_pinned` fails before recording that pin (for example a busy scanner, stale scope or failed factory). After the dispatch lock has unwound, the exception handler evaluates `any(run.pin is pin for run in self._runs.values())`. Another start can insert a run or an older supervisor can prune history during that Python generator iteration. All those mutations use `self._lock`; this read does not.
   - **Consequence:** Dictionary iterator invalidation raises `RuntimeError`, replacing the original failure before `_close_capability(pin)` runs. No `_Run` owns this fresh pin and the finite `_Operation` does not retain it. The normal operation-finally therefore retires its record, and `aclose()` cannot discover the lost pin. In production, the pin's `ExitStack` owns the retained `StorageLease`; `storage_admission._live_leases` strongly counts that lease until explicit close. Maintenance can consequently remain blocked even though index service close reports success. This is a cleanup/ownership defect, not source disclosure or a claim that every ordinary busy attempt leaks.
   - **Minimal fix:** Compute whether any run owns this exact pin while holding `self._lock`, or capture a tuple of runs under that lock. Release the lock before closing an unowned pin. Preserve the existing already-owned/uncertain-retirement behavior and do not put native closure under the service lock.
   - **Verification:** Add one coordinated focused failure-path control that pauses this ownership lookup while a real concurrent start/history mutation occurs, then proves the original error survives, an unrecorded fresh pin closes exactly once, the operation retires and service close/maintenance are not left with an invisible lease. Retain the already-recorded run-pin control so a started writer's ownership is not closed early. This is a source-established interleaving; it has not been executed or reproduced by this reviewer. I requested coordination with the parent; no extra check has been launched by this reviewer.

### Minor (nice to have)

None established. The old accepted M1 is not carried forward: the current source correction and meaningful native RED/GREEN address it.

## Plan/spec and quality assessment

Both requested deliveries are present. Source selection, saved Library policy/scope isolation, mixed citations, off-default per-binding keyword management, path/text indexing, run/read authority, exclusions, live hash validation, bounded work/storage, explicit recovery, detached cleanup and compatible committed Building coverage are represented in the implementation. The Important finding violates the planned complete native ownership/retirement contract on a failed-start race; it needs correction before this cycle closes.

The original canonical cached lookup p95 <=500ms remains the numeric gate. The expanded admitted boundary includes real registry/root/status/floor work and is a separately accepted unmet deviation, historically common 667/535ms. Keeping these boundaries separate is technically justified only with the admitted cost and miss still explicit; it does not establish a faster whole tool or remove authority checks. I accept the already recorded deviation for this optional local feature, conditional on actual current-source canonical/whole-tool qualification and retained adverse measurements. Historical 20k nested/wide build receipts remain source-qualified Windows evidence, not current-head build timings. The planned query-only reuse is appropriate because scanner/publication semantics did not change in this integration; a rebuild would add little evidence unless the fix exposes a concrete semantic change.

Corrected current-source index-plan, Textual-worker and timestamp guards each exited 0 in `derived-guards/corrected-guards-summary.json`. The one four-site card census row is supported by its sole await followed immediately by mounted/full-token fencing and synchronous lookups; it is not a blanket waiver. The plan-pinned directory index has a real EXPLAIN assertion without sqlite_stat1, and the three new writers use canonical UTC timestamps. Scoped Ruff/format/AST/whitespace receipts are passing; this does not claim unfiltered repository lint or a full suite is pristine.

## Recommendations

Address the single ownership race with the smallest lock/snapshot change and one focused causal check. Re-review only that correction and its actual evidence in this same report; do not repeat the complete source pass or already completed suites. Keep the existing 22-case run and query-only probe separately source-qualified if the new fix changes the head. Do not widen performance clocks, remove admission checks, regenerate unrelated inventories, or rewrite historical receipts to hide adverse results.

## Declined to judge

1. Semantic/vector/embedding support, SSH indexing, automatic watchers/Refresh and binary/OCR extraction: separate approved deliveries, explicitly outside this PR; they are not missing keyword features.
2. Generic storage/recovery/config redesign, cross-query authority caching and a machine-wide cleanup collector: outside the accepted local index boundary; the specific lost-pin defect above is within scope and is not deferred here.
3. Full-repository regression freedom, clean installation and unfiltered lint: only targeted verification is authorized and available. Existing baseline warnings/failures are not erased by this review.
4. POSIX/macOS native execution, other Windows filesystems/interpreters, human-terminal approval behavior and fresh real-provider/model reliability: not executed in this cycle. Windows/headless/native and earlier real-model receipts retain their stated limits.
5. Exact current-head 20k build duration, literal 1-GiB/250k exhaustion on every filesystem and exact memory peak: available scale evidence is historical/source-qualified, quota controls include scaled/SQLITE_FULL experiments, and memory observations are sampled. No unsupported current or universal claim is made.
6. Permanent drafts across Settings navigation, a per-binding Settings deep link and a general debounce redesign: accepted scope uses dirty-field preservation within the card, expected scope/job CAS and the existing Workspaces-category route. No demonstrated authoritative stale effect was set aside.
7. A universal whole-search numeric SLA or immediate cancellation of already issued native reads: the accepted feature specifies bounded candidates/output, per-worker retirement and honest whole-tool measurements, not such a universal guarantee. The current implementation must still retain actual operations through close; the concrete failed-start ownership race is reported above.
8. Successful simultaneous encrypted replacement against an actively held index owner: accepted recovery evidence uses staged retirement before replacement; this review does not infer concurrent replacement support.
9. Historical missing snapshots, interrupted UI runs and the cause of the earlier child-pass/outer-timeout: they cannot be reconstructed from later passing receipts. The timeout remains adverse/load-qualified, not an aggregate pass or a uniquely diagnosed cause.
10. Hosted required CI, current-head Qodo approval and final GitHub thread/queue state: parent-owned external gates. The reported Qodo billing/out-of-credits comment is not a review or approval, and zero threads alone does not satisfy that gate.

## Assessment

**Ready to merge? With fixes; evidence and external gates pending.**

The complete source review found one Important failed-start ownership race and no Critical or Minor issue. Fix it and verify its exact retirement boundary, then close the already agreed integration/query evidence and current-head hosted/Qodo/thread requirements before merge. Source quality otherwise supports the approved design without broadening file or Library authority. No task-Done or final CI/merge claim is made here.

## Scoped correction re-review - e16100e2b3 to 19d38fd32e

This is the single requested correction/evidence re-review. I reviewed the exact 23,545-byte package SHA256 `a093a7dd79d99649c3be7c9dbb5f8420aaf354f596a3eeec519a21f2b1fc3942`, source/test/task changes, exact RED/GREEN command manifests and retained child receipts, integration covering outcome, static receipts and the established private-bootstrap fixture mechanism. No complete source review, new check, helper or mutation outside this report was performed.

**Source/focused-evidence disposition: I1 ADDRESSED; 0 Critical, 0 Important, 0 Minor open.** Final query evidence is still pending at this checkpoint; this is not yet final merge readiness.

### I1 disposition

`ProjectIndexService._start_owned` now computes exact-pin ownership while holding the same service lock used for run insertion/pruning. The protected generator cannot be invalidated by those mutations. It releases the lock before `_close_capability(pin)`, so native storage retirement does not hold the dispatch/shutdown lock. Recorded pins remain owned by their runs; only an unrecorded pin is closed by the failed caller. The change preserves the original exception and existing unproven-retirement behavior without adding authority, storage interfaces or a new lifecycle owner.

The new regression uses a real production service, Workspace registry, contained scan, actual physically retired run and `_DataAuthorityPin`/StorageLease. Its coordinated `dict.values()` wrapper pauses the actual ownership generator; a test thread removes that retired run under the production mutation lock. The resulting RuntimeError is raised by Python's real dictionary iterator, not fabricated by the fixture. This is deliberate test-triggered pruning, not execution of `_retire`'s actual collector schedule; the existing twenty-completion collector and live-supervisor control remain the separate proof of that production path.

On e161 source the retained RED is meaningful: `RuntimeError('dictionary changed size during iteration')` replaced the injected original ValueError, the actual pin remained open and close_calls was zero. The outer receipt is 1 failed/1 warning in 25.21s; the native child failed in 20.34s. On corrected source, the same test retains the original exception object, observes exactly one pin close, proves another thread can acquire the service lock during native close, retires operations/history, closes the runtime and successfully drains actual storage admission. Test cleanup positively joins its threads and closes any retained resources. The GREEN child is 1 passed/1 warning in 20.39s. The complete affected selection is 5 passed/1 warning in 137.63s, including the real recorded-writer storage-pin lifetime control and failed-supervisor dispatch control. This closes the reported ownership defect.

### Latest-dev retry fixture disposition

The original agreed e161 integration covering receipt remains **2 failed, 20 passed, 2 warnings in 617.23s**. Both failures occur at controller construction: the real compaction-config reader reaches a collection-bound source participant after the generic per-test environment redirect selects another path, and the unchanged guard refuses `raw_source_selection_changed`. They do not reach the retry assertions and cannot count as passing retry behavior.

The correction adds only `@pytest.mark.bootstrap_profile` and its explanation to the exact parameterized retry wrapper. Existing `Tests/conftest.py` explicitly recognizes that per-node marker and retains its already isolated collection-time bootstrap profile. This does not disable admission, replace config/controller behavior or weaken any retry assertion. Both fresh/agent retry cases now execute their unchanged assertions and pass in the five-case corrected selection. No broader config-source redesign is needed to qualify these particular mounted real-config consumers. The earlier 20 covering passes remain e161 evidence; no single clean 22-case aggregate is invented.

### Correction quality and scope

The four changed files are the narrow production lock fix, causal ownership regression, fixture marker and task AC/plan amendment. AC14/plan23 records the issue before implementation and invokes existing ADR-241; no new architecture decision is introduced. Scoped service/test Ruff, format, AST/compile and whitespace checks pass. The large controller test file retains exactly its three baseline lint findings with zero introduced findings. No new scoped source defect was established.

The parent explicitly ruled on all ten declined-to-judge lines in this report in the current integration `progress.md`. Those limits remain accepted rather than silently dropped. The human's explicit waiver is **Qodo for PR #3204 only**; its old billing comment still is not a review. Independent review, actionable-thread resolution, required hosted CI, queue rules and actual merge remain separate obligations.

### Evidence awaiting final closure

The final same-cache query-only invocation is on `19d38fd32ed9009d9fd7c9ffb04e38fd989d370d`, with exact source/probe/profile/load data in `integration-idle-source.json`. The probe's body/clocks/assertions are unchanged apart from the already documented output destination; `-p Tests.conftest` supplies the same bootstrap plugin that the earlier probe directory's local conftest supplied. It retains the canonical 500ms gate, expanded admitted measurements, native Capture/public-owner retirement assertions and 900s bound. No current timing outcome is counted yet. Final evidence assessment will be appended here after the actual result.

### Final current-source query evidence closure

The single final invocation on `19d38fd32ed9009d9fd7c9ffb04e38fd989d370d` has physically retired and exited 0: **1 passed, 1 warning in 244.90s** (launcher elapsed 248.965s). I read the raw timing JSON/output/source manifest/summary and updated integration report. Independent read-only checks confirm all six cells contain 50 retained samples after five warmups, every common/rare/miss result count is respectively 100/1/0, all 42 actual Capture retirement records are positive, all 40 public returns have retired registry ownership, the service closes with zero searches/operations, and all 24 current source hashes still match the invocation manifest.

| Corpus/workload | Canonical lexical p95 ms | Expanded admitted p95 ms |
| --- | ---: | ---: |
| Nested/common | 273.264 | 644.722 |
| Nested/rare | 205.671 | 549.382 |
| Nested/miss | 169.099 | 454.642 |
| Wide/common | 244.297 | 617.648 |
| Wide/rare | 168.701 | 474.645 |
| Wide/miss | 164.260 | 435.502 |

All six canonical decorated lexical measurements pass the original <=500ms gate. The expanded admitted boundary **misses in three cells**, both common workloads and nested/rare. This differs from the historical common-only misses; the fresh rare 549.382ms result must remain explicit. The raw records and current summary preserve it. This is not an expanded-budget pass, a claim of improvement, or evidence that the lock correction caused a timing difference. There is no paired experiment establishing that causation.

Instrumented public whole-tool p95 is **1948.863ms nested / 2443.248ms wide**, on twenty actual marker calls per corpus. It includes registry/native/store/pin/guard/final-return work and the probe's native-retirement bookkeeping. The canonical component is therefore not the user-visible whole-call time. Candidate/store-open/setup exclusions and all real admission costs remain identified; no guard was removed and no raw-SQL substitute was used. The preserved fixed caches are existing completed 20k corpora; this query-only run does not establish fresh current-head scan/build duration, a clean install, other-platform behavior or arbitrary workload/hardware bounds.

I independently accept the expanded target's continued miss as the recorded, explicit delivery deviation for this optional local keyword feature. The original accepted canonical gate passes with substantial measured margin, evidence is live-validated and bounded, and the actual roughly 2-2.5s whole-tool p95 is disclosed. The additional rare-cell miss reinforces the admitted-cost limitation; it is not silently absorbed into a common-only claim. It does not establish a new scoped source defect or justify another optimization/rerun, changing the clocks, weakening admission, or rebuilding unchanged corpora. Parent release/PR notes should state all three current misses and both whole-tool timings.

### Final scoped assessment

**Code-review verdict: APPROVE. Open findings: 0 Critical, 0 Important, 0 Minor. Original fresh-review I1 is addressed.** This final disposition supersedes the earlier pending/With fixes checkpoint for the reviewed source.

**Spec:** The requested guidance and keyword behaviors are supported by the complete source review plus the scoped correction, source-qualified covering/focused evidence and final query result, with the expanded admitted-latency deviation explicitly retained. Historical build/platform/model/visual qualifications and the original adverse receipts remain intact. **Quality:** The sole failed-start ownership defect is corrected by the existing-lock synchronization and causally qualified without putting native closure under that lock; no new breakage was established in the scoped diff. **Readiness:** Local independent review/evidence is complete for source `19d38fd32e`. Actual merge remains conditional on parent verification of current-head required hosted CI, actionable thread resolution and the enabled queue mechanics; Qodo alone has the human's PR-3204-specific waiver. No hosted result, task-Done state or actual merge is certified by this read-only review.

No additional check, helper, production change or further review wave is requested. Carry the current three expanded misses, public timings and exact evidence limits into final metadata rather than treating approval as an all-green/full-suite claim.
