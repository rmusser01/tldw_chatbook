# Scoped final fix-wave re-review

FIX_BASE `2210c0b5f5b8f48ffaa02cfbba66c4db5b0ddba6` → FIX_HEAD `3b616994e6ff3d3237a08f2802f25c5b5633e693`.

**Verdict: APPROVE WITH ONE NONBLOCKING MINOR.** Remaining findings: **0 Critical, 0 Important, 1 Minor**. Original findings: **6 ADDRESSED, 1 NOT ADDRESSED in full (M1)**. No new independent Critical/Important finding was established in this exact fix diff.

**Spec:** all four blocking behavioral requirements are satisfied; the seven-finding fix brief is not completely satisfied because its explicit M1 still-running-supervisor condition remains unmet. The expanded admitted-latency target remains an accepted, measured deviation. **Quality:** acceptable with the narrowly described M1 debt. **Code-review readiness:** ready with that nonblocking limitation explicitly retained, not an unconditional all-seven closure. Root must record its M1 disposition; this report neither authorizes another fix wave nor certifies external CI/merge obligations.

This is the single scoped re-review, not a second whole-branch audit. I read the prepared re-review/fix briefs, original final review, accepted I2 contract, amended ADR-240, final fix report, exact product/test/documentation diff and retained qualifications. I reused the repository/architecture/lessons context from the original review. I independently verified the exact HEAD, the 155643-byte review package SHA256 `2c110474748493793bbc0a391763f6dfc98c26b54f861306064cf6ec4832970e`, clean owned working paths, and all 24 final timing source hashes. Root-owned metadata changes are outside the seventeen owned paths. No suite, native probe, benchmark, provider call, dependency installation, source/index/HEAD mutation or helper dispatch was performed. Only this report was written.

## Per-finding disposition

### I1 — Complete issued-operation shutdown: ADDRESSED

`service.py:125-161` now registers an index-only operation atomically against close, makes same-thread nested calls borrow that lifecycle record, and retains it through the outer strict registry context. It grants no authority. The existing public provider wrapper already uses `service.registry_connection_scope()`; the new implementation consequently includes provider preflight/return guards and actual outer registry close, not merely the inner search pin. Controller preparation and Console status now enter the same service context; the Settings card wraps its full off-thread action/load. Direct key derivation already enters that context around the pin/root capture/registry read. Direct service management/search/start paths enter it as well.

`aclose()` closes admission under the service lock off-loop, joins any cleanup worker, waits the admitted finite operations, then takes a final writer/search snapshot. An already-issued start can complete its dispatch without reopening new admission; the final snapshot catches its writer. Physical joins and waits remain off-loop and cancellation is shielded until drain finishes. The failed-supervisor path now leaves the dispatch lock before `_retire()` joins the started writer. The unstarted-writer branch marks positive retirement only after its pin/read owner closes. Internal writer/supervisor strict registry contexts remain independent of the new closed public admission, avoiding a shutdown deadlock.

Management store/capability closure failures retain actual objects and the data pin; strict registry failure marks the operation unresolved. A new call on the failed thread must obtain new admission and is refused instead of treating its old failed handle as harmless borrowed state. Truly externally borrowed registry handles retain their existing caller-owned contract. Repeated close still reports unresolved ownership. Ordinary positively retired operation errors do not permanently poison the service.

Evidence: actual provider and controller native outer-close barriers hold maintenance admission after inner search/pins retire, and now hold service close through repeated cancellation. Native-close failure withholds evidence (controller returns no optional scopes), preserves maintenance blocking and refuses same-thread reuse, both before and after close starts. The actual mounted card Clear barrier demonstrates close pending during cleanup outside the service lock, late-call refusal, responsive UI and eventual Off result. The final affected run also establishes failed-supervisor lock release, successful incremental dispatch/retirement, held native close and runtime composition drain. The synthetic reader failure case is correctly retained as synthetic; actual outer-native tests provide the native proof. M1 below is a separate nonblocking physical supervisor-tail bookkeeping residual, not an unresolved I1 native ownership defect.

### I2 — Detached ownership and retry: ADDRESSED under the accepted contract

The accepted cleanup-only receipt is implemented in the stable binding accounting group as `cleanup-owner-v1.lock`. Its closed version-1 schema validates bounded workspace/binding IDs, positive integer revision, exact group hash and a 4096-byte maximum. It contains no source locator/body or authority grant. Native existing-only open refuses missing detached receipts; no-follow/single-link and private-parent checks remain. Bounded reads/writes/truncation/fsync operate on a duplicate of the held native handle, not a reopened pathname. Duplicate descriptor/handle failures remain represented and prevent positive capability closure.

Enable and start acquire the current generation capability, capture live exact authority, enroll/repair ownership outside the registry transaction, and compare authority again before configuration/publication. Repair acquires observed generation lifetimes and preflights namespace ownership. Existing valid receipt metadata is reused. The receipt and creation reserve count toward the binding budget. Live Clear does not require allocating this receipt or migrating a validated legacy cache.

Retry holds current data admission, the existing receipt, every observed generation lifetime and the current generation lifetime. It repeats bounded complete inventory, validates namespace cleanup eligibility, and uses a later durable absent-binding/archived-workspace witness through `project_index_detached_fence`. The registry transaction spans only that bounded witness check; no native acquisition, flush or bulk reclamation is inside it. Existing `clear_owned_namespace` and reclamation/headroom checks remove proven derived content; empty control/CAS/receipt metadata may remain. Same-ID recreation cannot Enable while cleanup owns the current capability; a distinct replacement ID has its own accounting group. No source authority or matching-path inference is used.

Detach/archive callbacks queue a coalesced off-thread attempt; Settings Workspaces exposes explicit retry without the removed row. A busy initial hint need not guarantee automatic eventual reclamation: the durable receipt and reachable retry are the accepted contract. Missing/corrupt/unproven detached legacy receipts refuse truthfully; ordinary busy/reclamation failures report pending; native-close failures retain actual owners/pins and fail close. Disable intentionally retains content with live Clear available, as documented.

Evidence: real cache detach/archive, immediate withheld public evidence, injected reclaim failure and subsequent retry; a distinct replacement's bytes remain unchanged. A separate cold child reads only recorded ownership and reclaims without source authority. Real-scan detachment is pending until writer/process/pipes retire, then succeeds. Same-ID recreation/Enable is blocked during held reclamation. Native receipt-close failure keeps a real exclusive capability/pin/operation, refuses reacquisition/subsequent management and fails service close. Existing-only/4096-byte/truncation/alias and missing/corrupt receipt checks are retained. Legacy mounted Clear-under-pressure and old-generation Clear regressions remain qualified. POSIX native execution is not claimed.

### I3 — Scanner/SQL prefix agreement: ADDRESSED

`Workspace_Project_Index_DB.py:929-988` casefolds configured prefixes and candidate paths in the shared eligibility predicate. Exact and slash-descendant matching now agree with the unchanged scanner, including Unicode casefolding. Peer-prefix rejection, suffix behavior and current/requesting-run exclusions still precede rank/LIMIT; physical containment and permission floors are unchanged. Actual scan-to-public-search regressions cover `Src`/`src` and `Straße`/`STRASSE`, with peer, other-directory and excluded-file controls. The initially invalid fixture REDs are not counted as defect proof; the corrected four-case RED entered product behavior and failed before the fix.

### I4 — Committed Building coverage: ADDRESSED

Controller preparation checks committed compatible Building coverage using the bool-only metadata existence seam. The service admits Building at entry, candidate fences and final result gates, returns partial unless final state is Ready, and preserves exact scope/root/chain, current/run floors, manifest revision and live Capture identity/hash validation. `has_compatible_files()` uses the same eligibility predicate as ranked candidates under an enabled exact scope. Zero initial coverage, current-policy-filtered old rows and run-floor-filtered rows do not advertise useful Building scope or return a falsely ready query.

Real scanner barriers after committed capture exercise both initial build and Refresh through actual Console preparation/public provider invocation; evidence is partial while held, source edits yield no stale hit, and revoked authority yields unavailable. Additional held-before-publication cases cover empty/narrowed/run-floor-filtered coverage. The service/controller lookup injections supply the test rig, not fabricated successful guard/authority values. This fixes the original several-minute discovery outage while preserving publication/read concurrency bounds.

### M1 — Bounded completed history: NOT ADDRESSED in full; remaining Minor

The unbounded-history problem is substantially fixed: the normal join-before-next-run regression performs twenty Refresh cycles, retains at most sixteen entries and drops exception tracebacks. Unresolved native owners are not selected for eviction. However, the explicit final brief condition “never evict ... a still-running supervisor just because a flag was set” is unmet.

**Exact location:** `tldw_chatbook/Workspaces/project_index/service.py:1168-1179`, read by close's final `_runs` snapshot/join at `:1250-1272`.

**Interleaving:** Supervisor A closes all native owners/read admission and sets `run.native_retired = True` at line 1168, then is suspended before its final service-lock block at 1171 (or while returning after it). New starts are permitted because they check the native-retired flag. Sixteen later runs can finish, and the history collector at 1173-1179 selects A solely by that flag and removes its only service-owned join record. A subsequent close cannot join A even though its physical thread is still alive; A may later resume its final bookkeeping/`_bind` unwind after close returns. The current stress regression joins every supervisor before the next start, so it excludes this schedule by construction.

**Severity and limits:** Minor, nonblocking. The triggering point is after DB/registry/scanner, stable lifetime, pin and read-budget retirement; no remaining native capability, cached-text disclosure, data corruption or important shutdown failure is established. The distinction matters: native retirement is proven, physical supervisor exit is not. Do not mark the specific final M1 condition fully addressed or claim pruning proves every owner thread has exited.

**Minimal future correction:** only evict a natively retired record after its supervisor is absent or positively no longer alive (or after its join), retaining references needed by close/waiters. A small bounded cleanup point on later operations can keep the history bound without pruning unresolved or still-running owners. This is a documented residual for root disposition, not a request for another wave under this review's one-wave cap. No additional probe was run.

### M2 — Scanner public contract: ADDRESSED

`scanner.py:294-310` types and documents both the singleton iterator/ScanPage compatibility mode and bounded callback/ScanUnit production handoff. The executable scan path is unchanged; no abstraction or protocol expansion was introduced.

### M3 — Dirty processing choices: ADDRESSED

The card records the last painted choices and the exact input text at dispatch. Status/initial completion repaints only unchanged painted inputs; successful Enable/Apply repaints only text still matching that request's submission. Newer edits survive both callbacks. Existing mounted/request/workspace/binding fencing remains, and malformed draft validation stays specific to Enable/Apply. The three mounted initial/reload/apply tests enter actual completion behavior and preserve later edits. Local boundary copy explicitly distinguishes processing choices requiring Apply from immediate workspace lifecycle actions. No permanent navigation-draft or generic debounce system was added.

## Evidence and source applicability

- Retained REDs remain distinct: initial 10 failures include four fixture failures; corrected I3/I4 four failures entered product. Cleanup/history four failures, zero-compatible Building one failure and held-dispatch-lock one failure are separately recorded. The initial 14-pass subset predates later hardening and is not the final gate.
- The original covering result is **4 failed, 75 passed, 1 warning in 1684.90 s**, not all green. Its two native-close assertions expected the wrong refusal after shutdown had already closed admission; separate pre-shutdown reused-thread cases now test unproven retirement. The synthetic-reader teardown no longer erases retained unresolved state. Compact UI failure occurred before focus scrolling settled; the correction waits for focus/scroll while retaining the painted/focused assertions. No native proof or visibility assertion was weakened.
- **12 passed, 1 warning in 374.84 s** in `final-fix-affected.txt` supersedes those four failed fixtures and covers the narrow later failed-dispatch change and successful control. Provider/UI test edits during the original serial covering run are explicitly qualified: later children may import revised test modules. The final affected hashes match current source; this is not one combined final-source all-green sweep. The M1 interrupted-supervisor schedule is not tested by these receipts.
- **1 passed, 1 warning in 233.46 s** is the one final same-retained-cache query run. I independently checked the 24 source hashes, parsed raw counts/ownership receipts and recalculated all twelve p95 values from sorted 50-sample arrays at index 47. Each cell has five warmups; common/rare/miss counts are 100/1/0. No timing rerun was performed.

| Shape/workload | Canonical decorated lexical p95 ms | Expanded admitted p95 ms |
| --- | ---: | ---: |
| Nested common | 311.691 | 666.935 |
| Nested rare | 151.309 | 426.705 |
| Nested miss | 153.433 | 424.228 |
| Wide common | 228.370 | 534.810 |
| Wide rare | 163.944 | 462.229 |
| Wide miss | 164.826 | 440.040 |

All six canonical cells meet the original 500 ms gate. Both expanded common cells still exceed 500 ms. **I continue to accept the explicit expanded-target deviation as nonblocking**, independently of root's acceptance: the feature's original decorated lexical target passes, actual broader costs remain measured, and no authority/security check was removed or cached across queries. This is not a pass for the expanded target or a claimed performance improvement. Earlier adverse measurements, including the earlier canonical miss and broader admitted misses, remain part of the delivery history.

The final forty public marker calls entered without borrowed registry handles and returned with native registry ownership retired. All forty entry/return booleans are positive. Forty-two native Capture receipts (including two authority warmups) show process gone, worker retired, pipes closed and an observed return code. Final service close leaves zero searches and zero operations. Instrumented whole-public p95 is 1782.163 ms nested / 1754.915 ms wide, including probe retirement bookkeeping; it is not the canonical clock or a pure uninstrumented user-latency promise. No competing owned test/renderer was active; lightweight metadata work and uncontrolled OS background remain qualified.

The two historical full-pipeline 20,000-file nested/wide builds remain source-qualified for unchanged worker/protocol/chunking/migration and DB publication paths. Scanner executable behavior is unchanged apart from types/docs. Final code adds bounded receipt enrollment and lifecycle work before scanning; **there was no exact-final-source full rebuild and the historical build durations are not final-source durations**. Earlier Ready/coverage, sampled RSS/storage, throughput, absent-embedding/blocked-network and physical retirement qualifications remain as recorded in the original final review. The new query predicate/admission source has its own final measurement. No corpus/profile reconstruction or cleanup was performed.

The retained 45-artifact manifest includes raw/source/command/visual receipts rather than private profile DB/payload copies. Final scoped Ruff checks and formatting pass for ten owned core/test files; the shared comparison reports 91 inherited findings and zero introduced against FIX_BASE. This differs from the older whole-delivery baseline attribution, not a claim that unrelated lint is now clean. No full-suite/hosted-CI claim is made.

I independently viewed both final cleanup PNGs: actual dark 80x24 focus/scroll exposes the cleanup action and wrapped result, and actual light 170x48 exposes cleanup plus settled Off controls. Current mounted supersession is the final affected run. Clear-focus/recomposition frames alone are not settled-control proof; the later cleanup frame and mounted assertions are. The older adverse compact focus exports and startup-overridden theme evidence remain historical. Windows/headless Textual evidence does not establish real-terminal/human approval behavior.

## Carried deferred and declined dispositions

No original lifecycle question is left silently untriaged: finite management and outer invocation drain are I1; detached reclamation versus access revocation is I2; active-build availability is I4; historical run retention is M1; dirty local inputs are M3. K1 adapter typing, old whole-tree attempted sets, earlier malformed-draft Clear/Disable and test-factory retirement-adapter defects remain resolved; this exact diff does not reopen them. Generic debounce still has no separate demonstrated authoritative stale-effect defect because expected scope/job CAS and lifetime admission remain.

The original eight declined judgments remain explicit:

1. Semantic/vector/embedding, SSH index, watchers, automatic Refresh and binary/OCR extraction remain separate, unimplemented deliveries.
2. General storage/recovery/config redesign, machine-wide scheduling and cross-query authority caching are outside this correction; none is required to hide the admitted-latency miss.
3. Unrelated whole-repository and hosted regressions are unqualified because targeted-only verification remains binding. Earlier private-SQLite census failures retain exact-baseline attribution; governance retains its source-attributed failing result, not an all-green or exact-baseline-runtime claim.
4. POSIX/macOS native behavior, other interpreters, real terminal/human approvals and real-model project answers remain unexecuted. Native receipt/scan proof is Windows-only. Earlier symlink-privilege/setup failure and platform skips are not security passes; later actual junction proof stays separate. Prior guidance is not new keyword proof.
5. Encrypted recovery evidence remains the accepted actual staged sequence after query/native retirement, not successful simultaneous replacement against the same live service.
6. Settings category navigation with manual workspace-row selection remains accepted; this wave adds no per-binding deep link.
7. No universal whole-tool numeric SLA or arbitrary-hardware/corpus guarantee is invented. Canonical and expanded clocks, real public costs and the accepted deviation stay separate.
8. Generic debounce, permanent drafts across navigation and a general cache collector remain outside scope. The new cleanup receipt/retry is the accepted narrow index-only remedy. Missing/corrupt detached legacy ownership remains an explicit refusal, not an inferred success.

Source-qualified historical scale evidence, simulated optional-dependency absence with zero network attempts, sampled rather than exact memory peaks, scaled quota/SQLITE_FULL experiments rather than literal exhaustion on all filesystems, and prior recovery/platform/visual limitations all remain unchanged. No pending receipt has been counted as passing.

## Final readiness statement

The consolidated wave fixes the four blockers and two of three original minors without weakening index authority, native ownership, output bounds or Library isolation. One residual Minor in completed-history pruning needs explicit root disposition; it does not establish native ownership loss and does not block my code-review approval. Preserve that residual, the admitted-target deviation and all source/evidence qualifications in closure metadata. This completes the authorized single scoped re-review; no second review or further fix wave is requested here.
