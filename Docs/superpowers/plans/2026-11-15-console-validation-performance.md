# Current Console validation and performance implementation plan

> For agentic workers: use systematic debugging and targeted RED/GREEN checks; root coordinates independent investigation and final review.

**Goal:** Resolve current reproductions of the merged PR3166 validation/CI failures, then reduce the next freshly measured Console return cost and open a new PR with auto-merge off.

**Architecture:** Retain current finite preparation, original native source/owner checks and physical retirement. Repair existing owners and fixtures at the actual cause; reuse the guarded real-app Home/Console journey for measurements.

**Tech stack:** Python3.12, Textual8, SQLite and native Windows custody.

**Spec:** User request in this chat dated2026-11-15; existing TASK35679 and this investigation's follow-up task.

ADR required: no for investigation and routine contract repairs.
ADR path: N/A; existing backlog/decisions/126-complete-local-backup-and-recovery.md and230-console-send-preparation-and-io-ownership.md apply.
Reason: no new storage/schema, permissions, source/ownership, cache, service or runtime boundary. Reassess before selecting the measured product change.

## Constraints and review focus

- Keep storage admission, source freshness, permissions, custom-callback affinity, cancellation and native physical retirement checks intact.
- No timing-budget increases, safety bypasses, blind inventory updates or replacement of production callables in performance evidence.
- Run targeted selections only; no full local test sweep without the user's opt-in.
- Distinguish stale historical failures, current regressions and launch/setup failures. Retain every run and adverse result.
- Preserve drafts, render/input semantics, complete attachment and source identity across navigation; do not treat composer presence as attachment.
- Use a new branch and PR; no merge or auto-merge, including when CI becomes green.

## Task1: Attribute validation and CI failures

- [ ] Retrieve the original failing job assertions and group their causes.
- [ ] Compare exact test/product owners against currentdevcc6006fcf59328353004206bba8724c399216749.
- [ ] Run the three original test_console_tab_publication_read_barrier parameter cases serially under a genuine native-private profile.
- [ ] Reproduce the relevant structural, UI and performance contracts on current sources. Keep original node IDs and failing output.
- [ ] For every current red, record the data flow and minimal hypothesis, apply the smallest root fix, then rerun that contract and its connected checks. No change for an already-fixed historical cause.

## Task2: Measure and reduce one current Console cost

- [ ] Restore only the existing guarded six-return journey, remove unsupported historical experimental modes, and qualify observer code targets without replacing callables.
- [ ] Wait for tests and other owned resource-heavy commands to finish. Run one plain current baseline and a separate diagnostic with original-code wall/CPU spans plus main-thread samples.
- [ ] Rank costs within navigation-to-input and navigation-to-complete-attachment boundaries; distinguish nested CPU work from waits and queues.
- [ ] Append the selected root cause, exact files, smallest correction, causal RED test and ADR ruling here before product editing.
- [ ] Implement the bounded correction; run the causal GREEN and connected source/ownership/custom/cancellation controls.
- [ ] Freeze product hashes and run serial matched before/after cohorts. Preserve all journeys, exact exits, guard counts and unfavorable measurements; do not infer a global speedup from one removed call.

## Task3: Review and publish

- [ ] Run scoped lint/format checks, generated-artifact checks where affected, and diffcheck.
- [ ] Independently review source, safety preservation and measurement claims; address required findings with checks.
- [ ] Record the CI-failure inventory, validation results, profile comparison and limits; update Backlog notes and ACs honestly.
- [ ] Push the new branch, open and attach its PR, and verify remote head plus autoMergeRequest=null.
- [ ] Address newly exposed failures in the new PR's relevant CI, retaining targeted local verification and reporting any external blocks.

## Investigation ledger

- Current baseline is freshly fetched devcc6006fcf5 in the managed isolated worktree; primary checkout remains untouched.
- Historical three tab gaps were repaired in9f07ca956d. Fresh verification remains required; no additional product fix inferred from the historical gap labels.
- First requested native test launch refused ambient Windows TEMP mode0766 before collection. That launch is not counted as a test failure; supported native-private fixture roots are required.
- Task CLI completed: TASK-35679.1 records scope, ACs and implementation plan. Slow first-time repository scan did not create duplicate tasks.

- Harness independent review: retain exact owned profile roots for root-managed validated cleanup; protect native fixture owner with a module-lifetime context and put observer installation/constructor inside its finally. Async CPU is attributed only between paired resume/start and yield/return events, never whole-coroutine elapsed time. Enforce an external process deadline before execution.

### Selected current CI correction: modal classification

The targeted current static selection returned123passed/1failed. The sole failure is a newly added PluginRestoreAdoptionModal, absent from the reverse inventory. Its compose uses the exact existing #plugin-review-shell; _plugins.tcss gives that shell90% viewport width/height, matching the four already-documented plugin modal exclusions and exceeding the85% wide tier. Add exactly that class with the same verified geometry reason. Preserve the scanner, stale-entry checks, every CSS rule and plugin recovery behavior. Existing failing reverse-inventory test is RED; rerun it plus connected registry coverage as GREEN. ADR required:no; routine test-inventory repair, no product/visual or authority change.

The first temporary tab observer could not install PY_UNWIND as a local event under Python3.12; this is a diagnostic setup failure, not a product result. Remove that unsupported local event; retain original entry/return observations and final state.

### Current tab fixture diagnosis and correction

Original unobserved after_close failed before installing its native read gate: first _show_tabs rendered only the keeper. The original-code observed rerun records the first FULL returningFalse while attach is running and replay_whole_sync=True; a later publication paints both tabs. Its cleanup observer incorrectly read console.workers after app exit and masked the final test outcome; retain this as a diagnostic error, not a passing control. This still demonstrates the early deferred-publication setup hazard; it does not reconstruct the unobserved run's complete scheduling history.

Move the existing startup/publication and readiness precondition before session creation and the first _show_tabs, for every phase. Use one shared original10s deadline for those prerequisites (including after_close's existing readiness budget), retaining every native reader/proof, original FULL, real mouse Close/DOM, held Surface/read,10s waits,30s gate and8s cleanup. No production or storage change. Remove the now-misplaced after-first-show readiness block rather than adding a second readiness budget. Run original three parameter nodes without observation after the correction. ADR required:no, existing fixture lifecycle ordering only.

### Native close causal qualification

The startup-corrected unobserved Close failed at finalize_session_close with RuntimeError; the subsequent observed run passed33.05s with zero pending controller reads at finalization. Do not attribute the original refusal to a particular raise site without its traceback. Source analysis exposes separately testable gaps: new finite context reads observe only the controller set after runtime binding (observe_preparation_reads only copies existing handles); their currentness predicate does not reject the controller's committed close fence. Existing reads intentionally survive cancellation until native retirement.

Before product changes, add two causal tests using real context/native callbacks and real runtime.close_session: hold an existing original SQLite version query through its native callback, and require Close to retain it through physical retirement; separately hold the existing hook-cleanup await after drain and require new context admission to refuse a committed close fence before any native handle. Observe code with sys.monitoring only; use events and bounded cleanup. These qualify the defects independently and are not retroactive attribution of the mounted failure. Select the smallest owner-bound correction only after actual REDs. Existing ADR126/230 applies; no new lifetime, permission, storage or cancellation policy.

### Selected native lifetime correction after causal RED

Both new causal controls fail on unchanged production sources in9.41s: real Close reaches finalize_session_close with `A native preparation still owns this session`; the late original context call succeeds and observes one newly admitted native handle while both close fences are committed. Keep these distinct from the unobserved mounted refusal's unconfirmed line.

Correction: after its existing runtime-owned physical drain, ConsoleRuntime._close_session_after_voice_drain also drains the captured controller's original preparation-read set when it is a distinct set. Use existing drain_preparation_reads and preserve cancellation delivery, order, ticket validation and finalization guards. In the original qualified context reader's existing currentness predicate, also reject the controller's already-committed session-close generation. Custom callback fallback remains in its original pre-predicate route; memory selection, source checks, callback/worker binding, native scope and positive physical retirement remain unchanged. No new observer registry, cache, deadline or service interface. ADR required:no, existing ADR126/230 physical ownership and close fencing implemented consistently; routine correctness fix.

Current boot/runtime selection:9passed/1failed in470.49s. The modal correction is GREEN; boot worker, preimport and custody assertions pass. Remaining census is1041>1040, with provider_setup_facts and the mandatory Windows files helper beyond its snapshot. Do not relax the ratchet or defer mandatory native checks; analyze a legitimate first-use import. Parent launcher also hit cp1252 while echoing its completed UTF8 log; child receipt/results remain valid. Future launches use -Xutf8 for the parent too. Standalone markup checker passes unchanged.

- Independent native-close review found no product blocker. Extend the real held-read case with repeated Close cancellation; require retirement and deletion before delivery of CancelledError. No cancellation policy changes.

### Current module-census root cause and causal test

The pure canonical endpoint resolver loads provider_setup_facts for every non-engine endpoint solely to find the one existing ROUTE_SUFFIXES table. This brings setup/key-help data onto ordinary startup. Mandatory Utils.windows_files must remain. Add one isolated import-closure scenario: real OpenAI/local resolution must retain exact URLs without loading either setup-facts module; later explicit setup use must share the same suffix table object and preserve ZAI routing. Run RED before changing providers. Candidate mechanical correction: define the same single table in the already-resident canonical endpoint module and re-export it from setup_facts; no new table, routes, public values, dependency/runtime interface or caching policy. Existing one-owner route and wizard endpoint tests remain. ADR required:no, routine internal dependency relocation within Provider_Contract preserves canonical API/data identities under ADR097/179; document and test the re-export. Re-run the unchanged1040 module ratchet, never raise its budget.

- After-close and coalesced original controls pass39.20/57.17s after native fixes. Same-session fails at a source-owner assertion before gating: drain_preparation_reads moved to Backup_Recovery.runtime_producer_lifetime, but the fixture still compares its globals to the Chat re-export module. Bind the check to its actual defining module and additionally assert the public alias is that exact function; retain original compiled-source/code/return/retirement checks. This is a stale source-owner test seam, not a production drain failure.
- Keep the unchanged7078 runtime size budget: place the two-owner drain in the existing Console finite-preparation module and call it in the original runtime drain slot. This is the same captured-set/cancellation operation, with no new task or owner registry; no pin increases or unrelated size-only cleanup.

- Endpoint closure RED confirmed2.96s: actualOpenAI/local resolution loadsProvider_Contract.provider_setup_facts before any setup caller. Move the single suffix table to its already-resident endpoint owner and re-export exact identity; then run original routing/facts/wizard checks and the unchanged module census.

- Before first profiling execution, review catches provenance-map IO inside intermediate journey save(): move fullbefore/afterPython maps into sidecars written before app imports/after shutdown, retaining names/digests/count in small reports. Current pair setup hashes warm filesystem cache; make no OS-cold or historicalstartupcomparison. Observed import_ms includes observer-targetimports and is not compared toplain startup. Main six-return windows stayidenticalacrossplainpairs.

### Connected context-fixture repairs

The connected selection89P/3F exposes old fixture assumptions. Cold/expired spy replaces context_control_inputs and therefore intentionally fails exact callback qualification, selecting the custom fallback it then forbids. Observe the original callable locally with sys.monitoring; retain original thread/read counts and assert no live entries. The two compaction cases reach the correct PAUSED/CONTEXT_COMPACTION hold but fail before resume: current no-memory RAW optimization correctly skips version reads. Seed an actual RESET selection after one real accepted seed Send, before threshold-reaching Send captures its scope. RAW semantics stay unchanged, but the original lineage/version path becomes necessary. Keep original hold/echo/source-key/in-flight rejection and resume assertions. No production change for these fixtures; no Send refusal attribution from the expected Ask hold log.

### Final contract verification and publication fixture

Current final contracts:11passed/1failed; unchanged UI-ready census passes1040/1040. The remaining workbench Send test times out in bare nonblocking `_sync_controls` before any readiness comparison. Its sibling payload tests already use `_sync_console_control_bar_now`, which awaits original warm-up and obtains native config locks before checked publication. Use that helper only for the two workbench agreement assertions; retain bare polling in the tests of refusal and deferred replay. This changes no product gate or timeout. The exact acquisition that prevented the original four-second polling run is unobserved. Repaired agreement plus the entire P0 gate:179passed in172.07s. Pin the new native Close file in both the existing admission-sensitive workflow and P0 inventory.

### Profiling qualification and current measurements

Retain plain-baseline-plain-1 as failed setup: sandboxed creation under the real home did not reach the app before the300s deadline. Windows mkdtemp can repeatedly retry denied creation; use one unique native mkdir to fail promptly and run the required native-private fixture with authorized host access. The next native launch created its private root immediately. Deadline cleanup now kills the owned Windows redirector tree while its parent still exists; forced termination never counts as normal retirement.

Retain plain-baseline-native-1 as adverse: first return input1014ms, attachment exceeded10s, exit1, guards0/sourceunchanged. It did not prove the initial attachment had settled. Require a completed initial attachment before measuring returns, without changing the return's10s bound. Diagnostic-baseline-native-2 completes all6 returns with guards0/unchanged product and CSS sources: attachment2491–4098ms, input-to-draft3.5–58.3ms, navigation-to-input52.5–714.5ms. Its first ScreenResume restyle consumes604ms wall/469ms thread CPU; later synchronous spans are under21ms. Preparation includes substantial worker waits; no deadlock, stale-read reuse, or permission-policy change is justified. Collect a separate first-return cProfile to rank candidate-generation CPU before selecting the smallest correction; instrumented timings are not compared with plain latencies.
