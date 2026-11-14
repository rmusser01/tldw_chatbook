# Replacement-checkout continuation: Chat mark ownership and latency

Latest bounded rendering work and qualification state are recorded in
[Approved row staging](#approved-row-staging). Historical pending/failure
checkpoints below remain retained, not silently rewritten as passing evidence.

The user explicitly approved the replacement checkout and continued latency /
retained-Chat work. This isolated branch starts at preserved `a445143bde`;
the missing checkout and old raw evidence were not recreated. Dirty main, the
original ownership branch and all other sessions/worktrees remain untouched.
No other session was contacted or controlled; no push or integration occurred.

## Proven native survivor and scoped repair

Four full original diagnostic switcher journeys completed all 70 records on
unchanged `a445143bde`. The first three reported no captured Chat survivor after
physical Runner/default-executor join. The fourth avoids the earlier expensive
pre-retirement census and reproduces one native registered handle, still open
after its `asyncio_4` owner exits. Its exact opening path is receipt hydration,
fleet-mark reconciliation and `list_marked_conversation_ids`. Strong native
references exclude GC cleanup. Observation perturbs scheduling; earlier clean
runs do not disprove the survivor, and the original unmount count alone was not
physical-join evidence. No foreign SQL or close was used to attribute it.

Under accepted ADR225, the shared materialized list read now uses the same exact
file-backed `operation_owned_connection` guard as the adjacent unread readers.
Cache hits, query, order, limit, cache-generation check and error propagation are
unchanged. Warm transactions, custom owners and memory databases stay borrowed.
The existing `run_owned_db_call` also bounds the complete receipt hydration
callback to its captured marks database: reconciliation can open another handle
while clearing a stale mark or restoring fallback evidence. AgentRuns' original
finally-close, result and cancellation/publication conditions remain intact.
No new service interface, global registry, sweep, foreign close or GC policy.

## RED, partial repair and affected verification

- Shared readers: **two failed / 16 passed**. Success left native SQL usable;
  a SQL failure retained an extra registered handle. Both failures are the new
  list case; existing borrower/memory/custom controls pass.
- Corrected original hydration: **three failed** after real executor join.
  The first probe incorrectly included the constructor's main-thread borrower
  in its exited-worker census; that setup-invalid failure is retained separately.
- Reader-only repair: **23 passed / two failed / 14 deselected**. Empty hydration
  closes, but actual stale/fallback reconciliation still leaks; not green.
- Complete bounded repair: **23 passed / 18 deselected in 13.80s**, including
  native cold/error/cache reads, actual joined hydration and warm/degraded
  worker transaction preservation. Native assertions precede fixture cleanup.
- Original affected five-module run: **109 passed / three failed in 79.53s**.
  All three Fleet bridge cases refuse `raw_source_selection_changed` at prompt
  config lookup, before the changed mark/hydration paths. The exact original
  bodies pass **three tests in 5.76s** using existing `bootstrap_profile` mode.
  Only those three nodes now opt into that fixture; no body/assertion/getter,
  deadline or admission guard changed. Final ordinary five-module selection:
  **112 passed in 80.86s**, strict ResourceWarning/unraisable filters, no pytest
  warnings. The failed aggregate is retained, not relabelled.
- Size/extracted-binding controls: **58 passed in 15.54s**, ceilings unchanged.
  All eleven artifact guards pass, including pinned-public-input Mermaid
  reproduction. Changed tests and shared marks reader are Ruff clean; runtime's
  30 and Fleet test's three inherited code/message diagnostics are unchanged.
  New test formatting and changed production function/range formatting pass;
  unrelated inherited whole-file formatting is not rewritten or declared clean.

Actual app/native Interrupt selection passes **five tests in 89.48s**: cold,
warm, replacement, three repeated Runner/default-executor joins before cleanup,
and the original mounted Stop/Interrupt entry. Each of the four private-profile
children retains one sandbox pytest-cache permission warning; a clean parent
summary is not a warning-free child claim. Clean-commit scale remeasurement is
pending this checkpoint. TASK34661 remains In Progress; no whole-app retirement
or broader qualification claim.

## Latency remains non-green

Fresh uninstrumented baseline: scale corpus 10000 conversations / 250000 eligible
messages; Keyword 300-query P95 **164.848875ms**, loop **41.327625ms**, both pass.
Original 70-record UI matrix fails 12 unchanged 50ms interval gates, max
**179.211708ms**; busy **48.664792ms** stays below 100ms. No app exception.

Call-through config census measures 883 actual reads in 24 passes, combined
0.971251–5.224750ms per pass, not enough to explain the large stalls. Passive GC
and callback CPU observations preserve collector thresholds/enabled state and
original callbacks. A later diagnostic records 301 callbacks over 5ms and 36
paired gen-2 events, no drops/unpaired events. Long 80–105ms callbacks often use
nearly matching thread CPU; automatic GC accounts for a substantial portion of
some, but not all, intervals. A wide preparation Worker uses 55.964ms CPU in
60.320ms wall with no gen-2 overlap. Trigger locations/code samples are not
exclusive attribution or proof of retained-object ownership.

One small row-construction probe fails before mounting with the test harness's
missing theme-variable import; zero rows, setup-invalid, no performance claim.
No speculative memo/cache/GC or rendering change has been implemented. Further
latency work needs a causal bounded design; all 50ms/100ms limits stand.
Native ordinary quit/viewport, Windows Terminal, three unfamiliar participants,
and broader aggregate qualifications remain unwaived. Prior Terminal UI denial
is not bypassed. No full repository sweep or fresh-context review is claimed.

## Local raw provenance

Full raw logs/dumps remain LOCAL under
`/private/tmp/console-latency-continuation-fT3VrM`; none is copied into Git.
Older missing raw originals were not reverified. SHA256 values below were read
from the retained current originals; this is an audited outcome/hash receipt,
not a full-copy claim or permission to publish environment metadata.

| Local receipt | SHA256 |
| --- | --- |
| marks-red.log | c95233a9983836752a7c34e1d2aad89bd506490b6026d3a8f49e99d5bce8c42f |
| hydration-red-corrected.log | dad1fd8017b380f28f01aeb6f69b9fa714ad545007d9a76ee295c11b7179dfd5 |
| marks-partial.log | eedb428c0775bcdc4a8d378e86ef8d1aebd03b6b9cda9d6bd559dec12c04e304 |
| marks-green.log | 45f66ba81c8df18f51ab26c7ac0d86e546b0a055e483a3adefd2b555c5a05376 |
| marks-affected.log | b3d22a4776321fcedd43219047695a320e07fd414f66ea97f21dc6853293d066 |
| marks-affected-final.log | 12fdc9a6558639188a07e87a5592cc18e1ad3559ea4d27830ab22cec9011ecc0 |
| chat-handle-diagnostic-4.json | 69817543bb897036ad237b95167a523ad1bfcb8ea4afe2fbb65b4af61ca0c6cf |
| callback-cpu-diagnostic.json | e39b9f209da1917f8bb2de2041ef47cc1e2b399eb09be6875ba90db6afa2429d |
| size-green.log | 1b91a54015d653fb7ef3040973b9fa7999344a548c663d4bbccd8d49fe3cd58d |
| artifact-guards.log | 1944f8706ce642d48a4b070c3b9259a213d4c8985370d5f5e0d5a65a873102f5 |
| mermaid-guard.log | 7d735cc6cda40478416e3df9eda1dc7919845f1109841377210d3f7eda661b73 |
| native-app-final.log | 60bdc13e3795e990c3c97f7afec987c05d1d811bdc82909692079dbe265265e1 |
| native-app-final repeated private child pytest.log | f9681357ef5558f35d98b875c50932905cb7d2ceca1ff0131b5140a543fe0c43 |

## Clean-commit remeasurement — source `c260d3f0aa`

This supersedes the pending-remeasurement wording above. The ownership repair
is committed locally as `c260d3f0aa4919ab0555e1d359f8676ba3863075`. The original
guard correctly rejects reusing a corpus whose receipt names another head;
that setup refusal is retained, not bypassed or counted as a timing result.
A fresh source-bound scale corpus builds 10000 conversations / 250004 messages /
10000 search documents, 250000 eligible messages, ready/integrity OK; seed
275.03s and index 7.04s. Registered handles after cleanup: zero.

Original guarded Keyword: **300 samples pass**, P95 **212.082250ms** against
300ms, maximum loop **24.124542ms** against 50ms. Correctness, source and corpus
guards hold; owned database descriptors and registered handles are zero.

Original full UI: **70 records complete, 11 unchanged 50ms interval failures**,
maximum **233.984250ms**. Busy maximum **38.499333ms** passes 100ms. All 60 searches
and eight exact activations complete; no app exception. Source/corpus stay exact.
There is no callback/paint/collector instrumentation during this measurement.
After the original CLI's Runner/default-executor join and final source guard,
a local wrapper inspects only the original failed coroutine's known database
reference: **zero registered Chat handles and no database descriptor basenames**.
It performs no SQL, close, collection or test cleanup. Five regular descriptors
remain (the process instance lock, storage lease, fault log and two references to
the measurement log); they are not falsely labelled a whole-process zero census.
This resolves the demonstrated Chat survivor, not the failing latency matrix.

A corrected small rendering diagnostic imports the same theme-variable
registration as the existing test fixture and awaits actual painted content.
The real styled modal completes two 50-row constructor blocks, **154.616084ms**
and **148.271501ms** thread CPU (162.455459ms / 155.401671ms wall). Those numbers
include any GC during construction and use synthetic loader rows; they are not
exclusive CPU, scale/native or before/after optimization evidence. Both earlier
setup-invalid probes (missing theme and premature paint assertion) remain local.

Proposed next latency work is a bounded staged row build with complete/current
generation publication, non-actionable pending state and unchanged focus,
pointer/keyboard authority, cancellation and timing limits. A simple yield alone
is unsafe: readiness currently precedes awaited commit. This concrete UI design
is awaiting user approval; no UI implementation, GC/cache policy or new limit is
applied. The UI-performance skill reinforces measurement and preservation of
interaction contracts, not a redesign or design-context repair.

These additional full originals remain LOCAL in the same retained root:

| Local receipt | SHA256 |
| --- | --- |
| scale-c260/build-receipt.json | de3c9b387ac4b2dd5242302e8470f9f3df822a1acb80f9060043448102d7d961 |
| keyword-c260-fresh/keyword-receipt.json | 0c8f7e431540bb9d21461bb191fb8109f73f02e86ba0b3a2a7a3c5ea57279741 |
| ui-c260/ui-evidence/ui-latency-evidence.json | 0154b64d63f3b13c24e94647aadf5ce6d5dacca972014ec8e14d32aafe81e5c2 |
| post-fix-physical-boundary.json | 86ce568f9166da02b42364e54041f38230190a9b1f5addcbc8e0301235d3645c |
| measure_fixed_ui.py | 2209ef4a496c590d564545c2e0f1939c82a9be2e30a71caf70027808b789b457 |
| row-cost-c260-2/row-cost.json | 9a649d973732703d9acd6a80a4c7fb0231032548798900c0d8af60519db847bd |
| row-cost-c260-2/frame.txt | d0f620c02c052078e8642a5cc6f705b82613c8b7a9057c3e8095a52e40845766 |
| observe_row_cost.py | 2b4bdce97c7c86f543ebb1cde2e9e4a6b524d45aa4eeaa3a5b6c6d132afa09fe |

## Approved row staging

The user explicitly approved bounded row staging, superseding the preceding
approval-pending checkpoint. The existing fifty-row page is constructed in
five-row yielding batches. Existing pending state now lasts through the complete
hidden mount; rows are disabled and their payload generation is published only
after the original current-query/authority fences still match. Search,
reconciliation and immediate Enter serialize publication within this modal;
stale construction or mount cannot expose an obsolete page. Focus, keyboard and
pointer identity, mode/page restoration, cancellation, all timing limits and
the original fifty-result cap remain. No global GC/cache policy, row reuse,
virtualization, dependency, token, stylesheet or activation change.

Mounted baseline RED: four failures in 4.55s reproduce uninterrupted fifty-row
construction and false readiness before mount completion. An earlier ordinary
attempt had four config-source setup refusals, not behavioral RED. The same
native styled harness uses the existing isolated bootstrap-profile fixture.
First implementation run retained two test-observation errors (focus had not
settled and the debounce was not awaited); after correcting those observations,
one remaining check incorrectly expected a JSON stable key to end with a bare
conversation ID. These are retained, not credited production failures.

Seven mounted controls pass in 6.76s. The broader affected selection passes
**198 tests in 250.18s**, including original Character/Active/History, trust,
reuse/mode, archive, geometry, size/extraction and token-governance contracts.
Original source-bound UI bodies use the same bootstrap-profile fixture via a
local collection-only plugin; no getter, assertion, admission or cleanup changes.
All eight actual governance child logs also report a pass without pytest warnings.
Final strengthened queued-Pressed controls pass **seven tests in 5.43s** under
strict ResourceWarning/unraisable filters, after import-order-only cleanup.
Both touched Python files are Ruff and formatter clean. All eleven unchanged
artifact guard commands pass. The short staging suite joins the existing PR
census (153 files, floor 153); no coverage is removed. Whitespace is clean.

Two additional changed-authority snapshot controls cover construction and mount.
Final complete staging selection: **nine passed in 8.20s**, including queued
Pressed events, no pytest warnings under the same strict resource filters.

Inline review covers the sole commit caller, all yielding/mount boundaries,
shared mode rendering, payload generation, focus and cancellation. No other
session/agent was contacted; this is not a fresh-context review. ADR required:
no, existing ADR120/150/161/198 preserve the internal rendering contract and
ADR225 still governs ownership. Tasks remain In Progress with qualification
criteria open. Fresh clean-commit scale/Keyword/full UI measurement is next;
targeted correctness passes are not a 50ms latency or native/Windows/human waiver.

Additional full originals remain LOCAL in the same root; only audited outcomes
and hashes are published:

| Local receipt | SHA256 |
| --- | --- |
| staging-red.log (setup refusal) | 0f635da3244859ebc93b7607f610b5f9180433d1525d9fe60cc921e786c4a62e |
| staging-red-native.log | 9e187081ff42fcc35ef933297ec5c87a238aed7795333a2b5e677d7754539eac |
| staging-green.log | 30800620ccb21ee56b0d32f82f61ddd4c5ad19873e5048372f3f624607269f8c |
| staging-green2.log | 292a090de1ee0c613e6cffdb042f22b8e249ce14ea214959dc471e81e3333b85 |
| staging-green3.log | 3a005d08a76e18f466bcb3794778c717d9ccd90fe6ae40b61d7c8189c8470e07 |
| staging-affected.log | 2ec866aa004ca4f527da04e5dfe1e5436b7fd2a25ba8a78953406ee24e4634ba |
| staging-final.log | 3a20106c2d26ca70d996b6670f3e5421e96337f0f9b46705ad49ad035ff4399f |
| staging-guards.json | a1d7ed92758e3d22447a2826beef3331612f1d751e9b08ae798ea1bd903da551 |
| staging_ui_bootstrap.py | 4c008ea6563e7e7f94d81cd8aae3222e37c51fae803cb932263cc8e3fe454a66 |
| staging-final-authority.log | bd8e14c62817d8f0c59a76f7397a46e0042411dbc53f19cef5eb5ccdb2758503 |

### Frozen staged-source qualification — `2a4ffebfdd`

The bounded patch is committed locally as
`2a4ffebfdd4e960bda542cf7de713c3b361ff270`. No push or integration. The first
build command supplied an abbreviated expected head, which the unchanged guard
correctly refused before reserving a corpus. That setup failure is retained.
The full exact hash then built a new source-bound corpus: 10000 conversations /
250004 messages / 10000 search documents, 250000 eligible messages, ready index,
integrity OK; seed172.386s/index3.804s, registered cleanup handles0.

Original Keyword qualification **passes all300 samples**: P95147.540250ms
against300ms, maximum loop28.758375ms against50ms, zero owned cleanup handles
or database descriptors. Source/corpus checks are true. This is standalone
Keyword clearance only, not UI/native/Windows/human clearance.

Original uninstrumented UI completes all70 records:60 exact searches, eight
exact OPENED activations, two preparation windows. It **FAILS nine unchanged
50ms intervals**, maximum150.403542ms. All eight activation windows fail
(66.709459–150.403542ms), as does `120x50-body-7` (69.416000ms). Both preparation
windows pass (49.030584/28.452000ms) and every busy-paint check passes
(maximum44.972541ms against100ms). Source/corpus guards hold; app exceptionNone,
registered Chat handles0 at unmount. This is not overall latency qualification
or a controlled cross-run attribution of the smaller maxima to row staging.

The same observation-only post-failure wrapper inspects the original database
only after the original CLI's Runner/default-executor physical join and final
source guard. Actual database observed, native registered Chat handles0, app
exceptionNone; Darwin descriptors contain no database basename. Five regular
lock/lease/log descriptors remain, not a whole-process-zero claim. No measured
callback, SQL, close, forced collection or cleanup changed. This reinforces the
bounded hydration repair; it is not universal application-retirement proof.

One further diagnostic repeats the original matrix with call-through callback
wall/thread-CPU timings and passive generation2 observations, without the older
strong-handle observer. It retains70 records,413 callbacks over5ms,36 paired
gen2 events (32 main-thread), no dropped/unpaired records, observer maximum
1.492583ms. Thresholds remain700/10/10 and GC remains enabled. It also FAILS:
11 windows/12 individual failures, loop maximum263.510125ms and busy maximum
643.170125ms. These instrumented timings are **not qualification evidence**.

Six of the eight longest observed activation callbacks overlap main-thread GC
by41.243–72.988ms; other activation callbacks use roughly50–61ms wall time with
no such overlap, including the serialized activation coordinator. Long
non-GC preparation callbacks remain too. Inclusive CPU, trigger/callback names
and overlap do not establish exclusive object ownership, unique cause, or which
pause occurred in the original uninstrumented failing search. The residual
failures cannot truthfully be labelled "just GC". No activation-contract or GC
policy change was made; further correction needs separate cause/design work.

All benchmark/diagnostic processes finished. Source stays unchanged; tasks remain
In Progress and their qualification criteria stay open. Native ordinary quit,
Windows Terminal, unfamiliar participants and integration remain unwaived;
other sessions/worktrees and the prior Terminal UI denial were not touched.
Only audited outcomes/hashes below are published; full new dumps remain LOCAL.

| Local receipt | SHA256 |
| --- | --- |
| scale-staging-build.log (abbreviated-head refusal) | cee7fd0b75beff1ef6da61e38c4db84798a71eb10c6eec85150c316de538f6df |
| scale-staging-2a4ffeb/build-receipt.json | 781b7ec3c293f3f89ab70d08e3df2b77cbf6e12628ec5fbbe2e4f0ffa52bab30 |
| keyword-staging-2a4ffeb/keyword-receipt.json | 7063e7c32e184726b7b1e019f01e84e99e68ab5d07fec197fd0b0df2d675fdcf |
| ui-staging-2a4ffeb/ui-evidence/ui-latency-evidence.json | 25da38789ab2437ee135e0c8c9a537d22555a541537f18cce89639df74c9a232 |
| ui-staging.log | af38396163464c942bd193090521d956db97b103d0e7633e5d17bb58e0b551a9 |
| post-staging-physical-boundary.json | 54560b4565645a8c11d57404cf6f55860ba91779574e01e6de6027bda52ec4e3 |
| measure_staged_ui.py | 20180f921a8531bf98ed150a34309b71b31b70bda01f2721867ecceac18862e9 |
| staging-callback-diagnostic.json | 2cec5ab4b2b57e80d7863baa718bf583ed62333a7746f8babd15a04b0d887aa1 |
| ui-staging-diagnostic-2a4ffeb/ui-evidence/ui-latency-evidence.json | d7bc81540abf21699f4129703f4bad0073001fe945a240767f031714e0f38a0c |
| staging-callback-diagnostic.log | 81fda3bb60ea12884f5c5c8a140fbc262b3f806f0a04ca2cdcdcfa552721c9fc |
| observe_staged_callbacks.py | fe68283a0e5910ae69c85704f876d70cca3ef2a281c523918ddca57af1ccd91e |

## Approved guarded refresh phases — 2026-10-20

The user explicitly approved cooperative yields between existing Console refresh
phases, outside native lock sections, with teardown checks and unchanged exact
activation, cancellation, projections and timing limits. The shared native UI
refresh now yields after Character scope refresh, checked rail/control projection
and native tab synchronization. Each new boundary rejects a torn-down screen;
existing checked config lifetimes, task-scoped fingerprint validation, coalesced
replay and final activation proof remain. No GC/cache/lifetime policy, appearance,
profile/getter, dependency or measurement contract changed. Condensing the same
teardown rationale keeps the screen at25204 lines/756 methods; no ceiling rises.
ADR required: no new ADR; existing ADR120/150/161/198/225 apply to this bounded fix.

Pre-change direct clocks separate non-GC activation work from automatic gen2:
the combined diagnostic records185 activation callbacks over5ms and33 paired gen2
events, with valid accounting and no drops/unpaired observations. Long timer
callbacks overlap roughly40–49ms of main-thread GC; non-GC activation callbacks
also bundle roughly40–47ms. All eight original activation windows still fail in
that diagnostic. Nested clocks/overlap are not exclusive cost or unique-cause
proof. Both selective cProfile attempts are rejected for impossible accounting.
The first method probe is setup-invalid; no invalid measurement earns a pass.

Native constructor/partial-root diagnostics identify substantial compositor
Strip/FIFOCache allocation and cache graphs, not a leak. The opaque-backdrop
control reduces activation creations by about61% but still fails all8 activations
and changes appearance: it is not shipped. The existing derivation-scope control
is inconclusive and is not shipped. Full new dumps stay LOCAL; audited outcomes
and the [23-entry hash manifest](phase-yields-sha256.txt) retain provenance without
publishing environment metadata. Local root remains the existing continuation
directory `/private/tmp/console-latency-continuation-fT3VrM`.

Native mounted RED: the corrected unchanged-source run fails all9 queued-input,
teardown and cancellation cases; the final admitted-controls RED fails all3
selected cases. Earlier invalid busy/background observer assertions and original
Splash/cache warnings stay retained. Final9 native cases pass in86.08s with no
pytest warnings; their logs may still contain existing application warnings.
The observer now waits through the installed bounded quiescence helper before
checking idle/cache retirement, rather than mistaking a trailing refresh for the
target task. A competing writer defeated an earlier cross-thread availability
assertion; direct native ownership proves the UI loop owns neither lock at the
new boundaries. The test still checks real fresh transcript publication. New
coverage joins the existing PR census, growing its floor153→154.

Affected verification is **NON-GREEN**, not waived: the first combined run was
interrupted at295.45s, with1 failed/5 passed and later workspace cases incomplete.
The zero-read failure counts two off-loop Files-availability registry reads;
exact prior-refresh-method and current-method controls both reproduce it, with
all observed binding reads on executor threads. This is a method-only control,
not a complete prior-tree qualification. A separate old-method session-create
control reports a passed body but stalls during worker settlement; interrupted
exit2 at86.02s remains NON-GREEN. Both interrupted stacks enter the existing
workspace Files snapshot and `_run_console_default_worker_settled`; no unique
cause, identical interruption phase or blanket lifetime clearance is claimed.

The remaining affected modules complete116 passes/one failure in135.49s. The
failure is the unchanged Library screen36654 lines against unchanged35855 budget;
both inputs are byte-identical to the baseline. No unrelated Library change or
budget increase is made. The Console size node separately passes (one cache
permission warning retained). New test Ruff/format pass; screen inherited Ruff
189 findings and census-script inherited EXE001 remain unchanged, not whole-file
lint-clean claims. Original qualification on a fresh committed source is still
pending at this checkpoint. TASK31966/34661 remain In Progress with qualification
and native/Windows/participant/integration gaps open; no other session controlled.

### Fresh committed-source qualification — guarded phases

Source commit `1b68cbae382b6fb1eacf567b7e2b41b5e177bb89` is clean and exact
through the original build, Keyword and UI source guards. Fresh scale corpus:
10000 conversations/250004 messages/10000 indexed documents, ready/integrity OK,
seed109.599254s/index3.729940s, registered cleanup handles0. No older receipt was
rebound. Original Keyword300 passes: P95107.171333ms against300ms and maximum
loop6.875916ms against50ms; native registered cleanup handles0.

Original **uninstrumented UI remains NON-GREEN**. All70 records complete:
60 exact searches, eight exact OPENED activations and two preparation windows.
All searches and preparation windows meet50ms, and every busy paint meets100ms
(maximum45.244208ms). Five activation windows fail50ms, maximum79.496000ms.
The eight activation maxima are72.582666,49.773708,47.883417,72.891625,
23.067041,69.777083,76.165917,79.496000ms. Source/corpus guards hold and app
exception isNone. This single run does not establish a controlled cross-run
speedup, statistical improvement or full latency qualification.

Observation only after the original CLI's Runner/default-executor physical join
and final source guard finds the actual Chat database, registered handles0,
no database descriptors and app exceptionNone. Five regular process-owned
log/lock/lease descriptors remain, not a whole-process-zero claim. No measured
callback, SQL, close or collection was changed by the post-failure observer.

One separate passive call-through diagnostic retains the original70 records but
also FAILS: eight intervals, including preparation/search, maximum95.136167ms;
busy maximum80.916125ms. It records191 callbacks over5ms,177 during activation,
33 paired gen2 events (30 main-thread), zero drops/unpaired, observer maximum
0.021666ms. GC remains enabled at unchanged700/10/10 thresholds. Among observed
activation callbacks without main-thread gen2 overlap, the largest is38.432ms
and the serialized activation coordinator peaks34.545ms. A73.256ms timer
callback overlaps39.117ms of gen2; an observed Markdown callback overlaps46.017ms
of gen2. These are instrumented inclusive observations, not qualification,
exclusive object cost, or proof that every original failure was GC-caused.

All owned verification processes finished. The bounded phase-yield implementation
and its regression checks are retained; GC/render allocation and the recorded
workspace/Library verification gaps are not waived. The35-entry hash manifest
verifies all listed local receipts. A concluding evidence-only commit changes no
tested production/test/script/workflow/package input. Both tasks remain In
Progress; no push, integration, GC tuning, full repository sweep or other-session
activity. Git's existing gc.log/unreachable-object warnings remain untouched.
