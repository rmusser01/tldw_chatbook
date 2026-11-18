# Owner-free style colors implementation plan

Execute inline under executing-plans; user forbids contacting other sessions.
Goal: remove the proven native color-cache owner retention without changing paint.
Spec: [approved ADR361](../../../backlog/decisions/361-owner-free-textual-background-color-cache.md).
Tech: Python3.12, pinned Textual8.2.8, stdlib LRU1024.

ADR required: yes
ADR path: backlog/decisions/361-owner-free-textual-background-color-cache.md
Reason: explicitly approved process-local native cache-key policy.

Global constraints: unchanged50ms loop/100ms busy bounds; no collector/shared
dependency/appearance/activation/lifetime policy edits, no other sessions,
raw diagnostic dumps local only, no whole-repository sweep/push/merge.

### Task 1: Native color-cache ownership

Files: create `tldw_chatbook/Utils/textual_style_colors.py`, create
`Tests/CI/test_textual_style_color_ownership.py`, modify the existing
`tldw_chatbook.__init__._install_textual_compatibility_shims` entry only.
Interface: `install_owner_free_style_colors() -> bool`, using the original native
method body with its unused first argument set to None and two immutable colors.

- [ ] Add native tests: after32 `StylesCache().get_inner_outer(base, bg)` calls,
  weakrefs to discarded owners are all None WITHOUT collection; equal inputs
  across owners reuse computation, not mutable ANSI memos. Run targeted file,
  expect native retention RED.
- [ ] Implement a guarded installer: inspect native body/1024 untyped LRU and
  reject any body reading `cls`; install static color-only stdlib LRU calling
  `native.__wrapped__(None, base, background)`. Detach native dataclass fields
  per result to preserve renderer-specific ANSI state. Invoke from existing shim.
- [ ] Run the entire new file: expect owner proof,1024-entry promotion/eviction,
  unchanged old lines/subclass/error/idempotence/mismatch and real native render
  parity all PASS. Run existing Strip/GC/renderer affected contracts, lint/format
  and eleven artifact guards; inspect and address actual failures.
- [ ] Record audited outcomes/hashes, self-review exact diff and commit locally.
  Do not mark parent task Done; old failures and external gates stay open.
- [ ] Build fresh clean exact-head scale corpus, run original Keyword300 and UI70
  sequentially; expect all fences intact, retain any remaining timing failures
  and continue scoped root-cause work without changing limits.

Pre-flight: no shared task interfaces; one bounded task. No inactive-decoration
or identity-divide production changes. Fresh-context review via other sessions
is excluded by the user's boundary; perform inline self-review.

### Task 2: Proven native credential-poll work

ADR required: no
ADR path: N/A; reuse existing checked-config/derivation contracts and ADR225.
Reason: routine caller correction, no new runtime/cache/lifetime policy.

- [x] Reproduce actual REBUILD/FILE blocking, preserve watchdog/holder evidence.
- [x] Reuse nonblocking checked entry and existing whole-sync fresh replay.
- [x] Pin four duplicate equivalent native workspace builds, then reuse existing
  task-owned fingerprint scope. Keep revision/status/version, all original
  getters, admission, expiry/cadence and timing limits.
- [x] Verify original subscription/endpoint/Resend/tick cases; retain inherited
  setup/resource failures and calibrate only existing opt-in exact-owner fixtures.
  Final65 parent/6 actual private children pass strict DB retirement.
- [x] Self-review, static/no-added-lint, affected size ratchet and11 guards.
- [ ] Commit locally and run fresh exact-source scale/Keyword/UI qualification;
  continue any remaining GC/render latency failures without changing limits.

### Task 3: Approved owner-local result reuse and plain Character labels

ADR required: yes for native result reuse; no new ADR for app-owned labels.
ADR path: backlog/decisions/361-owner-free-textual-background-color-cache.md.
Reason: explicit owner approval refines the existing native compatibility policy;
native immutable Content is an existing UI render value, not a cache policy.

- [x] Pin actual same-owner allocation RED; retain cross-owner ANSI/weakref,
  cache-clear invalidation and every-call global LRU promotion/eviction controls.
- [x] Reuse one detached pair on exact native StylesCache only; no owner keys,
  reverse owner refs, subclass mutation, collector or timing-limit changes.
- [x] Replace plain Character labels' Rich Text round trip with native Content;
  pin allocation, exact paint/Unicode/clipping, focus and target parity. Keep
  styled Active/History labels and all authority/staging contracts unchanged.
- [ ] Run affected original contracts, static/artifact guards and inline review;
  record safe outcomes/hashes, commit locally, fresh exact-source qualification.
  Preserve residual failures and continue; no other sessions or integration.

### Task 4: Due-timer admission within the approved phases

ADR required: no
ADR path: N/A; existing guarded-phase design and ADR225 ownership.
Reason: scheduling correction at the same three lock-free boundaries, no new
runtime/cache/collector/lifetime policy. No claim this alone clears latency.

- [x] Align original sentinel intervals with passive clocks; preserve six
  non-gen2 intervals and separate GC/render evidence, no unique attribution.
- [x] Expand real mounted phase tests to ready and due timers; old native RED3.
- [x] Positive1ms timer yields at the same three boundaries, every original
  teardown/cancellation/error/freshness guard and operation/timing bound retained.
- [x] Retain17/18 failure and admission controls; test retry observes existing
  real readiness within unchanged10s. Final exact-old oracle still RED3.
- [x] Final50 caller/phase tests, static/size/eleven guards and inline review.
- [ ] Local commit, fresh exact-head original qualification and continued
  isolation of remaining native render/GC and scheduling failures.

### Task 5: Plain session-tab render values

ADR required: no
ADR path: N/A; ADR150/161 native render-value patterns.
Reason: remove only two Rich-to-native label conversions; no new policy.

- [x] Read both creation/update paths and native Button conversion; mounted
  allocation RED2 with native paint/markers/metadata/focus/target parity.
- [x] Use native Content at only those two plain paths; styled switcher,
  transcript header/coach-mark, authority and every timing/GC bound unchanged.
- [x] Final25 native/tab tests plus18 shipped-style/overflow/selection callers.
- [x] Broader targeted45 SQLite/closed-cursor/Interrupt retirement contracts,
  including all4 actual native-app private children; no aggregate/native waiver.
- [x] Static20 inherited/no additions, format/eleven guards/inline review.
- [ ] Local commit and fresh exact-head original scale/Keyword/UI matrix;
  continue residual failures. BoxModel policy remains separately unapproved.

### Task 6: Skip discarded new-chat settings during active draft refresh

ADR required: no
ADR path: N/A; existing ADR095 and Store.ensure_session creation contract.
Reason: remove an ignored derivation, not a memo, cache or ownership policy.

- [x] Trace every shared draft caller, actual ensure_session and the sibling
  creation-only settings path; real mounted call-through RED sees ten builds.
- [x] Derive defaults only when creating a session. Preserve actual creation
  provenance, generation, existing settings identity and draft/undo authority.
- [x] Run original cold/handoff/session/undo/snapshot callers, compare any
  failure to the exact parent method before claiming a regression or baseline.
- [x] Static, eleven artifact guards and inline self-review; safe outcomes and
  hashes only.
- [ ] Local commit, fresh exact-head original scale/Keyword/UI matrix.

Verification checkpoint:48 initial regression/session tests pass. Expanded
callers102pass/1failure/+571FD warning; exact parent reproduces the empty-redo
assertion. Adjacent native draft-settlement setup reused, all asserts retained.
Existing captured-owner fixture opt-in only in the three proven leaking harness
modules: native required-census RED with3body passes, GREEN4, final130PASS222.59s/
no pytestwarnings and all130case DB-descriptor observations0. No production
lifetime change or aggregate/native waiver. New test lint/format clean;
controller47 inherited lint and one prior format difference unchanged, caller
lint1/2/3 unchanged and formats pass. Eleven final guards and inline review pass.
Local commit and its fresh original qualification are the remaining Task6 steps.

### Task 7: Equal mode copy and exact original-caller ownership

ADR required: no
ADR path: N/A; existing ADR150/161 rendering and ADR225 test-owner scopes.
Reason: redundant update and missing test-owner scopes, no new policy.

- [x] Real native invalidation observation/mounted RED2; guard only existing
  mode copy, retain every run-chip projection and original error path.
- [x] Exact parent reproduces caller failure; scope complete raw test request
  Threads, capture exact navigation alias/injected Chat with existing fixture.
- [x] Final70 PASS with required parent/child retirement, original-body AST,
  static/size/guards/inline review and safe outcomes/hashes.
- [ ] Local commit/fresh exact-head qualification; continue approved BoxModel
  separately. All timing/external qualification gates unchanged.

Task6 committed75e26e347477779f3399e9b81b58a8dbcb3ea14b. Keyword300 PASS
105.4345ms P95/10.214209ms loop; UI70 NON-GREEN74.916/62.749/73.630ms, all
outcome/busy/source/corpus/physical Chat0/noDB fences intact. Human now approved
bounded ADR-backed BoxModel reuse; earlier unapproved notes are historical only.

### Task 8: Approved owner-local native BoxModel results

ADR required: yes
ADR path: backlog/decisions/362-owner-local-textual-box-model-results.md
Reason: explicitly approved, pinned native runtime/cache compatibility boundary.

1. Native allocation RED; preserve exact original method, all native cache
   operations/side effects and subclasses. Reuse only equal immutable previous
   native MRU values after a miss; no new owner/global memo.
2. Native geometry/paint, LRU hits/misses/order/eviction/clear/replacement,
   override/custom/unsupported/idempotence and GC-invariance controls.
3. Affected callers/static/guards/inline review; safe notes/hashes/local commit.
4. Fresh exact-head scale/Keyword/original UI70; continue remaining proven
   allocation/latency work with all timing/native/external gates unchanged.

Task7 committed0f0c2f0fbb3e9926e301eb66a30b1ad6dcb79af2. Keyword300PASS
109.071834ms P95/6.047625ms loop; UI70 NON-GREEN seven activation intervals
60.780/72.915/64.716/56.483/59.832/92.408/83.312ms. All60search/eight exact
OPENED/twoprep/busy/source/corpus/physical resource fences retained. No unique
attribution or latency clearance from the mode-copy update guard.

Task8 implementation: native identity RED1/GREEN1; final native53PASS and
native geometry/paint/LRU/clear/replacement/subclass/error/GC controls. Verifier
source-retention RED fixed without touching global cache policy. Warm startup
1036/1033 FAIL is fixed by co-location at existing lazy package entry, public
exports retained, old adapter AST unchanged. Native+boot62PASS/two headroom
warnings: warm1033/1033/cold681/686, limits unchanged. Actual teardown caller
empty-injection failure reproduced with native BoxModel disabled and actual
REBUILD-held coalesced replay. Both preconditions observe four flags/exact owned
Worker within unchanged10s; identical control GREEN. FINAL120PASS119.50s,
required parent census/no privateSQLite, no pytestwarnings. Static/AST/inline
review and affected guards; safe QA/hashes only. Local commit/new original
qualification next; all parent/external gaps and previous failures retained.

### Task 9: Approved render-pass-only Console backdrop tints

ADR required: yes
ADR path: backlog/decisions/363-console-render-pass-backdrop-tints.md
Reason: explicitly approved native rendering/allocation compatibility boundary.

1. Actual modal/native allocation RED, then one app-owned backdrop subclass in
   the existing lazy switcher module. Preserve super render/context semantics.
2. Only one live render iterator owns a 128-entry immutable color-pair memo;
   native compositor, every cleanup/add LRU touch and all appearance stay.
3. Native parity/lifetime/fallback/overflow tests and original affected callers,
   startup/static/artifact guards, inline review and safe notes/hashes.
4. Local commit/fresh exact-source scale/Keyword/original UI70. Continue proven
   GC/non-GC work; no timing relaxation, other sessions or integration.

Task8 committed c79dc443. Fresh scale intact; Keyword300PASS107.959958ms P95/
6.634208ms loop. Original UI70 NON-GREEN four activation intervals64.9885/
84.863/57.530/61.436ms, all original outcomes/busy/source/resource fences intact.
Separate disposable 10ms phase-spacing control still fails six intervals; no
production scheduling change. Approved per-render tint control is allocation
evidence only, not a latency pass. Canonical ADR363 precedes implementation.

Task9 native allocation RED/GREEN, foreign-color cleanup and unsupported-method
import RED/GREEN; final20 native contracts. Original broader176 selection is
NON-GREEN171PASS/5pre-body config-source refusals plus85census observations of
four descriptors from the first real History reader. Original modal rendering
disabled control has zero renders, four native reads and1->2 registered handles;
body1PASS but process fails required descriptor gate. Do not waive or relabel it.
Final20native+five unchanged teardown cases in existing bootstrap mode25PASS/
32.88s/no pytestwarnings/no privateDB in all25censuses. Startup9PASS/two headroom
warnings, cold681/686/warm1033/1033. Static/design detector/eleven guards and
unchanged existing-modal AST pass. Safe QA/67hashes only. Local commit/fresh
exact-source qualification next; separately address finite History ownership
under existing ADR225 with native cold/warm/error/cancellation TDD, no foreign
close or broad service/lifetime policy. All parent/native/timing gaps remain.

### Task 10: Existing finite-owner boundary for History reads

ADR required: no new ADR
ADR path: backlog/decisions/225-app-scoped-sqlite-retirement.md
Reason: direct use of its approved operation-owned native reader contract.

Planned on TASK34661 before implementation: actual cold/warm transaction/error/
cancellation controls, one existing file-backed synchronous dispatch through
run_owned_db_call, original callers and native census, static/guards/inline
review/safe hashes/local commit/fresh exact-source qualification. No new global
owner, foreign close, production lifetime policy or deadline.

Task9 committed d64e5c9057. Original fresh UI70 still seven failures including
one preparation interval; Keyword300PASS111.760583ms P95/13.735958ms loop,
physical Chat0/noDB/source/corpus/busy fences retained. Task10 corrected native
RED4FAIL/2warmPASS precedes one-line fix; initial two warm proxy/native identity
setup errors retained. GREEN48PASS15.55s and final24PASS69.95s; required census
all48/all24 has no privateDB, six native cases repeated. New test static clean,
Workspace format clean/69 inherited Ruff/no additions; exact other-code AST and
eleven guards pass. Safe QA/38hashes; inline author review, not independent.
Commit/new qualification next, then continue GC/non-GC architecture attribution.
All original50ms/100ms and external/resource gates stay open; no other sessions.

### Remaining-latency attribution checkpoint; no new production policy

Task10 committed f53af7edbe822b5d93f9766f2e43abaf7baada5d. Fresh Keyword300
PASS109.877708ms P95/47.550375ms loop; original UI70 NON-GREEN seven activation
intervals114.386/51.387/52.377/62.856/78.620/57.655/58.972ms. All original
outcome/busy/source/corpus and actual post-Runner Chat0/no privateDB fences stay.
Not aggregateFD0, native whole-app qualification or task completion.

Row-cohort/class-sync hypotheses are not supported as the large measured cause.
Bounded callback/main-stack clocks separate actual automatic gen2 from no-gen2
native admission/SQL/poll/paint work. Separate storage observer sees53 startup
waits but none in activation; five activation failures remain. Minimal native
Styles.node owner cycle is proven without GC calls. Finalized-pair and obsolete
virtual-component controls still fail four/six intervals. No lifecycle policy
adopted; the latter is NOT in the pending finalized-widget design scope.

Two independent process-only controls avoid repeated backdrop version metadata
(132 requests/one native read; four failed intervals) and discarded transcript
global policy (34 avoided reads, native memory selector intact; three failures).
Their bounded version-guard and memory-only designs are awaiting approval, as is
the separate finalized-owner lifecycle request. Do not infer authorization from
these controls or silently bundle the obsolete virtual-node policy. Any approved
lifecycle boundary needs its canonical ADR before production implementation.

Safe QA: Docs/QA/task-34661/latency-attribution-2026-10-21.md and49 verified
hashes. All full raw dumps remain local. Author evidence review only; task states,
GC policy,50ms/100ms limits and native/Windows/quit/human/olderFD gaps unchanged.

Checkpoint: completed Task5 is committed as5f1c83a099e2a84399017d0843eaf6ea50fefbfe.
Fresh scale integrity/cleanup fences pass; Keyword300 PASS109.088875ms P95,
loop5.874166ms. Original70 remains NON-GREEN at three activation intervals
61.475/59.386/59.880ms. All outcomes/busy/source/corpus/actual post-Runner
Chat0/noDB fences remain. Passive same-head trace separates two main-gen2
intervals from one no-gen2 native-paint/MessagePump interval; not unique cause
of the original unrecorded gaps. No production BoxModel policy is approved.

### Authorized remaining-candidate comparison — 2026-11-11

The owner now authorizes investigating and trying all three presented candidates.
Keep existing historical approval/evidence statements above as history.

1. Under ADR363, snapshot the loaded Textual version in the existing lazy modal
   module. Retain explicit version override and native source/getter/method
   fallbacks. Verify native metadata-read RED/GREEN and live rendering parity.
2. Under ADR052, share the existing branch-valid selector through a memory-only
   controller read. Use it only for transcript projection; settings/send retain
   policy reads, validation, errors and native resource ownership. Verify real
   persisted memory and full refresh/cancellation/teardown consumers.
3. Test finalized Style retirement against late borrowing, copying and remount.
   ADR364 records its rejection after the native remount width control failed;
   retain native ownership and GC policy. No global style interception is adopted.
4. Freeze the checked source, rebuild an exact-head disposable scale corpus, and
   compare native controls (neither, each alone, both) with unchanged original
   UI70/Keyword300 and 50ms/100ms limits. Controls are diagnostic, not source-head
   qualification. Keep all failures, source hashes and resource observations.

ADR required: existing ADR052/229 govern the two bounded source changes; ADR364
records the rejected lifecycle alternative before any production ownership edit.
ADR paths: backlog/decisions/052-console-conversation-memory-and-compaction-policy.md;
backlog/decisions/363-console-render-pass-backdrop-tints.md;
backlog/decisions/364-finalized-style-owner-retirement-rejected.md.
No other sessions, shared environment changes, new timing limits or integration.
Raw evidence is local under /private/tmp/console-stability-20261111-zy2TL1.

### Shared synchronous control derivation — 2026-11-11

The three-candidate comparison is complete: both small corrections preserve
native behavior, but no original/diagnostic UI70 run clears all latency gates.
Style owner retirement and lazy restoration fail native late-access/remount or
copy ownership; no lifecycle change ships. A separate native parser-reuse
control reduces construction81 to17 and still fails six intervals; do not ship
its extra pooling/ownership complexity.

The existing credential refresh already scopes synchronous derivation. A
call-through control applies the same scope to the shared control projection:
19–27 config loads become one per pass. Full diagnostic latency remains red.

1. Mounted native RED for duplicate config reads and absent pass memo; preserve
   saved-setting freshness, exact UI error and next-refresh retirement.
2. Reuse `_console_derivation_scope` only around the shared synchronous control
   projection, with no awaits/new lifetime. Give the old synthetic test surface
   its existing real class default; preserve all native lifetime assertions.
3. Native regression/checked-lifetime/phase tests, inherited static and affected
   guards; local commit followed by fresh original exact-source qualification.

ADR required: no new ADR.
ADR path: N/A; ADR120/198 and existing TASK15452 derivation scope apply.
Reason: reuse the installed synchronous per-pass memo without new runtime,
ownership, GC, cache capacity, admission, appearance or timing policy.

### Exact-head outcome and further candidate disposition — 2026-11-11

The shared-scope change is committed as 9d1db30b01. Fresh original scale and
Keyword300 pass (119.258708ms P95 / 21.296375ms maximum loop); original UI70
still fails four activation intervals, maximum 77.431083ms. All exact targets,
busy paints, source/corpus fences, native Chat-handle zero and post-Runner
private DB retirement pass. All parent/external/aggregate gaps remain open.

Further process-only heap, LRU-clear, cycle-free-LRU, transcript batching and
obsolete-query-generation controls do not clear the latency gates. Successful
native exit does not make style-line caches disposable: the remount control
paints blank after their removal. ADR364 records the additional rejection.
No additional production policy or prototype is adopted. Retain full failures
locally and audited outcomes/hashes in the stability-candidates QA receipt.
This is evidence reconciliation only, not a task-completion or optimality claim.
