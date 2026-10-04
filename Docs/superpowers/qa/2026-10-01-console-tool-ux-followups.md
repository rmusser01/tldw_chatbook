# Console tool UX qualification — 2026-10-01

Scope: TASK-33625.2 approval layout, TASK-33621.16 confirmed Close and
TASK-32367 pending-kind projection. TASK-33625.1 Stop was already merged
in PR #2934; its 33 focused run-control checks pass unchanged.

## Review and implementation

Existing Select/Button instances reflow by card width and the existing
Console compact-height mode. Compact rows use the existing scroll region,
bounded to eight rows; bulk actions remain visible while Tab reveals the
optional reason input. Tall/wide layout retains its original order and
15-row row limit. Height-only resizing preserves choice, focus and identity.

Close names the tab, omits zero consequences, states pending cancellation,
defaults to Stay, and reports/reconfirms recoverable failures. Cleanup revokes and wakes
exact-session questions; hosts retain teardown. Inspector uses existing
kind-aware copy and a locked session approval-round count including queued
rounds. Questions and confirmations do not count as tool approvals.

Independent review found crowded Close copy could hide the title at 80×24.
Shorter consequence lines and removing a repeated question corrected the
27/29-row overflow; short and 60-character titles now fit. Final review
checked scrolling reason-input focus, height transitions and control reuse.
No unresolved local review finding. Qodo follow-ups add all five pending kinds
to Close copy, validate QA arguments before profile mutation, reconcile exact
provisional fleet fences, and route review to the visible decision before a
queued approval. A separate failed-provisional marker blocks unsafe retries
while preserving surviving-child usage in the retained session.

## Targeted receipts

- Approval/card/denial-reason/budget/token: **52 passed**;
  /tmp/console-tool-ux-approval-qualified-final.txt.
- Final seven compact cases plus UI latency/responsiveness: **25 passed**;
  /tmp/console-tool-ux-height-final.txt.
- Close/impact: **17 passed**; affected routing: **3 passed**;
  final real-kind Close and maximum-risk geometry: **6 passed**.
- Pending-kind projection: **18 passed**, including two real full-app cases
  with worker rounds, queued count 2→1, precedence, sibling isolation,
  navigation and fresh remount.
- Stop: **33 passed**; /tmp/console-tool-ux-stop.txt.
- Derived-artifact preflight passes; generated CSS and UI census122 verified.
  Both new suites are added to that existing gate.
- New Python files pass Ruff/format. Modified legacy files add zero diagnostics;
  [comparison](2026-10-01-console-approval-layout/lint-delta.json).

RED receipts retain actual control clipping, question Inspector mismatch,
orphan-question cleanup and dialog overflow. Three inherited approval-test
assumptions were repaired: extracted CSS ownership, token width and real
private-profile/round ownership. Native refreshed second-tab retries exposed
scope wrapping and short-height chrome pushing Submit off-screen; existing
scrolling/height mode fixed them. Failed receipts remain in /tmp.

The CSS formatting follow-up restores unrelated multiline rules and full
original rationale comments from dev. Consistent two-space property indentation
in the owning stylesheet pays the unchanged startup budget; all six new compact
rules remain readable.
Comment/whitespace-normalized selectors and declarations match the preceding
stylesheet exactly: /tmp/console-tool-ux-css-formatting-proof.json. Generated
CSS was rebuilt.

## Review follow-up receipts

- Real mounted Close file: **19 passed**. Final provisional retry/refusal and
  surviving-usage correction: **2 passed**; close/usage boundaries: **21 passed**.
- All three keyboard/Inspector/tab review routes with a real skill confirmation
  and queued approval: **3 mounted cases passed**.
- Native Close runner malformed CLI: **7 passed** without traceback or profile writes.
- Independent review repeated full-finalize usage 120→165 and same-generation
  retry refusal; no remaining actionable finding.
- Diagnostic inventory reviewed against latest dev: one new rollback warning
  records only `type(exc).__name__`, with no exception text, user content, paths
  or secrets and no new sink. Its existing owner pin was regenerated after
  reviewing /tmp/console-tool-ux-diagnostic-review.txt.
- Preceding combined Console, Close, controller-attribute and latency invocation:
  **44 passed**, one expected empty-exemption parameter set skipped;
  /tmp/console-tool-ux-final-dev-qualified.txt. Fresh final preflight passes:
  /tmp/console-tool-ux-final-dev-preflight-qualified.txt.
- Latest-dev combined token/startup-budget checks: **9 passed**. Declaration-neutral
  whitespace in the owning sheet preserves the unchanged 608,090-byte cap;
  final census is **608,077 bytes**. Dev's model-switcher and both new UI suites
  remain in the 122-file census.

Receipts: /tmp/console-tool-ux-close/qodo-{close,usage,usage-boundary,args}-green.log,
/tmp/console-tool-ux-labels/qodo-focus-final.log,
/tmp/console-tool-ux-latest-dev-tokens.txt. No full-suite sweep.

## Preceding healthy-run integration

The clean rebase onto dev84247cb843 includes the healthy-run/provider-readiness
fix from PR #2948. Shared pending-kind, focus and layout behavior is retained.
Compact approval layout, startup ratchets and design-token governance pass:
**26 tests**, /tmp/console-tool-ux-dev842-layout-qualified.txt. Fresh preflight
passes, /tmp/console-tool-ux-dev842-preflight.txt; Ruff comparison against this
base remains identical with zero new diagnostics in 16 modified Python files.
All six mounted pending-projection and latest-dev readiness cases pass, including
the healthy-run, held-regenerate and missing-key control. Exact-node private
child receipts are /tmp/console-tool-ux-combined-rfotoln6/run-000..005/pytest.log;
aggregate /tmp/console-tool-ux-labels/combined-readiness.log. Each child uses
the designated temporary profile and disables pytest caching.
The broader focused status/composer/rail run has passing evidence for all
130 cases. One composer case first exhausted its inherited five-second stream
startup wait before behavior assertions; a fresh private-profile rerun passed
unchanged. Both receipts are retained; no timeout was widened:
/tmp/console-tool-ux-labels/combined-readiness-receipt.json and
/tmp/console-tool-ux-labels/combined-readiness-retry-receipt.json.

Native approval run08 and Close run9 requalify the changed Console screen;
source hashes match the combined tree and the real profile stays unchanged.

## CI timeout and recovery follow-up

UI Fast Lane job 110655142114 on head 70eadf9119 exceeded its unchanged
20-minute limit at 78%, with no earlier assertion failure. Exact job logs
are retained at /tmp/console-tool-ux-head70-ui-fast-lane.log. The fix reduces
repeated private-child interpreter/app startup without removing scenarios,
changing the census, weakening assertions or modifying CI/timeouts.

- Compact geometry: all five scenarios share one real screen, with explicit
  resize, rail preference, distinct round and focus reset. Reflow and height-only
  cases remain separate. **3 tests pass**, 135.68s;
  /tmp/console-tool-ux-ci-compact-reused-qualified.txt.
- Close: five pending kinds, two title lengths and two rollback outcomes stay
  in three isolated children, each with fresh app teardown. **All nine scenarios
  pass**, 212.512s wall versus 341.27s for the preceding nine-child run;
  /tmp/console-tool-ux-close/timing-consolidation/after-green.json.
- The retained failed-provisional marker now raises one authored recovery
  refusal. Confirmed Close reports it and ends the flow; a manual new close
  still requires its initial consent. Exact generation/fleet fence and surviving
  child usage6 remain unchanged. RED before the fix, then mounted GREEN:
  /tmp/console-tool-ux-close/timing-consolidation/retry-red.log and after-green.log.
- CSS formatting proof preserves selectors/declarations and restores unrelated
  rules. **19 startup-budget/token checks pass**,
  /tmp/console-tool-ux-qodo-css-qualified.txt; fresh preflight passes,
  /tmp/console-tool-ux-qodo-preflight.txt. Ruff adds zero diagnostics against
  immutable dev84247cb843 in 16 modified Python files. These 19 checks cover
  ratchet policy and tokens; the actual byte-cap check is recorded below.
- Independent source review confirms scenario/reset preservation, safe authored
  copy, cancellation propagation and retained usage/fence semantics.

The exact-head UI Fast Lane must still pass within its existing 20-minute cap.

## Preceding combined-tree qualification

Rebased cleanly onto dev27e718f01d81, including the compaction failure-copy
and v74 lineage/recovery fix. Range-diff confirms all four patches unchanged;
independent inspection found no upstream changes to pending, focus or Close
ownership. **7 targeted combined checks pass**: three mounted projections,
two compaction copy/recovery cases and two fresh/v73-upgrade schema cases.
Nine exercised Python/test source hashes remained stable:
/tmp/console-tool-ux-labels/combined-compaction-receipt.json.

The actual CSS-byte guard first failed at 608,749 versus the unchanged 608,090
cap after multiline restoration; shortening source comments did not help because
the builder strips them. The final two-space property indentation preserves all
364 ordered rules, including descendant-selector spaces, and restores the full
original rationale. **9 actual byte-cap/token checks pass**, with five parsed
sources totaling **607,057 bytes** (1,033 headroom):
/tmp/console-tool-ux-dev27e-css-final-qualified.txt. The failing receipt remains
/tmp/console-tool-ux-dev27e-css-budget-qualified.txt. Formatting passes for all
six modified test/native helper files; Ruff introduces zero diagnostics across
16 modified Python files against immutable dev27e718f01d81.

Final preflight passes: /tmp/console-tool-ux-final-preflight.txt. Current
native source-hash qualification is recorded below.
The preceding rebase onto dev922440b93e83 adds only ADR-210 documentation.
All five patches remain identical by range-diff, and production/native hashes
remain valid. ADR-210 keeps the approval card as the decision surface and
requires these TASK-33625 safety repairs before its separate region migration.
The exact-head GitHub gates and fresh Qodo resolution remain merge prerequisites.

## Preceding warm-config integration

Clean rebase onto devab4df9995954 includes PR #2903's guarded warm settings
and runtime-snapshot paths. All six patches remain unchanged by range-diff.
The config miss, forced reload, file replacement and symlink paths remain guarded;
no Console decision ownership, layout or execution interface changed upstream.
**16 targeted checks pass, zero skips**: three mounted pending projections and
thirteen warm-config safety cases, each using a private profile before imports.
Eight exercised sources remained stable and current; the real profile was unchanged.

Fresh native approval11 repeats all nine Ask-gated fs_read journeys at 80×24
Inspect open/closed and 235×52. Fresh Close12 repeats four owning-worker closes
and the explicitly synthetic five-kind, six-loss, 60-character-title geometry.
Both app/process exits are0; no network attempts. UI and new config-source hashes
match the combined tree. Real config and data-file mtimes remain unchanged
(533 files for approval; 515 for Close). Three approval raster exports and five
Close exports were inspected with the recorded Cairo font-fallback limitation.
Committed images below remain evidence from preceding runs with unchanged UI
source hashes; the new replay exports remain in temporary directories.

[Sanitized integration receipt](2026-10-01-console-tool-ux-config-integration.json)
records the verified tree, source pins, individual check results and fixture limits.
Temporary full receipts are /tmp/console-tool-ux-labels/combined-warm-config-receipt.json,
/private/tmp/console-tool-ux-approval-native-11/evidence/result.json and
/tmp/console-tool-ux-close-native/run12/qualification.json.

Fresh combined-tree preflight passes, including generated CSS, pinned Mermaid
assets, diagnostic inventory and the unchanged122-file UI census:
/tmp/console-tool-ux-devab4-preflight.txt. Ruff adds zero diagnostics in16 modified
Python files against immutable devab4df9995954:
/tmp/console-tool-ux-devab4-lint-qualified.json. Task closeout also removes a
duplicate unchecked criterion and preserves explicit approval denial/cancellation
while distinguishing recoverable Close failures from the restart refusal.

Published preceding head166e89a359 passed PR Fast Lane, UI Fast Lane, UI latency
and derived artifacts. UI Fast Lane completed921 tests in17m52s within its unchanged
20-minute budget. Those receipts are historical: all four gates and fresh Qodo
review must pass on the new published head before normal merge.

## Final short-height review qualification

Fresh Qodo review of heada663415450 identified missing fixture docs and a real
short-height Close overflow. The mounted RED case measured22 rows at80x18.
The canonical confirmation body now uses native VerticalScroll capped by the
existing viewport token, with the action row docked separately. Explicit cancel
autofocus preserves Stay. The selector matches both the new scroll body and
existing specialized Container bodies, retaining the original60-cell frame.
Independent review caught the selector mismatch before final qualification;
the incident is recorded in lessons-testing-evidence.md.

**27 targeted checks pass**: one mounted case covering both titles and
80x24→80x18→160x44 resize,17 literal/safe-dismissal/nested-callback contracts
and9 actual byte-cap/token checks. Keyboard Shift+Tab/End reaches the final
worktree consequence with both buttons painted and hit-testable; Home/Tab
restores Stay. Same button identities survive resize. Added fixture docs cover
all new public tests; executable AST comparisons show documentation-only changes.
Ruff introduces zero diagnostics in17 modified Python files against devab4df9995954.
Fresh preflight passes; census122 and the608,090-byte cap remain unchanged,
with607,219 boot-parsed CSS bytes. No full sweep or CI-setting change.

Native approval12 repeats all9 actual Ask-gated fs_read journeys with16 current
source/config/CSS/helper pins. Native Close13 repeats all4 actual owning-worker
closes and the synthetic maximum-risk geometry at80x24 and80x18; native scroll
moves0→6, paints the last consequence, and returns focus to Stay. Both actual
app/process exits are0; owned sockets released; no network or real-profile change
(approval533 files, Close515). [Updated Close receipts](2026-10-01-console-close-followup/README.md)
and the [integration receipt](2026-10-01-console-tool-ux-config-integration.json)
record source pins and limitations. Approval12 full exports remain temporary;
the earlier committed approval images below are historical. Fresh published-head
GitHub gates and resolved Qodo review remain mandatory before normal merge.

## Native evidence

[Approval result](2026-10-01-console-approval-layout/native/result.json):
**9/9** actual TldwCli/LinuxDriver/TTY journeys at 80×24 Inspect open/closed
and 235×52 Inspect open. At each size, fast Deny and Deny all + Submit return
authoritative refusal; Approve once reads the disposable file through the
production local executor. Deny all leaves its worker pending until Submit.
All five actions have complete painted labels and native hit/clip checks.
App/process exit0; no exception or network attempt; source/runner hashes recorded.

The real controller-composed LocalToolProvider and private permission store
gate fs_read as Ask. A disposable workspace is supplied explicitly at provider
composition; no accepted model turn or external MCP server is claimed.
Session projection is synchronized and its pending count asserted before capture.
[80×24 denial](2026-10-01-console-approval-layout/native/80x24-inspect-True-deny-all-decision.svg),
[80×24 Inspect closed](2026-10-01-console-approval-layout/native/80x24-inspect-False-approve-once-pending.svg),
[235×52 wide](2026-10-01-console-approval-layout/native/235x52-inspect-True-fast-deny-pending.svg).
Matching terminal text is retained. Approval choices use Pilot key events in the live native app; Close uses tmux SGR mouse input. Raster previews were inspected locally; Cairo font fallback is not pixel-perfect terminal evidence. Attempts01–10 remain under
/tmp/console-tool-ux-approval-native-*; run10 requalified all nine journeys
after the recovery refusal, dev27e718f01d81 rebase and declaration-neutral
CSS budget/formatting fix. The card/controller sources remain unchanged; the shared bundle was later
refreshed for the short-height Close fix. For run10, the
real config and all 533 tracked data-file mtimes remain unchanged.

[Close report](2026-10-01-console-close-followup/README.md): four real decision
worker closes, owning task cancellation, sibling isolation and explicitly
synthetic maximum-risk geometry. No provider/server execution is claimed
for Close. Final run13 requalifies the four real closes, all-five-kind geometry
and native80x18 keyboard scrolling after the shared confirmation repair.
Original60-cell frame and same actions survive resize; source pins match current
runtime/CSS/config/helper files. Exact target fleet/wake fences release; its
isolation receipt verifies the real profile was unchanged.

## Inherited optional-governance failures

Two broader component-pattern checks fail on original dev83c2c9810d:
dimension_literal_ratchet has14 unchanged diagnostics in agentic/settings/
workflows sheets; python_style_ratchet flags an unchanged Library skill-pane
width assignment. The [comparison](2026-10-01-console-approval-layout/baseline-governance.json)
proves identical matched declarations and byte-identical Python source.
This patch introduces neither violation. Checks were not removed, relaxed
or suppressed; affected token/budget checks pass. No full-suite sweep.

ADR required: no. Existing ADR-043/150/161 layout, ADR-067/094 lifetime and
ADR-195 live-tool boundaries apply; task plans link them. Final PR checks
and merge verification are tracked in the PR.


## Latest-dev logging integration — 2026-10-02

Rebased onto dev92a95170a540 (PR2904 logging changes). All68 unaffected
feature blobs, including every Python/CSS feature file, stayed byte-identical.
The overlapping diagnostic inventory merged automatically and passed fresh
preflight; the append-only lesson conflict retained both entries. No feature
behavior, architecture boundary, dependency or CI setting changed.

Fresh combined-tree verification passed19 targeted cases with no skips:
nine Console layout/projection/Close cases, eight logging contracts and two
worker-event cases. The two original unwrapped worker probes first hit the
documented profile-rebind admission before their worker assertions; fresh
per-case reruns used the existing `bootstrap_profile` opt-in, preserving the
original assertions and config guards. The
[receipt](2026-10-01-console-tool-ux-config-integration.json) keeps both the
initial probe logs and passing reruns. The prior17-file Ruff result remains
applicable because all modified Python feature bytes are unchanged.

Approval native run13 passed all nine journeys, including three actual local
reads and six denials, with22 current source pins. Close native run14 passed
four real worker closes and the crowded80x18 keyboard/resize journey, with17
current pins; its existing exports and [report](2026-10-01-console-close-followup/README.md)
are refreshed. Both apps exited0, attempted no network, released their exact
owned processes/sockets, and left real config/data unchanged. This qualifies
combined app behavior, not logging performance. Earlier receipts remain
historical; final published-head CI/Qodo and a verified normal merge remain
required.

## Final retained-ID and cancellation review corrections

Qodo on `6f82aca15073` reproduced two issues. The cancellation subclass still
styled only a direct Container after the shared body became VerticalScroll;
its existing primary-border selector now also matches `.confirmation-scroll`.
All four computed edges, literal prose, safe default and Enter/Escape dismissal
are covered by the production-CSS regression.

Retained committed native IDs already remain fenced by the runtime and the
controller's late-usage ownership. Close now uses the existing authored restart
refusal at both boundaries, before voice ownership. The actual mounted Close
flow ends after one confirmation and reports the named tab. Same-ID create and
restore probes retain the exact generation and tombstones; both immediate and
already-queued old drains leave restored usage/source unchanged. Normal saved
resume and new sessions continue to allocate fresh UUIDs. Recycling a closed
native ID within the same runtime is not added by this correction.

The final private-basetemp/XML run passes10 Close/lifecycle/privacy cases,
including the unrelated-error fixture's original type-only redaction and actual
successful retry. Eleven affected confirmation cases and9 design-token/actual
boot-byte checks also pass: **30 distinct targeted checks, zero failures/errors/
skips**. The two actual private-profile UI children pass. Preflight passes with
census122; boot CSS remains607,219 bytes under the unchanged608,090 cap. Ruff
adds zero diagnostics across21 modified Python files against dev92a; an
inherited F811 source-line reference is normalized for comparison. Independent
production/test review is clear.

Fresh native approval14 passes nine Ask-gated fs_read journeys (three reads,
six denials), with24 current/stable source pins. Native Close15 passes four real
decision-worker closes, target cleanup and sibling isolation, plus60-cell
80×24→80×18 crowded geometry and native Shift+Tab/End/Home/Tab scrolling, with19
current/stable pins. Both apps/processes/socket owners exit0, attempt no network
and leave real profiles unchanged (533/515 data file mtimes respectively).
The first Close15 profile was rejected by the existing path validator before
app imports; contained fixture paths were corrected before the valid launch.
The existing native exports remain historical; current receipts and source pins
are in `review_followups_2026_10_02` of the sanitized integration receipt.

These checks qualify the pinned working source bytes on dev92a, before the
review-fix commit. Final published-head GitHub gates and fresh clean/resolved
Qodo remain the separate normal-merge checkpoint. No full local sweep, new
incarnation machinery, cleared authority, dependency or CI-limit change.

## Latest dev queue, trace and GC integration

Clean rebase onto `ee1c1e7365c2` preserves all10 feature/review patches with no
conflict. Dev adds queue recovery and app-owned manager actions (PR2943),
parked trace-maintenance wake scheduling (PR2914), boot/pre-import heap freezing
(PR2913/ADR-198) and an ADR-126 documentation amendment (PR2911).

**47 plain targeted checks pass**, zero skips:30 review/lifecycle/style checks,
six current approval/projection cases, two GC contracts including actual
`_ui_ready` heap freezing, four trace wake/interval cases and five joined queue
ownership/drain checks. Two supplemental manager removal/clear and app-owned
Resume probes also pass with the existing `bootstrap_profile` opt-in in separate
private processes. Their initial `raw_source_selection_changed` setup errors are
retained; repository test bodies, assertions and markers remain unchanged.
Final plain GitHub UI Fast Lane is still required.

Fresh preflight passes, census122, and current boot CSS measures607,171 bytes
under the unchanged608,090 cap. Ruff adds zero diagnostics in21 modified Python
files against this dev. Read-only combined-tree review found no actionable
queue/pending-kind/modal/Close/trace-disposal collision.

Native approval15 passes all nine actual Ask-gated read/deny journeys with30
current/stable pins. Close16 passes the four actual worker closes and80×18
60-cell keyboard-scroll matrix with28 current/stable pins. Both apps/processes/
sockets exit0, attempt no network and leave real config/data unchanged
(533/515 file mtimes). The approval receipt's loose-object lookup failed after
Git packed the commit; read-only Git recovered the exact unchanged identity,
separate from the successful native journey. Its failure log is retained.
Current source hashes and scope are in `latest_dev_queue_gc_integration` of the
sanitized receipt; historical native exports remain identified as such.

### Final authored-refusal review correction

Qodo found that an accepted Temporary turn's authored refusal was shown while
a replacement Close dialog immediately covered the tab. The existing mounted
regression now requires the close worker's claim to end and pending work to
remain accessible; it fails before the one-line allowlist correction. All12
affected plain cases pass, including transient explicit retry, retained fences
and private-error redaction. Native Close18 qualifies the corrected source;
approval16 is unchanged. The storage-admission base adds47 passed checks and
one intentional unbound-profile skip. Exact receipts, pins and limits are in
`actionable_refusal_review_followup` in the combined JSON. Subsequent dev
rebases require fresh published-commit CI/review before PR2953 merges.


### Final CI settled-idle census correction

On dev eba4305d83, the captured trace scheduler left its fresh migration pending.
The exact plain guard failed; a call-through diagnostic counted3 admissions for
its first completion and2 for each of seven later checks (17/8, over ceiling2).
The test now asserts one real completion in uncounted setup before all eight
measured idle batches. Production, real seams, canaries and limits are unchanged.
The plain guard and four cold-completion/read-only/parking/wake contracts pass,
with no failures, setup errors or skips. The guard measured1.75 admissions and
0.75 helpers per tick; owned connection reuse can lower those counts. Existing
22-file Ruff baseline has zero new diagnostics; the census file retains its
pre-existing import/format diagnostics. Independent setup review is clear.
Current production remains qualified by30 plain cases and native approval17/
Close19 on this base. All four published-head CI gates and fresh resolved Qodo
remain required before normal merge. See settled_idle_census_followup in JSON.


Final chat-confirmation follow-up (2026-10-02): source head
`261887810a60cf091d5e9fe2b40111d0ff5b50b2`, dev
`fccf70d3b0cd21b0d44a906ec3a68b0633887684`. Ninety focused Console/consent/
Close/quit-neighbor checks pass without skips, followed by the ordinary storage
census with its private real startup cleanup completed before counting.
Native approval22 passes nine actual Ask-gated reads/denials; Close25 passes
six actual decision-worker closes (including chat creation with no owning turn)
and six-kind maximum-risk geometry at 80x24 resized to 80x18. All current
production pins, process exit0, no-egress and real-profile invariants verified.
Crowded geometry uses a synthetic impact; wider hooks/plugins/MCP behavior
retains only the stated targeted checks and source review. The corrected Close
runner removes an unsupported absence assertion: a pending decision counts as
an occupied lifecycle slot even without a parent stream task. These fixtures
and task notes are the only later changes; PR2953 requires fresh published-head
CI and clean resolved Qodo before normal merge.


Confirmed-create/Close race follow-up (2026-10-02, dev `f80d3e0090`):
Qodo finding `ece44de5-bd8f-4ef5-97a1-77cebdc2e262` reproduced a remembered
grant resurrecting after actual Close and both new-chat/fork-chat executors
creating durable rows while a committed Close retained its source. Final
verdict/grant and resolver writes now share the existing standalone lock; the
shared executor refuses the exact committed generation before DB/UI work.
Failed-provisional markers remain outside this guard.

Sixty-four unique affected cases pass:47 consent/execution cases,16 Close/UI
neighbors and the ordinary exact storage census. Three real race regressions
fail on the published source and pass on final bytes. Native approval23 passes
nine actual Ask-gated reads/denials; Close26 passes six actual worker closes,
sibling isolation and six-kind80x24→80x18 geometry. Their original byte pins
are preserved; only a later whitespace-only return-dict format differs, with
whole-controller AST equality proven separately. The47 consent cases reran
after stabilizing the protected-worker revision snapshot; the16 neighbors do
not import that test module. Apps/processes/socket cleanup exit0, no egress,
and real profiles unchanged. Root viewed fresh80x24 approval/Inspect/chat-create
and80x18 scrolled Close captures. Native does not force the executor/grant
interleaving or exercise broader hooks/plugins/providers/MCP.

Final-byte preflight passes(census123);25 changed Python files add zero Ruff
diagnostics; changed ranges format cleanly; independent source review is clear.
The earlier90-case combined hooks/plugins/quit verification is historical
preceding-candidate evidence. Current receipts and explicit equivalence limits:
`/tmp/console-tool-ux-chat-create-race-final-manifest.json`,
`/tmp/console-tool-ux-approval-run23-sanitized/isolation-qualified.json`, and
`/tmp/console-tool-ux-close-native/run26/qualification.json`.
Fresh final-head Qodo/all four GitHub gates and a verified normal merge remain
required. No full local sweep or CI/protection bypass.


In-flight creation follow-up (2026-10-02, Qodo59151d44 and helper-doc rulee7e76a0):
qualified code/test head `1bf350c729859069e093672c3dcf8d45b1ef41c9`, combined dev
`bb865f5cfeae4c9d8c068f588ee0d85ff28a0e13`. Entry-only refusal reproduced late
new-chat/fork-chat creation; queued UI dispatch refusal also stranded a live
row. The shared executor now qualifies the source after worker I/O and on the
synchronous UI thread, resolves the current view sink, and uses existing
worker-side best-effort orphan soft-delete on refusal. A local admission flag
preserves already-placed chats and the original UI exception if the source
retires before exception delivery. No lock or transaction spans the UI hop.
Four creation/handoff race cases, two refused-dispatch cases and one admitted
exception case fail on preceding candidates; all nine race/view cases pass in
the final combined run. Raw deleted rows remain; this is not physical rollback.

78 unique checks pass without failures/errors/skips:56 full consent/execution,
16 Close/UI/runtime neighbors,5 actual Resend integration neighbors and the
ordinary exact storage census. All85 affected-source pins stay unchanged.
Derived-artifact preflight passes with UI census124;25 changed Python files
add zero Ruff diagnostics; changed ranges format cleanly; independent source
and upstream-integration reviews are clear. Rebase retained all20 patches;
only census conflict resolutions preserve all upstream/PR paths and floor124.

Native approval24 passes9 Ask-gated local reads/denials at80x24 and235x52;
Close27 passes6 real approval/question/standalone-chat-create worker closes
and six-kind80x24→80x18 geometry. Apps/PTYs exit0, owned sockets/observer process
are gone, no egress, real config/data unchanged, and exercised source pins match
this final combined code. Root inspected current compact approval/Inspect,
chat-create Close and scrolled short Close captures. AX is unqualified because
Accessibility permission was unavailable; the completed owned Terminal viewer
window may remain. No TCC changes or broad window cleanup. The replay does not
force the durable worker race or exercise Resend/provider/hooks/plugins/MCP;
the focused real-DB and mounted integration checks cover this patch's races
and Resend neighbors. No full local suite. Earlier replays remain historical.

Current receipts: `/tmp/console-tool-ux-inflight-create-final-manifest.json`,
`/tmp/console-tool-ux-approval-run24-sanitized/isolation-qualified.json`,
`/tmp/console-tool-ux-close-native/run27/qualification.json`.
Fresh final-published-head Qodo and all four GitHub gates, latest-dev integration
and verified normal merge remain required.


## Current readiness-base and final CI corrections

Clean rebase onto dev `185c845fe836bf452e4beaaf8853162ce49b1e8d` preserves
all22 patches, including the upstream model-readiness integration. The existing
pure Resend eligibility helpers moved into the resident message-action owner;
execution imports only on explicit Resend, with compatibility exports and
executable helper AST preserved. Fresh projection and real warm-start guards
fail before and pass after; module ceiling and snapshot remain1033 unchanged.

Qodo496a9bd8 now retains only primary Close error class and raising function
before the existing authored recovery refusal. Successful/refused/raised rollback
regressions preserve exact fences, generation, session/grants and retry refusal,
with no exception content/capture, private IDs/paths or new sink. The reviewed
one-record diagnostic inventory delta preserves all upstream entries.

**230 scoped cases pass**, without errors/skips:193 behavior/grouped/cache
cases and37 exact three-group Perf Guard cases, including both new readiness
storage-census variants. The inherited isolated normalization test first counted
3669 versus9:3660 belonged to cold static support-set initialization. One real
no-evidence build outside profiling makes both comparisons equivalent; positive
baseline, exact equality and first shared-owner lookup remain measured. Five
comparison/keyed-evidence/cache contracts pass. Production readiness is unchanged.

Four projection journeys and two no-owning-turn Close kinds now use two bounded
private children, retaining fresh apps/controllers/stores/workers and all original
body assertions/cleanup. AST proof permits only two loop-local default bindings.
One paired local sample improves91.55 to64.84 seconds; the final bound Close
rerun passes in23.55 seconds. This is not a CI deadline guarantee; the180-second
child and20-minute workflow limits, census, counter seams, canaries and ceilings
remain unchanged. The preceding derived failure was the unsuccessful UI dependency,
not an artifact reproduction error.

Approval26 passes nine Ask-gated reads/denials; Close28 passes six actual worker
closes plus six-kind80x24→80x18 crowded geometry. All111/183 source pins stay
current;65/94 naturally loaded origins, including15 Close readiness origins,
are verified. Apps/PTYs/sockets exit0, real profiles unchanged, zero egress. Root
viewed fresh compact approval/Inspect, chat-create Close and scrolled80x18 frames.
The initial Close private-path prelaunch rejection is retained; only that external
fixture was corrected. AX remains unqualified and no Terminal/TCC/viewer was opened.
Native excludes readiness requests, Resend execution and forced race/rollback paths;
focused real-DB/runtime/UI checks cover the selected changes. Earlier broad action
failures and their baseline-comparison limits remain recorded separately.

Current combined preflight passes (UI census124); all34 changed Python files
introduce zero Ruff diagnostics and edited ranges format cleanly. Independent
source/isolation/privacy review is clear. Exact receipts, byte pins and scope:
`readiness_base_final_followup` in the combined integration JSON and
`/tmp/console-tool-ux-readiness-integrated-final-performance.json`. Fresh published-head
Qodo/all four GitHub gates, latest dev and verified normal merge remain pending.


Final overlap integration: dev advanced to `ef8fd5d38a512be299af17b1e0d5b367a352a5d6`
with PR2962's deferred Resend imports. The23-patch rebase resolves only equivalent
import placement/monkeypatch targets, preserving the exact qualified production
and test bytes, including the stronger execution-free broken-row projection.
All225 qualification pins remain identical. The final tree differs from the
preceding230-case/native/preflight candidate only in upstream lesson and task
metadata. Original receipt identities remain unchanged; explicit rebase equivalence
is in `overlapping_resend_rebase`. Published-head CI/Qodo remain required.


## Remaining UI Fast Lane startup cost

Published45189cd3 reached97% without an assertion failure, then its unchanged
20-minute UI job expired; all14 Close nodes had passed in273.95s. The derived
check fails explicitly on that cancelled required dependency, while its
artifact steps pass. Exact logs are retained in /tmp.

Nine lightweight Close navigation/recovery bodies now run in two bounded
private children instead of nine. Every original body/signature and all63
assertions remain identical; the five heavier cases stay separate. Fresh
apps/controllers/stores/workers, distinct real database paths, per-scenario
patch contexts, original cleanup and factory drains/unfreeze/GC remain.
Ordinary private wrappers pass all nine journeys: one paired macOS sample
95.55s ->28.42+38.44=66.86s. This is not a hosted CI completion guarantee.
All180s child/20m job deadlines, gates and census floor124 are unchanged.
Independent isolation review clear; edited-range format and zero-new Ruff34
pass. Receipts: /tmp/console-tool-ux-close-grouping-{before,after}.json and
/tmp/console-tool-ux-close-grouping-final-proof.json. Latest dev advanced to
ecc0a531c8 (PERF-07 config/path memo); qualification of that combined tree
and fresh final-head CI/Qodo remain required before merge.


## Final PERF-07 base qualification

Code anchor `17235d9d87` on dev `ecc0a531c8` incorporates PERF-07 unchanged:
all eight upstream files match dev and all 25 PR patches rebase unchanged.
Of 225 preceding source pins, 222 remain byte-identical. Changed config and
config_participants are upstream; Close tests have the qualified grouping.
The original 230-case/native receipts retain their original identities.

Fresh combined-tree checks: **65 pass**, zero failures, errors or skips:
all seven Close children (including the nine grouped navigation/recovery
journeys), the four-journey projection child, three compact approval cases,
12 upstream path-memo contracts, five readiness/cache comparisons, and all
37 cases from the exact three Perf Guard groups. Warm census remains
1033/1033. Real counter seams, canaries, ticks, ceilings, snapshots and
deadlines remain. Current preflight passes with census 124; zero-new Ruff
across 34 files, edited-range format and diff checks pass. Receipts:
/tmp/console-tool-ux-perf07-integrated-{affected,performance}.json and
/tmp/console-tool-ux-perf07-preflight.txt.

Native approval27 passes all nine real Ask-gated read/deny journeys:
three executed reads and six denials, with 113 pins and 69 natural origins.
Close29 passes six real approval/question/standalone-chat-create worker
closes with sibling isolation and target fence release. It also passes
six-kind long-title geometry at 80x24 and 80x18, including default Stay,
action hit tests and native keyboard scrolling: 186 pins and 97 origins.
Both apps, PTYs and sockets exit and clean up normally with code 0. Zero
egress and the real 21 config entries/533 data-file mtimes are unchanged.
Root independently verifies all current pins/origins and views fresh compact
approval, chat-create Close and scrolled short-height captures. Existing
helper bytes remain unchanged.

The native Close app completed successfully before an external shell
here-document capacity error during qualification. Its saved results were
intact; 3.7 GiB later became available without pruning or a rerun. The separate
environmental receipt is retained. Native AX, broader requests and forced
races remain outside scope; naturally loaded remote-worker code is import
evidence only. No full local suite. QA `path_memo_base_final_followup` records
exact identity, receipts and limits. Final published-head CI/Qodo and
protected normal merge remain required.


Final Qodo outcome-kind correction and release-base integration (2026-10-02)

Qodo247e7df5 now uses one private `_CHAT_CREATE_SESSION_GONE` constant in
all three executor refusal returns. The full controller AST matches the
preceding published c814 head after substituting that constant's literal and
removing its sole assignment; creation, admission and cleanup are unchanged.
Existing chat-create integration/race and real warm-census checks: **46 pass**,
zero failures/errors/skips. No additional test or runtime abstraction.

Rebased onto dev f3aeb32fb3d230c0774c7fc349729d9f75c96366 (release0.2.3).
All26 patches retain their code;25 are range-diff unchanged and one only shifts
an appended lesson's context/blank line, preserving both entries. Of230 prior
source pins,227 are unchanged. The three differences are the private outcome
constant, upstream pyproject version and upstream Resend-absent census guard.
Upstream package-version and unrelated evaluation changes remain upstream-owned.
Zero-new Ruff34, constant-range format, diff check and fresh full preflight pass.

Prior native27/29 and65/230-case receipts retain their original identities.
The new controller has semantic equivalence, not byte identity; release version
and census changes are identified separately. Native was not replayed for this
literal extraction. Current qualification and limits are recorded under
`outcome_kind_release_followup` in the combined QA JSON and
/tmp/console-tool-ux-outcome-qualified.json. Final published-head gates and fresh
resolved Qodo remain required before protected normal merge in PR2953.


Latest-dev known trace-work integration (PR2959/TASK33801, 2026-10-03 UTC)

Dev e6ab66b0a9f950a56de80fb6b1bfa7dfbee3df7d skips an extra idle-check
admission on known-work maintenance passes. All27 PR patches rebase unchanged.
The changed runtime worker method and maintenance class match upstream ASTs;
all remaining module ASTs and the controller remain unchanged from1695e988.
The two upstream regression files retain exact upstream bytes.

**147 fresh scoped checks pass**, zero failures/errors/skips:110 full trace
migration/parking, runtime-shutdown and chat-create integration contracts,
followed serially by all37 exact Perf Guard cases (22 boot,13 latency/stall,
and both ordinary storage variants). Real write-during-pass and signal-wake
SQLite workers, idle read-only completion, fences, canaries, ticks and existing
ceilings remain verified. Preflight and zero-new Ruff34 pass.

Of230 preceding pins,228 remain unchanged; the two differences are upstream
trace owners. Their two upstream regression files are additionally pinned,
for232 current source pins. Code anchor:2942fc3a8a315c12289f9dd153fe2f5a249afe45.
Original native27/29 and46/65/230-case receipts keep prior head/byte identities. No current native replay is claimed for the inherited worker optimization. Current runtime/maintenance owners are qualified by real SQLite worker/write regressions, lifetime contracts and mounted performance tests. No new production patch, budget, snapshot, CI setting, authority or full-suite sweep; fresh published-head gates/Qodo still required.
QA trace_work_base_final_followup records receipts and explicit scope.


Qodo fixture-documentation correction (c3c3fc77, 2026-10-03 UTC)

Three existing Close button tests now have Google-style summaries and Args
entries for every fixture, including request and the saved-history tmp_path.
The entire executable module AST, decorators and signatures are identical
after removing only those three new strings. Compile and private collection
of all three requested nodes pass; an audit finds no remaining missing request
docs among modified private-profile tests. Zero-new Ruff34, doc-range format,
diff check and fresh full artifact preflight pass.

All232 trace-work qualification pins remain byte-identical; the documented
test file is additionally pinned (233 current). The147-case proof and original
native/46/65/230 receipts keep their identities. Collection is not claimed as
functional replay, and no runtime/native/performance rerun was made solely
for docstrings. QA fixture_doc_review_followup records exact scope and hashes.
Fresh final-published-head CI/Qodo still precede protected normal merge.


Audit-only latest-dev rebase (PR2925, 2026-10-03 UTC)

Dev2612fc56b26510630190ecacdc3855c4bffb1786 changes only seven audit
paths. All29 Console patches rebase unchanged; all233 qualified source pins
retain exact bytes. Fresh preflight and zero-new Ruff across34 modified
Python files pass. No application, test, CI, boundary or budget change.
Original147-case and native receipts retain their code/head identities;
no functional replay is claimed for the audit-only base. QA
`audit_only_dev_rebase` records exact paths, hashes and range-diff proof.

The preceding c88 UI job hit its unchanged20-minute limit at88%, with no
assertion failure reported. Comparable groups took25–34% longer than on
the preceding successful c814 run. Limits and coverage remain unchanged;
fresh final-head CI and clean/resolved Qodo are required before normal merge.


Qodo public Close contract correction (d97b6290, 2026-10-03 UTC)

`ConsoleRuntime.close_session` now documents its existing arguments, result
and RuntimeError refusals, including a retained session admission fence. The
entire runtime executable AST is unchanged after restoring only the prior
docstring; compile and range formatting pass. Of233 preceding pins,232
retain exact bytes and only this docstring changes the runtime file hash.
Fresh full preflight and zero-new Ruff34 pass. Original147-case and native
receipts retain identities; no functional/native replay is claimed for strings.
QA `close_contract_doc_review_followup` records exact byte/semantic proof.
Fresh final-head CI/Qodo and current dev still precede normal protected merge.


Remaining private-startup reduction and paint synchronization (2026-10-03 UTC)

C88 and998d UI jobs hit the unchanged20-minute limit at88%/96%, with no
reported assertion failure; both Close files cost about251s. Eight remaining
separate pending/race/fleet/geometry and compact-approval scenarios now run in
two ordinary private wrappers. All eight complete bodies and145 assertions
remain; the older nine Close navigation/recovery scenarios retain their two
existing wrappers. Each scenario retains fresh owners, scoped patches, distinct
fleet DB paths and existing factory drain/unfreeze/GC cleanup. No production,
CI, timeout, budget, counter or canary change.

The initial timing baseline records7 passes/1 geometry failure in234.08s and
is not a successful paired timing baseline. End had started scrolling before
the final consequence was painted. The sole scenario-body correction awaits
Textual's existing scheduled-animation completion before unchanged pixel
assertions; the original geometry node then passes both titles in28.84s.
Both complete grouped files pass all17 existing journeys in4 private wrappers,
zero failures/errors/skips, in225.76s; slowest wrapper102.851s under the unchanged
180s cap. These samples cannot guarantee the hosted20-minute job completes.

Independent read-only AST/isolation review has no actionable findings. All34
preexisting function signatures/bodies match apart from the one animation wait.
Of233 current preceding pins,232 retain bytes, the Close test changes and the
approval test is additionally pinned (234 total). Other production/helper source pins and
original147-case/native receipts keep identities; no replay is claimed beyond
these affected files. Fresh preflight, zero-new Ruff34, range format and diff
checks pass. QA remaining_private_startup_followup records exact hashes,
real red/green receipts, review and limitations. Fresh final-head gates/Qodo
and current dev remain required before protected normal merge.


Latest-dev Roleplay quit integration (PR2963, 2026-10-03 UTC)

All32 Console patches rebase unchanged onto dev2d34cbf8. Eight upstream files
retain exact dev bytes; the combined diagnostic inventory additionally retains
the existing Console-controller diagnostics and passes the actual rebuild check.
Of234 preceding pins,233 remain byte-identical; the sole inherited change is
the expanded quit choke-point test. The Roleplay screen, lazy guard and its
upstream regression file are additionally pinned (237 current).

All42 fresh targeted cases pass with zero failures/errors/skips: five expanded
quit architecture cases and all37 exact Perf Guard cases, including both ordinary
storage variants. Fresh full artifact preflight and zero-new Ruff34 pass.
Code anchor:c0c459d5654c59481a88e685593210efb059d02d.
Original grouped17-journey,147-case and native27/29 receipts retain their original head/byte identities. All grouped Console test source pins and production/runtime pins retain bytes across this rebase. The inherited Roleplay quit behavior is not claimed as a new native or mounted Roleplay replay here; its upstream files are exact and the expanded shared quit architecture/37 mounted and startup guards pass. No budget, snapshot, counter, canary, tick, deadline, authority or global CI changes; no full sweep. Final published-head CI/Qodo and current dev remain required.
QA roleplay_quit_base_final_followup records the exact source manifest and limits.


Qodo chat-create synchronization deadline naming (4ae8306e, 2026-10-03 UTC)

The existing five-second event/worker deadline is now named by one private
constant in each of the two Chat test modules. All16 related waits/joins use
the same value. Both complete modules pass56 cases, zero failures/errors/skips.
Range format, zero-new Ruff34 and full artifact preflight pass.
Only two test-module source pins change; all235 other source pins, production/runtime, grouped UI fixtures and helper owners retain bytes. Whole affected module executable ASTs are identical after replacing the private names with5 and removing their assignments. Current56-case receipts qualify real workers/SQLite in both complete modules; original42/147/native/grouped17 receipts keep recorded identities with this mechanical equivalence proof. Deadlines remain5s/180s/20m; no full suite, new abstraction/dependency or CI/budget change. Fresh published-head review/all four checks/latest dev required before normal merge.
QA chat_create_timeout_review_followup records current237pins, receipts and limits.


Qodo late question admission after Close (d9c05b6a, 2026-10-03 UTC)

Close publishes its committed generation under the existing shared interrupt
lock. Question admission checks that same persistent fence in its existing
check/register section. An earlier round is swept; a delayed request returns
cancelled without a retained card or indefinite wait. Sibling questions remain
answerable. Callbacks and cancellation stay outside the non-reentrant lock.
Existing ADR-094/067 apply; no new owner, authority, persistence or service boundary.

The initial unmarked regression raised raw_source_selection_changed and is
invalid race evidence. With the existing bootstrap_profile ownership marker,
the real host/store complete-Close test reproduces exactly two late-worker waits
(before registration and already closed); the already-registered case passes.
All three pass after the fix and again on final formatted bytes with only their
checked-in marker, without a temporary plugin. This regression uses the existing
persisted-store helper and fake UI dispatcher; it is not itself real-SQLite or
mounted-runtime-drain qualification.

The first194-neighbor run passes186 and has eight setup failures. Restoring only
the two reviewed controller methods in an otherwise identical combined tree
reproduces all eight exact failures; it is not a pure-dev comparison. All194
neighbors then pass with explicit existing collection-profile ownership for the
question, interrupt-attention and durable-postcommit modules, without getter
mocking or admission bypass. The separate final ordinary three-case regression
also passes. Preserve this conditional fixture mode when interpreting the full run.

All37 exact Perf Guard cases pass with unchanged counters, canaries, ticks,
ceilings and snapshots. Later question-test formatting preserves the entire
module AST and does not change controller/runtime/guard bytes. Both complete
mounted approval/Close files pass4 ordinary private wrappers covering17 existing
journeys, zero failures/errors/skips, in258.01s; the slowest wrapper is115.408s
under the unchanged180s cap. Fresh full preflight, changed-range format and
zero-new Ruff across35 changed Python files pass. Independent source review has
no actionable findings. Existing question-test AST is unchanged after removing
only the added regression and its private synchronization constant. Only two
controller methods change; all236 other preceding pins retain bytes, with238
current source pins. Original native27/29 receipts retain their historical
identities; native was not replayed for this race correction. No full sweep.
Fresh published-head Qodo/all four gates/current dev still precede normal merge.
QA question_close_admission_review_followup records all source and fixture
identities, valid/invalid red receipts, exact baseline comparison and limitations.


Latest-dev cloud-fixture rebase (PR2927, 2026-10-03 UTC)

All35 Console patches rebase unchanged onto devacc45cdc. Its38 changed paths
are cloud capture/replay fixtures, four LLM test modules and one task; every
inherited path matches dev exactly, including the removed legacy capture helper.
All238 qualified source pins retain exact bytes. No production, shared conftest,
dependency, CI, budget or qualified-source change. Fresh full artifact preflight
and zero-new Ruff35 pass. Earlier question194+ordinary3, exact37 Perf, mounted
17-journey and native receipts retain their recorded code/head/fixture identities;
no functional/native replay is claimed for this test-only rebase. QA
cloud_fixture_dev_rebase records paths, exact range-diff and source proof.
Fresh final-head CI/Qodo/current dev still precede normal protected merge.


Latest-dev Roleplay design-docs rebase (PR2960, 2026-10-03 UTC)

All36 Console patches rebase unchanged onto dev1d566b9d. Its31 changed paths
are Markdown Roleplay design/review/task files; every inherited path matches
dev exactly.
All238 qualified source pins retain exact bytes. No production, shared conftest,
dependency, CI, budget or qualified-source change. Fresh full artifact preflight
and zero-new Ruff35 pass. Earlier question194+ordinary3, exact37 Perf, mounted
17-journey and native receipts retain their recorded code/head/fixture identities;
no functional/native replay is claimed for this documentation-only rebase. QA
roleplay_design_docs_dev_rebase records paths, exact range-diff and source proof.
Fresh final-head CI/Qodo/current dev still precede normal protected merge.


Chat-create UI dispatch currentness (PR2953 review investigation, 2026-10-03 UTC)

Qodo 6f41bacd proposed a register-after-sweep race. Six call-through interleavings
on the unchanged reviewed 54c controller, plus all 56 existing Chat cases, pass 62
checks: admission and the Close sweep share the standalone registry lock. Close
publishes its permanent generation before sweeping. Registration first is swept;
Close first is rejected at admission. The initial probe called Close from a
foreign thread and raised the prompt-queue owner violation, so it is invalid
race evidence. The corrected probe runs Close on its real owner and observes
the original lock/check/insert/sweep. Qodo accepted the rebuttal: its canonical
overview updated at 05:33:14 UTC reports 0 bugs/rules/cross-repo conflicts on 54c.

Separate ordinary mounted-source review reproduced a valid presentation bug on
the unchanged published controller: a worker pauses after registration before
UI dispatch, navigation or complete Close places a live sibling confirmation,
then the obsolete source payload replaces its actual pending state/card. Close
still denies and ends the source worker; this is a presentation race, not an
admission/wait/permission race. Initial NoMatches from querying before the card
mounted is retained as invalid setup evidence. Valid navigation and Close red
receipts verify the sibling request identity after the actual callback completes.

The existing marshal now resolves the current UI sink and qualifies the live
round, committed Close and scoped active session on the UI owner. External
setters remain outside locks. Delayed None clears rederive the active parked
head instead of erasing it. Restoring only the unconditional clear branch in
an otherwise corrected in-process controller reproduces a missing sibling state;
this is an isolated branch comparison, not pure dev or the whole original
controller. An earlier malformed temporary patch is not functional evidence.
The final regression explicitly checks that the survivor payload exists.

A naive active-session check also drops the only unparked legacy card after
navigation; its Close control passes. A trusted internal session_scoped bit
preserves legacy session_id=None initial projection and keeps Close denial.
This is the only justified deviation from the initial marshal-only plan; it
adds no public payload/API, owner, permission, persistence or lock. Existing
ADR-094/150 apply; no new ADR.

Final formatted-source qualification passes 64 Chat cases: all 58 checked-in
consent/execution cases plus the six unchanged call-through probes added at
collection, with existing explicit bootstrap-profile ownership. No getter
mocking or admission bypass. A separate ordinary 58-case consent receipt is
retained. Both complete Close/approval files and the complete projection file
pass 5 ordinary private wrappers covering 23 fresh-app journeys in 369.62s, with
zero failures/errors/skips. The slowest parent wrapper is 124.114s and child
109.938s under unchanged 180s limits. Original journeys/assertions and the public
wrapper census remain intact; new Close/navigation cases use separate fresh
apps and DBs, release both marshal gates and join every worker. All 37 exact
Perf Guards pass with unchanged budgets, counters, canaries, snapshots, ticks
and limits. Zero-new Ruff across 35 files, changed-range formatting, diff checks and
independent read-only source review pass.

Only two controller methods and two test modules change versus 54c; all 235
other source pins retain exact bytes (238 current). Original controller/test
AST outside the explicit added behavior/cases is preserved. Native 27/29
receipts retain their original identities; native was not replayed for this
callback correction. The incident lesson and Console guide record current
source/teardown behavior. QA chat_create_ui_projection_review_followup pins
valid/invalid comparisons, fixture modes, exact sources and fresh run receipts.
Fresh full artifact preflight passes (exit 0, all 238 source pins stable).
Published-head Qodo/all four CI gates/current dev and verified protected
normal merge remain the completion checkpoints.


Latest-dev queue-button integration (PR2964, 2026-10-03 UTC)

All 38 Console patches rebase unchanged onto dev 74bd039d. Its queue-region
runtime and two generated-style pins change; all 235 other preceding source
pins retain exact bytes, including the reviewed callback fix and its tests.
The inherited queue regression module is additionally pinned (239 current).
Six inherited paths match dev exactly; the shared incident lesson retains both
independent sections through exact patch replay. No own runtime, fixture,
style, authority, CI, budget or timeout change. Existing ADR-094/150 apply;
no new ADR is required.

The combined mounted run passes all 23 Console projection/Close/approval
journeys in five ordinary private wrappers, plus all 17 inherited queue-button
paint/composer-width cases: 22 top-level cases, zero failures/errors/skips,
in 419.22 seconds. Public wrapper-file census is retained; no temporary plugin
or source replacement. Every original journey body/assertion remains intact.
The slowest ordinary wrapper remains below its unchanged 180-second cap.
Zero-new Ruff across 35 changed Python files and diff checks pass.

The preceding 64-case Chat/probe receipt retains its original source/head
identity; its controller, Chat tests and worker/store dependencies remain exact.
This rebase changes UI-region/generated-style bytes and therefore has separate
mounted and performance qualification. Original native receipts remain
historical; no fresh native replay is claimed. QA queue_button_dev_integration
records exact source/provenance and current coupled receipts. All 37 exact
Perf Guards pass on this combination, with unchanged limits and 239 stable
source pins. Fresh full artifact preflight passes (exit0; all239 source pins
stable), with the unchanged124-file UI census. Published-head review/all four
CI gates, latest dev and verified normal merge remain required. TASK-32367
remains In Progress until hosted checks pass.


Legacy tool-approval count review follow-up (Qodo62959751, 2026-10-03 UTC)

A real request_mcp_approvals(session_id=None) leaves a live host round and
answerable mounted tool card but intentionally no kind/badge entry. Its UI
count returned zero. Valid RED reproduces that count in the mounted view and
one pure legacy control; five finishing/positive/global-metadata controls pass.
The initial GREEN test also asserted unrelated legacy activity copy; that new
fixture assertion was corrected to actual Inspector and Files count surfaces,
without changing production activity projection. Partial selection is retained
as regression evidence only; ordinary full-wrapper qualification follows.

Change only ChatScreen._console_pending_approval_count: preserve a positive
active-session registry count; on zero, count only the typed pending tool card,
excluding finishing state. Unrelated app/global counts remain unreachable on
this real-registry path. Older-controller fallback and all other screen AST,
original UI helpers/assertions and private-wrapper census are unchanged. Of 239
source pins, 237 retain bytes; only screen and projection tests change. Existing
ADR-067/094/195 apply, no new owner/interface/authority or ADR.

Complete projection/Close/compact-approval files pass all 24 fresh-app journeys
in five ordinary private wrappers plus six boundary controls (11 top-level
cases; zero failures/errors/skips; 352.17s). No selection
plugin; maximum child report 102.668s under the unchanged 180s cap.
All 99 host/consent cases and all 37 exact Perf Guards pass.
No budgets, snapshots, counters, ticks or limits change. Fresh full artifact
preflight, zero-new Ruff across 35 files and range-format/diff checks pass. Independent review
has no remaining actionable findings. The mixed legacy/registered compatibility
rule preserves the registered count rather than claiming an additive deduplicated
total; AC8 explicitly bounds that case. QA legacy_approval_count_review_followup
records exact bytes, receipts, fixture modes and earlier evidence limits.
No fresh native/queue replay or full local-suite claim. Task remains In Progress
until published-head review and all four hosted gates pass before normal merge.


Public boundary-test typing (Qodo324cc873, 2026-10-03 UTC)

Add only int, dict[str, object] | None and int annotations plus -> None to the
new public six-case count test. All six cases re-pass. Normalizing only those
annotations reproduces the entire reviewed19 module AST; every test body,
helper and private wrapper is unchanged, as are all238 other source pins and
all production bytes. Range format, zero-new Ruff35, diff/backlog checks and
unchanged124-file UI census pass. No new ADR is required for test-only typing.
QA legacy_approval_test_typing_followup records the exact new test pin and
proof. Complete mounted24/host99/Perf37/preflight receipts remain at their
original19 source/fixture identities; no fresh broad/native replay is claimed.
Task remains In Progress pending fresh final-head review/all four hosted gates.


Latest runtime-dev integration and hosted UI limit (2026-10-03 UTC)

GitHub's UI Fast Lane annotation on reviewed 8a confirms the 20-minute job
limit. No pytest failure was reported before cancellation at 90%. Every
derived-artifact reproduction step passed; its required gate failed solely
because UI Fast Lane did not finish. No current-head job was manually cancelled,
and no CI budget, timeout, gate or protection is changed.

Rebase all 41 patches unchanged onto dev 3c439d606e. All 18 inherited production
paths match dev exactly, all owned production bytes retain reviewed identity.
Thirteen preceding source pins change through dev, 226 stay exact; 10 upstream
source/snapshot dependencies join the pins (249 current). Inherited work includes
hook visit/preimport paydown, durable recovery and legacy flat deletion, provider
copy/support and the Voice-step extraction. Existing ADR-067/094/097/150/195
continue to govern our unchanged implementation; no new ADR is required.

Fresh combined-tree checks pass: 24 fresh-app journeys in the same five ordinary
private wrappers, six count boundary controls and 13 inherited hook UI cases
(24 top-level cases, zero failures/errors/skips; 307.34 seconds). Maximum wrapper 86.446 seconds,
child 76.041 seconds; the 180-second cap and every original journey remain unchanged.
All 170 host/consent/relaunch/Resend cases and all 37 exact Perf Guards pass. The
stricter upstream hook/preimport ratchets remain intact. Full artifact preflight
passes in 105.96 seconds with all 249 source pins stable; zero-new Ruff across 35 files and diff checks
pass. No additional production fix was needed.

Additional question/shutdown/hook qualification preserves both modes: the
unqualified 130-case run has 121 passes and nine question setup failures at
raw_source_selection_changed, before the awaited question round can arm. The
previously documented temporary collection plugin supplies the existing real
bootstrap_profile owner to the question module. All 130 pass under that owner;
no getter/admission bypass, production patch, assertion change or timeout raise.
This is conditional fixture evidence, not a claim that the unmarked command
is green. Original native 27/29 and queue 17 receipts retain their historical
identities; none is replayed for this rebase.

QA oct03_runtime_dev_integration records exact pins, patch equivalence, gate
cause, both fixture modes and fresh receipts. TASK-32367 stays In Progress
until exact published-head review and all four hosted gates pass; then final
metadata-head checks/review and verified protected normal merge remain required.


Provider-validation latest-dev integration (2026-10-03 UTC)

The prepublication guard caught dev advancing from 3c439d606e to e355b58c1e.
Rebase all 42 patches unchanged. The four inherited provider-error files match
dev exactly and retain identical import ASTs; only one preceding pin changes.
Three inherited production files and the new regression join the source pins
(253 total). No own implementation, test, fixture, CI, timeout or budget change;
no new ADR is required for this patch-equivalent integration.

The full targeted gateway/validation/Resend command has 447 passes and 17
failures. An exact latest-dev source-only control runs the same 464 cases in
the same order and produces the same 17 failing nodes with identical messages:
existing provider assertions and mixed-profile setup. The 16 new validation
and coupled Resend UI cases pass. No broader-green claim or suppression of
those 17 failures. The temporary control changes no branch or checkout.

All 37 exact Perf Guards, fresh full artifact preflight, zero-new Ruff across
35 modified Python files and diff/backlog/UI-census checks pass on this final
combination. All 253 source pins remain stable. Previous full mounted 24,
host/consent/relaunch 170 and conditionally qualified 130 receipts retain
their original 3c source/fixture identities; no fresh full replay of them,
native or queue work is claimed for the provider-error delta.

QA oct03_provider_validation_dev_integration pins current sources, passing
subsets, exact baseline failures and fresh guard receipts. TASK-32367 remains
In Progress until published-head Qodo and all four hosted gates pass, followed
by final task-metadata-head checks/review and verified protected normal merge.


Mechanical final-review follow-up (2026-10-03 UTC)

Qodo d40a53fd and 038a1411 are addressed by one shared
CONSOLE_PENDING_CHAT_CREATE_KIND in the already-imported models module and
fixture/stage/return annotations on the new question/Close test. The scalar
remains exactly chat_create. All four full module ASTs equal reviewed 87e53655
after only substituting that scalar/removing the added aliases and annotations;
249 other pins remain exact (253 current). No runtime behavior or module import
edge changes, and existing ADR-067/094/150/195 continue to apply.

An initial formatter call misread line:column as a start/end range and changed
unrelated source formatting. The preflight correctly rejected the altered
controller diagnostic digest. Remove the unrelated churn; retain the inventory
unchanged, rather than regenerating it. Preserve that failed preflight as
discarded-candidate evidence. Correct explicit start:column-end:column range
checks pass for every actual edit.

The minimal candidate passes all 102 question/interrupt/chat-create contracts,
eight top-level projection/Close cases (six boundary controls and both complete
ordinary private wrappers), and 13 import/module/LOC guards with zero
failures/errors/skips. No collection plugin or getter bypass is added; the
question test keeps its existing bootstrap_profile owner. The only later source
change sorts its new test-only Callable import; all bodies/assertions and every
production byte remain exact. All three question/Close stages re-pass on final
test bytes. Maximum ordinary wrapper 66.045s remains below the unchanged 180s cap.
Full artifact preflight passes in 135.00s; all 253 pins
remain stable. Zero-new Ruff across 35 changed Python files and
diff/backlog checks pass. The earlier full UI/host/37-Perf/native/queue receipts
retain their original identities; no fresh full or native replay is claimed.

QA oct03_mechanical_review_followup records current pins once, narrow full-AST
proof, exact receipts, fixture modes, limits and the discarded formatter attempt.
Keep TASK-32367 In Progress until latest-dev publication, clean resolved Qodo and
all four hosted gates, followed by final metadata-head review/checks and verified
protected normal merge.


Remaining public chat-test typing (2026-10-03 UTC)

Qodo46dc223b reports two untyped new integration tests. Audit all new public
module-level tests introduced by this PR and annotate both reported tests plus
four matching new Close/view/legacy tests in the same two modules. Existing
controller/DB/pytest types and one test-only Callable import suffice. The audit
now finds no newly introduced untyped public test. After removing only these
new annotations/import, both full module ASTs equal published 5c6262a8; every
body, assertion, marker, fixture, helper, stage and deadline stays exact. All
251 other source pins, including all production, remain unchanged (253 current).
No new ADR is required for test-only annotations; existing contracts apply.

Both complete real SQLite/worker modules pass all58 cases with zero
failures/errors/skips in 33.13s, under the existing
profile owners and without an added collection plugin or admission bypass.
Explicit changed-range format, zero-new Ruff35, diff/Backlog checks and full
artifact preflight (135.46s) pass with all253 pins
stable. QA oct03_public_chat_test_typing retains exact proof and receipts; prior
mounted/host/37-Perf/native/queue evidence retains original source/fixture
identity, without fresh broad/native replay claims. Keep TASK32367 In Progress
until latest-dev publication, exact-head clean resolved Qodo/all four hosted
gates, final metadata-head review/checks and verified protected normal merge.


Compact approval class naming (2026-10-03 UTC)

Qodo7d8b4330 requests one name for the four approval-compact references.
_COMPACT_APPROVAL_CLASS joins the existing fast-control class constants; its
value remains exactly approval-compact. Full widget module AST equals published
d346275a after substituting only that scalar and removing its declaration.
All imports, stylesheet/token/test bytes and252 other source pins remain exact
(253 current). No layout, focus, permission, deadline or visual value changes.
Existing ADR043/150/161 apply; no new ADR is required.

Both complete approval-card/compact Console modules, design-token governance
and CSS byte budget pass42 top-level cases with zero failures/errors/skips
(77.78s), preserving ordinary private wrappers
and their caps. Fresh artifact preflight passes in
87.36s with all253 pins stable. All five changed
ranges format cleanly and zero-new Ruff35/Backlog/diff checks pass.
QA oct03_compact_approval_class_review preserves exact proof/receipts. Prior
broad/native/37-Perf evidence retains its original identities; no fresh replay
of it is claimed. TASK33625.2 is reopened In Progress for final review
qualification, alongside TASK32367. Both need exact published-head clean/resolved
Qodo/all four hosted gates, followed by final Done-metadata-head review/checks
and verified protected normal merge.


Provider rate-limit dev integration (2026-10-03 UTC)

Rebase all46 own patches unchanged onto upstream PR2981 at d294c4871ff5b7585ad746bd6991c0934f61c639.
Provider capture/parser/gateway/Hugging Face streaming and tooltip code are
inherited unchanged; ChatScreen adds exactly its upstream spend wire line.
Six inherited production files equal dev;250 previous pins stay exact, three
inherited pins change and six are added, yielding259 current pins. No new ADR
is required to consume this existing boundary; ADR094/097/195 and inherited
provider contracts apply. No owner, authority, permission, fixture, CSS/token,
assertion, budget or CI-setting changes are made.

All24 top-level mounted checks pass (280.45s), including all five complete
ordinary private wrappers, six count controls and13 hook cases. The maximum
ordinary wrapper 78.849s and child 72.376s remain below180s.
The full targeted provider/validation/Resend/telemetry command has470 passes
and17 failures; an exact latest-dev source-only control reproduces the same
487 cases and17 failure nodes/messages. All23 new telemetry cases pass.

The unmarked mounted cost-tooltip test fails raw_source_selection_changed on
both candidate and exact dev. Under only the existing real bootstrap_profile
owner, its full mounted send/tooltip case passes in
10.35s. Record both modes: the temporary collection plugin adds
that marker to exactly this node; no getter/admission replacement or egress
occurs. Do not claim the ordinary upstream fixture is green.

All37 exact performance guards (22 startup,13 latency/responsiveness, two
storage ratchets) pass. Full artifact preflight passes in 97.35s
with all259 source pins stable. Zero-new Ruff across35 owned Python files,
diff and Backlog checks pass. Previous native/broad receipts retain their
original identities; no fresh native replay or full local suite is claimed.
QA oct03_rate_limit_dev_integration records exact proof, receipts and limits.
Keep TASK32367 and TASK33625.2 In Progress pending latest-dev publication,
clean/resolved exact-head Qodo and all four hosted gates, then fresh final
Done-metadata-head review/checks and verified protected normal merge.


Pending approval kind / changed public test signatures (2026-10-03 UTC)

Qodo b4d98154 asks that the three related pending-round approval defaults
share one kind. CONSOLE_PENDING_APPROVAL_KIND names the identical approval
scalar in the existing shared models; registration fallback, shared copy and
Close consequences reuse it. Qodo65fea6a1/d934f324/0dd121c7 identify five
changed existing test signatures. Add their existing controller/pytest types
and test-only pathlib.Path. Audit all new AND changed public test signatures
across the PR: none now omit input/return annotations. No new ADR is required;
existing ADR043/067/094/150/161/195 apply.

All six full module ASTs equal published835e9bbe after only this exact scalar
substitution/new declaration-import removal and the five new annotations/
test-only Path import removal. All other production imports/statements and
every test body/assertion/fixture/marker/deadline stay exact. All253 other
source pins are unchanged (259 total). Explicit format checks identify only
three signature/tuple wraps; apply those exact wraps without unrelated churn.
All16 final changed ranges format cleanly and zero-new Ruff35 passes.

The full six-module targeted command has110 passes/18 failures
(67.75s). All six cases from the five changed signatures pass in ordinary
mode, including the four production-shaped private wrappers. The18 failures
are unchanged unwrapped routing cases refused at config binding with
raw_source_selection_changed. An exact preceding-reviewed-head source-only
control reproduces the same128 cases and18 failure nodes/messages; this is
published835e9bbe, not pristine dev. Apply only the existing real
bootstrap_profile owner to those18 nodes through a temporary collection
plugin: all18 pass (72.23s). Retain both modes; no getter/admission replacement,
assertion/permission/cap/budget changes, and no whole-command-green claim.

All13 targeted import/module/LOC guards pass. Full artifact preflight passes
in 133.76s with all259 source pins stable. Backlog and diff checks pass.
Prior full37-Perf/mounted/native receipts retain their exact original identities;
no fresh broad/native/full-suite replay is claimed. QA oct03_pending_kind_and_changed_test_types
preserves the proof, both fixture modes and their controls. Keep TASK32367 and
TASK33625.2 In Progress pending latest-dev safe publication, fixed-thread
resolution, fresh exact-head Qodo/all four hosted gates, then final
Done-metadata-head review/checks and verified protected normal merge.


Latest-dev guard / Close first-paint qualification (2026-10-03 UTC)

Rebase onto dev 9b28ce1479efed6bca687cfbb33d261322f83abe preserves all48 Console patches. Three
inherited fixture pins change and two guard pins are added, all exact dev;
all256 other prior pins and production remain exact (261 total). The earlier
four maintenance fixes retain their exact scalar/type AST proof. No new ADR:
consume the inherited test guard and correct only test input readiness.

Guard tests55 pass/2 macOS xattr skips. Full128 affected contracts110 pass/
18 configuration-binding failures. A unique reviewed835 Console source-only
snapshot plus exactly current9b fixtures/guard reproduces these18 nodes and
messages; only those18 pass with their existing real bootstrap_profile owner.
This is a controlled preceding-Console/current-fixtures snapshot, not pristine
dev or a whole-command-green claim. No guard off switch, real-home override,
config getter/admission replacement, assertion change or budget relaxation.

All37 exact Perf Guards pass:22 startup,13 latency/responsiveness,2 storage.
The full24 mounted command has21 passes/3 Close failures (622.79s).
A queryable ConfirmationDialog entered the stack before its first paint:
two clicks see an empty compositor/NoWidget at0,0, and compact geometry sees
width0 despite Stay focus. Retain all RED messages and the21 passing cases.
Fix only shared _wait_for_confirmation to require non-empty Confirm/Stay
regions with the actual buttons hit-testable at their centres. Existing200
polls/0.01s, real clicks, every geometry/focus/worker assertion and180s caps
stay unchanged. Restoring that helper and its one test-only NoWidget import
makes the entire module AST exact; all260 other source pins stay unchanged.
The concrete readiness incident is recorded in lessons-testing-evidence.

All three complete Close groups now pass (206.8s), including all original scenarios and both geometry title variants.
The complete21 routing cases pass (198.62s) with existing bootstrap-profile
ownership only on18 unchanged legacy cases; the three typed private wrappers
keep ordinary ownership. The longest fresh wrapper/child takes
111.834s/98.992s, below unchanged180s.
Passing evidence covers all24 original mounted nodes across explicitly
identified pre-/post-helper sources; no fresh whole24/37 replay is claimed.
Production/guard/fixture/performance bytes remain exact; earlier provider,
SQLite and native receipts retain original identities, no native/full suite.

Fresh final-helper artifact preflight passes (172.02s), zero-new Ruff35
and4 exact helper format ranges pass, all261 final pins stable. Earlier16
maintenance-format signatures/scalar fragments remain exact. QA oct03_real_profile_guard_dev_integration
and oct03_close_first_paint_readiness retain source identity, failures, controls and final receipts.
The preceding835 hosted latency job passed timing but failed a startup
trace-callout mount; the fresh full22 local startup run passes unchanged.
That establishes neither cause nor fix for the older hosted failure; final
hosted latency success remains required, no speculative production/CI change.
The Close task is reopened alongside the pending/compact tasks until fresh
exact-head clean/resolved Qodo/all four hosted gates, then final task metadata
review/checks and verified protected normal merge. Auto-merge remains off.


Latest-dev storage/GC ratchets (2026-10-03 UTC)

Rebase onto dev a5600366381f19d99dbd0dada655113316ede9f7 consumes PR2969's platform open
ceilings, per-run census log and first eligible GC pass, including collection.
47 preceding patches are exact; two differ only in surrounding context:
the additive upstream lesson before our existing containment section, and
upstream burst-count lines after our real media-cleanup pass. Removing only
the upstream lesson restores the whole prior lesson document. Removing only
the same four unchanged owned settlement blocks makes the full census module
exact dev, both before and after rebase. The workflow is exact dev, no own
CI change. All260 other prior source pins, production/UI/guard/fixtures and
paint-readiness/test scenarios stay exact; add workflow pin for262 total.

Both exact hosted storage-ratchet variants plus platform/log-path controls
pass4 cases (86.13s), normal pytest-equivalent path depth, no pin/cap change.
Each first GC pass records0 config admissions/9 storage admissions/3 helpers/
54 opens, matching the canonical macOS pins. The census retains upstream
held backup-probe, completed collection/owned-call/drain assertions and Linux
ceilings. Fresh artifact preflight passes (162.75s), zero-new Ruff35/
Backlog/diff checks pass with262 stable pins; changed public-test signature
audit remains clean. No new ADR: existing ADR097/125/126 apply to inherited
measurement, production contracts unchanged. QA oct03_gc_census_dev_integration records proof/counts.

Earlier22 startup/13 latency and all24 mounted-node passing evidence retain
their explicit original source identities; production/fixtures/test scenarios
are exact across this rebase. The earlier RED3 and corrected Close/routing
passes remain separately recorded, no fresh whole24/37/native/full-suite
claim. All three selected tasks remain In Progress until exact clean/resolved
Qodo/all four hosted gates, final Done metadata-head review/checks and verified
normal protected merge. Auto-merge stays off.


Backlog-only dev rebase (2026-10-03 UTC)

Dev af138397408d596a1c72c1e8e9987ea95d497b94 adds only unimplemented TASK-34117, a separate trace-GC maintenance follow-up. All50 preceding PR patches are exact; whole old/new tree delta is only that inherited task. All262 qualified source pins, every production/test/fixture/guard/CI/derived-artifact source and existing passing evidence retain exact identities. Fresh Backlog ID/readability checks pass on the only changed input. No repeat full preflight/performance/mounted/native/full-suite claim. QA oct03_gc_followup_task_dev_rebase records complete range/tree proof. No new ADR: documentation-only integration. All three selected tasks stay In Progress; fresh exact-head clean/resolved Qodo and all four hosted gates, final Done metadata-head review/checks and protected normal merge remain required. Auto-merge off.


Final review: approval ownership and pre-composition resize (2026-10-03 UTC)

The13-case RED records11 passes and two failures: a stale other-session
approval counted as1, and a real uncomposed ChatApprovalCard raised NoMatches
after setting its compact class. Approval payloads already carry session_id;
use that owner only in the zero-registry fallback, preserving true unscoped,
matching, finishing and positive registry controls. Reflow now resolves all
existing controls before changing the compact class, returning only on the
existing NoMatches so a later normal resize can retry. No new owner or state
field, permission/decision path, token, import edge, budget or cap change.

Widget-only GREEN1, combined GREEN13, the complete47 affected UI cases
(347.34s),22 startup,13 latency and four
storage/platform/log controls all pass on stable source identities. Full
artifact preflight passes (180.27s). After those commands
finish, only the new test import alias is reordered to resolve its Ruff I001;
whole-module normalized AST equality proves all bodies/assertions exact.
The complete45 affected collection cases then pass on final262 pins
(52.57s); the pending/layout mounted
receipts and all runtime/fixture/targeted performance inputs stay exact.
There is no new whole47/39/preflight/native/full-suite claim after that
import-only correction. QA oct03_review_session_count_and_precompose_resize records separate exact sources/receipts.

Full production modules restore preceding AST after only the two targeted
methods are restored; test modules after only the six ownership rows and
one real-card regression/import are removed. All258 other source pins,
profile guards, fixtures, worker owners and existing caps are unchanged.
Zero-new Ruff35, changed-range format, final UI census/Backlog/diff checks
and the new/changed public signature audit pass. Existing ADR-043/067/094/150
apply, no new ADR. TASK-32367 AC9 and TASK-33625.2 AC6 are verified; all three
selected tasks remain In Progress until fresh exact-head Qodo/all four
hosted gates and final Done metadata-head review/checks precede normal merge.
Auto-merge stays off; prior scoped raw config failures/native/provider/Close
receipts remain under their explicit earlier identities.


Startup-cleanup and mounted-CSS dev integration (2026-10-03 UTC)

Rebased onto dev efea5e45cf2348dd0da50fedac215cd849143822; all52 preceding patches are exact.
The inherited startup timer capture supersedes three old media setup blocks.
Both real storage variants reproduce the missing-callback assertion before
removing only those blocks. The full census then restores byte-for-byte to dev
after removing only the existing trace-migration settlement; every real timer,
candidate-query/helper/first-GC assertion and platform/path ceiling stays exact.

The real fully mounted CSS census measured275 above the unchanged274 ceiling.
Its failure-frame probe names the owned bare Button approval selector. Replace
only that selector with the three existing Button IDs, preserving parsed(1,1,1)
specificity and every token declaration. Rebuild the app bundle from source;
whole source and generated bundle restore to the rebased anchor after only
that selector swap. Mounted census and complete approval journeys pass.

Final268-source qualification: 22startup,
13latency, 5ordinary storage/timer/platform/log controls,
47affected UI, 26SDK/service/loopback-transport,
1tooltip, and 13token/bundle-governance cases pass.
Fresh derived-artifact preflight passes. Zero-new Ruff35, changed-range format
and changed public-test signature audit pass;259 prior pins retain identities.
The inherited SDK's whole AST restores after only its diagnostics export method
is restored; no selected Console production method or guard changes here.

One earlier final-source tested warm visit measured10 helpers above ceiling9.
Both call-through traced variants pass at9, lighter tested capture at7, then
the ordinary final pair and controls pass without instrumentation. Real stacks
name character-scope metadata, conversation reads and workspace availability.
The original extra helper was not captured: exact cause remains unconfirmed,
no generic-flake fix or budget/worker/profile suppression is claimed. QA retains
the failure, actual stacks, source identities and final ordinary receipts;
fresh hosted performance success remains required.

Existing ADR097(boot budget)/125/126/150 apply; no new ADR. All three selected
tasks remain In Progress until fresh exact-head Qodo/all four hosted gates,
then final Done metadata-head review/checks before protected normal merge.
Auto-merge off. Earlier native/Close/provider evidence retains its original
source identity; no new native/full-suite/live-server-UAT claim.


Latest request-capture and UI-lane integration (2026-10-03 UTC)

Rebased onto dev 01a2020981c6197e5cd9945e5287567ad977edfe (PR2975).51of53 patches are exact; the
other two differ only in generated diagnostic inventory and additive lesson
context. Reversing only the incoming Python diff restores all20 complete
sources to the preceding published head. A checkout-owned Python3.12.11
interpreter imports this worktree under-I; dependencies are reused read-only.

Two preceding hosted UI lanes reached the unchanged20-minute cap at92%/83%
without an observed assertion failure. Scoped Close/compact harnesses now
use the shipping TldwCli.CSS_PATH with inherited widget/modal defaults.
Whole original test ASTs and every real geometry/action assertion remain.
The pending batch now drains existing factory patches/directories and runs
existing unfreeze/GC cleanup between fresh journeys. Its probe drains7actual
patches/directories and passes;6old apps remain reachable, so no app-retention
fix is claimed. Timings are observed comparisons, not a benchmark guarantee.

Final ordinary affected UI:53pass at180s per-case cap, no instrumentation,
247.48s elapsed. Startup22, latency13, real storage5 and
governance13 pass with unchanged runtime/measurement/style source closure;
earlier receipt source maps remain explicit. Incoming469/472 and Close71/77
pass: QA retains9untouched-dev fixture/profile/entry-admission failures and
ordinary repeat/whole-source or entry-AST identities; they are not green or
fixed. No admission/config/worker/guard or assertion bypass is used.

Artifact preflight passes; zero-new Ruff35, signatures, changed-range format,
Backlog/UI-census and diff checks pass. Final285source hashes are recorded.
No production, CSS declaration/token, census/workflow or budget change in the
timeout fix. Existing boot/token/component/GC ADRs apply; no new ADR. Earlier
native/provider evidence retains its identity. All3selected tasks remain
In Progress pending all4fresh hosted gates and clean/resolved Qodo, then final
Done-metadata-head review/checks and verified protected normal merge.


### 2026-10-04: latest-dev admission identity compatibility

Rebased all 54 Console patches exactly onto dev 81c7c94f486491d6fde09584710dc3a3172225a0 (PR2994 / TASK-34200). All incoming Python files match dev. Five inherited pins change; eight pins join, including the first-user-data-binding test whose entire file matches the immutable qualification anchor and dev. The 293-pin final map reconstructs exactly from the preceding 285-map plus those deltas. The five targeted receipts retain their actual 292-map identities.

All 135 targeted checks pass: incoming admission/identity 42, affected ordinary UI 53, startup guards 22, latency/responsiveness 13 and real SQLite storage/timer/platform/log 5. Every case retains actual private profiles, worker/admission/getter guards and original ceilings/caps; maximum UI group is 131.60 seconds under 180. Artifact preflight passes in 114.79 seconds, zero-new Ruff35/signature/Backlog/diff/124-file census checks pass. The 35 owned Python sources remain exact, so prior changed-range formatting retains its identity.

The preceding published docs-only head passed all four hosted gates with clean Qodo and resolved threads. Its first UI job hit the 20-minute limit at 95%, without a logged assertion failure; its first Linux Perf run billed 10.125 os.open/tick to tested credential polling. Same-head retry passed at 6.75/tick for both variants. Local ordinary and call-through pairs passed; traced local idle opens came from native backup-pause checks on executor threads, 37 opens per probe. No failed-Linux stack was captured; no guard was suppressed, budget raised, or universal flake fix claimed. The initial local runner missing its private profile directory executed zero cases and is preserved as invalid setup evidence.

Existing ADR-126 amendment and ADR094/097/195 apply; no new ADR. Earlier broader fixture/entry failures and native/provider evidence retain their original identities, with no fresh resolution/replay claim. No full suite or new live-server/native replay. All three selected tasks remain In Progress pending fresh exact published-head clean/resolved Qodo/all four hosted gates, then final metadata-head qualification and verified protected normal merge. Structured evidence: oct04_admission_device_identity_integration in the companion JSON.

## 2026-10-04: queue-shelf dev integration and indentation review

Integrated dev `225fbeba0ac1832921df95d20970a6d2b826a515` (PR #2986). All 55 existing Console patches replay exactly. The three incoming Python files match dev; two prior source pins change and the dispatch-recovery test is added, reconstructing all 294 final pins from the preceding admission map. The first post-rebase proof stopped because that third file was not in the earlier map; the corrected proof checks both existing changes and the added file. No application patch was needed.

All **162 targeted cases pass**: 74 complete queue/recovery cases, 53 approval/pending-decision/Close cases, 22 startup guards and 13 latency/responsiveness cases. Receipts use all 294 source identities and preserve private profiles, actual clicks/geometry, worker settlement, admissions/getters, counters, budgets and timeouts. The longest affected UI group is 90.32 seconds under the unchanged 180-second cap. Artifact preflight passes in 109.95 seconds. Zero-new Ruff35, public signatures, Backlog/diff and the unchanged 124-file UI census pass.

Qodo's stylesheet-indentation proposal was checked rather than applied blindly. Restoring four-space property indentation preserves every non-indentation line but the actual private-profile CSS byte guard fails at **608,858 bytes**, 768 over the unchanged 608,090 cap. Both source and generated candidate files were restored byte-for-byte before fresh green checks; the retained census is **607,166 bytes**. The two-space indentation was already documented above as a declaration-neutral ADR-097 paydown. [Review response](https://github.com/rmusser01/tldw_chatbook/pull/2953#issuecomment-5976245329) supplied that evidence; Qodo returned to zero active bugs/rule violations/cross-repo conflicts on the preceding published head, with all 37 threads resolved. All four hosted gates passed on that head.

QA key `oct04_queue_shelf_dev_integration_and_indent_review` retains exact delta pins, candidate failure/restoration, all fresh receipts and the review/gate evidence. Original admission/storage, broader inherited-fixture failures and native/provider receipts retain their original identities; this is not a fresh native/live-server/full-suite replay or a claim that those broader failures were fixed. Existing ADR043/094/097/150/161/195/210 apply; no new ADR. All three selected tasks remain In Progress until fresh latest-dev published-head review/all four gates, then final metadata qualification and verified protected normal merge. Auto-merge stays off.

## 2026-10-04: shared pending-kind keys and Settings dev integration

Qodo identified four Close labels repeating pending-kind strings. Four shared keys now live beside the existing approval key in the pure Console model module and are reused by pending-copy, interrupt-host contracts and Close consequences. All three whole modules retain the preceding published executable AST after expanding only the named keys and removing only their new declarations/import aliases. The added explicit model import edge is qualified by actual startup guards. Existing formatter ratchets show no debt growth or format-required overlap with changed ranges. No test source, visible copy, locking, callback, getter, fence or worker behavior changed.

The cleanup passed **136 targeted cases** on **296 stable source pins**: models/host48, complete owned UI53, startup22 and latency/responsiveness13. The longest owned UI group is71.71 seconds under the unchanged180-second cap. These receipts retain their original identities.

Then integrated dev `74fffc99f99cf47a39c1e67e0011768e57a55587` (PR #2990 Anthropic Settings sign-in selection). All57 existing patch bodies/context replay exactly;56 raw patches are exact and only the first patch's census index hashes differ. All four incoming Python files match dev. The composed census retains the existing two Console PR entries and the exact incoming Auth test entry:125 paths, floor124 unchanged. The three shared-key modules remain byte-identical to the qualified cleanup. Initial proof attempts stopped on raw census index hashes and then an incorrect expectation that the composed census should equal bare dev; the corrected proof checks patch bodies and reverses only the incoming census entry. No patch body or code changed during proof recovery.

Fresh **103 targeted cases pass on303 stable pins**: incoming provider-boundary5 and Settings UI10, models/host48, startup22, latency/responsiveness13 and real-storage/timer/platform/log5. Actual artifact preflight passes in122.76 seconds. Zero-new Ruff35, public signatures, Backlog/diff and125-file UI census pass. All original admissions/getters, private profiles, workers, counters, budgets and caps remain. The owned UI53 receipt is reused through exact source identity, not relabeled as a new303-pin replay.

Following qualification, dev advanced to `a418fa7967e765d5f4fee3da6c2bbfd29f2981d4` through PR #2988, which changes only the unrelated TASK-26022 metadata. All57 raw Console patches replay exactly on this docs-only rebase; the entire tree differs only in that incoming task file and all303 source pins remain exact. The actual103-case/preflight/static receipts retain runtime anchor `9e166e4138468d622c53aed9acf8caeab2d9aa13` and Settings base `74fffc99f99cf47a39c1e67e0011768e57a55587` rather than being relabeled as a new replay.

QA key `oct04_shared_kind_review_and_settings_dev_integration` reconstructs the final303 pins from the shelf294 map, three shared-key changes/two test additions, and two Settings changes/seven incoming/helper additions. It retains exact review, AST, formatter, rebase, receipt and preflight evidence. The upstream Auth task documents46 broader provider/QwenCloud fixture failures; those were not freshly reproduced or fixed here. Its new UI module retains the upstream bootstrap_profile marker. No native Auth/sign-in, credential access, new live-server or full-suite verification is claimed. Existing ADR012/033/094/097/150/195 apply; no new ADR. All three selected tasks remain In Progress pending fresh exact-head clean/resolved review/all four gates, final metadata qualification and verified protected normal merge; auto-merge off.


## Formatter/runtime and owner UI sharding integration

Runtime anchor `a20f4758dbb4b3c2cd6a5eca901e2dbbc9558018` on dev `ca2992cb10b24307fbae050643472ffb0a4388e7` inherits PR #2993. The prior59 Console commits were consolidated with an exact original tree and recoverable backup refs; no selected feature was dropped. Four overlapping Console modules preserve their complete feature AST and all comment tokens after upstream pinned Ruff0.15.22 formatting. Incoming nonowned files match dev; nonoverlap feature files and native PNG/SVG bytes match the original feature tree. Full upstream lessons and both additive feature lesson blocks survive. The diagnostic inventory was regenerated from actual combined sources. The original10 generated SVG whitespace diagnostics remain exact; no new whitespace finding is claimed. A proof initially decoded PNGs as text and was corrected to byte comparison without source changes.

Fresh runtime qualification: **257 passed,2 existing xfailed on313 stable source pins** (models/host48; worker-checker plus SQLite writer97pass/2xfail; incoming Settings19; owned UI53; startup22; latency13; storage/timer/platform/log5). The two xfails are existing W003 split-across-functions detector misses; three startup headroom warnings remain. Owned UI's longest case118.27s stays within180s. Actual313-pin artifact preflight passes in143.75s. No exact timing-cause or universal flake-fix claim is made. All real profiles, admission/getter/worker/canary checks, counters, caps and ceilings remain.

Published `d16d22fcbb2629f53aa5a79692279fcca80cc9c1` UI Fast Lane reached85% without a reported assertion failure; GitHub explicitly annotated its unchanged20-minute job cap. Integrated owner-merged PR #2999 on dev `ffb038115789cc8bf061f5cf520b5843b7074de6`. Both existing Console patch bodies/context replay exactly, with only census-checker index hashes differing. The workflow and2 CI test files match dev; the checker retains the already governed floor124. Full census remains125 files, partitioned by the actual helper. All runtime sources remain byte-exact to the qualified313-pin tree. The316-pin map uses one checker override and3 inherited additions. **34 fresh CI contract cases pass**, including real shard census coverage, invalid/empty shard handling and workflow checks. Fresh316-pin artifact preflight passes in166.06s; zero-new Ruff35, public signatures, Backlog and working diff checks pass.

QA key `oct04_formatter_runtime_and_ui_shard_dev_integration` retains actual anchors, exact map deltas, receipts and proofs. **Both UI Fast Lane matrix shards must pass**, alongside PR Fast Lane, Derived Artifacts and UI latency guardrails;20-minute cap, fail-fast false and125/floor124 census remain unchanged. Existing ADRs apply; incoming security ADR206/207 and owner sharding task34353 are retained without modification. Earlier fixture/native receipts preserve original identities; no new native Auth/sign-in, whole Architecture A/B, live-server or full-suite verification is claimed. All3 selected tasks remain In Progress pending final-head review/gates, then fresh task-metadata-head qualification and verified protected normal merge. Auto-merge is off.


## Latest-dev shared pane and Tab integration

Integrated owner shared-pane B0 on dev `83c264f28663d76b712e8a1baa6b6193bf2a0fa8`; actual runtime anchor `b4bda22edc0e25666a107840a00b5dcde36dce19`. ADR `backlog/decisions/212-shared-adaptive-pane-shell.md` and design-language amendments were read before the task plans and rebase. All owned Console runtime/native files are byte-exact to the preceding qualified tree. All incoming non-composed files match dev; the component catalog restores upstream exactly after reversing only the existing Console paragraph. Three metadata patches replay raw-exact. Retained all3 incoming UI entries and2 Console entries:128 entries, floor128 (owner floor125 plus the existing additive3). The owner pre-import557 ledger/source is inherited exactly; no authored budget or workflow change. Source-map reconstruction updates4 prior pins and adds18, yielding334.

Fresh targeted result: **918 passed out of919 collected** on334 stable pins. New pane/rail/Tab and pure resolver770; four Library ownership pins4; complete owned Console UI53; startup22; latency13; actual shard/workflow CI34; mounted route group23 includes21 real route-Tab passes and the MRO runtime pass. The remaining MRO AST guard fails at four unchanged modal sites (`buddy_conversation_modal.py`, `buddy_management_modal.py`, `buddy_character_review.py`, `console_endpoint_template_modal.py`). The actual unmodified guard assertion was executed on a complete archived dev package and failed with the identical sites; those four files and the guard are byte-exact dev. No assertion/allowlist/getter/admission/worker/cap exception was added; that inherited red is retained, not reported as fixed or green. Its baseline probe is an isolated pure AST case, not a runtime/full-suite/native replay.

Fresh artifact preflight passes in133.93s; zero-new Ruff35, signatures, Backlog and working diff checks pass. Both hosted UI shards, PR Fast Lane, latency and Derived Artifacts passed on preceding published `7f53a9837fefc6c83ff1352b4a06cb84fa4fc6fe`; Qodo was clean/resolved there. This latest combined tree needs its own fresh hosted checks and review. QA `oct04_shared_pane_latest_dev_integration` records actual anchors, maps, receipts and limits. Owner-documented full Library/order-dependent failures were not replayed; no new native Auth/sign-in, credential, live-server or full-suite verification is claimed. Previous native/failure receipts retain original identities. All3tasks stay In Progress until fresh exact-head clean/resolved Qodo and all5 checks, followed by final metadata-head checks/review and protected normal merge. Auto off.


## Following setup-documentation dev rebase

Rebased all5 Console patches raw-exact onto docs-only dev `67fc5310471823262eb6ce344a539e699784bed8` (PR #2982, setup-shape spec, ADR217/amendments and task records). The entire tree differs only by those incoming documents; every one of334 pinned sources remains exact. Incoming ADR217 and dated012/029/033 amendments were read; they specify future setup delivery and change no selected Console implementation or acceptance. Fresh artifact preflight passes on actual docs-tree anchor `65c7b1f2bea6f8ebf696e25b65a201680baf764b` in136.99s. Actual918-pass runtime receipts, one baseline-confirmed inherited MRO red, lint/static and original native receipts remain on their original anchors, reused by source identity rather than relabelled as new runs. QA `oct04_following_setup_docs_dev_rebase` retains the whole-tree/patch proof and actual preflight. No new runtime/full-suite/native claim. All3 tasks stay In Progress pending fresh final-head clean/resolved review/all5checks, followed by final metadata qualification and protected normal merge; auto off.


## Library reader freshness integration from latest dev

Rebased onto dev `f1f80847a410525ce1b00d27ea0e8be824a98422` (owner PR #2989, TASK33628.10, Library ADR076). The actual runtime anchor is `5523d633901e44624e5766b2c851b53f408b9239`. All owned Console runtime/native sources remain byte-exact to the preceding qualified tree; incoming non-composed paths, original profile markers and the reader controller's 967-line ratchet match dev. All six metadata patches replay raw-exact. The real diagnostic generator reproduces the combined inventory. The UI census contains 130 entries with its 128 floor unchanged, retaining every incoming and both Console entries. Two source-pin overrides plus ten new dependencies yield 344 pins. No new ADR or selected feature change.

Fresh targeted result: **84 passed of 86 collected** on 344 stable source pins: reader freshness and real file-backed Console Delete/Undo→Library journeys 12, startup 22, latency 13, shard/workflow CI 34, and exactly three controller size-ratchet cases. The two modified reader cases fail with `RecoveryRequired: raw_source_selection_changed` before owned Console behavior. Both actual tests reproduce this refusal on a complete archived, unmodified dev package with ordinary conftest/profile contracts and verified module origin. The first baseline origin proof stopped before pytest on a `/var` versus `/private/var` path comparison; resolving both paths corrected the proof while preserving the untouched archive and application/guard code. No added marker, fake getter or admission bypass. These failures are recorded, not claimed fixed or green.

Fresh artifact preflight passes in 126.64s. Zero-new Ruff across 35 modified Python files, public signatures, Backlog IDs and working diff checks pass. QA `oct04_library_reader_delete_refresh_dev_integration` stores maps, exact receipts, rebase proof and baseline limitations. Previous 918 passes/one inherited MRO guard red and the 53 owned Console UI passes retain their original 334-pin anchor; source identity permits reuse without claiming a new full run. Native captures remain unchanged. No new native Auth/sign-in, credential, live-server, full Library, Architecture A/B or full-suite verification. All three tasks remain In Progress until fresh final-head clean/resolved Qodo and all five hosted gates, including both UI shards, then final metadata qualification and protected normal merge. Auto-merge is off.


## Database schema owner integration from latest dev

Rebased onto dev `14a14c4a7ab0da85831afb290ab90d75170fbb79` (owner PR #3002, TASK19565/19566, ADR208). Actual runtime anchor `b0b9d5a70f14ab5a16d98975b08bd980339ce6be`. The owner’s SQL-file migration authority, ChaChaNotes v76, Evals v6 and exact restricted recovery authorizer remain intact. All 81 owned Console runtime/native paths are byte-exact; all incoming non-composed paths and eight metadata patches are exact. The actual diagnostic generator reproduces the combined inventory without a following delta. Six changed shared-source pins and 113 additions yield 457 pins, including all 73 retained migration SQL files. UI census 130/floor128, original profiles, admissions/getters, workers/canaries, budgets and caps are preserved. No new ADR.

Fresh targeted result: **345 passed of 345 collected** on 457 stable pins: schema/migration/recovery 163, Console models 48, complete owned Console UI 53, real file-backed Delete/Undo→Library 12, startup 22, latency 13 and CI 34. The database group verifies authoritative migration files, trigger bodies, historical replay/atomicity, v74/v75 receipts, fresh v76 schema/capture/FTS and exact Evals v5→v6 restricted recovery. Artifact preflight passes in 109.7s. Zero-new Ruff across 35 modified Python files, signatures, Backlog and working diff checks pass. QA `oct04_schema_migrations_latest_dev_integration` stores maps, rebase proof and actual receipts.

Earlier reader admission failures on dev f1f80847 and broader MRO guard red on dev83c264f2 remain on their original test/baseline anchors. Neither earlier command is relabelled as a new 457-pin run, and no full Library/Architecture green claim is made. Previous native captures and W003 xfail receipts retain actual identities. No new native Auth/sign-in, credential, live-server, installed-wheel build, full Library, Architecture A/B or full-suite verification. All three tasks remain In Progress until final exact-head clean/resolved Qodo and all five hosted gates including both UI shards, followed by final metadata qualification and protected normal merge. Auto-merge off.


## Encryption startup owner integration from latest dev

Rebased onto dev `057e62eaec5343ffce9445cfb32881e756dd7634` (owner PR #3000, TASK34100.4 part1). Actual runtime anchor `abe69825078ad5c3accf61e51f736e7eba06db10`. Incoming one strict-decrypt startup unlock, config lifecycle, provider readiness and canonical Settings Encryption card are retained exactly, along with the owner's real profile contracts. All81 owned Console runtime/native paths are unchanged; ten metadata patches replay raw-exact. The real diagnostic generator reproduces the combined inventory. Eight pin overrides and25 new dependencies yield482 pins, retaining all73 migration SQL files. Census130/floor128, original guards and budgets/caps unchanged. No new ADR; selected Console ADR094/097/150/195/212 and inherited database ADR208 remain applicable.

Fresh targeted result: **368 passed of368 collected** on482 stable pins: six owner encryption suites180, three modified Privacy hub cases3, affected CLI private-path3, Console models48/UI53, real file-backed Delete/Undo→Library12, startup22, latency13 and CI34. Owner profile markers are exact upstream; none were added or changed here. Artifact preflight passes in 88.0s. Zero-new Ruff35, signatures, Backlog and working diff checks pass. QA `oct04_encryption_startup_latest_dev_integration` preserves exact receipts/maps/proof.

The previous DB163/combined345 passes retain their actual457-pin b0b9d5a7 anchor and source identities; unchanged database/migration/recovery schema sources are combined with independently verified new config/startup/Console paths. Prior reader admission failures, MRO guard red, native captures and W003xfail receipts retain original anchors and are not relabelled as fresh482-pin runs. Owner-reported broader Console-session-settings/config-admission failure sets were not replayed; no blanket Library/Architecture/Settings green claim. The owner's deferred wizard and non-LLM credential TASK34100.4.1 work stays outside the selected Console tasks. No fresh full-suite, full Library/Architecture A-B, installed-wheel build, native Auth/sign-in, live-key/live-server or external-credential claim. All3 tasks remain In Progress until fresh exact-head clean/resolvedQodo/all5hostedgates including both UI shards, followed by final metadata qualification and protected normal merge; auto off.


## Chat settings Phase6 integration from latest dev

Integrated latest dev df2ba424de63576d36d9c3e38387c84285303f1c, Chat settings Phase6 PR2992, under existing ADR095/031/033/097/150/161/195/212. No new architecture or selected AC.75 owned non-overlap runtime/native paths byte-exact; reverse feature Python/lessons patches recover exact dev; shared approval CSS semantics retained with obsolete owner cap removed.12 metadata patches exact; actual generators reproduce;533 stable pins (22 overrides+51 additions),73SQL pins; census135/floor133; owner budgets/ratchets and ordinary profile/admission/worker/canary guards retained. Fresh targeted 387 passed/389 collected, 2 inherited failures reproduced exactly on unmodified dev; real preflight 127.78s; zero-newRuff35/signatures/Backlog checks pass. QA oct04_chat_settings_phase6_latest_dev_integration retains actual receipts/source maps and prior anchors; no full-suite/fullLibrary/Architecture A-B/newnative/Auth/livekey/server claim. Owner deferred riders and broader settings/admission failures are outside selected work. Remain In Progress pending fresh final-head all5checks including bothUIshards/clean resolvedQodo, then final Done-metadata qualification and protected normalmerge; auto off.

Actual runtime anchor `8e54a7063428f631f0f9f67394396b6830e5984f`. Incoming Phase6 owner layout, pair-only model picker, hidden-field disclosure and saved-default staging retained. Owner deferred riders remain outside these three selected Console tasks. No new architecture or AC. 75 non-overlap owned source/native paths byte-exact; reversing the original feature changes recovers exact dev for both shared Python modules and the lessons appendix. Shared approval CSS semantics are unchanged with the obsolete wide settings cap removed.12 metadata patches raw-exact. Source map533 reconstructs482+22 overrides+51 additions, with all73 migration SQL pins. Actual CSS build and diagnostic generator reproduce; census135/floor133 retains all prior and incoming entries. Incoming budget snapshot and module ratchets are exact dev; no existing token value changed. The new checks use the ordinary unchanged profile/admission/getter/worker/canary/budget/cap contracts. Prior encryption368, DB345, reader/MRO/native/W003 receipts retain actual original anchors; none are relabelled as a new533-pin run. No new full-suite, full Library/Architecture A-B, installed-wheel, native Auth/sign-in, live-key/server or external credential verification. Owner-reported broader settings/admission reds and deferred riders are not claimed fixed. Three tasks stay In Progress until fresh exact-head resolved clean review and all5 hosted gates, followed by final metadata qualification and protected normal merge; auto off. The two full component governance checks fail on inherited dimension/Python-style offenders; actual unmodified dev reproduces the exact same failures and offenders under unchanged ordinary profile guards. No assertion, marker, floor or allowlist was suppressed.
