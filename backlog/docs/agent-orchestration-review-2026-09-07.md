# Agent orchestration review and repair ledger — 2026-09-07

> This ledger preserves the original findings and historical verification below.
> The workstream has now been integrated with current dev's lifecycle and controller
> boundaries. Read the [PR integration status](../../Docs/superpowers/reviews/2026-09-11-agent-orchestration-pr-integration.md)
> for current verification and merge gates. TASK-3070 and the HandsFree ownership
> follow-up were completed upstream; their older status notes are historical.
> The later [remaining-work status](agent-orchestration-followups-2026-09-12.md)
> now records completed Settings/cap delivery, repaired integration defects and
> ordinary-Git worktree confirmation/recovery. The remaining tasks and fleet
> parent are Done after final correction e135a085f2 and independent re-review;
> see the [restoration record](../../Docs/superpowers/reviews/2026-09-12-agent-worktree-restoration.md)
> for current evidence, accepted limits and retained guard debt.

Reviewed working tree based on `bc745f854`, including the in-flight Console
controller extraction. This is a scoped engineering review, not an exhaustive
security audit or a live-provider certification.

## Findings and disposition

| ID | Finding | Evidence | Disposition |
| --- | --- | --- | --- |
| TASK-32483 | Steering admission has no aggregate bound; retention excludes unread steering from its size check. | 100 valid 4,000-character messages retained a 401,471-character payload under the default 200,000-character cap. | Fixed: 32-entry / 64,000-character queue, complete payload sizing, explicit refusal with preserved draft. |
| TASK-32484 | Accepted final-turn steering can remain unread while the queued indicator disappears. | Real loop/bridge probe: one model call, accepted steering unseen by model, terminal queue zero, one retained unread entry. | Fixed: terminal unread count and current continuation availability; explicit continuation preserved. |
| TASK-32485 | Wake safety tests omit their helper module's legacy-session fixture. | Three original failures and one further headless dispatch test fail at binding preflight; supplying legacy-session setup restores the original assertions. | Fixed: fixture imported in safety tests; headless dispatch explicitly selects legacy setup for its recording bridge. No production authority change. |
| TASK-32486 | Coordinator lifecycle events accumulate without a production consumer. | 100 reserve/finish/prune cycles leave zero handles but 200 events. Source search finds no production `drain_events` caller. | Fixed: events expire with pruned handles; live survivor events remain ordered. |
| TASK-32487 | Retained transcript snapshots share nested objects with callers. | Mutating a returned native `tool_calls[0].id` changes the subsequent stored snapshot. | Fixed: deep copies on admission and read; failed copies refuse retention before claiming steering. |
| TASK-32018 | Refused wakes retry immediately and can starve other conversations. | Real controller: unavailable provider resolved 183 times in 150 ms. Two-session probe: first refused 143 times, ready second never attempted. | Fixed: one-second retry delay per refused conversation; ready conversations proceed. |
| TASK-18312 | Pruned child handle IDs lose their useful terminal refusal; same-process run IDs can be called an earlier session. | Original resolution ladder consulted live handles, retained transcripts, unpruned handles, and DB run IDs. | Fixed: bounded payload-free terminal identities, accurate status/no-retention refusal, honest DB-only and expired-handle copy. |
| TASK-18311 | Per-child spend disappears at prune; original and continued runs have no durable aggregate. | Original handle-only budget rollup and continuation characterization reproduced the loss. | Fixed: schema v16 saves nullable budget counters; historical rows and selected continuation ancestry show known/partial totals without refeeding billing. |
| TASK-15665 | Live provider client pools lack deterministic closure after teardown. | Real bridge reproduction: child completes after parent teardown and closes its loop, but its HTTP client remains open. | Fixed: primary and child lifelines drain pending work and close their own pool before stopping the loop; injected clients remain caller-owned. |
| TASK-15200 | Historical fleet rows lose elapsed/result detail and stuck/cancelled status styling. | Real DB/bridge and rendered Console regressions reproduced missing child-owned detail and timestamp spans. | Fixed: shared historical projection restores bounded saved detail and budget counts; valid terminal timestamps show an explicitly approximate span; warning/muted status colors. |
| TASK-15201 | Long fleets lack a full-history action and expansion does not reveal the controls. | Compositor checks reproduced nested-scroll clipping; a childless later primary also hid the history entry. | Fixed: four-row preview with complete counts, expansion reveals the action, and a read-only 50-row paged picker reaches earlier/superseded children even when the current preview is empty. |
| TASK-32019 | Concurrency and spend caps are per conversation/run, not global or wake-chain bounds. | Four conversations sustain eight gated children; six successive wakes complete without a user send. A terminalized child can still occupy a worker. | Design complete under ADR-134 with user-selected conservative defaults. Operation ownership and shared tool/child admission implemented in TASK-32034/32035; durable chains plus runtime enforcement are implemented in TASK-32036/32037. |
| TASK-32020 | Fast-child results wait for the slowest sibling; a blocked wake holds global wake serialization. | Two-child and real approval probes both hold ready delivery for the full 350 ms gate, despite unused primary capacity. | Design complete in ADR-135: individual notifications, fixed batching window, two bounded wake slots with manual reserve and round-robin fairness. Implemented in TASK-32037; real gated-sibling and concurrent-approval tests pass. |
| TASK-32021 | A crash or failed stamp after wake work can replay it; missing attention marks can also strand automatic delivery. | A finished wake with an injected stamp failure triggers a second provider call when durable state is claimed. Source confirms marks are the discovery index. | Design complete in ADR-135: atomic attempt claims, required acceptance fence, FULL-synchronized ledger mutations, and conservative manual review. False exactly-once/one-extra-replay claims corrected. Durable claims and attempt transitions are implemented in TASK-32036; runtime acceptance and owner revocation are implemented in TASK-32037. |
| TASK-32022 | No child-to-parent progress messages or direct sibling messages in the original review baseline. | Relay-first implementation now has bounded child reports, explicit supervisor collection, and existing steering for relay; direct peer messaging remains outside scope. | Implemented under ADR-136 in TASK-32489/32490/32491. All three task reviews approved. Final review found a Save identity defect; the stable-owner and stale-dispatch correction passed scoped re-review with no new findings. [Implementation and review complete](../../Docs/superpowers/reviews/2026-09-10-scoped-agent-messaging-implementation.md). |
| TASK-32034 | Timed-out tool workers remain active with no shared admission bound. | Four sequential timeout returns leave four gated tool workers alive. Caller timeout and worker completion are different events. | Fixed: app-owned capacity follows worker/driver completion, including delayed cleanup. Eight tool slots, two manual reserves, and per-run refusal while a timed-out worker remains alive. |
| TASK-32035 | Per-conversation handles allow excess aggregate children and can free logical capacity before worker cleanup. | Four conversations previously held eight children; gated terminal-child tests retain physical work after row completion. Failed-start tests also exposed the pure loop's unrefunded unnamed-spawn counter. | Fixed: six shared child slots, two manual reserves, ownership through cleanup, and typed pre-dispatch refusals that spend no spawn allowance. |
| TASK-32036 | Automatic generations lack durable shared allowance and causal chain identity. | Real SQLite tests cover concurrent last-slot admissions, interrupted attempt claims, mixed-lineage wakes, token overruns, and clock/handle recovery. | Implemented: schema v17, FULL-synchronized ledger APIs, immutable manual/descendant lineage, exact live wake authorization, and separated chain batches. Dispatch enforcement and schema v18 runtime-owner revocation are supplied by TASK-32037. |
| TASK-32037 | Automatic-chain limits and durable attempts were not enforced at runtime. | Real dispatch, startup, approval, cancellation, owner-replacement, and rendered pause regressions. | Implemented: shared finite budgets, acceptance/owner fences, incremental fair delivery, badge-independent recovery, and explicit saved-result pauses. |
| TASK-32488 (C1) | Concurrent cold starts can disable a run log after a private-data-directory creation race. | Barrier-controlled real opens reproduce `FileExistsError` and an inactive writer through config data-root resolution. | Fixed: reopen the competing entry without following links, then retain owner/type/mode validation. Both real writers now persist their records. |
| TASK-32492 | Acceptance refresh creates a coroutine before Textual admits its worker. | Strict warning regression and allocation trace reproduce an orphan UI coroutine when the unmounted screen cannot start a worker. | Fixed: submit the bound async callable so allocation occurs when the worker runs. Original runtime-gate regression now fails on leaks; 47 affected and two mounted echo/teardown cases pass. Independent review approved all criteria with zero findings; task Done. |
| Hands-free ownership follow-up | Existing `_sync_hands_free_switch` queries the Screen DOM from inside `ConsoleHandsFreeController`, contrary to DESIGN.md section 7. | TASK-3070.10 before-image matches HEAD `187a1892d0`; the method directly calls `self._screen.query_one` and then updates the Switch. | Completed upstream. The integrated implementation retains dev's injected `_sync_hands_free_state_fn` presentation callback; no separate repair remains. |

TASK-3070.7 restored the adjacent Console architecture gate under the
[approved controller boundary](../../Docs/superpowers/plans/2026-09-10-task-3070-7-console-character-controller.md).
Moving the six remaining character-policy methods reduces ChatScreen from
17,656 lines / 594 methods to **17,483 / 588**; the ratchet is lowered from
17,570 / 591 to those earned counts. Task and final scoped reviews approved;
TASK-3070.7 is Done.
TASK-3070.9 subsequently extracted first-chat ownership and lowered the caps
again. Its follow-up below records the passing extraction checks and a later
40-line gate regression from concurrent TASK-32309 Character rail handlers.
TASK-3070.10's reviewed auto-speak extraction now leaves the Screen at
17,171 lines / 582 methods: two lines above the unchanged 17,169-line cap.
That child is Done; parent TASK-3070 remains open for final rebased-wave closeout.

A child's wall-clock budget is checked between
model calls: an in-flight provider call also needs its own timeout/cancellation
contract; cover that in the lifecycle/admission design rather than claiming
the loop's wall-clock ceiling interrupts an active network request.

## What is implemented correctly within the tested scope

- One conversation-owned coordinator survives supervisor turns; children use
  independent threads and handles with a per-conversation live cap.
- Supervisor and user steering share a locked FIFO mailbox. Delivery occurs
  only before a model call and after a complete tool-result batch.
- Source labels are mechanism-generated. Steering does not cancel a run,
  resolve an approval, or grant another agent's tools.
- Finished-agent continuation creates a new lineage-linked run, with retained
  history in process memory. Restart resurrection is explicitly unsupported.
- Completion wake, headless delivery, durable notification marks, and the
  delivery ledger exist. Shared session tasks have stable IDs and CAS updates.

### Automatic runtime enforcement and recovery (TASK-32037)

The accepted defaults now apply on actual dispatch: three accepted wakes, six
automatic child launches, 32 shared model calls, 500,000 budget tokens, 8,192
output tokens per call, and 900 elapsed seconds. Exact prepared calls, helpers,
and surviving children share one causal allowance. A fixed 250 ms grouping
window releases individual survivors before slow siblings; two conversations
can wake concurrently with one primary slot reserved for manual work and
round-robin selection. Final fleet usage reconciliation keeps its drain boundary.

Required FULL acceptance precedes preparation and dispatch. Exact result claims
prevent accepted work from being replayed after errors or interrupted completion.
Startup discovers results without badge authority. Schema v18 adds a durable
runtime-owner fence so a child of a completed wake cannot act under replaced
ownership; late usage can still settle. The inspector shows a wrapped pause
notice and saved results remain accessible through Run history. Viewing clears
attention while preserving the execution pause; explicit manual work does not
replenish old chains.

Independent review reproduced and drove fixes for preparation cancellation
stranding a primary slot, stale completed-parent authority, and provider workers
starting after an executor wait had invalidated their authority. The UI/startup
review also removed a permanent-unread badge and made an empty-startup recovery
failure visible to subsequently resumed sessions. Each correction has a targeted
regression; no external provider or power-loss guarantee is claimed.

Final backend verification passed **1,344 tests**, excluding two local-socket
cases that passed separately with loopback permission. The seven affected legacy
UI modules passed **34 tests**; launch passed **11**, startup **14**, and the
independent runtime re-review **65**. These suites overlap and their counts must
not be added. Final UI/re-review results are recorded in TASK-32037's notes.
New modules pass Ruff/format checks; legacy lint counts do not increase.
The five reviewed diagnostic owners match the inventory. Existing unrelated TTS
inventory drift and the Console screen-size ratchet remain outside this patch;
this work does not raise the ratchet or refresh unrelated diagnostic entries.
No full test sweep or external provider call was performed.

## Verification evidence

### Messaging design baseline — 2026-09-09 (TASK-32022)

The user selected relay-first on 2026-09-09 and requested a further review.
[Seven design findings](../../Docs/superpowers/reviews/2026-09-09-scoped-agent-messaging-review.md)
are resolved in ADR-136/spec: serialized-envelope sizing, productive-reader cycle
detection, user recovery from blocked queues, private continuation history,
body-free step projections, restored pending-call authority, and collection
guidance. Fresh targeted runtime/continuation verification passed **106 tests**;
these validate existing integration constraints, not the proposed messaging tools.
An independent reviewer rechecked authority, continuation, and projection fixes.

The accepted ADR-136 and spec have been reviewed for authority, bounds,
lifecycle, delivery wording, and compatibility with ADR-134/135. Local links,
source-map paths, placeholder checks, and changed-document whitespace pass.
The design does not establish implementation evidence for the proposed tools.

The targeted existing steering/mailbox/task-store run returned **216 passed,
6 failed**. All six failures occurred while constructing Python's multiprocessing
queue lock (`SemLock`, `OSError: [Errno 28] No space left on device`), before the
application regression body. An isolated `get_context('spawn').Lock()` failed
the same way with 22.2 GiB of disk free. A targeted rerun with sandbox escalation
also failed at that same prerequisite (6 failed, 179 deselected). This does not
establish a sandbox-only cause or a task-store defect; those six regressions
remain unverified on this host. No application/test code was changed to conceal
the failure, and no full suite was run.

Commands and raw output:

```text
.venv/bin/python -m pytest Tests/Agents/test_fleet_send_to_agent.py Tests/Agents/test_fleet_steering_mailbox.py Tests/Agents/test_session_todo_store.py -q --tb=short
/tmp/task32022-messaging-baseline.txt
/tmp/task32022-multiprocessing-regressions.txt
```

### Scoped messaging implementation — 2026-09-10 (TASK-32022; TASK-32489/32490/32491)

The approved [ADR-136](../decisions/136-scoped-child-progress-and-supervisor-relay.md)
contract is implemented in the working tree. Children can enqueue bounded,
session-only progress for explicit supervisor collection; onward relay uses the
existing steering mailbox. The inbox captures run/conversation ownership, checks
serialized-envelope bounds and lifetime limits, rejects stale owners, and keeps
report bodies out of ordinary step metadata. Native continuation treats collection
as a productive read, retains the disclosed private continuation history, and
refuses restored pending messaging calls. Reports do not trigger model wakes.

The existing Console Agent section exposes queued progress independently of child
row retention or agent mode. An explicit modal shows literal report bodies and
discards only checked report IDs from the captured owner. Inspection does not
consume reports or clear completion attention. Saved and unsaved conversation
navigation shows queued counts separately from historical child counts; callbacks
cannot follow a closed/reopened owner. User guidance describes session-only loss,
private history, and conditional shared-capacity recovery without changing the
reviewed body-free tool refusal format.

Evidence for the [implementation review](../../Docs/superpowers/reviews/2026-09-10-scoped-agent-messaging-implementation.md):

- The parent ran **586 passing backend tests across 14 targeted files**, covering
  store/coordinator, runtime/service, message tools, native continuation, bridge,
  ownership/preparation, and steering. The two backend implementation tasks were
  independently approved before the Console extension.
- The final Console feature run has **6 passing rendered tests**. It verifies
  exact selected discard with a concurrent arrival, unchanged lifetime counts and
  completion attention, saved/unsaved identities, same-ID owner replacement,
  model collection while the modal is open, literal bodies, polling cleanup,
  finish/prune and mode-off access, navigation changes, and actual regional badge
  paint at **180×48 and 100×36**. Six synthetic-data Console/modal/navigation
  captures use production modular CSS; SVG and PNG evidence accompanies the
  task report. Regional compositor assertions prevent a repeated Agent label
  from masking a clipped conversation-row badge.
- **220 existing UI/navigation regression cases passed across recorded targeted
  runs and a focused fixture rerun** (96 fleet/inspector/browser, 84 workspace/left
  rail, 40 agent rail/controller). DB-less affected harnesses now use the existing
  real DB helper; production fail-closed recovery and original assertions remain.
  These figures describe separate evidence groups, not one full-suite run, and
  should not be added to the historical repair-run counts elsewhere in this ledger.
- After review correction I1, changed Python files introduce **zero Ruff
  diagnostics** against preserved baselines; both new Python files pass focused
  Ruff and formatter checks. The initial zero-diagnostic claim was incorrect:
  its JSON recorded SIM102 in the new rendered test helper. The nested condition
  is now flattened with the same short-circuit bounds guard, and the regenerated
  comparison exits nonzero for any new-file or introduced diagnostic. Original
  JSON and correction evidence remain preserved in the task evidence directory.
  All five generated
  CSS bundles reproduce. The existing screen-size ratchet remains failing:
  this pass changes ChatScreen from **17,653 lines / 594 methods** to **17,656 /
  594**, against the unchanged **17,570 / 591** ceiling. The three added lines are
  callback wiring; no new screen methods were added.

Detailed commands, RED/GREEN logs, baseline comparison, screenshot paths, and
self-review are in
`.superpowers/sdd/2026-09-10-scoped-agent-messaging/task-3-report.md`; the parent
backend evidence is `final-backend-tests.log` and `final-backend-manifest.json`
in that directory. Final combined review found one Important Save identity defect.
The correction preserves an eager native progress owner through successful,
failed, cancelled-await and concurrent-close Save paths; restored live siblings
share it until last close. Controller/bridge expected-owner checks also prevent
queued or partially prepared submissions from following a reused native ID.
Real submitted reports, later collection, live senders and rendered selected
views retain their original inbox and source IDs. The single scoped re-review
closed I1 with zero new findings.

Final correction verification: **400 passed** across nine affected files, including
14 identity and seven styled progress cases; separately **84 passed** for queue,
coordinator and tool checks. The nine corrected Python files add no Ruff findings
(204 inherited before/after) and have no formatting differences overlapping the
change. These counts overlap earlier evidence and must not be added. Exact
commands and logs are in `final-fix-1-report.md`; independent disposition is in
`final-fix-1-review.md`. Parent verified the frozen source/test hashes and no
unexpected file changes since the combined review.

Known Requests dependency and unrelated pytest cleanup warnings remain. The final
run also reported an unawaited `ChatScreen._sync_native_console_chat_ui` coroutine
in the existing runtime-gate swap test. The subsequent TASK-32492 investigation
reproduced the actual allocation/refusal boundary and fixed eager coroutine
creation at the acceptance hook; see the follow-up evidence below. No full suite, external provider call,
staging or commit was performed for this extension.

### Acceptance refresh follow-up — 2026-09-10 (TASK-32492)

The runtime-gate test's unmounted screen called the real submission-acceptance
hook. That hook allocated `_sync_native_console_chat_ui()` before `run_worker`
could reject the absent application context. The controller contained the hook
exception to preserve the accepted send; cleanup then emitted the unawaited
coroutine warning. Promoting RuntimeWarning and pytest's unraisable warning to
errors produced a real failing regression, with the allocation trace pointing to
the hook. The existing test now retains these strict filters.

Passing the bound async callable to Textual defers coroutine creation until worker
execution. The one-line fix preserves its group, nonexclusive scheduling,
best-effort error handling, and composer clearing. **47 affected tests passed**
(runtime swaps, composer clearing and next-draft preservation); separately **two
mounted tests passed**, proving acceptance-time echo with transcript polling
disabled/provider output held, and safe post-teardown sync. The isolated strict
regression is already included among the 47. No unawaited warning remains in those
runs. Ruff findings are unchanged (150 in ChatScreen, 10 in the legacy swap test
file), with no formatting differences on edited ranges. Screen size remains
17,656 lines / 594 methods; its existing 17,570 / 591 budgets are unchanged.

Evidence, exact commands, RED/GREEN logs and scoped review are retained under
`.superpowers/orchestration-followups-2026-09-10/`. Requests compatibility and
unrelated pytest-cleanup warnings remain. The standalone semaphore failure was
reconfirmed without repeating six tests that cannot reach application code. No
full suite, provider network call, dependency change, host cleanup, staging or
commit was performed. Independent review approved all criteria with zero findings;
TASK-32492 is Done. TASK-3070 retains its existing reviewed multi-task
extraction plan; this correction does not claim to complete that separate work.

### Character boundary and history fixture — 2026-09-10 (TASK-3070.7)

The six remaining character picker, identity, card lookup, and session-choice
methods now belong to `ConsoleCharacterController`. Named lazy dependencies
preserve current session/store lookups; picker presentation and worker admission
stay on the screen. Prompt seeding, handoff ordering, notifications, avatar
refresh, and existing failure behavior are preserved. A completed-owner AST guard
prevents these methods from returning to ChatScreen. The screen loses **173 lines
and six methods**, and the size gate passes at its newly lowered 17,483 / 588 caps.

Targeted runs and corrective reruns cover **323 distinct cases with passing latest
outcomes**, including plain-fake controller behavior, real session handoffs,
mounted consumers, and architecture gates. This is a deduplicated set, not a
single full-suite run. The separate 49-case avatar run also passes. Root comparison
confirms equivalent policy ASTs after dependency substitution and conservation of
all 95 diagnostic calls across the affected production files. Mutation probes
prove the ownership and worker-dispatch checks detect their intended regressions.

Two mounted history cases initially failed in both the current tree and a
before-image source overlay: startup ran before the test bound its real bridge,
leaving recovery unavailable. The local fixture now binds the bridge before mount
and positively asserts recovery, retaining all preview/paging/drilldown assertions.
A bare native-session test also now supplies the real character owner. No production
recovery policy was changed. The extra history acceptance criterion was recorded
before the fixture repair.

No new Ruff findings or formatter changes overlap the owned edits; existing
whole-file static debt remains. The global diagnostic inventory command still
fails on unrelated archive, workspace, and TTS entries. Substituting this task's
before-images preserves exactly that external delta, so only affected inventory
rows were updated. Requests compatibility and unrelated pytest-cleanup warnings
remain. No full suite, provider call, host cleanup, staging, or commit was performed.

Evidence and exact diff hashes are retained under
`.superpowers/sdd/2026-09-10-task-3070-7-console-character-controller/`.
Task and final scoped reviews approved with no Critical/Important findings. Four
stale ownership comments were corrected and passed scoped re-review; executable
AST equivalence confirms that final cleanup changed no behavior. TASK-3070.7 is
Done. ADR required: no; this implements the existing Wave 6 / DESIGN.md section 7
boundary without changing a service, storage, security, or UX contract.

### First-chat ownership follow-up — 2026-09-10 (TASK-3070.9)

The eight first-chat handoff methods now belong to `ConsoleSessionController`
under the [approved extraction plan](../../Docs/superpowers/plans/2026-09-10-task-3070-9-console-first-chat-controller.md).
Eligibility, exact claims/configuration fences, acknowledgement, rollback and retry
retain their original policy. Named control/focus callbacks keep widget operations
on ChatScreen, while wizard and mount/resume callers address the Session owner.
The controller owns notification deduplication state; an assignable Screen descriptor
preserves compatibility. Rollback workers receive a partial async callable, delaying
coroutine creation until worker execution. The reviewed extraction reduced the screen from **17,483 lines /
588 methods to 17,169 / 584**; ratchets are lowered to the earned counts.

**135 distinct targeted cases passed on the reviewed extraction**, covering isolated
controller behavior, existing first-chat/wizard integration, handoff storage,
Session/wiring, the complete Wave 6 inventory, and relevant decomposition/CSS gates.
Mounted rollback tests retain their original state and focus assertions. This is a
deduplicated set of targeted runs, not the full repository suite. The initial
post-move config-guard test failure was a stale module lookup; repointing it to the
Session owner preserved the original assertion and restored the case.

Mutation checks catch removal of both configuration-generation and exact-claim
fences. The first config mutation initially survived an outcome-only test because
later rollback restored the final state. The strengthened test now also asserts
that a stale handoff never creates a session; the mutant makes two forbidden create
calls. Ownership and mount/resume dispatch guards also detect their intended
removals. This actual incident is recorded in the testing lessons.

Root AST comparison confirms all eight policy bodies match after the documented
control/focus/worker substitutions. All **159 diagnostic calls** across the affected
files are conserved. Static comparison reports no newly introduced Ruff findings;
existing whole-file debt remains. Compilation and scoped formatting/diff checks
pass. The lower ratchet passed on the reviewed extraction. During final review,
concurrent TASK-32309 Character rail handlers added **40 Screen lines**, leaving
the shared checkout at **17,209 lines / 584 methods**. A fresh ratchet run reports
**one failure / one pass** against the unchanged **17,169 / 584** caps. Exact
before-image comparison isolates those external handlers from this extraction;
they are preserved and the earned cap is not raised. The 135-case result does
not claim that the current shared checkout passes every architecture gate.

The global diagnostic inventory still fails on unrelated
custom-endpoint, archive, workspace and TTS entries. Restoring the task-owned rows
to their before-image equivalents on the same final external tree leaves exactly
the same global delta. No unrelated entries were refreshed.

Task and final integration reviews approved with no Critical/Important findings.
The final focus-token annotation cleanup passed scoped re-review; its runtime
body and executable AST are unchanged. TASK-3070.9 is Done, qualified by the
external gate failure above. Existing ADR-033 and the Wave 6 / DESIGN.md section 7 boundary apply;
no new ADR is needed. Evidence and exact diffs/hashes are retained in
`.superpowers/sdd/2026-09-10-task-3070-9-console-first-chat-controller/`.
At that checkpoint, auto-speak and final parent rebased-wave verification remained open. Existing test
infrastructure warnings and the six earlier host-blocked multiprocessing cases
remain outside this child; no full suite, provider call, host cleanup, staging or
commit was performed.

### Auto-speak ownership follow-up — 2026-09-11 (TASK-3070.10)

The destination resolver and auto-speak control/command targets now belong to
`ConsoleHandsFreeController` under the
[approved extraction plan](../../Docs/superpowers/plans/2026-09-11-task-3070-10-console-auto-speak-controller.md).
Three decorated Screen handlers stop their events and delegate once. Explicit
callables resolve the current coordinator and HandsFree owner at call time; a
named local callback in screen wiring retains the control-bar query and update.
Consent, queue ownership, retry/resume, worker scheduling and mount/unmount stay
with the existing coordinator.

The fresh baseline was **168 passed / one failure**, with ChatScreen at
17,209 lines / 584 methods against caps of 17,169 / 584. The extraction removes
**38 lines and two methods**, leaving **17,171 / 582**. The method cap is lowered
to 582; the line cap stays 17,169. The subsequent targeted run reports
**229 passed / one failure**: only the remaining **two-line overage**. The mounted
real-event/render test passes. No full-suite or live-provider validation is claimed.

The destination resolver AST is unchanged, and the projection body matches after
the `self`-to-`screen` substitution. All **94 diagnostic calls** and their sinks
across the three affected source files are conserved, requiring no inventory-row
update. The current global inventory mismatch is in Settings provider/view files;
that failure is separate from this extraction. The older hands-free Switch DOM
reach-through is also recorded separately in the findings table above.

All six mutations were caught: the three Screen delegates, resolver arguments,
factory exception propagation, and cancellation propagation. Identical-filename
Ruff comparison improves 153 inherited findings to 151 with no new finding;
compilation and scoped diff checks pass. Review found only formatting in three
owned ranges. That cleanup preserves the full executable AST of each file and
passes 17 focused cases; inherited formatting elsewhere remains unchanged.
Scoped re-review and final integration review approved with no remaining findings;
TASK-3070.10 is Done. Existing ADR-033 and Wave 6 / DESIGN.md section 7 apply;
no new ADR is required. Evidence is retained in
`.superpowers/sdd/2026-09-11-task-3070-10-console-auto-speak-controller/`.
Parent TASK-3070 and final rebased-wave closeout TASK-3070.11 remain open.

### Durable automatic-work ledger (TASK-32036)

Schema v17 adds immutable causal chains, atomic resource reservations, typed
wake attempts, and unique result claims. Normal run billing remains separate.
Only the first committed reservation or accepted attempt grants dispatch
permission. Proven pre-start aborts refund their own reservation; interrupted
or unknown work retains charges and requires review. FULL synchronization is
verified on reopened and alternate-thread connections, with rollback and exact
batch completion tests. The standalone v16 upgrade preserves legacy results
and unknown lineage without issuing new allowance.

Accepted manual sends use durable conversation identity and create a new chain
off the event loop. Children retain their spawning chain after newer manual
work. Plain wakes carry the same private identity; mixed chains are delivered
separately, and an earlier wake token cannot authorize a later delivery.
A real SQLite write lock leaves the Console responsive; a failed write dispatches
no provider call and releases stream ownership.

Independent review exposed token-overage admission across resource types, a
legacy-parent scope bypass, and monotonic elapsed-time loss across database
handles/replacement processes. Each has a failing regression and a verified fix.
Process-tagged clock anchors carry forward elapsed time without changing the
original deadline. Unknown clock state requires review; expiry stays durable.

Verification: **1,043 targeted DB/service/controller/fleet/provider tests passed**,
with **two loopback HTTP lifecycle tests passing separately** after the sandbox
blocked their local sockets. The final clock correction and its additional
regressions passed **60 focused ledger/migration/lineage tests** (overlapping the
combined run). New modules pass Ruff lint/format. Existing edited production
files introduce no lint findings, and their diagnostic-call ASTs are unchanged.
No full suite, external provider calls, or killed-process/power-loss certification
was performed. Runtime automatic call/child admission, acceptance fencing,
startup recovery, incremental notifications, scheduling, and pause UI remain
TASK-32037 supplied these runtime paths in the later enforcement pass below;
this earlier ledger checkpoint alone did not close the replay gap.


### Delivery/recovery design and startup log race (TASK-32020/32021/32488)

The log race is now deterministic: both callers observe the private directory
missing before either attempts creation. Before the fix, both direct creation
scenarios failed and one actual log writer lost its record. The losing creator
now reopens and validates the competing directory. File/symlink replacements
and a shared-writable intermediate directory still fail without changing the
replacement target. **135 targeted private-path and run-log tests passed.**
The new tests and edited private-path file pass formatting; the helper retains
one pre-existing Ruff import finding and adds none.
The final combined private-path, run-log, wake, approval-safety, and retry run
passed **163 tests**; the 135 private-path/run-log cases are included in that total.

Three explicit design probes also passed. A ready child waited through a
353 ms slow-sibling hold, and a second conversation produced no transcript
rows during a 350 ms real approval hold. After an injected stamp failure, a
durable-state claim increased provider calls from one to two for the same
result. [The baseline report](agent-fleet-delivery-baseline-2026-09-08.json)
contains the command, measurements, and limitations. This is local scripted
execution and in-process reconstruction, not a killed-process, power-loss,
or live-provider certification.

[ADR-135](../decisions/135-fleet-completion-delivery-and-crash-recovery.md)
and its linked plan complete the delivery and crash-policy design dependencies.
The implementation task criteria now include unique attempt claims, a required
acceptance fence before helper/main model or tool work, FULL-synchronized ledger
writes, and mark-independent recovery. Ordinary DB `synchronous=NORMAL` writes
can lose recent commits after an OS/power failure and cannot authorize automatic
execution. This is an implementation requirement; no DB setting changed here.

The guide and wake module now describe completion-time stamping accurately and
remove the false exactly-once and at-most-one-replay guarantees. AST comparison
confirms the wake module and its existing tests changed only in documentation.
The opt-in probe module passes Ruff lint/format. No full suite was run.

### Shared child admission (TASK-32035)

Four concurrently submitted conversations launch exactly six real child workers.
Automatic work is limited to four of those slots, preserving two for manual work.
Reservations precede handles, rows, and threads across spawn, continuation,
inline/skill dispatch, and old survivors. Terminalizing or pruning a handle does
not release its runtime slot while its thread/tool/model cleanup remains owned.

The new failed-start and retry tests exposed an additional mismatch: the service
refunded refused launches but the pure loop counted unnamed attempts as spent.
An internal `SpawnAdmissionRefusal` now preserves both counters without parsing
error copy or refunding a child that actually ran. Thread-constructor/start and
inline-model-start failures unwind ownership and the spawn allowance.

Verification: **451 targeted service, fleet-runtime, owner, tool, bridge, and
lifecycle tests passed**, plus **93 targeted admission, continuation, pure-loop,
and boundary tests**. New modules pass Ruff lint/format; existing changed files
add no lint findings. The two reindented multiline warnings preserve their AST,
level, arguments, and count; only the agent-service inventory digest was updated.
The prior unrelated TTS inventory drift remains outside this change. No full
suite or live-provider test was run.

Follow-up resolved in TASK-32488 above: the simultaneous cold-start probe logged
`PrivatePathError: operation_failed: FileExistsError` from secure sandbox-root
creation. `RunLogWriter.bind` disables that run's log after root resolution fails.
The isolated reproduction confirmed the mkdir race; the fix reopens and validates
the existing directory without weakening path checks. This was a separate log
availability failure, not a capacity failure.

### Physical operation ownership and tool admission (TASK-32034)

Real SQLite/service tests leave eight tool workers gated after their runs reach
`done`. Retries within those runs launch no worker; a ninth run is refused.
Six automatic workers leave two slots for manual submissions, across separate
conversations. A zero-timeout inline invocation also holds capacity.

The runtime retains model drivers through delayed cleanup after a bounded join,
including inline children. Model ownership survives bridge replacement and
runtime disposal; admission closes before shutdown drains work. Submit origin
is passed explicitly from the controller and inherited by children. Dynamic
limits apply to future admissions without cancelling occupied slots. The
snapshot contains counts, run/conversation identity, origin, and stopping state;
it contains no message bodies, tool arguments/results, or billing entries.

Verification: **382 targeted service, bridge, continuation, runtime, headless,
and wake-safety tests passed**, followed by **145 targeted ownership, worker,
fleet-runtime, lifecycle, boundary, and headless-dispatch tests** after final
integration. The latter includes one previously counted headless test. Another
**40 targeted controller tests passed**. New files
pass Ruff lint/format; edited production files add no lint findings relative to
the starting checkout. Scoped whitespace checks pass. The repository-wide
diagnostic gate is red only for `Event_Handlers/TTS_Events/tts_events.py`; all
five edited production files preserve their starting diagnostic/sink entries,
and the new capacity module has none. That unrelated TTS drift was not rewritten.
No full suite or live provider was run.

The timed-out-worker baseline probe now asserts enforcement (one worker remains
alive after four attempts from one run). The JSON baseline below remains the
historical pre-change measurement. At this checkpoint child admission was pending; TASK-32035 above now supplies
it. TASK-32037 now supplies cumulative automatic-wake budgets.

### Aggregate admission design and additional fixture repair

At the design checkpoint, four deterministic baseline probes recorded the missing bounds;
they do **not** assert that the proposed limits are enforced. Reproducible
commands, measured counts, and scope limitations are in
[the baseline report](agent-fleet-budget-baseline-2026-09-08.json). Gates are
released and worker threads joined before DB cleanup. The successive-wake
probe injects completions through the real coordinator/controller; it is not
a live-model recursive-spawn claim.

The wider contract run initially passed 160 tests and failed one headless
dispatch test before its manual send was accepted (`binding_unavailable`).
Applying only the existing legacy-session fixture diagnostically restored it.
The test now selects that setup explicitly for its recording bridge; its
manual/wake dispatch, cancellation, and budget assertions are unchanged.
The final combined targeted run passes **168 tests**, including the four new
probes and all 16 headless/safety checks. The new probe module passes Ruff
lint/format; the existing headless module adds no lint findings and its changed
range is formatted. Scoped whitespace checks pass. No full suite or live
provider was run. Production settings,
project-instruction preflight, and execution behavior are unchanged in this pass.

[ADR-134](../decisions/134-fleet-admission-and-automatic-work-budgets.md) defines
six child slots with two reserved for manual work, eight tool slots with two
manual reserves, and a shared automatic chain allowance of three wakes, six
child launches, 32 model calls, 500,000 budget tokens, and 900 elapsed seconds.
It also covers unknown usage, active reservations, old survivors, restarts,
mixed-chain completions, and visible exhaustion. The eight tool slots and two manual reserves are now effective under TASK-32034.
The child limits are effective under TASK-32035. At this design checkpoint automatic-chain limits
remain accepted design defaults. Both admission paths use actual ownership,
including cleanup still running after a bounded join.

### Durable accounting repair pass

**656 targeted tests passed** across three disjoint groups: 282 DB/service/
runtime/continuation/coordinator tests, 288 bridge/usage-reattachment/cost/lifeline
tests, and 86 fleet/agent/cost/steering UI tests. Strengthened continuation-detail
checks then passed **2 tests**, proving painted accounting survives real handle
pruning and a new primary record while the live billing feed falls to zero.
Those two are part of the UI group, not additional unique tests.

The initial regressions reproduced missing persisted counters and absent UI
accounting. Coverage includes legacy migration and reopen, known zero versus
unknown, late cancellation accounting without status overwrite, repeated-outcome
idempotence, invalid counters, cache-weighted and estimated service usage,
unknown exception outcomes, sibling/foreign ancestry exclusion, cycles, missing
parents, and sums above SQLite's integer accumulator range. Compositor checks
verify actual per-run and complete/partial chain text. No live provider or full
suite was run; the existing request-library dependency warning remains.

New files pass Ruff lint/format. Changed-range formatting, scoped whitespace,
and the persistent diagnostic inventory check pass. The six existing source/test
files checked against this pass's starting working tree add no lint findings.
See [ADR-131](../decisions/131-durable-agent-budget-accounting.md): these are
run-budget counters that may include cache weighting and estimates, not raw
provider usage or currency. Billing and the panel's existing latest-run scope
were unchanged in that pass. Historical elapsed/result detail and navigation
are addressed in the later fleet UI repair pass below.

### Fleet historical detail and navigation repair pass

**398 targeted tests passed** in four disjoint groups: 291 database/bridge
tests, 62 fleet/history/agent UI tests, 15 historical-detail/parallel-run
UI checks, and 30 Inspector component/CSS build checks. This is not a
full-suite or live-provider result.

New regressions cover bounded metadata pages without hydrating steps/results,
same-timestamp cursor ordering and concurrent inserts, conversation/kind scope,
superseded history, malformed limits/cursors, bounded task excerpts, approximate
terminal timestamps, child-owned saved detail, and single-pass truncation.
Rendered Console checks at 180x48 and 120x35 exercise expansion, the four-row
preview, page navigation, actual keyboard/mouse selection of an older child,
safe dismissal during loading, error/empty recovery controls, and selection
guards after a conversation switch. An additional regression reproduced the
hidden history entry after a childless primary; the selected conversation now
keeps its read-only history entry even with no current preview rows.

The first wider UI run found a stale five-glyph summary expectation, updated
to assert complete counts and retained rows under the new bounded summary.
It also found the pre-existing TASK-3070 screen-size failure: 17,600 lines at
pass start versus a 17,570 ceiling; now 17,601 lines and the same 591 methods.
That architecture gate is still red and its budget was not raised.

New files pass Ruff lint/format. Changed ranges are formatted, the six existing
production files add no lint findings relative to this pass's starting tree,
scoped whitespace checks pass, and the persistent diagnostic inventory remains
unchanged and valid. ADR-132 records the metadata/navigation contract. Saved
timestamps are explicitly approximate because last-update bookkeeping can
extend them; this pass does not claim immutable completion timestamps.

### Lifecycle and identity repair pass

**710 targeted tests passed** across three disjoint groups: 471 gateway/bridge/
lifeline tests, 195 identity/continuation/coordinator/runtime tests, and 44
teardown/close-session/runtime-lifetime/fanout/stop tests. Two of the 471 initially
could not bind their localhost server inside the sandbox; both passed when
rerun with the required localhost permission. These were fixture permission
errors, not assertion failures. No real provider or full suite was run.

The new tests first reproduced an open pool after real child completion and
incorrect post-prune refusals. Further regression coverage checks pending-call
draining before closure, same-loop cleanup, repeated shutdown, delayed cleanup
after the bounded join, cleanup failure, failed thread start, payload-free
identity bounds, ID collision ordering, and foreign-conversation isolation.
Self-review caught an idle-loop gap in the initial cleanup shape; a failing
regression showed `[open, closed]` at successive driver stops. Cleanup now
completes before the loop first stops, removing the competing-sweep window.

New modules pass Ruff lint/format; the four changed production modules add no
lint findings relative to the working tree at the start of this pass. The
diagnostic inventory change is limited to one constant cleanup-failure warning
in `console_agent_bridge.py` and its aggregate count (no new diagnostic payload).

### Mailbox and wake repair pass

After repairs, the combined targeted run passed **295 tests** in 101.23 seconds.
It covered fleet reliability, coordinator, mailbox, send-to-agent, continuation,
runtime, stop semantics, bridge steering, retry scheduling, wake safety,
staleness, view marks, autowake, close-session handling, fanout, wake ledger,
and steering UI. The final defensive-copy refusal check and all affected
coordinator/continuation/mailbox tests then passed **84 tests**. Strengthened
compositor assertions for both terminal recovery states and the full-queue
refusal passed **3 tests**, checking the actual painted words within each
widget's region. These counts overlap and are not additive.

New regression modules and the repaired safety module pass Ruff lint; the new
modules pass formatting. Changed-range formatting and scoped `git diff --check`
pass. Existing lint findings in legacy files and the unrelated in-flight
Console extraction were preserved; this is not a whole-repository lint claim.
The persistent diagnostic inventory check also passes. Each behavioral repair
was preceded by a failing regression. No full suite or real provider was run.

### Baseline and expanded-review evidence

Initial targeted mailbox/runtime/continuation/UI run: **229 passed**.
Wake/ledger/close/fanout run: **42 passed, 3 failed** (TASK-32485). Clean HEAD
reproduced those three failures; supplying the missing legacy fixture yielded
**3 passed** without production changes.

Expanded run-log isolation, task-store, fanout, usage, wake-staleness,
view-mark, and cancel-all run: **221 passed, 6 failed**. All six failures occur
while constructing a multiprocessing semaphore, before the target behavior.
A standalone `multiprocessing.get_context('spawn').Lock()` reproduced
`OSError: [Errno 28] No space left on device`. This is an environment limitation,
not evidence against the task-store implementation. No full suite or real
provider was invoked. Request-library dependency warnings also remain.

## Remaining work

The correctness repairs and approved admission, accounting, delivery, recovery,
and relay-first messaging scope are implemented. The current integration adds
schema versions 16–18 while preserving dev's indexed steps, spawn identities,
and activity receipts. Colliding task IDs were reassigned to TASK-32483–32492;
canonical upstream TASK-3070 records and controller ownership were retained.

PR #2631 merged into dev as `8ab21ecaf3` on 2026-09-12 UTC, after final
independent review, resolution of all four Qodo findings, and fresh required CI
on head `a8d548ecf8`. The [remaining-work inventory](agent-orchestration-followups-2026-09-12.md)
tracks older orchestration tickets outside this completed 25-item audit scope.
TASK-32493 is Done: the existing PR fast lane passed all 1,125 cases on a clean
runner, including the twelve process cases blocked by this host's semaphore
allocation. The required derived-artifact job also passed on that head. Existing Console architecture debt remains upstream; no ratchet cap was
raised. The current integration status records exact checks and measurements.

Direct peer addressing, progress-triggered wakes, and durable progress inboxes
remain optional design work outside [ADR-136](../decisions/136-scoped-child-progress-and-supervisor-relay.md).
The implemented channels are process-local bounded steering and progress,
explicit supervisor collection/relay, and shared versioned session tasks.
No full repository suite, live provider call, or power-loss certification is claimed.
