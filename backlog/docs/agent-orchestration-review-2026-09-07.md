# Agent orchestration review and repair ledger — 2026-09-07

Reviewed working tree based on `bc745f854`, including the in-flight Console
controller extraction. This is a scoped engineering review, not an exhaustive
security audit or a live-provider certification.

## Findings and disposition

| ID | Finding | Evidence | Disposition |
| --- | --- | --- | --- |
| TASK-32013 | Steering admission has no aggregate bound; retention excludes unread steering from its size check. | 100 valid 4,000-character messages retained a 401,471-character payload under the default 200,000-character cap. | Fixed: 32-entry / 64,000-character queue, complete payload sizing, explicit refusal with preserved draft. |
| TASK-32014 | Accepted final-turn steering can remain unread while the queued indicator disappears. | Real loop/bridge probe: one model call, accepted steering unseen by model, terminal queue zero, one retained unread entry. | Fixed: terminal unread count and current continuation availability; explicit continuation preserved. |
| TASK-32015 | Wake safety tests omit their helper module's legacy-session fixture. | Three original failures and one further headless dispatch test fail at binding preflight; supplying legacy-session setup restores the original assertions. | Fixed: fixture imported in safety tests; headless dispatch explicitly selects legacy setup for its recording bridge. No production authority change. |
| TASK-32016 | Coordinator lifecycle events accumulate without a production consumer. | 100 reserve/finish/prune cycles leave zero handles but 200 events. Source search finds no production `drain_events` caller. | Fixed: events expire with pruned handles; live survivor events remain ordered. |
| TASK-32017 | Retained transcript snapshots share nested objects with callers. | Mutating a returned native `tool_calls[0].id` changes the subsequent stored snapshot. | Fixed: deep copies on admission and read; failed copies refuse retention before claiming steering. |
| TASK-32018 | Refused wakes retry immediately and can starve other conversations. | Real controller: unavailable provider resolved 183 times in 150 ms. Two-session probe: first refused 143 times, ready second never attempted. | Fixed: one-second retry delay per refused conversation; ready conversations proceed. |
| TASK-18312 | Pruned child handle IDs lose their useful terminal refusal; same-process run IDs can be called an earlier session. | Original resolution ladder consulted live handles, retained transcripts, unpruned handles, and DB run IDs. | Fixed: bounded payload-free terminal identities, accurate status/no-retention refusal, honest DB-only and expired-handle copy. |
| TASK-18311 | Per-child spend disappears at prune; original and continued runs have no durable aggregate. | Original handle-only budget rollup and continuation characterization reproduced the loss. | Fixed: schema v13 saves nullable budget counters; historical rows and selected continuation ancestry show known/partial totals without refeeding billing. |
| TASK-15665 | Live provider client pools lack deterministic closure after teardown. | Real bridge reproduction: child completes after parent teardown and closes its loop, but its HTTP client remains open. | Fixed: primary and child lifelines drain pending work and close their own pool before stopping the loop; injected clients remain caller-owned. |
| TASK-15200 | Historical fleet rows lose elapsed/result detail and stuck/cancelled status styling. | Real DB/bridge and rendered Console regressions reproduced missing child-owned detail and timestamp spans. | Fixed: shared historical projection restores bounded saved detail and budget counts; valid terminal timestamps show an explicitly approximate span; warning/muted status colors. |
| TASK-15201 | Long fleets lack a full-history action and expansion does not reveal the controls. | Compositor checks reproduced nested-scroll clipping; a childless later primary also hid the history entry. | Fixed: four-row preview with complete counts, expansion reveals the action, and a read-only 50-row paged picker reaches earlier/superseded children even when the current preview is empty. |
| TASK-32019 | Concurrency and spend caps are per conversation/run, not global or wake-chain bounds. | Four conversations sustain eight gated children; six successive wakes complete without a user send. A terminalized child can still occupy a worker. | Design complete under ADR-134 with user-selected conservative defaults. Operation ownership and shared tool/child admission implemented in TASK-32034/32035; durable chains plus runtime enforcement are implemented in TASK-32036/32037. |
| TASK-32020 | Fast-child results wait for the slowest sibling; a blocked wake holds global wake serialization. | Two-child and real approval probes both hold ready delivery for the full 350 ms gate, despite unused primary capacity. | Design complete in ADR-135: individual notifications, fixed batching window, two bounded wake slots with manual reserve and round-robin fairness. Implemented in TASK-32037; real gated-sibling and concurrent-approval tests pass. |
| TASK-32021 | A crash or failed stamp after wake work can replay it; missing attention marks can also strand automatic delivery. | A finished wake with an injected stamp failure triggers a second provider call when durable state is claimed. Source confirms marks are the discovery index. | Design complete in ADR-135: atomic attempt claims, required acceptance fence, FULL-synchronized ledger mutations, and conservative manual review. False exactly-once/one-extra-replay claims corrected. Durable claims and attempt transitions are implemented in TASK-32036; runtime acceptance and owner revocation are implemented in TASK-32037. |
| TASK-32022 | No child-to-parent progress messages or direct sibling messages. | Fleet tools are structurally primary-only; children share a versioned task store instead. | Capability proposal, not a failed implementation. Proposed ADR-136 specifies bounded child reports and explicit supervisor collection/relay; design review is pending and no messaging runtime was added. |
| TASK-32034 | Timed-out tool workers remain active with no shared admission bound. | Four sequential timeout returns leave four gated tool workers alive. Caller timeout and worker completion are different events. | Fixed: app-owned capacity follows worker/driver completion, including delayed cleanup. Eight tool slots, two manual reserves, and per-run refusal while a timed-out worker remains alive. |
| TASK-32035 | Per-conversation handles allow excess aggregate children and can free logical capacity before worker cleanup. | Four conversations previously held eight children; gated terminal-child tests retain physical work after row completion. Failed-start tests also exposed the pure loop's unrefunded unnamed-spawn counter. | Fixed: six shared child slots, two manual reserves, ownership through cleanup, and typed pre-dispatch refusals that spend no spawn allowance. |
| TASK-32036 | Automatic generations lack durable shared allowance and causal chain identity. | Real SQLite tests cover concurrent last-slot admissions, interrupted attempt claims, mixed-lineage wakes, token overruns, and clock/handle recovery. | Implemented: schema v14, FULL-synchronized ledger APIs, immutable manual/descendant lineage, exact live wake authorization, and separated chain batches. Dispatch enforcement and schema v15 runtime-owner revocation are supplied by TASK-32037. |
| TASK-32037 | Automatic-chain limits and durable attempts were not enforced at runtime. | Real dispatch, startup, approval, cancellation, owner-replacement, and rendered pause regressions. | Implemented: shared finite budgets, acceptance/owner fences, incremental fair delivery, badge-independent recovery, and explicit saved-result pauses. |
| TASK-32039 (C1) | Concurrent cold starts can disable a run log after a private-data-directory creation race. | Barrier-controlled real opens reproduce `FileExistsError` and an inactive writer through config data-root resolution. | Fixed: reopen the competing entry without following links, then retain owner/type/mode validation. Both real writers now persist their records. |

An adjacent architecture check remains open under existing TASK-3070: the
Console screen exceeded its 17,570-line ceiling before this UI pass (17,600
lines / 591 methods). The history callback adds one line and no methods;
the ceiling remains unchanged. This is recorded separately from fleet
behavior and is not a claim of a fully green architecture gate.

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
Startup discovers results without badge authority. Schema v15 adds a durable
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

### Messaging design and current-behavior check (TASK-32022)

The proposed ADR-136 and spec have been self-reviewed for authority, bounds,
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

### Durable automatic-work ledger (TASK-32036)

Schema v14 adds immutable causal chains, atomic resource reservations, typed
wake attempts, and unique result claims. Normal run billing remains separate.
Only the first committed reservation or accepted attempt grants dispatch
permission. Proven pre-start aborts refund their own reservation; interrupted
or unknown work retains charges and requires review. FULL synchronization is
verified on reopened and alternate-thread connections, with rollback and exact
batch completion tests. The standalone v13 upgrade preserves legacy results
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


### Delivery/recovery design and startup log race (TASK-32020/32021/32039)

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

Follow-up resolved in TASK-32039 above: the simultaneous cold-start probe logged
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
Wake/ledger/close/fanout run: **42 passed, 3 failed** (TASK-32015). Clean HEAD
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

The listed correctness repairs and the approved admission, budget, delivery, and
recovery changes are implemented in the working tree. TASK-32019/32020/32021 are
completed decisions; TASK-32034/32035/32036/32037 provide their runtime behavior.
TASK-32039 also closes the confirmed cold-start log-directory race.

**TASK-32022 has a concrete proposal awaiting design review:**
[ADR-136](../decisions/136-scoped-child-progress-and-supervisor-relay.md) and its
[spec](../../Docs/superpowers/specs/2026-09-08-scoped-agent-messaging-design.md)
recommend bounded child progress reports, explicit supervisor collection, and
relay through existing steering. The proposed first version adds no direct peer
addressing, progress-triggered wake, or durable inbox. It specifies identity,
permissions, backpressure, lifecycle, honest receipts, and a future acceptance
matrix. No runtime behavior has been added for this proposal.

The existing supervisor/user-to-child steering mailbox is implemented and
bounded; it is not a general bidirectional message bus. Shared versioned task
state remains the current child coordination mechanism. This design pass also
corrected stale ADR-129/134/135 status text about implementation pending and the
old one-global-wake policy.

Task files carry acceptance criteria and implementation notes. The verification
limits above remain explicit; the whole test suite was not requested.
