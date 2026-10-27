# Lessons: Console screen-to-runtime wiring

Working knowledge about wiring ChatScreen callables into the Console
controller/store runtime. Not decisions (see `backlog/decisions/`) — these
are traps that have actually cost time here, kept so the next person does
not rediscover them.

**Every entry states the incident that produced it.** A lesson without its
evidence decays into folklore, and folklore gets ignored. If you add one,
bring the incident.

---

## Moving a stale-scope check onto the worker leaves the hop to the UI loop open (TASK-33620.15.1.6, 2026-10-27)

**Incident.** The Console Environment panel's `_land` used to read the
workspace root on the UI loop and drop a landing gathered for another root.
To take that consent read (a registry SELECT) off the loop, the gather worker
re-read the root after gathering and passed it along. Review reproduced the
gap on the head in two probes: the root changed from `/w/A` to `/w/B`, or
consent was revoked, after the worker's re-read and before the landing ran
on the UI loop. The old root painted, and its deferred `gh` fetch was issued
for a folder that was no longer the conversation's. Both probes passed on
origin/dev, whose read was in `_land` itself. The task's own notes conceded
the edge; the AC that promised the guard still held was ticked anyway.

**What to do.** A check made on a worker and an action taken on the UI loop
are not atomic, however late the worker checks. Carry a cheap in-memory
stamp of the check's inputs (here the session's workspace, the registry's
`mutation_generation` and the consent service's `admission_revision`), take
it on the worker *before* the re-read, and compare it at the action on the
loop. Test the hop itself: block the worker inside `marshal_to_ui`, change
the state on the loop, then release, and assert both that nothing stale
painted and that the loop made no database read.

---

## A warning-free batch can still retain worker-owned database files

**TASK-31245 / TASK-33620.9, 2026-10-04.** After explicit Console fixture
retirement, a 144-case affected batch passed without warnings but a read-only
post-teardown census still counted 75 workspace SQLite descriptors plus WAL/SHM
files. Constructor owners had been captured correctly; `WorkspaceDB.close()`
only closes the calling thread's cache. Allocation stacks identified finite
scope reads and a saved-chat membership projection retry opening their own
executor caches. Real file-backed REDs asserted that each callback left one
extra registered handle, including the projection's retryable-failure path.

Use the installed operation-owned boundary **inside** the finite callback's
thread. Preserve pre-existing caller caches, memory/custom owners and work still
running after await cancellation. Do not close foreign-thread handles from a
fixture or lower the warning threshold. The corrected projection test and
filtered mounted rename leave no workspace files after teardown without GC or
observer cleanup. A passing warning sentinel proves only that its threshold was
not exceeded; inspect ownership directly before claiming terminal retirement.

**TASK-34411, 2026-10-05.** A parked real character-metadata callback entered
the finite guard with a borrowed worker handle. Exact-file quiescence closed
that object, then resumed acquisition; the callback reopened a different handle.
After physical completion it remained executable and registered because cleanup
remembered only the entry-time borrowed flag. Cold entry passed, warm entry
failed; installed AgentRuns, Workspace and Collections replacement controls
failed on success and SQL error while unchanged active transactions stayed live.
Capture the exact original native identity: preserve it if still current, but
retire a replacement acquired within the completed operation. Do not infer that
an object is caller-owned merely because its predecessor was borrowed, or claim
this deterministic defect explains an intermittent failure not yet captured.

**PR3024, 2026-10-06.** After test-owner adoption, a complete runtime batch
still retained runs.db leases. A forwarding native-allocation trace identified
exited executor owners with no active operations at the run-log availability
probe's shared parent-run metadata lookup. The adjacent target resolver already
retired finite reads, but all three log readers bypassed that boundary. Six real
cold success/error controls failed while six unchanged warm transactions passed.
Guarding the shared lookup with the installed helper clears the complete 101-case
strict resource gate without adding cross-thread fixture closes or a new cache policy.
Trace the first acquisition in every sibling caller; a later guarded read may
correctly regard an already leaked handle as borrowed.

The initial observer covered constructor profiles, not every test `tmp_path`.
Adding the current run's explicit temporary root exposed Chat handles that the
first filter could not see. Local quiescence then retired them, but the shared
fixture called runtime disposal again: `detach_view` refreshed attention/local
marks and reopened the same file. The strict 152-case run passed every body but
exited 1 with eight Chat SQLite handles. Register exact test-owned databases for
terminal retirement **after the fixture's last successful disposal**; do not
treat an early `_disposed` admission latch as a completion receipt. Keep the
observer read-only and distinguish successful test bodies from its process gate.

Saved-sidebar construction separately admitted an unmounted save timer because
ordinary reactive assignment treated hydration as an edit. Two restored-state
REDs caught the timer; watcher-free `set_reactive` removed only that admission.
Retain the mounted user-edit/debounce/quit-flush checks when changing hydration.

---

## Awaiting a coalesced UI sync does not prove exact transcript publication

**TASK-31245, 2026-10-04, dev f800952214.** Warm History-to-Character reopening
intermittently left the switcher open despite selecting an existing runtime.
State probes captured the target active and composer focused while the
transcript still owned the preceding runtime, with both Console-sync flags
set. The full sync had returned after coalescing into an in-flight pass; the
canonical opener's strict readiness check correctly rejected the stale
transcript and rolled back. Individual retries passed, and reconciliation
could clear the failure before the test's five-second wait expired.

The deterministic installed regression holds the full sync before transcript
publication and opens the existing exact target. It returned postcommit
`FAILED` before repair and `OPENED` after awaiting the existing locked transcript
renderer before the readiness proof. A queued refresh request is not a render
completion receipt. Activation must await the publication it promises, without
weakening identity, visibility, focus, or overlay-owner checks. Capture the
typed outcome and transcript owner at the failure boundary; a later timeout
alone cannot distinguish a rejected activation from a slow successful one.

Qodo's PR #3009 review exposed a second seam in this repair: direct cancellation
of the postcommit child during publication stranded a newly hydrated runtime
before its ownership token was returned. A real-store regression failed for
the cold case and passed for the warm case. Shielding a caller is not shielding
the child itself. An opener adding a suspension after acquiring ownership must
settle that exact ownership on cancellation before propagating it; the repair
uses the existing synchronous exact-instance guard, retaining warm sessions and
letting the caller restore prior UI. Both cases then passed.

---

## Dispatch-recovery Retry replays the accepted turn; a setting change cannot reach it

**TASK-34100.5, 2026-10-03.** A durable send refused before dispatch (it could
not fit the model) keeps its fail-closed dispatch owner, and the first fix made
that owner's Retry re-enable once the session's model or reply limit changed:
"Settings changed — Retry sends it again." A unit test with a real store and
controller passed. Live, on a fresh profile with OpenAI's template default
`gpt-5.6-terra`, Alt+M to gpt-4.1 then Retry failed again — under the *old*
model. `retry_dispatch_recovery` resumes the frozen durable continuation: its
provider, model, reply limit and attached context window were fixed when the
turn was accepted. Resend on the user message re-plans with current settings and
got a reply. The card now keeps Retry disabled and says "Change a setting above,
then Discard and Resend the message."

Before offering Retry as the fix for a refusal, check what the retried turn is
built from. A test that only asserts the action is *enabled* proves nothing about
whether pressing it can change the outcome — drive the retry and read what it
sent.

## An unchanged recovery projection can still need its click latch released

**TASK-32819, GitHub #2708, 2026-09-18.** A failed recovery returned the store to
the same actionable state before the UI painted its in-flight projection.
`sync_recovery()` skipped identical display state before clearing the widget's
local click latch, leaving Retry and Discard enabled-looking but inert. A mounted
restored-conversation test with a real SQLite settlement failure reproduced it;
after the database failure was removed, another Discard still did nothing.

Release the widget's pending click explicitly when its action completes, then
refresh after exceptional or cancelled actions as well as successful returns.
Qodo review reproduced the opposite race in the first fix: an unrelated repaint
of unchanged model state released a click before its worker had claimed the
action. Use a per-click completion token so a stale worker cannot release a
newer click or owner. Preserve the action's original exception if repainting also
fails. Test a failed attempt followed by a second click through the real
dispatcher; store-only assertions cannot detect a stranded widget latch.

The dev rebase also exposed stale harness assumptions: full-app tests must use
`private_profile_test` once config sources are lifetime-bound, and controller-only
gateway doubles must not replace the runtime's UI metadata gateway. Wait for the
actual Queue label rather than an already-disabled empty composer. Pure widget
projection tests should override the full-app catalog fixture, as other isolated
UI harnesses do, so running them alone does not import and bind the application.

## A `console_view_hooks()` entry with no declared slot is silently inert

**TASK-32482 (tasks 6 and 7), 2026-09-11.** The chat-fork feature wired two
new screen→controller callables — `set_pending_chat_create` (arming the
confirm card) and `complete_agent_chat_create` (landing the created chat as
a session) — and in both tasks the wiring passed its own new tests while
doing nothing in production. The cause: `_bind_view_hooks`
(`tldw_chatbook/Chat/console_runtime.py`) does not copy whatever the view's
`console_view_hooks()` map returns. It iterates the declared
`CONSOLE_VIEW_HOOK_SLOTS` tuple and sets
`hooks.get(slot.name, slot.viewless_default)` — an entry the map provides
that has no matching declared slot is silently dropped, leaving the
controller attribute at its (inert or fail-closed) viewless default. The
guard is exhaustive in both directions:
`test_attach_and_detach_cover_exactly_the_same_slot_set`
(`Tests/UI/test_console_runtime_ownership.py`) asserts set equality between
what `ChatScreen.console_view_hooks()` provides and what the slot list
declares — so the omission is caught, but only if that test runs.

**What to do.** When adding any kwargs entry to
`ChatScreen.console_view_hooks()`, add the matching `ConsoleViewHookSlot`
entry to `CONSOLE_VIEW_HOOK_SLOTS` in the same change, with a `why` naming
the production read site that makes the chosen `viewless_default` correct
(`None` only where the read site's own guard makes it inert or fail-closed;
an explicit callable everywhere else). Then run the slot-set test —
`.venv/bin/python -m pytest Tests/UI/test_console_runtime_ownership.py` —
before trusting any new hook wiring, however green its feature tests are:
they exercise the callable directly and never go through the bind.

---

## Test endpoint creation with the real settings rebaser and queued adapters

**TASK-32566, 2026-09-16.** An isolated parent/child modal test found the named
endpoint probe's family/identity mismatch, but the production controller then
exposed two more failures: rebasing normalized the entry's hyphenated slug, and
a delayed nonblank `Select.Changed` event restored the old model after a successful
single-model listing. Mirrored input events also cancelled the automatic probe.
The mounted controller test plus explicit queued adapter events caught these;
filtering against the control's current value before cancelling work preserved
the actual current selection and evidence. Keep the real rebaser in flow tests,
and treat queued widget echoes as potentially stale even when their value is
nonblank. Coverage: `Tests/UI/test_console_endpoint_discovery.py`.

## Credential polling must compare with the last rendered state

**TASK-32711, 2026-09-17.** The first persona readiness poll initialized its
remembered block reason to `None`. The inspector painted a pending credential
read, but if the worker completed before the first timer tick, the ready reason
was also `None` and the poll skipped repainting. The UI stayed pending despite
completed work. Recording the state when rendering fixed the race; the mounted
regression holds back the first poll until the credential worker has finished.
Settings separately forces an initial projection. A second deterministic case
completed the read between rendering the header and inspector: remembering only
the inspector left the header blocked. Track both rendered projections so the
next poll reconciles them. Compare credential status as well as completion: an
expiry can happen inside the cache TTL without a new completion revision.
Coverage: `Tests/UI/test_personas_subscription_readiness.py` and
`Tests/UI/test_settings_subscription_readiness.py`.

## Start the transcript sync timer only after an active run status is set

**TASK-33661, 2026-10-01.** Resend on a refused echo ran in a `console-run-*`
worker that, like Retry, called `_start_console_transcript_sync_timer()` first and
then sent the echo through `_dispatch_console_draft_send`. In two live runs the
app log showed the resent turn completing while the transcript stayed on
"Generating…" and Send still read Queue. The poll stops itself when the viewed
session is not active. The session still read blocked from the refusal while the
send path awaited hook admission, so the poll could stop before the turn ran.
The start that the send path makes at custody returns early while a timer object
exists. The control (the shelf's Restore, then Enter) never froze, because that
path starts nothing before custody. Leaving the timer to the send path fixed both
live runs. The mounted pilot tests passed before the fix: they poll
`_visible_text` themselves, so a stopped timer is invisible to them.

**What to do:** start the sync timer only in code that sets an active run status
before its first await, such as Retry, Continue or a controller re-run. When
routing through the normal send path, let that path start the timer. To check a
poll timer, read a live capture after the log reports the turn complete; a pilot
test cannot catch a stopped timer.

## A lock that looks redundant may be the barrier something else promises on

**TASK-32801.4, 2026-09-18.** The core review found `_capture_quiescence_lock`
held across the persistence adapter's own `BEGIN IMMEDIATE`, inverting against
`_dispatch_branch_mutation`, and recommended narrowing it to the in-memory
merge. Narrowing alone would have been a data bug. `begin_capture_quiescence`
never waits on anything explicitly: it inherits its guarantee from the lock, so
blocking on it means no exchange write is mid-flight, and
`commit_full_capture_purge` deletes rows on that promise. Narrow the lock and a
writer that passed the quiescence check before the fence armed re-adds rows the
purge has just deleted -- a stall traded for silent capture resurrection. The
sibling `_trajectory_lock` genuinely was redundant (the DB assigns `seq` inside
its own transaction and the adapter retries the loser) and was deleted; the
fence lock instead became explicit -- admit and count the writer under the lock,
write outside it, and have the fence arm first and then drain the admitted
writers on a condition. Before deleting or narrowing a lock, find every caller
that treats *acquiring* it as proof of quiescence; a barrier inherited from
mutual exclusion has no other name in the code to grep for.

## A suspended Chat settings draft restores its edits along two paths

**TASK-33003.5 fix rounds 1 and 2, 2026-09-29.** The unsaved-edits guard decided which
fields "opened already changed" by comparing the modal's opening `settings`
with the chat's committed settings. The review read modal `__init__`, saw
`settings = suspended_draft.settings`, and concluded that a credential
round-trip's edits would count. They did not. `capture_suspended_draft` stores
`settings=self._settings`, the settings the first modal *opened* with (equal
to the committed settings for a plain open). The edits travel only in
`raw_values` and the provider drafts, so the reopened modal's `settings` matched
the committed settings and Esc dropped the restored edit silently.

Round 1 then fixed it by reading the controls just before
`_restore_suspended_draft` and treating those as the unedited values. That held
for Temperature, the only field its test edited, and nothing else. There are
**two** restore paths. `_restore_suspended_draft` writes the generation Inputs
after compose, but `__init__` applies Provider (`_active_provider` from
`raw_values`), Model and Endpoint (the provider drafts) and Streaming
(`_streaming_draft`) *before* compose, so those controls are born edited. The
round-2 review's probe showed Model, Endpoint, Streaming and a provider switch
all reopening with `labels=()`. **What to do:** do not reconstruct a first
modal's derived state from the reopened modal. Carry it in the snapshot (the
guard's baseline is now `ConsoleSettingsDraftSnapshot.unsaved_baseline`). A
round-trip test must edit at least one field from each restore path; the
parametrized `test_suspended_draft_round_trip_keeps_its_edits_unsaved` and the
production-router `test_credential_round_trip_keeps_the_restored_edit_unsaved`
in `Tests/UI/test_console_settings_unsaved_guard.py` do.

## A provider failure leaves Console by two exits; classify it at both

**TASK-32369, 2026-10-03.** task-32342 stopped a status-less
`ChatConfigurationError` (a request that never reached the provider) from being
reported as a provider outage. The fix sat in `ConsoleProviderGateway`'s
non-stream path, which re-raises such an error untouched. `stream_chat` has a
second exit: its worker thread turns any exception into a queue error item, and
the consumer raised `ChatProviderError(item.text, status_code=... or 502)` for
every item. So the same local failure, streamed, still read "provider returned
HTTP 502". Nothing pinned the stream path, so this went unnoticed for three
weeks. It turned up only because TASK-32369 drove a real handler through
`stream_chat` instead of calling `safe_provider_error_copy` directly.
**What to do:** a change to how Console classifies or words a provider failure
must cover both the non-stream re-raise and the stream queue consumer
(`_QueueItem.error` and its `item.kind == "error"` branch). Its test should drive
the real failure through `stream_chat`, which is the path Console sends on.

## A parent descriptor census does not survey private test children

**TASK-31966 Model-row verification, 2026-10-04.** A parent-only census observed
no retained database files after private-profile cases, but loading the same
read-only observer in the children made all six mounted Model processes fail
retirement on harness-owned Library/Workspace SQLite files even while every
body passed. Those cases had not imported the existing opt-in ownership fixture;
the new cases also called a factory binding in an imported helper, outside its
capture. Reusing the module's factory and `owned_console_apps` retired the actual
owners after harness/workers stopped. The final 15-test run passed with required
child and parent gates; each mounted child census was empty. Activate resource
observers in the process that executes the app (`PYTEST_PLUGINS` survives the
private helper's disabled autoload), inspect child receipts, and do not promote
a clean parent census into child or terminal application-lifetime evidence.

## Pilot's whole-screen barrier can distort a measured activation

**TASK-31966, 2026-10-04, clean ac3a4f1.** A bounded native trigger trace found
a 73.944541ms gen-2 pause inside `Pilot._wait_for_screen` after Enter delivery;
the barrier queued up to 751 descendant callbacks. A real mounted regression
received the correct Input submission but failed on four barrier registrations.
Measured Enter now uses the same installed `App._press_keys` dispatch (including
its native idle/animation waits), then the existing real activation/readiness
and modal-removal gates, without that artificial fan-out. The 48 affected tests
pass; fresh source-bound latency qualification is still required. Trace the
driver as well as the app before choosing a production remedy. Never subtract
observer cost from failed timings, weaken real settlement, or confuse an
allocating trigger with ownership of the heap a collection scans.

At clean 81c0d22, the corrected driver still made 3–18 benchmark full-modal
captures per activation. A trigger trace found gen-2 allocation inside an extra
`render_strips()` capture. A real mounted RED proved a later native repaint
increased full captures from one to two even after the first busy receipt;
search remained a separate control. Stop redundant activation captures only
after actual busy paint, while retaining native display, whole-operation loop
observation and exact final readiness/transcript capture. The strict 50-test
run passes without retained DB files; fresh scale qualification is still
required. Removing observer allocation is not evidence that application GC
pauses disappeared, and moving a collection is not retiring its heap owners.

## Textual shutdown and executor join are different ownership checkpoints

**TASK-31966 / TASK-31245, 2026-10-05, clean d7b1e915dc.** The ordinary
native quit returned with 37 database descriptors. A separate guarded real-app
headless probe still retained native-open Chat connections after their default
executor threads actually exited, plus original constructor caches. One attempt
had a storage operation and three acquisitions live immediately after Textual
shutdown but none after the runner joined. Neither a cancelled worker nor an
unmounted app proves all physical work has stopped or all database owners closed.

Keep the census read-only and distinguish these boundaries. Match allocation
origins with weak, monotonically assigned tokens for the actual connection;
scalar object IDs can be reused after close. The final probe matched ten retained
native handles with no dropped registrations, while explicitly preserving the
headless/native distinction and a still-present separate theme executor. Retire
finite callbacks on their own thread through the installed owner boundary; do
not turn the fixture's blanket retirement into a production lifetime policy.

## Committed saved titles are not owned by older send snapshots

**TASK-33620.9, 2026-10-05, publication review of c7e5558.** The new rename
publication test cancelled a pending send before publishing its new title, so
it missed the reverse order. Publishing first and cancelling afterward restored
the old `pre_send_title`. The same stale snapshot existed in optimistic-send
rollback and successful delayed durable identity publication; SQLite kept the
new name while live surfaces reverted. The strict regression run produced six
failures with five rollback controls passing, and the successful-publication
check produced two more failures (including replay after first persistence).

Saved sends do not auto-title. Keep the current title when their saved binding
is unchanged; retain genuine scratch/rebound rollback and first-save naming.
Test both event orders and the shared successful-send publication helper, not
only cancellation inside the rename worker. Also recheck sanitized title input
before writing: a raw nonblank control byte can become blank during sanitization
and otherwise erase the saved title before live publication rejects it.

## An empty selected-worker wait is not an empty Textual wait

**TASK-33620.9, PR3024, 2026-10-05.** A real rename refusal finished and
reported its error, but its test failed in `WorkerCancelled`: its selected
rename-worker list was already empty, and installed Textual's
`wait_for_complete` uses `(workers or self)`. It therefore waited for unrelated
cancelled background work. A completed real refusal plus a separately cancelled
worker reproduces the failure deterministically; stdlib `asyncio.gather` over
only selected `worker.wait()` calls repairs all 30 affected controls without
catching genuine selected-worker failures or changing production behavior.
Guard empty selections or gather their exact waits; never silently broaden a
settled operation to every app worker.

## Closed-loop submit retirement includes maintenance admission

During PR3024 qualification, the closed-loop fixture initially assumed 20
zero-delay ticks reached COMMITTING despite real off-thread hook admission.
Waiting for the actual history event exposed a real ownership leak: permanent
shutdown removed the Task from submit/preparation maps but maintenance admission
still retained it until an impossible closed-loop finalizer. TASK-34412's two
RED controls prove that path; one exact-key removal under the existing closed
loop guard preserves the live-loop peer. Check every admission ledger when
retiring an unreachable owner, not only its ticket's original registry. The
25-body-pass receipt still reports two ContextVar warnings and retained SQLite
files; ledger retirement is not normal shutdown or resource qualification.

## Check empty authority before admitting a registry owner

**TASK-34410, PR3024, 2026-10-05.** After exact fixture-owner retirement,
the original agent shutdown control still retained three workspace SQLite/WAL/SHM
descriptors while its direct control retained none. Allocation tracing found
`frozen_workspace_roots` constructing the shared registry with an empty captured
binding maximum, only to return no roots. Four real-SQLite tuple/iterator tests
failed on default database creation or supplied-handle reopening. Returning
before admission for empty authority repaired the strict 22-case batch without
closing the shared cache or weakening nonempty live-binding validation.

Trace why a cache is admitted before inventing its teardown policy. An empty
authority result does not need a database read; unnecessary construction can
look like a lifetime leak. Keep the existing nonempty/retarget controls and
close only the regression's own database handles.

## Abandoned test Tasks still own their ContextVar reset context

**TASK-34412, PR3024, 2026-10-05.** The emergency fixture correctly dropped
its closed-loop Task but collected it outside the Task's copied Context. Its
diagnostic/manual-authority finalizers raised two genuine token-reset errors.
The warning-as-error control fails. Merely collecting inside the exact public
Task Context also fails: Python 3.12's custom exception handler tries to enter
that same Context again, replacing the expected pending diagnostic with an
unhandled-handler error whose nested text still contains the pending message.

For this deliberately abandoned test owner, use its captured public Context
and observe the real default handler. Require exact first-line messages/counts,
both binding restorations, original weakref/ledger assertions and no unraisable
warnings; substring matching would accept the failed experiment. The 114-case
covering run passes without warnings or retained database files. This is fixture
ownership only: production emergency detachment still cannot make a closed-loop
Task terminal. Do not catch authority-reset errors or claim normal native quit.

## Installed runtime identity survives a refresh; controller-only fakes do not

**PR3024, 2026-10-06.** The surviving-child Close fixture passed prepared authority
into a controller-only namespace bridge. Real Console synchronization then rebound
the controller from its installed runtime, invalidating that authority. Installing
the incumbent complete bridge through the runtime and explicitly refreshing before
the original assertions repaired the intended workflow without weakening any of
its 208 assertions. Bind the owner that normal synchronization actually reads.

## Zero registered handles does not prove every native allocation retired

**PR3024, 2026-10-06.** Strict Close bodies passed while Chat SQLite files remained.
Forwarding traces proved successful quiescence, zero registrations and no later
registered opens, but a live native lease still belonged to an exited worker.
The actual getter captured maintenance rejecting an initialization PRAGMA after
allocation and before publication; the SQLite/path-only cleanup missed RuntimeError.
A real acquisition/barrier RED and the existing initializer-owned cleanup repair
cleared the 21-case strict gate. Trace unpublished allocations and the first opener
(including requester attribution before later source reads), not just caches or
registry counts. Preserve the original error and failed-close evidence; do not
invent a global closer or GC policy to hide a failed initialization owner.

## Local not-sent copy must stay separate from agent diagnostic custody

**PR2918 strict-base integration, 2026-10-03.** Shipped TASK32369 introduced
Anthropic/Cohere pre-network refusals and a field-only `not sent` presentation.
Its two stream tests read the former generic-wrapper copy through
`describe_stream_failure`. The existing ADR211 composition projects typed
errors to content-free diagnostics and carries sanitized Console presentation
separately. The unchanged incoming selection therefore had **3 passes / 2
failures in 1.580s XML**. Moving the local wrapper ahead of the typed payload
would make those assertions pass by putting presentation into exception
message/STEP_ERROR, violating the approved custody boundary.

The preserving queue composition carries both values and consumes the typed
payload first. Only the two incoming display oracles now read
`describe_console_stream_failure`; status/no-HTTP checks, real handlers,
bootstrap admission, no-network fixture and the three passing cases stay
intact. Separate diagnostic checks and two actual Console controls exercise
direct toast/row and default-agent row, real SQLite STEP_ERROR and admitted
run-log files. The initial new selection was **4 passes / 3 failures in
2.104s XML**: its expected `ChatConfigurationError()` defaults to HTTP500,
whereas these failures explicitly have `status_code=None`. Correcting only
those three expected constructors yields **7 passes in 1.819s XML**; unchanged
typed status/redaction/audit neighbors pass **16 in 4.072s XML**. No production
repair, diagnostic leak, marker/config/readiness bypass or live provider run
followed. Raw/XML and command/exit receipts remain
`/private/tmp/pr2918-localcheck-{original-local,local-green,local-final,provider-neighbors}.*`.

The adjacent unchanged Voice selection is **NON-GREEN: 58 passes / one summary
failure in 32.607s XML**. The exact incoming failed node reproduces the same
`RecoveryRequired('raw_source_selection_changed')` message in **3.539s XML**.
That is source/baseline evidence, not a timing/profile cause or broad wizard
certificate. The moved class AST, eight registered handlers and six logger
call ASTs are exact; the wizard's lowered 9866 pin passes both scoped size
cases. Artifact counting corrected the prospective thirteen test-patch count
to fourteen actual calls (thirteen OmniVoice plus one setup-resume) before
review, with stopped metadata invocations retained. All positive/NON-GREEN
raw/XML, late stale-cleanup tails and physical/service/provisioning/playback
limits remain. Overlapping selections are never summed.

## Busy configuration locks must not park the Console event loop

**TASK-34415, 2026-10-06.** Forwarding observations found real REBUILD waits
outside the control render callback. A held-native-lock regression made both
refresh callers block until its two-second watchdog. Reuse the installed
coalesced fresh-state retry, not a config cache: probe the same REBUILD then
FILE RLocks nonblocking and retain acquired locks across unchanged checked
entry/retirement, so a writer cannot win between probe and entry. Release a
partial REBUILD acquisition when FILE is busy. Real-holder, fresh-source and
cleanup assertions establish this repair; they do not prove the full 50ms
activation matrix or native terminal qualification.

## A completed one-shot timer need not remain discoverable

**TASK-34415, PR3034 UI1, 2026-10-07.** Approval geometry failed before its
layout assertions because the readiness helper required a startup timer object.
Installed Textual's timer registry is a WeakSet. Observing real successful
projection and natural timer retirement reproduced the exact None assertion.
Await a still-present timer, then drain its queued callback in either case and
retain actual readiness/rendered-state assertions. The same retired-timer probe
and a genuinely held pending-timer control passed without extending deadlines
or changing production startup/profile behavior.

## A drained callback is not a completed coalesced refresh

**TASK-34415, PR3034 final-head UI1, 2026-10-07.** The same approval geometry
helper next failed its original requested=False assertion after one callback
drain. A real REBUILD holder and second native sync reproduced the assertion
with clean holder retirement. The installed trailing replay also clears its
flags before starting its async worker; a real pending-worker control exposed
readiness returning in that gap. Retain the callback drain, then observe both
the real flags and unfinished same-screen console-sync workers within the
existing projection deadline. Native and worker GREEN, all six original
consumers, and a frozen-replay rejection preserve the assertions and bound.
This proves supported deferral, not the hosted runner's specific lock cause.
Review then caught Pilot.pause's independent 30-second screen drain inside
that nominally ten-second loop. Use a remaining-budget-capped asyncio yield,
not another unbounded-to-this-phase drain. A call-path guard is RED on the
initial poll and GREEN with real native/worker replay; all original controls
and the frozen-replay rejection pass their intended outcomes afterward.

## Refusal text in controls is not transcript publication

**TASK-34415, PR3034 final-head UI3, 2026-10-07.** The refused-send Resend
journey saw provider refusal text anywhere on screen, then selected an echo not
yet in the transcript. A 250ms call-through publication delay reproduced the
exact missing-button failure: selection of an absent row is intentionally a
no-op, and later publication does not retry it. Scope the existing text wait
to the actual transcript before selection, retaining its bounds and assertions.
The same delayed original node and thirteen affected cases pass; no production
selection semantics or timeout change is needed. This controlled proof does
not attribute unrecorded hosted ordering or certify full qualification.

## A fence held across an await is visible to app exit

**TASK-33628.5, 2026-10-05.** Moving Console Delete/Undo's durable write off
the event loop meant holding the store's fork-source and voice-promotion
admissions across an `await`, from before the write starts until its result
is applied. On dev the delete ran inline, so nothing else on the loop could
run inside that window. On the branch, app exit could: Ctrl+Q, Ctrl+Q,
**Quit anyway** during a 3,002-message save reached `ConsoleRuntime.dispose`,
whose `ConsoleChatStore.end_app_runtime` takes the voice-promotion
*replacement* fence. That fence refuses while any admission is held
("Voice promotion state prevents store replacement."), and dispose logged the
refusal and skipped the whole store teardown: the trace-settlement drain, both
executor shutdowns and the teardown retries. The delete itself still
committed, so nothing looked wrong in the chat; only the app log showed it.
The suite was green; the branch's own final gate found it by reading which
other code takes the same fence. **What to do:** when a change adds an `await`
inside a section that holds a lock, fence or admission, list every path that
takes the same fence -- app exit and store replacement especially -- and
decide what each does when it arrives mid-window. Here app exit now closes
admission, waits a bounded time for the writes in flight
(`Chat/console_durable_writes.py`), and past the bound still runs every step
that does not replace state. Pin it with the real app: press the real
Ctrl+Q twice with the write held (`Tests/UI/test_console_message_delete_quit.py`),
and assert the teardown steps ran, not only that the data landed.
## Recovery copy needs the actual rollback outcome and every input path

**TASK-33622.20, 2026-10-06.** A switcher-owned quit question deliberately
refused the synchronous Character reveal; rollback succeeded but generic failure
copy replaced the user's intent. The first recovery attempt then failed again:
cold hydration had persisted settings and advanced the immutable search revision.
Refreshing only the Retry button left Enter and row click equally stale. Valid
mounted RED controls pinned all three before they converged at the common
activation entry on a fenced same-target refresh.

The first refusal flag also masked failed prior-session restoration: the real
best-effort repaint helper swallowed repaint/focus errors, so a coordinator-only
exception control never observed them. Actual-helper RED controls required its
existing adapter to report a bounded success signal; exceptions or explicit
failed recovery clear refusal provenance. Keep closed outcomes and strict owner
proofs unchanged. Live Active updates must refresh badges without silently
erasing retained recovery, and a missing target must never donate Retry to its
neighbor. Mounted evidence does not establish native quit or resource retirement.

## Guard observer entry before fresh configuration reads

**PR3020 required UI qualification / TASK-34807, 2026-10-13.** A real Rules
maintenance/Stop case blocked in the credential timer before control-bar refresh
reached its existing configuration guard. Forcing the armed timer with the real
cold rebuild lock held reproduced both maintenance and rebuild failures; the
watchdog stack reached provider selection, load_settings and native admission.
Route the entire observer through the existing nonblocking checked config
lifetime before it derives readiness. Preserve revision, time-based status and
evidence gates, then prove the ordinary timer refreshes their fresh values after
maintenance; increasing the Stop deadline would conceal the entry defect.

**Same incident, hosted follow-up.** The first guarded poll at5fc passed Stop
controls but reintroduced one config and two storage admissions on every warm
tick. Perf Guard38454091568 measured57.5 native opens against Linux6.75×1.05.
The ordinary5fc macOS census measured1/2 admissions and61.75 opens. A matched
comparison installed the original poll only after the census seeded its config:
both old and corrected callbacks paid0/0 admissions and zero native opens;
3.75–10.5 opens per tick came entirely from existing cadence probes. An earlier
autouse import changed startup, so its lower baseline was rejected. Preserve
PERF06’s source/posture-checked
warm read under nonblocking native lock probes; only a genuine cold miss needs
the full checked lifetime. Publish the credential snapshot only after the real
control refresh returns True: a native pause after summary but before controls
otherwise consumes the evidence revision without publishing its controls.
Idle hides the rule chip but keeps its old label; Stop tests must assert no
visible active copy within the original deadline, rather than clearing hidden
widget content.


## Pending registration and Inspector rows have different publication boundaries

**TASK-34807, 2026-10-13.** PR3020 UI shard4 on3a865 failed the second
pending-interrupt journey: transcript and run chip said Waiting for your approval,
but Inspector still showed Generating and zero approvals. The fixture waited on
the private pending set, bypassing the lock protecting its subsequent kind write,
then ignored the full refresh's boolean completion. An actual held configuration
rebuild on90bf reproduced False/Inspector0 with public rounds2; releasing it let
the existing coalesced replay publish state2, before Textual finished recomposing
the rows. The relevant owners and original fixture matched accepted84861.

Use the locked public pending-round count for setup. Request the existing refresh
once, then observe its actual Inspector count and fully rendered copy under the
original bounded wait. A pause or updated state alone does not prove row publication.
Keep the original assertions and real worker cleanup; do not retry rendering from
the observation predicate or change native admission/deadlines. This qualifies the
current full-refresh contract, not a new native-free partial Inspector projection.

## Console typing waits behind a busy screen too; a click-time composer snapshot cannot order queued keys (TASK-35640, 2026-10-19)

**Incident.** TASK-35640's AC#2 first read "keys typed after each of two
queued tab clicks go to the tab clicked just before them". A replay design was
built for it: each click record also noted how many draft carries had already
happened, and each activation replayed the later switches to find where its
own keys began in the composer. The strict mounted test timed out waiting for
the composer to show the first key while A's activation was held. Printable
keys reach the composer through `ChatScreen.on_key` ->
`composer.handle_console_key`, so they wait in the screen's message queue
behind the held `on_button_pressed`, exactly like the presses. The record a
second click takes therefore cannot contain keys typed before it, and both
records were identical; no replay can split them. The arrival order itself
is not fixed either. Injected back to back (the test harness), the keys
reached the composer ahead of the presses. Live on a loaded host (click
Chat 1, click Chat 2, "x", click Chat 3, "y" at 0.15 s gaps, from Chat 4),
six runs gave two outcomes. In five (two on the fix, three on dev) both keys
ended in Chat 3, so they reached the composer after every queued press;
typing order would have put "x" in Chat 2. In one run on the fix both keys,
"y" included, ended in Chat 2 ("two:xy"), so they reached the composer
before Chat 3's press. Not one run applied them in the order they were
typed.

**What to do.** Before ordering keystrokes against clicks in the Console,
remember both travel to the screen's queue and are handled in arrival order;
a widget-side snapshot sees only what the screen has already applied. Pin such
behaviour with assertions that hold for every arrival order (nothing stays in
the chat being left, nothing is lost, order kept). Write the User Guide
claim from several live runs, not one. After the first run, the guide said
"text typed between the two clicks can land in either"; the third run put a
key typed after the second click into the first tab, and the sentence had
to widen.
