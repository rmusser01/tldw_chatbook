# Lessons: Console screen-to-runtime wiring

Working knowledge about wiring ChatScreen callables into the Console
controller/store runtime. Not decisions (see `backlog/decisions/`) — these
are traps that have actually cost time here, kept so the next person does
not rediscover them.

**Every entry states the incident that produced it.** A lesson without its
evidence decays into folklore, and folklore gets ignored. If you add one,
bring the incident.

---

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
