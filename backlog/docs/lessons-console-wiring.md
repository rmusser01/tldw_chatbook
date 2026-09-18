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
