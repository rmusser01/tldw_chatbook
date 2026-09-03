# Task 4 report — return safely from provider credential settings

## Scope

Implemented only Task 4 from the approved Conversation settings return plan.
The Task 2 typed contracts and Task 3 snapshot/staging semantics remain the
source contracts; no new state owner, schema, persistence, or ADR was added.

## Assumptions and interfaces

- `ProviderSettingsNavigationTarget` is the only typed return-aware route into
  Providers & Models. Existing non-return provider links remain backward
  compatible.
- Same-provider identity is compared through the existing canonical provider
  key. Its draft remains mounted and only fixed dirty-field display names are
  disclosed.
- A different-provider target is staged behind the explicit Review, Discard,
  and Return actions; the deep-link never overwrites the existing draft.
- Settings retains only the typed provider target, opaque handoff revision, and
  fixed outcome enum in the existing process-local screen-state handoff. The
  private Console snapshot remains on `ChatScreen` and in the existing native
  `ScreenStateStore` representation.
- Settings produces `settings-provider-return`, `settings-provider-stay`, and
  `settings-provider-return-without-save`, plus
  `settings-provider-conflict-review`, `settings-provider-conflict-discard`,
  and `settings-provider-conflict-return`.
- `ChatScreen.apply_navigation_context()` parses and stages an exact
  `ConsoleSettingsReturnTarget` without taking handoff ownership. The mounted,
  stack-owning deferred consumer claims and validates the coordinates and
  origin session/settings revision, restores the exact suspended modal snapshot
  and logical focus, then acknowledges only restoration or terminal rejection.
  Transient modal failures and unmount cancellation release the exact claim
  while retaining the safe retry coordinate; a later mount or resume reclaims
  that same revision.

## Changes

- Added typed provider deep-link application with exact provider/model/API-key
  focus. A narrow mount-projection guard prevents the provider context-window
  control's initial `Changed` event from manufacturing a dirty draft before
  the navigation callback lands.
- Added same-provider dirty disclosure and different-provider conflict regions.
  Review preserves and focuses the prior draft, Discard explicitly reverts it
  before applying the staged target, and Return preserves unrelated edits.
- Added mutation-aware continuation after a fully applied provider save:
  credential-only saves return `credential_saved`; all broader provider saves
  return `provider_settings_saved`. Save failure keeps the draft and handoff.
- Added Return without saving through the existing confirmation-dialog discard
  semantics and Stay as an exact handoff acknowledgement/abandonment.
- Added Console claim/restore settlement for valid, stale, deleted, temporary,
  superseded, consumed, and transient-failure routes. Return status text is
  selected from fixed screen-owned copy, while canonical readiness is rendered
  separately. An absent configured environment credential remains blocked and
  adds the required export/relaunch recovery.
- Added mounted Textual journey coverage for the Settings and Console sides,
  including focus fallback and an exact six-key return-route allowlist.
- Preserved an explicit first-run `model=None` target instead of replacing it
  with a configured model, and restored safe Settings continuation state across
  ordinary fresh-screen navigation without replaying it on intentional Return.
- Made Return the focused, primary, scrolled-into-view post-save action at a
  compact viewport, renamed the secondary action to `Stay in Settings`, and
  used fixed readiness-oriented outcome and recovery copy.
- Added a production-router journey from the real Console Configure action,
  through a fresh Settings screen and its navigation completion callback, to a
  fresh Console screen's deferred modal restoration.
- Moved the return claim from pre-mount navigation-context application into the
  mounted deferred worker. Router veto therefore leaves the revision pending,
  and both worker cancellation and screen unmount release only that worker's
  exact claim without disturbing a newer staged return.
- Made every Settings Return entry point exact single-flight at the handler
  boundary as well as visibly disabled while navigation is outstanding. A
  failed/vetoed completion clears the fence and restores the continuation;
  successful navigation settles the outgoing continuation state once.
- Replaced state-dependent superseded wording with the fixed truthful copy
  `This return was superseded by a newer request.`

## RED evidence

Required command before production edits:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/UI/test_settings_configuration_hub.py Tests/UI/test_console_native_chat_flow.py -k 'conversation_settings_return or provider_navigation_conflict' -q
13 failed, 670 deselected, 1 warning in 13.07s
```

The failures were contract-expected: the Settings conflict/continuation nodes
and Console consumer did not exist. There were no fixture or collection errors.

Round-1 review regressions were also run before their production changes:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/UI/test_settings_configuration_hub.py Tests/UI/test_console_native_chat_flow.py -k 'explicit_unselected_model or continuation_survives_fresh_settings_screen or focuses_primary_return_above_compact_fold or terminal_rejection_consumes_once or superseded_route_preserves_latest_handoff or consumed_route_clears_stale_snapshot_with_copy or releases_transient_modal_mount_failure or real_navigation_restores_fresh_console_modal' -q --tb=short
10 failed, 682 deselected, 1 warning

PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py -k 'real_navigation_restores_fresh_console_modal' -q --tb=short
1 failed, 323 deselected, 1 warning

PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py -k 'claims_exact_revision_and_restores_mounted_draft' -q --tb=short
3 failed, 323 deselected, 1 warning in 5.90s

PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/State/test_pending_handoff_store.py Tests/UI/test_console_native_chat_flow.py -k 'exact_revision_status_distinguishes or waits_for_exact_in_flight_claim' -q --tb=short
2 failed, 415 deselected, 1 warning in 2.97s
```

The first combined run exposed one test-authoring patch-target error in the new
real-router case; that seam was corrected before any production edit and was
not accepted as RED. Its isolated rerun then traversed the production router to
the fresh Console modal and failed on the old outcome copy. The remaining
faithful failures showed the old configured-model fallback, absent lifecycle state,
hidden/non-primary continuation, silent terminal outcomes, lost retry target,
old fixed success wording, and the final in-flight/settled ambiguity. The
test-authoring error was not counted as evidence of missing production behavior.

Round-2 review regressions were run before their production changes:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/UI/test_settings_configuration_hub.py Tests/UI/test_console_native_chat_flow.py -k 'single_flight_and_retries_after_failed_navigation or superseded_route_preserves_latest_handoff or router_failure_before_mount_keeps_handoff_pending or rapid_unmount_before_consumer_keeps_handoff_pending or unmount_releases_acquired_exact_claim' -q --tb=short
5 failed, 694 deselected, 1 warning

PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/UI/test_settings_configuration_hub.py -k 'conversation_settings_return_is_single_flight_and_retries_after_failed_navigation' -q --tb=short
1 failed, 368 deselected, 1 warning in 2.19s
```

The five failures matched the review findings: duplicate Return navigation,
state-dependent superseded copy, a pre-mount router-veto claim, a rapid-unmount
claim, and an acquired claim stranded during unmount cancellation. The first
combined command was interrupted only after pytest had reported all five
failures because the new cancellation test's failed assertion skipped its
release signal; its cleanup was moved into `finally` before production edits.
There were no fixture or collection failures. The isolated RED then exercised
a queued production `Button.Pressed` directly and proved that disabled visual
state alone did not enforce the single-flight fence.

## GREEN evidence

Required journey selection:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/UI/test_settings_configuration_hub.py Tests/UI/test_console_native_chat_flow.py -k 'conversation_settings_return or provider_navigation_conflict' -q --tb=short
23 passed, 671 deselected, 1 warning in 35.74s
```

Focused return-contract suite:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/State/test_pending_handoff_store.py Tests/State/test_screen_state_store.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_session_settings.py -k 'conversation_settings or provider_navigation or credential or screen_state' -q --tb=short
92 passed, 1025 deselected, 1 warning in 45.62s
```

The final in-flight ownership regression and complete handoff-store file:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/State/test_pending_handoff_store.py Tests/UI/test_console_native_chat_flow.py -k 'exact_revision_status_distinguishes or waits_for_exact_in_flight_claim' -q --tb=short
2 passed, 415 deselected, 1 warning in 2.82s

PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/State/test_pending_handoff_store.py -q --tb=short
90 passed, 1 warning in 0.46s
```

Round-2 review slice and cumulative Task 4 selection:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/UI/test_settings_configuration_hub.py Tests/UI/test_console_native_chat_flow.py -k 'single_flight_and_retries_after_failed_navigation or superseded_route_preserves_latest_handoff or router_failure_before_mount_keeps_handoff_pending or rapid_unmount_before_consumer_keeps_handoff_pending or unmount_releases_acquired_exact_claim' -q --tb=short
5 passed, 694 deselected, 1 warning in 30.90s

PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/State/test_pending_handoff_store.py Tests/State/test_screen_state_store.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_session_settings.py -k 'conversation_settings or provider_navigation or credential or screen_state' -q --tb=short
96 passed, 1025 deselected, 1 warning in 76.21s

PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/State/test_pending_handoff_store.py -q --tb=short
90 passed, 1 warning in 0.62s
```

The warning in these runs is the repository environment's existing Requests
dependency-version warning. No full suite was run, per repository and task
instructions.

Static verification:

```text
../../.venv/bin/python -m ruff check tldw_chatbook/UI/Navigation/pending_handoff_store.py tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/UI/Screens/chat_screen.py Tests/State/test_pending_handoff_store.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_console_native_chat_flow.py
All checks passed!

../../.venv/bin/python -m py_compile tldw_chatbook/UI/Navigation/pending_handoff_store.py tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/UI/Screens/chat_screen.py Tests/State/test_pending_handoff_store.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_console_native_chat_flow.py
exit 0

git diff --check
exit 0
```

## Privacy and failure closure

- The return navigation mapping contains exactly `session_id`,
  `settings_revision`, `active_view`, `focus_control_id`, `return_revision`, and
  the outcome enum value.
- No API key, prompt, prefill, raw endpoint, transcript, provider draft value,
  or arbitrary result copy is placed in route context or status copy.
- Stale/deleted/temporary and coordinate-mismatched exact claims are terminally
  acknowledged and their obsolete snapshot is removed with fixed visible
  recovery copy. Superseded/repeated routes do not consume a newer handoff. A
  modal mount failure releases the exact claim, retains the suspended snapshot
  and safe typed target, restores the previously active session, and retries by
  exact revision on a later mount/resume.
- Applying a route before the destination mounts never claims the return.
  Cancellation and unmount settle only the opaque claim object acquired by that
  mounted worker; a replacement revision staged during the in-flight attempt
  remains pending and claimable.

## Files

- `tldw_chatbook/UI/Screens/settings_screen.py`
- `tldw_chatbook/UI/Screens/chat_screen.py`
- `tldw_chatbook/UI/Navigation/pending_handoff_store.py`
- `Tests/State/test_pending_handoff_store.py`
- `Tests/UI/test_settings_configuration_hub.py`
- `Tests/UI/test_console_native_chat_flow.py`
- `Docs/superpowers/plans/2026-09-02-task-30010-conversation-settings-return-contract.md`
- `.superpowers/sdd/2026-09-02-task-30010-conversation-settings-return-contract/task-4-report.md`

## Deviations and ADR check

One narrow lifecycle scope amendment was required after final diff review:
`PendingHandoffStore.claim()` intentionally returns `None` for both settled
absence and an exact claim held in flight by an outgoing screen. The existing
store therefore gains a value-free exact revision-state query so a fresh
Console does not destroy retry ownership during that race. No handoff value,
schema, persistence boundary, or new owner is exposed. The focused suite
exposed no unrelated baseline failure. No generalized testing lesson was
added; the mount-projection issue is a narrow instance of the repository's
already documented mounted-evidence guidance.

ADR required: no new ADR

ADR paths: `backlog/decisions/012-provider-credential-settings-boundary.md` and
`backlog/decisions/033-application-session-state-ownership.md`

Reason: this task directly implements the approved ownership, privacy, and
single-slot handoff decisions without changing their boundaries.

## Round-3 review fixes

Resolved the three Important lifecycle findings without adding a state owner,
store API, persistence change, or ADR:

- Console now checks the captured return revision before claiming the channel,
  clears local target/snapshot coordinates only when the exact captured target
  object is still current, and schedules one retained replacement only when
  that replacement's exact revision remains pending and the Console still owns
  the mounted stack top. An older restore can settle or release only its own
  claim; a replacement staged before claim, during terminal rejection, or
  before a successful acknowledgement remains reclaimable.
- Dirty **Return without saving** acquires the existing return-navigation fence
  before opening its confirmation dialog. All return actions project the
  disabled state, queued/direct duplicates are rejected, cancel clears the
  fence, and confirm carries the same fence into the existing typed navigation
  path without self-blocking or opening another dialog.
- Transient, cancellation, and exception repair captures the existing
  `ConsoleChatStore.active_session_epoch()` immediately after selecting the
  origin session. It restores the prior active session only while both the
  selected session ID and activation epoch still match, preserving a later
  user or newer-target selection even across an A→C→A sequence.

### Round-3 RED evidence

Before production edits:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py Tests/UI/test_settings_configuration_hub.py -k 'replacement_before_claim_consumes_replacement or replacement_during_terminal_restore_is_consumed or success_preserves_replacement_staged_during_restore or transient_cleanup_preserves_new_session_selection or return_without_saving_is_single_flight_on_confirm or return_without_saving_cancel_allows_retry' -q --tb=line --show-capture=no
6 failed, 698 deselected, 1 warning in 17.22s
```

The six failures were the expected product behaviors: no replacement modal
after B arrived before A's claim; no replacement modal after B arrived during
A's terminal settings-revision rejection; A success cleared B's local target;
stale transient cleanup selected the prior session instead of C; duplicate
direct dirty-return events stacked three confirmation dialogs; and cancel had
no pre-dialog fence to clear. There were no fixture or collection failures.

The first GREEN run reported four passes and two residual failures. Both were
test-oracle timing defects: the assertions stopped when B's modal first mounted,
before B's same worker completed status mounting, acknowledgement, and local
target settlement. Waiting for both modal ownership and target settlement fixed
the oracle without changing production behavior.

### Round-3 GREEN evidence

Exact review regressions:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py Tests/UI/test_settings_configuration_hub.py -k 'replacement_before_claim_consumes_replacement or replacement_during_terminal_restore_is_consumed or success_preserves_replacement_staged_during_restore or transient_cleanup_preserves_new_session_selection or return_without_saving_is_single_flight_on_confirm or return_without_saving_cancel_allows_retry' -q --tb=short --show-capture=no
6 passed, 698 deselected, 1 warning in 7.52s
```

Prior round-2 lifecycle plus discard-return regression slice:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/UI/test_settings_configuration_hub.py Tests/UI/test_console_native_chat_flow.py -k 'single_flight_and_retries_after_failed_navigation or superseded_route_preserves_latest_handoff or router_failure_before_mount_keeps_handoff_pending or rapid_unmount_before_consumer_keeps_handoff_pending or unmount_releases_acquired_exact_claim or return_without_saving' -q --tb=short --show-capture=no
7 passed, 697 deselected, 1 warning in 32.32s
```

Fresh cumulative Task 4 selection and complete handoff-store suite:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/State/test_pending_handoff_store.py Tests/State/test_screen_state_store.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_session_settings.py -k 'conversation_settings or provider_navigation or credential or screen_state' -q --tb=short --show-capture=no
101 passed, 1025 deselected, 1 warning in 80.67s

PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/State/test_pending_handoff_store.py -q --tb=short --show-capture=no
90 passed, 1 warning in 0.47s
```

The warning is the already documented Requests dependency-version warning from
the shared root virtualenv. No full suite was run, per repository and task
instructions.

Static verification:

```text
../../.venv/bin/python -m ruff check tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/Screens/settings_screen.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_settings_configuration_hub.py
All checks passed!

../../.venv/bin/python -m py_compile tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/Screens/settings_screen.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_settings_configuration_hub.py
exit 0

git diff --check
exit 0
```

### Round-3 files and self-review

- `tldw_chatbook/UI/Screens/chat_screen.py`
- `tldw_chatbook/UI/Screens/settings_screen.py`
- `Tests/UI/test_console_native_chat_flow.py`
- `Tests/UI/test_settings_configuration_hub.py`
- `.superpowers/sdd/2026-09-02-task-30010-conversation-settings-return-contract/task-4-report.md`

Self-review confirmed that a same-target transient failure remains passive (no
automatic retry loop), only a distinct exact pending replacement is scheduled,
and modal ownership still prevents a replacement from opening over A's
successfully mounted modal. The terminal replacement test advances the real
session settings revision and exercises the production rejection classifier;
the test doubles only inject otherwise unobservable interleaving points. The
existing privacy/copy contract is unchanged. No general lesson or new ADR was
needed.

## Round-4 review fix

Resolved the Important forced-dismiss lifecycle finding entirely inside
`SettingsScreen`; the generic confirmation dialog and application router are
unchanged.

- Added a distinct confirmation-open phase for dirty **Return without saving**.
  Confirmation-open and committed return navigation both disable every Return
  action and reject duplicate events, but only committed navigation suppresses
  the safe continuation from the outgoing Settings snapshot.
- Cancel clears only the confirmation-open phase and re-enables retry. Confirm
  synchronously changes confirmation-open to false and committed-navigation to
  true before claiming/posting the exact typed return, so there is no enabled
  gap and no self-block.
- Added a production-router mounted regression that starts from the real
  Console Configure action, opens a dirty discard confirmation, navigates to
  Home through `TldwCli.handle_screen_navigation()`, and returns to a fresh
  Settings screen. It verifies the router's raw overlay dismissal cannot orphan
  the handoff: the dirty draft and exact safe continuation return, the revision
  remains pending, and a second confirmed attempt restores the originating
  Conversation settings modal.
- Strengthened the existing duplicate-confirm and cancel/retry tests to assert
  the two phases independently and to prove all Return actions are disabled
  while confirmation is open. The existing committed-return real-router test
  still proves the outgoing snapshot omits continuation and a later Settings
  visit cannot replay stale controls.

### Round-4 RED evidence

The first isolated run reached the real navigation handler but asserted its
internal result; `handle_screen_navigation()` intentionally returns `None` and
settles through the message callback/stack. That test-oracle error was corrected
before accepting RED. The corrected isolated regression then failed at the
product boundary:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py -k 'dirty_return_confirmation_survives_real_router_navigation_away_and_back' -q --tb=short --show-capture=no
FAILED ... assert fresh_settings._provider_return_target is not None
1 failed, 334 deselected, 1 warning in 12.32s
```

The combined pre-production run additionally proved the requested phase split
did not yet exist while reproducing the same fresh-screen loss:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/UI/test_settings_configuration_hub.py Tests/UI/test_console_native_chat_flow.py -k 'return_without_saving_is_single_flight_on_confirm or return_without_saving_cancel_allows_retry or dirty_return_confirmation_survives_real_router_navigation_away_and_back' -q --tb=short --show-capture=no
3 failed, 702 deselected, 1 warning in 14.15s
```

The two focused Settings failures were the expected missing
`_provider_return_confirmation_open` state; the router failure was the missing
restored return target. There were no fixture or collection failures.

### Round-4 GREEN evidence

Exact phase and real-router regressions:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/UI/test_settings_configuration_hub.py Tests/UI/test_console_native_chat_flow.py -k 'return_without_saving_is_single_flight_on_confirm or return_without_saving_cancel_allows_retry or dirty_return_confirmation_survives_real_router_navigation_away_and_back' -q --tb=short --show-capture=no
3 passed, 702 deselected, 1 warning in 15.90s

PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py -k 'dirty_return_confirmation_survives_real_router_navigation_away_and_back' -q --tb=short --show-capture=no
1 passed, 334 deselected, 1 warning in 14.23s
```

Task 4 journey filter and cumulative selection:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/UI/test_settings_configuration_hub.py Tests/UI/test_console_native_chat_flow.py -k 'conversation_settings_return or provider_navigation_conflict or dirty_return_confirmation' -q --tb=short --show-capture=no
34 passed, 671 deselected, 1 warning in 85.26s

PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/State/test_pending_handoff_store.py Tests/State/test_screen_state_store.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_session_settings.py -k 'conversation_settings or provider_navigation or credential or screen_state or dirty_return_confirmation' -q --tb=short --show-capture=no
102 passed, 1025 deselected, 1 warning in 94.06s
```

The warning is the already documented Requests dependency-version warning from
the shared root virtualenv. No full suite was run, per repository and task
instructions.

Static verification:

```text
../../.venv/bin/python -m ruff check tldw_chatbook/UI/Screens/settings_screen.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_console_native_chat_flow.py
All checks passed!

../../.venv/bin/python -m py_compile tldw_chatbook/UI/Screens/settings_screen.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_console_native_chat_flow.py
exit 0

git diff --check
exit 0
```

### Round-4 files, ADR check, and self-review

- `tldw_chatbook/UI/Screens/settings_screen.py`
- `Tests/UI/test_settings_configuration_hub.py`
- `Tests/UI/test_console_native_chat_flow.py`
- `.superpowers/sdd/2026-09-02-task-30010-conversation-settings-return-contract/task-4-report.md`

ADR required: no new ADR

ADR paths: `backlog/decisions/012-provider-credential-settings-boundary.md` and
`backlog/decisions/033-application-session-state-ownership.md`

Reason: the fix separates two local UI lifecycle phases without changing the
credential owner, handoff owner, typed payload, persistence boundary, router,
or confirmation-dialog contract.

Self-review traced each mutation: merging the two flags again loses the
continuation in the real-router regression; failing to disable either phase is
caught by direct duplicate events; clearing the wrong phase breaks cancel or
confirm assertions; and serializing committed navigation replays stale controls
in the existing real-router return test. No unrelated code, controller ledger,
general lesson, schema, or ADR changed.

## Round-5 review fix

Resolved the Important post-transfer cancellation finding without changing the
router, handoff store, modal persistence contract, or cancellation propagation:

- `_reopen_suspended_console_settings()` now returns whether the exact modal
  restoration took ownership instead of requiring its caller to infer success
  from the mutable screen-local snapshot slot. A replacement B may therefore
  occupy that slot immediately after A transfers without making A look like a
  transient failure.
- The return worker acknowledges A, clears its active claim pointer, and
  identity-clears only A's local target synchronously at the successful modal
  transfer boundary. Optional status mounting happens afterwards. Failure,
  cancellation, and unmount during that optional work cannot release or
  recreate the already-settled claim.
- Added a production-app regression that blocks optional status after the
  restored modal owns A, navigates away through the real overlay-dismiss/router
  unmount path, and revisits Console. It proves A was settled before dismissal,
  the outgoing snapshot carries no stale A retry coordinates, and the revisit
  neither reopens nor terminally rejects A.
- Added a mounted interleaving regression that stages B immediately after the
  real A modal transfer and before optional status work. A reaches the
  post-transfer path with no retained claim, while B's exact target and private
  snapshot remain pending; dismissing A then exercises the normal resume path
  and restores/settles B.

The forced-dismiss behavior remains intentionally bounded: while present, A's
modal is the sole owner of A's private draft. A real router dismissal may end
that modal and its draft; this fix does not invent persistence after dismissal.
The completed A handoff must nevertheless stay settled and must not replay as a
stale return on the next Console visit.

### Round-5 RED evidence

Initial exact transfer/status and production-router regressions against the
pre-fix implementation:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py -k 'settles_at_transfer_before_status_and_keeps_replacement or real_router_unmount_after_transfer_does_not_replay_stale_handoff' -q --tb=short --show-capture=no
2 failed, 335 deselected, 1 warning in 13.99s
```

Both failures observed `in_flight` instead of `settled` after the modal had
already restored A and optional status was blocked. The router test continued
through real forced dismissal, unmount, and revisit before making its final
assertions; no fixture or collection failure occurred.

After tightening the B regression to stage B inside the exact post-transfer,
pre-status gap, the final test shape was re-run with the production fix removed:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py -k 'settles_at_transfer_before_status_and_keeps_replacement' -q --tb=short --show-capture=no
1 failed, 336 deselected, 1 warning in 2.29s
```

The old mutable-slot inference saw B, classified A as not restored, and never
entered A's post-transfer status boundary (`None`, expected the exact
newer-revision classification reached after settlement). This was the expected
interleaving failure.

### Round-5 GREEN evidence

Final exact regressions:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py -k 'settles_at_transfer_before_status_and_keeps_replacement or real_router_unmount_after_transfer_does_not_replay_stale_handoff' -q --tb=short --show-capture=no
2 passed, 335 deselected, 1 warning in 14.04s
```

Prior transfer, transient, cancellation, unmount, and source-reopen slice:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_session_settings.py -k 'settles_at_transfer_before_status_and_keeps_replacement or real_router_unmount_after_transfer_does_not_replay_stale_handoff or success_preserves_replacement_staged_during_restore or releases_transient_modal_mount_failure or unmount_releases_acquired_exact_claim or rapid_unmount_before_consumer_keeps_handoff_pending or real_navigation_restores_fresh_console_modal or failed_source_reopen_retains_suspended_snapshot_and_token or cancelled_source_reopen_retains_suspended_snapshot_and_token or covered_cancelled_source_reopen_transfers_exact_draft_to_modal or source_reopen_revalidates_exact_owner_after_model_resolution' -q --tb=short --show-capture=no
11 passed, 616 deselected, 1 warning in 48.87s
```

Cumulative Task 4/state slice and complete handoff-store suite:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/State/test_pending_handoff_store.py Tests/State/test_screen_state_store.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_session_settings.py -k 'conversation_settings or provider_navigation or credential or screen_state or dirty_return_confirmation' -q --tb=short --show-capture=no
104 passed, 1025 deselected, 1 warning in 107.91s

PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/State/test_pending_handoff_store.py -q --tb=short --show-capture=no
90 passed, 1 warning in 0.48s
```

The warning is the already documented Requests dependency-version warning from
the shared root virtualenv. No full suite was run, per repository and task
instructions.

Static verification before the report update:

```text
../../.venv/bin/python -m ruff check tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_native_chat_flow.py
All checks passed!

../../.venv/bin/python -m py_compile tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_native_chat_flow.py
exit 0

git diff --check
exit 0
```

### Round-5 files, ADR check, and self-review

- `tldw_chatbook/UI/Screens/chat_screen.py`
- `Tests/UI/test_console_native_chat_flow.py`
- `.superpowers/sdd/2026-09-02-task-30010-conversation-settings-return-contract/task-4-report.md`

ADR required: no new ADR

ADR paths: `backlog/decisions/012-provider-credential-settings-boundary.md` and
`backlog/decisions/033-application-session-state-ownership.md`

Reason: this correction moves settlement to the ownership boundary already
required by ADR-033 and leaves ADR-012's credential owner unchanged.

Self-review confirmed that no await exists between successful A transfer and
exact acknowledgement/identity-guarded local cleanup. The local claim is
cleared before optional UI work, so every later release path receives `None`.
Pre-transfer failure still returns false and follows the existing release/session
repair path; cancellation still propagates. B can neither be acknowledged by A's
claim nor cleared by A's identity check, and normal Console resume remains the
only scheduling path after A's modal leaves. No controller ledger, task state,
general lesson, schema, persistence owner, or ADR changed.

## Round-6 review fix

Resolved the Important covered-cancellation ownership gap at the existing modal
transfer boundary:

- `_open_console_settings()` accepts a private, optional transfer callback. It
  reports successful modal ownership exactly once after a normal push or before
  re-raising cancellation when the exact modal remains covered on the stack.
  Default callers retain the prior behavior, and the callback captures no draft
  snapshot.
- The Task 4 return worker commits A synchronously from that callback: it
  acknowledges the exact claim, identity-clears only A's local target, and
  removes A's active claim pointer before cancellation can reach any release
  path. A normal return invokes the same single-shot callback as a fallback, so
  the Round-5 pre-status ordering remains unchanged.
- A failed exact acknowledgement is not represented as success. The worker
  inspects only A's exact store status; if A was concurrently requeued pending,
  it exact-discards that owner now that the modal holds the draft. If the store
  still reports A in flight, the source relinquishes its duplicate ownership but
  does not release a snapshot-less retry. This deliberately fails closed and
  leaves the store observably unsettled for diagnosis.
- Added a mounted production-app regression that lets the real Conversation
  settings modal mount, covers it with a newer real overlay while the push await
  is still active, cancels the return worker, then navigates away and revisits
  Console. It proves cancellation propagates, A settles before unmount, the
  modal is the sole draft owner, and no stale retry survives.
- Added an outer-worker pre-transfer cancellation regression and an exact
  acknowledgement-failure regression. Strengthened the Round-5 B interleaving
  test to stage B inside the transfer callback and assert the callback is
  single-shot, A alone is acknowledged/cleared, and B remains pending with its
  private snapshot.

### Round-6 RED evidence

The first combined run exposed a test-oracle identity error in the new
pre-transfer test; that test-only error was corrected before accepting RED. The
production-faithful covered-cancellation regression then failed at the intended
product boundary while the pre-transfer regression already passed:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py -k 'covered_mount_cancellation_settles_before_unmount or pretransfer_cancellation_retains_retry' -q --tb=short --show-capture=no
FAILED ... assert status_after_cancel == "settled"
1 failed, 1 passed, 337 deselected, 1 warning in 12.44s
```

The observed exact status was `pending`: cancellation had cleared the screen
snapshot through the Task 3 covered-modal path, then Task 4 released A because
it could not observe that ownership transfer.

### Round-6 GREEN evidence

Exact RED pair after the implementation:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py -k 'covered_mount_cancellation_settles_before_unmount or pretransfer_cancellation_retains_retry' -q --tb=short --show-capture=no
2 passed, 337 deselected, 1 warning in 12.04s
```

The final mounted test also captures the still-covered modal's draft after
cancellation, proving the modal retains A while the source slot is empty:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py -k 'covered_mount_cancellation_settles_before_unmount' -q --tb=short --show-capture=no
1 passed, 339 deselected, 1 warning in 11.42s
```

Exact Task 3 covered cancellation, mounted Task 4 cancellation, outer
pre-transfer cancellation, B replacement, and acknowledgement-failure cases:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_session_settings.py -k 'covered_cancelled_source_reopen_transfers_exact_draft_to_modal or covered_mount_cancellation_settles_before_unmount or pretransfer_cancellation_retains_retry or settles_at_transfer_before_status_and_keeps_replacement or ack_failure_is_not_reported_as_settled' -q --tb=short --show-capture=no
5 passed, 625 deselected, 1 warning in 15.34s
```

Focused transfer, cancellation, transient failure, unmount, navigation, and B
replacement slice:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_session_settings.py -k 'covered_mount_cancellation_settles_before_unmount or pretransfer_cancellation_retains_retry or ack_failure_is_not_reported_as_settled or settles_at_transfer_before_status_and_keeps_replacement or real_router_unmount_after_transfer_does_not_replay_stale_handoff or success_preserves_replacement_staged_during_restore or releases_transient_modal_mount_failure or unmount_releases_acquired_exact_claim or rapid_unmount_before_consumer_keeps_handoff_pending or real_navigation_restores_fresh_console_modal or failed_source_reopen_retains_suspended_snapshot_and_token or cancelled_source_reopen_retains_suspended_snapshot_and_token or covered_cancelled_source_reopen_transfers_exact_draft_to_modal or source_reopen_revalidates_exact_owner_after_model_resolution' -q --tb=short --show-capture=no
14 passed, 616 deselected, 1 warning in 61.65s
```

Cumulative Task 4/state selection and complete handoff-store suite:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/State/test_pending_handoff_store.py Tests/State/test_screen_state_store.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_session_settings.py -k 'conversation_settings or provider_navigation or credential or screen_state or dirty_return_confirmation' -q --tb=short --show-capture=no
107 passed, 1025 deselected, 1 warning in 119.15s

PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/State/test_pending_handoff_store.py -q --tb=short --show-capture=no
90 passed, 1 warning in 0.46s
```

The warning is the already documented Requests dependency-version warning from
the shared root virtualenv. No full suite was run, per repository and task
instructions.

Static verification:

```text
../../.venv/bin/python -m ruff check tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_native_chat_flow.py
All checks passed!

../../.venv/bin/python -m py_compile tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_native_chat_flow.py
exit 0

git diff --check
exit 0
```

### Round-6 files, ADR check, and self-review

- `tldw_chatbook/UI/Screens/chat_screen.py`
- `Tests/UI/test_console_native_chat_flow.py`
- `.superpowers/sdd/2026-09-02-task-30010-conversation-settings-return-contract/task-4-report.md`

ADR required: no new ADR

ADR paths: `backlog/decisions/012-provider-credential-settings-boundary.md` and
`backlog/decisions/033-application-session-state-ownership.md`

Reason: the change exposes the modal-ownership transition already required by
ADR-033 to its exact handoff consumer; it does not move the credential owner,
introduce persistence, or change the typed handoff/store boundary.

Self-review confirmed that the transfer callback is private, optional,
single-shot, synchronous, and does not retain the private draft. Covered
cancellation invokes it before a bare re-raise, while pre-transfer cancellation
never invokes it and therefore releases A with the exact retry state intact.
The callback nulls the local A claim even when exact acknowledgement fails, so
no `except` or `finally` path can release a snapshot-less retry. Exact identity
checks preserve B throughout. The acknowledgement-failure case intentionally
leaves an unexplained in-flight A observably in flight rather than claiming
settlement or duplicating modal/store ownership. No controller ledger, task
state, general lesson, schema, persistence owner, or ADR changed.

## Round-7 review fix

Resolved the Important acknowledgement-recovery leak without adding a store
API, state owner, persistence change, or ADR:

- The modal-transfer settlement path still attempts the exact acknowledgement
  first. If it returns false or raises before mutation, it now releases that
  exact claim through `PendingHandoffStore.release()`.
- When A is still the latest revision, release requeues exact A and the transfer
  path immediately removes only `(A.revision, A.value)` with
  `discard_pending_exact()`. The modal remains the sole draft owner and no
  snapshot-less retry survives.
- When newer B is already pending, release removes A from in-flight ownership
  without requeueing it. A's exact discard is a no-op and B retains its original
  revision, value, and claimability.
- When the failed acknowledgement result arrives after real store mutation has
  already settled or superseded A, exact release returns false and the
  value-free status check records that terminal state without touching B or a
  different current claim.
- Replaced the prior test that asserted the leaked `in_flight` state with
  mounted false/exception recovery cases grounded in the real store's release,
  replacement, settlement, and claim transitions. Existing covered-cancel,
  pre-transfer retry, and single-shot transfer regressions remain unchanged.

### Round-7 RED evidence

The final new test shapes were run before the production change:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py -k 'recovers_pre_mutation_ack_failure or ack_failure_preserves_pending_replacement or false_ack_after_real_settlement_preserves_owner' -q --tb=short --show-capture=no
4 failed, 2 passed, 339 deselected, 1 warning in 8.46s
```

The four expected failures were the pre-mutation false/exception cases with A
alone and with B already pending: A remained `in_flight` instead of becoming
`settled`/`superseded`. The two characterization cases passed because their
real store mutations had already settled or superseded A; they guard the fix
from releasing or discarding a different owner. There were no fixture or
collection failures.

### Round-7 GREEN evidence

Exact acknowledgement-failure regressions:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py -k 'recovers_pre_mutation_ack_failure or ack_failure_preserves_pending_replacement or false_ack_after_real_settlement_preserves_owner' -q --tb=short --show-capture=no
6 passed, 339 deselected, 1 warning in 7.90s
```

Acknowledgement, replacement, single-shot transfer, covered cancellation, and
pre-transfer retry slice:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_session_settings.py -k 'recovers_pre_mutation_ack_failure or ack_failure_preserves_pending_replacement or false_ack_after_real_settlement_preserves_owner or settles_at_transfer_before_status_and_keeps_replacement or covered_mount_cancellation_settles_before_unmount or pretransfer_cancellation_retains_retry or covered_cancelled_source_reopen_transfers_exact_draft_to_modal' -q --tb=short --show-capture=no
10 passed, 625 deselected, 1 warning in 20.93s
```

Cumulative Task 4/state selection and complete handoff-store suite:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/State/test_pending_handoff_store.py Tests/State/test_screen_state_store.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_session_settings.py -k 'conversation_settings or provider_navigation or credential or screen_state or dirty_return_confirmation' -q --tb=short --show-capture=no
112 passed, 1025 deselected, 1 warning in 127.44s

PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/State/test_pending_handoff_store.py -q --tb=short --show-capture=no
90 passed, 1 warning in 0.46s
```

The warning is the already documented Requests dependency-version warning from
the shared root virtualenv. No full suite was run, per repository and task
instructions.

Static verification before the report update:

```text
../../.venv/bin/python -m ruff check tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_native_chat_flow.py
All checks passed!

../../.venv/bin/python -m py_compile tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_native_chat_flow.py
exit 0

git diff --check
exit 0
```

### Round-7 files, ADR check, and self-review

- `tldw_chatbook/UI/Screens/chat_screen.py`
- `Tests/UI/test_console_native_chat_flow.py`
- `.superpowers/sdd/2026-09-02-task-30010-conversation-settings-return-contract/task-4-report.md`

ADR required: no new ADR

ADR paths: `backlog/decisions/012-provider-credential-settings-boundary.md` and
`backlog/decisions/033-application-session-state-ownership.md`

Reason: this correction uses ADR-033's existing exact release/discard semantics
at the already-approved modal ownership boundary and does not move ADR-012's
credential owner.

Self-review confirmed the recovery never inspects or logs a handoff value. A's
exact release is the only operation that can requeue it; exact discard runs
only after that release succeeds. B is preserved both pending and under a
different in-flight claim. Direct acknowledgement success remains unchanged,
covered cancellation still propagates after a single transfer callback, and
pre-transfer cancellation still releases A for an exact retry. No controller
ledger, task state, general lesson, schema, persistence owner, or ADR changed.
