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
