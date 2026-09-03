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
- `ChatScreen.apply_navigation_context()` consumes an exact
  `ConsoleSettingsReturnTarget`, validates the claim coordinates and origin
  session/settings revision, restores the exact suspended modal snapshot and
  logical focus, then acknowledges only restoration or terminal rejection.
  Transient modal failures release the claim while retaining the exact safe
  retry target; a later mount or resume reclaims that same revision.

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
