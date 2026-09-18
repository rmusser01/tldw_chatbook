# Model-thinking visibility Settings review — TASK-32763

This review covers the canonical immediate-save checkbox in Console Behavior.
It preserves ADR-090's device-local presentation contract: hiding rows does not
change stored thinking, public answers, exchange capture or replay policy. No
provider request was issued. The broader feature review remains open, and draft
PR #2704 still requires the user's visual review and explicit merge approval.

ADR required: no. This repairs existing behavior under ADR-090, ADR-150 and
ADR-161 without changing storage, capture, provider or replay boundaries.

## Repair

The existing app-owned Console toggle queue now owns thinking-visibility writes.
It serializes the newest choice across actual Settings removal/recreation and
rebases the saved value after a configuration reload. The obsolete screen-local
writer was removed. Failed writes restore the confirmed value and allow a new
keyboard toggle to retry; saved-file/cache-refresh warnings retain the saved
value. A superseded completion cannot replace the current outcome.

The outcome also lives with the app-owned state. A dedicated token-backed status
row restores it beside the checkbox on return, avoiding unrelated staged-field
events overwriting the visible receipt. Compact failure copy fits the row's
content width, including its existing padding. No stylesheet or token changed.

## Automated evidence

`ui-final.txt`: **49 affected Settings/transcript cases pass**, including shared
immediate-toggle regressions. Together with 32 governance cases below, **81
distinct targeted cases pass**.
The 15 new private-profile journeys cover reload, overlapping writes, screen
recreation, completions while Settings is absent, failed writes and keyboard
retry at both widths, cache-publication warnings, and mounted Console rows.
Existing visibility tests now intercept the config-writer boundary instead of
the removed private worker; their value, refresh-count, no-op, ordering and
rollback assertions remain. Two transcript tests use private process profiles
to respect interpreter-lifetime profile binding. No production safety guard was
disabled. Shared remote-image/status-position queue regressions are included.

`governance.txt`: 32 token/component, bundle-sync and boot-byte checks pass.
All seven derived-artifact guards pass in `preflight.txt`; the diagnostic inventory
has no drift. Changed tests and the native runner pass Ruff check and format.
The Settings owner retains its 116 inherited lint findings and 83 inherited
format edits, with none introduced (`static-large.json`); source compilation
passes. Read-only independent review is clear (`review.txt`). The final task-ID
scan found this task only in this worktree across 314 refs and 29
worktrees. No full test suite was run.

Negative evidence is retained:

- `red-reload-departure.txt` demonstrates stale disk values after config reload
  and screen departure. Its cache-warning variant used a call counter across a
  held write; that fixture was corrected to capture the call ordinal before
  waiting. Do not use that original variant as independent cache-failure proof.
- `red-receipt.txt` demonstrates outcomes lost on Settings recreation. Its
  separate mounted-Console failure was a fixture error: it first used the wrong
  destination harness, then modified a returned message snapshot. The final
  fixture uses the production-style Console harness and canonical store mutation.
- `red-compact-paint.txt` proves the failure receipt's third row was clipped;
  `red-compact-padding.txt` rejects the first shortened copy because its content
  width was still too narrow. Final assertions require the whole receipt inside
  the compositor clip and the recovery instruction in painted text.

Failure/conflict/exception cases substitute the config-mutation boundary;
post-replacement cache failures use the real writer's publication callback.
Successful journeys inspect actual private TOML. Synthetic displayable thinking
qualifies presentation and stored-content preservation, not provider execution.

## Native evidence

Final private profile `/private/tmp/tldw-32763-native-003`, PID 50930, passes
dark/light × 190×55/80×24. `native-result.json` pins the exact runner and source
hashes. Real `TldwCli.app.run` uses LinuxDriver, with all rendering streams attached
to the owned tmux terminal and private HOME/config/data selected before imports.

Each cell makes the private config directory read-only, verifies real write
refusal and rollback, restores permissions, retries by keyboard, leaves for
Console and recreates Settings, then verifies the saved value and retained
receipt. A real external config change followed by Diagnostics → Reload Config
rebases the next toggle correctly. Every cell restores the original Off value.
Seven live database object identities remain unchanged.

All eight final SVGs were visually inspected with their native terminal captures.
Both compact themes show the full checkbox and the complete rollback/retry text;
wide and compact saved states retain the receipt after Settings recreation.
The generic category banner continues to describe staged controls; the thinking
control's adjacent help identifies its immediate-save behavior.

`lifecycle.json` verifies normal app return, exit 0, PID absent, instance lock
reacquired, owned terminal closed, 11 healthy private databases, zero
conversations/messages, empty faulthandler logs and three unchanged default-file
fingerprints. The only four errors are expected `PrivatePathError` refusals:
one lock-phase and three pre-replacement errors. Native cache-publication failure
is not claimed; the mounted tests cover that boundary.

The `first-pass/` evidence records the initial clipped compact captures and the
failed containment confirmation (PID 49488, exit 1, clean resource release).
One launch was refused before app imports because private database parent
directories were missing; those directories were created and validated before
launching again. These unsuccessful attempts are not counted as passing native
qualification. Capture manifests preserve original and whitespace-normalized
stored hashes.

| Theme / size | Failed save and retry guidance | Saved after Settings recreation |
| --- | --- | --- |
| Dark 190×55 | [View](textual-dark-190x55-failed.svg) | [View](textual-dark-190x55-saved.svg) |
| Dark 80×24 | [View](textual-dark-80x24-failed.svg) | [View](textual-dark-80x24-saved.svg) |
| Light 190×55 | [View](textual-light-190x55-failed.svg) | [View](textual-light-190x55-saved.svg) |
| Light 80×24 | [View](textual-light-80x24-failed.svg) | [View](textual-light-80x24-saved.svg) |

Exchange-capture execution/consent, agent budgets, backgrounds, context/memory
and the other Settings categories remain separate review gates in the completion
ledger. This receipt does not qualify the whole Console destination.
