# Console Behavior and Storage Settings review — 2026-09-17

TASK-32761, starting from `96d4ca2b96` on draft PR #2704 against dev.
Existing ADR-004 preserves the next-launch Storage boundary; ADR-150/161 govern
control tokens and layout; ADR-020 supplies the established immediate-save
precedent. No schema, provider, permission or storage-migration boundary changes.

## Repairs

Compact Console/Storage rows now stack labels over full-width fields. Console
checkboxes use the established compact border/spacing contract, and three labels
are shortened while explanatory text remains. Textual checkboxes always measure
one content row; enabling wrapping alone did not make long labels visible.
The context/memory jump can wrap. Source styles were rebuilt, including the small
shared-field change in the boot bundle and the lazy Settings sheet.

Remote-image and status-row buttons previously ignored failed writes and allowed
older threaded writes to overwrite a newer choice. Their app-owned, per-preference
queue now serializes/coalesces admitted writes across Settings removal/recreation,
refreshes the idle baseline after configuration reload, restores the last saved
runtime value after pre-replace failure, and retains a file-replaced value with
an explicit cache-refresh warning. Controls stay focused and editable. Completion
messages belong to the latest choice, including enable/disable/enable sequences.
The public structured configuration mutation boundary remains the writer.

Original Console side-chat, reasoning, placement and Storage assertions run under
the existing exact-node private-profile helper. The Storage shortcut fixture now
uses the profile chosen before imports. The production recovery guard is intact.
Storage application logic is unchanged: Check/validation creates nothing, and Save
changes next-launch defaults without moving files or reconnecting active handles.

## Automated evidence

**92 distinct targeted cases pass**, without counting overlapping attempts:

- `instant-final.txt`: 18 immediate-save cases, including failure/retry, held older
  writes, navigation, real screen removal/recreation, reload, cache-refresh failure,
  and latest-choice receipts. Successful persistence uses real private TOML; failures
  are injected at the public mutation boundary.
- `geometry-final.txt`: eight production-style field traversals, Console/Storage ×
  190×55/80×24 × dark/light. Checks focus, complete compositor containment, full
  control labels/selected options, at least 12 input cells, and visible value suffixes.
  The local reasoning target is present so its override controls are enabled.
- `staged-recovery-final.txt`: four staged keyboard journeys, Console paste threshold
  and Storage workspace path at both sizes, plus four overlapping compact paint
  cases. Validates invalid-save refusal, navigation retention, cancel/confirm Revert,
  failed save retention, retry and persisted values. Storage checks/saves leave the
  destination directory absent and preserve the existing factory app DB identities.
- `originals-final.txt`: 30 original Console/Storage/remote-image cases with their
  original payload, runtime, failure, validation and Revert assertions retained.
- `governance-final.txt`: 32 token, component, generated-bundle and boot-byte checks.

Red logs preserve the ignored failure, stale-write, screen-departure and stale-receipt
reproductions. `test-log-manifest.json` pins original and trailing-whitespace-normalized
log bytes. Six derived-artifact checks pass in `preflight.txt`; its Mermaid input
fetch was unavailable in the restricted run. `mermaid-final.txt` records the seventh
check passing with declared hash-verified public inputs. No artifacts were regenerated
to mask that environmental failure.

Ruff check/format pass for the four focused test files and final native runner.
The two large legacy files retain their pre-existing lint findings (69 and 9 unique
findings respectively); `static-large.json` verifies none introduced. The diagnostic
inventory changes only for two removed generic warning calls, replaced by visible
structured save results (65 Settings calls; total TASK-494 calls 7761). The fresh
316-ref/29-worktree scan finds TASK-32761 only in this worktree.

## Native evidence

Final run `/private/tmp/tldw-32761-native-004`, PID 14447, passed all four cells.
All eight SVGs and matching terminal captures were inspected. `native-result.json`
pins the exact runner and production sources; `capture-manifest.json` pins capture
bytes. `lifecycle.json` proves app.run returned, exit 0, PID absent, instance lock
reacquired, owned terminal closed, structured app_stopping present, no ERROR log
records, empty faulthandler, 11 healthy private databases, zero conversations/messages,
and unchanged default-profile fingerprints.
The runner selects private HOME/USERPROFILE/config/data before imports, validates
all data roots, probes terminal capabilities, and runs the real TldwCli with
LinuxDriver and all rendering streams attached to an owned tmux terminal.

Each dark/light × 190×55/80×24 cell uses real Settings navigation, Console threshold
validation and confirmed Revert, both immediate toggles saved and restored, and
Storage invalid-path refusal, non-mutating Check, category navigation, confirmed
Revert and a real private next-launch Save. Seven active database object identities
are checked after each Save; target directories remain absent. Eight SVGs and
matching terminal captures show Console and Storage states.

Attempts 001/002 remain diagnostic evidence: 001 sent a repeat Enter during the
Button's 0.2-second active effect; 002 checked label paint under stacked transient
notifications. Both exited normally with failed assertions. The runner waits for
button readiness and unobscured paint. Attempt 003 passed four cells before final
runner lint cleanup; the final run uses the exact lint-clean runner saved here.

Independent review found and then cleared screen-lifetime, reload-baseline and
latest-receipt ownership issues. Its final read-only probes also confirmed preservation
of equal-value cache warnings and failure receipts. Native proof is separate from
those injected persistence interleavings.

## Remaining scope

This qualifies the reviewed controls and representative workflows, not every
Console feature. Permission summaries, model-thinking visibility under departure,
exchange-capture execution/consent, agent budgets, backgrounds and context/memory
flows still need bounded behavioral reviews. No LLM call, remote-image fetch,
permission-summary request, storage migration or restart into the new paths was
performed. The native flow does not inject disk failure; automated cases do.

The broader feature/component review remains active. PR #2704 stays draft; the user
requires visual confirmation of conflict choices before approving a merge into dev.
See the separate [historical conflict review](../../reports/2026-09-17-pr-2704-conflict-review.md).

## Captures

| Theme / terminal | Console | Storage |
| --- | --- | --- |
| Dark 190×55 | [View](textual-dark-190x55-console.svg) | [View](textual-dark-190x55-storage.svg) |
| Dark 80×24 | [View](textual-dark-80x24-console.svg) | [View](textual-dark-80x24-storage.svg) |
| Light 190×55 | [View](textual-light-190x55-console.svg) | [View](textual-light-190x55-storage.svg) |
| Light 80×24 | [View](textual-light-80x24-console.svg) | [View](textual-light-80x24-storage.svg) |
