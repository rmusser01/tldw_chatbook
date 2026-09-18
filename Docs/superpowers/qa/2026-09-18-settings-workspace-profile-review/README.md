# Workspace imported-profile review — TASK-32774

First-bind review now belongs to the Apply action that requested it. Leaving
and returning to the workspace/category, suspending Settings, or changing the
staged persona/profile invalidates a delayed review or confirmation token. The
owned review modal preserves that intent while still clearing the separate
memory acknowledgement. Already-submitted local saves keep their existing
semantics. Persona names render literally in the status and picker.

Existing ADR-107/079 govern binding authority and assistant defaults;
ADR-150/161 govern presentation. No new token, schema, permission rule or ADR
was required.

## Targeted verification

**70 distinct cases pass across final and corrective runs:** ten new cases,
28 existing assistant/memory/field-edit cases and 32 binding/modal cases.
[Four delayed-token cases](delayed-confirmation-tests.txt) and four delayed-review
cases in the [related run](related-tests.txt) cover the same four
navigation/staging boundaries. [Failed Clear](clear-test.txt) preserves the
existing memory confirmation and label. [Literal-name paint](literal-test.txt)
verifies both rendered consumers. The [binding matrix](binding-tests.txt)
retains exact-token, replay, expiry, changed-authority and memory boundaries.

The related run passed 27 cases and failed one test before it could press Apply:
it queried the control during pane replacement. Its only change is a bounded
pane-ready wait. The [final related run](final-related-tests.txt) passes that
case and four existing dark/light compact/wide field-edit journeys. Counts
include the corrected case once. All 15 original assistant-default test
functions and assertions remain intact.

[Four initial failures](delayed-review-red.txt) reproduce stale dialog
publication. [The literal-name failure](literal-red.txt) reproduces the
bracketed suffix disappearing from painted text. [Static checks](static.txt)
show no new legacy Ruff diagnostics, clean new files, scoped formatting and
diff hygiene. [Independent review](review.md) found and then verified the
failed-Clear correction; no introduced blocker remains. No full suite or
provider/tool execution was run.

## Native review

The [runner](native_check.py) starts real TldwCli/LinuxDriver with TTY-backed
rendering and a private HOME/config/data selected before imports. Fixture setup
exports the real local permission catalog, serializes an archive, inspects it
and installs a separate unbound imported profile for each cell through the real
ToolPackService and receipt store. The UI then stages that profile in Workspaces,
reads target and expanded policy details, cancels, returns to a changed-policy
review, rejects its stale revision, and retries with an exact successful bind.
Real registry and strict permission-store reads verify defaults and marker state.
The fixture has Ask/Deny rules and no Allow grants; detailed Allow semantics are
covered by the binding tests, not by these captures.

Controls are focused before real keyboard activation; Tab reaches the scroll
body, Enter expands details and End reaches the confirmation boundary. This is
first-bind UI evidence. Tool Profiles file-picker/import/export/removal journeys,
Persona auto-creation and project-context interviewing remain separate reviews.

| View | Dark compact | Light compact | Dark wide | Light wide |
| --- | --- | --- | --- | --- |
| Exact target | ![textual-dark 80x24](textual-dark-80x24-target.svg) | ![textual-light 80x24](textual-light-80x24-target.svg) | ![textual-dark 170x48](textual-dark-170x48-target.svg) | ![textual-light 170x48](textual-light-170x48-target.svg) |
| Policy details and confirmation boundary | ![textual-dark 80x24](textual-dark-80x24-policy.svg) | ![textual-light 80x24](textual-light-80x24-policy.svg) | ![textual-dark 170x48](textual-dark-170x48-policy.svg) | ![textual-light 170x48](textual-light-170x48-policy.svg) |
| Cancelled | ![textual-dark 80x24](textual-dark-80x24-cancelled.svg) | ![textual-light 80x24](textual-light-80x24-cancelled.svg) | ![textual-dark 170x48](textual-dark-170x48-cancelled.svg) | ![textual-light 170x48](textual-light-170x48-cancelled.svg) |
| Changed policy refused | ![textual-dark 80x24](textual-dark-80x24-stale.svg) | ![textual-light 80x24](textual-light-80x24-stale.svg) | ![textual-dark 170x48](textual-dark-170x48-stale.svg) | ![textual-light 170x48](textual-light-170x48-stale.svg) |
| Applied with literal name | ![textual-dark 80x24](textual-dark-80x24-applied.svg) | ![textual-light 80x24](textual-light-80x24-applied.svg) | ![textual-dark 170x48](textual-dark-170x48-applied.svg) | ![textual-light 170x48](textual-light-170x48-applied.svg) |

All 20 final SVG captures were rendered and visually inspected. Run003 used the
final source and runner hashes recorded in [native-result.json](native-result.json);
[capture-manifest.json](capture-manifest.json) records all 40 SVG/text hashes.
The [lifecycle receipt](lifecycle.json) verifies normal keyboard shutdown, exit 0,
PID 56632 absent before closing its owned terminal session, 11 healthy SQLite
databases, reacquired instance lock, zero durable conversations/messages, no
error or faulthandler output, and unchanged default-profile fingerprints.

Earlier run001 passed behavior but its captures exposed the missing bracketed
name; pre-literal receipts and one capture retain that evidence. Run002 stopped
when the probe checked a Collapsible heading before focus-scroll animation
settled. Escape/Ctrl+Q did not complete that failed probe; SIGINT returned from
the app with exit 1, absent PID, healthy private databases and unchanged defaults.
Its receipt explicitly does not claim successful keyboard shutdown. The final
runner waits for that animation and cancels an open review before failure cleanup.
