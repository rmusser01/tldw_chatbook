# Library import recovery — TASK-32663

Reviewed on `feat/component-pattern-library`, based on `eda1d6bd65`.

Import media's fixed one-row Start explanation cut off recovery and consent
messages at compact widths. The gate now grows with its text while retaining
one minimum row when empty. Start remains in the docked commit bar; the screen
continues updating the existing widgets in place. No copy, submission policy,
authority, dependency or token value changed. Source CSS was rebuilt into the
Library screen sheet.

ADR required: no. Existing
[014](../../../../backlog/decisions/014-library-ingest-service-authority-and-recovery.md),
[150](../../../../backlog/decisions/150-design-token-system-and-design-language.md)
and [161](../../../../backlog/decisions/161-component-pattern-library.md) apply.

## Automated evidence

**271 distinct targeted checks pass.** [verification.json](verification.json)
records commands, run outcomes and the count basis. Eight baseline cases failed
on clipped missing-path, empty-folder, invalid-option and consent messages at
72×18 in both themes. All 24 new cases pass after the repair. Wide gate cases
use 170×48; full Library journeys use 170×48 and 80×24 in both themes.

The full Library journeys use real local file preflight and a temporary SQLite
media database. They check initial keyboard entry, Tab to Browse and Clear,
Enter on Clear, immediate typing into the returned path field, retained metadata,
and ready-state recovery without submitting a job. Existing keyboard, consent,
canvas, structural and token/bundle checks cover the neighboring behavior.

One existing check still fails: the Parakeet model-directory Browse button ends
at column 94 beyond its row's column 78 at an 80-column viewport. The same test
fails with all modified production files restored to the base commit. That
option-row issue belongs to the next review, not this gate repair.

Two test setup corrections are recorded rather than counted as product fixes:
the first full-screen fixture lacked a media database and exercised unavailable
state; a later lint cleanup used a tuple for Textual's list-valued `CSS_PATH`,
preventing 20 new cases from mounting. The final run uses a typed list and passes.
Two inherited pytest garbage-directory cleanup warnings remain.

[Static comparison](static-comparison.json) reports zero new Ruff diagnostics
(six inherited in the widget). New files pass full Ruff and formatting; changed
existing ranges are formatted. [Size comparison](size-comparison.json) confirms
the widget shrank by three lines and controller/screen budgets are unchanged.
The inherited LibraryScreen ceiling failure recorded by TASK-32662 was not rerun.
Independent read-only review found no actionable defect. No full suite was run.

## Native evidence

[native_check.py](native_check.py) ran actual TldwCli with LinuxDriver in an
[isolated profile](isolation.json), exclusive instance lock and null keyring.
It inspected one missing path, an empty folder and a 46-byte text file using
real preflight. Paths were assigned directly; metadata, Clear activation,
re-entry typing and navigation to Start used keyboard input. Start was focused
but never activated.

Six captures were rendered and inspected in one batched pass:

| State | 170×48 dark | 80×24 light |
|---|---|---|
| Missing-path recovery | [capture](missing-170.svg) | [capture](missing-80.svg) |
| Empty-folder recovery | [capture](empty-170.svg) | [capture](empty-80.svg) |
| Ready source, Start focused | [capture](ready-170.svg) | [capture](ready-80.svg) |

The compact empty-folder message wraps in full above Start. Native keyboard
Clear returns focus to the path; typing goes there, the metadata survives, and
Tab reaches a visibly focused Start. Re-entering Import preserves the staged
source and metadata. Run-001 stopped on the runner's incorrect expectation of
an empty placeholder after re-entry; corrected run-002 passes both sizes.

[Run result](result.json) and [read-only persistence checks](persistence.json)
confirm zero media records, messages and ingest jobs, ten healthy SQLite
databases, and unchanged source content. Normal terminal Quit returned exit 0
after repeating it once after the autopilot finished; the shell was observed
and the owned session closed. The app log has no ERROR/CRITICAL lines.

## Next review

Continue with per-type ingest options, starting with the inherited Parakeet
directory-row overflow, then queue activity and recovery. This slice does not
qualify file-picker operation, real import execution, remote authorities,
provider/extraction actions or restart. No push or integration into `dev`.
