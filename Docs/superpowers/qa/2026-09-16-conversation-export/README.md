# Conversations Export — TASK-32704

Conversations Export passes the bounded native journey at `faed7c3d3d`:
whole-source and selected-row scopes write the advertised contents, the real
destination picker supports Cancel and `.zip` normalization, an explicit submit
replaces the chosen disposable archive, and Escape returns to the retained query.
No production code or CSS change was needed. Three stale automated failures were
resolved by updating two test files to the existing behavior.

ADR required: no. Existing [ADR-011](../../../../backlog/decisions/011-chatbook-workbench-ui-system.md),
[ADR-147](../../../../backlog/decisions/147-conversation-archive-and-exact-resume.md)
and [ADR-150](../../../../backlog/decisions/150-design-token-system-and-design-language.md)
govern this verification. The Impeccable audit was adapted to Textual controls,
terminal geometry and compositor paint; no mobile or overall accessibility score
is claimed.

## Scope and native results

The [runner](native_check.py) uses real `TldwCli`, its LinuxDriver, FileSave,
export workers, LocalChatbookService and SQLite. It validates private data paths
before importing the app, verifies the imported checkout and requires exclusive
profile ownership. It seeds Alpha and Beta as active conversations, Gamma as
archived, and one unrelated note. Each conversation has a User/Assistant pair
with brackets, an accented character and a newline; [fixtures](fixtures.json)
record exact IDs and bodies.

[Run-002](result.json) covers textual-dark and textual-light at 170×48 and 80×24.
Each of the four cells performs two real writes:

1. Filter Conversations to `Alpha`, open Export, and verify the form advertises
   **Conversations · 2 items**. This existing whole-source contract ignores the
   text filter and includes both active conversations; archived Gamma is excluded.
2. Cancel the real destination picker and verify no destination is accepted.
   Choose a private `.txt` path and verify normalization to `.zip` before submit.
3. Export, inspect the ZIP, and match the receipt's item count and byte size to
   the written artifact. Escape returns to the same `Alpha` query and one result.
4. Check Alpha and use Export selected. The form advertises one selected
   conversation. Choose the same `.txt` path: the form names the normalized
   existing ZIP in its **Overwrites** notice, while that file remains unchanged.
5. Submit deliberately, verify the replacement holds only Alpha, and return to
   the same query again.

Overwrite has an inline informational notice and the ordinary Export submit;
this flow does not introduce or claim a separate confirmation dialog. The prior
successful receipt remains visible until the next successful write replaces it.

The native runner compares complete source conversation/message snapshots before
and after. Its Console check compares active session ID, session ID list and the
active draft within this process. The fresh profile's draft is empty; populated
drafts, attachments and staged source context are not qualified here.

## Artifact and persistence checks

After normal exit, the independent [artifact checker](verify_artifacts.py)
reopened preserved copies of all eight writes. The [receipt](artifact-verification.json)
checks archive integrity, exact manifest length and IDs, no duplicate ZIP entries,
and each conversation's exact message length, ordered IDs, bodies, roles and
order fields. This strengthens the native runner's ID-set/body-map checks, which
alone would not detect duplication or reordering. All four final destination
files match their selected-export hashes. Neither archived Gamma nor the
unrelated note appears in these bundles.

[Lifecycle](lifecycle.json): PID 71650 returned normally from `app.run()`, exited
0, and was independently confirmed absent before only its owned terminal was
closed. [Persistence](persistence.json): all ten private SQLite databases pass
integrity checks; Alpha/Beta remain active at version 1, Gamma remains archived
at version 2, all six source messages are unchanged, and the unrelated note is
intact. The default profile's three configuration/state hashes are unchanged.
The app log has no ERROR/CRITICAL lines or traceback headers; faulthandler is empty.
No external model request was made.

An early native attempt stopped on a harness-only comparison of a multiline
marked-up row label against compositor text. It exited 1 normally and its PID
was confirmed absent. The final run uses the identifying title plus focused-widget
checks on a fresh profile. A transient post-exit SQLite opening failure and its
successful final read-only verification are separately recorded in
[verification](verification.json); neither is counted as a successful initial run.

## Capture review

All twelve SVGs were rendered and inspected in one round. [Inspection](inspection.json)
records raw/stored hashes; only whitespace-only SVG source lines were normalized.

| Theme and size | Whole-source receipt | Before replacement | Selected receipt |
| --- | --- | --- | --- |
| Dark 170×48 | [2 items](bundle-textual-dark-170-whole-receipt.svg) | [Overwrite](bundle-textual-dark-170-overwrite.svg) | [1 item](bundle-textual-dark-170-selected-receipt.svg) |
| Dark 80×24 | [2 items](bundle-textual-dark-80-whole-receipt.svg) | [Overwrite](bundle-textual-dark-80-overwrite.svg) | [1 item](bundle-textual-dark-80-selected-receipt.svg) |
| Light 170×48 | [2 items](bundle-textual-light-170-whole-receipt.svg) | [Overwrite](bundle-textual-light-170-overwrite.svg) | [1 item](bundle-textual-light-170-selected-receipt.svg) |
| Light 80×24 | [2 items](bundle-textual-light-80-whole-receipt.svg) | [Overwrite](bundle-textual-light-80-overwrite.svg) | [1 item](bundle-textual-light-80-selected-receipt.svg) |

Receipts, overwrite notices and focused submit controls are readable in both
themes. At 80×24, the form scrolls to the focused action and long paths wrap;
the header/scope can be above that viewport. No actionable defect was observed
in the captured states. Controls are explicitly focused and then activated with
Enter, with a Tab step from the picker filename to Save. Complete Tab traversal,
mouse navigation and automatic focus placement are not established by this run.

## Automated verification and test repairs

The [baseline selection](baseline-targeted-tests.txt) returned **120 passed,
3 failed**. The failures predate this task:

- A fake-self row test lacks `query`, now used for the archive/restore controls.
  Its assertions were moved into the existing mounted selection journey, which
  now verifies checked identity, unchanged loaded transcript, disabled reader
  actions and enabled bulk Export/Archive/Restore controls. This removes the
  duplicate fake test while preserving its behavior checks.
- Two destination tests expected an untouched form plus a toast.
  [TASK-32251](../../../../backlog/tasks/task-32251%20-%20Library-Notes-path-fields-concatenate-a-typed-absolute-path-and-the-export-destination-then-accepts-the-corrupted-value-unvalidated.md)
  instead clears a refused destination and stores its inline reason. The updated
  tests check that contract, retain unrelated form fields and still drive the
  real shared validator for the traversal-shaped path.

The [final selection](targeted-tests.txt) passes **122 tests in 40.39s**, exit 0;
[verification](verification.json) records the exact eight-file command. Existing
tests cover cancellation ownership, stale callbacks, destination validation,
uncapped scope resolution, receipts and real service/importer round trips.
[Lint comparison](lint-comparison.json) shows no new diagnostics; the multi-select
file retains its one inherited finding. Changed-range/helper formatting and
evidence checks are in [static checks](static.json). [Independent review](review.json)
records the final assessment.

The tiny native exports finish too quickly to qualify cancellation during an
active write or failures from disk exhaustion/permissions. Selected archived
exports, attachments, branching/variant graphs, citations, private provider
continuation, thinking payloads, arbitrary library sizes and exact Resume remain
outside this native fixture. The automated round trips have their own narrower
fixtures; they do not expand this native claim. No full suite was run.

## Reproduce

Prepare a new private profile with `data/`, `data/db/` and `config/`, every
configured database/data path beneath it, no secrets and catalog refresh
disabled. The runner reuses the fail-closed [profile guard](../2026-09-16-ingest-lifecycle/native_check.py)
and refuses existing evidence/export directories. In an owned tmux session
sized 170×48, set `PYTHONPATH` to this checkout's absolute root and run:

```text
.venv/bin/python Docs/superpowers/qa/2026-09-16-conversation-export/native_check.py PROFILE SOCKET SESSION
```

Record the shell exit and confirm process absence, then run:

```text
.venv/bin/python Docs/superpowers/qa/2026-09-16-conversation-export/verify_artifacts.py PROFILE
```

Keep independent read-only database, log and default-profile checks alongside
the runner receipts. The terminal capability probe is primed before stdin
ownership, so ordinary unprimed startup timing is not qualified. Raw ZIP copies,
logs, databases and rendered PNGs remain in disposable audit storage.
