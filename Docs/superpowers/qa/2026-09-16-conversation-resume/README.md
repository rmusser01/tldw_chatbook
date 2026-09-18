# Conversations exact Resume — 2026-09-16

TASK-32705 qualifies active local conversations through Library's **Resume
conversation** action. Four cold loads and four existing-tab returns passed
across textual-dark/textual-light at 170×48 and 80×24. The original identity,
older selected branch, newer off-path sibling and an unrelated populated draft
are preserved. No production code or CSS change was needed.

ADR required: no new ADR. This verifies existing
[ADR-147](../../../../backlog/decisions/147-conversation-archive-and-exact-resume.md)
and [ADR-086](../../../../backlog/decisions/086-library-adaptive-reader-shell.md).
The Impeccable audit workflow is adapted to Textual focus, compositor paint,
terminal geometry and truthful state; no mobile/accessibility score is claimed.

## Native qualification

The [runner](native_check.py) uses real TldwCli, LinuxDriver, Library navigation,
the Resume handoff, Console store and SQLite. Private configuration/data paths
are validated before app imports, the imported checkout is verified and the
profile lock must be exclusive. No generation is requested.

[Fixtures](fixtures.json) contain four distinct global conversations. Each has
a user question and two assistant siblings with increasing timestamps; the
persisted active leaf points at the **older** answer. The fresh native process
starts with an unrelated Console tab. Paste is delivered through the app to its
focused composer, containing brackets and an accented character.

For each theme/size cell, [run-005](result.json) proves:

1. No Console session yet owns the fixture. Library is opened with Ctrl+3,
   filtered to that fixture, and its actual row is focused and opened with Enter.
2. The loaded original reader enables Resume. Its focused label is painted;
   Enter activates the production action. The new handoff revision must reach
   `settled`, and Console must activate the exact persisted conversation ID.
3. The store contains exactly the two expected active-path message IDs, roles
   and bodies, with the older answer also painted. All three tree nodes remain
   loaded; sibling enumeration reports index 0 of 2 and both original IDs.
4. The original unrelated tab is activated through its real tab button. Its
   complete canonical draft and stored draft match the original string, and its
   text is painted, wrapping to three lines at 80×24.
5. The same Library Resume action is repeated from the unrelated tab. Console
   reuses the exact session ID, adds no duplicate tab and preserves the draft
   again when its tab is revisited.

Here **cold** means a saved conversation with no existing Console session in
this process. It does not mean restarting the application between matrix cells.
Warm reuse checks the same selected branch; this runner does not switch siblings.
Programmatic focus followed by Enter exercises mounted actions, with painted
focus checks. Complete Tab traversal, the `c` shortcut and mouse interaction
are not established by this run.

## Persistence and lifecycle

[Before](source-before.json) and [after](source-after.json) record complete source
conversation/message snapshots. They match without ignored fields. After normal
Ctrl+Q exit, the independent [checker](verify_profile.py) opens SQLite read-only
and compares every field in those snapshots against storage, normalizing only
equivalent datetime string encodings. [Persistence](persistence.json) confirms:

- Exactly four original conversations and 12 original messages; no copied chats
  or new messages, and unchanged versions, active leaves, ownership and content.
- All ten private databases pass `PRAGMA integrity_check`.
- The default profile's config, UI state and runtime policy hashes are unchanged.
- Zero ERROR/CRITICAL log lines, traceback headers or faulthandler output.

[Lifecycle](lifecycle.json) records PID 14457, normal `app.run()` return, shell
exit 0 and independent process-absence verification. Only the owned terminal
was closed afterward. The populated draft is qualified during navigation in
this process; draft durability across application shutdown is outside scope.

Four [exploratory attempts](exploratory-runs.json) exited normally with status 1
while developing the harness: a method/property mismatch (plus an early stored
draft check), the remaining method mismatch, JSON datetime serialization, and a
single-line paint assertion against a wrapped draft. They are not counted as
successful qualification. Review also replaced an insufficient `has_pending`
check with exact revision settlement before the final run. No product behavior
was changed to make these probes pass.

## Capture review

All twelve captures were rendered and inspected in one batch.
[Inspection](inspection.json) records observations; [hashes](capture-hashes.json)
distinguish raw captures from stored SVGs with trailing source whitespace removed.

| Theme and size | Library Resume | Original branch | Unrelated draft |
| --- | --- | --- | --- |
| Dark 170×48 | [Reader](textual-dark-170-reader.svg) | [Console](textual-dark-170-resumed.svg) | [Draft](textual-dark-170-draft.svg) |
| Dark 80×24 | [Reader](textual-dark-80-reader.svg) | [Console](textual-dark-80-resumed.svg) | [Draft](textual-dark-80-draft.svg) |
| Light 170×48 | [Reader](textual-light-170-reader.svg) | [Console](textual-light-170-resumed.svg) | [Draft](textual-light-170-draft.svg) |
| Light 80×24 | [Reader](textual-light-80-reader.svg) | [Console](textual-light-80-resumed.svg) | [Draft](textual-light-80-draft.svg) |

The focused Resume action, selected answer and restored draft are readable in
both themes/sizes. At 80×24 Items is explicitly collapsed to expose the reader;
its transcript is below the viewport while the action is focused. Console hides
the context rail, truncates long tab labels and scrolls the tab strip. The active
conversation title and selected answer remain visible. No actionable Resume
defect was observed in these states.

## Automated verification and limits

The [baseline](baseline-targeted-tests.txt) had 80 passes and three stale failures.
The two mounted restore-and-resume cases called a removed private submit method;
they now activate the actual Send button and wait for the expected durable answer.
Existing exact-history, original-ID, archive-state and unrelated-draft assertions
remain. The consumer-registration census now includes the existing exact Resume
consumer, retaining its check for missing or unexpected registrations.

The [final targeted selection](targeted-tests.txt) passes **83 tests in 76.07s**,
exit 0. It covers branch reconstruction, recovery failures/navigation ownership,
consumer registration and mounted recovery. The send cases use a deterministic
fake provider and real storage. [Verification](verification.json) records the
command and static checks; [review](review.json) records independent review.
No full suite was run.

Native fixtures do not qualify archived workspace restoration, rename collisions,
roleplay/server conversations, live or queued work, attachments, staged sources,
thinking/provider continuation or a real provider send. The targeted tests have
their own narrower fixtures and do not expand this native claim. Terminal
capability probing is primed before app input ownership; ordinary unprimed startup
timing is also outside scope.

## Reproduce

Prepare a new no-secret private profile with `data/`, `data/db/`, `config/`,
catalog refresh disabled, and every configured database/data path beneath it.
The runner reuses the fail-closed [profile guard](../2026-09-16-ingest-lifecycle/native_check.py)
and refuses an existing evidence directory. In an owned tmux terminal sized
170×48, set `PYTHONPATH` to the checkout and run:

```text
.venv/bin/python Docs/superpowers/qa/2026-09-16-conversation-resume/native_check.py PROFILE SOCKET SESSION
```

Record the shell exit and independently verify process absence. Retain a
pre-launch `default-before.json` mapping the three default-profile file paths
to SHA-256 hashes (or null if absent), then run:

```text
.venv/bin/python Docs/superpowers/qa/2026-09-16-conversation-resume/verify_profile.py PROFILE
```

Raw private run data remains at `/private/tmp/tldw-32705-run-005` on this host.
Next bounded journey: Conversations **Use as source**, whose context-staging
contract is separate from exact Resume, followed by remaining Library destinations.
