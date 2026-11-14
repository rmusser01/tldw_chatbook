# Notes export qualification

Baseline: merged dev `7884e11a496a677c8180cc5d67234a57b7d659b9`.
No product or existing tracked test was modified.

The final `book3283-notes-03` control uses the production Library screen on the
established bundle-CSS LibraryHarness, seeded Notes service stand-ins, actual
Export Markdown Button, native FileSave filename/Tab/Enter journey, native App
event handler, and real atomic writer. It settles source workers before opening
the editor; the visible source Retry is available if needed. Neither final
timeline required Retry. Unrelated unavailable optional count backends are
omitted. This does not assert a real Notes SQLite or full live TldwCli journey.

Ordinary export passes and the writer runs once on the main thread. The held
case reaches that same writer on the main thread (both recorded thread IDs
130324), posts an App F12 event, and waits for the actual App handler to release
it. Because the write blocks the loop, the three-second watchdog releases it
first. Recorded order:

1. Writer entered and key posted: 608201.687.
2. Watchdog released writer: 608204.687.
3. Real atomic write completed: 608204.703.
4. App key acknowledged: 608205.203, before-release false.

Correct captured Markdown, Export complete status, one writer invocation and no
temporary residue verify after release. The final expected assertion is
`[False] == [True]` for dispatch before release. Parent and actual child XML
agree: two cases, one ordinary pass, one intended failure, zero errors/skips.
The parent run took 260.66 seconds including private-process/fixture overhead.
This is a scheduling-mechanism witness, not a three-second responsiveness SLA,
natural storage timing distribution or source-database mutation claim.

`qualified/` contains losslessly archived actual XML plus original report hashes,
command/verification receipts and both writer observations. The root archive map
records compressed/decoded SHA256. The retained source is `test_note_export.py.txt`.

Notes01 failed startup before reaching either writer. Notes02 passed ordinary
export, but its held arm again failed source readiness. Both are excluded from
the blocking-dispatch conclusion. Their raw reports are retained under the
excluded directories. The exact cause of those startup deadline misses remains
unqualified; the older source's optional-error-formatting comment is not causal
evidence. No product deadline or privacy guard was changed.

Current path: `UI/Screens/library_screen.py:21886–21967` supplies the validated
FileSave callback; `UI/Library_Modules/library_file_export.py:344–429` synchronously
invokes its Notes writer. The Prompt sibling already admits a native threaded
write worker under TASK-37033. TASK-34000.3 retains Replace/atomic/no-clobber
outcomes; remote-only TASK-34000.97 owns a different destination-error boundary;
TASK-2834 excludes text writes. PR3096 changes source/newline fidelity but leaves
this writer synchronous. TASK-37268 owns the bounded Notes scheduling outcome
under existing ADR-033/036/031; no new ADR is expected.
