# Local Import success and restart recovery — TASK-32700

Completed against `d576cb93c7` on `feat/component-pattern-library`, using four
fresh native app processes on one disposable profile. No application code changed
in this resumed qualification. The earlier semaphore failure did not recur.

| Acceptance criterion | Native evidence |
| --- | --- |
| Text/Markdown success and usable Open | [Success](success-result.json): exact persisted content and mounted Open at 170×48 and 80×24 in textual-dark/light. |
| Duplicate resolution and successful Retry | Alpha's second import resolves media 1 without adding content. A real permission failure followed by restored permissions and mounted Retry creates media 3 with durable lineage. |
| Clear, supported actions, interruption | Clear leaves no job/media. General text exposes no Cancel. [Interrupt](interrupt-result.json) quits during real worker chunking; [recover](recover-result.json) shows “Interrupted by app restart” and successfully retries it. |
| Fresh-process persistence | [Reopen](reopen-result.json) checks both retry histories and opens all three small fixtures in both themes/sizes. [Persistence](persistence.json) verifies all four exact contents and database integrity. |

The real application, parse pool, parser, writer and durable job registry run
unchanged. Mounted controls submit, clear, retry and open; the harness assigns
form values and uses native terminal resizing. No parser/pool/writer substitute,
worker pause, or forced process kill was used. Quit occurs after real worker IPC
reports chunking an 18 MB text fixture, with the job still parsing. Its durable
pre-restart state remains parsing with no media; the next process reconciles it
to an explicit retryable failure.

[Lifecycle receipts](lifecycle.json) record normal `app.run()` return, shell exit
0 and independently checked process absence for all four apps, including the
interrupted worker. Only the owned tmux session was closed after those exits.

Final read-only database checks found ten healthy private databases, four media
and four document versions with exact source hashes, and seven durable attempts
(five current rows plus two superseded failures). Both Retry successors retain
`retry_count=1` and their predecessor IDs. Only the large fixture has chunks:
2,778 unvectorized chunks, none processed. Analysis and embeddings stayed off;
conversations/messages remain empty. Source bodies and the three recorded
default-profile config/state hashes are unchanged.

## Verification and visual scope

[Targeted verification](verification.json) records **83 passed in 80.95s**, including
the real spawn-pool test, restore/database checks, retry/duplicate/shutdown paths,
entry/Recent imports journeys and the profile guard. No full suite ran.
[Independent review](review.json) has no remaining actionable harness findings.

[Inspection](inspection.json) records one batch of 16 final native captures:
completed queue, permission Retry, interrupted Retry and saved Markdown reader,
each at both sizes/themes. Open/Retry labels, focused actions, failure causes and
saved body text are visible. The permission summary abbreviates its long absolute
path; Show details content was not inspected. Compact captures scroll the queue
into view. Long reader metadata may wrap/clip, and the large fixture was checked
by its complete persisted content rather than rendered in full. These checks do
not qualify other import formats, external providers, transcription cancellation,
all navigation layouts, performance targets or crash/power-loss durability.

The private app log contains **three caught ChatScreen startup errors**:
`Failed to load sidebar state: 'ChatScreen' object has no attribute '_sidebar_state_save_timer'`.
They occur on the three restarts, outside the Import path; they did not prevent
these journeys. This is not a claim of error-free application startup. The log
contains no traceback header, and the private faulthandler log is empty. The
diagnostics are retained in [persistence.json](persistence.json).

Two earlier harness attempts are excluded from passing evidence: run-004 compared
the reader's normalized presentation with a trailing source newline, and run-005
assumed Escape returned from the reader to the Library hub. The final runner uses
normalized presentation plus a painted content assertion and mounted Import
navigation. Database content comparison remains exact. Both failed attempts
returned normally with exit 1 and process absence; their raw receipts remain in
ignored scratch. All four final run-006 phases used the same recorded runner hash.

## Reproduction

Use [native_success_check.py](../native_success_check.py), not the historical
failure-specific runner. Prepare a fresh private profile as described in the
[original profile setup](../README.md#historical-failure-runner), with every
configured database/data path beneath it, catalog refresh disabled and one parse
worker. Run from this checkout with `PYTHONPATH` set to its absolute root (the
recorded run exported that path before launching Python). The runner validates
isolation before app imports, verifies the imported application belongs to this
checkout, and refuses reused phase directories.

Create these UTF-8 fixtures under the profile's `sources/` directory:

```python
fixtures = {
    "alpha.txt": "Alpha import verification. This private local text must survive restart.\n",
    "beta.md": "# Beta import verification\n\nThis **Markdown** stays intact after a fresh process.\n",
    "permission.txt": "Permission recovered. This local file must import through Retry.\n",
    "interrupt.txt": "Interrupted import verification has a real parser and durable recovery.\n" * 250_000,
}
```

In an owned tmux session initially sized 170×48, run each phase as a separate
process in this order, waiting for the prior process to exit and recording its
exit code before continuing:

```text
.venv/bin/python Docs/superpowers/qa/2026-09-16-ingest-lifecycle/native_success_check.py PROFILE SOCKET SESSION success
.venv/bin/python Docs/superpowers/qa/2026-09-16-ingest-lifecycle/native_success_check.py PROFILE SOCKET SESSION interrupt
.venv/bin/python Docs/superpowers/qa/2026-09-16-ingest-lifecycle/native_success_check.py PROFILE SOCKET SESSION recover
.venv/bin/python Docs/superpowers/qa/2026-09-16-ingest-lifecycle/native_success_check.py PROFILE SOCKET SESSION reopen
```

The permission phase needs a normal user account on a filesystem honoring mode
bits. The interrupt phase asserts that the real worker is still parsing; on a
different machine, a fixture that finishes before Quit is not interruption proof.
Retain independent process/worker exit checks and read-only persistence checks
alongside the runner output. SVG source normalization only removes whitespace on
otherwise empty lines; [inspection.json](inspection.json) retains both hashes.

ADR required: no. This verifies the existing ADR-014/065/150/161 contracts linked
from the [parent record](../README.md); no authority, schema or UX boundary changed.
