# Import provider recovery continuity — TASK-32698

Baseline: `69060a097c` on `feat/component-pattern-library`.

A successful GGUF picker result rebuilt the entire Library screen. The result
handler now uses the existing in-place audio/video option synchronizer, which
updates provider status and the chooser without replacing the next draft's
editors. The existing queue listener handles retry transitions; a removed row
control falls back to the source field. The synchronizer's route guard leaves
newer navigation intact, and current focus remains authoritative.

ADR required: no. This repairs presentation under
[ADR-041](../../../../backlog/decisions/041-direct-local-gguf-before-managed-acquisition.md),
[ADR-150](../../../../backlog/decisions/150-design-token-system-and-design-language.md)
and [ADR-161](../../../../backlog/decisions/161-component-pattern-library.md).
Model admission, configuration persistence, retry authority and execution
boundaries are unchanged. No stylesheet or token value changed.

## Verification

[Verification](verification.json) records **104 passing targeted checks**:
16 new recovery journeys and 88 neighboring queue, option, provider and
governance checks. The new journeys open the real mounted GGUF picker from both
the failed row and provider form, return through its callback and real worker,
and cover cancellation, rejection and success at 170×48 dark and 80×24 light.
They preserve draft text, selection, widget identity and painted focus; delayed
success also respects a move to Author or the Library hub. Explicit
faster-whisper recovery exercises the production app routing and real registry
requeue while replacing parse-pool dispatch.

[Regressions](regressions.json) records the original mounted-canvas replacement
failure and the fixture corrections. External model admission/persistence is
replaced deliberately; these checks qualify presentation and routing, not GGUF
admission, model execution or transcript correctness.

[Static comparison](static-comparison.json) records zero new Ruff diagnostics.
The 205 inherited LibraryScreen diagnostics remain; new Python files are clean
and formatted. [Size comparison](size-comparison.json) records unchanged line
and function counts. The inherited LibraryScreen size-ceiling failure was not
rerun or claimed repaired; no budget was raised. [Independent review](review.json)
found no actionable code issues. No full repository suite ran. Existing pytest
cleanup and component-governance SyntaxWarnings remain.

## Native evidence and limits

[native_check.py](native_check.py) uses real TldwCli/LinuxDriver, a private
exclusive profile, private database paths and null keyring. It keeps the actual
picker, worker, result handler, provider retry and registry routing. Only model
admission/persistence and parse-pool dispatch are replaced. The synthetic queue
has neither a store nor a runner; no model, import or network operation runs.

The journey covers row cancellation, rejected selection, successful form
configuration, successful row retry and explicit faster-whisper retry in both
terminal sizes. Full draft values and selection are checked independently from
visible substrings: long single-line fields scroll horizontally.

Final run-004 passed all **10 journeys** and exited normally with status 0:
[result](result.json), [isolation](isolation.json), [lifecycle](lifecycle.json).
Six final captures were rendered and inspected in the confirmation batch;
[inspection](inspection.json) records the observations and separate picker
finding. [Persistence](persistence.json) records ten healthy private databases,
zero media/messages/ingest jobs, unchanged fixture bytes and default-profile
hashes, and no app ERROR/CRITICAL log lines. Full logs and private databases
remain in ignored scratch.

| State | 170×48 dark | 80×24 light |
|---|---|---|
| GGUF picker | [capture](picker-170.svg) | [capture](picker-80.svg) |
| Provider configured, chooser focused | [capture](configured-170.svg) | [capture](configured-80.svg) |
| Retained draft after retry | [capture](draft-after-recovery-170.svg) | [capture](draft-after-recovery-80.svg) |

**Separate finding:** the filtered file picker's compact footer squeezes the
filename input to about one column and puts Cancel outside the dialog. Escape
still cancels and keyboard path submission reaches the result handler. This is
a shared picker layout defect, retained in the compact capture and recorded as
a separate Backlog repair; this task does not claim that picker layout is fixed.

Earlier native attempts corrected runner assumptions: clearing the audio source
can replace its path widget, a long title need not fit wholly inside an Input,
and transient Toasts can cover compositor text. The final runner reacquires the
mounted path field and waits for notices before asserting unobscured path text.
The actual optional terminal probe runs before app startup, qualifying
probe-primed startup. No restart, push or dev integration was performed.
