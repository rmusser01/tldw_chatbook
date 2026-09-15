# Library audit evidence

See the [audit report](../../reports/2026-09-14-library-workflow-audit.md) for findings, priorities and coverage limits. Baseline: `2939afda63`.

## What is retained

- `rail-fold-focus-*`: 80×24 production-styled screen captures and measured focus/cue geometry in both themes. The focused Create toggle is hidden beneath the cue.
- `workspace-projection-mismatch*`: 120×45 production-styled screen with a correct four-source projection and a stale painted Handoff row.
- `notes-*` and `media-*`: representative responsive reader captures. Native Notes ANSI files show real local save, resize restoration and reopening.
- `rail-*.ansi`: native Search/RAG focus, hidden Create focus, and the collapsed Create section after Enter followed by widening.
- `query-cost-*.json`: nearest Library production query call sites in the two failing budget tests. Counts are not latency measurements.
- `test-results.json`: exact summary lines and failing node IDs from the targeted runs, including overlapping diagnostic reruns. Raw logs are retained locally under ignored `.superpowers/sdd/2026-09-14-library-ui-audit/`.
- `startup-exit.ansi` and `native-log-review.txt`: native exit code 0, startup warnings, and the one application ERROR record. Rendering successfully did not imply a clean startup log.

SVGs are Textual exports; TXT files are compositor strips. Trailing line padding and final blank lines in TXT/ANSI files were trimmed for repository whitespace hygiene; JSON paint crops preserve measured spaces exactly. All five retained SVGs were rendered with macOS Quick Look and visually inspected. The native ANSI captures provide an independent terminal check. The complete 20-case matrix and private app databases remain in scratch and are not included here.

## Reproduce the targeted selection

Run from the repository root in the project's Python environment. This is the original targeted selection, **not** a full suite. It is expected to reproduce the failures at the audited baseline.

```python
from pathlib import Path
import subprocess

selection = Path("Docs/superpowers/qa/2026-09-14-library-workflow-audit")
for name in ("workflow-selection.txt", "rag-responsiveness-selection.txt"):
    nodes = (selection / name).read_text().splitlines()
    subprocess.run([".venv/bin/python", "-m", "pytest", "-q", *nodes], check=False)
```

## Disposable differential probes

The two Python files under `probes/` are historical diagnostic sources, not newly installed regression tests. To reproduce them, copy both files to `Tests/UI/` with their existing names. They rely on the existing test fixtures and pytest isolation there; do not run them directly from this evidence directory. Remove only those copied files when finished.

```text
.venv/bin/python -m pytest -q Tests/UI/_library_audit_capture.py
.venv/bin/python -m pytest -q Tests/UI/_library_audit_diagnostics.py
```

The capture probe uses service fixtures with production Library styles and records 120×45 → 80×24 → 120×45. Its final source corrects the first run's empty-Search setup mistake. The diagnostic probe changes only the Workspace harness stylesheet list, instruments the existing query-count tests, and asserts expected focus paint and Workspace projection/receipt agreement. Its failing assertions intentionally expose the audit findings; they are not evidence of a new implementation regression. Import ordering, formatting and dictionary-literal style were normalized when archiving the probes; their diagnostic behavior is unchanged.

## Native reproduction

The audit ran the actual app in tmux at 120×45 with a private config redirecting all configured database paths and base data directories. For the focus defect: open Library, expand the full tool rail, resize to 80×24, Tab to Search/RAG, then Tab once more. Create is focused invisibly; Enter collapses it, revealed after resizing back to 120×45.

For Notes: create a blank note, enter the title “Library audit evidence” and body “Audit-only note. Verify retained text after resizing and reopening.” Wait for Saved, resize 120→80→120, Escape to the list, and Enter to reopen. The retained files show the title and body in each state. The audit's private terminal session was closed after the app exited normally.
