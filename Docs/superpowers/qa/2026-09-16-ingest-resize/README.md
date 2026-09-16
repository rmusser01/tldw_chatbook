# Import focus during terminal resizing — TASK-32697

Baseline: `d45f55cde0` on `feat/component-pattern-library`.

Resizing with focus inside Import replayed the Notes semantic focus tuple, which
fell back to a Library rail row. Import now shares the existing Prompt exclusion
from that replay. After its viewport changes, the canvas reveals the currently
focused control using its existing dock-aware scroll helper. The callback reads
current focus and respects movement outside Import.

ADR required: no. This restores existing behavior under
[ADR-086](../../../../backlog/decisions/086-library-adaptive-reader-shell.md),
[ADR-150](../../../../backlog/decisions/150-design-token-system-and-design-language.md)
and [ADR-161](../../../../backlog/decisions/161-component-pattern-library.md).
No CSS, token values, layout policy, persistence or queue execution changed.

## Verification

[Verification](verification.json) records **98 passing targeted checks**: 92
Import, Prompt, resize-cost and governance cases plus six Notes focus neighbors.
Twelve new cases cover metadata, options, queue Details and Recent imports in
both themes across 80×24, 170×48 and 170×24 transitions, plus newer focus inside
and outside Import. They assert widget identity, painted content, draft text,
selection, disclosure and registry retention, with no source refresh, preflight
reload or preference write. Fixtures use temporary SQLite and local preflight;
their synthetic queue has no worker.

[Regressions](regressions.json) records two load-bearing failures: the original
code moved title focus to the rail; removing only the new canvas resize reveal
kept focus but let the title move outside the painted viewport. A test fixture
initially targeted the compact rail container; it now targets its focusable
Open button. That fixture correction required no product change.

[Static comparison](static-comparison.json) records zero new Ruff diagnostics;
205 inherited LibraryScreen and six canvas diagnostics remain. New Python files
and changed production ranges are formatted. The LibraryScreen size ceiling
still fails: 35,215 lines versus 33,204 allowed, with 35,214 at baseline.
The companion ratchet check passes. [Sizes](size-comparison.json) records the
one-line addition in each production file; no budget was raised.
[Review](review.json) found no actionable issues. No full repository suite ran.
Existing pytest cleanup and component-governance SyntaxWarnings remain.

## Native evidence

[native_check.py](native_check.py) runs actual TldwCli/LinuxDriver with an exclusive
private profile, private database paths and null keyring. A synthetic failed job
in each theme supplies queue controls; the registry has no store or runner.
Actual tmux resize events exercise all four controls through 80×24 → 170×48 →
170×24 → 170×48 in both themes, for **32 passing steps**. Each step checks retained
visible focus, draft and selection, unchanged preflight and composition/source
generations, unchanged configuration bytes and registry entries.

| Focused state | Dark | Light |
|---|---|---|
| Title, 80×24 | [capture](title-textual-dark-80.svg) | [capture](title-textual-light-80.svg) |
| Recent imports, 80×24 | [capture](recent-textual-dark-80.svg) | [capture](recent-textual-light-80.svg) |
| Recent imports, 170×48 | [capture](recent-textual-dark-170.svg) | [capture](recent-textual-light-170.svg) |

Run-001 passed and exited normally with status 0:
[result](result.json), [isolation](isolation.json), [lifecycle](lifecycle.json).
All six SVGs were rendered and inspected; [inspection](inspection.json) records
the visible focus, readable content and unobscured controls. Screenshots are
representative states; the other transitions are covered by runtime assertions.
[Persistence](persistence.json) records ten healthy private databases, zero
media/messages/ingest jobs, unchanged source bytes and default-profile hashes,
and no app ERROR/CRITICAL log lines. Private databases and full logs remain in
ignored scratch.

The real optional image terminal probe runs before startup, qualifying
probe-primed startup. No actual import, provider request, server request,
installation, restart, push or dev integration was performed. Provider-specific
Import recovery remains for review.
