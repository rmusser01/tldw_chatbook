# TASK-34414 — exact Notes folder readiness

Approved test-only follow-up in PR [#3029](https://github.com/rmusser01/tldw_chatbook/pull/3029).
No production Library, sync, config, ownership, timing threshold or retry change.
ADR required: no; test-only settlement correction, N/A.

## Failure and correction

Exact-head `8a5dc579ce770631df3b340ffdaf062eb92aa70c` CI
[UI Fast Lane (2)](https://github.com/rmusser01/tldw_chatbook/actions/runs/37434243096/job/112172032018)
failed with `StopIteration` selecting VSync after waiting for any folder row.
The required aggregate guard failed because this UI lane failed, not artifact drift.
The unchanged isolated test passed (1 test, 13.85 s); that did not clear CI or prove a race.

The controlled scenario adds a real Unfiled note and holds only the real root-folder
service response. The native test wait is observed without replacing its behavior;
navigation must remain pending while Unfiled is visible but `folder-1` is absent.
All rows, SQLite writes, vault bytes, Delete/Undo controls and sync statuses remain real.
The first timing-dependent control passed and is explicitly **not RED**.
The tightened handoff fails against the original wait with the CI's
`RuntimeError('coroutine raised StopIteration')` footprint.

The helper now uses the existing 30-second predicate wait for exact `folder-1`,
then re-queries and presses its live row without another yield. The normal and
controlled scenarios both retain every original Delete/Undo/idle status assertion.

## Evidence

- [Initial non-RED control](initial-control.log): 2 passed, 19.91 s.
- [Valid RED](red.log): 1 failed, 5.75 s; Unfiled incorrectly completed navigation.
- [GREEN](green.log): 2 passed, 17.66 s, exit 0, no pytest warnings.
- Changed test: Ruff check with `--no-cache` and format check pass; whitespace clean.
- Independent read-only review: no Critical, Important or Minor findings; no tests
  run by the reviewer and no global merge-ready claim.
- First artifact run: ten guards passed; Mermaid alone failed because the supplied
  cache path did not exist. This is retained as an invocation failure, not code drift.
  [Normal pinned-input Mermaid rerun](mermaid-guard.log) passes, exit 0: six outputs
  verified. All eleven guards have passing observations; this is not one clean
  eleven-guard invocation. Final task-ID/readability guards pass after closeout.

Original local evidence is retained at `/tmp/library-exact-folder-red-rXziWD`,
`/tmp/task34414-library-green-RhR1fV`, `/tmp/task34414-preflight-hvn8UQ`, and the
pre-renumber initial-control root `/tmp/task34413-library-red-VXDeSJ`.
The unpublished CLI ID collided with an existing remote task; only our record was
renumbered to TASK-34414 after a remote/worktree ID sweep.

New-head CI, current-base integration and actual external review are separate gates.
The aggregate FD warning, native/Windows/participant/full latency matrix and final
application retirement remain follow-ups; none are waived by these two test passes.
