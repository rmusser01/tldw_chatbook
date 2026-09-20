# Static source/fix delta attribution

Base: `cf61cb68505fc62991a0488c964a78cb7ab31cbb` (before Tasks 1–5).
Same root-venv Ruff/config used for both revisions. `static-delta.json` records
all 25 explicit changed Python paths, full unmatched diagnostics and formatter
edits. The checker maps *complete* diagnostic spans and exact formatter edits
through unchanged source, not net diagnostic counts.

Baseline/current lint totals: 1,203 / 1,201. No introduced formatter edit.
All diagnostics match unchanged baseline spans except these three manual
attributions on the authorized Notes import sort:

| Diagnostic | Base source | Current source | Attribution |
| --- | --- | --- | --- |
| UP035 `typing.Dict` | Notes_Library.py:19 | Notes_Library.py:20 | Same deprecated symbol; import order only |
| UP035 `typing.List` | Notes_Library.py:19 | Notes_Library.py:20 | Same deprecated symbol; import order only |
| UP035 `typing.Sequence` | Notes_Library.py:19 | Notes_Library.py:20 | Same symbol and collections.abc migration suggestion; import order only |

Exact old statement:
`from typing import TYPE_CHECKING, List, Dict, Optional, Any, Sequence, Union`

Exact new statement:
`from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union`

Imported symbols are identical. The checker correctly refuses to automatically
match a reordered line; the displayed source comparison attributes each retained
diagnostic rather than suppressing it. The existing nested import I001 is retained
on its unchanged source span; it was not authorized for cleanup. The module-level
I001 whose interior Task 3 changed is cleared by the permitted import sort.
No annotations, other imports, Notes behavior or whole-file formatting changed.

The new Task 6 integration/probe file is checked separately and must be fully
lint/format clean. Legacy files are not claimed whole-file clean.

Final scope extension (`static-final-delta.json`): 28 explicit paths, including
the new Task6 integration, approved logging-test helper and canvas readiness fix.
1,232 baseline / 1,230 current lint findings; 101 baseline / 101 current formatter
edit spans. The same three Notes import symbols above are the only unmatched
diagnostics. All canvas/logging diagnostics and formatter edits map to unchanged
baseline source spans. `static-final-current-ruff.txt` preserves the full current
output; `task6-final-{lint,format}.txt` record the new file's clean checks.

Commit-scope extension (`static-commit-delta.json`) adds both approved editor
files, for 30 paths. Totals remain 1,232 / 1,230 lint and 101 / 101 formatter
edits, with exactly the same three manual attributions above. Both editor files
are entirely clean; no changed-span exception is needed for that fix.
