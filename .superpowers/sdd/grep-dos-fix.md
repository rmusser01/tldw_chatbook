# grep_files ReDoS mitigation (PR #953 review findings 1 & 2)

## Scope

Two review findings on PR #953, fixed on `feat/agent-runtime-substrate` in
`tldw_chatbook/Tools/file_operation_tools.py`:

1. `GrepFiles.execute` compiled a model-supplied regex and ran it against the
   **full** line with no bound, on a thread `_call_with_timeout`
   (`Agents/agent_service.py`) cannot kill on timeout.
2. Import-grouping nit: `from loguru import logger` sat with no blank line
   separating it from the stdlib imports.

## Finding 1 -- what was changed

Three additions, all scoped to `GrepFiles`:

- `_MAX_GREP_LINE_SEARCH_CHARS = 500` -- the slice of each line actually
  handed to `regex.search`. Previously the search ran on the full line while
  only the *stored* result was truncated to 500 chars; since
  `_MAX_GREP_FILE_BYTES` (5,000,000) bounds a *file*, not a *line*, a
  no-newline file could hand `re.search` a ~5,000,000-character line.
- `_MAX_GREP_LINES_SCANNED = 200_000` -- total lines read across **all**
  files in one call, independent of `_MAX_CANDIDATES` (files examined) and
  `_MAX_MATCHES` (matches returned). Neither of those bounds the aggregate
  line count a wide corpus of small-line files can produce.
- `GrepFiles.timeout_seconds = 20.0` -- overrides the `Tool` base's `0.0`
  default via the per-tool override seam (`Tool.timeout_seconds`, resolved
  through `ToolCatalogRegistry.timeout_for`).

### Reasoning / values chosen

**`_MAX_GREP_LINE_SEARCH_CHARS = 500`.** Measured growth of a classic
catastrophic pattern on this machine:

```
re.compile(r'(a+)+$').search('a'*n + 'X')
n=18  0.012s   n=24  0.747s   n=27   6.0s
n=20  0.047s   n=25  1.484s   n=28  11.9s
n=22  0.185s   n=26  2.985s   n=30  47.5s   (growth ~2x per extra char)
```

This means **no** line-length cap that is still useful for real grep
searches (tens to hundreds of characters) makes a deliberately adversarial
pattern *fast* -- the growth is exponential in input length, so even 500
chars is enormous for a maximally adversarial pattern. 500 was chosen to
match the existing stored-result truncation (so a match found is always
inside what's returned/displayed) and because it is a ~10,000x reduction
from the previous effective bound (`_MAX_GREP_FILE_BYTES` = 5,000,000
chars for a no-newline file) -- turning "scales with file size, effectively
unbounded" into "bounded by a small, fixed constant." The docstrings say
this plainly: it does **not** make catastrophic backtracking fast, only
finite and independent of file size. The real backstops for the residual
risk are (a) `_MAX_GREP_LINES_SCANNED` bounding aggregate *normal*-case
cost, (b) `timeout_seconds` freeing the agent loop (not the thread) sooner,
and (c) `grep_files`'s `"reads"` risk tag flooring it to `ask`, so a human
approves every individual call -- limiting how often an adversarial pattern
can even be tried. A complete fix needs a regex engine with match timeouts
or a killable subprocess; noted in the source comments.

**`_MAX_GREP_LINES_SCANNED = 200_000`.** In-memory benchmark of a realistic
(non-pathological) pattern against 200,000 lines, each pre-capped to 300
chars: **0.064s**. This is a different concern from the line-length cap --
it bounds the *aggregate* cost of many ordinary per-line searches (a corpus
of small-line files, each individually under `_MAX_GREP_FILE_BYTES`, that
could otherwise sum to an enormous total line count), not the cost of any
single catastrophic search.

**`GrepFiles.timeout_seconds = 20.0`.** A legitimate search over the
candidate bound (`_MAX_CANDIDATES` = 20,000 files, `_MAX_GREP_LINES_SCANNED`
= 200,000 lines) is comfortably fast per the benchmark above (well under a
second for the regex+iteration cost; real disk I/O for a few tens of MB
adds at most low single-digit seconds on a loaded system). 20s leaves
generous headroom above that while being far tighter than the run's own
default (`RunBudget.max_tool_call_seconds`, 300s at defaults) -- so a
pathological call is reported back to the agent as timed out 15x sooner.
It does **not** stop the search itself: `_call_with_timeout` abandons the
worker thread rather than killing it (Python cannot kill a thread), so a
pathological pattern keeps burning CPU in the background regardless of
this value -- stated explicitly in the property's docstring.

### Before/after timing on the pathological pattern

Reproduced with `_MAX_GREP_LINE_SEARCH_CHARS` monkeypatched to `10` and a
28-character run of `a` + `X` (`"a"*28 + "X\n"`), via `GrepFiles.execute`:

| | full line searched (before fix / mutated) | capped to 10 chars (after fix) |
|---|---|---|
| `regex.search(r"(a+)+$", ...)` | **11.64s**, no match anywhere in the string | **<0.001s**, matches (capped slice is pure `a`, greedy match succeeds immediately) |

(The isolated, non-tool-wrapped repro for the same pattern/length pair was
measured separately at 11.9s -- see the growth table above; both numbers
agree.)

## Finding 2 -- import grouping

```python
import re
from pathlib import Path, PureWindowsPath
from typing import Dict, Any

from loguru import logger          # <- now its own third-party group

from . import Tool
...
```

## Mutation verification (per task instructions, not shipped)

Both new caps are covered by tests that were verified to actually fail when
the corresponding cap is removed -- confirmed by temporarily mutating the
source, running the single test, observing the failure, then reverting:

1. **Line-length cap**
   (`test_grep_files_search_input_is_bounded_to_a_capped_line_length`):
   temporarily changed `regex.search(line[:_MAX_GREP_LINE_SEARCH_CHARS])` to
   `regex.search(line)`. Re-ran just that test:
   `AssertionError: regex.search took 11.64s -- is the line-length cap
   applied?` (also would have failed on `len(result["matches"]) == 1`,
   since the uncapped search finds no match at all). Reverted; suite green
   again.
2. **Total-lines-scanned cap**
   (`test_grep_files_bounds_total_lines_scanned_across_the_whole_call`):
   temporarily replaced both `lines_scanned >= _MAX_GREP_LINES_SCANNED`
   guards with `False`. Re-ran just that test:
   `AssertionError: assert 100 == 30` (all 100 lines across the 5 test
   files matched, instead of stopping at the 30-line cap). Reverted; suite
   green again.

This was **not** done by literally reverting the fix and running the whole
suite (the line-cap scenario would then hang for ~12s per invocation and
risks a much longer hang for a slightly larger repro), only by isolating
each single test with `pytest -k <name>` during the manual verification
pass.

## Tests added

`Tests/Tools/test_glob_grep_files.py` (new section, 4 tests):

- `test_grep_files_search_input_is_bounded_to_a_capped_line_length` --
  proves the line-length cap is applied (timing + match-count assertions).
- `test_grep_files_bounds_total_lines_scanned_across_the_whole_call` --
  proves the total-scan cap binds, with `_MAX_MATCHES`/`_MAX_CANDIDATES`
  set generously high so neither of those could explain the bounded result.
- `test_grep_files_declares_a_nonzero_timeout` -- `GrepFiles().timeout_seconds
  > 0.0`.
- `test_grep_files_timeout_resolves_through_the_tool_catalog_registry` --
  builds a real `BuiltinToolProvider` + `ToolCatalogRegistry`, registers a
  real `GrepFiles` instance, and asserts
  `registry.timeout_for("grep_files") == 20.0`.

No existing behavior was touched: streaming reads, `_MAX_GREP_FILE_BYTES`,
`_MAX_MATCHES`/`_MAX_CANDIDATES`, containment (`is_within`), the
sensitive-path denylist, the hidden-component rule, and the dotted-root
rule are all unchanged and still covered by the pre-existing tests in the
same file.

## Test runs

```
$ .venv/bin/python -m pytest Tests/Tools/test_glob_grep_files.py -q
34 passed in 1.04s

$ .venv/bin/python -m pytest Tests/Agents/ Tests/Tools/ Tests/Utils/ -q
879 passed, 12 warnings in 114.66s
```

No failures. (The task's noted baselines -- missing `pytest-mock`/`numpy`,
six pre-existing `test_chat_api_key_*` failures -- live in `Tests/Chat/`,
outside this run's scope, and were not encountered.)
