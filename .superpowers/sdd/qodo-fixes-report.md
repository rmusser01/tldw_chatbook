# Qodo fixes -- PR #936

Six findings from the automated Qodo review of PR #936, all independent. Fixed on
branch `docs/builtin-tool-packs-spec`.

## Finding 1+2 (CRITICAL) -- glob_files/grep_files inconsistent with read_file's hidden-file refusal

`GlobFiles`/`GrepFiles` in `tldw_chatbook/Agents/builtin_packs/files.py` filtered
candidates with `is_within()` only. `is_within()` applies the credential/app-state
denylist (`Utils/sensitive_paths.py`) but never applied the hidden-component rule
`Utils/path_validation.validate_path()` enforces for `read_file`/`write_file` (any
dot-prefixed component in the user-supplied portion of the path is refused). Live
repro pre-fix:

```
read_file('.env')                       -> {'error': 'Access to hidden files/directories is not allowed'}
grep_files('API_KEY', glob='**/.env')   -> returns the line 'API_KEY=supersecret123'
```

### Approach chosen: inline check, not `validate_path` per candidate

Benchmarked over a 1,500-file sandbox tree (`bench_validate_path.py`, run three
times, best-of-three reported), calling `is_within()` alone as today vs. adding
either (a) a full `validate_path(candidate, root)` call per candidate or (b) an
inline hidden-component check against the already-resolved path:

| Approach | best time / 1500 candidates | ms / candidate | delta vs. baseline |
|---|---|---|---|
| `is_within` only (current) | 0.2459s | 0.1639 | -- |
| `is_within` + `validate_path` per candidate | 0.3605s | 0.2403 | **+46.6%** |
| `is_within` + inline hidden-component check | 0.2948s | 0.1965 | +19.9% |

`validate_path`'s per-call overhead (exception-based control flow, timing calls,
recomputing `base_directory.name.startswith(".")` and re-resolving `base_directory`
every call) is significant for a candidate-filtering loop, and it also **raises**
on rejection -- these tools must never raise, so every call would need its own
try/except anyway. Chose the inline check: a new `_is_hidden_within(resolved,
root_resolved)` helper in `files.py` that mirrors `validate_path`'s hidden-component
rule against the already-resolved candidate and the once-resolved root, called
right after `is_within()` succeeds, in both `GlobFiles.execute` and
`GrepFiles.execute`. `is_within()` itself is unchanged and still called first (it
owns the denylist/`SensitivePathContext` check, which `validate_path` deliberately
does not perform).

Did not bake the hidden-component rule into `is_within()` itself: `is_within()` is
also used by `ListDirectoryTool`'s recursive-descent guard, which must still be
able to descend into a dotted directory when the caller passes
`include_hidden=True` -- a feature `glob_files`/`grep_files` don't have an
equivalent opt-in for. Scoping the new rule to `files.py`'s two tools avoids
regressing that.

### Regression tests (`Tests/Agents/test_builtin_packs.py`)

- `test_glob_files_hides_a_dotfile_in_the_sandbox`
- `test_glob_files_hides_a_file_under_a_dotted_directory` (nested, e.g. `.git/config`)
- `test_grep_files_cannot_read_a_dotfile_in_the_sandbox` (exact live repro, `glob='**/.env'`)
- `test_grep_files_cannot_read_a_dotfile_via_a_broad_glob` (default glob, no dotfile named explicitly)

## Finding 5 (Important) -- grep_files reads whole files into memory

`GrepFiles.execute` did `path.read_text()` then `.splitlines()`, materializing the
whole file (plus a second, line-split copy) for every candidate. Changed to stream:
`with path.open("r", encoding="utf-8", errors="replace") as fh: for number, line in
enumerate(fh, start=1): ...`, breaking out once `_MAX_MATCHES` is reached, exactly
as asked. `errors="replace"` decoding behavior is preserved.

Also added a per-file byte cap, `_MAX_GREP_FILE_BYTES = 5_000_000`, checked via
`path.stat().st_size` before opening the file and skipping (not erroring) files
above it. This bounds the worst case independent of `_MAX_CANDIDATES`/`_MAX_MATCHES`
(which bound file/match *counts*, not the size of any one file) -- a single
pathological file with no newline characters would otherwise still force one giant
line to be buffered in full even with the streaming change.

### Tests

- `test_grep_files_skips_a_file_over_the_size_cap`
- `test_grep_files_still_matches_within_the_size_cap`

## Finding 6 (Important) -- Windows absolute patterns bypass `_rejects_traversal`

`_rejects_traversal()` checked `pattern.startswith("/")`, which misses Windows
drive-letter (`C:\...`) and UNC (`\\server\share\...`) forms -- an OS-dependent
gap (`is_within()` still guards every candidate regardless, so this was a
cost/consistency issue, not an actual escape, as the finding notes).

Verified empirically that neither `Path(pattern).is_absolute()` alone nor
`PureWindowsPath(pattern).is_absolute()` alone covers both forms on a POSIX host
(this repo's dev/test environment is macOS):

```
'/etc/passwd'              Path.is_absolute=True   PureWindowsPath.is_absolute=False
'C:\Users\x'                Path.is_absolute=False  PureWindowsPath.is_absolute=True
'\\server\share\path'       Path.is_absolute=False  PureWindowsPath.is_absolute=True
```

Fixed by checking both: `Path(pattern).is_absolute() or
PureWindowsPath(pattern).is_absolute() or ".." in Path(pattern).parts`.
`PureWindowsPath` is a "pure" path class (no filesystem calls), so this works
correctly regardless of the host OS actually running the process. Left the `".."
in Path(pattern).parts` traversal-component check untouched, per the finding's
note.

### Tests

- `test_glob_files_refuses_windows_drive_letter_pattern`
- `test_glob_files_refuses_windows_unc_pattern`
- `test_grep_files_refuses_windows_drive_letter_glob`
- `test_grep_files_refuses_windows_unc_glob`
- `test_rejects_traversal_recognizes_windows_absolute_forms` (direct unit test of the helper)

## Finding 3 (Minor) -- import grouping

`files.py` had `import re` / `from pathlib import Path` (stdlib), then three
separate local-import blocks each separated by a blank line. Merged into the
repo's three-contiguous-groups convention (stdlib, third-party, local; no blank
line inside a group). This file has no third-party imports, so the result is two
groups: stdlib (`re`, `pathlib`), then one contiguous local-import block ordered
`Agents.builtin_services`, `Tools.base`, `Tools.file_operation_tools`,
`Utils.sensitive_paths`.

Added `PureWindowsPath` to the existing `from pathlib import Path` line for
finding 6 above.

## Finding 4 (Minor) -- to_openai_format() has zero callers; deleted rather than documented

Qodo asked for a Google-style `Returns:` section on `Tool.to_openai_format()` in
`tldw_chatbook/Tools/base.py`. Verified the zero-caller claim:

```
$ grep -rn "to_openai_format" . 2>/dev/null | grep -v "\.git/"
tldw_chatbook/Tools/base.py:81:    def to_openai_format(self) -> dict:
```

The only hit is the definition itself -- no callers anywhere in `tldw_chatbook/`,
`Tests/`, or any other file in the repo (the `ToolExecutor` that used to consume it
was already removed on this branch, per commit `63eb693c3`'s message: "remove the
ToolExecutor, code audit tool, and settings switches"). Deleted the method instead
of documenting dead code.

## Verification

```
$ source .venv/bin/activate
$ python -m pytest Tests/Agents/ Tests/Tools/ Tests/Utils/ Tests/Chat/test_console_agent_bridge.py -q
879 passed, 2 warnings in 71.19s (0:01:11)
```

(The 2 warnings are pre-existing `RuntimeWarning: coroutine ... was never awaited`
noise from `Tests/Agents/test_mcp_tool_provider.py`, unrelated to this change.)

```
$ python -m pytest Tests/Agents/test_builtin_packs.py Tests/Agents/test_builtin_pack_config.py -q
41 passed in 0.80s
```

`ruff check` on all three touched files: all checks passed.

## Files changed

- `tldw_chatbook/Agents/builtin_packs/files.py` -- findings 1, 2, 3, 5, 6
- `tldw_chatbook/Tools/base.py` -- finding 4 (`to_openai_format` deleted)
- `Tests/Agents/test_builtin_packs.py` -- regression tests for findings 1, 2, 5, 6
