# Sensitive-path hardening + glob_files/grep_files port report

Source: `wt-builtin-tool-packs` (`docs/builtin-tool-packs-spec`), read-only.
Target: `wt-agent-substrate` (`feat/agent-runtime-substrate`).

The reference branch's `builtin_packs/` architecture (pack registry,
`builtin_pack_config.py`, `Agents/builtin_packs/files.py`, `Tools/base.py`)
does not exist on this base and was NOT re-created. Everything below is
re-homed directly onto dev's existing `Agents/tool_catalog.py`
`_GATEABLE_BUILTINS` table + `Tools/file_operation_tools.py` structure.

## 1. `Utils/sensitive_paths.py`

Ported essentially verbatim. `_DB_PATH_ACCESSOR_NAMES` (11 accessors) matches
dev's `config.py` 1:1 — no accessor renames were needed. Only the module
docstring and a couple of cross-references were reworded to point at dev's
real file tools instead of the retired `Agents/builtin_packs/files.py`.

`SensitivePathContext`/`resolve_sensitive_context`/`is_sensitive_path` are
unchanged: uncached per-call resolution, `context=None` default that still
enforces the denylist (never "nothing is sensitive"), fail-closed on an
unresolvable path, DB sidecar (`-wal`/`-shm`/`-journal`) refusal by exact
constructed name.

## 2. Wiring into the file tools

Dev's `ReadFileTool`/`WriteFileTool`/`ListDirectoryTool` already differ from
the reference: they validate against `validate_path_multi(...,
allowed_file_roots(write=.., sandbox_root=...))` — a workspace-folder-roots
feature (`Tools/workspace_file_roots.py`, ADR-028) the reference branch does
not have. The sensitive-path check was inserted immediately after that
multi-root validation and before any filesystem access, in each tool,
returning that tool's own error-dict shape (`"Refused: '<path>' is a
protected path and cannot be {read,written,listed}"`).

`ListDirectoryTool` resolves one `SensitivePathContext` at the top of
`execute()` and reuses it for: the top-level target check, `containment_root`
selection, the recursive-descent guard, and the per-entry filter inside the
walk. The per-entry filter is what stops `mcp_permissions.json` and a DB's
`-wal` from being listed by name even when they merely sit inside an
otherwise-ordinary, non-descended-into directory.

Dev's private `_is_within(candidate, root)` (containment-only) was renamed
to a public `is_within(candidate, root, context=None)` that now also
enforces `is_sensitive_path`, matching the reference's combined
containment+sensitivity helper. Its two existing call sites
(`containment_root` selection, recursive-descent guard) were updated to pass
the resolved context. This is a small strengthening of existing behavior
(a widened sandbox root could previously recurse into `~/.ssh` if nothing
else stopped it) but is fully covered by the new tests and did not disturb
any of the workspace-folder-roots tests (`Tests/Tools/
test_file_tools_workspace_roots.py`, still 100% passing).

**Note tools checked, no change needed.** `Tools/note_management_tools.py`
(`CreateNoteTool`/`UpdateNoteTool`/`SearchNotesTool`) never resolves a
user-supplied filesystem path — they only call `NotesInteropService` methods
against the ChaChaNotes DB. There is no path input to gate, so no
sensitive-path check was added there.

## 3. `glob_files` / `grep_files`

Implemented as `GlobFiles`/`GrepFiles` `Tool` subclasses directly in
`Tools/file_operation_tools.py` (not a new package — dev has no pack layer
to put them in). Both are scoped to `_tool_sandbox_root()` only — **not**
dev's multi-root `allowed_file_roots()` workspace bindings. This is a
deliberate scoping decision, flagged rather than silently made: the
reference implementation these were ported from has no workspace-roots
concept at all, so there was no "exact" behavior to preserve for that case.
Extending glob/grep to search bound workspace folders too (for parity with
`list_directory`) is a reasonable follow-up but out of scope for a straight
port — happy to file a task if wanted.

Behavior preserved from the reference exactly: `_MAX_MATCHES = 200` (result
cap), `_MAX_CANDIDATES = 20_000` (examined-candidate cap, independent of
matches — `Path.glob("../**/*")` doesn't raise and was measured yielding
~1.4M paths, so a match-only bound never trips), `_MAX_GREP_FILE_BYTES =
5_000_000` (per-file streaming cap), up-front `_rejects_traversal` (POSIX
absolute, Windows drive-letter, Windows UNC, `..` component) before any
globbing, lazy-iteration guarding (`next(candidates)` inside its own narrow
`try`, so a `ValueError` from body code — `is_within`, the streamed read,
`regex.search` — is never misreported as a bad pattern), no
`sorted(candidates)` in `grep_files` (would materialize the generator and
defeat `_MAX_CANDIDATES`), and per-file-streamed grep (never
`read_text()`+`splitlines()`).

Both tools resolve one `SensitivePathContext` per call (`resolve_sensitive_
context()`) and reuse it across every candidate via `is_within(path, root,
context=sensitive_ctx)`, plus the inline `_is_hidden_within` dotfile check
(item 4 below). Risk tag: `("reads",)`, matching dev's existing convention
for `ReadFileTool`/`ListDirectoryTool` (dev already carries concrete risk
tags that the reference branch, at the point this was forked from, did not
— dev is ahead here, so this follows dev).

### Registration

Two new `GateableTool` entries in `Agents/tool_catalog.py`'s
`_GATEABLE_BUILTINS`:

```python
GateableTool("glob_files_enabled", "file_operation_tools", "GlobFiles", "glob_files"),
GateableTool("grep_files_enabled", "file_operation_tools", "GrepFiles", "grep_files"),
```

Both default OFF (absent from `[tools]` config), like every sibling entry.
Also added to `Tools/__init__.py`'s lazy `__all__`/`_SUBMODULE_BY_NAME` map
for consistency with `ReadFileTool`/`ListDirectoryTool`/`WriteFileTool`
(`build_gateable_tool` doesn't need this — it imports the submodule
directly — but every other file tool has an entry here and omitting these
two would be a silent inconsistency for anyone auditing that table).

Added `"glob_files"`/`"grep_files"` to `Library/library_skills_state.py`'s
`_SHADOWED_BUILTIN_NAMES` frozenset — required by the existing drift-guard
tests (`test_gated_tool_names_are_covered_by_the_shadow_guard` and its P2
sibling), which build a `BuiltinToolProvider` with default config and so
cannot see config-gated names on their own; they must be listed explicitly
or a skill named `glob_files`/`grep_files` would silently shadow the real
builtin the moment a user enables the gate.

### Settings UI — no changes needed, confirmed

`UI/Tools_Settings_Window.py`'s `_compose_tool_settings`/`_save_tool_
settings`/`_reset_tool_settings` all iterate `gateable_builtin_tools()`
directly and build `Switch(id=f"tool-switch-{entry.tool_name}")` rows from
the table — there is no hardcoded tool list anywhere in that window. The
two new entries are picked up automatically. Verified via the existing
structural test `Tests/UI/test_settings_tools_section.py::
test_every_gateable_tool_gets_a_switch_id` (asserts the compose loop
literally contains `"gateable_builtin_tools()"` and the switch-id f-string)
plus a live check: `BuiltinToolProvider().list_catalog()` with
`glob_files_enabled=True`/`grep_files_enabled=True` set now includes both
tools with their `("reads",)` risk tags, same as every other gated tool.

## 4. Hidden-file consistency fix

Applied `_is_hidden_within(resolved, root_resolved)` inline in both
`GlobFiles.execute` and `GrepFiles.execute`, run against the *already-
resolved* candidate (not by calling `validate_path` per candidate), for the
same reasons as the reference: `validate_path` raises (these tools must
return `{"error": ...}`, never raise) and it is measurably more expensive
per candidate. Not folded into `is_within`/the shared containment helper,
because `ListDirectoryTool`'s `include_hidden` opt-in must keep working —
folding it in there would make every hidden-aware caller un-optionally
hidden-blind.

**Re-measured on this base** (1,500-file sandbox tree, best of 3 iterations,
`Utils/path_validation.validate_path` vs the inline `_is_hidden_within`,
both layered on top of the same `is_within` baseline):

| variant | ms/candidate | vs baseline |
|---|---|---|
| `is_within` alone (baseline) | 0.1471 | — |
| `is_within` + inline `_is_hidden_within` | 0.1764 | **+20.0%** |
| `is_within` + per-candidate `validate_path` | 0.2100 | **+42.8%** |

Consistent with the reference branch's own reported +19.9%/+46.6% — the
inline check remains the right call on this base too.

Regression coverage: `test_glob_files_hides_a_dotfile_in_the_sandbox`,
`test_glob_files_hides_a_file_under_a_dotted_directory`,
`test_grep_files_cannot_read_a_dotfile_in_the_sandbox`,
`test_grep_files_cannot_read_a_dotfile_via_a_broad_glob` all reproduce the
exact live finding (`grep_files('API_KEY', glob='**/.env')` used to return
the secret line while `read_file('.env')` refused it) and pin the fix.

## Tests brought across (all new files/additions, adapted import paths)

- `Tests/Utils/test_sensitive_paths.py` — new file. Reference's suite plus
  two additions: full 11-accessor coverage
  (`test_all_eleven_db_accessors_resolve_to_sensitive_paths`, reference only
  covered 3) and an explicit context-reuse-vs-fresh-resolution parity test.
- `Tests/Tools/test_file_tool_sandbox.py` — **already existed** on this base
  (4 tests: sandbox-root-is-real-dir, traversal rejection, in-sandbox read,
  dotted-ancestor-root regression). Appended the 6 tool-level sensitive-path
  integration tests from the reference (`read_file`/`write_file`/
  `list_directory` each refusing a sensitive target/overwrite, the DB path,
  the DB's `-wal` sidecar, and the recursive-listing per-entry filter test),
  plus an autouse fixture that forces `allowed_file_roots` to its
  sandbox-only fallback (these tests are about the denylist, not
  workspace-roots, and this also sidesteps `workspace_file_roots.py`'s
  process-wide registry-instance cache leaking state across tests with
  different isolated `HOME`s).
- `Tests/Tools/test_glob_grep_files.py` — new file, all of the reference's
  glob/grep tests (recursive match, grep line reporting, bad-regex handling,
  parent-traversal + absolute + Windows drive-letter/UNC refusal for both
  the up-front check and via direct `_rejects_traversal` unit test,
  lazy-invalid-pattern reaching real iteration for both tools, the
  `_MAX_CANDIDATES` test that shrinks the bound and proves it — not just the
  larger `_MAX_MATCHES` — is what cuts the walk off, unusable-sandbox-root
  error-dict-not-raise, sensitive-context-resolved-once-per-call call
  counting, dotfile/dotted-dir hiding, grep file-size-cap skip/match), plus
  two additional tool-level integration tests
  (`test_glob_files_hides_this_apps_own_sqlite_db`,
  `test_grep_files_cannot_read_this_apps_own_sqlite_db_wal_sidecar`) with the
  sandbox root configured to *contain* the sensitive path — the one
  configuration in which the bug is actually observable, per the task's
  explicit ask.
- `Tests/Agents/test_builtin_file_tools.py` — appended 6 tests: absent by
  default, per-gate-key appears (parametrized), gate independence, risk-tag
  carry-through, and shadow-guard coverage for the two new names.

## Test commands run (foreground; this worktree has no `.venv` of its own —
confirmed via `sys.path`/`tldw_chatbook.__file__` that the main checkout's
venv resolves the package from THIS worktree's source when cwd is inside it)

```
source /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/activate
cd /Users/macbook-dev/Documents/GitHub/wt-agent-substrate
python -m pytest Tests/Utils/test_sensitive_paths.py Tests/Tools/test_file_tool_sandbox.py Tests/Tools/test_glob_grep_files.py Tests/Agents/test_builtin_file_tools.py -q
# 75 passed in 1.92s

python -m pytest Tests/Agents/ Tests/Tools/ Tests/Utils/ Tests/UI/test_tools_settings_window.py -q
# 877 passed, 6 failed, 16 skipped in 129.30s
```

The 6 failures are pre-existing and unrelated (`test_chat_api_key_*` in
`Tests/UI/test_tools_settings_window.py` — Chat API key field/save
behavior). Confirmed by `git stash -u` and re-running one of them against
the unmodified tree: identical failure (`assert '' == 'test-configured-key'`).
877 passed vs. the pre-change baseline's 821 passed = the 56 new/added
tests above, all green; no regressions.

Did not run the full suite per instructions (documented to exceed three
hours here). `pytest-mock` and `numpy` remain absent from this venv per the
stated baseline — neither was needed for anything touched here.

## Files touched

- `tldw_chatbook/Utils/sensitive_paths.py` (new)
- `tldw_chatbook/Tools/file_operation_tools.py` (sensitive-path wiring +
  `is_within` rename/extension + `GlobFiles`/`GrepFiles`)
- `tldw_chatbook/Tools/__init__.py` (lazy re-export entries)
- `tldw_chatbook/Agents/tool_catalog.py` (two new `GateableTool` entries)
- `tldw_chatbook/Library/library_skills_state.py` (`_SHADOWED_BUILTIN_NAMES`)
- `Tests/Utils/test_sensitive_paths.py` (new)
- `Tests/Tools/test_file_tool_sandbox.py` (extended)
- `Tests/Tools/test_glob_grep_files.py` (new)
- `Tests/Agents/test_builtin_file_tools.py` (extended)
