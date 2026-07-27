# TASK-963..966 -- dev baseline test-failure fixes

Worktree: `/Users/macbook-dev/Documents/GitHub/wt-baseline`, branch `fix/dev-baseline-test-failures`,
fast-forwarded from a stale 19-behind snapshot to `origin/dev@a73b9b46f` before starting (branch had
no local commits, so the fast-forward was safe and matches the "cut from current origin/dev" intent).

## TASK-963 -- raw-connection census test

**Root cause.** Two production sites bypassed the `connect_private_sqlite` seam with a raw
`sqlite3.connect(...)`:
- `tldw_chatbook/DB/Subscriptions_DB.py::ensure_site_configs_schema` (the exact site named in the task).
- `tldw_chatbook/Notes/file_notes_replica.py::FileNotesReplica.__init__` -- a **second** site the census
  also flagged, introduced by the File Notes feature that merged into `dev` after this task was triaged
  (picked up by the fast-forward). Not in the task's original description; found by re-running the
  census on current dev rather than assuming only one site was outstanding.

**Fix: production code**, both sites. Each genuinely bypassed the private-path boundary the seam exists
to enforce -- qualifying was the correct answer, not silencing the census.
- Added `db.subscriptions.site_configs` (`private_file`) and `notes.file_replica`
  (`private_file` + `memory`) owner policies to `SQLITE_OWNER_REGISTRY` in
  `tldw_chatbook/DB/private_sqlite.py`.
- Both call sites now route through `connect_private_sqlite(owner_id, ...)`.
- Added inventory rows C36/C37 to `backlog/docs/sqlite-private-owner-inventory.md`.
- Updated `Tests/DB/test_private_sqlite_inventory.py`'s id-range assertion from C01..C35 to C01..C37
  (a legitimate shape change now that two new documented sites exist, not a relaxed assertion).

**Before/after:** `Tests/DB/test_private_sqlite_inventory.py`: 1 failed / 20 passed -> 21 passed.
Also verified clean: `Tests/DB/test_subscriptions_db_site_configs.py` (20 passed),
`Tests/Notes/test_file_notes_replica.py`, `Tests/DB/` (600 passed, 1 skipped),
`Tests/Subscriptions/` + `Tests/Notes/` (296 passed, 1 skipped).

## TASK-964 -- workspace folder-binding test

**Root cause.** The full traceback (not just the exception type) shows the `PrivatePathError` does
**not** come from `add_folder_binding`'s own root-validation logic. `add_folder_binding` already
requires the candidate folder to exist (`resolved.is_dir()`, else a domain `WorkspaceRegistryServiceError`)
-- production already treats binding a not-yet-existing root as **not legitimate**, unconditionally.

The actual raise happens in the test's own `build_registry()` helper: it constructs a second,
independent `WorkspaceDB` under `tmp_path / "second-db"`, a directory the test never created. The
private-paths hardening removed `BaseDB.__init__`'s old auto-`mkdir` of its own parent directory
(inventory row P06, now "migrated"), so opening a database whose containing directory doesn't exist
now raises instead of silently creating it. Real callers never hit this because `WorkspaceDB` is
always constructed under `get_user_data_dir()`, which is guaranteed to exist first.

**Fix: test code only.** `build_registry()` in `Tests/Workspaces/test_workspace_folder_bindings.py`
now does `tmp_path.mkdir(parents=True, exist_ok=True)` before constructing `WorkspaceDB`, mirroring
what a real caller's directory is guaranteed to have. No production change -- both the hardening and
`add_folder_binding`'s existing rejection of nonexistent folders are correct as-is.

**Before/after:** `test_workspace_folder_bindings.py`: 1 failed / 10 passed -> 11 passed.
`Tests/Workspaces/` overall: 138 passed, no regressions.

## TASK-965 -- 33 failing Skills tests

**Already fixed on current `origin/dev`; no code change made.** `Tests/Skills/` ran clean twice in a
row on this worktree: 379 passed, 0 failed both times (re-run specifically to rule out flakiness given
the task's claimed 33-failed baseline).

This worktree started 26 commits behind `origin/dev` and was fast-forwarded to catch up. Among the
commits picked up, `ee49881d2` ("test: make sensitive-path and skills-fixture tests re-derive paths",
TASK-866, landed the same day, after this task's triage) rewrote `Tests/conftest.py`'s
`make_trust_service` fixture and `Tests/Skills/test_skills_library_flow.py`'s trust-service builders,
and as a side effect of deriving `trust_dir` from the real accessor, added
`trust_dir.mkdir(parents=True, exist_ok=True)` before constructing the trust store. That is exactly
root cause #3 (a test not pre-creating a config parent directory) -- fixed incidentally by an unrelated
hygiene task before this one started.

Root causes #1 (`current_runtime_backend` read-only property) and #2 (`provider_model_resolution`'s
`persisted_defaults must be a mapping`) do **not** appear anywhere in `Tests/Skills/` or the production
code it exercises -- grepped `Skills_Interop/`, the Skills UI screen, and `Tests/Skills/` for both
symbols with zero hits; no `TldwCli(...)` construction and no `resolve_effective_provider_model` call
anywhere in that directory. Both symptoms are real, but they reproduce in `Tests/UI/` (this same batch's
TASK-966 file sets `current_runtime_backend` on a stub `App`, and several other `Tests/UI/*` files
construct a real `TldwCli` or call `resolve_effective_provider_model` directly) -- not in `Tests/Skills`.
Likely explanation: the original triage ran a larger/combined failure set and misfiled two UI-suite
causes under the Skills task.

No test relaxed; nothing needed changing in `Tests/Skills/` itself.

**Before/after:** claimed 33 failed / 342 passed -> confirmed 379 passed / 0 failed (x2 runs) on current dev.

## TASK-966 -- six chat-API-key tests, `KeyError: 'openai'`

**Root cause**, confirmed via full traceback + captured logs (not the `KeyError` alone):
`mount_settings_window()`'s `mock_config_path` fixture monkeypatches
`tldw_chatbook.config.DEFAULT_CONFIG_PATH` to `temp_config_path`. But every code path these six tests
exercise (`get_provider_readiness`, `save_setting_to_cli_config` -> `apply_settings_mutation_to_cli_config`,
`load_cli_config_and_ensure_existence`) resolves its path via `_get_effective_config_path()`, which
checks the `TLDW_CONFIG_PATH` **environment variable first** and only falls back to
`DEFAULT_CONFIG_PATH` when unset. `Tests/conftest.py`'s autouse `isolate_test_environment` fixture
always sets `TLDW_CONFIG_PATH` (to a sandbox path under a *different* per-test directory) for every
test in the whole suite -- so the `DEFAULT_CONFIG_PATH` patch was silently a no-op for these tests: they
read/wrote a config file the test never touched, while asserting against `temp_config_path`, which was
never written by app code.

**Fix: test code only.** `mock_config_path` (autouse, applies to the whole file) now also does
`monkeypatch.setenv("TLDW_CONFIG_PATH", str(temp_config_path))`, so it actually controls the path every
real code path resolves. No production change -- the env-var-over-module-constant precedence is
intentional and load-bearing (it's exactly what `Tests/conftest.py`'s own sandboxing depends on).

**Before/after:** `Tests/UI/test_tools_settings_window.py`: 6 failed / 40 passed / 16 skipped ->
46 passed / 16 skipped (unrelated, pre-existing "AppTest not available in this version of Textual"
skips), 0 failed.

## Left failing

Nothing. All four tasks are green on this worktree; TASK-965 required no code change (already fixed
upstream). Known, pre-accepted environment gaps (`pytest-mock`/`numpy` absent) were not touched and are
out of scope per the task instructions.

## Files touched

- `tldw_chatbook/DB/private_sqlite.py` (production: two new owner-registry entries)
- `tldw_chatbook/DB/Subscriptions_DB.py` (production: qualify `ensure_site_configs_schema`)
- `tldw_chatbook/Notes/file_notes_replica.py` (production: qualify `FileNotesReplica.__init__`)
- `backlog/docs/sqlite-private-owner-inventory.md` (docs: C36/C37 rows)
- `Tests/DB/test_private_sqlite_inventory.py` (test: id-range assertion widened to match new rows)
- `Tests/Workspaces/test_workspace_folder_bindings.py` (test: `build_registry` creates its own dir)
- `Tests/UI/test_tools_settings_window.py` (test: `mock_config_path` also pins `TLDW_CONFIG_PATH`)
- Four task files under `backlog/tasks/` updated with Implementation Notes and status Done.
