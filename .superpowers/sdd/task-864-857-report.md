# TASK-864 + TASK-857 — implementation report

Worktree: `/Users/macbook-dev/Documents/GitHub/wt-864-857`
Branch: `fix/sql-validation-and-workspace-denylist` (cut from `origin/dev`)
Commits:
- `a3d0a5189` — fix(db): reconcile sql_validation VALID_TABLES/VALID_COLUMNS with live schema
- `b234c92ff` — fix(workspaces): consult the sensitive-path denylist at the folder-binding gate

Shared theme: a check that cannot reject anything, or that rejects something
real, is worse than no check. Both fixes below were reproduced broken first,
fixed, and reproduced fixed, with the exact commands and output kept here.

---

## TASK-864 — `sql_validation.VALID_TABLES` / `VALID_COLUMNS`

### Reproduction — before

Ran from the worktree, `tldw_chatbook.__file__` confirmed resolving to the
worktree (not the main checkout), `HOME`/`TLDW_CONFIG_PATH` redirected to a
scratch temp dir:

```
PYTHONPATH=/Users/macbook-dev/Documents/GitHub/wt-864-857 \
HOME=<scratch tmp> TLDW_CONFIG_PATH=<scratch tmp>/config.toml \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python - <<'EOF'
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
db = CharactersRAGDB(":memory:", client_id="repro-client")
cid = db.add_keyword_collection("Coll A")
print("created collection id:", cid)
db.update_keyword_collection(cid, {"name": "Coll B"}, expected_version=1)
EOF
```

Output (before the fix, on the branch point / `origin/dev`):

```
created collection id: 1
WARNING  sql_validation:validate_table_name:287 - Table 'keyword_collections' not in whitelist for chachanotes database
UPDATE FAILED: ValueError Invalid table name: keyword_collections
```

Confirms the task's claim exactly: `add_keyword_collection` succeeds,
`update_keyword_collection` raises unconditionally.

### Reproduction — after

Same script, after the fix:

```
created collection id: 1
UPDATE SUCCEEDED: True
[{'id': 1, 'name': 'Coll B', 'parent_id': None, ..., 'version': 2}]
```

### How the table list was reconciled

Instantiated a real, fully-migrated `CharactersRAGDB(":memory:")` (runs
`_FULL_SCHEMA_SQL_V4` plus every `_migrate_from_vX_to_vY` step up to
`_CURRENT_SCHEMA_VERSION = 27`) and read `sqlite_master` directly:

```python
db = CharactersRAGDB(":memory:", client_id="probe")
rows = db.get_connection().execute(
    "SELECT name, type FROM sqlite_master WHERE type IN ('table','view')"
).fetchall()
```

This produced 108 `sqlite_master` rows. Filtering out `sqlite_sequence` and
every name containing `_fts` (the FTS5 virtual tables and their
`_data`/`_idx`/`_docsize`/`_config` shadow tables — written to exclusively by
SQL triggers, never through a generic CRUD helper that calls
`validate_table_name`) leaves **47 substantive tables**.

This is where deriving from the *live* schema mattered rather than trusting
the task's own audit text: the audit listed 26 missing names
(`keyword_collections` + 25 others). The live schema actually has **38**
tables missing from the old 9-name allowlist — the 26 the audit found, plus
**12 more `rag_*` tables** (`rag_answer_attempt_payloads`,
`rag_artifact_owner_leases`, `rag_artifact_owner_operations`,
`rag_citation_traces`, `rag_evidence_runs`, `rag_evidence_snapshots`,
`rag_identity_context`, `rag_legacy_migration_journal`,
`rag_message_trace_owners`, `rag_payload_tombstones`,
`rag_source_observations`, `rag_trace_evidence_refs`) that schema v27 (RAG
citation provenance, merged since the audit ran) added. A hand-copy of the
audit's own list would already have been incomplete on today's `dev`.

### Was the list derived, or hand-reconciled?

**Hand-reconciled**, deliberately, not derived at import/runtime. Reasoning
kept as a comment above `VALID_TABLES` in `sql_validation.py`:

- Building a live `CharactersRAGDB(":memory:")` as a side effect of
  validating a table name would mean a lightweight, dependency-free SQL
  identifier validator — shared by three otherwise-unrelated DB modules
  (chachanotes/media/prompts, each with its own schema) — running all 27
  migrations (with ~30 log lines) every time it needs the chachanotes table
  set.
- It would also require a lazy in-function import back into
  `ChaChaNotes_DB.py` (which itself imports from `sql_validation` at module
  scope) to avoid a circular import — workable, but adds a hidden runtime
  dependency in the wrong direction for what should be the app's most
  foundational, dependency-free validation module.

Instead: the allowlist stays a hand-maintained `set`, and
`Tests/DB/test_sql_validation.py::TestChachanotesValidTablesMatchesLiveSchema`
derives the real table set the same way (`sqlite_master` on a live migrated
DB) and asserts equality in **both** directions — `test_no_missing_tables`
and `test_no_stale_tables` — so the next schema/allowlist divergence fails a
test immediately, in CI, rather than surfacing as a user's `ValueError`. This
is what the task's own AC #3 asked for ("if deriving is not practical, say
why, and add a test that compares the list against the real schema").

### `VALID_COLUMNS` decision (AC #4)

Chose **both** options together, not either alone:

1. Enumerated every real call site of `validate_column_name(column,
   table_name)` across the repo. The only tables ever passed with a concrete
   `table_name` (i.e. real code paths, not the two `_get_next_version` dead
   helpers in `Prompts_DB.py`/`Client_Media_DB_v2.py` that have zero
   callers) are: `character_cards`, `conversations`, `messages`, `notes`,
   `keywords` (already present); `keyword_collections`
   (`update_keyword_collection`/`soft_delete_keyword_collection`);
   `sync_profile_state` (`Sync_Interop/sync_state_repository.py`'s
   `_ensure_sync_v2_profile_columns`, immediately before an `ALTER TABLE ...
   ADD COLUMN` f-string); and `Transcripts`/`MediaChunks`/
   `UnvectorizedMediaChunks`/`DocumentVersions` (`Client_Media_DB_v2.py`'s
   soft-delete/undelete cascade loops).
2. Added `VALID_COLUMNS` entries for all six, columns verified against each
   table's real `CREATE TABLE` statement (and `PRAGMA table_info` for
   `keyword_collections`, to be certain — it has no `uuid`/`deleted_at`
   unlike its siblings).
3. Changed `validate_column_name`'s fallback from "table not in
   `VALID_COLUMNS` → skip the per-table check, return whatever
   `validate_identifier` alone decided" to "table not in `VALID_COLUMNS` →
   return `False`" (fail closed). `table_name=None` (no schema context) is
   unaffected and still skips the per-table check entirely.

Justification: failing closed with nothing backfilled would have repeated
the exact TASK-864 pattern — an absent allowlist entry unconditionally
breaking a currently-working call site (this time `sync_profile_state`'s
migration path, or the Media cascade tables). Backfilling without failing
closed leaves the exact silent-no-op gap the audit flagged, where a future
caller passing an unregistered table gets no column-specific check at all
and no signal that it's missing. Doing both closes the gap without
repeating 864's mistake. The two dead `_get_next_version` helpers were left
untouched (no live callers today); if ever revived for a table without a
`VALID_COLUMNS` entry, failing closed is the correct, safe behavior for them
too — not a regression, since nothing calls them currently.

### Test results

- `Tests/DB/test_sql_validation.py` — 23 passed (was 21; 2 new test methods
  in `TestChachanotesValidTablesMatchesLiveSchema`).
- `Tests/ChaChaNotesDB/test_chachanotes_db.py` + `Tests/DB/` — 769 passed, 1
  skipped, 1 pre-existing failure (`test_private_sqlite_inventory.py`,
  confirmed failing identically on pristine `origin/dev` via `git stash` —
  unrelated raw-`sqlite3.connect()` census drift in `Subscriptions_DB.py`,
  not touched by this change).

---

## TASK-857 — workspace folder-binding vs. the sensitive-path denylist

### Reproduction — before

```
PYTHONPATH=/Users/macbook-dev/Documents/GitHub/wt-864-857 \
HOME=<scratch tmp> TLDW_CONFIG_PATH=<scratch tmp>/.config/tldw_cli/config.toml \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python - <<'EOF'
from tldw_chatbook.config import get_user_data_dir, _get_effective_config_path
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService
registry = LocalWorkspaceRegistryService(WorkspaceDB(<tmp path>, client_id="repro"))
registry.ensure_default_workspace()
registry.create_workspace(workspace_id="ws-a", name="Client A")
registry.add_folder_binding("ws-a", get_user_data_dir())
registry.add_folder_binding("ws-a", _get_effective_config_path().parent)
registry.add_folder_binding("ws-a", Path.home() / ".ssh")
EOF
```

Output (before the fix, on the branch point):

```
BOUND get_user_data_dir() SUCCESSFULLY: .../.local/share/tldw_cli/default_user
BOUND ~/.config/tldw_cli SUCCESSFULLY: .../.config/tldw_cli
BOUND ~/.ssh SUCCESSFULLY: .../.ssh
```

All three — this app's own database directory, its own config/API-key
directory, and the user's SSH key directory — bound as workspace folder
roots without any rejection. An ordinary project folder also bound
successfully (as it must continue to).

### Reproduction — after

Same script, after the fix, plus two more cases (ancestor-of and
subdirectory-of `get_user_data_dir()`):

```
REJECTED get_user_data_dir(): '.../default_user' cannot be bound: it is, or
  contains, the protected path '.../default_user'. Choose a folder that does
  not overlap this application's own data, configuration, or credential
  directories.
REJECTED ~/.config/tldw_cli: '.../.config/tldw_cli' cannot be bound: it is,
  or contains, the protected path '.../.config/tldw_cli'. ...
REJECTED ~/.ssh: '.../.ssh' cannot be bound: it is, or contains, the
  protected path '.../.ssh'. ...
REJECTED parent-of-user-data-dir: '.../.local/share/tldw_cli' cannot be
  bound: it is, or contains, the protected path
  '.../.local/share/tldw_cli/default_user'. ...
REJECTED subdir-of-user-data-dir: '.../default_user/some_new_subdir' cannot
  be bound: it is, or contains, the protected path '.../default_user'. ...
BOUND ordinary project dir: <tmp>/Projects/myapp
```

Every message names the exact conflicting path (see "priority ordering"
below for why `get_user_data_dir()` itself is reported as *itself*, not as
containing the skill-trust subtree three levels down).

### The containment rule chosen

Reject a candidate root `R` (already resolved) if `R`:

1. **IS** one of: the fixed sensitive directories (`~/.ssh`, `~/.aws`,
   `~/.gnupg`, `~/.config/gcloud`, `~/.docker`, `~/.kube`,
   `~/.local/share/keyrings`, plus the skill-trust subtree), or one of this
   app's own state-container directories (`get_user_data_dir()`, the
   effective config directory, the ChromaDB persist directory, the
   RAG-profile directory), or one of the app's sensitive single
   files/database paths (+ WAL/SHM/journal sidecars); **or**
2. is **NESTED INSIDE** one of those directories (a subdirectory of
   `get_user_data_dir()`, even one that doesn't look sensitive by name);
   **or**
3. **CONTAINS** one of those directories or files as a descendant (e.g.
   `~/.local/share`, which contains `get_user_data_dir()`; `~/.config`,
   which contains `~/.config/tldw_cli`).

Implemented as `Utils.sensitive_paths.find_root_binding_conflict()`, reusing
the exact same `SensitivePathContext` (`resolve_sensitive_context()`) that
`is_sensitive_path` already resolves for read/write-time checks — so the
binding gate can never enumerate a different protected-path set than the
file tools do.

**Why "any subdirectory of `get_user_data_dir()`", not just the specific
files/dirs `is_sensitive_path` would flag on a per-path basis:** `
is_sensitive_path`'s direct-child-file rule deliberately treats an
*existing directory* nested inside `get_user_data_dir()` (e.g.
`tool_sandbox`) as exempt, so individual file reads under it stay reachable
— that rule exists to catch stray loose files landing directly in the data
directory, not to certify the whole directory safe as a *binding root*.
Binding grants blanket, recursive reachability to everything under the
root; nothing legitimate needs to bind `get_user_data_dir()` (or anything
inside it) directly in the first place, because
`Tools.workspace_file_roots.allowed_file_roots` already includes the
sandbox root (`get_user_data_dir()/tool_sandbox`) automatically, without
ever going through `add_folder_binding`. So the binding gate is
deliberately *coarser* than the per-path read-time check — refusing the
whole container and everything in it — while the per-path check stays
exactly as permissive as before for actual reads/writes (nothing about
`is_sensitive_path` itself changed).

**Why this doesn't degenerate into "refuse everything above home":**
nothing hardcodes home as a boundary. The rule is purely "is, is nested in,
or contains one of the *concrete* protected paths" — `~/.local/share` gets
refused because it happens to contain `get_user_data_dir()`, not because of
any home-relative rule. An ordinary project folder elsewhere on disk
(`~/Projects/myapp`, or anywhere else that isn't an ancestor/descendant of
one of the concrete protected paths) is untouched. In practice, on this
platform, refusing every root that would reach one of the protected
directories does end up refusing most coarse ancestors up to and including
home — because most of the protected directories live under home — but
that is a consequence of where the protected paths happen to live, not a
rule written in terms of home itself.

**Priority/tie-breaking when several protected paths match at once:**
checked in three ordered passes — (1) exact self-match, (2) nested-inside
(tie-broken toward the *deepest*/closest enclosing directory), (3) contains
(tie-broken toward the *shallowest*/closest contained path). This is why
binding `get_user_data_dir()` itself is reported as case (1) — "it IS the
protected path" — even though it also technically *contains* the
skill-trust subtree nested three levels down; and why binbinding its
*parent* is reported as containing `get_user_data_dir()` itself (the
nearest contained protected item), not the more deeply nested skill-trust
subtree beneath it. Verified with a dedicated regression test
(`test_find_root_binding_conflict_when_root_is_the_user_data_dir` /
`..._when_root_contains_a_container`) after an initial implementation
picked the wrong (more obscure) match on the first pass.

### Rejection message / no silent failure

`add_folder_binding` raises `WorkspaceRegistryServiceError` with the exact
conflicting path named:

```
'{resolved}' cannot be bound: it is, or contains, the protected path
'{conflict}'. Choose a folder that does not overlap this application's own
data, configuration, or credential directories.
```

`UI/Screens/settings_screen.py`'s existing handler for this exact call site
already catches `WorkspaceRegistryServiceError` and displays `str(exc)`
verbatim in the Settings → Workspaces pane — no UI change was needed for
the message to reach the user; it was already wired to surface whatever
`add_folder_binding` raises.

### Read-time behavior, and existing bindings

**Unchanged.** `is_sensitive_path`'s own per-path checks (and the
direct-child-file exemption for existing containers like `tool_sandbox`)
were not touched — this fix is scoped to the binding *gate*, per the task.
Consequence worth stating plainly: a folder bound **before** this fix that
is, or contains, a protected path is **not retroactively unbound or
re-validated**. Such a binding could still be unsafe until a user removes
it (or re-adds it, which now goes through the new check) — no migration or
background re-validation was added, since the task scoped this to the
binding gate and asked to say so rather than silently attempt a migration.

### Test results

- `Tests/Utils/test_sensitive_paths.py` — all pass (5 new tests for
  `find_root_binding_conflict`, covering all three cases plus the
  ordinary-folder no-conflict case).
- `Tests/Workspaces/test_workspace_folder_bindings.py` — all pass except one
  **pre-existing** failure (`test_add_folder_binding_rejects_duplicates_and_nesting`),
  confirmed failing identically on pristine `origin/dev` via `git stash`
  (unrelated `private_paths` "missing_directory" issue when a second
  `WorkspaceDB` is constructed under a not-yet-created subdirectory — not
  caused by this change). One pre-existing test
  (`test_add_folder_binding_validation_matrix`) needed a one-line update:
  it previously bound `tmp_path` directly to test unknown-workspace
  precedence; this suite's own `Tests/conftest.py` autouse HOME-redirection
  fixture nests its fake config directory under `tmp_path`
  (`tmp_path/test_data/config`), so `tmp_path` itself is now *correctly*
  rejected as a folder-binding root under the new rule. Updated to bind a
  freshly created subdirectory instead, which is what the test actually
  intended to exercise.
- `Tests/DB/` — 596 passed, 1 skipped, the same one pre-existing
  `test_private_sqlite_inventory.py` failure as above (unrelated).
- `Tests/Tools/test_workspace_file_roots.py` +
  `test_file_tools_workspace_roots.py` + `test_file_tool_sandbox.py` — 31
  passed.
- `Tests/Utils/` (full directory) — 553 passed.

---

## Environment notes for anyone re-running these reproductions

- Pin the interpreter and `PYTHONPATH` to the worktree explicitly:
  `PYTHONPATH=/Users/macbook-dev/Documents/GitHub/wt-864-857
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`. The
  worktree has no `.venv` of its own, and an editable install resolves
  `tldw_chatbook` to the main checkout unless `PYTHONPATH` wins.
  `tldw_chatbook.__file__` was checked before trusting every probe in this
  session.
- Every ad hoc probe redirected `HOME` to a fresh scratch temp directory and
  set `TLDW_CONFIG_PATH` to a file inside it — never the real user config or
  databases. `WorkspaceDB`/`CharactersRAGDB` construction with a
  `TLDW_CONFIG_PATH` under a not-yet-existing directory needs that
  directory pre-created (`private_paths` verification fails closed
  otherwise) — e.g. `mkdir -p "$HOME/.config/tldw_cli"` before setting
  `TLDW_CONFIG_PATH=$HOME/.config/tldw_cli/config.toml`.
- `CharactersRAGDB(":memory:")` works fine as a single-process probe;
  `WorkspaceDB(":memory:")` does not (each connection gets its own private
  `:memory:` database) — use a real temp file path for `WorkspaceDB` in ad
  hoc scripts, as the existing test suite already does.
