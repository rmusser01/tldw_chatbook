# DB migrations

`.sql` steps for the SQLite schemas, applied by the `_migrate_from_vX_to_vY`
methods in the matching `DB/*.py` module. Filenames are
`<db>_v<from>_to_v<to>_<what>.sql`.

## Adding a migration

1. Bump `_CURRENT_SCHEMA_VERSION` in the owning DB module.
2. Add `<db>_v<n>_to_v<n+1>_<what>.sql` here **and** the
   `_migrate_from_v<n>_to_v<n+1>` step that runs it.
3. Packaging: **nothing to do** (TASK-19860). `pyproject.toml` matches
   `migrations/*.sql` and `MANIFEST.in` does `recursive-include
   tldw_chatbook/DB/migrations *.sql`, so a new script ships the moment it
   lands. Do not re-introduce a per-file list, and do not add a fifth one.

   The four hand-written lists this replaced (`MANIFEST.in`,
   `[tool.setuptools.package-data]`, `Packaging/check_manifest.py`, and
   `Tests/Packaging/test_installed_distribution.py`) agreed with each other
   and with nothing else: of the 32 files present they named 13, 11, 13 and
   13, and two files the schema runner actually reads —
   `chachanotes_v40_to_v41_persona_visual.sql` and
   `chachanotes_v45_to_v46_sync_log_retention.sql` — were in none of them, so
   a `pip install` walled at V40 with a `SchemaError` and the app did not
   start. Both checkers now *derive* the requirement (from the `.sql` files in
   the checkout, and from the `.sql` names the artifact's own
   `ChaChaNotes_DB.py` opens) and assert it against the built wheel and sdist.
4. **If the migration contains `CREATE TABLE`, add every new table name to
   `VALID_TABLES['chachanotes']` in `DB/sql_validation.py`, in the same
   commit.** See below — this is the step that keeps getting missed.
5. **If it contains `CREATE INDEX`, add the index to
   `EXPECTED_CHACHANOTES_INDEXES` in `Tests/ChaChaNotesDB/test_index_census.py`.**
6. Run `./scripts/preflight.sh`. It checks 4 and reports exactly what to paste.

## The table-allowlist trap (TASK-20971)

`VALID_TABLES` is a hand-maintained allowlist keyed by database. It is not
decoration: `validate_table_name()` rejects any name that is not in it, so
every generic CRUD helper routed through it raises unconditionally for a table
you forgot to list. TASK-864 was filed because `keyword_collections` was one of
38 tables missing from it, which made `update_keyword_collection()` raise for
everyone.

It has now gone stale three times, and the third time is the reason this
section exists:

| When | What |
| --- | --- |
| TASK-864 | 9 of ~47 tables listed. Added `Tests/DB/test_sql_validation.py::TestChachanotesValidTablesMatchesLiveSchema` as a pin. |
| TASK-19568, merged `aaec11812` 2026-08-22 00:16 -0700 | Pin had gone red; entry repaired. |
| TASK-19057, merged `2fe6ca20f` 2026-08-22 14:51 -0700 | v44→v45 added `actor_portable_identities` and `actor_pack_persona_intents`. Pin red again. **Fourteen and a half hours.** |

The instructive part is *why* TASK-19057 updated one hand-maintained literal
and not the other. It correctly added `idx_actor_pack_persona_intents_state` to
the index census, and it correctly bumped three schema-version pins under
`Tests/DB/`. Measured on that branch, those two literals were reachable by the
two things the author actually did:

* the index census lives in `Tests/ChaChaNotesDB/`, beside the migration test
  they wrote — running that directory turned it red; and
* the version pins contain the schema version number, which a grep for it
  finds.

`VALID_TABLES` is reachable by neither. It names no schema version, and
nothing else in `Tests/DB/test_sql_validation.py` mentions the feature. Its
guard survived by geography for years and stopped surviving the moment a
migration landed from a directory that did not happen to sit next to it.

So the pin is not the fix on its own — a guard that only reports to whoever
runs it reports after the merge. The authoring-time guard is:

```
python3 scripts/check_schema_table_allowlist.py     # also in ./scripts/preflight.sh
```

It statically scans the `CREATE TABLE` statements in
`migrations/chachanotes_*.sql` and in the SQL string literals of
`ChaChaNotes_DB.py`, requires each name to be in `VALID_TABLES['chachanotes']`
in both directions, and prints the lines to paste. Stdlib-only, no database
built, ~0.1 s, and it is a step in the required `derived-artifacts` CI job.

Its expectation comes from the migration SQL, never from `VALID_TABLES` — the
rule TASK-19045 established for the index census: a census that re-derives its
expectation from the artifact it checks is the identity function on exactly the
defect class it exists to catch. That is also why neither literal is
auto-generated, and why "just generate it from the schema" is the wrong repair.

`VALID_TABLES['media']` and `VALID_TABLES['prompts']` are drifted in both
directions and are **not** covered by this checker yet; TASK-19867 owns them.
