---
id: TASK-19860
title: >-
  Migration SQL files are missing from the built wheel and sdist
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - packaging
  - bug
  - database
  - release-blocker
priority: high
dependencies: []
---

## Description

Source: surfaced by the reviewer of **TASK-19632/19633**, who found
`Tests/Packaging/test_mcp_unified_distribution.py` red on dev for a reason
unrelated to that branch. Re-verified at `3605bd52d`.

A user who installs `tldw_chatbook` from PyPI cannot start the application. The
main conversations/notes/characters database refuses to initialize, because the
migration scripts it needs were never packaged.

The cause is that the migration scripts are listed **one file at a time** in
both `pyproject.toml`'s `package-data` map and `MANIFEST.in`, rather than being
matched by a glob. Every new migration since the lists were last touched is
therefore absent from the artifact. Measured at `3605bd52d`:

- 31 `.sql` files exist under `tldw_chatbook/DB/migrations/`
- 11 are enumerated in `pyproject.toml` → **20 missing from the wheel**
- 9 are enumerated in `MANIFEST.in` → **22 missing from the sdist**

The current schema version is 44 (`ChaChaNotes_DB.py:450`). Three of the
missing files sit directly on the fresh-install upgrade chain:

- `chachanotes_v40_to_v41_persona_visual.sql`
- `chachanotes_v41_to_v42_console_project_context.sql`
- `chachanotes_v43_to_v44_sync_conflict_preservation.sql`

A PyPI install dies at the first of them with
`SchemaError: Migration from V40 to V41 failed ... No such file or directory:
chachanotes_v40_to_v41_persona_visual.sql`. The existing packaging test only
ever reports that one, because the migration chain aborts there and the two
later gaps are never reached — so the visible symptom badly under-reports the
size of the hole.

This compounds the holistic review's standing finding that migrations brick the
DB on a partial apply (TASK-19551 family): here the partial apply is caused by
the packaging step itself, on a completely ordinary install.

The mechanical fix is a `*.sql` glob in both files. That is necessary but not
sufficient as a task outcome: **enumeration trailing reality is the root
cause**, and the same trap will re-form for any other asset directory that is
listed by hand. What is missing is a check that the *built artifact* — not the
source tree, and not the config file — contains every migration the code can
ask for.

## Acceptance Criteria

- [ ] A wheel built from a clean checkout contains every `.sql` file present
      under `tldw_chatbook/DB/migrations/`
- [ ] An sdist built from a clean checkout contains every `.sql` file present
      under `tldw_chatbook/DB/migrations/`
- [ ] Installing the built wheel into an empty environment and initializing the
      ChaChaNotes database from scratch reaches the current schema version
      without a `SchemaError`
- [ ] A test fails when a migration file exists in the source tree but is
      absent from the built artifact, and it reports **all** missing files
      rather than aborting at the first — mutation-checked by removing one
      migration from the packaging config and confirming the test names it
- [ ] The test asserts against the contents of the built artifact, not against
      the text of `pyproject.toml` / `MANIFEST.in`, so a future packaging
      mechanism change cannot make it vacuously pass
- [ ] Other hand-enumerated asset lists in `pyproject.toml` / `MANIFEST.in` are
      audited for the same enumeration-trails-reality shape and the result is
      recorded (either globbed, or documented as deliberately explicit)

## Notes

Rate this as a release blocker rather than a defect: the failure mode is "the
application does not start after a normal install", and it has been shippable
for as long as the enumeration has been stale.
