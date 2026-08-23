---
id: TASK-19860
title: >-
  Migration SQL files are missing from the built wheel and sdist
status: Done
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

- [x] A wheel built from a clean checkout contains every `.sql` file present
      under `tldw_chatbook/DB/migrations/`
- [x] An sdist built from a clean checkout contains every `.sql` file present
      under `tldw_chatbook/DB/migrations/`
- [x] Installing the built wheel into an empty environment and initializing the
      ChaChaNotes database from scratch reaches the current schema version
      without a `SchemaError`
- [x] A test fails when a migration file exists in the source tree but is
      absent from the built artifact, and it reports **all** missing files
      rather than aborting at the first — mutation-checked by removing one
      migration from the packaging config and confirming the test names it
- [x] The test asserts against the contents of the built artifact, not against
      the text of `pyproject.toml` / `MANIFEST.in`, so a future packaging
      mechanism change cannot make it vacuously pass
- [x] Other hand-enumerated asset lists in `pyproject.toml` / `MANIFEST.in` are
      audited for the same enumeration-trails-reality shape and the result is
      recorded (either globbed, or documented as deliberately explicit)

## Implementation Plan

1. Re-measure at the base commit: count `.sql` files in the source tree and in
   **every** hand-enumerated list, then build a wheel and an sdist from a clean
   `git archive` export and enumerate the missing files from the artifacts'
   own contents (born-red evidence).
2. Replace the enumerations with globs: `migrations/*.sql` in
   `pyproject.toml`'s `package-data`, `recursive-include ... *.sql` in
   `MANIFEST.in`.
3. Make `Packaging/check_manifest.py` derive its required migration set
   instead of listing it: from the adjacent source tree, and independently
   from the `.sql` names the artifact's own `ChaChaNotes_DB.py` opens. Fail
   closed when either source is unavailable or empty.
4. Make `Tests/Packaging/test_installed_distribution.py` derive the same two
   sets and assert them against wheel and sdist **members**, reporting every
   missing file at once instead of aborting at the first.
5. Extend the installed-wheel probe to initialize a ChaChaNotes database
   **from scratch** (no version pin) and read the reached schema version back
   out of `db_schema_version`.
6. Mutation-check: drop one migration from the packaging config, rebuild,
   confirm the test names that file, Edit-restore, confirm a clean `git diff`.
7. Audit every other hand-enumerated asset list in `pyproject.toml` /
   `MANIFEST.in` against the built artifacts and record the verdict for each.

## Implementation Notes

Replaced four hand-maintained migration lists with two derivations, and moved
the packaging assertion off the config text onto the built artifacts.

**Re-measured at `d60ebe1d0` (the numbers in the Description were already
stale, which is the defect).** 32 `.sql` files exist; `pyproject.toml` named
13, `MANIFEST.in` named 11, and **two further lists the task did not know
about** named 13 each: `Packaging/check_manifest.py`'s
`REQUIRED_SDIST_PATHS`/`REQUIRED_WHEEL_PATHS`, and
`Tests/Packaging/test_installed_distribution.py`'s `RUNTIME_MIGRATION_PATHS`
(built from 15 constants, one of which — `TRANSCRIPT_ANNOTATIONS_MIGRATION_PATH`
— was defined twice). All four agreed with each other and with nothing else,
which is why ~90 packaging tests were green over a release blocker. Wheel and
sdist each carried 13 of 32 (`MANIFEST.in`'s 11 is masked: setuptools also
folds `package-data` into the sdist). `ChaChaNotes_DB.py` opens **15** scripts
at runtime; two of them — `chachanotes_v40_to_v41_persona_visual.sql` and
`chachanotes_v45_to_v46_sync_log_retention.sql` — were in no list at all.

**Changes.** `pyproject.toml` → `"tldw_chatbook.DB" = ["migrations/*.sql"]`;
`MANIFEST.in` → `recursive-include tldw_chatbook/DB/migrations *.sql`.
`check_manifest.py` no longer lists migrations: it derives them from the
source tree beside it *and*, independently, from the `.sql` names the
**artifact's own** `ChaChaNotes_DB.py` opens (so the requirement survives
being run where no checkout exists). Both derivations fail closed — an absent
migrations directory or a regex that stops matching is a reported error, not
an empty requirement. The test file derives the same two sets and asserts them
against `ZipFile`/`TarFile` members, listing every missing file in one
message.

**Evidence.** Born-red at base: a wheel and sdist built from a clean
`git archive` export each omitted the same **19** files (named in full in the
run log); the base `check_manifest.py` accepted that wheel, the new one names
all 19 per archive. A clean-environment install of the base wheel
(`pip install --no-deps --no-index` into an empty venv) reproduced the user's
failure exactly — `SchemaError: Migration from V40 to V41 failed ... No such
file or directory: chachanotes_v40_to_v41_persona_visual.sql` — confirming the
chain walls at v40 and under-reports 18 further gaps. The same probe on the
fixed wheel imported from the fresh venv's `site-packages`, initialized a
ChaChaNotes DB **from scratch**, and read **version 46 out of
`db_schema_version`** (130 tables), matching `_CURRENT_SCHEMA_VERSION` without
that constant being the assertion's source. Mutation: enumerating 31 of 32 in
`pyproject.toml` plus one `MANIFEST.in` `exclude`, then rebuilding, made four
tests name `chachanotes_v45_to_v46_sync_log_retention.sql` and killed the
installed-wheel probe with the real `SchemaError`; Edit-restored, `git diff`
shows only the intended change. `Tests/Packaging/` **97 passed** (was 2 failed
at base: `test_mcp_unified_distribution.py::test_mcp_extra_installs_and_runs_
from_each_isolated_artifact[wheel|sdist]`, red for this defect). Repo-wide
`--collect-only -q`: 56,668 collected, 1 error —
`Tests/UI/test_library_file_notes_workspace.py`, the known dev red (TASK-20972).

**Asset-list audit (last AC).** Every non-`.py` file under `tldw_chatbook/`
(189) was grouped by directory + extension and diffed against both artifacts:
60 groups, of which only two were partial. Migrations were the defect. The
other, `Config_Files/*.toml`, is deliberate — `embedding_configs_examples.toml`
is in `FORBIDDEN_WHEEL_PATHS`, which is exactly why `rag_pipelines.toml` must
stay a literal rather than becoming a `*.toml` glob. Three single-file lists
stay explicit with the reason now written beside them in `pyproject.toml`: the
pinned `TTS/audio_cpp_artifact_manifest.json` (its repository/commit/package
count are pinned by tests), and the vendored `LICENSE` notices (a fixed legal
obligation per package, not a growing directory). Everything else was already
complete or excluded from both artifacts by design.

**Deliberately not done.** `test_installed_distribution_migrates_v35_database_
to_current` keeps its name despite now also covering the from-scratch path;
two backlog documents cite it by name and the rename buys nothing.

**Files:** `pyproject.toml`, `MANIFEST.in`, `Packaging/check_manifest.py`,
`Packaging/PACKAGING_CHECKLIST.md`,
`Tests/Packaging/test_installed_distribution.py`,
`backlog/docs/lessons-testing-evidence.md`.

## Notes

Rate this as a release blocker rather than a defect: the failure mode is "the
application does not start after a normal install", and it has been shippable
for as long as the enumeration has been stale.
