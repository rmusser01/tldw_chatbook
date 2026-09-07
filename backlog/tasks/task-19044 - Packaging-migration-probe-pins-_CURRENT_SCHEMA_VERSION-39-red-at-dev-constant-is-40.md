---
id: TASK-19044
title: >-
  Packaging migration probe pins _CURRENT_SCHEMA_VERSION == 39 — red at dev
  (constant is 40)
status: Done
assignee: []
created_date: '2026-08-20 08:40'
labels:
  - test-health
  - db
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-16840's investigation flagged two schema-version equality pins that were
already red at the wave's base `cef56efaf` (constant 39, tests asserting 38):
`Tests/DB/test_chachanotes_trajectory_metadata_migration.py:44` and `:220`.
Those two were fixed on dev by commit `46945ebbe` (v39→v40,
`transcript_annotations`), which converted them to
`== CharactersRAGDB._CURRENT_SCHEMA_VERSION`.

But the same commit bumped the constant to 40 and left a third literal pin
behind: `Tests/Packaging/test_installed_distribution.py:807`
(`INSTALLED_MIGRATION_PROBE`) reads the constant into
`current_schema_version` and then asserts `current_schema_version == 39` —
red whenever the integration-marked packaging suite runs at dev `1bf7f234e`
(the constant is 40). History shows the literal being hand-bumped each time
(37→39 at `4a2d48046`) and missed on this bump — the exact
equality-pin-on-a-moving-constant fragility the trajectory fix just removed.
The probe's real job (monkeypatch to v35, reopen, prove the installed
migration chain reaches the current version) does not need the literal at all.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The installed-migration probe passes at current dev with no hand-maintained schema-version literal remaining in the file (owner ruling: durable over re-pinning the new number)
- [x] #2 The probe still proves what the literal stood in for: a v35 database migrates to whatever `_CURRENT_SCHEMA_VERSION` is, under the installed distribution
- [x] #3 The affected packaging test is demonstrated green at dev (run evidence, not collection)
- [x] #4 The v39→v40 migration SQL ships in the built distributions — discovered while making #1/#3 true: `chachanotes_v39_to_v40_transcript_annotations.sql` is absent from `[tool.setuptools.package-data]` (and from `Packaging/check_manifest.py` + the test's `RUNTIME_MIGRATION_PATHS` contract), so the installed migration chain cannot reach v40 regardless of how the probe asserts; the probe can only go green once the wheel/sdist actually carry the file
<!-- AC:END -->

## Implementation Plan

1. Born-red evidence: run `Tests/Packaging/test_installed_distribution.py::test_installed_distribution_migrates_v35_database_to_v38` unmodified at dev base and capture the `== 39` failure signature to a file.
2. Probe fix (AC #1/#2): drop the `assert current_schema_version == 39` pin. The probe already reads `CharactersRAGDB._CURRENT_SCHEMA_VERSION` *inside the child process against the installed distribution* and asserts the upgraded DB's version equals it (line 818) — that comparison is the real proof and stays. Add `assert current_schema_version > 35` so the v35 downgrade-monkeypatch remains a genuine downgrade (35 is the fixed fixture baseline, not a moving pin).
3. Same-class literals in the file: the sentinel pair is version-baked AND self-inconsistent since `4a2d48046` (probe prints `installed-wheel-v35-to-v39-ok`, outer test asserts `...-v38-ok`, function named `..._to_v38`). Make the sentinel version-agnostic (`installed-wheel-v35-to-current-ok`) in both places and rename the test `...migrates_v35_database_to_current`.
4. Packaging gap (AC #4): add `chachanotes_v39_to_v40_transcript_annotations.sql` to `[tool.setuptools.package-data]` in `pyproject.toml` (the mechanism that carries the two prior migrations into both wheel and sdist; MANIFEST.in stops at v36→v37 for all of them), to both hand lists in `Packaging/check_manifest.py`, and as a `RUNTIME_MIGRATION_PATHS` row in the test so the release-checker mutation tests cover it.
5. Green evidence: re-run the migration test (both wheel_source params) plus the release-checker tests that consume `RUNTIME_MIGRATION_PATHS`, output redirected to files, read in full.

## Implementation Notes

Rewrote the installed-migration probe to compare against the constant in the
environment the assertion runs in, and fixed the real packaging regression the
red probe had been masking.

**Probe (AC #1/#2).** Deleted `assert current_schema_version == 39`. The probe
already reads `CharactersRAGDB._CURRENT_SCHEMA_VERSION` inside the `python -c`
child against the installed wheel and asserts the migrated DB's version equals
it — that in-environment comparison is the proof and is unchanged. The only
literal kept is `assert current_schema_version > 35`, guarding that the fixed
v35 fixture baseline stays a genuine downgrade (35 is the fixture, not a
moving pin).

**Same-class literals in the file.** Commit `4a2d48046` had left the sentinel
pair self-inconsistent — probe printed `installed-wheel-v35-to-v39-ok`, outer
test asserted `...-v38-ok` (so the test was doubly red at dev), function named
`..._to_v38`. All three are now version-agnostic: sentinel
`installed-wheel-v35-to-current-ok v{current_schema_version}` (f-string
resolved in the child), outer assert matches the version-agnostic prefix, test
renamed `test_installed_distribution_migrates_v35_database_to_current`. Swept
the repo for references to the old name/sentinels: none.

**Shipped packaging bug (AC #4, discovered).** `46945ebbe` (v39→v40) never
added `chachanotes_v39_to_v40_transcript_annotations.sql` to
`[tool.setuptools.package-data]` (`include-package-data = false`), so built
wheels/sdists did not carry it and `_migrate_from_v39_to_v40` — which reads
that file from the installed package — failed `FileNotFoundError → SchemaError`:
installed users could not migrate an existing DB past v39. Proven by control
run (pyproject line reverted → probe red through the real chain; restored →
green). Added the file to pyproject package-data, both hand lists in
`Packaging/check_manifest.py`, and the test's `RUNTIME_MIGRATION_PATHS`
(release-checker mutation coverage). MANIFEST.in intentionally untouched: the
v37→v38/v38→v39 precedent ships via package-data alone, confirmed by the
sdist-sourced param passing.

**Run evidence** (all read from redirected files, venv + PYTHONPATH pinned to
the worktree):
- Born-red at dev base: `test_installed_distribution_migrates_v35_database_to_v38`
  → 2 failed in 214.94s, child `AssertionError` at the `== 39` pin.
- Fixed: 7 passed in 212.23s — migration probe `[source]` + `[sdist]`,
  `test_built_artifacts_match_distribution_contract`,
  `test_release_checker_accepts_fresh_artifacts`, probe-content test, and both
  new release-checker mutation params for the v39→v40 file.
- Control: pyproject line reverted → 2 failed in 61.12s
  (`Migration from V39 to V40 failed ... No such file or directory`), line
  restored (Edit-based).
- `--collect-only`: 37 tests collected, no errors.

**Files:** `Tests/Packaging/test_installed_distribution.py`, `pyproject.toml`,
`Packaging/check_manifest.py`, `backlog/docs/lessons-testing-evidence.md`
(new entry: a probe left red on a stale pin stops guarding), this task file.
