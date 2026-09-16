# PyPI Packaging Checklist for tldw_chatbook

Use this checklist to verify a distribution before release. Artifact checks
must start from a fresh output directory; do not treat an existing checkout
`dist/` as release evidence.

## Project declarations

- [x] `pyproject.toml` is the source of truth for dependencies and entry points.
- [x] `tldw-cli` and `tldw-serve` are declared in `[project.scripts]`.
- [x] `Packaging/common/version.py` derives its release version from
  `pyproject.toml`.
- [x] `[tool.setuptools.package-data]` explicitly owns wheel runtime data.
- [x] `include-package-data = false` keeps sdist-only files out of wheels.
- [x] Asset directories that grow are matched by pattern, never enumerated.
  Only three lists name files on purpose, each for a reason recorded beside
  it in `pyproject.toml`: `Config_Files/rag_pipelines.toml` (a `*.toml` glob
  would ship the forbidden example TOML), `TTS/audio_cpp_artifact_manifest.json`
  (one pinned manifest), and the licence notices. Adding a new
  runtime asset to one of those three means adding its name too.
- [x] The licence list is complete, not merely explicit. The reason it stays
  enumerated — one fixed obligation per subtree — is also what makes an
  omission a licence breach, and it had two: `LLM_Calls/LICENSE` and
  `tldw_api/LICENSE` (Apache-2.0 subtrees whose modules ship) were in no
  artifact. Re-derive it by listing every `LICEN*`/`NOTICE`/`COPYING` file
  under `tldw_chatbook/` and diffing against the built archives, not by
  reading `pyproject.toml`.
- [x] The project license uses the `AGPL-3.0-or-later` SPDX expression and
  declares `LICENSE`.

## Distribution content

- [x] Root `MANIFEST.in` is the canonical setuptools sdist manifest.
- [x] The sdist contains release metadata, runtime data, TCSS modules, the
  source-only `stats_screen.css` input, and project/vendored licenses.
- [x] The wheel contains the compiled CSS bundle, RAG pipeline configuration,
  eval configuration, configuration resources, **every** `.sql` file under
  `tldw_chatbook/DB/migrations/`, **every** `.json` under
  `tldw_chatbook/Evals/eval_datasets/` (read at runtime by the bundled
  research template), and every license notice.
- [x] The wheel excludes source-only CSS, example TOML, development Markdown,
  the namespace-discovered chunking example, tests, caches, and OS metadata.
- [x] Wheel and sdist metadata use Core Metadata 2.4 and declare the project
  license file.

## Fresh artifact gate

Build into a newly created empty output directory:

```bash
python -m build --sdist --wheel --no-isolation --outdir fresh-dist
python Packaging/check_manifest.py fresh-dist
```

`check_manifest.py` requires exactly one sdist and one wheel. It checks
required and forbidden archive paths, entry points, SPDX metadata, the project
license, and vendored notices. Its migration requirement is derived twice and
fails closed: from the `.sql` files in the checkout beside it, and from the
`.sql` names the artifact's own `ChaChaNotes_DB.py` opens. It names every
missing script, not just the first.

Run the focused release metadata gate and installed-wheel regression:

```bash
python -m pytest \
  Tests/Packaging/test_release_metadata.py \
  -q
python -m pytest \
  Tests/Packaging/test_installed_distribution.py \
  -m integration -q -p no:cacheprovider
```

The regression builds from a temporary source copy, installs the wheel with
`--no-deps`, loads packaged resources outside the checkout, runs both installed
help commands under private temporary state, and verifies that every installed
target file hash remains unchanged.

## Pre-release steps

### Local backup and recovery qualification

Complete capture and replacement require separate release decisions. The helper
and native filesystem declarations qualify only their recorded operations; neither
alone establishes that a packaged application can recover an installation.

Before advertising either capability for a distribution:

- [ ] Verify the bundled helper's protocol, digest, native pipe behavior and
  installed-wheel behavior following [helper qualification](backup_age/README.md).
- [ ] Match the actual OS, architecture, Python version, filesystem and native
  protocol to installed evidence. Keep untested combinations unavailable.
- [ ] Verify every supported owner-inventory row using the synthetic multi-profile
  fixture, including shared stores, directory metadata, selected optional assets,
  deleted data and excluded credentials.
- [ ] Exercise the packaged F9 create/inspect/isolated-restore/open flow. Qualify
  replacement and later rollback separately, including preservation of edits made
  after restore in the new encrypted safety copy.
- [ ] Verify startup-independent recovery and the applicable interruption tests,
  including SQLite sidecars, changed credentials and multiple-volume behavior.
- [ ] Confirm restored execution remains inactive until the applicable explicit
  owner review, and distinguish archive verification, installation validation,
  successful opening and remaining setup requirements in the displayed results.
- [ ] Record exact commands, revision, artifact hashes, native identity and actual
  results in the [release evidence ledger](../backlog/docs/backup-recovery-release-evidence.md).
  Do not combine results from different source snapshots into a single build claim.
- [ ] Review [user recovery instructions](../Docs/Backup-and-Recovery.md) against
  the qualified artifact, including password loss, plaintext staging, retained
  copies, credential/model exclusions and upgrade interoperability.

Use disposable profiles, private HOME/XDG/config paths and fixture credential
stores; contain network access before application imports. Run the named checks
in [Task 26](../Docs/superpowers/plans/2026-09-07-backup-recovery-06-release-evidence.md)
and its applicable native matrix. A missing test or unavailable runner leaves its
gate open; skipped checks and the boolean conjunction regression are not product
qualification. These checklist items remain unchecked until demonstrated against
the exact release build.

The manual `backup-recovery-qualification.yml` workflow defines the finite host
checks and retains their logs and artifacts. It does not promote installed evidence
or qualify another platform automatically. The development evidence ledger records
each tested revision separately; rerun the applicable checks against the distribution
being published. Defining the workflow is not evidence that a CI run succeeded.

### Build and publish

1. Update the version in `pyproject.toml` and
   `tldw_chatbook/__init__.py`, then update `CHANGELOG.md`.
2. Build and pass the fresh artifact and installed-wheel gates above.
3. Check package metadata:

   ```bash
   twine check fresh-dist/*
   ```

4. Smoke-test the wheel in a disposable environment:

   ```bash
   python -m venv test_env
   source test_env/bin/activate
   pip install fresh-dist/tldw_chatbook-*.whl
   tldw-cli --help
   tldw-serve --help
   deactivate
   ```

5. Inspect the exact artifacts when diagnosing a contract failure:

   ```bash
   tar -tzf fresh-dist/tldw_chatbook-*.tar.gz | less
   unzip -l fresh-dist/tldw_chatbook-*.whl | less
   ```

Do not upload artifacts until every gate above passes against the same fresh
build.
