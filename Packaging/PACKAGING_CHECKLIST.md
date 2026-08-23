# PyPI Packaging Checklist for tldw_chatbook

Use this checklist to verify a distribution before release. Artifact checks
must start from a fresh output directory; do not treat an existing checkout
`dist/` as release evidence.

## Project declarations

- [x] `pyproject.toml` is the source of truth for dependencies and entry points.
- [x] `tldw-cli` and `tldw-serve` are declared in `[project.scripts]`.
- [x] `[tool.setuptools.package-data]` explicitly owns wheel runtime data.
- [x] `include-package-data = false` keeps sdist-only files out of wheels.
- [x] Asset directories that grow are matched by pattern, never enumerated.
  Only three lists name files on purpose, each for a reason recorded beside
  it in `pyproject.toml`: `Config_Files/rag_pipelines.toml` (a `*.toml` glob
  would ship the forbidden example TOML), `TTS/audio_cpp_artifact_manifest.json`
  (one pinned manifest), and the vendored `LICENSE` notices. Adding a new
  runtime asset to one of those three means adding its name too.
- [x] The project license uses the `AGPL-3.0-or-later` SPDX expression and
  declares `LICENSE`.

## Distribution content

- [x] Root `MANIFEST.in` is the canonical setuptools sdist manifest.
- [x] The sdist contains release metadata, runtime data, TCSS modules, the
  source-only `stats_screen.css` input, and project/vendored licenses.
- [x] The wheel contains the compiled CSS bundle, RAG pipeline configuration,
  eval configuration, configuration resources, **every** `.sql` file under
  `tldw_chatbook/DB/migrations/`, and both vendored license notices.
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

Run the isolated installed-wheel regression:

```bash
python -m pytest \
  Tests/Packaging/test_installed_distribution.py \
  -m integration -q -p no:cacheprovider
```

The regression builds from a temporary source copy, installs the wheel with
`--no-deps`, loads packaged resources outside the checkout, runs both installed
help commands under private temporary state, and verifies that every installed
target file hash remains unchanged.

## Pre-release steps

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
