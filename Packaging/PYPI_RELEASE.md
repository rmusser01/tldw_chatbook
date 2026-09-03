# PyPI Release Guide for tldw_chatbook

Use GitHub Actions trusted publishing for normal releases. Do not upload with a
long-lived PyPI token unless the trusted-publishing path is unavailable.

## Package Identity

- Source distribution name: `tldw_chatbook`
- Normalized PyPI name: `tldw-chatbook`
- Install command: `pip install tldw_chatbook`
- Console scripts: `tldw-cli`, `tldw-serve`
- Version source of truth: `pyproject.toml`
- Package runtime version: `tldw_chatbook/__init__.py`
- Native packaging version helper: `Packaging/common/version.py`, derived from
  `pyproject.toml`

## One-Time PyPI Setup

Create trusted publishers on TestPyPI and PyPI before the first workflow upload.

For TestPyPI:

- Owner: `rmusser01`
- Repository: `tldw_chatbook`
- Workflow: `publish-pypi.yml`
- Environment: `testpypi`

For PyPI:

- Owner: `rmusser01`
- Repository: `tldw_chatbook`
- Workflow: `publish-pypi.yml`
- Environment: `pypi`

If the project does not exist yet, create a pending trusted publisher for the
same owner, repository, workflow, and environment.

Protect the `pypi` GitHub environment and the production `v*` tag pattern before
allowing production publishing. The publish job only downloads built artifacts
and requests the PyPI OIDC token.

## Local Release Gates

1. Update `pyproject.toml`, `tldw_chatbook/__init__.py`, and `CHANGELOG.md`.
2. Install release tools in the active environment:

   ```bash
   python -m pip install "setuptools>=77.0" build twine wheel
   ```

3. Build fresh artifacts:

   ```bash
   Packaging/build_dist.sh
   ```

4. Run the focused metadata and installed-distribution gates:

   ```bash
   python -m pytest Tests/Packaging/test_release_metadata.py -q
   python -m pytest Tests/Packaging/test_installed_distribution.py -m integration -q -p no:cacheprovider
   ```

5. Smoke-test the wheel in a disposable environment:

   ```bash
   python -m venv test_env
   source test_env/bin/activate
   pip install dist/tldw_chatbook-*.whl
   tldw-cli --help
   tldw-serve --help
   deactivate
   ```

Run the full suite only when the release manager explicitly wants a full sweep.

## Publish to TestPyPI

1. Open the `Publish Python package` workflow in GitHub Actions.
2. Run it manually from the intended release branch.
3. The workflow builds, checks, uploads artifacts, and publishes to TestPyPI
   through the `testpypi` environment.
4. Test installation from TestPyPI:

   ```bash
   python -m venv testpypi_env
   source testpypi_env/bin/activate
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "tldw_chatbook==<version>"
   tldw-cli --help
   tldw-serve --help
   deactivate
   ```

## Publish to PyPI

1. Confirm the same commit passed the local gates and TestPyPI smoke test.
2. Create and push a protected production tag:

   ```bash
   git tag -a v<version> -m "Release v<version>"
   git push origin v<version>
   ```

3. The `Publish Python package` workflow publishes the checked artifacts to PyPI
   only when the `v<version>` tag is protected and matches the package version.
   Publishing still goes through the protected `pypi` environment.
4. Verify:

   ```bash
   python -m venv pypi_env
   source pypi_env/bin/activate
   pip install "tldw_chatbook==<version>"
   tldw-cli --help
   tldw-serve --help
   deactivate
   ```

## Emergency Manual Upload

Use this only if trusted publishing is unavailable and the same `dist/` artifacts
passed the local gates:

```bash
python -m twine upload --repository testpypi dist/*
python -m twine upload dist/*
```

Manual uploads require an account or project-scoped PyPI token. Never commit
tokens, `.pypirc`, or upload logs containing credentials.
