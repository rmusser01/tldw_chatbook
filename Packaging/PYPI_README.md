# PyPI Distribution Notes for tldw_chatbook

`Packaging/PYPI_RELEASE.md` is the release runbook. This file is the short
maintainer checklist.

## Required Gates

```bash
python -m pip install "setuptools>=77.0" build twine wheel
Packaging/build_dist.sh
python -m pytest Tests/Packaging/test_release_metadata.py -q
python -m pytest Tests/Packaging/test_installed_distribution.py -m integration -q -p no:cacheprovider
```

`Packaging/build_dist.sh` builds a fresh sdist and wheel, runs `twine check`,
and verifies both artifacts with `Packaging/check_manifest.py`.

## Entry Points

The PyPI package installs these console scripts:

- `tldw-cli`
- `tldw-serve`

Use `tldw-cli --help` and `tldw-serve --help` for installation smoke tests.

## Publishing

Normal publishing is done by `.github/workflows/publish-pypi.yml`:

- Manual workflow dispatch from protected `dev` publishes to TestPyPI through
  the `testpypi` environment.
- Protected `main` pushes publish to PyPI through the `pypi` environment only
  when the built version does not already exist on PyPI. Tag pushes do not
  publish.

Do not use `.pypirc` or long-lived API tokens for routine releases.
