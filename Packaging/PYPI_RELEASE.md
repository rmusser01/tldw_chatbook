# PyPI Release Guide for tldw_chatbook

This guide outlines the process for building and releasing tldw_chatbook to PyPI.

## Prerequisites

1. Ensure you have the necessary tools installed:
   ```bash
   pip install build twine
   ```

2. Create PyPI account at https://pypi.org/account/register/

3. Create API token at https://pypi.org/manage/account/token/
   - Save the token securely
   - You'll use it as your password when uploading

4. (Optional) Create TestPyPI account at https://test.pypi.org/account/register/
   - Recommended for testing releases before publishing to production PyPI

## Pre-Release Checklist

- [ ] Update version in `pyproject.toml`
- [ ] Update version in `tldw_chatbook/__init__.py`
- [ ] Update `CHANGELOG.md` with release notes
- [ ] Run all tests: `pytest`
- [ ] Review and update documentation if needed
- [ ] Commit all changes
- [ ] Create git tag: `git tag v0.1.0`

## Building the Distribution

1. Clean previous builds:
   ```bash
   rm -rf dist/ build/ *.egg-info
   ```

2. Build source distribution and wheel:
   ```bash
   python -m build
   ```

   This creates:
   - `dist/tldw_chatbook-0.1.0.tar.gz` (source distribution)
   - `dist/tldw_chatbook-0.1.0-py3-none-any.whl` (wheel)

3. Verify the distributions:
   ```bash
   twine check dist/*
   ```

## Testing the Package

1. Create a test virtual environment:
   ```bash
   python -m venv test_env
   source test_env/bin/activate  # On Windows: test_env\Scripts\activate
   ```

2. Install from built wheel:
   ```bash
   pip install dist/tldw_chatbook-0.1.0-py3-none-any.whl
   ```

3. Test the installation:
   ```bash
   tldw-cli --version
   ```

4. Test with optional dependencies:
   ```bash
   pip install dist/tldw_chatbook-0.1.0-py3-none-any.whl[embeddings_rag,websearch]
   ```

5. Deactivate and clean up:
   ```bash
   deactivate
   rm -rf test_env
   ```

## Uploading to TestPyPI (Recommended First)

1. Upload to TestPyPI:
   ```bash
   twine upload --repository testpypi dist/*
   ```

2. Test installation from TestPyPI:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ tldw_chatbook
   ```

   Note: The `--extra-index-url` is needed because TestPyPI doesn't have all dependencies.

## Uploading to PyPI

1. Upload to PyPI:
   ```bash
   twine upload dist/*
   ```

   When prompted:
   - Username: `__token__`
   - Password: Your PyPI API token (including the `pypi-` prefix)

2. Verify the package at: https://pypi.org/project/tldw_chatbook/

3. Test installation:
   ```bash
   pip install tldw_chatbook
   ```

## Post-Release

1. Push the git tag:
   ```bash
   git push origin v0.1.0
   ```

2. Create GitHub release:
   - Go to https://github.com/rmusser01/tldw_chatbook/releases
   - Click "Create a new release"
   - Select the tag
   - Copy release notes from CHANGELOG.md
   - Attach the built distributions from `dist/`

3. Update version for next development:
   - Increment version in `pyproject.toml` (e.g., to 0.2.0.dev0)
   - Update `tldw_chatbook/__init__.py`
   - Add new "Unreleased" section to CHANGELOG.md
   - Commit with message: "Bump version for development"

## Troubleshooting

### Common Issues

1. **Authentication failed**: Ensure you're using `__token__` as username and your full API token as password

2. **Package name already taken**: The name might be too similar to existing packages. Consider a unique name.

3. **Missing files in distribution**: Check MANIFEST.in and ensure all necessary files are included

4. **Import errors after installation**: Verify all dependencies are listed in pyproject.toml

### Useful Commands

- View package contents: `tar -tzf dist/tldw_chatbook-0.1.0.tar.gz`
- View wheel contents: `unzip -l dist/tldw_chatbook-0.1.0-py3-none-any.whl`
- Check package metadata: `twine check dist/*`

## Security Notes

- Never commit PyPI tokens to version control
- Use environment variables or keyring for token storage
- Consider using GitHub Actions for automated releases
- Review all included files before uploading