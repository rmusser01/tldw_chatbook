# PyPI Distribution Notes for tldw_chatbook

This document contains notes for maintainers about the PyPI distribution process.

## Pre-release Checklist

1. **Update Version Number**
   - Update version in `pyproject.toml`
   - Update version in `tldw_chatbook/__init__.py`
   - Update CHANGELOG.md with release notes

2. **Test Installation**
   ```bash
   # Create a fresh virtual environment
   python -m venv test_env
   source test_env/bin/activate  # Windows: test_env\Scripts\activate
   
   # Install in development mode
   pip install -e .
   
   # Run import tests
   python test_import.py
   
   # Test the CLI
   tldw-cli --help
   ```

3. **Run Full Test Suite**
   ```bash
   pytest
   ```

4. **Build Distribution**
   ```bash
   ./build_dist.sh
   ```

5. **Test Distribution Locally**
   ```bash
   # Create another fresh environment
   python -m venv dist_test
   source dist_test/bin/activate
   
   # Install from wheel
   pip install dist/tldw_chatbook-*.whl
   
   # Run import tests
   python test_import.py
   
   # Test CLI
   tldw-cli --help
   ```

## Upload to TestPyPI First

1. **Configure .pypirc** (if not already done)
   - Copy `.pypirc.template` to `~/.pypirc`
   - Add your PyPI tokens

2. **Upload to TestPyPI**
   ```bash
   python -m twine upload --repository testpypi dist/*
   ```

3. **Test from TestPyPI**
   ```bash
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ tldw_chatbook
   ```

## Upload to PyPI

Once testing is complete:

```bash
python -m twine upload dist/*
```

## Post-release

1. Create a git tag:
   ```bash
   git tag -a v0.1.0 -m "Release version 0.1.0"
   git push origin v0.1.0
   ```

2. Create a GitHub release with the changelog

3. Update version in `pyproject.toml` and `__init__.py` to next development version (e.g., 0.2.0.dev0)

## Package Structure Verification

The distribution should include:
- All Python modules under `tldw_chatbook/`
- CSS files (*.tcss) in `tldw_chatbook/css/`
- Theme files in `tldw_chatbook/css/Themes/`
- JSON templates in `tldw_chatbook/Config_Files/`
- LICENSE file
- README.md

## Common Issues

1. **Missing Data Files**: Check MANIFEST.in and pyproject.toml package-data
2. **Import Errors**: Ensure all __init__.py files are present
3. **Entry Point Not Working**: Verify the console_scripts configuration
4. **Dependencies Missing**: Compare pyproject.toml with requirements.txt