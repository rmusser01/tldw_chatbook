# PyPI Packaging Checklist for tldw_chatbook

This checklist ensures the project is ready for PyPI distribution.

## ✅ Project Structure
- [x] `pyproject.toml` is the source of truth for dependencies
- [x] Version synchronized between `pyproject.toml` and `__init__.py`
- [x] All package data paths defined in `pyproject.toml`
- [x] Entry point `tldw-cli` defined in `[project.scripts]`

## ✅ Packaging Files
- [x] **MANIFEST.in** includes:
  - Documentation files (LICENSE, README.md, CHANGELOG.md, CLAUDE.md)
  - Config file templates (*.json, *.md in Config_Files/)
  - CSS files (*.tcss in all subdirectories)
  - Third-party licenses
  - Excludes all build artifacts and OS-specific files

- [x] **check_manifest.py** verifies:
  - All expected files are in distribution
  - CSS subdirectories (core/, features/, layout/)
  - Config files and documentation
  - Provides file count summary

- [x] **build_dist.sh**:
  - Cleans Python artifacts before building
  - Runs twine check
  - Runs manifest verification
  - Shows clear next steps

- [x] **clean.sh**:
  - Removes all __pycache__ directories
  - Removes all .pyc/.pyo files
  - Removes .DS_Store files
  - Cleans build/dist directories

## ✅ Development Files
- [x] `.gitignore` includes all necessary exclusions
- [x] `requirements.txt` has header explaining pyproject.toml is authoritative
- [x] GitHub Actions workflows support cross-platform builds

## 📋 Pre-Release Steps

1. **Clean the project**:
   ```bash
   ./Packaging/clean.sh
   ```

2. **Update version** (if releasing new version):
   - Edit version in `pyproject.toml`
   - Edit version in `tldw_chatbook/__init__.py`
   - Update `CHANGELOG.md`

3. **Build the distribution**:
   ```bash
   ./Packaging/build_dist.sh
   ```

4. **Test locally**:
   ```bash
   python -m venv test_env
   source test_env/bin/activate
   pip install dist/tldw_chatbook-*.whl
   tldw-chatbook --help
   deactivate
   rm -rf test_env
   ```

5. **Upload to TestPyPI**:
   ```bash
   python -m twine upload --repository testpypi dist/*
   ```

6. **Test from TestPyPI**:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ tldw_chatbook
   ```

7. **Upload to PyPI** (when ready):
   ```bash
   python -m twine upload dist/*
   ```

## 🔍 Verification Commands

Check what files will be included:
```bash
python Packaging/check_manifest.py
```

Check package metadata:
```bash
twine check dist/*
```

View distribution contents:
```bash
# Source distribution
tar -tzf dist/tldw_chatbook-*.tar.gz | less

# Wheel
unzip -l dist/tldw_chatbook-*.whl | less
```

## ⚠️ Common Issues

1. **Missing files**: Check MANIFEST.in and pyproject.toml package-data
2. **Import errors**: Ensure all __init__.py files exist
3. **Entry point not working**: Verify console_scripts in pyproject.toml
4. **Build artifacts included**: Run clean.sh before building

## 🚀 Ready for PyPI!

The project has been prepared with:
- Proper dependency management via pyproject.toml
- Comprehensive artifact exclusion
- Build verification scripts
- Cross-platform support
- Clear documentation

Run `./Packaging/build_dist.sh` to create your distribution!