# GitHub Actions Cross-Platform Compatibility Fixes

## Issues Found and Fixed

### 1. **Shell Specification**
- **Issue**: Commands with line continuations (`\`) don't work on Windows CMD
- **Fix**: Added `shell: bash` to all run steps or set as default
- **Files Updated**: 
  - `test.yml`: Added default shell configuration
  - `python-app.yml`: Added shell specification to run steps

### 2. **Default Shell Configuration**
Added to `test.yml`:
```yaml
defaults:
  run:
    shell: bash
```
This ensures all commands use bash even on Windows (via Git Bash).

### 3. **Windows-Specific Dependencies**
Added conditional installation step for Windows:
```yaml
- name: Install Windows-specific dependencies
  if: runner.os == 'Windows'
  run: |
    pip install windows-curses || echo "windows-curses not needed"
```

### 4. **Python Script Compatibility**
- Added UTF-8 encoding declaration to `generate_test_summary.py`
- Imported `os` module for better path handling

## Best Practices for Cross-Platform GitHub Actions

### 1. **Always Specify Shell**
Use `shell: bash` for consistency across platforms, or set it as default.

### 2. **Use Forward Slashes**
Always use `/` for paths, even on Windows. GitHub Actions handles conversion.

### 3. **Conditional Steps**
Use `if: runner.os == 'Windows'` for platform-specific steps.

### 4. **Environment Variables**
Set these for consistent behavior:
```yaml
env:
  PYTHONDONTWRITEBYTECODE: 1
  PYTHONUNBUFFERED: 1
  FORCE_COLOR: 1
```

### 5. **Line Continuations**
When using bash shell, backslash continuations work on all platforms:
```yaml
shell: bash
run: |
  pytest -m unit \
    --json-report \
    --cov=tldw_chatbook
```

### 6. **File Operations**
Use Python's `pathlib` or `os.path` for file operations in scripts.

## Remaining Considerations

### 1. **Textual on Windows**
Textual may have different behavior on Windows terminals. Consider:
- Testing with Windows Terminal vs CMD vs PowerShell
- May need to set `TERM` environment variable

### 2. **Path Length Limits**
Windows has a 260-character path limit by default. Consider:
- Keeping artifact names short
- Using shorter test names

### 3. **Case Sensitivity**
Windows filesystem is case-insensitive. Ensure:
- Import statements match actual file casing
- No files differ only by case

## Testing the Fixes

To verify cross-platform compatibility:
1. Push changes to a branch
2. Create a PR to trigger workflows
3. Check that all OS matrix jobs pass
4. Review logs for any platform-specific warnings