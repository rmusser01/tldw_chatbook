# Local Test Runner for tldw_chatbook

A comprehensive local test runner that provides CI/CD-like functionality for running tests with detailed reporting, history tracking, and multiple test modes.

## Features

- **Multiple Test Modes**: Quick, Full, Coverage, and Smoke tests
- **Parallel Execution**: Run test groups in parallel for faster results
- **HTML Reports**: Beautiful HTML reports with charts and detailed results
- **History Tracking**: Track test results over time with trend analysis
- **Group Selection**: Run specific test groups or all tests
- **Coverage Reports**: Generate code coverage reports (requires pytest-cov)
- **Colored Output**: Clear, colored terminal output for easy reading

## Installation

Ensure you have the required dependencies:

```bash
pip install pytest pytest-json-report pytest-cov
```

## Usage

### Basic Usage

```bash
# Run quick tests (critical groups only)
python run_tests.py

# Run all tests
python run_tests.py --mode full

# Run smoke tests (minimal set for quick validation)
python run_tests.py --mode smoke

# Run with coverage
python run_tests.py --mode coverage
```

### Advanced Options

```bash
# Run tests in parallel
python run_tests.py --parallel

# Run specific test groups
python run_tests.py --groups database,core

# Verbose output
python run_tests.py --verbose

# Skip HTML report
python run_tests.py --no-html

# Skip history tracking  
python run_tests.py --no-track
```

## Test Groups

The runner organizes tests into logical groups:

- **database**: Database layer tests (ChaChaNotesDB, DB, Prompts_DB, Media_DB)
- **core**: Core features (Character_Chat, Chat, Notes, Evals)
- **integration**: Integration tests (Event_Handlers, LLM_Management, RAG)
- **ui**: UI and Widget tests
- **utility**: Utility and infrastructure tests

## Output

### Console Output
- Colored status indicators (✓ passed, ✗ failed, ⚠ warning)
- Real-time progress updates
- Summary statistics
- Trend analysis (last 5 runs)

### Reports Directory
All reports are saved in `test_reports/` with timestamps:

```
test_reports/
├── 20250106_143022/
│   ├── report.html          # Main HTML report
│   ├── summary.txt          # Text summary
│   ├── database_output.txt  # Group outputs
│   ├── database_report.json # Detailed JSON data
│   └── coverage_database/   # Coverage HTML (if enabled)
└── test_history.json        # Historical tracking
```

### HTML Report
Open `test_reports/[timestamp]/report.html` to view:
- Visual progress bars
- Detailed statistics per test group
- Failed test listings
- Platform and environment info

## Test Modes

### Quick Mode (default)
- Runs only critical test groups
- Fastest option for development
- Good for pre-commit checks

### Full Mode
- Runs all test groups
- Comprehensive testing
- Use before merging to main

### Coverage Mode
- Runs tests with code coverage
- Generates HTML coverage reports
- Identifies untested code paths

### Smoke Mode
- Runs minimal set of critical tests
- Ultra-fast validation
- Good for post-deployment checks

## Continuous Monitoring

The test runner tracks history automatically. After 5+ runs, you'll see trend analysis:

```
Test Trend (Last 5 Runs)
Date                 Mode       Pass Rate  Duration  
--------------------------------------------------
2025-01-06 14:30     quick       85.2%      45.3s
2025-01-06 15:45     full        87.1%     120.5s
2025-01-06 16:20     quick       88.5%      43.1s
2025-01-06 17:00     coverage    89.2%     150.2s
2025-01-06 17:30     quick       91.3%      42.8s
```

## Integration with Development Workflow

### Pre-commit Hook
Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
python run_tests.py --mode quick
```

### Continuous Testing
Run in watch mode (requires pytest-watch):

```bash
pip install pytest-watch
ptw -- --mode quick
```

### VS Code Task
Add to `.vscode/tasks.json`:

```json
{
    "label": "Run Tests",
    "type": "shell",
    "command": "python run_tests.py",
    "problemMatcher": "$python",
    "group": {
        "kind": "test",
        "isDefault": true
    }
}
```

## Troubleshooting

### Missing Dependencies
If you see import errors, install test dependencies:
```bash
pip install -r requirements-test.txt
```

### Permission Errors
Make the script executable:
```bash
chmod +x run_tests.py
```

### Parallel Execution Issues
If parallel mode fails, try sequential:
```bash
python run_tests.py --mode full
```

## Contributing

When adding new tests:
1. Place them in the appropriate test group directory
2. Follow existing naming conventions (`test_*.py`)
3. Update TEST_GROUPS in run_tests.py if adding new categories
4. Add critical tests to SMOKE_TESTS list