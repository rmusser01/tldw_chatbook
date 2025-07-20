# Test Runner Documentation

## Overview

The project includes a comprehensive test runner (`run_all_tests_with_report.py`) that provides detailed reporting and analysis of all tests in the project.

## Running Tests

### Run All Tests
```bash
python run_all_tests_with_report.py
```

### Run Specific Test Modules
```bash
# Run only Chat and Database tests
python run_all_tests_with_report.py --modules Chat Database

# Run all UI-related tests
python run_all_tests_with_report.py --modules UI Widgets
```

### Run RAG Tests

The RAG tests are now organized into three categories:

1. **RAG_Simplified** - New simplified RAG implementation tests (Tests/RAG/simplified/)
2. **RAG_Legacy** - Legacy RAG_Search tests (Tests/RAG_Search/)
3. **RAG_Other** - Other RAG tests including enhanced_rag.py

#### Run All RAG Tests with Detailed Report
```bash
# Using the convenience script
python run_rag_tests.py

# Or using the main runner
python run_all_tests_with_report.py --modules RAG_Simplified RAG_Legacy RAG_Other --rag-detailed
```

#### Run Only New RAG Tests
```bash
python run_all_tests_with_report.py --modules RAG_Simplified
```

#### Run Only Legacy RAG Tests
```bash
python run_all_tests_with_report.py --modules RAG_Legacy
```

### Advanced Options

#### Parallel Execution
```bash
# Run tests with 4 parallel workers
python run_all_tests_with_report.py -n 4
```

#### Run Tests by Marker
```bash
# Run only unit tests
python run_all_tests_with_report.py -m unit

# Run only integration tests
python run_all_tests_with_report.py -m integration
```

#### Output Formats
```bash
# Generate console report only (default)
python run_all_tests_with_report.py --format console

# Generate all report formats
python run_all_tests_with_report.py --format all

# Generate JSON and Markdown reports
python run_all_tests_with_report.py --format json markdown
```

## Test Categories

The test runner organizes tests into the following categories:

- **Core**: Basic smoke tests and unit tests
- **Chat**: Chat functionality tests
- **Character_Chat**: Character chat system tests
- **Database**: All database-related tests
- **UI**: User interface tests
- **RAG_Simplified**: New simplified RAG implementation (V2)
- **RAG_Legacy**: Legacy RAG_Search tests
- **RAG_Other**: Other RAG tests (enhanced, integration)
- **Notes**: Notes system tests
- **Event_Handlers**: Event handling tests
- **Evals**: Evaluation system tests
- **LLM_Management**: LLM management tests
- **Local_Ingestion**: Local media ingestion tests
- **Transcription**: Transcription service tests
- **Web_Scraping**: Web scraping functionality tests
- **Utils**: Utility function tests
- **Chunking**: Text chunking tests
- **TTS**: Text-to-speech tests
- **API**: API tests
- **Integration**: Integration tests

## Report Features

### Console Report
- Overall test summary with pass/fail statistics
- Module-by-module breakdown
- Failed test details with error messages
- Execution time per module

### Detailed RAG Report (--rag-detailed)
When running RAG tests with the `--rag-detailed` flag, you get:
- RAG category breakdown (Simplified, Legacy, Other)
- Per-file test statistics for simplified RAG
- Failure analysis grouped by error type:
  - ChromaDB Configuration issues
  - Import errors
  - API changes
  - Other errors

### JSON Report
Detailed JSON output containing:
- Complete test results
- Individual test timings
- Failure messages and stack traces
- Module statistics

### Markdown Report
Human-readable markdown report with:
- Summary tables
- Module breakdown
- Failed test listings
- Suitable for documentation or issue reports

## Output Location

All reports are saved to: `Tests/test_reports/`

- JSON: `test_report_YYYYMMDD_HHMMSS.json`
- Markdown: `test_report_YYYYMMDD_HHMMSS.md`

## Exit Codes

- `0`: All tests passed
- `1`: One or more tests failed or errored

## Requirements

- Python 3.8+
- pytest
- pytest-xdist (for parallel execution)
- rich (optional, for enhanced console output)

Install requirements:
```bash
pip install pytest pytest-xdist rich
```

## Troubleshooting

### Tests Not Found
Ensure you're running from the project root directory.

### Import Errors
Make sure all project dependencies are installed:
```bash
pip install -e ".[dev]"
```

### RAG Test Dependencies
For RAG tests, ensure embeddings dependencies are installed:
```bash
pip install -e ".[embeddings_rag]"
```

## Recent RAG Test Updates

As of the latest update, the RAG tests have been reorganized and fixed:

1. **Dependency Fix**: Fixed the lazy loading issue that was preventing tests from running
2. **Import Fixes**: Updated all legacy tests to use proper relative imports
3. **API Updates**: Migrated all tests to use the new V2 profile-based API
4. **Async Marking**: All async tests now properly marked with @pytest.mark.asyncio
5. **Configuration**: Tests now use in-memory vector stores to avoid persist_directory issues

Current RAG test status:
- Simplified RAG: ~95% pass rate (212/224 tests passing)
- Legacy RAG: ~84% pass rate (estimated)
- Overall improvement: From ~74% to ~93% pass rate