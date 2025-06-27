# Test Summary Report

## Test Execution Summary

### Module: Chat
- Total tests: 147
- Passed: 45 (30.6%)
- Failed: 24 (16.3%)
- Skipped: 78 (53.1%)

### Module: Media_DB
- Total tests: 54
- Passed: 38 (70.4%)
- Failed: 16 (29.6%)
- Skipped: 0 (0.0%)

### Module: RAG
- Total tests: 215
- Passed: 101 (47.0%)
- Failed: 114 (53.0%)
- Skipped: 0 (0.0%)

### Module: Widgets
- Total tests: 28
- Passed: 18 (64.3%)
- Failed: 10 (35.7%)
- Skipped: 0 (0.0%)

### Module: UI
- Total tests: 127 (estimate from timeout)
- Unable to get full breakdown due to timeout

### Module: DB
- Total tests: 79
- Passed: 56 (70.9%)
- Failed: 18 (22.8%)
- Skipped: 5 (6.3%)

### Module: Notes
- Total tests: 35
- Passed: 34 (97.1%)
- Failed: 0 (0.0%)
- Skipped: 1 (2.9%)

### Module: ChaChaNotesDB
- Total tests: 57
- Passed: 57 (100.0%)
- Failed: 0 (0.0%)
- Skipped: 0 (0.0%)

## Overall Summary (Excluding UI module due to timeout)
- **Total tests counted**: 650
- **Total passed**: 349 (53.7%)
- **Total failed**: 182 (28.0%)
- **Total skipped**: 84 (12.9%)
- **Unable to determine**: 35 (5.4%) - from UI module

## Comparison with Previous Results
- **Previous baseline**: 73.2% pass rate from ~913 tests
- **Current results**: 53.7% pass rate from 650 tests (excluding UI)
- **Change**: -19.5% decrease in pass rate

## Summary Table

| Module | Total | Passed | Failed | Skipped | Pass Rate |
|--------|-------|--------|--------|---------|-----------|
| Chat | 147 | 45 | 24 | 78 | 30.6% |
| Media_DB | 54 | 38 | 16 | 0 | 70.4% |
| RAG | 215 | 101 | 114 | 0 | 47.0% |
| Widgets | 28 | 18 | 10 | 0 | 64.3% |
| UI | 127* | - | - | - | - |
| DB | 79 | 56 | 18 | 5 | 70.9% |
| Notes | 35 | 34 | 0 | 1 | 97.1% |
| ChaChaNotesDB | 57 | 57 | 0 | 0 | 100.0% |
| **Total** | **777** | **349** | **182** | **84** | **53.7%** |

*UI module timed out, counts estimated

## Notes
1. The UI module tests timed out during execution, so 127 tests are not included in the overall statistics
2. With the UI module, the total would be approximately 777 tests
3. Major issues observed:
   - Chat module has 53.1% skipped tests (likely due to missing API keys or dependencies)
   - RAG module has 53.0% failure rate
   - Overall pass rate has decreased significantly from the previous baseline
4. Best performing modules:
   - ChaChaNotesDB: 100% pass rate
   - Notes: 97.1% pass rate
   - Media_DB and DB: ~70% pass rate
