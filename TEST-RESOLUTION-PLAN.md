# Test Suite Resolution Plan
Date: 2025-06-19

## Executive Summary

This plan addresses the remaining test failures in the tldw_chatbook project, prioritizing high-impact fixes that will improve the test suite from ~51.7% to a target of 85%+ pass rate.

## Priority 1: Critical Failures (High Impact)

### 1. RAG Module API Mismatches (81 failing tests)
**Current State**: 59 passing, 81 failing, 0 skipped
**Target**: 120+ passing (85%+ pass rate)

**Issues**:
- Service interface mismatches between implementation and tests
- Mock configuration for embeddings needs updating
- ChromaDB integration failures

**Action Items**:
1. Audit RAG service interfaces and align with implementation:
   - Review `App_Functions/RAG/` modules for actual method signatures
   - Update test expectations in `Tests/RAG/`
   - Fix parameter naming inconsistencies
2. Update embeddings mock configuration:
   - Review SentenceTransformer usage patterns
   - Update mock paths and return values
3. Fix ChromaDB test configuration:
   - Ensure proper initialization for test databases
   - Mock external dependencies appropriately

**Files to modify**:
- `Tests/RAG/test_embeddings_service.py`
- `Tests/RAG/test_rag_service.py`
- `Tests/RAG/test_chromadb_utils.py`

### 2. Evals Module Failures (~90 failing tests)
**Current State**: 47 passing, 91 failing, 7 errors
**Target**: 110+ passing (75%+ pass rate)

**Issues**:
- Task loader implementation issues
- Integration test failures across all evaluation types
- Database operation mismatches

**Action Items**:
1. Fix task loader implementation:
   - Review `App_Functions/Evals/task_loaders.py`
   - Ensure proper JSON/YAML parsing
   - Fix dataset loading mechanisms
2. Align database operations:
   - Verify CRUD operations match test expectations
   - Fix transaction handling in tests
3. Update integration test fixtures:
   - Create proper test datasets
   - Mock external dependencies

**Files to modify**:
- `App_Functions/Evals/task_loaders.py`
- `Tests/Evals/test_task_loaders.py`
- `Tests/Evals/test_evals_integration.py`

### 3. Chat Module Failures (8 failed, 39 skipped)
**Current State**: 41 passing, 8 failing, 39 skipped
**Target**: 80+ passing (90%+ pass rate)

**Issues**:
- KoboldCPP integration failures
- Template rendering security test failure
- Media sidebar search async issues

**Action Items**:
1. Fix KoboldCPP integration:
   - Review API endpoint configuration
   - Update mock responses for tests
   - Fix authentication/connection handling
2. Fix template rendering security:
   - Review Jinja2 sandbox configuration
   - Update security test expectations
3. Enable skipped tests:
   - Identify why 39 tests are skipped
   - Install missing dependencies or fix configurations

**Files to modify**:
- `App_Functions/LLM_Calls/Local_APIs/koboldcpp_api_calls.py`
- `Tests/Chat/test_koboldcpp_integration.py`
- `Tests/Chat/test_template_security.py`

## Priority 2: Infrastructure Improvements (Medium Impact)

### 4. UI/Widget Test Infrastructure
**Current State**: UI: 35/67 passing (52.2%), Widgets: 23/43 passing (53.5%)
**Target**: 70%+ pass rate for both

**Issues**:
- Textual app context requirements
- Async fixture configuration
- Command palette test failures

**Action Items**:
1. Create proper Textual test harness:
   ```python
   # Create Tests/fixtures/textual_test_app.py
   class TestApp(App):
       """Minimal test app for widget testing"""
       def compose(self):
           yield Container()
   ```
2. Standardize widget test patterns:
   - Use `async_test` decorator consistently
   - Mock app context properly
   - Handle compose() method requirements
3. Fix command palette tests:
   - Review command registration
   - Fix action handling in tests

**New files to create**:
- `Tests/fixtures/textual_test_app.py`
- `Tests/fixtures/widget_test_utils.py`

### 5. Event Handlers Async Issues
**Current State**: 28 passing, 21 failing
**Target**: 40+ passing (80%+ pass rate)

**Issues**:
- Async coroutine handling errors
- Missing mock UI components
- Media search functionality issues

**Action Items**:
1. Fix async/await patterns:
   - Audit all event handler tests
   - Ensure proper awaiting of coroutines
   - Use AsyncMock where appropriate
2. Complete mock UI components:
   - Add missing display area mocks
   - Fix media button state handling
3. Fix media search functionality:
   - Update search return value tuples
   - Fix pagination handling

**Files to modify**:
- `Tests/Event_Handlers/test_chat_events_sidebar.py`
- `Tests/Event_Handlers/test_media_search_events.py`

### 6. Integration Test Failures
**Current State**: 20 passing, 14 failing, 3 errors
**Target**: 30+ passing (80%+ pass rate)

**Issues**:
- File operation validation failures
- Import structure issues
- Core functionality test failures

**Action Items**:
1. Fix file operation tests:
   - Review path validation logic
   - Handle edge cases properly (e.g., ".." in filenames)
2. Update import tests:
   - Ensure all modules are importable
   - Fix circular import issues
3. Update integration mocks:
   - Match actual application architecture
   - Fix dependency injection patterns

## Priority 3: Minor Improvements (Low Impact)

### 7. Method Name Alignment
**Affected modules**: Prompts_DB, various test files

**Action Items**:
1. Create mapping of test vs implementation method names
2. Systematically update test files
3. Consider adding method aliases for backward compatibility

### 8. Database Module Standardization
**Issues**: Inconsistent Path vs string handling

**Action Items**:
1. Create DB base class with standardized path handling
2. Update all DB modules to inherit from base
3. Add comprehensive path validation tests

## Implementation Timeline

### Week 1 (Immediate)
- Day 1-2: RAG API alignment (Priority 1.1)
- Day 3-4: Evals task loader fixes (Priority 1.2)
- Day 5: Chat module KoboldCPP fixes (Priority 1.3)

### Week 2 (Short-term)
- Day 1-2: Textual test infrastructure (Priority 2.4)
- Day 3-4: Event handler async fixes (Priority 2.5)
- Day 5: Integration test updates (Priority 2.6)

### Week 3 (Cleanup)
- Day 1-2: Method name alignment (Priority 3.7)
- Day 3-4: Database standardization (Priority 3.8)
- Day 5: Final testing and documentation

## Success Metrics

1. **Overall Pass Rate**: Achieve 85%+ (from current 51.7%)
2. **Zero Skipped Tests**: All tests should run (currently 0 after fixes)
3. **Minimal Warnings**: Keep under 20 warnings total
4. **No Import Errors**: All modules should be importable
5. **CI/CD Ready**: All tests should pass in CI environment

## Technical Debt Prevention

1. **Test Writing Guidelines**:
   - Create `Tests/TESTING_GUIDE.md` with patterns and examples
   - Document Textual widget testing approach
   - Include async testing best practices

2. **Continuous Integration**:
   - Add pre-commit hooks for test execution
   - Set up GitHub Actions for test automation
   - Require tests for all new features

3. **Dependency Management**:
   - Make "optional" dependencies truly optional with feature flags
   - Document all test dependencies in `Tests/requirements-test.txt`
   - Add dependency checking to CI pipeline

## Risk Mitigation

1. **Breaking Changes**: 
   - Run full test suite after each fix
   - Keep changes isolated to test files where possible
   - Document any API changes required

2. **Time Constraints**:
   - Focus on high-impact fixes first
   - Consider temporary test skipping for complex issues
   - Parallelize work where possible

3. **Regression Prevention**:
   - Add integration tests for fixed issues
   - Document fix patterns for future reference
   - Review fixes in code review process

## Conclusion

This resolution plan provides a systematic approach to improving the test suite health from ~51.7% to 85%+ pass rate. By focusing on high-impact areas first and establishing proper testing infrastructure, we can achieve a robust and maintainable test suite that supports continued development of the tldw_chatbook project.