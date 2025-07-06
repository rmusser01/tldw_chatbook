# Evaluation System Error Handling Improvements

## Overview

This document outlines the comprehensive error handling improvements for the evaluation system in tldw_chatbook. The improvements focus on better error categorization, user-friendly notifications, recovery mechanisms, and detailed error tracking.

## New Components Created

### 1. `eval_errors.py` - Centralized Error Management
- **Purpose**: Provides a unified error handling framework for the entire evaluation system
- **Key Features**:
  - Error categorization with `ErrorCategory` enum
  - Severity levels with `ErrorSeverity` enum
  - `ErrorContext` dataclass for rich error information
  - Specialized exception classes for different error types
  - Global `ErrorHandler` for tracking and summarizing errors

### 2. `eval_runner_enhanced.py` - Enhanced Runner Error Handling
- **Purpose**: Improves error handling in the evaluation runner
- **Key Features**:
  - `EnhancedDatasetLoader` with detailed dataset validation
  - Better error messages for missing fields, format issues, and file problems
  - `EnhancedErrorHandler` with provider health tracking
  - Batch error handling for parallel operations
  - Partial failure recovery

### 3. `eval_orchestrator_enhanced.py` - Orchestrator Improvements
- **Purpose**: Adds robust error handling to the orchestrator
- **Key Features**:
  - `ConcurrentRunManager` to prevent conflicting runs
  - `ConfigurationValidator` for pre-flight validation
  - Database operation error handling with retry logic
  - Safe export operations with atomic file writes
  - Comprehensive error context preservation

### 4. `eval_error_dialog.py` - User Interface Enhancements
- **Purpose**: Provides user-friendly error display and interaction
- **Key Features**:
  - `ErrorDetailsDialog` modal with expandable technical details
  - Retry options for recoverable errors
  - Copy error details functionality
  - `ErrorSummaryWidget` for ongoing error tracking
  - Enhanced notification helpers

## Key Improvements by Area

### 1. Error Categorization and Recovery (eval_runner.py)

**Improvements Needed:**
- ✅ Dataset loading errors now properly categorized
- ✅ Error context preserved through retry chains
- ✅ Partial batch failure handling implemented
- ✅ Actionable suggestions included in error messages

**Implementation Highlights:**
```python
# Better dataset error handling
if not path.exists():
    raise DatasetLoadingError.file_not_found(str(path))

# Enhanced retry with provider health tracking
health_status = error_handler.get_provider_health(provider)
if health_status['consecutive_errors'] > 5:
    # Add delay to avoid hammering failing provider
    rate_limit_delays[provider] = 10.0
```

### 2. Edge Case Handling (eval_orchestrator.py)

**Improvements Needed:**
- ✅ Concurrent run detection and prevention
- ✅ Configuration validation before execution
- ✅ Database retry logic for locked databases
- ✅ Atomic file operations for exports

**Implementation Highlights:**
```python
# Prevent concurrent runs
await concurrent_manager.register_run(run_id, task_id, model_id)

# Validate configurations before starting
ConfigurationValidator.validate_task_config(task_config)
ConfigurationValidator.validate_model_config(model_config)

# Atomic file export
temp_path.replace(output_path)  # Atomic rename
```

### 3. Runner-Specific Error Handling (specialized_runners.py)

**Improvements Needed:**
- ✅ Security error detection and reporting
- ✅ Timeout errors with better context
- ✅ Language detection fallbacks
- ✅ Safety check error logging

**Implementation Highlights:**
```python
# Security error for code execution
if contains_dangerous_code(code):
    raise SecurityError.unsafe_code_execution(code_snippet)

# Timeout with context
raise ExecutionError.timeout(task_name, timeout_seconds)
```

### 4. User-Friendly Notifications (UI Event Handlers)

**Improvements Needed:**
- ✅ Error severity levels in notifications
- ✅ Error details dialog for technical users
- ✅ Retry progress shown in UI
- ✅ Error history tracking

**Implementation Highlights:**
```python
# Show appropriate error dialog
show_error_dialog(app, error, allow_retry=error.context.is_retryable)

# Enhanced notifications with severity
notify_error(app, error, title="Evaluation Failed")
```

## Error Categories

The new system defines clear error categories:

1. **DATASET_LOADING** - Issues loading or parsing datasets
2. **MODEL_CONFIGURATION** - Model setup problems
3. **API_COMMUNICATION** - Network and API issues
4. **AUTHENTICATION** - Credential problems
5. **RATE_LIMITING** - API rate limit issues
6. **TIMEOUT** - Operation timeouts
7. **RESOURCE_EXHAUSTION** - Memory/disk issues
8. **VALIDATION** - Configuration validation failures
9. **FILE_SYSTEM** - File access problems
10. **DATABASE** - Database operation failures
11. **NETWORK** - Connection issues
12. **SECURITY** - Security policy violations

## Usage Examples

### 1. Handling Dataset Errors
```python
try:
    samples = EnhancedDatasetLoader.load_dataset_samples(task_config)
except DatasetLoadingError as e:
    # User sees: "Dataset file not found: /path/to/data.json
    #             Suggestion: Check the file path and ensure the file exists"
    show_error_dialog(app, e, allow_retry=False)
```

### 2. Handling API Errors with Retry
```python
async def run_with_retry():
    try:
        result = await error_handler.with_retry(
            operation=lambda: llm_interface.generate(prompt),
            sample_id="test_001",
            provider="openai"
        )
    except APIError as e:
        if e.context.category == ErrorCategory.RATE_LIMITING:
            # Wait and retry automatically handled
            pass
```

### 3. Validation Before Execution
```python
try:
    ConfigurationValidator.validate_run_config(task_config, model_config, run_config)
except ValidationError as e:
    # Prevents invalid configurations from starting
    notify_error(app, e)
    return
```

## Integration Steps

To integrate these improvements into the existing codebase:

1. **Copy the new files** to their respective directories
2. **Import error classes** in existing modules:
   ```python
   from .eval_errors import (
       DatasetLoadingError, APIError, ValidationError,
       get_error_handler, show_error_dialog
   )
   ```

3. **Replace error handling** in key locations:
   - Dataset loading in `DatasetLoader`
   - Retry logic in `ErrorHandler`
   - Validation in `run_evaluation`
   - UI notifications in event handlers

4. **Add error tracking** to the UI:
   ```python
   # In EvalsWindow
   self.error_summary = ErrorSummaryWidget()
   yield self.error_summary
   ```

5. **Test error scenarios**:
   - Missing dataset files
   - Invalid API keys
   - Network timeouts
   - Concurrent run attempts

## Benefits

1. **Better User Experience**
   - Clear, actionable error messages
   - Suggestions for fixing problems
   - Retry options for transient failures

2. **Improved Debugging**
   - Detailed error context preservation
   - Technical details available on demand
   - Error history tracking

3. **System Reliability**
   - Automatic retry for recoverable errors
   - Provider health monitoring
   - Graceful degradation

4. **Developer Experience**
   - Consistent error handling patterns
   - Easy to add new error types
   - Comprehensive error categorization

## Future Enhancements

1. **Error Analytics**
   - Track error patterns over time
   - Identify problematic providers
   - Suggest configuration improvements

2. **Advanced Recovery**
   - Checkpoint/resume for long evaluations
   - Automatic fallback providers
   - Smart retry scheduling

3. **Error Reporting**
   - Export error summaries
   - Integration with monitoring tools
   - Automated error reports

## Testing Recommendations

1. **Unit Tests** for error classes and handlers
2. **Integration Tests** for error flows
3. **UI Tests** for error dialogs and notifications
4. **Stress Tests** for error recovery mechanisms

The improved error handling system makes the evaluation system more robust, user-friendly, and maintainable.