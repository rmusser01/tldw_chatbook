# test_eval_errors.py
# Description: Unit tests for the unified error handling system
#
"""
Test Evaluation Error Handling
------------------------------

Tests for the consolidated error handling system with retry logic and budget monitoring.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from tldw_chatbook.Evals.eval_errors import (
    ErrorHandler, EvaluationError, ErrorContext, ErrorCategory, ErrorSeverity,
    BudgetMonitor, with_error_handling, get_error_handler,
    DatasetLoadingError, ModelConfigurationError, APIError, ExecutionError
)


class TestErrorContext:
    """Test ErrorContext dataclass."""
    
    def test_error_context_creation(self):
        """Test creating an error context."""
        context = ErrorContext(
            category=ErrorCategory.API_COMMUNICATION,
            severity=ErrorSeverity.ERROR,
            message="Test error",
            details="Some details",
            suggestion="Try again",
            is_retryable=True,
            retry_after=5.0
        )
        
        assert context.category == ErrorCategory.API_COMMUNICATION
        assert context.severity == ErrorSeverity.ERROR
        assert context.message == "Test error"
        assert context.is_retryable is True
        assert context.retry_after == 5.0
        assert context.timestamp is not None
    
    def test_error_context_to_dict(self):
        """Test converting error context to dictionary."""
        context = ErrorContext(
            category=ErrorCategory.DATASET_LOADING,
            severity=ErrorSeverity.WARNING,
            message="Dataset warning"
        )
        
        result = context.to_dict()
        
        assert result['category'] == 'dataset_loading'
        assert result['severity'] == 'warning'
        assert result['message'] == 'Dataset warning'
        assert 'timestamp' in result


class TestEvaluationError:
    """Test EvaluationError exception class."""
    
    def test_evaluation_error_creation(self):
        """Test creating an evaluation error."""
        context = ErrorContext(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.ERROR,
            message="Validation failed"
        )
        
        error = EvaluationError(context)
        
        assert error.context == context
        assert str(error) == "Validation failed"
    
    def test_evaluation_error_with_original(self):
        """Test evaluation error with original exception."""
        original = ValueError("Original error")
        context = ErrorContext(
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.ERROR,
            message="Wrapped error"
        )
        
        error = EvaluationError(context, original)
        
        assert error.original_error == original
        assert error.context.message == "Wrapped error"
    
    def test_get_user_message(self):
        """Test getting user-friendly error message."""
        context = ErrorContext(
            category=ErrorCategory.API_COMMUNICATION,
            severity=ErrorSeverity.ERROR,
            message="Connection failed",
            suggestion="Check your internet connection"
        )
        
        error = EvaluationError(context)
        user_msg = error.get_user_message()
        
        assert "Connection failed" in user_msg
        assert "Check your internet connection" in user_msg


class TestSpecificErrors:
    """Test specific error types."""
    
    def test_dataset_loading_error_file_not_found(self):
        """Test dataset loading error for file not found."""
        error = DatasetLoadingError.file_not_found("/path/to/missing.json")
        
        assert error.context.category == ErrorCategory.DATASET_LOADING
        assert error.context.severity == ErrorSeverity.ERROR
        assert "/path/to/missing.json" in error.context.message
        assert error.context.is_retryable is False
    
    def test_model_configuration_error_missing_api_key(self):
        """Test model configuration error for missing API key."""
        error = ModelConfigurationError.missing_api_key("openai")
        
        assert error.context.category == ErrorCategory.MODEL_CONFIGURATION
        assert "openai" in error.context.message
        assert error.context.error_code == "MISSING_API_KEY"
        assert error.context.is_retryable is False
    
    def test_api_error_rate_limit(self):
        """Test API error for rate limit."""
        error = APIError.rate_limit_exceeded("anthropic", retry_after=30.0)
        
        assert error.context.category == ErrorCategory.RATE_LIMITING
        assert error.context.severity == ErrorSeverity.WARNING
        assert "anthropic" in error.context.message
        assert error.context.is_retryable is True
        assert error.context.retry_after == 30.0
    
    def test_execution_error_timeout(self):
        """Test execution error for timeout."""
        error = ExecutionError.timeout("long_task", 60.0)
        
        assert error.context.category == ErrorCategory.TIMEOUT
        assert "long_task" in error.context.message
        assert "60" in error.context.message
        assert error.context.is_retryable is True


class TestErrorHandler:
    """Test ErrorHandler class."""
    
    @pytest.fixture
    def handler(self):
        """Create an error handler instance."""
        return ErrorHandler()
    
    def test_error_handler_initialization(self, handler):
        """Test error handler initialization."""
        assert handler.error_history == []
        assert handler.max_history_size == 100
    
    def test_handle_error_with_evaluation_error(self, handler):
        """Test handling an EvaluationError."""
        context = ErrorContext(
            category=ErrorCategory.API_COMMUNICATION,
            severity=ErrorSeverity.ERROR,
            message="Test error",
            is_retryable=True
        )
        error = EvaluationError(context)
        
        result = handler.handle_error(error, {'operation': 'test'})
        
        assert result == context
        assert len(handler.error_history) == 1
        assert handler.error_history[0] == error
    
    def test_handle_error_with_standard_exception(self, handler):
        """Test handling a standard exception."""
        error = FileNotFoundError("test.txt")
        
        result = handler.handle_error(error, {'operation': 'load'})
        
        assert result.category == ErrorCategory.FILE_SYSTEM
        assert result.severity == ErrorSeverity.ERROR
        assert "test.txt" in result.message
        assert result.is_retryable is False
    
    def test_error_history_limit(self, handler):
        """Test that error history respects size limit."""
        # Add more errors than the limit
        for i in range(150):
            error = EvaluationError(ErrorContext(
                category=ErrorCategory.UNKNOWN,
                severity=ErrorSeverity.ERROR,
                message=f"Error {i}"
            ))
            handler.handle_error(error, {})
        
        assert len(handler.error_history) == 100  # Should be capped at max_history_size
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff_success(self, handler):
        """Test retry with backoff when operation succeeds."""
        mock_func = AsyncMock(return_value="success")
        
        result = await handler.retry_with_backoff(
            mock_func,
            max_retries=3,
            base_delay=0.01  # Short delay for testing
        )
        
        assert result == "success"
        mock_func.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff_eventual_success(self, handler):
        """Test retry with backoff when operation eventually succeeds."""
        mock_func = AsyncMock(side_effect=[
            Exception("First failure"),
            Exception("Second failure"),
            "success"
        ])
        
        result = await handler.retry_with_backoff(
            mock_func,
            max_retries=3,
            base_delay=0.01
        )
        
        assert result == "success"
        assert mock_func.call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff_all_failures(self, handler):
        """Test retry with backoff when all attempts fail."""
        mock_func = AsyncMock(side_effect=Exception("Always fails"))
        
        with pytest.raises(Exception) as exc_info:
            await handler.retry_with_backoff(
                mock_func,
                max_retries=2,
                base_delay=0.01
            )
        
        assert "Always fails" in str(exc_info.value)
        assert mock_func.call_count == 3  # Initial + 2 retries
    
    @pytest.mark.asyncio
    async def test_retry_with_non_retryable_error(self, handler):
        """Test retry stops on non-retryable error."""
        context = ErrorContext(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.ERROR,
            message="Invalid input",
            is_retryable=False
        )
        error = EvaluationError(context)
        
        mock_func = AsyncMock(side_effect=error)
        
        with pytest.raises(EvaluationError) as exc_info:
            await handler.retry_with_backoff(
                mock_func,
                max_retries=3,
                base_delay=0.01
            )
        
        assert exc_info.value == error
        mock_func.assert_called_once()  # Should not retry
    
    def test_get_error_summary(self, handler):
        """Test getting error summary."""
        # Add some errors
        for i in range(3):
            handler.handle_error(
                EvaluationError(ErrorContext(
                    category=ErrorCategory.API_COMMUNICATION,
                    severity=ErrorSeverity.ERROR,
                    message=f"API error {i}"
                )),
                {}
            )
        
        handler.handle_error(
            EvaluationError(ErrorContext(
                category=ErrorCategory.DATASET_LOADING,
                severity=ErrorSeverity.ERROR,
                message="Dataset error"
            )),
            {}
        )
        
        summary = handler.get_error_summary()
        
        assert summary['total_errors'] == 4
        assert summary['categories']['api_communication'] == 3
        assert summary['categories']['dataset_loading'] == 1
        assert len(summary['recent_errors']) <= 5


class TestBudgetMonitor:
    """Test BudgetMonitor class."""
    
    @pytest.fixture
    def monitor(self):
        """Create a budget monitor instance."""
        return BudgetMonitor(budget_limit=10.0, warning_threshold=0.8)
    
    def test_budget_monitor_initialization(self, monitor):
        """Test budget monitor initialization."""
        assert monitor.budget_limit == 10.0
        assert monitor.warning_threshold == 0.8
        assert monitor.current_cost == 0.0
        assert monitor._warning_sent is False
    
    def test_update_cost_under_limit(self, monitor):
        """Test updating cost when under limit."""
        monitor.update_cost(5.0)
        
        assert monitor.current_cost == 5.0
        assert monitor.get_remaining_budget() == 5.0
    
    def test_update_cost_warning_threshold(self, monitor):
        """Test warning when approaching budget limit."""
        callback_called = False
        warning_context = None
        
        def callback(context):
            nonlocal callback_called, warning_context
            callback_called = True
            warning_context = context
        
        monitor.callback = callback
        
        # Add cost to trigger warning (80% of 10.0 = 8.0)
        monitor.update_cost(8.5)
        
        assert callback_called is True
        assert warning_context.severity == ErrorSeverity.WARNING
        assert "Approaching budget limit" in warning_context.message
        assert monitor._warning_sent is True
    
    def test_update_cost_exceeded(self, monitor):
        """Test error when budget is exceeded."""
        with pytest.raises(EvaluationError) as exc_info:
            monitor.update_cost(11.0)
        
        error = exc_info.value
        assert error.context.category == ErrorCategory.RESOURCE_EXHAUSTION
        assert error.context.severity == ErrorSeverity.CRITICAL
        assert "exceeded" in error.context.message
        assert error.context.is_retryable is False
    
    def test_no_budget_limit(self):
        """Test monitor with no budget limit."""
        monitor = BudgetMonitor(budget_limit=0.0)
        
        # Should not raise even with high cost
        monitor.update_cost(1000.0)
        assert monitor.current_cost == 1000.0
    
    def test_reset(self, monitor):
        """Test resetting the budget monitor."""
        monitor.update_cost(5.0)
        monitor._warning_sent = True
        
        monitor.reset()
        
        assert monitor.current_cost == 0.0
        assert monitor._warning_sent is False


class TestErrorHandlingDecorator:
    """Test the with_error_handling decorator."""
    
    @pytest.mark.asyncio
    async def test_decorator_success(self):
        """Test decorator with successful function."""
        @with_error_handling(max_retries=2)
        async def successful_func():
            return "success"
        
        result = await successful_func()
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_decorator_with_retries(self):
        """Test decorator with retries."""
        call_count = 0
        
        @with_error_handling(max_retries=2)
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary failure")
            return "success"
        
        result = await flaky_func()
        assert result == "success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_decorator_with_specific_error_types(self):
        """Test decorator with specific error types."""
        @with_error_handling(error_types=[ValueError], max_retries=2)
        async def func_with_value_error():
            raise ValueError("Test error")
        
        with pytest.raises(EvaluationError) as exc_info:
            await func_with_value_error()
        
        assert exc_info.value.context.message == "Test error"
    
    @pytest.mark.asyncio
    async def test_decorator_unhandled_error(self):
        """Test decorator with unhandled error type."""
        @with_error_handling(error_types=[ValueError], max_retries=2)
        async def func_with_type_error():
            raise TypeError("Wrong type")
        
        with pytest.raises(TypeError) as exc_info:
            await func_with_type_error()
        
        assert "Wrong type" in str(exc_info.value)


class TestGlobalErrorHandler:
    """Test global error handler singleton."""
    
    def test_get_error_handler_singleton(self):
        """Test that get_error_handler returns singleton."""
        handler1 = get_error_handler()
        handler2 = get_error_handler()
        
        assert handler1 is handler2  # Same instance