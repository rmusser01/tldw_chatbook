# unified_error_handler.py
# Description: Unified error handling for the entire Evals module
#
"""
Unified Error Handler
---------------------

Centralized error handling for all evaluation components.
Uses existing Chat error types for consistency.
"""

import asyncio
import time
from typing import Any, Callable, Optional, Dict, Type, Tuple
from functools import wraps
from contextlib import asynccontextmanager
from loguru import logger

# Use existing error types from Chat module for consistency
from tldw_chatbook.Chat.Chat_Deps import (
    ChatProviderError,
    ChatAPIError,
    ChatAuthenticationError,
    ChatRateLimitError,
    ChatBadRequestError,
    ChatConfigurationError
)


class EvaluationError(Exception):
    """Base class for all evaluation-specific errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None,
                 suggestion: Optional[str] = None, is_retryable: bool = False):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.suggestion = suggestion
        self.is_retryable = is_retryable
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/storage."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'details': self.details,
            'suggestion': self.suggestion,
            'is_retryable': self.is_retryable
        }


class TaskLoadingError(EvaluationError):
    """Error loading or parsing evaluation tasks."""
    pass


class DatasetError(EvaluationError):
    """Error loading or processing datasets."""
    pass


class MetricsError(EvaluationError):
    """Error calculating or storing metrics."""
    pass


class RunnerError(EvaluationError):
    """Error during evaluation execution."""
    pass


class UnifiedErrorHandler:
    """
    Unified error handler for all evaluation operations.
    Provides consistent error handling, retry logic, and recovery.
    """
    
    # Map external errors to evaluation errors
    ERROR_MAPPING = {
        ChatAuthenticationError: (TaskLoadingError, "Authentication failed", False),
        ChatRateLimitError: (RunnerError, "Rate limit exceeded", True),
        ChatBadRequestError: (RunnerError, "Invalid request", False),
        ChatConfigurationError: (TaskLoadingError, "Configuration error", False),
        ChatAPIError: (RunnerError, "API error", True),
        ChatProviderError: (RunnerError, "Provider error", True),
        FileNotFoundError: (DatasetError, "File not found", False),
        json.JSONDecodeError: (DatasetError, "Invalid JSON format", False),
        ValueError: (MetricsError, "Invalid value", False),
        KeyError: (MetricsError, "Missing required field", False),
    }
    
    def __init__(self, 
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 exponential_backoff: bool = True):
        """
        Initialize the error handler.
        
        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (seconds)
            exponential_backoff: Whether to use exponential backoff
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.exponential_backoff = exponential_backoff
        self.error_counts = {}  # Track errors by type
        
    def handle_error(self, error: Exception, context: str = "") -> EvaluationError:
        """
        Convert any error to an EvaluationError with context.
        
        Args:
            error: The original exception
            context: Additional context about where the error occurred
            
        Returns:
            An appropriate EvaluationError subclass
        """
        error_type = type(error)
        
        # Track error frequency
        error_name = error_type.__name__
        self.error_counts[error_name] = self.error_counts.get(error_name, 0) + 1
        
        # Map to evaluation error
        if error_type in self.ERROR_MAPPING:
            eval_error_class, message, is_retryable = self.ERROR_MAPPING[error_type]
            
            details = {
                'original_error': str(error),
                'error_type': error_name,
                'context': context,
                'occurrence_count': self.error_counts[error_name]
            }
            
            # Add specific details based on error type
            if hasattr(error, 'response'):
                details['response'] = str(error.response)
            if hasattr(error, 'status_code'):
                details['status_code'] = error.status_code
                
            suggestion = self._get_suggestion(error_type, error)
            
            return eval_error_class(
                message=f"{message}: {str(error)}",
                details=details,
                suggestion=suggestion,
                is_retryable=is_retryable
            )
        
        # Default handling for unknown errors
        return RunnerError(
            message=f"Unexpected error in {context}: {str(error)}",
            details={
                'original_error': str(error),
                'error_type': error_name,
                'context': context
            },
            suggestion="Check logs for details",
            is_retryable=False
        )
    
    def _get_suggestion(self, error_type: Type[Exception], error: Exception) -> str:
        """Get helpful suggestion based on error type."""
        suggestions = {
            ChatAuthenticationError: "Check your API key in the configuration",
            ChatRateLimitError: "Wait a moment and try again, or reduce request rate",
            FileNotFoundError: f"Ensure the file exists: {error}",
            json.JSONDecodeError: "Validate the JSON format of your input file",
            ValueError: "Check input values match expected format",
            KeyError: f"Ensure required field is present: {error}",
        }
        
        return suggestions.get(error_type, "Review the error details and try again")
    
    async def with_retry(self, 
                        operation: Callable,
                        operation_name: str = "operation",
                        **kwargs) -> Tuple[Any, int]:
        """
        Execute an operation with retry logic.
        
        Args:
            operation: Async callable to execute
            operation_name: Name for logging
            **kwargs: Arguments to pass to operation
            
        Returns:
            Tuple of (result, retry_count)
            
        Raises:
            EvaluationError: If all retries fail
        """
        last_error = None
        retry_count = 0
        
        for attempt in range(self.max_retries + 1):
            try:
                # Execute the operation
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(**kwargs)
                else:
                    result = operation(**kwargs)
                    
                # Success - log if we had retries
                if attempt > 0:
                    logger.info(f"Operation '{operation_name}' succeeded after {attempt} retries")
                    
                return result, attempt
                
            except Exception as e:
                retry_count = attempt
                last_error = self.handle_error(e, operation_name)
                
                # Check if retryable
                if not last_error.is_retryable or attempt >= self.max_retries:
                    logger.error(f"Operation '{operation_name}' failed after {attempt + 1} attempts: {last_error.message}")
                    raise last_error
                
                # Calculate delay with exponential backoff
                delay = self.retry_delay * (2 ** attempt if self.exponential_backoff else 1)
                
                logger.warning(f"Operation '{operation_name}' failed (attempt {attempt + 1}/{self.max_retries + 1}), "
                             f"retrying in {delay}s: {e}")
                
                await asyncio.sleep(delay)
        
        # Should never reach here, but for safety
        raise last_error or RunnerError("Operation failed with unknown error")
    
    @asynccontextmanager
    async def error_context(self, context_name: str):
        """
        Context manager for consistent error handling.
        
        Usage:
            async with error_handler.error_context("loading dataset"):
                # operations that might fail
        """
        try:
            yield
        except EvaluationError:
            # Re-raise evaluation errors as-is
            raise
        except Exception as e:
            # Convert other errors
            eval_error = self.handle_error(e, context_name)
            logger.error(f"Error in {context_name}: {eval_error.to_dict()}")
            raise eval_error
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors encountered."""
        return {
            'total_errors': sum(self.error_counts.values()),
            'error_counts': self.error_counts.copy(),
            'most_common': max(self.error_counts.items(), key=lambda x: x[1])[0] if self.error_counts else None
        }
    
    def reset_error_counts(self):
        """Reset error tracking."""
        self.error_counts.clear()


def handle_eval_errors(max_retries: int = 3):
    """
    Decorator for consistent error handling on methods.
    
    Usage:
        @handle_eval_errors(max_retries=3)
        async def my_method(self, ...):
            # method implementation
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            handler = UnifiedErrorHandler(max_retries=max_retries)
            
            try:
                # Try to get operation name from function
                operation_name = func.__name__.replace('_', ' ')
                
                # Execute with retry
                result, retries = await handler.with_retry(
                    func,
                    operation_name=operation_name,
                    *args,
                    **kwargs
                )
                
                return result
                
            except EvaluationError as e:
                # Log and re-raise evaluation errors
                logger.error(f"Evaluation error in {func.__name__}: {e.to_dict()}")
                raise
            except Exception as e:
                # Convert unexpected errors
                eval_error = handler.handle_error(e, func.__name__)
                logger.error(f"Unexpected error in {func.__name__}: {eval_error.to_dict()}")
                raise eval_error
                
        return wrapper
    return decorator


# Singleton instance for module-wide use
_global_handler = UnifiedErrorHandler()

def get_error_handler() -> UnifiedErrorHandler:
    """Get the global error handler instance."""
    return _global_handler


# Import json for error mapping
import json