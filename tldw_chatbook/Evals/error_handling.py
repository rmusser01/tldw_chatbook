# error_handling.py
# Description: Enhanced error handling for the evaluation system
#
"""
Evaluation Error Handling
------------------------

Provides comprehensive error handling for the evaluation system including:
- API connection failures
- Invalid dataset formats
- Token limit exceeded
- Budget warnings and limits
- Retry logic with exponential backoff
"""

import asyncio
from typing import Optional, Dict, Any, Callable, Type
from datetime import datetime
from enum import Enum
from loguru import logger
from functools import wraps


class EvalErrorType(Enum):
    """Types of errors that can occur during evaluation."""
    API_CONNECTION = "api_connection"
    API_RATE_LIMIT = "api_rate_limit"
    API_AUTHENTICATION = "api_authentication"
    DATASET_INVALID = "dataset_invalid"
    DATASET_MISSING = "dataset_missing"
    TOKEN_LIMIT_EXCEEDED = "token_limit_exceeded"
    BUDGET_EXCEEDED = "budget_exceeded"
    BUDGET_WARNING = "budget_warning"
    TIMEOUT = "timeout"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


class EvaluationError(Exception):
    """Base exception for evaluation errors."""
    
    def __init__(
        self,
        message: str,
        error_type: EvalErrorType = EvalErrorType.UNKNOWN,
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = True
    ):
        super().__init__(message)
        self.error_type = error_type
        self.details = details or {}
        self.recoverable = recoverable
        self.timestamp = datetime.now()


class APIConnectionError(EvaluationError):
    """Error connecting to LLM API."""
    
    def __init__(self, message: str, provider: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message,
            EvalErrorType.API_CONNECTION,
            details or {},
            recoverable=True
        )
        self.provider = provider


class TokenLimitError(EvaluationError):
    """Token limit exceeded error."""
    
    def __init__(
        self,
        message: str,
        tokens_used: int,
        token_limit: int,
        sample_index: Optional[int] = None
    ):
        details = {
            "tokens_used": tokens_used,
            "token_limit": token_limit,
            "sample_index": sample_index
        }
        super().__init__(
            message,
            EvalErrorType.TOKEN_LIMIT_EXCEEDED,
            details,
            recoverable=False
        )


class BudgetError(EvaluationError):
    """Budget exceeded or warning."""
    
    def __init__(
        self,
        message: str,
        current_cost: float,
        budget_limit: float,
        is_warning: bool = False
    ):
        details = {
            "current_cost": current_cost,
            "budget_limit": budget_limit,
            "percentage": (current_cost / budget_limit * 100) if budget_limit > 0 else 0
        }
        error_type = EvalErrorType.BUDGET_WARNING if is_warning else EvalErrorType.BUDGET_EXCEEDED
        super().__init__(
            message,
            error_type,
            details,
            recoverable=is_warning
        )


class DatasetError(EvaluationError):
    """Dataset-related error."""
    
    def __init__(
        self,
        message: str,
        dataset_path: str,
        error_type: EvalErrorType = EvalErrorType.DATASET_INVALID,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message,
            error_type,
            details or {},
            recoverable=False
        )
        self.dataset_path = dataset_path


class ErrorHandler:
    """Handles errors during evaluation with retry logic and reporting."""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        error_callback: Optional[Callable[[EvaluationError], None]] = None
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.error_callback = error_callback
        self.error_history: List[EvaluationError] = []
    
    async def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> bool:
        """
        Handle an error and determine if operation should be retried.
        
        Args:
            error: The exception that occurred
            context: Context information about where error occurred
            
        Returns:
            bool: True if operation should be retried, False otherwise
        """
        # Convert to EvaluationError if needed
        if not isinstance(error, EvaluationError):
            eval_error = self._convert_to_eval_error(error, context)
        else:
            eval_error = error
        
        # Log error
        logger.error(
            f"Evaluation error: {eval_error.message}",
            error_type=eval_error.error_type.value,
            details=eval_error.details,
            context=context
        )
        
        # Record error
        self.error_history.append(eval_error)
        
        # Call error callback
        if self.error_callback:
            try:
                self.error_callback(eval_error)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
        
        # Determine if retry is appropriate
        return eval_error.recoverable and context.get('retry_count', 0) < self.max_retries
    
    def _convert_to_eval_error(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> EvaluationError:
        """Convert generic exception to EvaluationError."""
        error_str = str(error).lower()
        
        # API errors
        if "connection" in error_str or "timeout" in error_str:
            return APIConnectionError(
                str(error),
                provider=context.get('provider', 'unknown')
            )
        elif "rate limit" in error_str or "429" in error_str:
            return EvaluationError(
                str(error),
                EvalErrorType.API_RATE_LIMIT,
                recoverable=True
            )
        elif "unauthorized" in error_str or "401" in error_str:
            return EvaluationError(
                str(error),
                EvalErrorType.API_AUTHENTICATION,
                recoverable=False
            )
        elif "token" in error_str and "limit" in error_str:
            return TokenLimitError(
                str(error),
                tokens_used=context.get('tokens_used', 0),
                token_limit=context.get('token_limit', 0)
            )
        else:
            return EvaluationError(str(error))
    
    async def retry_with_backoff(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with exponential backoff retry.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Result from func
            
        Raises:
            Last exception if all retries fail
        """
        retry_count = 0
        last_error = None
        
        while retry_count <= self.max_retries:
            try:
                return await func(*args, **kwargs)
            
            except Exception as e:
                last_error = e
                context = {
                    'retry_count': retry_count,
                    'function': func.__name__,
                    **kwargs.get('context', {})
                }
                
                should_retry = await self.handle_error(e, context)
                
                if not should_retry:
                    raise
                
                # Calculate delay with exponential backoff
                delay = min(
                    self.base_delay * (2 ** retry_count),
                    self.max_delay
                )
                
                logger.info(
                    f"Retrying {func.__name__} after {delay}s "
                    f"(attempt {retry_count + 1}/{self.max_retries})"
                )
                
                await asyncio.sleep(delay)
                retry_count += 1
        
        # All retries exhausted
        raise last_error


def with_error_handling(
    error_types: List[Type[Exception]] = None,
    max_retries: int = 3,
    recoverable: bool = True
):
    """
    Decorator for adding error handling to async functions.
    
    Args:
        error_types: List of exception types to handle
        max_retries: Maximum number of retries
        recoverable: Whether errors are recoverable by default
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            handler = ErrorHandler(max_retries=max_retries)
            
            try:
                return await handler.retry_with_backoff(func, *args, **kwargs)
            except Exception as e:
                # Re-raise if not in handled types
                if error_types and not any(isinstance(e, t) for t in error_types):
                    raise
                
                # Convert and re-raise
                if not isinstance(e, EvaluationError):
                    raise EvaluationError(
                        str(e),
                        recoverable=recoverable
                    )
                raise
        
        return wrapper
    return decorator


class BudgetMonitor:
    """Monitors evaluation costs against budget limits."""
    
    def __init__(
        self,
        budget_limit: float,
        warning_threshold: float = 0.8,
        callback: Optional[Callable[[BudgetError], None]] = None
    ):
        self.budget_limit = budget_limit
        self.warning_threshold = warning_threshold
        self.callback = callback
        self.current_cost = 0.0
        self._warning_sent = False
    
    def update_cost(self, additional_cost: float) -> None:
        """
        Update current cost and check budget.
        
        Args:
            additional_cost: Cost to add to current total
            
        Raises:
            BudgetError: If budget is exceeded
        """
        self.current_cost += additional_cost
        
        if self.budget_limit <= 0:
            return  # No budget limit set
        
        percentage = self.current_cost / self.budget_limit
        
        # Check for budget exceeded
        if percentage >= 1.0:
            error = BudgetError(
                f"Budget limit of ${self.budget_limit:.2f} exceeded. "
                f"Current cost: ${self.current_cost:.2f}",
                current_cost=self.current_cost,
                budget_limit=self.budget_limit,
                is_warning=False
            )
            
            if self.callback:
                self.callback(error)
            
            raise error
        
        # Check for warning threshold
        elif percentage >= self.warning_threshold and not self._warning_sent:
            self._warning_sent = True
            warning = BudgetError(
                f"Approaching budget limit: ${self.current_cost:.2f} "
                f"of ${self.budget_limit:.2f} ({percentage*100:.1f}%)",
                current_cost=self.current_cost,
                budget_limit=self.budget_limit,
                is_warning=True
            )
            
            if self.callback:
                self.callback(warning)
            
            logger.warning(warning.message)
    
    def get_remaining_budget(self) -> float:
        """Get remaining budget."""
        return max(0, self.budget_limit - self.current_cost)
    
    def reset(self) -> None:
        """Reset cost tracking."""
        self.current_cost = 0.0
        self._warning_sent = False