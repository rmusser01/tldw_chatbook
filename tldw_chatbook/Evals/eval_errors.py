# eval_errors.py
# Description: Enhanced error handling for the evaluation system
#
"""
Evaluation Error Handling
------------------------

Provides comprehensive error handling for the evaluation system with:
- Specific error categories
- User-friendly error messages
- Recovery suggestions
- Error context preservation
"""

from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

class ErrorSeverity(Enum):
    """Error severity levels for UI display."""
    INFO = "information"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Categories of errors for better handling."""
    DATASET_LOADING = "dataset_loading"
    MODEL_CONFIGURATION = "model_configuration"
    API_COMMUNICATION = "api_communication"
    AUTHENTICATION = "authentication"
    RATE_LIMITING = "rate_limiting"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    VALIDATION = "validation"
    FILE_SYSTEM = "file_system"
    DATABASE = "database"
    NETWORK = "network"
    SECURITY = "security"
    UNKNOWN = "unknown"

@dataclass
class ErrorContext:
    """Context information for an error."""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Optional[str] = None
    suggestion: Optional[str] = None
    is_retryable: bool = True
    retry_after: Optional[float] = None
    error_code: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/display."""
        return {
            'category': self.category.value,
            'severity': self.severity.value,
            'message': self.message,
            'details': self.details,
            'suggestion': self.suggestion,
            'is_retryable': self.is_retryable,
            'retry_after': self.retry_after,
            'error_code': self.error_code,
            'timestamp': self.timestamp.isoformat()
        }

class EvaluationError(Exception):
    """Base exception for evaluation errors with context."""
    
    def __init__(self, context: ErrorContext, original_error: Optional[Exception] = None):
        self.context = context
        self.original_error = original_error
        super().__init__(context.message)
    
    def get_user_message(self) -> str:
        """Get user-friendly error message."""
        msg = self.context.message
        if self.context.suggestion:
            msg += f"\n\nSuggestion: {self.context.suggestion}"
        return msg
    
    def get_technical_details(self) -> str:
        """Get technical error details for logging/debugging."""
        details = [
            f"Category: {self.context.category.value}",
            f"Severity: {self.context.severity.value}",
            f"Message: {self.context.message}"
        ]
        
        if self.context.details:
            details.append(f"Details: {self.context.details}")
        
        if self.context.error_code:
            details.append(f"Error Code: {self.context.error_code}")
        
        if self.original_error:
            details.append(f"Original Error: {type(self.original_error).__name__}: {str(self.original_error)}")
        
        return "\n".join(details)

class DatasetLoadingError(EvaluationError):
    """Error loading or processing dataset."""
    
    @staticmethod
    def file_not_found(file_path: str) -> 'DatasetLoadingError':
        return DatasetLoadingError(ErrorContext(
            category=ErrorCategory.DATASET_LOADING,
            severity=ErrorSeverity.ERROR,
            message=f"Dataset file not found: {file_path}",
            suggestion="Check the file path and ensure the file exists",
            is_retryable=False
        ))
    
    @staticmethod
    def invalid_format(file_path: str, format_error: str) -> 'DatasetLoadingError':
        return DatasetLoadingError(ErrorContext(
            category=ErrorCategory.DATASET_LOADING,
            severity=ErrorSeverity.ERROR,
            message=f"Invalid dataset format in {file_path}",
            details=format_error,
            suggestion="Ensure the dataset follows the expected format (JSON array or CSV with headers)",
            is_retryable=False
        ))
    
    @staticmethod
    def missing_required_fields(missing_fields: List[str]) -> 'DatasetLoadingError':
        return DatasetLoadingError(ErrorContext(
            category=ErrorCategory.DATASET_LOADING,
            severity=ErrorSeverity.ERROR,
            message=f"Dataset missing required fields: {', '.join(missing_fields)}",
            suggestion="Add the missing fields to your dataset or update the task configuration",
            is_retryable=False
        ))

class ModelConfigurationError(EvaluationError):
    """Error in model configuration."""
    
    @staticmethod
    def invalid_provider(provider: str) -> 'ModelConfigurationError':
        return ModelConfigurationError(ErrorContext(
            category=ErrorCategory.MODEL_CONFIGURATION,
            severity=ErrorSeverity.ERROR,
            message=f"Invalid model provider: {provider}",
            suggestion="Check the provider name and ensure it's supported",
            is_retryable=False
        ))
    
    @staticmethod
    def missing_api_key(provider: str) -> 'ModelConfigurationError':
        return ModelConfigurationError(ErrorContext(
            category=ErrorCategory.MODEL_CONFIGURATION,
            severity=ErrorSeverity.ERROR,
            message=f"API key not found for {provider}",
            suggestion=f"Set the API key in settings or environment variable",
            is_retryable=False,
            error_code="MISSING_API_KEY"
        ))

class APIError(EvaluationError):
    """API communication error."""
    
    @staticmethod
    def connection_failed(provider: str, details: str) -> 'APIError':
        return APIError(ErrorContext(
            category=ErrorCategory.API_COMMUNICATION,
            severity=ErrorSeverity.ERROR,
            message=f"Failed to connect to {provider} API",
            details=details,
            suggestion="Check your internet connection and API endpoint",
            is_retryable=True
        ))
    
    @staticmethod
    def rate_limit_exceeded(provider: str, retry_after: Optional[float] = None) -> 'APIError':
        return APIError(ErrorContext(
            category=ErrorCategory.RATE_LIMITING,
            severity=ErrorSeverity.WARNING,
            message=f"{provider} rate limit exceeded",
            suggestion="Wait a moment before retrying or reduce request frequency",
            is_retryable=True,
            retry_after=retry_after
        ))
    
    @staticmethod
    def authentication_failed(provider: str) -> 'APIError':
        return APIError(ErrorContext(
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.ERROR,
            message=f"Authentication failed for {provider}",
            suggestion="Check your API credentials and ensure they're valid",
            is_retryable=False,
            error_code="AUTH_FAILED"
        ))

class ExecutionError(EvaluationError):
    """Error during evaluation execution."""
    
    @staticmethod
    def timeout(task_name: str, timeout_seconds: float) -> 'ExecutionError':
        return ExecutionError(ErrorContext(
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.WARNING,
            message=f"Task '{task_name}' timed out after {timeout_seconds} seconds",
            suggestion="Consider increasing the timeout or optimizing the task",
            is_retryable=True
        ))
    
    @staticmethod
    def resource_exhausted(resource: str) -> 'ExecutionError':
        return ExecutionError(ErrorContext(
            category=ErrorCategory.RESOURCE_EXHAUSTION,
            severity=ErrorSeverity.ERROR,
            message=f"Insufficient {resource} to complete evaluation",
            suggestion=f"Free up {resource} or reduce the evaluation size",
            is_retryable=True
        ))

class ValidationError(EvaluationError):
    """Configuration or input validation error."""
    
    @staticmethod
    def invalid_configuration(field: str, reason: str) -> 'ValidationError':
        return ValidationError(ErrorContext(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.ERROR,
            message=f"Invalid configuration for '{field}': {reason}",
            suggestion="Review the configuration and fix the invalid values",
            is_retryable=False
        ))
    
    @staticmethod
    def conflicting_parameters(param1: str, param2: str) -> 'ValidationError':
        return ValidationError(ErrorContext(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.ERROR,
            message=f"Conflicting parameters: '{param1}' and '{param2}'",
            suggestion="Remove or adjust one of the conflicting parameters",
            is_retryable=False
        ))

class DatabaseError(EvaluationError):
    """Database operation error."""
    
    @staticmethod
    def connection_failed(details: str) -> 'DatabaseError':
        return DatabaseError(ErrorContext(
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.CRITICAL,
            message="Failed to connect to evaluation database",
            details=details,
            suggestion="Check database file permissions and disk space",
            is_retryable=True
        ))
    
    @staticmethod
    def operation_failed(operation: str, details: str) -> 'DatabaseError':
        return DatabaseError(ErrorContext(
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.ERROR,
            message=f"Database {operation} failed",
            details=details,
            suggestion="Check database integrity and available disk space",
            is_retryable=True
        ))

class FileSystemError(EvaluationError):
    """File system operation error."""
    
    @staticmethod
    def permission_denied(path: str) -> 'FileSystemError':
        return FileSystemError(ErrorContext(
            category=ErrorCategory.FILE_SYSTEM,
            severity=ErrorSeverity.ERROR,
            message=f"Permission denied accessing: {path}",
            suggestion="Check file permissions and ensure write access",
            is_retryable=False
        ))
    
    @staticmethod
    def disk_full(path: str) -> 'FileSystemError':
        return FileSystemError(ErrorContext(
            category=ErrorCategory.FILE_SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            message=f"Insufficient disk space at: {path}",
            suggestion="Free up disk space before continuing",
            is_retryable=False
        ))

class SecurityError(EvaluationError):
    """Security-related error."""
    
    @staticmethod
    def unsafe_code_execution(code_snippet: str) -> 'SecurityError':
        return SecurityError(ErrorContext(
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.CRITICAL,
            message="Attempted to execute potentially unsafe code",
            details=f"Code contained restricted operations",
            suggestion="Review the code for security issues",
            is_retryable=False,
            error_code="UNSAFE_CODE"
        ))

class ErrorHandler:
    """Central error handler for the evaluation system."""
    
    def __init__(self):
        self.error_history: List[EvaluationError] = []
        self.max_history_size = 100
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorContext:
        """Handle an error and return appropriate context."""
        if isinstance(error, EvaluationError):
            self._add_to_history(error)
            return error.context
        
        # Map common exceptions to our error types
        error_context = self._map_exception_to_context(error, context)
        eval_error = EvaluationError(error_context, error)
        self._add_to_history(eval_error)
        
        return error_context
    
    def _map_exception_to_context(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorContext:
        """Map standard exceptions to error contexts."""
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Network errors
        if 'connection' in error_str or 'network' in error_str:
            return ErrorContext(
                category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.ERROR,
                message="Network connection error",
                details=str(error),
                suggestion="Check your internet connection",
                is_retryable=True
            )
        
        # File errors
        elif isinstance(error, FileNotFoundError):
            return ErrorContext(
                category=ErrorCategory.FILE_SYSTEM,
                severity=ErrorSeverity.ERROR,
                message=f"File not found: {error.filename if hasattr(error, 'filename') else 'unknown'}",
                suggestion="Check the file path",
                is_retryable=False
            )
        
        # Permission errors
        elif isinstance(error, PermissionError):
            return ErrorContext(
                category=ErrorCategory.FILE_SYSTEM,
                severity=ErrorSeverity.ERROR,
                message="Permission denied",
                details=str(error),
                suggestion="Check file permissions",
                is_retryable=False
            )
        
        # Value errors (often validation)
        elif isinstance(error, ValueError):
            return ErrorContext(
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.ERROR,
                message="Invalid value provided",
                details=str(error),
                suggestion="Check input parameters",
                is_retryable=False
            )
        
        # Generic fallback
        else:
            return ErrorContext(
                category=ErrorCategory.UNKNOWN,
                severity=ErrorSeverity.ERROR,
                message=f"Unexpected error: {error_type}",
                details=str(error),
                suggestion="Check logs for more details",
                is_retryable=True
            )
    
    def _add_to_history(self, error: EvaluationError):
        """Add error to history, maintaining size limit."""
        self.error_history.append(error)
        if len(self.error_history) > self.max_history_size:
            self.error_history.pop(0)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors."""
        if not self.error_history:
            return {"total_errors": 0, "categories": {}}
        
        category_counts = {}
        for error in self.error_history:
            category = error.context.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "categories": category_counts,
            "recent_errors": [
                {
                    "timestamp": error.context.timestamp.isoformat(),
                    "category": error.context.category.value,
                    "message": error.context.message
                }
                for error in self.error_history[-5:]  # Last 5 errors
            ]
        }
    
    def clear_history(self):
        """Clear error history."""
        self.error_history.clear()

# Global error handler instance
_error_handler = None

def get_error_handler() -> ErrorHandler:
    """Get or create the global error handler."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler