# eval_runner_enhanced.py
# Description: Enhanced error handling for eval_runner.py
#
"""
Enhanced Error Handling for Evaluation Runner
--------------------------------------------

This module provides enhanced error handling patches for eval_runner.py
These can be integrated into the main eval_runner.py file.
"""

from typing import Dict, List, Any, Optional
import asyncio
import traceback
from loguru import logger

from .eval_errors import (
    get_error_handler, EvaluationError, DatasetLoadingError, 
    ModelConfigurationError, APIError, ExecutionError, 
    ValidationError, ErrorContext, ErrorCategory, ErrorSeverity
)
from .eval_runner import DatasetLoader, EvalSample

class EnhancedDatasetLoader(DatasetLoader):
    """Enhanced dataset loader with better error handling."""
    
    @staticmethod
    def load_dataset_samples(task_config, split: str = None, max_samples: int = None) -> List[EvalSample]:
        """Load samples with enhanced error handling."""
        error_handler = get_error_handler()
        
        try:
            if split is None:
                split = task_config.split
            
            dataset_name = task_config.dataset_name
            
            # Validate dataset name
            if not dataset_name:
                raise DatasetLoadingError.missing_required_fields(['dataset_name'])
            
            # Handle different dataset sources with specific error handling
            if Path(dataset_name).exists():
                return EnhancedDatasetLoader._load_local_dataset_safe(dataset_name, max_samples)
            elif HF_DATASETS_AVAILABLE and '/' in dataset_name:
                return EnhancedDatasetLoader._load_huggingface_dataset_safe(task_config, split, max_samples)
            else:
                raise DatasetLoadingError(ErrorContext(
                    category=ErrorCategory.DATASET_LOADING,
                    severity=ErrorSeverity.ERROR,
                    message=f"Cannot determine dataset type for: {dataset_name}",
                    suggestion="Provide a valid local file path or HuggingFace dataset name (format: 'owner/dataset')",
                    is_retryable=False
                ))
                
        except EvaluationError:
            raise
        except Exception as e:
            error_context = error_handler.handle_error(e, {
                'dataset_name': dataset_name,
                'split': split,
                'max_samples': max_samples
            })
            raise EvaluationError(error_context, e)
    
    @staticmethod
    def _load_local_dataset_safe(dataset_path: str, max_samples: int = None) -> List[EvalSample]:
        """Load local dataset with enhanced error handling."""
        path = Path(dataset_path)
        
        # Check file exists
        if not path.exists():
            raise DatasetLoadingError.file_not_found(str(path))
        
        # Check file is readable
        if not path.is_file():
            raise DatasetLoadingError(ErrorContext(
                category=ErrorCategory.DATASET_LOADING,
                severity=ErrorSeverity.ERROR,
                message=f"Path is not a file: {path}",
                suggestion="Provide a path to a valid dataset file",
                is_retryable=False
            ))
        
        # Check file size
        file_size = path.stat().st_size
        if file_size == 0:
            raise DatasetLoadingError(ErrorContext(
                category=ErrorCategory.DATASET_LOADING,
                severity=ErrorSeverity.ERROR,
                message=f"Dataset file is empty: {path}",
                suggestion="Ensure the dataset file contains data",
                is_retryable=False
            ))
        
        if file_size > 1_000_000_000:  # 1GB warning
            logger.warning(f"Large dataset file ({file_size / 1_000_000:.1f} MB): {path}")
        
        # Load based on extension
        try:
            if path.suffix.lower() == '.json':
                return EnhancedDatasetLoader._load_json_dataset_safe(path, max_samples)
            elif path.suffix.lower() in ['.csv', '.tsv']:
                return EnhancedDatasetLoader._load_csv_dataset_safe(path, max_samples)
            else:
                raise DatasetLoadingError(ErrorContext(
                    category=ErrorCategory.DATASET_LOADING,
                    severity=ErrorSeverity.ERROR,
                    message=f"Unsupported file format: {path.suffix}",
                    suggestion="Use JSON (.json) or CSV/TSV (.csv, .tsv) format",
                    is_retryable=False
                ))
        except EvaluationError:
            raise
        except Exception as e:
            raise DatasetLoadingError.invalid_format(str(path), str(e))
    
    @staticmethod
    def _load_json_dataset_safe(path: Path, max_samples: int = None) -> List[EvalSample]:
        """Load JSON dataset with enhanced error handling."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise DatasetLoadingError.invalid_format(
                str(path), 
                f"Invalid JSON at line {e.lineno}, column {e.colno}: {e.msg}"
            )
        except UnicodeDecodeError as e:
            raise DatasetLoadingError(ErrorContext(
                category=ErrorCategory.DATASET_LOADING,
                severity=ErrorSeverity.ERROR,
                message=f"File encoding error in {path}",
                details=str(e),
                suggestion="Ensure the file is UTF-8 encoded",
                is_retryable=False
            ))
        
        if not isinstance(data, list):
            raise DatasetLoadingError.invalid_format(
                str(path),
                "JSON file must contain an array of samples at the root level"
            )
        
        if len(data) == 0:
            raise DatasetLoadingError(ErrorContext(
                category=ErrorCategory.DATASET_LOADING,
                severity=ErrorSeverity.ERROR,
                message=f"No samples found in {path}",
                suggestion="Add sample data to the JSON array",
                is_retryable=False
            ))
        
        samples = []
        errors = []
        
        for i, item in enumerate(data[:max_samples] if max_samples else data):
            try:
                # Validate required fields
                if not isinstance(item, dict):
                    errors.append(f"Sample {i}: Not a JSON object")
                    continue
                
                sample_id = item.get('id', str(i))
                input_text = item.get('input', item.get('question', item.get('text', '')))
                
                if not input_text:
                    errors.append(f"Sample {sample_id}: Missing input text (checked 'input', 'question', 'text' fields)")
                    continue
                
                expected_output = item.get('output', item.get('answer', item.get('target')))
                choices = item.get('choices', item.get('options'))
                
                samples.append(EvalSample(
                    id=sample_id,
                    input_text=input_text,
                    expected_output=expected_output,
                    choices=choices,
                    metadata=item
                ))
                
            except Exception as e:
                errors.append(f"Sample {i}: {str(e)}")
        
        # Report errors if any samples failed
        if errors:
            error_summary = "\n".join(errors[:10])  # Show first 10 errors
            if len(errors) > 10:
                error_summary += f"\n... and {len(errors) - 10} more errors"
            
            logger.warning(f"Failed to load {len(errors)} samples from {path}:\n{error_summary}")
        
        if not samples:
            raise DatasetLoadingError(ErrorContext(
                category=ErrorCategory.DATASET_LOADING,
                severity=ErrorSeverity.ERROR,
                message=f"No valid samples could be loaded from {path}",
                details=error_summary if errors else None,
                suggestion="Fix the sample format errors and try again",
                is_retryable=False
            ))
        
        return samples

class EnhancedErrorHandler:
    """Enhanced error handler with better retry logic and error classification."""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor
        self.rate_limit_delays = {}  # Provider -> delay time
        self.error_handler = get_error_handler()
        self.consecutive_errors = {}  # Track consecutive errors per provider
    
    async def with_retry(self, operation, sample_id: str, provider: str = None):
        """Execute operation with enhanced retry logic and error handling."""
        last_exception = None
        retry_count = 0
        
        # Reset consecutive errors on new operation
        if provider and provider in self.consecutive_errors:
            self.consecutive_errors[provider] = 0
        
        for attempt in range(self.max_retries + 1):
            try:
                # Check rate limiting
                if provider and provider in self.rate_limit_delays:
                    delay = self.rate_limit_delays[provider]
                    if delay > 0:
                        logger.info(f"Rate limited for {provider}, waiting {delay:.1f}s")
                        await asyncio.sleep(delay)
                        # Reduce delay for next attempt
                        self.rate_limit_delays[provider] = max(0, delay - 1.0)
                
                # Execute operation
                result = await operation()
                
                # Clear rate limit and error counts if successful
                if provider:
                    if provider in self.rate_limit_delays:
                        self.rate_limit_delays[provider] = 0
                    if provider in self.consecutive_errors:
                        self.consecutive_errors[provider] = 0
                
                return result, retry_count
                
            except EvalRateLimitError as e:
                logger.warning(f"Rate limit hit for {provider}: {e}")
                
                # Track consecutive errors
                if provider:
                    self.consecutive_errors[provider] = self.consecutive_errors.get(provider, 0) + 1
                    
                    # Exponential backoff for rate limits
                    base_delay = 5.0 * (attempt + 1)
                    max_delay = 60.0
                    self.rate_limit_delays[provider] = min(max_delay, base_delay)
                
                if attempt < self.max_retries:
                    delay = self.retry_delay * (self.backoff_factor ** attempt)
                    logger.info(f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                    await asyncio.sleep(delay)
                    retry_count += 1
                    last_exception = e
                else:
                    # Convert to evaluation error with context
                    raise APIError.rate_limit_exceeded(provider or "unknown", self.rate_limit_delays.get(provider))
                    
            except EvalAuthenticationError as e:
                logger.error(f"Authentication error for {provider}: {e}")
                # Don't retry auth errors
                raise APIError.authentication_failed(provider or "unknown")
                
            except (EvalAPIError, EvalProviderError) as e:
                logger.warning(f"API error for sample {sample_id}: {e}")
                
                # Track consecutive errors
                if provider:
                    self.consecutive_errors[provider] = self.consecutive_errors.get(provider, 0) + 1
                    
                    # If too many consecutive errors, increase delay
                    if self.consecutive_errors[provider] > 5:
                        logger.warning(f"Multiple consecutive errors for {provider}, adding delay")
                        self.rate_limit_delays[provider] = 10.0
                
                if attempt < self.max_retries:
                    delay = self.retry_delay * (self.backoff_factor ** attempt)
                    logger.info(f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                    await asyncio.sleep(delay)
                    retry_count += 1
                    last_exception = e
                else:
                    # Convert to evaluation error
                    raise APIError.connection_failed(provider or "unknown", str(e))
                    
            except asyncio.TimeoutError:
                logger.warning(f"Timeout for sample {sample_id}")
                
                if attempt < self.max_retries:
                    delay = self.retry_delay * (self.backoff_factor ** attempt)
                    logger.info(f"Retrying after timeout in {delay:.1f}s")
                    await asyncio.sleep(delay)
                    retry_count += 1
                    last_exception = ExecutionError.timeout(sample_id, 30.0)  # Default timeout
                else:
                    raise ExecutionError.timeout(sample_id, 30.0)
                    
            except Exception as e:
                logger.error(f"Unexpected error for sample {sample_id}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                
                # Use error handler to classify
                error_context = self.error_handler.handle_error(e, {
                    'sample_id': sample_id,
                    'provider': provider,
                    'attempt': attempt
                })
                
                if error_context.is_retryable and attempt < self.max_retries:
                    delay = self.retry_delay * (self.backoff_factor ** attempt)
                    logger.info(f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                    await asyncio.sleep(delay)
                    retry_count += 1
                    last_exception = EvaluationError(error_context, e)
                else:
                    raise EvaluationError(error_context, e)
        
        # If we get here, all retries failed
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError(f"All retries failed for sample {sample_id}")
    
    def classify_error(self, error: Exception) -> Dict[str, Any]:
        """Enhanced error classification."""
        if isinstance(error, EvaluationError):
            return error.context.to_dict()
        
        # Use centralized error handler
        error_context = self.error_handler.handle_error(error)
        return error_context.to_dict()
    
    def get_provider_health(self, provider: str) -> Dict[str, Any]:
        """Get health status for a provider."""
        return {
            'provider': provider,
            'rate_limited': self.rate_limit_delays.get(provider, 0) > 0,
            'rate_limit_expires': self.rate_limit_delays.get(provider, 0),
            'consecutive_errors': self.consecutive_errors.get(provider, 0),
            'health_status': 'healthy' if self.consecutive_errors.get(provider, 0) < 3 else 'degraded'
        }

# Additional helper functions for better error reporting

def format_error_for_user(error: Exception) -> str:
    """Format an error for user display."""
    if isinstance(error, EvaluationError):
        return error.get_user_message()
    
    # Generic formatting
    return f"An error occurred: {str(error)}"

def format_error_for_log(error: Exception) -> str:
    """Format an error for logging with full details."""
    if isinstance(error, EvaluationError):
        return error.get_technical_details()
    
    # Generic formatting with traceback
    return f"{type(error).__name__}: {str(error)}\n{traceback.format_exc()}"

async def handle_batch_errors(operations: List[asyncio.Task], 
                            continue_on_error: bool = True) -> tuple[List[Any], List[Exception]]:
    """Handle errors in batch operations."""
    results = []
    errors = []
    
    if continue_on_error:
        # Gather with return_exceptions to continue on errors
        outcomes = await asyncio.gather(*operations, return_exceptions=True)
        
        for outcome in outcomes:
            if isinstance(outcome, Exception):
                errors.append(outcome)
                results.append(None)
            else:
                results.append(outcome)
                errors.append(None)
    else:
        # Stop on first error
        try:
            results = await asyncio.gather(*operations)
            errors = [None] * len(results)
        except Exception as e:
            # Find which task failed
            for i, task in enumerate(operations):
                if task.done() and task.exception():
                    errors.append(task.exception())
                    results.append(None)
                elif task.done():
                    results.append(task.result())
                    errors.append(None)
                else:
                    # Task was cancelled
                    errors.append(asyncio.CancelledError())
                    results.append(None)
            raise
    
    return results, errors