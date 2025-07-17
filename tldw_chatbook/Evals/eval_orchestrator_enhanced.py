# eval_orchestrator_enhanced.py
# Description: Enhanced error handling for eval_orchestrator.py
#
"""
Enhanced Error Handling for Evaluation Orchestrator
--------------------------------------------------

This module provides enhanced error handling patches for eval_orchestrator.py
These improvements can be integrated into the main orchestrator.
"""

import asyncio
import json
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timezone
from contextlib import contextmanager

from loguru import logger

from .eval_errors import (
    get_error_handler, EvaluationError, ValidationError, 
    DatabaseError, FileSystemError, ModelConfigurationError,
    ErrorContext, ErrorCategory, ErrorSeverity
)

class ConcurrentRunManager:
    """Manages concurrent evaluation runs to prevent conflicts."""
    
    def __init__(self):
        self._active_runs: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def register_run(self, run_id: str, task_id: str, model_id: str) -> bool:
        """Register a new run, checking for conflicts."""
        async with self._lock:
            # Check for existing runs with same task/model
            for existing_id, run_info in self._active_runs.items():
                if (run_info['task_id'] == task_id and 
                    run_info['model_id'] == model_id and
                    existing_id != run_id):
                    raise ValidationError(ErrorContext(
                        category=ErrorCategory.VALIDATION,
                        severity=ErrorSeverity.WARNING,
                        message=f"An evaluation is already running for this task and model combination",
                        details=f"Existing run: {existing_id}",
                        suggestion="Wait for the existing run to complete or cancel it",
                        is_retryable=True
                    ))
            
            # Register the new run
            self._active_runs[run_id] = {
                'task_id': task_id,
                'model_id': model_id,
                'start_time': datetime.now(),
                'status': 'running'
            }
            return True
    
    async def unregister_run(self, run_id: str):
        """Remove a run from active tracking."""
        async with self._lock:
            if run_id in self._active_runs:
                del self._active_runs[run_id]
    
    async def get_active_runs(self) -> List[Dict[str, Any]]:
        """Get list of active runs."""
        async with self._lock:
            return [
                {'run_id': run_id, **info}
                for run_id, info in self._active_runs.items()
            ]

class ConfigurationValidator:
    """Validates evaluation configurations before execution."""
    
    @staticmethod
    def validate_task_config(task_config: Dict[str, Any]):
        """Validate task configuration."""
        errors = []
        
        # Required fields
        required_fields = ['name', 'task_type']
        for field in required_fields:
            if field not in task_config or not task_config[field]:
                errors.append(f"Missing required field: {field}")
        
        # Task type validation
        valid_task_types = ['question_answer', 'classification', 'generation', 'logprob']
        if task_config.get('task_type') not in valid_task_types:
            errors.append(f"Invalid task_type: {task_config.get('task_type')}. Must be one of {valid_task_types}")
        
        # Validate generation parameters
        gen_kwargs = task_config.get('generation_kwargs', {})
        if 'temperature' in gen_kwargs:
            temp = gen_kwargs['temperature']
            if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                errors.append(f"Invalid temperature: {temp}. Must be between 0 and 2")
        
        if 'max_tokens' in gen_kwargs:
            max_tokens = gen_kwargs['max_tokens']
            if not isinstance(max_tokens, int) or max_tokens < 1:
                errors.append(f"Invalid max_tokens: {max_tokens}. Must be positive integer")
        
        if errors:
            raise ValidationError(ErrorContext(
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.ERROR,
                message="Task configuration validation failed",
                details="\n".join(errors),
                suggestion="Fix the configuration errors and try again",
                is_retryable=False
            ))
    
    @staticmethod
    def validate_model_config(model_config: Dict[str, Any]):
        """Validate model configuration."""
        errors = []
        
        # Required fields
        if not model_config.get('provider'):
            errors.append("Missing provider name")
        
        if not model_config.get('model_id'):
            errors.append("Missing model ID")
        
        # Provider-specific validation
        provider = model_config.get('provider', '').lower()
        if provider in ['openai', 'anthropic', 'cohere'] and not model_config.get('api_key'):
            # Check for API key in config
            from ...config import load_settings
            settings = load_settings()
            if not settings.get(f'{provider}_api_key'):
                errors.append(f"API key required for {provider}")
        
        if errors:
            raise ModelConfigurationError(ErrorContext(
                category=ErrorCategory.MODEL_CONFIGURATION,
                severity=ErrorSeverity.ERROR,
                message="Model configuration validation failed",
                details="\n".join(errors),
                suggestion="Provide all required configuration parameters",
                is_retryable=False
            ))
    
    @staticmethod
    def validate_run_config(task_config: Dict[str, Any], model_config: Dict[str, Any], 
                          run_config: Dict[str, Any]):
        """Validate run configuration for conflicts."""
        # Check for conflicting parameters
        if run_config.get('max_samples'):
            max_samples = run_config['max_samples']
            if not isinstance(max_samples, int) or max_samples < 1:
                raise ValidationError.invalid_configuration(
                    'max_samples', 
                    'Must be a positive integer'
                )
        
        # Check task/model compatibility
        task_type = task_config.get('task_type')
        provider = model_config.get('provider')
        
        # Some providers don't support certain task types
        incompatible = {
            'logprob': ['anthropic', 'google'],  # These don't support logprobs
        }
        
        if task_type in incompatible and provider in incompatible[task_type]:
            raise ValidationError(ErrorContext(
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.ERROR,
                message=f"Provider '{provider}' does not support task type '{task_type}'",
                suggestion=f"Use a different provider or task type",
                is_retryable=False
            ))

class EnhancedEvaluationOrchestrator:
    """Enhanced orchestrator with better error handling."""
    
    def __init__(self, db_path: str = None, client_id: str = "eval_orchestrator"):
        self.concurrent_manager = ConcurrentRunManager()
        self.error_handler = get_error_handler()
        self._db_path = db_path
        self._client_id = client_id
        self._db = None
        self._init_database()
    
    def _init_database(self):
        """Initialize database with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self._db_path is None:
                    # Use default path
                    from ...config import load_settings
                    settings = load_settings()
                    user_data_dir = Path(settings.get('user_data_dir', '~/.local/share/tldw_cli')).expanduser()
                    self._db_path = user_data_dir / 'default_user' / 'evals.db'
                
                # Ensure directory exists
                Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
                
                # Initialize database
                from ...DB.Evals_DB import EvalsDB
                self._db = EvalsDB(db_path=str(self._db_path), client_id=self._client_id)
                
                logger.info(f"Database initialized successfully: {self._db_path}")
                return
                
            except sqlite3.OperationalError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Database initialization attempt {attempt + 1} failed: {e}")
                    asyncio.sleep(1)  # Brief delay before retry
                else:
                    raise DatabaseError.connection_failed(str(e))
            except Exception as e:
                raise DatabaseError.connection_failed(str(e))
    
    @contextmanager
    def _db_operation(self, operation_name: str):
        """Context manager for database operations with error handling."""
        try:
            yield self._db
        except sqlite3.OperationalError as e:
            if 'locked' in str(e):
                raise DatabaseError(ErrorContext(
                    category=ErrorCategory.DATABASE,
                    severity=ErrorSeverity.WARNING,
                    message=f"Database is locked during {operation_name}",
                    suggestion="Another process may be using the database. Try again in a moment",
                    is_retryable=True
                ))
            else:
                raise DatabaseError.operation_failed(operation_name, str(e))
        except Exception as e:
            raise DatabaseError.operation_failed(operation_name, str(e))
    
    async def run_evaluation_safe(self, 
                                 task_id: str, 
                                 model_id: str, 
                                 run_name: str = None,
                                 max_samples: int = None,
                                 config_overrides: Dict[str, Any] = None,
                                 progress_callback: Callable = None) -> str:
        """Run evaluation with comprehensive error handling."""
        run_id = None
        
        try:
            # Load configurations
            with self._db_operation("loading task configuration"):
                task_data = self._db.get_task(task_id)
                if not task_data:
                    raise ValidationError(ErrorContext(
                        category=ErrorCategory.VALIDATION,
                        severity=ErrorSeverity.ERROR,
                        message=f"Task not found: {task_id}",
                        suggestion="Check the task ID or create the task first",
                        is_retryable=False
                    ))
            
            with self._db_operation("loading model configuration"):
                model_data = self._db.get_model(model_id)
                if not model_data:
                    raise ModelConfigurationError(ErrorContext(
                        category=ErrorCategory.MODEL_CONFIGURATION,
                        severity=ErrorSeverity.ERROR,
                        message=f"Model configuration not found: {model_id}",
                        suggestion="Check the model ID or configure the model first",
                        is_retryable=False
                    ))
            
            # Validate configurations
            task_config = task_data['config_data'].copy()
            if config_overrides:
                task_config.update(config_overrides)
            
            ConfigurationValidator.validate_task_config(task_config)
            ConfigurationValidator.validate_model_config(model_data)
            ConfigurationValidator.validate_run_config(
                task_config, 
                model_data, 
                {'max_samples': max_samples}
            )
            
            # Generate run name if needed
            if not run_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = f"{task_data['name']}_{model_data['name']}_{timestamp}"
            
            # Create run record
            with self._db_operation("creating evaluation run"):
                run_id = self._db.create_run(
                    name=run_name,
                    task_id=task_id,
                    model_id=model_id,
                    config_overrides=config_overrides or {}
                )
            
            # Register with concurrent manager
            await self.concurrent_manager.register_run(run_id, task_id, model_id)
            
            try:
                # Update status
                with self._db_operation("updating run status"):
                    self._db.update_run_status(run_id, 'running')
                
                # Execute evaluation (main logic would go here)
                # ... evaluation execution ...
                
                # Mark as completed
                with self._db_operation("completing evaluation"):
                    self._db.update_run_status(run_id, 'completed')
                
                return run_id
                
            finally:
                # Always unregister the run
                await self.concurrent_manager.unregister_run(run_id)
                
        except EvaluationError:
            # Already properly formatted
            if run_id:
                with self._db_operation("marking run as failed"):
                    self._db.update_run_status(run_id, 'failed', str(e))
            raise
            
        except Exception as e:
            # Unexpected error - wrap it
            error_context = self.error_handler.handle_error(e, {
                'task_id': task_id,
                'model_id': model_id,
                'run_name': run_name
            })
            
            if run_id:
                with self._db_operation("marking run as failed"):
                    self._db.update_run_status(run_id, 'failed', error_context.message)
            
            raise EvaluationError(error_context, e)
    
    def export_results_safe(self, run_id: str, output_path: str, format_type: str = 'json'):
        """Export results with enhanced error handling."""
        try:
            # Validate format
            valid_formats = ['json', 'csv']
            if format_type.lower() not in valid_formats:
                raise ValidationError(ErrorContext(
                    category=ErrorCategory.VALIDATION,
                    severity=ErrorSeverity.ERROR,
                    message=f"Invalid export format: {format_type}",
                    suggestion=f"Use one of: {', '.join(valid_formats)}",
                    is_retryable=False
                ))
            
            # Get run data
            with self._db_operation("loading run summary"):
                run_summary = self.get_run_summary(run_id)
                results = self.get_run_results(run_id)
            
            # Prepare export path
            output_path = Path(output_path)
            
            # Check parent directory
            if not output_path.parent.exists():
                try:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                except PermissionError:
                    raise FileSystemError.permission_denied(str(output_path.parent))
                except OSError as e:
                    if 'No space left' in str(e):
                        raise FileSystemError.disk_full(str(output_path.parent))
                    raise
            
            # Check disk space (rough estimate)
            try:
                import shutil
                free_space = shutil.disk_usage(output_path.parent).free
                estimated_size = len(json.dumps(results)) * 2  # Conservative estimate
                
                if free_space < estimated_size:
                    raise FileSystemError.disk_full(str(output_path.parent))
            except:
                pass  # Ignore if we can't check
            
            # Export based on format
            try:
                if format_type.lower() == 'json':
                    self._export_json_safe(run_summary, results, output_path)
                else:  # csv
                    self._export_csv_safe(results, output_path)
                    
                logger.info(f"Results exported successfully to: {output_path}")
                
            except PermissionError:
                raise FileSystemError.permission_denied(str(output_path))
            except OSError as e:
                if 'No space left' in str(e):
                    raise FileSystemError.disk_full(str(output_path))
                raise FileSystemError(ErrorContext(
                    category=ErrorCategory.FILE_SYSTEM,
                    severity=ErrorSeverity.ERROR,
                    message=f"Failed to write export file",
                    details=str(e),
                    suggestion="Check file permissions and disk space",
                    is_retryable=True
                ))
                
        except EvaluationError:
            raise
        except Exception as e:
            error_context = self.error_handler.handle_error(e, {
                'run_id': run_id,
                'output_path': output_path,
                'format_type': format_type
            })
            raise EvaluationError(error_context, e)
    
    def _export_json_safe(self, run_summary: Dict[str, Any], 
                         results: List[Dict[str, Any]], output_path: Path):
        """Export to JSON with proper error handling."""
        export_data = {
            'run_summary': run_summary,
            'results': results,
            'export_timestamp': datetime.now(timezone.utc).isoformat(),
            'error_summary': self.error_handler.get_error_summary()
        }
        
        # Write to temporary file first
        temp_path = output_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            # Atomic rename
            temp_path.replace(output_path)
            
        finally:
            # Clean up temp file if it exists
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass
    
    def _export_csv_safe(self, results: List[Dict[str, Any]], output_path: Path):
        """Export to CSV with proper error handling."""
        import csv
        
        if not results:
            raise ValidationError(ErrorContext(
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.WARNING,
                message="No results to export",
                suggestion="Run the evaluation first to generate results",
                is_retryable=False
            ))
        
        # Prepare fieldnames
        fieldnames = ['sample_id', 'input_text', 'expected_output', 'actual_output']
        if results[0].get('metrics'):
            fieldnames.extend(results[0]['metrics'].keys())
        
        # Add error info fields
        fieldnames.extend(['has_error', 'error_category', 'error_message'])
        
        # Write to temporary file first
        temp_path = output_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                
                for result in results:
                    row = {
                        'sample_id': result['sample_id'],
                        'input_text': result.get('input_data', {}).get('input', ''),
                        'expected_output': result.get('expected_output', ''),
                        'actual_output': result.get('actual_output', ''),
                        'has_error': 'error' in result.get('metrics', {}),
                        'error_category': result.get('metadata', {}).get('error_info', {}).get('category', ''),
                        'error_message': result.get('metadata', {}).get('error_info', {}).get('message', '')
                    }
                    
                    # Add metrics
                    if result.get('metrics'):
                        row.update(result['metrics'])
                    
                    writer.writerow(row)
            
            # Atomic rename
            temp_path.replace(output_path)
            
        finally:
            # Clean up temp file
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass