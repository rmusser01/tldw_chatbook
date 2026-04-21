# configuration_validator.py
# Description: Validates evaluation configurations before execution
#
"""
Configuration Validator for Evaluations
---------------------------------------

Validates task, model, and run configurations to ensure
they meet requirements before evaluation execution.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
from loguru import logger

from .eval_errors import ValidationError, ErrorContext, ErrorCategory, ErrorSeverity
from tldw_chatbook.Utils.path_validation import validate_path_simple
from .config_loader import get_eval_config


class ConfigurationValidator:
    """Validates evaluation configurations before execution."""
    
    def __init__(self):
        """Initialize validator with configuration."""
        self.config = get_eval_config()
        
        # Load configuration values
        self.VALID_TASK_TYPES = set(self.config.get_task_types())
        self.VALID_METRICS = {}
        for task_type in self.VALID_TASK_TYPES:
            self.VALID_METRICS[task_type] = set(self.config.get_metrics_for_task(task_type))
        
        self.REQUIRED_FIELDS = {
            'task': self.config.get_required_fields('task'),
            'model': self.config.get_required_fields('model'),
            'run': self.config.get_required_fields('run')
        }
    
    def validate_task_config(self, task_config: Dict[str, Any]) -> List[str]:
        """
        Validate task configuration.
        
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        # Check required fields
        for field in self.REQUIRED_FIELDS['task']:
            if field not in task_config or not task_config[field]:
                errors.append(f"Missing required field: {field}")
        
        # Validate task type
        if 'task_type' in task_config:
            if task_config['task_type'] not in self.VALID_TASK_TYPES:
                errors.append(
                    f"Invalid task_type: {task_config['task_type']}. "
                    f"Must be one of: {', '.join(self.VALID_TASK_TYPES)}"
                )
        
        # Validate metric if specified
        if 'metric' in task_config and 'task_type' in task_config:
            task_type = task_config['task_type']
            if task_type in self.VALID_METRICS:
                valid_metrics = self.VALID_METRICS[task_type]
                if task_config['metric'] not in valid_metrics:
                    errors.append(
                        f"Invalid metric '{task_config['metric']}' for task type '{task_type}'. "
                        f"Valid metrics: {', '.join(valid_metrics)}"
                    )
        
        # Validate dataset path if specified
        if 'dataset_name' in task_config:
            dataset_path = Path(task_config['dataset_name'])
            if dataset_path.exists() and dataset_path.is_file():
                try:
                    validate_path_simple(str(dataset_path))
                except Exception as e:
                    errors.append(f"Invalid dataset path: {e}")
        
        # Validate generation kwargs
        if 'generation_kwargs' in task_config:
            gen_kwargs = task_config['generation_kwargs']
            if not isinstance(gen_kwargs, dict):
                errors.append("generation_kwargs must be a dictionary")
            else:
                # Validate temperature
                if 'temperature' in gen_kwargs:
                    temp = gen_kwargs['temperature']
                    if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                        errors.append("temperature must be between 0 and 2")
                
                # Validate max_tokens
                if 'max_tokens' in gen_kwargs:
                    max_tokens = gen_kwargs['max_tokens']
                    if not isinstance(max_tokens, int) or max_tokens < 1:
                        errors.append("max_tokens must be a positive integer")
        
        return errors
    
    def validate_model_config(self, model_config: Dict[str, Any]) -> List[str]:
        """
        Validate model configuration.
        
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        # Check required fields
        for field in self.REQUIRED_FIELDS['model']:
            if field not in model_config or not model_config[field]:
                errors.append(f"Missing required field: {field}")
        
        # Validate provider
        if 'provider' in model_config:
            provider = model_config['provider'].lower()
            # Could add list of valid providers here
        
        # Validate API key if required (for non-local providers)
        if 'provider' in model_config:
            provider = model_config['provider'].lower()
            local_providers = {'ollama', 'vllm', 'llama', 'kobold', 'tabbyapi', 'local-llm'}
            if provider not in local_providers and 'api_key' not in model_config:
                errors.append(f"API key required for provider: {provider}")
        
        return errors
    
    def validate_run_config(self, run_config: Dict[str, Any]) -> List[str]:
        """
        Validate run configuration.
        
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        # Check required fields
        for field in self.REQUIRED_FIELDS['run']:
            if field not in run_config or not run_config[field]:
                errors.append(f"Missing required field: {field}")
        
        # Validate max_samples if specified
        if 'max_samples' in run_config:
            max_samples = run_config['max_samples']
            if not isinstance(max_samples, int) or max_samples < 1:
                errors.append("max_samples must be a positive integer")
        
        # Validate run name length if specified
        if 'name' in run_config:
            if len(run_config['name']) > 200:
                errors.append("Run name must be less than 200 characters")
        
        return errors
    
    @classmethod
    def raise_if_invalid(cls, config: Dict[str, Any], config_type: str):
        """
        Validate configuration and raise if invalid.
        
        Args:
            config: Configuration to validate
            config_type: Type of configuration ('task', 'model', or 'run')
        
        Raises:
            ValidationError: If configuration is invalid
        """
        validators = {
            'task': cls.validate_task_config,
            'model': cls.validate_model_config,
            'run': cls.validate_run_config
        }
        
        if config_type not in validators:
            raise ValueError(f"Unknown config type: {config_type}")
        
        errors = validators[config_type](config)
        
        if errors:
            raise ValidationError(ErrorContext(
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.ERROR,
                message=f"Invalid {config_type} configuration",
                details="\n".join(errors),
                suggestion="Fix the validation errors and try again",
                is_retryable=False
            ))