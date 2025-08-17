# config_loader.py
# Description: Configuration loader for the evaluation module
#
"""
Configuration Loader
--------------------

Loads and manages configuration from YAML files for the evaluation system.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger


class EvalConfigLoader:
    """Loads and manages evaluation configuration."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Optional path to configuration file
        """
        if config_path is None:
            # Default to config directory relative to this file
            config_dir = Path(__file__).parent / "config"
            config_path = config_dir / "eval_config.yaml"
        
        self.config_path = Path(config_path)
        self._config = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Configuration file not found: {self.config_path}")
            self._config = self._get_default_config()
            return
        
        try:
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file not found."""
        return {
            'task_types': ['question_answer', 'generation', 'classification', 'logprob'],
            'metrics': {
                'question_answer': ['exact_match', 'f1', 'contains', 'accuracy'],
                'generation': ['bleu', 'rouge', 'perplexity', 'coherence'],
                'classification': ['accuracy', 'f1', 'precision', 'recall'],
                'logprob': ['perplexity', 'log_likelihood', 'accuracy']
            },
            'required_fields': {
                'task': ['name', 'task_type'],
                'model': ['provider', 'model_id'],
                'run': ['task_id', 'model_id']
            },
            'error_handling': {
                'max_retries': 3,
                'retry_delay_seconds': 1.0,
                'exponential_backoff': True,
                'max_delay_seconds': 60.0
            },
            'budget': {
                'warning_threshold': 0.8,
                'default_limit': 10.0
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        if self._config is None:
            return default
        
        # Support dot notation for nested keys
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_task_types(self) -> List[str]:
        """Get list of valid task types."""
        return self.get('task_types', ['question_answer', 'generation', 'classification', 'logprob'])
    
    def get_metrics_for_task(self, task_type: str) -> List[str]:
        """
        Get valid metrics for a task type.
        
        Args:
            task_type: Type of task
            
        Returns:
            List of valid metric names
        """
        metrics = self.get('metrics', {})
        return metrics.get(task_type, ['accuracy'])
    
    def get_required_fields(self, config_type: str) -> List[str]:
        """
        Get required fields for a configuration type.
        
        Args:
            config_type: Type of configuration ('task', 'model', 'run')
            
        Returns:
            List of required field names
        """
        required = self.get('required_fields', {})
        return required.get(config_type, [])
    
    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """
        Get configuration for a specific provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Provider configuration dictionary
        """
        providers = self.get('providers', {})
        return providers.get(provider, {})
    
    def get_error_config(self) -> Dict[str, Any]:
        """Get error handling configuration."""
        return self.get('error_handling', {
            'max_retries': 3,
            'retry_delay_seconds': 1.0,
            'exponential_backoff': True,
            'max_delay_seconds': 60.0
        })
    
    def get_budget_config(self) -> Dict[str, Any]:
        """Get budget monitoring configuration."""
        return self.get('budget', {
            'warning_threshold': 0.8,
            'default_limit': 10.0
        })
    
    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation configuration."""
        return self.get('validation', {
            'max_dataset_size_mb': 1000,
            'max_samples_per_run': 10000,
            'max_concurrent_runs': 5,
            'min_samples_for_statistics': 30,
            'confidence_level': 0.95
        })
    
    def is_feature_enabled(self, feature: str) -> bool:
        """
        Check if a feature is enabled.
        
        Args:
            feature: Feature name
            
        Returns:
            True if feature is enabled
        """
        features = self.get('features', {})
        return features.get(feature, False)
    
    def reload(self):
        """Reload configuration from file."""
        self._load_config()
    
    def update(self, updates: Dict[str, Any]):
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply
        """
        if self._config is None:
            self._config = {}
        
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self._config = deep_update(self._config, updates)
    
    def save(self, path: Optional[str] = None):
        """
        Save configuration to file.
        
        Args:
            path: Optional path to save to (defaults to original path)
        """
        save_path = Path(path) if path else self.config_path
        
        try:
            # Ensure directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Saved configuration to {save_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")


# Global configuration instance
_config_loader = None


def get_eval_config() -> EvalConfigLoader:
    """Get or create the global configuration loader."""
    global _config_loader
    if _config_loader is None:
        _config_loader = EvalConfigLoader()
    return _config_loader


def reload_config():
    """Reload the global configuration."""
    global _config_loader
    if _config_loader:
        _config_loader.reload()