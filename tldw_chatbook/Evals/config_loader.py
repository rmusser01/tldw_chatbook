# config_loader.py
# Description: Configuration loader for the evaluation module
#
"""
Configuration Loader
--------------------

Loads and manages configuration from YAML files for the evaluation system.
"""

import copy
from pathlib import Path
from typing import Dict, Any, List, Optional
from weakref import WeakValueDictionary

from loguru import logger
import yaml

from tldw_chatbook.Backup_Recovery import raw_participants as raw


def _merge(base, updates):
    from tldw_chatbook.config import deep_merge_dicts

    return deep_merge_dicts(base, copy.deepcopy(updates))


def _capture_draft(overrides, expected, draft):
    """Capture mutable getter edits without pinning unchanged inherited values."""
    result = copy.deepcopy(overrides)
    for key in expected.keys() - draft.keys():
        result.pop(key, None)
    for key, value in draft.items():
        if key in expected and value == expected[key]:
            continue
        if isinstance(value, dict) and isinstance(expected.get(key), dict):
            nested = _capture_draft(result.get(key, {}), expected[key], value)
            if nested:
                result[key] = nested
            else:
                result.pop(key, None)
        else:
            result[key] = copy.deepcopy(value)
    return result


class EvalConfigLoader:
    """Loads and manages evaluation configuration."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.

        Args:
            config_path: Optional full configuration file. Omission selects
                shipped defaults with private profile overrides.
        """
        self._uses_profile_overrides = config_path is None
        if self._uses_profile_overrides:
            from . import _override_config_path

            config_path = _override_config_path()

        self.config_path = Path(config_path)
        self._config = None
        self._persisted_config = None
        self._defaults = {}
        self._overrides = {}
        self._persisted_overrides = {}
        self.persistence_error = None
        self._load_config()

    def _load_config(self):
        """Load configuration from YAML file."""
        previous_config = self._config
        previous_persisted = self._persisted_config
        defaults = None
        try:
            if self._uses_profile_overrides:
                from . import _default_config_path

                # Distribution assets are read-only resources, not private state.
                with _default_config_path().open(encoding="utf-8") as stream:
                    defaults = yaml.safe_load(stream)
                if not isinstance(defaults, dict):
                    raise ValueError("eval_defaults_not_mapping")
            with raw._scope(self, "eval_config") as operation:
                selected = raw._selected(operation)
                if not selected.exists():
                    if not self._uses_profile_overrides:
                        logger.warning(f"Configuration file not found: {selected}")
                        self._config = self._get_default_config()
                        self._persisted_config = None
                        return
                    loaded = {}
                else:
                    with raw._file(operation, selected, "r") as f:
                        loaded = yaml.safe_load(f)
                    if self._uses_profile_overrides and loaded is None:
                        loaded = {}
                if self._uses_profile_overrides:
                    if not isinstance(loaded, dict):
                        raise ValueError("eval_overrides_not_mapping")
                    effective = _merge(defaults, loaded)
                else:
                    effective = loaded
                # Commit baselines only after the private operation closes cleanly.
            self._config = effective
            self._persisted_config = copy.deepcopy(effective)
            if self._uses_profile_overrides:
                self._defaults = defaults
                self._overrides = copy.deepcopy(loaded)
                self._persisted_overrides = copy.deepcopy(loaded)
            self.persistence_error = None
            logger.debug(f"Loaded evaluation configuration from {selected}")
        except Exception as e:
            self._config = previous_config
            self._persisted_config = previous_persisted
            self.persistence_error = "eval_load_failed"
            logger.error(f"Error loading configuration: {e}")
            # Refusal never discards an existing mutable draft.
            if self._config is None:
                if isinstance(defaults, dict):
                    self._defaults = defaults
                    self._config = copy.deepcopy(defaults)
                else:
                    self._config = self._get_default_config()

    def persistence_safe_point(self):
        """Report the actual mutable draft, including edits through get()."""
        if self._config != self._persisted_config or (
            self._uses_profile_overrides
            and self._overrides != self._persisted_overrides
        ):
            return "needs_user_save_or_discard"
        return "persistence_failed" if self.persistence_error else "ready"

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file not found."""
        return {
            "task_types": [
                "question_answer",
                "generation",
                "classification",
                "logprob",
            ],
            "metrics": {
                "question_answer": ["exact_match", "f1", "contains", "accuracy"],
                "generation": ["bleu", "rouge", "perplexity", "coherence"],
                "classification": ["accuracy", "f1", "precision", "recall"],
                "logprob": ["perplexity", "log_likelihood", "accuracy"],
            },
            "required_fields": {
                "task": ["name", "task_type"],
                "model": ["provider", "model_id"],
                "run": ["task_id", "model_id"],
            },
            "error_handling": {
                "max_retries": 3,
                "retry_delay_seconds": 1.0,
                "exponential_backoff": True,
                "max_delay_seconds": 60.0,
            },
            "budget": {"warning_threshold": 0.8, "default_limit": 10.0},
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
        keys = key.split(".")
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
        return self.get(
            "task_types", ["question_answer", "generation", "classification", "logprob"]
        )

    def get_metrics_for_task(self, task_type: str) -> List[str]:
        """
        Get valid metrics for a task type.

        Args:
            task_type: Type of task

        Returns:
            List of valid metric names
        """
        metrics = self.get("metrics", {})
        return metrics.get(task_type, ["accuracy"])

    def get_required_fields(self, config_type: str) -> List[str]:
        """
        Get required fields for a configuration type.

        Args:
            config_type: Type of configuration ('task', 'model', 'run')

        Returns:
            List of required field names
        """
        required = self.get("required_fields", {})
        return required.get(config_type, [])

    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """
        Get configuration for a specific provider.

        Args:
            provider: Provider name

        Returns:
            Provider configuration dictionary
        """
        providers = self.get("providers", {})
        return providers.get(provider, {})

    def get_error_config(self) -> Dict[str, Any]:
        """Get error handling configuration."""
        return self.get(
            "error_handling",
            {
                "max_retries": 3,
                "retry_delay_seconds": 1.0,
                "exponential_backoff": True,
                "max_delay_seconds": 60.0,
            },
        )

    def get_budget_config(self) -> Dict[str, Any]:
        """Get budget monitoring configuration."""
        return self.get("budget", {"warning_threshold": 0.8, "default_limit": 10.0})

    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation configuration."""
        return self.get(
            "validation",
            {
                "max_dataset_size_mb": 1000,
                "max_samples_per_run": 10000,
                "max_concurrent_runs": 5,
                "min_samples_for_statistics": 30,
                "confidence_level": 0.95,
            },
        )

    def is_feature_enabled(self, feature: str) -> bool:
        """
        Check if a feature is enabled.

        Args:
            feature: Feature name

        Returns:
            True if feature is enabled
        """
        features = self.get("features", {})
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

        if self._uses_profile_overrides:
            self._overrides = _capture_draft(
                self._overrides, _merge(self._defaults, self._overrides), self._config
            )
            self._overrides = _merge(self._overrides, updates)
        self._config = _merge(self._config, updates)

    def save(self, path: Optional[str] = None):
        """
        Save configuration to file.

        Args:
            path: Optional full-config export destination. Saving to the
                profile override path writes only explicit overrides.
        """
        save_path = Path(path) if path else self.config_path
        previous_persisted = self._persisted_config
        try:
            with raw._scope(
                self, "eval_config", writing=True, selected_read=save_path
            ) as operation:
                selected = raw._selected(operation)
                primary = selected == raw.lexical_path(self.config_path)
                sparse = self._uses_profile_overrides and primary
                overrides = (
                    _capture_draft(
                        self._overrides,
                        _merge(self._defaults, self._overrides),
                        self._config,
                    )
                    if sparse
                    else None
                )
                snapshot = copy.deepcopy(overrides if sparse else self._config)
                raw._mkdirs(operation)
                temporary = selected.with_suffix(selected.suffix + ".tmp")
                try:
                    with raw._file(operation, temporary, "w") as f:
                        yaml.safe_dump(
                            snapshot, f, default_flow_style=False, sort_keys=False
                        )
                    raw._replace(operation, temporary, selected)
                finally:
                    raw._remove_temporary(operation, temporary)
            if primary:
                if sparse:
                    self._overrides = overrides
                    self._persisted_overrides = copy.deepcopy(overrides)
                    self._config = _merge(self._defaults, overrides)
                self._persisted_config = copy.deepcopy(self._config)
                self.persistence_error = None
            logger.info(f"Saved configuration to {selected}")
        except Exception as e:
            self._persisted_config = previous_persisted
            self.persistence_error = "eval_save_failed"
            logger.error(f"Error saving configuration: {e}")


# Global configuration instance
_config_loader = None
_profile_loaders = WeakValueDictionary()


def get_eval_config() -> EvalConfigLoader:
    """Get or create the global configuration loader."""
    global _config_loader
    from . import _override_config_path

    selected = _override_config_path()
    if _config_loader is None or _config_loader.config_path != selected:
        _config_loader = _profile_loaders.get(selected)
        if _config_loader is None:
            _config_loader = EvalConfigLoader()
            _profile_loaders[selected] = _config_loader
    return _config_loader


def reload_config():
    """Reload the global configuration."""
    get_eval_config().reload()
