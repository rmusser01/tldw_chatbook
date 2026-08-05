# eval_orchestrator.py
# Description: Orchestrates evaluation runs and manages results storage
#
"""
Evaluation Orchestrator
----------------------

Coordinates the complete evaluation pipeline:
1. Task and model configuration validation
2. Evaluation execution with progress tracking
3. Results storage and aggregation
4. Error handling and recovery

This is the main entry point for running evaluations.
"""

import asyncio
import time
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Callable, Tuple
from pathlib import Path
from contextlib import contextmanager

from loguru import logger

from .task_loader import TaskLoader, TaskConfig
from .eval_runner import EvalRunner, EvalSampleResult, _invoke_callback

# Use existing infrastructure
from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Metrics.metrics_logger import log_counter, log_histogram
from tldw_chatbook.Utils.path_validation import validate_path_simple
from tldw_chatbook.Utils.private_paths import (
    lexical_path,
    verify_trusted_directory,
)
from .eval_errors import (
    get_error_handler,
    EvaluationError,
    ValidationError,
    DatabaseError,
    FileSystemError,
    ModelConfigurationError,
    ErrorContext,
    ErrorCategory,
    ErrorSeverity,
)

# Import refactored components
from .concurrency_manager import ConcurrentRunManager
from .configuration_validator import ConfigurationValidator


class EvaluationOrchestrator:
    """
    Orchestrates evaluation runs from start to finish.

    Single Responsibility: Coordinates the evaluation workflow by delegating
    specific tasks to specialized components.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        client_id: str = "eval_orchestrator",
    ):
        """
        Initialize the evaluation orchestrator.

        Args:
            db_path: Path to evaluation database (defaults to user data directory)
            client_id: Client identifier for audit trail
        """
        # Initialize specialized components (composition over inheritance)
        self.concurrent_manager = ConcurrentRunManager()
        self.validator = ConfigurationValidator()
        self.error_handler = get_error_handler()
        self._client_id = client_id

        # Initialize active tasks tracking for cancellation
        self._active_tasks = {}

        # Initialize database connection
        self.db = self._initialize_database(db_path, client_id)

        # Initialize task loader
        self.task_loader = TaskLoader()

    def _initialize_database(
        self,
        db_path: str | Path | None,
        client_id: str,
    ) -> EvalsDB:
        """Initialize database connection with proper path."""
        if db_path is None:
            # Resolve the default path through the same shared, profile-aware
            # mechanism every other DB in the app uses. Do NOT re-derive the
            # profile / data-root from settings.get(...) with guessed keys --
            # load_settings() publishes the profile name as "USERS_NAME"
            # (not "user_id"/"username"), and it does not publish
            # "user_data_dir" at all, so both lookups used to silently miss.
            #
            # get_evals_db_path() is the single source of truth for this path
            # (TASK-899) -- other consumers (e.g. the Settings
            # database-maintenance panel) resolve the Evals DB through it too,
            # so they cannot drift from what the orchestrator actually opens.
            from tldw_chatbook.config import get_evals_db_path

            selected = get_evals_db_path()

            self._warn_if_legacy_data_exists(selected)
        elif db_path == ":memory:":
            selected = ":memory:"
        else:
            selected = lexical_path(db_path)
            verify_trusted_directory(
                selected.parent,
                allow_shared_sticky=False,
            )

        return EvalsDB(db_path=str(selected), client_id=client_id)

    @staticmethod
    def _safe_validate_existence_check_path(
        path: Path, purpose: str
    ) -> Optional[Path]:
        """Validate a path through validate_path_simple before an existence check.

        This exists solely to decide whether to emit an informational warning
        about legacy Evals data -- it must never turn into a startup failure.
        validate_path_simple can raise on paths it has no business rejecting
        here (see TASK-838: its raw-string scan for parent-directory patterns
        runs before normalization and over-rejects some legitimate Windows
        paths, e.g. drive-relative paths containing "..\\" style segments).
        If validation raises for any reason, log why at debug level and
        return None so the caller degrades to "skip the warning" instead of
        propagating the error into _initialize_database.
        """
        try:
            return validate_path_simple(path)
        except Exception as e:
            logger.debug(
                "Skipping legacy-Evals-data check for {} path {}: {}",
                purpose,
                path,
                e,
            )
            return None

    @staticmethod
    def _warn_if_legacy_data_exists(resolved_path: Path) -> None:
        """Warn once if the profile-resolved evals.db is missing but legacy data exists.

        Before this fix, EVERY profile shared a single Evals database at the
        literal, hardcoded path "~/.local/share/tldw_cli/default_user/evals.db"
        -- not "<configured_data_root>/default_user/evals.db". The buggy code
        read `settings.get("user_data_dir", "~/.local/share/tldw_cli")`, and
        `load_settings()` never publishes a "user_data_dir" key, so that
        fallback literal always won even for users with a custom
        `[paths] data_dir`. The legacy location this function checks must
        therefore stay hardcoded too, or it would miss exactly the users most
        likely to be surprised (those who already customized their data root).

        If the newly-resolved, profile-specific path has no data yet but that
        legacy file does, the user's existing benches/datasets/runs are not
        lost -- they are simply still sitting at the old, hardcoded path.
        Surface that clearly instead of silently starting a brand-new, empty
        database.

        This function never copies, moves, or deletes anything.
        """
        legacy_path = (
            Path("~/.local/share/tldw_cli").expanduser() / "default_user" / "evals.db"
        )

        validated_resolved_path = EvaluationOrchestrator._safe_validate_existence_check_path(
            resolved_path, "resolved Evals database"
        )
        if validated_resolved_path is None:
            # Validation failed (e.g. TASK-838 over-rejection). This function
            # only ever emits an informational warning, so degrade to
            # "nothing to warn about" rather than risk breaking Evals
            # startup over a path we can't safely check.
            return
        resolved_path = validated_resolved_path

        validated_legacy_path = EvaluationOrchestrator._safe_validate_existence_check_path(
            legacy_path, "legacy Evals database"
        )
        if validated_legacy_path is None:
            return
        legacy_path = validated_legacy_path

        if resolved_path == legacy_path:
            # The configured profile IS "default_user" (with no custom data
            # root) -- this IS the legacy location, nothing to warn about.
            return

        if resolved_path.exists():
            # This profile already has its own data; nothing to warn about.
            return

        if legacy_path.exists():
            logger.warning(
                "No Evals database found for the current profile at {}. "
                "Older versions of this app ignored the configured profile and "
                "always wrote Evals data under the 'default_user' profile -- "
                "your existing benches, datasets and runs are still there: {}. "
                "Nothing has been moved or copied automatically; if you want "
                "that data under this profile, copy the file there yourself.",
                resolved_path,
                legacy_path,
            )

    @contextmanager
    def _db_operation(self, operation_name: str):
        """Context manager for database operations with error handling."""
        try:
            yield self.db
        except sqlite3.OperationalError as e:
            if "locked" in str(e):
                raise DatabaseError(
                    ErrorContext(
                        category=ErrorCategory.DATABASE,
                        severity=ErrorSeverity.WARNING,
                        message=f"Database is locked during {operation_name}",
                        suggestion="Another process may be using the database. Try again in a moment",
                        is_retryable=True,
                    )
                )
            else:
                raise DatabaseError(
                    ErrorContext(
                        category=ErrorCategory.DATABASE,
                        severity=ErrorSeverity.ERROR,
                        message=f"Database operation failed: {operation_name}",
                        details=str(e),
                        is_retryable=False,
                    )
                )
        except Exception as e:
            raise DatabaseError(
                ErrorContext(
                    category=ErrorCategory.DATABASE,
                    severity=ErrorSeverity.ERROR,
                    message=f"Database operation failed: {operation_name}",
                    details=str(e),
                    is_retryable=False,
                )
            )

    async def create_task_from_file(
        self, task_file_path: str, format_type: str = "auto"
    ) -> str:
        """
        Create a task from a file and store it in the database.

        Args:
            task_file_path: Path to task configuration file
            format_type: Format type ('eleuther', 'custom', 'auto')

        Returns:
            Task ID
        """
        logger.info(f"Creating task from file: {task_file_path}")
        start_time = time.time()

        log_counter(
            "eval_task_creation_attempt",
            labels={"format_type": format_type, "source": "file"},
        )

        try:
            # Validate and normalize the file path
            validated_path = validate_path_simple(task_file_path)
            if not Path(validated_path).exists():
                raise FileSystemError(
                    ErrorContext(
                        category=ErrorCategory.FILE_SYSTEM,
                        severity=ErrorSeverity.ERROR,
                        message=f"Task file not found: {task_file_path}",
                        suggestion="Check the file path and ensure the file exists",
                        is_retryable=False,
                    )
                )

            # Load task configuration
            task_config = self.task_loader.load_task(validated_path, format_type)

            # Validate task
            validation_issues = self.task_loader.validate_task(task_config)
            if validation_issues:
                log_counter(
                    "eval_task_validation_failed",
                    labels={
                        "format_type": format_type,
                        "task_type": task_config.task_type
                        if task_config
                        else "unknown",
                    },
                )
                raise ValueError(f"Task validation failed: {validation_issues}")

            # Store in database
            task_id = self.db.create_task(
                name=task_config.name,
                description=task_config.description,
                task_type=task_config.task_type,
                config_format=task_config.metadata.get("format", "custom"),
                config_data=task_config.__dict__,
                dataset_id=None,  # Will be handled separately if needed
            )

            duration = time.time() - start_time
            log_histogram(
                "eval_task_creation_duration",
                duration,
                labels={"format_type": format_type, "task_type": task_config.task_type},
            )
            log_counter(
                "eval_task_creation_success",
                labels={"format_type": format_type, "task_type": task_config.task_type},
            )

            logger.info(f"Created task: {task_config.name} ({task_id})")
            return task_id

        except Exception as e:
            duration = time.time() - start_time
            log_counter(
                "eval_task_creation_error",
                labels={"format_type": format_type, "error_type": type(e).__name__},
            )
            log_histogram(
                "eval_task_creation_error_duration",
                duration,
                labels={"format_type": format_type},
            )
            raise

    def create_model_config(
        self, name: str, provider: str, model_id: str, config: Dict[str, Any] = None
    ) -> str:
        """
        Create a model configuration and store it in the database.

        Args:
            name: Human-readable name for the model
            provider: Provider name (openai, anthropic, etc.)
            model_id: Model identifier
            config: Additional configuration parameters

        Returns:
            Model configuration ID
        """
        logger.info(f"Creating model config: {name} ({provider}/{model_id})")

        # Validate model configuration
        model_data = {
            "name": name,
            "provider": provider,
            "model_id": model_id,
            "config": config or {},
            "api_key": (config or {}).get("api_key"),
        }
        self.validator.raise_if_invalid(model_data, "model")

        with self._db_operation("creating model configuration"):
            model_config_id = self.db.create_model(
                name=name, provider=provider, model_id=model_id, config=config or {}
            )

        logger.info(f"Created model config: {name} ({model_config_id})")
        return model_config_id

    def _create_model_config(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create model configuration from database data."""
        return {
            "provider": model_data.get("provider", ""),
            "model_id": model_data.get("model_id", ""),
            "api_key": model_data.get("config", {}).get("api_key"),
            **model_data.get("config", {}),
        }

    async def run_evaluation(
        self,
        task_id: str,
        model_id: str,
        run_name: str = None,
        max_samples: int = None,
        config_overrides: Dict[str, Any] = None,
        progress_callback: Callable = None,
        run_started_callback: Callable = None,
    ) -> str:
        """
        Run a complete evaluation.

        Args:
            task_id: ID of the task to run
            model_id: ID of the model configuration to use
            run_name: Name for this evaluation run
            max_samples: Maximum number of samples to evaluate (None for all)
            config_overrides: Override task configuration parameters
            progress_callback: Function to call with progress updates
            run_started_callback: Sync or async callback receiving the durable run ID

        Returns:
            Evaluation run ID
        """
        run_id = None

        try:
            # Load task and model configurations with error handling
            with self._db_operation("loading task configuration"):
                task_data = self.db.get_task(task_id)
                if not task_data:
                    raise ValidationError(
                        ErrorContext(
                            category=ErrorCategory.VALIDATION,
                            severity=ErrorSeverity.ERROR,
                            message=f"Task not found: {task_id}",
                            suggestion="Check the task ID or create the task first",
                            is_retryable=False,
                        )
                    )

            with self._db_operation("loading model configuration"):
                model_data = self.db.get_model(model_id)
                if not model_data:
                    raise ModelConfigurationError(
                        ErrorContext(
                            category=ErrorCategory.MODEL_CONFIGURATION,
                            severity=ErrorSeverity.ERROR,
                            message=f"Model configuration not found: {model_id}",
                            suggestion="Check the model ID or configure the model first",
                            is_retryable=False,
                        )
                    )

            # Create TaskConfig from database data
            # Merge database fields with config_data to ensure all required fields are present
            task_config_dict = {
                "name": task_data["name"],
                "description": task_data.get("description", ""),
                "task_type": task_data.get("task_type", "question_answer"),
                "dataset_name": task_data.get("dataset_name", "custom"),
            }
            # Update with any additional config from config_data
            if task_data.get("config_data"):
                task_config_dict.update(task_data["config_data"])

            # Store model-specific overrides separately
            model_overrides = {}
            if config_overrides:
                # Separate model params from task params
                model_params = ["temperature", "max_tokens", "top_p", "top_k"]
                for key in list(config_overrides.keys()):
                    if key in model_params:
                        model_overrides[key] = config_overrides.pop(key)
                # Update task config with remaining overrides
                task_config_dict.update(config_overrides)

            # Validate configurations
            self.validator.raise_if_invalid(task_config_dict, "task")
            self.validator.raise_if_invalid(model_data, "model")

            # Filter task_config_dict to only include TaskConfig fields
            from inspect import signature

            task_config_fields = set(signature(TaskConfig).parameters.keys())
            filtered_task_config = {
                k: v for k, v in task_config_dict.items() if k in task_config_fields
            }

            task_config = TaskConfig(**filtered_task_config)

            # Generate run name if not provided
            if not run_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = f"{task_data['name']}_{model_data['name']}_{timestamp}"

            logger.info(f"Starting evaluation run: {run_name}")
            eval_start_time = time.time()

            # Log evaluation start metrics
            log_counter(
                "eval_run_started",
                labels={
                    "task_type": task_config.task_type,
                    "provider": model_data["provider"],
                    "model": model_data["model_id"],
                },
            )

            # Create evaluation run record
            with self._db_operation("creating evaluation run"):
                run_id = self.db.create_run(
                    name=run_name,
                    task_id=task_id,
                    model_id=model_id,
                    config_overrides=config_overrides or {},
                )

            # Register with concurrent manager
            await self.concurrent_manager.register_run(run_id, task_id, model_id)
            # Update run status to running
            self.db.update_run_status(run_id, "running")

            owner_task = asyncio.current_task()
            if owner_task is None:
                raise RuntimeError("Evaluation must run inside an asyncio task")
            self._active_tasks[run_id] = owner_task
            await _invoke_callback(run_started_callback, run_id)

            # Create evaluation runner with model config
            model_config = {
                "provider": model_data["provider"],
                "model_id": model_data["model_id"],
                **model_data.get("config", {}),
                **model_overrides,  # Include the temperature, max_tokens, etc.
            }
            eval_runner = EvalRunner(task_config, model_config)

            # Create progress callback wrapper
            async def progress_wrapper(
                completed: int, total: int, result: EvalSampleResult
            ):
                # Store individual result
                self.db.store_result(
                    run_id=run_id,
                    sample_id=result.sample_id,
                    input_data={"input": result.input_text},
                    actual_output=result.actual_output,
                    expected_output=result.expected_output,
                    logprobs=result.logprobs,
                    metrics=result.metrics,
                    metadata=result.metadata,
                )

                # Log sample metrics
                log_counter(
                    "eval_sample_processed",
                    labels={
                        "task_type": task_config.task_type,
                        "provider": model_data["provider"],
                        "model": model_data["model_id"],
                        "success": "true"
                        if not result.error_info and "error" not in result.metrics
                        else "false",
                    },
                )

                # Log individual metric values
                if result.metrics:
                    for metric_name, metric_value in result.metrics.items():
                        if isinstance(metric_value, (int, float)):
                            log_histogram(
                                f"eval_sample_metric_{metric_name}",
                                metric_value,
                                labels={
                                    "task_type": task_config.task_type,
                                    "provider": model_data["provider"],
                                    "model": model_data["model_id"],
                                },
                            )

                # Call user progress callback if provided
                await _invoke_callback(
                    progress_callback, completed, total, result
                )

                # Log progress
                if completed % 10 == 0 or completed == total:
                    logger.info(
                        f"Evaluation progress: {completed}/{total} samples completed"
                    )
                    log_histogram(
                        "eval_run_progress_percentage",
                        (completed / total) * 100,
                        labels={
                            "task_type": task_config.task_type,
                            "provider": model_data["provider"],
                            "model": model_data["model_id"],
                        },
                    )

            # Run evaluation
            results = await eval_runner.run_evaluation(
                max_samples=max_samples, progress_callback=progress_wrapper
            )

            # Calculate and store aggregate metrics
            aggregate_metrics = eval_runner.calculate_aggregate_metrics(results)

            # Convert to required format for database
            db_metrics = {}
            for metric_name, value in aggregate_metrics.items():
                metric_type = "custom"
                if "accuracy" in metric_name:
                    metric_type = "accuracy"
                elif "f1" in metric_name:
                    metric_type = "f1"
                elif "bleu" in metric_name:
                    metric_type = "bleu"

                db_metrics[metric_name] = (value, metric_type)

                # Log aggregate metrics
                log_histogram(
                    f"eval_run_metric_{metric_name}",
                    value,
                    labels={
                        "task_type": task_config.task_type,
                        "provider": model_data["provider"],
                        "model": model_data["model_id"],
                    },
                )

            self.db.store_run_metrics(run_id, db_metrics)

            stored_result_count = len(self.db.get_results_for_run(run_id))
            expected_result_count = len(results)
            sample_error_count = sum(
                1
                for result in results
                if result.error_info or "error" in result.metrics
            )
            terminal_issues = []
            if sample_error_count:
                terminal_issues.append(
                    f"{sample_error_count} of {expected_result_count} "
                    "evaluation samples failed"
                )
            if stored_result_count != expected_result_count:
                terminal_issues.append(
                    f"Only stored {stored_result_count} of "
                    f"{expected_result_count} evaluation results"
                )
            final_status = "failed" if terminal_issues else "completed"
            final_error = "; ".join(terminal_issues) or None

            self.db.update_run_status(run_id, final_status, final_error)

            # Log completion metrics
            eval_duration = time.time() - eval_start_time
            log_histogram(
                "eval_run_duration",
                eval_duration,
                labels={
                    "task_type": task_config.task_type,
                    "provider": model_data["provider"],
                    "model": model_data["model_id"],
                    "status": final_status,
                },
            )
            log_counter(
                f"eval_run_{final_status}",
                labels={
                    "task_type": task_config.task_type,
                    "provider": model_data["provider"],
                    "model": model_data["model_id"],
                },
            )
            log_histogram(
                "eval_run_total_samples",
                len(results),
                labels={
                    "task_type": task_config.task_type,
                    "provider": model_data["provider"],
                    "model": model_data["model_id"],
                },
            )

            logger.info(
                f"Evaluation finished with status {final_status}: {run_name} ({run_id})"
            )
            logger.info(f"Results summary: {aggregate_metrics}")

            return run_id

        except asyncio.CancelledError:
            if run_id:
                try:
                    self.db.update_run_status(run_id, "cancelled")
                except Exception as status_error:
                    logger.error(
                        f"Could not persist cancellation for {run_id}: "
                        f"{status_error}"
                    )
            raise

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")

            # Log failure metrics if we have the necessary data
            if "eval_start_time" in locals():
                eval_duration = time.time() - eval_start_time
                log_histogram(
                    "eval_run_duration",
                    eval_duration,
                    labels={
                        "task_type": task_config_dict.get("task_type", "unknown")
                        if "task_config_dict" in locals()
                        else "unknown",
                        "provider": model_data["provider"]
                        if "model_data" in locals()
                        else "unknown",
                        "model": model_data["model_id"]
                        if "model_data" in locals()
                        else "unknown",
                        "status": "failed",
                    },
                )
                log_counter(
                    "eval_run_failed",
                    labels={
                        "task_type": task_config_dict.get("task_type", "unknown")
                        if "task_config_dict" in locals()
                        else "unknown",
                        "provider": model_data["provider"]
                        if "model_data" in locals()
                        else "unknown",
                        "model": model_data["model_id"]
                        if "model_data" in locals()
                        else "unknown",
                        "error_type": type(e).__name__,
                    },
                )

            # Update run status to failed if we have a run_id
            if run_id:
                with self._db_operation("updating run status to failed"):
                    self.db.update_run_status(run_id, "failed", str(e))

            # Re-raise with enhanced error context if not already an EvaluationError
            if not isinstance(e, EvaluationError):
                raise EvaluationError(
                    ErrorContext(
                        category=ErrorCategory.UNKNOWN,
                        severity=ErrorSeverity.ERROR,
                        message="Evaluation failed",
                        details=str(e),
                        is_retryable=False,
                    )
                ) from e
            raise

        finally:
            # Always unregister the run from concurrent manager
            if run_id:
                if self._active_tasks.get(run_id) is asyncio.current_task():
                    self._active_tasks.pop(run_id, None)
                await self.concurrent_manager.unregister_run(run_id)

    def get_run_summary(self, run_id: str) -> Dict[str, Any]:
        """Get a comprehensive summary of an evaluation run."""
        run_data = self.db.get_run(run_id)
        if not run_data:
            raise ValueError(f"Run not found: {run_id}")

        metrics = self.db.get_run_metrics(run_id)

        summary = {
            "run_info": run_data,
            "metrics": metrics,
            "sample_count": run_data["completed_samples"],
            "status": run_data["status"],
        }

        # Calculate duration if completed
        if run_data["start_time"] and run_data["end_time"]:
            start = datetime.fromisoformat(run_data["start_time"])
            end = datetime.fromisoformat(run_data["end_time"])
            summary["duration_seconds"] = (end - start).total_seconds()

        return summary

    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare metrics across multiple evaluation runs."""
        return self.db.compare_runs(run_ids)

    def list_tasks(self, **kwargs) -> List[Dict[str, Any]]:
        """List available tasks."""
        return self.db.list_tasks(**kwargs)

    def list_models(self, **kwargs) -> List[Dict[str, Any]]:
        """List available model configurations."""
        return self.db.list_models(**kwargs)

    def list_runs(self, **kwargs) -> List[Dict[str, Any]]:
        """List evaluation runs."""
        return self.db.list_runs(**kwargs)

    def list_datasets(self, **kwargs) -> List[Dict[str, Any]]:
        """List available datasets."""
        return self.db.list_datasets(**kwargs)

    def search_tasks(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Search tasks using full-text search."""
        return self.db.search_tasks(query, **kwargs)

    def get_task_details(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a task."""
        return self.db.get_task(task_id)

    def get_model_details(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a model configuration."""
        return self.db.get_model(model_id)

    def get_run_results(self, run_id: str, **kwargs) -> List[Dict[str, Any]]:
        """Get detailed results for a run."""
        return self.db.get_run_results(run_id, **kwargs)

    def export_results(self, run_id: str, output_path: str, format_type: str = "json"):
        """Export evaluation results to file."""
        run_summary = self.get_run_summary(run_id)
        results = self.get_run_results(run_id)

        export_data = {
            "run_summary": run_summary,
            "results": results,
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        output_path = Path(output_path)

        if format_type.lower() == "json":
            import json

            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

        elif format_type.lower() == "csv":
            import csv

            # Export results as CSV
            with open(output_path, "w", newline="") as f:
                if not results:
                    return

                fieldnames = [
                    "sample_id",
                    "input_text",
                    "expected_output",
                    "actual_output",
                ] + list(results[0]["metrics"].keys())

                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for result in results:
                    row = {
                        "sample_id": result["sample_id"],
                        "input_text": result["input_data"].get("input", ""),
                        "expected_output": result["expected_output"],
                        "actual_output": result["actual_output"],
                    }
                    row.update(result["metrics"])
                    writer.writerow(row)

        else:
            raise ValueError(f"Unsupported export format: {format_type}")

        logger.info(f"Results exported to: {output_path}")

    def create_dataset_from_file(
        self, name: str, file_path: str, description: str = None
    ) -> str:
        """Create a dataset record from a local file."""
        path = Path(file_path)
        if not path.exists():
            raise ValueError(f"Dataset file not found: {file_path}")

        # Determine format from file extension
        format_type = "json"
        if path.suffix.lower() in [".csv", ".tsv"]:
            format_type = "csv"
        elif path.suffix.lower() in [".yaml", ".yml"]:
            format_type = "yaml"

        dataset_id = self.db.create_dataset(
            name=name,
            description=description or f"Dataset from {path.name}",
            format=format_type,
            source_path=str(path),
            metadata={"file_size": path.stat().st_size, "created_from": "local_file"},
        )

        logger.info(f"Created dataset: {name} ({dataset_id})")
        return dataset_id

    def create_task_from_template(
        self, template_name: str, output_dir: str = None, **kwargs
    ) -> Tuple[str, str]:
        """
        Create a task and sample dataset from a template.

        Args:
            template_name: Name of the evaluation template
            output_dir: Directory to save template files (optional)
            **kwargs: Override parameters for the template

        Returns:
            Tuple of (task_id, dataset_id)
        """
        logger.info(f"Creating task from template: {template_name}")

        # Load template and create task config
        task_config = self.task_loader.create_task_from_template(
            template_name, **kwargs
        )

        # Create task in database
        task_id = self.db.create_task(
            name=task_config.name,
            description=task_config.description,
            task_type=task_config.task_type,
            config_format="template",
            config_data=task_config.__dict__,
        )

        # Create sample dataset if template has sample problems
        dataset_id = None
        try:
            from .eval_templates import get_eval_templates

            template_manager = get_eval_templates()

            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

                # Export sample dataset
                dataset_path = output_dir / f"{template_name}_samples.json"
                num_samples = template_manager.create_sample_dataset(
                    template_name, str(dataset_path), num_samples=20
                )

                # Create dataset record
                dataset_id = self.create_dataset_from_file(
                    name=f"{task_config.name} Samples",
                    file_path=str(dataset_path),
                    description=f"Sample dataset for {task_config.name} with {num_samples} examples",
                )

                logger.info(
                    f"Created sample dataset with {num_samples} examples: {dataset_path}"
                )

        except Exception as e:
            logger.warning(
                f"Could not create sample dataset for template {template_name}: {e}"
            )

        logger.info(f"Created task from template: {task_config.name} ({task_id})")
        return task_id, dataset_id

    def list_available_templates(self) -> List[Dict[str, Any]]:
        """List all available evaluation templates."""
        return self.task_loader.list_available_templates()

    def get_templates_by_category(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get evaluation templates organized by category."""
        templates = self.list_available_templates()
        categories = {}

        for template in templates:
            category = template.get("category", "general")
            if category not in categories:
                categories[category] = []
            categories[category].append(template)

        return categories

    async def cancel_evaluation(self, run_id: str) -> bool:
        """
        Cancel an active evaluation run.

        Args:
            run_id: ID of the evaluation run to cancel

        Returns:
            True if successfully cancelled, False otherwise
        """
        task = self._active_tasks.get(run_id)
        if task is None or task.done():
            logger.warning(f"Evaluation run not found or not active: {run_id}")
            return False

        task.cancel()
        logger.info(f"Cancellation requested for evaluation run: {run_id}")
        if task is not asyncio.current_task():
            try:
                await task
            except asyncio.CancelledError:
                pass
        return True

    def get_run_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status and details of an evaluation run.

        Args:
            run_id: ID of the evaluation run

        Returns:
            Dictionary with run details or None if not found
        """
        try:
            run_data = self.db.get_run(run_id)
            if run_data:
                # Add additional status information
                run_data["is_active"] = run_id in self._active_tasks

                # Get results if available
                results = self.db.get_run_results(run_id)
                if results:
                    run_data["results"] = results
                    run_data["samples_evaluated"] = len(results)
                else:
                    run_data["results"] = []
                    run_data["samples_evaluated"] = 0

                return run_data
            return None
        except Exception as e:
            logger.error(f"Error getting run status for {run_id}: {e}")
            return None

    def list_available_tasks(
        self, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List all available evaluation tasks.

        Args:
            limit: Maximum number of tasks to return
            offset: Number of tasks to skip

        Returns:
            List of task dictionaries
        """
        try:
            tasks = self.db.list_tasks(limit=limit, offset=offset)

            # Add source information to database tasks
            all_tasks = []
            for task in tasks:
                task["source"] = "database"
                all_tasks.append(task)

            # Try to add template information if available
            try:
                if hasattr(self.task_loader, "list_available_templates"):
                    template_tasks = self.task_loader.list_available_templates()

                    # Add template tasks (if not already in database)
                    task_names = {t.get("name") for t in tasks if "name" in t}
                    for template in template_tasks:
                        if template.get("name") and template["name"] not in task_names:
                            template["source"] = "template"
                            all_tasks.append(template)
            except Exception as template_error:
                # Log but don't fail if templates aren't available
                logger.debug(f"Could not load template tasks: {template_error}")

            return all_tasks
        except Exception as e:
            logger.error(f"Error listing available tasks: {e}")
            # Return empty list but don't crash
            return []

    async def aclose(self) -> None:
        """Cancel and drain active evaluations before closing the database."""
        tasks = list(self._active_tasks.values())
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.db.close()

    def close(self) -> None:
        """Close the database when no evaluation owns active work."""
        if self._active_tasks:
            raise RuntimeError(
                "Cannot close orchestrator while an active evaluation is running; "
                "await aclose() instead"
            )
        self.db.close()


# Convenience functions for common operations


async def quick_eval(
    task_file: str,
    provider: str,
    model: str,
    max_samples: int = 100,
    output_dir: str = None,
) -> Dict[str, Any]:
    """
    Quick evaluation helper function.

    Args:
        task_file: Path to task configuration file
        provider: LLM provider name
        model: Model identifier
        max_samples: Maximum samples to evaluate
        output_dir: Directory to save results (optional)

    Returns:
        Evaluation summary
    """
    orchestrator = EvaluationOrchestrator()

    try:
        # Create task
        task_id = await orchestrator.create_task_from_file(task_file)

        # Create model config
        model_id = orchestrator.create_model_config(
            name=f"{provider}_{model}", provider=provider, model_id=model
        )

        # Run evaluation
        run_name = f"quick_eval_{int(time.time())}"
        run_id = await orchestrator.run_evaluation(
            task_id=task_id,
            model_id=model_id,
            run_name=run_name,
            max_samples=max_samples,
        )

        # Get summary
        summary = orchestrator.get_run_summary(run_id)

        # Export results if output directory specified
        if output_dir:
            output_path = Path(output_dir) / f"{run_name}_results.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            orchestrator.export_results(run_id, str(output_path))
            summary["exported_to"] = str(output_path)

        return summary

    finally:
        orchestrator.close()


def create_task_template(task_type: str, name: str, output_path: str):
    """Create a task template file."""
    loader = TaskLoader()

    task_config = loader.create_task_from_template(
        template_name=task_type,
        name=name,
        description=f"Template for {task_type} evaluation",
    )

    loader.export_task(task_config, output_path, format_type="custom")
    logger.info(f"Created task template: {output_path}")
