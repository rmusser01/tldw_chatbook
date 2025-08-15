"""
Evaluation Messages - Custom message definitions for component communication
Following Textual message patterns for type-safe event handling
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from textual.message import Message


@dataclass
class EvaluationMessage(Message):
    """Base class for all evaluation messages."""
    
    def __init__(self, **kwargs):
        """Initialize with optional metadata."""
        super().__init__()
        self.metadata = kwargs


# Task Messages

@dataclass
class TaskSelected(EvaluationMessage):
    """Message sent when a task is selected."""
    
    def __init__(self, task_id: str, task_name: str, **kwargs):
        """
        Initialize task selected message.
        
        Args:
            task_id: The ID of the selected task
            task_name: The name of the selected task
            **kwargs: Additional metadata
        """
        super().__init__(**kwargs)
        self.task_id = task_id
        self.task_name = task_name


@dataclass 
class TaskCreated(EvaluationMessage):
    """Message sent when a new task is created."""
    
    def __init__(self, task_id: str, task_config: Dict[str, Any], **kwargs):
        """
        Initialize task created message.
        
        Args:
            task_id: The ID of the created task
            task_config: The task configuration
            **kwargs: Additional metadata
        """
        super().__init__(**kwargs)
        self.task_id = task_id
        self.task_config = task_config


@dataclass
class TaskDeleted(EvaluationMessage):
    """Message sent when a task is deleted."""
    
    def __init__(self, task_id: str, **kwargs):
        """
        Initialize task deleted message.
        
        Args:
            task_id: The ID of the deleted task
            **kwargs: Additional metadata
        """
        super().__init__(**kwargs)
        self.task_id = task_id


# Model Messages

@dataclass
class ModelSelected(EvaluationMessage):
    """Message sent when a model is selected."""
    
    def __init__(self, model_id: str, model_name: str, provider: str, **kwargs):
        """
        Initialize model selected message.
        
        Args:
            model_id: The ID of the selected model
            model_name: The name of the selected model
            provider: The model provider
            **kwargs: Additional metadata
        """
        super().__init__(**kwargs)
        self.model_id = model_id
        self.model_name = model_name
        self.provider = provider


@dataclass
class ModelAdded(EvaluationMessage):
    """Message sent when a new model is added."""
    
    def __init__(self, model_id: str, model_config: Dict[str, Any], **kwargs):
        """
        Initialize model added message.
        
        Args:
            model_id: The ID of the added model
            model_config: The model configuration
            **kwargs: Additional metadata
        """
        super().__init__(**kwargs)
        self.model_id = model_id
        self.model_config = model_config


@dataclass
class ModelTested(EvaluationMessage):
    """Message sent when a model connection is tested."""
    
    def __init__(self, model_id: str, success: bool, latency_ms: Optional[float] = None, error: Optional[str] = None, **kwargs):
        """
        Initialize model tested message.
        
        Args:
            model_id: The ID of the tested model
            success: Whether the test was successful
            latency_ms: Test latency in milliseconds
            error: Error message if test failed
            **kwargs: Additional metadata
        """
        super().__init__(**kwargs)
        self.model_id = model_id
        self.success = success
        self.latency_ms = latency_ms
        self.error = error


# Dataset Messages

@dataclass
class DatasetSelected(EvaluationMessage):
    """Message sent when a dataset is selected."""
    
    def __init__(self, dataset_id: str, dataset_name: str, sample_count: int, **kwargs):
        """
        Initialize dataset selected message.
        
        Args:
            dataset_id: The ID of the selected dataset
            dataset_name: The name of the selected dataset
            sample_count: Number of samples in the dataset
            **kwargs: Additional metadata
        """
        super().__init__(**kwargs)
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.sample_count = sample_count


@dataclass
class DatasetUploaded(EvaluationMessage):
    """Message sent when a dataset is uploaded."""
    
    def __init__(self, dataset_id: str, file_path: str, format: str, **kwargs):
        """
        Initialize dataset uploaded message.
        
        Args:
            dataset_id: The ID of the uploaded dataset
            file_path: Path to the uploaded file
            format: Dataset format (csv, json, parquet, etc.)
            **kwargs: Additional metadata
        """
        super().__init__(**kwargs)
        self.dataset_id = dataset_id
        self.file_path = file_path
        self.format = format


@dataclass
class DatasetValidated(EvaluationMessage):
    """Message sent when a dataset is validated."""
    
    def __init__(self, dataset_id: str, valid: bool, errors: List[str], warnings: List[str], **kwargs):
        """
        Initialize dataset validated message.
        
        Args:
            dataset_id: The ID of the validated dataset
            valid: Whether the dataset is valid
            errors: List of validation errors
            warnings: List of validation warnings
            **kwargs: Additional metadata
        """
        super().__init__(**kwargs)
        self.dataset_id = dataset_id
        self.valid = valid
        self.errors = errors
        self.warnings = warnings


# Evaluation Messages

@dataclass
class EvaluationStarted(EvaluationMessage):
    """Message sent when an evaluation starts."""
    
    def __init__(self, evaluation_id: Optional[str] = None, quick_start: bool = False, **kwargs):
        """
        Initialize evaluation started message.
        
        Args:
            evaluation_id: The ID of the evaluation
            quick_start: Whether this is a quick start evaluation
            **kwargs: Additional metadata
        """
        super().__init__(**kwargs)
        self.evaluation_id = evaluation_id
        self.quick_start = quick_start


@dataclass
class EvaluationPaused(EvaluationMessage):
    """Message sent when an evaluation is paused."""
    
    def __init__(self, evaluation_id: str, **kwargs):
        """
        Initialize evaluation paused message.
        
        Args:
            evaluation_id: The ID of the paused evaluation
            **kwargs: Additional metadata
        """
        super().__init__(**kwargs)
        self.evaluation_id = evaluation_id


@dataclass
class EvaluationResumed(EvaluationMessage):
    """Message sent when an evaluation is resumed."""
    
    def __init__(self, evaluation_id: str, **kwargs):
        """
        Initialize evaluation resumed message.
        
        Args:
            evaluation_id: The ID of the resumed evaluation
            **kwargs: Additional metadata
        """
        super().__init__(**kwargs)
        self.evaluation_id = evaluation_id


@dataclass
class EvaluationCancelled(EvaluationMessage):
    """Message sent when an evaluation is cancelled."""
    
    def __init__(self, evaluation_id: str, reason: Optional[str] = None, **kwargs):
        """
        Initialize evaluation cancelled message.
        
        Args:
            evaluation_id: The ID of the cancelled evaluation
            reason: Reason for cancellation
            **kwargs: Additional metadata
        """
        super().__init__(**kwargs)
        self.evaluation_id = evaluation_id
        self.reason = reason


@dataclass
class EvaluationCompleted(EvaluationMessage):
    """Message sent when an evaluation completes."""
    
    def __init__(self, evaluation_id: Optional[str] = None, metrics: Optional[Dict[str, float]] = None, **kwargs):
        """
        Initialize evaluation completed message.
        
        Args:
            evaluation_id: The ID of the completed evaluation
            metrics: Final evaluation metrics
            **kwargs: Additional metadata
        """
        super().__init__(**kwargs)
        self.evaluation_id = evaluation_id
        self.metrics = metrics or {}


# Progress Messages

@dataclass
class ProgressUpdate(EvaluationMessage):
    """Message sent to update progress."""
    
    def __init__(self, percentage: float, message: str = "", current_sample: int = 0, total_samples: int = 0, **kwargs):
        """
        Initialize progress update message.
        
        Args:
            percentage: Progress percentage (0-100)
            message: Progress message
            current_sample: Current sample being processed
            total_samples: Total number of samples
            **kwargs: Additional metadata
        """
        super().__init__(**kwargs)
        self.percentage = percentage
        self.message = message
        self.current_sample = current_sample
        self.total_samples = total_samples


@dataclass
class SampleProcessed(EvaluationMessage):
    """Message sent when a sample is processed."""
    
    def __init__(self, sample_id: str, success: bool, metrics: Dict[str, float], error: Optional[str] = None, **kwargs):
        """
        Initialize sample processed message.
        
        Args:
            sample_id: The ID of the processed sample
            success: Whether processing was successful
            metrics: Sample metrics
            error: Error message if processing failed
            **kwargs: Additional metadata
        """
        super().__init__(**kwargs)
        self.sample_id = sample_id
        self.success = success
        self.metrics = metrics
        self.error = error


# Cost Messages

@dataclass
class CostEstimated(EvaluationMessage):
    """Message sent when cost is estimated."""
    
    def __init__(self, estimated_cost: float, tokens: int, warnings: List[str], **kwargs):
        """
        Initialize cost estimated message.
        
        Args:
            estimated_cost: Estimated cost in dollars
            tokens: Estimated token usage
            warnings: Cost-related warnings
            **kwargs: Additional metadata
        """
        super().__init__(**kwargs)
        self.estimated_cost = estimated_cost
        self.tokens = tokens
        self.warnings = warnings


@dataclass
class CostUpdated(EvaluationMessage):
    """Message sent when actual cost is updated."""
    
    def __init__(self, actual_cost: float, tokens_used: int, remaining_budget: float, **kwargs):
        """
        Initialize cost updated message.
        
        Args:
            actual_cost: Actual cost so far in dollars
            tokens_used: Actual tokens used
            remaining_budget: Remaining budget
            **kwargs: Additional metadata
        """
        super().__init__(**kwargs)
        self.actual_cost = actual_cost
        self.tokens_used = tokens_used
        self.remaining_budget = remaining_budget


# Error Messages

@dataclass
class ErrorOccurred(EvaluationMessage):
    """Message sent when an error occurs."""
    
    def __init__(self, error: Exception, context: str = "", recoverable: bool = True, **kwargs):
        """
        Initialize error occurred message.
        
        Args:
            error: The exception that occurred
            context: Context where error occurred
            recoverable: Whether the error is recoverable
            **kwargs: Additional metadata
        """
        super().__init__(**kwargs)
        self.error = error
        self.context = context
        self.recoverable = recoverable


@dataclass
class WarningRaised(EvaluationMessage):
    """Message sent when a warning is raised."""
    
    def __init__(self, warning: str, severity: str = "low", **kwargs):
        """
        Initialize warning raised message.
        
        Args:
            warning: Warning message
            severity: Warning severity (low, medium, high)
            **kwargs: Additional metadata
        """
        super().__init__(**kwargs)
        self.warning = warning
        self.severity = severity


# Configuration Messages

@dataclass
class ConfigurationSaved(EvaluationMessage):
    """Message sent when configuration is saved."""
    
    def __init__(self, config_id: str, file_path: str, **kwargs):
        """
        Initialize configuration saved message.
        
        Args:
            config_id: The ID of the saved configuration
            file_path: Path where configuration was saved
            **kwargs: Additional metadata
        """
        super().__init__(**kwargs)
        self.config_id = config_id
        self.file_path = file_path


@dataclass
class ConfigurationLoaded(EvaluationMessage):
    """Message sent when configuration is loaded."""
    
    def __init__(self, config_id: str, file_path: str, **kwargs):
        """
        Initialize configuration loaded message.
        
        Args:
            config_id: The ID of the loaded configuration
            file_path: Path from where configuration was loaded
            **kwargs: Additional metadata
        """
        super().__init__(**kwargs)
        self.config_id = config_id
        self.file_path = file_path


@dataclass
class ConfigurationValidated(EvaluationMessage):
    """Message sent when configuration is validated."""
    
    def __init__(self, valid: bool, errors: List[str], warnings: List[str], **kwargs):
        """
        Initialize configuration validated message.
        
        Args:
            valid: Whether configuration is valid
            errors: List of validation errors
            warnings: List of validation warnings
            **kwargs: Additional metadata
        """
        super().__init__(**kwargs)
        self.valid = valid
        self.errors = errors
        self.warnings = warnings


# Export Messages

@dataclass
class ResultsExported(EvaluationMessage):
    """Message sent when results are exported."""
    
    def __init__(self, export_path: str, format: str, result_count: int, **kwargs):
        """
        Initialize results exported message.
        
        Args:
            export_path: Path where results were exported
            format: Export format (csv, json, html, etc.)
            result_count: Number of results exported
            **kwargs: Additional metadata
        """
        super().__init__(**kwargs)
        self.export_path = export_path
        self.format = format
        self.result_count = result_count


# Template Messages

@dataclass
class TemplateLoaded(EvaluationMessage):
    """Message sent when a template is loaded."""
    
    def __init__(self, template_id: str, template_type: str, **kwargs):
        """
        Initialize template loaded message.
        
        Args:
            template_id: The ID of the loaded template
            template_type: Type of template (task, model, dataset, config)
            **kwargs: Additional metadata
        """
        super().__init__(**kwargs)
        self.template_id = template_id
        self.template_type = template_type


@dataclass
class TemplateSaved(EvaluationMessage):
    """Message sent when a template is saved."""
    
    def __init__(self, template_id: str, template_type: str, file_path: str, **kwargs):
        """
        Initialize template saved message.
        
        Args:
            template_id: The ID of the saved template
            template_type: Type of template (task, model, dataset, config)
            file_path: Path where template was saved
            **kwargs: Additional metadata
        """
        super().__init__(**kwargs)
        self.template_id = template_id
        self.template_type = template_type
        self.file_path = file_path