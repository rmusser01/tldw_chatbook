"""
Evaluation UI Module - Modern Textual-based evaluation interface
"""

from .evals_screen import EvaluationScreen
from .evals_state import (
    EvaluationState,
    TaskConfig,
    ModelConfig,
    DatasetConfig,
    EvaluationConfig,
    EvaluationResult
)
from .evals_messages import (
    # Task messages
    TaskSelected,
    TaskCreated,
    TaskDeleted,
    # Model messages
    ModelSelected,
    ModelAdded,
    ModelTested,
    # Dataset messages
    DatasetSelected,
    DatasetUploaded,
    DatasetValidated,
    # Evaluation messages
    EvaluationStarted,
    EvaluationPaused,
    EvaluationResumed,
    EvaluationCancelled,
    EvaluationCompleted,
    # Progress messages
    ProgressUpdate,
    SampleProcessed,
    # Cost messages
    CostEstimated,
    CostUpdated,
    # Error messages
    ErrorOccurred,
    WarningRaised,
    # Configuration messages
    ConfigurationSaved,
    ConfigurationLoaded,
    ConfigurationValidated,
    # Export messages
    ResultsExported,
    # Template messages
    TemplateLoaded,
    TemplateSaved
)

__all__ = [
    # Screen
    "EvaluationScreen",
    # State
    "EvaluationState",
    "TaskConfig",
    "ModelConfig",
    "DatasetConfig",
    "EvaluationConfig",
    "EvaluationResult",
    # Messages
    "TaskSelected",
    "TaskCreated",
    "TaskDeleted",
    "ModelSelected",
    "ModelAdded",
    "ModelTested",
    "DatasetSelected",
    "DatasetUploaded",
    "DatasetValidated",
    "EvaluationStarted",
    "EvaluationPaused",
    "EvaluationResumed",
    "EvaluationCancelled",
    "EvaluationCompleted",
    "ProgressUpdate",
    "SampleProcessed",
    "CostEstimated",
    "CostUpdated",
    "ErrorOccurred",
    "WarningRaised",
    "ConfigurationSaved",
    "ConfigurationLoaded",
    "ConfigurationValidated",
    "ResultsExported",
    "TemplateLoaded",
    "TemplateSaved"
]