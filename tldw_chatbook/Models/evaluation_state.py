# evaluation_state.py
# Description: Centralized state management for the evaluation system
#
"""
Evaluation State Management
---------------------------

Provides centralized state management for the evaluation system:
- Single source of truth for all evaluation-related state
- Type-safe state updates
- Observable state changes
- Serializable for persistence
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
import json
from loguru import logger

# Configure logger
logger = logger.bind(module="evaluation_state")


class RunStatus(Enum):
    """Evaluation run status enumeration."""
    IDLE = auto()
    CONFIGURING = auto()
    STARTING = auto()
    RUNNING = auto()
    PAUSED = auto()
    CANCELLING = auto()
    COMPLETED = auto()
    ERROR = auto()
    CANCELLED = auto()


class ViewType(Enum):
    """Available view types in the evaluation system."""
    SETUP = "setup"
    RESULTS = "results"
    MODELS = "models"
    DATASETS = "datasets"


@dataclass
class Model:
    """Model configuration."""
    id: str
    name: str
    provider: str
    model_id: str
    api_key_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "model_id": self.model_id,
            "api_key_name": self.api_key_name,
            "parameters": self.parameters,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Model':
        """Create from dictionary."""
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('last_used'):
            data['last_used'] = datetime.fromisoformat(data['last_used'])
        return cls(**data)


@dataclass
class Dataset:
    """Dataset configuration."""
    id: str
    name: str
    format: str  # csv, json, parquet, etc.
    path: Optional[Path] = None
    url: Optional[str] = None
    sample_count: int = 0
    columns: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "format": self.format,
            "path": str(self.path) if self.path else None,
            "url": self.url,
            "sample_count": self.sample_count,
            "columns": self.columns,
            "created_at": self.created_at.isoformat(),
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Dataset':
        """Create from dictionary."""
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('last_modified'):
            data['last_modified'] = datetime.fromisoformat(data['last_modified'])
        if data.get('path'):
            data['path'] = Path(data['path'])
        return cls(**data)


@dataclass
class EvaluationTask:
    """Evaluation task configuration."""
    id: str
    name: str
    description: str
    task_type: str  # classification, generation, etc.
    metrics: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "task_type": self.task_type,
            "metrics": self.metrics,
            "parameters": self.parameters,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationTask':
        """Create from dictionary."""
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


@dataclass
class EvaluationConfig:
    """Configuration for an evaluation run."""
    task_id: str
    model_id: str
    dataset_id: str
    max_samples: Optional[int] = None
    batch_size: int = 1
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    timeout_seconds: int = 30
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """
        Validate the configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if not self.task_id:
            errors.append("Task ID is required")
        if not self.model_id:
            errors.append("Model ID is required")
        if not self.dataset_id:
            errors.append("Dataset ID is required")
        if self.max_samples is not None and self.max_samples <= 0:
            errors.append("Max samples must be positive")
        if self.batch_size <= 0:
            errors.append("Batch size must be positive")
        if not 0 <= self.temperature <= 2:
            errors.append("Temperature must be between 0 and 2")
        if self.timeout_seconds <= 0:
            errors.append("Timeout must be positive")
        
        return errors


@dataclass
class EvaluationRun:
    """Information about an evaluation run."""
    id: str
    name: str
    config: EvaluationConfig
    status: RunStatus = RunStatus.IDLE
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_samples: int = 0
    completed_samples: int = 0
    successful_samples: int = 0
    failed_samples: int = 0
    current_sample: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    results: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_samples == 0:
            return 0.0
        return (self.completed_samples / self.total_samples) * 100
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.completed_samples == 0:
            return 0.0
        return (self.successful_samples / self.completed_samples) * 100
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate run duration in seconds."""
        if not self.started_at:
            return None
        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "config": self.config.__dict__,
            "status": self.status.name,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_samples": self.total_samples,
            "completed_samples": self.completed_samples,
            "successful_samples": self.successful_samples,
            "failed_samples": self.failed_samples,
            "current_sample": self.current_sample,
            "error_message": self.error_message,
            "results": self.results,
            "metrics": self.metrics
        }


@dataclass
class EvaluationState:
    """
    Centralized state for the evaluation system.
    
    This class serves as the single source of truth for all evaluation-related state.
    It provides methods for state updates and change notifications.
    """
    
    # Navigation
    active_view: ViewType = ViewType.SETUP
    sidebar_collapsed: bool = False
    
    # Current evaluation run
    current_run: Optional[EvaluationRun] = None
    
    # Data collections
    models: Dict[str, Model] = field(default_factory=dict)
    datasets: Dict[str, Dataset] = field(default_factory=dict)
    tasks: Dict[str, EvaluationTask] = field(default_factory=dict)
    recent_runs: List[EvaluationRun] = field(default_factory=list)
    
    # UI state
    loading_states: Dict[str, bool] = field(default_factory=dict)
    error_messages: Dict[str, str] = field(default_factory=dict)
    selected_items: Dict[str, Set[str]] = field(default_factory=dict)
    
    # Configuration being built
    draft_config: Optional[EvaluationConfig] = None
    
    # Change tracking
    _observers: List[Any] = field(default_factory=list, init=False, repr=False)
    _dirty_fields: Set[str] = field(default_factory=set, init=False, repr=False)
    
    def __post_init__(self):
        """Initialize change tracking."""
        self._last_state_hash = self._compute_state_hash()
    
    def _compute_state_hash(self) -> int:
        """Compute a hash of the current state for change detection."""
        # Simple hash based on key fields
        return hash((
            self.active_view,
            self.sidebar_collapsed,
            self.current_run.id if self.current_run else None,
            len(self.models),
            len(self.datasets),
            len(self.tasks),
            len(self.recent_runs)
        ))
    
    def add_observer(self, observer: Any) -> None:
        """Add an observer for state changes."""
        if observer not in self._observers:
            self._observers.append(observer)
    
    def remove_observer(self, observer: Any) -> None:
        """Remove an observer."""
        if observer in self._observers:
            self._observers.remove(observer)
    
    def _notify_observers(self, field: str, old_value: Any, new_value: Any) -> None:
        """Notify observers of a state change."""
        for observer in self._observers:
            if hasattr(observer, 'on_state_change'):
                try:
                    observer.on_state_change(field, old_value, new_value)
                except Exception as e:
                    logger.error(f"Error notifying observer: {e}")
    
    # Navigation methods
    
    def set_active_view(self, view: ViewType) -> None:
        """Set the active view."""
        if view != self.active_view:
            old_view = self.active_view
            self.active_view = view
            self._notify_observers('active_view', old_view, view)
    
    def toggle_sidebar(self) -> None:
        """Toggle sidebar collapsed state."""
        self.sidebar_collapsed = not self.sidebar_collapsed
        self._notify_observers('sidebar_collapsed', not self.sidebar_collapsed, self.sidebar_collapsed)
    
    # Model management
    
    def add_model(self, model: Model) -> None:
        """Add a model configuration."""
        self.models[model.id] = model
        self._notify_observers('models', None, model)
    
    def remove_model(self, model_id: str) -> None:
        """Remove a model configuration."""
        if model_id in self.models:
            model = self.models.pop(model_id)
            self._notify_observers('models', model, None)
    
    def get_model(self, model_id: str) -> Optional[Model]:
        """Get a model by ID."""
        return self.models.get(model_id)
    
    # Dataset management
    
    def add_dataset(self, dataset: Dataset) -> None:
        """Add a dataset."""
        self.datasets[dataset.id] = dataset
        self._notify_observers('datasets', None, dataset)
    
    def remove_dataset(self, dataset_id: str) -> None:
        """Remove a dataset."""
        if dataset_id in self.datasets:
            dataset = self.datasets.pop(dataset_id)
            self._notify_observers('datasets', dataset, None)
    
    def get_dataset(self, dataset_id: str) -> Optional[Dataset]:
        """Get a dataset by ID."""
        return self.datasets.get(dataset_id)
    
    # Task management
    
    def add_task(self, task: EvaluationTask) -> None:
        """Add an evaluation task."""
        self.tasks[task.id] = task
        self._notify_observers('tasks', None, task)
    
    def remove_task(self, task_id: str) -> None:
        """Remove a task."""
        if task_id in self.tasks:
            task = self.tasks.pop(task_id)
            self._notify_observers('tasks', task, None)
    
    def get_task(self, task_id: str) -> Optional[EvaluationTask]:
        """Get a task by ID."""
        return self.tasks.get(task_id)
    
    # Evaluation run management
    
    def start_evaluation(self, run: EvaluationRun) -> None:
        """Start a new evaluation run."""
        old_run = self.current_run
        self.current_run = run
        run.status = RunStatus.STARTING
        run.started_at = datetime.now()
        self._notify_observers('current_run', old_run, run)
    
    def update_run_progress(
        self,
        completed: int,
        successful: int,
        failed: int,
        current_sample: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update current run progress."""
        if self.current_run:
            self.current_run.completed_samples = completed
            self.current_run.successful_samples = successful
            self.current_run.failed_samples = failed
            self.current_run.current_sample = current_sample
            self._notify_observers('current_run', self.current_run, self.current_run)
    
    def complete_evaluation(self, metrics: Dict[str, Any]) -> None:
        """Mark current evaluation as complete."""
        if self.current_run:
            self.current_run.status = RunStatus.COMPLETED
            self.current_run.completed_at = datetime.now()
            self.current_run.metrics = metrics
            self.recent_runs.insert(0, self.current_run)
            
            # Keep only last 100 runs
            if len(self.recent_runs) > 100:
                self.recent_runs = self.recent_runs[:100]
            
            old_run = self.current_run
            self.current_run = None
            self._notify_observers('current_run', old_run, None)
    
    def cancel_evaluation(self) -> None:
        """Cancel the current evaluation."""
        if self.current_run:
            self.current_run.status = RunStatus.CANCELLED
            self.current_run.completed_at = datetime.now()
            self.recent_runs.insert(0, self.current_run)
            
            old_run = self.current_run
            self.current_run = None
            self._notify_observers('current_run', old_run, None)
    
    def set_evaluation_error(self, error_message: str) -> None:
        """Set an error on the current evaluation."""
        if self.current_run:
            self.current_run.status = RunStatus.ERROR
            self.current_run.error_message = error_message
            self.current_run.completed_at = datetime.now()
            self._notify_observers('current_run', self.current_run, self.current_run)
    
    # Loading state management
    
    def set_loading(self, key: str, is_loading: bool) -> None:
        """Set loading state for a specific operation."""
        old_value = self.loading_states.get(key, False)
        self.loading_states[key] = is_loading
        if old_value != is_loading:
            self._notify_observers('loading_states', {key: old_value}, {key: is_loading})
    
    def is_loading(self, key: str) -> bool:
        """Check if a specific operation is loading."""
        return self.loading_states.get(key, False)
    
    # Error management
    
    def set_error(self, key: str, error_message: str) -> None:
        """Set an error message."""
        self.error_messages[key] = error_message
        self._notify_observers('error_messages', None, {key: error_message})
    
    def clear_error(self, key: str) -> None:
        """Clear an error message."""
        if key in self.error_messages:
            old_message = self.error_messages.pop(key)
            self._notify_observers('error_messages', {key: old_message}, None)
    
    def get_error(self, key: str) -> Optional[str]:
        """Get an error message."""
        return self.error_messages.get(key)
    
    # Selection management
    
    def select_item(self, category: str, item_id: str) -> None:
        """Select an item in a category."""
        if category not in self.selected_items:
            self.selected_items[category] = set()
        self.selected_items[category].add(item_id)
        self._notify_observers('selected_items', None, {category: item_id})
    
    def deselect_item(self, category: str, item_id: str) -> None:
        """Deselect an item."""
        if category in self.selected_items:
            self.selected_items[category].discard(item_id)
            self._notify_observers('selected_items', {category: item_id}, None)
    
    def get_selected_items(self, category: str) -> Set[str]:
        """Get selected items in a category."""
        return self.selected_items.get(category, set())
    
    def clear_selection(self, category: str) -> None:
        """Clear all selections in a category."""
        if category in self.selected_items:
            old_items = self.selected_items[category].copy()
            self.selected_items[category].clear()
            self._notify_observers('selected_items', {category: old_items}, {category: set()})
    
    # Configuration management
    
    def start_draft_config(self) -> None:
        """Start building a new configuration."""
        self.draft_config = EvaluationConfig(
            task_id="",
            model_id="",
            dataset_id=""
        )
        self._notify_observers('draft_config', None, self.draft_config)
    
    def update_draft_config(self, **kwargs) -> None:
        """Update the draft configuration."""
        if self.draft_config:
            for key, value in kwargs.items():
                if hasattr(self.draft_config, key):
                    setattr(self.draft_config, key, value)
            self._notify_observers('draft_config', self.draft_config, self.draft_config)
    
    def clear_draft_config(self) -> None:
        """Clear the draft configuration."""
        old_config = self.draft_config
        self.draft_config = None
        self._notify_observers('draft_config', old_config, None)
    
    # Persistence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "active_view": self.active_view.value,
            "sidebar_collapsed": self.sidebar_collapsed,
            "models": {k: v.to_dict() for k, v in self.models.items()},
            "datasets": {k: v.to_dict() for k, v in self.datasets.items()},
            "tasks": {k: v.to_dict() for k, v in self.tasks.items()},
            "recent_runs": [run.to_dict() for run in self.recent_runs[:20]],  # Keep last 20
        }
    
    def save_to_file(self, filepath: Path) -> None:
        """Save state to a JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"State saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    @classmethod
    def load_from_file(cls, filepath: Path) -> 'EvaluationState':
        """Load state from a JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            state = cls()
            
            # Restore view
            state.active_view = ViewType(data.get('active_view', 'setup'))
            state.sidebar_collapsed = data.get('sidebar_collapsed', False)
            
            # Restore models
            for model_data in data.get('models', {}).values():
                model = Model.from_dict(model_data)
                state.models[model.id] = model
            
            # Restore datasets
            for dataset_data in data.get('datasets', {}).values():
                dataset = Dataset.from_dict(dataset_data)
                state.datasets[dataset.id] = dataset
            
            # Restore tasks
            for task_data in data.get('tasks', {}).values():
                task = EvaluationTask.from_dict(task_data)
                state.tasks[task.id] = task
            
            logger.info(f"State loaded from {filepath}")
            return state
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return cls()


# Export main classes
__all__ = [
    'EvaluationState',
    'RunStatus',
    'ViewType',
    'Model',
    'Dataset',
    'EvaluationTask',
    'EvaluationConfig',
    'EvaluationRun',
]