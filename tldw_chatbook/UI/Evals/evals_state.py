"""
Evaluation State Management - Centralized reactive state for evaluations
Following Textual reactive patterns for predictable state updates
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

from textual.reactive import reactive, Reactive
from loguru import logger

logger = logger.bind(module="evals_state")


@dataclass
class TaskConfig:
    """Configuration for an evaluation task."""
    task_id: str
    task_name: str
    task_type: str  # "multiple_choice", "generation", "classification", "code_generation"
    prompt_template: str
    metrics: List[str] = field(default_factory=list)
    success_threshold: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Configuration for a model."""
    model_id: str
    model_name: str
    provider: str
    api_key_env: Optional[str] = None
    endpoint: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 30
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    dataset_id: str
    dataset_name: str
    dataset_type: str  # "csv", "json", "parquet", "huggingface"
    path: Optional[str] = None
    sample_count: int = 0
    columns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class EvaluationConfig:
    """Complete evaluation configuration."""
    config_id: str
    task: Optional[TaskConfig] = None
    model: Optional[ModelConfig] = None
    dataset: Optional[DatasetConfig] = None
    max_samples: Optional[int] = None
    batch_size: int = 1
    parallel_requests: int = 5
    save_responses: bool = True
    auto_export: bool = True
    enable_caching: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """
        Validate the configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if not self.task:
            errors.append("No task selected")
        if not self.model:
            errors.append("No model selected")
        if not self.dataset:
            errors.append("No dataset selected")
            
        if self.batch_size < 1:
            errors.append("Batch size must be at least 1")
        if self.parallel_requests < 1:
            errors.append("Parallel requests must be at least 1")
            
        if self.max_samples and self.max_samples < 1:
            errors.append("Max samples must be at least 1")
            
        return errors


@dataclass
class EvaluationResult:
    """Results from an evaluation run."""
    result_id: str
    config_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "pending"  # "pending", "running", "completed", "failed", "cancelled"
    total_samples: int = 0
    completed_samples: int = 0
    failed_samples: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_samples == 0:
            return 0.0
        return (self.completed_samples / self.total_samples) * 100
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate duration in seconds."""
        if not self.end_time:
            return None
        return (self.end_time - self.start_time).total_seconds()


class EvaluationState:
    """
    Centralized state management for evaluations.
    
    Uses Textual's reactive attributes for automatic UI updates.
    Provides single source of truth for evaluation state.
    """
    
    # Reactive attributes for UI binding
    current_config: Reactive[Optional[EvaluationConfig]] = reactive(None)
    evaluation_status: Reactive[str] = reactive("idle")  # "idle", "configuring", "running", "completed", "error"
    selected_task: Reactive[Optional[TaskConfig]] = reactive(None)
    selected_model: Reactive[Optional[ModelConfig]] = reactive(None)
    selected_dataset: Reactive[Optional[DatasetConfig]] = reactive(None)
    
    # Collections
    available_tasks: Reactive[Dict[str, TaskConfig]] = reactive({})
    available_models: Reactive[Dict[str, ModelConfig]] = reactive({})
    available_datasets: Reactive[Dict[str, DatasetConfig]] = reactive({})
    
    # Results
    current_result: Reactive[Optional[EvaluationResult]] = reactive(None)
    recent_results: Reactive[List[EvaluationResult]] = reactive([])
    
    # Progress tracking
    evaluation_progress: Reactive[float] = reactive(0.0)
    progress_message: Reactive[str] = reactive("")
    
    # Cost tracking
    estimated_cost: Reactive[float] = reactive(0.0)
    actual_cost: Reactive[float] = reactive(0.0)
    daily_budget: Reactive[float] = reactive(100.0)
    daily_spent: Reactive[float] = reactive(0.0)
    
    # Errors and warnings
    errors: Reactive[List[str]] = reactive([])
    warnings: Reactive[List[str]] = reactive([])
    
    def __init__(self):
        """Initialize the evaluation state."""
        self._storage_path = self._get_storage_path()
        self._initialize_defaults()
        
    def _get_storage_path(self) -> Path:
        """Get the path for state storage."""
        from ...config import get_user_data_dir
        return get_user_data_dir() / "evals" / "state.json"
    
    def _initialize_defaults(self) -> None:
        """Initialize default values."""
        # Create default configuration
        self.current_config = EvaluationConfig(
            config_id="default",
            batch_size=1,
            parallel_requests=5
        )
        
    # Task Management
    
    def select_task(self, task_id: str) -> None:
        """
        Select a task by ID.
        
        Args:
            task_id: The task ID to select
        """
        if task_id in self.available_tasks:
            self.selected_task = self.available_tasks[task_id]
            if self.current_config:
                self.current_config.task = self.selected_task
            logger.info(f"Task selected: {task_id}")
        else:
            logger.warning(f"Task not found: {task_id}")
            self.errors.append(f"Task not found: {task_id}")
    
    def add_task(self, task: TaskConfig) -> None:
        """
        Add a new task.
        
        Args:
            task: The task configuration to add
        """
        self.available_tasks[task.task_id] = task
        logger.info(f"Task added: {task.task_id}")
    
    # Model Management
    
    def select_model(self, model_id: str) -> None:
        """
        Select a model by ID.
        
        Args:
            model_id: The model ID to select
        """
        if model_id in self.available_models:
            self.selected_model = self.available_models[model_id]
            if self.current_config:
                self.current_config.model = self.selected_model
            logger.info(f"Model selected: {model_id}")
        else:
            logger.warning(f"Model not found: {model_id}")
            self.errors.append(f"Model not found: {model_id}")
    
    def add_model(self, model: ModelConfig) -> None:
        """
        Add a new model.
        
        Args:
            model: The model configuration to add
        """
        self.available_models[model.model_id] = model
        logger.info(f"Model added: {model.model_id}")
    
    # Dataset Management
    
    def select_dataset(self, dataset_id: str) -> None:
        """
        Select a dataset by ID.
        
        Args:
            dataset_id: The dataset ID to select
        """
        if dataset_id in self.available_datasets:
            self.selected_dataset = self.available_datasets[dataset_id]
            if self.current_config:
                self.current_config.dataset = self.selected_dataset
            logger.info(f"Dataset selected: {dataset_id}")
        else:
            logger.warning(f"Dataset not found: {dataset_id}")
            self.errors.append(f"Dataset not found: {dataset_id}")
    
    def add_dataset(self, dataset: DatasetConfig) -> None:
        """
        Add a new dataset.
        
        Args:
            dataset: The dataset configuration to add
        """
        self.available_datasets[dataset.dataset_id] = dataset
        logger.info(f"Dataset added: {dataset.dataset_id}")
    
    # Configuration Management
    
    def is_valid(self) -> bool:
        """
        Check if current configuration is valid.
        
        Returns:
            True if configuration is valid
        """
        if not self.current_config:
            return False
        return len(self.current_config.validate()) == 0
    
    def get_validation_errors(self) -> List[str]:
        """
        Get validation errors for current configuration.
        
        Returns:
            List of validation errors
        """
        if not self.current_config:
            return ["No configuration available"]
        return self.current_config.validate()
    
    def create_new_config(self) -> EvaluationConfig:
        """
        Create a new evaluation configuration.
        
        Returns:
            New evaluation configuration
        """
        import uuid
        config = EvaluationConfig(
            config_id=str(uuid.uuid4()),
            task=self.selected_task,
            model=self.selected_model,
            dataset=self.selected_dataset
        )
        self.current_config = config
        return config
    
    # Result Management
    
    def start_evaluation(self) -> EvaluationResult:
        """
        Start a new evaluation.
        
        Returns:
            New evaluation result object
        """
        import uuid
        
        if not self.current_config:
            raise ValueError("No configuration available")
            
        result = EvaluationResult(
            result_id=str(uuid.uuid4()),
            config_id=self.current_config.config_id,
            start_time=datetime.now(),
            status="running"
        )
        
        self.current_result = result
        self.evaluation_status = "running"
        self.evaluation_progress = 0.0
        
        return result
    
    def update_progress(self, progress: float, message: str = "") -> None:
        """
        Update evaluation progress.
        
        Args:
            progress: Progress percentage (0-100)
            message: Optional progress message
        """
        self.evaluation_progress = max(0, min(100, progress))
        self.progress_message = message
        
        if self.current_result:
            self.current_result.metadata["last_progress"] = progress
            self.current_result.metadata["last_message"] = message
    
    def complete_evaluation(self, metrics: Dict[str, float]) -> None:
        """
        Complete the current evaluation.
        
        Args:
            metrics: Final evaluation metrics
        """
        if not self.current_result:
            return
            
        self.current_result.end_time = datetime.now()
        self.current_result.status = "completed"
        self.current_result.metrics = metrics
        
        self.recent_results.insert(0, self.current_result)
        if len(self.recent_results) > 100:  # Keep last 100 results
            self.recent_results = self.recent_results[:100]
            
        self.evaluation_status = "completed"
        self.evaluation_progress = 100.0
    
    def fail_evaluation(self, error: str) -> None:
        """
        Mark current evaluation as failed.
        
        Args:
            error: Error message
        """
        if not self.current_result:
            return
            
        self.current_result.end_time = datetime.now()
        self.current_result.status = "failed"
        self.current_result.errors.append(error)
        
        self.recent_results.insert(0, self.current_result)
        self.evaluation_status = "error"
        self.errors.append(error)
    
    # Cost Management
    
    def estimate_cost(self) -> float:
        """
        Estimate cost for current configuration.
        
        Returns:
            Estimated cost in dollars
        """
        if not self.current_config:
            return 0.0
            
        # Simple estimation (will be replaced with actual logic)
        samples = self.current_config.max_samples or 100
        
        # Rough estimates per provider
        cost_per_sample = {
            "openai": 0.003,
            "anthropic": 0.004,
            "groq": 0.001,
            "ollama": 0.0,
        }.get(self.selected_model.provider if self.selected_model else "openai", 0.002)
        
        estimated = samples * cost_per_sample
        self.estimated_cost = estimated
        
        return estimated
    
    def update_actual_cost(self, cost: float) -> None:
        """
        Update actual cost spent.
        
        Args:
            cost: Cost in dollars
        """
        self.actual_cost += cost
        self.daily_spent += cost
        
        if self.daily_spent > self.daily_budget:
            self.warnings.append(f"Daily budget exceeded: ${self.daily_spent:.2f} / ${self.daily_budget:.2f}")
    
    # Storage
    
    def save_to_storage(self) -> None:
        """Save state to persistent storage."""
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            state_data = {
                "evaluation_status": self.evaluation_status,
                "recent_results": [asdict(r) for r in self.recent_results[:20]],
                "daily_spent": self.daily_spent,
                "daily_budget": self.daily_budget,
            }
            
            with open(self._storage_path, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
                
            logger.info("State saved to storage")
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def load_from_storage(self) -> None:
        """Load state from persistent storage."""
        try:
            if not self._storage_path.exists():
                return
                
            with open(self._storage_path, 'r') as f:
                state_data = json.load(f)
                
            self.evaluation_status = state_data.get("evaluation_status", "idle")
            self.daily_spent = state_data.get("daily_spent", 0.0)
            self.daily_budget = state_data.get("daily_budget", 100.0)
            
            # Load recent results
            for result_data in state_data.get("recent_results", []):
                # Convert datetime strings back to datetime objects
                result_data["start_time"] = datetime.fromisoformat(result_data["start_time"])
                if result_data.get("end_time"):
                    result_data["end_time"] = datetime.fromisoformat(result_data["end_time"])
                self.recent_results.append(EvaluationResult(**result_data))
                
            logger.info("State loaded from storage")
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
    
    def save_config(self, path: Path) -> None:
        """
        Save current configuration to file.
        
        Args:
            path: Path to save configuration
        """
        if not self.current_config:
            raise ValueError("No configuration to save")
            
        config_data = asdict(self.current_config)
        
        with open(path, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
            
        logger.info(f"Configuration saved to {path}")
    
    def load_config(self, path: Path) -> None:
        """
        Load configuration from file.
        
        Args:
            path: Path to load configuration from
        """
        with open(path, 'r') as f:
            config_data = json.load(f)
            
        # Reconstruct nested objects
        if config_data.get("task"):
            config_data["task"] = TaskConfig(**config_data["task"])
        if config_data.get("model"):
            config_data["model"] = ModelConfig(**config_data["model"])
        if config_data.get("dataset"):
            config_data["dataset"] = DatasetConfig(**config_data["dataset"])
            
        config_data["created_at"] = datetime.fromisoformat(config_data["created_at"])
        
        self.current_config = EvaluationConfig(**config_data)
        self.selected_task = self.current_config.task
        self.selected_model = self.current_config.model
        self.selected_dataset = self.current_config.dataset
        
        logger.info(f"Configuration loaded from {path}")
    
    def clear_errors(self) -> None:
        """Clear all errors."""
        self.errors = []
    
    def clear_warnings(self) -> None:
        """Clear all warnings."""
        self.warnings = []