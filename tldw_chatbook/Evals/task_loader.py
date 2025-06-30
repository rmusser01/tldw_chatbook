# task_loader.py
# Description: Evaluation task loader for multiple formats (Eleuther, custom)
#
"""
Task Loader for LLM Evaluations
-------------------------------

Supports loading and parsing evaluation tasks from various formats:
- Eleuther AI evaluation harness YAML format
- Custom JSON format
- HuggingFace datasets
- CSV/TSV files

Handles task configuration, dataset loading, and validation.
"""

import json
import yaml
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from loguru import logger

try:
    from datasets import load_dataset, Dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    logger.warning("HuggingFace datasets not available. Install with: pip install datasets")

class ValidationError(Exception):
    """Raised when task validation fails."""
    pass

@dataclass
class TaskConfig:
    """Standardized task configuration structure."""
    name: str
    description: str
    task_type: str  # 'question_answer', 'logprob', 'generation', 'classification'
    dataset_name: str
    dataset_config: Optional[str] = None
    split: str = 'test'
    few_shot_split: Optional[str] = None
    num_fewshot: int = 0
    
    # Generation parameters
    generation_kwargs: Dict[str, Any] = None
    stop_sequences: List[str] = None
    
    # Evaluation parameters
    metric: str = 'exact_match'
    filter_list: List[Dict[str, Any]] = None
    
    # Format templates
    doc_to_text: Optional[str] = None
    doc_to_target: Optional[str] = None
    doc_to_choice: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.generation_kwargs is None:
            self.generation_kwargs = {}
        if self.stop_sequences is None:
            self.stop_sequences = []
        if self.filter_list is None:
            self.filter_list = []
        if self.metadata is None:
            self.metadata = {}

class TaskLoadError(Exception):
    """Exception raised when task loading fails."""
    pass

class TaskLoader:
    """Loads and parses evaluation tasks from various formats."""
    
    def __init__(self):
        self.supported_formats = ['eleuther', 'custom', 'huggingface', 'csv']
        logger.info("TaskLoader initialized")
    
    def load_task(self, source: Union[str, Path, Dict], format_type: str = 'auto') -> TaskConfig:
        """
        Load a task from various sources.
        
        Args:
            source: Path to file, URL, or dict containing task config
            format_type: Format type ('eleuther', 'custom', 'huggingface', 'csv', 'auto')
            
        Returns:
            TaskConfig object
        """
        if format_type == 'auto':
            format_type = self._detect_format(source)
        
        if format_type not in self.supported_formats:
            raise TaskLoadError(f"Unsupported format: {format_type}")
        
        logger.info(f"Loading task from {source} with format {format_type}")
        
        if format_type == 'eleuther':
            return self._load_eleuther_task(source)
        elif format_type == 'custom':
            return self._load_custom_task(source)
        elif format_type == 'huggingface':
            return self._load_huggingface_task(source)
        elif format_type == 'csv':
            return self._load_csv_task(source)
        else:
            raise TaskLoadError(f"Format {format_type} not implemented")
    
    def _detect_format(self, source: Union[str, Path, Dict]) -> str:
        """Auto-detect the format of the task source."""
        if isinstance(source, dict):
            # Check for Eleuther-style keys
            if any(key in source for key in ['dataset_name', 'dataset_path', 'test_split']):
                return 'eleuther'
            return 'custom'
        
        if isinstance(source, (str, Path)):
            path = Path(source)
            
            if path.suffix.lower() in ['.yaml', '.yml']:
                # Try to peek inside to distinguish eleuther vs custom
                try:
                    with open(path, 'r') as f:
                        content = yaml.safe_load(f)
                    if isinstance(content, dict) and any(key in content for key in 
                                                       ['dataset_name', 'dataset_path', 'test_split']):
                        return 'eleuther'
                    return 'custom'
                except Exception:
                    return 'eleuther'  # Default assumption
            
            elif path.suffix.lower() == '.json':
                return 'custom'
            elif path.suffix.lower() in ['.csv', '.tsv']:
                return 'csv'
        
        # If it looks like a HuggingFace dataset path
        if isinstance(source, str) and '/' in source and not Path(source).exists():
            return 'huggingface'
        
        return 'custom'  # Default fallback
    
    def _load_eleuther_task(self, source: Union[str, Path, Dict]) -> TaskConfig:
        """Load task in Eleuther AI evaluation harness format."""
        if isinstance(source, dict):
            config_data = source
        else:
            path = Path(source)
            if not path.exists():
                raise TaskLoadError(f"Eleuther task file not found: {path}")
            
            with open(path, 'r') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
        
        # Extract required fields
        try:
            name = config_data.get('task', path.stem if isinstance(source, Path) else 'unnamed_task')
            dataset_name = config_data.get('dataset_name') or config_data.get('dataset_path', '')
            
            # Determine task type from Eleuther config
            task_type = self._infer_task_type_from_eleuther(config_data)
            
            # Extract generation parameters
            generation_kwargs = {}
            if 'generation_kwargs' in config_data:
                generation_kwargs = config_data['generation_kwargs']
            
            # Extract stop sequences
            stop_sequences = []
            if 'output_type' in config_data and config_data['output_type'] == 'generate_until':
                stop_sequences = config_data.get('until', [])
            
            # Extract metric information
            metric = 'exact_match'  # Default
            if 'metric_list' in config_data:
                metrics = config_data['metric_list']
                if metrics and isinstance(metrics[0], dict):
                    metric = metrics[0].get('metric', 'exact_match')
                elif metrics:
                    metric = metrics[0]
            
            # Extract filter information
            filter_list = config_data.get('filter_list', [])
            
            task_config = TaskConfig(
                name=name,
                description=config_data.get('description', f"Eleuther task: {name}"),
                task_type=task_type,
                dataset_name=dataset_name,
                dataset_config=config_data.get('dataset_config_name'),
                split=config_data.get('test_split', 'test'),
                few_shot_split=config_data.get('fewshot_split', 'dev'),
                num_fewshot=config_data.get('num_fewshot', 0),
                generation_kwargs=generation_kwargs,
                stop_sequences=stop_sequences,
                metric=metric,
                filter_list=filter_list,
                doc_to_text=config_data.get('doc_to_text'),
                doc_to_target=config_data.get('doc_to_target'),
                doc_to_choice=config_data.get('doc_to_choice'),
                metadata={
                    'original_config': config_data,
                    'format': 'eleuther'
                }
            )
            
            logger.info(f"Loaded Eleuther task: {name}")
            return task_config
            
        except KeyError as e:
            raise TaskLoadError(f"Missing required field in Eleuther config: {e}")
        except Exception as e:
            raise TaskLoadError(f"Failed to parse Eleuther config: {e}")
    
    def _infer_task_type_from_eleuther(self, config: Dict[str, Any]) -> str:
        """Infer task type from Eleuther configuration."""
        # Check output type
        output_type = config.get('output_type', '')
        
        if output_type == 'loglikelihood':
            return 'logprob'
        elif output_type == 'generate_until':
            return 'generation'
        elif output_type == 'multiple_choice':
            return 'classification'
        
        # Check for common patterns
        if config.get('doc_to_choice') or 'choice' in str(config).lower():
            return 'classification'
        
        # Default assumption
        return 'question_answer'
    
    def _load_custom_task(self, source: Union[str, Path, Dict]) -> TaskConfig:
        """Load task in custom JSON format."""
        if isinstance(source, dict):
            config_data = source
        else:
            path = Path(source)
            if not path.exists():
                raise TaskLoadError(f"Custom task file not found: {path}")
            
            with open(path, 'r') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
        
        try:
            # Custom format should be more direct
            task_config = TaskConfig(
                name=config_data['name'],
                description=config_data.get('description', ''),
                task_type=config_data['task_type'],
                dataset_name=config_data['dataset_name'],
                dataset_config=config_data.get('dataset_config'),
                split=config_data.get('split', 'test'),
                few_shot_split=config_data.get('few_shot_split'),
                num_fewshot=config_data.get('num_fewshot', 0),
                generation_kwargs=config_data.get('generation_kwargs', {}),
                stop_sequences=config_data.get('stop_sequences', []),
                metric=config_data.get('metric', 'exact_match'),
                filter_list=config_data.get('filter_list', []),
                doc_to_text=config_data.get('doc_to_text'),
                doc_to_target=config_data.get('doc_to_target'),
                doc_to_choice=config_data.get('doc_to_choice'),
                metadata=config_data.get('metadata', {})
            )
            
            task_config.metadata['format'] = 'custom'
            
            logger.info(f"Loaded custom task: {task_config.name}")
            return task_config
            
        except KeyError as e:
            raise TaskLoadError(f"Missing required field in custom config: {e}")
        except Exception as e:
            raise TaskLoadError(f"Failed to parse custom config: {e}")
    
    def _load_huggingface_task(self, source: str) -> TaskConfig:
        """Load task from HuggingFace dataset."""
        if not HF_DATASETS_AVAILABLE:
            raise TaskLoadError("HuggingFace datasets not available. Install with: pip install datasets")
        
        try:
            # Parse dataset path and config
            parts = source.split(':')
            dataset_name = parts[0]
            dataset_config = parts[1] if len(parts) > 1 else None
            
            # Try to load dataset info
            dataset = load_dataset(dataset_name, dataset_config, split='train[:1]')  # Load minimal sample
            
            # Infer task type from dataset structure
            task_type = self._infer_task_type_from_dataset(dataset)
            
            task_config = TaskConfig(
                name=f"hf_{dataset_name.replace('/', '_')}",
                description=f"HuggingFace dataset: {dataset_name}",
                task_type=task_type,
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                split='test',
                metadata={'format': 'huggingface', 'dataset_info': str(dataset.info) if hasattr(dataset, 'info') else ''}
            )
            
            logger.info(f"Loaded HuggingFace task: {dataset_name}")
            return task_config
            
        except Exception as e:
            raise TaskLoadError(f"Failed to load HuggingFace dataset {source}: {e}")
    
    def _infer_task_type_from_dataset(self, dataset: 'Dataset') -> str:
        """Infer task type from dataset structure."""
        if len(dataset) == 0:
            return 'question_answer'
        
        sample = dataset[0]
        
        # Check for common field names
        if any(field in sample for field in ['choices', 'options', 'answers']):
            return 'classification'
        elif any(field in sample for field in ['question', 'input', 'prompt']):
            return 'question_answer'
        
        return 'question_answer'  # Default
    
    def _load_csv_task(self, source: Union[str, Path]) -> TaskConfig:
        """Load task from CSV/TSV file."""
        path = Path(source)
        if not path.exists():
            raise TaskLoadError(f"CSV file not found: {path}")
        
        delimiter = '\t' if path.suffix.lower() == '.tsv' else ','
        
        try:
            with open(path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                rows = list(reader)
            
            if not rows:
                raise TaskLoadError("CSV file is empty")
            
            # Infer task type from columns
            columns = set(rows[0].keys())
            task_type = self._infer_task_type_from_columns(columns)
            
            task_config = TaskConfig(
                name=path.stem,
                description=f"CSV task from {path.name}",
                task_type=task_type,
                dataset_name=str(path),
                split='all',  # CSV files don't have built-in splits
                metadata={
                    'format': 'csv',
                    'columns': list(columns),
                    'num_samples': len(rows)
                }
            )
            
            logger.info(f"Loaded CSV task: {path.stem} ({len(rows)} samples)")
            return task_config
            
        except Exception as e:
            raise TaskLoadError(f"Failed to load CSV file {path}: {e}")
    
    def _infer_task_type_from_columns(self, columns: set) -> str:
        """Infer task type from CSV column names."""
        columns_lower = {col.lower() for col in columns}
        
        if any(col in columns_lower for col in ['choices', 'options', 'answers', 'labels']):
            return 'classification'
        elif any(col in columns_lower for col in ['question', 'query', 'input', 'prompt']):
            return 'question_answer'
        
        return 'question_answer'  # Default
    
    def validate_task(self, task_config: TaskConfig) -> List[str]:
        """
        Validate a task configuration and return list of issues.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []
        
        # Required fields
        if not task_config.name or not task_config.name.strip():
            issues.append("Task name is required")
        
        if not task_config.dataset_name:
            issues.append("Dataset name is required")
        
        if task_config.task_type not in ['question_answer', 'logprob', 'generation', 'classification']:
            issues.append(f"Invalid task_type: {task_config.task_type}")
        
        # Logical validations
        if task_config.num_fewshot > 0 and not task_config.few_shot_split:
            issues.append("few_shot_split required when num_fewshot > 0")
        
        if task_config.num_fewshot < 0:
            issues.append("num_fewshot cannot be negative")
        
        # Format-specific validations
        if task_config.task_type == 'classification' and not task_config.doc_to_choice:
            issues.append("doc_to_choice required for classification tasks")
        
        return issues
    
    def create_task_from_template(self, template_name: str, **kwargs) -> TaskConfig:
        """Create a task from a built-in template."""
        # Import here to avoid circular imports
        from .eval_templates import get_eval_templates
        
        template_manager = get_eval_templates()
        
        # Try to get from extended templates first
        try:
            return template_manager.create_task_config(template_name, **kwargs)
        except ValueError:
            pass
        
        # Fallback to basic templates
        basic_templates = {
            'simple_qa': {
                'name': 'Simple Q&A',
                'description': 'Simple question answering task',
                'task_type': 'question_answer',
                'dataset_name': 'custom',
                'metric': 'exact_match'
            },
            'multiple_choice': {
                'name': 'Multiple Choice',
                'description': 'Multiple choice classification task',
                'task_type': 'classification',
                'dataset_name': 'custom',
                'metric': 'accuracy'
            },
            'text_generation': {
                'name': 'Text Generation',
                'description': 'Text generation task',
                'task_type': 'generation',
                'dataset_name': 'custom',
                'metric': 'bleu',
                'generation_kwargs': {'max_length': 100, 'temperature': 0.7}
            }
        }
        
        if template_name not in basic_templates:
            raise TaskLoadError(f"Unknown template: {template_name}")
        
        template = basic_templates[template_name].copy()
        template.update(kwargs)
        
        return TaskConfig(**template)
    
    def list_available_templates(self) -> List[Dict[str, Any]]:
        """List all available evaluation templates."""
        from .eval_templates import get_eval_templates
        
        template_manager = get_eval_templates()
        return template_manager.list_templates()
    
    def export_task(self, task_config: TaskConfig, output_path: Union[str, Path], 
                   format_type: str = 'custom') -> None:
        """Export task configuration to file."""
        path = Path(output_path)
        
        if format_type == 'custom':
            data = asdict(task_config)
            
            if path.suffix.lower() in ['.yaml', '.yml']:
                with open(path, 'w') as f:
                    yaml.dump(data, f, default_flow_style=False)
            else:
                with open(path, 'w') as f:
                    json.dump(data, f, indent=2)
        
        elif format_type == 'eleuther':
            # Convert to Eleuther format
            eleuther_config = self._convert_to_eleuther_format(task_config)
            
            with open(path, 'w') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(eleuther_config, f, default_flow_style=False)
                else:
                    json.dump(eleuther_config, f, indent=2)
        
        else:
            raise TaskLoadError(f"Export format {format_type} not supported")
        
        logger.info(f"Exported task to {path}")
    
    def _convert_to_eleuther_format(self, task_config: TaskConfig) -> Dict[str, Any]:
        """Convert TaskConfig to Eleuther format."""
        config = {
            'task': task_config.name,
            'dataset_name': task_config.dataset_name,
            'test_split': task_config.split,
            'num_fewshot': task_config.num_fewshot
        }
        
        if task_config.dataset_config:
            config['dataset_config_name'] = task_config.dataset_config
        
        if task_config.few_shot_split:
            config['fewshot_split'] = task_config.few_shot_split
        
        if task_config.description:
            config['description'] = task_config.description
        
        # Convert task type to output type
        if task_config.task_type == 'logprob':
            config['output_type'] = 'loglikelihood'
        elif task_config.task_type == 'generation':
            config['output_type'] = 'generate_until'
            if task_config.stop_sequences:
                config['until'] = task_config.stop_sequences
        elif task_config.task_type == 'classification':
            config['output_type'] = 'multiple_choice'
        
        if task_config.generation_kwargs:
            config['generation_kwargs'] = task_config.generation_kwargs
        
        if task_config.filter_list:
            config['filter_list'] = task_config.filter_list
        
        config['metric_list'] = [task_config.metric]
        
        if task_config.doc_to_text:
            config['doc_to_text'] = task_config.doc_to_text
        if task_config.doc_to_target:
            config['doc_to_target'] = task_config.doc_to_target
        if task_config.doc_to_choice:
            config['doc_to_choice'] = task_config.doc_to_choice
        
        return config