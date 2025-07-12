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
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from loguru import logger
from tldw_chatbook.Metrics.metrics_logger import log_counter, log_histogram

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
        start_time = time.time()
        
        if format_type == 'auto':
            format_type = self._detect_format(source)
        
        if format_type not in self.supported_formats:
            raise TaskLoadError(f"Unsupported format: {format_type}")
        
        logger.info(f"Loading task from {source} with format {format_type}")
        
        # Log task load attempt
        log_counter("eval_task_load_attempt", labels={
            "format_type": format_type,
            "source_type": "dict" if isinstance(source, dict) else "file"
        })
        
        try:
            if format_type == 'eleuther':
                task_config = self._load_eleuther_task(source)
            elif format_type == 'custom':
                task_config = self._load_custom_task(source)
            elif format_type == 'huggingface':
                task_config = self._load_huggingface_task(source)
            elif format_type == 'csv':
                task_config = self._load_csv_task(source)
            else:
                raise TaskLoadError(f"Format {format_type} not implemented")
            
            # Log successful load
            duration = time.time() - start_time
            log_histogram("eval_task_load_duration", duration, labels={
                "format_type": format_type,
                "task_type": task_config.task_type
            })
            log_counter("eval_task_load_success", labels={
                "format_type": format_type,
                "task_type": task_config.task_type
            })
            
            return task_config
            
        except Exception as e:
            duration = time.time() - start_time
            log_counter("eval_task_load_error", labels={
                "format_type": format_type,
                "error_type": type(e).__name__
            })
            log_histogram("eval_task_load_error_duration", duration, labels={
                "format_type": format_type
            })
            raise
    
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
        parse_start = time.time()
        
        if isinstance(source, dict):
            config_data = source
        else:
            path = Path(source)
            if not path.exists():
                raise TaskLoadError(f"Eleuther task file not found: {path}")
            
            try:
                with open(path, 'r') as f:
                    if path.suffix.lower() in ['.yaml', '.yml']:
                        config_data = yaml.safe_load(f)
                        file_format = "yaml"
                    else:
                        config_data = json.load(f)
                        file_format = "json"
                
                # Log file parsing metrics
                parse_duration = time.time() - parse_start
                log_histogram("eval_task_file_parse_duration", parse_duration, labels={
                    "format": "eleuther",
                    "file_format": file_format
                })
                log_histogram("eval_task_file_size", path.stat().st_size, labels={
                    "format": "eleuther",
                    "file_format": file_format
                })
                
            except (json.JSONDecodeError, yaml.YAMLError) as e:
                parse_duration = time.time() - parse_start
                log_counter("eval_task_file_parse_error", labels={
                    "format": "eleuther",
                    "error_type": type(e).__name__
                })
                raise TaskLoadError(f"Failed to parse Eleuther task file {path}: {e}")
        
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
            return 'generate_until'
        elif output_type == 'multiple_choice':
            return 'classification'  # Map multiple_choice to classification
        
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
            
            try:
                with open(path, 'r') as f:
                    if path.suffix.lower() in ['.yaml', '.yml']:
                        config_data = yaml.safe_load(f)
                    else:
                        config_data = json.load(f)
            except (json.JSONDecodeError, yaml.YAMLError) as e:
                raise TaskLoadError(f"Failed to parse task file {path}: {e}")
        
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
        
        # Check if source is a file path containing HuggingFace config
        path = Path(source)
        if path.exists() and path.is_file():
            # Load configuration from file
            try:
                with open(path, 'r') as f:
                    if path.suffix.lower() in ['.yaml', '.yml']:
                        config_data = yaml.safe_load(f)
                    else:
                        config_data = json.load(f)
                
                # Extract configuration
                task_config = TaskConfig(
                    name=config_data.get('name', f"HF Task {path.stem}"),
                    description=config_data.get('description', ''),
                    task_type=config_data.get('task_type', 'question_answer'),
                    dataset_name=config_data.get('dataset_name', ''),
                    dataset_config=config_data.get('dataset_config'),
                    split=config_data.get('split', 'test'),
                    metric=config_data.get('metric', 'exact_match'),
                    metadata={'format': 'huggingface', 'source': str(path)}
                )
                
                logger.info(f"Loaded HuggingFace task from file: {path}")
                return task_config
                
            except Exception as e:
                raise TaskLoadError(f"Failed to load HuggingFace config from {path}: {e}")
        
        # Otherwise, treat as HuggingFace dataset identifier
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
    
    
    def create_task_from_template(self, template_name: str, **kwargs) -> TaskConfig:
        """Create a task from a built-in template."""
        start_time = time.time()
        
        # Log template creation attempt
        log_counter("eval_task_template_create_attempt", labels={
            "template_name": template_name
        })
        
        # Import here to avoid circular imports
        from .eval_templates import get_eval_templates
        
        template_manager = get_eval_templates()
        
        # Try to get from extended templates first
        try:
            task_config = template_manager.create_task_config(template_name, **kwargs)
            
            # Log successful template creation
            duration = time.time() - start_time
            log_histogram("eval_task_template_create_duration", duration, labels={
                "template_name": template_name,
                "task_type": task_config.task_type
            })
            log_counter("eval_task_template_create_success", labels={
                "template_name": template_name,
                "task_type": task_config.task_type
            })
            
            return task_config
            
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
            'question_answer': {  # Alias for simple_qa
                'name': 'Question Answering',
                'description': 'Question answering task',
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
            },
            'code_generation': {
                'name': 'Code Generation',
                'description': 'Code generation task',
                'task_type': 'code_generation',
                'dataset_name': 'custom',
                'metric': 'execution_pass_rate',
                'generation_kwargs': {'max_length': 200, 'temperature': 0.2, 'language': 'python'}
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
        start_time = time.time()
        path = Path(output_path)
        
        # Log export attempt
        log_counter("eval_task_export_attempt", labels={
            "format_type": format_type,
            "task_type": task_config.task_type
        })
        
        try:
            if format_type == 'custom':
                data = asdict(task_config)
                
                if path.suffix.lower() in ['.yaml', '.yml']:
                    with open(path, 'w') as f:
                        yaml.dump(data, f, default_flow_style=False)
                    file_format = "yaml"
                else:
                    with open(path, 'w') as f:
                        json.dump(data, f, indent=2)
                    file_format = "json"
            
            elif format_type == 'eleuther':
                # Convert to Eleuther format
                eleuther_config = self._convert_to_eleuther_format(task_config)
                
                with open(path, 'w') as f:
                    if path.suffix.lower() in ['.yaml', '.yml']:
                        yaml.dump(eleuther_config, f, default_flow_style=False)
                        file_format = "yaml"
                    else:
                        json.dump(eleuther_config, f, indent=2)
                        file_format = "json"
            
            else:
                raise TaskLoadError(f"Export format {format_type} not supported")
            
            # Log successful export
            duration = time.time() - start_time
            file_size = path.stat().st_size
            log_histogram("eval_task_export_duration", duration, labels={
                "format_type": format_type,
                "file_format": file_format,
                "task_type": task_config.task_type
            })
            log_histogram("eval_task_export_file_size", file_size, labels={
                "format_type": format_type,
                "file_format": file_format
            })
            log_counter("eval_task_export_success", labels={
                "format_type": format_type,
                "task_type": task_config.task_type
            })
            
            logger.info(f"Exported task to {path}")
            
        except Exception as e:
            duration = time.time() - start_time
            log_counter("eval_task_export_error", labels={
                "format_type": format_type,
                "error_type": type(e).__name__
            })
            log_histogram("eval_task_export_error_duration", duration, labels={
                "format_type": format_type
            })
            raise
    
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
    
    def convert_to_custom_format(self, task_config: TaskConfig) -> Dict[str, Any]:
        """Convert TaskConfig to custom format dict."""
        return asdict(task_config)
    
    def convert_to_eleuther_format(self, task_config: TaskConfig) -> Dict[str, Any]:
        """Convert TaskConfig to Eleuther format dict."""
        return self._convert_to_eleuther_format(task_config)
    
    def generate_template(self, template_type: str, **kwargs) -> Dict[str, Any]:
        """Generate a task template of the specified type."""
        task_config = self.create_task_from_template(template_type, **kwargs)
        return asdict(task_config)
    
    def generate_eleuther_template(self, task_type: str) -> Dict[str, Any]:
        """Generate an Eleuther format template for the specified task type."""
        # Create a basic task config
        task_config = self.create_task_from_template(task_type)
        # Convert to Eleuther format
        return self._convert_to_eleuther_format(task_config)
    
    def _detect_file_format(self, path: str) -> str:
        """Detect the format of a task configuration file."""
        p = Path(path)
        
        # For CSV/TSV, just check extension
        if p.suffix.lower() in ['.csv', '.tsv']:
            return 'csv'
        
        # For YAML/JSON, check content
        try:
            with open(p, 'r') as f:
                if p.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                    # Check for Eleuther format indicators
                    if 'task' in data or 'output_type' in data or 'dataset_name' in data:
                        return 'eleuther'
                    return 'custom'
                else:  # JSON
                    data = json.load(f)
                    # JSON files with 'name' field are custom format
                    if 'name' in data and 'task_type' in data:
                        return 'custom'
                    # Otherwise could be eleuther
                    if 'task' in data or 'output_type' in data:
                        return 'eleuther'
                    return 'custom'
        except:
            # If we can't parse, default to custom
            return 'custom'
    
    def _normalize_task_type(self, task_type: str) -> str:
        """Normalize task type string."""
        # Map common variations to standard types
        type_map = {
            'qa': 'question_answer',
            'q&a': 'question_answer',
            'question-answer': 'question_answer',
            'multiple choice': 'multiple_choice',
            'multiple-choice': 'multiple_choice',
            'mc': 'multiple_choice',
            'classification': 'multiple_choice',
            'generation': 'generation',
            'text_generation': 'generation',
            'text-generation': 'generation',
            'code': 'code_generation',
            'coding': 'code_generation',
            'code-generation': 'code_generation',
            'generate-until': 'generate_until',
            'generate_until': 'generate_until'
        }
        # Also handle generic case where hyphen should be replaced with underscore
        normalized = type_map.get(task_type.lower(), task_type.lower())
        # If not in map, try replacing hyphens with underscores
        if normalized == task_type.lower() and '-' in normalized:
            normalized = normalized.replace('-', '_')
        return normalized
    
    def _validate_dataset_path(self, path: str) -> bool:
        """Validate that a dataset path exists and is accessible."""
        if not path:
            return False
        
        # Check if it's a HuggingFace dataset path (contains /)
        if '/' in path and not path.startswith('/') and not path.startswith('./'):
            # Looks like a HuggingFace dataset identifier (e.g., "huggingface/dataset")
            return True
        
        # Check if it's a valid file path format
        # Accept any non-empty path that doesn't contain invalid characters
        invalid_chars = ['<', '>', '|', '?', '*', '\0']
        if any(char in path for char in invalid_chars):
            return False
        
        # For local files, check if they exist (but also accept if they just look valid)
        p = Path(path)
        # If it exists, it must be a file
        if p.exists():
            return p.is_file()
        
        # If it doesn't exist, accept if it has a valid file extension
        valid_extensions = ['.csv', '.tsv', '.json', '.jsonl', '.txt', '.parquet']
        return any(path.lower().endswith(ext) for ext in valid_extensions)
    
    def _merge_generation_kwargs(self, defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Merge generation kwargs, with overrides taking precedence."""
        merged = defaults.copy()
        merged.update(overrides)
        return merged
    
    def validate_task(self, task_config: TaskConfig) -> List[str]:
        """Validate a task configuration and return list of issues."""
        start_time = time.time()
        issues = []
        
        # Validate required fields
        if not task_config.name or not task_config.name.strip():
            issues.append("Task name cannot be empty")
        
        if not task_config.dataset_name or not task_config.dataset_name.strip():
            issues.append("Dataset name cannot be empty")
        
        if not task_config.metric or not task_config.metric.strip():
            issues.append("Metric cannot be empty")
        
        # Validate task type
        valid_task_types = [
            'question_answer', 'multiple_choice', 'classification', 
            'generation', 'logprob', 'code_generation', 'generate_until'
        ]
        if task_config.task_type not in valid_task_types:
            issues.append(f"Invalid task type: {task_config.task_type}")
        
        # Validate generation kwargs if present
        if task_config.generation_kwargs:
            if 'temperature' in task_config.generation_kwargs:
                temp = task_config.generation_kwargs['temperature']
                if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                    issues.append("Temperature must be a number between 0 and 2")
            
            if 'max_length' in task_config.generation_kwargs:
                max_len = task_config.generation_kwargs['max_length']
                if not isinstance(max_len, int) or max_len <= 0:
                    issues.append("max_length must be a positive integer")
        
        # Validate code task requirements
        if task_config.task_type == 'code_generation':
            if not task_config.generation_kwargs:
                issues.append("Code generation tasks must specify generation_kwargs")
            elif 'language' not in task_config.generation_kwargs:
                issues.append("Code generation tasks must specify language in generation_kwargs")
        
        # Log validation metrics
        duration = time.time() - start_time
        log_histogram("eval_task_validation_duration", duration, labels={
            "task_type": task_config.task_type
        })
        log_histogram("eval_task_validation_issue_count", len(issues), labels={
            "task_type": task_config.task_type
        })
        if len(issues) > 0:
            log_counter("eval_task_validation_failed", labels={
                "task_type": task_config.task_type,
                "issue_count": str(len(issues))
            })
        else:
            log_counter("eval_task_validation_passed", labels={
                "task_type": task_config.task_type
            })
        
        return issues