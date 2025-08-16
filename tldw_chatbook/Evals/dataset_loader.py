# dataset_loader.py
# Description: Dataset loading utilities for evaluation tasks
#
"""
Dataset Loader
--------------

Loads evaluation datasets from various sources including local files and HuggingFace.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

from .base_runner import EvalSample
from .task_loader import TaskConfig
from .eval_errors import (
    get_error_handler, EvaluationError, DatasetLoadingError,
    ErrorContext, ErrorCategory, ErrorSeverity
)

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False


class DatasetLoader:
    """Loads datasets from various sources."""
    
    @staticmethod
    def load_dataset_samples(
        task_config: TaskConfig,
        split: str = None,
        max_samples: int = None
    ) -> List[EvalSample]:
        """
        Load samples from dataset based on task configuration.
        
        Args:
            task_config: Task configuration with dataset information
            split: Dataset split to load (train/test/val)
            max_samples: Maximum number of samples to load
            
        Returns:
            List of evaluation samples
        """
        error_handler = get_error_handler()
        
        try:
            if split is None:
                split = task_config.split
            
            dataset_name = task_config.dataset_name
            
            # Validate dataset name
            if not dataset_name:
                raise DatasetLoadingError.missing_required_fields(['dataset_name'])
            
            # Handle different dataset sources
            if Path(dataset_name).exists():
                return DatasetLoader._load_local_dataset(dataset_name, max_samples)
            elif HF_DATASETS_AVAILABLE and '/' in dataset_name:
                return DatasetLoader._load_huggingface_dataset(task_config, split, max_samples)
            else:
                raise DatasetLoadingError(ErrorContext(
                    category=ErrorCategory.DATASET_LOADING,
                    severity=ErrorSeverity.ERROR,
                    message=f"Cannot determine dataset type for: {dataset_name}",
                    suggestion="Provide a valid local file path or HuggingFace dataset name (format: 'owner/dataset')",
                    is_retryable=False
                ))
                
        except EvaluationError:
            raise
        except Exception as e:
            error_context = error_handler.handle_error(e, {
                'dataset_name': dataset_name,
                'split': split,
                'max_samples': max_samples
            })
            raise EvaluationError(error_context, e)
    
    @staticmethod
    def _load_local_dataset(dataset_path: str, max_samples: int = None) -> List[EvalSample]:
        """Load dataset from local file."""
        path = Path(dataset_path)
        
        # Check file exists
        if not path.exists():
            raise DatasetLoadingError.file_not_found(str(path))
        
        # Check file is readable
        if not path.is_file():
            raise DatasetLoadingError(ErrorContext(
                category=ErrorCategory.DATASET_LOADING,
                severity=ErrorSeverity.ERROR,
                message=f"Path is not a file: {path}",
                suggestion="Provide a path to a valid dataset file",
                is_retryable=False
            ))
        
        # Check file size
        file_size = path.stat().st_size
        if file_size == 0:
            raise DatasetLoadingError(ErrorContext(
                category=ErrorCategory.DATASET_LOADING,
                severity=ErrorSeverity.ERROR,
                message=f"Dataset file is empty: {path}",
                suggestion="Ensure the dataset file contains data",
                is_retryable=False
            ))
        
        if file_size > 1_000_000_000:  # 1GB warning
            logger.warning(f"Large dataset file ({file_size / 1_000_000:.1f} MB): {path}")
        
        # Load based on extension
        try:
            if path.suffix.lower() == '.json':
                return DatasetLoader._load_json_dataset(path, max_samples)
            elif path.suffix.lower() in ['.csv', '.tsv']:
                return DatasetLoader._load_csv_dataset(path, max_samples)
            else:
                raise DatasetLoadingError(ErrorContext(
                    category=ErrorCategory.DATASET_LOADING,
                    severity=ErrorSeverity.ERROR,
                    message=f"Unsupported file format: {path.suffix}",
                    suggestion="Use JSON (.json) or CSV/TSV (.csv, .tsv) format",
                    is_retryable=False
                ))
        except EvaluationError:
            raise
        except Exception as e:
            raise DatasetLoadingError.invalid_format(str(path), str(e))
    
    @staticmethod
    def _load_json_dataset(path: Path, max_samples: int = None) -> List[EvalSample]:
        """Load JSON dataset."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise DatasetLoadingError.invalid_format(
                str(path), 
                f"Invalid JSON at line {e.lineno}, column {e.colno}: {e.msg}"
            )
        except UnicodeDecodeError as e:
            raise DatasetLoadingError(ErrorContext(
                category=ErrorCategory.DATASET_LOADING,
                severity=ErrorSeverity.ERROR,
                message=f"File encoding error in {path}",
                details=str(e),
                suggestion="Ensure the file is UTF-8 encoded",
                is_retryable=False
            ))
        
        if not isinstance(data, list):
            raise DatasetLoadingError.invalid_format(
                str(path),
                "JSON file must contain an array of samples at the root level"
            )
        
        if len(data) == 0:
            raise DatasetLoadingError(ErrorContext(
                category=ErrorCategory.DATASET_LOADING,
                severity=ErrorSeverity.ERROR,
                message=f"No samples found in {path}",
                suggestion="Add sample data to the JSON array",
                is_retryable=False
            ))
        
        samples = []
        errors = []
        
        for i, item in enumerate(data[:max_samples] if max_samples else data):
            try:
                # Validate required fields
                if not isinstance(item, dict):
                    errors.append(f"Sample {i}: Not a JSON object")
                    continue
                
                sample_id = item.get('id', str(i))
                input_text = item.get('input', item.get('question', item.get('text', '')))
                
                if not input_text:
                    errors.append(f"Sample {sample_id}: Missing input text (checked 'input', 'question', 'text' fields)")
                    continue
                
                expected_output = item.get('output', item.get('answer', item.get('target')))
                choices = item.get('choices', item.get('options'))
                
                samples.append(EvalSample(
                    id=sample_id,
                    input_text=input_text,
                    expected_output=expected_output,
                    choices=choices,
                    metadata=item
                ))
                
            except Exception as e:
                errors.append(f"Sample {i}: {str(e)}")
        
        # Report errors if any samples failed
        if errors:
            error_summary = "\n".join(errors[:10])  # Show first 10 errors
            if len(errors) > 10:
                error_summary += f"\n... and {len(errors) - 10} more errors"
            
            logger.warning(f"Failed to load {len(errors)} samples from {path}:\n{error_summary}")
        
        if not samples:
            raise DatasetLoadingError(ErrorContext(
                category=ErrorCategory.DATASET_LOADING,
                severity=ErrorSeverity.ERROR,
                message=f"No valid samples could be loaded from {path}",
                details=error_summary if errors else None,
                suggestion="Fix the sample format errors and try again",
                is_retryable=False
            ))
        
        return samples
    
    @staticmethod
    def _load_csv_dataset(path: Path, max_samples: int = None) -> List[EvalSample]:
        """Load CSV/TSV dataset."""
        delimiter = '\t' if path.suffix.lower() == '.tsv' else ','
        
        try:
            with open(path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                rows = list(reader)
        except Exception as e:
            raise DatasetLoadingError.invalid_format(str(path), f"CSV read error: {str(e)}")
        
        if not rows:
            raise DatasetLoadingError(ErrorContext(
                category=ErrorCategory.DATASET_LOADING,
                severity=ErrorSeverity.ERROR,
                message=f"No data rows found in {path}",
                suggestion="Ensure the CSV file contains data rows after the header",
                is_retryable=False
            ))
        
        samples = []
        for i, row in enumerate(rows[:max_samples] if max_samples else rows):
            sample_id = row.get('id', str(i))
            
            # Try common column names for input
            input_text = (row.get('input') or row.get('question') or 
                         row.get('text') or row.get('prompt', ''))
            
            # Try common column names for expected output
            expected_output = (row.get('output') or row.get('answer') or 
                             row.get('target') or row.get('label'))
            
            # Handle choices for multiple choice
            choices = None
            choice_keys = [k for k in row.keys() if k.startswith('choice') or k.startswith('option')]
            if choice_keys:
                choices = [row[k] for k in sorted(choice_keys) if row[k]]
            
            samples.append(EvalSample(
                id=sample_id,
                input_text=input_text,
                expected_output=expected_output,
                choices=choices,
                metadata=dict(row)
            ))
        
        return samples
    
    @staticmethod
    def _load_huggingface_dataset(
        task_config: TaskConfig,
        split: str,
        max_samples: int = None
    ) -> List[EvalSample]:
        """Load HuggingFace dataset."""
        if not HF_DATASETS_AVAILABLE:
            raise ImportError("HuggingFace datasets not available. Install with: pip install datasets")
        
        try:
            dataset = load_dataset(
                task_config.dataset_name,
                task_config.dataset_config,
                split=split
            )
            
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            samples = []
            for i, item in enumerate(dataset):
                sample_id = item.get('id', str(i))
                
                # Apply doc_to_text template if provided
                if task_config.doc_to_text:
                    input_text = DatasetLoader._apply_template(task_config.doc_to_text, item)
                else:
                    input_text = item.get('input', item.get('question', item.get('text', str(item))))
                
                # Apply doc_to_target template if provided
                expected_output = None
                if task_config.doc_to_target:
                    expected_output = DatasetLoader._apply_template(task_config.doc_to_target, item)
                else:
                    expected_output = item.get('output', item.get('answer', item.get('target')))
                
                # Handle choices for classification tasks
                choices = None
                if task_config.doc_to_choice:
                    choices_text = DatasetLoader._apply_template(task_config.doc_to_choice, item)
                    choices = choices_text.split('\n') if choices_text else None
                elif 'choices' in item:
                    choices = item['choices']
                
                samples.append(EvalSample(
                    id=sample_id,
                    input_text=input_text,
                    expected_output=expected_output,
                    choices=choices,
                    metadata=dict(item)
                ))
            
            return samples
            
        except Exception as e:
            raise DatasetLoadingError(ErrorContext(
                category=ErrorCategory.DATASET_LOADING,
                severity=ErrorSeverity.ERROR,
                message=f"Failed to load HuggingFace dataset {task_config.dataset_name}",
                details=str(e),
                suggestion="Check the dataset name and ensure you have internet access",
                is_retryable=True
            ))
    
    @staticmethod
    def _apply_template(template: str, item: Dict[str, Any]) -> str:
        """Apply template to dataset item."""
        try:
            # Check if this is a Jinja2-style template
            if '{{' in template and '}}' in template:
                # Use Jinja2 for template rendering
                from jinja2 import Template
                jinja_template = Template(template)
                return jinja_template.render(**item)
            else:
                # Simple template substitution - replace {field} with item[field]
                result = template
                for key, value in item.items():
                    placeholder = f"{{{key}}}"
                    if placeholder in result:
                        result = result.replace(placeholder, str(value))
                return result
        except Exception as e:
            logger.warning(f"Template application failed: {e}")
            return str(item)