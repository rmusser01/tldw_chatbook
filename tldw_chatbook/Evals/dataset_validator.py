# dataset_validator.py
# Description: Dataset validation functionality for evaluation tasks
#
"""
Dataset Validator
-----------------

Provides comprehensive validation for evaluation datasets including:
- Schema validation
- Data type checking
- Format consistency
- Sample completeness
- Balance analysis
- Quality metrics
"""

import json
import csv
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import statistics
from loguru import logger

from tldw_chatbook.Utils.path_validation import validate_path_simple

@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    severity: str  # 'error', 'warning', 'info'
    category: str  # 'schema', 'format', 'data', 'balance', 'quality'
    message: str
    sample_index: Optional[int] = None
    field_name: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationReport:
    """Complete validation report for a dataset."""
    dataset_name: str
    total_samples: int
    valid_samples: int
    issues: List[ValidationIssue]
    statistics: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def error_count(self) -> int:
        return len([i for i in self.issues if i.severity == 'error'])
    
    @property
    def warning_count(self) -> int:
        return len([i for i in self.issues if i.severity == 'warning'])
    
    @property
    def is_valid(self) -> bool:
        return self.error_count == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary format."""
        return {
            'dataset_name': self.dataset_name,
            'total_samples': self.total_samples,
            'valid_samples': self.valid_samples,
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'is_valid': self.is_valid,
            'issues': [
                {
                    'severity': issue.severity,
                    'category': issue.category,
                    'message': issue.message,
                    'sample_index': issue.sample_index,
                    'field_name': issue.field_name,
                    'details': issue.details
                }
                for issue in self.issues
            ],
            'statistics': self.statistics,
            'metadata': self.metadata
        }

class DatasetValidator:
    """Validates evaluation datasets for consistency and quality."""
    
    def __init__(self):
        """Initialize the dataset validator."""
        self.required_fields = {
            'classification': ['question', 'choices', 'answer'],
            'question_answer': ['question', 'answer'],
            'generation': ['input', 'expected_output'],
            'code_generation': ['problem', 'solution', 'test_cases']
        }
        
        self.optional_fields = {
            'all': ['id', 'metadata', 'difficulty', 'topic', 'source'],
            'classification': ['explanation'],
            'question_answer': ['context', 'reference'],
            'generation': ['max_length', 'stop_sequences'],
            'code_generation': ['language', 'timeout']
        }
    
    def validate_dataset(self, 
                        dataset_path: Union[str, Path],
                        task_type: str = 'auto',
                        strict: bool = False) -> ValidationReport:
        """
        Validate a dataset file.
        
        Args:
            dataset_path: Path to the dataset file
            task_type: Type of task ('classification', 'question_answer', etc.) or 'auto'
            strict: If True, warnings are treated as errors
            
        Returns:
            ValidationReport with detailed findings
        """
        path = validate_path_simple(dataset_path, require_exists=True)
        issues = []
        statistics = {}
        
        # Load dataset
        try:
            samples = self._load_dataset(path)
            if not samples:
                issues.append(ValidationIssue(
                    severity='error',
                    category='data',
                    message='Dataset is empty'
                ))
                return ValidationReport(
                    dataset_name=path.name,
                    total_samples=0,
                    valid_samples=0,
                    issues=issues,
                    statistics={}
                )
        except Exception as e:
            issues.append(ValidationIssue(
                severity='error',
                category='format',
                message=f'Failed to load dataset: {str(e)}'
            ))
            return ValidationReport(
                dataset_name=path.name,
                total_samples=0,
                valid_samples=0,
                issues=issues,
                statistics={}
            )
        
        # Auto-detect task type if needed
        if task_type == 'auto':
            task_type = self._detect_task_type(samples)
            issues.append(ValidationIssue(
                severity='info',
                category='schema',
                message=f'Auto-detected task type: {task_type}'
            ))
        
        # Validate schema
        schema_issues = self._validate_schema(samples, task_type)
        issues.extend(schema_issues)
        
        # Validate data types
        type_issues = self._validate_data_types(samples, task_type)
        issues.extend(type_issues)
        
        # Check format consistency
        format_issues = self._check_format_consistency(samples, task_type)
        issues.extend(format_issues)
        
        # Analyze sample completeness
        completeness_issues = self._check_completeness(samples, task_type)
        issues.extend(completeness_issues)
        
        # Analyze balance (for classification)
        if task_type == 'classification':
            balance_issues, balance_stats = self._analyze_balance(samples)
            issues.extend(balance_issues)
            statistics['balance'] = balance_stats
        
        # Calculate quality metrics
        quality_stats = self._calculate_quality_metrics(samples, task_type)
        statistics['quality'] = quality_stats
        
        # Count valid samples
        valid_samples = len(samples) - len(set(
            issue.sample_index for issue in issues 
            if issue.severity == 'error' and issue.sample_index is not None
        ))
        
        # Convert warnings to errors if strict mode
        if strict:
            for issue in issues:
                if issue.severity == 'warning':
                    issue.severity = 'error'
        
        return ValidationReport(
            dataset_name=path.name,
            total_samples=len(samples),
            valid_samples=valid_samples,
            issues=issues,
            statistics=statistics,
            metadata={
                'task_type': task_type,
                'file_format': path.suffix[1:],
                'validated_at': str(Path.ctime(path))
            }
        )
    
    def _load_dataset(self, path: Path) -> List[Dict[str, Any]]:
        """Load dataset from file."""
        if path.suffix.lower() == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, list) else [data]
        
        elif path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return data if isinstance(data, list) else [data]
        
        elif path.suffix.lower() in ['.csv', '.tsv']:
            delimiter = '\t' if path.suffix.lower() == '.tsv' else ','
            samples = []
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                for row in reader:
                    # Parse JSON fields if present
                    if 'choices' in row and row['choices'].startswith('['):
                        try:
                            row['choices'] = json.loads(row['choices'])
                        except:
                            pass
                    samples.append(row)
            return samples
        
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def _detect_task_type(self, samples: List[Dict[str, Any]]) -> str:
        """Auto-detect task type from dataset structure."""
        if not samples:
            return 'unknown'
        
        sample = samples[0]
        
        # Check for classification markers
        if 'choices' in sample or any(k.startswith('choice') for k in sample.keys()):
            return 'classification'
        
        # Check for code generation markers
        if 'test_cases' in sample or 'solution' in sample:
            return 'code_generation'
        
        # Check for generation markers
        if 'expected_output' in sample or 'target' in sample:
            return 'generation'
        
        # Default to question_answer
        return 'question_answer'
    
    def _validate_schema(self, samples: List[Dict[str, Any]], task_type: str) -> List[ValidationIssue]:
        """Validate dataset schema."""
        issues = []
        required = self.required_fields.get(task_type, [])
        optional = self.optional_fields.get('all', []) + self.optional_fields.get(task_type, [])
        
        # Check each sample
        for i, sample in enumerate(samples):
            # Check required fields
            for field in required:
                if field not in sample:
                    issues.append(ValidationIssue(
                        severity='error',
                        category='schema',
                        message=f'Missing required field: {field}',
                        sample_index=i,
                        field_name=field
                    ))
            
            # Check for unknown fields
            known_fields = set(required + optional)
            unknown_fields = set(sample.keys()) - known_fields
            if unknown_fields and i == 0:  # Only report once
                issues.append(ValidationIssue(
                    severity='info',
                    category='schema',
                    message=f'Unknown fields detected: {", ".join(unknown_fields)}',
                    details={'fields': list(unknown_fields)}
                ))
        
        return issues
    
    def _validate_data_types(self, samples: List[Dict[str, Any]], task_type: str) -> List[ValidationIssue]:
        """Validate data types of fields."""
        issues = []
        
        for i, sample in enumerate(samples):
            # Check choices for classification
            if task_type == 'classification' and 'choices' in sample:
                if not isinstance(sample['choices'], list):
                    issues.append(ValidationIssue(
                        severity='error',
                        category='data',
                        message='Choices must be a list',
                        sample_index=i,
                        field_name='choices'
                    ))
                elif len(sample['choices']) < 2:
                    issues.append(ValidationIssue(
                        severity='warning',
                        category='data',
                        message=f'Only {len(sample["choices"])} choices provided (minimum 2 recommended)',
                        sample_index=i,
                        field_name='choices'
                    ))
            
            # Check answer validity for classification
            if task_type == 'classification' and 'answer' in sample and 'choices' in sample:
                if isinstance(sample['choices'], list):
                    valid_answers = ['A', 'B', 'C', 'D'][:len(sample['choices'])]
                    if sample['answer'] not in valid_answers:
                        issues.append(ValidationIssue(
                            severity='error',
                            category='data',
                            message=f'Invalid answer: {sample["answer"]} (must be one of {valid_answers})',
                            sample_index=i,
                            field_name='answer'
                        ))
            
            # Check string fields
            string_fields = ['question', 'answer', 'input', 'expected_output']
            for field in string_fields:
                if field in sample and not isinstance(sample[field], str):
                    issues.append(ValidationIssue(
                        severity='error',
                        category='data',
                        message=f'{field} must be a string',
                        sample_index=i,
                        field_name=field
                    ))
        
        return issues
    
    def _check_format_consistency(self, samples: List[Dict[str, Any]], task_type: str) -> List[ValidationIssue]:
        """Check format consistency across samples."""
        issues = []
        
        if not samples:
            return issues
        
        # Get field structure from first sample
        first_fields = set(samples[0].keys())
        
        # Check consistency across samples
        for i, sample in enumerate(samples[1:], 1):
            current_fields = set(sample.keys())
            if current_fields != first_fields:
                missing = first_fields - current_fields
                extra = current_fields - first_fields
                
                if missing:
                    issues.append(ValidationIssue(
                        severity='warning',
                        category='format',
                        message=f'Inconsistent fields - missing: {missing}',
                        sample_index=i,
                        details={'missing': list(missing), 'extra': list(extra)}
                    ))
        
        # Check choice count consistency for classification
        if task_type == 'classification':
            choice_counts = [len(s.get('choices', [])) for s in samples if 'choices' in s]
            if choice_counts and len(set(choice_counts)) > 1:
                issues.append(ValidationIssue(
                    severity='warning',
                    category='format',
                    message=f'Inconsistent number of choices: {dict(Counter(choice_counts))}',
                    details={'distribution': dict(Counter(choice_counts))}
                ))
        
        return issues
    
    def _check_completeness(self, samples: List[Dict[str, Any]], task_type: str) -> List[ValidationIssue]:
        """Check sample completeness."""
        issues = []
        
        for i, sample in enumerate(samples):
            # Check for empty values
            for field, value in sample.items():
                if value == '' or value is None:
                    issues.append(ValidationIssue(
                        severity='warning',
                        category='data',
                        message=f'Empty value for field: {field}',
                        sample_index=i,
                        field_name=field
                    ))
            
            # Check for whitespace-only values
            for field, value in sample.items():
                if isinstance(value, str) and value.strip() == '':
                    issues.append(ValidationIssue(
                        severity='warning',
                        category='data',
                        message=f'Whitespace-only value for field: {field}',
                        sample_index=i,
                        field_name=field
                    ))
        
        return issues
    
    def _analyze_balance(self, samples: List[Dict[str, Any]]) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
        """Analyze class balance for classification tasks."""
        issues = []
        stats = {}
        
        # Count answer distribution
        answers = [s.get('answer') for s in samples if 'answer' in s]
        answer_counts = Counter(answers)
        stats['answer_distribution'] = dict(answer_counts)
        
        if answer_counts:
            # Calculate imbalance ratio
            max_count = max(answer_counts.values())
            min_count = min(answer_counts.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            stats['imbalance_ratio'] = imbalance_ratio
            
            # Check for severe imbalance
            if imbalance_ratio > 3:
                issues.append(ValidationIssue(
                    severity='warning',
                    category='balance',
                    message=f'Dataset is imbalanced (ratio: {imbalance_ratio:.2f})',
                    details={'distribution': dict(answer_counts)}
                ))
            
            # Check for missing classes
            if 'choices' in samples[0]:
                num_choices = len(samples[0]['choices'])
                expected_answers = set(['A', 'B', 'C', 'D'][:num_choices])
                actual_answers = set(answer_counts.keys())
                missing_answers = expected_answers - actual_answers
                
                if missing_answers:
                    issues.append(ValidationIssue(
                        severity='warning',
                        category='balance',
                        message=f'Missing answers in dataset: {missing_answers}',
                        details={'missing': list(missing_answers)}
                    ))
        
        return issues, stats
    
    def _calculate_quality_metrics(self, samples: List[Dict[str, Any]], task_type: str) -> Dict[str, Any]:
        """Calculate quality metrics for the dataset."""
        metrics = {}
        
        # Sample count
        metrics['total_samples'] = len(samples)
        
        # Average lengths
        if task_type in ['question_answer', 'classification']:
            question_lengths = [len(s.get('question', '')) for s in samples if 'question' in s]
            if question_lengths:
                metrics['avg_question_length'] = statistics.mean(question_lengths)
                metrics['min_question_length'] = min(question_lengths)
                metrics['max_question_length'] = max(question_lengths)
        
        if task_type == 'classification' and samples and 'choices' in samples[0]:
            choice_lengths = []
            for sample in samples:
                if 'choices' in sample and isinstance(sample['choices'], list):
                    choice_lengths.extend([len(str(c)) for c in sample['choices']])
            
            if choice_lengths:
                metrics['avg_choice_length'] = statistics.mean(choice_lengths)
        
        # Unique questions (check for duplicates)
        if 'question' in samples[0]:
            questions = [s.get('question', '') for s in samples]
            metrics['unique_questions'] = len(set(questions))
            metrics['duplicate_questions'] = len(questions) - metrics['unique_questions']
        
        # Difficulty distribution if available
        if any('difficulty' in s for s in samples):
            difficulties = [s.get('difficulty') for s in samples if 'difficulty' in s]
            metrics['difficulty_distribution'] = dict(Counter(difficulties))
        
        # Topic distribution if available
        if any('topic' in s for s in samples):
            topics = [s.get('topic') for s in samples if 'topic' in s]
            metrics['topic_distribution'] = dict(Counter(topics))
            metrics['num_topics'] = len(set(topics))
        
        return metrics
    
    def validate_batch(self, dataset_paths: List[Union[str, Path]], 
                      task_type: str = 'auto',
                      strict: bool = False) -> Dict[str, ValidationReport]:
        """
        Validate multiple datasets.
        
        Args:
            dataset_paths: List of paths to dataset files
            task_type: Type of task or 'auto' for each
            strict: If True, warnings are treated as errors
            
        Returns:
            Dictionary mapping dataset names to validation reports
        """
        reports = {}
        
        for path in dataset_paths:
            try:
                report = self.validate_dataset(path, task_type, strict)
                reports[Path(path).name] = report
            except Exception as e:
                logger.error(f"Failed to validate {path}: {e}")
                reports[Path(path).name] = ValidationReport(
                    dataset_name=Path(path).name,
                    total_samples=0,
                    valid_samples=0,
                    issues=[ValidationIssue(
                        severity='error',
                        category='system',
                        message=f'Validation failed: {str(e)}'
                    )],
                    statistics={}
                )
        
        return reports
    
    def suggest_fixes(self, report: ValidationReport) -> List[str]:
        """Generate fix suggestions based on validation report."""
        suggestions = []
        
        # Group issues by category
        issues_by_category = defaultdict(list)
        for issue in report.issues:
            issues_by_category[issue.category].append(issue)
        
        # Schema suggestions
        if 'schema' in issues_by_category:
            missing_fields = set()
            for issue in issues_by_category['schema']:
                if 'Missing required field' in issue.message and issue.field_name:
                    missing_fields.add(issue.field_name)
            
            if missing_fields:
                suggestions.append(f"Add missing required fields: {', '.join(missing_fields)}")
        
        # Balance suggestions
        if 'balance' in issues_by_category:
            for issue in issues_by_category['balance']:
                if 'imbalanced' in issue.message:
                    suggestions.append("Consider balancing the dataset by adding more samples for underrepresented classes")
                elif 'Missing answers' in issue.message:
                    suggestions.append("Add samples for all possible answer choices")
        
        # Format suggestions
        if 'format' in issues_by_category:
            suggestions.append("Ensure all samples have consistent field structure")
        
        # Data quality suggestions
        if 'data' in issues_by_category:
            empty_fields = set()
            for issue in issues_by_category['data']:
                if 'Empty value' in issue.message and issue.field_name:
                    empty_fields.add(issue.field_name)
            
            if empty_fields:
                suggestions.append(f"Fill in empty values for fields: {', '.join(empty_fields)}")
        
        # Duplicate suggestions
        if report.statistics.get('quality', {}).get('duplicate_questions', 0) > 0:
            suggestions.append("Remove or modify duplicate questions to increase dataset diversity")
        
        return suggestions