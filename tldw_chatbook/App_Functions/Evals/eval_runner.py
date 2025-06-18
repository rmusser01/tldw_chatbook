# eval_runner.py
# Description: Evaluation runner for different LLM task types
#
"""
Evaluation Runner for LLM Tasks
------------------------------

Executes evaluation tasks against LLM providers with support for:
- Question-Answer tasks
- Log probability evaluation
- Text generation tasks
- Classification tasks

Handles dataset loading, prompt formatting, model inference, and metric calculation.
"""

import asyncio
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union, Tuple, AsyncIterator
from pathlib import Path
import csv
from contextlib import asynccontextmanager

from loguru import logger

try:
    from datasets import load_dataset, Dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False

from .task_loader import TaskConfig
from .llm_interface import LLMInterface, EvalProviderError, EvalAPIError, EvalAuthenticationError, EvalRateLimitError

@dataclass
class EvalSample:
    """Individual evaluation sample."""
    id: str
    input_text: str
    expected_output: Optional[str] = None
    choices: Optional[List[str]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class EvalResult:
    """Result of evaluating a single sample."""
    sample_id: str
    input_text: str
    expected_output: Optional[str]
    actual_output: str
    logprobs: Optional[Dict[str, Any]] = None
    metrics: Dict[str, float] = None
    metadata: Dict[str, Any] = None
    processing_time: float = 0.0
    error_info: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.metadata is None:
            self.metadata = {}
        if self.error_info is None:
            self.error_info = {}

class DatasetLoader:
    """Loads datasets from various sources."""
    
    @staticmethod
    def load_dataset_samples(task_config: TaskConfig, split: str = None, max_samples: int = None) -> List[EvalSample]:
        """Load samples from dataset based on task configuration."""
        if split is None:
            split = task_config.split
        
        dataset_name = task_config.dataset_name
        
        # Handle different dataset sources
        if Path(dataset_name).exists():
            return DatasetLoader._load_local_dataset(dataset_name, max_samples)
        elif HF_DATASETS_AVAILABLE and '/' in dataset_name:
            return DatasetLoader._load_huggingface_dataset(task_config, split, max_samples)
        else:
            raise ValueError(f"Cannot load dataset: {dataset_name}")
    
    @staticmethod
    def _load_local_dataset(dataset_path: str, max_samples: int = None) -> List[EvalSample]:
        """Load dataset from local file (CSV, JSON, etc.)."""
        path = Path(dataset_path)
        
        if path.suffix.lower() == '.json':
            return DatasetLoader._load_json_dataset(path, max_samples)
        elif path.suffix.lower() in ['.csv', '.tsv']:
            return DatasetLoader._load_csv_dataset(path, max_samples)
        else:
            raise ValueError(f"Unsupported local dataset format: {path.suffix}")
    
    @staticmethod
    def _load_json_dataset(path: Path, max_samples: int = None) -> List[EvalSample]:
        """Load JSON dataset."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("JSON dataset must be a list of samples")
        
        samples = []
        for i, item in enumerate(data[:max_samples] if max_samples else data):
            sample_id = item.get('id', str(i))
            input_text = item.get('input', item.get('question', item.get('text', '')))
            expected_output = item.get('output', item.get('answer', item.get('target')))
            choices = item.get('choices', item.get('options'))
            
            samples.append(EvalSample(
                id=sample_id,
                input_text=input_text,
                expected_output=expected_output,
                choices=choices,
                metadata=item
            ))
        
        return samples
    
    @staticmethod
    def _load_csv_dataset(path: Path, max_samples: int = None) -> List[EvalSample]:
        """Load CSV/TSV dataset."""
        delimiter = '\t' if path.suffix.lower() == '.tsv' else ','
        
        with open(path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            rows = list(reader)
        
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
    def _load_huggingface_dataset(task_config: TaskConfig, split: str, max_samples: int = None) -> List[EvalSample]:
        """Load HuggingFace dataset."""
        if not HF_DATASETS_AVAILABLE:
            raise ImportError("HuggingFace datasets not available. Install with: pip install datasets")
        
        try:
            dataset = load_dataset(task_config.dataset_name, task_config.dataset_config, split=split)
            
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
            raise ValueError(f"Failed to load HuggingFace dataset {task_config.dataset_name}: {e}")
    
    @staticmethod
    def _apply_template(template: str, item: Dict[str, Any]) -> str:
        """Apply template to dataset item."""
        try:
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

class MetricsCalculator:
    """Calculates evaluation metrics."""
    
    @staticmethod
    def calculate_exact_match(predicted: str, expected: str) -> float:
        """Calculate exact match accuracy."""
        if expected is None:
            return 0.0
        return 1.0 if predicted.strip().lower() == expected.strip().lower() else 0.0
    
    @staticmethod
    def calculate_contains_match(predicted: str, expected: str) -> float:
        """Check if expected answer is contained in prediction."""
        if expected is None:
            return 0.0
        return 1.0 if expected.strip().lower() in predicted.strip().lower() else 0.0
    
    @staticmethod
    def calculate_regex_match(predicted: str, expected: str, pattern: str = None) -> float:
        """Calculate match using regex pattern."""
        if expected is None or pattern is None:
            return 0.0
        
        try:
            if re.search(pattern, predicted, re.IGNORECASE):
                return 1.0
            return 0.0
        except re.error:
            logger.warning(f"Invalid regex pattern: {pattern}")
            return 0.0
    
    @staticmethod
    def calculate_f1_score(predicted: str, expected: str) -> float:
        """Calculate F1 score based on token overlap."""
        if expected is None:
            return 0.0
        
        pred_tokens = set(predicted.lower().split())
        expected_tokens = set(expected.lower().split())
        
        if not expected_tokens:
            return 1.0 if not pred_tokens else 0.0
        
        intersection = pred_tokens & expected_tokens
        if not intersection:
            return 0.0
        
        precision = len(intersection) / len(pred_tokens) if pred_tokens else 0.0
        recall = len(intersection) / len(expected_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def calculate_bleu_score(predicted: str, expected: str) -> float:
        """Simple BLEU-like score based on n-gram overlap."""
        if expected is None:
            return 0.0
        
        pred_tokens = predicted.lower().split()
        expected_tokens = expected.lower().split()
        
        if not expected_tokens:
            return 1.0 if not pred_tokens else 0.0
        
        # Simple unigram overlap (BLEU-1 approximation)
        matches = 0
        for token in pred_tokens:
            if token in expected_tokens:
                matches += 1
        
        return matches / len(expected_tokens) if expected_tokens else 0.0

class ErrorHandler:
    """Handles retries and error classification for evaluation runs."""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor
        self.rate_limit_delays = {}  # Provider -> delay time
    
    async def with_retry(self, operation, sample_id: str, provider: str = None):
        """Execute operation with retry logic and error handling."""
        last_exception = None
        retry_count = 0
        
        for attempt in range(self.max_retries + 1):
            try:
                # Check rate limiting
                if provider and provider in self.rate_limit_delays:
                    delay = self.rate_limit_delays[provider]
                    if delay > 0:
                        logger.info(f"Rate limited for {provider}, waiting {delay:.1f}s")
                        await asyncio.sleep(delay)
                        # Reduce delay for next attempt
                        self.rate_limit_delays[provider] = max(0, delay - 1.0)
                
                # Execute operation
                result = await operation()
                
                # Clear rate limit if successful
                if provider and provider in self.rate_limit_delays:
                    self.rate_limit_delays[provider] = 0
                
                return result, retry_count
                
            except EvalRateLimitError as e:
                logger.warning(f"Rate limit hit for {provider}: {e}")
                
                # Set rate limit delay
                if provider:
                    self.rate_limit_delays[provider] = min(60.0, 5.0 * (attempt + 1))
                
                if attempt < self.max_retries:
                    delay = self.retry_delay * (self.backoff_factor ** attempt)
                    logger.info(f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                    await asyncio.sleep(delay)
                    retry_count += 1
                    last_exception = e
                else:
                    raise
                    
            except EvalAuthenticationError as e:
                logger.error(f"Authentication error for {provider}: {e}")
                # Don't retry auth errors
                raise
                
            except (EvalAPIError, EvalProviderError) as e:
                logger.warning(f"API error for sample {sample_id}: {e}")
                
                if attempt < self.max_retries:
                    delay = self.retry_delay * (self.backoff_factor ** attempt)
                    logger.info(f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                    await asyncio.sleep(delay)
                    retry_count += 1
                    last_exception = e
                else:
                    raise
                    
            except Exception as e:
                logger.error(f"Unexpected error for sample {sample_id}: {e}")
                
                if attempt < self.max_retries:
                    delay = self.retry_delay * (self.backoff_factor ** attempt)
                    logger.info(f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                    await asyncio.sleep(delay)
                    retry_count += 1
                    last_exception = e
                else:
                    raise
        
        # If we get here, all retries failed
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError(f"All retries failed for sample {sample_id}")
    
    def classify_error(self, error: Exception) -> Dict[str, Any]:
        """Classify error for better reporting."""
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'is_retryable': True,
            'error_category': 'unknown'
        }
        
        if isinstance(error, EvalAuthenticationError):
            error_info.update({
                'is_retryable': False,
                'error_category': 'authentication',
                'suggested_action': 'Check API keys and credentials'
            })
        elif isinstance(error, EvalRateLimitError):
            error_info.update({
                'error_category': 'rate_limit',
                'suggested_action': 'Reduce request rate or wait'
            })
        elif isinstance(error, EvalAPIError):
            error_info.update({
                'error_category': 'api_error',
                'suggested_action': 'Check API endpoint and parameters'
            })
        elif isinstance(error, EvalProviderError):
            error_info.update({
                'error_category': 'provider_error',
                'suggested_action': 'Check provider configuration'
            })
        elif 'timeout' in str(error).lower():
            error_info.update({
                'error_category': 'timeout',
                'suggested_action': 'Increase timeout or check network'
            })
        elif 'connection' in str(error).lower():
            error_info.update({
                'error_category': 'network',
                'suggested_action': 'Check network connectivity'
            })
        else:
            error_info.update({
                'is_retryable': False,
                'error_category': 'unknown',
                'suggested_action': 'Review error details and task configuration'
            })
        
        return error_info

class BaseEvalRunner(ABC):
    """Base class for evaluation runners."""
    
    def __init__(self, task_config: TaskConfig, model_config: Dict[str, Any]):
        self.task_config = task_config
        self.model_config = model_config
        self.llm_interface = LLMInterface(
            provider_name=model_config['provider'],
            model_id=model_config['model_id'],
            config=model_config
        )
        self.error_handler = ErrorHandler(
            max_retries=task_config.metadata.get('max_retries', 3),
            retry_delay=task_config.metadata.get('retry_delay', 1.0)
        )
        
    @abstractmethod
    async def run_sample(self, sample: EvalSample) -> EvalResult:
        """Run evaluation on a single sample."""
        pass
    
    def calculate_metrics(self, predicted: str, expected: str, choices: List[str] = None) -> Dict[str, float]:
        """Calculate metrics for a prediction."""
        metrics = {}
        
        if self.task_config.metric == 'exact_match':
            metrics['exact_match'] = MetricsCalculator.calculate_exact_match(predicted, expected)
        elif self.task_config.metric == 'contains':
            metrics['contains'] = MetricsCalculator.calculate_contains_match(predicted, expected)
        elif self.task_config.metric == 'f1':
            metrics['f1'] = MetricsCalculator.calculate_f1_score(predicted, expected)
        elif self.task_config.metric == 'bleu':
            metrics['bleu'] = MetricsCalculator.calculate_bleu_score(predicted, expected)
        elif self.task_config.metric == 'accuracy' and choices:
            # For multiple choice
            metrics['accuracy'] = MetricsCalculator.calculate_exact_match(predicted, expected)
        else:
            # Default to exact match
            metrics['exact_match'] = MetricsCalculator.calculate_exact_match(predicted, expected)
        
        return metrics
    
    def apply_filters(self, output: str) -> str:
        """Apply post-processing filters to model output."""
        filtered_output = output
        
        for filter_config in self.task_config.filter_list:
            if filter_config.get('filter') == 'regex':
                pattern = filter_config.get('regex_pattern')
                group = filter_config.get('group', 0)
                
                if pattern:
                    try:
                        match = re.search(pattern, filtered_output, re.IGNORECASE | re.DOTALL)
                        if match:
                            filtered_output = match.group(group)
                    except re.error as e:
                        logger.warning(f"Regex filter error: {e}")
            
            elif filter_config.get('filter') == 'take_first_line':
                filtered_output = filtered_output.split('\n')[0].strip()
            
            elif filter_config.get('filter') == 'strip':
                filtered_output = filtered_output.strip()
        
        return filtered_output

class QuestionAnswerRunner(BaseEvalRunner):
    """Runner for question-answer tasks."""
    
    async def run_sample(self, sample: EvalSample) -> EvalResult:
        """Run Q&A evaluation on a single sample."""
        start_time = time.time()
        
        async def _run_sample_operation():
            # Format the prompt
            prompt = sample.input_text
            
            # Add few-shot examples if configured
            if self.task_config.num_fewshot > 0:
                few_shot_samples = await self._get_few_shot_examples()
                prompt = self._format_few_shot_prompt(few_shot_samples, sample.input_text)
            
            # Generate response
            response = await self._generate_response(prompt)
            
            # Apply filters
            filtered_response = self.apply_filters(response)
            
            # Calculate metrics
            metrics = self.calculate_metrics(filtered_response, sample.expected_output)
            
            return {
                'response': filtered_response,
                'raw_output': response,
                'prompt': prompt,
                'metrics': metrics
            }
        
        try:
            # Use error handler with retry logic
            result_data, retry_count = await self.error_handler.with_retry(
                _run_sample_operation,
                sample.id,
                self.model_config.get('provider')
            )
            
            processing_time = time.time() - start_time
            
            return EvalResult(
                sample_id=sample.id,
                input_text=sample.input_text,
                expected_output=sample.expected_output,
                actual_output=result_data['response'],
                metrics=result_data['metrics'],
                metadata={
                    'raw_output': result_data['raw_output'], 
                    'prompt': result_data['prompt'],
                    'provider': self.model_config.get('provider')
                },
                processing_time=processing_time,
                retry_count=retry_count
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_info = self.error_handler.classify_error(e)
            
            logger.error(f"Error processing sample {sample.id} after retries: {e}")
            logger.error(f"Error classification: {error_info}")
            
            return EvalResult(
                sample_id=sample.id,
                input_text=sample.input_text,
                expected_output=sample.expected_output,
                actual_output=f"ERROR: {str(e)}",
                metrics={'error': 1.0},
                processing_time=processing_time,
                error_info=error_info
            )
    
    async def _generate_response(self, prompt: str) -> str:
        """Generate response from LLM."""
        try:
            config = self.model_config.copy()
            config.update(self.task_config.generation_kwargs)
            
            response = await self.llm_interface.generate(
                prompt=prompt,
                **config
            )
            
            return response
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
    
    async def _get_few_shot_examples(self) -> List[EvalSample]:
        """Get few-shot examples from validation set."""
        if not self.task_config.few_shot_split:
            return []
        
        try:
            samples = DatasetLoader.load_dataset_samples(
                self.task_config, 
                split=self.task_config.few_shot_split,
                max_samples=self.task_config.num_fewshot
            )
            return samples
        except Exception as e:
            logger.warning(f"Failed to load few-shot examples: {e}")
            return []
    
    def _format_few_shot_prompt(self, few_shot_samples: List[EvalSample], query: str) -> str:
        """Format prompt with few-shot examples."""
        prompt_parts = []
        
        for sample in few_shot_samples:
            prompt_parts.append(f"Q: {sample.input_text}")
            if sample.expected_output:
                prompt_parts.append(f"A: {sample.expected_output}")
            prompt_parts.append("")  # Empty line
        
        prompt_parts.append(f"Q: {query}")
        prompt_parts.append("A:")
        
        return "\n".join(prompt_parts)

class ClassificationRunner(BaseEvalRunner):
    """Runner for classification/multiple-choice tasks."""
    
    async def run_sample(self, sample: EvalSample) -> EvalResult:
        """Run classification evaluation on a single sample."""
        start_time = time.time()
        
        async def _run_classification_operation():
            # Format multiple choice prompt
            prompt = self._format_classification_prompt(sample)
            
            # Generate response
            response = await self._generate_response(prompt)
            
            # Extract choice from response
            predicted_choice = self._extract_choice(response, sample.choices)
            
            # Calculate metrics
            metrics = self.calculate_metrics(predicted_choice, sample.expected_output, sample.choices)
            
            return {
                'predicted_choice': predicted_choice,
                'raw_output': response,
                'prompt': prompt,
                'metrics': metrics
            }
        
        try:
            # Use error handler with retry logic
            result_data, retry_count = await self.error_handler.with_retry(
                _run_classification_operation,
                sample.id,
                self.model_config.get('provider')
            )
            
            processing_time = time.time() - start_time
            
            return EvalResult(
                sample_id=sample.id,
                input_text=sample.input_text,
                expected_output=sample.expected_output,
                actual_output=result_data['predicted_choice'],
                metrics=result_data['metrics'],
                metadata={
                    'raw_output': result_data['raw_output'], 
                    'prompt': result_data['prompt'],
                    'choices': sample.choices,
                    'provider': self.model_config.get('provider')
                },
                processing_time=processing_time,
                retry_count=retry_count
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_info = self.error_handler.classify_error(e)
            
            logger.error(f"Error processing classification sample {sample.id} after retries: {e}")
            logger.error(f"Error classification: {error_info}")
            
            return EvalResult(
                sample_id=sample.id,
                input_text=sample.input_text,
                expected_output=sample.expected_output,
                actual_output=f"ERROR: {str(e)}",
                metrics={'error': 1.0},
                processing_time=processing_time,
                error_info=error_info
            )
    
    def _format_classification_prompt(self, sample: EvalSample) -> str:
        """Format prompt for classification task."""
        if not sample.choices:
            return sample.input_text
        
        prompt_parts = [sample.input_text, ""]
        
        # Add choices
        for i, choice in enumerate(sample.choices):
            label = chr(ord('A') + i)  # A, B, C, D...
            prompt_parts.append(f"{label}. {choice}")
        
        prompt_parts.append("")
        prompt_parts.append("Answer:")
        
        return "\n".join(prompt_parts)
    
    def _extract_choice(self, response: str, choices: List[str]) -> str:
        """Extract the chosen answer from model response."""
        if not choices:
            return response.strip()
        
        response_clean = response.strip().upper()
        
        # Look for pattern like "A", "B)", "(C)", etc.
        for i, choice in enumerate(choices):
            label = chr(ord('A') + i)
            
            # Check various patterns
            patterns = [
                f"^{label}$",  # Just the letter
                f"^{label}\\.",  # A.
                f"^{label}\\)",  # A)
                f"^\\({label}\\)",  # (A)
                f"answer\\s*:?\\s*{label}",  # Answer: A
                f"choice\\s*:?\\s*{label}",  # Choice: A
            ]
            
            for pattern in patterns:
                if re.search(pattern, response_clean):
                    return choice
        
        # If no clear choice found, try to match choice text
        for choice in choices:
            if choice.lower() in response.lower():
                return choice
        
        # Default to first choice if nothing matches
        return choices[0] if choices else response.strip()
    
    async def _generate_response(self, prompt: str) -> str:
        """Generate response from LLM for classification."""
        try:
            config = self.model_config.copy()
            config.update(self.task_config.generation_kwargs)
            
            # For classification, we usually want shorter responses
            config.setdefault('max_tokens', 10)
            config.setdefault('temperature', 0.0)  # More deterministic
            
            response = await self.llm_interface.generate(
                prompt=prompt,
                **config
            )
            
            return response
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

class LogProbRunner(BaseEvalRunner):
    """Runner for log probability evaluation tasks."""
    
    async def run_sample(self, sample: EvalSample) -> EvalResult:
        """Run log probability evaluation on a single sample."""
        start_time = time.time()
        
        async def _run_logprob_operation():
            # For logprob tasks, we typically evaluate the likelihood of different continuations
            prompt = sample.input_text
            
            if sample.choices:
                # Multiple choice logprob
                logprobs = await self._get_choice_logprobs(prompt, sample.choices)
                
                # Select choice with highest logprob
                best_choice_idx = max(range(len(logprobs)), key=lambda i: logprobs[i])
                predicted = sample.choices[best_choice_idx]
                
                metrics = self.calculate_metrics(predicted, sample.expected_output, sample.choices)
                
                return {
                    'predicted': predicted,
                    'logprobs': {'choice_logprobs': logprobs, 'choice_names': sample.choices},
                    'metrics': metrics
                }
            
            else:
                # Single continuation logprob
                if not sample.expected_output:
                    raise ValueError("Expected output required for logprob evaluation")
                
                logprob = await self._get_continuation_logprob(prompt, sample.expected_output)
                
                return {
                    'predicted': sample.expected_output,
                    'logprobs': {'continuation_logprob': logprob},
                    'metrics': {'logprob': logprob}
                }
        
        try:
            # Use error handler with retry logic
            result_data, retry_count = await self.error_handler.with_retry(
                _run_logprob_operation,
                sample.id,
                self.model_config.get('provider')
            )
            
            processing_time = time.time() - start_time
            
            return EvalResult(
                sample_id=sample.id,
                input_text=sample.input_text,
                expected_output=sample.expected_output,
                actual_output=result_data['predicted'],
                logprobs=result_data['logprobs'],
                metrics=result_data['metrics'],
                metadata={'provider': self.model_config.get('provider')},
                processing_time=processing_time,
                retry_count=retry_count
            )
        
        except Exception as e:
            processing_time = time.time() - start_time
            error_info = self.error_handler.classify_error(e)
            
            logger.error(f"Error processing logprob sample {sample.id} after retries: {e}")
            logger.error(f"Error classification: {error_info}")
            
            return EvalResult(
                sample_id=sample.id,
                input_text=sample.input_text,
                expected_output=sample.expected_output,
                actual_output=f"ERROR: {str(e)}",
                metrics={'error': 1.0},
                processing_time=processing_time,
                error_info=error_info
            )
    
    async def _get_choice_logprobs(self, prompt: str, choices: List[str]) -> List[float]:
        """Get log probabilities for each choice."""
        logprobs = []
        for choice in choices:
            full_text = prompt + choice
            logprob = await self._get_text_logprob(full_text)
            logprobs.append(logprob)
        return logprobs
    
    async def _get_continuation_logprob(self, prompt: str, continuation: str) -> float:
        """Get log probability of a continuation given prompt."""
        full_text = prompt + continuation
        return await self._get_text_logprob(full_text)
    
    async def _get_text_logprob(self, text: str) -> float:
        """Get log probability of text from LLM."""
        try:
            # This would need to be implemented based on the LLM provider's API
            # For now, return a placeholder
            logger.warning("Log probability calculation not yet implemented for this provider")
            return 0.0
            
        except Exception as e:
            logger.error(f"Failed to get logprob: {e}")
            return float('-inf')

class GenerationRunner(BaseEvalRunner):
    """Runner for text generation tasks."""
    
    async def run_sample(self, sample: EvalSample) -> EvalResult:
        """Run text generation evaluation on a single sample."""
        start_time = time.time()
        
        async def _run_generation_operation():
            prompt = sample.input_text
            
            # Generate response
            response = await self._generate_response(prompt)
            
            # Apply filters
            filtered_response = self.apply_filters(response)
            
            # Calculate metrics
            metrics = self.calculate_metrics(filtered_response, sample.expected_output)
            
            return {
                'response': filtered_response,
                'raw_output': response,
                'prompt': prompt,
                'metrics': metrics
            }
        
        try:
            # Use error handler with retry logic
            result_data, retry_count = await self.error_handler.with_retry(
                _run_generation_operation,
                sample.id,
                self.model_config.get('provider')
            )
            
            processing_time = time.time() - start_time
            
            return EvalResult(
                sample_id=sample.id,
                input_text=sample.input_text,
                expected_output=sample.expected_output,
                actual_output=result_data['response'],
                metrics=result_data['metrics'],
                metadata={
                    'raw_output': result_data['raw_output'], 
                    'prompt': result_data['prompt'],
                    'provider': self.model_config.get('provider')
                },
                processing_time=processing_time,
                retry_count=retry_count
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_info = self.error_handler.classify_error(e)
            
            logger.error(f"Error processing generation sample {sample.id} after retries: {e}")
            logger.error(f"Error classification: {error_info}")
            
            return EvalResult(
                sample_id=sample.id,
                input_text=sample.input_text,
                expected_output=sample.expected_output,
                actual_output=f"ERROR: {str(e)}",
                metrics={'error': 1.0},
                processing_time=processing_time,
                error_info=error_info
            )
    
    async def _generate_response(self, prompt: str) -> str:
        """Generate response from LLM for text generation."""
        try:
            config = self.model_config.copy()
            config.update(self.task_config.generation_kwargs)
            
            # Set stop sequences if specified
            if self.task_config.stop_sequences:
                config['stop'] = self.task_config.stop_sequences
            
            response = await self.llm_interface.generate(
                prompt=prompt,
                **config
            )
            
            return response
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

class EvalRunner:
    """Main evaluation runner that coordinates different task types."""
    
    def __init__(self, task_config: TaskConfig, model_config: Dict[str, Any]):
        self.task_config = task_config
        self.model_config = model_config
        
        # Import specialized runners here to avoid circular imports
        try:
            from .specialized_runners import (
                CodeExecutionRunner, SafetyEvaluationRunner, 
                MultilingualEvaluationRunner, CreativeEvaluationRunner
            )
            specialized_available = True
        except ImportError:
            specialized_available = False
        
        # Determine runner based on task metadata and type
        category = task_config.metadata.get('category', '')
        subcategory = task_config.metadata.get('subcategory', '')
        
        # Use specialized runners when available and appropriate
        if specialized_available:
            if category == 'coding' or subcategory in ['function_implementation', 'algorithms', 'code_completion']:
                self.runner = CodeExecutionRunner(task_config, model_config)
            elif category == 'safety' or subcategory in ['harmfulness', 'bias', 'truthfulness']:
                self.runner = SafetyEvaluationRunner(task_config, model_config)
            elif subcategory in ['translation', 'cross_lingual_qa', 'multilingual']:
                self.runner = MultilingualEvaluationRunner(task_config, model_config)
            elif category == 'creative' or subcategory in ['creative_writing', 'story_completion', 'dialogue_generation']:
                self.runner = CreativeEvaluationRunner(task_config, model_config)
            else:
                self.runner = self._create_basic_runner(task_config, model_config)
        else:
            self.runner = self._create_basic_runner(task_config, model_config)
    
    def _create_basic_runner(self, task_config: TaskConfig, model_config: Dict[str, Any]):
        """Create basic runner based on task type."""
        if task_config.task_type == 'question_answer':
            return QuestionAnswerRunner(task_config, model_config)
        elif task_config.task_type == 'classification':
            return ClassificationRunner(task_config, model_config)
        elif task_config.task_type == 'logprob':
            return LogProbRunner(task_config, model_config)
        elif task_config.task_type == 'generation':
            return GenerationRunner(task_config, model_config)
        else:
            raise ValueError(f"Unsupported task type: {task_config.task_type}")
    
    async def run_evaluation(self, max_samples: int = None, 
                           progress_callback: callable = None) -> List[EvalResult]:
        """Run evaluation on the task dataset."""
        logger.info(f"Starting evaluation: {self.task_config.name}")
        
        # Load dataset samples
        samples = DatasetLoader.load_dataset_samples(self.task_config, max_samples=max_samples)
        logger.info(f"Loaded {len(samples)} samples")
        
        results = []
        error_count = 0
        retry_count = 0
        
        for i, sample in enumerate(samples):
            try:
                result = await self.runner.run_sample(sample)
                results.append(result)
                
                # Track retry statistics
                if hasattr(result, 'retry_count'):
                    retry_count += result.retry_count
                
                # Track error statistics
                if result.error_info:
                    error_count += 1
                    logger.warning(f"Sample {sample.id} completed with errors: {result.error_info.get('error_category', 'unknown')}")
                
                if progress_callback:
                    progress_callback(i + 1, len(samples), result)
                
                # Log progress periodically with enhanced info
                if (i + 1) % 10 == 0:
                    success_rate = ((i + 1 - error_count) / (i + 1)) * 100
                    logger.info(f"Processed {i + 1}/{len(samples)} samples | Success rate: {success_rate:.1f}% | Total retries: {retry_count}")
                    
            except Exception as e:
                logger.error(f"Fatal error processing sample {sample.id}: {e}")
                error_count += 1
                
                # Create error result
                error_result = EvalResult(
                    sample_id=sample.id,
                    input_text=sample.input_text,
                    expected_output=sample.expected_output,
                    actual_output=f"FATAL_ERROR: {str(e)}",
                    metrics={'error': 1.0},
                    error_info={
                        'error_type': type(e).__name__,
                        'error_message': str(e),
                        'error_category': 'fatal',
                        'is_retryable': False
                    }
                )
                results.append(error_result)
        
        # Log final statistics
        success_rate = ((len(results) - error_count) / len(results)) * 100 if results else 0
        logger.info(f"Evaluation completed: {len(results)} results")
        logger.info(f"Success rate: {success_rate:.1f}% | Errors: {error_count} | Total retries: {retry_count}")
        
        return results
    
    def calculate_aggregate_metrics(self, results: List[EvalResult]) -> Dict[str, float]:
        """Calculate aggregate metrics across all results."""
        if not results:
            return {}
        
        # Collect all metric names
        all_metrics = set()
        for result in results:
            all_metrics.update(result.metrics.keys())
        
        aggregate_metrics = {}
        
        for metric_name in all_metrics:
            values = [result.metrics.get(metric_name, 0.0) for result in results 
                     if metric_name in result.metrics]
            
            if values:
                aggregate_metrics[f"{metric_name}_mean"] = sum(values) / len(values)
                aggregate_metrics[f"{metric_name}_count"] = len(values)
        
        # Add summary stats
        aggregate_metrics['total_samples'] = len(results)
        aggregate_metrics['error_count'] = sum(1 for r in results if 'error' in r.metrics)
        aggregate_metrics['success_rate'] = (len(results) - aggregate_metrics['error_count']) / len(results)
        
        return aggregate_metrics