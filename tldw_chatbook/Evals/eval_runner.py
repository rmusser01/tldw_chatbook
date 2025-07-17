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
from tldw_chatbook.Metrics.metrics_logger import log_counter, log_histogram

class EvalError(Exception):
    """Base exception for evaluation errors."""
    pass

@dataclass
class EvalProgress:
    """Progress tracking for evaluation runs."""
    current: int
    total: int
    current_task: Optional[str] = None
    
    @property
    def percentage(self) -> float:
        return (self.current / self.total * 100) if self.total > 0 else 0

@dataclass
class EvalRunResult:
    """Result of an evaluation run."""
    task_name: str
    metrics: Dict[str, Any]
    samples_evaluated: int
    duration_seconds: float
    timestamp: str
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []

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
class EvalSampleResult:
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
        """Calculate exact match accuracy (case-sensitive)."""
        if expected is None:
            return 0.0
        return 1.0 if predicted.strip() == expected.strip() else 0.0
    
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
    def calculate_bleu_score(predicted: str, expected: str, n: int = 1) -> float:
        """Calculate BLEU score with n-gram support."""
        if expected is None:
            return 0.0
        
        def get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
            """Get n-grams from token list."""
            if n <= 0 or n > len(tokens):
                return []
            return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        
        pred_tokens = predicted.lower().split()
        expected_tokens = expected.lower().split()
        
        if not expected_tokens:
            return 1.0 if not pred_tokens else 0.0
        
        if not pred_tokens:
            return 0.0
        
        # Calculate n-gram precision
        total_precision = 0.0
        for i in range(1, min(n + 1, len(expected_tokens) + 1)):
            pred_ngrams = get_ngrams(pred_tokens, i)
            expected_ngrams = get_ngrams(expected_tokens, i)
            
            if not pred_ngrams:
                continue
                
            matches = 0
            expected_ngram_counts = {}
            for ngram in expected_ngrams:
                expected_ngram_counts[ngram] = expected_ngram_counts.get(ngram, 0) + 1
            
            for ngram in pred_ngrams:
                if ngram in expected_ngram_counts and expected_ngram_counts[ngram] > 0:
                    matches += 1
                    expected_ngram_counts[ngram] -= 1
            
            precision = matches / len(pred_ngrams) if pred_ngrams else 0.0
            total_precision += precision
        
        # Average precision across n-grams
        avg_precision = total_precision / min(n, len(expected_tokens))
        
        # Brevity penalty
        bp = 1.0
        if len(pred_tokens) < len(expected_tokens):
            bp = min(1.0, (len(pred_tokens) / len(expected_tokens)) ** 0.5)
        
        return bp * avg_precision
    
    @staticmethod
    def calculate_rouge_1(predicted: str, expected: str) -> float:
        """Calculate ROUGE-1 (unigram) F1 score."""
        if expected is None:
            return 0.0
        
        pred_tokens = set(predicted.lower().split())
        expected_tokens = set(expected.lower().split())
        
        if not expected_tokens:
            return 1.0 if not pred_tokens else 0.0
        
        if not pred_tokens:
            return 0.0
        
        # Calculate overlap
        overlap = pred_tokens & expected_tokens
        
        if not overlap:
            return 0.0
        
        # Calculate precision and recall
        precision = len(overlap) / len(pred_tokens)
        recall = len(overlap) / len(expected_tokens)
        
        # Calculate F1 score
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    @staticmethod
    def calculate_rouge_2(predicted: str, expected: str) -> float:
        """Calculate ROUGE-2 (bigram) F1 score."""
        if expected is None:
            return 0.0
        
        def get_bigrams(tokens: List[str]) -> List[Tuple[str, str]]:
            """Get bigrams from token list."""
            if len(tokens) < 2:
                return []
            return [(tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)]
        
        pred_tokens = predicted.lower().split()
        expected_tokens = expected.lower().split()
        
        if len(expected_tokens) < 2:
            return 1.0 if len(pred_tokens) < 2 else 0.0
        
        if len(pred_tokens) < 2:
            return 0.0
        
        # Get bigrams
        pred_bigrams = set(get_bigrams(pred_tokens))
        expected_bigrams = set(get_bigrams(expected_tokens))
        
        if not expected_bigrams:
            return 1.0 if not pred_bigrams else 0.0
        
        # Calculate overlap
        overlap = pred_bigrams & expected_bigrams
        
        if not overlap:
            return 0.0
        
        # Calculate precision and recall
        precision = len(overlap) / len(pred_bigrams)
        recall = len(overlap) / len(expected_bigrams)
        
        # Calculate F1 score
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    @staticmethod
    def calculate_rouge_l(predicted: str, expected: str) -> float:
        """Calculate ROUGE-L (Longest Common Subsequence) F1 score."""
        if expected is None:
            return 0.0
        
        def lcs_length(x: List[str], y: List[str]) -> int:
            """Calculate length of longest common subsequence."""
            m, n = len(x), len(y)
            if m == 0 or n == 0:
                return 0
            
            # Create DP table
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            # Fill DP table
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i-1] == y[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        pred_tokens = predicted.lower().split()
        expected_tokens = expected.lower().split()
        
        if not expected_tokens:
            return 1.0 if not pred_tokens else 0.0
        
        if not pred_tokens:
            return 0.0
        
        # Calculate LCS
        lcs_len = lcs_length(pred_tokens, expected_tokens)
        
        if lcs_len == 0:
            return 0.0
        
        # Calculate precision and recall
        precision = lcs_len / len(pred_tokens)
        recall = lcs_len / len(expected_tokens)
        
        # Calculate F1 score
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    @staticmethod
    def calculate_semantic_similarity(predicted: str, expected: str, embedding_model=None) -> float:
        """Calculate semantic similarity using embeddings if available."""
        if expected is None:
            return 0.0
        
        if not predicted and not expected:
            return 1.0
        
        if not predicted or not expected:
            return 0.0
        
        # Try to use sentence transformers if available
        try:
            if embedding_model is None:
                from sentence_transformers import SentenceTransformer
                # Use a small, fast model by default
                embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Get embeddings
            embeddings = embedding_model.encode([predicted, expected])
            pred_embedding = embeddings[0]
            exp_embedding = embeddings[1]
            
            # Calculate cosine similarity
            try:
                from numpy import dot
                from numpy.linalg import norm
                cosine_sim = dot(pred_embedding, exp_embedding) / (norm(pred_embedding) * norm(exp_embedding))
            except ImportError:
                # Fallback to pure Python cosine similarity
                def cosine_similarity(v1, v2):
                    dot_product = sum(a * b for a, b in zip(v1, v2))
                    norm1 = sum(a * a for a in v1) ** 0.5
                    norm2 = sum(b * b for b in v2) ** 0.5
                    return dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0
                cosine_sim = cosine_similarity(pred_embedding, exp_embedding)
            
            # Normalize to 0-1 range
            return max(0.0, min(1.0, float(cosine_sim)))
            
        except ImportError:
            # Fallback to token overlap if embeddings not available
            logger.debug("Sentence transformers not available, using token overlap for semantic similarity")
            return MetricsCalculator.calculate_f1_score(predicted, expected)
        except Exception as e:
            logger.warning(f"Error calculating semantic similarity: {e}")
            return MetricsCalculator.calculate_f1_score(predicted, expected)
    
    @staticmethod
    def calculate_perplexity(logprobs: List[float]) -> float:
        """Calculate perplexity from log probabilities."""
        if not logprobs:
            return float('inf')
        
        try:
            import math
            # Perplexity = exp(average negative log probability)
            avg_neg_logprob = -sum(logprobs) / len(logprobs)
            perplexity = math.exp(avg_neg_logprob)
            return perplexity
        except (ValueError, OverflowError):
            return float('inf')

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
    async def run_sample(self, sample: EvalSample) -> EvalSampleResult:
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
        elif self.task_config.metric == 'bleu-1':
            metrics['bleu-1'] = MetricsCalculator.calculate_bleu_score(predicted, expected, n=1)
        elif self.task_config.metric == 'bleu-2':
            metrics['bleu-2'] = MetricsCalculator.calculate_bleu_score(predicted, expected, n=2)
        elif self.task_config.metric == 'bleu-3':
            metrics['bleu-3'] = MetricsCalculator.calculate_bleu_score(predicted, expected, n=3)
        elif self.task_config.metric == 'bleu-4':
            metrics['bleu-4'] = MetricsCalculator.calculate_bleu_score(predicted, expected, n=4)
        elif self.task_config.metric == 'rouge-1':
            metrics['rouge-1'] = MetricsCalculator.calculate_rouge_1(predicted, expected)
        elif self.task_config.metric == 'rouge-2':
            metrics['rouge-2'] = MetricsCalculator.calculate_rouge_2(predicted, expected)
        elif self.task_config.metric == 'rouge-l':
            metrics['rouge-l'] = MetricsCalculator.calculate_rouge_l(predicted, expected)
        elif self.task_config.metric == 'rouge':
            # Calculate all ROUGE metrics
            metrics['rouge-1'] = MetricsCalculator.calculate_rouge_1(predicted, expected)
            metrics['rouge-2'] = MetricsCalculator.calculate_rouge_2(predicted, expected)
            metrics['rouge-l'] = MetricsCalculator.calculate_rouge_l(predicted, expected)
        elif self.task_config.metric == 'semantic_similarity':
            metrics['semantic_similarity'] = MetricsCalculator.calculate_semantic_similarity(predicted, expected)
        elif self.task_config.metric == 'bertscore':
            # Alias for semantic similarity
            metrics['bertscore'] = MetricsCalculator.calculate_semantic_similarity(predicted, expected)
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
    
    async def run_sample(self, sample: EvalSample) -> EvalSampleResult:
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
            
            return EvalSampleResult(
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
            
            return EvalSampleResult(
                sample_id=sample.id,
                input_text=sample.input_text,
                expected_output=sample.expected_output,
                actual_output=None,
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
    
    def calculate_metrics(self, predicted: str, expected: str, choices: List[str] = None) -> Dict[str, float]:
        """Calculate classification-specific metrics including accuracy."""
        # Get base metrics
        metrics = super().calculate_metrics(predicted, expected)
        
        # Add accuracy metric (1.0 if correct, 0.0 if incorrect)
        metrics['accuracy'] = 1.0 if predicted == expected else 0.0
        
        # Add classification-specific metrics if choices are provided
        if choices and expected in choices:
            # Add position-based metrics
            predicted_idx = choices.index(predicted) if predicted in choices else -1
            expected_idx = choices.index(expected)
            
            # Distance metric (normalized)
            if predicted_idx >= 0:
                distance = abs(predicted_idx - expected_idx) / (len(choices) - 1) if len(choices) > 1 else 0
                metrics['choice_distance'] = 1.0 - distance  # Higher is better
            else:
                metrics['choice_distance'] = 0.0
            
            # Random baseline (for comparison)
            metrics['random_baseline'] = 1.0 / len(choices) if choices else 0.0
        
        return metrics
    
    async def run_sample(self, sample: EvalSample) -> EvalSampleResult:
        """Run classification evaluation on a single sample."""
        start_time = time.time()
        
        # Log sample processing start
        log_counter("eval_sample_processing_started", labels={
            "task_type": "classification",
            "provider": self.model_config.get('provider', 'unknown'),
            "model": self.model_config.get('model_id', 'unknown')
        })
        
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
            
            # Log successful sample processing
            log_histogram("eval_sample_processing_duration", processing_time, labels={
                "task_type": "classification",
                "provider": self.model_config.get('provider', 'unknown'),
                "model": self.model_config.get('model_id', 'unknown'),
                "status": "success"
            })
            log_counter("eval_sample_processing_success", labels={
                "task_type": "classification",
                "provider": self.model_config.get('provider', 'unknown'),
                "model": self.model_config.get('model_id', 'unknown')
            })
            if retry_count > 0:
                log_histogram("eval_sample_retry_count", retry_count, labels={
                    "task_type": "classification",
                    "provider": self.model_config.get('provider', 'unknown')
                })
            
            # Log metric values
            for metric_name, metric_value in result_data['metrics'].items():
                if isinstance(metric_value, (int, float)):
                    log_histogram(f"eval_metric_{metric_name}", metric_value, labels={
                        "task_type": "classification",
                        "provider": self.model_config.get('provider', 'unknown'),
                        "model": self.model_config.get('model_id', 'unknown')
                    })
            
            return EvalSampleResult(
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
            
            # Log error metrics
            log_histogram("eval_sample_processing_duration", processing_time, labels={
                "task_type": "classification",
                "provider": self.model_config.get('provider', 'unknown'),
                "model": self.model_config.get('model_id', 'unknown'),
                "status": "error"
            })
            log_counter("eval_sample_processing_error", labels={
                "task_type": "classification",
                "provider": self.model_config.get('provider', 'unknown'),
                "model": self.model_config.get('model_id', 'unknown'),
                "error_category": error_info.get('error_category', 'unknown'),
                "error_type": error_info.get('error_type', 'unknown')
            })
            
            return EvalSampleResult(
                sample_id=sample.id,
                input_text=sample.input_text,
                expected_output=sample.expected_output,
                actual_output=None,
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
        generation_start = time.time()
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
            
            # Log generation metrics
            generation_duration = time.time() - generation_start
            log_histogram("eval_llm_generation_duration", generation_duration, labels={
                "task_type": "classification",
                "provider": self.model_config.get('provider', 'unknown'),
                "model": self.model_config.get('model_id', 'unknown')
            })
            log_histogram("eval_llm_response_length", len(response), labels={
                "task_type": "classification",
                "provider": self.model_config.get('provider', 'unknown'),
                "model": self.model_config.get('model_id', 'unknown')
            })
            
            return response
            
        except Exception as e:
            generation_duration = time.time() - generation_start
            log_counter("eval_llm_generation_error", labels={
                "task_type": "classification",
                "provider": self.model_config.get('provider', 'unknown'),
                "model": self.model_config.get('model_id', 'unknown'),
                "error_type": type(e).__name__
            })
            log_histogram("eval_llm_generation_error_duration", generation_duration, labels={
                "task_type": "classification",
                "provider": self.model_config.get('provider', 'unknown')
            })
            logger.error(f"LLM generation failed: {e}")
            raise

class LogProbRunner(BaseEvalRunner):
    """Runner for log probability evaluation tasks."""
    
    async def run_sample(self, sample: EvalSample) -> EvalSampleResult:
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
            
            return EvalSampleResult(
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
            
            return EvalSampleResult(
                sample_id=sample.id,
                input_text=sample.input_text,
                expected_output=sample.expected_output,
                actual_output=None,
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
            # Use the LLM interface to get logprobs
            result = await self.llm_interface.get_logprobs(text)
            
            if result and 'logprobs' in result and result['logprobs']:
                # Sum the log probabilities
                total_logprob = sum(result['logprobs'])
                return total_logprob
            else:
                logger.warning("No logprobs returned from LLM provider")
                return float('-inf')
            
        except Exception as e:
            logger.error(f"Failed to get logprob: {e}")
            return float('-inf')
    
    async def _get_continuation_logprob(self, prompt: str, continuation: str) -> float:
        """Get log probability of a continuation given prompt."""
        try:
            # Use the LLM interface to get completion logprobs
            result = await self.llm_interface.get_completion_logprobs(prompt, continuation)
            
            if result and 'logprobs' in result and result['logprobs']:
                # Sum the log probabilities for the continuation
                total_logprob = sum(result['logprobs'])
                return total_logprob
            else:
                # Fallback to full text logprob
                return await self._get_text_logprob(prompt + continuation)
            
        except Exception as e:
            logger.error(f"Failed to get continuation logprob: {e}")
            return float('-inf')

class GenerationRunner(BaseEvalRunner):
    """Runner for text generation tasks."""
    
    async def run_sample(self, sample: EvalSample) -> EvalSampleResult:
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
            
            return EvalSampleResult(
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
            
            return EvalSampleResult(
                sample_id=sample.id,
                input_text=sample.input_text,
                expected_output=sample.expected_output,
                actual_output=None,
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
        
        # Expose configuration values as properties for tests
        self.max_concurrent_requests = model_config.get('max_concurrent_requests', 10)
        self.request_timeout = model_config.get('request_timeout', 30.0)
        self.retry_attempts = model_config.get('retry_attempts', 3)
        
        # Store LLM interface for tests
        self.llm_interface = None
        
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
    
    async def run_single_sample(self, task_config: TaskConfig, sample: Any) -> EvalSampleResult:
        """Run evaluation on a single sample."""
        return await self.runner.run_sample(sample)
    
    async def run_evaluation(self, max_samples: int = None, 
                           progress_callback: callable = None) -> List[EvalSampleResult]:
        """Run evaluation on the task dataset."""
        return await self.runner.run_evaluation(max_samples=max_samples, 
                                               progress_callback=progress_callback)
    
    def calculate_aggregate_metrics(self, results: List[EvalSampleResult]) -> Dict[str, float]:
        """Calculate aggregate metrics from results."""
        if hasattr(self.runner, 'calculate_aggregate_metrics'):
            return self.runner.calculate_aggregate_metrics(results)
        
        # Default implementation
        if not results:
            return {}
        
        metrics = {}
        for key in results[0].metrics.keys():
            values = [r.metrics.get(key, 0) for r in results if not r.error_info]
            if values:
                metrics[f"{key}_mean"] = sum(values) / len(values)
        
        return metrics
    
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
                           progress_callback: callable = None) -> List[EvalSampleResult]:
        """Run evaluation on the task dataset."""
        logger.info(f"Starting evaluation: {self.task_config.name}")
        eval_run_start = time.time()
        
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
                error_result = EvalSampleResult(
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
        
        # Log evaluation run completion metrics
        eval_run_duration = time.time() - eval_run_start
        log_histogram("eval_runner_run_duration", eval_run_duration, labels={
            "task_type": self.task_config.task_type,
            "provider": self.model_config.get('provider', 'unknown'),
            "model": self.model_config.get('model_id', 'unknown'),
            "status": "completed" if error_count < len(samples) else "failed"
        })
        log_counter("eval_runner_run_completed", labels={
            "task_type": self.task_config.task_type,
            "provider": self.model_config.get('provider', 'unknown'),
            "model": self.model_config.get('model_id', 'unknown')
        })
        log_histogram("eval_runner_sample_count", len(results), labels={
            "task_type": self.task_config.task_type,
            "provider": self.model_config.get('provider', 'unknown'),
            "model": self.model_config.get('model_id', 'unknown')
        })
        log_histogram("eval_runner_error_count", error_count, labels={
            "task_type": self.task_config.task_type,
            "provider": self.model_config.get('provider', 'unknown'),
            "model": self.model_config.get('model_id', 'unknown')
        })
        log_histogram("eval_runner_total_retries", retry_count, labels={
            "task_type": self.task_config.task_type,
            "provider": self.model_config.get('provider', 'unknown'),
            "model": self.model_config.get('model_id', 'unknown')
        })
        
        # Calculate and log success rate
        if len(results) > 0:
            log_histogram("eval_runner_success_rate", success_rate / 100.0, labels={
                "task_type": self.task_config.task_type,
                "provider": self.model_config.get('provider', 'unknown'),
                "model": self.model_config.get('model_id', 'unknown')
            })
        
        return results
    
    def calculate_aggregate_metrics(self, results: List[EvalSampleResult]) -> Dict[str, float]:
        """Calculate aggregate metrics across all results."""
        start_time = time.time()
        
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
        
        # Log aggregation metrics
        duration = time.time() - start_time
        log_histogram("eval_aggregation_duration", duration, labels={
            "result_count": str(len(results)),
            "metric_count": str(len(aggregate_metrics))
        })
        
        # Log aggregate metric values
        for metric_name, metric_value in aggregate_metrics.items():
            if isinstance(metric_value, (int, float)):
                log_histogram(f"eval_aggregate_{metric_name}", metric_value, labels={
                    "sample_count": str(len(results))
                })
        
        return aggregate_metrics