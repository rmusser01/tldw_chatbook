# base_runner.py
# Description: Base classes and data structures for evaluation runners
#
"""
Base Evaluation Runner
----------------------

Provides abstract base class and common data structures for all evaluation runners.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, AsyncIterator
from datetime import datetime
from pathlib import Path

from loguru import logger

from .eval_errors import (
    get_error_handler, EvaluationError, ExecutionError,
    ErrorContext, ErrorCategory, ErrorSeverity
)


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
    metrics: Dict[str, float]
    latency_ms: float
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseEvalRunner(ABC):
    """
    Abstract base class for all evaluation runners.
    
    Provides common functionality for:
    - Progress tracking
    - Error handling
    - Metric calculation
    - Result aggregation
    """
    
    def __init__(self, task_config: Dict[str, Any], model_config: Dict[str, Any]):
        """
        Initialize base runner.
        
        Args:
            task_config: Task configuration including dataset and metrics
            model_config: Model configuration including provider and parameters
        """
        self.task_config = task_config
        self.model_config = model_config
        self.error_handler = get_error_handler()
        
        # Progress tracking
        self._progress_callback = None
        self._current_progress = 0
        self._total_samples = 0
        
        # Results storage
        self.sample_results: List[EvalSampleResult] = []
        self.errors: List[str] = []
    
    @abstractmethod
    async def evaluate_sample(self, sample: EvalSample) -> EvalSampleResult:
        """
        Evaluate a single sample.
        
        Args:
            sample: The sample to evaluate
            
        Returns:
            Result of evaluating the sample
        """
        pass
    
    @abstractmethod
    def calculate_metrics(self, expected: str, actual: str) -> Dict[str, float]:
        """
        Calculate metrics for a sample.
        
        Args:
            expected: Expected output
            actual: Actual model output
            
        Returns:
            Dictionary of metric names to values
        """
        pass
    
    async def run(
        self, 
        samples: List[EvalSample],
        progress_callback: Optional[callable] = None,
        max_concurrent: int = 1
    ) -> EvalRunResult:
        """
        Run evaluation on all samples.
        
        Args:
            samples: List of samples to evaluate
            progress_callback: Optional callback for progress updates
            max_concurrent: Maximum concurrent evaluations
            
        Returns:
            Aggregated evaluation results
        """
        self._progress_callback = progress_callback
        self._total_samples = len(samples)
        self._current_progress = 0
        
        start_time = datetime.now()
        
        # Process samples with concurrency control
        if max_concurrent == 1:
            # Sequential processing
            for sample in samples:
                result = await self._evaluate_with_progress(sample)
                self.sample_results.append(result)
        else:
            # Concurrent processing
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_sample(sample):
                async with semaphore:
                    return await self._evaluate_with_progress(sample)
            
            tasks = [process_sample(sample) for sample in samples]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    self.errors.append(str(result))
                else:
                    self.sample_results.append(result)
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics()
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return EvalRunResult(
            task_name=self.task_config.get('name', 'Unknown Task'),
            metrics=aggregate_metrics,
            samples_evaluated=len(self.sample_results),
            duration_seconds=duration,
            timestamp=datetime.now().isoformat(),
            errors=self.errors
        )
    
    async def _evaluate_with_progress(self, sample: EvalSample) -> EvalSampleResult:
        """Evaluate sample and update progress."""
        try:
            result = await self.evaluate_sample(sample)
            self._current_progress += 1
            
            if self._progress_callback:
                progress = EvalProgress(
                    current=self._current_progress,
                    total=self._total_samples,
                    current_task=f"Sample {sample.id}"
                )
                await self._progress_callback(progress)
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating sample {sample.id}: {e}")
            self.errors.append(f"Sample {sample.id}: {str(e)}")
            
            # Return error result
            return EvalSampleResult(
                sample_id=sample.id,
                input_text=sample.input_text,
                expected_output=sample.expected_output,
                actual_output="",
                metrics={},
                latency_ms=0,
                error=str(e)
            )
    
    def _calculate_aggregate_metrics(self) -> Dict[str, Any]:
        """Calculate aggregate metrics from sample results."""
        if not self.sample_results:
            return {}
        
        # Collect all metric names
        metric_names = set()
        for result in self.sample_results:
            if result.metrics:
                metric_names.update(result.metrics.keys())
        
        # Calculate averages for each metric
        aggregate = {}
        for metric_name in metric_names:
            values = [
                result.metrics.get(metric_name, 0)
                for result in self.sample_results
                if result.metrics and not result.error
            ]
            
            if values:
                aggregate[f"{metric_name}_mean"] = sum(values) / len(values)
                aggregate[f"{metric_name}_min"] = min(values)
                aggregate[f"{metric_name}_max"] = max(values)
        
        # Add performance metrics
        latencies = [r.latency_ms for r in self.sample_results if r.latency_ms > 0]
        if latencies:
            aggregate["latency_mean_ms"] = sum(latencies) / len(latencies)
            aggregate["latency_p95_ms"] = sorted(latencies)[int(len(latencies) * 0.95)]
        
        # Add cost metrics
        costs = [r.cost for r in self.sample_results if r.cost is not None]
        if costs:
            aggregate["total_cost"] = sum(costs)
            aggregate["cost_per_sample"] = sum(costs) / len(costs)
        
        # Add error rate
        error_count = sum(1 for r in self.sample_results if r.error)
        aggregate["error_rate"] = error_count / len(self.sample_results)
        
        return aggregate
    
    def get_failed_samples(self) -> List[EvalSampleResult]:
        """Get list of samples that failed evaluation."""
        return [r for r in self.sample_results if r.error]
    
    def get_successful_samples(self) -> List[EvalSampleResult]:
        """Get list of samples that succeeded evaluation."""
        return [r for r in self.sample_results if not r.error]