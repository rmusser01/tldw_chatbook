# ab_testing.py
# Description: A/B testing functionality for comparing model performance
#
"""
A/B Testing for LLM Evaluations
-------------------------------

Provides functionality for:
- Running evaluations on multiple models simultaneously
- Statistical comparison of results
- Significance testing
- Performance visualization
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import statistics
import math
from loguru import logger

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available. Statistical tests will be limited.")

from .eval_orchestrator import EvaluationOrchestrator
from .eval_runner import EvalSampleResult
from tldw_chatbook.Metrics.metrics_logger import log_counter, log_histogram

@dataclass
class ABTestConfig:
    """Configuration for an A/B test."""
    name: str
    description: str
    task_id: str
    model_a_id: str
    model_b_id: str
    sample_size: Optional[int] = None
    confidence_level: float = 0.95
    minimum_effect_size: float = 0.05
    stratify_by: Optional[str] = None  # Field to stratify sampling
    metrics_to_compare: List[str] = field(default_factory=lambda: ['accuracy', 'f1_score'])
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class ABTestResult:
    """Results from an A/B test comparison."""
    test_id: str
    test_name: str
    model_a_name: str
    model_b_name: str
    sample_size: int
    
    # Metrics for each model
    model_a_metrics: Dict[str, float]
    model_b_metrics: Dict[str, float]
    
    # Statistical analysis
    statistical_tests: Dict[str, Dict[str, Any]]
    winner: Optional[str] = None  # 'model_a', 'model_b', or None (no significant difference)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Performance data
    model_a_latency: float = 0.0
    model_b_latency: float = 0.0
    model_a_cost: float = 0.0
    model_b_cost: float = 0.0
    
    # Sample-level results for detailed analysis
    sample_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class ABTestRunner:
    """Runs A/B tests comparing two models on the same evaluation task."""
    
    def __init__(self, orchestrator: EvaluationOrchestrator):
        """
        Initialize the A/B test runner.
        
        Args:
            orchestrator: Evaluation orchestrator instance
        """
        self.orchestrator = orchestrator
        logger.info("ABTestRunner initialized")
    
    async def run_ab_test(self, 
                         config: ABTestConfig,
                         progress_callback: Optional[Callable[[int, int, str], None]] = None) -> ABTestResult:
        """
        Run an A/B test comparing two models.
        
        Args:
            config: A/B test configuration
            progress_callback: Optional callback for progress updates (completed, total, status)
            
        Returns:
            ABTestResult with comparison data
        """
        logger.info(f"Starting A/B test: {config.name}")
        start_time = time.time()
        
        # Log test start
        log_counter("ab_test_started", labels={
            "task_id": config.task_id,
            "model_a": config.model_a_id,
            "model_b": config.model_b_id
        })
        
        # Generate test ID
        test_id = f"ab_test_{int(time.time())}"
        
        # Get model names
        model_a_info = self.orchestrator.db.get_model(config.model_a_id)
        model_b_info = self.orchestrator.db.get_model(config.model_b_id)
        
        model_a_name = model_a_info['name'] if model_a_info else config.model_a_id
        model_b_name = model_b_info['name'] if model_b_info else config.model_b_id
        
        # Initialize result
        result = ABTestResult(
            test_id=test_id,
            test_name=config.name,
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            sample_size=0,
            model_a_metrics={},
            model_b_metrics={},
            statistical_tests={}
        )
        
        try:
            # Run evaluations in parallel
            if progress_callback:
                progress_callback(0, 2, "Starting model evaluations...")
            
            # Create progress wrappers
            def model_a_progress(completed: int, total: int, sample_result):
                if progress_callback:
                    # Scale to first half of progress
                    overall_progress = int(completed / total * 50)
                    progress_callback(overall_progress, 100, f"Model A: {completed}/{total}")
            
            def model_b_progress(completed: int, total: int, sample_result):
                if progress_callback:
                    # Scale to second half of progress
                    overall_progress = 50 + int(completed / total * 50)
                    progress_callback(overall_progress, 100, f"Model B: {completed}/{total}")
            
            # Run both evaluations
            model_a_task = asyncio.create_task(
                self.orchestrator.run_evaluation(
                    task_id=config.task_id,
                    model_id=config.model_a_id,
                    run_name=f"{config.name} - Model A",
                    max_samples=config.sample_size,
                    progress_callback=model_a_progress
                )
            )
            
            model_b_task = asyncio.create_task(
                self.orchestrator.run_evaluation(
                    task_id=config.task_id,
                    model_id=config.model_b_id,
                    run_name=f"{config.name} - Model B",
                    max_samples=config.sample_size,
                    progress_callback=model_b_progress
                )
            )
            
            # Wait for both to complete
            run_a_id, run_b_id = await asyncio.gather(model_a_task, model_b_task)
            
            if progress_callback:
                progress_callback(100, 100, "Analyzing results...")
            
            # Get results
            results_a = self.orchestrator.get_run_results(run_a_id)
            results_b = self.orchestrator.get_run_results(run_b_id)
            
            # Get run summaries for metrics and timing
            summary_a = self.orchestrator.get_run_summary(run_a_id)
            summary_b = self.orchestrator.get_run_summary(run_b_id)
            
            # Update result with metrics
            result.model_a_metrics = summary_a.get('metrics', {})
            result.model_b_metrics = summary_b.get('metrics', {})
            result.sample_size = len(results_a)
            
            # Calculate latencies
            if summary_a.get('duration'):
                result.model_a_latency = summary_a['duration'] / len(results_a) if results_a else 0
            if summary_b.get('duration'):
                result.model_b_latency = summary_b['duration'] / len(results_b) if results_b else 0
            
            # Perform statistical analysis
            result.statistical_tests = self._perform_statistical_tests(
                results_a, results_b, config
            )
            
            # Determine winner
            result.winner = self._determine_winner(
                result.statistical_tests, 
                config.metrics_to_compare,
                config.confidence_level
            )
            
            # Calculate confidence intervals
            result.confidence_intervals = self._calculate_confidence_intervals(
                results_a, results_b, config.metrics_to_compare
            )
            
            # Store sample-level results for detailed analysis
            result.sample_results = self._merge_sample_results(results_a, results_b)
            
            # Mark completion
            result.completed_at = datetime.now()
            
            # Log completion
            duration = time.time() - start_time
            log_histogram("ab_test_duration", duration, labels={
                "winner": result.winner or "tie"
            })
            log_counter("ab_test_completed", labels={
                "winner": result.winner or "tie"
            })
            
            logger.info(f"A/B test completed. Winner: {result.winner or 'No significant difference'}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error running A/B test: {e}")
            log_counter("ab_test_error", labels={
                "error_type": type(e).__name__
            })
            raise
    
    def _perform_statistical_tests(self, 
                                  results_a: List[Dict[str, Any]], 
                                  results_b: List[Dict[str, Any]],
                                  config: ABTestConfig) -> Dict[str, Dict[str, Any]]:
        """Perform statistical tests on the results."""
        tests = {}
        
        for metric in config.metrics_to_compare:
            # Extract metric values
            values_a = []
            values_b = []
            
            for res_a, res_b in zip(results_a, results_b):
                # Get metric value, handle nested structure
                metric_a = res_a.get('metrics', {}).get(metric)
                metric_b = res_b.get('metrics', {}).get(metric)
                
                if metric_a is not None and metric_b is not None:
                    # Handle dict metrics (e.g., {'value': 0.95})
                    if isinstance(metric_a, dict):
                        metric_a = metric_a.get('value', 0)
                    if isinstance(metric_b, dict):
                        metric_b = metric_b.get('value', 0)
                    
                    values_a.append(float(metric_a))
                    values_b.append(float(metric_b))
            
            if not values_a or not values_b:
                logger.warning(f"No values found for metric: {metric}")
                continue
            
            # Calculate basic statistics
            mean_a = statistics.mean(values_a)
            mean_b = statistics.mean(values_b)
            std_a = statistics.stdev(values_a) if len(values_a) > 1 else 0
            std_b = statistics.stdev(values_b) if len(values_b) > 1 else 0
            
            # Perform statistical tests if scipy is available
            if SCIPY_AVAILABLE:
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(values_a, values_b)
                
                # Perform Mann-Whitney U test (non-parametric alternative)
                u_stat, u_p_value = stats.mannwhitneyu(values_a, values_b, alternative='two-sided')
            else:
                # Simple approximation without scipy
                # Using Welch's t-test approximation
                n_a, n_b = len(values_a), len(values_b)
                var_a, var_b = std_a**2, std_b**2
                
                t_stat = (mean_a - mean_b) / math.sqrt(var_a/n_a + var_b/n_b) if (var_a/n_a + var_b/n_b) > 0 else 0
                
                # Approximate p-value using normal distribution
                # This is a rough approximation
                z_score = abs(t_stat)
                p_value = 2 * (1 - self._normal_cdf(z_score))
                
                u_stat = 0  # Not calculated without scipy
                u_p_value = p_value  # Use same approximation
            
            # Calculate effect size (Cohen's d)
            pooled_std = math.sqrt((std_a**2 + std_b**2) / 2) if std_a > 0 or std_b > 0 else 1
            effect_size = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0
            
            tests[metric] = {
                'mean_a': mean_a,
                'mean_b': mean_b,
                'std_a': std_a,
                'std_b': std_b,
                't_statistic': t_stat,
                'p_value': p_value,
                'effect_size': effect_size,
                'u_statistic': u_stat,
                'u_p_value': u_p_value,
                'sample_size_a': len(values_a),
                'sample_size_b': len(values_b),
                'is_significant': p_value < (1 - config.confidence_level),
                'difference': mean_a - mean_b,
                'relative_difference': ((mean_a - mean_b) / mean_b * 100) if mean_b != 0 else 0
            }
        
        return tests
    
    def _determine_winner(self, 
                         statistical_tests: Dict[str, Dict[str, Any]], 
                         metrics: List[str],
                         confidence_level: float) -> Optional[str]:
        """Determine the winner based on statistical tests."""
        model_a_wins = 0
        model_b_wins = 0
        
        for metric in metrics:
            if metric in statistical_tests:
                test = statistical_tests[metric]
                
                # Check if difference is significant
                if test['is_significant']:
                    # Check which model is better
                    if test['difference'] > 0:
                        model_a_wins += 1
                    else:
                        model_b_wins += 1
        
        # Determine overall winner
        if model_a_wins > model_b_wins:
            return 'model_a'
        elif model_b_wins > model_a_wins:
            return 'model_b'
        else:
            return None  # Tie or no significant differences
    
    def _calculate_confidence_intervals(self,
                                      results_a: List[Dict[str, Any]],
                                      results_b: List[Dict[str, Any]],
                                      metrics: List[str]) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for the difference in metrics."""
        intervals = {}
        
        for metric in metrics:
            values_a = []
            values_b = []
            
            for res_a, res_b in zip(results_a, results_b):
                metric_a = res_a.get('metrics', {}).get(metric)
                metric_b = res_b.get('metrics', {}).get(metric)
                
                if metric_a is not None and metric_b is not None:
                    if isinstance(metric_a, dict):
                        metric_a = metric_a.get('value', 0)
                    if isinstance(metric_b, dict):
                        metric_b = metric_b.get('value', 0)
                    
                    values_a.append(float(metric_a))
                    values_b.append(float(metric_b))
            
            if values_a and values_b:
                # Calculate difference for each pair
                differences = [a - b for a, b in zip(values_a, values_b)]
                
                # Calculate confidence interval
                mean_diff = statistics.mean(differences)
                std_diff = statistics.stdev(differences) if len(differences) > 1 else 0
                n = len(differences)
                
                # 95% confidence interval
                margin = 1.96 * (std_diff / math.sqrt(n)) if n > 0 else 0
                
                intervals[f"{metric}_difference"] = (mean_diff - margin, mean_diff + margin)
        
        return intervals
    
    def _merge_sample_results(self, 
                            results_a: List[Dict[str, Any]], 
                            results_b: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge sample results for detailed analysis."""
        merged = []
        
        for res_a, res_b in zip(results_a, results_b):
            merged.append({
                'sample_id': res_a.get('sample_id'),
                'input': res_a.get('input_data', {}).get('input', ''),
                'expected': res_a.get('expected_output'),
                'model_a_output': res_a.get('actual_output'),
                'model_b_output': res_b.get('actual_output'),
                'model_a_metrics': res_a.get('metrics', {}),
                'model_b_metrics': res_b.get('metrics', {}),
                'model_a_correct': res_a.get('metrics', {}).get('exact_match', 0) == 1,
                'model_b_correct': res_b.get('metrics', {}).get('exact_match', 0) == 1
            })
        
        return merged
    
    def _normal_cdf(self, z: float) -> float:
        """
        Approximate normal cumulative distribution function.
        Uses the error function approximation when scipy is not available.
        """
        # Approximation of the error function
        # erf(x) ≈ sign(x) * sqrt(1 - exp(-x^2 * (4/π + ax^2) / (1 + ax^2)))
        # where a ≈ 0.147
        a = 0.147
        x = z / math.sqrt(2)
        
        # Calculate erf approximation
        x_squared = x * x
        numerator = 4.0 / math.pi + a * x_squared
        denominator = 1.0 + a * x_squared
        ratio = -x_squared * numerator / denominator
        erf_approx = math.copysign(math.sqrt(1 - math.exp(ratio)), x)
        
        # Convert to CDF: Φ(z) = 0.5 * (1 + erf(z/√2))
        return 0.5 * (1 + erf_approx)

class ABTestOrchestrator:
    """Orchestrates multiple A/B tests and manages results."""
    
    def __init__(self, eval_orchestrator: EvaluationOrchestrator):
        """Initialize the A/B test orchestrator."""
        self.eval_orchestrator = eval_orchestrator
        self.runner = ABTestRunner(eval_orchestrator)
        self.active_tests = {}
    
    async def create_and_run_ab_test(self, config: ABTestConfig) -> str:
        """Create and run an A/B test, returning the test ID."""
        test_id = f"ab_{int(time.time())}_{config.model_a_id}_{config.model_b_id}"
        
        # Store in database
        self.eval_orchestrator.db.create_ab_test(
            test_id=test_id,
            name=config.name,
            description=config.description,
            task_id=config.task_id,
            model_a_id=config.model_a_id,
            model_b_id=config.model_b_id,
            config=config.__dict__
        )
        
        # Run test
        async def run_test():
            try:
                result = await self.runner.run_ab_test(config)
                # Store results
                self.eval_orchestrator.db.update_ab_test_results(test_id, result)
                return result
            except Exception as e:
                logger.error(f"A/B test {test_id} failed: {e}")
                self.eval_orchestrator.db.update_ab_test_status(test_id, 'failed', str(e))
                raise
        
        # Track active test
        self.active_tests[test_id] = asyncio.create_task(run_test())
        
        return test_id
    
    def get_test_status(self, test_id: str) -> Dict[str, Any]:
        """Get the status of an A/B test."""
        if test_id in self.active_tests:
            task = self.active_tests[test_id]
            if task.done():
                return {'status': 'completed', 'result': task.result()}
            else:
                return {'status': 'running'}
        else:
            # Check database
            return self.eval_orchestrator.db.get_ab_test_status(test_id)
    
    def list_ab_tests(self, **filters) -> List[Dict[str, Any]]:
        """List all A/B tests with optional filters."""
        return self.eval_orchestrator.db.list_ab_tests(**filters)