# Evals Window Implementation Guide

## Executive Summary

This guide provides a comprehensive blueprint for implementing the Evals (Evaluation) Window from scratch. The Evals system enables benchmarking of LLM models across various tasks, datasets, and metrics to assess performance, compare models, and track improvements over time.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Core Components](#core-components)
3. [Window Implementation](#window-implementation)
4. [Task Management](#task-management)
5. [Evaluation Runner](#evaluation-runner)
6. [Results Analysis](#results-analysis)
7. [Backend Integration](#backend-integration)
8. [Testing Strategies](#testing-strategies)
9. [Implementation Roadmap](#implementation-roadmap)
10. [Advanced Features](#advanced-features)

---

## System Architecture

### Overview

The Evals system follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                     Evals Window (UI)                       │
│  ┌─────────────┬──────────────┬──────────────┬──────────┐ │
│  │ Quick Setup │ Task Creator │ Active Runs  │ Results  │ │
│  └──────┬──────┴──────┬───────┴──────┬───────┴──────┬───┘ │
└─────────┼─────────────┼──────────────┼──────────────┼─────┘
          │             │              │              │
┌─────────▼─────────────▼──────────────▼──────────────▼─────┐
│              Evaluation Orchestrator                        │
│  ┌────────────┬─────────────┬──────────────┬────────────┐ │
│  │ Task Loader│ Config Mgr  │ Run Manager  │ Error Mgr  │ │
│  └────────────┴─────────────┴──────────────┴────────────┘ │
└─────────────────────────────────────────────────────────────┘
          │                    │                     │
┌─────────▼────────┐ ┌────────▼────────┐ ┌─────────▼────────┐
│   Task Runners   │ │  LLM Providers  │ │   Metrics Calc   │
│ ┌──────────────┐ │ │ ┌────────────┐ │ │ ┌──────────────┐ │
│ │ QA Runner    │ │ │ │ OpenAI     │ │ │ │ Accuracy     │ │
│ │ Generation   │ │ │ │ Anthropic  │ │ │ │ F1 Score     │ │
│ │ Code Runner  │ │ │ │ Local LLMs │ │ │ │ BLEU         │ │
│ │ Multi-Choice │ │ │ │ Custom     │ │ │ │ Custom       │ │
│ └──────────────┘ │ │ └────────────┘ │ │ └──────────────┘ │
└──────────────────┘ └─────────────────┘ └──────────────────┘
          │                    │                     │
┌─────────▼────────────────────▼─────────────────────▼───────┐
│                      Evals Database                         │
│  ┌──────────┬────────────┬───────────┬─────────────────┐  │
│  │ Tasks    │ Models     │ Runs      │ Results         │  │
│  │ Datasets │ Templates  │ Metrics   │ Configurations  │  │
│  └──────────┴────────────┴───────────┴─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Modularity**: Each component has a single responsibility
2. **Extensibility**: Easy to add new task types, metrics, and providers
3. **Reliability**: Comprehensive error handling and recovery
4. **Performance**: Async operations, parallel execution, caching
5. **Usability**: Intuitive UI with progressive disclosure

---

## Core Components

### 1. TaskLoader (`task_loader.py`)

**Purpose**: Load and parse evaluation tasks from various formats

```python
class TaskLoader:
    """Loads evaluation tasks from multiple formats"""
    
    def load_task(self, source: Union[str, Path, Dict], 
                  format_type: str = 'auto') -> TaskConfig:
        """Load task from file or configuration"""
        
    def create_task_from_template(self, template_name: str, 
                                 **kwargs) -> TaskConfig:
        """Create task from built-in template"""
        
    def validate_task(self, task_config: TaskConfig) -> List[str]:
        """Validate task configuration"""
```

**Supported Formats**:
- Eleuther AI Evaluation Harness YAML
- Custom JSON/YAML format
- HuggingFace datasets
- CSV/TSV files

### 2. EvalRunner (`eval_runner.py`)

**Purpose**: Execute evaluation tasks against LLM models

```python
class EvalRunner:
    """Executes evaluation tasks"""
    
    async def run_evaluation(self, max_samples: int = None,
                            progress_callback: Callable = None) -> List[EvalSampleResult]:
        """Run complete evaluation"""
        
    def calculate_aggregate_metrics(self, results: List[EvalSampleResult]) -> Dict[str, float]:
        """Calculate aggregate metrics from results"""
```

**Key Features**:
- Parallel sample processing
- Progress tracking
- Error recovery
- Multiple task type support

### 3. EvaluationOrchestrator (`eval_orchestrator.py`)

**Purpose**: Coordinate the complete evaluation pipeline

```python
class EvaluationOrchestrator:
    """Orchestrates evaluation workflow"""
    
    async def run_evaluation(self, task_id: str, model_id: str,
                            run_name: str = None,
                            max_samples: int = None,
                            config_overrides: Dict = None,
                            progress_callback: Callable = None) -> str:
        """Run complete evaluation"""
        
    def create_model_config(self, name: str, provider: str, 
                          model_id: str, config: Dict = None) -> str:
        """Create model configuration"""
```

### 4. Task Configuration Structure

```python
@dataclass
class TaskConfig:
    name: str
    description: str
    task_type: str  # 'question_answer', 'generation', 'classification', 'code_generation'
    dataset_name: str
    dataset_config: Optional[str] = None
    split: str = 'test'
    num_fewshot: int = 0
    
    # Generation parameters
    generation_kwargs: Dict[str, Any] = None
    stop_sequences: List[str] = None
    
    # Evaluation parameters
    metric: str = 'exact_match'
    
    # Format templates
    doc_to_text: Optional[str] = None
    doc_to_target: Optional[str] = None
    doc_to_choice: Optional[str] = None
```

---

## Window Implementation

### Main Window Structure (`Evals_Window_v3_unified.py`)

```python
class EvalsWindow(Container):
    """Unified evaluation dashboard"""
    
    # Reactive state management
    current_run_status = reactive("idle")
    active_run_id = reactive(None)
    evaluation_progress = reactive(0.0)
    selected_provider = reactive(None)
    selected_model = reactive(None)
    selected_dataset = reactive(None)
    
    def compose(self) -> ComposeResult:
        """Build UI components"""
        # Quick start bar
        # Collapsible sections
        # Results dashboard
        # Status bar
```

### UI Sections

#### 1. Quick Start Bar
```python
with Container(classes="quick-start-bar"):
    yield Static("🧪 Evaluation Lab", classes="dashboard-title")
    with Horizontal(classes="quick-actions"):
        yield Button("🚀 Run MMLU on GPT-4", id="quick-mmlu")
        yield Button("📊 Compare Claude vs GPT", id="quick-compare")
        yield Button("🔄 Rerun Last Test", id="quick-rerun")
```

#### 2. Task Creation Section
```python
with Collapsible(title="➕ Create New Task", id="task-creation-section"):
    # Task name, type, prompt template
    # Metrics selection
    # Success criteria
    # Import/save options
```

#### 3. Quick Configuration
```python
with Collapsible(title="⚡ Quick Setup", collapsed=False):
    # Task/Model/Dataset selectors
    # Sample count
    # Cost estimation
    # Template cards
```

#### 4. Active Evaluations Monitor
```python
with Collapsible(title="🔄 Active Evaluations", id="active-eval-section"):
    # Progress bars
    # Live metrics
    # Cancel buttons
    # Log viewer
```

#### 5. Results Dashboard
```python
with Container(classes="results-dashboard"):
    # Results list
    # Quick stats grid
    # Comparison tools
    # Export options
```

### Reactive State Management

```python
class EvalsWindow(Container):
    def watch_current_run_status(self, old: str, new: str):
        """React to status changes"""
        if new == "running":
            self.expand_active_section()
            self.start_progress_updates()
        elif new == "completed":
            self.update_results_dashboard()
            self.show_completion_notification()
            
    def watch_evaluation_progress(self, old: float, new: float):
        """Update progress displays"""
        self.update_progress_bar(new)
        self.update_cost_estimate(new)
```

---

## Task Management

### Task Types

#### 1. Question Answering
```python
class QuestionAnswerRunner(BaseTaskRunner):
    """Handles Q&A evaluation tasks"""
    
    def format_prompt(self, sample: Dict) -> str:
        return f"Question: {sample['question']}\nAnswer:"
        
    def evaluate_response(self, predicted: str, expected: str) -> Dict:
        return {
            'exact_match': predicted.strip() == expected.strip(),
            'f1_score': calculate_f1(predicted, expected)
        }
```

#### 2. Multiple Choice
```python
class MultipleChoiceRunner(BaseTaskRunner):
    """Handles multiple choice tasks"""
    
    def format_prompt(self, sample: Dict) -> str:
        choices = "\n".join([f"{i}. {c}" for i, c in enumerate(sample['choices'])])
        return f"Question: {sample['question']}\n{choices}\nAnswer:"
```

#### 3. Code Generation
```python
class CodeGenerationRunner(BaseTaskRunner):
    """Handles code generation tasks"""
    
    async def evaluate_response(self, predicted: str, test_cases: List) -> Dict:
        results = await run_code_tests(predicted, test_cases)
        return {
            'pass_rate': sum(r['passed'] for r in results) / len(results),
            'execution_time': avg([r['time'] for r in results])
        }
```

#### 4. Text Generation
```python
class TextGenerationRunner(BaseTaskRunner):
    """Handles text generation tasks"""
    
    def evaluate_response(self, predicted: str, reference: str) -> Dict:
        return {
            'bleu': calculate_bleu(predicted, reference),
            'rouge': calculate_rouge(predicted, reference),
            'length_ratio': len(predicted) / len(reference)
        }
```

### Task Templates

```python
TASK_TEMPLATES = {
    'mmlu': {
        'name': 'MMLU Benchmark',
        'description': 'Massive Multitask Language Understanding',
        'task_type': 'multiple_choice',
        'dataset_name': 'hendrycks/mmlu',
        'metric': 'accuracy',
        'num_fewshot': 5
    },
    'humaneval': {
        'name': 'HumanEval',
        'description': 'Code generation benchmark',
        'task_type': 'code_generation',
        'dataset_name': 'openai/humaneval',
        'metric': 'pass_rate',
        'generation_kwargs': {
            'max_length': 500,
            'temperature': 0.2
        }
    },
    'gsm8k': {
        'name': 'GSM8K',
        'description': 'Grade school math problems',
        'task_type': 'question_answer',
        'dataset_name': 'gsm8k',
        'metric': 'exact_match'
    }
}
```

---

## Evaluation Runner

### Core Evaluation Loop

```python
class EvalRunner:
    async def run_evaluation(self, max_samples: int = None,
                            progress_callback: Callable = None) -> List[EvalSampleResult]:
        """Execute evaluation"""
        
        # Load dataset
        dataset = await self.load_dataset()
        
        # Prepare samples
        samples = self.prepare_samples(dataset, max_samples)
        
        # Initialize runners
        runner = self.get_task_runner(self.task_config.task_type)
        
        # Process samples
        results = []
        async with self.create_session() as session:
            for i, sample in enumerate(samples):
                # Format prompt
                prompt = runner.format_prompt(sample)
                
                # Get model response
                response = await self.get_model_response(prompt, session)
                
                # Evaluate response
                metrics = runner.evaluate_response(
                    response, 
                    sample.get('answer', sample.get('target'))
                )
                
                # Create result
                result = EvalSampleResult(
                    sample_id=str(i),
                    input_text=prompt,
                    expected_output=sample.get('answer'),
                    actual_output=response,
                    metrics=metrics
                )
                
                results.append(result)
                
                # Report progress
                if progress_callback:
                    progress_callback(i + 1, len(samples), result)
        
        return results
```

### Parallel Processing

```python
class ParallelEvalRunner(EvalRunner):
    async def run_evaluation(self, max_samples: int = None,
                            max_concurrent: int = 5) -> List[EvalSampleResult]:
        """Run evaluation with parallel processing"""
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_sample(sample, index):
            async with semaphore:
                return await self.evaluate_single_sample(sample, index)
        
        tasks = [
            process_sample(sample, i) 
            for i, sample in enumerate(samples)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                # Log error and create error result
                logger.error(f"Sample evaluation failed: {result}")
                processed_results.append(self.create_error_result(result))
            else:
                processed_results.append(result)
        
        return processed_results
```

---

## Results Analysis

### Metrics Calculation

```python
class MetricsCalculator:
    """Calculate evaluation metrics"""
    
    def calculate_accuracy(self, results: List[EvalSampleResult]) -> float:
        """Calculate accuracy metric"""
        correct = sum(1 for r in results if r.metrics.get('correct', False))
        return correct / len(results) if results else 0.0
    
    def calculate_f1_score(self, results: List[EvalSampleResult]) -> float:
        """Calculate F1 score"""
        # Implementation for F1 calculation
        
    def calculate_bleu(self, results: List[EvalSampleResult]) -> float:
        """Calculate BLEU score for generation tasks"""
        # Implementation for BLEU calculation
        
    def calculate_pass_rate(self, results: List[EvalSampleResult]) -> float:
        """Calculate pass rate for code tasks"""
        passed = sum(1 for r in results if r.metrics.get('passed', False))
        return passed / len(results) if results else 0.0
```

### Results Aggregation

```python
def calculate_aggregate_metrics(results: List[EvalSampleResult]) -> Dict[str, float]:
    """Aggregate results into summary metrics"""
    
    metrics = {}
    
    # Basic stats
    metrics['total_samples'] = len(results)
    metrics['successful_samples'] = sum(1 for r in results if not r.error_info)
    
    # Task-specific metrics
    if all('accuracy' in r.metrics for r in results):
        metrics['accuracy'] = np.mean([r.metrics['accuracy'] for r in results])
    
    if all('f1_score' in r.metrics for r in results):
        metrics['f1_score'] = np.mean([r.metrics['f1_score'] for r in results])
    
    if all('bleu' in r.metrics for r in results):
        metrics['bleu'] = np.mean([r.metrics['bleu'] for r in results])
    
    # Performance metrics
    if all('latency' in r.metadata for r in results):
        latencies = [r.metadata['latency'] for r in results]
        metrics['avg_latency'] = np.mean(latencies)
        metrics['p95_latency'] = np.percentile(latencies, 95)
    
    return metrics
```

### Comparison Tools

```python
class EvaluationComparator:
    """Compare evaluation runs"""
    
    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple evaluation runs"""
        
        runs = [self.db.get_run(run_id) for run_id in run_ids]
        metrics = [self.db.get_run_metrics(run_id) for run_id in run_ids]
        
        comparison = {
            'runs': runs,
            'metrics_comparison': self.compare_metrics(metrics),
            'best_performer': self.identify_best_performer(runs, metrics),
            'statistical_significance': self.calculate_significance(metrics)
        }
        
        return comparison
    
    def generate_comparison_chart(self, comparison: Dict) -> str:
        """Generate comparison visualization"""
        # Implementation for chart generation
```

---

## Backend Integration

### Database Schema

```sql
-- Tasks table
CREATE TABLE eval_tasks (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    task_type TEXT NOT NULL,
    config_format TEXT,
    config_data JSON,
    dataset_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Models table  
CREATE TABLE eval_models (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    provider TEXT NOT NULL,
    model_id TEXT NOT NULL,
    config JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Runs table
CREATE TABLE eval_runs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    task_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    status TEXT DEFAULT 'pending',
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    total_samples INTEGER,
    completed_samples INTEGER,
    config_overrides JSON,
    error_message TEXT,
    FOREIGN KEY (task_id) REFERENCES eval_tasks(id),
    FOREIGN KEY (model_id) REFERENCES eval_models(id)
);

-- Results table
CREATE TABLE eval_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    input_data JSON,
    expected_output TEXT,
    actual_output TEXT,
    metrics JSON,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES eval_runs(id)
);

-- Metrics table
CREATE TABLE eval_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL,
    metric_type TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES eval_runs(id)
);
```

### API Integration

```python
class LLMProviderInterface:
    """Unified interface for LLM providers"""
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from LLM"""
        raise NotImplementedError
    
    async def get_logprobs(self, prompt: str, completion: str) -> List[float]:
        """Get log probabilities for completion"""
        raise NotImplementedError

class OpenAIProvider(LLMProviderInterface):
    """OpenAI API integration"""
    
    async def generate(self, prompt: str, **kwargs) -> str:
        response = await self.client.completions.create(
            model=self.model_id,
            prompt=prompt,
            **kwargs
        )
        return response.choices[0].text

class AnthropicProvider(LLMProviderInterface):
    """Anthropic API integration"""
    
    async def generate(self, prompt: str, **kwargs) -> str:
        response = await self.client.messages.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.content[0].text
```

---

## Testing Strategies

### Unit Tests

```python
# test_task_loader.py
import pytest
from tldw_chatbook.Evals.task_loader import TaskLoader, TaskConfig

class TestTaskLoader:
    def test_load_eleuther_format(self, tmp_path):
        """Test loading Eleuther format tasks"""
        config = {
            'task': 'test_task',
            'dataset_name': 'test_dataset',
            'output_type': 'multiple_choice'
        }
        
        loader = TaskLoader()
        task = loader.load_task(config, format_type='eleuther')
        
        assert task.name == 'test_task'
        assert task.task_type == 'classification'
    
    def test_template_creation(self):
        """Test creating task from template"""
        loader = TaskLoader()
        task = loader.create_task_from_template('mmlu')
        
        assert task.task_type == 'multiple_choice'
        assert task.metric == 'accuracy'
    
    def test_validation(self):
        """Test task validation"""
        task = TaskConfig(
            name='',  # Invalid: empty name
            task_type='invalid_type',  # Invalid type
            dataset_name='test'
        )
        
        loader = TaskLoader()
        issues = loader.validate_task(task)
        
        assert len(issues) >= 2
        assert any('name' in issue for issue in issues)
```

### Integration Tests

```python
# test_eval_orchestrator.py
import pytest
from tldw_chatbook.Evals.eval_orchestrator import EvaluationOrchestrator

@pytest.mark.asyncio
class TestEvaluationOrchestrator:
    async def test_full_evaluation_flow(self, tmp_path):
        """Test complete evaluation workflow"""
        orchestrator = EvaluationOrchestrator(db_path=tmp_path / 'test.db')
        
        # Create task
        task_id = await orchestrator.create_task_from_file(
            'fixtures/test_task.yaml'
        )
        
        # Create model
        model_id = orchestrator.create_model_config(
            name='test_model',
            provider='mock',
            model_id='mock-1'
        )
        
        # Run evaluation
        run_id = await orchestrator.run_evaluation(
            task_id=task_id,
            model_id=model_id,
            max_samples=10
        )
        
        # Check results
        summary = orchestrator.get_run_summary(run_id)
        assert summary['status'] == 'completed'
        assert summary['sample_count'] == 10
    
    async def test_parallel_evaluations(self):
        """Test running multiple evaluations in parallel"""
        # Implementation for parallel testing
    
    async def test_error_recovery(self):
        """Test error handling and recovery"""
        # Implementation for error testing
```

### UI Tests

```python
# test_evals_window.py
import pytest
from textual.testing import AppTest
from tldw_chatbook.UI.Evals_Window_v3_unified import EvalsWindow

@pytest.mark.asyncio
class TestEvalsWindow:
    async def test_window_initialization(self):
        """Test window loads correctly"""
        async with AppTest.run_test() as pilot:
            app = pilot.app
            window = EvalsWindow(app)
            await pilot.mount(window)
            
            # Check key sections exist
            assert window.query_one("#task-creation-section")
            assert window.query_one("#quick-setup-section")
            assert window.query_one("#active-eval-section")
    
    async def test_task_creation_flow(self):
        """Test creating a new task"""
        async with AppTest.run_test() as pilot:
            # Mount window
            # Fill task form
            # Click create button
            # Verify task saved
    
    async def test_evaluation_lifecycle(self):
        """Test complete evaluation lifecycle"""
        # Start evaluation
        # Monitor progress
        # Check results
        # Export data
```

### Performance Tests

```python
# test_eval_performance.py
import pytest
import time
from tldw_chatbook.Evals.eval_runner import EvalRunner

class TestEvaluationPerformance:
    @pytest.mark.benchmark
    async def test_large_dataset_processing(self, benchmark):
        """Test performance with large datasets"""
        runner = EvalRunner(mock_task_config(), mock_model_config())
        
        async def run_eval():
            return await runner.run_evaluation(max_samples=1000)
        
        results = benchmark(run_eval)
        assert len(results) == 1000
    
    async def test_parallel_execution_speedup(self):
        """Test parallel execution improves performance"""
        # Serial execution
        start = time.time()
        serial_results = await run_serial_evaluation(100)
        serial_time = time.time() - start
        
        # Parallel execution
        start = time.time()
        parallel_results = await run_parallel_evaluation(100, max_concurrent=5)
        parallel_time = time.time() - start
        
        # Should be faster
        assert parallel_time < serial_time * 0.5
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1)

**Goal**: Set up core infrastructure

- [ ] Database schema implementation
- [ ] Basic TaskLoader with JSON support
- [ ] Simple EvalRunner for Q&A tasks
- [ ] Basic UI skeleton with sections

**Deliverables**:
- Working database layer
- Task loading from JSON
- Simple Q&A evaluation
- UI with basic layout

### Phase 2: Core Features (Week 2)

**Goal**: Implement essential evaluation capabilities

- [ ] Multiple task type support
- [ ] Model configuration management
- [ ] Evaluation orchestrator
- [ ] Progress tracking
- [ ] Basic metrics calculation

**Deliverables**:
- Support for 3+ task types
- Model management UI
- Working evaluation flow
- Real-time progress updates

### Phase 3: Advanced Features (Week 3)

**Goal**: Add advanced capabilities

- [ ] Parallel evaluation execution
- [ ] Template system
- [ ] Comparison tools
- [ ] Export functionality
- [ ] Cost estimation

**Deliverables**:
- 5x performance improvement
- 10+ built-in templates
- Model comparison features
- Multiple export formats

### Phase 4: Polish & Testing (Week 4)

**Goal**: Production readiness

- [ ] Comprehensive error handling
- [ ] Performance optimization
- [ ] Full test coverage
- [ ] Documentation
- [ ] UI polish

**Deliverables**:
- 90% test coverage
- Complete documentation
- Production-ready system
- Performance benchmarks

---

## Advanced Features

### 1. Adaptive Sampling

```python
class AdaptiveSampler:
    """Intelligently sample from datasets"""
    
    def select_samples(self, dataset, target_count: int) -> List:
        """Select representative samples"""
        # Stratified sampling
        # Difficulty-based selection
        # Error-prone case prioritization
```

### 2. A/B Testing Framework

```python
class ABTestRunner:
    """Run A/B tests between models"""
    
    async def run_ab_test(self, model_a: str, model_b: str,
                         task_id: str, confidence_level: float = 0.95):
        """Run statistical A/B test"""
        # Parallel evaluation
        # Statistical significance testing
        # Early stopping on clear winner
```

### 3. Custom Metrics

```python
class CustomMetricEvaluator:
    """Support custom evaluation metrics"""
    
    def register_metric(self, name: str, 
                       evaluator: Callable[[str, str], float]):
        """Register custom metric function"""
        
    def evaluate(self, predicted: str, expected: str) -> Dict[str, float]:
        """Evaluate with all registered metrics"""
```

### 4. Continuous Evaluation

```python
class ContinuousEvaluator:
    """Continuous model evaluation"""
    
    async def monitor_model(self, model_id: str, 
                           test_suite: List[str],
                           interval: int = 3600):
        """Continuously evaluate model performance"""
        # Scheduled evaluations
        # Drift detection
        # Alert on degradation
```

### 5. Evaluation Caching

```python
class EvaluationCache:
    """Cache evaluation results"""
    
    def get_cached_result(self, prompt_hash: str, 
                         model_version: str) -> Optional[EvalSampleResult]:
        """Retrieve cached result if available"""
        
    def cache_result(self, prompt_hash: str, 
                    model_version: str, 
                    result: EvalSampleResult):
        """Cache evaluation result"""
```

---

## Best Practices

### 1. Task Design
- Use clear, unambiguous prompts
- Include diverse test cases
- Balance difficulty levels
- Version control task configurations

### 2. Evaluation Methodology
- Use sufficient sample sizes
- Control for randomness with seeds
- Run multiple iterations
- Document evaluation parameters

### 3. Performance Optimization
- Batch API requests
- Use connection pooling
- Cache repeated computations
- Profile bottlenecks

### 4. Error Handling
- Implement retry logic
- Log all errors with context
- Graceful degradation
- User-friendly error messages

### 5. Security
- Validate all inputs
- Sanitize code execution
- Secure API key storage
- Rate limiting

---

## Troubleshooting Guide

### Common Issues

#### 1. Slow Evaluations
**Problem**: Evaluations taking too long
**Solutions**:
- Increase parallel workers
- Use smaller sample sizes
- Enable result caching
- Check API rate limits

#### 2. Memory Issues
**Problem**: Running out of memory with large datasets
**Solutions**:
- Use streaming/chunking
- Reduce batch sizes
- Clear caches periodically
- Use disk-based storage

#### 3. Inconsistent Results
**Problem**: Results vary between runs
**Solutions**:
- Set random seeds
- Use temperature=0 for deterministic output
- Increase sample size
- Check for API changes

#### 4. API Errors
**Problem**: Frequent API failures
**Solutions**:
- Implement exponential backoff
- Check rate limits
- Validate API keys
- Use fallback providers

---

## Example Implementation

### Minimal Working Example

```python
# minimal_eval.py
import asyncio
from tldw_chatbook.Evals import EvaluationOrchestrator

async def run_simple_evaluation():
    """Run a simple evaluation"""
    
    # Initialize orchestrator
    orchestrator = EvaluationOrchestrator()
    
    # Create a simple task
    task_config = {
        'name': 'Simple Math',
        'task_type': 'question_answer',
        'dataset_name': 'custom',
        'questions': [
            {'question': '2 + 2 = ?', 'answer': '4'},
            {'question': '5 * 3 = ?', 'answer': '15'}
        ]
    }
    
    task_id = await orchestrator.create_task_from_dict(task_config)
    
    # Configure model
    model_id = orchestrator.create_model_config(
        name='GPT-3.5',
        provider='openai',
        model_id='gpt-3.5-turbo'
    )
    
    # Run evaluation
    run_id = await orchestrator.run_evaluation(
        task_id=task_id,
        model_id=model_id,
        run_name='Simple Math Test'
    )
    
    # Get results
    summary = orchestrator.get_run_summary(run_id)
    print(f"Accuracy: {summary['metrics']['accuracy']:.2%}")

if __name__ == '__main__':
    asyncio.run(run_simple_evaluation())
```

---

## Conclusion

The Evals Window provides a comprehensive system for evaluating LLM performance. By following this guide, you can implement a robust evaluation framework that supports multiple task types, provides detailed metrics, and enables model comparison.

Key success factors:
1. Start with core functionality and iterate
2. Focus on reliability and error handling
3. Design for extensibility
4. Provide clear user feedback
5. Test thoroughly at all levels

The modular architecture ensures that new features can be added without disrupting existing functionality, while the comprehensive testing strategy ensures reliability in production environments.