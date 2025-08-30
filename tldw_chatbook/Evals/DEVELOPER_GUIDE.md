# Evals Module Developer Guide

## Architecture Overview

The Evals module follows a modular, extensible architecture designed for evaluating Large Language Models (LLMs). This guide provides detailed information for developers working with or extending the module.

## Core Components

### 1. EvaluationOrchestrator (`eval_orchestrator.py`)

The main entry point that coordinates all evaluation activities.

```python
class EvaluationOrchestrator:
    def __init__(self, db_path: str = None):
        self.db = EvalsDB(db_path)
        self.task_loader = TaskLoader(self.db)
        self.concurrent_manager = ConcurrentRunManager()
        self.validator = ConfigurationValidator()
        self.error_handler = get_error_handler()
        self._active_tasks = {}  # Critical: Tracks running evaluations
        self._client_id = "eval_orchestrator"
```

**Key Responsibilities:**
- Task creation and management
- Evaluation run orchestration
- Concurrent run management
- Error handling coordination
- Result aggregation

**Critical Bug Fix (v2.0.0):**
The `_active_tasks` dictionary must be initialized in `__init__` to prevent `AttributeError` when calling `cancel_evaluation()`.

### 2. Error Handling System (`eval_errors.py`)

Unified error handling with retry logic and budget monitoring.

```python
class ErrorHandler:
    """Singleton error handler with retry logic."""
    
    async def retry_with_backoff(self, func: Callable, 
                                max_retries: int = 3,
                                base_delay: float = 1.0):
        """Execute function with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                return await func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2 ** attempt)
                await asyncio.sleep(delay)

class BudgetMonitor:
    """Monitor and enforce budget limits."""
    
    def __init__(self, budget_limit: float = 10.0):
        self.budget_limit = budget_limit
        self.current_cost = 0.0
        self.warning_threshold = 0.8
```

**Error Categories:**
- `DATASET_LOADING` - Dataset file issues
- `MODEL_CONFIGURATION` - Model config problems
- `API_ERROR` - API call failures
- `RATE_LIMIT` - Rate limiting
- `BUDGET_EXCEEDED` - Cost limits
- `EXECUTION` - Runtime errors
- `VALIDATION` - Input validation
- `FILE_SYSTEM` - File I/O errors

### 3. Runner System

#### Base Runner (`base_runner.py`)

Abstract base class for all evaluation runners.

```python
@dataclass
class EvalSample:
    """Single evaluation sample."""
    id: str
    input_text: str
    expected_output: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvalSampleResult:
    """Result of evaluating a single sample."""
    sample_id: str
    input_text: str
    expected_output: Optional[str]
    actual_output: str
    metrics: Dict[str, float]
    latency_ms: float
    error: Optional[str] = None

class BaseEvalRunner(ABC):
    """Abstract base class for evaluation runners."""
    
    @abstractmethod
    async def evaluate_sample(self, sample: EvalSample) -> EvalSampleResult:
        """Evaluate a single sample."""
        pass
```

#### Standard Runner (`eval_runner.py`)

Default implementation for most evaluation tasks.

```python
class StandardEvalRunner(BaseEvalRunner):
    """Standard evaluation runner."""
    
    async def evaluate_sample(self, sample: EvalSample) -> EvalSampleResult:
        # Call LLM
        response = await self._call_llm(sample.input_text)
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(
            expected=sample.expected_output,
            actual=response,
            metric_names=self.task_config.get('metrics', ['exact_match'])
        )
        
        return EvalSampleResult(...)
```

### 4. Metrics System (`metrics_calculator.py`)

Comprehensive metrics calculation for various task types.

```python
class MetricsCalculator:
    """Calculate evaluation metrics."""
    
    def calculate_exact_match(self, expected: str, actual: str) -> float:
        """Exact string match."""
        return 1.0 if expected.strip() == actual.strip() else 0.0
    
    def calculate_f1_score(self, expected: str, actual: str) -> float:
        """Token-level F1 score."""
        expected_tokens = set(expected.lower().split())
        actual_tokens = set(actual.lower().split())
        
        if not expected_tokens and not actual_tokens:
            return 1.0
        if not expected_tokens or not actual_tokens:
            return 0.0
            
        precision = len(expected_tokens & actual_tokens) / len(actual_tokens)
        recall = len(expected_tokens & actual_tokens) / len(expected_tokens)
        
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
```

**Available Metrics:**
- Text: `exact_match`, `f1`, `rouge_*`, `bleu`, `semantic_similarity`
- Classification: `accuracy`, `precision`, `recall`, `confusion_matrix`
- Code: `pass_rate`, `syntax_valid`, `execution_success`
- Safety: `toxicity_level`, `bias_score`, `safety_score`

### 5. Dataset Loading (`dataset_loader.py`)

Handles multiple dataset formats with validation.

```python
class DatasetLoader:
    """Load and validate datasets."""
    
    @staticmethod
    def load_dataset_samples(task_config: TaskConfig) -> List[EvalSample]:
        """Load dataset based on task configuration."""
        dataset_path = task_config.dataset_name
        
        if dataset_path.endswith('.json'):
            return DatasetLoader._load_json(dataset_path)
        elif dataset_path.endswith('.csv'):
            return DatasetLoader._load_csv(dataset_path)
        elif dataset_path.endswith('.jsonl'):
            return DatasetLoader._load_jsonl(dataset_path)
        else:
            # Try HuggingFace datasets
            return DatasetLoader._load_huggingface(dataset_path)
```

**Supported Formats:**
- JSON: Array of objects with `id`, `input`, `output`
- CSV: Headers must include `id`, `input`, `output`
- JSONL: One JSON object per line
- HuggingFace: Direct dataset names

### 6. Export System (`exporters.py`)

Unified export system with polymorphic dispatch.

```python
class EvaluationExporter:
    """Export evaluation results in various formats."""
    
    def export(self, result: Any, output_path: Union[str, Path], 
              format: str = 'csv'):
        """Export results with automatic format detection."""
        
        # Determine result type
        if self._is_ab_test_result(result):
            return self._export_ab_test(result, output_path, format)
        elif self._is_standard_run(result):
            return self._export_standard_run(result, output_path, format)
        else:
            raise ValueError(f"Unknown result type: {type(result)}")
```

**Export Formats:**
- CSV: Tabular data with headers
- JSON: Complete structured data
- Markdown: Human-readable reports
- LaTeX: Academic paper format
- HTML: Web-viewable reports

### 7. Template System (`eval_templates/`)

Organized template package structure.

```python
# eval_templates/__init__.py
class TemplateManager:
    """Manage evaluation templates."""
    
    def __init__(self):
        self.templates = {}
        self._load_all_templates()
    
    def get_template(self, name: str) -> Dict[str, Any]:
        """Get template by name."""
        return self.templates.get(name)
    
    def get_templates_by_category(self, category: str) -> List[Dict]:
        """Get all templates in a category."""
        return [t for t in self.templates.values() 
                if t.get('category') == category]
```

**Template Categories:**
- `reasoning.py` - Mathematical and logical reasoning
- `language.py` - Translation, grammar, paraphrasing
- `coding.py` - Code generation and review
- `safety.py` - Safety and bias evaluation
- `creative.py` - Creative writing tasks
- `multimodal.py` - Image and visual tasks

### 8. Configuration Management (`config_loader.py`)

External YAML configuration with runtime updates.

```python
class EvalConfigLoader:
    """Load and manage evaluation configuration."""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or self._get_default_path()
        self.config = self._load_config()
        self._last_modified = os.path.getmtime(self.config_path)
    
    def reload(self):
        """Reload configuration if file changed."""
        current_mtime = os.path.getmtime(self.config_path)
        if current_mtime > self._last_modified:
            self.config = self._load_config()
            self._last_modified = current_mtime
```

## Database Schema

### Tables Structure

```sql
-- Schema version tracking
CREATE TABLE schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Evaluation tasks
CREATE TABLE tasks (
    task_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    task_type TEXT NOT NULL,
    dataset_path TEXT,
    dataset_hash TEXT,
    metric TEXT,
    metadata TEXT,  -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP
);

-- Evaluation runs
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    model_config TEXT NOT NULL,  -- JSON
    run_config TEXT,  -- JSON
    status TEXT DEFAULT 'pending',
    progress REAL DEFAULT 0.0,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
);

-- Individual results
CREATE TABLE results (
    result_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    input_text TEXT,
    expected_output TEXT,
    actual_output TEXT,
    metrics TEXT,  -- JSON
    latency_ms REAL,
    error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

-- Aggregated metrics
CREATE TABLE run_metrics (
    run_id TEXT PRIMARY KEY,
    metrics TEXT NOT NULL,  -- JSON
    summary_stats TEXT,  -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

-- Indexes for performance
CREATE INDEX idx_runs_task_id ON runs(task_id);
CREATE INDEX idx_runs_status ON runs(status);
CREATE INDEX idx_results_run_id ON results(run_id);
CREATE INDEX idx_results_sample_id ON results(sample_id);
```

## Extension Points

### Creating Custom Task Types

1. Define the task configuration:

```python
# In configuration_validator.py
VALID_TASK_TYPES.append('custom_task')
VALID_METRICS['custom_task'] = ['custom_metric1', 'custom_metric2']
```

2. Create a specialized runner:

```python
# In specialized_runners/custom_runner.py
class CustomTaskRunner(BaseEvalRunner):
    """Runner for custom task type."""
    
    async def evaluate_sample(self, sample: EvalSample) -> EvalSampleResult:
        # Custom evaluation logic
        response = await self.custom_evaluation(sample)
        
        # Custom metrics
        metrics = self.calculate_custom_metrics(
            sample.expected_output, 
            response
        )
        
        return EvalSampleResult(...)
```

3. Register in runner factory:

```python
# In eval_runner.py
def create_runner(task_type: str, **kwargs) -> BaseEvalRunner:
    if task_type == 'custom_task':
        return CustomTaskRunner(**kwargs)
    # ... other runners
```

### Adding Custom Metrics

1. Extend the metrics calculator:

```python
# In metrics_calculator.py or custom module
class ExtendedMetricsCalculator(MetricsCalculator):
    
    def calculate_semantic_similarity(self, expected: str, 
                                     actual: str) -> float:
        """Calculate semantic similarity using embeddings."""
        # Get embeddings
        expected_embedding = self.get_embedding(expected)
        actual_embedding = self.get_embedding(actual)
        
        # Calculate cosine similarity
        return self.cosine_similarity(expected_embedding, actual_embedding)
```

2. Register metric in configuration:

```yaml
# In config/eval_config.yaml
metrics:
  custom_task:
    - semantic_similarity
    - perplexity
```

### Creating Custom Exporters

1. Extend the exporter class:

```python
# In custom_exporter.py
class CustomExporter(EvaluationExporter):
    
    def export_to_dashboard(self, result: Any, 
                           dashboard_url: str):
        """Export to custom dashboard."""
        formatted_data = self.format_for_dashboard(result)
        response = requests.post(
            dashboard_url,
            json=formatted_data
        )
        return response.status_code == 200
```

### Implementing Caching

1. Create cache manager:

```python
# In cache_manager.py
class EvaluationCache:
    """Cache evaluation results."""
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or self._get_default_cache_dir()
        self.cache = {}
        self._load_cache()
    
    def get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result if available."""
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if not self._is_expired(entry):
                return entry['result']
        return None
    
    def cache_result(self, cache_key: str, result: Any):
        """Cache evaluation result."""
        self.cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
        self._save_cache()
```

## Testing Guidelines

### Unit Testing

Test individual components in isolation:

```python
# test_metrics_calculator.py
class TestMetricsCalculator:
    def test_exact_match(self):
        calculator = MetricsCalculator()
        assert calculator.calculate_exact_match("hello", "hello") == 1.0
        assert calculator.calculate_exact_match("hello", "world") == 0.0
    
    def test_f1_score(self):
        calculator = MetricsCalculator()
        score = calculator.calculate_f1_score(
            "the quick brown fox",
            "the brown fox"
        )
        assert 0.5 < score < 1.0  # Partial match
```

### Integration Testing

Test component interactions:

```python
# test_integration.py
@pytest.mark.asyncio
async def test_full_evaluation_pipeline():
    orchestrator = EvaluationOrchestrator(":memory:")
    
    # Create task
    task_id = await orchestrator.create_task_from_file(
        "test_dataset.json",
        "Test Task"
    )
    
    # Run evaluation
    with patch('llm_api.call') as mock_call:
        mock_call.return_value = "mocked response"
        
        run_id = await orchestrator.run_evaluation(
            task_id=task_id,
            model_configs=[test_model_config],
            max_samples=10
        )
        
        # Verify results
        status = orchestrator.get_run_status(run_id)
        assert status['status'] == 'completed'
```

### Performance Testing

```python
# test_performance.py
@pytest.mark.benchmark
async def test_large_dataset_performance():
    """Test performance with large datasets."""
    samples = [create_sample(i) for i in range(10000)]
    
    start_time = time.time()
    results = await process_samples(samples)
    duration = time.time() - start_time
    
    assert duration < 60  # Should complete in under 1 minute
    assert len(results) == len(samples)
```

## Performance Optimization

### 1. Batch Processing

```python
async def evaluate_batch(self, samples: List[EvalSample], 
                        batch_size: int = 10):
    """Process samples in batches."""
    results = []
    
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        batch_tasks = [
            self.evaluate_sample(sample) 
            for sample in batch
        ]
        batch_results = await asyncio.gather(*batch_tasks)
        results.extend(batch_results)
    
    return results
```

### 2. Connection Pooling

```python
class LLMConnectionPool:
    """Manage LLM API connections."""
    
    def __init__(self, max_connections: int = 10):
        self.semaphore = asyncio.Semaphore(max_connections)
        self.session = aiohttp.ClientSession()
    
    async def call_with_limit(self, *args, **kwargs):
        async with self.semaphore:
            return await self.call_llm(*args, **kwargs)
```

### 3. Result Streaming

```python
async def stream_results(self, run_id: str):
    """Stream results as they complete."""
    async for result in self.evaluate_streaming(run_id):
        yield result
        # Save intermediate result
        self.db.save_result(result)
```

## Security Considerations

### 1. Input Validation

```python
def validate_task_config(config: Dict[str, Any]):
    """Validate task configuration."""
    # Check required fields
    required = ['name', 'task_type', 'dataset_name']
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate task type
    if config['task_type'] not in VALID_TASK_TYPES:
        raise ValueError(f"Invalid task type: {config['task_type']}")
    
    # Sanitize paths
    dataset_path = config['dataset_name']
    if not is_safe_path(dataset_path):
        raise ValueError("Invalid dataset path")
```

### 2. API Key Management

```python
def get_api_key(provider: str) -> str:
    """Securely retrieve API key."""
    # Try environment variable first
    env_key = f"{provider.upper()}_API_KEY"
    if env_key in os.environ:
        return os.environ[env_key]
    
    # Try secure keyring
    try:
        import keyring
        return keyring.get_password("evals", provider)
    except ImportError:
        pass
    
    # Fall back to config file (least secure)
    return config.get('api_keys', {}).get(provider)
```

### 3. Rate Limiting

```python
class RateLimiter:
    """Enforce rate limits."""
    
    def __init__(self, max_requests: int = 100, 
                 window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()
    
    async def acquire(self):
        """Wait if rate limit exceeded."""
        now = time.time()
        
        # Remove old requests
        while self.requests and self.requests[0] < now - self.window_seconds:
            self.requests.popleft()
        
        # Check limit
        if len(self.requests) >= self.max_requests:
            sleep_time = self.window_seconds - (now - self.requests[0])
            await asyncio.sleep(sleep_time)
            return await self.acquire()
        
        self.requests.append(now)
```

## Debugging

### Enable Debug Logging

```python
import logging
from loguru import logger

# Set debug level
logger.add("debug.log", level="DEBUG")

# Add to specific module
logger.debug(f"Evaluating sample: {sample.id}")
logger.debug(f"LLM response: {response}")
logger.debug(f"Calculated metrics: {metrics}")
```

### Trace Execution

```python
@trace_execution
async def evaluate_sample(self, sample: EvalSample):
    """Traced evaluation."""
    # Automatic logging of entry/exit and timing
    pass

def trace_execution(func):
    """Decorator for execution tracing."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        logger.debug(f"Entering {func.__name__}")
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            logger.debug(f"Exiting {func.__name__} (took {time.time()-start:.2f}s)")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise
    return wrapper
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
def process_large_dataset(dataset_path: str):
    """Memory-profiled function."""
    samples = load_dataset(dataset_path)
    results = evaluate_all(samples)
    return results
```

## Migration Guide

### From v1.0 to v2.0

1. **Update imports:**
```python
# Old
from tldw_chatbook.Evals.eval_runner import EvalRunner

# New
from tldw_chatbook.Evals.eval_orchestrator import EvaluationOrchestrator
```

2. **Fix _active_tasks usage:**
```python
# Old (would crash)
orchestrator.cancel_evaluation(run_id)

# New (works)
orchestrator.cancel_evaluation(run_id)  # _active_tasks initialized
```

3. **Use unified error handling:**
```python
# Old
try:
    # evaluation code
except Exception as e:
    print(f"Error: {e}")

# New
from tldw_chatbook.Evals.eval_errors import get_error_handler

error_handler = get_error_handler()
try:
    # evaluation code
except Exception as e:
    context = error_handler.handle_error(e)
    print(context.get_user_message())
```

4. **Update configuration:**
```yaml
# Move from hardcoded to YAML config
# Old: Hardcoded in Python files
# New: config/eval_config.yaml
```

## Best Practices

1. **Always use type hints:**
```python
async def evaluate(task_id: str, 
                  model_configs: List[Dict[str, Any]],
                  max_samples: Optional[int] = None) -> str:
```

2. **Handle errors gracefully:**
```python
try:
    result = await evaluate_sample(sample)
except EvaluationError as e:
    logger.error(f"Evaluation failed: {e.get_user_message()}")
    result = create_error_result(sample, e)
```

3. **Use async/await properly:**
```python
# Good
results = await asyncio.gather(*tasks)

# Bad
results = [await task for task in tasks]  # Sequential
```

4. **Document complex logic:**
```python
def calculate_metric(expected: str, actual: str) -> float:
    """
    Calculate custom metric.
    
    This metric considers:
    1. Exact match (weight: 0.5)
    2. Fuzzy match (weight: 0.3)
    3. Semantic similarity (weight: 0.2)
    
    Returns:
        Float between 0 and 1
    """
```

5. **Test edge cases:**
```python
@pytest.mark.parametrize("input,expected", [
    ("", 0.0),  # Empty string
    (None, 0.0),  # None value
    ("test" * 1000, 1.0),  # Long string
    ("❤️", 1.0),  # Unicode
])
def test_edge_cases(input, expected):
    assert process(input) == expected
```

## Monitoring and Observability

### Metrics Collection

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
eval_counter = Counter('evaluations_total', 'Total evaluations')
eval_duration = Histogram('evaluation_duration_seconds', 'Evaluation duration')
active_evaluations = Gauge('active_evaluations', 'Currently running evaluations')

# Use in code
@eval_duration.time()
async def run_evaluation(...):
    eval_counter.inc()
    active_evaluations.inc()
    try:
        # evaluation logic
    finally:
        active_evaluations.dec()
```

### Health Checks

```python
async def health_check() -> Dict[str, Any]:
    """System health check."""
    return {
        'status': 'healthy',
        'database': check_database_connection(),
        'active_runs': len(orchestrator._active_tasks),
        'memory_usage': get_memory_usage(),
        'uptime': get_uptime()
    }
```

## Contributing

### Code Style

- Follow PEP 8
- Use type hints
- Write docstrings (Google style)
- Add unit tests for new features
- Update documentation

### Pull Request Process

1. Create feature branch
2. Write tests first (TDD)
3. Implement feature
4. Run test suite
5. Update documentation
6. Submit PR with description

### Review Checklist

- [ ] Tests pass
- [ ] Documentation updated
- [ ] Type hints added
- [ ] Error handling implemented
- [ ] Performance considered
- [ ] Security reviewed
- [ ] Breaking changes documented

## Resources

- [Module README](README.md) - User documentation
- [Test Documentation](../../Tests/Evals/TESTING_SUMMARY.md) - Testing details
- [Refactoring Notes](REFACTORING_COMPLETE.md) - Recent changes
- [Configuration Reference](config/eval_config.yaml) - Configuration options

## Support

For development questions:
1. Check this developer guide
2. Review source code comments
3. Check test implementations
4. Open a development issue