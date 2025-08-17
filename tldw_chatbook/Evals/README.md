# Evals Module Documentation

## Overview

The Evals module provides a comprehensive framework for evaluating Large Language Models (LLMs) across various tasks, metrics, and providers. It supports single model evaluation, A/B testing, custom datasets, and extensive configuration options.

## Table of Contents

- [Quick Start](#quick-start)
- [User Guide](#user-guide)
- [Developer Guide](#developer-guide)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Templates](#templates)
- [Testing](#testing)

## Quick Start

### Basic Evaluation

```python
from tldw_chatbook.Evals.eval_orchestrator import EvaluationOrchestrator

# Initialize orchestrator
orchestrator = EvaluationOrchestrator()

# Create a task from a dataset file
task_id = await orchestrator.create_task_from_file(
    "datasets/qa_test.json",
    "Question Answering Test"
)

# Configure model
model_config = {
    'provider': 'openai',
    'model_id': 'gpt-3.5-turbo',
    'name': 'GPT-3.5',
    'api_key': 'your-api-key'  # Or use environment variable
}

# Run evaluation
run_id = await orchestrator.run_evaluation(
    task_id=task_id,
    model_configs=[model_config],
    max_samples=100
)

# Check status
status = orchestrator.get_run_status(run_id)
print(f"Evaluation status: {status['status']}")

# Export results
from tldw_chatbook.Evals.exporters import EvaluationExporter
exporter = EvaluationExporter()
exporter.export(status, "results.csv", format="csv")
```

### Using Templates

```python
from tldw_chatbook.Evals.eval_templates import get_eval_templates

# Get available templates
templates = get_eval_templates()

# List all templates
all_templates = templates.list_templates()
print(f"Available templates: {', '.join(all_templates)}")

# Get a specific template
gsm8k = templates.get_template('gsm8k')
print(f"Template: {gsm8k['name']}")
print(f"Task type: {gsm8k['task_type']}")
print(f"Metric: {gsm8k['metric']}")

# Get templates by category
reasoning_templates = templates.get_templates_by_category('reasoning')
```

## User Guide

### Dataset Formats

The module supports multiple dataset formats:

#### JSON Format
```json
[
    {
        "id": "1",
        "input": "What is the capital of France?",
        "output": "Paris",
        "metadata": {"category": "geography"}
    }
]
```

#### CSV Format
```csv
id,input,output
1,"What is 2+2?","4"
2,"What is the capital of France?","Paris"
```

#### JSONL Format
```jsonl
{"id": "1", "input": "Question 1", "output": "Answer 1"}
{"id": "2", "input": "Question 2", "output": "Answer 2"}
```

### Task Types

Supported task types:
- `question_answer` - Q&A evaluation with exact match or F1 scoring
- `generation` - Text generation quality assessment
- `classification` - Multi-class or binary classification
- `code_execution` - Code generation and execution testing
- `safety_evaluation` - Safety and bias testing
- `multilingual_evaluation` - Cross-language performance
- `creative_evaluation` - Creative writing assessment
- `math_reasoning` - Mathematical problem solving
- `summarization` - Document summarization quality
- `dialogue` - Conversational ability testing

### Metrics

Available metrics per task type:

| Task Type | Available Metrics |
|-----------|------------------|
| question_answer | exact_match, f1, rouge_1, rouge_2, rouge_l, semantic_similarity |
| generation | bleu, rouge_*, perplexity, coherence, creativity_score |
| classification | accuracy, f1, precision, recall, confusion_matrix |
| code_execution | pass_rate, syntax_valid, execution_success, test_pass_rate |
| safety_evaluation | safety_score, toxicity_level, bias_score |

### A/B Testing

Compare multiple models:

```python
# Configure multiple models
models = [
    {
        'provider': 'openai',
        'model_id': 'gpt-3.5-turbo',
        'name': 'GPT-3.5'
    },
    {
        'provider': 'anthropic',
        'model_id': 'claude-3-haiku',
        'name': 'Claude Haiku'
    }
]

# Run A/B test
run_id = await orchestrator.run_evaluation(
    task_id=task_id,
    model_configs=models,
    max_samples=100,
    run_config={'type': 'ab_test'}
)

# Export comparison report
exporter.export(results, "ab_test_report.md", format="markdown")
```

### Budget Management

Set cost limits:

```python
# Configure budget limits
run_config = {
    'budget_limit': 10.0,  # $10 USD
    'warning_threshold': 0.8,  # Warn at 80%
    'track_by': 'cost'  # or 'tokens'
}

run_id = await orchestrator.run_evaluation(
    task_id=task_id,
    model_configs=[model_config],
    run_config=run_config
)
```

### Error Handling

The module provides automatic retry with exponential backoff:

```python
# Errors are automatically retried
# You can also handle specific errors:
from tldw_chatbook.Evals.eval_errors import EvaluationError, ErrorCategory

try:
    run_id = await orchestrator.run_evaluation(...)
except EvaluationError as e:
    if e.context.category == ErrorCategory.RATE_LIMIT:
        print(f"Rate limited. Retry after: {e.context.retry_after}")
    elif e.context.category == ErrorCategory.BUDGET_EXCEEDED:
        print("Budget limit reached")
    else:
        print(f"Error: {e.get_user_message()}")
```

## Developer Guide

### Architecture

```
┌─────────────────────────────────────────────────┐
│              EvaluationOrchestrator              │
│  (Main entry point, coordinates evaluation)      │
└─────────────────┬───────────────────────────────┘
                  │
    ┌─────────────┴─────────────┬─────────────────┐
    ▼                           ▼                 ▼
┌──────────────┐    ┌──────────────────┐  ┌──────────────┐
│ TaskLoader   │    │ ConcurrentManager │  │ ErrorHandler │
│ (Load tasks) │    │ (Manage runs)     │  │ (Handle errs)│
└──────────────┘    └──────────────────┘  └──────────────┘
    │                           │                 │
    ▼                           ▼                 ▼
┌──────────────┐    ┌──────────────────┐  ┌──────────────┐
│DatasetLoader │    │   EvalRunner     │  │   Metrics    │
│ (Load data)  │    │  (Execute eval)  │  │ (Calculate)  │
└──────────────┘    └──────────────────┘  └──────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │    Exporters     │
                    │ (Export results) │
                    └──────────────────┘
```

### Creating Custom Runners

Extend the base runner class:

```python
from tldw_chatbook.Evals.base_runner import BaseEvalRunner, EvalSample, EvalSampleResult
from typing import Dict, Any

class CustomRunner(BaseEvalRunner):
    """Custom evaluation runner."""
    
    async def evaluate_sample(self, sample: EvalSample) -> EvalSampleResult:
        """Evaluate a single sample."""
        # Your custom evaluation logic
        response = await self.call_model(sample.input_text)
        
        # Calculate metrics
        metrics = self.calculate_metrics(
            expected=sample.expected_output,
            actual=response
        )
        
        return EvalSampleResult(
            sample_id=sample.id,
            input_text=sample.input_text,
            expected_output=sample.expected_output,
            actual_output=response,
            metrics=metrics,
            latency_ms=100.0
        )
    
    def calculate_metrics(self, expected: str, actual: str) -> Dict[str, float]:
        """Calculate custom metrics."""
        return {
            'custom_metric': self.custom_calculation(expected, actual)
        }
```

### Adding New Templates

Create a template in the appropriate category:

```python
# In eval_templates/reasoning.py
def get_custom_template():
    return {
        'name': 'custom_reasoning',
        'task_type': 'question_answer',
        'dataset_name': 'custom_dataset',
        'metric': 'exact_match',
        'description': 'Custom reasoning evaluation',
        'generation_kwargs': {
            'temperature': 0.7,
            'max_tokens': 512
        },
        'prompt_template': """
        Solve this problem step by step:
        {input}
        
        Answer:
        """
    }

# Register in __init__.py
TEMPLATES['custom_reasoning'] = get_custom_template()
```

### Extending Metrics

Add custom metrics to the calculator:

```python
from tldw_chatbook.Evals.metrics_calculator import MetricsCalculator

class CustomMetricsCalculator(MetricsCalculator):
    """Extended metrics calculator."""
    
    def calculate_custom_metric(self, expected: str, actual: str) -> float:
        """Calculate a custom metric."""
        # Your metric logic
        return score
    
    def calculate_all_metrics(self, expected: str, actual: str, 
                            metric_names: List[str]) -> Dict[str, float]:
        """Calculate all requested metrics."""
        metrics = super().calculate_all_metrics(expected, actual, metric_names)
        
        if 'custom_metric' in metric_names:
            metrics['custom_metric'] = self.calculate_custom_metric(expected, actual)
        
        return metrics
```

### Database Schema

The module uses SQLite with the following main tables:

```sql
-- Tasks table
CREATE TABLE tasks (
    task_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    task_type TEXT NOT NULL,
    dataset_path TEXT,
    metric TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Runs table
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    task_id TEXT,
    model_config TEXT,
    status TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
);

-- Results table
CREATE TABLE results (
    result_id TEXT PRIMARY KEY,
    run_id TEXT,
    sample_id TEXT,
    input_text TEXT,
    expected_output TEXT,
    actual_output TEXT,
    metrics TEXT,  -- JSON
    latency_ms REAL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
```

### Event System

The module emits events for monitoring:

```python
from tldw_chatbook.Evals.eval_events import EvalEvent, EvalEventType

# Listen for events
@on(EvalEvent)
def handle_eval_event(event: EvalEvent):
    if event.type == EvalEventType.RUN_STARTED:
        print(f"Evaluation started: {event.run_id}")
    elif event.type == EvalEventType.SAMPLE_COMPLETED:
        print(f"Sample {event.sample_id} completed")
    elif event.type == EvalEventType.RUN_COMPLETED:
        print(f"Evaluation completed: {event.metrics}")
```

## API Reference

### EvaluationOrchestrator

Main orchestrator class for managing evaluations.

#### Methods

##### `__init__(db_path: str = None)`
Initialize the orchestrator.
- `db_path`: Path to SQLite database (optional, defaults to user data directory)

##### `async create_task_from_file(file_path: str, name: str, **kwargs) -> str`
Create an evaluation task from a dataset file.
- `file_path`: Path to dataset file
- `name`: Display name for the task
- Returns: Task ID

##### `async run_evaluation(task_id: str, model_configs: List[Dict], **kwargs) -> str`
Run an evaluation.
- `task_id`: Task to evaluate
- `model_configs`: List of model configurations
- `max_samples`: Maximum samples to evaluate (optional)
- `run_config`: Additional run configuration (optional)
- Returns: Run ID

##### `get_run_status(run_id: str) -> Dict`
Get the status of an evaluation run.
- `run_id`: Run identifier
- Returns: Status dictionary with progress, metrics, etc.

##### `cancel_evaluation(run_id: str) -> bool`
Cancel a running evaluation.
- `run_id`: Run to cancel
- Returns: True if cancelled successfully

##### `list_available_tasks() -> List[Dict]`
List all available evaluation tasks.
- Returns: List of task dictionaries

### EvaluationExporter

Export evaluation results in various formats.

#### Methods

##### `export(result: Any, output_path: Union[str, Path], format: str = 'csv')`
Export evaluation results.
- `result`: Evaluation result object
- `output_path`: Where to save the export
- `format`: Export format ('csv', 'json', 'markdown', 'latex', 'html')

### Error Classes

#### `EvaluationError`
Base exception for evaluation errors.

Properties:
- `context`: ErrorContext object with details
- `original_error`: Original exception if wrapped

Methods:
- `get_user_message()`: Get user-friendly error message

#### `ErrorContext`
Context information for errors.

Properties:
- `category`: ErrorCategory enum
- `severity`: ErrorSeverity enum
- `message`: Error message
- `details`: Additional details
- `suggestion`: Suggested action
- `is_retryable`: Whether to retry
- `retry_after`: Seconds to wait before retry

### Configuration Classes

#### `EvalConfigLoader`
Load and manage configuration.

Methods:
- `get_task_types()`: Get valid task types
- `get_metrics(task_type)`: Get metrics for task type
- `is_feature_enabled(feature)`: Check feature flag
- `reload()`: Reload configuration from file

## Configuration

### Configuration File

The module uses YAML configuration at `config/eval_config.yaml`:

```yaml
# Task types and metrics
task_types:
  - question_answer
  - generation
  - classification

metrics:
  question_answer:
    - exact_match
    - f1
    - rouge_1

# Provider settings
providers:
  openai:
    models:
      - gpt-3.5-turbo
      - gpt-4
    max_tokens: 4096
    supports_streaming: true

# Error handling
error_handling:
  max_retries: 3
  retry_delay_seconds: 1.0
  exponential_backoff: true

# Budget monitoring
budget:
  warning_threshold: 0.8
  default_limit: 10.0

# Performance
performance:
  batch_size: 10
  cache_results: true

# Feature flags
features:
  enable_streaming: true
  enable_caching: true
  enable_parallel_processing: true
```

### Environment Variables

Set API keys and configuration via environment:

```bash
# API Keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Configuration
export EVAL_BUDGET_LIMIT="50.0"
export EVAL_MAX_CONCURRENT="5"
export EVAL_CACHE_DIR="/path/to/cache"
```

## Templates

### Available Template Categories

1. **Reasoning** (`eval_templates/reasoning.py`)
   - gsm8k - Grade school math problems
   - math_word_problems - Complex word problems
   - logical_reasoning - Logic puzzles
   - chain_of_thought - Step-by-step reasoning

2. **Language** (`eval_templates/language.py`)
   - translation - Language translation
   - grammar_correction - Grammar fixing
   - paraphrasing - Text rewriting
   - sentiment_analysis - Emotion detection

3. **Coding** (`eval_templates/coding.py`)
   - humaneval - Python code generation
   - code_review - Code quality assessment
   - bug_detection - Find bugs in code
   - code_explanation - Explain code snippets

4. **Safety** (`eval_templates/safety.py`)
   - toxicity_detection - Harmful content detection
   - bias_evaluation - Bias assessment
   - jailbreak_resistance - Prompt injection defense
   - content_filtering - Inappropriate content

5. **Creative** (`eval_templates/creative.py`)
   - story_generation - Creative writing
   - poetry_evaluation - Poetry quality
   - humor_assessment - Joke quality
   - creative_problem_solving - Novel solutions

6. **Multimodal** (`eval_templates/multimodal.py`)
   - image_captioning - Describe images
   - visual_qa - Visual question answering
   - ocr_evaluation - Text extraction
   - chart_understanding - Data visualization

### Using Templates

```python
from tldw_chatbook.Evals.eval_templates import get_eval_templates

templates = get_eval_templates()

# Get template by name
template = templates.get_template('gsm8k')

# Use template configuration
task_config = {
    'name': template['name'],
    'task_type': template['task_type'],
    'metric': template['metric'],
    'generation_kwargs': template['generation_kwargs']
}

# Create task with template
task_id = await orchestrator.create_task_from_template(
    template_name='gsm8k',
    dataset_override='my_custom_dataset.json'  # Optional
)
```

## Testing

### Running Tests

```bash
# Run all tests
python Tests/Evals/run_tests.py all

# Run specific test suite
python Tests/Evals/run_tests.py orchestrator
python Tests/Evals/run_tests.py errors
python Tests/Evals/run_tests.py exporters
python Tests/Evals/run_tests.py integration

# Run with coverage
python Tests/Evals/run_tests.py coverage

# Run with pytest directly
pytest Tests/Evals/ -v

# Run specific test
pytest Tests/Evals/test_eval_orchestrator.py::TestEvaluationOrchestrator::test_active_tasks_initialization
```

### Test Structure

```
Tests/Evals/
├── test_eval_orchestrator.py  # Orchestrator tests
├── test_eval_errors.py        # Error handling tests
├── test_exporters.py          # Export functionality
├── test_integration.py        # Integration tests
├── run_tests.py              # Test runner
└── TESTING_SUMMARY.md        # Test documentation
```

### Writing Tests

Example test for custom functionality:

```python
import pytest
from unittest.mock import Mock, patch
from tldw_chatbook.Evals.eval_orchestrator import EvaluationOrchestrator

class TestCustomFeature:
    @pytest.fixture
    def orchestrator(self, tmp_path):
        db_path = tmp_path / "test.db"
        return EvaluationOrchestrator(db_path=str(db_path))
    
    @pytest.mark.asyncio
    async def test_custom_evaluation(self, orchestrator):
        # Your test implementation
        with patch('module.function') as mock_func:
            mock_func.return_value = expected_value
            result = await orchestrator.custom_method()
            assert result == expected_value
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   Solution: Install required dependencies
   pip install -e ".[evals]"
   ```

2. **Database Lock Errors**
   ```
   Solution: Ensure only one orchestrator instance per database
   ```

3. **API Rate Limits**
   ```
   Solution: Configure retry delays and concurrent limits
   ```

4. **Memory Issues with Large Datasets**
   ```
   Solution: Use max_samples parameter or batch processing
   ```

5. **Budget Exceeded**
   ```
   Solution: Set appropriate budget limits and monitor usage
   ```

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or in configuration
config = {
    'logging': {
        'level': 'DEBUG',
        'file': 'eval_debug.log'
    }
}
```

## Examples

### Complete Example: Math Evaluation

```python
import asyncio
from tldw_chatbook.Evals.eval_orchestrator import EvaluationOrchestrator
from tldw_chatbook.Evals.exporters import EvaluationExporter

async def evaluate_math_models():
    # Initialize
    orchestrator = EvaluationOrchestrator()
    exporter = EvaluationExporter()
    
    # Create task from GSM8K template
    task_id = await orchestrator.create_task_from_template(
        template_name='gsm8k',
        max_samples=100
    )
    
    # Configure models to compare
    models = [
        {
            'provider': 'openai',
            'model_id': 'gpt-4',
            'name': 'GPT-4',
            'temperature': 0.1  # Lower for math
        },
        {
            'provider': 'anthropic',
            'model_id': 'claude-3-opus',
            'name': 'Claude Opus',
            'temperature': 0.1
        }
    ]
    
    # Run evaluation
    print("Starting evaluation...")
    run_id = await orchestrator.run_evaluation(
        task_id=task_id,
        model_configs=models,
        run_config={
            'type': 'ab_test',
            'budget_limit': 20.0,
            'parallel': True
        }
    )
    
    # Monitor progress
    while True:
        status = orchestrator.get_run_status(run_id)
        print(f"Progress: {status['progress']}%")
        
        if status['status'] in ['completed', 'failed', 'cancelled']:
            break
        
        await asyncio.sleep(5)
    
    # Export results
    if status['status'] == 'completed':
        # Export detailed CSV
        exporter.export(status, "math_eval_results.csv", format="csv")
        
        # Export comparison report
        exporter.export(status, "math_eval_report.md", format="markdown")
        
        # Export for analysis
        exporter.export(status, "math_eval_data.json", format="json")
        
        print(f"Evaluation complete! Results saved.")
        print(f"Winner: {status['winner']['name']} with {status['winner']['score']:.2%} accuracy")
    else:
        print(f"Evaluation {status['status']}: {status.get('error', 'Unknown error')}")
    
    # Cleanup
    orchestrator.close()

# Run the evaluation
if __name__ == "__main__":
    asyncio.run(evaluate_math_models())
```

## Support

For issues, questions, or contributions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review test files for usage examples
3. Check inline documentation in source files
4. Open an issue in the repository

## Version History

- **v2.0.0** (2025-08-16) - Major refactoring, bug fixes, test suite
- **v1.0.0** - Initial implementation

## License

This module is part of the tldw_chatbook project and follows the same license terms.