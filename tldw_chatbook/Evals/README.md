# Evaluation Module

## Overview

The Evaluation module provides a comprehensive framework for benchmarking Large Language Model (LLM) performance across various tasks. It supports 30+ LLM providers and 27+ evaluation task types with statistical analysis and A/B testing capabilities.

## Architecture

### Core Components

- **`eval_orchestrator.py`** - Main orchestrator that coordinates evaluation runs
- **`eval_runner.py`** - Base runners for standard evaluation tasks (QA, Classification, Generation, LogProb)
- **`specialized_runners.py`** - Advanced runners for specialized tasks (Code, Safety, Multilingual, Creative, etc.)
- **`eval_errors.py`** - Unified error handling with retry logic and budget monitoring
- **`exporters.py`** - Unified export functionality for all result types (CSV, JSON, Markdown, LaTeX)
- **`ab_testing.py`** - A/B testing with statistical analysis
- **`task_loader.py`** - Task configuration loading and management
- **`eval_templates.py`** - Prompt templates for various evaluation types
- **`configuration_validator.py`** - Configuration validation
- **`dataset_validator.py`** - Dataset validation and verification
- **`concurrency_manager.py`** - Manages concurrent evaluation runs
- **`ui_integration.py`** - Integration with the Textual UI

### Error Handling

All error handling is centralized in `eval_errors.py`:
- Structured error contexts with severity levels
- Automatic retry logic with exponential backoff
- Budget monitoring to prevent cost overruns
- User-friendly error messages with recovery suggestions

### Data Flow

1. **Task Loading**: Tasks are loaded via `TaskLoader` from JSON/CSV files
2. **Validation**: Configuration and datasets are validated before execution
3. **Execution**: `EvaluationOrchestrator` manages the evaluation workflow
4. **Processing**: Appropriate runner processes each sample
5. **Storage**: Results are stored in the SQLite database
6. **Export**: Results can be exported in multiple formats

## Usage

### Basic Evaluation

```python
from tldw_chatbook.Evals import EvaluationOrchestrator

# Initialize orchestrator
orchestrator = EvaluationOrchestrator()

# Load task
task_id = await orchestrator.create_task_from_file(
    "path/to/task.json",
    "My Evaluation Task"
)

# Configure model
model_config = {
    "provider": "openai",
    "model_id": "gpt-3.5-turbo",
    "name": "GPT-3.5"
}

# Run evaluation
run_id = await orchestrator.run_evaluation(
    task_id=task_id,
    model_configs=[model_config],
    max_samples=100
)

# Export results
await orchestrator.export_results(
    run_ids=[run_id],
    output_path="results.csv",
    format="csv"
)
```

### A/B Testing

```python
from tldw_chatbook.Evals.ab_testing import ABTestRunner

# Initialize A/B test runner
ab_runner = ABTestRunner(orchestrator)

# Run A/B test
result = await ab_runner.run_ab_test(
    task_id=task_id,
    model_a_config={"provider": "openai", "model_id": "gpt-3.5-turbo"},
    model_b_config={"provider": "anthropic", "model_id": "claude-2"},
    sample_size=100,
    significance_level=0.05
)

# Export results
from tldw_chatbook.Evals.exporters import EvaluationExporter
exporter = EvaluationExporter()
exporter.export(result, "ab_test_results.md", format="markdown")
```

## Supported Task Types

### Standard Tasks
- **Question-Answer**: Evaluate factual Q&A performance
- **Classification**: Text classification accuracy
- **Generation**: Open-ended text generation quality
- **LogProb**: Log probability evaluation

### Specialized Tasks (via `specialized_runners.py`)
- **Code Execution**: Evaluate code generation with execution
- **Safety Evaluation**: Test for harmful content handling
- **Multilingual**: Translation and cross-lingual tasks
- **Creative**: Creative writing evaluation
- **Robustness**: Adversarial and edge case testing
- **Math Reasoning**: Mathematical problem solving
- **Summarization**: Document summarization quality
- **Dialogue**: Conversational ability evaluation

## Configuration

### Task Configuration (JSON)

```json
{
  "name": "Math Problem Solving",
  "task_type": "question_answer",
  "dataset": [
    {
      "id": "1",
      "input": "What is 2 + 2?",
      "expected_output": "4"
    }
  ],
  "metric": "exact_match",
  "prompt_template": "Solve this math problem: {input}"
}
```

### Model Configuration

```python
{
    "provider": "openai",           # Required: LLM provider
    "model_id": "gpt-4",           # Required: Model identifier
    "name": "GPT-4",                # Optional: Display name
    "api_key": "...",              # Optional: API key (uses config if not provided)
    "temperature": 0.7,             # Optional: Sampling temperature
    "max_tokens": 1000             # Optional: Max response tokens
}
```

## Metrics

### Text Matching
- **exact_match**: Exact string matching
- **contains**: Substring matching
- **f1**: Token-level F1 score

### Similarity Metrics
- **bleu**: BLEU score for translation quality
- **rouge**: ROUGE scores for summarization
- **semantic_similarity**: Embedding-based similarity

### Statistical Metrics
- **perplexity**: From log probabilities
- **accuracy**: Classification accuracy
- **precision/recall**: For classification tasks

## Error Recovery

The module includes comprehensive error handling:

```python
# Automatic retry with exponential backoff
@with_error_handling(max_retries=3)
async def evaluation_function():
    # Your evaluation code
    pass

# Budget monitoring
monitor = BudgetMonitor(budget_limit=10.0)
monitor.update_cost(0.01)  # Track costs
```

## Performance Considerations

- **Concurrency**: Managed by `ConcurrentRunManager` to prevent conflicts
- **Rate Limiting**: Automatic handling with retry logic
- **Memory**: Streaming support for large datasets
- **Cost Control**: Budget monitoring prevents unexpected charges

## Testing

Run tests with:
```bash
pytest Tests/Evals/ -v
```

## Recent Refactoring (2025)

The module was refactored to address:
1. **Fixed Bug**: Added missing `_active_tasks` initialization in `eval_orchestrator.py`
2. **Consolidated Error Handling**: Merged three separate error systems into `eval_errors.py`
3. **Removed Redundancy**: Deleted `simplified_runners.py`, `error_handling.py`, `unified_error_handler.py`
4. **Unified Exporters**: Merged `ab_test_exporter.py` and `export_handlers.py` into `exporters.py`
5. **Improved Documentation**: Added this comprehensive README

## Future Improvements

- [ ] Split `eval_runner.py` (2180 lines) into smaller modules
- [ ] Refactor `eval_templates.py` into a package structure
- [ ] Externalize hardcoded configurations to YAML
- [ ] Add more comprehensive integration tests
- [ ] Implement caching for repeated evaluations
- [ ] Add support for streaming evaluations

## Contributing

When adding new features:
1. Extend appropriate base classes in `eval_runner.py`
2. Add error handling using `eval_errors.py` patterns
3. Validate configurations using existing validators
4. Export results through the unified `exporters.py`
5. Add tests for new functionality

## License

Part of tldw_chatbook project under AGPLv3+ license.