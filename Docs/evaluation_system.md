# Evaluation System Documentation

## Overview

The evaluation system in tldw_chatbook provides a comprehensive framework for evaluating Language Model (LLM) performance across various tasks and metrics. It supports specialized evaluation types including multilingual, creative, and robustness evaluations.

## Core Components

### 1. Evaluation Runners

#### Base Runner (`BaseEvalRunner`)
- Abstract base class for all evaluation runners
- Provides common functionality for metrics calculation
- Handles LLM interface integration

#### Specialized Runners

##### MultilingualEvaluationRunner
- **Purpose**: Evaluates multilingual capabilities including translation, cross-lingual QA, and language detection
- **Key Features**:
  - Language detection with confidence scoring
  - Script analysis (Latin, Chinese, Japanese, Arabic, etc.)
  - Fluency metrics calculation
  - Translation quality assessment (BLEU score)
  - Mixed language detection

##### CreativeEvaluationRunner
- **Purpose**: Evaluates creative text generation including stories, poetry, and dialogue
- **Key Features**:
  - Creativity scoring based on vocabulary diversity
  - Detection of narrative elements
  - Emotional language analysis
  - Dialogue quality assessment
  - Structural coherence evaluation

##### RobustnessEvaluationRunner
- **Purpose**: Tests model robustness against various perturbations and adversarial inputs
- **Key Features**:
  - Input perturbation testing (typos, case changes, punctuation)
  - Adversarial question detection
  - Format robustness testing
  - Consistency scoring across variations

### 2. Evaluation Metrics

#### Standard Metrics
- **Accuracy**: Exact match and F1 scores
- **BLEU**: Translation quality
- **Perplexity**: Language modeling quality

#### Custom Metrics
- **Instruction Adherence**: Measures how well outputs follow specific instructions
- **Format Compliance**: Validates output format (JSON, CSV, lists, etc.)
- **Coherence Score**: Evaluates text structure and flow
- **Dialogue Quality**: Assesses conversation naturalness and relevance

### 3. Database Schema

The evaluation system uses SQLite with the following key tables:
- `eval_tasks`: Task definitions and configurations
- `eval_runs`: Evaluation execution records
- `eval_results`: Individual sample results
- `eval_models`: Model configurations
- `eval_datasets`: Dataset metadata

### 4. UI Components

#### EvalsLab
- Unified dashboard for evaluation management
- Quick template selection
- Real-time progress monitoring
- Result visualization and comparison
- Task creation dialog

## Usage Guide

### Running an Evaluation

```python
from tldw_chatbook.Evals.specialized_runners import MultilingualEvaluationRunner
from tldw_chatbook.Evals.eval_runner import TaskConfig

# Configure task
task_config = TaskConfig(
    name="French Translation",
    task_type="generation",
    metric="bleu",
    metadata={"target_language": "french", "subcategory": "translation"}
)

# Initialize runner
runner = MultilingualEvaluationRunner(task_config, model_config)

# Run evaluation
result = await runner.run_sample(sample)
```

### Creating Custom Tasks

1. Through the UI:
   - Click "Create Task" in EvalsLab
   - Fill in task details
   - Select evaluation type
   - Configure metrics

2. Programmatically:
   ```python
   task_id = db.create_task(
       name="Custom Task",
       task_type="generation",
       config_format="custom",
       config_data={...}
   )
   ```

### Analyzing Results

Results include:
- Per-sample metrics
- Aggregate statistics
- Language/creative analysis metadata
- Error tracking

## Configuration

### Task Types
- `question_answer`: Q&A tasks
- `generation`: Text generation tasks
- `classification`: Classification tasks
- `logprob`: Probability-based tasks

### Metric Selection
Choose metrics based on task:
- Translation: `bleu`, `ter`, `meteor`
- Creative: `creativity_score`, `quality_score`
- Robustness: `robustness_score`, `consistency`
- General: `accuracy`, `f1`, `perplexity`

## Best Practices

1. **Task Design**
   - Clear, specific instructions
   - Appropriate metric selection
   - Sufficient sample size

2. **Model Configuration**
   - Consistent temperature settings
   - Appropriate max tokens
   - Proper prompt formatting

3. **Result Analysis**
   - Consider multiple metrics
   - Check for outliers
   - Validate against human judgment

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all dependencies installed: `pip install -e ".[dev]"`
   - Check Python version ≥ 3.11

2. **Database Errors**
   - Valid task types: `question_answer`, `generation`, `classification`, `logprob`
   - Config format must be `eleuther` or `custom`

3. **UI Issues**
   - DOM elements load asynchronously
   - Use `call_after_refresh` for initialization

## API Reference

### TaskConfig
```python
@dataclass
class TaskConfig:
    name: str
    description: str
    task_type: str
    dataset_name: str
    metric: str
    generation_kwargs: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### EvalSample
```python
@dataclass
class EvalSample:
    id: str
    input_text: str
    expected_output: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Runner Methods
- `run_sample(sample: EvalSample) -> EvalSampleResult`
- `calculate_metrics(predicted: str, expected: str, sample: EvalSample) -> Dict[str, float]`

## Future Enhancements

- Additional language support
- More creative evaluation metrics
- Advanced robustness testing
- Integration with external benchmarks
- Real-time collaboration features