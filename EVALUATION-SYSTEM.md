# tldw_chatbook Evaluation System Documentation

**Last Updated**: 2025-07-06  
**System Status**: Backend Complete, UI Integration Pending  
**Version**: 1.0

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Implementation Status](#implementation-status)
4. [Supported Features](#supported-features)
5. [API Reference](#api-reference)
6. [Development Guide](#development-guide)
7. [Testing](#testing)
8. [Roadmap](#roadmap)

## Overview

The tldw_chatbook evaluation system provides a comprehensive framework for benchmarking Large Language Model (LLM) performance across various tasks. The system features a robust backend implementation with support for multiple evaluation formats, providers, and task types.

### Key Capabilities
- **Multi-format support**: Eleuther AI YAML, Custom JSON, CSV/TSV, HuggingFace datasets
- **30+ LLM providers**: Both commercial (OpenAI, Anthropic, etc.) and local (Ollama, vLLM, etc.)
- **7 evaluation categories** with 27+ specialized task types
- **Advanced metrics**: ROUGE, BLEU, semantic similarity, perplexity, custom metrics
- **Secure code execution**: Sandboxed environment for code evaluation tasks
- **Comprehensive database**: SQLite with FTS5 for result storage and analysis

## Architecture

### System Components

```
tldw_chatbook/
├── DB/
│   └── Evals_DB.py              # Complete database implementation
├── Evals/
│   ├── __init__.py              
│   ├── eval_orchestrator.py     # Pipeline coordination
│   ├── eval_runner.py           # Core evaluation engine
│   ├── specialized_runners.py   # Task-specific runners
│   ├── llm_interface.py         # Provider integrations
│   └── eval_templates.py        # Prompt templates
├── UI/
│   └── Evals_Window.py          # UI framework (incomplete)
├── Event_Handlers/
│   └── eval_events.py           # Event routing
└── Widgets/
    └── file_picker_dialog.py    # File selection dialogs
```

### Data Flow
1. **Task Loading** → Load evaluation tasks from various formats
2. **Configuration** → Set up models and evaluation parameters
3. **Orchestration** → Coordinate evaluation pipeline
4. **Execution** → Run evaluations with appropriate runners
5. **Storage** → Save results to database
6. **Analysis** → Aggregate and export results

## Implementation Status

### ✅ Fully Implemented (Backend)

#### Database Layer (`DB/Evals_DB.py`)
- **Complete schema** with 6 core tables:
  - `eval_tasks`: Task definitions and configurations
  - `eval_datasets`: Dataset management
  - `eval_models`: Model configurations
  - `eval_runs`: Evaluation run metadata
  - `eval_results`: Individual evaluation results
  - `eval_run_metrics`: Aggregated metrics
- **Features**:
  - Thread-safe SQLite with WAL mode
  - FTS5 full-text search
  - Optimistic locking with version fields
  - Soft deletion support
  - Schema versioning and migrations
  - Comprehensive CRUD operations

#### LLM Interface (`Evals/llm_interface.py`)
Complete support for 30+ providers:

**Commercial Providers**:
- OpenAI (GPT-3.5/4 with logprobs)
- Anthropic (Claude models)
- Google (Gemini)
- Cohere (Command models)
- DeepSeek
- Mistral
- Groq (high-speed inference)
- HuggingFace Hub
- OpenRouter (multi-provider gateway)

**Local Providers**:
- Ollama
- Llama.cpp
- vLLM (with logprobs)
- Kobold.cpp
- TabbyAPI
- Aphrodite (vLLM fork)
- MLX-LM (Apple Silicon)
- Custom OpenAI-compatible
- ONNX Runtime
- Transformers library

#### Evaluation Runners

**Base Runners** (`eval_runner.py`):
- `QuestionAnswerRunner`: Q&A tasks with exact match/F1
- `ClassificationRunner`: Multiple choice with accuracy metrics
- `LogProbRunner`: Perplexity calculation
- `GenerationRunner`: Open-ended generation

**Specialized Runners** (`specialized_runners.py`):
- `CodeExecutionRunner`: Secure Python code execution with test cases
- `SafetyEvaluationRunner`: Harmfulness and bias detection
- `MultilingualEvaluationRunner`: Translation quality assessment
- `CreativeEvaluationRunner`: Creative writing evaluation
- `MathReasoningRunner`: Mathematical problem solving
- `SummarizationRunner`: Summary quality with ROUGE scores
- `DialogueRunner`: Conversational coherence

#### Metrics Implementation
- **Text Matching**: Exact match, contains, F1 score
- **ROUGE Metrics**: ROUGE-1, ROUGE-2, ROUGE-L with n-gram analysis
- **BLEU Score**: 1-4 gram support with brevity penalty
- **Semantic Similarity**: Sentence transformers integration
- **Perplexity**: From log probabilities
- **Custom Metrics**: Extensible metric framework

#### Task Loading System
- **Format Detection**: Automatic format identification
- **Supported Formats**:
  - Eleuther AI YAML with full template support
  - Custom JSON with flexible schema
  - CSV/TSV with automatic column mapping
  - HuggingFace datasets (when available)
- **Template Processing**: Jinja2-style templates for prompts

### ⚠️ Partially Implemented (UI Layer)

#### UI Framework (`UI/Evals_Window.py`)
- ✅ Layout structure with sidebar navigation
- ✅ Four main views (Setup, Results, Models, Datasets)
- ✅ Event handler connections
- ❌ Actual functionality not connected to backend
- ❌ Progress updates not implemented
- ❌ Results display incomplete

#### Event Handlers (`Event_Handlers/eval_events.py`)
- ✅ Handler structure and routing
- ✅ Backend integration code
- ⚠️ Depends on missing UI widget implementations
- ⚠️ File picker integration incomplete

#### File Picker Dialogs (`Widgets/file_picker_dialog.py`)
- ✅ Classes defined: `TaskFilePickerDialog`, `DatasetFilePickerDialog`, `ExportFilePickerDialog`
- ⚠️ Implementation may be incomplete or not properly integrated

### ❌ Missing Components

1. **Configuration Dialogs**:
   - `ModelConfigDialog`
   - `TaskConfigDialog`
   - `RunConfigDialog`

2. **Results Visualization**:
   - Result table population
   - Metrics charts
   - Comparison views

3. **Template Management UI**:
   - Template creation/editing
   - Template selection

## Supported Features

### Task Types by Category

#### 1. Text Understanding (6 types)
- Question Answering
- Reading Comprehension
- Multiple Choice Questions
- Text Classification
- Named Entity Recognition
- Sentiment Analysis

#### 2. Generation Tasks (5 types)
- Text Completion
- Creative Writing
- Dialogue Generation
- Code Generation
- Structured Output Generation

#### 3. Reasoning Tasks (4 types)
- Mathematical Reasoning
- Logical Reasoning
- Common Sense Reasoning
- Causal Reasoning

#### 4. Language Understanding (4 types)
- Translation
- Summarization
- Paraphrasing
- Grammar Correction

#### 5. Specialized Domains (3 types)
- Code Execution & Testing
- Scientific QA
- Medical QA

#### 6. Safety & Alignment (3 types)
- Harmful Content Detection
- Bias Evaluation
- Instruction Following

#### 7. Multimodal Tasks (2 types)
- Image Captioning (when vision models available)
- Visual Question Answering

### Configuration Options

```python
# Task Configuration
{
    "name": "My Evaluation",
    "task_type": "question_answer",
    "metric": "exact_match",  # or f1, bleu, rouge, semantic_similarity
    "few_shot": 3,            # Number of examples
    "temperature": 0.0,       # For consistent results
    "max_tokens": 512,
    "system_prompt": "You are a helpful assistant.",
    "timeout": 30.0          # Seconds per sample
}

# Model Configuration
{
    "name": "GPT-4 Turbo",
    "provider": "openai",
    "model_id": "gpt-4-turbo-preview",
    "api_key": "from_config",  # or explicit key
    "parameters": {
        "temperature": 0.0,
        "max_tokens": 2048
    }
}
```

## API Reference

### Orchestrator API

```python
from tldw_chatbook.Evals import EvaluationOrchestrator

# Initialize
orchestrator = EvaluationOrchestrator()

# Create task from file
task_id = await orchestrator.create_task_from_file(
    file_path="path/to/task.json",
    task_name="My Task"
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
    max_samples=100,
    progress_callback=my_callback
)

# Export results
await orchestrator.export_results(
    run_ids=[run_id],
    output_path="results.csv",
    format="csv"
)
```

### Direct Runner Usage

```python
from tldw_chatbook.Evals.eval_runner import create_runner

# Create appropriate runner
runner = create_runner(task_config, model_config, db_path)

# Run evaluation
results = await runner.run_evaluation(
    samples=dataset_samples,
    progress_callback=callback
)
```

## Development Guide

### Adding a New Provider

1. Add provider function to `llm_interface.py`:
```python
async def generate_your_provider(
    messages: List[Dict],
    model_config: Dict,
    **kwargs
) -> Union[str, Dict]:
    # Implementation
    return {"text": response_text}
```

2. Register in `PROVIDER_HANDLERS`:
```python
PROVIDER_HANDLERS = {
    # ...
    'your_provider': generate_your_provider,
}
```

### Adding a New Runner

1. Extend `BaseEvalRunner`:
```python
class YourTaskRunner(BaseEvalRunner):
    def get_prompt(self, sample: Dict) -> str:
        # Format prompt for your task
        
    async def evaluate_response(
        self, 
        response: str, 
        expected: Any,
        sample: Dict
    ) -> Dict[str, float]:
        # Calculate metrics
```

2. Register in `RUNNER_MAPPING`:
```python
RUNNER_MAPPING = {
    # ...
    'your_task': YourTaskRunner,
}
```

### Adding a New Metric

1. Implement in `calculate_metrics`:
```python
def calculate_metrics(
    metric_type: str,
    generated: str,
    expected: str,
    **kwargs
) -> float:
    if metric_type == "your_metric":
        # Calculate and return score
```

## Testing

### Test Coverage
- **Unit Tests**: 200+ test cases
- **Integration Tests**: End-to-end evaluation flows
- **Property Tests**: Edge case discovery with Hypothesis

### Running Tests
```bash
# All evaluation tests
pytest Tests/Evals/

# Specific test file
pytest Tests/Evals/test_eval_runner.py

# With coverage
pytest Tests/Evals/ --cov=tldw_chatbook.Evals
```

### Test Files
- `test_evals_db.py`: Database operations
- `test_eval_runner.py`: Runner functionality
- `test_eval_integration.py`: Full pipeline tests
- `test_eval_properties.py`: Property-based testing

## Roadmap

### Phase 1: UI Completion (Immediate)
- [ ] Implement missing configuration dialogs
- [ ] Connect UI to backend orchestrator
- [ ] Add real-time progress updates
- [ ] Implement results visualization

### Phase 2: Enhanced Features (Short-term)
- [ ] Batch evaluation management
- [ ] Cost tracking and estimation
- [ ] Advanced filtering and search
- [ ] Evaluation templates library

### Phase 3: Advanced Capabilities (Medium-term)
- [ ] Distributed evaluation support
- [ ] Custom metric plugins
- [ ] Evaluation scheduling
- [ ] A/B testing framework

### Phase 4: Enterprise Features (Long-term)
- [ ] Multi-user support
- [ ] Evaluation versioning
- [ ] Audit trails
- [ ] API endpoint for external integration

## Migration Notes

For users of previous documentation:
- `EVALUATIONS-STATUS.md` is now superseded by this document
- `EVALS_SYSTEM_REFERENCE.md` content is incorporated here
- `EVAL-IMPLEMENTATION-SUMMARY.md` details are in the Implementation Status section
- `EVALUATIONS-QUICKSTART.md` remains as the user guide

## Conclusion

The evaluation system backend is production-ready with comprehensive functionality. The primary remaining work is UI integration to make the system accessible through the Textual interface. The modular architecture allows for easy extension and the robust testing ensures reliability.

For immediate use, the system can be accessed programmatically through the orchestrator API or by directly using the runners. Full UI integration would unlock the complete user experience envisioned in the original design.