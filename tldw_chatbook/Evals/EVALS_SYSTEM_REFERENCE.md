# EVALS System Reference Document

**Date Created**: 2025-06-18  
**Last Updated**: 2025-07-06  
**Project**: tldw_chatbook  
**Feature**: LLM Evaluation Framework  

> **Note**: This document has been updated and consolidated. For the most current and comprehensive documentation, please refer to [EVALUATION-SYSTEM.md](../../Docs/Development/EVALUATION-SYSTEM.md) in the project root.

## Overview

The LLM evaluation system in tldw_chatbook provides a comprehensive framework for benchmarking Large Language Model performance across various tasks. The backend is fully implemented and production-ready, supporting 30+ LLM providers and 27+ evaluation task types.

## Current Status

**Backend**: ✅ COMPLETE  
**UI Integration**: ⚠️ PENDING  
**Testing**: ✅ COMPREHENSIVE (200+ tests)  
**Documentation**: ✅ UPDATED  

## Quick Links

- **Comprehensive Documentation**: [EVALUATION-SYSTEM.md](../../Docs/Development/EVALUATION-SYSTEM.md)
- **Implementation Status**: [EVALUATIONS-STATUS.md](../../Docs/Development/EVALUATIONS-STATUS.md)
- **User Guide**: [EVALUATIONS-QUICKSTART.md](../../Docs/Development/EVALUATIONS-QUICKSTART.md)
- **Sample Tasks**: [/sample_evaluation_tasks/](../../sample_evaluation_tasks/)

## Module Structure

```
Evals/
├── __init__.py                    # Package exports
├── eval_orchestrator.py           # High-level API (✅ Complete)
├── eval_runner.py                 # Core runners (✅ Complete)
├── specialized_runners.py         # 7 specialized runners (✅ Complete)
├── llm_interface.py              # 30+ provider support (✅ Complete)
├── eval_templates.py             # Prompt templates (✅ Complete)
└── EVALS_SYSTEM_REFERENCE.md    # This file
```

## Key Features

### Supported Providers (30+)
**Commercial**: OpenAI, Anthropic, Google, Cohere, Groq, Mistral, DeepSeek, HuggingFace, OpenRouter  
**Local**: Ollama, Llama.cpp, vLLM, Kobold, TabbyAPI, Aphrodite, MLX-LM, ONNX, Transformers

### Task Types (27+)
- **Text Understanding**: Q&A, Reading Comprehension, Classification
- **Generation**: Text Completion, Creative Writing, Code Generation
- **Reasoning**: Mathematical, Logical, Common Sense
- **Language**: Translation, Summarization, Paraphrasing
- **Specialized**: Code Execution, Scientific/Medical Q&A
- **Safety**: Harmful Content Detection, Bias Evaluation
- **Multimodal**: Vision tasks (when supported)

### Metrics
- **Text Matching**: Exact match, F1, Contains
- **ROUGE**: ROUGE-1, ROUGE-2, ROUGE-L
- **BLEU**: 1-4 gram support with brevity penalty
- **Semantic**: Sentence transformer similarity
- **Perplexity**: From log probabilities
- **Custom**: Extensible metric framework

## API Usage

### Quick Start
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

### Direct Runner Usage
```python
from tldw_chatbook.Evals.eval_runner import create_runner

# Create runner for specific task type
runner = create_runner(task_config, model_config)

# Run evaluation
results = await runner.run_evaluation(
    samples=your_samples,
    progress_callback=your_callback
)
```

## File Formats

### Eleuther AI YAML
```yaml
task: my_task_name
dataset_name: my_dataset
task_type: question_answer
metric: exact_match
doc_to_text: "Question: {{question}}"
doc_to_target: "{{answer}}"
```

### Custom JSON
```json
{
  "name": "My Task",
  "task_type": "question_answer",
  "metric": "f1",
  "samples": [
    {
      "input": "What is the capital of France?",
      "expected": "Paris"
    }
  ]
}
```

### CSV Format
```csv
input,expected
"What is 2+2?","4"
"Capital of Japan?","Tokyo"
```

## Development

### Adding a New Provider
See [llm_interface.py](./llm_interface.py) - add your provider function and register in `PROVIDER_HANDLERS`.

### Adding a New Runner
See [specialized_runners.py](./specialized_runners.py) - extend `BaseEvalRunner` and implement required methods.

### Adding a New Metric
See [eval_runner.py](./eval_runner.py) - add to `calculate_metrics` function.

## Testing

Run all evaluation tests:
```bash
pytest Tests/Evals/ -v
```

## Database Schema

The evaluation system uses 6 core tables:
- `eval_tasks`: Task definitions
- `eval_datasets`: Dataset storage
- `eval_models`: Model configurations
- `eval_runs`: Evaluation runs
- `eval_results`: Individual results
- `eval_run_metrics`: Aggregated metrics

See [Evals_DB.py](../DB/Evals_DB.py) for complete schema.

## Future Work

1. **UI Integration** (2-3 weeks)
   - Configuration dialogs
   - Progress visualization
   - Results display
   
2. **Enhanced Features**
   - Batch evaluation management
   - Cost tracking
   - Evaluation templates library

3. **Advanced Capabilities**
   - Distributed evaluation
   - Custom metric plugins
   - A/B testing framework

For detailed roadmap, see [EVALUATION-SYSTEM.md](../../Docs/Development/EVALUATION-SYSTEM.md#roadmap).

## Support

- **Issues**: Report in GitHub Issues
- **Documentation**: See linked documents above
- **Examples**: Check `/sample_evaluation_tasks/`