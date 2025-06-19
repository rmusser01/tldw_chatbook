# Sample Evaluation Tasks

This directory contains sample evaluation task files demonstrating different formats and evaluation types supported by the tldw_chatbook evaluation system.

## Directory Structure

```
sample_evaluation_tasks/
├── README.md                          # This file
├── eleuther_format/                   # Eleuther AI format tasks
│   ├── mmlu_sample.yaml              # MMLU multiple choice
│   ├── gsm8k_sample.yaml             # GSM8K math problems
│   ├── humaneval_sample.yaml         # Code generation
│   └── arc_sample.yaml               # ARC reasoning
├── custom_format/                     # Custom JSON format tasks
│   ├── qa_basic.json                 # Basic Q&A
│   ├── code_generation.json          # Python code generation
│   ├── safety_evaluation.json        # AI safety testing
│   ├── multilingual_qa.json          # Cross-lingual evaluation
│   └── creative_writing.json         # Creative content generation
├── csv_datasets/                      # CSV format datasets
│   ├── simple_qa.csv                 # Simple question-answer pairs
│   ├── math_problems.csv             # Mathematical reasoning
│   ├── science_facts.csv             # Science knowledge
│   └── trivia_questions.csv          # General trivia
├── huggingface_configs/              # HuggingFace dataset configurations
│   ├── squad_config.json            # SQuAD reading comprehension
│   ├── glue_sst2_config.json        # GLUE sentiment analysis
│   └── xnli_config.json             # Cross-lingual NLI
└── specialized_tasks/                # Specialized evaluation types
    ├── bias_evaluation.json          # Bias detection
    ├── robustness_test.json          # Adversarial robustness
    ├── few_shot_learning.json        # Few-shot learning
    └── chain_of_thought.json         # Reasoning evaluation
```

## Usage Examples

### Loading a Task File

```python
from tldw_chatbook.App_Functions.Evals.task_loader import TaskLoader

loader = TaskLoader()

# Load Eleuther format
config = loader.load_task("eleuther_format/mmlu_sample.yaml", "eleuther")

# Load custom format
config = loader.load_task("custom_format/qa_basic.json", "custom")

# Auto-detect format
config = loader.load_task("csv_datasets/simple_qa.csv", "auto")
```

### Running an Evaluation

```python
from tldw_chatbook.App_Functions.Evals.eval_orchestrator import EvaluationOrchestrator

orchestrator = EvaluationOrchestrator()

# Create task from file
task_id = await orchestrator.create_task_from_file("custom_format/qa_basic.json", "custom")

# Create model config
model_id = orchestrator.create_model_config(
    name="GPT-4",
    provider="openai",
    model_id="gpt-4"
)

# Run evaluation
run_id = await orchestrator.run_evaluation(
    task_id=task_id,
    model_id=model_id,
    max_samples=50
)
```

## Task Format Reference

### Eleuther AI Format
- Based on the evaluation harness format
- Uses YAML configuration
- Supports templates and complex evaluation logic
- Best for standardized benchmarks

### Custom JSON Format
- Simplified configuration
- Easy to create and modify
- Good for custom evaluations
- Flexible metadata support

### CSV Format
- Direct dataset specification
- Automatic column detection
- Good for simple tabular data
- Supports custom delimiters

### HuggingFace Format
- Leverages HuggingFace datasets
- Automatic data loading
- Supports dataset configurations
- Access to thousands of datasets

## Evaluation Types Covered

1. **Question Answering** - Basic Q&A, reading comprehension
2. **Multiple Choice** - MMLU-style academic benchmarks
3. **Code Generation** - Programming and algorithm tasks
4. **Mathematical Reasoning** - Math word problems, calculations
5. **Safety Evaluation** - Harmful content detection, bias testing
6. **Multilingual** - Cross-language capabilities
7. **Creative Tasks** - Story writing, content generation
8. **Robustness** - Adversarial and stress testing
9. **Reasoning** - Logical reasoning, chain of thought
10. **Domain-Specific** - Science, history, specialized knowledge

## Getting Started

1. **Choose a task type** based on what you want to evaluate
2. **Select the appropriate format** for your needs
3. **Customize the configuration** with your specific requirements
4. **Load and run** the evaluation using the orchestrator
5. **Analyze results** through the UI or export functionality

## Contributing New Tasks

When adding new sample tasks:

1. Follow the existing directory structure
2. Include comprehensive metadata
3. Provide clear descriptions and examples
4. Test with multiple models
5. Document any special requirements
6. Consider different difficulty levels

## Notes

- All sample files are functional examples that can be run directly
- Datasets referenced may need to be downloaded separately
- Some tasks require specific model capabilities (e.g., code generation)
- Performance may vary significantly across different models
- Consider rate limits and costs when running large evaluations