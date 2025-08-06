# Evaluation Task Formats Guide

This guide provides comprehensive documentation on creating evaluation tasks for the LLM evaluation system, including detailed examples for MMLU-style tasks.

## Table of Contents

1. [Overview](#overview)
2. [Supported Formats](#supported-formats)
3. [Task Types](#task-types)
4. [Format Specifications](#format-specifications)
5. [MMLU Examples](#mmlu-examples)
6. [Creating Custom Tasks](#creating-custom-tasks)
7. [Best Practices](#best-practices)
8. [Common Pitfalls](#common-pitfalls)

## Overview

The evaluation system supports multiple task formats to accommodate different evaluation scenarios:

- **Eleuther AI Format**: Compatible with the lm-evaluation-harness
- **Custom JSON/YAML**: Flexible format for custom evaluations
- **CSV/TSV**: Simple tabular format for basic tasks
- **HuggingFace Datasets**: Direct integration with HF datasets

## Supported Formats

### 1. Eleuther AI Format (YAML)

The most comprehensive format, compatible with the Eleuther AI evaluation harness.

```yaml
# Task configuration
task: mmlu_abstract_algebra
dataset_path: hendrycks/mmlu
dataset_name: abstract_algebra
output_type: multiple_choice
num_fewshot: 5
metric_list:
  - metric: acc
    aggregation: mean
  - metric: acc_norm
    aggregation: mean

# Dataset configuration  
training_split: auxiliary_train
validation_split: val
test_split: test

# Prompt templates
doc_to_text: |
  Question: {{question}}
  A) {{choices[0]}}
  B) {{choices[1]}}
  C) {{choices[2]}}
  D) {{choices[3]}}
  Answer:

doc_to_choice: ["A", "B", "C", "D"]
doc_to_target: answer

# Generation parameters
generation_kwargs:
  max_gen_toks: 1
  temperature: 0
```

### 2. Custom JSON Format

A simplified format for quick task creation.

```json
{
  "name": "MMLU Physics Questions",
  "description": "Multiple choice physics questions from MMLU",
  "task_type": "classification",
  "dataset_name": "custom_mmlu_physics.json",
  "metric": "accuracy",
  "generation_kwargs": {
    "max_tokens": 1,
    "temperature": 0
  },
  "doc_to_text": "Question: {question}\nA) {choice_a}\nB) {choice_b}\nC) {choice_c}\nD) {choice_d}\nAnswer:",
  "doc_to_target": "{answer}",
  "doc_to_choice": ["A", "B", "C", "D"]
}
```

### 3. CSV Format

Simple format for basic question-answer tasks.

```csv
question,choice_a,choice_b,choice_c,choice_d,answer
"What is the speed of light?","299,792 km/s","299,792,458 m/s","300,000 km/s","299,792,458 km/s","B"
"What is Newton's second law?","F = ma","E = mc²","p = mv","W = Fd","A"
```

## Task Types

### 1. Classification (Multiple Choice)

Used for MMLU-style tasks where the model selects from predefined options.

**Key Properties:**
- `task_type`: "classification"
- `doc_to_choice`: List of valid choices
- `metric`: Usually "accuracy"

### 2. Question Answer

Open-ended questions with specific expected answers.

**Key Properties:**
- `task_type`: "question_answer"
- `metric`: "exact_match" or "f1_score"

### 3. Generation

Tasks requiring longer-form generation.

**Key Properties:**
- `task_type`: "generation"
- `stop_sequences`: List of strings to stop generation
- `metric`: "bleu", "rouge", or custom metrics

### 4. Code Generation

Programming tasks with execution validation.

**Key Properties:**
- `task_type": "code_generation"
- `test_cases`: List of input/output pairs
- `metric`: "pass@k" or "execution_accuracy"

## Format Specifications

### Required Fields

All task formats must include:

1. **name**: Unique identifier for the task
2. **task_type**: One of the supported types
3. **dataset_name**: Path to dataset or HuggingFace identifier
4. **metric**: Evaluation metric to use

### Optional Fields

- **description**: Human-readable description
- **num_fewshot**: Number of examples for few-shot prompting
- **generation_kwargs**: Model generation parameters
- **doc_to_text**: Template for formatting inputs
- **doc_to_target**: Template for extracting expected output
- **doc_to_choice**: List of valid choices (classification only)

## MMLU Examples

### Example 1: MMLU Physics (Eleuther Format)

```yaml
task: mmlu_physics
dataset_path: hendrycks/mmlu
dataset_name: high_school_physics
output_type: multiple_choice
num_fewshot: 5

training_split: auxiliary_train
validation_split: val
test_split: test

doc_to_text: |
  The following is a multiple choice question about physics.
  
  Question: {{question}}
  A) {{choices[0]}}
  B) {{choices[1]}}
  C) {{choices[2]}}
  D) {{choices[3]}}
  Answer:

doc_to_choice: ["A", "B", "C", "D"]
doc_to_target: answer

metric_list:
  - metric: acc
    aggregation: mean

generation_kwargs:
  max_gen_toks: 1
  temperature: 0
  do_sample: false
```

### Example 2: Custom MMLU Format (JSON)

```json
{
  "name": "Custom MMLU History",
  "description": "World history questions in MMLU style",
  "task_type": "classification",
  "dataset_name": "custom_history_questions.json",
  "metric": "accuracy",
  "num_fewshot": 3,
  "doc_to_text": "Question: {question}\n\nChoices:\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}\n\nThe correct answer is:",
  "doc_to_choice": ["A", "B", "C", "D"],
  "doc_to_target": "{answer}",
  "generation_kwargs": {
    "max_tokens": 1,
    "temperature": 0,
    "top_p": 1.0
  }
}
```

### Example 3: MMLU with Custom Scoring

```yaml
task: mmlu_with_partial_credit
dataset_path: custom_mmlu_dataset.json
output_type: multiple_choice

doc_to_text: |
  Subject: {{subject}}
  
  {{question}}
  
  Options:
  A) {{choices[0]}}
  B) {{choices[1]}}
  C) {{choices[2]}}
  D) {{choices[3]}}
  
  Select the best answer (A, B, C, or D):

doc_to_choice: ["A", "B", "C", "D"]
doc_to_target: answer

# Custom metrics for partial credit
metric_list:
  - metric: acc
    aggregation: mean
  - metric: acc_norm  # Normalized by question difficulty
    aggregation: mean

# Metadata for difficulty-based scoring
metadata:
  difficulty_field: difficulty
  partial_credit: true
```

## Creating Custom Tasks

### Step 1: Define Your Task Structure

Decide on:
- Task type (classification, QA, generation)
- Input format
- Expected output format
- Evaluation metric

### Step 2: Create the Task Configuration

Choose between YAML or JSON based on complexity:
- Use YAML for Eleuther-compatible tasks
- Use JSON for simpler custom tasks

### Step 3: Prepare Your Dataset

Ensure your dataset includes:
- All required fields referenced in templates
- Consistent formatting
- Proper escaping of special characters

### Step 4: Test Your Task

1. Validate the configuration:
   ```python
   python -m tldw_chatbook.Evals.task_loader validate my_task.yaml
   ```

2. Run a small test:
   ```python
   python -m tldw_chatbook.Evals.eval_runner test my_task.yaml --max-samples 10
   ```

## Best Practices

### 1. Template Design

- **Clear Instructions**: Make prompts unambiguous
- **Consistent Formatting**: Use the same format across all samples
- **Avoid Bias**: Don't hint at correct answers in prompts

### 2. Dataset Quality

- **Balanced Classes**: Ensure equal distribution of answers
- **Diverse Examples**: Cover edge cases and variations
- **Quality Control**: Manually review a sample of your data

### 3. Few-Shot Examples

- **Representative**: Choose examples that cover different cases
- **Ordered**: Place examples from simple to complex
- **Relevant**: Ensure examples match the task domain

### 4. Metric Selection

- **Appropriate**: Choose metrics that match your task type
- **Multiple Metrics**: Use several metrics for comprehensive evaluation
- **Custom Metrics**: Implement domain-specific metrics when needed

## Common Pitfalls

### 1. Format Errors

**Problem**: Inconsistent field names between config and dataset
```yaml
# Config expects 'question' but dataset has 'text'
doc_to_text: "{{question}}"  # Will fail!
```

**Solution**: Verify field names match exactly
```yaml
doc_to_text: "{{text}}"  # Matches dataset field
```

### 2. Template Issues

**Problem**: Missing space after prompt
```yaml
doc_to_text: "Question: {{question}}Answer:"  # No space!
```

**Solution**: Add appropriate spacing
```yaml
doc_to_text: "Question: {{question}}\nAnswer: "
```

### 3. Choice Formatting

**Problem**: Inconsistent choice labeling
```json
{
  "doc_to_choice": ["A", "B", "C", "D"],
  "dataset_answers": ["a", "b", "c", "d"]  // Lowercase!
}
```

**Solution**: Ensure case matches
```json
{
  "doc_to_choice": ["a", "b", "c", "d"],
  "dataset_answers": ["a", "b", "c", "d"]
}
```

### 4. Metric Misalignment

**Problem**: Using exact_match for multiple choice
```yaml
task_type: classification
metric: exact_match  # Should be 'accuracy'
```

**Solution**: Use appropriate metrics
```yaml
task_type: classification  
metric: accuracy
```

## Advanced Features

### 1. Conditional Templates

Use Jinja2 conditionals for complex formatting:
```yaml
doc_to_text: |
  {% if subject %}Subject: {{subject}}{% endif %}
  Question: {{question}}
  {% for i, choice in enumerate(choices) %}
  {{chr(65+i)}}) {{choice}}
  {% endfor %}
```

### 2. Custom Preprocessing

Add preprocessing functions in your config:
```json
{
  "preprocessing": {
    "lowercase_choices": true,
    "strip_punctuation": true,
    "normalize_numbers": true
  }
}
```

### 3. Multi-Stage Evaluation

Define tasks with multiple evaluation stages:
```yaml
stages:
  - name: initial_answer
    doc_to_text: "{{question}}"
    metric: exact_match
  - name: explanation
    doc_to_text: "Explain your answer to: {{question}}"
    metric: rouge_l
```

## Conclusion

Creating effective evaluation tasks requires careful consideration of format, metrics, and data quality. Start with the provided templates and customize based on your specific needs. Always validate your tasks with small samples before running large-scale evaluations.

For more examples, see the `examples/eval_tasks/` directory in the codebase.