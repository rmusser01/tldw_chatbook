# Evaluation System Quick Start Guide

## Overview

The tldw_chatbook evaluation system allows you to benchmark LLM performance on various tasks. The system is **fully functional** and ready to use!

## Prerequisites

1. **API Keys**: Configure at least one LLM provider in your `~/.config/tldw_cli/config.toml`:
   ```toml
   [api_settings.openai]
   api_key = "your-openai-api-key"
   ```

2. **Sample Tasks**: Located in `/sample_evaluation_tasks/` with examples for:
   - Question answering
   - Code generation
   - Math problems
   - Safety evaluation
   - Multilingual tasks

## Quick Start Steps

### 1. Run the Application
```bash
python3 -m tldw_chatbook.app
```

### 2. Navigate to Evaluations Tab
Click on the "Evaluations" tab in the top navigation.

### 3. Upload a Task File

1. Click **"Upload Task"** button
2. Navigate to `sample_evaluation_tasks/custom_format/`
3. Select `qa_basic.json` for a simple Q&A evaluation
4. Click **"Select"**

### 4. Configure a Model

1. Click **"Add Model"** button
2. Fill in the configuration:
   - **Name**: "GPT-3.5 Test"
   - **Provider**: "openai"
   - **Model ID**: "gpt-3.5-turbo"
3. Click **"Save"**

### 5. Run an Evaluation

1. Click **"Start Evaluation"** button
2. Select your task and model
3. Set **Max Samples**: 5 (for quick testing)
4. Click **"Run"**

### 6. View Results

Results appear in real-time in the Results view showing:
- ✅ Successful evaluations
- Task and model information
- Number of samples evaluated
- Success/failure status

### 7. Export Results

Click **"Export"** button to save results as:
- CSV format for spreadsheets
- JSON format for programmatic access

## Task Formats Supported

### 1. Eleuther AI Format (YAML)
```yaml
task: my_task
dataset_name: dataset/name
task_type: question_answer
metric: exact_match
```

### 2. Custom JSON Format
```json
{
  "name": "My Task",
  "task_type": "question_answer",
  "samples": [
    {
      "input": "What is 2+2?",
      "expected": "4"
    }
  ]
}
```

### 3. CSV Format
```csv
input,expected
"What is 2+2?","4"
"Capital of France?","Paris"
```

## Supported Task Types

1. **Question-Answer**: General Q&A tasks
2. **Classification**: Multiple choice questions
3. **Generation**: Open-ended text generation
4. **Code Generation**: Programming tasks (with execution)

## Supported Providers

- OpenAI (GPT-3.5, GPT-4)
- Anthropic (Claude)
- Cohere
- Groq
- OpenRouter
- Local models (Ollama, llama.cpp, etc.)

## Troubleshooting

### "No API Key" Error
Add your API key to `~/.config/tldw_cli/config.toml`:
```toml
[api_settings.openai]
api_key = "sk-..."
```

### "Task Load Failed" Error
Ensure your task file follows one of the supported formats above.

### Results Not Showing
Click the **"Refresh"** button in the Results view.

## Advanced Features

- **Few-shot prompting**: Include examples in your task configuration
- **Custom metrics**: Define evaluation criteria
- **Batch processing**: Evaluate multiple models/tasks
- **Progress tracking**: Real-time updates during evaluation

## Integration Test

Run the included test to verify your setup:
```bash
python3 test_eval_integration.py
```

This will confirm all components are working correctly.

## Next Steps

1. Explore different task types in `/sample_evaluation_tasks/`
2. Create custom evaluation tasks for your use cases
3. Compare performance across different models
4. Export and analyze results

The evaluation system is ready for production use. Happy benchmarking! 🎉