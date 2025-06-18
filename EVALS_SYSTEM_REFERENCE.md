# EVALS System Reference Document

**Date Created**: 2025-06-18  
**Project**: tldw_chatbook  
**Feature**: LLM Evaluation Framework  

## Overview

This document provides a comprehensive reference for the newly implemented LLM evaluation system in tldw_chatbook. The system enables users to upload evaluation tasks in multiple formats (particularly Eleuther AI's format), run them against various LLM APIs, and analyze the results.

## System Architecture

### Core Components

The evaluation system consists of several key components organized in a modular architecture:

1. **Database Layer** (`tldw_chatbook/DB/Evals_DB.py`)
2. **Task Management** (`tldw_chatbook/App_Functions/Evals/task_loader.py`)
3. **Evaluation Execution** (`tldw_chatbook/App_Functions/Evals/eval_runner.py`)
4. **LLM Interface** (`tldw_chatbook/App_Functions/Evals/llm_interface.py`)
5. **Orchestration** (`tldw_chatbook/App_Functions/Evals/eval_orchestrator.py`)
6. **UI Components** (`tldw_chatbook/UI/Evals_Window.py`)

## New Files Created

### Database Layer
- **`tldw_chatbook/DB/Evals_DB.py`** - Complete SQLite-based database for evaluation management
  - Tables: `eval_tasks`, `eval_datasets`, `eval_models`, `eval_runs`, `eval_results`, `eval_run_metrics`
  - Full-text search capabilities with FTS5
  - Thread-safe connections with WAL mode
  - Versioning and audit trail support

### Core Evaluation Framework
- **`tldw_chatbook/App_Functions/Evals/__init__.py`** - Package initialization
- **`tldw_chatbook/App_Functions/Evals/task_loader.py`** - Multi-format task loading system
- **`tldw_chatbook/App_Functions/Evals/eval_runner.py`** - Evaluation execution engine
- **`tldw_chatbook/App_Functions/Evals/llm_interface.py`** - Unified LLM provider interface
- **`tldw_chatbook/App_Functions/Evals/eval_orchestrator.py`** - High-level orchestration layer
- **`tldw_chatbook/App_Functions/Evals/eval_templates.py`** - Comprehensive evaluation template library
- **`tldw_chatbook/App_Functions/Evals/specialized_runners.py`** - Specialized evaluation runners for advanced tasks

## Key Features Implemented

### 1. Multi-Format Task Support

The system supports multiple evaluation task formats:

#### Eleuther AI Format
- **Full compatibility** with Eleuther AI evaluation harness YAML format
- Support for MMLU-Pro-Plus and similar benchmarks
- Automatic task type detection (logprob, generation, multiple_choice)
- Template system support (`doc_to_text`, `doc_to_target`, `doc_to_choice`)

#### Custom JSON Format
- Simplified JSON configuration format
- Direct specification of task parameters
- Easier to create and modify than Eleuther format

#### HuggingFace Integration
- Direct loading from HuggingFace datasets
- Automatic task type inference from dataset structure
- Support for dataset configurations and splits

#### CSV/TSV Support
- Local file support for custom datasets
- Automatic column mapping detection
- Support for both delimited formats

### 2. Comprehensive Evaluation Task Types

The system now supports **27+ specialized evaluation types** across **7 major categories**:

#### 🧠 Reasoning & Mathematical Evaluations
- **GSM8K Math Problems**: Grade school math word problems requiring multi-step reasoning
- **Logical Reasoning**: Syllogisms, deduction, and formal reasoning tasks
- **Arithmetic Reasoning**: Multi-step arithmetic problems with reasoning components
- **Chain of Thought**: Step-by-step reasoning evaluation with process assessment
- **Analogy Reasoning**: Pattern recognition and analogical reasoning tasks
- **Math Word Problems**: Custom mathematical problems of varying difficulty

#### 🛡️ Safety & Alignment Evaluations  
- **Harmfulness Detection**: Identify and refuse harmful requests appropriately
- **Bias Evaluation**: Test for demographic, gender, racial, and social biases
- **Truthfulness QA**: Evaluate factual accuracy and resistance to misinformation
- **Jailbreak Resistance**: Test resistance to prompt injection and safety bypasses
- **Privacy Leakage Detection**: Identify potential privacy violations and data leakage
- **Ethical Reasoning**: Evaluate ethical reasoning and moral judgment capabilities

#### 💻 Code Generation & Programming
- **HumanEval Coding**: Python function implementation with execution testing
- **Code Completion**: Complete partially written code snippets
- **Bug Detection**: Identify bugs and issues in code snippets
- **Algorithm Implementation**: Implement standard algorithms and data structures
- **Code Explanation**: Explain what code snippets do and how they work
- **SQL Generation**: Generate SQL queries from natural language descriptions

#### 🌍 Multilingual & Translation
- **Translation Quality**: Evaluate translation accuracy across language pairs
- **Cross-lingual QA**: Question answering in different languages
- **Multilingual Sentiment**: Sentiment analysis across multiple languages
- **Code Switching**: Handle mixed-language inputs and responses

#### 🎓 Domain-Specific Knowledge
- **Medical QA**: Medical knowledge and reasoning evaluation
- **Legal Reasoning**: Legal concepts, case analysis, and jurisprudence
- **Scientific Reasoning**: Scientific knowledge and methodology evaluation
- **Financial Analysis**: Financial concepts and market analysis
- **Historical Knowledge**: Historical facts, timelines, and causation

#### 🎯 Robustness & Adversarial Testing
- **Adversarial QA**: Challenging questions designed to test robustness
- **Input Perturbation**: Response consistency under input variations
- **Context Length Stress**: Performance with very long contexts
- **Instruction Following**: Adherence to complex, multi-step instructions
- **Format Robustness**: Consistent performance across different input formats

#### 🎨 Creative & Open-ended Tasks
- **Creative Writing**: Original story and content generation
- **Story Completion**: Continue and complete narrative pieces
- **Dialogue Generation**: Generate realistic conversations and interactions
- **Summarization Quality**: Extract key information and create summaries
- **Open-ended QA**: Handle questions without definitive answers

### 3. Specialized Evaluation Capabilities

#### 🔧 Code Execution & Testing
- **Real Python Execution**: Code is actually executed in sandboxed environment
- **Test Case Validation**: Automated test running with pass/fail metrics
- **Syntax Checking**: AST parsing for syntax validation
- **Performance Metrics**: Execution time and efficiency measurement
- **Error Analysis**: Detailed error reporting and debugging information
- **Security**: Timeout protection and safe execution environment

#### 🛡️ Advanced Safety Analysis
- **Keyword-based Detection**: Multi-category harmful content identification
- **Pattern Recognition**: Regex-based detection of sensitive information (emails, phones, SSNs)
- **Refusal Assessment**: Evaluation of appropriate response refusal
- **Bias Quantification**: Systematic bias measurement across demographics
- **Privacy Protection**: Detection of potential personal information leakage
- **Ethical Reasoning**: Complex moral scenario evaluation

#### 🌐 Multilingual Assessment
- **Language Detection**: Automatic identification of response languages
- **Script Analysis**: Support for Latin, Chinese, Japanese, Arabic scripts
- **Fluency Metrics**: Word count, sentence structure, punctuation analysis
- **Cross-lingual Consistency**: Response quality across language boundaries
- **Translation Evaluation**: BLEU-like scoring for translation tasks

#### 🎨 Creative Content Analysis
- **Vocabulary Diversity**: Unique word ratio and lexical richness
- **Narrative Structure**: Story elements, dialogue detection, narrative flow
- **Coherence Metrics**: Sentence and paragraph structure analysis
- **Creativity Indicators**: Descriptive language, emotional content, originality markers
- **Quality Assessment**: Multi-dimensional scoring for creative output

### 4. LLM Provider Integration

Unified interface supporting:
- **OpenAI** (GPT models)
- **Anthropic** (Claude models)  
- **Cohere** (Command models)
- **Groq** (Fast inference)
- **OpenRouter** (Multi-provider access)

Each provider adapter handles:
- Async operations for performance
- API key management from config
- Error handling and retries
- Format conversion for existing API functions

### 4. Results Management

Comprehensive results storage and analysis:
- Individual sample results with full metadata
- Aggregated run-level metrics
- Performance tracking (latency, success rates)
- Export capabilities (JSON, CSV)
- Run comparison functionality

### 5. User Interface Enhancements

The existing Evals tab has been significantly enhanced:

#### Evaluation Setup View
- Task file upload functionality
- Model configuration management
- Run parameter configuration
- Status monitoring

#### Results Dashboard View
- Recent evaluations display
- Run comparison tools
- Export functionality
- Metrics visualization

#### Model Management View
- Provider-specific setup
- Model configuration templates
- Quick setup for major providers
- Configuration validation

#### Dataset Management View
- Dataset upload and management
- HuggingFace integration
- Template creation
- Dataset validation

## Database Schema

### Core Tables

1. **`eval_tasks`** - Task definitions and configurations
2. **`eval_datasets`** - Dataset metadata and source information
3. **`eval_models`** - LLM model configurations
4. **`eval_runs`** - Evaluation run tracking
5. **`eval_results`** - Individual sample results
6. **`eval_run_metrics`** - Aggregated run metrics

### Key Features
- **FTS5 Search** on tasks and datasets
- **Optimistic Locking** with version fields
- **Soft Deletion** with `deleted_at` timestamps
- **Audit Trail** with client_id tracking
- **Foreign Key Constraints** for data integrity

## Configuration Integration

The system integrates with the existing tldw_chatbook configuration:

### Database Location
- Default: `~/.local/share/tldw_cli/evals.db`
- Configurable via settings
- Automatic directory creation

### API Key Management
- Uses existing config.toml structure
- Environment variable fallback
- Per-provider key storage

### User Data Directory
- Follows existing patterns
- Export functionality uses configured paths
- Results storage in user space

## Task Configuration Examples

### Eleuther AI Format Example
```yaml
task: mmlu_pro_plus_biology
dataset_name: saeidasgari/mmlu-pro-plus
dataset_config_name: biology
test_split: test
fewshot_split: validation
num_fewshot: 5
output_type: generate_until
until: ["</s>", "Q:"]
generation_kwargs:
  temperature: 0.0
  max_tokens: 32
metric_list: ["exact_match"]
filter_list:
  - filter: "regex"
    regex_pattern: "Answer: ([A-Z])"
    group: 1
```

### Custom JSON Format Example
```json
{
  "name": "Custom Q&A Task",
  "description": "Simple question answering evaluation",
  "task_type": "question_answer",
  "dataset_name": "local_qa_dataset.json",
  "split": "test",
  "metric": "exact_match",
  "generation_kwargs": {
    "temperature": 0.0,
    "max_tokens": 50
  }
}
```

## Usage Examples

### Basic Evaluation
```python
from tldw_chatbook.App_Functions.Evals.eval_orchestrator import quick_eval

# Run a quick evaluation
result = await quick_eval(
    task_file="mmlu_sample.yaml",
    provider="openai",
    model="gpt-3.5-turbo",
    max_samples=100,
    output_dir="./results"
)
```

### Full Orchestration
```python
from tldw_chatbook.App_Functions.Evals.eval_orchestrator import EvaluationOrchestrator

orchestrator = EvaluationOrchestrator()

# Create task from file
task_id = await orchestrator.create_task_from_file("task.yaml", "eleuther")

# Create model configuration
model_id = orchestrator.create_model_config(
    name="GPT-4",
    provider="openai", 
    model_id="gpt-4"
)

# Run evaluation
run_id = await orchestrator.run_evaluation(
    task_id=task_id,
    model_id=model_id,
    max_samples=500
)

# Get results
summary = orchestrator.get_run_summary(run_id)
```

## Performance Considerations

### Async Operations
- All LLM calls are asynchronous
- Concurrent sample processing capability
- Non-blocking UI operations

### Database Optimization
- Indexed queries for performance
- WAL mode for concurrent access
- Prepared statements for safety

### Memory Management
- Streaming results processing
- Configurable batch sizes
- Lazy loading of large datasets

## Future Enhancement Opportunities

### Immediate Next Steps
1. **File Upload UI** - Implement actual file dialogs in the UI
2. **Real-time Progress** - Add progress bars for long-running evaluations
3. **Result Visualization** - Charts and graphs for metrics
4. **Model Comparison** - Side-by-side comparison interfaces

### Advanced Features
1. **Distributed Evaluation** - Multi-worker support for large datasets
2. **Custom Metrics** - User-defined evaluation metrics
3. **Benchmark Suites** - Pre-configured evaluation suites
4. **A/B Testing** - Systematic model comparison workflows

### Integration Enhancements
1. **Local Model Support** - Integration with local inference engines
2. **Cost Tracking** - API usage and cost monitoring
3. **Scheduled Evaluations** - Automated evaluation runs
4. **Notification System** - Completion alerts and summaries

## Technical Design Decisions

### Database Choice
- **SQLite** chosen for simplicity and portability
- Single-file database fits existing architecture
- FTS5 provides powerful search without external dependencies

### Async Architecture
- Enables responsive UI during long evaluations
- Supports concurrent API calls for performance
- Future-ready for distributed processing

### Provider Abstraction
- Unified interface enables easy provider switching
- Standardized configuration across providers
- Extensible for future provider additions

### Modular Design
- Clear separation of concerns
- Independent testing of components
- Easy maintenance and extension

## Error Handling Strategy

### Graceful Degradation
- Individual sample failures don't stop evaluation
- Error tracking in results database
- Comprehensive logging for debugging

### User Feedback
- Clear error messages in UI
- Status updates during processing
- Recovery suggestions when possible

### Data Integrity
- Transaction-based operations
- Foreign key constraints
- Validation at multiple layers

## Security Considerations

### API Key Management
- No hardcoded keys in any files
- Environment variable support
- Config file encryption ready

### Input Validation
- SQL injection prevention
- File path validation
- Configuration parameter sanitization

### Access Control
- Client ID tracking for audit
- User data directory isolation
- No cross-user data access

## Testing Strategy

### Unit Tests
- Database operations
- Task loading and validation
- Metric calculations
- Provider interfaces

### Integration Tests
- End-to-end evaluation flows
- Multi-provider testing
- Database migration testing
- UI component testing

### Performance Tests
- Large dataset handling
- Concurrent evaluation runs
- Memory usage validation
- API rate limit handling

## Documentation Requirements

### User Documentation
- Getting started guide
- Task format specifications
- Provider setup instructions
- Troubleshooting guide

### Developer Documentation
- API reference
- Extension guidelines
- Database schema documentation
- Architecture overview

This evaluation system represents a significant enhancement to tldw_chatbook, providing comprehensive LLM evaluation capabilities that support both research and production use cases while maintaining the application's focus on usability and reliability.