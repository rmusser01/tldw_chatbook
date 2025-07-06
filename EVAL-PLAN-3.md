# Evaluation System Completion Plan

## Overview
The evaluation system has a solid foundation with database, UI, and architecture in place. This plan outlines the remaining implementation work needed to make it fully functional.

## Current State Analysis

### ✅ Already Implemented:
1. **Database Layer** (Evals_DB.py) - Complete with tables, CRUD operations, FTS5 search
2. **Task Loading** (task_loader.py) - Supports multiple formats (Eleuther AI, JSON, CSV, HuggingFace)
3. **Evaluation Templates** (eval_templates.py) - 30+ templates across various categories
4. **UI Components** - Complete windows, widgets, and dialogs
5. **Event Handlers** - Basic event routing and handling

### 🚧 Partially Implemented:
1. **LLM Interface** (llm_interface.py) - Base structure exists, missing provider implementations
2. **Evaluation Runner** (eval_runner.py) - Core loop exists, missing metrics
3. **Specialized Runners** (specialized_runners.py) - Structure exists, implementations incomplete

### ❌ Missing:
1. Most LLM provider integrations
2. Advanced metric calculations
3. Actual code execution for coding tasks
4. Logprob support
5. Streaming evaluation progress

## Implementation Plan

### Phase 1: LLM Provider Integration (Priority: Critical)

#### 1.1 Leverage Existing Providers
The codebase already has these providers in `LLM_Calls/LLM_API_Calls.py` and `LLM_API_Calls_Local.py`:

**Commercial Providers:**
- ✅ OpenAI (chat_with_openai)
- ✅ Anthropic (chat_with_anthropic)
- ✅ Cohere (chat_with_cohere)
- ✅ Groq (chat_with_groq)
- ✅ OpenRouter (chat_with_openrouter)
- ✅ DeepSeek (chat_with_deepseek)
- ✅ Mistral (chat_with_mistral)
- ✅ Google (chat_with_google)
- ✅ HuggingFace (chat_with_huggingface)

**Local Providers:**
- ✅ Ollama (chat_with_ollama)
- ✅ Llama.cpp (chat_with_llamacpp)
- ✅ Kobold.cpp (chat_with_koboldcpp)
- ✅ vLLM (chat_with_vllm)
- ✅ Aphrodite (chat_with_aphrodite)
- ✅ TabbyAPI (chat_with_tabbyapi)
- ✅ Custom OpenAI (chat_with_custom_openai)

#### 1.2 Integration Tasks
1. **Update llm_interface.py** to import and wrap existing providers:
   ```python
   # Import from existing LLM_API_Calls modules
   from tldw_chatbook.LLM_Calls.LLM_API_Calls import (
       chat_with_openai, chat_with_anthropic, chat_with_cohere, ...
   )
   ```

2. **Implement provider classes** that wrap existing functions:
   - Use consistent interface for all providers
   - Handle streaming vs non-streaming
   - Add logprob support where available
   - Implement proper error handling

3. **Add missing provider integrations**:
   - MLX (if not present)
   - ONNX (if not present)
   - Transformers (if not present)
   - Llamafile (if not present)

### Phase 2: Metric Implementation (Priority: High)

#### 2.1 Enhance MetricsCalculator in eval_runner.py

**Basic Metrics** (Already have some):
- ✅ Exact match
- ✅ Contains
- ✅ Regex match
- ✅ F1 score
- ✅ Simple BLEU

**Add Advanced Metrics**:
1. **ROUGE Scores**:
   - ROUGE-1 (unigram overlap)
   - ROUGE-2 (bigram overlap)
   - ROUGE-L (longest common subsequence)
   - Use rouge-score library if available

2. **Enhanced BLEU**:
   - Full n-gram support (1-4)
   - Smoothing for short texts
   - Corpus-level BLEU

3. **Semantic Metrics**:
   - Cosine similarity using embeddings
   - BERTScore if transformers available
   - Semantic textual similarity

4. **Code Metrics**:
   - Pass@k for code generation
   - Syntax validation
   - Test case execution

5. **Perplexity**:
   - Token-level perplexity
   - Requires logprob support

#### 2.2 Implement Logprob Calculation
- Update calculate_logprob() to use actual probabilities
- Add provider-specific logprob extraction
- Handle providers that don't support logprobs

### Phase 3: Specialized Runners (Priority: Medium)

#### 3.1 Complete Existing Runners

**CodeExecutionRunner**:
- Implement safe code execution using subprocess with timeout
- Add language-specific execution (Python, JavaScript, etc.)
- Capture stdout, stderr, and return codes
- Add sandboxing for security

**SafetyEvaluationRunner**:
- Implement comprehensive keyword lists
- Add context-aware safety detection
- Support for different safety categories

**MultilingualEvaluationRunner**:
- Integrate proper language detection (langdetect)
- Add language-specific metrics
- Support for translation evaluation

#### 3.2 Add New Runners

**MathReasoningRunner**:
- Parse mathematical expressions
- Validate numerical answers
- Support for different answer formats

**SummarizationRunner**:
- ROUGE-based evaluation
- Length constraints checking
- Key information retention

**DialogueRunner**:
- Multi-turn consistency
- Context maintenance
- Response relevance

### Phase 4: UI and Integration (Priority: Medium)

#### 4.1 Progress Visualization
- Real-time progress bars
- ETA calculation
- Live metric updates

#### 4.2 Error Handling UI
- Graceful error recovery
- Retry failed evaluations
- Skip problematic samples

#### 4.3 Advanced Features
- Batch evaluation setup
- Model comparison views
- Statistical significance testing

### Phase 5: Advanced Features (Priority: Low)

#### 5.1 Performance Optimization
- Implement caching for repeated evaluations
- Add batch processing for efficiency
- GPU memory management

#### 5.2 Analysis Tools
- Error categorization
- Performance regression detection
- Bias and fairness metrics

## Implementation Order

1. **Week 1**: LLM Provider Integration
   - Day 1-2: Update llm_interface.py to use existing providers
   - Day 3-4: Add logprob support where available
   - Day 5: Test all provider integrations

2. **Week 2**: Core Metrics
   - Day 1-2: Implement ROUGE scores
   - Day 3: Enhanced BLEU implementation
   - Day 4-5: Semantic similarity metrics

3. **Week 3**: Specialized Runners
   - Day 1-2: Complete CodeExecutionRunner
   - Day 3-4: Finish safety and multilingual runners
   - Day 5: Add new domain-specific runners

4. **Week 4**: Polish and Testing
   - Day 1-2: UI enhancements
   - Day 3-4: Integration testing
   - Day 5: Documentation and examples

## Testing Strategy

1. **Unit Tests**: For each metric and runner
2. **Integration Tests**: End-to-end evaluation flows
3. **Provider Tests**: Mock responses for each LLM
4. **Performance Tests**: Large-scale evaluation handling

## Success Criteria

1. All major LLM providers integrated and working
2. Standard NLP metrics (ROUGE, BLEU, etc.) implemented
3. Code evaluation with actual execution
4. UI shows real-time progress and results
5. System can handle large evaluation sets (1000+ samples)
6. Results are reproducible and accurate

## Dependencies to Check

- rouge-score (for ROUGE metrics)
- sacrebleu (for better BLEU)
- langdetect (for language detection)
- sentence-transformers (for semantic similarity)
- Any sandboxing libraries for code execution

## Notes

- Leverage existing code as much as possible
- Maintain backward compatibility with existing evaluations
- Focus on accuracy over performance initially
- Document all new metrics and their interpretations