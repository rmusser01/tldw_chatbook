# LLM API Metrics Analysis Report

## Overview
This report analyzes the LLM API call patterns in the tldw_chatbook codebase to identify areas that need metrics instrumentation for better observability.

## Current State of Metrics

### Commercial LLM Providers (LLM_API_Calls.py)

#### ✅ OpenAI - FULLY INSTRUMENTED
The OpenAI provider (`chat_with_openai`) is the only commercial provider with comprehensive metrics:
- **Request metrics**: `openai_api_request` counter with model and streaming labels
- **Response time**: `openai_api_response_time` histogram with model, streaming, and status_code labels
- **Success metrics**: `openai_api_success` counter
- **Token usage**: 
  - `openai_api_prompt_tokens` histogram
  - `openai_api_completion_tokens` histogram
  - `openai_api_total_tokens` histogram
- **Error tracking**: `openai_api_error` counter with error_type and status_code labels
- **Error response time**: `openai_api_error_response_time` histogram

#### ❌ Other Commercial Providers - NO METRICS
The following providers have NO metrics instrumentation:
- `chat_with_anthropic` - No metrics
- `chat_with_cohere` - No metrics
- `chat_with_deepseek` - No metrics
- `chat_with_google` - No metrics
- `chat_with_groq` - No metrics
- `chat_with_huggingface` - No metrics
- `chat_with_mistral` - No metrics
- `chat_with_openrouter` - No metrics
- `get_openai_embeddings` - No metrics (embeddings endpoint)

### Local LLM Providers (LLM_API_Calls_Local.py)

#### ❌ All Local Providers - NO METRICS
None of the local providers have any metrics instrumentation:
- `_chat_with_openai_compatible_local_server` - Generic handler, no metrics
- `chat_with_local_llm` - No metrics
- `chat_with_llama` - No metrics
- `chat_with_kobold` - No metrics
- `chat_with_oobabooga` - No metrics
- `chat_with_vllm` - No metrics
- `chat_with_tabbyapi` - No metrics
- `chat_with_aphrodite` - No metrics
- `chat_with_ollama` - No metrics
- `chat_with_mlx_lm` - No metrics
- `chat_with_custom_openai` - No metrics
- `chat_with_custom_openai_2` - No metrics

### Local Inference Operations
- `mlx_lm_inference_local.py` - Server start/stop operations, no metrics
- `ollama_model_mgmt.py` - Model management operations, no metrics

## Key Patterns to Instrument

### 1. API Request/Response Patterns
All providers follow similar patterns that should be instrumented:
- **Request initiation**: Log counter with provider, model, streaming mode
- **Response time**: Measure duration from request start to completion
- **Token usage**: Extract from provider-specific response formats
- **Error handling**: Track error types, status codes, and error response times

### 2. Provider-Specific Considerations

#### Anthropic
- Uses different response format than OpenAI
- Token usage in `usage` field of response
- Supports multimodal content (images)

#### Cohere
- Token usage in `meta.billed_units` (input_tokens, output_tokens)
- Has unique fields like `generation_id`
- Supports tool calls with different format

#### Google
- Different endpoint structure
- Token usage may be in different format
- Supports Gemini models with specific parameters

#### Local Providers
- May not always return token usage
- Network errors more common (connection refused)
- Model loading time should be tracked separately
- Memory usage could be relevant

### 3. Streaming Operations
- Both commercial and local providers support streaming
- Need to track:
  - Stream initiation
  - Chunk delivery metrics
  - Stream completion/interruption
  - Total streaming duration

### 4. Retry Logic
- Many providers have retry configuration
- Should track:
  - Retry attempts
  - Retry reasons (rate limit, server error, etc.)
  - Total time including retries

### 5. Special Operations
- **Embeddings**: Track dimension size, batch size, response time
- **Tool Calls**: Track tool usage frequency, success/failure
- **Multimodal**: Track image processing, additional latency

## Recommendations

### 1. Standardize Metrics Across All Providers
Create a common set of metrics for all providers:
```python
# Request metrics
{provider}_api_request_total
{provider}_api_response_time_seconds
{provider}_api_success_total
{provider}_api_error_total

# Token metrics
{provider}_api_prompt_tokens_total
{provider}_api_completion_tokens_total
{provider}_api_total_tokens_total

# Streaming metrics
{provider}_api_stream_duration_seconds
{provider}_api_stream_chunks_total
```

### 2. Add Provider-Agnostic Metrics
For high-level monitoring:
```python
llm_api_request_total{provider, model, streaming}
llm_api_response_time_seconds{provider, model, streaming}
llm_api_tokens_total{provider, model, type}
llm_api_error_total{provider, error_type}
```

### 3. Local Model Specific Metrics
```python
local_llm_model_load_time_seconds{provider, model}
local_llm_memory_usage_bytes{provider, model}
local_llm_inference_time_seconds{provider, model}
```

### 4. Cost Tracking Metrics
For commercial providers:
```python
llm_api_estimated_cost_dollars{provider, model, operation}
```

## Implementation Priority

1. **High Priority** - Commercial providers without metrics (Anthropic, Cohere, Google)
2. **Medium Priority** - Local providers (especially popular ones like Ollama, Llama.cpp)
3. **Low Priority** - Specialized metrics (embeddings, tool calls, multimodal)

## Technical Debt
- Inconsistent error handling across providers
- No unified response format transformation
- Token counting logic duplicated or missing
- Streaming implementation varies significantly