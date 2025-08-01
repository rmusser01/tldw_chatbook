# OpenTelemetry Metrics Implementation

## Overview

This document provides a comprehensive technical reference for the OpenTelemetry (OTel) metrics implementation in the TLDW Chatbook application. The metrics system provides observability into application performance, user behavior, and system health.

## Architecture

### Core Components

1. **Metrics Logger** (`tldw_chatbook/Metrics/metrics_logger.py`)
   - Central abstraction layer for all metrics operations
   - Provides `log_counter()` and `log_histogram()` functions
   - Handles graceful degradation when prometheus_client is not available
   - Manages metric naming conventions and label validation

2. **OpenTelemetry Integration** (`tldw_chatbook/Metrics/Otel_Metrics.py`)
   - Configures OpenTelemetry SDK
   - Sets up Prometheus exporter
   - Manages metric providers and exporters
   - Handles environment-based configuration

3. **Prometheus Server** (Optional)
   - Runs on port 9090 by default
   - Scrapes metrics from the application
   - Provides query interface for metric analysis

### Design Principles

1. **Graceful Degradation**: The application continues to function normally if metrics libraries are missing or misconfigured
2. **Minimal Performance Impact**: Metrics collection is designed to have negligible impact on application performance
3. **Consistent Naming**: All metrics follow a consistent naming convention with meaningful prefixes
4. **Rich Labels**: Metrics include contextual labels for detailed analysis and filtering
5. **Separation of Concerns**: Metrics logic is separated from business logic

## Implementation Details

### Metric Types

#### Counters
- Monotonically increasing values
- Used for counting events (e.g., API calls, errors, user actions)
- Example: `chat_message_sent`, `eval_run_started`

#### Histograms
- Distribution of values
- Used for measuring durations, sizes, or other numeric values
- Automatically provides count, sum, and bucket information
- Example: `chat_response_duration`, `eval_sample_processing_duration`

### Naming Conventions

```
<module>_<component>_<metric>
```

- **module**: Top-level module (e.g., `chat`, `eval`, `media`, `notes`)
- **component**: Specific component or feature
- **metric**: What is being measured

Examples:
- `chat_message_sent` - Chat module, message component, sent action
- `eval_run_duration` - Eval module, run component, duration measurement
- `media_ingestion_file_size` - Media module, ingestion component, file size measurement

### Label Guidelines

1. **Consistency**: Use the same label names across related metrics
2. **Cardinality**: Avoid high-cardinality labels (e.g., user IDs, timestamps)
3. **Meaningful Values**: Label values should be categorical and meaningful
4. **Common Labels**:
   - `status`: success, error, cancelled
   - `provider`: openai, anthropic, groq, etc.
   - `error_type`: Specific error class name
   - `operation`: Type of operation being performed
   - `source`: Origin of the request or data

### Error Handling

```python
try:
    # Metric logging attempt
    log_counter("operation_success", labels={"type": "example"})
except Exception:
    # Silently fail - metrics should never break the application
    pass
```

## Module-Specific Implementations

### Chat Module Metrics

Location: Various files in `tldw_chatbook/Chat/` and `tldw_chatbook/LLM_Calls/`

Key Metrics:
- `chat_message_sent` - User messages sent
- `chat_response_received` - LLM responses received
- `chat_response_duration` - Response generation time
- `chat_token_usage` - Token consumption
- `chat_streaming_chunk` - Streaming response chunks
- `chat_error` - Chat-related errors
- `document_generator_request` - Document generation attempts (timeline, study guide, briefing)
- `document_generator_duration` - Document generation time
- `document_generator_token_usage` - Tokens used for document generation

Implementation Notes:
- Streaming responses track both chunks and completion
- Token usage is tracked per provider and model
- Response duration includes full end-to-end time
- Document generation tracks type, provider, model, and streaming mode

### Evaluation Module Metrics

Location: `tldw_chatbook/Evals/` directory

Key Metrics:
- `eval_run_*` - Evaluation run lifecycle
- `eval_sample_*` - Individual sample processing
- `eval_llm_*` - LLM API calls during evaluation
- `eval_cost_*` - Cost tracking and estimation
- `eval_specialized_runner_*` - Domain-specific evaluation metrics

Implementation Notes:
- Comprehensive tracking of evaluation lifecycle
- Specialized runners have domain-specific metrics
- Cost tracking integrated with evaluation flow
- Database operations tracked separately

### Media Module Metrics

Location: `tldw_chatbook/Local_Ingestion/` and related files

Key Metrics:
- `media_ingestion_*` - Media file ingestion
- `media_processing_*` - Processing operations
- `media_search_*` - Media search operations
- `media_transcription_*` - Transcription operations
- `xml_ingestion_parse_duration` - XML parsing time
- `xml_ingestion_element_count` - Number of XML elements
- `local_file_ingestion_attempt` - File ingestion attempts with type
- `video_processing_download_duration` - Video download time
- `video_processing_file_size_bytes` - Size of processed videos

Implementation Notes:
- File size and processing duration tracked
- Different metrics for different media types
- Chunking operations tracked for large files
- Batch processing metrics for multiple files

### Notes Module Metrics

Location: `tldw_chatbook/Notes/` directory

Key Metrics:
- `notes_created` - Note creation
- `notes_sync_*` - Synchronization operations
- `notes_template_*` - Template usage
- `notes_search_*` - Note search operations
- `notes_library_add_note_duration` - Time to add a note
- `notes_library_note_content_length` - Length of note content
- `sync_engine_scan_directory_duration` - Directory scanning time
- `sync_engine_sync_duration` - Full sync operation time
- `sync_engine_sync_conflicts` - Number of sync conflicts
- `sync_engine_sync_errors` - Number of sync errors

Implementation Notes:
- Sync operations track conflicts and resolutions
- Template usage helps identify popular templates
- Search performance tracked for optimization
- Bidirectional sync tracks created/updated files and notes
- Version conflict detection and resolution tracked

### RAG (Retrieval-Augmented Generation) Metrics

Location: `tldw_chatbook/RAG_Search/` directory

Key Metrics:
- `rag_search_*` - Search operations
- `rag_embedding_*` - Embedding generation
- `rag_retrieval_*` - Document retrieval
- `rag_rerank_*` - Result reranking

Implementation Notes:
- Tracks both keyword and semantic search
- Embedding generation performance monitored
- Retrieval accuracy metrics included

## Configuration

### Environment Variables

```bash
# Enable/disable metrics collection
OTEL_METRICS_ENABLED=true

# Prometheus exporter configuration
OTEL_EXPORTER_PROMETHEUS_HOST=localhost
OTEL_EXPORTER_PROMETHEUS_PORT=9090

# Service identification
OTEL_SERVICE_NAME=tldw-chatbook
OTEL_SERVICE_VERSION=1.0.0
```

### Configuration File

In `config.toml`:

```toml
[telemetry]
enabled = true
export_interval = 60  # seconds
prometheus_port = 9090

[telemetry.labels]
# Global labels added to all metrics
environment = "production"
region = "us-east-1"
```

### Conditional Imports

The application uses conditional imports to handle optional dependencies:

```python
try:
    from opentelemetry import metrics
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
```

## Debugging Guide

### Common Issues

1. **Metrics Not Appearing**
   - Check if prometheus_client is installed: `pip install prometheus_client`
   - Verify metrics are enabled in configuration
   - Check for import errors in logs
   - Ensure Prometheus server is running and accessible

2. **High Memory Usage**
   - Review label cardinality - too many unique label combinations
   - Check histogram bucket configurations
   - Monitor metric retention settings

3. **Performance Impact**
   - Use sampling for high-frequency metrics
   - Batch metric updates where possible
   - Review histogram bucket counts

### Debug Commands

```python
# Check if metrics are available
from tldw_chatbook.Metrics.metrics_logger import METRICS_AVAILABLE
print(f"Metrics available: {METRICS_AVAILABLE}")

# List all registered metrics
from prometheus_client import REGISTRY
for metric in REGISTRY.collect():
    print(f"{metric.name}: {metric.type}")

# Get current metric values
from prometheus_client import generate_latest
print(generate_latest().decode('utf-8'))
```

### Metric Validation

```python
# Validate metric naming
import re
def validate_metric_name(name):
    pattern = r'^[a-zA-Z_:][a-zA-Z0-9_:]*$'
    return bool(re.match(pattern, name))

# Check label cardinality
def estimate_cardinality(labels):
    cardinality = 1
    for label, values in labels.items():
        cardinality *= len(values)
    return cardinality
```

## Query Examples

### Prometheus Queries

```promql
# Request rate by module
sum by (module) (rate(tldw_chatbook_requests_total[5m]))

# Error rate
rate(tldw_chatbook_errors_total[5m]) / rate(tldw_chatbook_requests_total[5m])

# P95 latency
histogram_quantile(0.95, rate(tldw_chatbook_duration_seconds_bucket[5m]))

# Active evaluations
eval_run_active_total

# Cost by provider
sum by (provider) (rate(eval_cost_sample_cost_sum[1h]))
```

### Grafana Dashboard Examples

1. **Application Overview**
   - Request rate
   - Error rate
   - Response time percentiles
   - Active users

2. **LLM Performance**
   - API call latency by provider
   - Token usage and costs
   - Error rates by provider
   - Rate limit occurrences

3. **Evaluation Metrics**
   - Active evaluations
   - Sample processing rate
   - Cost tracking
   - Success/failure rates

## Best Practices

### When to Add Metrics

1. **User Actions**: Track significant user interactions
2. **External API Calls**: Monitor latency and errors
3. **Resource Usage**: Track memory, CPU, disk operations
4. **Business Events**: Monitor key business metrics
5. **Error Conditions**: Track all error types and frequencies

### When NOT to Add Metrics

1. **Sensitive Data**: Never include PII or sensitive data in metrics
2. **High-Frequency Loops**: Avoid metrics in tight loops
3. **Temporary Debugging**: Use logs for temporary debugging
4. **Unbounded Labels**: Avoid labels with unlimited possible values

### Metric Lifecycle

1. **Development**: Add metric with descriptive name and labels
2. **Testing**: Verify metric appears correctly
3. **Documentation**: Update this document and relevant module docs
4. **Monitoring**: Create alerts for important metrics
5. **Deprecation**: Mark as deprecated before removal

## Integration with Other Systems

### Prometheus Integration

The application exposes metrics in Prometheus format on port 9090 by default:

```bash
# View raw metrics
curl http://localhost:9090/metrics

# Configure Prometheus scraping
# prometheus.yml
scrape_configs:
  - job_name: 'tldw-chatbook'
    static_configs:
      - targets: ['localhost:9090']
```

### Grafana Integration

Import dashboard JSON from `Docs/Monitoring/grafana-dashboards/`:
- `overview-dashboard.json` - Application overview
- `llm-performance-dashboard.json` - LLM metrics
- `evaluation-dashboard.json` - Evaluation system metrics

### AlertManager Integration

Example alert rules in `Docs/Monitoring/alerts/`:
- High error rate
- Slow response times
- Cost thresholds
- Resource usage

## Future Enhancements

1. **Tracing Integration**: Add OpenTelemetry tracing for request flow
2. **Custom Metrics API**: Allow users to define custom metrics
3. **Metric Aggregation**: Pre-aggregate common queries
4. **Adaptive Sampling**: Dynamic sampling based on load
5. **Metric Export**: Export metrics to various backends (CloudWatch, Datadog, etc.)

## Migration Guide

### From Direct Prometheus to OpenTelemetry

```python
# Old approach
from prometheus_client import Counter
my_counter = Counter('my_metric', 'Description')
my_counter.inc()

# New approach
from tldw_chatbook.Metrics.metrics_logger import log_counter
log_counter('my_metric', labels={'key': 'value'})
```

### Adding Metrics to New Modules

1. Import the metrics logger:
   ```python
   from tldw_chatbook.Metrics.metrics_logger import log_counter, log_histogram
   ```

2. Add metrics at key points:
   ```python
   # Count events
   log_counter('module_event_occurred', labels={'type': 'example'})
   
   # Measure duration
   start_time = time.time()
   # ... operation ...
   duration = time.time() - start_time
   log_histogram('module_operation_duration', duration, labels={'op': 'example'})
   ```

3. Follow naming conventions and label guidelines

4. Test metrics collection

5. Update documentation

## References

- [OpenTelemetry Python Documentation](https://opentelemetry-python.readthedocs.io/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Metric Naming Conventions](https://prometheus.io/docs/practices/naming/)

### LLM Provider Metrics

Location: `tldw_chatbook/LLM_Calls/` directory

#### Commercial Providers
- `anthropic_api_request` - Anthropic API requests
- `cohere_api_request` - Cohere API requests  
- `deepseek_api_request` - DeepSeek API requests
- `google_api_request` - Google API requests
- `groq_api_request` - Groq API requests
- `huggingface_api_request` - HuggingFace API requests
- `mistral_api_request` - Mistral API requests
- `openrouter_api_request` - OpenRouter API requests

Each provider tracks:
- Request count by model and streaming mode
- Response time histograms
- Token usage (input/output)
- Error rates and types

#### Local Providers
- `local_openai_compatible_api_request` - Shared for most local providers
- `kobold_api_request` - Kobold-specific requests

Local providers tracked:
- Local-llm, Llama.cpp, Ollama, vLLM, Aphrodite
- TabbyAPI, Oobabooga, Custom OpenAI endpoints
- MLX-LM support

### Database Module Metrics

Location: `tldw_chatbook/DB/` directory

#### ChaChaNotes_DB
- `chachanotes_db_query_duration` - Query execution time
- `chachanotes_db_character_card_operation_duration` - Character CRUD operations
- `chachanotes_db_conversation_operation_duration` - Conversation operations

#### Client_Media_DB_v2
- `client_media_db_media_operation_count` - Media operations with path tracking
- `client_media_db_operation_duration` - Operation timing
- Path tracking: create_new, update_content, update_metadata_only, etc.

#### RAG_Indexing_DB
- `rag_indexing_db_operation_duration` - RAG indexing operations
- `rag_indexing_db_collection_completion_rate` - Collection indexing progress

#### Other Databases
- `search_history_db_*` - Search history tracking
- `prompts_db_*` - Prompt management operations
- `subscriptions_db_*` - Subscription management

### Web Scraping Module Metrics

Location: `tldw_chatbook/Web_Scraping/` directory

- `websearch_generate_search_duration` - Search generation time
- `websearch_perform_search_duration` - Search execution time
- `article_scraper_*` - Article scraping operations
- `crawler_*` - Site crawling metrics
- `processor_*` - Content processing pipelines
- `importer_*` - Bookmark import operations
- `cookie_cloner_*` - Cookie extraction metrics

### Chunking Module Metrics

Location: `tldw_chatbook/Chunking/` directory

- `chunking_text_attempt` - Chunking attempts by method
- `chunking_duration` - Time to chunk text
- `chunking_chunk_count` - Number of chunks created
- `chunking_language_detection_*` - Language detection operations

### Utils Module Metrics

Location: `tldw_chatbook/Utils/` directory

- `path_validation_*` - Path security validation
- `input_validation_*` - User input validation for various types
- `secure_temp_files_*` - Secure temporary file operations
- `config_encryption_*` - Configuration encryption operations
- `custom_tokenizers_*` - Custom tokenizer loading and usage
- `terminal_utils_*` - Terminal capability detection

### Local Inference Module Metrics

Location: `tldw_chatbook/Local_Inference/` directory

- `mlx_lm_server_*` - MLX-LM server lifecycle
- `ollama_api_*` - Ollama API operations
- `ollama_pull_model_*` - Model download operations
- `ollama_delete_model_*` - Model deletion
- `ollama_running_models_count` - Active model count
- `ollama_running_model_memory_bytes` - Memory usage per model

## Appendix: Metric Inventory

See `EVAL_METRICS_SUMMARY.md` for a complete list of evaluation metrics.
See individual module documentation for module-specific metrics.

For a real-time inventory of all metrics:
```bash
curl -s http://localhost:9090/metrics | grep -E "^[a-zA-Z]" | cut -d'{' -f1 | sort | uniq
```

### Summary of Implementation Progress

Completed metric instrumentation across all major modules:
- ✅ Phase 1: LLM Calls (Commercial & Local) and Database modules
- ✅ Phase 2: Chat and Web Scraping modules  
- ✅ Phase 3: Local Ingestion and Chunking modules
- ✅ Phase 4: Notes, Utils, and Local Inference modules

Total metrics added: 200+ across 40+ files