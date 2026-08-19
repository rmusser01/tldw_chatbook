import pytest

# --- Ported (chunking-engine-parity Task 4) ---------------------------------
# Upstream file: tldw_Server_API/tests/Chunking/test_chunker_process_metrics.py
# Skipped: server Metrics registry not vendored; engine degrades gracefully to no-op metrics. Remove this block when the module is vendored in
# its own sub-project and re-sync the test from upstream.
pytest.importorskip("tldw_chatbook.NoSuchDeferredModule",
                    reason="skipped: server Metrics registry not vendored; engine degrades gracefully to no-op metrics")

from tldw_chatbook.Chunking.engine.chunker import Chunker
from tldw_chatbook.Chunking._shims.Metrics import get_metrics_registry


def test_chunker_process_metrics_registered_and_recorded():
    registry = get_metrics_registry()
    metric_names = [
        "chunker_process_total",
        "chunker_frontmatter_duration_seconds",
        "chunker_header_extract_seconds",
        "chunker_chunking_duration_seconds",
        "chunker_normalization_seconds",
        "chunker_last_chunk_count",
        "chunker_output_bytes",
        "chunker_input_bytes",
        "chunker_process_total_seconds",
    ]

    for name in metric_names:
        assert name in registry.metrics
        registry.values[name].clear()

    chunker = Chunker()
    chunker.process_text("One sentence for metrics.")

    for name in metric_names:
        assert registry.values[name], f"Expected metric samples for {name}"
