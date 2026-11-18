"""Keep the real admitted private config for exact native adapter owner tests."""
from pathlib import Path
import pytest

NATIVE_OWNERS = {
    "test_anthropic_native_tools.py",
    "test_anthropic_prefix_stability.py",
    "test_anthropic_streaming_usage.py",
}


def pytest_collection_modifyitems(items):
    for item in items:
        if Path(str(item.path)).name in NATIVE_OWNERS:
            item.add_marker(pytest.mark.bootstrap_profile)
