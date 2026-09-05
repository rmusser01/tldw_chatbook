"""Real-Chromium qualification for the measured Canvas quota defaults."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from tldw_chatbook.Canvas.limits import CanvasLimits

ROOT = Path(__file__).resolve().parents[3]
PROBE_PATH = ROOT / "scripts" / "canvas_runtime_quota_probe.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "canvas_runtime_quota_browser_probe", PROBE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.loopback_network
def test_real_chromium_probe_enforces_final_runtime_boundaries() -> None:
    probe = _load_probe()

    result = probe.run_browser_probe(CanvasLimits(), samples=1)
    cases = {item["fixture_id"]: item for item in result["runtime_cases"]}

    assert cases["representative-interactive"]["state"] == "ready"
    assert cases["adversarial-combined-at-limit"]["state"] == "ready"
    assert cases["adversarial-startup-timeout"]["code"] == "runtime-timeout"
    assert cases["adversarial-event-timeout"]["code"] == "runtime-timeout"
    assert cases["adversarial-stack-pressure"]["state"] == "failed"
    assert cases["adversarial-heap-pressure"]["state"] == "failed"
    assert cases["adversarial-patches-at-limit"]["state"] == "ready"
    assert cases["adversarial-patches-over-limit"]["code"] == "patch-limit"
    assert cases["adversarial-patches-at-limit"]["patches"] == 500
    assert cases["adversarial-patches-over-limit"]["patches"] == 501

    memory = result["process_tree_rss"]
    assert set(memory) == {
        "warmed_blank_mib",
        "trusted_runtime_mib",
        "representative_mib",
        "near_limit_mib",
    }
    assert all(isinstance(value, float) and value > 0 for value in memory.values())
    resources = result["quickjs_resources"]
    assert resources["heap_limit_bytes"] == 32 * 1024 * 1024
    assert resources["accepted_allocation_bytes"] == 16 * 1024 * 1024
    assert resources["oversized_allocation_bytes"] == 32 * 1024 * 1024
    assert resources["oversized_allocation_outcome"] == "rejected"
    assert (
        resources["accepted_memory_used_bytes"]
        > resources["baseline_memory_used_bytes"]
    )
    assert resources["stack_limit_bytes"] == 512 * 1024
    assert resources["recursion_outcome"] == "rejected"
    assert resources["recursion_depth"] > 0
    assert result["browser_engine"] == "chromium"
    assert result["samples"] == 1
