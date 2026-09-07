"""Real-Chromium qualification for the measured Canvas quota defaults."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
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


def _replace_once(body: bytes, old: bytes, new: bytes) -> bytes:
    assert body.count(old) == 1
    return body.replace(old, new, 1)


def _run_direct_probe_with_engine_html(
    probe, monkeypatch: pytest.MonkeyPatch, transform: Callable[[bytes], bytes]
) -> None:
    playwright_module = probe.importlib.import_module("playwright.sync_api")
    engine_html = transform(probe._engine_probe_html())
    monkeypatch.setattr(probe, "_engine_probe_html", lambda: engine_html)
    with (
        probe._serve_runtime_assets() as server,
        playwright_module.sync_playwright() as playwright,
    ):
        executable = probe._chromium_executable(playwright.chromium)
        browser = playwright.chromium.launch(headless=True, executable_path=executable)
        try:
            probe._run_trusted_quickjs_probe(browser, server.origin, CanvasLimits())
        finally:
            browser.close()


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
    assert resources["accepted_allocation_outcome"] == "accepted"
    assert resources["oversized_allocation_bytes"] == 32 * 1024 * 1024
    assert resources["oversized_allocation_outcome"] == "heap-limit"
    assert (
        resources["accepted_memory_used_bytes"]
        > resources["baseline_memory_used_bytes"]
    )
    assert resources["stack_limit_bytes"] == 512 * 1024
    assert resources["recursion_outcome"] == "stack-engine-trap"
    assert resources["maximum_accepted_recursion_depth"] >= 1
    assert resources["recursion_depth"] > 0
    assert result["browser_engine"] == "chromium"
    assert result["samples"] == 1


@pytest.mark.loopback_network
@pytest.mark.parametrize(
    "transform",
    [
        lambda body: _replace_once(
            body,
            b"result = context.evalCode(source, filename);",
            b'throw new RangeError("Maximum call stack size exceeded");',
        ),
        lambda body: _replace_once(
            body,
            b"result.error.dispose();",
            b'result.error.dispose(); throw new Error("PRIVATE-DISPOSAL-FAILURE");',
        ),
    ],
    ids=["host-api", "guest-error-disposal"],
)
def test_direct_quickjs_probe_sanitizes_unexpected_host_failures(
    monkeypatch: pytest.MonkeyPatch, transform: Callable[[bytes], bytes]
) -> None:
    probe = _load_probe()

    with pytest.raises(
        probe.ProbeError, match="trusted QuickJS resource probe failed unexpectedly"
    ) as raised:
        _run_direct_probe_with_engine_html(probe, monkeypatch, transform)

    assert "PRIVATE" not in str(raised.value)


@pytest.mark.loopback_network
def test_direct_quickjs_probe_requires_positive_recursion_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_probe()

    def reject_every_depth(body: bytes) -> bytes:
        return _replace_once(
            body,
            b"`function descend(n){if(n>0)descend(n-1)}descend(${depth});`,",
            b'`throw new Error("PRIVATE-FORCED-RECURSION-REFUSAL");`,',
        )

    with pytest.raises(
        probe.ProbeError, match="trusted QuickJS resource probe failed unexpectedly"
    ) as raised:
        _run_direct_probe_with_engine_html(probe, monkeypatch, reject_every_depth)

    assert "PRIVATE" not in str(raised.value)
