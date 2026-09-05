"""Behavioral coverage for the reproducible Canvas runtime quota probe."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tldw_chatbook.Canvas.compiler import CanvasCompileError, compile_canvas_document
from tldw_chatbook.Canvas.limits import CanvasLimits

ROOT = Path(__file__).resolve().parents[2]
PROBE_PATH = ROOT / "scripts" / "canvas_runtime_quota_probe.py"


def _load_probe() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "canvas_runtime_quota_probe", PROBE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_synthetic_fixtures_exercise_exact_compiler_boundaries() -> None:
    probe = _load_probe()
    limits = CanvasLimits()
    fixtures = {
        item.identifier: item for item in probe.build_synthetic_fixtures(limits)
    }

    at_dom = compile_canvas_document(fixtures["adversarial-dom-at-limit"].source)
    assert probe.count_plan_nodes(at_dom.root) == limits.dom_nodes
    with pytest.raises(CanvasCompileError, match="incompatible") as dom_error:
        compile_canvas_document(fixtures["adversarial-dom-over-limit"].source)
    assert dom_error.value.issues[0].code == "dom-limit"

    at_css = compile_canvas_document(fixtures["adversarial-css-at-limit"].source)
    assert len(at_css.css_rules) == limits.css_rules
    with pytest.raises(CanvasCompileError, match="incompatible") as css_error:
        compile_canvas_document(fixtures["adversarial-css-over-limit"].source)
    assert css_error.value.issues[0].code == "css-rule-limit"

    at_script = compile_canvas_document(fixtures["adversarial-script-at-limit"].source)
    assert (
        sum(len(script.encode("utf-8")) for script in at_script.scripts)
        == limits.script_bytes
    )
    with pytest.raises(CanvasCompileError, match="incompatible") as script_error:
        compile_canvas_document(fixtures["adversarial-script-over-limit"].source)
    assert script_error.value.issues[0].code == "script-limit"


def test_runtime_limit_mirrors_are_checked_as_one_contract() -> None:
    probe = _load_probe()
    limits = CanvasLimits()
    worker = (ROOT / "tldw_chatbook/Canvas/static/canvas_runtime_worker.js").read_text()
    renderer = (ROOT / "tldw_chatbook/Canvas/static/canvas_renderer.js").read_text()

    observed = probe.validate_runtime_limit_mirrors(
        limits=limits,
        worker_source=worker,
        renderer_source=renderer,
    )

    assert observed["python.dom_nodes"] == limits.dom_nodes
    assert observed["worker.dom_nodes"] == limits.dom_nodes
    assert observed["virtual_facade.dom_nodes"] == limits.dom_nodes
    assert observed["renderer.dom_nodes"] == limits.dom_nodes
    with pytest.raises(probe.ProbeError, match="worker.dom_nodes"):
        probe.validate_runtime_limit_mirrors(
            limits=limits,
            worker_source=worker.replace(
                f"domNodes: {limits.dom_nodes}",
                f"domNodes: {limits.dom_nodes + 1}",
                1,
            ),
            renderer_source=renderer,
        )


def test_content_free_summary_excludes_fixture_source_and_runtime_messages() -> None:
    probe = _load_probe()
    fixture = probe.ProbeFixture(
        identifier="synthetic-summary-contract",
        category="representative",
        source="PRIVATE-SENTINEL",
        expected="accepted",
    )
    summary = probe.build_summary(
        compiler_results=[
            probe.CompilerResult(
                fixture_id=fixture.identifier,
                category=fixture.category,
                source_bytes=len(fixture.source),
                plan_bytes=42,
                plan_nodes=7,
                css_rules=2,
                script_bytes=3,
                median_milliseconds=1.25,
                p95_milliseconds=1.5,
                maximum_milliseconds=1.75,
            )
        ],
        compiler_boundaries={"dom_over_limit": "dom-limit"},
        browser_results={"fixture_id": fixture.identifier, "state": "ready"},
        environment={"platform": "synthetic-test"},
        mirrors={"python.dom_nodes": 1_800},
    )
    encoded = json.dumps(summary, sort_keys=True)

    assert summary["fixture_provenance"] == "synthetic-agent-authored"
    assert "PRIVATE-SENTINEL" not in encoded
    assert "source" not in summary["compiler"]["fixtures"][0]
    assert "messages" not in encoded


def test_percentile_uses_nearest_rank_without_interpolating_measurements() -> None:
    probe = _load_probe()

    assert probe.percentile([5.0, 1.0, 4.0, 2.0, 3.0], 95) == 5.0
    assert probe.percentile([9.0], 95) == 9.0
