"""The standalone evidence script diagnoses unavailable optional SVG support."""

import runpy
import sys
from pathlib import Path

import pytest

from tldw_chatbook.Utils import optional_deps


async def test_capture_without_svg_reports_installation_guidance(monkeypatch):
    script = (
        Path(__file__).resolve().parents[2]
        / "Docs/superpowers/qa/workflows-authoring-dev/capture.py"
    )
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(optional_deps, "ensure_svg_rendering", lambda: False)
    capture = runpy.run_path(str(script))["capture"]
    monkeypatch.setitem(sys.modules, "cairocffi", None)
    monkeypatch.setitem(sys.modules, "cairosvg", None)

    with pytest.raises(ImportError, match=r"tldw_chatbook\[svg\]"):
        await capture()
