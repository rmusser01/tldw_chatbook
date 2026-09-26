"""LLAMA_GGUF override containment for the long-tail capture harness.

Qodo finding 6 (ADR-179 fix wave): the capture script's model override is
untrusted environment input feeding a ``llama-server`` process, so it is
resolved through ``tldw_chatbook.Utils.path_validation``'s read-mode
containment against explicit allowed roots (the home model directories by
default). Only a validated, existing file inside a root is ever used;
traversal and out-of-root values are refused.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_CAPTURE_LOCAL = Path(__file__).resolve().parent / "capture_local.py"
_spec = importlib.util.spec_from_file_location("capture_local", _CAPTURE_LOCAL)
assert _spec is not None and _spec.loader is not None
capture_local = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("capture_local", capture_local)
_spec.loader.exec_module(capture_local)


def _confine_roots(monkeypatch, *roots: Path) -> None:
    monkeypatch.setattr(capture_local, "_gguf_allowed_roots", lambda: tuple(roots))


def test_override_inside_root_is_used(tmp_path, monkeypatch) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"x" * 16)
    _confine_roots(monkeypatch, tmp_path)

    assert capture_local._llama_gguf_override(str(model)) == model


def test_traversal_out_of_root_is_refused(tmp_path, monkeypatch) -> None:
    outside = tmp_path.parent / "secret.gguf"
    outside.write_bytes(b"x" * 16)
    try:
        _confine_roots(monkeypatch, tmp_path)
        traversal = str(tmp_path / ".." / "secret.gguf")

        assert capture_local._llama_gguf_override(traversal) is None
    finally:
        outside.unlink()


def test_absolute_path_outside_root_is_refused(tmp_path, monkeypatch) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    model = outside / "model.gguf"
    model.write_bytes(b"x" * 16)
    _confine_roots(monkeypatch, root)  # root excludes the sibling dir

    assert capture_local._llama_gguf_override(str(model)) is None
    assert capture_local._llama_gguf_override(str(outside / ".." / "outside" / "model.gguf")) is None


def test_missing_file_inside_root_is_refused(tmp_path, monkeypatch) -> None:
    _confine_roots(monkeypatch, tmp_path)

    assert capture_local._llama_gguf_override(str(tmp_path / "absent.gguf")) is None


def test_default_roots_are_home_model_directories() -> None:
    home = Path.home()
    assert capture_local._gguf_allowed_roots() == (
        home / ".ollama" / "models",
        home / ".lmstudio" / "models",
        home / ".cache" / "llama.cpp",
        home / "models",
    )
