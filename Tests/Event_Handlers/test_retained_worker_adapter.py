from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tldw_chatbook.Event_Handlers import worker_events

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_ROOT = PROJECT_ROOT / "tldw_chatbook"
RETAINED_CALLER_PATHS = (PROJECT_ROOT / "tldw_chatbook" / "UI" / "MediaWindow_v2.py",)
EXPECTED_CALLS_BY_PATH = {
    RETAINED_CALLER_PATHS[0]: 1,
}


def _iter_chat_wrapper_calls(source_root: Path):
    for path in sorted(source_root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "chat_wrapper" not in source:
            continue
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "chat_wrapper"
            ):
                yield path, node


def test_chat_wrapper_caller_scan_covers_the_production_tree(tmp_path: Path) -> None:
    approved = tmp_path / "approved.py"
    approved.write_text(
        "def run(app):\n    return app.chat_wrapper(streaming=False)\n",
        encoding="utf-8",
    )
    nested = tmp_path / "nested"
    nested.mkdir()
    unauthorized = nested / "unauthorized.py"
    unauthorized.write_text(
        "def run(app):\n    return app.chat_wrapper(streaming=True)\n",
        encoding="utf-8",
    )

    assert {
        (path.relative_to(tmp_path), call.lineno)
        for path, call in _iter_chat_wrapper_calls(tmp_path)
    } == {
        (Path("approved.py"), 2),
        (Path("nested/unauthorized.py"), 2),
    }


def test_retained_worker_adapter_delegates_non_streaming_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    expected = object()

    def core_chat_function(**kwargs):
        observed.update(kwargs)
        return expected

    monkeypatch.setattr(worker_events, "core_chat_function", core_chat_function)

    result = worker_events.chat_wrapper_function(
        None,
        strip_thinking_tags=False,
        message="retained destination",
        streaming=False,
    )

    assert result is expected
    assert observed == {
        "strip_thinking_tags": False,
        "message": "retained destination",
        "streaming": False,
    }


def test_retained_worker_adapter_rejects_legacy_streaming_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def core_chat_function(**kwargs):
        nonlocal called
        called = True
        return kwargs

    monkeypatch.setattr(worker_events, "core_chat_function", core_chat_function)

    with pytest.raises(ValueError, match="no longer owns streaming"):
        worker_events.chat_wrapper_function(
            None,
            message="must use native Console",
            streaming=True,
        )

    assert called is False


def test_every_retained_chat_wrapper_caller_is_explicitly_non_streaming() -> None:
    calls_by_path = dict.fromkeys(RETAINED_CALLER_PATHS, 0)

    for path, node in _iter_chat_wrapper_calls(PRODUCTION_ROOT):
        assert path in EXPECTED_CALLS_BY_PATH, (
            f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} is not an approved "
            "TldwCli.chat_wrapper destination"
        )
        streaming = next(
            (keyword.value for keyword in node.keywords if keyword.arg == "streaming"),
            None,
        )
        assert isinstance(streaming, ast.Constant) and streaming.value is False, (
            f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} must declare "
            "streaming=False"
        )
        calls_by_path[path] += 1

    assert calls_by_path == EXPECTED_CALLS_BY_PATH
