from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tldw_chatbook.Event_Handlers import worker_events

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RETAINED_CALLER_PATHS = (
    PROJECT_ROOT / "tldw_chatbook" / "Event_Handlers" / "conv_char_events.py",
    PROJECT_ROOT / "tldw_chatbook" / "UI" / "MediaWindow_v2.py",
)


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
    callsites: list[tuple[str, int]] = []

    for path in RETAINED_CALLER_PATHS:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "chat_wrapper"
            ):
                continue
            streaming = next(
                (
                    keyword.value
                    for keyword in node.keywords
                    if keyword.arg == "streaming"
                ),
                None,
            )
            assert isinstance(streaming, ast.Constant) and streaming.value is False, (
                f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} must declare "
                "streaming=False"
            )
            callsites.append((str(path.relative_to(PROJECT_ROOT)), node.lineno))

    assert len(callsites) == 7
