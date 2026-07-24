from __future__ import annotations

import ast
from collections.abc import AsyncIterator, Iterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from Tests.TTS.adapter_fakes import FakeAdapterFactory, provider_spec
from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSEventHandler,
    STTSPlaygroundGenerateEvent,
)
from tldw_chatbook.TTS.adapter_types import ProgressSink, TTSProgress
from tldw_chatbook.TTS.TTS_Generation import (
    get_tts_service,
    reset_tts_service_binding,
)
from tldw_chatbook.app import TldwCli


REPO_ROOT = Path(__file__).resolve().parents[2]


class FakeOwnedService:
    def __init__(self) -> None:
        self.close_calls = 0
        self.wait_closed_calls = 0

    async def close(self) -> None:
        self.close_calls += 1

    async def wait_closed(self) -> None:
        self.wait_closed_calls += 1


@pytest.fixture(autouse=True)
def isolated_tts_binding() -> Iterator[None]:
    reset_tts_service_binding()
    yield
    reset_tts_service_binding()


def _method_node(
    path: Path,
    class_name: str,
    method_name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )


def _self_method_calls(node: ast.AST, method_name: str) -> list[ast.Call]:
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "self"
        and call.func.attr == method_name
    ]


def _isolate_constructor_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "tldw_chatbook.app.get_library_collections_db_path",
        lambda: tmp_path / "library_collections.sqlite",
    )
    monkeypatch.setattr(
        "tldw_chatbook.app.get_library_ingest_jobs_db_path",
        lambda: tmp_path / "library_ingest_jobs.sqlite",
    )
    monkeypatch.setattr(
        "tldw_chatbook.app.get_scheduled_tasks_db_path",
        lambda: tmp_path / "scheduled_tasks.sqlite",
    )


def test_app_constructs_one_tts_service(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = FakeOwnedService()
    builder = Mock(return_value=service)
    monkeypatch.setattr("tldw_chatbook.app.build_default_tts_service", builder)
    _isolate_constructor_paths(monkeypatch, tmp_path)

    app = _build_test_app()

    assert app.tts_service is service
    assert app._tts_binding_active is False
    builder.assert_called_once_with(app.app_config)


def test_app_construction_does_not_materialize_an_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    factory = FakeAdapterFactory("openai")

    def provider_specs(
        config: Mapping[str, Any],
    ) -> tuple[Any, ...]:
        del config
        return (provider_spec("openai", factory),)

    monkeypatch.setattr(
        "tldw_chatbook.TTS.adapter_bootstrap.legacy_provider_specs",
        provider_specs,
    )
    _isolate_constructor_paths(monkeypatch, tmp_path)

    app = _build_test_app()

    assert app.tts_service.registry.descriptors()[0].provider_id == "openai"
    assert factory.calls == 0


@pytest.mark.asyncio
async def test_app_binding_and_close_are_explicit_and_idempotent() -> None:
    service = FakeOwnedService()
    owner = SimpleNamespace(tts_service=service, _tts_binding_active=False)

    TldwCli._bind_tts_service(owner)
    TldwCli._bind_tts_service(owner)
    assert await get_tts_service() is service

    await TldwCli._close_tts_service(owner)
    await TldwCli._close_tts_service(owner)

    assert service.close_calls == 1
    assert service.wait_closed_calls == 1
    assert owner._tts_binding_active is False
    with pytest.raises(RuntimeError, match="not bound"):
        await get_tts_service()


def test_existing_mount_binds_before_screen_work() -> None:
    method = _method_node(REPO_ROOT / "tldw_chatbook/app.py", "TldwCli", "on_mount")
    bind_calls = _self_method_calls(method, "_bind_tts_service")
    restore_calls = _self_method_calls(method, "_restore_ingest_jobs")

    assert len(bind_calls) == 1
    assert len(restore_calls) == 1
    assert bind_calls[0].lineno < restore_calls[0].lineno


def test_unmount_closes_owned_service_without_handler_guard() -> None:
    method = _method_node(REPO_ROOT / "tldw_chatbook/app.py", "TldwCli", "on_unmount")
    close_calls = _self_method_calls(method, "_close_tts_service")
    parent_by_node = {
        child: parent
        for parent in ast.walk(method)
        for child in ast.iter_child_nodes(parent)
    }

    assert len(close_calls) == 1
    parent = parent_by_node[close_calls[0]]
    try_ancestors = 0
    while parent is not method:
        assert not isinstance(parent, ast.If)
        try_ancestors += isinstance(parent, ast.Try)
        parent = parent_by_node[parent]
    assert try_ancestors == 1


def test_application_and_stts_do_not_reach_through_to_backend_manager() -> None:
    paths = (
        REPO_ROOT / "tldw_chatbook/app.py",
        REPO_ROOT / "tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py",
    )
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        accesses = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr == "backend_manager"
        ]
        assert accesses == [], f"{path} reaches through to backend_manager"


class CapturingStreamService:
    def __init__(self) -> None:
        self.progress_sink: ProgressSink | None = None

    async def generate_audio_stream(
        self,
        request: object,
        internal_model_id: str,
        progress_sink: ProgressSink | None = None,
    ) -> AsyncIterator[bytes]:
        del request, internal_model_id
        self.progress_sink = progress_sink
        assert progress_sink is not None
        await progress_sink(
            TTSProgress(
                status="Generating",
                fraction=0.5,
                processed=1,
                total=2,
            )
        )
        yield b"RIFF"


@pytest.mark.asyncio
async def test_stts_forwards_typed_progress_sink_to_service() -> None:
    app = SimpleNamespace(notify=Mock())
    handler = STTSEventHandler(app=app)
    service = CapturingStreamService()
    handler._stts_service = service
    status_text = SimpleNamespace(update=Mock())
    progress_bar = SimpleNamespace(update=Mock())
    generation_log = SimpleNamespace(write=Mock())
    status_container = SimpleNamespace(remove_class=Mock(), add_class=Mock())
    audio_status = SimpleNamespace(update=Mock())
    generate_button = SimpleNamespace(disabled=True)
    play_button = SimpleNamespace(disabled=True)
    export_button = SimpleNamespace(disabled=True)
    widgets = {
        "#generation-status-container": status_container,
        "#generation-progress": progress_bar,
        "#generation-status-text": status_text,
        "#tts-generation-log": generation_log,
        "#audio-play-btn": play_button,
        "#audio-export-btn": export_button,
        "#audio-player-status": audio_status,
        "#tts-generate-btn": generate_button,
    }
    scheduled: list[object] = []

    def query_one(selector: str, widget_type: object = None) -> object:
        del widget_type
        return widgets[selector]

    def call_from_thread(callback: object, *args: object) -> None:
        scheduled.append(callback)
        assert callable(callback)
        callback(*args)

    playground = SimpleNamespace(
        query_one=query_one,
        call_from_thread=call_from_thread,
    )
    event = STTSPlaygroundGenerateEvent(
        text="hello",
        provider="openai",
        voice="alloy",
        model="tts-1",
        speed=1.0,
        format="wav",
    )

    await handler._generate_tts_worker(event, playground)

    assert service.progress_sink is not None
    assert scheduled
    status_text.update.assert_called_with("Generating")
    progress_bar.update.assert_any_call(progress=50.0)
    generation_log.write.assert_any_call("[dim]Processed 1/2 item(s)[/dim]")
    assert handler._current_audio_file is not None
    handler._current_audio_file.unlink()
