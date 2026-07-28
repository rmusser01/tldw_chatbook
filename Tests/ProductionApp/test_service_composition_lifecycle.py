from __future__ import annotations

from collections import Counter
import ast
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import tldw_chatbook.app as app_module
from tldw_chatbook.app import TldwCli


WIRING_METHODS = (
    "_wire_writing_services",
    "_wire_chat_conversation_services",
)
EXPECTED_WIRING_CALLS = Counter({name: 1 for name in WIRING_METHODS})
SYNC_CONSUMER_CLASSES = (
    app_module.ChatConversationScopeService,
    app_module.MediaReadingScopeService,
)
SERVICE_ATTRIBUTES = (
    "local_writing_service",
    "server_writing_service",
    "writing_scope_service",
    "local_chat_conversation_service",
    "conversation_local_marks_service",
    "server_chat_conversation_service",
    "chat_conversation_scope_service",
    "citation_trace_repository",
    "citation_legacy_migration_service",
    "citation_artifact_ownership_coordinator",
    "media_reading_scope_service",
    "sync_scope_service",
)
APP_PATH = Path(app_module.__file__).resolve()


def _constructor_wiring_calls() -> Counter[str]:
    tree = ast.parse(
        APP_PATH.read_text(encoding="utf-8"),
        filename=str(APP_PATH),
    )
    app_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TldwCli"
    )
    init_method = next(
        node
        for node in app_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    return Counter(
        node.func.attr
        for node in ast.walk(init_method)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr in WIRING_METHODS
        )
    )


def _disable_splash(monkeypatch: pytest.MonkeyPatch) -> None:
    real_get_cli_setting = app_module.get_cli_setting

    def get_cli_setting_without_splash(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return real_get_cli_setting(section, key, default)

    monkeypatch.setattr(app_module, "get_cli_setting", get_cli_setting_without_splash)


def _service_identities(app: TldwCli) -> tuple[object, ...]:
    return tuple(getattr(app, name) for name in SERVICE_ATTRIBUTES)


def _assert_service_identities(
    app: TldwCli,
    expected: tuple[object, ...],
) -> None:
    current = _service_identities(app)
    assert len(current) == len(expected)
    assert all(
        actual is original for actual, original in zip(current, expected, strict=True)
    )


def _assert_service_graph(app: TldwCli) -> None:
    assert app.writing_scope_service.local_service is app.local_writing_service
    assert app.writing_scope_service.server_service is app.server_writing_service
    assert app.server_writing_service.client_provider is app.server_context_provider
    assert (
        app.chat_conversation_scope_service.local_service
        is app.local_chat_conversation_service
    )
    assert (
        app.chat_conversation_scope_service.server_service
        is app.server_chat_conversation_service
    )
    assert (
        app.chat_conversation_scope_service.sync_scope_service is app.sync_scope_service
    )
    assert app.media_reading_scope_service.sync_scope_service is app.sync_scope_service
    assert (
        app.local_chat_conversation_service.citation_legacy_migration
        is app.citation_legacy_migration_service
    )
    assert (
        app.citation_artifact_ownership_coordinator.trace_repository
        is app.citation_trace_repository
    )
    assert (
        app.citation_artifact_ownership_coordinator.artifact_store
        is app.local_chatbook_service
    )


async def _close_production_app(app: TldwCli) -> None:
    try:
        if app._rich_log_handler:
            await app._rich_log_handler.stop_processor()
            logging.getLogger().removeHandler(app._rich_log_handler)
            app._rich_log_handler.close()
        await app.on_shutdown_request()
        await app.on_unmount()
    except Exception:
        pass


def test_constructor_contains_one_call_for_each_guarded_composition_helper() -> None:
    assert _constructor_wiring_calls() == EXPECTED_WIRING_CALLS


@pytest.mark.asyncio
async def test_production_app_composes_one_stable_dependency_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: Counter[str] = Counter()
    initial_sync_arguments: dict[str, list[object]] = {
        consumer.__name__: [] for consumer in SYNC_CONSUMER_CLASSES
    }
    for method_name in WIRING_METHODS:
        original = getattr(TldwCli, method_name)

        def counted(
            self: TldwCli,
            _original=original,
            _method_name=method_name,
        ) -> None:
            calls[_method_name] += 1
            _original(self)

        monkeypatch.setattr(TldwCli, method_name, counted)

    for consumer in SYNC_CONSUMER_CLASSES:
        original_init = consumer.__init__

        def captured_init(
            self,
            *args,
            _original=original_init,
            _consumer_name=consumer.__name__,
            **kwargs,
        ) -> None:
            initial_sync_arguments[_consumer_name].append(
                kwargs.get("sync_scope_service")
            )
            _original(self, *args, **kwargs)

        monkeypatch.setattr(consumer, "__init__", captured_init)

    _disable_splash(monkeypatch)
    app = TldwCli()
    app.app_config["_first_run"] = False
    identities = _service_identities(app)

    try:
        assert calls == EXPECTED_WIRING_CALLS
        assert initial_sync_arguments == {
            consumer.__name__: [app.sync_scope_service]
            for consumer in SYNC_CONSUMER_CLASSES
        }
        _assert_service_graph(app)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert calls == EXPECTED_WIRING_CALLS
            _assert_service_identities(app, identities)
            _assert_service_graph(app)
        assert calls == EXPECTED_WIRING_CALLS
        _assert_service_identities(app, identities)
        _assert_service_graph(app)
    finally:
        await _close_production_app(app)


@pytest.mark.asyncio
async def test_production_app_scheduler_worker_settles_without_contract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unmount must join the real Textual worker through its public API."""
    _disable_splash(monkeypatch)
    app = TldwCli()
    app.app_config["_first_run"] = False
    logger = MagicMock()
    app.loguru_logger = logger

    try:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            scheduler_worker = app.scheduler_worker
            assert not scheduler_worker.is_finished

        scheduler_errors = [
            call
            for call in logger.error.call_args_list
            if "Error stopping scheduler worker" in str(call)
        ]
        assert scheduler_errors == []
        assert scheduler_worker.is_finished
    finally:
        await _close_production_app(app)
