from __future__ import annotations

import ast
import logging
import os
import subprocess  # nosec B404
import sys
from collections import Counter
from pathlib import Path

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
    "server_sync_service",
    "local_first_sync_service",
    "manual_sync_control_service",
    "sync_v2_dataset_keys",
    "sync_state_repository",
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


def _server_sync_config_factory_calls() -> list[int]:
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"), filename=str(APP_PATH))
    app_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TldwCli"
    )
    return [
        node.lineno
        for node in ast.walk(app_class)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_config"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "ServerSyncService"
        )
    ]


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
    assert app.server_sync_service.client is None
    assert app.server_sync_service.client_provider is app.server_context_provider
    assert app.server_sync_service.state_repository is app.sync_state_repository
    assert app.sync_scope_service.server_service is app.server_sync_service
    assert app.sync_scope_service.state_repository is app.sync_state_repository
    assert app.local_first_sync_service.server_service is app.server_sync_service
    assert app.local_first_sync_service.state_repository is app.sync_state_repository
    assert app.local_first_sync_service.local_store is None
    assert app.local_first_sync_service.dataset_keys is app.sync_v2_dataset_keys
    assert app.sync_v2_dataset_keys == {}
    assert (
        app.manual_sync_control_service.local_first_sync_service
        is app.local_first_sync_service
    )
    assert app.manual_sync_control_service.state_repository is app.sync_state_repository
    assert app.manual_sync_control_service.dataset_keys is app.sync_v2_dataset_keys


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


def test_app_composition_does_not_use_sync_config_factory() -> None:
    assert _server_sync_config_factory_calls() == []


async def _composes_one_stable_dependency_graph(
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
    app.app_config.setdefault("first_run", {})["setup_completed"] = True
    identities = _service_identities(app)
    provider_close_calls = 0
    original_close_cached_client = app.server_context_provider.close_cached_client

    async def counted_close_cached_client() -> None:
        nonlocal provider_close_calls
        provider_close_calls += 1
        await original_close_cached_client()

    monkeypatch.setattr(
        app.server_context_provider,
        "close_cached_client",
        counted_close_cached_client,
    )

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
        assert provider_close_calls == 1
    finally:
        await _close_production_app(app)


async def _scheduler_worker_settles_without_contract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unmount must join the real Textual worker through its public API."""
    _disable_splash(monkeypatch)
    app = TldwCli()
    app.app_config["_first_run"] = False
    app.app_config.setdefault("first_run", {})["setup_completed"] = True
    scheduler_errors: list[str] = []
    sink_id: int | None = None

    try:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            scheduler_worker = app.scheduler_worker
            assert not scheduler_worker.is_finished
            sink_id = app.loguru_logger.add(
                lambda message: scheduler_errors.append(message.record["message"]),
                level="ERROR",
            )

        assert not any(
            "Error stopping scheduler worker" in message for message in scheduler_errors
        )
        assert scheduler_worker.is_finished
    finally:
        if sink_id is not None:
            app.loguru_logger.remove(sink_id)
        await _close_production_app(app)


_CHILD = r"""
import asyncio
import sys
from Tests.network_guard import install, blocked_attempts
install()
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
import pytest
from Tests.ProductionApp import test_service_composition_lifecycle as cases
from tldw_chatbook.Chat import local_server_discovery
from tldw_chatbook.UI.Screens import llm_screen

async def offline_refresh(_app):
    return None

# Match Tests/conftest.py::_no_local_server_probes without retargeting config.
async def no_endpoint(http_client, url, timeout, display):
    return None, f"No models endpoint at {display}."

async def no_ollama(host="127.0.0.1", port=11434):
    return False

with pytest.MonkeyPatch.context() as monkeypatch:
    monkeypatch.setattr(local_server_discovery, "_get_models_payload", no_endpoint)
    monkeypatch.setattr(llm_screen, "_probe_local_server", no_ollama)
    monkeypatch.setattr(cases.TldwCli, "_refresh_model_catalogs", offline_refresh)
    selected = {
        "graph": cases._composes_one_stable_dependency_graph,
        "scheduler": cases._scheduler_worker_settles_without_contract_error,
    }[sys.argv[1]]
    asyncio.run(selected(monkeypatch))
assert not blocked_attempts(), blocked_attempts()
print("LIFECYCLE_CASE_COMPLETE")
"""


def _run_lifecycle_case(tmp_path: Path, case: str) -> None:
    """Import the actual app only after the child's private selection is fixed."""
    directories = {
        name: tmp_path / name for name in ("home", "config", "data", "cache", "tmp")
    }
    for directory in directories.values():
        directory.mkdir(mode=0o700)
    selector = directories["config"] / "config.toml"
    selector.write_text(
        '[general]\nusers_name="default_user"\n'
        "[first_run]\nsetup_completed=true\n"
        "[splash_screen]\nenabled=false\n",
        encoding="utf-8",
    )
    selector.chmod(0o600)
    repo_root = Path(__file__).resolve().parents[2]
    env = {
        name: os.environ[name]
        for name in ("PATH", "LANG", "LC_ALL", "TERM", "COLORTERM")
        if name in os.environ
    }
    env.update(
        HOME=str(directories["home"]),
        USERPROFILE=str(directories["home"]),
        XDG_CONFIG_HOME=str(directories["config"]),
        XDG_DATA_HOME=str(directories["data"]),
        XDG_CACHE_HOME=str(directories["cache"]),
        TMPDIR=str(directories["tmp"]),
        TLDW_CONFIG_PATH=str(selector),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
        PYTHONPATH=str(repo_root),
    )
    output_path = tmp_path / "child-output.log"
    with output_path.open("w", encoding="utf-8") as output:
        # Fixed interpreter, script and case; no shell or external executable.
        result = subprocess.run(  # nosec B603
            [sys.executable, "-c", _CHILD, case],
            cwd=repo_root,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=120,
        )
    captured = output_path.read_text(encoding="utf-8")
    assert result.returncode == 0, captured
    assert "LIFECYCLE_CASE_COMPLETE" in captured, captured


def test_production_app_composes_one_stable_dependency_graph(tmp_path: Path) -> None:
    _run_lifecycle_case(tmp_path, "graph")


def test_production_app_scheduler_worker_settles_without_contract_error(
    tmp_path: Path,
) -> None:
    _run_lifecycle_case(tmp_path, "scheduler")
