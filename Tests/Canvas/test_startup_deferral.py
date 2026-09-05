"""First-use boundaries for Canvas native and served runtime owners."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

CONTROL_ENV = {
    "CHATBOOK_CANVAS_CONTROL_HOST": "127.0.0.1",
    "CHATBOOK_CANVAS_CONTROL_PORT": "32123",
    "CHATBOOK_CANVAS_CONTROL_CHILD_ID": "child-a",
    "CHATBOOK_CANVAS_CONTROL_SECRET": "s" * 32,
    "CHATBOOK_CANVAS_CONTROL_VERSION": "1",
}
REPO_ROOT = Path(__file__).resolve().parents[2]
USER_OPEN_DIALOG_MODULES = frozenset(
    {
        "tldw_chatbook.Widgets.Console.console_prompt_queue_modal",
        "tldw_chatbook.Widgets.Console.console_review_notes_modal",
        "tldw_chatbook.Widgets.Console.console_side_chat_modal",
    }
)
CHATBOOK_CONFLICT_MODULE = "tldw_chatbook.Chatbooks.conflict_resolver"


def _isolated_environment(tmp_path: Path) -> dict[str, str]:
    home = tmp_path / "home"
    data_home = tmp_path / "data"
    config_home = tmp_path / "config"
    for path in (home, data_home, config_home):
        path.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "HOME": str(home),
        "PYTHONPATH": str(REPO_ROOT),
        "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring",
        "TLDW_CONFIG_PATH": str(config_home / "config.toml"),
        "TLDW_SCREEN_PREIMPORT": "0",
        "TLDW_TEST_MODE": "1",
        "USERPROFILE": str(home),
        "XDG_CONFIG_HOME": str(config_home),
        "XDG_DATA_HOME": str(data_home),
    }
    environment.pop("PYTEST_CURRENT_TEST", None)
    return environment


@pytest.mark.parametrize(
    ("environment", "expects_handler", "expects_client"),
    [
        ({}, False, False),
        *[({key: value}, True, False) for key, value in CONTROL_ENV.items()],
        (CONTROL_ENV, True, True),
    ],
    ids=["native", *[f"partial-{key}" for key in CONTROL_ENV], "served"],
)
def test_app_constructs_served_control_only_for_present_spawn_environment(
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
    expects_handler: bool,
    expects_client: bool,
) -> None:
    for key in CONTROL_ENV:
        monkeypatch.delenv(key, raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    app = _build_test_app()

    assert (app.served_canvas_handler is not None) is expects_handler
    assert (app.served_canvas_control is not None) is expects_client
    if expects_client:
        assert app.served_canvas_control.child_id == "child-a"


def test_native_app_does_not_import_served_control_transport(tmp_path: Path) -> None:
    environment = _isolated_environment(tmp_path)
    for key in CONTROL_ENV:
        environment.pop(key, None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import sys

                from Tests.UI.app_factory import _build_test_app

                app = _build_test_app()
                assert app.served_canvas_handler is None
                assert app.served_canvas_control is None
                assert "tldw_chatbook.Canvas.control_protocol" not in sys.modules
                assert "tldw_chatbook.Canvas.gateway" not in sys.modules
                """
            ),
        ],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_execution_only_config_read_does_not_import_web_auth(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import sys

                from tldw_chatbook import config

                config.load_cli_config_and_ensure_existence = lambda: {
                    "canvas": {"enabled": True, "auto_open_on_create": False},
                    "web_server": {
                        "host": "0.0.0.0",
                        "public_url": "https://chatbook.example",
                    },
                }
                assert "tldw_chatbook.Canvas.web_auth" not in sys.modules
                assert config.get_canvas_execution_enabled() is True
                assert "tldw_chatbook.Canvas.web_auth" not in sys.modules

                policy = config.build_canvas_config_policy(
                    {"web_server": {"host": "127.0.0.1"}}, environ={}
                )
                assert policy.remote_access_status == "loopback"
                assert "tldw_chatbook.Canvas.web_auth" in sys.modules
                """
            ),
        ],
        cwd=REPO_ROOT,
        env=_isolated_environment(tmp_path),
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_canvas_models_load_shared_wire_validation_only_on_first_decode(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import sys

                from tldw_chatbook.Canvas.models import CanvasBridgeRequest

                module_name = "tldw_chatbook.Utils.input_validation"
                assert module_name not in sys.modules
                request = CanvasBridgeRequest.from_wire(
                    {
                        "version": "canvas-v1",
                        "request_id": "request-first-use",
                        "kind": "submit",
                        "value": "synthetic",
                    }
                )
                assert request.submit_text() == "synthetic"
                assert module_name in sys.modules
                """
            ),
        ],
        cwd=REPO_ROOT,
        env=_isolated_environment(tmp_path),
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_console_owners_do_not_import_compiler_until_first_compile(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import sys

                import tldw_chatbook.Canvas.service as service
                import tldw_chatbook.Chat.console_canvas_controller as controller
                import tldw_chatbook.Chat.console_message_actions
                import tldw_chatbook.UI.Console_Modules.message

                compiler_name = "tldw_chatbook.Canvas.compiler"
                assert compiler_name not in sys.modules
                assert callable(service.compile_canvas_document)
                assert callable(controller.compile_canvas_document)

                plan = controller.compile_canvas_document(
                    "<!doctype html><title>Synthetic</title><p>ready</p>"
                )
                assert plan.runtime_profile == "canvas-v1"
                assert compiler_name in sys.modules
                """
            ),
        ],
        cwd=REPO_ROOT,
        env=_isolated_environment(tmp_path),
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_console_screen_defers_user_open_only_dialog_modules(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import sys

                import tldw_chatbook.UI.Screens.chat_screen

                dialog_modules = {set(USER_OPEN_DIALOG_MODULES)!r}
                loaded = sorted(dialog_modules.intersection(sys.modules))
                assert loaded == [], loaded
                """
            ),
        ],
        cwd=REPO_ROOT,
        env=_isolated_environment(tmp_path),
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_local_chatbook_service_defers_conflict_policy_until_import(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import sys

                from tldw_chatbook.Chatbooks.local_chatbook_service import (
                    LocalChatbookService,
                )

                assert {CHATBOOK_CONFLICT_MODULE!r} not in sys.modules
                service = LocalChatbookService({{}}, registry_path="registry.json")
                assert service.db_paths == {{}}
                assert {CHATBOOK_CONFLICT_MODULE!r} not in sys.modules
                """
            ),
        ],
        cwd=REPO_ROOT,
        env=_isolated_environment(tmp_path),
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def _bind_canvas_view(runtime: ConsoleRuntime, **hooks: object) -> None:
    binder = getattr(runtime, "bind_canvas_native_view", None)
    assert callable(binder), "ConsoleRuntime must expose lazy Canvas view binding"
    binder(**hooks)


def _canvas_scope(
    session_id: str, active_message_ids: tuple[str, ...], run_id: str
) -> Any:
    from tldw_chatbook.Canvas.models import CanvasScope

    return CanvasScope(
        session_id=session_id,
        conversation_id=session_id,
        active_message_ids=active_message_ids,
        selected_canvas_id=None,
        selected_revision_id=None,
        run_id=run_id,
    )


def _publish_root_canvas(
    runtime: ConsoleRuntime,
    store: Any,
    session_id: str,
    run_id: str,
    scopes: dict[str, Any],
) -> tuple[Any, Any]:
    assistant = store.append_message(
        session_id,
        role="assistant",
        content="synthetic assistant",
    )
    scope = _canvas_scope(
        session_id,
        tuple(store.active_path_message_ids(session_id)),
        run_id,
    )
    scopes[session_id] = scope
    run = runtime.canvas_controller.register_run(
        scope,
        assistant_message_id=assistant.id,
        temporary=True,
    )
    created = run.create_canvas(
        scope,
        tool_call_id=f"{run_id}-create",
        title="Synthetic Canvas",
        html="<!doctype html><p>synthetic</p>",
    )
    settlement = run.finish_assistant_run(
        assistant.id,
        actual_run_id=run_id,
        terminal_status="done",
    )
    assert settlement is not None
    assert runtime.canvas_controller.confirm_exact_settlement(settlement) is True
    return created, settlement


def test_canvas_view_binding_builds_authority_on_first_publication_and_rebinds(
    tmp_path,
) -> None:
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    database = CharactersRAGDB(tmp_path / "lazy-canvas.sqlite", "lazy-canvas")
    runtime = ConsoleRuntime(SimpleNamespace(chachanotes_db=database))
    try:
        store = runtime.ensure_chat_store()
        session = store.create_session(ephemeral=True)
        first_opened: list[tuple[str, str]] = []
        scopes = {}

        _bind_canvas_view(
            runtime,
            scope_resolver=lambda requested: scopes[requested],
            auto_open=lambda requested, info: first_opened.append(
                (requested, info.revision_id)
            ),
        )
        assert runtime._canvas_native_authority is None
        assert runtime.canvas_controller._settlement_listeners == [
            runtime._canvas_settlement_listener
        ]

        first, first_settlement = _publish_root_canvas(
            runtime, store, session.id, "first-run", scopes
        )

        assert runtime._canvas_native_authority is not None
        assert first_opened == [(session.id, first.revision.revision_id)]
        assert (
            runtime.canvas_controller.confirm_exact_settlement(first_settlement) is True
        )
        assert first_opened == [(session.id, first.revision.revision_id)]

        second_opened: list[tuple[str, str]] = []
        _bind_canvas_view(
            runtime,
            scope_resolver=lambda requested: scopes[requested],
            auto_open=lambda requested, info: second_opened.append(
                (requested, info.revision_id)
            ),
        )
        assert runtime.canvas_controller._settlement_listeners == [
            runtime._canvas_settlement_listener
        ]
        second, second_settlement = _publish_root_canvas(
            runtime, store, session.id, "second-run", scopes
        )

        assert first_opened == [(session.id, first.revision.revision_id)]
        assert second_opened == [(session.id, second.revision.revision_id)]
        assert (
            runtime.canvas_controller.confirm_exact_settlement(second_settlement)
            is True
        )
        assert second_opened == [(session.id, second.revision.revision_id)]
    finally:
        asyncio.run(runtime.dispose())
        database.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("closed", ["disabled", "disposed"])
async def test_canvas_view_binding_cannot_build_authority_after_runtime_closes(
    closed: str,
) -> None:
    enabled = [True]
    runtime = ConsoleRuntime(
        SimpleNamespace(chachanotes_db=None),
        canvas_enabled_reader=lambda: enabled[0],
    )
    runtime.ensure_chat_store()
    if closed == "disabled":
        enabled[0] = False
        assert runtime.canvas_enabled() is False
    else:
        await runtime.dispose()

    _bind_canvas_view(
        runtime,
        scope_resolver=lambda _requested: SimpleNamespace(),
        auto_open=lambda *_args: pytest.fail("closed Canvas opened"),
    )

    assert runtime._canvas_native_authority is None
    if closed != "disposed":
        await runtime.dispose()


def test_canvas_view_rebind_registers_listener_on_rebuilt_controller() -> None:
    runtime = ConsoleRuntime(SimpleNamespace(chachanotes_db=None))
    first_store = runtime.ensure_chat_store()
    _bind_canvas_view(
        runtime,
        scope_resolver=lambda _requested: SimpleNamespace(),
    )
    first_controller = runtime.canvas_controller

    runtime.set_chat_store(None)
    second_store = runtime.ensure_chat_store()
    second_controller = runtime.canvas_controller
    _bind_canvas_view(
        runtime,
        scope_resolver=lambda _requested: SimpleNamespace(),
    )

    try:
        assert second_store is not first_store
        assert second_controller is not first_controller
        assert first_controller._settlement_listeners == [
            runtime._canvas_settlement_listener
        ]
        assert second_controller._settlement_listeners == [
            runtime._canvas_settlement_listener
        ]
        assert runtime._canvas_native_authority is None
    finally:
        first_controller.close_runtime()
        asyncio.run(runtime.dispose())


def test_concurrent_first_canvas_ensure_constructs_one_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.Canvas import native_authority

    runtime = ConsoleRuntime(SimpleNamespace(chachanotes_db=None))
    runtime.ensure_chat_store()
    _bind_canvas_view(
        runtime,
        scope_resolver=lambda _requested: SimpleNamespace(),
    )
    real_authority = native_authority.NativeConsoleCanvasAuthority
    first_entered = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    count_lock = threading.Lock()
    construction_count = 0

    def controlled_authority(*args: Any, **kwargs: Any) -> Any:
        nonlocal construction_count
        with count_lock:
            construction_count += 1
            current = construction_count
        if current == 1:
            first_entered.set()
            assert release_first.wait(2), "first authority construction not released"
        else:
            second_entered.set()
        return real_authority(*args, **kwargs)

    monkeypatch.setattr(
        native_authority,
        "NativeConsoleCanvasAuthority",
        controlled_authority,
    )
    results: list[Any] = []

    def ensure(*, announce: bool = False) -> None:
        if announce:
            second_started.set()
        results.append(
            runtime.ensure_canvas_native_authority(
                scope_resolver=lambda _requested: SimpleNamespace(),
            )
        )

    first = threading.Thread(target=ensure)
    second = threading.Thread(target=ensure, kwargs={"announce": True})
    try:
        first.start()
        try:
            assert first_entered.wait(1), "first authority construction did not start"
            second.start()
            assert second_started.wait(1), "second authority ensure did not start"
            second_entered.wait(0.25)
        finally:
            release_first.set()
            if first.ident is not None:
                first.join(2)
            if second.ident is not None:
                second.join(2)
        assert not first.is_alive()
        assert not second.is_alive()
        assert construction_count == 1
        assert len(results) == 2
        assert results[0] is results[1]
    finally:
        release_first.set()
        asyncio.run(runtime.dispose())
