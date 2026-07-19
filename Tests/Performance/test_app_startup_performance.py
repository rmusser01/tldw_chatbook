"""Performance guardrails for startup and import-time behavior."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


async def _wait_until(
    condition,
    *,
    pause,
    timeout_seconds: float = 3.0,
    interval_seconds: float = 0.05,
) -> None:
    """Wait for a test-app condition without sleeping the host process."""

    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        if condition():
            return
        await pause(interval_seconds)
    if condition():
        return
    raise AssertionError(f"condition was not met within {timeout_seconds:.1f}s")


def _run_isolated_python(
    tmp_path: Path,
    code: str,
    *,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a Python snippet with isolated Chatbook config/data directories."""

    data_home = tmp_path / "data"
    config_home = tmp_path / "config"
    home = tmp_path / "home"
    for path in (data_home, config_home, home):
        path.mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "TLDW_TEST_MODE": "1",
        "XDG_DATA_HOME": str(data_home),
        "XDG_CONFIG_HOME": str(config_home),
        "HOME": str(home),
        "PYTHONPATH": str(REPO_ROOT),
    }
    env.pop("PYTEST_CURRENT_TEST", None)
    if extra_env:
        env.update(extra_env)

    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_optional_deps_import_does_not_eagerly_check_embeddings(tmp_path: Path) -> None:
    """Importing optional_deps should not import heavyweight embeddings packages."""

    result = _run_isolated_python(
        tmp_path,
        """
        import json
        import sys

        import tldw_chatbook.Utils.optional_deps  # noqa: F401

        guards = ("torch", "transformers", "chromadb", "sentence_transformers")
        print(json.dumps({"loaded": [name for name in guards if name in sys.modules]}))
        """,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["loaded"] == []
    assert "Checking embeddings dependencies early" not in result.stderr


def test_optional_deps_eager_env_still_initializes_dependency_checks(tmp_path: Path) -> None:
    """Explicit eager dependency mode should still run dependency initialization."""

    result = _run_isolated_python(
        tmp_path,
        """
        import json

        import tldw_chatbook.Utils.optional_deps as optional_deps

        print(json.dumps({"initialized": optional_deps._initialized}))
        """,
        extra_env={"TLDW_EAGER_DEPENDENCY_CHECK": "true"},
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["initialized"] is True
    assert "Eager dependency checking enabled via TLDW_EAGER_DEPENDENCY_CHECK" in result.stderr


def test_app_import_does_not_load_legacy_feature_windows(tmp_path: Path) -> None:
    """Plain app import should not load heavy destination/legacy windows."""

    result = _run_isolated_python(
        tmp_path,
        """
        import json
        import sys

        import tldw_chatbook.app  # noqa: F401

        guards = (
            "tldw_chatbook.UI.Evals.evals_window_v3",
            "tldw_chatbook.UI.STTS_Window",
            "tldw_chatbook.UI.MediaWindow_v2",
            "tldw_chatbook.Utils.Splash_Screens.classic.glitch_reveal",
            "tldw_chatbook.Utils.Splash_Screens.tech.code_scroll",
        )
        print(json.dumps({"loaded": [name for name in guards if name in sys.modules]}))
        """,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["loaded"] == []


@pytest.mark.asyncio
async def test_ui_ready_before_nonessential_startup_services_finish(monkeypatch) -> None:
    """Optional audio/DB/cleanup startup work should not gate initial UI readiness."""

    from Tests.UI.test_screen_navigation import _build_test_app
    from tldw_chatbook.Utils.db_status_manager import DBStatusManager
    from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import TTSEventHandler
    from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import STTSEventHandler
    from tldw_chatbook.app import TldwCli

    tts_started = asyncio.Event()
    stts_started = asyncio.Event()
    db_size_started = asyncio.Event()
    media_cleanup_started = asyncio.Event()
    release_optional_work = asyncio.Event()

    async def blocked_tts_init(self) -> None:
        tts_started.set()
        await release_optional_work.wait()

    async def blocked_stts_init(self) -> None:
        stts_started.set()
        await release_optional_work.wait()

    async def blocked_db_size_update(self) -> None:
        db_size_started.set()
        await release_optional_work.wait()

    async def blocked_media_cleanup(self) -> None:
        media_cleanup_started.set()
        await release_optional_work.wait()

    def test_cli_setting(section: str, key: str | None = None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        if section == "media_cleanup" and key == "enabled":
            return True
        if section == "media_cleanup" and key == "cleanup_on_startup":
            return True
        return default

    monkeypatch.setattr(TTSEventHandler, "initialize_tts", blocked_tts_init)
    monkeypatch.setattr(STTSEventHandler, "initialize_stts", blocked_stts_init)
    monkeypatch.setattr(DBStatusManager, "update_db_sizes", blocked_db_size_update)
    monkeypatch.setattr(TldwCli, "perform_media_cleanup", blocked_media_cleanup)
    monkeypatch.setattr("tldw_chatbook.app.get_cli_setting", test_cli_setting)

    app = _build_test_app()

    async with app.run_test(size=(120, 36)) as pilot:
        try:
            await _wait_until(lambda: app._ui_ready, pause=pilot.pause)
            await _wait_until(
                lambda: (
                    tts_started.is_set()
                    and stts_started.is_set()
                    and db_size_started.is_set()
                ),
                pause=pilot.pause,
            )
            assert app._ui_ready is True

            await asyncio.sleep(0.2)
            assert media_cleanup_started.is_set() is False
        finally:
            release_optional_work.set()
            await pilot.pause(0.05)


@pytest.mark.asyncio
async def test_tts_handler_initializes_on_first_use(monkeypatch) -> None:
    """TTS event paths can initialize the handler lazily after startup."""

    from Tests.UI.test_screen_navigation import _build_test_app
    from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import TTSEventHandler

    initialized = asyncio.Event()

    async def initialize_tts(self) -> None:
        initialized.set()

    monkeypatch.setattr(TTSEventHandler, "initialize_tts", initialize_tts)

    app = _build_test_app()

    assert app._tts_handler is None
    handler = await app._ensure_tts_handler()

    assert initialized.is_set()
    assert handler is app._tts_handler


@pytest.mark.asyncio
async def test_stts_handler_initializes_on_first_use(monkeypatch) -> None:
    """S/TT/S command paths can initialize the handler lazily after startup."""

    from Tests.UI.test_screen_navigation import _build_test_app
    from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import STTSEventHandler

    initialized = asyncio.Event()

    async def initialize_stts(self) -> None:
        initialized.set()

    monkeypatch.setattr(STTSEventHandler, "initialize_stts", initialize_stts)

    app = _build_test_app()

    assert app._stts_handler is None
    handler = await app._ensure_stts_handler()

    assert initialized.is_set()
    assert handler is app._stts_handler
