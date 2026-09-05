"""Canvas integration failures must not persist exception payloads or locals."""

from types import SimpleNamespace

import pytest
from loguru import logger

from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.config import RuntimeConfigSnapshot
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen


@pytest.fixture
def diagnostic_records():
    records = []
    sink = logger.add(lambda message: records.append(message.record))
    try:
        yield records
    finally:
        logger.remove(sink)


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["gateway", "authority"])
async def test_canvas_disposal_diagnostics_omit_exception_content(
    owner, diagnostic_records
):
    """Reintroducing traceback logging would expose the owned object's payload."""
    runtime = ConsoleRuntime(SimpleNamespace(chachanotes_db=None))

    def fail():
        raise RuntimeError("PRIVATE-CANVAS-EXCEPTION-CONTENT")

    runtime._canvas_gateway = (
        SimpleNamespace(aclose=fail) if owner == "gateway" else None
    )
    runtime._canvas_native_authority = (
        SimpleNamespace(dispose=fail) if owner == "authority" else None
    )
    await runtime.dispose()
    records = [record for record in diagnostic_records if "Canvas" in record["message"]]
    assert records
    assert all(record["exception"] is None for record in records)
    assert all("PRIVATE-CANVAS" not in record["message"] for record in records)


@pytest.mark.parametrize("stage", ["snapshot", "generation"])
def test_canvas_config_failure_diagnostics_omit_exception_content(
    stage, monkeypatch, caplog
):
    """Config refresh failures must fail closed without serializing secret values."""
    app = _build_test_app()
    screen = SettingsScreen(app)
    snapshot = RuntimeConfigSnapshot(1, {"canvas": {"enabled": True}})

    def fail(*args, **kwargs):
        raise RuntimeError("PRIVATE-CANVAS-CONFIG-CONTENT")

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_screen.get_runtime_config_snapshot",
        fail if stage == "snapshot" else lambda: snapshot,
    )
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_screen.run_if_runtime_config_generation_current",
        fail,
    )
    policy, stable = screen._reconcile_canvas_runtime_policy(
        None if stage == "snapshot" else snapshot
    )
    assert stable is False
    assert policy.enabled is False
    records = [record for record in caplog.records if "Canvas" in record.getMessage()]
    assert records
    assert all(record.exc_info is None for record in records)
    assert all("PRIVATE-CANVAS" not in record.getMessage() for record in records)
