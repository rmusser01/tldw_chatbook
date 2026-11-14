"""The checked control refresh shares derivation only for one synchronous pass."""

import asyncio

import pytest

from Tests.UI.console_fixture_ownership import owned_console_apps  # noqa: F401
from Tests.UI.test_console_mcp_approval import (
    _build_ready_production_console_app as _ready_host,
)
from Tests.UI.test_console_mcp_approval import (
    _wait_for_production_console_ready,
)
from tldw_chatbook import config
from tldw_chatbook.UI.Screens import chat_screen

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.mark.parametrize("raising_leg", [False, True])
@pytest.mark.parametrize(
    "refresh_name", ["_sync_console_control_bar", "_sync_console_rail_and_controls"]
)
async def test_control_derivation_retires_before_the_next_refresh(
    monkeypatch, raising_leg, refresh_name
):
    app = _ready_host()
    async with app.run_test(size=(120, 50)) as pilot:
        screen = await _wait_for_production_console_ready(app, pilot)
        reads = []
        original = screen._build_console_provider_selection

        def selection(*args, **kwargs):
            result = original(*args, **kwargs)
            reads.append(screen._provider_readiness_app_config())
            return result

        async def refresh(inspector=None):
            async with asyncio.timeout(10):
                while True:
                    reads.clear()
                    with monkeypatch.context() as attempt:
                        if inspector is not None:
                            attempt.setattr(
                                screen, "_build_console_inspector_state", inspector
                            )
                        complete = getattr(screen, refresh_name)()
                    if complete:
                        return
                    await _wait_for_production_console_ready(app, pilot)

        with monkeypatch.context() as observed:
            observed.setattr(screen, "_build_console_provider_selection", selection)
            if raising_leg:
                failure = RuntimeError("control projection failed")
                native_inspector = screen._build_console_inspector_state
                memo_seen = []

                def fail(*args, **kwargs):
                    native_inspector(*args, **kwargs)
                    memo_seen.append(bool(screen._console_derivation_memo))
                    raise failure

                with pytest.raises(RuntimeError) as caught:
                    await refresh(fail)
                assert caught.value is failure
                assert memo_seen == [True]
                assert screen._console_derivation_memo is None

            await refresh()
            assert reads and all(value is reads[0] for value in reads)
            assert screen._console_derivation_memo is None
            next_name = "Fresh control projection"
            assert config.save_setting_to_cli_config("general", "users_name", next_name)
            await refresh()
            assert reads and all(value is reads[0] for value in reads)
            assert reads[0]["general"]["users_name"] == next_name
            assert screen._console_derivation_memo is None

        await _wait_for_production_console_ready(app, pilot)
