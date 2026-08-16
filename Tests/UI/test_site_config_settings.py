"""TASK-16841: `SiteConfigSettings`'s `#auth-type-select` is backwards.

The TASK-15991 review (scratchpad/review15991.md) found `#auth-type-select`
(``UI/SiteConfigSettings.py``, `_compose_auth_settings`) composed with
``(value, label)``-ordered option tuples -- ``("none", "None"), ("basic",
"Basic Auth"), ...`` -- backwards against Textual's ``(label, value)``
contract. This is the same bug class TASK-15772 (six STTS sites) and
TASK-15991 (two ScraperBuilderWindow sites) fixed piecemeal.

The consumer, `display_config`, sets ``auth_select.value = config.auth_type
or "none"`` where `config.auth_type` is always a machine token (``"basic"``,
``"bearer"``, ``"api_key"``, or ``None`` -- see
`Subscriptions/site_config_manager.py::SiteConfig.__init__`'s own comment:
``# 'basic', 'bearer', 'api_key'``). With the tuples reversed, the Select's
real *values* are the display labels ("None", "Basic Auth", ...), so setting
``.value = "basic"`` raises ``InvalidSelectValueError``. That call happens
inside `load_site_config`'s ``@work(thread=True)`` body, wrapped in a bare
``try/except Exception: logger.error(f"...: {str(e)}")`` -- so the crash was
silently swallowed with a one-line, traceback-free log message, not
surfaced to the user or the test suite.

Born red at HEAD (pre-fix): `test_display_config_restores_saved_auth_type_without_raising`
raises `InvalidSelectValueError` when `auth-type-select` still ships the
reversed tuples. Green once the tuples are swapped to `(label, value)`.

`test_load_site_config_worker_narrows_the_swallow_to_a_traceback` proves the
narrowed ``load_site_config`` except now logs a record carrying a real
traceback (``record["exception"] is not None``) instead of a bare
``str(e)`` line -- so a *future* instance of this bug class cannot hide the
same way again.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from loguru import logger as loguru_logger
from textual.app import App, ComposeResult
from textual.widgets import Select
from textual.widgets._select import InvalidSelectValueError

from tldw_chatbook.Subscriptions.site_config_manager import (
    SiteConfig,
    SiteConfigManager,
)
from tldw_chatbook.UI import SiteConfigSettings as site_config_settings_module
from tldw_chatbook.UI.SiteConfigSettings import SiteConfigSettings

BUNDLE = (
    Path(__file__).resolve().parents[2]
    / "tldw_chatbook"
    / "css"
    / "tldw_cli_modular.tcss"
)


class _Harness(App[None]):
    CSS_PATH = str(BUNDLE)

    def compose(self) -> ComposeResult:
        yield SiteConfigSettings()


@pytest.fixture()
def isolated_manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SiteConfigManager:
    """A real `SiteConfigManager` against a tmp_path DB -- never the user's."""
    manager = SiteConfigManager(str(tmp_path / "subs.db"))
    monkeypatch.setattr(
        site_config_settings_module, "get_site_config_manager", lambda: manager
    )
    return manager


@pytest.mark.asyncio
async def test_display_config_restores_saved_auth_type_without_raising(
    isolated_manager: SiteConfigManager,
) -> None:
    """AC#1 born-red: a saved 'basic' auth_type must round-trip through the Select."""
    config = SiteConfig("example.com", {"auth_type": "basic"})
    assert isolated_manager.save_config(config) is True

    app = _Harness()
    async with app.run_test(size=(160, 48)) as pilot:
        widget = app.query_one(SiteConfigSettings)
        await pilot.pause()

        # This is the exact call site that raised InvalidSelectValueError
        # pre-fix (UI/SiteConfigSettings.py::display_config, the
        # `auth_select.value = config.auth_type or "none"` line).
        widget.display_config(config)

        auth_select = widget.query_one("#auth-type-select", Select)
        assert auth_select.value == "basic"
        assert not widget.query_one("#basic-auth-container").has_class("hidden")
        assert widget.query_one("#bearer-auth-container").has_class("hidden")
        assert widget.query_one("#api-key-container").has_class("hidden")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "auth_type,container_id",
    [
        ("none", None),
        ("basic", "basic-auth-container"),
        ("bearer", "bearer-auth-container"),
        ("api_key", "api-key-container"),
    ],
)
async def test_all_machine_tokens_are_accepted_select_values(
    isolated_manager: SiteConfigManager, auth_type: str, container_id: str | None
) -> None:
    """Every real `auth_type` machine token must be a valid Select value.

    `SiteConfig.auth_type` is always one of these four tokens (see the
    class's own inline comment) -- never the display label. A Select whose
    real values are the labels would raise `InvalidSelectValueError` for
    every one of these.
    """
    config = SiteConfig("example.com", {"auth_type": auth_type})

    app = _Harness()
    async with app.run_test(size=(160, 48)) as pilot:
        widget = app.query_one(SiteConfigSettings)
        await pilot.pause()

        widget.display_config(config)

        auth_select = widget.query_one("#auth-type-select", Select)
        assert auth_select.value == auth_type
        if container_id is not None:
            assert not widget.query_one(f"#{container_id}").has_class("hidden")


@pytest.mark.asyncio
async def test_display_labels_are_not_valid_select_values(
    isolated_manager: SiteConfigManager,
) -> None:
    """Guards the *direction* of the fix: the labels must NOT be values.

    A future accidental re-reversal (or a double-swap) would make the
    display text itself acceptable as `.value`; this pins the option tuples
    to (label, value), not the other way around.
    """
    app = _Harness()
    async with app.run_test(size=(160, 48)) as pilot:
        widget = app.query_one(SiteConfigSettings)
        await pilot.pause()

        auth_select = widget.query_one("#auth-type-select", Select)
        for label in ("None", "Basic Auth", "Bearer Token", "API Key"):
            with pytest.raises(InvalidSelectValueError):
                auth_select.value = label


@pytest.mark.asyncio
async def test_load_site_config_worker_reaches_display_config(
    isolated_manager: SiteConfigManager,
) -> None:
    """End-to-end: the real `load_site_config` -> call_from_thread(display_config) path."""
    config = SiteConfig("api-key.example.com", {"auth_type": "api_key"})
    assert isolated_manager.save_config(config) is True

    app = _Harness()
    async with app.run_test(size=(160, 48)) as pilot:
        widget = app.query_one(SiteConfigSettings)
        await pilot.pause()

        widget.load_site_config("api-key.example.com")
        await app.workers.wait_for_complete()
        await pilot.pause()

        auth_select = widget.query_one("#auth-type-select", Select)
        assert auth_select.value == "api_key"


@pytest.mark.asyncio
async def test_load_site_config_worker_narrows_the_swallow_to_a_traceback(
    isolated_manager: SiteConfigManager,
) -> None:
    """The narrowed except must log a real traceback, not a bare str(e) line.

    Pre-fix, `load_site_config` wrapped BOTH the config fetch and the
    `call_from_thread(display_config, ...)` UI update in one bare
    `except Exception as e: logger.error(f"...: {str(e)}")` -- so a bug in
    `display_config` (like the reversed Select) logged one line with no
    stack trace and no way to tell it apart from a legitimate data-loading
    failure. This drives a synthetic `display_config` failure through the
    real worker and asserts the resulting log record actually carries an
    exception/traceback.
    """
    config = SiteConfig("boom.example.com", {"auth_type": "basic"})
    assert isolated_manager.save_config(config) is True

    app = _Harness()
    async with app.run_test(size=(160, 48)) as pilot:
        widget = app.query_one(SiteConfigSettings)
        await pilot.pause()

        def _boom(_config: Any) -> None:
            raise RuntimeError("synthetic display_config failure")

        widget.display_config = _boom  # type: ignore[method-assign]

        records: list[dict[str, Any]] = []
        sink_id = loguru_logger.add(
            lambda message: records.append(message.record), level="ERROR"
        )
        try:
            widget.load_site_config("boom.example.com")
            await app.workers.wait_for_complete()
            await pilot.pause()
        finally:
            loguru_logger.remove(sink_id)

        assert records, "expected the narrowed except to log something"
        assert any(record["exception"] is not None for record in records), (
            "the display_config failure must be logged with a traceback, "
            f"got: {[r['message'] for r in records]}"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
