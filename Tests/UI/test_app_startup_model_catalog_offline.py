"""Full-app boots never reach the ADR-020 catalog network seam (task-16198).

Incident this pins: ``Tests/UI/test_product_maturity_phase3_knowledge_entry.py
::test_study_screen_consumes_pending_initial_section`` failed on pristine dev
with the egress guard's teardown error naming ``104.18.3.115:443`` and
``104.18.2.115:443`` — openrouter.ai's two A records (httpx tries each in
turn). Chain: ``TldwCli.on_mount`` → ``_schedule_startup_model_catalog_refresh``
→ worker ``_refresh_model_catalogs`` →
``LocalLLMProviderCatalogService.refresh_stale_configured_providers``, which
exempts OpenRouter from the missing-credentials skip and fetches
``https://openrouter.ai/api/v1/models`` keylessly whenever the loaded settings
carry ``refresh_consent_recorded = true``. The per-test sandbox defaults
consent off, so the leak only fired when consented settings leaked into the
process — making the red intermittent and environment-shaped.

This test drives the WORST CASE on purpose: consented settings pinned into
the app module, a real ``TldwCli`` boot, and the startup schedule + its
worker awaited to completion — then asserts the guard recorded zero egress
attempts. It is green because ``Tests/UI/conftest.py``'s autouse
``_disable_model_catalog_refresh`` pins the seam shut; with that fixture
removed, this test reproduces the incident byte-for-byte (blocked
``socket.connect`` to openrouter.ai's A records at teardown).
"""

from __future__ import annotations

import logging

import pytest

import tldw_chatbook.app as app_module
from tldw_chatbook.app import TldwCli

from Tests import network_guard


def _pin_consented_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Overlay consent-on/auto-refresh-on onto the real sandbox settings."""
    real_load_settings = app_module.load_settings

    def load_settings_consented():
        settings = dict(real_load_settings())
        section = dict(settings.get("model_catalog") or {})
        section["refresh_consent_recorded"] = True
        section["auto_refresh_enabled"] = True
        settings["model_catalog"] = section
        return settings

    monkeypatch.setattr(app_module, "load_settings", load_settings_consented)


def _disable_splash(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable only the production splash setting."""
    real_get_cli_setting = app_module.get_cli_setting

    def get_cli_setting_without_splash(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return real_get_cli_setting(section, key, default)

    monkeypatch.setattr(app_module, "get_cli_setting", get_cli_setting_without_splash)


async def _close_production_app(app: TldwCli) -> None:
    """Release production-app resources even when the assertion fails."""
    try:
        if app._rich_log_handler:
            await app._rich_log_handler.stop_processor()
            logging.getLogger().removeHandler(app._rich_log_handler)
            app._rich_log_handler.close()
        await app.on_shutdown_request()
        await app.on_unmount()
    except Exception:
        pass


@pytest.mark.asyncio
async def test_full_boot_with_consented_settings_makes_no_catalog_egress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_splash(monkeypatch)
    _pin_consented_settings(monkeypatch)
    app = TldwCli()
    app.app_config["_first_run"] = False
    app.app_config.setdefault("first_run", {})["setup_completed"] = True
    app._initial_tab_value = "home"

    try:
        async with app.run_test() as pilot:
            # Wait for on_mount to reach the startup schedule. With consent
            # pinned on, this takes the worker branch (not the consent modal).
            for _ in range(1500):
                if getattr(app, "_startup_model_catalog_refresh_scheduled", False):
                    break
                await pilot.pause(0.02)
            else:
                raise AssertionError(
                    "startup model catalog refresh was never scheduled; the "
                    "seam this test pins was not exercised."
                )

            # Let the refresh worker drain. Textual removes finished workers,
            # so "no worker left in the group" is completion. Do NOT use
            # workers.wait_for_complete(): the scheduler-loop worker never ends.
            for _ in range(1500):
                if not any(
                    worker.group == "model-catalog-refresh" for worker in app.workers
                ):
                    break
                await pilot.pause(0.02)
            else:
                raise AssertionError("model-catalog-refresh worker never finished.")
            await pilot.pause(0.2)

            attempts = network_guard.blocked_attempts()
            assert attempts == (), (
                "full-app boot reached the catalog network seam despite the "
                f"suite's offline stub: {attempts!r} — see task-16198; the "
                "keyless OpenRouter fetch must never run in tests."
            )
    finally:
        await _close_production_app(app)
