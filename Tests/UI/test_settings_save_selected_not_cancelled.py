"""Settings ▸ Providers & Models: Save selected is never cancelled (review of
#3057, R2 "outside the diff", ruling C9).

Discover, Save selected and Clear shared one exclusive worker group, so a
press on Discover or Clear while Save selected was writing cancelled a
durable write.
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from textual.widgets import Button, SelectionList

from Tests.private_profile import private_profile_test
from Tests.UI.test_settings_default_model_picker import (
    _app,
    _CatalogScope,
    _open_providers,
    _settle,
)
from Tests.UI.test_settings_discovery_listing_dropped import (
    _SERVED,
    _SIZE,
    _discover,
)
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
    PersistenceResult,
)
from tldw_chatbook.UI.Screens import settings_screen


class _HeldSaveScope(_CatalogScope):
    """Save selected waits until the test lets it finish."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.saved: tuple[str, ...] = ()

    async def persist_discovered_models_to_settings(self, **kwargs):
        self.persist_calls.append(kwargs)
        self.started.set()
        await self.release.wait()
        self.saved = tuple(kwargs["model_ids"])
        return PersistenceResult(
            provider="openai",
            provider_list_key="OpenAI",
            status="saved",
            saved_model_ids=self.saved,
        )


class _RefusingSaveScope(_HeldSaveScope):
    """Save selected is held, then refused (``outcome`` is a status) or raises."""

    def __init__(self, *args, outcome: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.outcome = outcome

    async def persist_discovered_models_to_settings(self, **kwargs):
        self.persist_calls.append(kwargs)
        self.started.set()
        await self.release.wait()
        if self.outcome == "raise":
            raise OSError("config.toml is read-only")
        return PersistenceResult(
            provider="openai",
            provider_list_key=None,
            status=self.outcome,
            message="Could not save discovered models to the providers list.",
        )


_INTERRUPTS = [
    "#settings-discover-provider-models",
    "#settings-clear-discovered-provider-models",
]
# Capture fix 3 round 3: a provider change also takes the line (it resets
# discovery) without cancelling the save, so the toast must name the
# provider the save was for, not the one the form now holds.
_SWITCH_PROVIDER = "llama_cpp"


async def _save_then_press(monkeypatch, scope, interrupt):
    """Press Save selected, then ``interrupt`` (a button, or a provider to
    switch to) while the write is held.

    Returns the app, the toasts as (message, severity) and the picker
    syncs as (provider, reload).
    """
    app = _app("openai", "gpt-4o", catalog=("gpt-4o",), served=(_SERVED,))
    app.llm_provider_catalog_scope_service = scope
    host = _SettingsCssHarness(app, "settings")
    toasts: list[tuple[str, str]] = []
    reloads: list[tuple[object, bool]] = []

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        await _discover(host, pilot, screen)
        screen.query_one("#settings-discovered-models-list", SelectionList).select_all()
        await pilot.pause()
        monkeypatch.setattr(
            host,
            "notify",
            lambda message, severity="information", **_: toasts.append(
                (message, severity)
            ),
        )
        sync_picker = screen._sync_model_picker

        def spy(provider=None, *, reload=False):
            reloads.append((provider, reload))
            sync_picker(provider, reload=reload)

        monkeypatch.setattr(screen, "_sync_model_picker", spy)

        screen.query_one("#settings-save-discovered-provider-models", Button).press()
        await asyncio.wait_for(scope.started.wait(), timeout=10)
        if interrupt.startswith("#"):
            screen.query_one(interrupt, Button).press()
        else:
            screen._apply_provider_value_change(interrupt)
        await pilot.pause()
        await pilot.pause()
        scope.release.set()
        await _settle(host, pilot)
    return app, toasts, reloads


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("interrupt", _INTERRUPTS)
async def test_save_selected_is_not_cancelled_by_discover_or_clear(
    request, monkeypatch, interrupt
):
    """Fix round 1, finding 13: the write landed, but with Discover or Clear
    pressed meanwhile it said nothing and the Default model picker went on
    listing the ids as unsaved; after Clear the only toast read as if the
    save had been dropped."""
    scope = _HeldSaveScope("openai", catalog=("gpt-4o",), served=(_SERVED,))
    app, toasts, reloads = await _save_then_press(monkeypatch, scope, interrupt)

    assert scope.saved == (_SERVED,)
    assert _SERVED in app.providers_models["OpenAI"]
    # Capture fix 3 round 1, finding 4, rewritten on purpose: the line
    # belongs to the Discover or Clear pressed since, so the toast is the
    # one report and says what the line would have said.
    receipt = ("Saved 1 discovered model to OpenAI.", "information")
    assert toasts.count(receipt) == 1, toasts
    assert ("openai", True) in reloads, reloads


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("interrupt", [*_INTERRUPTS, _SWITCH_PROVIDER])
@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (
            "ambiguous_provider_key",
            (
                "OpenAI: Multiple provider entries match this provider. Rename "
                "or remove duplicates before saving discovered models."
            ),
        ),
        ("error", "OpenAI: Could not save discovered models to the providers list."),
        (
            "raise",
            (
                "OpenAI: Could not save the discovered models (OSError). Discover "
                "and select them again, then press Save selected; if it keeps "
                "failing, check that your config file is writable. Details are "
                "in Logs (F3)."
            ),
        ),
    ],
    ids=["ambiguous", "error", "raise"],
)
async def test_save_selected_refused_after_discover_or_clear_still_reports(
    request, monkeypatch, interrupt, outcome, expected
):
    """Capture fix 3 round 2, finding 1: with Discover or Clear pressed while
    Save selected was writing, a save that failed said nothing at all (and a
    raised one logged nothing), so the user believed the models were saved.
    Its one report is a warning toast in the line's words.

    Capture fix 3 round 3, rewritten on purpose: the toast names the provider
    the save was for (a provider change also takes the line, and "this
    provider" then read as the form's new one), and a raised save's next step
    can be followed: Discover, Clear and a provider change all drop the
    checked rows, so "Try Save again" could not be."""
    scope = _RefusingSaveScope(
        "openai", catalog=("gpt-4o",), served=(_SERVED,), outcome=outcome
    )
    logs: list[str] = []
    handler = logging.Handler(logging.WARNING)
    handler.emit = lambda record: logs.append(record.getMessage())
    screen_logger = logging.getLogger(settings_screen.__name__)
    screen_logger.addHandler(handler)
    try:
        app, toasts, _reloads = await _save_then_press(monkeypatch, scope, interrupt)
    finally:
        screen_logger.removeHandler(handler)

    assert scope.saved == ()
    assert _SERVED not in app.providers_models.get("OpenAI", ())
    warnings = [message for message, severity in toasts if severity == "warning"]
    assert warnings == [expected], toasts
    if outcome == "raise":
        assert any("persistence failed: OSError" in line for line in logs), logs
