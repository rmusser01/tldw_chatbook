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
    """Save selected is held, then refused (``outcome`` is a status) or raises.
    ``clear`` is "ok", "fails" (Clear raises) or "held" (Clear is still
    writing when the save fails, and succeeds after)."""

    def __init__(self, *args, outcome: str, clear: str = "ok", **kwargs):
        super().__init__(*args, **kwargs)
        self.outcome = outcome
        self.clear = clear
        self.clear_release = asyncio.Event()

    async def clear_discovered_models(self, **kwargs):
        self.clear_calls.append(kwargs)
        if self.clear == "held":
            await self.clear_release.wait()
        if self.clear == "fails":
            raise OSError("the model cache is read-only")

    async def persist_discovered_models_to_settings(self, **kwargs):
        self.persist_calls.append(kwargs)
        self.started.set()
        await self.release.wait()
        # A held Clear resumes on a later loop step, after the screen has
        # handled this save's outcome.
        self.clear_release.set()
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

    Returns the app, the toasts as (message, severity), the picker syncs
    as (provider, reload) and the ids still checked at the end.
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
        checked = [
            str(model_id)
            for listing in screen.query("#settings-discovered-models-list").results(
                SelectionList
            )
            for model_id in listing.selected
        ]
    return app, toasts, reloads, checked


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
    app, toasts, reloads, _checked = await _save_then_press(
        monkeypatch, scope, interrupt
    )

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
    ],
    ids=["ambiguous", "error"],
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
    provider" then read as the form's new one). A raised save's next step
    depends on what took the line: see the test below."""
    scope = _RefusingSaveScope(
        "openai", catalog=("gpt-4o",), served=(_SERVED,), outcome=outcome
    )
    app, toasts, _reloads, _checked = await _save_then_press(
        monkeypatch, scope, interrupt
    )

    assert scope.saved == ()
    assert _SERVED not in app.providers_models.get("OpenAI", ())
    warnings = [message for message, severity in toasts if severity == "warning"]
    assert warnings == [expected], toasts


_CLEAR = "#settings-clear-discovered-provider-models"
_RESELECT = (
    "Select them again (Discover first if they are no longer listed), then press "
    "Save selected"
)


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize(
    ("interrupt", "clear", "next_step", "still_checked"),
    [
        pytest.param(_INTERRUPTS[0], "ok", _RESELECT, [], id="discover"),
        pytest.param(_CLEAR, "ok", _RESELECT, [], id="clear"),
        pytest.param(_CLEAR, "fails", "Try Save again", [_SERVED], id="failed-clear"),
        pytest.param(_CLEAR, "held", _RESELECT, [], id="held-clear"),
        pytest.param(
            _SWITCH_PROVIDER,
            "ok",
            "Switch back to OpenAI, Discover and select them again, then press "
            "Save selected",
            [],
            id="provider-change",
        ),
    ],
)
async def test_save_selected_raised_after_another_action_names_a_true_next_step(
    request, monkeypatch, interrupt, clear, next_step, still_checked
):
    """Capture fix 3 round 4, finding 1 (the raise row above, rewritten on
    purpose): round 3 picked "Discover and select them again" from the
    revision alone, claiming every action that takes the line drops the
    checked rows. A Clear that fails keeps them listed and checked, so the
    toast told the user to re-discover rows still on screen; and after a
    provider change "Discover" ran against the form's new provider. The step
    now follows the rows and the provider the form holds when the save
    fails.

    Capture fix 3 round 5, finding 1 ("held-clear"): a Clear still writing
    when the save failed has not dropped the rows yet, so the step read "Try
    Save again"; the Clear then succeeded and took the list and Save selected
    away. A Clear in flight counts as dropping the rows."""
    scope = _RefusingSaveScope(
        "openai",
        catalog=("gpt-4o",),
        served=(_SERVED,),
        outcome="raise",
        clear=clear,
    )
    logs: list[str] = []
    handler = logging.Handler(logging.WARNING)
    handler.emit = lambda record: logs.append(record.getMessage())
    screen_logger = logging.getLogger(settings_screen.__name__)
    screen_logger.addHandler(handler)
    try:
        app, toasts, _reloads, checked = await _save_then_press(
            monkeypatch, scope, interrupt
        )
    finally:
        screen_logger.removeHandler(handler)

    assert scope.saved == ()
    assert _SERVED not in app.providers_models.get("OpenAI", ())
    assert checked == still_checked
    warnings = [message for message, severity in toasts if severity == "warning"]
    assert warnings == [
        (
            f"OpenAI: Could not save the discovered models (OSError). {next_step}; "
            "if it keeps failing, check that your config file is writable. "
            "Details are in Logs (F3)."
        )
    ], toasts
    assert any("persistence failed: OSError" in line for line in logs), logs
