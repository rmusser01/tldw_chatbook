"""TASK-34100.1 AC#5 (cross-cutting-14): a slow Next says what it is doing.

The 2026-10-02 review measured Nexts of 2.5-4 s, 3-6 s when choosing Full,
and up to 30 s on Voice. The only cue was nav buttons whose disabled state
changed colour alone. These tests cover four things. A Next that runs past
about 400 ms shows a busy line naming the work, with elapsed seconds on long
waits. A fast Next shows nothing, so nothing flickers. Disabled nav buttons
carry the terminal's dim attribute, not only a colour. Back and Forward over
the same provider reuse its model discovery instead of re-running it.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, RadioButton, RadioSet

from tldw_chatbook.UI.Wizards.first_run_setup_state import (
    STEP_MODEL,
    STEP_PROVIDER,
    STEP_VOICE,
    TRACK_QUICK,
)
from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
    FirstRunSetupWizard,
    ModelStep,
    ProviderStep,
    SetupWizardContainer,
    WelcomeStep,
)

pytestmark = pytest.mark.bootstrap_profile

_BUNDLE = (
    Path(__file__).resolve().parents[2] / "tldw_chatbook/css/tldw_cli_modular.tcss"
)


class _Host(App):
    def __init__(self, wizard: FirstRunSetupWizard) -> None:
        super().__init__()
        self._wizard = wizard

    def compose(self) -> ComposeResult:
        yield from ()

    async def on_mount(self) -> None:
        self.push_screen(self._wizard)


class _StyledHost(_Host):
    CSS_PATH = str(_BUNDLE)


def _wizard() -> FirstRunSetupWizard:
    app_instance = MagicMock()
    app_instance.app_config = {}
    return FirstRunSetupWizard(app_instance)


def _busy_text(wizard: FirstRunSetupWizard) -> str | None:
    """The busy line's text while it is shown, else None."""
    from textual.widgets import Static

    line = wizard.query_one("#setup-busy-status", Static)
    if not line.display or line.has_class("hidden"):
        return None
    return str(line.content)


def _slow_checkpoint(container: SetupWizardContainer, seconds: float) -> None:
    """Make every Next's checkpoint write take ``seconds`` (a slow disk)."""

    async def slow(_step_id: str) -> bool:
        await asyncio.sleep(seconds)
        return True

    container.persist_setup_checkpoint = slow


async def _until_settled(pilot, container, limit: float = 6.0) -> None:
    for _ in range(int(limit / 0.05)):
        if not container._advancing:
            return
        await pilot.pause(0.05)
    raise AssertionError("the Next never settled")


def _record_busy_lines(monkeypatch) -> list[tuple[float, str]]:
    """Record every text the busy line shows, with seconds since its start.

    Timing is measured from ``SetupBusyStatus.start`` itself: ``pilot.press``
    returns only once the app is idle, which can already be past 400 ms.
    """
    import time

    from tldw_chatbook.UI.Wizards.first_run_busy_status import SetupBusyStatus

    shown: list[tuple[float, str]] = []
    real_show = SetupBusyStatus._show

    def recording_show(self, text: str) -> None:
        started = self._started_at
        elapsed = time.monotonic() - started if started is not None else -1.0
        if text != self._shown:
            shown.append((elapsed, text))
        real_show(self, text)

    monkeypatch.setattr(SetupBusyStatus, "_show", recording_show)
    return shown


@pytest.mark.asyncio
async def test_a_fast_next_shows_no_busy_line(monkeypatch):
    shown = _record_busy_lines(monkeypatch)
    wizard = _wizard()
    app = _Host(wizard)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        _slow_checkpoint(container, 0.15)

        await pilot.press("ctrl+n")
        await _until_settled(pilot, container)
        await pilot.pause(0.2)

        assert [text for _elapsed, text in shown if text] == []
        assert _busy_text(wizard) is None


@pytest.mark.asyncio
async def test_choosing_full_names_the_work_then_counts_the_seconds(monkeypatch):
    shown = _record_busy_lines(monkeypatch)
    wizard = _wizard()
    app = _Host(wizard)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        wizard.query_one("#setup-track-full", RadioButton).value = True
        await pilot.pause(0.05)
        _slow_checkpoint(container, 3.0)

        await pilot.press("ctrl+n")
        for _ in range(60):
            if _busy_text(wizard):
                break
            await pilot.pause(0.05)
        assert _busy_text(wizard) == "Preparing the Full setup…"
        # The nav buttons are fenced while the line explains why.
        assert wizard.query_one("#wizard-next", Button).disabled

        for _ in range(60):
            if re.search(r" \d+ s$", _busy_text(wizard) or ""):
                break
            await pilot.pause(0.05)
        assert re.fullmatch(r"Preparing the Full setup… \d+ s", _busy_text(wizard) or "")

        await _until_settled(pilot, container)
        assert _busy_text(wizard) is None
        assert container.steps[container.current_step].config.id == STEP_PROVIDER

    texts = [text for _elapsed, text in shown if text]
    first_reveal = next(elapsed for elapsed, text in shown if text)
    first_count = next(elapsed for elapsed, text in shown if text.endswith(" s"))
    assert texts[0] == "Preparing the Full setup…"
    assert first_reveal >= 0.4, "the line must not flash on a quick Next"
    assert first_count >= 2.0
    assert shown[-1][1] == ""


@pytest.mark.asyncio
async def test_choosing_full_never_blocks_the_screen_on_local_discovery(monkeypatch):
    """The loop stays free while the localhost scan does its blocking setup.

    Review round 1: choosing Full lands on Provider, whose first show starts
    the localhost scan. Its storage admission and TLS client setup are
    synchronous file and SSL work. On the UI loop they froze Welcome for
    0.4-1.3 s after the Next itself had finished, so neither Provider nor the
    busy line (a timer) could paint. A slow path that awaits
    (``asyncio.sleep``) leaves the loop free and cannot catch that; this one
    blocks whichever thread runs it.

    Review round 2: the property is asserted directly (the scan ran off the
    main thread). The loop-gap bound is a second, looser check, measured with
    the garbage collector frozen: in a process that has already run other
    wizard apps, a GC pause alone passed the old 0.5 s bar.
    """
    import gc
    import threading
    import time

    started = threading.Event()
    ran_on: list[threading.Thread] = []

    async def blocking_discovery(*_args, **_kwargs):
        ran_on.append(threading.current_thread())
        started.set()
        time.sleep(1.0)  # synchronous, like admission and SSL context setup
        return ()

    monkeypatch.setattr(
        "tldw_chatbook.Chat.local_server_discovery.discover_local_servers",
        blocking_discovery,
    )
    wizard = _wizard()
    app = _Host(wizard)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        wizard.query_one("#setup-track-full", RadioButton).value = True
        await pilot.pause(0.05)

        gc.collect()
        gc.disable()
        try:
            container.action_next()
            gaps: list[float] = []
            last = time.monotonic()
            deadline = last + 1.6
            while time.monotonic() < deadline:
                await asyncio.sleep(0.02)
                now = time.monotonic()
                gaps.append(now - last)
                last = now
        finally:
            gc.enable()

        assert started.is_set(), "Provider never started the localhost scan"
        assert ran_on and ran_on[0] is not threading.main_thread(), (
            "the localhost scan ran on the UI thread"
        )
        # Below the scan's own 1.0 s block: a blocked loop shows the whole of it.
        assert max(gaps) < 0.9, f"the UI loop was blocked for {max(gaps):.2f} s"
        assert isinstance(container.steps[container.current_step], ProviderStep)
        provider = container.steps[container.current_step]
        for _ in range(60):
            if provider._local_discovery_state == "complete":
                break
            await pilot.pause(0.05)
        assert provider._local_discovery_state == "complete"


@pytest.mark.asyncio
async def test_each_step_names_its_own_work():
    from tldw_chatbook.UI.Wizards.first_run_busy_status import busy_label_for

    wizard = _wizard()
    app = _Host(wizard)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        by_id = {step.config.id: step for step in container.steps}

        welcome = by_id["welcome"]
        assert isinstance(welcome, WelcomeStep)
        assert busy_label_for(welcome) == "Preparing the Quick setup…"
        assert busy_label_for(by_id[STEP_VOICE]) == "Saving voice settings…"
        assert busy_label_for(by_id[STEP_MODEL]) == "Saving the provider and model…"
        assert busy_label_for(by_id["tools"]) == "Saving Tools settings…"


@pytest.mark.asyncio
async def test_disabled_nav_buttons_carry_the_dim_attribute():
    """A colour-only disabled state vanished in the reviewers' terminals."""
    wizard = _wizard()
    app = _StyledHost(wizard)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        for selector in ("#wizard-back", "#wizard-next", "#wizard-cancel"):
            button = wizard.query_one(selector, Button)
            styles = {}
            for disabled in (False, True):
                button.disabled = disabled
                await pilot.pause(0.05)
                label = next(
                    segment.style
                    for segment in button.render_line(1)
                    if segment.text.strip()
                )
                styles[disabled] = label
            assert styles[True].dim, f"{selector} disabled is not dim"
            assert not styles[True].bold, f"{selector} disabled is still bold"
            assert not styles[False].dim, f"{selector} enabled is dim"


@pytest.mark.parametrize("outcome", ["models", "failure"])
@pytest.mark.asyncio
async def test_back_and_forward_reuse_the_providers_model_discovery(outcome: str):
    """A model list is fetched once per provider identity.

    ``[models]``: Back to Provider and Forward again reuse the list (review:
    "reruns on every Forward/Back"). ``[failure]``: moving between the steps
    after Model does not ask the server again. Going back to Provider does
    (see the next test): its notice says "Check it's running, then
    continue", so continuing must check. Model's Retry also asks again.

    Both cases already passed on the base code, which reused a completed
    discovery; they stay as regression pins for that reuse.
    """
    wizard = _wizard()
    wizard.app_instance.app_config = {
        "api_settings": {"custom": {"api_url": "https://cache.example.test/v1"}}
    }
    scope_service = MagicMock()
    if outcome == "models":
        scope_service.discover_models = AsyncMock(
            return_value=_typed_result("custom", "cached-model")
        )
    else:
        scope_service.discover_models = AsyncMock(side_effect=OSError("down"))
    wizard.app_instance.llm_provider_catalog_scope_service = scope_service
    app = _Host(wizard)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        provider_index = container._step_index_for_id(STEP_PROVIDER)
        model_index = container._step_index_for_id(STEP_MODEL)
        container.show_step(provider_index)
        provider = container.steps[provider_index]
        assert isinstance(provider, ProviderStep)
        provider.select_provider("custom")
        for _ in range(40):
            if provider._selected_discovery_state in {"complete", "failed"}:
                break
            await pilot.pause(0.05)

        model = container.steps[model_index]
        assert isinstance(model, ModelStep)

        async def model_settled() -> None:
            if outcome == "models":
                await _until_models(pilot, model, "cached-model")
            else:
                await _until_selector(pilot, model, "#setup-model-connection-failed")

        await container._advance()
        await model_settled()
        # A failure seen on Provider is asked again by its Next.
        asked = scope_service.discover_models.await_count
        assert asked == (1 if outcome == "models" else 2)

        if outcome == "models":
            # Back to Provider, then Forward again: same provider identity.
            await pilot.press("ctrl+b")
            await pilot.pause(0.3)
            assert container.current_step == provider_index
            await container._advance()
            await model_settled()

        # Model -> Voice -> Back to Model: still the same identity.
        await container._advance()
        await pilot.pause(0.2)
        await pilot.press("ctrl+b")
        await pilot.pause(0.3)
        assert container.current_step == model_index
        await model_settled()
        assert scope_service.discover_models.await_count == asked

        if outcome == "failure":
            # Asking again is still one press away.
            model.query_one("#setup-model-retry", Button).press()
            for _ in range(40):
                if scope_service.discover_models.await_count == asked + 1:
                    break
                await pilot.pause(0.05)
            assert scope_service.discover_models.await_count == asked + 1


@pytest.mark.parametrize("fixed_via", ["back_then_next", "test_connection"])
@pytest.mark.asyncio
async def test_a_server_fixed_after_a_failed_discovery_is_seen_on_continue(
    fixed_via: str,
):
    """Review round 2: a failed model list must not outlive the user's fix.

    The server is down, so discovery fails. The user starts the server and
    continues from Provider, as its notice asks ("Check it's running, then
    continue"). ``back_then_next``: the failure showed on Model, the user
    went Back and pressed Next. ``test_connection``: the failure showed on
    Provider, and a Test connection succeeded before Next. Either way Next
    must ask the server again, so Model lists its models and does not hold
    Next behind "The server couldn't be reached… Continue anyway?".
    """
    from tldw_chatbook.UI.Screens.settings_endpoint_probe import (
        SettingsEndpointProbeOutcome,
    )

    wizard = _wizard()
    wizard.app_instance.app_config = {
        "api_settings": {"custom": {"api_url": "https://cache.example.test/v1"}}
    }
    server_up = False

    async def discover(**_kwargs):
        if not server_up:
            raise OSError("down")
        return _typed_result("custom", "now-up")

    scope_service = MagicMock()
    scope_service.discover_models = AsyncMock(side_effect=discover)
    wizard.app_instance.llm_provider_catalog_scope_service = scope_service
    app = _Host(wizard)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        provider_index = container._step_index_for_id(STEP_PROVIDER)
        model_index = container._step_index_for_id(STEP_MODEL)
        container.show_step(provider_index)
        provider = container.steps[provider_index]
        assert isinstance(provider, ProviderStep)
        provider._probe = AsyncMock(
            return_value=SettingsEndpointProbeOutcome(
                state="reachable", summary="reachable (1 models)", model_ids=("now-up",)
            )
        )
        provider.select_provider("custom")
        for _ in range(40):
            if provider._selected_discovery_state == "failed":
                break
            await pilot.pause(0.05)
        assert provider._selected_discovery_state == "failed"
        model = container.steps[model_index]
        assert isinstance(model, ModelStep)

        if fixed_via == "back_then_next":
            await container._advance()
            await _until_selector(pilot, model, "#setup-model-connection-failed")
            server_up = True
            await pilot.press("ctrl+b")
            await pilot.pause(0.3)
            assert container.current_step == provider_index
        else:
            server_up = True
            provider.query_one("#setup-provider-test", Button).press()
            for _ in range(40):
                if provider._probe.await_count:
                    break
                await pilot.pause(0.05)
            await pilot.pause(0.1)
        asked_before = scope_service.discover_models.await_count

        await container._advance()
        assert container.current_step == model_index
        await _until_models(pilot, model, "now-up")

        assert scope_service.discover_models.await_count > asked_before
        assert model.current_probe_failure() == ""
        assert model.confirm_before_advance() is None


async def _until_selector(pilot, model: ModelStep, selector: str) -> None:
    for _ in range(80):
        if list(model.query(selector)):
            return
        await pilot.pause(0.05)
    raise AssertionError(f"{selector} never rendered")


def _typed_result(provider: str, *model_ids: str):
    from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
        DiscoveredModel,
        ModelDiscoveryResult,
    )

    return ModelDiscoveryResult(
        provider=provider,
        provider_list_key=provider,
        endpoint_fingerprint=f"https://{provider}.example.test/v1",
        status="success",
        models=tuple(
            DiscoveredModel(
                provider=provider,
                provider_list_key=provider,
                model_id=model_id,
                display_name=model_id,
                source="runtime_discovered",
                endpoint_fingerprint=f"https://{provider}.example.test/v1",
                discovered_at="2026-10-03T00:00:00Z",
            )
            for model_id in model_ids
        ),
    )


async def _until_models(pilot, model: ModelStep, model_id: str) -> None:
    for _ in range(60):
        radios = model.query_one("#setup-model-choice", RadioSet).query(RadioButton)
        if any(getattr(button, "_model_id", "") == model_id for button in radios):
            return
        await pilot.pause(0.05)
    raise AssertionError(f"{model_id} never rendered")
