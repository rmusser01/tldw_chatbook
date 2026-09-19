"""First-run readiness never reads borrowed credentials on the UI thread."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.widgets import Input, Static

from Tests.Wizards.test_first_run_setup_wizard import (
    _provider_step,
    _StepHost,
)
from tldw_chatbook.LLM_Calls import anthropic_subscription as subscription
from tldw_chatbook.UI.Wizards.first_run_setup_state import (
    FirstRunProviderDraft,
    ProviderCredentialDraft,
    build_first_run_provider_commit,
)


@pytest.fixture
def credential_reader(monkeypatch, tmp_path):
    cache = subscription._SubscriptionReadinessCache()
    monkeypatch.setattr(subscription, "_SUBSCRIPTION_READINESS_CACHE", cache)
    monkeypatch.setattr(
        subscription, "DEFAULT_CREDENTIALS_PATH", tmp_path / "credential"
    )
    reader = SimpleNamespace(
        entered=threading.Event(),
        release=threading.Event(),
        threads=[],
        credential=subscription.SubscriptionCredential(access_token="private-token"),
        cache=cache,
    )
    ui_thread = threading.get_ident()

    def read(*args, **kwargs):
        reader.threads.append(threading.get_ident())
        reader.entered.set()
        # Fail the UI-thread assertion promptly on the old synchronous path.
        if threading.get_ident() != ui_thread:
            assert reader.release.wait(5)
        return reader.credential

    monkeypatch.setattr(subscription, "read_claude_code_credential", read)
    yield reader
    reader.release.set()
    if cache._worker is not None:
        cache._worker.join(timeout=3)
        assert not cache._worker.is_alive()


def _subscription_step():
    wizard = SimpleNamespace(
        app_instance=SimpleNamespace(
            app_config={
                "api_settings": {
                    "anthropic": {"auth_source": "claude_subscription"},
                },
            }
        ),
        note_key_entered=MagicMock(),
        stage_provider_setup=MagicMock(return_value=True),
        rerun=False,
    )
    step = _provider_step(
        wizard=wizard,
        discover=AsyncMock(return_value=()),
    )
    return step, _StepHost(step)


async def _wait_for_status(status, text):
    async with asyncio.timeout(3):
        while text not in str(status.renderable):
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["ready", "missing", "expired"])
async def test_subscription_selection_stays_responsive_and_refreshes_without_click(
    credential_reader, outcome
):
    reader = credential_reader
    if outcome == "missing":
        reader.credential = None
    elif outcome == "expired":
        reader.credential = subscription.SubscriptionCredential(
            access_token="private-token", expires_at_ms=1
        )
    step, app = _subscription_step()
    async with app.run_test(size=(120, 40)) as pilot:
        step.select_provider("anthropic")
        await pilot.pause()
        assert reader.entered.is_set()
        assert reader.threads == [reader.threads[0]]
        assert threading.get_ident() not in reader.threads
        status = step.query_one("#setup-provider-key-status", Static)
        assert "Checking Claude subscription" in str(status.renderable)
        allowed, explanation = await step.commit()
        assert not allowed
        assert "Checking Claude subscription" in explanation
        assert "API key required" not in explanation
        # A mounted control and the UI loop remain usable while the read waits.
        key_input = step.query_one("#setup-provider-api-key", Input)
        key_input.focus()
        await pilot.pause()
        assert key_input.has_focus
        assert not reader.release.is_set()
        reader.release.set()
        expected = {
            "ready": "Ready (Claude subscription)",
            "missing": "No Claude subscription credential found",
            "expired": "Claude subscription credential is expired",
        }[outcome]
        await _wait_for_status(status, expected)
        allowed, explanation = await step.commit()
        assert allowed is (outcome == "ready")
        if allowed:
            draft = step.wizard.stage_provider_setup.call_args.args[0]
            assert draft.credential.source == "none"
            assert "private-token" not in repr(draft)
            step.wizard.note_key_entered.assert_not_called()
        else:
            assert expected in explanation
        assert "private-token" not in app.export_screenshot()
        assert len(reader.threads) == 1


@pytest.mark.asyncio
async def test_completed_subscription_read_does_not_repaint_another_provider(
    credential_reader,
):
    step, app = _subscription_step()
    async with app.run_test(size=(120, 40)) as pilot:
        step.select_provider("anthropic")
        await pilot.pause()
        assert threading.get_ident() not in credential_reader.threads
        step.select_provider("openai")
        await pilot.pause()
        status = step.query_one("#setup-provider-key-status", Static)
        expected = str(status.renderable)
        credential_reader.release.set()
        await asyncio.sleep(0.35)
        assert step.selected_provider_key == "openai"
        assert str(status.renderable) == expected
        assert "Claude" not in expected


@pytest.mark.asyncio
async def test_subscription_expiry_refreshes_within_cache_lifetime(credential_reader):
    credential_reader.release.set()
    step, app = _subscription_step()
    async with app.run_test(size=(120, 40)) as pilot:
        step.select_provider("anthropic")
        await pilot.pause()
        status = step.query_one("#setup-provider-key-status", Static)
        await _wait_for_status(status, "Ready (Claude subscription)")
        credential_reader.cache._expires_at_ms = 1
        await _wait_for_status(status, "Claude subscription credential is expired")
        assert not (await step.commit())[0]
        assert len(credential_reader.threads) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("leave", ["hide", "unmount"])
async def test_late_credential_completion_does_not_repaint_inactive_step(
    credential_reader, leave
):
    step, app = _subscription_step()
    async with app.run_test(size=(120, 40)) as pilot:
        step.select_provider("anthropic")
        await pilot.pause()
        status = step.query_one("#setup-provider-key-status", Static)
        assert "Checking Claude subscription" in str(status.renderable)
        if leave == "hide":
            step.display = False
        else:
            await step.remove()
        await pilot.pause()
        expected = str(status.renderable)
        credential_reader.release.set()
        await asyncio.sleep(0.35)
        step._refresh_subscription_readiness()
        assert str(status.renderable) == expected
        if leave == "hide":
            step.display = True
            await _wait_for_status(status, "Ready (Claude subscription)")


@pytest.mark.asyncio
async def test_cached_subscription_is_rechecked_while_step_stays_open(
    credential_reader,
):
    credential_reader.release.set()
    step, app = _subscription_step()
    async with app.run_test(size=(120, 40)) as pilot:
        step.select_provider("anthropic")
        await pilot.pause()
        status = step.query_one("#setup-provider-key-status", Static)
        await _wait_for_status(status, "Ready (Claude subscription)")
        credential_reader.credential = None
        credential_reader.cache._completed_at = 0
        await _wait_for_status(status, "No Claude subscription credential found")
        assert not (await step.commit())[0]
        assert len(credential_reader.threads) == 2
        assert threading.get_ident() not in credential_reader.threads


def test_first_run_commit_builder_does_not_read_or_persist_borrowed_token(
    credential_reader,
):
    draft = FirstRunProviderDraft(
        provider="anthropic",
        endpoint="",
        credential=ProviderCredentialDraft("none", ""),
    )
    mutation = build_first_run_provider_commit(
        draft,
        "claude-test",
        {"api_settings": {"anthropic": {"auth_source": "claude_subscription"}}},
    )
    assert threading.get_ident() not in credential_reader.threads
    assert "private-token" not in repr(mutation)
    assert "api_key" not in mutation.section_values.get("api_settings.anthropic", {})
    assert mutation.section_values["chat_defaults"]["provider"] == "anthropic"
