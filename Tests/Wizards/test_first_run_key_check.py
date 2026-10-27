"""TASK-34100.2 AC#3, AC#5, AC#8: setup shows the provider's own key verdict.

Live (pass 1, g2-r2-orbad/01, g2-v1-gem/02): the Provider step's Enter ran
Settings' OpenAI-compatible probe. For a wrong OpenRouter key it printed '✓
Reached the server' (OpenRouter lists models for any key); for Gemini it
printed 'Enter a valid endpoint before testing.' under 'Key staged -- it will
be checked when you continue. Connection testing is unavailable for this
provider.' The shared discovery already checks these keys (Google's native
list, OpenRouter's /api/v1/key, Hugging Face's whoami-v2, Anthropic's
authenticated listing); Enter now runs it and shows its verdict, a
public-listing provider without a key endpoint (NVIDIA NIM, Novita) reads
'key not checked' and never ✓, and the Model step and Summary use the same
words.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.widgets import Input, Static

from Tests.Wizards.test_first_run_setup_wizard import (
    _provider_step,
    _StepHost,
    _typed_model_discovery_result,
)
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
    ModelDiscoveryError,
    ModelDiscoveryResult,
)
from tldw_chatbook.UI.Wizards import first_run_key_check as key_check

# The mounted Summary reads config through the participant admission, which
# the per-test redirect refuses (TE-1: without this the Summary wiring test
# failed every time in the normal harness), as test_first_run_setup_wizard.py.
pytestmark = pytest.mark.bootstrap_profile


def _verified(provider: str, *models: str) -> ModelDiscoveryResult:
    from dataclasses import replace

    return replace(_typed_model_discovery_result(provider, *models), key_verified=True)


def _rejected(provider: str, message: str, reason: str) -> ModelDiscoveryResult:
    return ModelDiscoveryResult(
        provider=provider,
        provider_list_key=provider,
        endpoint_fingerprint=None,
        status="error",
        error=ModelDiscoveryError(
            kind="missing_credentials",
            message=message,
            recovery_hint="Check the API key configured for this provider.",
            category="unauthorized",
            provider_reason=reason,
        ),
    )


#: What a fresh profile's loaded config carries for the engine presets (the
#: discovery identity needs the API base).
_PRESET_CONFIG = {
    "api_settings": {"nvidia": {"api_base_url": "https://integrate.api.nvidia.com/v1"}}
}


def _wizard_with(result: ModelDiscoveryResult) -> SimpleNamespace:
    scope = MagicMock()
    scope.discover_models = AsyncMock(return_value=result)
    app_instance = MagicMock(app_config=_PRESET_CONFIG)
    app_instance.llm_provider_catalog_scope_service = scope
    return SimpleNamespace(
        app_instance=app_instance,
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )


TYPED_KEY = "typed-test-key-0123456789"


def _assert_the_typed_key_reached_discovery(scope, provider: str) -> None:
    """Pass 3 review round 1 (TE-3): the shared discovery is a mock here, so
    a verdict alone proves only the wizard's copy. Pin the seam: the check
    ran for this provider with the key just typed (a draft credential),
    never a saved or empty one."""
    calls = [call.kwargs for call in scope.discover_models.await_args_list]
    assert calls, "the shared discovery was never asked"
    checked = [kwargs for kwargs in calls if kwargs.get("provider") == provider]
    assert checked, calls
    staged = checked[-1]["staged_settings"]["api_settings"][provider]
    assert staged["credential_source"] == "draft"
    assert staged["api_key"] == TYPED_KEY
    assert checked[-1]["use_shared_cache"] is False


async def _enter_key(provider: str, result: ModelDiscoveryResult):
    wizard = _wizard_with(result)
    probe = AsyncMock()
    step = _provider_step(wizard=wizard, probe=probe)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider(provider)
        await pilot.pause()
        key_input = step.query_one("#setup-provider-api-key", Input)
        key_input.value = TYPED_KEY
        await pilot.pause()
        staged = str(step.query_one("#setup-provider-key-status", Static).renderable)
        key_input.focus()
        await pilot.press("enter")
        for _ in range(20):
            await pilot.pause(0.05)
        status = str(step.query_one("#setup-provider-probe-status", Static).renderable)
    scope = wizard.app_instance.llm_provider_catalog_scope_service
    _assert_the_typed_key_reached_discovery(scope, provider)
    return staged, status, probe, scope


@pytest.mark.asyncio
async def test_gemini_enter_checks_the_key_with_google() -> None:
    staged, status, probe, scope = await _enter_key(
        "google", _verified("google", "gemini-flash-latest", "gemini-3.8-flash")
    )

    assert status == "✓ Google accepted this key — found 2 model(s)."
    assert "Connection testing is unavailable" not in staged
    assert "Enter a valid endpoint" not in status
    assert scope.discover_models.await_count >= 1
    probe.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_wrong_openrouter_key_reads_openrouters_reason() -> None:
    staged, status, probe, _scope = await _enter_key(
        "openrouter",
        _rejected(
            "openrouter", "OpenRouter rejected this key: User not found.", "User not found."
        ),
    )

    assert status == "✗ OpenRouter rejected this key: User not found."
    assert "Reached the server" not in status
    probe.assert_not_awaited()  # the public listing can't fail a key
    assert "press Enter to check it with OpenRouter" in staged


@pytest.mark.asyncio
async def test_nvidia_nim_reads_key_not_checked_never_a_tick() -> None:
    staged, status, _probe, _scope = await _enter_key(
        "nvidia", _typed_model_discovery_result("nvidia", "meta/llama-3.1-8b")
    )

    assert "key not checked" in staged
    assert "key not checked" in status
    assert "✓" not in status


@pytest.mark.asyncio
async def test_anthropic_enter_checks_the_key_with_its_listing() -> None:
    staged, status, probe, _scope = await _enter_key(
        "anthropic", _typed_model_discovery_result("anthropic", "claude-sonnet-5")
    )

    assert status == "✓ Anthropic accepted this key — found 1 model(s)."
    assert "Connection testing is unavailable" not in staged
    probe.assert_not_awaited()


def test_the_model_step_quotes_the_providers_reason() -> None:
    from collections import namedtuple

    discovery_key = namedtuple("Key", "provider_key")("openrouter")
    wizard = SimpleNamespace(
        _first_run_selected_provider_outcomes={
            discovery_key: _rejected(
                "openrouter",
                "OpenRouter rejected this key: User not found.",
                "User not found.",
            )
        }
    )

    assert key_check.rejected_text(wizard, discovery_key) == (
        "OpenRouter rejected this key: User not found. Go Back to fix it, or "
        "enter a model ID below."
    )
    assert key_check.continue_prompt(wizard, discovery_key) == (
        "OpenRouter rejected this key: User not found. This model setup is "
        "unverified. Continue anyway?"
    )


def test_a_forbidden_listing_is_not_a_rejected_key() -> None:
    """Review round 2 (R2-8, wizard side): a Gemini 403 is 'refused', not a
    rejected key, as Settings shows it ('listing not permitted'). Dev's
    TASK-33008 D2 keeps the 403's "forbidden" for its keyless hint."""
    from tldw_chatbook.UI.Wizards.first_run_model_discovery import (
        _model_discovery_ui_outcome,
    )

    refused = ModelDiscoveryResult(
        provider="google",
        provider_list_key="google",
        endpoint_fingerprint=None,
        status="error",
        error=ModelDiscoveryError(
            kind="missing_credentials",
            message="Google refused this request: Permission denied.",
            recovery_hint="",
            category="forbidden",
            provider_reason="Permission denied.",
        ),
    )

    assert _model_discovery_ui_outcome(refused) == (
        [],
        "listing_unavailable",
        "forbidden",
    )
    # Only a 401 rejects a key: no surface words a 403 as a verdict on it.
    assert key_check.rejected_key_message("Google", refused) == ""


def test_the_summary_never_ticks_a_key_nothing_checked() -> None:
    from tldw_chatbook.UI.Wizards.first_run_setup_state import (
        ROW_CONFIGURED,
        SummaryRow,
    )

    rows = (
        SummaryRow("Provider", ROW_CONFIGURED, "nvidia"),
        SummaryRow("Default model", ROW_CONFIGURED, "meta/llama-3.1-8b"),
    )
    checked = (SummaryRow("Provider", ROW_CONFIGURED, "openrouter"),)

    marked = key_check.mark_unchecked_listing_keys(rows)

    assert marked[0].glyph == "–"
    assert marked[0].detail == "nvidia (key not checked)"
    assert marked[1] == rows[1]
    # OpenRouter has a key endpoint, but only a recorded pass ticks it.
    assert key_check.mark_unchecked_listing_keys(
        checked, key_verified_for=lambda _provider: True
    ) == checked
    assert key_check.mark_unchecked_listing_keys(checked)[0].detail == (
        "openrouter (key not checked)"
    )


def test_the_model_step_says_analysis_follows_the_chat_model() -> None:
    """AC#8: setup says document analysis follows the chat model -- while
    [analysis_defaults] still equals the shipped template -- and names the
    model analysis keeps when the user set one."""
    from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML
    from tldw_chatbook.UI.Wizards.first_run_model_notes import analysis_note

    template = {"analysis_defaults": dict(DEFAULT_CONFIG_FROM_TOML["analysis_defaults"])}
    own = {"analysis_defaults": {"provider": "anthropic", "model": "claude-haiku-4-5"}}

    assert analysis_note(template) == (
        "Document analysis and summaries (Library import, Media, Reader) use "
        "this model too."
    )
    assert analysis_note(own) == (
        "Document analysis keeps its own model: anthropic · claude-haiku-4-5."
    )


@pytest.mark.asyncio
async def test_the_mounted_summary_reads_key_not_checked_for_nvidia() -> None:
    """The wiring: the Summary step applies the overlay (mutating the call
    out of first_run_summary_step fails this)."""
    from tldw_chatbook.UI.Wizards.BaseWizard import WizardStepConfig
    from tldw_chatbook.UI.Wizards.first_run_summary_step import SummaryStep

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
        wizard_data={"welcome": {"track": "quick"}},
    )
    step = SummaryStep(
        wizard=wizard,
        config=WizardStepConfig(id="summary", title="Summary", step_number=9),
        load_config=lambda: {
            "api_settings": {
                "nvidia": {
                    "api_key": "nvapi-test",
                    "api_base_url": "https://integrate.api.nvidia.com/v1",
                }
            },
            "chat_defaults": {"provider": "nvidia", "model": "meta/llama-3.1-8b"},
        },
        rag_deps_installed=lambda: False,
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await app.workers.wait_for_complete()
        rendered = str(step.query_one("#setup-summary-rows", Static).render())

    assert "nvidia (key not checked)" in rendered
    assert "✓ Provider" not in rendered


# --- Review round 1 of pass 2 ----------------------------------------------


def _unauthorized_listing(provider: str) -> ModelDiscoveryResult:
    """What an unauthenticated listing returns for a keyed provider."""
    return _rejected(provider, f"{provider} listing needs a key.", "")


def test_a_keyed_listing_with_no_key_sent_is_never_a_rejected_key() -> None:
    """Live F1: selecting OpenAI or Anthropic, before any key was typed, read
    '✗ OpenAI rejected this key.' -- the unauthenticated listing's 401 was
    worded as a verdict on a key nobody sent. A real wrong key read the same,
    so 'no key yet' and 'wrong key' were indistinguishable."""
    no_key = key_check.discovery_status_copy(
        "openai",
        "OpenAI",
        _unauthorized_listing("openai"),
        models=(),
        failed=True,
        attempted=True,
        key_sent=False,
        failure_copy=lambda: "Couldn't discover models for OpenAI. Add a key. Or go Back.",
    )
    wrong_key = key_check.discovery_status_copy(
        "openai",
        "OpenAI",
        _unauthorized_listing("openai"),
        models=(),
        failed=True,
        attempted=True,
        key_sent=True,
        failure_copy=lambda: "unused",
    )

    assert "rejected" not in no_key
    assert no_key == "Couldn't discover models for OpenAI. Add a key. Or go Back."
    assert wrong_key == "✗ OpenAI rejected this key."


@pytest.mark.asyncio
async def test_selecting_a_keyed_provider_with_no_key_never_says_rejected() -> None:
    wizard = _wizard_with(_unauthorized_listing("anthropic"))
    step = _provider_step(wizard=wizard, probe=AsyncMock())
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("anthropic")
        for _ in range(20):
            await pilot.pause(0.05)
        status = str(step.query_one("#setup-provider-probe-status", Static).render())

    assert "rejected" not in status


@pytest.mark.parametrize(
    "reason",
    [
        "Incorrect API key provided: sk-ab***[redacted] is invalid.",
        "see [/docs] now",
        "[link=https://evil.example]click[/link]",
        "[bold red]PAY NOW[/bold red]",
    ],
)
def test_a_providers_reason_is_shown_as_text_never_as_markup(reason: str) -> None:
    """F4: the wizard rendered the provider's own words as Textual markup. A
    masked key's '[redacted]' vanished as a style tag, a '[/docs]' raised
    MarkupError inside the handler (the status never updated), and a reason
    could style itself."""
    from collections import namedtuple

    from textual.content import Content

    result = _rejected("openai", f"OpenAI rejected this key: {reason}", reason)
    status = key_check.discovery_status_copy(
        "openai",
        "OpenAI",
        result,
        models=(),
        failed=True,
        attempted=True,
        key_sent=True,
        failure_copy=lambda: "unused",
    )
    discovery_key = namedtuple("Key", "provider_key")("openai")
    wizard = SimpleNamespace(_first_run_selected_provider_outcomes={discovery_key: result})

    # The Provider step's status (a Static) parses markup: its text must
    # survive it verbatim.
    assert reason in Content.from_markup(status).plain
    # The Model step's status line (TASK-33008.5) and the 'Continue anyway?'
    # body render with markup off: plain text, never escaped.
    for plain in (
        key_check.rejected_text(wizard, discovery_key),
        key_check.continue_prompt(wizard, discovery_key),
    ):
        assert reason in plain
        assert "\\[" not in plain


def _openrouter_draft():
    from tldw_chatbook.UI.Wizards import first_run_setup_state as setup_state

    return setup_state.FirstRunProviderDraft(
        provider="openrouter",
        endpoint="https://openrouter.ai/api/v1",
        credential=setup_state.ProviderCredentialDraft(
            "draft", "sk-or-test-0123456789", revision=3
        ),
    )


@pytest.mark.parametrize(
    ("key_verified", "expected"),
    [(True, "openrouter"), (False, "openrouter (key not checked)")],
)
def test_the_summary_ticks_a_listing_key_only_when_its_check_passed(
    key_verified: bool, expected: str
) -> None:
    """F5: OpenRouter and Hugging Face have a key endpoint, so the Summary
    ticked them from the provider record alone. When that endpoint answers
    403/429/5xx, the network fails, or the endpoint is another host, the
    discovery verifies nothing (the Provider step correctly reads 'key not
    checked') -- and the Summary still read '✓ Provider — openrouter'."""
    from dataclasses import replace

    from tldw_chatbook.UI.Wizards import first_run_setup_state as setup_state
    from tldw_chatbook.UI.Wizards.first_run_setup_state import (
        ROW_CONFIGURED,
        SummaryRow,
    )

    draft = _openrouter_draft()
    discovery_key = setup_state.build_first_run_model_discovery_key(draft)
    outcome = replace(
        _typed_model_discovery_result("openrouter", "openai/gpt-4o-mini"),
        key_verified=key_verified,
    )
    wizard = SimpleNamespace(
        staged_provider_draft=draft,
        _first_run_selected_provider_outcomes={discovery_key: outcome},
    )
    rows = (SummaryRow("Provider", ROW_CONFIGURED, "openrouter"),)

    marked = key_check.mark_unchecked_listing_keys(
        rows,
        key_verified_for=lambda provider: key_check.listing_key_verified(
            wizard, provider
        ),
    )

    assert marked[0].detail == expected
    # Nothing recorded for the current key (a stored key, another revision):
    # nothing checked it.
    assert key_check.listing_key_verified(SimpleNamespace(), "openrouter") is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("analysis_defaults", "expected"),
    [
        (None, "use this model too"),
        (
            {"provider": "anthropic", "model": "claude-haiku-4-5"},
            "keeps its own model: anthropic · claude-haiku-4-5",
        ),
    ],
)
async def test_the_mounted_model_step_shows_the_analysis_note(
    analysis_defaults, expected: str
) -> None:
    """TE-3: only the pure ``analysis_note`` was tested, so dropping the
    widget from the Model step's compose left every test green."""
    from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML
    from tldw_chatbook.UI.Wizards.first_run_model_notes import ANALYSIS_NOTE_ID

    from Tests.Wizards.test_first_run_setup_wizard import _model_step

    app_config = {
        "analysis_defaults": dict(
            analysis_defaults or DEFAULT_CONFIG_FROM_TOML["analysis_defaults"]
        )
    }
    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config=app_config),
        wizard_data={"provider": {"provider_key": "openai", "provider_value": "OpenAI"}},
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _model_step(wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause()
        note = str(step.query_one(f"#{ANALYSIS_NOTE_ID}", Static).render())

    assert expected in note
