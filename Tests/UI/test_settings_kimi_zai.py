"""Focused F9 Settings contracts for hosted Kimi and GLM providers."""

from __future__ import annotations

import pytest
from textual.widgets import Input, Select, Static

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
)
from Tests.UI.test_settings_configuration_hub import _open_settings_category
from Tests.UI.test_settings_configuration_hub import _settle_settings_mount_storm
from tldw_chatbook.UI.Screens.settings_screen import SettingsCategoryId


def _option_values(select: Select) -> set[str]:
    return {
        str(option[1]) for option in select._options if option[1] is not Select.NULL
    }


def _hosted_app(provider: str, model: str):
    app = _build_test_app()
    provider_key = "moonshot" if provider == "Moonshot" else "zai"
    endpoint = (
        "https://api.moonshot.ai/v1"
        if provider_key == "moonshot"
        else "https://api.z.ai/api/paas/v4"
    )
    env_var = "MOONSHOT_API_KEY" if provider_key == "moonshot" else "ZAI_API_KEY"
    app.app_config["chat_defaults"] = {"provider": provider, "model": model}
    app.app_config["api_settings"] = {
        provider_key: {
            "api_key": "settings-test-key",
            "api_key_env_var": env_var,
            "api_base_url": endpoint,
            "model": model,
            "streaming": True,
        }
    }
    app.providers_models = {
        "Moonshot": ["kimi-k3", "moonshot-v1-128k"],
        "ZAI": ["glm-5.2", "glm-4.5"],
    }
    return app


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "model", "expected_endpoint", "expected_env", "reasoning"),
    [
        (
            "Moonshot",
            "kimi-k3",
            "https://api.moonshot.ai/v1",
            "MOONSHOT_API_KEY",
            # "medium" wire-verified in TASK-18803
            # (chatcmpl-6a872b62bea2d202c1d3f6fa); this pin predates that
            # constant change and was red on dev until TASK-19170.
            {"low", "medium", "high", "max"},
        ),
        # TASK-19170 AC #2: a non-literal family member gets the curated
        # list without a code edit -- the options follow the 18803 family
        # predicates, not exact-id pins.
        (
            "Moonshot",
            "kimi-k2.6",
            "https://api.moonshot.ai/v1",
            "MOONSHOT_API_KEY",
            {"low", "medium", "high", "max"},
        ),
        (
            "ZAI",
            "glm-5.2",
            "https://api.z.ai/api/paas/v4",
            "ZAI_API_KEY",
            {"none", "minimal", "low", "medium", "high", "xhigh", "max"},
        ),
        (
            "ZAI",
            "glm-5.3",
            "https://api.z.ai/api/paas/v4",
            "ZAI_API_KEY",
            {"none", "minimal", "low", "medium", "high", "xhigh", "max"},
        ),
    ],
)
async def test_settings_hosted_provider_controls_are_exact_and_actionable(
    provider, model, expected_endpoint, expected_env, reasoning
):
    host = DestinationHarness(_hosted_app(provider, model), "settings")

    async with host.run_test(size=(190, 58)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)

        assert screen.query_one("#settings-model-value", Input).value == model
        assert (
            screen.query_one("#settings-provider-endpoint-value", Input).value
            == expected_endpoint
        )
        assert (
            screen.query_one("#settings-provider-credential-env-var", Input).value
            == expected_env
        )
        api_mode_row = screen.query_one("#settings-provider-api-mode-row")
        assert api_mode_row.has_class("settings-gated-profile-hidden")
        assert screen.query_one("#settings-provider-api-mode", Select).disabled

        reasoning_select = screen.query_one(
            "#settings-model-profile-reasoning-effort", Select
        )
        assert reasoning_select.disabled is False
        assert _option_values(reasoning_select) == reasoning

        guidance = str(
            screen.query_one("#settings-hosted-provider-guidance", Static).renderable
        )
        if provider == "Moonshot":
            assert "international, China, or custom" in guidance
            assert "Preserved Thinking" in guidance
            assert "private" in guidance
        else:
            assert "general API" in guidance
            assert "coding-only" in guidance
            assert "private" in guidance


@pytest.mark.asyncio
async def test_settings_hosted_provider_switch_isolates_drafts_and_keeps_old_model():
    app = _hosted_app("Moonshot", "moonshot-v1-128k")
    app.app_config["api_settings"]["zai"] = {
        "api_key": "settings-zai-key",
        "api_key_env_var": "ZAI_API_KEY",
        "api_base_url": "https://api.z.ai/api/paas/v4",
        "model": "glm-5.2",
        "streaming": True,
    }
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 58)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        endpoint.value = "https://unsaved-moonshot.example/v1"
        await pilot.pause()

        screen._apply_provider_value_change("zai")
        await pilot.pause()
        assert endpoint.value == "https://api.z.ai/api/paas/v4"
        assert screen.query_one("#settings-model-value", Input).value == "glm-5.2"
        assert _option_values(
            screen.query_one("#settings-model-profile-reasoning-effort", Select)
        ) == {"none", "minimal", "low", "medium", "high", "xhigh", "max"}

        screen._apply_provider_value_change("moonshot")
        await pilot.pause()
        assert endpoint.value == "https://api.moonshot.ai/v1"
        assert (
            screen.query_one("#settings-model-value", Input).value == "moonshot-v1-128k"
        )
        assert "Verify reasoning support" in str(
            screen.query_one("#settings-hosted-provider-guidance", Static).renderable
        )


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        # Curated Kimi list follows the 18803 request-side family predicate
        # (the values build_moonshot_chat_payload accepts), so any kimi-series
        # release gets it without a code edit (TASK-19170 AC #2).
        ("Moonshot", "kimi-k3", ("low", "medium", "high", "max")),
        ("Moonshot", "kimi-k2.6", ("low", "medium", "high", "max")),
        ("Moonshot", "kimi-k3-turbo", ("low", "medium", "high", "max")),
        # reasoning_effort answers 200 on kimi-latest
        # (TASK-18803, chatcmpl-6a872ac016ceb0c0ae780b0c) and the builder
        # accepts it: the curated list applies even though kimi-latest
        # returns no reasoning_content.
        ("Moonshot", "kimi-latest", ("low", "medium", "high", "max")),
        # The builder client-side-rejects reasoning_effort for moonshot-v1:
        # outside the kimi series the generic list is unchanged behavior.
        ("Moonshot", "moonshot-v1-8k", None),
        ("ZAI", "glm-5.2", ("none", "minimal", "low", "medium", "high", "xhigh", "max")),
        ("ZAI", "glm-5.3", ("none", "minimal", "low", "medium", "high", "xhigh", "max")),
        (
            "ZAI",
            "glm-5.2-air",
            ("none", "minimal", "low", "medium", "high", "xhigh", "max"),
        ),
        ("ZAI", "glm-4.6", None),
        ("OpenAI", "gpt-5", None),
    ],
)
def test_model_profile_reasoning_effort_options_follow_family_predicates(
    provider, model, expected
):
    from tldw_chatbook.UI.Screens.settings_screen import (
        REASONING_EFFORT_SELECT_OPTIONS,
        SettingsScreen,
    )

    options = SettingsScreen._model_profile_reasoning_effort_options(provider, model)
    assert options == (
        expected if expected is not None else REASONING_EFFORT_SELECT_OPTIONS
    )


@pytest.mark.asyncio
async def test_settings_hosted_navigation_focuses_reasoning_control():
    host = DestinationHarness(_hosted_app("Moonshot", "kimi-k3"), "settings")

    async with host.run_test(size=(190, 58)) as pilot:
        await _settle_settings_mount_storm(pilot)
        screen = _active_destination_screen(host)
        screen.apply_navigation_context(
            {
                "category": SettingsCategoryId.PROVIDERS_MODELS.value,
                "provider": "moonshot",
                "model": "kimi-k3",
                "field": "reasoning",
            }
        )
        await pilot.pause()
        await pilot.pause()

        assert screen.query_one(
            "#settings-model-profile-reasoning-effort", Select
        ).has_focus
