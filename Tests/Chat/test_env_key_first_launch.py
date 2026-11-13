"""TASK-34100.2 AC#7 / AC#12 (coverage-04, E2): an honest env-key launch.

With a provider key exported, a fresh install skips setup and used to say
"Found ANTHROPIC_API_KEY — you're ready to chat" without checking anything,
while new chats still pointed at an unkeyed OpenAI (and an OpenAI-only user's
first send was refused for context capacity). The notice now says "ready to
chat" only when the shared readiness verdict -- capacity included -- is ready
for the provider new chats use; otherwise it names the key it found and the
unkeyed default. Console's Get started card offers each env-keyed provider
("Found ANTHROPIC_API_KEY · Use Anthropic"), and choosing one commits it as the
chat default with credential_source = environment in one mutation. Nothing
here writes config.
"""

from __future__ import annotations

import copy

import pytest

from tldw_chatbook.Chat.console_onboarding_state import (
    build_console_env_key_actions,
    build_env_key_first_launch_notice,
    build_env_key_provider_mutation,
    env_keyed_providers,
)
from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML

ANTHROPIC_ENV = {"ANTHROPIC_API_KEY": "sk-ant-test-env-first-launch"}
OPENAI_ENV = {"OPENAI_API_KEY": "sk-test-env-first-launch-openai"}


def _fresh_profile() -> dict:
    return copy.deepcopy(DEFAULT_CONFIG_FROM_TOML)


def test_a_found_key_is_named_with_its_provider_and_curated_model() -> None:
    (found,) = env_keyed_providers(_fresh_profile(), ANTHROPIC_ENV)

    assert (found.provider_key, found.env_var) == ("anthropic", "ANTHROPIC_API_KEY")
    assert found.display_name == "Anthropic"
    assert found.model == DEFAULT_CONFIG_FROM_TOML["api_settings"]["anthropic"]["model"]


def test_a_key_for_another_provider_is_not_called_ready() -> None:
    notice = build_env_key_first_launch_notice(_fresh_profile(), ANTHROPIC_ENV)

    assert notice is not None
    assert notice.ready is False
    assert "ready to chat" not in notice.message
    assert notice.message.startswith(
        "Found ANTHROPIC_API_KEY, but new chats use OpenAI, which has no key."
    )
    assert "Use Anthropic" in notice.message  # the Get started card's action
    assert "Set up provider" in notice.message


def test_a_key_for_the_default_provider_is_ready_only_when_readiness_says_so() -> None:
    notice = build_env_key_first_launch_notice(_fresh_profile(), OPENAI_ENV)

    assert notice is not None
    assert notice.ready is True
    model = DEFAULT_CONFIG_FROM_TOML["api_settings"]["openai"]["model"]
    assert notice.message.startswith(
        f"Found OPENAI_API_KEY — you're ready to chat with OpenAI · {model}."
    )


def test_a_capacity_blocked_default_is_not_ready_and_says_why() -> None:
    profile = _fresh_profile()
    profile["chat_defaults"] = {"provider": "openai", "model": "gpt-4o"}
    profile["api_settings"]["openai"]["max_tokens"] = 128_000

    notice = build_env_key_first_launch_notice(profile, OPENAI_ENV)

    assert notice is not None and notice.ready is False
    assert "window too small" in notice.message


def test_no_env_key_means_no_notice() -> None:
    assert build_env_key_first_launch_notice(_fresh_profile(), {}) is None


def test_the_get_started_card_offers_each_env_keyed_provider() -> None:
    environ = {**ANTHROPIC_ENV, "GROQ_API_KEY": "gsk-test-env-first-launch"}

    actions = build_console_env_key_actions(
        _fresh_profile(), environ, card_mode="card"
    )

    assert [(action.label, action.detail) for action in actions] == [
        ("Use Anthropic", "Found ANTHROPIC_API_KEY"),
        ("Use Groq", "Found GROQ_API_KEY"),
    ]
    assert build_console_env_key_actions(
        _fresh_profile(), environ, card_mode="ready_line"
    ) == ()


def test_the_default_providers_own_key_needs_no_action() -> None:
    assert build_console_env_key_actions(
        _fresh_profile(), OPENAI_ENV, card_mode="card"
    ) == ()


def test_choosing_one_commits_it_with_its_environment_key_in_one_mutation() -> None:
    profile = _fresh_profile()
    (action,) = build_console_env_key_actions(profile, ANTHROPIC_ENV, card_mode="card")

    mutation = build_env_key_provider_mutation(action, profile)

    section = mutation.section_values["api_settings.anthropic"]
    assert section["credential_source"] == "environment"
    assert section["api_key_env_var"] == "ANTHROPIC_API_KEY"
    assert section["model"] == action.model
    assert dict(mutation.section_values["chat_defaults"]) == {
        "provider": "anthropic",
        "model": action.model,
    }
    # The secret itself is never written, only the variable's name.
    assert ANTHROPIC_ENV["ANTHROPIC_API_KEY"] not in repr(dict(section))


@pytest.mark.parametrize("value", ["", "   ", "<API_KEY_HERE>"])
def test_a_placeholder_env_value_is_not_a_found_key(value) -> None:
    assert env_keyed_providers(_fresh_profile(), {"ANTHROPIC_API_KEY": value}) == ()


def test_a_local_servers_key_is_named_but_never_offered() -> None:
    environ = {"VLLM_API_KEY": "vllm-test-env-first-launch"}

    notice = build_env_key_first_launch_notice(_fresh_profile(), environ)

    assert notice is not None and notice.ready is False
    assert notice.message.startswith("Found VLLM_API_KEY, but new chats use OpenAI")
    assert "Use vLLM" not in notice.message
    assert notice.message.endswith("choose Set up provider.")
    assert build_console_env_key_actions(_fresh_profile(), environ, card_mode="card") == ()


def test_the_card_offers_at_most_three_providers() -> None:
    environ = {
        "ANTHROPIC_API_KEY": "sk-ant-test-env-first-launch",
        "GROQ_API_KEY": "gsk-test-env-first-launch",
        "MISTRAL_API_KEY": "mistral-test-env-first-launch",
        "DEEPSEEK_API_KEY": "deepseek-test-env-first-launch",
    }

    actions = build_console_env_key_actions(_fresh_profile(), environ, card_mode="card")

    assert len(actions) == 3
    assert all(action.workbench_action.startswith("use-env-key:") for action in actions)


# --- Review round 1 of pass 2 ----------------------------------------------


@pytest.mark.parametrize(
    "stored",
    ["sk-ant-stored-in-settings-0123456789", "enc:v1:c2VjcmV0LWNpcGhlcnRleHQ="],
)
def test_a_provider_with_a_stored_key_is_never_offered_its_env_key(stored) -> None:
    """F3: 'Use Anthropic' commits credential_source = environment, and the
    shared persistence then deletes api_settings.anthropic.api_key. For a
    provider already set up in Settings (its stored key, or a still-encrypted
    one, outranks the env var) the offer silently deleted that key and
    changed which key is spent."""
    profile = _fresh_profile()
    profile["api_settings"]["anthropic"]["api_key"] = stored

    assert build_console_env_key_actions(profile, ANTHROPIC_ENV, card_mode="card") == ()
    notice = build_env_key_first_launch_notice(profile, ANTHROPIC_ENV)
    # The notice never names an offer the card does not show.
    assert notice is not None and "Use Anthropic" not in notice.message


def test_the_notice_names_only_offers_the_card_shows() -> None:
    """F10: the 'choose Use X or ...' list skipped the card's model filter,
    so it could name an offer the card did not show."""
    profile = _fresh_profile()
    profile["api_settings"]["anthropic"]["model"] = ""
    environ = {**ANTHROPIC_ENV, "GROQ_API_KEY": "gsk-test-env-first-launch"}

    notice = build_env_key_first_launch_notice(profile, environ)
    shown = build_console_env_key_actions(profile, environ, card_mode="card")

    assert [action.label for action in shown] == ["Use Groq"]
    assert notice is not None
    assert "Use Groq" in notice.message
    assert "Use Anthropic" not in notice.message


def test_a_capacity_blocked_default_points_at_fix_model_limits() -> None:
    """F10: a capacity blocker renders the empty chat's one line with Fix
    model limits, not the Get started card the notice pointed at."""
    profile = _fresh_profile()
    profile["chat_defaults"] = {"provider": "openai", "model": "gpt-4o"}
    profile["api_settings"]["openai"]["max_tokens"] = 128_000

    notice = build_env_key_first_launch_notice(profile, OPENAI_ENV)

    assert notice is not None
    assert "Get started card" not in notice.message
    assert "Fix model limits" in notice.message


from Tests.private_profile import private_profile_test  # noqa: E402


@private_profile_test
def test_a_committed_env_offer_lands_in_config_toml_without_the_secret(
    request, monkeypatch
) -> None:
    """TE-8: 'one atomic commit' and 'credential_source = environment' were
    checked only on the mutation object. This commits 'Use Anthropic'
    through the real ``persist_provider_setup`` into this test's private
    profile and reads the file back."""
    import os
    import tomllib
    from pathlib import Path

    from tldw_chatbook import config as config_module
    from tldw_chatbook.Chat.provider_setup_persistence import persist_provider_setup

    path = Path(config_module.get_cli_config_path())
    # The private profile owns both HOME and the config directory. Review
    # round 2 (R3-7): the old check, path under HOME's parent, also held for
    # a real ~/.config/tldw_cli under /Users; the profile root does not.
    root = os.environ.get("TLDW_TEST_CONFIG_ROOT", "")
    assert root, "only a private profile"
    profile_root = Path(root).resolve()
    assert path.resolve().is_relative_to(profile_root), "only a private profile"
    assert Path(os.environ["HOME"]).resolve().is_relative_to(profile_root)
    settings = config_module.load_settings(force_reload=True)
    (action,) = build_console_env_key_actions(settings, ANTHROPIC_ENV, card_mode="card")

    result = persist_provider_setup(build_env_key_provider_mutation(action, settings))

    assert result.fully_applied is True
    text = path.read_text(encoding="utf-8")
    saved = tomllib.loads(text)
    anthropic = saved["api_settings"]["anthropic"]
    assert anthropic["credential_source"] == "environment"
    assert anthropic["api_key_env_var"] == "ANTHROPIC_API_KEY"
    assert "api_key" not in anthropic
    assert saved["chat_defaults"]["provider"] == "anthropic"
    assert saved["chat_defaults"]["model"] == action.model
    assert ANTHROPIC_ENV["ANTHROPIC_API_KEY"] not in text
