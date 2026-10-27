"""TASK-34100.2 AC#10 (entry-exit-handoff-25): a local server streams.

LM Studio users set up through the Custom OpenAI-compatible provider got a
long silent wait and then a wall of text: the template shipped
``[api_settings.custom] streaming = false`` and setup never chose a streaming
mode. The template now ships streaming on for the local OpenAI-compatible
Custom sections, and setup turns streaming on when it commits a local
provider; a cloud provider's streaming setting is left alone (task-33620.10).
"""

from __future__ import annotations

import copy

import pytest

from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML
from tldw_chatbook.UI.Wizards import first_run_setup_state as setup_state


def test_the_template_streams_local_openai_compatible_sections() -> None:
    api = DEFAULT_CONFIG_FROM_TOML["api_settings"]

    assert api["custom"]["streaming"] is True
    assert api["custom_2"]["streaming"] is True


def test_the_streaming_set_comes_from_the_registrys_local_servers() -> None:
    keys = setup_state.FIRST_RUN_LOCAL_STREAMING_PROVIDER_KEYS

    assert {
        "custom", "custom_2", "llama_cpp", "ollama", "vllm", "tabbyapi",
        "aphrodite", "oobabooga", "local_llm", "local_mlx_lm",
    } <= keys
    assert "koboldcpp" not in keys  # its native handler forces streaming off
    assert not keys & {"openai", "anthropic", "google", "openrouter"}


def _commit(provider: str, endpoint: str, *, streaming: bool = False):
    config = copy.deepcopy(DEFAULT_CONFIG_FROM_TOML)
    # An older profile that still carries the old template value.
    config["api_settings"][provider]["streaming"] = streaming
    draft = setup_state.FirstRunProviderDraft(
        provider=provider,
        endpoint=endpoint,
        credential=setup_state.ProviderCredentialDraft("none", "", 0),
    )
    resolved = setup_state.resolve_first_run_provider_draft(draft, config)
    return setup_state.build_first_run_provider_commit(resolved, "qwen2.5-0.5b", config)


@pytest.mark.parametrize(
    ("provider", "endpoint"),
    [
        ("custom", "http://127.0.0.1:9099/v1"),
        ("custom_2", "http://127.0.0.1:1234/v1"),
        ("llama_cpp", "http://127.0.0.1:8181"),
    ],
)
def test_setup_turns_streaming_on_when_it_commits_a_local_provider(
    provider, endpoint
) -> None:
    mutation = _commit(provider, endpoint)

    assert mutation.section_values[f"api_settings.{provider}"]["streaming"] is True


def test_setup_leaves_a_cloud_providers_streaming_alone() -> None:
    config = copy.deepcopy(DEFAULT_CONFIG_FROM_TOML)
    draft = setup_state.FirstRunProviderDraft(
        provider="openai",
        endpoint="",
        credential=setup_state.ProviderCredentialDraft("draft", "sk-test-local-streaming", 1),
    )
    resolved = setup_state.resolve_first_run_provider_draft(draft, config)
    mutation = setup_state.build_first_run_provider_commit(resolved, "gpt-4.1", config)

    assert "streaming" not in mutation.section_values["api_settings.openai"]


def test_saving_a_template_sample_provider_clears_its_marker() -> None:
    """AC#9: setup's save makes the provider the user's own (the marker is
    written false -- deleting it would let the template's true merge back)."""
    mutation = _commit("custom", "http://127.0.0.1:9099/v1")

    assert mutation.section_values["api_settings.custom"]["template_sample"] is False
