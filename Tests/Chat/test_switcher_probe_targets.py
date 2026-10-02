"""Which connections Switch model may probe on its own (TASK-33005.5).

Real config resolution and the real readiness builder. A connection is probed
automatically only when it is keyless and URL-based, sends no credential, and
sits on a loopback or private-network address (D2, ADR-012, AC#4).
"""

from datetime import UTC, datetime

import pytest

from tldw_chatbook.Chat.console_session_settings import (
    build_console_settings_readiness,
    build_target_default_console_session_settings,
    console_send_connection,
)
from tldw_chatbook.Chat.provider_test_evidence import (
    ProviderConnectionEvidence,
    ProviderTestEvidence,
)
from tldw_chatbook.UI.Console_Modules import connection_probe
from tldw_chatbook.UI.Console_Modules.connection_probe import (
    switcher_connection_prober,
    switcher_probe_plan,
)

_ENTRY = {
    "display_name": "Vale endpoint",
    "base_url": "http://127.0.0.1:9999/v1",
    "family": "openai_compatible",
    "models": ["vale-model"],
}


def _config(provider: str, settings: dict) -> dict:
    if provider.startswith("custom-ep:"):
        return {"api_settings": {}, "custom_endpoints": {"vale": settings}}
    return {"api_settings": {provider: settings}}


PROBED = [
    ("llama_cpp", {}),  # nothing saved: a new chat sends to 127.0.0.1:9099
    ("llama_cpp", {"api_url": "http://localhost:8080"}),
    ("ollama", {"api_url": "http://127.0.0.1:11434"}),
    ("vllm", {"api_url": "http://192.168.1.20:8000"}),
    ("koboldcpp", {"api_url": "http://10.0.0.7:5001"}),
    ("custom", {"api_url": "http://[::1]:8080/v1"}),
    ("custom-ep:vale", _ENTRY),
]
NEVER = [
    # A keyed cloud provider is never contacted, even on loopback (D2).
    ("qwencloud", {"api_key": "sk-cloud", "api_base_url": "http://127.0.0.1:8000"}),
    ("openai", {"api_key": "sk-cloud"}),
    ("anthropic", {"api_key": "sk-cloud"}),
    # Keyless, but on a public host, a name (never resolved) or link-local.
    ("ollama", {"api_url": "http://8.8.8.8:11434"}),
    ("ollama", {"api_url": "http://ollama.example.com:11434"}),
    ("vllm", {"api_url": "http://[::ffff:8.8.8.8]:8000"}),
    ("vllm", {"api_url": "http://169.254.169.254"}),
    # Local, but a probe would send the credential a send uses.
    ("vllm", {"api_url": "http://127.0.0.1:8000", "api_key": "sk-local"}),
    ("custom", {"api_url": "http://127.0.0.1:8080/v1", "api_key": "sk-custom"}),
    ("custom-ep:vale", {**_ENTRY, "api_key": "sk-entry"}),
    # In-process providers have no models route.
    ("mlx_lm", {}),
    ("local_llm", {}),
]


def _ids(cases):
    return [f"{provider}-{index}" for index, (provider, _settings) in enumerate(cases)]


@pytest.mark.parametrize(("provider", "settings"), PROBED, ids=_ids(PROBED))
def test_a_keyless_local_server_that_sends_no_key_is_probed(provider, settings):
    """AC#1: probed, under the exact connection a send uses."""
    config = _config(provider, settings)

    plan = switcher_probe_plan(config, {provider: "model-a"})

    expected = console_send_connection(
        build_target_default_console_session_settings(config, provider, "model-a"),
        app_config=config,
    )
    assert plan == {expected: [provider]}
    assert expected.credential_revision == 0


@pytest.mark.parametrize(("provider", "settings"), NEVER, ids=_ids(NEVER))
def test_cloud_keyed_public_and_in_process_endpoints_are_never_probed(
    provider, settings
):
    """AC#4: qwencloud, key-requiring providers, public hosts and any endpoint
    that resolves a credential are never contacted automatically."""
    assert switcher_probe_plan(_config(provider, settings), {provider: "model-a"}) == {}


@pytest.mark.parametrize(
    ("provider", "settings"),
    [*PROBED, NEVER[7], NEVER[9]],
    ids=[*_ids(PROBED), "vllm-keyed", "entry-keyed"],
)
def test_the_probed_connection_is_the_one_readiness_reads(provider, settings):
    """AC#6: a result settled for the probed connection is the one the
    readiness builder (switcher rows, Console status) reads back."""
    config = _config(provider, settings)
    target = build_target_default_console_session_settings(config, provider, "model-a")
    identity = console_send_connection(target, app_config=config, environ={})
    owner = ProviderConnectionEvidence()
    owner.publish(
        ProviderTestEvidence(
            identity,
            "unreachable",
            (),
            "connection_refused",
            observed_at=datetime(2026, 10, 2, 9, 30, tzinfo=UTC),
        ),
        order=1,
    )

    readiness = build_console_settings_readiness(
        target, app_config=config, environ={}, connection_evidence=owner
    )

    assert readiness.connection == identity
    assert readiness.endpoint_category == "connection_refused"


def test_no_test_directory_opens_switch_model_onto_the_network():
    """The autouse guard in Tests/conftest.py, not one directory's conftest,
    shuts the switcher's probe seam: Tests/ProductionApp opens the real Switch
    model from the palette on the shipped profile, which lists eight localhost
    servers. The real seam is reached only by a by-name import, as here."""
    assert connection_probe.switcher_connection_prober is not switcher_connection_prober
