"""A local provider whose saved key is still `enc:` ciphertext is never Ready.

TASK-34100.4 AC #10 (part 2 residue). Part 1 blocked a stranded `enc:` key in
`[api_settings.<provider>]`, including keyless local providers. A local server
is just as often saved as an ADR-146 custom endpoint, and that entry's own
`api_key` is encrypted the same way (``encrypt_api_keys_in_config`` walks every
table). Its readiness fell back to the keyless family: a llama.cpp entry whose
key was still `enc:` read "Ready · not tested", and the send went out with no
key and failed on the server's bare HTTP 401.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    build_console_settings_readiness,
    readiness_words,
)
from tldw_chatbook.Chat.custom_endpoint_registry import entry_for, resolve_entry_credential
from tldw_chatbook.Chat.provider_readiness import (
    STILL_ENCRYPTED_REASON,
    get_provider_readiness,
)
from tldw_chatbook.Utils.config_encryption import ConfigEncryption

PASSWORD = "earlier-master-pw"
LOCAL_KEY = "llama-local-server-key"


def _ciphertext() -> str:
    return ConfigEncryption().encrypt_value(LOCAL_KEY, PASSWORD)


def _entry_config(family: str, *, api_key: str, **extra: object) -> dict:
    return {
        "custom_endpoints": {
            "home": {
                "display_name": "Home llama",
                "family": family,
                "base_url": "http://127.0.0.1:8181",
                "api_key": api_key,
                **extra,
            }
        }
    }


def test_wizard_saved_llama_cpp_key_still_encrypted_reads_not_ready():
    # The shape the setup wizard writes for a local server with a key.
    config = {
        "chat_defaults": {"provider": "llama_cpp", "model": "qwen"},
        "api_settings": {
            "llama_cpp": {
                "api_url": "http://127.0.0.1:8181",
                "api_key": _ciphertext(),
                "credential_source": "stored",
            }
        },
    }
    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(provider="llama_cpp", model="qwen"),
        app_config=config,
        environ={},
    )

    assert readiness.operability == "not_ready"
    assert readiness_words(readiness) == "Not ready · no key"
    assert "still encrypted" in readiness.detail


@pytest.mark.parametrize("family", ["llama_cpp", "ollama"])
def test_custom_endpoint_whose_saved_key_is_still_encrypted_is_not_ready(family):
    config = _entry_config(family, api_key=_ciphertext())

    readiness = get_provider_readiness("custom-ep:home", config, environ={})

    assert readiness.ready is False
    assert readiness.api_key is None
    assert readiness.reason == STILL_ENCRYPTED_REASON
    assert readiness.configuration_issue == "credential_missing"
    assert "Custom endpoints" in readiness.recovery


def test_console_header_for_a_stranded_custom_endpoint_names_the_missing_key():
    config = _entry_config("llama_cpp", api_key=_ciphertext())

    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(
            provider="custom-ep:home", model="qwen", base_url="http://127.0.0.1:8181"
        ),
        app_config=config,
        environ={},
    )

    assert readiness.operability == "not_ready"
    assert readiness_words(readiness) == "Not ready · no key"
    assert readiness_words(readiness) != "Ready · not tested"
    assert "still encrypted" in readiness.detail


def test_custom_endpoint_env_key_still_beats_a_stranded_saved_key():
    config = _entry_config(
        "llama_cpp", api_key=_ciphertext(), api_key_env="HOME_LLAMA_KEY"
    )

    readiness = get_provider_readiness(
        "custom-ep:home", config, environ={"HOME_LLAMA_KEY": "env-local-key"}
    )

    assert readiness.ready is True


def test_custom_endpoint_without_any_key_stays_keyless_ready():
    config = {
        "custom_endpoints": {
            "home": {
                "display_name": "Home llama",
                "family": "llama_cpp",
                "base_url": "http://127.0.0.1:8181",
            }
        }
    }

    readiness = get_provider_readiness("custom-ep:home", config, environ={})

    assert readiness.ready is True


def test_a_stranded_entry_key_is_never_resolved_as_its_credential():
    entry = entry_for(_entry_config("llama_cpp", api_key=_ciphertext()), "custom-ep:home")

    assert entry is not None
    assert resolve_entry_credential(entry, environ={}) == (None, None)
