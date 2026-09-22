"""TASK-32894: the two API TTS backends must obey ADR-012's key precedence.

ADR-012's 2026-09-19 amendment (TASK-32806.2) rules that an explicit
`api_settings.<provider>.api_key` OUTRANKS the environment variable it
names, and `config.get_api_key` is the shared accessor that implements it
-- screening every branch through `resolve_provider_api_key` so a
placeholder or a whitespace-padded value is never handed back.

Both TTS backends resolved credentials themselves instead:

* `TTS/backends/openai.py` read `os.getenv("OPENAI_API_KEY")` FIRST, so a
  stale shell export outranked the key typed into Settings -- the exact
  disagreement the amendment exists to end -- and no branch screened the
  shipped `<API_KEY_HERE>` placeholder.
* `TTS/backends/elevenlabs.py` never looked at
  `api_settings.elevenlabs.api_key` at all, so a key saved through
  Settings was invisible to it, and it too accepted a placeholder.

The real `config.get_api_key` runs here; only its config SOURCE is stubbed,
so these assert the shipped precedence rule rather than a stub of it.
"""

from __future__ import annotations

import pytest

from tldw_chatbook import config as config_module
from tldw_chatbook.TTS.backends.elevenlabs import ElevenLabsTTSBackend
from tldw_chatbook.TTS.backends.openai import OpenAITTSBackend

STORED = "sk-stored-in-settings"
FROM_ENV = "sk-exported-in-a-shell-profile"


@pytest.fixture
def stub_config(monkeypatch):
    """Drive the real `get_api_key` off a controlled settings mapping."""

    def _install(api_settings: dict, legacy_api: dict | None = None, app_tts=None):
        monkeypatch.setattr(
            config_module,
            "load_settings",
            lambda *a, **k: {"api_settings": api_settings},
        )

        def _fake_cli_setting(section, key=None, default=None):
            if section == "API":
                table = legacy_api or {}
                return table.get(key, default) if key else table
            if section == "app_tts":
                table = app_tts or {}
                return table.get(key, default) if key else table
            return default

        monkeypatch.setattr(config_module, "get_cli_setting", _fake_cli_setting)
        # `APITTSBackend.__init__` builds an httpx client, whose TLS setting
        # read trips ADR-126's recovery gate in a clean worktree. Unrelated
        # to the credential rule under test.
        from tldw_chatbook.Utils import tls_trust

        monkeypatch.setattr(tls_trust, "get_cli_setting", _fake_cli_setting)
        for module_name in (
            "tldw_chatbook.TTS.backends.openai",
            "tldw_chatbook.TTS.backends.elevenlabs",
        ):
            import sys

            module = sys.modules[module_name]
            if hasattr(module, "get_cli_setting"):
                monkeypatch.setattr(module, "get_cli_setting", _fake_cli_setting)

    return _install


def test_openai_stored_key_outranks_the_environment_variable(stub_config, monkeypatch):
    stub_config(
        {"openai": {"api_key": STORED, "api_key_env_var": "OPENAI_API_KEY"}}
    )
    monkeypatch.setenv("OPENAI_API_KEY", FROM_ENV)
    assert OpenAITTSBackend({}).api_key == STORED


def test_openai_environment_variable_is_still_the_fallback(stub_config, monkeypatch):
    stub_config({"openai": {"api_key_env_var": "OPENAI_API_KEY"}})
    monkeypatch.setenv("OPENAI_API_KEY", FROM_ENV)
    assert OpenAITTSBackend({}).api_key == FROM_ENV


def test_openai_refuses_the_shipped_placeholder(stub_config, monkeypatch):
    stub_config({"openai": {"api_key": "<API_KEY_HERE>"}})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert OpenAITTSBackend({}).api_key is None


def test_openai_strips_a_padded_key(stub_config, monkeypatch):
    stub_config({"openai": {"api_key": f"  {STORED}  "}})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert OpenAITTSBackend({}).api_key == STORED


def test_elevenlabs_reads_the_modern_api_settings_table(stub_config, monkeypatch):
    stub_config({"elevenlabs": {"api_key": STORED}})
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    assert ElevenLabsTTSBackend({}).api_key == STORED


def test_elevenlabs_stored_key_outranks_the_environment_variable(
    stub_config, monkeypatch
):
    stub_config(
        {"elevenlabs": {"api_key": STORED, "api_key_env_var": "ELEVENLABS_API_KEY"}}
    )
    monkeypatch.setenv("ELEVENLABS_API_KEY", FROM_ENV)
    assert ElevenLabsTTSBackend({}).api_key == STORED


def test_elevenlabs_refuses_the_shipped_placeholder(stub_config, monkeypatch):
    stub_config({"elevenlabs": {"api_key": "<API_KEY_HERE>"}})
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    assert ElevenLabsTTSBackend({}).api_key is None


def test_elevenlabs_still_honours_the_legacy_api_section(stub_config, monkeypatch):
    stub_config({}, legacy_api={"elevenlabs_api_key": STORED})
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    assert ElevenLabsTTSBackend({}).api_key == STORED
