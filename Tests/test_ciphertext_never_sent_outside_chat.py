"""`enc:` ciphertext is never sent as a credential by these non-chat services.

TASK-34100.4 AC #10 (part 2 residue). Part 1 screened ciphertext out of
provider-key resolution, readiness and every LLM chat handler. Five services
outside the chat layer still read their key straight from config, so a value
left `enc:` (encryption off with a stranded value, or a session that never
unlocked) went out as the key, token or bearer:

* the TTS backend manager's OpenAI and ElevenLabs keys,
* remote Whisper transcription,
* the image/video generation secret resolver,
* the SwarmUI client,
* the tldw server API client.

Each test records what reaches the request or the client. A still-encrypted
value reads as absent, so the next source (an environment variable, the
keyring, a legacy table) is used, or nothing is sent.

TTS is defence in depth (review round 1, R1-F8/TE-3): the OpenAI and
ElevenLabs backends already ran their key through `resolve_provider_api_key`,
so on the pre-fix code the ciphertext was *held* in the manager's backend
config, not put on the wire. Those tests pin what the manager (and the
legacy bridge that feeds it in production) passes to the backend.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.Utils.config_encryption import ConfigEncryption

PASSWORD = "stranded-master-pw"


def _enc(value: str) -> str:
    return ConfigEncryption().encrypt_value(value, PASSWORD)


# --- TTS -------------------------------------------------------------------


def _tts_config(app_config: dict) -> dict:
    from tldw_chatbook.TTS.TTS_Backends import TTSBackendManager

    return TTSBackendManager(app_config)._prepare_backend_config("openai_official_tts")


def test_tts_openai_key_still_encrypted_is_not_passed_to_the_backend():
    config = _tts_config({"api_settings.openai": {"api_key": _enc("sk-tts-real")}})

    assert "OPENAI_API_KEY" not in config


def test_tts_openai_encrypted_key_falls_through_to_the_next_config_source():
    config = _tts_config(
        {
            "api_settings.openai": {"api_key": _enc("sk-tts-real")},
            "openai_api": {"api_key": "sk-legacy-plain"},
        }
    )

    assert config["OPENAI_API_KEY"] == "sk-legacy-plain"


def test_tts_openai_plain_key_is_still_passed():
    config = _tts_config({"api_settings.openai": {"api_key": "sk-plain"}})

    assert config["OPENAI_API_KEY"] == "sk-plain"


def test_tts_elevenlabs_key_still_encrypted_is_not_passed_to_the_backend():
    from tldw_chatbook.TTS.TTS_Backends import TTSBackendManager

    manager = TTSBackendManager({"API": {"elevenlabs_api_key": _enc("el-real")}})
    config = manager._prepare_backend_config("elevenlabs")

    assert "ELEVENLABS_API_KEY" not in config


def test_tts_elevenlabs_encrypted_key_falls_through_to_the_normalized_table():
    from tldw_chatbook.TTS.TTS_Backends import TTSBackendManager

    manager = TTSBackendManager(
        {
            "API": {"elevenlabs_api_key": _enc("el-real")},
            "elevenlabs_api": {"api_key": "el-plain"},
        }
    )
    config = manager._prepare_backend_config("elevenlabs")

    assert config["ELEVENLABS_API_KEY"] == "el-plain"


def test_tts_openai_legacy_slot_still_encrypted_is_not_passed_to_the_backend():
    # TE-3: `openai_api` is the slot the production bridge fills.
    config = _tts_config({"openai_api": {"api_key": _enc("sk-tts-real")}})

    assert "OPENAI_API_KEY" not in config


def test_tts_elevenlabs_normalized_slot_still_encrypted_is_not_passed():
    from tldw_chatbook.TTS.TTS_Backends import TTSBackendManager

    manager = TTSBackendManager({"elevenlabs_api": {"api_key": _enc("el-real")}})

    assert "ELEVENLABS_API_KEY" not in manager._prepare_backend_config("elevenlabs")


def _bridged_backend_config(provider_id: str, backend_id: str, app_config: dict):
    """The production path: legacy bridge projection -> manager -> backend."""
    from tldw_chatbook.TTS.legacy_bridge import legacy_provider_config
    from tldw_chatbook.TTS.TTS_Backends import TTSBackendManager

    projected = legacy_provider_config(provider_id, app_config)["app_config"]
    return TTSBackendManager(projected)._prepare_backend_config(backend_id)


@pytest.mark.parametrize(
    ("provider_id", "backend_id", "table", "env_var", "config_key"),
    [
        ("openai", "openai_official_tts", "openai", "OPENAI_API_KEY", "OPENAI_API_KEY"),
        (
            "elevenlabs",
            "elevenlabs",
            "elevenlabs",
            "ELEVENLABS_API_KEY",
            "ELEVENLABS_API_KEY",
        ),
    ],
    ids=["openai", "elevenlabs"],
)
def test_tts_bridge_skips_a_still_encrypted_stored_key_for_the_env_var(
    monkeypatch, provider_id, backend_id, table, env_var, config_key
):
    # TE-3: the bridge took the first truthy stored value, so ciphertext
    # short-circuited the environment variable that should come next.
    monkeypatch.setenv(env_var, "env-plain-key")
    config = _bridged_backend_config(
        provider_id,
        backend_id,
        {"api_settings": {table: {"api_key": _enc("stored-real")}}},
    )

    assert config.get(config_key) == "env-plain-key"


def test_tts_bridge_never_projects_a_still_encrypted_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = _bridged_backend_config(
        "openai",
        "openai_official_tts",
        {"api_settings": {"openai": {"api_key": _enc("stored-real")}}},
    )

    assert "OPENAI_API_KEY" not in config


# --- Remote Whisper --------------------------------------------------------


class _RequestSent(Exception):
    pass


def _remote_whisper_headers(monkeypatch, tmp_path, remote_config, environ):
    import tldw_chatbook.config as config_module
    from tldw_chatbook.Local_Ingestion import transcription_service

    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF0000WAVE")
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        config_module,
        "get_cli_setting",
        lambda key, default=None: (
            remote_config if key == "transcription.remote_whisper" else default
        ),
    )
    captured: dict = {}
    posted: list[str] = []

    def _post(url, **kwargs):
        posted.append(url)
        captured.update(kwargs.get("headers") or {})
        raise _RequestSent()

    monkeypatch.setattr(transcription_service.requests, "post", _post)
    with pytest.raises(Exception):
        transcription_service._LegacyTranscriptionBackend._transcribe_with_remote_whisper(
            SimpleNamespace(), str(audio)
        )
    # TE-7: the request was really attempted, so "no Authorization header"
    # is a statement about what was sent, not about an early exit.
    assert posted == [remote_config["api_endpoint"]]
    return captured


def test_remote_whisper_encrypted_key_is_never_sent(monkeypatch, tmp_path):
    monkeypatch.delenv("REMOTE_WHISPER_API_KEY", raising=False)
    headers = _remote_whisper_headers(
        monkeypatch,
        tmp_path,
        {
            "enabled": True,
            "api_endpoint": "http://127.0.0.1:9/v1/audio/transcriptions",
            "api_key": _enc("whisper-real"),
        },
        {},
    )

    assert "Authorization" not in headers


def test_remote_whisper_encrypted_key_falls_back_to_its_env_var(monkeypatch, tmp_path):
    headers = _remote_whisper_headers(
        monkeypatch,
        tmp_path,
        {
            "enabled": True,
            "api_endpoint": "http://127.0.0.1:9/v1/audio/transcriptions",
            "api_key": _enc("whisper-real"),
            "api_key_env_var": "REMOTE_WHISPER_API_KEY",
        },
        {"REMOTE_WHISPER_API_KEY": "whisper-env-key"},
    )

    assert headers.get("Authorization") == "Bearer whisper-env-key"


# --- Image / video generation secrets ---------------------------------------


def _tables():
    from tldw_chatbook.Media_Generation.config_machinery import (
        ModalityConfigTables,
        SecretSpec,
    )

    return ModalityConfigTables(
        section_name="image_generation",
        keyring_namespace="tldw-test-imagegen",
        global_keys=(),
        secrets={
            "swarmui": SecretSpec(
                flat_field="swarmui_token",
                env_vars=("TLDW_TEST_SWARM_TOKEN",),
                keyring_id="swarmui",
                config_key="swarm_token",
            )
        },
    )


@pytest.mark.parametrize("toml_key", ["swarm_token", "api_key"])
def test_generation_secret_still_encrypted_falls_through_to_the_keyring(
    monkeypatch, toml_key
):
    from tldw_chatbook.Media_Generation.config_machinery import resolve_secret

    monkeypatch.delenv("TLDW_TEST_SWARM_TOKEN", raising=False)
    field, value, source = resolve_secret(
        "swarmui",
        {toml_key: _enc("swarm-real")},
        _tables(),
        keyring_lookup=lambda _id: "keyring-token",
    )

    assert (field, value, source) == ("swarmui_token", "keyring-token", "keyring")


def test_generation_secret_still_encrypted_with_no_other_source_is_missing(
    monkeypatch,
):
    from tldw_chatbook.Media_Generation.config_machinery import resolve_secret

    monkeypatch.delenv("TLDW_TEST_SWARM_TOKEN", raising=False)
    _field, value, source = resolve_secret(
        "swarmui",
        {"swarm_token": _enc("swarm-real")},
        _tables(),
        keyring_lookup=lambda _id: None,
    )

    assert (value, source) == (None, "missing")


# --- SwarmUI client ----------------------------------------------------------


def _swarm_client(monkeypatch, settings, **kwargs):
    from tldw_chatbook.Media_Creation import swarmui_client

    monkeypatch.setattr(swarmui_client, "load_settings", lambda: settings)
    return swarmui_client.SwarmUIClient(**kwargs)


def test_swarmui_client_never_holds_an_encrypted_config_key(monkeypatch):
    client = _swarm_client(
        monkeypatch,
        {"media_creation": {"swarmui": {"api_key": _enc("swarm-real")}}},
    )

    assert client.api_key == ""


def test_swarmui_client_never_holds_an_encrypted_explicit_key(monkeypatch):
    client = _swarm_client(
        monkeypatch,
        {"media_creation": {"swarmui": {"api_key": "swarm-plain"}}},
        api_key=_enc("swarm-real"),
    )

    assert client.api_key == "swarm-plain"


# --- tldw server client -------------------------------------------------------


def _server_client(monkeypatch, api_config: dict):
    from tldw_chatbook.runtime_policy import bootstrap

    # TLS trust reads the profile's [network] table; the credential choice
    # under test does not depend on it.
    monkeypatch.setattr(bootstrap, "httpx_verify", lambda: True)
    return bootstrap.build_runtime_api_client(
        app_config={"tldw_api": {"base_url": "http://127.0.0.1:9", **api_config}}
    )


def test_server_client_never_sends_an_encrypted_token(monkeypatch):
    client = _server_client(monkeypatch, {"auth_token": _enc("server-real")})

    assert client.token is None
    assert client.bearer_token is None


def test_server_client_skips_an_encrypted_token_for_the_next_plain_one(monkeypatch):
    client = _server_client(
        monkeypatch, {"auth_token": _enc("server-real"), "api_key": "server-plain"}
    )

    assert client.token == "server-plain"


def test_server_client_never_sends_an_encrypted_bearer_token(monkeypatch):
    client = _server_client(
        monkeypatch, {"auth_mode": "bearer", "bearer_token": _enc("server-bearer")}
    )

    assert client.bearer_token is None


def test_server_client_never_sends_an_encrypted_explicit_token(monkeypatch):
    # TE-4: server_context.build_client and the unified-MCP factory pass
    # auth_token= explicitly; that path had no pin.
    from tldw_chatbook.runtime_policy import bootstrap

    monkeypatch.setattr(bootstrap, "httpx_verify", lambda: True)
    client = bootstrap.build_runtime_api_client(
        endpoint_url="http://127.0.0.1:9",
        auth_token=_enc("server-explicit"),
        app_config={"tldw_api": {"base_url": "http://127.0.0.1:9"}},
    )

    assert client.token is None
    assert client.bearer_token is None


def test_a_stranded_api_key_does_not_change_how_the_auth_token_is_sent(monkeypatch):
    # Review v5r1 (R1-7): auth_mode unset, a plain auth_token, a stranded
    # `enc:` api_key and a plain bearer_token. The auth_token always went as
    # the API key (the api_key's presence picks api-key mode); screening the
    # api_key flipped the inferred mode to bearer, so the same valid token
    # went out as a Bearer header instead -- a working server could 401.
    client = _server_client(
        monkeypatch,
        {
            "auth_token": "server-plain",
            "api_key": _enc("server-real"),
            "bearer_token": "server-bearer",
        },
    )

    assert client.token == "server-plain"
    assert client.bearer_token is None


def test_a_bearer_token_used_in_place_of_a_stranded_key_is_sent_as_bearer(
    monkeypatch,
):
    # When the bearer token is the credential actually chosen, it must go as
    # a Bearer header, never in the API-key header.
    client = _server_client(
        monkeypatch, {"api_key": _enc("server-real"), "bearer_token": "server-bearer"}
    )

    assert client.bearer_token == "server-bearer"
    assert client.token is None


@pytest.mark.parametrize(
    ("api_config", "token", "bearer"),
    [
        ({"auth_token": "t", "bearer_token": "b"}, None, "t"),
        ({"api_key": "k", "bearer_token": "b"}, "k", None),
        ({"bearer_token": "b"}, None, "b"),
        ({"auth_token": "t", "api_key": "k"}, "t", None),
    ],
    ids=["token+bearer", "key+bearer", "bearer-only", "token+key"],
)
def test_server_client_auth_mode_is_unchanged_without_ciphertext(
    monkeypatch, api_config, token, bearer
):
    # Guard: with no `enc:` value the inferred mode is exactly the old rule
    # ("bearer" iff a bearer_token and no api_key).
    client = _server_client(monkeypatch, api_config)

    assert (client.token, client.bearer_token) == (token, bearer)


def test_server_client_still_sends_a_plain_token(monkeypatch):
    client = _server_client(monkeypatch, {"auth_token": "server-plain"})

    assert client.token == "server-plain"
