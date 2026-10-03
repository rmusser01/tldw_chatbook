"""Key-encryption lifecycle safety (TASK-34100.4).

The first-run review (2026-10-02, protect-summary-04) found that a second
``enable_config_encryption`` call with a different password rewrote the
password verifier over keys still encrypted under the first password, then
failed to publish -- leaving the file changed, the session poisoned and both
passwords locked out. ``disable_config_encryption`` and
``change_encryption_password`` decrypted non-strictly, so a file whose keys did
not match its verifier was rewritten with ciphertext stored as plain values.

These tests pin the repaired contract against a real temp config file:

* enabling refuses an already-encrypted file and never writes it;
* every lifecycle call validates before writing and, if anything fails after
  the write, restores the previous file bytes and in-process password;
* a reset strips every ``enc:`` value and the ``[encryption]`` table;
* an ``enc:`` value is never a usable provider credential.
"""

from __future__ import annotations

import tomllib

import pytest
import toml

import tldw_chatbook.config as _bootstrap_config
from Tests.Backup_Recovery.config_test_support import install_config_source
from tldw_chatbook.Utils.config_encryption import ConfigEncryption

PASSWORD_A = "first-master-pw"
PASSWORD_B = "second-master-pw"
PLAINTEXT_KEY = "sk-proj-lifecycle-plaintext-openai-key"


def _reset_config_state(module) -> None:
    module.clear_encryption_password()
    module._SETTINGS_CACHE = None
    module._SETTINGS_CACHE_SOURCE = None
    module._CONFIG_CACHE = None
    module._CONFIG_CACHE_SOURCE = None


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """A real config module whose selected source is an isolated temp file.

    The file holds one plaintext provider key. ``install_config_source``
    imports a fresh module bound to the per-test selection, so config writes
    pass the ADR-126 participant admission.
    """
    path = tmp_path / "profile" / "config.toml"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(
        "[general]\nusers_name = \"lifecycle\"\n\n"
        f"[api_settings.openai]\napi_key = \"{PLAINTEXT_KEY}\"\nmodel = \"gpt-4o\"\n"
    )
    path.chmod(0o600)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(path))
    module = install_config_source(monkeypatch)
    monkeypatch.setattr(
        module, "DEFAULT_CONFIG_PATH", tmp_path / "decoy" / "config.toml"
    )
    _reset_config_state(module)
    module.lifecycle_test_path = path
    yield module
    _reset_config_state(module)


@pytest.fixture
def config_path(cfg):
    return cfg.lifecycle_test_path


def _stranded_document() -> dict:
    """Keys encrypted under A while the verifier was rewritten for B.

    This is exactly the state the second-password enable produced.
    """
    engine = ConfigEncryption()
    return {
        "general": {"users_name": "lifecycle"},
        "api_settings": {
            "openai": {
                "api_key": engine.encrypt_value(PLAINTEXT_KEY, PASSWORD_A),
                "model": "gpt-4o",
            }
        },
        "encryption": {
            "enabled": True,
            "method": "AES-256-GCM-scrypt",
            "version": 1,
            "password_verifier": engine.create_password_verifier(PASSWORD_B),
        },
    }


def test_second_enable_with_another_password_is_refused_byte_identical(cfg, config_path):
    assert cfg.enable_config_encryption(PASSWORD_A) is True
    encrypted_bytes = config_path.read_bytes()

    assert cfg.enable_config_encryption(PASSWORD_B) is False

    assert config_path.read_bytes() == encrypted_bytes
    assert cfg.get_encryption_password() == PASSWORD_A
    document = tomllib.loads(encrypted_bytes.decode("utf-8"))
    decrypted = ConfigEncryption().decrypt_config_strict(document, PASSWORD_A)
    assert decrypted["api_settings"]["openai"]["api_key"] == PLAINTEXT_KEY


def test_second_enable_with_the_same_password_is_an_idempotent_no_op(
    cfg, config_path
):
    # Review round 2 (R2-F4): re-entering the SAME password after a successful
    # enable (the pre-fix wizard reopens its dialog on Enter) used to return
    # False, and the wizard then claimed the keys were "unchanged (plain
    # text)" while they were encrypted. The same password changes nothing on
    # disk, so it is a success that writes nothing.
    assert cfg.enable_config_encryption(PASSWORD_A) is True
    encrypted_bytes = config_path.read_bytes()
    cfg.clear_encryption_password()

    assert cfg.enable_config_encryption(PASSWORD_A) is True

    assert config_path.read_bytes() == encrypted_bytes
    assert cfg.get_encryption_password() == PASSWORD_A


def test_same_password_enable_over_a_plaintext_secret_is_refused(cfg, config_path):
    # The no-op is only for a file that is already fully encrypted under the
    # typed password; one that still holds a plaintext secret is refused
    # untouched rather than reported as encrypted.
    assert cfg.enable_config_encryption(PASSWORD_A) is True
    document = tomllib.loads(config_path.read_text())
    document["api_settings"]["anthropic"] = {"api_key": "sk-ant-plain-sentinel"}
    config_path.write_text(toml.dumps(document))
    before = config_path.read_bytes()

    assert cfg.enable_config_encryption(PASSWORD_A) is False

    assert config_path.read_bytes() == before


def test_same_password_enable_over_stranded_keys_is_refused(cfg, config_path):
    # Verifier for B, keys under A: B passes the verifier but cannot read the
    # keys, so "already encrypted with this password" would be false.
    config_path.write_text(toml.dumps(_stranded_document()))
    before = config_path.read_bytes()

    assert cfg.enable_config_encryption(PASSWORD_B) is False

    assert config_path.read_bytes() == before


def test_enable_validates_before_writing(cfg, config_path, monkeypatch):
    """A document that would not strict-decrypt is never written."""
    original = config_path.read_bytes()
    writes = []
    real_write = cfg._write_raw_cli_config_unlocked

    def spy(*args, **kwargs):
        writes.append(args)
        return real_write(*args, **kwargs)

    monkeypatch.setattr(cfg, "_write_raw_cli_config_unlocked", spy)
    engine = cfg.get_encryption_module()
    monkeypatch.setattr(
        engine, "encrypt_value", lambda value, password: "enc:not-a-real-ciphertext"
    )

    assert cfg.enable_config_encryption(PASSWORD_A) is False
    assert writes == []
    assert config_path.read_bytes() == original
    assert cfg.get_encryption_password() is None


def test_enable_publish_failure_restores_file_bytes_and_password(
    cfg, config_path, monkeypatch
):
    original = config_path.read_bytes()

    def failing_publish(*_args, **_kwargs):
        raise ValueError("Configuration runtime reload failed")

    monkeypatch.setattr(cfg, "_publish_runtime_config_unlocked", failing_publish)

    assert cfg.enable_config_encryption(PASSWORD_A) is False
    assert config_path.read_bytes() == original
    assert cfg.get_encryption_password() is None


def test_enable_on_a_missing_file_removes_it_again_on_failure(
    cfg, config_path, monkeypatch
):
    config_path.unlink()

    def failing_publish(*_args, **_kwargs):
        raise ValueError("Configuration runtime reload failed")

    monkeypatch.setattr(cfg, "_publish_runtime_config_unlocked", failing_publish)

    assert cfg.enable_config_encryption(PASSWORD_A) is False
    assert not config_path.exists()


def test_disable_refuses_a_stranded_file_without_writing(cfg, config_path):
    config_path.write_text(toml.dumps(_stranded_document()))
    stranded = config_path.read_bytes()

    # B passes the verifier but cannot decrypt the key: the old non-strict
    # path wrote the ciphertext back as a plain value and dropped encryption.
    assert cfg.disable_config_encryption(PASSWORD_B) is False

    assert config_path.read_bytes() == stranded
    assert cfg.get_encryption_password() is None


def test_change_password_refuses_a_stranded_file_without_writing(cfg, config_path):
    config_path.write_text(toml.dumps(_stranded_document()))
    stranded = config_path.read_bytes()

    assert cfg.change_encryption_password(PASSWORD_B, "third-master-pw") is False

    assert config_path.read_bytes() == stranded
    assert cfg.get_encryption_password() is None


def test_disable_publish_failure_restores_file_bytes_and_password(
    cfg, config_path, monkeypatch
):
    assert cfg.enable_config_encryption(PASSWORD_A) is True
    encrypted = config_path.read_bytes()

    def failing_publish(*_args, **_kwargs):
        raise ValueError("Configuration runtime reload failed")

    monkeypatch.setattr(cfg, "_publish_runtime_config_unlocked", failing_publish)

    assert cfg.disable_config_encryption(PASSWORD_A) is False
    assert config_path.read_bytes() == encrypted
    assert cfg.get_encryption_password() == PASSWORD_A


def test_change_password_publish_failure_restores_file_bytes_and_password(
    cfg, config_path, monkeypatch
):
    assert cfg.enable_config_encryption(PASSWORD_A) is True
    encrypted = config_path.read_bytes()

    def failing_publish(*_args, **_kwargs):
        raise ValueError("Configuration runtime reload failed")

    monkeypatch.setattr(cfg, "_publish_runtime_config_unlocked", failing_publish)

    assert cfg.change_encryption_password(PASSWORD_A, PASSWORD_B) is False
    assert config_path.read_bytes() == encrypted
    assert cfg.get_encryption_password() == PASSWORD_A


def _double_fault(cfg, monkeypatch) -> None:
    """The publish fails AND putting the previous bytes back fails too."""

    def failing_publish(*_args, **_kwargs):
        raise ValueError("Configuration runtime reload failed")

    def failing_restore(*_args, **_kwargs):
        raise OSError("restore refused")

    monkeypatch.setattr(cfg, "_publish_runtime_config_unlocked", failing_publish)
    monkeypatch.setattr(cfg, "_restore_previous_config_unlocked", failing_restore)


def test_change_password_double_fault_keeps_the_password_the_file_holds(
    cfg, config_path, monkeypatch
):
    # Review round 1 (F7): when the rollback itself fails the file holds the
    # NEW document. Reverting the session to the old password would encrypt
    # the next save under A beside a verifier for B -- the stranded state.
    assert cfg.enable_config_encryption(PASSWORD_A) is True
    _double_fault(cfg, monkeypatch)

    assert cfg.change_encryption_password(PASSWORD_A, PASSWORD_B) is False

    document = tomllib.loads(config_path.read_text())
    ConfigEncryption().decrypt_config_strict(document, PASSWORD_B)
    assert cfg.get_encryption_password() == PASSWORD_B


def test_disable_double_fault_keeps_the_session_unlocked_state_of_the_file(
    cfg, config_path, monkeypatch
):
    assert cfg.enable_config_encryption(PASSWORD_A) is True
    _double_fault(cfg, monkeypatch)

    assert cfg.disable_config_encryption(PASSWORD_A) is False

    assert "encryption" not in tomllib.loads(config_path.read_text())
    assert cfg.get_encryption_password() is None


def test_enable_double_fault_keeps_the_password_the_file_holds(
    cfg, config_path, monkeypatch
):
    _double_fault(cfg, monkeypatch)

    assert cfg.enable_config_encryption(PASSWORD_A) is False

    document = tomllib.loads(config_path.read_text())
    assert document["encryption"]["enabled"] is True
    assert cfg.get_encryption_password() == PASSWORD_A


def test_stranded_values_are_named_when_encryption_is_off(cfg, config_path):
    # Review round 1 (G4-R1-F5): the Settings card names the stuck value
    # when it refuses to encrypt over it.
    document = _stranded_document()
    del document["encryption"]
    config_path.write_text(toml.dumps(document))

    stranded = config_path.read_bytes()

    assert cfg.encrypted_value_paths_on_disk() == ["api_settings.openai.api_key"]
    # The key is under A; encrypting with B would strand it again.
    assert cfg.enable_config_encryption(PASSWORD_B) is False
    assert config_path.read_bytes() == stranded


def test_change_password_then_disable_round_trips_the_key(cfg, config_path):
    assert cfg.enable_config_encryption(PASSWORD_A) is True
    assert cfg.change_encryption_password(PASSWORD_A, PASSWORD_B) is True
    assert cfg.get_encryption_password() == PASSWORD_B
    assert cfg.disable_config_encryption(PASSWORD_A) is False
    assert cfg.disable_config_encryption(PASSWORD_B) is True

    restored = tomllib.loads(config_path.read_text())
    assert "encryption" not in restored
    assert restored["api_settings"]["openai"]["api_key"] == PLAINTEXT_KEY
    assert cfg.get_encryption_password() is None


def test_verify_encryption_password_reads_the_saved_verifier(cfg, config_path):
    assert cfg.verify_config_encryption_password(PASSWORD_A) is False
    assert cfg.enable_config_encryption(PASSWORD_A) is True
    assert cfg.verify_config_encryption_password(PASSWORD_A) is True
    assert cfg.verify_config_encryption_password(PASSWORD_B) is False


def test_reset_strips_every_encrypted_value_and_the_encryption_table(cfg, config_path):
    engine = ConfigEncryption()
    document = _stranded_document()
    document["API"] = {"anthropic_api_key": engine.encrypt_value("sk-ant", PASSWORD_A)}
    document["search_engines"] = {
        "serper_api_key": engine.encrypt_value("serper-secret", PASSWORD_A),
        "default": "serper",
    }
    config_path.write_text(toml.dumps(document))
    cfg.set_encryption_password(PASSWORD_A)

    assert cfg.reset_encrypted_config_values() is True

    text = config_path.read_text()
    assert "enc:" not in text
    restored = tomllib.loads(text)
    assert "encryption" not in restored
    assert "api_key" not in restored["api_settings"]["openai"]
    assert restored["api_settings"]["openai"]["model"] == "gpt-4o"
    assert "anthropic_api_key" not in restored["API"]
    assert restored["search_engines"] == {"default": "serper"}
    assert restored["general"] == {"users_name": "lifecycle"}
    assert cfg.get_encryption_password() is None


def test_rekey_adopts_an_earlier_password_that_reads_every_key(cfg, config_path):
    # Review round 2 (R2-F5): in the stranded state (verifier for B, keys
    # under A) the user who remembers A could only reset -- deleting keys A
    # still reads. A password that strict-decrypts every saved value becomes
    # the master password again; the old verifier (B) stops working.
    config_path.write_text(toml.dumps(_stranded_document()))

    assert cfg.rekey_encryption_verifier(PASSWORD_A) is True

    document = tomllib.loads(config_path.read_text())
    engine = ConfigEncryption()
    verifier = document["encryption"]["password_verifier"]
    assert engine.verify_password(PASSWORD_A, verifier) is True
    assert engine.verify_password(PASSWORD_B, verifier) is False
    decrypted = engine.decrypt_config_strict(document, PASSWORD_A)
    assert decrypted["api_settings"]["openai"]["api_key"] == PLAINTEXT_KEY
    assert document["encryption"]["enabled"] is True
    assert cfg.get_encryption_password() == PASSWORD_A


def test_rekey_repairs_a_missing_verifier(cfg, config_path):
    document = _stranded_document()
    del document["encryption"]["password_verifier"]
    config_path.write_text(toml.dumps(document))

    assert cfg.rekey_encryption_verifier(PASSWORD_A) is True

    verifier = tomllib.loads(config_path.read_text())["encryption"][
        "password_verifier"
    ]
    assert ConfigEncryption().verify_password(PASSWORD_A, verifier) is True


@pytest.mark.parametrize("password", [PASSWORD_B, "not-any-password"])
def test_rekey_refuses_a_password_that_cannot_read_the_keys(
    cfg, config_path, password
):
    config_path.write_text(toml.dumps(_stranded_document()))
    before = config_path.read_bytes()

    assert cfg.rekey_encryption_verifier(password) is False

    assert config_path.read_bytes() == before
    assert cfg.get_encryption_password() is None


def test_rekey_refuses_when_nothing_is_encrypted(cfg, config_path):
    # With no encrypted value every password "reads every key"; adopting one
    # would let any guess replace the verifier.
    document = _stranded_document()
    document["api_settings"]["openai"]["api_key"] = ""
    config_path.write_text(toml.dumps(document))
    before = config_path.read_bytes()

    assert cfg.rekey_encryption_verifier("any-guess-at-all") is False

    assert config_path.read_bytes() == before


def test_rekey_refuses_when_encryption_is_off(cfg, config_path):
    before = config_path.read_bytes()

    assert cfg.rekey_encryption_verifier(PASSWORD_A) is False

    assert config_path.read_bytes() == before


@pytest.mark.parametrize(
    "value",
    [
        "enc:AbCdEf0123456789==",
        "  enc:AbCdEf0123456789==  ",
    ],
)
def test_ciphertext_is_never_a_usable_provider_key(value):
    assert _bootstrap_config.resolve_provider_api_key(value) is None
    assert _bootstrap_config.is_valid_provider_api_key(value) is False


def test_encrypted_prefix_matches_the_encryption_engine():
    assert (
        _bootstrap_config.ENCRYPTED_CONFIG_VALUE_PREFIX
        == ConfigEncryption.ENCRYPTION_PREFIX
    )


@pytest.mark.parametrize(
    ("provider", "key"),
    [("llama_cpp", "llama_cpp"), ("Ollama", "ollama"), ("OpenAI", "openai")],
)
def test_a_still_encrypted_saved_key_is_never_ready(provider, key):
    # Review round 2 (F-R2-2): a local provider needs no key, so a saved key
    # that was still `enc:` read "Ready" -- and the send then failed with a
    # bare HTTP 401 from a server that does require it. Any provider with a
    # still-encrypted saved key (and no other credential) is blocked with
    # copy that says why and where to fix it.
    from tldw_chatbook.Chat.provider_readiness import get_provider_readiness

    ciphertext = ConfigEncryption().encrypt_value(PLAINTEXT_KEY, PASSWORD_A)
    readiness = get_provider_readiness(
        provider,
        {
            "api_settings": {
                key: {"api_key": ciphertext, "api_url": "http://127.0.0.1:9/v1"}
            }
        },
        environ={},
    )

    assert readiness.ready is False
    assert readiness.api_key is None
    assert readiness.reason == "Saved API key is still encrypted"
    assert readiness.configuration_issue == "credential_missing"
    assert "Providers & Models" in readiness.recovery


@pytest.mark.parametrize("provider", ["llama_cpp", "ollama"])
def test_console_blocks_a_keyless_provider_whose_saved_key_is_still_encrypted(
    provider,
):
    # Found live in review round 2: Console mapped every keyless provider's
    # credential to "not_required", so the blocked readiness above produced
    # no blocker while its configuration was incomplete -- and the Console
    # screen failed to load ("Ready Console settings contain blocking
    # facets").
    from tldw_chatbook.Chat.console_session_settings import (
        ConsoleSessionSettings,
        build_console_settings_readiness,
    )

    ciphertext = ConfigEncryption().encrypt_value(PLAINTEXT_KEY, PASSWORD_A)
    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(provider=provider, model="local-model-sentinel"),
        app_config={"api_settings": {provider: {"api_key": ciphertext}}},
        environ={},
    )

    assert readiness.operability == "not_ready"
    assert readiness.blocker == "credential_missing"
    assert readiness.label == "Missing key"
    assert "still encrypted" in readiness.detail
    assert "Providers & Models" in readiness.detail


def test_an_environment_key_still_beats_a_stranded_saved_key():
    from tldw_chatbook.Chat.provider_readiness import get_provider_readiness

    ciphertext = ConfigEncryption().encrypt_value(PLAINTEXT_KEY, PASSWORD_A)
    readiness = get_provider_readiness(
        "OpenAI",
        {"api_settings": {"openai": {"api_key": ciphertext}}},
        environ={"OPENAI_API_KEY": "sk-env-sentinel-key"},
    )

    assert readiness.ready is True
    assert readiness.api_key == "sk-env-sentinel-key"


def test_a_deliberately_keyless_provider_ignores_a_stranded_saved_key():
    from tldw_chatbook.Chat.provider_readiness import get_provider_readiness

    ciphertext = ConfigEncryption().encrypt_value(PLAINTEXT_KEY, PASSWORD_A)
    readiness = get_provider_readiness(
        "llama_cpp",
        {"api_settings": {"llama_cpp": {"api_key": ciphertext, "credential_source": "none"}}},
        environ={},
    )

    assert readiness.ready is True


def test_locked_config_reads_its_provider_as_key_missing():
    from tldw_chatbook.Chat.provider_readiness import get_provider_readiness

    ciphertext = ConfigEncryption().encrypt_value(PLAINTEXT_KEY, PASSWORD_A)
    readiness = get_provider_readiness(
        "OpenAI",
        {"api_settings": {"openai": {"api_key": ciphertext}}},
        environ={},
    )
    assert readiness.ready is False
    assert readiness.api_key is None
