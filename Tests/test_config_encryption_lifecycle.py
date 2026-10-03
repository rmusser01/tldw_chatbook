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


def test_second_enable_with_the_same_password_is_also_refused(cfg, config_path):
    assert cfg.enable_config_encryption(PASSWORD_A) is True
    encrypted_bytes = config_path.read_bytes()

    assert cfg.enable_config_encryption(PASSWORD_A) is False
    assert config_path.read_bytes() == encrypted_bytes


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
