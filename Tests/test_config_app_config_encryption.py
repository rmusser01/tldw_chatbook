"""Regression tests for backlog task-151.

With config encryption enabled, ``load_settings()`` (which populates
``app.app_config``) must return DECRYPTED ``api_settings`` values, never ``enc:``
ciphertext — otherwise the Chat send path passes the ciphertext to providers
verbatim and auth fails. Also verifies that ``set_encryption_password`` drops the
stale ciphertext cache primed before the startup unlock prompt.
"""

import toml
import pytest

import tldw_chatbook.config as cfg
from tldw_chatbook.Utils.config_encryption import config_encryption

PASSWORD = "test-master-pw"
PLAINTEXT_KEY = "test-plaintext-openai-key"


def _write_encrypted_config(path):
    """Write a config whose api_settings.openai.api_key is encrypted on disk."""
    plain = {
        "encryption": {"enabled": True},
        "api_settings": {"openai": {"api_key": PLAINTEXT_KEY}},
    }
    encrypted = cfg.encrypt_api_keys_in_config(plain, PASSWORD)
    path.write_text(toml.dumps(encrypted))
    on_disk = toml.load(path)["api_settings"]["openai"]["api_key"]
    assert config_encryption.is_encrypted(on_disk), "precondition: on-disk key must be ciphertext"


@pytest.fixture
def reset_config_state():
    """Ensure global encryption/cache state never leaks into other tests."""
    yield
    cfg.clear_encryption_password()
    cfg._SETTINGS_CACHE = None
    cfg._SETTINGS_CACHE_SOURCE = None
    cfg._CONFIG_CACHE = None
    cfg._CONFIG_CACHE_SOURCE = None


def test_load_settings_decrypts_api_settings_when_encrypted(tmp_path, monkeypatch, reset_config_state):
    cfg_path = tmp_path / "config.toml"
    _write_encrypted_config(cfg_path)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(cfg_path))
    cfg.set_encryption_password(PASSWORD)

    result = cfg.load_settings(force_reload=True)

    key = result["api_settings"]["openai"]["api_key"]
    assert key == PLAINTEXT_KEY
    assert not config_encryption.is_encrypted(key)


def test_set_encryption_password_invalidates_stale_ciphertext_cache(tmp_path, monkeypatch, reset_config_state):
    """Mirrors app startup: config is cached (as ciphertext) before the unlock
    prompt, then the password is entered — the next load must decrypt."""
    cfg_path = tmp_path / "config.toml"
    _write_encrypted_config(cfg_path)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(cfg_path))

    # 1) No password yet: load primes the cache with ciphertext.
    cfg.clear_encryption_password()
    cfg._SETTINGS_CACHE = None
    cfg._SETTINGS_CACHE_SOURCE = None
    primed = cfg.load_settings(force_reload=True)
    assert config_encryption.is_encrypted(primed["api_settings"]["openai"]["api_key"])

    # 2) Entering the password must invalidate the stale cache...
    cfg.set_encryption_password(PASSWORD)
    # 3) ...so the next NON-forced load re-reads and decrypts.
    result = cfg.load_settings()

    assert result["api_settings"]["openai"]["api_key"] == PLAINTEXT_KEY
