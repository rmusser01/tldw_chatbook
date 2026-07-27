"""Regression tests for task-851.

``enable_config_encryption``, ``disable_config_encryption`` and
``change_encryption_password`` used to read/write ``DEFAULT_CONFIG_PATH``
directly instead of ``_get_effective_config_path()``. Any user running with a
``TLDW_CONFIG_PATH`` override active (e.g. the "Override config" profile
control) would have encryption silently operate on a *different* file than
the one actually in use: "enable encryption" reported success while leaving
the active profile's secrets in plaintext and rewriting an unrelated file.

These tests set ``TLDW_CONFIG_PATH`` to a profile file, point
``DEFAULT_CONFIG_PATH`` at a separate "decoy" path (standing in for the
wrong file the bug used to write), and assert the change lands only in the
file ``_get_effective_config_path()`` resolves to.

A companion test asserts the "enable" write path is atomic: a config.toml
holding every API key at once must never be left truncated by a crash
mid-write.
"""

import toml
import pytest

import tldw_chatbook.config as cfg
from tldw_chatbook.Utils.config_encryption import config_encryption

PASSWORD = "test-master-pw"
NEW_PASSWORD = "test-master-pw-2"
PLAINTEXT_KEY = "sk-proj-test-plaintext-openai-key"


@pytest.fixture
def isolated_config_paths(tmp_path, monkeypatch):
    """Point the active (effective) config at a profile file, and point
    DEFAULT_CONFIG_PATH at a distinct decoy file that a correct fix must
    never touch.

    Returns:
        (profile_path, decoy_path) tuple.
    """
    profile_path = tmp_path / "profile" / "config.toml"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    decoy_path = tmp_path / "default_home" / ".config" / "tldw_cli" / "config.toml"

    monkeypatch.setenv("TLDW_CONFIG_PATH", str(profile_path))
    monkeypatch.setattr(cfg, "DEFAULT_CONFIG_PATH", decoy_path)

    yield profile_path, decoy_path

    cfg.clear_encryption_password()
    cfg._SETTINGS_CACHE = None
    cfg._SETTINGS_CACHE_SOURCE = None
    cfg._CONFIG_CACHE = None
    cfg._CONFIG_CACHE_SOURCE = None


def _write_plain_profile(profile_path, extra_toml: str = "") -> None:
    profile_path.write_text(
        '[api_settings.openai]\n'
        f'api_key = "{PLAINTEXT_KEY}"\n' + extra_toml
    )


def test_enable_config_encryption_writes_active_file_not_default(
    isolated_config_paths,
):
    """enable_config_encryption must read/write the effective path."""
    profile_path, decoy_path = isolated_config_paths
    _write_plain_profile(profile_path)

    assert cfg.enable_config_encryption(PASSWORD) is True

    # The active (effective) file was rewritten with the secret encrypted.
    active_data = toml.load(profile_path)
    assert active_data["encryption"]["enabled"] is True
    active_key = active_data["api_settings"]["openai"]["api_key"]
    assert config_encryption.is_encrypted(active_key)
    assert active_key != PLAINTEXT_KEY

    # The decoy DEFAULT_CONFIG_PATH was never created or touched.
    assert not decoy_path.exists()


def test_disable_config_encryption_reads_and_writes_active_file(
    isolated_config_paths,
):
    """disable_config_encryption must read/write the effective path."""
    profile_path, decoy_path = isolated_config_paths
    _write_plain_profile(profile_path)
    assert cfg.enable_config_encryption(PASSWORD) is True

    assert cfg.disable_config_encryption(PASSWORD) is True

    active_data = toml.load(profile_path)
    assert "encryption" not in active_data
    assert active_data["api_settings"]["openai"]["api_key"] == PLAINTEXT_KEY
    assert not decoy_path.exists()


def test_change_encryption_password_reads_and_writes_active_file(
    isolated_config_paths,
):
    """change_encryption_password must read/write the effective path."""
    profile_path, decoy_path = isolated_config_paths
    _write_plain_profile(profile_path)
    assert cfg.enable_config_encryption(PASSWORD) is True

    assert cfg.change_encryption_password(PASSWORD, NEW_PASSWORD) is True

    active_data = toml.load(profile_path)
    verifier = active_data["encryption"]["password_verifier"]
    assert config_encryption.verify_password(NEW_PASSWORD, verifier) is True
    assert config_encryption.verify_password(PASSWORD, verifier) is False
    assert not decoy_path.exists()


def test_enable_disable_roundtrip_with_profile_active(isolated_config_paths):
    """Enabling then disabling encryption must restore the original secret,
    byte for byte, with a profile active via TLDW_CONFIG_PATH."""
    profile_path, _decoy_path = isolated_config_paths
    _write_plain_profile(profile_path)
    original = toml.load(profile_path)

    assert cfg.enable_config_encryption(PASSWORD) is True
    encrypted = toml.load(profile_path)
    assert config_encryption.is_encrypted(
        encrypted["api_settings"]["openai"]["api_key"]
    )

    assert cfg.disable_config_encryption(PASSWORD) is True
    restored = toml.load(profile_path)
    assert restored == original


def test_change_password_roundtrip_with_profile_active(isolated_config_paths):
    """Rotating the password must keep the config decryptable under the new
    password only, with a profile active via TLDW_CONFIG_PATH."""
    profile_path, _decoy_path = isolated_config_paths
    _write_plain_profile(profile_path)

    assert cfg.enable_config_encryption(PASSWORD) is True
    assert cfg.change_encryption_password(PASSWORD, NEW_PASSWORD) is True

    # Wrong (old) password must now fail to disable.
    assert cfg.disable_config_encryption(PASSWORD) is False
    active_data = toml.load(profile_path)
    assert active_data["encryption"]["enabled"] is True

    # New password disables and decrypts correctly.
    assert cfg.disable_config_encryption(NEW_PASSWORD) is True
    restored = toml.load(profile_path)
    assert restored["api_settings"]["openai"]["api_key"] == PLAINTEXT_KEY


def test_enable_config_encryption_write_is_atomic(tmp_path, monkeypatch):
    """A failure while serializing the encrypted config must never truncate
    the on-disk config: the write must go through a write-temp-then-replace
    helper, not a plain ``open(path, "w")`` (which truncates on open, before
    any new content -- or a raised exception -- ever reaches the file).

    Deliberately does not exercise the TLDW_CONFIG_PATH-vs-DEFAULT_CONFIG_PATH
    routing from the tests above: this isolates the atomic-write property by
    pointing DEFAULT_CONFIG_PATH itself straight at the target file, so the
    only variable under test is whether a mid-write failure can truncate it.
    """
    monkeypatch.delenv("TLDW_CONFIG_PATH", raising=False)
    config_path = tmp_path / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cfg, "DEFAULT_CONFIG_PATH", config_path)
    _write_plain_profile(config_path)
    original_bytes = config_path.read_bytes()

    def _boom(*_args, **_kwargs):
        raise RuntimeError("simulated crash while serializing config")

    # Patch both the string-returning and file-writing serializers: which
    # one is called is an implementation detail we are deliberately trying
    # not to assume here (a plain ``open(path, "w")`` implementation calls
    # ``toml.dump(data, fh)``; a write-temp-then-replace implementation
    # calls ``toml.dumps(data)`` first). Either way, this simulates the
    # serialization step failing before persistence completes.
    monkeypatch.setattr(cfg.toml, "dumps", _boom)
    monkeypatch.setattr(cfg.toml, "dump", _boom)

    assert cfg.enable_config_encryption(PASSWORD) is False
    # The file must be byte-for-byte unchanged: a plain open(path, "w")
    # would have already truncated it before toml.dumps ever ran.
    assert config_path.read_bytes() == original_bytes

    cfg.clear_encryption_password()
    cfg._SETTINGS_CACHE = None
    cfg._SETTINGS_CACHE_SOURCE = None
    cfg._CONFIG_CACHE = None
    cfg._CONFIG_CACHE_SOURCE = None
