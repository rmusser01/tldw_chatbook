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

import stat

import toml
import pytest

import tldw_chatbook.config as cfg
from tldw_chatbook.Utils.config_encryption import config_encryption


def _mode(path) -> int:
    return stat.S_IMODE(path.stat().st_mode)

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


def test_app_quit_save_writes_active_file_not_default(isolated_config_paths):
    """Regression test for task-851 review finding 1.

    ``enable_config_encryption``/``disable_config_encryption``/
    ``change_encryption_password`` are all user-initiated and were fixed to
    read/write ``_get_effective_config_path()`` above. But there is a
    fourth, *automatic* write path this task's original fix missed:
    ``TldwCli.action_quit()``'s on-exit "save encrypted config if enabled"
    block, which still hardcoded ``DEFAULT_CONFIG_PATH``.

    This is the dangerous one specifically *because* the other three were
    fixed: before that fix, enabling encryption under a profile wrote
    ``[encryption] enabled=true`` into the *default* file (which the
    profile-reading loader never saw), so this quit-time block -- reading
    the active profile via ``load_cli_config_and_ensure_existence()`` --
    never saw ``enabled=True`` either and stayed a no-op. Once enable/
    disable/change started reading and writing the *active* file, every
    quit began serializing the merged active-profile config over the
    user's separate default config file: silently destroying its content
    and replacing its identity with the profile's, while the profile
    itself was never re-encrypted.

    Reproduced live before fixing (see
    ``.superpowers/sdd/review-fixes-report.md``): the default config file
    grew from 56 to 31004 bytes and its original API key was replaced by
    the profile's merged/encrypted content.
    """
    profile_path, decoy_path = isolated_config_paths
    profile_path.write_text(
        "[encryption]\n"
        "enabled = true\n\n"
        "[api_settings.openai]\n"
        f'api_key = "{PLAINTEXT_KEY}"\n'
    )

    decoy_path.parent.mkdir(parents=True, exist_ok=True)
    decoy_path.write_text(
        '[api_settings.openai]\napi_key = "sk-DEFAULT-CONFIG-UNRELATED-KEY"\n'
    )
    decoy_before = decoy_path.read_bytes()

    cfg.set_encryption_password(PASSWORD)

    from tldw_chatbook.app import TldwCli

    app = TldwCli()
    try:
        app.action_quit()
    finally:
        cfg.clear_encryption_password()

    # The decoy DEFAULT_CONFIG_PATH file must be byte-for-byte untouched --
    # the on-quit save must never write it once a profile is active.
    assert decoy_path.read_bytes() == decoy_before

    # The active profile file is the one that got encrypted on exit.
    active_data = toml.load(profile_path)
    assert active_data["encryption"]["enabled"] is True
    active_key = active_data["api_settings"]["openai"]["api_key"]
    assert config_encryption.is_encrypted(active_key)


def test_encrypt_decrypt_preserve_pre_existing_restrictive_mode(
    isolated_config_paths,
):
    """Regression test for task-851 review finding 2.

    ``enable_config_encryption``/``disable_config_encryption`` write through
    ``atomic_write_text``, which used to always ``chmod`` the replacement
    file to its generic 0o644 default -- widening permissions on a config
    file the user (or a prior write) had tightened to 0o600. Measured live
    before fixing: 0600 -> 0644 after enabling encryption, and disabling
    again left it at 0644 while holding plaintext API keys.

    This asserts the file's pre-existing 0600 mode survives both the
    enable and the disable rewrite.
    """
    profile_path, _decoy_path = isolated_config_paths
    _write_plain_profile(profile_path)
    profile_path.chmod(0o600)
    assert _mode(profile_path) == 0o600

    assert cfg.enable_config_encryption(PASSWORD) is True
    assert _mode(profile_path) == 0o600

    assert cfg.disable_config_encryption(PASSWORD) is True
    assert _mode(profile_path) == 0o600
    # Still holds the (now decrypted) plaintext key, so the restrictive
    # mode matters just as much post-disable as pre-enable.
    assert toml.load(profile_path)["api_settings"]["openai"]["api_key"] == (
        PLAINTEXT_KEY
    )


def test_enable_config_encryption_creates_new_file_with_restrictive_mode(
    tmp_path, monkeypatch
):
    """When there is no pre-existing config file to preserve the mode of,
    a freshly created one must still get a restrictive (not 0o644) mode --
    it can hold plaintext API keys and the password verifier.
    """
    monkeypatch.delenv("TLDW_CONFIG_PATH", raising=False)
    config_path = tmp_path / "profile" / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cfg, "DEFAULT_CONFIG_PATH", config_path)
    assert not config_path.exists()

    try:
        assert cfg.enable_config_encryption(PASSWORD) is True
        assert _mode(config_path) == cfg.CONFIG_SECRETS_FILE_MODE
        assert _mode(config_path) == 0o600
    finally:
        cfg.clear_encryption_password()
        cfg._SETTINGS_CACHE = None
        cfg._SETTINGS_CACHE_SOURCE = None
        cfg._CONFIG_CACHE = None
        cfg._CONFIG_CACHE_SOURCE = None
