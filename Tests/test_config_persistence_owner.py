from __future__ import annotations

import base64
import inspect
import os
import stat
from pathlib import Path

import pytest
import toml

from tldw_chatbook import config
import sys
from Tests.Backup_Recovery.config_test_support import install_config_source
from tldw_chatbook.Utils.config_encryption import config_encryption


def _reset_config_state() -> None:
    config._CONFIG_CACHE = None
    config._CONFIG_CACHE_SOURCE = None
    config._SETTINGS_CACHE = None
    config._SETTINGS_CACHE_SOURCE = None
    config.clear_encryption_password()


@pytest.fixture(autouse=True)
def reset_config_state():
    _reset_config_state()
    yield
    _reset_config_state()


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")
def test_serialized_raw_replace_backup_and_downgrade_guard_use_effective_path(
    tmp_path,
    monkeypatch,
):
    password = "owner-boundary-password"
    config_path = tmp_path / "override" / "config.toml"
    config_path.parent.mkdir(mode=0o700)
    ignored_default = tmp_path / "default" / "config.toml"
    encrypted = config.encrypt_api_keys_in_config(
        {
            "encryption": {"enabled": True},
            "api_settings": {"openai": {"api_key": "old-SENTINEL"}},
        },
        password,
    )
    config_path.write_text(toml.dumps(encrypted), encoding="utf-8")
    config_path.chmod(0o644)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(sys.modules[__name__], "config", install_config_source(monkeypatch))
    monkeypatch.setattr(config, "DEFAULT_CONFIG_PATH", ignored_default)
    config.set_encryption_password(password)

    serialized = config.read_cli_config_serialized()
    loaded, backup = config.replace_cli_config_serialized(
        "[encryption]\nenabled = true\n"
        '[api_settings.openai]\napi_key = "new-SENTINEL"\n',
        create_backup=True,
    )

    assert "old-SENTINEL" not in serialized
    assert loaded["api_settings"]["openai"]["api_key"] == "new-SENTINEL"
    assert backup == config_path.with_suffix(".toml.bak")
    assert backup is not None
    assert "old-SENTINEL" not in backup.read_text(encoding="utf-8")
    assert config_encryption.is_encrypted(
        toml.load(backup)["api_settings"]["openai"]["api_key"]
    )
    assert config_encryption.is_encrypted(
        toml.load(config_path)["api_settings"]["openai"]["api_key"]
    )
    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    assert not ignored_default.exists()

    before = config_path.read_bytes()
    with pytest.raises(ValueError, match="disable encryption explicitly"):
        config.replace_cli_config_serialized(
            '[api_settings.openai]\napi_key = "plaintext-downgrade-SENTINEL"\n'
        )
    assert config_path.read_bytes() == before


@pytest.mark.skipif(os.name != "posix", reason="POSIX shutdown contract")
def test_shutdown_persistence_uses_only_effective_path(tmp_path, monkeypatch):
    password = "shutdown-password"
    target = tmp_path / "override" / "config.toml"
    target.parent.mkdir(mode=0o700)
    ignored_default = tmp_path / "ignored" / "config.toml"
    encrypted = config.encrypt_api_keys_in_config(
        {
            "encryption": {"enabled": True},
            "API": {"openai_api_key": "shutdown-SENTINEL"},
        },
        password,
    )
    # TASK-35835: a plaintext secret hand-added to the encrypted file is what
    # the quit still persists; with none, it writes nothing.
    encrypted["API"]["anthropic_api_key"] = "hand-added-SENTINEL"
    target.write_text(toml.dumps(encrypted), encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    monkeypatch.setattr(sys.modules[__name__], "config", install_config_source(monkeypatch))
    monkeypatch.setattr(config, "DEFAULT_CONFIG_PATH", ignored_default)
    config.set_encryption_password(password)
    config.load_cli_config_and_ensure_existence(force_reload=True)

    assert config.persist_cli_config_for_shutdown() is True

    for key in ("openai_api_key", "anthropic_api_key"):
        assert config_encryption.is_encrypted(toml.load(target)["API"][key])
    assert "SENTINEL" not in target.read_text(encoding="utf-8")
    assert "api_settings" not in toml.load(target)  # its own contents, no template
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert not ignored_default.exists()
    # Encrypted, unlocked and nothing left in plaintext: the next quit writes
    # nothing (AC#2 for an encrypted profile). The comment shows a rewrite.
    after = _HAND_COMMENT + target.read_bytes()
    target.write_bytes(after)
    assert config.persist_cli_config_for_shutdown() is True
    assert target.read_bytes() == after


_LEGACY_PASSWORD = "legacy-password"
# A rewrite drops comments, so a file that keeps one was not written.
_HAND_COMMENT = b"# hand-written\n"


def _ciphertext_version(value: str) -> int:
    return base64.b64decode(value[len("enc:") :])[0]


def _shape(table):
    """The document with every ``enc:`` value replaced by a marker."""
    return {
        key: _shape(value)
        if isinstance(value, dict)
        else ("<enc>" if config_encryption.is_encrypted(value) else value)
        for key, value in table.items()
    }


@pytest.fixture(scope="module")
def legacy_encrypted_toml() -> str:
    """A pre-v2 file: one v1 key and a v1 password check (scrypt 2^20 each)."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            type(config_encryption), "VERSION", config_encryption.LEGACY_VERSION
        )
        document = config.encrypt_api_keys_in_config(
            {
                "encryption": {"enabled": True},
                "API": {"openai_api_key": "legacy-SENTINEL"},
            },
            _LEGACY_PASSWORD,
        )
    return toml.dumps(document)


def _install_legacy_file(tmp_path, monkeypatch, serialized: str) -> Path:
    target = tmp_path / "override" / "config.toml"
    target.parent.mkdir(mode=0o700)
    target.write_bytes(_HAND_COMMENT + serialized.encode("utf-8"))
    target.chmod(0o600)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    monkeypatch.setattr(sys.modules[__name__], "config", install_config_source(monkeypatch))
    return target


def test_unlocked_quit_upgrades_legacy_ciphertext_in_place(
    tmp_path, monkeypatch, legacy_encrypted_toml
):
    """TASK-35835 review: the quit is the one pass that moves v1 values to v2.

    A v1 value costs a 2^20 scrypt derivation (about 1 GiB, 3 s) at every
    launch. Dev's quit re-encrypted everything as v2; the fixed quit
    re-encrypts only the v1 values, keeps every other byte of the document's
    shape, and leaves a v2 file alone on the next quit.
    """
    target = _install_legacy_file(tmp_path, monkeypatch, legacy_encrypted_toml)
    before = toml.loads(legacy_encrypted_toml)
    assert _ciphertext_version(before["API"]["openai_api_key"]) == 1
    assert _ciphertext_version(before["encryption"]["password_verifier"]) == 1
    config.set_encryption_password(_LEGACY_PASSWORD)

    assert config.persist_cli_config_for_shutdown() is True

    after = toml.load(target)
    assert _shape(after) == _shape(before)  # no template keys, nothing decrypted
    key, verifier = after["API"]["openai_api_key"], after["encryption"]["password_verifier"]
    assert (_ciphertext_version(key), _ciphertext_version(verifier)) == (2, 2)
    assert config_encryption.decrypt_value(key, _LEGACY_PASSWORD) == "legacy-SENTINEL"
    assert config_encryption.verify_password(_LEGACY_PASSWORD, verifier)
    upgraded = _HAND_COMMENT + target.read_bytes()
    target.write_bytes(upgraded)
    assert config.persist_cli_config_for_shutdown() is True
    assert target.read_bytes() == upgraded  # a v2-only file is left alone


def test_locked_quit_leaves_legacy_ciphertext_byte_identical(
    tmp_path, monkeypatch, legacy_encrypted_toml
):
    """Without the session password nothing can be re-encrypted, so nothing is written."""
    target = _install_legacy_file(tmp_path, monkeypatch, legacy_encrypted_toml)
    before = target.read_bytes()

    assert config.persist_cli_config_for_shutdown() is True

    assert target.read_bytes() == before


def test_quit_refuses_a_value_another_password_wrote_between_its_reads(
    tmp_path, monkeypatch
):
    """The document the quit writes is validated, not just the file it loaded.

    A hand edit after the load adds a plaintext key (so the quit writes) and a
    value under another password. Writing it would give the file a new check
    value that this other value cannot pass: the stranded state
    ``rekey_encryption_verifier`` repairs.
    """
    password = "session-password"
    target = tmp_path / "override" / "config.toml"
    target.parent.mkdir(mode=0o700)
    loaded = config.encrypt_api_keys_in_config(
        {"encryption": {"enabled": True}, "API": {"openai_api_key": "kept-SENTINEL"}},
        password,
    )
    target.write_text(toml.dumps(loaded), encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    monkeypatch.setattr(sys.modules[__name__], "config", install_config_source(monkeypatch))
    edited = toml.loads(toml.dumps(loaded))
    edited["API"]["anthropic_api_key"] = "hand-added-SENTINEL"
    edited["API"]["groq_api_key"] = config_encryption.encrypt_value(
        "foreign-SENTINEL", "another-password"
    )
    edited_text = toml.dumps(edited)
    real_decrypt = config._decrypt_config_section_with_status
    edits: list[str] = []

    def decrypt_then_hand_edit(*args, **kwargs):
        # The first call is inside the quit's load, after it read the file.
        result = real_decrypt(*args, **kwargs)
        if not edits:
            edits.append(edited_text)
            target.write_text(edited_text, encoding="utf-8")
        return result

    monkeypatch.setattr(config, "_decrypt_config_section_with_status", decrypt_then_hand_edit)
    config.set_encryption_password(password)

    assert config.persist_cli_config_for_shutdown() is False

    assert edits
    assert target.read_text(encoding="utf-8") == edited_text


def test_shutdown_persistence_leaves_a_fresh_template_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TASK-35835: a quit with nothing to encrypt writes nothing.

    It used to write the template merge back, which stripped a fresh file's
    comments on the first quit and restored keys a save had deleted.
    """
    target = tmp_path / "override" / "config.toml"
    target.parent.mkdir(mode=0o700)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    monkeypatch.setattr(sys.modules[__name__], "config", install_config_source(monkeypatch))
    config.load_cli_config_and_ensure_existence(force_reload=True)
    created = target.read_bytes()
    assert created.lstrip().startswith(b"#")

    assert config.persist_cli_config_for_shutdown() is True

    assert target.read_bytes() == created


@pytest.mark.parametrize("failure", ["malformed_toml", "decryption"])
def test_shutdown_persistence_preserves_unreadable_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    target = tmp_path / "override" / "config.toml"
    target.parent.mkdir(mode=0o700)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    monkeypatch.setattr(sys.modules[__name__], "config", install_config_source(monkeypatch))
    if failure == "malformed_toml":
        target.write_text("[broken\nvalue = 1", encoding="utf-8")
    else:
        encrypted = config.encrypt_api_keys_in_config(
            {
                "encryption": {"enabled": True},
                "API": {"openai_api_key": "preserve-SENTINEL"},
            },
            "correct-password",
        )
        target.write_text(toml.dumps(encrypted), encoding="utf-8")
        config.set_encryption_password("wrong-password")
    original = target.read_bytes()

    assert config.persist_cli_config_for_shutdown() is False

    assert target.read_bytes() == original


def test_production_config_persistence_has_one_owner_and_no_mutable_imports():
    package_root = Path(config.__file__).parent
    config_source = Path(config.__file__).read_text(encoding="utf-8")
    # TASK-33011: TldwCli's code is split across app.py and the app_*.py
    # modules extracted from it (entry points and the boot-time config load in
    # app_entry.py, the quit flow's config persistence in app_lifecycle.py, ...);
    # all of them are the app's config callers, so scan them all.
    app_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [package_root / "app.py", *sorted(package_root.glob("app_*.py"))]
    )
    settings_source = (
        package_root / "UI" / "Screens" / "settings_screen.py"
    ).read_text(encoding="utf-8")

    assert "atomic_private_write_text(" in config_source
    assert "atomic_write_text(\n        config_path" not in config_source
    assert "atomic_write_text(DEFAULT_CONFIG_PATH" not in app_source
    assert 'open(DEFAULT_CONFIG_PATH, "w"' not in app_source
    assert "DEFAULT_CONFIG_PATH.parent.mkdir" not in app_source
    assert "backup_path.write_text(" not in settings_source
    assert "tmp_path.replace(config_path)" not in settings_source
    assert "config_path.read_text(" not in settings_source

    offenders = []
    for source_path in package_root.rglob("*.py"):
        if source_path == Path(config.__file__):
            continue
        source = source_path.read_text(encoding="utf-8")
        if "config import settings" in source:
            offenders.append(source_path.relative_to(package_root).as_posix())
    assert offenders == []

    assert "persist_cli_config_for_shutdown" in app_source
    assert "load_cli_config_and_ensure_existence" in app_source
    assert inspect.isfunction(config.replace_cli_config_serialized)
