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
    # TASK-35835: in this v2-only file, a hand-added plaintext secret is the
    # one thing the quit persists (it would also re-encrypt v1 values as v2);
    # with nothing to do, it writes nothing.
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


# TASK-38987: a config write makes ~10,000 GIL-releasing syscalls (storage
# admission and private-path checks). Beside a thread running Python, each one
# waited out that thread's 5 ms switch interval, so a 0.09 s save took 5.7-6.4 s.
# Writes now run with a short interval, restored however they end.
_SHORT_INTERVAL = 0.001


@pytest.fixture
def default_interval():
    previous = sys.getswitchinterval()
    sys.setswitchinterval(0.005)
    try:
        yield sys.getswitchinterval()
    finally:
        sys.setswitchinterval(previous)


def _fresh_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    target = tmp_path / "override" / "config.toml"
    target.parent.mkdir(mode=0o700)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    module = install_config_source(monkeypatch)
    monkeypatch.setattr(sys.modules[__name__], "config", module)
    module.load_cli_config_and_ensure_existence(force_reload=True)
    return module


def _write(module, days: int, before_replace=None):
    return module.apply_settings_mutation_to_cli_config(
        {"sync": {"sync_log_retention_days": days}}, before_replace=before_replace
    )


def test_a_config_write_runs_with_a_short_switch_interval_then_restores_it(
    tmp_path, monkeypatch, default_interval
):
    module = _fresh_config(tmp_path, monkeypatch)
    seen = []

    assert _write(module, 91, lambda: seen.append(sys.getswitchinterval())).fully_applied
    assert seen and seen[0] < _SHORT_INTERVAL
    assert sys.getswitchinterval() == default_interval


def test_a_config_write_that_raises_restores_the_switch_interval(
    tmp_path, monkeypatch, default_interval
):
    module = _fresh_config(tmp_path, monkeypatch)

    class Stop(BaseException):
        pass

    seen = []

    def stop():
        seen.append(sys.getswitchinterval())
        raise Stop

    with pytest.raises(Stop):
        _write(module, 92, stop)
    assert seen and seen[0] < _SHORT_INTERVAL
    assert sys.getswitchinterval() == default_interval


def test_overlapping_config_writes_keep_the_short_interval_until_the_last_ends(
    tmp_path, monkeypatch, default_interval
):
    import threading

    module = _fresh_config(tmp_path, monkeypatch)
    first_in, release_first = threading.Event(), threading.Event()
    second_waits, second_in, release_second = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    real_lock = module._config_write_lock
    calls = []

    def lock(path):
        calls.append(path)
        if len(calls) == 2:
            second_waits.set()  # the second write is inside its scope
        return real_lock(path)

    monkeypatch.setattr(module, "_config_write_lock", lock)

    def hold(entered, release):
        def wait():
            entered.set()
            assert release.wait(10)

        return wait

    results = {}
    first = threading.Thread(
        target=lambda: results.update(a=_write(module, 93, hold(first_in, release_first)))
    )
    second = threading.Thread(
        target=lambda: results.update(
            b=_write(module, 94, hold(second_in, release_second))
        )
    )
    try:
        first.start()
        assert first_in.wait(10)
        second.start()
        assert second_waits.wait(10)
        release_first.set()
        first.join(10)
        assert second_in.wait(10)
        assert sys.getswitchinterval() < _SHORT_INTERVAL
        release_second.set()
        second.join(10)
    finally:
        release_first.set()
        release_second.set()
        first.join(10)
        second.join(10)
    assert results["a"].fully_applied and results["b"].fully_applied
    assert sys.getswitchinterval() == default_interval


def test_a_preparation_read_ending_inside_a_config_write_still_restores(
    tmp_path, monkeypatch, default_interval
):
    """The scopes overlap without nesting: the read began first, ends first."""
    import threading

    from tldw_chatbook.Chat.console_preparation_reads import _preparation_handoffs

    module = _fresh_config(tmp_path, monkeypatch)
    in_write, release = threading.Event(), threading.Event()

    def hold():
        in_write.set()
        assert release.wait(10)

    read = _preparation_handoffs()
    read.__enter__()
    writer = threading.Thread(target=lambda: _write(module, 95, hold))
    try:
        writer.start()
        assert in_write.wait(10)
        read.__exit__(None, None, None)
        release.set()
        writer.join(10)
    finally:
        release.set()
        writer.join(10)
    assert sys.getswitchinterval() == default_interval


def test_an_interval_other_code_sets_inside_a_config_write_is_kept(
    tmp_path, monkeypatch, default_interval
):
    module = _fresh_config(tmp_path, monkeypatch)

    assert _write(module, 96, lambda: sys.setswitchinterval(0.002)).fully_applied
    assert sys.getswitchinterval() == pytest.approx(0.002)


def test_a_write_waiting_on_another_process_lock_does_not_shorten_the_interval(
    tmp_path, monkeypatch, default_interval
):
    """Review r12s-rev1 M4: while another process holds the config's OS lock,
    the waiting write must not leave the whole process at the short interval
    (CPU-bound threads ran 28-47% slower). The write still runs short once it
    holds the lock."""
    import threading

    import portalocker

    module = _fresh_config(tmp_path, monkeypatch)
    lock_path = module._get_effective_config_path()
    lock_path = lock_path.with_name(f"{lock_path.name}.lock")
    waits: list[float] = []
    waiting = threading.Event()

    class _Event:
        def __init__(self) -> None:
            self._event = threading.Event()

        def wait(self, timeout=None):
            waits.append(sys.getswitchinterval())
            waiting.set()
            return self._event.wait(timeout)

    class _Threading:
        Event = _Event

        def __getattr__(self, name):
            return getattr(threading, name)

    seen = []
    results = {}
    with open(lock_path, "a", encoding="utf-8") as other_process:
        portalocker.lock(other_process, portalocker.LockFlags.EXCLUSIVE)
        monkeypatch.setattr(module, "_threading", _Threading())
        writer = threading.Thread(
            target=lambda: results.update(
                r=_write(module, 97, lambda: seen.append(sys.getswitchinterval()))
            )
        )
        writer.start()
        try:
            assert waiting.wait(10)
            assert waits[0] == default_interval
        finally:
            portalocker.unlock(other_process)
            writer.join(10)
    assert results["r"].fully_applied
    assert seen and seen[0] < _SHORT_INTERVAL
    assert sys.getswitchinterval() == default_interval


def test_switch_interval_releases_nest_and_reopen(default_interval):
    """Review r12s-rev2 M-b: a release inside a release, and a release after a
    scope opened inside one, raised ValueError, because the thread's open
    scopes still listed the limits the first release had closed."""
    from tldw_chatbook.Utils.switch_interval import (
        switch_interval_at_most,
        switch_interval_released,
    )

    with switch_interval_at_most(0.0001):
        with switch_interval_released():
            with switch_interval_released():
                assert sys.getswitchinterval() == default_interval
            with switch_interval_at_most(0.002):
                assert sys.getswitchinterval() == pytest.approx(0.002)
                with switch_interval_released():
                    assert sys.getswitchinterval() == default_interval
                assert sys.getswitchinterval() == pytest.approx(0.002)
            assert sys.getswitchinterval() == default_interval
        assert sys.getswitchinterval() < _SHORT_INTERVAL
    assert sys.getswitchinterval() == default_interval
