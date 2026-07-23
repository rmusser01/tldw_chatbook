import os
import stat
from pathlib import Path

import pytest

import tldw_chatbook.Utils.private_paths as private_paths
from tldw_chatbook.Utils.private_paths import (
    PrivateBinaryFile,
    PrivatePathError,
    PrivatePathResult,
    PrivatePathStatus,
    _classify_private_file_stat,
    lexical_path,
    open_private_binary,
    secure_private_directory,
)


@pytest.mark.parametrize(
    ("status", "verified_private", "usable"),
    [
        (PrivatePathStatus.CREATED_PRIVATE, True, True),
        (PrivatePathStatus.HARDENED_PRIVATE, True, True),
        (PrivatePathStatus.ALREADY_PRIVATE, True, True),
        (PrivatePathStatus.UNVERIFIED_PLATFORM, False, True),
        (PrivatePathStatus.UNSAFE_PARENT, False, False),
        (PrivatePathStatus.WRONG_OWNER, False, False),
        (PrivatePathStatus.LINK_OR_NON_REGULAR, False, False),
        (PrivatePathStatus.OPERATION_FAILED, False, False),
    ],
)
def test_private_path_result_classifies_posture(status, verified_private, usable):
    result = PrivatePathResult(Path("/tmp/config.toml"), status)

    assert result.verified_private is verified_private
    assert result.usable is usable


def test_private_path_error_exposes_bounded_result_without_original_exception():
    result = PrivatePathResult(
        Path("/tmp/config.toml"),
        PrivatePathStatus.UNSAFE_PARENT,
        reason="shared_writable_parent",
    )

    error = PrivatePathError(result)

    assert error.result is result
    assert "shared_writable_parent" in str(error)


def test_lexical_path_normalizes_without_resolving_symlink(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    monkeypatch.chdir(tmp_path)

    selected = lexical_path(Path("alias") / ".." / "alias" / "config.toml")

    assert selected == alias / "config.toml"
    assert selected != real / "config.toml"


def test_lexical_path_rejects_nul():
    with pytest.raises(ValueError, match="NUL"):
        lexical_path("bad\x00path")


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor contract")
def test_open_private_binary_hardens_before_read(tmp_path):
    target = tmp_path / "config.toml"
    target.write_bytes(b"[chat]\nstreaming = true\n")
    target.chmod(0o644)

    with open_private_binary(target) as opened:
        assert isinstance(opened, PrivateBinaryFile)
        assert opened.stream.read().startswith(b"[chat]")
        assert opened.result.status is PrivatePathStatus.HARDENED_PRIVATE
        assert stat.S_IMODE(os.fstat(opened.stream.fileno()).st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor contract")
def test_open_private_binary_rejects_final_symlink(tmp_path):
    outside = tmp_path / "outside.toml"
    outside.write_text("secret = true\n", encoding="utf-8")
    alias = tmp_path / "config.toml"
    alias.symlink_to(outside)

    with pytest.raises(PrivatePathError) as caught:
        with open_private_binary(alias):
            pass

    assert caught.value.result.status is PrivatePathStatus.LINK_OR_NON_REGULAR
    assert outside.stat().st_mode & 0o777 != 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor contract")
def test_open_private_binary_rejects_intermediate_symlink(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "config.toml").write_text("[chat]\n", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    with pytest.raises(PrivatePathError) as caught:
        with open_private_binary(alias / "config.toml"):
            pass

    assert caught.value.result.status is PrivatePathStatus.LINK_OR_NON_REGULAR


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor contract")
def test_open_private_binary_rejects_non_regular_leaf(tmp_path):
    target = tmp_path / "config.toml"
    target.mkdir()

    with pytest.raises(PrivatePathError) as caught:
        with open_private_binary(target):
            pass

    assert caught.value.result.status is PrivatePathStatus.LINK_OR_NON_REGULAR


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor contract")
@pytest.mark.timeout(2, method="signal")
def test_open_private_binary_rejects_fifo_without_blocking(tmp_path):
    target = tmp_path / "config.toml"
    os.mkfifo(target, mode=0o644)

    with pytest.raises(PrivatePathError) as caught:
        with open_private_binary(target):
            pass

    assert caught.value.result.status is PrivatePathStatus.LINK_OR_NON_REGULAR


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor contract")
def test_open_private_binary_rejects_multiply_linked_file_without_changing_alias(
    tmp_path,
):
    target = tmp_path / "config.toml"
    alias = tmp_path / "shared-alias.toml"
    content = b"shared private data"
    target.write_bytes(content)
    target.chmod(0o644)
    os.link(target, alias)

    with pytest.raises(PrivatePathError) as caught:
        with open_private_binary(target) as opened:
            opened.stream.read()

    assert caught.value.result.status is PrivatePathStatus.LINK_OR_NON_REGULAR
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
    assert stat.S_IMODE(alias.stat().st_mode) == 0o644
    assert target.read_bytes() == content
    assert alias.read_bytes() == content


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor contract")
def test_open_private_binary_reports_hardening_failure(tmp_path, monkeypatch):
    target = tmp_path / "config.toml"
    target.write_bytes(b"config")
    target.chmod(0o644)

    def fail_fchmod(file_fd, mode):
        raise OSError("simulated")

    monkeypatch.setattr(private_paths.os, "fchmod", fail_fchmod)

    with pytest.raises(PrivatePathError) as caught:
        with open_private_binary(target):
            pass

    assert caught.value.result.status is PrivatePathStatus.OPERATION_FAILED
    assert caught.value.result.reason == "OSError"


@pytest.mark.skipif(os.name != "posix", reason="POSIX postcondition contract")
def test_open_private_binary_fails_when_postcondition_is_not_verified(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "config.toml"
    target.write_bytes(b"config")
    target.chmod(0o600)
    monkeypatch.setattr(
        private_paths,
        "_private_file_postcondition_holds",
        lambda *args, **kwargs: False,
    )

    with pytest.raises(PrivatePathError) as caught:
        with open_private_binary(target):
            pass

    assert caught.value.result.status is PrivatePathStatus.OPERATION_FAILED
    assert caught.value.result.reason == "private_file_postcondition_failed"


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor contract")
def test_open_private_binary_keeps_opened_identity_when_name_is_replaced(tmp_path):
    target = tmp_path / "config.toml"
    replacement = tmp_path / "replacement.toml"
    target.write_bytes(b"trusted")
    replacement.write_bytes(b"replacement")

    with open_private_binary(target) as opened:
        replacement.replace(target)
        assert opened.stream.read() == b"trusted"


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor contract")
def test_open_private_binary_stays_on_pinned_parent_during_path_replacement(
    tmp_path,
    monkeypatch,
):
    selected_parent = tmp_path / "selected"
    child = selected_parent / "child"
    child.mkdir(parents=True)
    (child / "config.toml").write_bytes(b"trusted")
    displaced_parent = tmp_path / "selected-displaced"
    real_open_component = private_paths._open_directory_component
    raced = False

    def race_after_parent_is_pinned(parent_fd, component):
        nonlocal raced
        if component == "child" and not raced:
            raced = True
            selected_parent.rename(displaced_parent)
            replacement_child = selected_parent / "child"
            replacement_child.mkdir(parents=True)
            (replacement_child / "config.toml").write_bytes(b"replacement")
        return real_open_component(parent_fd, component)

    monkeypatch.setattr(
        private_paths,
        "_open_directory_component",
        race_after_parent_is_pinned,
    )

    with open_private_binary(selected_parent / "child" / "config.toml") as opened:
        assert opened.stream.read() == b"trusted"


def test_stat_classification_rejects_wrong_owner():
    fake = type(
        "FakeStat",
        (),
        {"st_mode": stat.S_IFREG | 0o600, "st_nlink": 1, "st_uid": 2222},
    )()

    assert (
        _classify_private_file_stat(fake, expected_uid=1111)
        is PrivatePathStatus.WRONG_OWNER
    )


@pytest.mark.skipif(os.name != "posix", reason="POSIX postcondition contract")
def test_private_directory_never_reports_success_after_failed_postcondition(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "application-config"
    monkeypatch.setattr(
        private_paths,
        "_private_directory_postcondition_holds",
        lambda *args, **kwargs: False,
    )

    with pytest.raises(PrivatePathError) as caught:
        secure_private_directory(
            target,
            create=True,
            application_owned=True,
        )

    assert caught.value.result.status is PrivatePathStatus.OPERATION_FAILED
    assert caught.value.result.reason == "private_directory_postcondition_failed"


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor contract")
def test_private_directory_closes_component_fd_when_entry_stat_disappears(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "application-config"
    target.mkdir()
    opened_components = set()
    real_open_component = private_paths._open_directory_component
    real_stat = private_paths.os.stat

    def track_open(parent_fd, component):
        opened = real_open_component(parent_fd, component)
        opened_components.add(opened)
        return opened

    def fail_nofollow_entry_stat(
        path,
        *,
        dir_fd=None,
        follow_symlinks=True,
    ):
        if dir_fd is not None and follow_symlinks is False:
            raise FileNotFoundError("simulated entry replacement")
        return real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(
        private_paths,
        "_open_directory_component",
        track_open,
    )
    monkeypatch.setattr(private_paths.os, "stat", fail_nofollow_entry_stat)
    monkeypatch.setattr(
        private_paths,
        "_posix_guards_available",
        lambda: True,
    )

    with pytest.raises(PrivatePathError) as caught:
        secure_private_directory(
            target,
            create=False,
            application_owned=True,
        )

    assert caught.value.result.status is PrivatePathStatus.OPERATION_FAILED
    assert caught.value.result.reason == "FileNotFoundError"
    for opened_fd in opened_components:
        with pytest.raises(OSError):
            os.fstat(opened_fd)


def test_private_directory_rejects_filesystem_root():
    with pytest.raises(ValueError, match="filesystem root"):
        secure_private_directory(
            Path(os.path.abspath(os.sep)),
            create=True,
            application_owned=True,
        )


def test_open_private_binary_fails_closed_when_posix_guards_are_unavailable(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "config.toml"
    target.write_bytes(b"private")
    target.chmod(0o644)
    monkeypatch.setattr(
        private_paths,
        "_posix_guards_available",
        lambda: False,
    )
    monkeypatch.setattr(private_paths, "_WINDOWS_PLATFORM", False)

    with pytest.raises(PrivatePathError) as caught:
        with open_private_binary(target):
            pass

    assert caught.value.result.status is PrivatePathStatus.OPERATION_FAILED
    assert caught.value.result.reason == "required_posix_guards_unavailable"
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
