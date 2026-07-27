import os
import shutil
import stat
import tempfile
from pathlib import Path

import pytest

import tldw_chatbook.Utils.private_paths as private_paths
from tldw_chatbook.Utils.private_paths import (
    PrivateBinaryFile,
    PrivatePathError,
    PrivatePathResult,
    PrivatePathStatus,
    _classify_private_file_stat,
    atomic_private_write_text,
    create_private_text,
    lexical_path,
    open_private_binary,
    secure_private_directory,
    verify_trusted_directory,
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
        (PrivatePathStatus.TRUSTED_DIRECTORY, False, True),
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


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")
def test_create_private_text_is_0600_under_0022_umask(tmp_path):
    target = tmp_path / "config.toml"
    previous = os.umask(0o022)
    try:
        result = create_private_text(target, "[chat]\n")
    finally:
        os.umask(previous)

    assert result.status is PrivatePathStatus.CREATED_PRIVATE
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX namespace contract")
def test_create_private_text_rejects_missing_leaf_in_shared_sticky_parent(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o1777)

    with pytest.raises(PrivatePathError) as caught:
        create_private_text(shared / "config.toml", "[chat]\n")

    assert caught.value.result.status is PrivatePathStatus.UNSAFE_PARENT
    assert not (shared / "config.toml").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX namespace contract")
def test_create_private_text_does_not_replace_existing_target(tmp_path):
    target = tmp_path / "config.toml"
    target.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        create_private_text(target, "replacement")

    assert target.read_text(encoding="utf-8") == "existing"


def test_unverified_platform_does_not_claim_private(tmp_path, monkeypatch):
    target = tmp_path / "config.toml"
    monkeypatch.setattr(
        private_paths,
        "_posix_guards_available",
        lambda: False,
    )
    monkeypatch.setattr(private_paths, "_WINDOWS_PLATFORM", True)

    result = create_private_text(target, "[chat]\n")

    assert result.status is PrivatePathStatus.UNVERIFIED_PLATFORM
    assert result.usable is True
    assert result.verified_private is False


@pytest.mark.skipif(os.name != "posix", reason="POSIX mutation contract")
def test_posix_encoding_failure_has_no_filesystem_residue(tmp_path):
    owned_directory = tmp_path / "application-config"
    target = owned_directory / "config.toml"

    with pytest.raises(UnicodeEncodeError):
        create_private_text(
            target,
            "\ud800",
            application_owned_directory=owned_directory,
        )

    assert not owned_directory.exists()
    assert not target.exists()


def test_windows_encoding_failure_has_no_filesystem_residue(
    tmp_path,
    monkeypatch,
):
    owned_directory = tmp_path / "application-config"
    target = owned_directory / "config.toml"
    monkeypatch.setattr(
        private_paths,
        "_posix_guards_available",
        lambda: False,
    )
    monkeypatch.setattr(private_paths, "_WINDOWS_PLATFORM", True)

    with pytest.raises(UnicodeEncodeError):
        create_private_text(
            target,
            "\ud800",
            application_owned_directory=owned_directory,
        )

    assert not owned_directory.exists()
    assert not target.exists()


def test_windows_atomic_write_does_not_create_custom_parent(
    tmp_path,
    monkeypatch,
):
    custom_parent = tmp_path / "custom"
    target = custom_parent / "config.toml"
    monkeypatch.setattr(
        private_paths,
        "_atomic_posix_guards_available",
        lambda: False,
    )
    monkeypatch.setattr(private_paths, "_WINDOWS_PLATFORM", True)

    with pytest.raises(PrivatePathError):
        atomic_private_write_text(target, "[chat]\n")

    assert not custom_parent.exists()
    assert not target.exists()


def test_windows_atomic_write_creates_explicit_application_owned_parent(
    tmp_path,
    monkeypatch,
):
    owned_parent = tmp_path / "application-config"
    target = owned_parent / "config.toml"
    monkeypatch.setattr(private_paths, "_posix_guards_available", lambda: False)
    monkeypatch.setattr(
        private_paths,
        "_atomic_posix_guards_available",
        lambda: False,
    )
    monkeypatch.setattr(private_paths, "_WINDOWS_PLATFORM", True)

    result = atomic_private_write_text(
        target,
        "[chat]\n",
        application_owned_directory=owned_parent,
    )

    assert result.status is PrivatePathStatus.UNVERIFIED_PLATFORM
    assert target.read_text(encoding="utf-8") == "[chat]\n"


def test_unsupported_posix_guards_fail_closed_without_creating(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "config.toml"
    monkeypatch.setattr(
        private_paths,
        "_posix_guards_available",
        lambda: False,
    )
    monkeypatch.setattr(private_paths, "_WINDOWS_PLATFORM", False)

    with pytest.raises(PrivatePathError) as caught:
        create_private_text(target, "[chat]\n")

    assert caught.value.result.status is PrivatePathStatus.OPERATION_FAILED
    assert caught.value.result.reason == "required_posix_guards_unavailable"
    assert not target.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX capability contract")
def test_missing_unlink_dir_fd_capability_does_not_block_creation(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "config.toml"
    restricted = frozenset(
        capability
        for capability in private_paths.os.supports_dir_fd
        if capability is not private_paths.os.unlink
    )
    monkeypatch.setattr(private_paths.os, "supports_dir_fd", restricted)
    monkeypatch.setattr(private_paths, "_WINDOWS_PLATFORM", False)

    assert private_paths._posix_guards_available() is True
    result = create_private_text(target, "[chat]\n")

    assert result.status is PrivatePathStatus.CREATED_PRIVATE
    assert target.read_text(encoding="utf-8") == "[chat]\n"


@pytest.mark.skipif(os.name != "posix", reason="POSIX postcondition contract")
def test_create_private_text_retains_private_entry_on_failed_postcondition(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "config.toml"
    monkeypatch.setattr(
        private_paths,
        "_private_file_postcondition_holds",
        lambda *args, **kwargs: False,
    )

    with pytest.raises(PrivatePathError) as caught:
        create_private_text(target, "[chat]\n")

    assert caught.value.result.status is PrivatePathStatus.OPERATION_FAILED
    assert caught.value.result.reason == "private_file_postcondition_failed"
    assert target.read_text(encoding="utf-8") == "[chat]\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX rollback race contract")
def test_create_private_text_never_unlinks_name_after_postcondition_failure(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "config.toml"
    unlink_calls = []

    def record_unlink(*args, **kwargs):
        unlink_calls.append((args, kwargs))

    monkeypatch.setattr(
        private_paths,
        "_private_file_postcondition_holds",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(private_paths.os, "unlink", record_unlink)
    monkeypatch.setattr(
        private_paths,
        "_posix_guards_available",
        lambda: True,
    )

    with pytest.raises(PrivatePathError) as caught:
        create_private_text(target, "[chat]\n")

    assert caught.value.result.status is PrivatePathStatus.OPERATION_FAILED
    assert caught.value.result.reason == "private_file_postcondition_failed"
    assert unlink_calls == []
    assert target.read_text(encoding="utf-8") == "[chat]\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX write contract")
@pytest.mark.timeout(2, method="signal")
def test_create_private_text_zero_byte_write_fails_without_spinning(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "config.toml"
    monkeypatch.setattr(private_paths.os, "write", lambda *args, **kwargs: 0)

    with pytest.raises(PrivatePathError) as caught:
        create_private_text(target, "[chat]\n")

    assert caught.value.result.status is PrivatePathStatus.OPERATION_FAILED
    assert caught.value.result.reason == "zero_byte_write"
    assert target.exists()
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX race contract")
def test_create_private_text_never_follows_raced_final_symlink(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "config.toml"
    outside = tmp_path / "outside.toml"
    outside.write_text("sentinel", encoding="utf-8")
    real_open = private_paths._open_leaf_for_create

    def raced_open(parent_fd, leaf):
        target.symlink_to(outside)
        return real_open(parent_fd, leaf)

    monkeypatch.setattr(private_paths, "_open_leaf_for_create", raced_open)

    with pytest.raises((FileExistsError, PrivatePathError)):
        create_private_text(target, "private")

    assert outside.read_text(encoding="utf-8") == "sentinel"


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


@pytest.mark.skipif(os.name != "posix", reason="POSIX capability contract")
@pytest.mark.parametrize("missing_capability", ["_NONBLOCK", "_NOCTTY"])
def test_open_private_binary_fails_before_traversal_when_leaf_guard_is_unavailable(
    tmp_path,
    monkeypatch,
    missing_capability,
):
    target = tmp_path / "config.toml"
    content = b"private"
    target.write_bytes(content)
    target.chmod(0o644)
    monkeypatch.setattr(
        private_paths,
        missing_capability,
        0,
        raising=False,
    )

    def fail_if_traversed(*args, **kwargs):
        pytest.fail("target traversal occurred without required leaf guards")

    monkeypatch.setattr(
        private_paths,
        "_open_verified_parent",
        fail_if_traversed,
    )

    with pytest.raises(PrivatePathError) as caught:
        with open_private_binary(target):
            pass

    assert caught.value.result.status is PrivatePathStatus.OPERATION_FAILED
    assert caught.value.result.reason == "required_posix_guards_unavailable"
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
    assert target.read_bytes() == content


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory contract")
def test_verify_trusted_directory_accepts_owned_non_writable_without_mutation(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "custom-databases"
    target.mkdir(mode=0o750)
    before = stat.S_IMODE(target.stat().st_mode)

    def fail_mutation(*args, **kwargs):
        pytest.fail("trusted-directory verification must not mutate the directory")

    monkeypatch.setattr(private_paths.os, "fchmod", fail_mutation)
    monkeypatch.setattr(private_paths.os, "mkdir", fail_mutation)
    monkeypatch.setattr(private_paths, "_posix_guards_available", lambda: True)

    result = verify_trusted_directory(target, allow_shared_sticky=False)

    assert result.status is PrivatePathStatus.TRUSTED_DIRECTORY
    assert result.usable is True
    assert result.verified_private is False
    assert stat.S_IMODE(target.stat().st_mode) == before


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory contract")
def test_verify_trusted_directory_allows_shared_sticky_only_when_requested(tmp_path):
    target = tmp_path / "shared"
    target.mkdir()
    target.chmod(0o1777)

    result = verify_trusted_directory(target, allow_shared_sticky=True)

    assert result.status is PrivatePathStatus.TRUSTED_DIRECTORY
    with pytest.raises(PrivatePathError) as caught:
        verify_trusted_directory(target, allow_shared_sticky=False)
    assert caught.value.result.status is PrivatePathStatus.UNSAFE_PARENT


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory contract")
@pytest.mark.parametrize("final_component", [False, True])
def test_verify_trusted_directory_rejects_directory_symlinks(
    tmp_path,
    final_component,
):
    real = tmp_path / "real"
    child = real / "child"
    child.mkdir(parents=True)
    alias = tmp_path / "alias"
    if final_component:
        alias.symlink_to(child, target_is_directory=True)
        selected = alias
    else:
        alias.symlink_to(real, target_is_directory=True)
        selected = alias / "child"

    with pytest.raises(PrivatePathError) as caught:
        verify_trusted_directory(selected, allow_shared_sticky=False)

    assert caught.value.result.status is PrivatePathStatus.LINK_OR_NON_REGULAR


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory contract")
def test_verify_trusted_directory_rejects_wrong_owner_simulation(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "custom-databases"
    target.mkdir()
    actual_euid = os.geteuid()
    monkeypatch.setattr(private_paths.os, "geteuid", lambda: actual_euid + 1000)

    with pytest.raises(PrivatePathError) as caught:
        verify_trusted_directory(target, allow_shared_sticky=False)

    assert caught.value.result.status in {
        PrivatePathStatus.UNSAFE_PARENT,
        PrivatePathStatus.WRONG_OWNER,
    }


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory contract")
@pytest.mark.parametrize("mode", [0o770, 0o707, 0o777])
def test_verify_trusted_directory_rejects_non_sticky_shared_writable(
    tmp_path,
    mode,
):
    target = tmp_path / "custom-databases"
    target.mkdir()
    target.chmod(mode)

    with pytest.raises(PrivatePathError) as caught:
        verify_trusted_directory(target, allow_shared_sticky=True)

    assert caught.value.result.status is PrivatePathStatus.UNSAFE_PARENT


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory contract")
def test_verify_trusted_directory_rejects_missing_directory_without_creation(tmp_path):
    target = tmp_path / "missing" / "databases"

    with pytest.raises(PrivatePathError):
        verify_trusted_directory(target, allow_shared_sticky=False)

    assert not target.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor contract")
def test_verify_trusted_directory_closes_every_component_descriptor(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "custom" / "databases"
    target.mkdir(parents=True)
    opened_components: set[int] = set()
    real_open_component = private_paths._open_directory_component

    def track_open(parent_fd, component):
        opened = real_open_component(parent_fd, component)
        opened_components.add(opened)
        return opened

    monkeypatch.setattr(private_paths, "_open_directory_component", track_open)

    verify_trusted_directory(target, allow_shared_sticky=False)

    for opened_fd in opened_components:
        with pytest.raises(OSError):
            os.fstat(opened_fd)


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor contract")
def test_verify_trusted_directory_closes_descriptors_on_error(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "custom" / "missing"
    target.parent.mkdir()
    opened_components: set[int] = set()
    real_open_component = private_paths._open_directory_component

    def track_open(parent_fd, component):
        opened = real_open_component(parent_fd, component)
        opened_components.add(opened)
        return opened

    monkeypatch.setattr(private_paths, "_open_directory_component", track_open)

    with pytest.raises(PrivatePathError):
        verify_trusted_directory(target, allow_shared_sticky=False)

    for opened_fd in opened_components:
        with pytest.raises(OSError):
            os.fstat(opened_fd)


@pytest.mark.skipif(os.name != "posix", reason="POSIX identity contract")
def test_verify_trusted_directory_rejects_entry_replacement_during_walk(
    tmp_path,
    monkeypatch,
):
    selected = tmp_path / "selected"
    selected.mkdir()
    displaced = tmp_path / "selected-displaced"
    real_open_component = private_paths._open_directory_component

    def replace_after_open(parent_fd, component):
        opened = real_open_component(parent_fd, component)
        if component == selected.name:
            selected.rename(displaced)
            selected.mkdir()
        return opened

    monkeypatch.setattr(
        private_paths,
        "_open_directory_component",
        replace_after_open,
    )

    with pytest.raises(PrivatePathError) as caught:
        verify_trusted_directory(selected, allow_shared_sticky=False)

    assert caught.value.result.status is PrivatePathStatus.OPERATION_FAILED
    assert caught.value.result.reason == "trusted_directory_postcondition_failed"


@pytest.mark.skipif(os.name != "posix", reason="POSIX identity contract")
def test_verify_trusted_directory_rejects_forced_final_postcondition_failure(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "custom-databases"
    target.mkdir()
    monkeypatch.setattr(
        private_paths,
        "_trusted_directory_postcondition_holds",
        lambda *args, **kwargs: False,
    )

    with pytest.raises(PrivatePathError) as caught:
        verify_trusted_directory(target, allow_shared_sticky=False)

    assert caught.value.result.status is PrivatePathStatus.OPERATION_FAILED
    assert caught.value.result.reason == "trusted_directory_postcondition_failed"


def test_verify_trusted_directory_reports_windows_as_unverified_without_mutation(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "custom-databases"
    target.mkdir(mode=0o755)
    before = stat.S_IMODE(target.stat().st_mode)
    monkeypatch.setattr(private_paths, "_posix_guards_available", lambda: False)
    monkeypatch.setattr(private_paths, "_WINDOWS_PLATFORM", True)

    result = verify_trusted_directory(target, allow_shared_sticky=False)

    assert result.status is PrivatePathStatus.UNVERIFIED_PLATFORM
    assert result.verified_private is False
    assert stat.S_IMODE(target.stat().st_mode) == before


# --- Trusted system symlink traversal (TASK-950) -----------------------------
#
# macOS ships `/var -> private/var` and `/tmp -> private/tmp`, and the platform
# temporary directory lives under `/var/folders`. The walk must be able to cross
# a *root-owned* symlink without ever crossing one an unprivileged user could
# have planted or repointed.


def _symlinked_components(path: Path) -> list[Path]:
    return [
        candidate
        for candidate in [path, *path.parents]
        if candidate.is_symlink()
    ]


def _relabelled_stat(
    original: os.stat_result,
    *,
    uid: int | None = None,
    mode: int | None = None,
) -> os.stat_result:
    fields = list(original)
    if mode is not None:
        fields[0] = mode
    if uid is not None:
        fields[4] = uid
    return os.stat_result(fields)


def _relabel_symlink_identity(
    monkeypatch,
    components: set[str],
    *,
    uid: int,
    mode: int,
) -> None:
    """Make named symlink components look like a differently-owned link.

    Only root can create a root-owned symlink, so the positive and negative
    ownership cases are exercised by relabelling the link's own ``lstat``.
    """

    real_stat = private_paths.os.stat

    def relabelled_stat(path, *, dir_fd=None, follow_symlinks=True):
        result = real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)
        if (
            follow_symlinks is False
            and path in components
            and stat.S_ISLNK(result.st_mode)
        ):
            return _relabelled_stat(result, uid=uid, mode=mode)
        return result

    monkeypatch.setattr(private_paths.os, "stat", relabelled_stat)
    # The replacement is not the real os.stat, so the dir_fd capability probe
    # would otherwise report the POSIX guards as unavailable.
    monkeypatch.setattr(private_paths, "_posix_guards_available", lambda: True)


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory contract")
def test_verify_trusted_directory_traverses_the_platform_temporary_directory():
    """The macOS `/var -> private/var` case, asserted against the real system."""

    temporary_root = tempfile.gettempdir()
    if not _symlinked_components(Path(temporary_root)):
        pytest.skip("platform temporary directory has no symlinked component")

    result = verify_trusted_directory(temporary_root, allow_shared_sticky=True)

    assert result.status is PrivatePathStatus.TRUSTED_DIRECTORY
    assert result.lexical_path == Path(temporary_root)


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory contract")
def test_secure_private_directory_creates_under_the_platform_temporary_directory():
    temporary_root = Path(tempfile.gettempdir())
    if not _symlinked_components(temporary_root):
        pytest.skip("platform temporary directory has no symlinked component")
    scratch = Path(tempfile.mkdtemp(prefix="tldw-950-"))
    try:
        target = scratch / "databases"

        result = secure_private_directory(
            target,
            create=True,
            application_owned=True,
        )

        assert result.verified_private is True
        assert stat.S_IMODE(target.stat().st_mode) == 0o700
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory contract")
def test_verify_trusted_directory_traverses_a_root_owned_symlink(
    tmp_path,
    monkeypatch,
):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    _relabel_symlink_identity(
        monkeypatch,
        {"alias"},
        uid=0,
        mode=stat.S_IFLNK | 0o755,
    )

    result = verify_trusted_directory(alias, allow_shared_sticky=False)

    assert result.status is PrivatePathStatus.TRUSTED_DIRECTORY
    assert result.lexical_path == alias


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory contract")
def test_verify_trusted_directory_traverses_a_relative_root_owned_symlink(
    tmp_path,
    monkeypatch,
):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to("real", target_is_directory=True)
    _relabel_symlink_identity(
        monkeypatch,
        {"alias"},
        uid=0,
        mode=stat.S_IFLNK | 0o755,
    )

    result = verify_trusted_directory(alias, allow_shared_sticky=False)

    assert result.status is PrivatePathStatus.TRUSTED_DIRECTORY


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory contract")
@pytest.mark.parametrize(
    ("uid_offset", "mode"),
    [
        (1000, stat.S_IFLNK | 0o755),
        (0, stat.S_IFLNK | 0o775),
        (0, stat.S_IFLNK | 0o757),
    ],
    ids=["foreign-owner", "group-writable", "world-writable"],
)
def test_verify_trusted_directory_rejects_an_untrusted_symlink(
    tmp_path,
    monkeypatch,
    uid_offset,
    mode,
):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    foreign_uid = os.geteuid() + 1000 if uid_offset else 0
    _relabel_symlink_identity(
        monkeypatch,
        {"alias"},
        uid=foreign_uid,
        mode=mode,
    )

    with pytest.raises(PrivatePathError) as caught:
        verify_trusted_directory(alias, allow_shared_sticky=False)

    assert caught.value.result.status is PrivatePathStatus.LINK_OR_NON_REGULAR


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory contract")
@pytest.mark.timeout(20, method="signal")
def test_verify_trusted_directory_stops_on_a_symlink_loop(tmp_path, monkeypatch):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.symlink_to(second, target_is_directory=True)
    second.symlink_to(first, target_is_directory=True)
    _relabel_symlink_identity(
        monkeypatch,
        {"first", "second"},
        uid=0,
        mode=stat.S_IFLNK | 0o755,
    )

    with pytest.raises(PrivatePathError) as caught:
        verify_trusted_directory(first, allow_shared_sticky=False)

    assert caught.value.result.status is PrivatePathStatus.LINK_OR_NON_REGULAR
    assert caught.value.result.reason == "symlink_hop_limit_exceeded"


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor contract")
def test_open_private_binary_traverses_a_root_owned_intermediate_symlink(
    tmp_path,
    monkeypatch,
):
    real = tmp_path / "real"
    real.mkdir()
    payload = b"[chat]\n"
    target = real / "config.toml"
    target.write_bytes(payload)
    target.chmod(0o600)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    _relabel_symlink_identity(
        monkeypatch,
        {"alias"},
        uid=0,
        mode=stat.S_IFLNK | 0o755,
    )

    with open_private_binary(alias / "config.toml") as opened:
        assert opened.stream.read() == payload
