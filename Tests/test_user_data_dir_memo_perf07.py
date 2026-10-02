"""PERF-07 (TASK-33266): the user data directory is memoized, never trusted blindly.

``get_user_data_dir()`` ran the default-root lock and a private-path walk on
every call (12 ms / 351 open() calls warm; ``resolve_sensitive_context``
called it ~19 times, 237 ms). The resolution is now reused while the config
generation, the settings it reads and the posture of every path component
from ``/`` are unchanged (ADR-126 amendment, D2). These tests warm the memo,
then change something the memo must not paper over, and require the
unmodified resolution to run and behave exactly as it does uncached.

Each test runs in a private-profile subprocess: the resolution's guarded
handshake refuses an in-process HOME swap (see TASK-33370).
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from Tests.private_profile import private_profile_test

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")


def _count_resolutions(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    from tldw_chatbook import config

    calls = [0]
    real = config._resolve_user_data_dir

    def counted() -> Path:
        calls[0] += 1
        return real()

    monkeypatch.setattr(config, "_resolve_user_data_dir", counted)
    return calls


def _warm(monkeypatch: pytest.MonkeyPatch) -> tuple[Path, list[int]]:
    """Resolve until the memo serves, and return the directory."""
    from tldw_chatbook import config

    calls = _count_resolutions(monkeypatch)
    first = config.get_user_data_dir()
    for _ in range(3):  # create/harden, then a bracketed resolution records it
        assert config.get_user_data_dir() == first
    calls[0] = 0
    assert config.get_user_data_dir() == first
    assert calls[0] == 0, "a warm call re-ran the resolution"
    return first, calls


@private_profile_test
def test_a_warm_call_reuses_the_resolution(request, monkeypatch):
    from tldw_chatbook import config
    from tldw_chatbook.Utils import sensitive_paths

    user_dir, calls = _warm(monkeypatch)
    for _ in range(5):
        assert config.get_user_data_dir() == user_dir
        assert sensitive_paths.resolve_sensitive_context().user_data_dir == user_dir.resolve()
    assert calls[0] == 0


@private_profile_test
def test_a_repermissioned_leaf_is_resolved_and_hardened_again(request, monkeypatch):
    from tldw_chatbook import config

    user_dir, calls = _warm(monkeypatch)
    user_dir.chmod(0o755)

    assert config.get_user_data_dir() == user_dir
    assert calls[0] == 1
    assert stat.S_IMODE(user_dir.stat().st_mode) == 0o700


@private_profile_test
def test_a_replaced_leaf_is_resolved_again(request, monkeypatch):
    from tldw_chatbook import config

    user_dir, calls = _warm(monkeypatch)
    before = user_dir.stat().st_ino
    user_dir.rename(user_dir.with_name(user_dir.name + ".old"))
    user_dir.mkdir(mode=0o755)

    assert config.get_user_data_dir() == user_dir
    assert calls[0] == 1
    assert user_dir.stat().st_ino != before
    assert stat.S_IMODE(user_dir.stat().st_mode) == 0o700


@private_profile_test
def test_a_group_writable_ancestor_is_refused_as_before(request, monkeypatch):
    """The memo must not hide an ancestor the resolution would now refuse."""
    from tldw_chatbook import config
    from tldw_chatbook.Utils.private_paths import PrivatePathError

    user_dir, calls = _warm(monkeypatch)
    ancestor = user_dir.parents[2]  # ~/.local
    ancestor.chmod(0o777)
    try:
        with pytest.raises(PrivatePathError):
            config.get_user_data_dir()
        assert calls[0] == 1
    finally:
        ancestor.chmod(0o700)


@private_profile_test
def test_a_config_reload_resolves_again(request, monkeypatch):
    from tldw_chatbook import config

    user_dir, calls = _warm(monkeypatch)
    config.load_settings(force_reload=True)

    assert config.get_user_data_dir() == user_dir
    assert calls[0] >= 1


@private_profile_test
def test_an_environment_override_is_seen_by_the_next_sensitive_context(request, monkeypatch, tmp_path):
    """``RAG_PERSIST_DIR`` outranks config, so the memo must not hide a change
    to it while the config and data directory stay the same (Qodo, #2924)."""
    from tldw_chatbook.Utils import sensitive_paths

    _warm(monkeypatch)
    sensitive_paths.resolve_sensitive_context()
    moved = tmp_path / "moved-chroma"
    moved.mkdir()
    monkeypatch.setenv("RAG_PERSIST_DIR", str(moved))

    context = sensitive_paths.resolve_sensitive_context()
    assert moved.resolve() in context.direct_child_denied_dirs
    assert sensitive_paths.is_sensitive_path(moved / "chroma.sqlite3", context)


@private_profile_test
def test_an_override_changed_while_the_inputs_build_is_not_memoized(request, monkeypatch, tmp_path):
    """The memo key is read before the inputs are built; if the environment moves
    during the build, those inputs describe another key and are not kept. An
    override switched and restored meanwhile must not leave a deny list missing
    the restored directory (Qodo, #2924).

    Args:
        request: pytest fixture the private-profile runner needs.
        monkeypatch: Sets the override and switches it mid-build.
        tmp_path: Holds the two Chroma directories.
    """
    from tldw_chatbook.Utils import sensitive_paths

    _warm(monkeypatch)
    original = tmp_path / "chroma-original"
    switched = tmp_path / "chroma-switched"
    original.mkdir()
    switched.mkdir()
    monkeypatch.setenv("RAG_PERSIST_DIR", str(original))
    real_containers = sensitive_paths._direct_child_rule_container_dirs

    def switching_containers():
        monkeypatch.setattr(
            sensitive_paths, "_direct_child_rule_container_dirs", real_containers
        )
        os.environ["RAG_PERSIST_DIR"] = str(switched)
        return real_containers()

    monkeypatch.setattr(
        sensitive_paths, "_direct_child_rule_container_dirs", switching_containers
    )
    sensitive_paths.resolve_sensitive_context()  # built against the switched value
    os.environ["RAG_PERSIST_DIR"] = str(original)  # restored: the original key again

    context = sensitive_paths.resolve_sensitive_context()
    assert original.resolve() in context.direct_child_denied_dirs
    assert sensitive_paths.is_sensitive_path(original / "chroma.sqlite3", context)


@private_profile_test
def test_a_warm_context_reuses_the_raw_inputs_until_their_key_moves(request, monkeypatch, tmp_path):
    """The memo is real: a second context does not re-run the database-path
    accessors, and an environment change makes the next one rebuild (Qodo, #2924).

    Args:
        request: pytest fixture the private-profile runner needs.
        monkeypatch: Counts the accessor calls and changes an override.
        tmp_path: Holds the override's directory.
    """
    from tldw_chatbook.Utils import sensitive_paths

    _warm(monkeypatch)
    builds = [0]
    real_db_paths = sensitive_paths._sensitive_db_paths

    def counted_db_paths():
        builds[0] += 1
        return real_db_paths()

    monkeypatch.setattr(sensitive_paths, "_sensitive_db_paths", counted_db_paths)
    sensitive_paths.resolve_sensitive_context()
    built = builds[0]
    sensitive_paths.resolve_sensitive_context()
    assert builds[0] == built, "a warm context re-ran the database-path accessors"

    moved = tmp_path / "moved-chroma"
    moved.mkdir()
    monkeypatch.setenv("RAG_PERSIST_DIR", str(moved))
    sensitive_paths.resolve_sensitive_context()
    assert builds[0] == built + 1, "an environment change did not rebuild the inputs"


@private_profile_test
def test_a_relative_database_override_follows_the_working_directory(request, monkeypatch, tmp_path):
    """A relative custom database path resolves against the working directory,
    so a directory change must rebuild the deny list (Qodo, #2924).

    Args:
        request: pytest fixture the private-profile runner needs.
        monkeypatch: Makes the database path relative and changes directory.
        tmp_path: Holds the two working directories.
    """
    from tldw_chatbook import config
    from tldw_chatbook.Utils import sensitive_paths

    _warm(monkeypatch)
    first, second = tmp_path / "first", tmp_path / "second"
    for directory in (first, second):
        (directory / "db").mkdir(parents=True)
    monkeypatch.setattr(
        config, "get_chachanotes_db_path", lambda: Path(os.path.abspath("db/ChaChaNotes.db"))
    )
    monkeypatch.chdir(first)
    sensitive_paths.resolve_sensitive_context()
    monkeypatch.chdir(second)

    context = sensitive_paths.resolve_sensitive_context()
    assert sensitive_paths.is_sensitive_path(second / "db" / "ChaChaNotes.db", context)
