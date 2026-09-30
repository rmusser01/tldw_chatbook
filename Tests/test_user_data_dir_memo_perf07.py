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
