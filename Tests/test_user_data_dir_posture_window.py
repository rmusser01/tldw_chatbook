"""``get_user_data_dir()`` looks at its path posture once a window, not once a call.

The companion of ``Tests/test_user_data_dir_memo_perf07.py`` for the Windows
observation window (TASK-40608, owner decisions O1 and O3; the helper's own
tests are ``Tests/Backup_Recovery/test_data_dir_posture_window.py``). Each test
sets the window itself, so it runs on every platform, in a private-profile
subprocess like the memo's tests.
"""

from __future__ import annotations

import pytest

from Tests.private_profile import private_profile_test


def _warm_and_count(monkeypatch: pytest.MonkeyPatch, window: float):
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import profile_paths
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    monkeypatch.setattr(profile_paths, "_DATA_DIR_HELD_SECONDS", window)
    monkeypatch.setattr(storage, "_EVIDENCE_OBSERVATION_SECONDS", window)
    first = config.get_user_data_dir()
    for _ in range(3):  # create/harden, then a bracketed resolution records it
        assert config.get_user_data_dir() == first
    looks = [0]
    real = config._user_data_dir_stamps

    def counted(paths):
        looks[0] += 1
        return real(paths)

    monkeypatch.setattr(config, "_user_data_dir_stamps", counted)
    resolutions = [0]
    resolve = config._resolve_user_data_dir

    def resolved():
        resolutions[0] += 1
        return resolve()

    monkeypatch.setattr(config, "_resolve_user_data_dir", resolved)
    return config, first, looks, resolutions


@private_profile_test
def test_warm_calls_inside_the_window_take_no_stamps(request, monkeypatch):
    config, first, looks, resolutions = _warm_and_count(monkeypatch, 30.0)
    assert config.get_user_data_dir() == first  # at most this one confirming look
    looks[0] = 0

    for _ in range(6):
        assert config.get_user_data_dir() == first

    assert looks[0] == 0
    assert resolutions[0] == 0


@private_profile_test
def test_with_no_window_every_warm_call_takes_its_stamps(request, monkeypatch):
    config, first, looks, resolutions = _warm_and_count(monkeypatch, 0.0)

    for _ in range(6):
        assert config.get_user_data_dir() == first

    assert looks[0] == 6
    assert resolutions[0] == 0


@private_profile_test
def test_a_config_reload_inside_the_window_resolves_again(request, monkeypatch):
    """The window covers the path posture only: a new generation is a new question."""
    config, first, looks, resolutions = _warm_and_count(monkeypatch, 30.0)
    assert config.get_user_data_dir() == first
    config.load_settings(force_reload=True)

    assert config.get_user_data_dir() == first

    assert resolutions[0] >= 1
