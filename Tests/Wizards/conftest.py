"""Shared fixtures for the first-run wizard suites."""

import os
from pathlib import Path

import pytest


class _SilentAudioPlayer:
    async def play(self, _path):
        return True

    async def stop(self):
        return None

    async def cleanup(self):
        return None


@pytest.fixture(autouse=True)
def _silent_audio_player(monkeypatch):
    """Voice "Test and Hear" creates the app's audio player on first use;
    keep wizard tests from spawning a real OS player. A test that needs a
    specific player sets app.audio_player or re-patches this class."""
    monkeypatch.setattr(
        "tldw_chatbook.TTS.audio_player.AsyncAudioPlayer", _SilentAudioPlayer
    )


@pytest.fixture(autouse=True)
def _restore_shared_bootstrap_config(request, isolate_test_environment):
    """Put the shared bootstrap ``config.toml`` back after a ``bootstrap_profile`` test.

    TASK-34100.1: a ``bootstrap_profile`` test keeps the collection-time
    profile, which every such test in the same worker shares. Several wizard
    tests write it for real (``apply_settings_mutation_to_cli_config``). When
    the whole wizard file was first marked, a key stored by one test made
    ``test_mounted_sparse_keyless_save_back_next_is_idempotent`` see a
    ``stored`` credential where it expects ``none``. The restore goes through
    the guarded writer, which also refreshes the config caches. A file that did
    not exist before the test is left alone: the next read recreates the
    same defaults.

    Args:
        request: The pytest request, used to read the node's markers.
        isolate_test_environment: Ordering only. The root conftest's sandbox
            sets ``TLDW_CONFIG_PATH`` to the bootstrap profile first.

    Yields:
        None, while the test runs.
    """
    del isolate_test_environment
    if request.node.get_closest_marker("bootstrap_profile") is None:
        yield
        return
    path = Path(os.environ["TLDW_CONFIG_PATH"])
    before = path.read_text(encoding="utf-8") if path.exists() else None
    yield
    if before is None or not path.exists():
        return
    if path.read_text(encoding="utf-8") == before:
        return
    from tldw_chatbook import config

    config.replace_cli_config_serialized(before, create_backup=False)
