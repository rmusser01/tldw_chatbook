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
    ``stored`` credential where it expects ``none``. ``put_back_bootstrap_config``
    does the restore: it removes a file the test created and writes back one
    it changed or deleted, refreshing the config caches either way.

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
    put_back_bootstrap_config(path, before)


def put_back_bootstrap_config(path: Path, before: str | None) -> None:
    """Return the config file at ``path`` to its state from before a test.

    Qodo (PR #3001): a file the test created used to be left in place, keys
    and all, for the next test in the worker. It is now removed and the
    config caches are republished from the absent file. A file the test
    changed or deleted is written back through the guarded writer, which
    refreshes the caches too.

    Args:
        path: The bootstrap ``config.toml`` (``TLDW_CONFIG_PATH``).
        before: Its text before the test, or None when it did not exist.

    Raises:
        RuntimeError: The caches could not be republished after removing a
            file the test created.
    """
    from tldw_chatbook import config

    if before is None:
        if not path.exists():
            return
        path.unlink()
        refreshed = config.refresh_runtime_config_from_cli_config()
        if not refreshed.caches_reloaded:
            raise RuntimeError(
                f"config caches not republished ({refreshed.failure_phase})"
            )
        return
    if path.exists() and path.read_text(encoding="utf-8") == before:
        return
    config.replace_cli_config_serialized(before, create_backup=False)
