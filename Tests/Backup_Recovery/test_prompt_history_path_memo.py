"""A Console send's prompt-history append reuses its resolved path (TASK-33620.15.5).

Measured live on origin/dev (Anthropic haiku, 160x45, 30 sampled sends): each
send's ``PromptHistory.append`` built a file job whose source selection called
``default_prompt_history_path()`` twice, and each call ran a guarded config
operation (storage admission, companion guard, pause probe) on the UI loop,
about 12 ms per send. The path is now kept while the resolution's inputs hold:
the installed config module, its identity (``current_config_identity()``:
generation and effective path) and ``HOME``, which the default data root
follows. The file job's worker scope still re-derives the selection from the
live config before any IO, so a stale path can only be refused there, never
written to.
"""

from __future__ import annotations

import json
import sys

import pytest

from Tests.Backup_Recovery.config_test_support import install_config_source
from Tests.Backup_Recovery.test_participant_lifetimes import local_root as local_root  # noqa: PLC0414 - exported pytest fixture
from tldw_chatbook.Backup_Recovery import config_participants
from tldw_chatbook.Chat import prompt_history
from tldw_chatbook.Chat.prompt_history import PromptHistory


class PathOperations:
    """Count guarded config operations entered to resolve the history path."""

    def __init__(self, monkeypatch) -> None:
        self.count = 0
        real = config_participants.operation

        def counted(*args, **kwargs):
            frame = sys._getframe(1)
            while frame is not None:
                if frame.f_code.co_name == "default_prompt_history_path":
                    self.count += 1
                    break
                frame = frame.f_back
            return real(*args, **kwargs)

        monkeypatch.setattr(config_participants, "operation", counted)


def _inputs(history: PromptHistory) -> list[str]:
    return [json.loads(line)["input"] for line in history.path.read_text().splitlines()]


@pytest.mark.asyncio
async def test_a_warm_append_runs_no_config_operation_for_its_path(
    local_root, monkeypatch
):
    """Red on origin/dev: the second append resolved the path twice more."""
    install_config_source(monkeypatch)
    history = PromptHistory(prompt_history.default_prompt_history_path())
    assert await history.append("first prompt")
    operations = PathOperations(monkeypatch)

    assert await history.append("second prompt")

    assert operations.count == 0, (
        f"the append re-resolved the history path {operations.count} times "
        "through a guarded config operation"
    )
    assert _inputs(history) == ["first prompt", "second prompt"]


@pytest.mark.asyncio
async def test_a_config_write_makes_the_next_append_resolve_its_path_again(
    local_root, monkeypatch
):
    config = install_config_source(monkeypatch)
    history = PromptHistory(prompt_history.default_prompt_history_path())
    assert await history.append("first prompt")
    identity = config.current_config_identity()
    assert config.save_setting_to_cli_config("general", "bd10rs_marker", "on")
    assert config.current_config_identity() != identity
    operations = PathOperations(monkeypatch)

    assert await history.append("second prompt")

    assert operations.count >= 1, "a config write did not invalidate the kept path"
    assert _inputs(history) == ["first prompt", "second prompt"]


def test_a_new_home_resolves_a_new_default_path(local_root, monkeypatch, tmp_path):
    """The default data root follows HOME, which the config identity does not see."""
    config = install_config_source(monkeypatch)
    if config.get_cli_setting("paths", "data_dir", None) or config.get_cli_setting(
        "Paths", "data_dir", None
    ):
        pytest.skip("the test profile configures a data_dir; HOME is not an input")
    before = prompt_history.default_prompt_history_path()
    home = tmp_path / "other-home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HOME", str(home))

    after = prompt_history.default_prompt_history_path()

    assert after != before
    assert after == config.get_user_data_dir() / "prompt_history.jsonl"


@pytest.mark.asyncio
async def test_a_profile_change_the_identity_misses_is_still_refused(
    local_root, monkeypatch
):
    """An external edit installs a new config without moving the identity.

    The kept path then names the old profile; the worker's scope compares it
    with the live config's selection and refuses, as origin/dev refuses (there
    the bound source no longer matches the freshly resolved path). Nothing is
    written for either profile.
    """
    config = install_config_source(monkeypatch)
    history = PromptHistory(prompt_history.default_prompt_history_path())
    assert await history.append("first prompt")
    identity = config.current_config_identity()
    monkeypatch.setattr(
        config,
        "_CONFIG_CACHE",
        {**config._CONFIG_CACHE, "general": {"users_name": "bd10rs_other"}},
    )
    assert config.current_config_identity() == identity

    assert await history.append("must refuse") is False

    assert _inputs(history) == ["first prompt"]
    assert history.persistence_error == "RecoveryRequired"
