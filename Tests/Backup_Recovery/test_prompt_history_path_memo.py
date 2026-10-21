"""A Console send's prompt-history append reuses its resolved path (TASK-33620.15.5).

Measured live on origin/dev (Anthropic haiku, 160x45, sampled sends): each
send's ``PromptHistory.append`` built a file job whose source selection called
``default_prompt_history_path()``, and each call ran guarded config operations
(storage admission, companion guard, pause probe) on the UI loop, inside run
start's longest callback; in this test profile one resolution costs about
10 ms. The path is now kept while the resolution's inputs hold: the config
module, the mapping it has installed (a forced reload installs a new one
without a new generation), its identity (``current_config_identity()``:
generation and effective path) and ``HOME``, which the default data root
follows. A bound history's state check still compares its path with the live
config's mapping before any IO.
"""

from __future__ import annotations

import json
import re
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
    """Red on origin/dev: the second append ran four guarded operations for its path."""
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


def test_a_reselected_default_data_root_resolves_a_new_default_path(
    local_root, monkeypatch, tmp_path
):
    """ADR-127's default root follows which root exists on disk, not config or HOME.

    With no configured data_dir, the default root is the conventional one
    unless only the fallback root exists. Red when the kept path ignored that
    (kept on config and HOME alone): it named the removed conventional root
    while ``get_user_data_dir()`` selected the fallback, so a bound history's
    state check compared the stale path with itself instead of refusing it.
    """
    from tldw_chatbook.Backup_Recovery import profile_paths

    config = install_config_source(monkeypatch)
    if config.get_cli_setting("paths", "data_dir", None) or config.get_cli_setting(
        "Paths", "data_dir", None
    ):
        pytest.skip("a data_dir is configured; the default root is unused")
    conventional = profile_paths.default_base_data_dir()
    # This test moves the default data root: never outside its own sandbox.
    assert conventional.is_relative_to(tmp_path), conventional
    fallback = conventional.parents[2] / profile_paths.DEFAULT_DATA_FALLBACK_DIRECTORY
    before = prompt_history.default_prompt_history_path()
    assert before.is_relative_to(conventional)
    assert not fallback.exists()
    conventional.rename(conventional.with_name("tldw_cli-moved-away"))
    fallback.mkdir(mode=0o700)

    after = prompt_history.default_prompt_history_path()

    assert after == config.get_user_data_dir() / "prompt_history.jsonl"
    assert after.is_relative_to(fallback), after


def _edit_users_name(config, name: str) -> None:
    """Edit the selected config file outside the app, then force a reload.

    A forced reload (Settings, briefings and others pass ``force_reload``)
    installs the edited file as a new mapping without a new generation, so
    ``current_config_identity()`` does not move.
    """
    source = config._get_effective_config_path()
    text = source.read_text()
    if re.search(r"^users_name\s*=", text, flags=re.M):
        text = re.sub(r'^users_name\s*=\s*"[^"]*"', f'users_name = "{name}"', text, flags=re.M)
    else:
        text = text.replace("[general]", f'[general]\nusers_name = "{name}"', 1)
    source.write_text(text)
    identity = config.current_config_identity()
    config.load_cli_config_and_ensure_existence(force_reload=True)
    assert config.current_config_identity() == identity
    assert config.get_cli_setting("general", "users_name", None) == name


@pytest.mark.asyncio
async def test_a_history_bound_before_a_reload_to_another_profile_is_refused(
    local_root, monkeypatch
):
    """The bound history's path names the old profile; its append is refused.

    As on origin/dev (there the bound source no longer matches the freshly
    resolved path). Nothing is written for either profile.
    """
    config = install_config_source(monkeypatch)
    history = PromptHistory(prompt_history.default_prompt_history_path())
    assert await history.append("first prompt")
    _edit_users_name(config, "bd10rs_other")

    assert await history.append("must refuse") is False

    assert _inputs(history) == ["first prompt"]
    assert history.persistence_error == "RecoveryRequired"


@pytest.mark.asyncio
async def test_a_history_opened_after_a_reload_binds_the_live_path(
    local_root, monkeypatch
):
    """A history opened after the reload binds the live profile's path.

    As on origin/dev, and it appends there. Red when the kept path outlived
    the installed config (kept on ``current_config_identity()`` alone): the
    new history bound the old profile's path and its first append was
    refused with ``RecoveryRequired``.
    """
    config = install_config_source(monkeypatch)
    history = PromptHistory(prompt_history.default_prompt_history_path())
    assert await history.append("first prompt")
    _edit_users_name(config, "bd10rs_other")
    live = config.get_user_data_dir() / "prompt_history.jsonl"
    assert live != history.path

    reopened = PromptHistory(prompt_history.default_prompt_history_path())

    assert reopened.path == live
    assert await reopened.append("after the edit")
    assert _inputs(reopened) == ["after the edit"]
    assert _inputs(history) == ["first prompt"]
