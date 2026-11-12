"""The one-time Full agent tools upgrade (TASK-34100.5.3.2).

Owner ruling 2026-10-29, "Keep them on if used": a profile that existed
before the light surface (TASK-34100.5.3) and already uses agent features
gets ``[tools] full_agent_tools_enabled = true`` written once; a fresh or
unused profile keeps the light set with nothing written to its config.

Everything here is real: the workspaces and runs databases are built by the
real ``WorkspaceDB``/``AgentRunsDB`` in ``tmp_path``, the config is a real
``config.toml`` selected through the same source seam the config suites use,
and the outcome is read back from the file bytes.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger

from Tests.Backup_Recovery.config_test_support import select_config_source
from tldw_chatbook.Agents.agent_models import AgentDefinition
from tldw_chatbook.Agents.full_agent_tools_upgrade import (
    FULL_AGENT_TOOLS_UPGRADE_MARKER,
    SIGNAL_AGENT_DEFINITION,
    SIGNAL_SUBAGENT_RUN,
    SIGNAL_TOOL_CALL,
    SIGNAL_WORKSPACE_BINDING,
    UpgradeOutcome,
    _write_full_agent_tools_on,
    run_for_app,
    run_full_agent_tools_upgrade,
)
from tldw_chatbook.Agents.plain_chat_tool_surface import light_surface_configured
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces.models import (
    RuntimeBindingKind,
    RuntimeBindingStatus,
    WorkspaceRuntimeBinding,
)
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

pytestmark = pytest.mark.unit

#: A pre-upgrade profile's config: hand comments, a tool gate the user
#: switched on, and no ``full_agent_tools_enabled``. Any rewrite drops the
#: comments, so "nothing written" is checked on these exact bytes.
PRE_UPGRADE_CONFIG = """\
# My notes; a rewrite of this file would drop them.
[general]
default_tab = "chat"  # inline comment

[first_run]
setup_started = true
setup_completed = true

[tools]
read_file_enabled = true
"""


@pytest.fixture
def profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A real pre-upgrade profile: config.toml plus the two databases."""
    config_path = tmp_path / "config" / "config.toml"
    config_path.parent.mkdir(mode=0o700)
    config_path.write_text(PRE_UPGRADE_CONFIG, encoding="utf-8")
    config_path.chmod(0o600)
    select_config_source(monkeypatch, str(config_path), globals())
    data = tmp_path / "data"
    data.mkdir(mode=0o700)
    workspace_db = WorkspaceDB(data / "workspaces.db", client_id="upgrade-test")
    runs_path = data / "agent_runs.db"
    profile = SimpleNamespace(
        config_path=config_path,
        data=data,
        workspace_db=workspace_db,
        registry=LocalWorkspaceRegistryService(workspace_db),
        runs_path=runs_path,
        tmp_path=tmp_path,
    )
    try:
        yield profile
    finally:
        workspace_db.close()


def _runs_db(profile) -> AgentRunsDB:
    return AgentRunsDB(profile.runs_path, client_id="upgrade-test")


def _run(profile, **kwargs):
    return run_full_agent_tools_upgrade(
        workspace_db=profile.workspace_db,
        agent_runs_path=profile.runs_path,
        **kwargs,
    )


def _raw(profile) -> dict:
    return tomllib.loads(profile.config_path.read_text(encoding="utf-8"))


def _without_key(raw: dict) -> dict:
    tools = dict(raw.get("tools", {}))
    tools.pop("full_agent_tools_enabled", None)
    return {**raw, "tools": tools}


def _marker_recorded(profile) -> bool:
    return profile.workspace_db.is_agent_backfill_complete(
        FULL_AGENT_TOOLS_UPGRADE_MARKER
    )


def _bind_folder(profile) -> None:
    profile.registry.create_workspace(workspace_id="ws-project", name="Project")
    folder = profile.tmp_path / "project"
    folder.mkdir()
    profile.registry.add_folder_binding("ws-project", folder)


def _bind_ssh(profile) -> None:
    profile.registry.create_workspace(workspace_id="ws-remote", name="Remote")
    profile.registry.save_runtime_binding(
        WorkspaceRuntimeBinding(
            workspace_id="ws-remote",
            binding_id="ssh-abc123",
            binding_kind=RuntimeBindingKind.SSH_FILESYSTEM,
            label="repo",
            locator="ssh://me@devbox.example.com/home/me/repo",
            status=RuntimeBindingStatus.READY,
            metadata={"access": "ro", "python": "python3"},
        )
    )


def _define_subagent(profile, *, deleted: bool = False) -> None:
    runs = _runs_db(profile)
    try:
        definition_id = runs.create_agent_definition(
            AgentDefinition(
                name="researcher",
                description="Finds sources.",
                instructions="Search, then summarize.",
            )
        )
        if deleted:
            runs.soft_delete_agent_definition(definition_id)
    finally:
        runs.close()


def _record_run(profile, *steps: dict, agent_kind: str = "primary") -> str:
    runs = _runs_db(profile)
    try:
        run_id = runs.create_run(conversation_id="conv-1", agent_kind=agent_kind)
        if steps:
            runs.insert_steps_at_indices(
                run_id,
                [
                    (index, {"index": index, **step})
                    for index, step in enumerate(steps)
                ],
            )
        return run_id
    finally:
        runs.close()


def _tool_call(name: str) -> dict:
    return {"kind": "tool_call", "tool_name": name, "summary": "", "args": None}


# --------------------------------------------------------------------------
# AC#1: a profile that uses agent features keeps them
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("seed", "signal"),
    [
        (_bind_folder, SIGNAL_WORKSPACE_BINDING),
        (_bind_ssh, SIGNAL_WORKSPACE_BINDING),
        (_define_subagent, SIGNAL_AGENT_DEFINITION),
        (
            lambda profile: _record_run(
                profile, {"kind": "model", "summary": "hi"}, _tool_call("fs_list")
            ),
            SIGNAL_TOOL_CALL,
        ),
        (
            lambda profile: _record_run(
                profile,
                {"kind": "spawn", "tool_name": "spawn_subagent", "summary": "x"},
            ),
            SIGNAL_TOOL_CALL,
        ),
        (
            lambda profile: _record_run(profile, agent_kind="subagent"),
            SIGNAL_SUBAGENT_RUN,
        ),
    ],
    ids=[
        "folder-binding",
        "ssh-binding",
        "subagent-definition",
        "tool-call-run",
        "spawn-step",
        "subagent-run",
    ],
)
def test_a_profile_that_uses_agent_features_gets_full_agent_tools_written_once(
    profile, seed, signal
) -> None:
    seed(profile)
    before = _raw(profile)
    assert light_surface_configured() is True

    result = _run(profile)

    assert (result.outcome, result.signal) == (UpgradeOutcome.KEPT_ON, signal)
    after = _raw(profile)
    assert after["tools"]["full_agent_tools_enabled"] is True
    # The one key, nothing else: every other value is as it was.
    assert _without_key(after) == before
    assert _marker_recorded(profile)
    # Plain chat now gets the full surface on its next send.
    assert light_surface_configured() is False

    # Written once: the next launch reads the marker and stops there.
    written = profile.config_path.read_bytes()
    again = _run(profile)
    assert again.outcome is UpgradeOutcome.ALREADY_RECORDED
    assert profile.config_path.read_bytes() == written


def test_a_pre_v13_run_whose_steps_live_in_the_json_blob_counts(profile) -> None:
    """Runs from before task-18601 kept every step in ``agent_runs.steps``."""
    run_id = _record_run(profile)
    runs = _runs_db(profile)
    try:
        with runs.transaction() as connection:
            connection.execute(
                "UPDATE agent_runs SET steps = ? WHERE id = ?",
                (json.dumps([{"index": 0, **_tool_call("web_search")}]), run_id),
            )
    finally:
        runs.close()

    result = _run(profile)

    assert (result.outcome, result.signal) == (
        UpgradeOutcome.KEPT_ON,
        SIGNAL_TOOL_CALL,
    )
    assert _raw(profile)["tools"]["full_agent_tools_enabled"] is True


# --------------------------------------------------------------------------
# AC#2: a fresh or unused profile stays light with nothing written
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seed",
    [
        lambda profile: None,
        lambda profile: _define_subagent(profile, deleted=True),
        lambda profile: _record_run(
            profile,
            {"kind": "model", "summary": "hi"},
            _tool_call("calculator"),
            _tool_call("get_current_datetime"),
            _tool_call("ask_user"),
        ),
        # A local ``!command`` is the user's shell, not a model tool call.
        lambda profile: _record_run(
            profile, _tool_call("raw_cli"), agent_kind="local_command"
        ),
        # A run that only talked: model steps, no tool call.
        lambda profile: _record_run(profile, {"kind": "model", "summary": "x"}),
    ],
    ids=[
        "nothing",
        "deleted-definition",
        "light-tools-only",
        "local-command",
        "model-steps-only",
    ],
)
def test_a_profile_without_agent_use_stays_light_and_its_config_is_untouched(
    profile, seed
) -> None:
    seed(profile)
    before = profile.config_path.read_bytes()

    result = _run(profile)

    assert (result.outcome, result.signal) == (UpgradeOutcome.LIGHT, None)
    assert profile.config_path.read_bytes() == before
    assert _marker_recorded(profile)
    assert light_surface_configured() is True


def test_a_fresh_profile_with_no_runs_database_is_not_given_one(profile) -> None:
    """The runs database is read only if it exists; never created."""
    assert not profile.runs_path.exists()

    result = _run(profile)

    assert result.outcome is UpgradeOutcome.LIGHT
    assert not profile.runs_path.exists()


def test_a_malformed_step_row_does_not_hide_a_later_tool_call(profile) -> None:
    run_id = _record_run(profile, _tool_call("fs_list"))
    runs = _runs_db(profile)
    try:
        with runs.transaction() as connection:
            connection.execute(
                "INSERT INTO agent_run_steps (run_id, seq, payload, created_at) "
                "VALUES (?, ?, ?, ?)",
                (run_id, 99, "{not json tool_call", "2026-11-05T00:00:00Z"),
            )
    finally:
        runs.close()

    result = _run(profile)

    assert result.outcome is UpgradeOutcome.KEPT_ON


# --------------------------------------------------------------------------
# AC#3: at most once; a later "off" is never switched back on
# --------------------------------------------------------------------------


def test_agent_use_that_starts_after_the_upgrade_ran_turns_nothing_on(
    profile,
) -> None:
    assert _run(profile).outcome is UpgradeOutcome.LIGHT
    before = profile.config_path.read_bytes()

    _bind_folder(profile)
    _define_subagent(profile)

    assert _run(profile).outcome is UpgradeOutcome.ALREADY_RECORDED
    assert profile.config_path.read_bytes() == before
    assert light_surface_configured() is True


def test_turning_full_agent_tools_off_after_the_upgrade_stays_off(profile) -> None:
    from tldw_chatbook.config import save_setting_to_cli_config

    _bind_folder(profile)
    assert _run(profile).outcome is UpgradeOutcome.KEPT_ON
    assert save_setting_to_cli_config("tools", "full_agent_tools_enabled", False)

    assert _run(profile).outcome is UpgradeOutcome.ALREADY_RECORDED
    assert _raw(profile)["tools"]["full_agent_tools_enabled"] is False

    # Even with the marker lost (a replaced workspaces database), the
    # explicit "off" is the user's and is left alone.
    fresh = WorkspaceDB(profile.data / "replaced.db", client_id="upgrade-test")
    try:
        result = run_full_agent_tools_upgrade(
            workspace_db=fresh, agent_runs_path=profile.runs_path
        )
    finally:
        fresh.close()
    assert result.outcome is UpgradeOutcome.EXPLICIT
    assert _raw(profile)["tools"]["full_agent_tools_enabled"] is False
    assert light_surface_configured() is True


# --------------------------------------------------------------------------
# AC#4: an explicit value is left exactly as it is
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["true", "false"])
def test_an_explicit_value_is_left_exactly_as_it_is(profile, value) -> None:
    profile.config_path.write_text(
        PRE_UPGRADE_CONFIG + f"full_agent_tools_enabled = {value}  # mine\n",
        encoding="utf-8",
    )
    before = profile.config_path.read_bytes()
    _bind_folder(profile)

    result = _run(profile)

    assert result.outcome is UpgradeOutcome.EXPLICIT
    assert profile.config_path.read_bytes() == before
    assert _marker_recorded(profile)


def test_a_value_set_while_the_upgrade_ran_wins_under_the_write_lock(
    profile,
) -> None:
    """The write re-reads the raw file under the lock and never overwrites."""
    from tldw_chatbook.config import save_setting_to_cli_config

    assert save_setting_to_cli_config("tools", "full_agent_tools_enabled", False)
    before = profile.config_path.read_bytes()

    assert _write_full_agent_tools_on() is False
    assert profile.config_path.read_bytes() == before


# --------------------------------------------------------------------------
# AC#5: failures stay light, write nothing, retry, and log no user content
# --------------------------------------------------------------------------


@pytest.fixture
def captured_logs():
    """What the upgrade module itself logs (the DB and config layers log
    their own paths at INFO; those are theirs, not this module's)."""
    lines: list[str] = []
    handle = logger.add(
        lines.append,
        format="{level} {message}",
        level="DEBUG",
        filter="tldw_chatbook.Agents.full_agent_tools_upgrade",
    )
    try:
        yield lines
    finally:
        logger.remove(handle)


def test_an_unreadable_runs_database_fails_closed_and_retries(
    profile, captured_logs
) -> None:
    profile.runs_path.write_bytes(b"this is not an sqlite database " * 64)
    profile.runs_path.chmod(0o600)
    before = profile.config_path.read_bytes()

    result = _run(profile)

    assert result.outcome is UpgradeOutcome.FAILED
    assert result.error_type
    assert profile.config_path.read_bytes() == before
    assert not _marker_recorded(profile)
    assert light_surface_configured() is True
    logged = "\n".join(captured_logs)
    assert "upgrade check failed" in logged
    assert str(profile.tmp_path) not in logged

    # Nothing was recorded, so the next launch tries again.
    profile.runs_path.unlink()
    assert _run(profile).outcome is UpgradeOutcome.LIGHT


def test_detection_past_its_time_budget_fails_closed(profile) -> None:
    _record_run(profile, _tool_call("fs_list"))
    before = profile.config_path.read_bytes()

    result = _run(profile, budget_seconds=0.0)

    assert (result.outcome, result.error_type) == (
        UpgradeOutcome.FAILED,
        "_DetectionTimeout",
    )
    assert profile.config_path.read_bytes() == before
    assert not _marker_recorded(profile)


def test_a_read_interrupted_mid_query_fails_closed(profile, monkeypatch) -> None:
    """The progress handler that enforces the budget can stop any query."""
    from tldw_chatbook.Agents import full_agent_tools_upgrade as upgrade

    _record_run(profile, _tool_call("fs_list"))
    before = profile.config_path.read_bytes()
    monkeypatch.setattr(upgrade, "_deadline_handler", lambda deadline: lambda: 1)
    monkeypatch.setattr(upgrade, "_BUDGET_CHECK_STEPS", 1)

    result = _run(profile)

    assert (result.outcome, result.error_type) == (
        UpgradeOutcome.FAILED,
        "OperationalError",
    )
    assert profile.config_path.read_bytes() == before
    assert not _marker_recorded(profile)


def test_a_config_write_that_does_not_land_records_no_marker(
    profile, monkeypatch
) -> None:
    import tldw_chatbook.config as config_module

    _bind_folder(profile)
    before = profile.config_path.read_bytes()
    monkeypatch.setattr(
        config_module,
        "apply_literal_settings_transaction_to_cli_config",
        lambda builder: config_module.LiteralConfigMutationResult(
            False, False, None, "before_replace"
        ),
    )

    result = _run(profile)

    assert result.outcome is UpgradeOutcome.FAILED
    assert profile.config_path.read_bytes() == before
    assert not _marker_recorded(profile)


def test_a_missing_workspaces_database_fails_closed(profile) -> None:
    result = run_full_agent_tools_upgrade(
        workspace_db=None, agent_runs_path=profile.runs_path
    )

    assert result.outcome is UpgradeOutcome.FAILED
    assert light_surface_configured() is True


def test_logs_name_the_outcome_and_signal_but_no_path_or_tool(
    profile, captured_logs
) -> None:
    _record_run(profile, _tool_call("mcp__my_private_server__lookup"))

    assert _run(profile).outcome is UpgradeOutcome.KEPT_ON

    logged = "\n".join(captured_logs)
    assert "outcome=kept_on signal=tool_call" in logged
    assert "my_private_server" not in logged
    assert str(profile.tmp_path) not in logged


# --------------------------------------------------------------------------
# The app's boot-worker body
# --------------------------------------------------------------------------


def test_the_boot_worker_body_mirrors_the_written_key_into_app_config(
    profile,
) -> None:
    _bind_folder(profile)
    calls: list[str] = []

    def call_from_thread(function, *args):
        calls.append(function.__name__)
        return function(*args)

    app = SimpleNamespace(
        local_workspace_db=profile.workspace_db,
        # The runs database sits beside ChaChaNotes.
        chachanotes_db=SimpleNamespace(db_path=profile.data / "ChaChaNotes.db"),
        app_config={"tools": {"read_file_enabled": True}},
        call_from_thread=call_from_thread,
    )

    result = run_for_app(app)

    assert result.outcome is UpgradeOutcome.KEPT_ON
    assert calls == ["_mirror_into_app_config"]
    assert app.app_config["tools"] == {
        "read_file_enabled": True,
        "full_agent_tools_enabled": True,
    }
