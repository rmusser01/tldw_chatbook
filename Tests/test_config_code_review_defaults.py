"""Saved teams stay empty by default and retain stable unresolved preset IDs."""

import tomllib

import pytest

from Tests.Backup_Recovery.config_test_support import select_config_source
from tldw_chatbook import config
from tldw_chatbook.Agents.code_review_models import ReviewRefused, get_code_review_team


def test_template_defaults_do_not_select_destinations():
    template = tomllib.loads(config.CONFIG_TOML_CONTENT)
    assert template["agents"]["code_review"] == {
        "reviewers": [],
        "verifier_preset_id": "",
    }
    assert template["tools"]["code_review_enabled"] is True
    assert "subagent_default_provider" not in template["agents"]["code_review"]


def test_missing_config_has_empty_team(tmp_path, monkeypatch):
    select_config_source(monkeypatch, str(tmp_path / "missing.toml"), globals())
    assert get_code_review_team().reviewers == ()
    assert get_code_review_team().verifier_preset_id is None


def test_config_merges_preserve_unrelated_keys_and_stable_missing_ids(
    tmp_path, monkeypatch
):
    path = tmp_path / "config.toml"
    path.write_text(
        '[agents]\nsubagent_default_provider = "llama_cpp"\n'
        '[agents.code_review]\nverifier_preset_id = "missing-verifier"\n'
        '[[agents.code_review.reviewers]]\npreset_id = "missing-reviewer"\n'
        'focus = "edge cases"\n[tools]\ncustom_enabled = false\n',
        encoding="utf-8",
    )
    select_config_source(monkeypatch, str(path), globals())
    team = get_code_review_team()
    assert team.reviewers[0].preset_id == "missing-reviewer"
    assert team.verifier_preset_id == "missing-verifier"
    assert config.get_cli_setting("agents", "subagent_default_provider") == "llama_cpp"
    assert config.get_cli_setting("tools", "custom_enabled") is False
    assert config.get_cli_setting("tools", "code_review_enabled") is True
    assert config.save_setting_to_cli_config(
        "agents.code_review", "verifier_preset_id", "renamed-id"
    )
    assert get_code_review_team().reviewers == team.reviewers
    assert get_code_review_team().verifier_preset_id == "renamed-id"


def test_malformed_saved_team_is_visible_as_refusal(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text(
        '[agents.code_review]\nreviewers = ["missing-id"]\n', encoding="utf-8"
    )
    select_config_source(monkeypatch, str(path), globals())
    with pytest.raises(ReviewRefused):
        get_code_review_team()


def test_settings_passes_team_and_tool_gate_through_without_losing_agent_keys(
    tmp_path, monkeypatch
):
    path = tmp_path / "config.toml"
    path.write_text(
        '[agents]\nsubagent_default_model = "chosen-model"\n'
        "[agents.code_review]\nreviewers = []\n"
        "[tools]\ncode_review_enabled = false\ncustom_enabled = true\n",
        encoding="utf-8",
    )
    select_config_source(monkeypatch, str(path), globals())
    settings = config.load_settings()
    assert settings["agents"]["subagent_default_model"] == "chosen-model"
    assert settings["agents"]["code_review"]["reviewers"] == []
    assert settings["tools"] == {"code_review_enabled": False, "custom_enabled": True}


def test_preset_rename_keeps_saved_id_and_deleted_id_remains_visible(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Agents.agent_models import AgentDefinition
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    db = AgentRunsDB(":memory:", client_id="review-defaults")
    try:
        preset_id = db.create_agent_definition(
            AgentDefinition(name="before", instructions="Review")
        )
        path = tmp_path / "config.toml"
        path.write_text(
            f'[agents.code_review]\nreviewers = [{{preset_id = "{preset_id}"}}]\n',
            encoding="utf-8",
        )
        select_config_source(monkeypatch, str(path), globals())
        db.update_agent_definition(
            preset_id, AgentDefinition(name="after", instructions="Review")
        )
        assert (
            db.get_agent_definition(get_code_review_team().reviewers[0].preset_id)[
                "name"
            ]
            == "after"
        )
        db.soft_delete_agent_definition(preset_id)
        assert db.get_agent_definition(preset_id) is None
        assert get_code_review_team().reviewers[0].preset_id == preset_id
    finally:
        db.close()
