# Tests/Skills/test_project_skills_prompt.py
from pathlib import Path

from tldw_chatbook.Skills_Interop.project_skills_prompt import (
    ProjectSkillsPromptLedger,
    should_offer_project_skills_prompt,
)


def test_gating_truth_table():
    assert should_offer_project_skills_prompt(False, None, "f1") is False
    assert should_offer_project_skills_prompt(True, None, "f1") is True
    assert should_offer_project_skills_prompt(True, ("never", "f0"), "f1") is False
    assert should_offer_project_skills_prompt(True, ("declined", "f1"), "f1") is False
    assert should_offer_project_skills_prompt(True, ("declined", "f0"), "f1") is True
    assert should_offer_project_skills_prompt(True, ("imported", "f0"), "f1") is True


def test_ledger_roundtrip_and_missing(tmp_path):
    ledger = ProjectSkillsPromptLedger(tmp_path)
    directory = Path("/some/project")
    assert ledger.decision_for(directory) is None
    ledger.record(directory, "declined", "f1")
    assert ledger.decision_for(directory) == ("declined", "f1")
    ledger.record(directory, "never", "f2")
    assert ledger.decision_for(directory) == ("never", "f2")


def test_ledger_survives_corrupt_file(tmp_path):
    path = tmp_path / "skills" / "project_prompts.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    ledger = ProjectSkillsPromptLedger(tmp_path)
    assert ledger.decision_for(Path("/x")) is None
    ledger.record(Path("/x"), "imported", "f1")
    assert ledger.decision_for(Path("/x")) == ("imported", "f1")


def test_ledger_key_normalizes_path_spellings(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "proj").mkdir()
    ledger = ProjectSkillsPromptLedger(tmp_path)
    unresolved = tmp_path / "x" / ".." / "proj"
    ledger.record(unresolved, "imported", "f1")
    assert ledger.decision_for(tmp_path / "proj") == ("imported", "f1")
