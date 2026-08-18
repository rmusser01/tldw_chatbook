# Tests/Skills/test_project_skills_startup_gate.py
from pathlib import Path

from tldw_chatbook.Skills_Interop.project_skills_prompt import (
    startup_discovery_for,
)


def _skill(root):
    d = root / ".SKILLS" / "alpha-skill"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("---\ndescription: x\n---\nB\n", encoding="utf-8")


def test_startup_discovery_found(tmp_path):
    _skill(tmp_path / "repo")
    (tmp_path / "repo" / ".git").mkdir()
    sub = tmp_path / "repo" / "src"
    sub.mkdir()
    discovery = startup_discovery_for(sub, enabled=True, ledger_dir=tmp_path / "data")
    assert discovery is not None and discovery.entries


def test_startup_discovery_disabled(tmp_path):
    _skill(tmp_path)
    assert startup_discovery_for(tmp_path, enabled=False, ledger_dir=tmp_path / "d") is None


def test_startup_discovery_respects_never(tmp_path):
    _skill(tmp_path)
    from tldw_chatbook.Skills_Interop.project_skills_prompt import (
        ProjectSkillsPromptLedger,
    )
    ledger = ProjectSkillsPromptLedger(tmp_path / "data")
    ledger.record(tmp_path.resolve(), "never", "anything")
    assert startup_discovery_for(tmp_path, enabled=True, ledger_dir=tmp_path / "data") is None
