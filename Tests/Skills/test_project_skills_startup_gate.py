# Tests/Skills/test_project_skills_startup_gate.py
from pathlib import Path
from types import SimpleNamespace

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


def test_discover_project_skills_for_startup_never_raises(monkeypatch, tmp_path):
    """The worker body must swallow ANY exception, not just OSError on cwd().

    ``TldwCli._discover_project_skills_for_startup`` runs on a worker
    thread with ``exit_on_error=False``, but that alone still surfaces an
    unhandled exception as a logged worker error -- an entirely optional
    startup nicety must degrade to a quiet no-op instead. Exercised by
    monkeypatching ``startup_discovery_for`` at its home module (the
    method imports it locally on every call, so patching the module
    attribute is visible to that import) to raise, then calling the
    method directly on a minimal stub object -- no real ``TldwCli``
    instance needed since the exception fires before anything else on
    ``self`` is touched.
    """
    import tldw_chatbook.Skills_Interop.project_skills_prompt as prompt_module
    from tldw_chatbook.app import TldwCli

    def _raise(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(prompt_module, "startup_discovery_for", _raise)
    monkeypatch.chdir(tmp_path)

    stub = SimpleNamespace()
    result = TldwCli._discover_project_skills_for_startup(stub)
    assert result is None
