"""TASK-858: EvalDBOperations' default path must agree with
config.get_evals_db_path() -- the same accessor
Evals.eval_orchestrator.EvaluationOrchestrator delegates to for its own
default case.

Before this fix, EvalDBOperations.__init__() hardcoded
``Path.home() / ".config" / "tldw_cli" / "evals.db"`` -- a profile-unaware
literal that named a different file than the one the app's real Evals
database (and the Settings screen's maintenance panel) opens.
"""

from __future__ import annotations

from pathlib import Path

from tldw_chatbook.Event_Handlers.eval_db_operations import EvalDBOperations


def test_default_db_path_matches_get_evals_db_path():
    from tldw_chatbook.config import get_evals_db_path

    ops = EvalDBOperations()

    assert Path(ops.db.db_path) == get_evals_db_path()


def test_default_db_path_tracks_a_retargeted_profile(monkeypatch, tmp_path):
    """The default must be resolved at construction time against the
    currently active profile, not a fixed HOME-relative literal."""
    from tldw_chatbook.config import get_evals_db_path

    retargeted = tmp_path / "profile-two" / "config.toml"
    retargeted.parent.mkdir()
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(retargeted))

    ops = EvalDBOperations()

    assert Path(ops.db.db_path) == get_evals_db_path()
    assert str(tmp_path) in str(ops.db.db_path)


def test_explicit_db_path_is_unaffected(tmp_path):
    explicit_path = tmp_path / "custom_evals.db"

    ops = EvalDBOperations(db_path=explicit_path)

    assert Path(ops.db.db_path) == explicit_path
