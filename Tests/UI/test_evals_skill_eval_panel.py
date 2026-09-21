"""SkillEvalPanel mount, estimate math, RunRequested payload -- plus the
``EvalsViewModel`` skill-eval reads (bench list/lookup, model targets) the
screen that owns this panel will drive it from.

The panel is DB-free by design (its owning screen owns engines and storage),
so its harness is a bare ``App`` hosting one ``SkillEvalPanel`` -- no
``EvalsHarness``/``_FakeAppInstance`` and no ``EvalsDB`` anywhere near the
panel tests, unlike ``test_evals_character_bench_editor.py``'s editor tests.
What IS reused from that module's shape is the pilot pattern itself: a real
mounted app, real widget interaction (options picked through the Select's
own dropdown overlay, buttons pressed via ``pilot.click``), and messages
captured by a handler appending to a plain list.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.skill_eval.models import (
    SkillEvalConfig,
    SkillEvalDepth,
)
from tldw_chatbook.Evals.skill_eval.runner import estimate_calls
from tldw_chatbook.Evals.skill_eval.storage import save_skill_eval_bench
from tldw_chatbook.UI.Evals.evals_state import EvalsViewModel
from tldw_chatbook.UI.Evals.skill_eval_panel import SkillEvalPanel

#: Same realistic default as the sibling Evals workbench tests -- plenty of
#: room for every panel row without scrolling.
_REALISTIC_SIZE = (120, 45)

_TARGETS = [
    {"id": "g1", "name": "gen", "provider": "llama_cpp", "model_id": "m"},
    {"id": "j1", "name": "jud", "provider": "llama_cpp", "model_id": "m2"},
]


class _PanelHarness(App):
    """Minimal app hosting one bare ``SkillEvalPanel`` and recording the
    messages it posts -- the panel's whole contract is "post RunRequested/
    CancelRequested/SubjectChanged upward", so a plain list per message
    type is the entire harness."""

    def __init__(self) -> None:
        super().__init__()
        self.run_requested: list[SkillEvalPanel.RunRequested] = []
        self.cancel_requested: list[SkillEvalPanel.CancelRequested] = []
        self.subject_changed: list[SkillEvalPanel.SubjectChanged] = []

    def compose(self) -> ComposeResult:
        yield SkillEvalPanel()

    def on_skill_eval_panel_run_requested(
        self, event: SkillEvalPanel.RunRequested
    ) -> None:
        self.run_requested.append(event)

    def on_skill_eval_panel_cancel_requested(
        self, event: SkillEvalPanel.CancelRequested
    ) -> None:
        self.cancel_requested.append(event)

    def on_skill_eval_panel_subject_changed(
        self, event: SkillEvalPanel.SubjectChanged
    ) -> None:
        self.subject_changed.append(event)


async def _pick_via_overlay(pilot, select_id: str, downs: int) -> None:
    """Selects an option through the Select's own dropdown, the way a user
    does: click to expand, arrow to the option, enter to pick. One ``down``
    is consumed by the NULL padding option Textual prepends when
    ``allow_blank`` is on, so ``downs=2`` reaches the first real option,
    ``downs=3`` the second."""
    await pilot.click(f"#{select_id}")
    await pilot.pause()
    await pilot.press(*(["down"] * downs), "enter")
    await pilot.pause()


def test_estimate_label_matches_depth():
    assert estimate_calls(SkillEvalDepth.QUICK) == 0
    assert estimate_calls(SkillEvalDepth.STANDARD) == 16
    assert estimate_calls(SkillEvalDepth.DEEP) == 67


@pytest.mark.asyncio
async def test_panel_mounts_and_posts_run_requested():
    app = _PanelHarness()
    async with app.run_test(size=_REALISTIC_SIZE) as pilot:
        panel = app.screen.query_one(SkillEvalPanel)
        panel.set_subject("csv-cleaner", "store")
        panel.set_targets(_TARGETS)
        await pilot.pause()

        # First real option (one extra down for the overlay's NULL padding).
        await _pick_via_overlay(pilot, "skill-eval-generator", downs=2)
        await _pick_via_overlay(pilot, "skill-eval-judge", downs=3)
        assert app.screen.query_one("#skill-eval-generator").value == "g1"
        assert app.screen.query_one("#skill-eval-judge").value == "j1"

        # STANDARD is the depth select's own default -- the estimate line
        # must already reflect it before Run is ever pressed.
        estimate = str(app.screen.query_one("#skill-eval-estimate").render())
        assert "16" in estimate

        await pilot.click("#skill-eval-run")
        await pilot.pause()

        assert len(app.run_requested) == 1
        msg = app.run_requested[0]
        assert msg.generator_target_id == "g1"
        assert msg.judge_target_id == "j1"
        assert msg.depth is SkillEvalDepth.STANDARD


@pytest.mark.asyncio
async def test_run_without_both_models_picks_posts_nothing():
    """Run with no model picked must warn and post nothing -- the screen
    must never receive a half-populated RunRequested it would have to
    half-handle."""
    app = _PanelHarness()
    async with app.run_test(size=_REALISTIC_SIZE) as pilot:
        panel = app.screen.query_one(SkillEvalPanel)
        panel.set_targets(_TARGETS)
        await pilot.pause()

        await pilot.click("#skill-eval-run")
        await pilot.pause()

        assert app.run_requested == []


@pytest.mark.asyncio
async def test_cancel_button_posts_cancel_requested():
    app = _PanelHarness()
    async with app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.click("#skill-eval-cancel")
        await pilot.pause()
        assert len(app.cancel_requested) == 1


_SUBJECTS = [
    {"name": "csv-cleaner", "description": "Tidy CSV exports.",
     "trust_status": "trusted"},
    {"name": "pdf-writer", "description": "Write PDFs.",
     "trust_status": "unverified"},
]


@pytest.mark.asyncio
async def test_subject_picker_lists_store_skills_and_posts_store_change():
    """``set_subjects`` feeds the picker (label ``name (trust)``); a pick
    through the overlay posts ``SubjectChanged(kind="store")`` carrying the
    bare skill name."""
    app = _PanelHarness()
    async with app.run_test(size=_REALISTIC_SIZE) as pilot:
        panel = app.screen.query_one(SkillEvalPanel)
        panel.set_subjects(_SUBJECTS)
        await pilot.pause()

        from textual.widgets import Select

        options = panel.query_one("#skill-eval-subject-picker", Select)._options
        # _options leads with the allow-blank NULL padding entry.
        values = [value for _label, value in options if value is not Select.NULL]
        assert values == ["csv-cleaner", "pdf-writer"]
        assert any("csv-cleaner (trusted)" in str(label)
                   for label, value in options if value is not Select.NULL)

        await _pick_via_overlay(pilot, "skill-eval-subject-picker", downs=2)
        assert [msg.subject_ref for msg in app.subject_changed] == ["csv-cleaner"]
        assert app.subject_changed[0].subject_kind == "store"


@pytest.mark.asyncio
async def test_last_touched_wins_between_picker_and_directory_input():
    """A store pick clears the directory Input; typing a path afterwards
    resets the Select to NULL and posts a ``directory`` change -- the
    documented last-touched-wins rule (one effective choice at a time)."""
    from textual.widgets import Input, Select

    app = _PanelHarness()
    async with app.run_test(size=_REALISTIC_SIZE) as pilot:
        panel = app.screen.query_one(SkillEvalPanel)
        panel.set_subjects(_SUBJECTS)
        await pilot.pause()

        await _pick_via_overlay(pilot, "skill-eval-subject-picker", downs=2)
        await pilot.pause()
        assert panel.query_one("#skill-eval-subject-dir", Input).value == ""

        panel.query_one("#skill-eval-subject-dir", Input).value = "/tmp/skill"
        await pilot.pause()
        assert panel.query_one("#skill-eval-subject-picker", Select).value is Select.NULL
        assert [(m.subject_ref, m.subject_kind) for m in app.subject_changed] == [
            ("csv-cleaner", "store"),
            ("/tmp/skill", "directory"),
        ]


@pytest.mark.asyncio
async def test_emptying_directory_input_posts_nothing():
    """Emptying the Input is not a subject choice (and resurrects no cleared
    store pick) -- the screen must never receive an empty-subject persist."""
    app = _PanelHarness()
    async with app.run_test(size=_REALISTIC_SIZE) as pilot:
        panel = app.screen.query_one(SkillEvalPanel)
        directory = panel.query_one("#skill-eval-subject-dir")
        directory.value = "/tmp/skill"
        await pilot.pause()
        assert len(app.subject_changed) == 1

        directory.value = ""
        await pilot.pause()
        assert len(app.subject_changed) == 1


# ---------------------------------------------------------------------------
# EvalsViewModel: the three skill-eval reads
# ---------------------------------------------------------------------------


@pytest.fixture
def evals_db() -> EvalsDB:
    return EvalsDB(db_path=":memory:", client_id="test")


def _make_skill_bench(evals_db: EvalsDB) -> str:
    return save_skill_eval_bench(
        evals_db,
        SkillEvalConfig(
            name="csv-cleaner eval",
            subject_ref="csv-cleaner",
            subject_kind="store",
            depth=SkillEvalDepth.STANDARD,
            generator_target_id="g1",
            judge_target_id="j1",
        ),
    )


def test_skill_eval_benches_returns_only_skill_eval_rows(evals_db):
    bench_id = _make_skill_bench(evals_db)
    evals_db.create_task(
        name="a classic task",
        task_type="generation",
        config_format="custom",
        config_data={},
    )

    rows = EvalsViewModel(evals_db).skill_eval_benches()

    assert [row["id"] for row in rows] == [bench_id]
    assert rows[0]["name"] == "csv-cleaner eval"


def test_skill_eval_benches_degrades_empty_without_db():
    assert EvalsViewModel(None).skill_eval_benches() == []


def test_skill_eval_bench_by_id_resolves_skill_benches_only(evals_db):
    bench_id = _make_skill_bench(evals_db)
    classic_id = evals_db.create_task(
        name="a classic task",
        task_type="generation",
        config_format="custom",
        config_data={},
    )

    view_model = EvalsViewModel(evals_db)
    assert (view_model.skill_eval_bench_by_id(bench_id) or {}).get("id") == bench_id
    assert view_model.skill_eval_bench_by_id(classic_id) is None
    assert view_model.skill_eval_bench_by_id("no-such-id") is None
    assert view_model.skill_eval_bench_by_id("") is None
    assert EvalsViewModel(None).skill_eval_bench_by_id(bench_id) is None


def test_skill_eval_targets_lists_models(evals_db):
    gen_id = evals_db.create_model(
        name="gen", provider="llama_cpp", model_id="m"
    )
    judge_id = evals_db.create_model(
        name="jud", provider="llama_cpp", model_id="m2"
    )

    rows = EvalsViewModel(evals_db).skill_eval_targets()

    assert {row["id"] for row in rows} == {gen_id, judge_id}
    assert EvalsViewModel(None).skill_eval_targets() == []
