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

#: This module's import graph (UI.Evals -> config getters) binds config
#: participants at collection, so the per-test env redirect trips the
#: config-participant admission (TASK-32628) and every node errors at setup
#: with RecoveryRequired("raw_source_selection_changed"). The sanctioned
#: per-file opt-in (TASK-32873) keeps the collection-time bootstrap profile
#: -- itself a hermetic conftest temp root -- for these DB-free panel tests.
pytestmark = pytest.mark.bootstrap_profile

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
        self.close_requested: list[SkillEvalPanel.CloseRequested] = []

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

    def on_skill_eval_panel_close_requested(
        self, event: SkillEvalPanel.CloseRequested
    ) -> None:
        self.close_requested.append(event)


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
async def test_run_without_subject_posts_nothing():
    """TASK-32885: Run with both models picked but no subject set must warn
    and post nothing -- a subjectless RunRequested previously became a real
    run that failed in the worker and left a failed row in the rail."""
    app = _PanelHarness()
    async with app.run_test(size=_REALISTIC_SIZE) as pilot:
        panel = app.screen.query_one(SkillEvalPanel)
        panel.set_targets(_TARGETS)
        await pilot.pause()

        await _pick_via_overlay(pilot, "skill-eval-generator", downs=2)
        await _pick_via_overlay(pilot, "skill-eval-judge", downs=2)
        await pilot.click("#skill-eval-run")
        await pilot.pause()
        assert app.run_requested == []

        # The screen's remount sentinel must NOT count as a subject.
        panel.set_subject("(no subject set)", "store")
        await pilot.click("#skill-eval-run")
        await pilot.pause()
        assert app.run_requested == []

        # A persisted subject remount makes Run dispatchable again.
        panel.set_subject("csv-cleaner", "store")
        await pilot.click("#skill-eval-run")
        await pilot.pause()
        assert len(app.run_requested) == 1


@pytest.mark.asyncio
async def test_directory_path_subject_enables_run(tmp_path):
    """TASK-32885: a typed directory path is as valid a subject as a store
    pick -- the guard must accept it without a picker choice. An INVALID
    path must not arm the guard (Qodo review)."""
    skill_dir = tmp_path / "csv"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# csv\n", encoding="utf-8")

    app = _PanelHarness()
    async with app.run_test(size=_REALISTIC_SIZE) as pilot:
        panel = app.screen.query_one(SkillEvalPanel)
        panel.set_targets(_TARGETS)
        await pilot.pause()

        # Invalid first: no SKILL.md -> no subject, Run posts nothing.
        directory = panel.query_one("#skill-eval-subject-dir")
        directory.value = str(tmp_path)
        await pilot.pause()
        await _pick_via_overlay(pilot, "skill-eval-generator", downs=2)
        await _pick_via_overlay(pilot, "skill-eval-judge", downs=2)
        await pilot.click("#skill-eval-run")
        await pilot.pause()
        assert app.run_requested == []
        # The guard's warning toast renders over the button for ~5s and
        # would swallow the second Run click -- clear it first.
        app.clear_notifications()
        await pilot.pause()
        directory.value = ""
        await pilot.pause()

        directory.value = str(skill_dir)
        await pilot.pause()
        # Models were picked during the invalid-path phase above and are
        # still held (a set Select's Enter no longer re-opens -- re-picking
        # through the overlay here would strand an open overlay that
        # swallows the Run click).
        await pilot.click("#skill-eval-run")
        await pilot.pause()
        assert len(app.run_requested) == 1


@pytest.mark.asyncio
async def test_stop_run_button_posts_cancel_requested_only_when_enabled():
    """TASK-32889: the panel's second action is 'Stop run', DISABLED while
    idle -- the old always-enabled 'Cancel' read as 'close this form' and
    silently no-opped when no run existed."""
    from textual.widgets import Button

    app = _PanelHarness()
    async with app.run_test(size=_REALISTIC_SIZE) as pilot:
        stop = app.screen.query_one("#skill-eval-cancel", Button)
        assert stop.disabled
        assert "Stop run" in str(stop.label)
        await pilot.click("#skill-eval-cancel")
        await pilot.pause()
        assert app.cancel_requested == []

        # Enabled by the screen while a run is in flight: press stops it.
        stop.disabled = False
        await pilot.click("#skill-eval-cancel")
        await pilot.pause()
        assert len(app.cancel_requested) == 1


@pytest.mark.asyncio
async def test_escape_on_panel_posts_close_requested():
    """TASK-32889: Escape on the launch panel asks the screen to close it
    (selection-level, never popping the Lab screen) -- previously there
    was no keyboard way to leave the panel at all."""
    app = _PanelHarness()
    async with app.run_test(size=_REALISTIC_SIZE) as pilot:
        panel = app.screen.query_one(SkillEvalPanel)
        panel.query_one("#skill-eval-subject-dir").focus()
        await pilot.press("escape")
        await pilot.pause()
        assert len(app.close_requested) == 1


@pytest.mark.asyncio
async def test_enter_on_a_set_select_no_ops_instead_of_reopening():
    """TASK-32889: Enter on a Select that already holds a value is a
    no-op -- it used to silently re-open the overlay, stranding keyboard
    users who pressed Enter to confirm/advance. Space still opens."""
    app = _PanelHarness()
    async with app.run_test(size=_REALISTIC_SIZE) as pilot:
        from textual.widgets import Select

        # The depth select mounts with STANDARD set.
        depth = app.screen.query_one("#skill-eval-depth", Select)
        depth.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert not depth.expanded

        await pilot.press("space")
        await pilot.pause()
        assert depth.expanded


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
async def test_last_touched_wins_between_picker_and_directory_input(tmp_path):
    """A store pick clears the directory Input; typing a VALID path
    afterwards resets the Select to NULL and posts a ``directory``
    change -- the documented last-touched-wins rule (one effective
    choice at a time)."""
    from textual.widgets import Input, Select

    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# my skill\n", encoding="utf-8")

    app = _PanelHarness()
    async with app.run_test(size=_REALISTIC_SIZE) as pilot:
        panel = app.screen.query_one(SkillEvalPanel)
        panel.set_subjects(_SUBJECTS)
        await pilot.pause()

        await _pick_via_overlay(pilot, "skill-eval-subject-picker", downs=2)
        await pilot.pause()
        assert panel.query_one("#skill-eval-subject-dir", Input).value == ""

        panel.query_one("#skill-eval-subject-dir", Input).value = str(skill_dir)
        await pilot.pause()
        assert panel.query_one("#skill-eval-subject-picker", Select).value is Select.NULL
        assert [(m.subject_ref, m.subject_kind) for m in app.subject_changed] == [
            ("csv-cleaner", "store"),
            (str(skill_dir), "directory"),
        ]


@pytest.mark.asyncio
async def test_emptying_directory_input_posts_nothing(tmp_path):
    """Emptying the Input is not a subject choice (and resurrects no cleared
    store pick) -- the screen must never receive an empty-subject persist."""
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# my skill\n", encoding="utf-8")

    app = _PanelHarness()
    async with app.run_test(size=_REALISTIC_SIZE) as pilot:
        panel = app.screen.query_one(SkillEvalPanel)
        directory = panel.query_one("#skill-eval-subject-dir")
        directory.value = str(skill_dir)
        await pilot.pause()
        assert len(app.subject_changed) == 1

        directory.value = ""
        await pilot.pause()
        assert len(app.subject_changed) == 1


@pytest.mark.asyncio
async def test_empty_store_picker_shows_guidance_not_a_blank_overlay():
    """TASK-32883: an empty skills store must not render a blank dead-end
    overlay -- the picker's prompt names the problem and the alternative,
    the overlay holds one guidance row, and picking that row posts nothing
    instead of silently selecting a fake subject."""
    from textual.widgets import Select

    app = _PanelHarness()
    async with app.run_test(size=_REALISTIC_SIZE) as pilot:
        panel = app.screen.query_one(SkillEvalPanel)
        panel.set_subjects([])
        await pilot.pause()

        picker = panel.query_one("#skill-eval-subject-picker", Select)
        assert "No skills" in str(picker.prompt)
        labels = [str(label) for label, value in picker._options
                  if value is not Select.NULL]
        assert any("No skills" in label for label in labels)

        await _pick_via_overlay(pilot, "skill-eval-subject-picker", downs=2)
        assert app.subject_changed == []
        assert picker.value is Select.NULL


@pytest.mark.asyncio
async def test_populated_store_picker_keeps_the_standard_prompt():
    """TASK-32883 guard: the guidance prompt swap must only happen for an
    actually-empty store -- populated pickers keep the neutral prompt."""
    app = _PanelHarness()
    async with app.run_test(size=_REALISTIC_SIZE) as pilot:
        panel = app.screen.query_one(SkillEvalPanel)
        panel.set_subjects(_SUBJECTS)
        await pilot.pause()

        from textual.widgets import Select

        picker = panel.query_one("#skill-eval-subject-picker", Select)
        assert str(picker.prompt) == "subject skill (store)"


@pytest.mark.asyncio
async def test_directory_input_hint_and_inline_validation(tmp_path):
    """TASK-32883: the path input names the expected layout inline, flags a
    directory without SKILL.md immediately, and validates a real skill
    directory clean -- no silent acceptance of unusable paths."""
    app = _PanelHarness()
    async with app.run_test(size=_REALISTIC_SIZE) as pilot:
        panel = app.screen.query_one(SkillEvalPanel)
        directory = panel.query_one("#skill-eval-subject-dir")
        hint = panel.query_one("#skill-eval-subject-dir-hint")
        assert "SKILL.md" in str(hint.render())

        directory.value = str(tmp_path)  # exists, but no SKILL.md inside
        await pilot.pause()
        assert not directory.is_valid
        assert "No SKILL.md" in str(hint.render())
        # Qodo review: an invalid path is not a subject choice -- nothing
        # posts and the (empty) effective subject does not arm Run.
        assert app.subject_changed == []

        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# my skill\n", encoding="utf-8")
        directory.value = str(skill_dir)
        await pilot.pause()
        assert directory.is_valid
        assert "No SKILL.md" not in str(hint.render())


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


# ---------------------------------------------------------------------------
# Qodo PR-review fixes: F1 (retry-aware estimate) + F4 (dispatchable targets)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_estimate_line_shows_the_retry_maximum():
    """F1: the estimate line carries the nominal count AND the worst case
    with judge retries (quick needs no parenthetical -- 0 max)."""
    from tldw_chatbook.Evals.skill_eval.runner import max_estimate_calls

    assert max_estimate_calls(SkillEvalDepth.STANDARD) == 32

    app = _PanelHarness()
    async with app.run_test(size=_REALISTIC_SIZE) as pilot:
        from textual.widgets import Select

        # STANDARD is the mounted default.
        estimate = str(app.screen.query_one("#skill-eval-estimate").render())
        assert "16" in estimate and "max 32" in estimate

        app.screen.query_one("#skill-eval-depth", Select).value = (
            SkillEvalDepth.DEEP
        )
        await pilot.pause()
        estimate = str(app.screen.query_one("#skill-eval-estimate").render())
        assert "67" in estimate and "max 84" in estimate

        app.screen.query_one("#skill-eval-depth", Select).value = (
            SkillEvalDepth.QUICK
        )
        await pilot.pause()
        estimate = str(app.screen.query_one("#skill-eval-estimate").render())
        assert "0" in estimate and "max" not in estimate


def test_skill_eval_targets_excludes_undispatchable_providers(evals_db):
    """F4: rows whose provider has no ``chat_api_call`` handler (e.g. the
    keyless ``local_transformers`` alias, which passes readiness) never
    reach the picker -- picking one doomed the run at its first call."""
    dispatchable = evals_db.create_model(
        name="gen", provider="llama_cpp", model_id="m"
    )
    evals_db.create_model(
        name="local", provider="local_transformers", model_id="t5"
    )
    evals_db.create_model(
        name="typo", provider="NoSuchProvider", model_id="x"
    )

    rows = EvalsViewModel(evals_db).skill_eval_targets()

    assert [row["id"] for row in rows] == [dispatchable]
    assert EvalsViewModel(None).skill_eval_targets() == []


@pytest.mark.asyncio
async def test_empty_model_pickers_show_bootstrap_guidance():
    """TASK-32884: with no eval models configured the pickers must name
    the problem and the way out, not render blank; picking the guidance
    row posts nothing and leaves the picker unset."""
    from textual.widgets import Select

    app = _PanelHarness()
    async with app.run_test(size=_REALISTIC_SIZE) as pilot:
        panel = app.screen.query_one(SkillEvalPanel)
        panel.set_targets([])
        await pilot.pause()

        for picker_id in ("skill-eval-generator", "skill-eval-judge"):
            picker = panel.query_one(f"#{picker_id}", Select)
            assert "no eval models" in str(picker.prompt)
            labels = [str(label) for label, value in picker._options
                      if value is not Select.NULL]
            assert any("No eval models" in label for label in labels)

        await _pick_via_overlay(pilot, "skill-eval-generator", downs=2)
        picker = panel.query_one("#skill-eval-generator", Select)
        assert picker.value is Select.NULL


@pytest.mark.asyncio
async def test_run_guard_toast_teaches_the_fix_when_no_models_exist():
    """TASK-32884: the Run guard's warning must name where eval models
    come from when none exist -- 'pick models first' alone strands a user
    who has nowhere to pick from."""
    app = _PanelHarness()
    async with app.run_test(size=_REALISTIC_SIZE) as pilot:
        panel = app.screen.query_one(SkillEvalPanel)
        panel.set_subject("csv-cleaner", "store")
        panel.set_targets([])
        await pilot.pause()

        await pilot.click("#skill-eval-run")
        await pilot.pause()
        messages = [n.message for n in app._notifications]
        assert any("No eval models" in m and "target" in m for m in messages), messages


@pytest.mark.asyncio
async def test_run_guard_toast_stays_plain_when_models_exist():
    """TASK-32884 guard: the bootstrap teaching belongs only to the
    zero-models case -- with models available the original message
    stands."""
    app = _PanelHarness()
    async with app.run_test(size=_REALISTIC_SIZE) as pilot:
        panel = app.screen.query_one(SkillEvalPanel)
        panel.set_subject("csv-cleaner", "store")
        panel.set_targets(_TARGETS)
        await pilot.pause()

        await pilot.click("#skill-eval-run")
        await pilot.pause()
        messages = [n.message for n in app._notifications]
        assert any("Pick generator and judge models first" in m for m in messages)
        assert not any("No eval models" in m for m in messages)
