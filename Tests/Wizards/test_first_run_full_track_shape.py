"""TASK-34100.13 (fulltrack-05): the Full track has no Notes step.

The step had no control and committed nothing; lasting Notes sync is set up
from Library ▸ Notes ▸ Add from files…, which the Summary now names. Drafts
saved by the eleven-step Full track (at the Notes step, or with the old RAG
and splash-card fields) still resume instead of being silently dropped.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state


def test_full_track_has_no_notes_step() -> None:
    full = wizard_state.active_step_ids(wizard_state.TRACK_FULL, key_entered=False)
    assert "notes" not in full
    assert len(full) == 10
    assert full[-1] == wizard_state.STEP_SUMMARY


def test_the_full_track_is_the_same_whatever_happens_mid_run() -> None:
    assert wizard_state.active_step_ids(
        wizard_state.TRACK_FULL, key_entered=False
    ) == wizard_state.active_step_ids(wizard_state.TRACK_FULL, key_entered=True)


def test_wizard_no_longer_owns_the_notes_section() -> None:
    assert "notes" not in wizard_state.WIZARD_OWNED_SECTIONS
    assert wizard_state.commit_sections_allowed({"notes": {"x": 1}}) is False


def test_summary_notes_row_points_at_add_from_files() -> None:
    """Review G13-R1-01 (coordinator decision 2026-11-05): sync roots live in
    the Library database, which the Summary never reads, so the row is an
    action pointer and claims no state -- "not set up" was false on a re-run
    after a folder had been added in Library."""
    rows = {
        row.label: row
        for row in wizard_state.build_summary_rows({}, {}, rag_deps_installed=True)
    }
    row = rows["Notes folder sync"]
    assert row.state == wizard_state.ROW_DEFAULT
    assert row.detail == "set up in Library ▸ Notes ▸ Add from files…"
    assert "not set up" not in row.detail


def _draft_config(active_step: str, values: dict) -> dict:
    return {
        "first_run": {
            "draft_version": wizard_state.SETUP_DRAFT_VERSION,
            "draft_track": wizard_state.TRACK_FULL,
            "active_step_id": active_step,
            "draft_values": values,
            "resume_attempted": False,
        }
    }


def test_a_draft_saved_at_the_retired_notes_step_resumes_at_the_next_step() -> None:
    draft = wizard_state.read_setup_draft(
        _draft_config("notes", {"welcome": {"track": "full"}, "notes": {}})
    )
    assert draft is not None
    assert draft.active_step_id == wizard_state.STEP_APPEARANCE
    assert "notes" not in draft.values


@pytest.mark.parametrize(
    ("step_id", "retired_field"),
    [("rag", "embedding_model"), ("appearance", "splash_card")],
)
def test_a_draft_with_a_retired_field_keeps_its_other_values(
    step_id: str, retired_field: str
) -> None:
    values = {
        "welcome": {"track": "full"},
        step_id: {retired_field: "something"},
        "model": {"model_id": "gpt-4.1-mini"},
    }
    draft = wizard_state.read_setup_draft(_draft_config("tools", values))
    assert draft is not None
    assert draft.values["model"] == {"model_id": "gpt-4.1-mini"}
    assert retired_field not in draft.values.get(step_id, {})


# -- In the real wizard frame ---------------------------------------------------


def _hosted_wizard(*, resume_draft=None, app_config=None):
    """The real FirstRunSetupWizard in the app stylesheet, without the app."""
    from pathlib import Path
    from unittest.mock import MagicMock

    from textual.app import ComposeResult

    from Tests.UI.consolidated_css import ConsolidatedCSSApp
    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import FirstRunSetupWizard

    bundle = str(
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook/css/tldw_cli_modular.tcss"
    )

    class _Host(ConsolidatedCSSApp):
        CSS_PATH = bundle

        def __init__(self, wizard) -> None:
            super().__init__()
            self._wizard = wizard

        def compose(self) -> ComposeResult:
            yield from ()

        async def on_mount(self) -> None:
            self.push_screen(self._wizard)

    app_instance = MagicMock()
    app_instance.app_config = {} if app_config is None else app_config
    wizard = FirstRunSetupWizard(app_instance, resume_draft=resume_draft)
    return _Host(wizard)


async def _progress_at(container, position: int) -> tuple[int, int]:
    """(tracker items, nav total) once the nav shows ``position``.

    Polls with plain sleeps: ``pilot.pause()`` waits up to 30 s for the whole
    screen to go idle, and on a loaded host a Summary or Speech ``on_show``
    worker overran that (WaitForScreenTimeout) although the counts were right.
    """
    import asyncio

    from tldw_chatbook.UI.Wizards.BaseWizard import WizardNavigation

    nav = container.query_one(".wizard-navigation", WizardNavigation)
    items = 0
    for _ in range(200):
        await asyncio.sleep(0.05)
        items = len(container.query(".setup-progress-item"))
        if nav.current_step == position + 1 and items == nav.total_steps:
            break
    return items, nav.total_steps


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_the_live_full_track_counts_ten_steps_across_next_and_back() -> None:
    """review-tests F9: the pure pin above held before Notes was removed too;
    this one walks the mounted wizard, forward and back, with a key typed
    part-way (the old trigger for a mid-run step-count change)."""
    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import SetupWizardContainer

    app = _hosted_wizard()
    seen: list[tuple[int, int]] = []
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        container = app.screen.query_one(SetupWizardContainer)
        container.select_track(wizard_state.TRACK_FULL)
        order = [container._step_index_for_id(step) for step in container.active_ids]
        for position, index in enumerate(order):
            if position == 2:
                container.note_key_entered()
            container.show_step(index)
            seen.append(await _progress_at(container, position))
        for position in reversed(range(len(order) - 1)):
            container.show_step(order[position])
            seen.append(await _progress_at(container, position))
        assert "notes" not in container.active_ids
    assert set(seen) == {(10, 10)}


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_a_resumed_draft_puts_search_and_appearance_choices_back(
    monkeypatch,
) -> None:
    """review-tests F4: the rewritten restore paths, without the app factory.

    The live-contract tests that cover them stop in the app factory on this
    machine, so this drives the real resume path (``_restore_resume_target``
    on mount) and then checks that Next would save what was restored.
    """
    from unittest.mock import AsyncMock

    from textual.widgets import Checkbox, RadioSet

    from tldw_chatbook.UI.Wizards import first_run_rag_state as rag_state
    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import SetupWizardContainer

    monkeypatch.setattr(
        "tldw_chatbook.Utils.optional_deps.embeddings_rag_deps_installed",
        lambda: True,
    )
    monkeypatch.setattr(
        rag_state,
        "resolve_active_embedding_model",
        lambda: rag_state.EmbeddingModelFacts(name="all-MiniLM-L6-v2", cloud=False),
    )
    draft = wizard_state.SetupDraft(
        version=wizard_state.SETUP_DRAFT_VERSION,
        track=wizard_state.TRACK_FULL,
        active_step_id=wizard_state.STEP_APPEARANCE,
        values={
            wizard_state.STEP_WELCOME: {"track": wizard_state.TRACK_FULL},
            wizard_state.STEP_RAG: {"auto_retrieve": True},
            wizard_state.STEP_APPEARANCE: {
                "startup": "short",
                "reduce_motion": True,
                "ascii_glyphs": True,
            },
        },
    )
    app = _hosted_wizard(resume_draft=draft)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.5)
        container = app.screen.query_one(SetupWizardContainer)
        rag = container.steps[container._step_index_for_id(wizard_state.STEP_RAG)]
        look = container.steps[
            container._step_index_for_id(wizard_state.STEP_APPEARANCE)
        ]
        rag_pressed = rag.query_one("#setup-rag-auto-retrieve", RadioSet).pressed_button
        startup = look.query_one("#setup-startup-choice", RadioSet).pressed_button
        reduce_box = look.query_one("#setup-appearance-reduce-motion", Checkbox).value
        ascii_box = look.query_one("#setup-appearance-ascii", Checkbox).value
        container.commit_config = AsyncMock(return_value=True)
        rag_ok, _ = await rag.commit()
        look_ok, _ = await look.commit()
        writes = [call.args[0] for call in container.commit_config.await_args_list]
    assert container.steps[container.current_step] is look
    assert rag_pressed is not None
    assert rag_pressed.id == "setup-rag-auto-retrieve-automatic"
    assert startup is not None and startup.id == "setup-startup-short"
    assert (reduce_box, ascii_box) == (True, True)
    assert rag_ok and look_ok
    assert {"chat_defaults": {"rag_auto_retrieve_on_send": True}} in writes
    appearance = next(write for write in writes if "splash_screen" in write)
    assert appearance["splash_screen"] == {"enabled": True, "duration": 2.0}
    assert appearance["appearance"] == {"reduce_motion": True, "ascii_glyphs": True}
