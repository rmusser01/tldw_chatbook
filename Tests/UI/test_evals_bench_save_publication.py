"""Queued word-bench Save publication must preserve later child input (TASK-37031)."""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest
from textual import on
from textual.events import Key
from textual.message import Message
from textual.widgets import Button, Input

from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.test_evals_screen import EvalsHarness, _FakeAppInstance
from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.word_bench.models import BenchConfig
from tldw_chatbook.Evals.word_bench.storage import load_bench, save_bench
from tldw_chatbook.UI.Evals.bench_editor import BenchEditor
from tldw_chatbook.UI.Evals.library_rail import LibraryRail
from tldw_chatbook.UI.Screens.evals_screen import EvalsScreen


async def _until(predicate):
    async with asyncio.timeout(8):
        while not predicate():
            await asyncio.sleep(0.01)


class _Hold(Message):
    pass


class _GatedScreen(EvalsScreen):
    def __init__(self, instance):
        super().__init__(instance)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.publication_consumed = asyncio.Event()
        self.hold_rail = False
        self.rail_entered = asyncio.Event()
        self.rail_release = asyncio.Event()

    @on(_Hold)
    async def hold(self, event):
        event.stop()
        self.entered.set()
        await self.release.wait()

    @on(BenchEditor.Saved)
    def observe_publication(self):
        # Observe native dispatch without replacing the production handler.
        self.publication_consumed.set()

    async def _replace_region(self, region_id, content):
        await super()._replace_region(region_id, content)
        if region_id == "#lab-rail" and self.hold_rail:
            self.rail_entered.set()
            await self.rail_release.wait()


class _Host(ConsolidatedCSSApp):
    CSS_PATH = EvalsHarness.CSS_PATH

    def __init__(self, instance):
        super().__init__()
        self.app_instance = instance

    async def on_mount(self):
        await self.push_screen(_GatedScreen(self.app_instance))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    ["ordinary", "later-input", "during-refresh", "departure", "departure-pending"],
)
@private_profile_test
async def test_save_publication_retains_later_editing_context(request, case):
    """Replacing the detail after Save loses a later draft or departed selection."""
    root = Path(__file__).resolve().parents[2]
    for owner in (EvalsScreen, BenchEditor, EvalsDB, load_bench):
        assert Path(inspect.getsourcefile(owner)).resolve().is_relative_to(root)
    db = EvalsDB(db_path=":memory:", client_id="save-publication")
    try:
        dataset_id = db.create_dataset(
            name="snippets",
            format="custom",
            source_path="inline:snippets",
            metadata={"sample_count": 1},
        )
        target_id = db.create_model(
            name="target", provider="llama_cpp", model_id="test"
        )
        bench_id = save_bench(
            db,
            BenchConfig(
                name="Original",
                dataset_id=dataset_id,
                target_ids=(target_id,),
                prompt_mode="raw",
                top_k=10,
                probes=(" x",),
            ),
        )
        app = _Host(_FakeAppInstance(db))
        async with app.run_test(size=(160, 45)) as pilot:
            screen = app.screen
            screen.select(kind="bench", id=bench_id)
            await app.workers.wait_for_complete()
            await pilot.pause()
            editor = screen.query_one(BenchEditor)
            field = editor.query_one("#evals-bench-name", Input)
            field.focus()
            submitted = "  Saved  " if case == "ordinary" else "Saved"
            await pilot.press("home", "shift+end", "backspace", *submitted)
            assert field.value == submitted
            assert field.has_focus
            later_input = case in {"later-input", "during-refresh"}
            screen.hold_rail = case == "during-refresh"
            screen.post_message(_Hold())
            selection_lock_held = False
            try:
                await _until(screen.entered.is_set)
                editor.query_one("#evals-bench-save", Button).press()
                await _until(lambda: load_bench(db, bench_id).name == "Saved")
                if case == "during-refresh":
                    screen.release.set()
                    await _until(screen.rail_entered.is_set)
                if later_input:
                    for char in "-new":
                        field.post_message(Key(char, char))
                    await _until(lambda: field.value == "Saved-new")
                    await _until(lambda: field.cursor_position == 9)
                    assert field.selection.start == 9
                    assert field.selection.end == 9
                    assert field.has_focus and editor.is_dirty()
                    before_selection = field.selection
                elif case.startswith("departure"):
                    if case == "departure-pending":
                        await screen._selection_swap_lock.acquire()
                        selection_lock_held = True
                    screen.select(kind="dataset", id=dataset_id)
                    assert screen._selection.kind == "dataset"
            finally:
                screen.release.set()
                screen.rail_release.set()
                try:
                    await _until(screen.publication_consumed.is_set)
                finally:
                    if selection_lock_held:
                        screen._selection_swap_lock.release()
                await app.workers.wait_for_complete()
                await pilot.pause()
            assert load_bench(db, bench_id).name == "Saved"
            labels = [
                str(button.label)
                for button in screen.query_one(LibraryRail).query(Button)
            ]
            assert any("Saved" in label for label in labels)
            if case.startswith("departure"):
                assert screen._selection.kind == "dataset"
                assert screen._selection.id == dataset_id
                assert not screen.query(BenchEditor)
                return
            assert "Saved" in str(
                screen.query_one("#evals-primary-action", Button).label
            )
            current = screen.query_one("#evals-bench-name", Input)
            assert current.value == (
                "Saved-new" if later_input else "Saved"
            ), "queued Save publication replaced newer accepted child input"
            if later_input:
                assert current is field
                assert app.focused is field
                assert field.cursor_position == 9
                assert field.selection == before_selection
                assert editor.is_dirty()
                current_editor = screen.query_one(BenchEditor)
                current_editor.query_one("#evals-bench-save", Button).press()
                await _until(lambda: load_bench(db, bench_id).name == "Saved-new")
                await app.workers.wait_for_complete()
                await pilot.pause()
                assert load_bench(db, bench_id).name == "Saved-new"
            else:
                assert current is not field
                assert not screen.query_one(BenchEditor).is_dirty()
    finally:
        db.close()
