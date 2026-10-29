"""Character Save publication must preserve later native input (TASK-37266)."""

from __future__ import annotations

import asyncio
import inspect
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual import on
from textual.events import Key
from textual.message import Message
from textual.widgets import Button, Input

from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.test_evals_screen import EvalsHarness, _FakeAppInstance
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.character_probe.models import (
    CharacterProbeConfig,
    Probe,
    ProbeSet,
)
from tldw_chatbook.Evals.character_probe.storage import (
    load_character_bench,
    save_character_bench,
    save_probe_set,
)
from tldw_chatbook.UI.Evals.card_picker import CardPicker, CardRow
from tldw_chatbook.UI.Evals.character_bench_editor import CharacterBenchEditor
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
        self.publication_count = 0
        self.hold_rail = False
        self.rail_entered = asyncio.Event()
        self.rail_release = asyncio.Event()

    @on(_Hold)
    async def hold(self, event):
        event.stop()
        self.entered.set()
        await self.release.wait()

    @on(CharacterBenchEditor.Saved)
    def observe_publication(self):
        # Observe native dispatch; the production handler still runs.
        self.publication_count += 1
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


@asynccontextmanager
async def _mounted_bench():
    root = Path(__file__).resolve().parents[2]
    for owner in (EvalsScreen, CharacterBenchEditor, EvalsDB, load_character_bench):
        assert Path(inspect.getsourcefile(owner)).resolve().is_relative_to(root)
    db = EvalsDB(db_path=":memory:", client_id="character-save-publication")
    cards = CharactersRAGDB(":memory:", "character-save-publication")
    try:
        first_card = cards.add_character_card({"name": "Vex"})
        second_card = cards.add_character_card({"name": "Marlow"})
        dataset_id = save_probe_set(
            db, "probe set", ProbeSet(probes=(Probe(turns=("Hello there.",)),))
        )
        target_id = db.create_model(
            name="target", provider="llama_cpp", model_id="test"
        )
        bench_id = save_character_bench(
            db,
            CharacterProbeConfig(
                name="Original",
                description="Submitted description",
                probe_set_id=dataset_id,
                character_ids=(first_card,),
                target_ids=(target_id,),
                seed=7,
                temperature=0.7,
                max_tokens=256,
            ),
        )
        app = _Host(_FakeAppInstance(db, chachanotes_db=cards))
        async with app.run_test(size=(160, 45)) as pilot:
            screen = app.screen
            screen.select(kind="character_bench", id=bench_id)
            await app.workers.wait_for_complete()
            await pilot.pause()
            try:
                yield SimpleNamespace(
                    app=app,
                    pilot=pilot,
                    screen=screen,
                    editor=screen.query_one(CharacterBenchEditor),
                    db=db,
                    bench_id=bench_id,
                    dataset_id=dataset_id,
                    first_card=first_card,
                    second_card=second_card,
                )
            finally:
                # Every hold is released before the App's own shutdown.
                screen.release.set()
                screen.rail_release.set()
                await app.workers.wait_for_complete()
                await pilot.pause()
    finally:
        db.close()
        cards.close()


async def _submit_held_save(
    ctx, *, during_refresh=False, ordinary=False, focus_selector=None
):
    field = ctx.editor.query_one("#evals-cb-name", Input)
    field.focus()
    submitted = "  Saved  " if ordinary else "Saved"
    await ctx.pilot.press("home", "shift+end", "backspace", *submitted)
    assert field.value == submitted and field.has_focus
    if focus_selector:
        field = ctx.editor.query_one(focus_selector, Input)
        field.focus()
        await ctx.pilot.press("home", "shift+end")
        assert field.has_focus
    ctx.screen.hold_rail = during_refresh
    ctx.screen.post_message(_Hold())
    await _until(ctx.screen.entered.is_set)
    ctx.editor.query_one("#evals-cb-save", Button).press()
    await _until(lambda: load_character_bench(ctx.db, ctx.bench_id).name == "Saved")
    if during_refresh:
        ctx.screen.release.set()
        await _until(ctx.screen.rail_entered.is_set)


async def _publish(ctx, *, count=1):
    ctx.screen.release.set()
    ctx.screen.rail_release.set()
    await _until(lambda: ctx.screen.publication_count >= count)
    await ctx.app.workers.wait_for_complete()
    await ctx.pilot.pause()


def _assert_submitted_metadata(ctx):
    stored = load_character_bench(ctx.db, ctx.bench_id)
    assert stored.name == "Saved"
    assert stored.description == "Submitted description"
    assert stored.character_ids == (ctx.first_card,)
    assert (
        stored.samples_per_cell,
        stored.seed,
        stored.temperature,
        stored.max_tokens,
    ) == (1, 7, 0.7, 256)
    labels = [str(row.label) for row in ctx.screen.query_one(LibraryRail).query(Button)]
    assert any("Saved" in label for label in labels)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["before-publication", "during-refresh"])
@pytest.mark.parametrize(
    ("field_id", "later_value"),
    [
        ("name", "Saved-new"),
        ("description", "Later description"),
        ("samples", "01"),
        ("seed", "07"),
        ("temperature", "0.70"),
        ("max-tokens", "0256"),
        ("cards", None),
        ("card-search", "Vex"),
    ],
)
@private_profile_test
async def test_character_save_preserves_later_draft(
    request, phase, field_id, later_value
):
    """A name-only/normalized snapshot or a pre-await guard loses these edits."""
    async with _mounted_bench() as ctx:
        selector = (
            "#evals-card-search"
            if field_id in {"cards", "card-search"}
            else f"#evals-cb-{field_id}"
        )
        await _submit_held_save(
            ctx, during_refresh=phase == "during-refresh", focus_selector=selector
        )
        picker = ctx.editor.query_one(CardPicker)
        field = ctx.editor.query_one(selector, Input)
        if field_id == "cards":
            row = next(
                row for row in picker.query(CardRow) if row.card_id == ctx.second_card
            )
            row.press()
            await _until(
                lambda: set(picker.selected_ids()) == {ctx.first_card, ctx.second_card}
            )
        else:
            # The screen pump is held, while the focused child's native pump
            # continues accepting edits. No Input.value assignment is used.
            for char in later_value:
                field.post_message(Key(char, char))
            await _until(
                lambda: field.value == later_value
                and field.cursor_position == len(later_value)
            )
        before = (field.value, field.cursor_position, field.selection)
        assert field.has_focus
        await _publish(ctx)
        _assert_submitted_metadata(ctx)
        assert (
            str(ctx.screen.query_one("#evals-primary-action", Button).label)
            == "Run Saved"
        )
        current = ctx.screen.query_one(selector, Input)
        assert (
            current is field
        ), "queued character Save replaced newer accepted form state"
        assert ctx.screen.query_one(CharacterBenchEditor) is ctx.editor
        assert ctx.app.focused is field
        assert (field.value, field.cursor_position, field.selection) == before
        expected_cards = (
            {ctx.first_card, ctx.second_card}
            if field_id == "cards"
            else {ctx.first_card}
        )
        assert set(picker.selected_ids()) == expected_cards
        # Numeric spelling and search are local context, not semantic dirtiness.
        assert ctx.editor.is_dirty() == (field_id in {"name", "description", "cards"})

        ctx.editor.query_one("#evals-cb-save", Button).press()
        await _until(
            lambda: bool(ctx.screen.query(CharacterBenchEditor))
            and ctx.screen.query_one(CharacterBenchEditor) is not ctx.editor
        )
        await ctx.app.workers.wait_for_complete()
        await ctx.pilot.pause()
        stored = load_character_bench(ctx.db, ctx.bench_id)
        assert stored.name == ("Saved-new" if field_id == "name" else "Saved")
        assert stored.description == (
            "Later description"
            if field_id == "description"
            else "Submitted description"
        )
        assert set(stored.character_ids) == expected_cards
        assert (
            stored.samples_per_cell,
            stored.seed,
            stored.temperature,
            stored.max_tokens,
        ) == (1, 7, 0.7, 256)
        assert not ctx.screen.query_one(CharacterBenchEditor).is_dirty()


@pytest.mark.asyncio
@pytest.mark.parametrize("pending", [False, True], ids=["completed", "lock-pending"])
@private_profile_test
async def test_character_save_preserves_newer_selection(request, pending):
    """A Save must neither reselect its bench nor supersede a pending region swap."""
    async with _mounted_bench() as ctx:
        await _submit_held_save(ctx)
        if pending:
            await ctx.screen._selection_swap_lock.acquire()
        try:
            ctx.screen.select(kind="dataset", id=ctx.dataset_id)
            if not pending:
                await ctx.app.workers.wait_for_complete()
                assert ctx.screen.query("#evals-probeset-detail-name")
            ctx.screen.release.set()
            await _until(ctx.screen.publication_consumed.is_set)
        finally:
            if pending:
                ctx.screen._selection_swap_lock.release()
        await _publish(ctx)
        _assert_submitted_metadata(ctx)
        assert (
            ctx.screen._selection.kind == "dataset"
        ), "older Save displaced newer shell selection"
        assert ctx.screen._selection.id == ctx.dataset_id
        assert ctx.screen.query("#evals-probeset-detail-name")
        assert not ctx.screen.query(CharacterBenchEditor)


@pytest.mark.asyncio
@private_profile_test
async def test_character_save_reloads_canonical_values(request):
    """Ordinary Save still refreshes the name, Catalog, Inspector and clean form."""
    async with _mounted_bench() as ctx:
        field = ctx.editor.query_one("#evals-cb-name", Input)
        await _submit_held_save(ctx, ordinary=True)
        await _publish(ctx)
        _assert_submitted_metadata(ctx)
        current = ctx.screen.query_one("#evals-cb-name", Input)
        assert current is not field
        assert current.value == "Saved"
        assert (
            str(ctx.screen.query_one("#evals-primary-action", Button).label)
            == "Run Saved"
        )
        assert not ctx.screen.query_one(CharacterBenchEditor).is_dirty()

        # The external Screen watch belongs to the retired editor. Textual
        # prunes a closing owner on the next focus change without an unwatch hack.
        retired_revision = ctx.editor._publication_revision
        assert ctx.editor._closing
        ctx.screen.set_focus(None)
        current.focus()
        await _until(lambda: current.has_focus)
        assert ctx.editor._publication_revision == retired_revision


@pytest.mark.asyncio
@private_profile_test
async def test_character_older_save_preserves_a_later_returned_draft(request):
    """An old raw-equal Save must not replace a draft made after another Save."""
    async with _mounted_bench() as ctx:
        await _submit_held_save(ctx, focus_selector="#evals-cb-name")
        field = ctx.editor.query_one("#evals-cb-name", Input)
        for char in "Second":
            field.post_message(Key(char, char))
        await _until(lambda: field.value == "Second")
        ctx.editor.query_one("#evals-cb-save", Button).press()
        await _until(
            lambda: load_character_bench(ctx.db, ctx.bench_id).name == "Second"
        )

        # Native focus delivery selects the existing text on the child pump;
        # the screen's held publication does not consume these new keys.
        ctx.screen.set_focus(None)
        field.focus()
        await _until(lambda: (field.selection.start, field.selection.end) == (0, 6))
        for char in "Saved":
            field.post_message(Key(char, char))
        await _until(lambda: field.value == "Saved" and field.cursor_position == 5)
        before = (field.value, field.cursor_position, field.selection)
        assert field.has_focus and ctx.editor.is_dirty()
        await _publish(ctx)

        assert load_character_bench(ctx.db, ctx.bench_id).name == "Second"
        labels = [
            str(row.label) for row in ctx.screen.query_one(LibraryRail).query(Button)
        ]
        assert any("Second" in label for label in labels)
        assert (
            str(ctx.screen.query_one("#evals-primary-action", Button).label)
            == "Run Second"
        )
        assert (
            ctx.screen.query_one("#evals-cb-name", Input) is field
        ), "older Save reloaded a later draft that returned to its raw text"
        assert ctx.app.focused is field
        assert (field.value, field.cursor_position, field.selection) == before
        assert ctx.editor.is_dirty()

        ctx.editor.query_one("#evals-cb-save", Button).press()
        await _until(lambda: load_character_bench(ctx.db, ctx.bench_id).name == "Saved")
        await ctx.app.workers.wait_for_complete()
        await ctx.pilot.pause()
        assert ctx.screen.query_one("#evals-cb-name", Input).value == "Saved"
        assert not ctx.screen.query_one(CharacterBenchEditor).is_dirty()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["before-publication", "during-refresh"])
@pytest.mark.parametrize(
    "case", ["equal-valued-save", "single-save-roundtrip", "focus-only"]
)
@private_profile_test
async def test_character_save_retains_returned_text_and_context(request, phase, case):
    """Equal final values do not make later native editing context disposable."""
    async with _mounted_bench() as ctx:
        await _submit_held_save(
            ctx,
            during_refresh=phase == "during-refresh",
            focus_selector="#evals-cb-samples",
        )
        field = ctx.editor.query_one("#evals-cb-samples", Input)
        first_baseline = ctx.editor._loaded_config
        if case == "focus-only":
            focused = ctx.editor.query_one("#evals-cb-revert", Button)
            focused.focus()
            await _until(lambda: focused.has_focus)
        else:
            for char in "01":
                field.post_message(Key(char, char))
            await _until(lambda: field.value == "01" and field.cursor_position == 2)
            if case == "equal-valued-save":
                ctx.editor.query_one("#evals-cb-save", Button).press()
                await _until(lambda: ctx.editor._loaded_config is not first_baseline)
                # Witness the review's distinct successful, value-equal Save B.
                assert ctx.editor._loaded_config == first_baseline
                assert load_character_bench(ctx.db, ctx.bench_id).samples_per_cell == 1
            ctx.screen.set_focus(None)
            field.focus()
            await _until(lambda: (field.selection.start, field.selection.end) == (0, 2))
            field.post_message(Key("1", "1"))
            await _until(lambda: field.value == "1" and field.cursor_position == 1)
            assert (field.selection.start, field.selection.end) == (1, 1)
            focused = field
        before = (field.value, field.cursor_position, field.selection)
        assert not ctx.editor.is_dirty()
        assert focused.has_focus
        await _publish(ctx, count=2 if case == "equal-valued-save" else 1)
        _assert_submitted_metadata(ctx)
        assert (
            str(ctx.screen.query_one("#evals-primary-action", Button).label)
            == "Run Saved"
        )
        assert (
            ctx.screen.query_one(CharacterBenchEditor) is ctx.editor
        ), "Save publication replaced later native roundtrip or focus context"
        assert ctx.screen.query_one("#evals-cb-samples", Input) is field
        assert (field.value, field.cursor_position, field.selection) == before
        assert ctx.app.focused is focused
        assert not ctx.editor.is_dirty()

        # An explicit Save of this unchanged semantic value still reloads cleanly.
        ctx.editor.query_one("#evals-cb-save", Button).press()
        await _until(
            lambda: bool(ctx.screen.query(CharacterBenchEditor))
            and ctx.screen.query_one(CharacterBenchEditor) is not ctx.editor
        )
        await ctx.app.workers.wait_for_complete()
        await ctx.pilot.pause()
        assert ctx.screen.query_one("#evals-cb-samples", Input).value == "1"
        assert not ctx.screen.query_one(CharacterBenchEditor).is_dirty()
