from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from tldw_chatbook.Actor_Packs.importer import ActorPackImportService
from tldw_chatbook.Actor_Packs.repository import ActorPackRepository
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Widgets.Persona_Widgets.actor_pack_import_review import (
    ActorPackImportReviewDialog,
)


FIXTURE = (
    Path(__file__).parents[1]
    / "Actor_Packs"
    / "fixtures"
    / "export-golden"
    / "minimal-character.tldw-actor-pack"
)


class _ReviewApp(App[None]):
    def compose(self) -> ComposeResult:
        yield Button("Open", id="open")


@pytest.fixture
def review_material(tmp_path: Path):
    db = CharactersRAGDB(tmp_path / "review.db", client_id="review-ui")
    importer = ActorPackImportService(
        ActorPackRepository(db),
        staging_root=tmp_path / "staging",
        profile_root=tmp_path,
    )
    review = importer.inspect_archive(FIXTURE.resolve())
    preview = importer.read_portrait_preview(review)
    yield review, preview
    importer.cleanup_review(review)
    db.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (140, 44)])
async def test_review_is_scrollable_visible_and_action_specific(
    review_material, size: tuple[int, int]
) -> None:
    review, preview = review_material
    app = _ReviewApp()
    async with app.run_test(size=size) as pilot:
        app.push_screen(ActorPackImportReviewDialog(review, preview))
        await pilot.pause()

        dialog = app.screen.query_one("#actor-pack-import-review")
        assert dialog.region.width > 0 and dialog.region.height > 0
        assert dialog.region.right <= size[0]
        assert dialog.region.bottom <= size[1]
        assert app.screen.query_one("#actor-pack-import-create-new", Button).display
        assert app.screen.query_one("#actor-pack-import-create-copy", Button).display
        assert not app.screen.query_one(
            "#actor-pack-import-update-existing", Button
        ).display
        for selector in (
            "#actor-pack-import-cancel",
            "#actor-pack-import-create-new",
            "#actor-pack-import-create-copy",
        ):
            button = app.screen.query_one(selector, Button)
            assert button.region.width > 0 and button.region.height > 0
            assert button.region.right <= size[0]
            assert button.region.bottom <= size[1]
        assert "Not included" in str(
            app.screen.query_one("#actor-pack-import-visuals", Static).renderable
        )
        assert "No warnings" in str(
            app.screen.query_one("#actor-pack-import-warnings", Static).renderable
        )


@pytest.mark.asyncio
async def test_untrusted_actor_text_is_rendered_as_plain_text(review_material) -> None:
    review, preview = review_material
    unsafe = dataclasses.replace(
        review,
        actor_fields=(("name", "[bold]Danger[/bold]\x1b[31m"),),
        warnings=("[link=https://example.invalid]Do not open[/link]",),
    )
    app = _ReviewApp()
    async with app.run_test() as pilot:
        app.push_screen(ActorPackImportReviewDialog(unsafe, preview))
        await pilot.pause()

        details = app.screen.query_one("#actor-pack-import-actor-fields", Static)
        assert "[bold]Danger[/bold]" in str(details.renderable)
        assert "\\u001b" in str(details.renderable)
        warnings = app.screen.query_one("#actor-pack-import-warnings", Static)
        assert "[link=https://example.invalid]" in str(warnings.renderable)


@pytest.mark.asyncio
async def test_cancel_restores_opener_focus(review_material) -> None:
    review, preview = review_material
    app = _ReviewApp()
    async with app.run_test() as pilot:
        opener = app.query_one("#open", Button)
        opener.focus()
        app.push_screen(ActorPackImportReviewDialog(review, preview))
        await pilot.pause()

        await pilot.click("#actor-pack-import-cancel")
        await pilot.pause()

        assert app.focused is opener


@pytest.mark.asyncio
async def test_compact_terminal_keyboard_confirms_focused_safe_action(
    review_material,
) -> None:
    review, preview = review_material
    app = _ReviewApp()
    result: list[str | None] = []
    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(
            ActorPackImportReviewDialog(review, preview), callback=result.append
        )
        await pilot.pause()

        assert app.focused is app.screen.query_one(
            "#actor-pack-import-create-new", Button
        )
        await pilot.press("enter")
        await pilot.pause()

        assert result == ["create_new"]
