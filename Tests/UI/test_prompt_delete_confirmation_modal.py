"""Contract tests for the reusable Library Prompt deletion confirmation modal."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from tldw_chatbook.Widgets.Library.prompt_delete_confirmation_modal import (
    PromptDeleteConfirmationModal,
    PromptDeleteDecision,
    PromptDeleteItem,
    PromptDeleteRequest,
)


class ModalHarness(App[None]):
    """Minimal host which captures the modal's typed dismissal result."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[PromptDeleteDecision] = []

    def compose(self) -> ComposeResult:
        yield Static("Library")

    def show(self, request: PromptDeleteRequest) -> None:
        self.push_screen(
            PromptDeleteConfirmationModal(request), callback=self.results.append
        )


def _item(name: str, artifact_type: str = "prompt") -> PromptDeleteItem:
    return PromptDeleteItem(name=name, artifact_type=artifact_type)


def test_request_and_decision_are_immutable_and_modal_keeps_frozen_request() -> None:
    request = PromptDeleteRequest(
        items=(_item("Draft"),), fingerprint="editor:42", dirty=True
    )
    modal = PromptDeleteConfirmationModal(request)

    assert modal.request is request
    assert isinstance(request.items, tuple)
    with pytest.raises(FrozenInstanceError):
        request.fingerprint = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        PromptDeleteDecision(True, "editor:42").confirmed = False  # type: ignore[misc]


@pytest.mark.parametrize(
    ("name", "artifact_type", "error"),
    [
        (1, "prompt", TypeError),
        ("Future artifact", "future", ValueError),
    ],
)
def test_delete_item_fails_closed_for_malformed_or_unknown_artifact_types(
    name: object, artifact_type: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        PromptDeleteItem(name=name, artifact_type=artifact_type)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("items", "fingerprint", "dirty", "preview_limit", "error"),
    [
        ([_item("Mutable")], None, False, 3, TypeError),
        (("not a delete item",), None, False, 3, TypeError),
        ((_item("Draft"),), 42, False, 3, TypeError),
        ((_item("Draft"),), None, 1, 3, TypeError),
        ((_item("Draft"),), None, False, True, TypeError),
        ((_item("Draft"),), None, False, 1.5, TypeError),
        ((_item("Draft"),), None, False, 0, ValueError),
    ],
)
def test_delete_request_rejects_malformed_public_data(
    items: object,
    fingerprint: object,
    dirty: object,
    preview_limit: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        PromptDeleteRequest(  # type: ignore[arg-type]
            items=items,
            fingerprint=fingerprint,
            dirty=dirty,
            preview_limit=preview_limit,
        )


@pytest.mark.parametrize(
    ("confirmed", "fingerprint"),
    [(1, None), (False, 42)],
)
def test_delete_decision_rejects_malformed_public_data(
    confirmed: object, fingerprint: object
) -> None:
    with pytest.raises(TypeError):
        PromptDeleteDecision(  # type: ignore[arg-type]
            confirmed=confirmed, fingerprint=fingerprint
        )


@pytest.mark.asyncio
async def test_single_prompt_copy_names_the_saved_prompt() -> None:
    app = ModalHarness()

    async with app.run_test(size=(90, 30)) as pilot:
        app.show(PromptDeleteRequest(items=(_item("Release notes"),)))
        await pilot.pause()
        modal = app.screen
        copy = str(modal.query_one("#prompt-delete-copy", Static).renderable)

        assert "Delete Prompt" in str(
            modal.query_one("#prompt-delete-title", Static).renderable
        )
        assert "Release notes" in copy
        assert "saved Prompt" in copy
        assert "unsaved working copy" not in copy
        assert "Undo" in copy
        assert "cannot be undone" not in copy.lower()


@pytest.mark.asyncio
async def test_single_recipe_copy_names_the_saved_recipe() -> None:
    app = ModalHarness()

    async with app.run_test(size=(90, 30)) as pilot:
        app.show(PromptDeleteRequest(items=(_item("Morning brief", "recipe"),)))
        await pilot.pause()
        modal = app.screen
        copy = str(modal.query_one("#prompt-delete-copy", Static).renderable)

        assert "Delete Recipe" in str(
            modal.query_one("#prompt-delete-title", Static).renderable
        )
        assert "Morning brief" in copy
        assert "saved Recipe" in copy


@pytest.mark.asyncio
async def test_dirty_single_delete_warns_about_saved_artifact_and_unsaved_working_copy() -> None:
    app = ModalHarness()

    async with app.run_test(size=(90, 30)) as pilot:
        app.show(PromptDeleteRequest(items=(_item("Draft"),), dirty=True))
        await pilot.pause()
        copy = str(app.screen.query_one("#prompt-delete-copy", Static).renderable)

        assert "saved Prompt" in copy
        assert "unsaved working copy" in copy
        assert "discarded" in copy
        assert "Undo" in copy
        assert "cannot be undone" not in copy.lower()


@pytest.mark.asyncio
async def test_bulk_copy_counts_types_and_bounds_name_preview() -> None:
    app = ModalHarness()

    async with app.run_test(size=(90, 30)) as pilot:
        app.show(
            PromptDeleteRequest(
                items=(
                    _item("One"),
                    _item("Two", "recipe"),
                    _item("Three"),
                    _item("Four"),
                    _item("Five", "recipe"),
                ),
                fingerprint="selection:scope",
                preview_limit=3,
            )
        )
        await pilot.pause()
        modal = app.screen
        copy = str(modal.query_one("#prompt-delete-copy", Static).renderable)
        preview = str(modal.query_one("#prompt-delete-preview", Static).renderable)

        assert "3 Prompts" in copy
        assert "2 Recipes" in copy
        assert "Undo" in copy
        assert "cannot be undone" not in copy.lower()
        assert "One" in preview
        assert "Two" in preview
        assert "Three" in preview
        assert "Four" not in preview
        assert "Five" not in preview
        assert "and 2 more" in preview


@pytest.mark.asyncio
async def test_cancel_dismisses_typed_negative_decision() -> None:
    app = ModalHarness()

    async with app.run_test(size=(90, 30)) as pilot:
        app.show(PromptDeleteRequest(items=(_item("Draft"),), fingerprint="editor:42"))
        await pilot.pause()
        await pilot.click("#prompt-delete-cancel")
        await pilot.pause()

    assert app.results == [PromptDeleteDecision(confirmed=False, fingerprint="editor:42")]


@pytest.mark.asyncio
async def test_confirm_dismisses_typed_positive_decision() -> None:
    app = ModalHarness()

    async with app.run_test(size=(90, 30)) as pilot:
        app.show(PromptDeleteRequest(items=(_item("Draft"),), fingerprint="editor:42"))
        await pilot.pause()
        await pilot.click("#prompt-delete-confirm")
        await pilot.pause()

    assert app.results == [PromptDeleteDecision(confirmed=True, fingerprint="editor:42")]


@pytest.mark.asyncio
async def test_markup_looking_names_render_literally() -> None:
    app = ModalHarness()
    name = "[bold magenta]not markup[/bold magenta]"

    async with app.run_test(size=(90, 30)) as pilot:
        app.show(PromptDeleteRequest(items=(_item(name),)))
        await pilot.pause()
        preview = app.screen.query_one("#prompt-delete-preview", Static)

        assert str(preview.renderable) == name
        assert preview._render_markup is False


@pytest.mark.asyncio
async def test_long_multiline_preview_keeps_delete_actions_visible_at_80_by_24() -> None:
    app = ModalHarness()
    long_name = ("very long artifact name " * 12 + "\n") * 5

    async with app.run_test(size=(80, 24)) as pilot:
        app.show(
            PromptDeleteRequest(
                items=(_item(long_name), _item(long_name), _item(long_name)),
            )
        )
        await pilot.pause()
        modal = app.screen
        preview = str(modal.query_one("#prompt-delete-preview", Static).renderable)
        cancel = modal.query_one("#prompt-delete-cancel", Button)
        confirm = modal.query_one("#prompt-delete-confirm", Button)

        assert cancel.region.y + cancel.region.height <= 24
        assert confirm.region.y + confirm.region.height <= 24
        assert all(len(line) <= 48 for line in preview.splitlines())
