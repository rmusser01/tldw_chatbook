"""Contract tests for the reusable Library Prompt deletion confirmation modal."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

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
