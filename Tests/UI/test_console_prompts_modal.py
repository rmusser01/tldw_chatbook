"""Unified Console Prompt Library Browse/Edit modal contracts."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Any

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Static, TextArea

from tldw_chatbook.Widgets.Console.console_prompts_browse import ConsolePromptsBrowse
from tldw_chatbook.Widgets.Console.console_prompts_modal import ConsolePromptsModal
from tldw_chatbook.Widgets.Console.console_prompts_state import (
    ConsolePromptsState,
    PromptBrowseResult,
)
from tldw_chatbook.Widgets.Prompts.prompt_block_editor import PromptBlockEditor


def _definition(kind: str = "block_prompt") -> dict[str, Any]:
    return {
        "kind": kind,
        "schema_version": 2,
        "lanes": [
            {
                "id": "system",
                "blocks": [
                    {
                        "id": "role",
                        "title": "Role",
                        "syntax": "markdown",
                        "content": "Be exact.",
                    }
                ],
            },
            {
                "id": "user",
                "blocks": [
                    {
                        "id": "goal",
                        "title": "Goal",
                        "syntax": "freeform",
                        "content": "Answer the question.",
                    }
                ],
            },
        ],
    }


def _detail(
    *,
    artifact_type: str = "prompt",
    prompt_format: str = "structured",
    schema_version: int = 2,
    definition: Any | None = None,
    identifier: str = "prompt-1",
    version: int = 4,
) -> dict[str, Any]:
    if definition is None:
        definition = _definition(
            "block_recipe" if artifact_type == "recipe" else "block_prompt"
        )
    return {
        "id": identifier,
        "name": "Precise answer",
        "artifact_type": artifact_type,
        "prompt_format": prompt_format,
        "prompt_schema_version": schema_version,
        "prompt_definition": definition,
        "system_prompt": "# Role\n\nBe exact.",
        "user_prompt": "Answer the question.",
        "version": version,
        "backend": "local",
    }


def _brief(identifier: str = "prompt-1", *, artifact_type: str = "prompt"):
    return {
        "id": identifier,
        "name": "Precise answer",
        "artifact_type": artifact_type,
        "has_system_prompt": True,
        "has_user_prompt": True,
        "updated_at": "2026-08-01T12:00:00Z",
        "backend": "local",
        "version": 4,
    }


class _PromptBackend:
    def __init__(self, *, pages: Mapping[int, Mapping[str, Any]] | None = None) -> None:
        self.pages = dict(pages or {})
        self.list_calls: list[tuple[str, int]] = []
        self.search_calls: list[tuple[str, str]] = []
        self.detail_calls: list[tuple[str, str]] = []
        self.save_calls: list[dict[str, Any]] = []
        self.model_calls = 0
        self.usage_mutations = 0
        self.search_result: Any = []
        self.detail_result: Any = _detail()
        self.save_result: Any = None
        self.list_error: Exception | None = None
        self.search_error: Exception | None = None
        self.detail_error: Exception | None = None
        self.capabilities_result: object = SimpleNamespace(
            structured_kinds=frozenset({(2, "block_prompt"), (2, "block_recipe")}),
            artifact_types=frozenset({"prompt", "recipe"}),
            conditional_update=True,
        )

    async def capabilities(self, source: str) -> object:
        return self.capabilities_result

    async def list_page(self, source: str, page: int) -> Mapping[str, Any]:
        self.list_calls.append((source, page))
        if self.list_error is not None:
            raise self.list_error
        return self.pages.get(
            page,
            {"items": [], "page": page, "total_pages": 1, "total_items": 0},
        )

    async def search(self, source: str, query: str) -> Any:
        self.search_calls.append((source, query))
        if self.search_error is not None:
            raise self.search_error
        return self.search_result

    async def detail(self, source: str, identifier: str) -> Any:
        self.detail_calls.append((source, identifier))
        if self.detail_error is not None:
            raise self.detail_error
        return self.detail_result

    async def save(self, **payload: Any) -> Any:
        self.save_calls.append(payload)
        return payload if self.save_result is None else self.save_result


class _Harness(App):
    def __init__(
        self,
        backend: _PromptBackend,
        *,
        improve_unavailable_reason: str = "",
        configure_provider: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.improve_unavailable_reason = improve_unavailable_reason
        self.configure_provider = configure_provider

    def compose(self) -> ComposeResult:
        yield Input(id="console-native-composer")

    async def on_mount(self) -> None:
        self.query_one("#console-native-composer", Input).focus()
        kwargs: dict[str, Any] = {}
        if self.configure_provider is not None:
            kwargs["configure_provider"] = self.configure_provider
        await self.push_screen(
            ConsolePromptsModal(
                capabilities=self.backend.capabilities,
                list_page=self.backend.list_page,
                search=self.backend.search,
                detail=self.backend.detail,
                save=self.backend.save,
                improve_unavailable_reason=self.improve_unavailable_reason,
                **kwargs,
            )
        )


@pytest.mark.unit
def test_state_owns_navigation_search_identity_and_stale_tokens() -> None:
    state = ConsolePromptsState.initial()
    state = (
        state.with_query("alpha")
        .with_page(3)
        .remember_focus("browse", "console-prompts-search")
    )
    state = state.select(identity="prompt-7", version=12).enter_mode("improve")
    stale_token = state.search_token
    state = state.begin_search().with_source("server")

    assert state.mode == "improve"
    assert state.query == "alpha"
    assert state.page == 1
    assert state.selected_identity == "prompt-7"
    assert state.selected_version == 12
    assert state.focus_for("browse") == "console-prompts-search"
    assert not state.accepts(stale_token, "local")
    assert state.accepts(state.search_token, "server")
    assert state.go_back().mode == "browse"


@pytest.mark.unit
def test_browse_result_is_source_scoped_and_immutable() -> None:
    result = PromptBrowseResult(
        source="local",
        items=(_brief(),),
        page=2,
        total_pages=4,
        total_items=31,
    )

    assert result.source == "local"
    assert result.items[0]["artifact_type"] == "prompt"
    assert result.page == 2
    assert result.total_pages == 4


@pytest.mark.asyncio
async def test_empty_local_library_and_improve_action_are_visible() -> None:
    backend = _PromptBackend()
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.pause()
        assert app.screen.query_one("#console-prompts-improve", Button)
        empty = app.screen.query_one("#console-prompts-browse-status", Static)
        assert "Local Prompt Library is empty" in str(empty.renderable)
        assert "Create or save a Prompt" in str(empty.renderable)

    assert backend.list_calls == [("local", 1)]


@pytest.mark.asyncio
async def test_improve_and_back_preserve_browse_state_and_focus() -> None:
    backend = _PromptBackend(
        pages={
            2: {
                "items": [_brief()],
                "page": 2,
                "total_pages": 2,
                "total_items": 11,
            }
        }
    )
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        modal.state = (
            modal.state.with_query("kept query")
            .with_page(2)
            .select(identity="prompt-1", version=4)
        )
        modal.query_one("#console-prompts-search", Input).focus()
        modal.query_one("#console-prompts-improve", Button).press()
        await pilot.pause()
        assert modal.state.mode == "improve"

        modal.query_one("#console-prompts-back", Button).press()
        await pilot.pause()

        assert modal.state.mode == "browse"
        assert modal.state.query == "kept query"
        assert modal.state.page == 2
        assert modal.state.selected_identity == "prompt-1"
        assert getattr(app.focused, "id", None) == "console-prompts-search"


@pytest.mark.asyncio
async def test_empty_query_paginates_and_nonempty_query_uses_backend_search() -> None:
    backend = _PromptBackend(
        pages={
            1: {
                "items": [_brief()],
                "page": 1,
                "total_pages": 2,
                "total_items": 12,
            },
            2: {
                "items": [_brief("prompt-2")],
                "page": 2,
                "total_pages": 2,
                "total_items": 12,
            },
        }
    )
    backend.search_result = [_brief("searched")]
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.screen.query_one("#console-prompts-next", Button).press()
        await pilot.pause()
        assert backend.list_calls[-1] == ("local", 2)

        search = app.screen.query_one("#console-prompts-search", Input)
        search.focus()
        await pilot.press("a", "l", "p", "h", "a")
        await asyncio.sleep(0.23)
        await pilot.pause()

    assert backend.search_calls == [("local", "alpha")]
    assert backend.list_calls == [("local", 1), ("local", 2)]


@pytest.mark.asyncio
async def test_source_switch_reloads_without_merging_results() -> None:
    backend = _PromptBackend()
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.screen.switch_source("server")
        await pilot.pause()

        assert app.screen.state.source == "server"
        assert backend.list_calls == [("local", 1), ("server", 1)]
        assert app.screen.browse_result.source == "server"


@pytest.mark.asyncio
async def test_late_source_completion_is_rejected() -> None:
    local_started = asyncio.Event()
    release_local = asyncio.Event()

    async def list_page(source: str, page: int) -> Mapping[str, Any]:
        if source == "local":
            local_started.set()
            await release_local.wait()
            return {
                "items": [_brief("late-local")],
                "page": page,
                "total_pages": 1,
                "total_items": 1,
            }
        return {
            "items": [{**_brief("server"), "backend": "server"}],
            "page": page,
            "total_pages": 1,
            "total_items": 1,
        }

    backend = _PromptBackend()
    app = _Harness(backend)
    async with app.run_test(size=(120, 40)) as pilot:
        modal = app.screen
        await pilot.pause()
        modal._list_page = list_page
        late_local = asyncio.create_task(modal.reload_browse())
        await local_started.wait()
        await modal.switch_source("server")
        release_local.set()
        await late_local
        await pilot.pause()
        await pilot.pause()

        assert modal.browse_result.source == "server"
        assert modal.browse_result.items[0]["id"] == "server"


@pytest.mark.asyncio
async def test_no_matches_retry_and_source_unavailable_are_explicit() -> None:
    backend = _PromptBackend()
    backend.search_result = []
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.screen.set_query("missing")
        await pilot.pause()
        status = app.screen.query_one("#console-prompts-browse-status", Static)
        assert "No matches" in str(status.renderable)
        assert "Change the query or switch source" in str(status.renderable)

        backend.search_error = RuntimeError("offline")
        await app.screen.reload_browse()
        await pilot.pause()
        assert app.screen.query_one("#console-prompts-retry", Button).display
        assert "Search failed" in str(status.renderable)

        backend.search_error = None
        app.screen.query_one("#console-prompts-retry", Button).press()
        await pilot.pause()
        assert backend.search_calls[-1] == ("local", "missing")

        backend.list_error = ValueError("Server prompt backend is unavailable.")
        await app.screen.set_query("")
        await pilot.pause()
        assert "source is unavailable" in str(status.renderable)
        assert "Retry or switch source" in str(status.renderable)


@pytest.mark.asyncio
async def test_selected_row_deleted_before_detail_fetch_stays_in_browse() -> None:
    backend = _PromptBackend(
        pages={1: {"items": [_brief()], "page": 1, "total_pages": 1, "total_items": 1}}
    )
    backend.detail_error = KeyError("deleted")
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.screen.query_one("#console-prompts-result-prompt-1", Button).press()
        await pilot.pause()

        assert app.screen.state.mode == "browse"
        assert "changed or deleted" in str(
            app.screen.query_one("#console-prompts-browse-status", Static).renderable
        )
        assert backend.detail_calls == [("local", "prompt-1")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("detail", "expected_mode", "unsaved"),
    [
        (_detail(), "edit", False),
        (_detail(artifact_type="recipe"), "edit", True),
        (
            _detail(
                prompt_format="legacy",
                schema_version=0,
                definition=None,
            ),
            "edit",
            False,
        ),
    ],
)
async def test_supported_prompt_recipe_copy_and_legacy_open_without_side_effects(
    detail: Mapping[str, Any], expected_mode: str, unsaved: bool
) -> None:
    backend = _PromptBackend(
        pages={1: {"items": [_brief()], "page": 1, "total_pages": 1, "total_items": 1}}
    )
    backend.detail_result = detail
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.screen.open_artifact("prompt-1")
        await pilot.pause()

        assert app.screen.state.mode == expected_mode
        assert app.screen.state.working_copy_unsaved is unsaved
        editor = app.screen.query_one(PromptBlockEditor)
        assert editor.state.artifact_type == "prompt"
        assert app.screen.state.selected_identity == "prompt-1"
        assert app.screen.state.selected_version == 4

    assert backend.model_calls == 0
    assert backend.usage_mutations == 0
    assert backend.save_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "detail",
    [
        _detail(schema_version=1, definition={"schema_version": 1, "blocks": []}),
        _detail(
            definition={"schema_version": 2, "definition_kind": "single_text_recipe"}
        ),
        _detail(schema_version=99, definition={"schema_version": 99, "kind": "future"}),
        _detail(definition="{not-json"),
        _detail(artifact_type="prompt", definition=_definition("block_recipe")),
        _detail(artifact_type="alien"),
    ],
)
async def test_foreign_future_malformed_and_mismatched_artifacts_are_guarded(
    detail: Mapping[str, Any],
) -> None:
    backend = _PromptBackend()
    backend.detail_result = detail
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.screen.open_artifact("prompt-1")
        await pilot.pause()

        assert app.screen.state.mode == "edit"
        assert not app.screen.query(PromptBlockEditor)
        assert app.screen.query_one("#console-prompts-compatibility", Static)
        convert = app.screen.query_one("#console-prompts-convert", Button)
        assert convert.label == "Convert and save as new"
        assert convert.disabled is False
        assert app.screen.query_one(
            "#console-prompts-compat-system", TextArea
        ).read_only
        assert app.screen.query_one("#console-prompts-compat-user", TextArea).read_only

    assert backend.model_calls == 0
    assert backend.usage_mutations == 0
    assert backend.save_calls == []


@pytest.mark.asyncio
async def test_guarded_artifact_without_compatibility_text_disables_conversion() -> (
    None
):
    backend = _PromptBackend()
    backend.detail_result = {
        **_detail(schema_version=99, definition={"schema_version": 99}),
        "system_prompt": "",
        "user_prompt": "",
    }
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.screen.open_artifact("prompt-1")
        await pilot.pause()

        button = app.screen.query_one("#console-prompts-convert", Button)
        assert button.disabled is True
        assert "no compatible System or User text" in str(button.tooltip)


@pytest.mark.asyncio
async def test_provider_unavailability_disables_only_improve() -> None:
    backend = _PromptBackend()
    app = _Harness(
        backend,
        improve_unavailable_reason="No active provider or model is configured.",
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        improve = app.screen.query_one("#console-prompts-improve", Button)
        assert improve.disabled is True
        assert "No active provider" in str(improve.tooltip)
        assert app.screen.query_one("#console-prompts-search", Input).disabled is False
        assert "Browse and manual editing remain available" in str(
            app.screen.query_one("#console-prompts-model-status", Static).renderable
        )


@pytest.mark.asyncio
async def test_source_capabilities_gate_only_unsupported_structured_saves() -> None:
    backend = _PromptBackend()
    backend.capabilities_result = SimpleNamespace(
        structured_kinds=frozenset({(2, "block_prompt")}),
        artifact_types=frozenset({"prompt"}),
        conditional_update=False,
    )
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.screen.open_artifact("prompt-1")
        await pilot.pause()

        assert (
            app.screen.query_one("#prompt-editor-save-prompt", Button).disabled is False
        )
        recipe = app.screen.query_one("#prompt-editor-save-recipe", Button)
        update = app.screen.query_one("#prompt-editor-update-original", Button)
        assert recipe.disabled is True
        assert "does not support block_recipe" in str(recipe.tooltip)
        assert update.disabled is True
        assert "conditional updates" in str(update.tooltip)


@pytest.mark.asyncio
async def test_dirty_back_offers_only_keep_editing_or_discard() -> None:
    backend = _PromptBackend()
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.screen.open_artifact("prompt-1")
        await pilot.pause()
        app.screen.mark_dirty()
        await pilot.press("escape")
        await pilot.pause()

        guard = app.screen.query_one("#console-prompts-dirty-guard")
        assert guard.display
        buttons = list(guard.query(Button))
        assert [str(button.label) for button in buttons] == [
            "Keep editing",
            "Discard changes",
        ]
        assert app.screen.state.mode == "edit"


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 40), (80, 24)])
async def test_modal_geometry_keeps_important_actions_in_bounds(
    size: tuple[int, int],
) -> None:
    backend = _PromptBackend()
    app = _Harness(backend)

    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        shell = app.screen.query_one("#console-prompts-modal")
        improve = app.screen.query_one("#console-prompts-improve")
        close = app.screen.query_one("#console-prompts-close")

        assert 0 <= shell.region.x < size[0]
        assert shell.region.x + shell.region.width <= size[0]
        assert 0 <= shell.region.y < size[1]
        assert shell.region.y + shell.region.height <= size[1]
        for widget in (improve, close):
            assert shell.region.contains_region(widget.region)


@pytest.mark.asyncio
async def test_root_escape_dismisses_and_restores_composer_focus() -> None:
    backend = _PromptBackend()
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()

        assert getattr(app.focused, "id", None) == "console-native-composer"


@pytest.mark.asyncio
async def test_late_local_detail_cannot_open_after_switching_to_server() -> None:
    backend = _PromptBackend()
    local_started = asyncio.Event()
    release_local = asyncio.Event()

    async def detail(source: str, identifier: str) -> Mapping[str, Any]:
        if source == "local":
            local_started.set()
            await release_local.wait()
        return {
            **_detail(identifier=identifier),
            "backend": source,
        }

    app = _Harness(backend)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        modal._detail = detail

        late_open = asyncio.create_task(modal.open_artifact("late-local"))
        await local_started.wait()
        await modal.switch_source("server")
        release_local.set()
        await late_open
        await pilot.pause()

        assert modal.state.source == "server"
        assert modal.state.mode == "browse"
        assert modal.state.selected_identity != "late-local"
        assert modal._selected_record is None


@pytest.mark.asyncio
async def test_late_first_detail_cannot_replace_newer_selection() -> None:
    backend = _PromptBackend()
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def detail(source: str, identifier: str) -> Mapping[str, Any]:
        if identifier == "prompt-a":
            first_started.set()
            await release_first.wait()
        return {
            **_detail(identifier=identifier),
            "name": f"Name {identifier}",
            "backend": source,
        }

    app = _Harness(backend)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        modal._detail = detail

        first_open = asyncio.create_task(modal.open_artifact("prompt-a"))
        await first_started.wait()
        await modal.open_artifact("prompt-b")
        release_first.set()
        await first_open
        await pilot.pause()

        assert modal.state.selected_identity == "prompt-b"
        assert modal._selected_record is not None
        assert modal._selected_record["name"] == "Name prompt-b"
        assert modal.state.mode_stack == ("browse", "edit")


@pytest.mark.asyncio
async def test_source_switch_clears_foreign_rows_before_unavailable_result() -> None:
    backend = _PromptBackend(
        pages={
            1: {
                "items": [_brief("local-only")],
                "page": 1,
                "total_pages": 1,
                "total_items": 1,
            }
        }
    )
    server_started = asyncio.Event()
    release_server = asyncio.Event()

    async def list_page(source: str, page: int) -> Mapping[str, Any]:
        if source == "server":
            server_started.set()
            await release_server.wait()
            raise ValueError("Server Prompt source is unavailable.")
        return backend.pages[page]

    app = _Harness(backend)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        modal._list_page = list_page
        assert modal.query("#console-prompts-result-local-only")

        switch = asyncio.create_task(modal.switch_source("server"))
        await server_started.wait()
        foreign_rows_visible_while_loading = bool(
            modal.query("#console-prompts-result-local-only")
        )
        owner_while_loading = modal.browse_result.source
        release_server.set()
        await switch
        await pilot.pause()

        assert foreign_rows_visible_while_loading is False
        assert owner_while_loading == "server"
        assert modal.browse_result.source == "server"
        assert modal.browse_result.items == ()
        assert not modal.query("#console-prompts-result-local-only")
        assert "Server Prompt source is unavailable" in str(
            modal.query_one("#console-prompts-browse-status", Static).renderable
        )


@pytest.mark.asyncio
async def test_recipe_save_as_prompt_becomes_the_guarded_saved_prompt() -> None:
    backend = _PromptBackend()
    backend.detail_result = _detail(artifact_type="recipe")
    backend.save_result = {
        **_detail(
            artifact_type="prompt",
            identifier="local:prompt:new-77",
            version=9,
        ),
        "source_id": "new-77",
        "name": "Saved Prompt",
        "backend": "local",
    }
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        await modal.open_artifact("recipe-1")
        await pilot.pause()
        assert modal.state.working_copy_unsaved is True

        modal.query_one("#prompt-editor-save-prompt", Button).press()
        await pilot.pause()
        await pilot.pause()

        assert modal.state.working_copy_unsaved is False
        assert modal.state.selected_identity == "local:prompt:new-77"
        assert modal.state.selected_version == 9
        assert modal.state.selected_source == "local"
        assert modal._selected_record is not None
        assert modal._selected_record["name"] == "Saved Prompt"
        assert "Saved Prompt" in str(
            modal.query_one("#console-prompts-location", Static).renderable
        )
        update = modal.query_one("#prompt-editor-update-original", Button)
        assert update.disabled is False

        update.press()
        await pilot.pause()
        await pilot.pause()

    assert backend.save_calls[1]["prompt_identifier"] == "local:prompt:new-77"
    assert backend.save_calls[1]["expected_version"] == 9
    assert backend.save_calls[1]["name"] == "Saved Prompt"


@pytest.mark.asyncio
async def test_stale_compiled_text_warns_that_definition_wins_and_save_repairs() -> (
    None
):
    backend = _PromptBackend()
    backend.detail_result = {
        **_detail(),
        "system_prompt": "STALE COMPILED SYSTEM",
        "user_prompt": "STALE COMPILED USER",
    }
    backend.save_result = _detail(
        identifier="local:prompt:repaired",
        version=5,
    )
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        await modal.open_artifact("prompt-1")
        await pilot.pause()

        warnings = list(modal.query("#console-prompts-compatibility-stale"))
        assert warnings
        warning_copy = str(warnings[0].renderable)
        assert "definition is authoritative" in warning_copy
        assert "Saving repairs" in warning_copy

        modal.query_one("#prompt-editor-save-prompt", Button).press()
        await pilot.pause()
        await pilot.pause()

        assert warnings[0].display is False

    assert backend.save_calls[0]["system_prompt"] != "STALE COMPILED SYSTEM"
    assert "Be exact." in backend.save_calls[0]["system_prompt"]
    assert backend.save_calls[0]["user_prompt"] == "Answer the question."


@pytest.mark.asyncio
async def test_host_apply_deferral_replaces_ready_copy_and_has_no_apply_path() -> None:
    backend = _PromptBackend()
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        await modal.open_artifact("prompt-1")
        await pilot.pause()

        apply_button = modal.query_one("#prompt-editor-apply", Button)
        apply_copy = str(
            modal.query_one("#prompt-editor-apply-reason", Static).renderable
        )
        assert apply_button.disabled is True
        assert "Apply unavailable" in apply_copy
        assert "save the Prompt" in apply_copy
        assert "Ready" not in apply_copy
        apply_button.press()
        await pilot.pause()

    assert backend.model_calls == 0
    assert backend.usage_mutations == 0
    assert backend.save_calls == []


@pytest.mark.asyncio
async def test_provider_unavailable_configure_action_is_focusable_and_injected() -> (
    None
):
    backend = _PromptBackend()
    configure_calls: list[bool] = []

    async def configure_provider() -> None:
        configure_calls.append(True)

    app = _Harness(
        backend,
        improve_unavailable_reason="No active provider or model is configured.",
        configure_provider=configure_provider,
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        configure = modal.query_one("#console-prompts-configure-provider", Button)
        assert configure.disabled is False
        configure.focus()
        await pilot.pause()
        assert app.focused is configure

        configure.press()
        await pilot.pause()

        assert configure_calls == [True]
        assert modal.state.mode == "browse"
        assert modal.query_one("#console-prompts-search", Input).disabled is False


@pytest.mark.asyncio
async def test_source_switch_cancels_pending_query_debounce() -> None:
    backend = _PromptBackend()
    app = _Harness(backend)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        modal._query_requested(ConsolePromptsBrowse.QueryChanged("alpha"))
        await modal.switch_source("server")
        await asyncio.sleep(0.23)
        await pilot.pause()

    assert backend.search_calls.count(("server", "alpha")) == 1
