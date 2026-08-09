"""Focused UI/controller contracts for TASK-198 Prompt collections."""

from __future__ import annotations

import asyncio
import inspect
from datetime import datetime, timezone
from typing import Any

import pytest
from textual.app import App
from textual.containers import VerticalScroll
from textual.widgets import Button, Checkbox, Input, Static

import tldw_chatbook.Library.library_prompts_state as prompts_state_module
from tldw_chatbook.Prompt_Management.prompt_scope_service import PromptScopeService
from tldw_chatbook.Widgets.Library.library_prompts_canvas import (
    LibraryPromptsListCanvas,
)

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_prompts_canvas import (
    _open_prompt_editor,
    _painted_contrast,
    _painted_style_of_text,
    _real_prompt_scope_service,
    _wire_empty_non_prompt_services,
)
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)


def _catalog_page(*, offset: int, total: int = 207, query: str = "") -> dict[str, Any]:
    stop = min(total, offset + 100)
    return {
        "collections": [
            {
                "id": f"local:prompt_collection:{collection_id}",
                "backend": "local",
                "collection_id": collection_id,
                "name": (
                    "[bold]" if collection_id in {1, 2} else f"集合 {collection_id}"
                ),
                "display_name": (
                    f"[bold] · #{collection_id}"
                    if collection_id in {1, 2}
                    else f"集合 {collection_id}"
                ),
                "description": None,
                "prompt_ids": [],
            }
            for collection_id in range(offset + 1, stop + 1)
        ],
        "limit": 100,
        "offset": offset,
        "total": total,
        "query": query,
    }


def test_controller_calls_exact_local_scope_signatures_off_ui_loop():
    from tldw_chatbook.UI.Library_Modules.prompt_collections import (
        LibraryPromptCollectionsController,
    )

    assert (
        "mode"
        in inspect.signature(PromptScopeService.list_prompt_collections).parameters
    )
    calls: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []

    class Service:
        async def list_prompt_collections(
            self, *, mode: str, query: str, limit: int, offset: int
        ) -> dict[str, Any]:
            assert mode == "local"
            return _catalog_page(offset=offset, query=query)

    async def run(call, *args, **kwargs):
        calls.append((call, args, kwargs))
        return await call(
            *args, **{k: v for k, v in kwargs.items() if k != "isolate_in_worker"}
        )

    controller = LibraryPromptCollectionsController(
        run_service_call=lambda: run,
        prompt_service=lambda: Service(),
        sync_memberships=lambda: lambda _state: None,
        current_prompt_id=lambda: None,
        current_prompt_detail=lambda: None,
        prompt_editor_active=lambda: False,
    )

    async def exercise():
        token = controller.begin_manager()
        state = await controller.load_catalog(
            manager_token=token, query="  [bold] ", offset=0
        )
        assert state is not None
        assert state.query == "[bold]"
        assert state.total == 207

    asyncio.run(exercise())
    assert calls[0][2] == {
        "mode": "local",
        "query": "[bold]",
        "limit": 100,
        "offset": 0,
        "isolate_in_worker": True,
    }


def test_controller_rejects_late_catalog_and_membership_results_after_identity_switch():
    from tldw_chatbook.UI.Library_Modules.prompt_collections import (
        LibraryPromptCollectionsController,
    )

    catalog_gate = asyncio.Event()
    membership_gate = asyncio.Event()
    prompt_id = [41]
    synced: list[Any] = []

    class Service:
        async def list_prompt_collections(self, **_kwargs):
            await catalog_gate.wait()
            return _catalog_page(offset=0)

        async def list_prompt_collection_memberships(self, **_kwargs):
            await membership_gate.wait()
            return {"prompt_id": 41, "collection_ids": (1,), "changed": False}

    async def run(call, *args, **kwargs):
        kwargs.pop("isolate_in_worker", None)
        return await call(*args, **kwargs)

    controller = LibraryPromptCollectionsController(
        run_service_call=lambda: run,
        prompt_service=lambda: Service(),
        sync_memberships=lambda: synced.append,
        current_prompt_id=lambda: prompt_id[0] if prompt_id else None,
        current_prompt_detail=lambda: {"backend": "local"},
        prompt_editor_active=lambda: True,
    )

    async def exercise():
        token = controller.begin_manager()
        catalog_task = asyncio.create_task(
            controller.load_catalog(manager_token=token, query="", offset=0)
        )
        membership_task = asyncio.create_task(controller.load_memberships())
        await asyncio.sleep(0)
        controller.end_manager(token)
        prompt_id[:] = [42]
        catalog_gate.set()
        membership_gate.set()
        assert await catalog_task is None
        await membership_task

    asyncio.run(exercise())
    assert all(state.prompt_id != 41 or state.status == "loading" for state in synced)


def test_controller_membership_apply_keeps_content_dirty_and_outcomes_separate():
    from tldw_chatbook.UI.Library_Modules.prompt_collections import (
        LibraryPromptCollectionsController,
    )

    dirty = [True]
    content_status = ["Name already in use"]
    synced: list[Any] = []

    class Service:
        async def list_prompt_collection_memberships(self, **_kwargs):
            return {"prompt_id": 41, "collection_ids": (1,), "changed": False}

        async def replace_prompt_collection_memberships(self, **kwargs):
            return {
                "prompt_id": kwargs["prompt_id"],
                "collection_ids": tuple(kwargs["collection_ids"]),
                "changed": True,
            }

    async def run(call, *args, **kwargs):
        kwargs.pop("isolate_in_worker", None)
        return await call(*args, **kwargs)

    controller = LibraryPromptCollectionsController(
        run_service_call=lambda: run,
        prompt_service=lambda: Service(),
        sync_memberships=lambda: synced.append,
        current_prompt_id=lambda: 41,
        current_prompt_detail=lambda: {"backend": "local"},
        prompt_editor_active=lambda: True,
    )

    async def exercise():
        await controller.load_memberships()
        controller.stage_memberships((1, 2))
        await controller.apply_memberships()

    asyncio.run(exercise())
    assert dirty == [True]
    assert content_status == ["Name already in use"]
    assert synced[-1].outcome == "Memberships applied."
    assert synced[-1].status == "success"


def test_controller_invalidate_rejects_old_modal_across_same_prompt_reopen():
    from tldw_chatbook.UI.Library_Modules.prompt_collections import (
        LibraryPromptCollectionsController,
    )

    controller = LibraryPromptCollectionsController(
        run_service_call=lambda: None,
        prompt_service=lambda: None,
        sync_memberships=lambda: lambda _state: None,
        current_prompt_id=lambda: 41,
        current_prompt_detail=lambda: {"backend": "local"},
        prompt_editor_active=lambda: True,
    )
    old_token = controller.begin_manager("membership")
    old_identity = controller.identity_for(41)

    controller.invalidate()

    assert not controller.manager_is_active(old_token, "membership")
    assert not controller.manager_context_is_active(
        old_token,
        mode="membership",
        prompt_identity=old_identity,
    )


def test_controller_membership_success_callback_runs_once_only_for_current_success():
    from tldw_chatbook.UI.Library_Modules.prompt_collections import (
        LibraryPromptCollectionsController,
    )

    refreshes: list[str] = []

    class Service:
        async def list_prompt_collection_memberships(self, **_kwargs):
            return {"collection_ids": (1,)}

        async def replace_prompt_collection_memberships(self, **kwargs):
            return {"collection_ids": kwargs["collection_ids"]}

    async def run(call, *args, **kwargs):
        kwargs.pop("isolate_in_worker", None)
        return await call(*args, **kwargs)

    controller = LibraryPromptCollectionsController(
        run_service_call=lambda: run,
        prompt_service=lambda: Service(),
        sync_memberships=lambda: lambda _state: None,
        current_prompt_id=lambda: 41,
        current_prompt_detail=lambda: {"backend": "local"},
        prompt_editor_active=lambda: True,
        membership_applied=lambda: lambda: refreshes.append("refresh"),
    )

    async def exercise():
        await controller.load_memberships()
        await controller.apply_memberships()  # no staged change: no callback
        controller.stage_memberships((1, 2))
        await controller.apply_memberships()
        await controller.apply_memberships()  # already current: no callback

    asyncio.run(exercise())
    assert refreshes == ["refresh"]


def test_controller_rename_refreshes_off_page_label_from_validated_response():
    from tldw_chatbook.UI.Library_Modules.prompt_collections import (
        LibraryPromptCollectionsController,
    )

    renamed = "[bold] renamed 集合"

    class Service:
        async def list_prompt_collections(self, **kwargs):
            page = _catalog_page(
                offset=kwargs["offset"], total=150, query=kwargs["query"]
            )
            for item in page["collections"]:
                if item["collection_id"] == 150:
                    item["name"] = "Old off-page label"
                    item["display_name"] = "Old off-page label"
            return page

        async def update_prompt_collection(self, **kwargs):
            assert kwargs == {
                "mode": "local",
                "collection_id": 150,
                "name": renamed,
            }
            return {
                "id": "local:prompt_collection:150",
                "backend": "local",
                "collection_id": 150,
                "name": renamed,
                "display_name": renamed,
                "description": None,
                "prompt_ids": [],
            }

    async def run(call, *args, **kwargs):
        kwargs.pop("isolate_in_worker", None)
        return await call(*args, **kwargs)

    controller = LibraryPromptCollectionsController(
        run_service_call=lambda: run,
        prompt_service=lambda: Service(),
        sync_memberships=lambda: lambda _state: None,
        current_prompt_id=lambda: None,
        current_prompt_detail=lambda: None,
        prompt_editor_active=lambda: False,
    )

    async def exercise():
        token = controller.begin_manager("browse")
        await controller.load_catalog(manager_token=token, query="", offset=0)
        await controller.load_catalog(manager_token=token, query="", offset=100)
        assert controller.collection_label(150) == "Old off-page label"
        await controller.rename_collection(
            manager_token=token,
            manager_mode="browse",
            collection_id=150,
            name=renamed,
        )

    asyncio.run(exercise())
    assert controller.collection_label(150) == renamed


class _ManagerHost(App):
    def __init__(self, *, mode: str, total: int = 207) -> None:
        super().__init__()
        self.mode = mode
        self.total = total
        self.calls: list[tuple[str, Any]] = []

    def on_mount(self) -> None:
        self.push_screen(self._modal())

    def _modal(self):
        from tldw_chatbook.UI.Library_Modules.prompt_collection_manager_modal import (
            PromptCollectionManagerModal,
        )

        async def load(*, query: str, offset: int):
            self.calls.append(("load", (query, offset)))
            current = prompts_state_module.begin_prompt_collection_catalog(
                query=query, request_token=len(self.calls)
            )
            if offset:
                first = prompts_state_module.apply_prompt_collection_catalog_page(
                    current,
                    _catalog_page(offset=0, total=self.total, query=query),
                    request_token=len(self.calls),
                )
                return prompts_state_module.apply_prompt_collection_catalog_page(
                    first,
                    _catalog_page(offset=offset, total=self.total, query=query),
                    request_token=len(self.calls),
                    append=True,
                )
            return prompts_state_module.apply_prompt_collection_catalog_page(
                current,
                _catalog_page(offset=0, total=self.total, query=query),
                request_token=len(self.calls),
            )

        async def create(name: str):
            self.calls.append(("create", name))
            return await load(query="", offset=0)

        async def rename(collection_id: int, name: str):
            self.calls.append(("rename", (collection_id, name)))
            return await load(query="", offset=0)

        return PromptCollectionManagerModal(
            mode=self.mode,
            selected_collection_id=None,
            staged_collection_ids=(1, 2),
            load_catalog=load,
            create_collection=create,
            rename_collection=rename,
        )


class _StyledManagerHost(_ManagerHost):
    CSS_PATH = LibraryHarness.CSS_PATH


class _MutationManagerHost(App):
    def __init__(self, *, failure: Exception | None = None) -> None:
        super().__init__()
        self.failure = failure
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.create_calls = 0

    def on_mount(self) -> None:
        self.push_screen(self._modal())

    def _modal(self):
        from tldw_chatbook.UI.Library_Modules.prompt_collection_manager_modal import (
            PromptCollectionManagerModal,
        )

        async def load(*, query: str, offset: int):
            current = prompts_state_module.begin_prompt_collection_catalog(
                query=query, request_token=1
            )
            return prompts_state_module.apply_prompt_collection_catalog_page(
                current,
                _catalog_page(offset=offset, total=2, query=query),
                request_token=1,
            )

        async def create(_name: str):
            self.create_calls += 1
            self.started.set()
            if self.failure is not None:
                raise self.failure
            await self.release.wait()
            return await load(query="", offset=0)

        async def rename(_collection_id: int, _name: str):
            raise AssertionError("rename was not requested")

        return PromptCollectionManagerModal(
            mode="browse",
            selected_collection_id=None,
            staged_collection_ids=(),
            load_catalog=load,
            create_collection=create,
            rename_collection=rename,
        )


class _CatalogErrorManagerHost(App):
    def __init__(self) -> None:
        super().__init__()
        self.load_calls = 0

    def on_mount(self) -> None:
        from tldw_chatbook.UI.Library_Modules.prompt_collection_manager_modal import (
            PromptCollectionManagerModal,
        )

        async def load(*, query: str, offset: int):
            self.load_calls += 1
            current = prompts_state_module.begin_prompt_collection_catalog(
                query=query, request_token=self.load_calls
            )
            return prompts_state_module.fail_prompt_collection_catalog(
                current,
                request_token=self.load_calls,
                error="Couldn't load collections. Retry.",
            )

        async def unused(*_args, **_kwargs):
            raise AssertionError("mutation was not requested")

        self.push_screen(
            PromptCollectionManagerModal(
                mode="browse",
                selected_collection_id=None,
                staged_collection_ids=(),
                load_catalog=load,
                create_collection=unused,
                rename_collection=unused,
            )
        )


class _CatalogMutationRaceHost(App):
    def __init__(self) -> None:
        super().__init__()
        self.load_started = asyncio.Event()
        self.release_load = asyncio.Event()
        self.create_calls = 0

    @staticmethod
    def _catalog(collection_id: int, name: str):
        current = prompts_state_module.begin_prompt_collection_catalog(
            query="", request_token=1
        )
        return prompts_state_module.apply_prompt_collection_catalog_page(
            current,
            {
                "collections": [
                    {
                        "id": f"local:prompt_collection:{collection_id}",
                        "backend": "local",
                        "collection_id": collection_id,
                        "name": name,
                        "display_name": name,
                        "description": None,
                        "prompt_ids": [],
                    }
                ],
                "limit": 100,
                "offset": 0,
                "total": 1,
                "query": "",
            },
            request_token=1,
        )

    def on_mount(self) -> None:
        from tldw_chatbook.UI.Library_Modules.prompt_collection_manager_modal import (
            PromptCollectionManagerModal,
        )

        async def load(*, query: str, offset: int):
            assert query == "" and offset == 0
            self.load_started.set()
            await self.release_load.wait()
            return self._catalog(1, "Older load")

        async def create(_name: str):
            self.create_calls += 1
            return self._catalog(99, "Created authoritative")

        async def unused(*_args, **_kwargs):
            raise AssertionError("rename was not requested")

        self.push_screen(
            PromptCollectionManagerModal(
                mode="browse",
                selected_collection_id=None,
                staged_collection_ids=(),
                load_catalog=load,
                create_collection=create,
                rename_collection=unused,
            )
        )


class _CollectionCanvasHost(App):
    def __init__(self, *, mode: str, membership_state=None) -> None:
        super().__init__()
        self._mode = mode
        self._membership_state = membership_state

    def compose(self):
        if self._mode == "list":
            browse = prompts_state_module.build_prompt_browse_result(
                prompts_state_module.PromptBrowseScope(page_size=2),
                {
                    "items": [
                        {
                            "id": 1,
                            "name": "Literal",
                            "artifact_type": "prompt",
                            "backend": "local",
                        }
                    ],
                    "total_items": 1,
                    "total_pages": 1,
                    "current_page": 1,
                    "per_page": 2,
                },
            )
            yield LibraryPromptsListCanvas(
                prompts_state_module.build_prompt_browse_list_state(
                    browse, now=datetime.now(timezone.utc)
                ),
                browse_result=browse,
                collection_label="[bold] · #7",
            )
            return
        editor = prompts_state_module.build_prompt_editor_state(
            {
                "id": 41,
                "name": "Dirty prompt",
                "backend": "local",
                "artifact_type": "prompt",
                "system_prompt": "System",
                "user_prompt": "User",
                "version": 3,
            }
        )
        yield LibraryPromptsListCanvas(
            mode="editor",
            editor_state=editor,
            dirty=True,
            can_update_original=True,
            membership_state=self._membership_state,
        )


@pytest.mark.asyncio
async def test_shared_manager_has_one_scroll_owner_literal_labels_no_delete_or_source():
    app = _ManagerHost(mode="browse")
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = app.screen
        assert len(screen.query(VerticalScroll)) == 1
        rows_scroll = screen.query_one(
            "#prompt-collection-manager-rows", VerticalScroll
        )
        load_more = screen.query_one("#prompt-collection-manager-load-more", Button)
        assert load_more.parent is not rows_scroll
        assert screen.query_one("#prompt-collection-manager-search", Input).has_focus
        assert (
            str(screen.query_one("#prompt-collection-manager-row-1", Button).label)
            == "[bold] · #1"
        )
        assert len(screen.query("#prompt-collection-manager-delete")) == 0
        visible_copy = " ".join(str(widget.render()) for widget in screen.query(Static))
        assert "server" not in visible_copy.casefold()
        assert "source" not in visible_copy.casefold()
        assert "local only" in visible_copy.casefold()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "selector", "widget_type", "size"),
    (
        ("browse", "#prompt-collection-manager-row-1", Button, (80, 24)),
        (
            "membership",
            "#prompt-collection-manager-member-1",
            Checkbox,
            (100, 30),
        ),
    ),
)
async def test_shared_manager_literal_rows_paint_readably_in_final_compositor(
    mode, selector, widget_type, size
):
    app = _StyledManagerHost(mode=mode)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        row = app.screen.query_one(selector, widget_type)

        painted = _painted_style_of_text(app, row.region, "[bold] · #1")
        assert painted is not None
        assert painted.color is not None and painted.bgcolor is not None
        assert _painted_contrast(painted.color, painted.bgcolor) >= 4.5


@pytest.mark.asyncio
async def test_shared_manager_load_more_beyond_207_and_membership_multiselect():
    browse_app = _ManagerHost(mode="browse")
    async with browse_app.run_test(size=(64, 24)) as pilot:
        await pilot.pause()
        await pilot.click("#prompt-collection-manager-load-more")
        await pilot.pause()
        assert len(browse_app.screen.query(".prompt-collection-manager-row")) == 201
        assert browse_app.screen.query_one(
            "#prompt-collection-manager-load-more", Button
        ).display

    membership_app = _ManagerHost(mode="membership")
    async with membership_app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        first = membership_app.screen.query_one(
            "#prompt-collection-manager-member-1", Checkbox
        )
        second = membership_app.screen.query_one(
            "#prompt-collection-manager-member-2", Checkbox
        )
        assert first.value is True and second.value is True
        await pilot.click("#prompt-collection-manager-member-1")
        assert first.value is False and second.value is True
        assert membership_app.screen.query_one(
            "#prompt-collection-manager-done", Button
        )


@pytest.mark.asyncio
async def test_manager_rename_requires_one_concrete_selection_and_new_has_explicit_outcome():
    app = _ManagerHost(mode="membership", total=2)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        rename = app.screen.query_one("#prompt-collection-manager-rename", Button)
        assert rename.disabled is True
        await pilot.click("#prompt-collection-manager-new-name")
        await pilot.press("space", "space")
        await pilot.click("#prompt-collection-manager-create")
        assert (
            "required"
            in str(
                app.screen.query_one(
                    "#prompt-collection-manager-outcome", Static
                ).render()
            ).casefold()
        )
        assert not any(call[0] == "create" for call in app.calls)


@pytest.mark.asyncio
async def test_manager_mutation_failure_copy_never_renders_exception_secrets():
    secret = "SECRET-COLLECTION-NAME-AND-SQL"
    app = _MutationManagerHost(failure=RuntimeError(secret))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.screen.query_one(
            "#prompt-collection-manager-new-name", Input
        ).value = "Private name"
        await pilot.click("#prompt-collection-manager-create")
        await _wait_for_condition(
            pilot,
            lambda: (
                "Retry"
                in str(
                    app.screen.query_one(
                        "#prompt-collection-manager-outcome", Static
                    ).render()
                )
            ),
            message="create failure never settled",
        )
        outcome = str(
            app.screen.query_one("#prompt-collection-manager-outcome", Static).render()
        )
        assert outcome == "Couldn't create collection. Retry."
        assert secret not in outcome
        assert app.screen.query_one("#prompt-collection-manager-retry", Button).display


@pytest.mark.asyncio
async def test_manager_catalog_error_state_always_exposes_exact_retry():
    app = _CatalogErrorManagerHost()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        retry = app.screen.query_one("#prompt-collection-manager-retry", Button)
        assert retry.display
        retry.press()
        await _wait_for_condition(
            pilot,
            lambda: app.load_calls == 2,
            message="catalog Retry did not repeat the exact root load",
        )


@pytest.mark.asyncio
async def test_manager_create_is_single_flight_under_rapid_double_submit():
    app = _MutationManagerHost()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one("#prompt-collection-manager-new-name", Input).value = "One"
        first = asyncio.create_task(modal._run_mutation("create", None, "One"))
        await app.started.wait()
        second = asyncio.create_task(modal._run_mutation("create", None, "One"))
        await pilot.pause()

        assert app.create_calls == 1
        assert modal.query_one("#prompt-collection-manager-create", Button).disabled
        assert modal.query_one("#prompt-collection-manager-rename", Button).disabled
        assert modal.query_one("#prompt-collection-manager-cancel", Button).disabled
        modal.action_cancel()
        await pilot.pause()
        assert app.screen is modal

        app.release.set()
        await asyncio.gather(first, second)
        await pilot.pause()
        assert app.create_calls == 1
        assert not modal.query_one("#prompt-collection-manager-create", Button).disabled
        assert not modal.query_one("#prompt-collection-manager-cancel", Button).disabled


@pytest.mark.asyncio
async def test_older_catalog_load_cannot_overwrite_completed_mutation():
    app = _CatalogMutationRaceHost()
    async with app.run_test(size=(80, 24)) as pilot:
        await app.load_started.wait()
        modal = app.screen
        await modal._run_mutation("create", None, "Created authoritative")
        await pilot.pause()
        assert app.create_calls == 1
        assert len(modal.query("#prompt-collection-manager-row-99")) == 1

        app.release_load.set()
        await pilot.pause()
        assert len(modal.query("#prompt-collection-manager-row-99")) == 1
        assert len(modal.query("#prompt-collection-manager-row-1")) == 0
        assert (
            str(modal.query_one("#prompt-collection-manager-outcome", Static).render())
            == "Collection created."
        )


def test_collection_manager_presentation_and_coordinator_are_split_by_concern():
    from tldw_chatbook.UI.Library_Modules.prompt_collection_manager_modal import (
        PromptCollectionManagerModal,
    )
    from tldw_chatbook.UI.Library_Modules.prompt_collections import (
        LibraryPromptCollectionsController,
    )

    assert PromptCollectionManagerModal.__module__.endswith(
        "prompt_collection_manager_modal"
    )
    assert callable(getattr(LibraryPromptCollectionsController, "open_manager"))


@pytest.mark.asyncio
async def test_prompt_canvas_starts_with_visible_literal_collection_control():
    app = _CollectionCanvasHost(mode="list")
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        collection = app.screen.query_one("#library-prompts-collection", Button)
        assert str(collection.label) == "collection: [bold] · #7 ▸"
        focus_ids = [widget.id for widget in app.screen.focus_chain]
        assert focus_ids.index("library-prompts-collection") < focus_ids.index(
            "library-prompts-sort"
        )
        visible = " ".join(str(widget.render()) for widget in app.screen.query(Static))
        assert "server" not in visible.casefold()
        assert "source" not in visible.casefold()


@pytest.mark.asyncio
async def test_prompt_editor_memberships_are_separate_from_dirty_content_save():
    loading = prompts_state_module.begin_prompt_memberships(
        prompt_id=41,
        identity_fingerprint="local:prompt:41",
        request_token=1,
    )
    ready = prompts_state_module.apply_prompt_memberships_loaded(
        loading,
        collection_ids=(1,),
        labels={1: "[bold] literal"},
        request_token=1,
    )
    staged = prompts_state_module.stage_prompt_memberships(ready, (1, 2))
    app = _CollectionCanvasHost(mode="editor", membership_state=staged)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        assert "Unsaved changes" in str(
            app.screen.query_one("#library-prompt-meta", Static).render()
        )
        assert "Staged" in str(
            app.screen.query_one("#library-prompt-memberships-summary", Static).render()
        )
        assert (
            app.screen.query_one("#library-prompt-memberships-manage", Button).disabled
            is False
        )
        assert (
            app.screen.query_one("#library-prompt-memberships-apply", Button).disabled
            is False
        )
        assert (
            str(app.screen.query_one("#library-prompt-save-status", Static).render())
            == ""
        )
        assert "Saved." not in str(
            app.screen.query_one("#library-prompt-memberships-status", Static).render()
        )


@pytest.mark.asyncio
async def test_prompt_editor_unsaved_identity_has_readable_apply_reason():
    disabled = prompts_state_module.disable_prompt_memberships(
        "Save this prompt before managing collections."
    )
    app = _CollectionCanvasHost(mode="editor", membership_state=disabled)

    async with app.run_test(size=(64, 24)) as pilot:
        await pilot.pause()
        assert (
            app.screen.query_one("#library-prompt-memberships-apply", Button).disabled
            is True
        )
        assert "Save this prompt" in str(
            app.screen.query_one("#library-prompt-memberships-status", Static).render()
        )


@pytest.mark.asyncio
async def test_library_screen_membership_load_retry_and_apply_retry_are_distinct(
    tmp_path, monkeypatch
):
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _prompt_uuid, _message = db.add_prompt(
        name="Retry prompt",
        author="A",
        details="Before",
        system_prompt="System",
        user_prompt="User",
    )
    first = await service.create_prompt_collection(
        mode="local", name="Existing", prompt_ids=[prompt_id]
    )
    second = await service.create_prompt_collection(mode="local", name="Staged")
    read_memberships = service.list_prompt_collection_memberships
    replace_memberships = service.replace_prompt_collection_memberships
    load_calls = 0
    replace_calls = 0

    async def flaky_load(**kwargs):
        nonlocal load_calls
        load_calls += 1
        if load_calls == 1:
            raise RuntimeError("private load details")
        return await read_memberships(**kwargs)

    async def flaky_replace(**kwargs):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 1:
            raise RuntimeError("private apply details")
        return await replace_memberships(**kwargs)

    monkeypatch.setattr(service, "list_prompt_collection_memberships", flaky_load)
    monkeypatch.setattr(service, "replace_prompt_collection_memberships", flaky_replace)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        controller = screen._library_prompt_collections_controller
        await _wait_for_condition(
            pilot,
            lambda: controller.membership_state.status == "load_error",
            message="membership load failure was not distinguished",
        )
        manage = screen.query_one("#library-prompt-memberships-manage", Button)
        apply = screen.query_one("#library-prompt-memberships-apply", Button)
        assert str(manage.label) == "Retry memberships"
        assert manage.disabled is False
        assert apply.disabled is True
        assert controller.open_manager("membership") is None

        controller.stage_memberships((second["collection_id"],))
        await controller.apply_memberships()
        assert replace_calls == 0
        assert controller.membership_state.status == "load_error"
        assert (await read_memberships(mode="local", prompt_id=prompt_id))[
            "collection_ids"
        ] == (first["collection_id"],)

        manage.press()
        await _wait_for_condition(
            pilot,
            lambda: controller.membership_state.status == "ready",
            message="Retry memberships did not reload the exact set",
        )
        assert controller.membership_state.applied_ids == (first["collection_id"],)
        assert load_calls == 2

        name_input = screen.query_one("#library-prompt-name", Input)
        name_input.value = "Retry prompt dirty"
        await pilot.pause()
        assert screen._library_prompt_dirty is True
        controller.stage_memberships(
            tuple(sorted((first["collection_id"], second["collection_id"])))
        )
        apply.press()
        await _wait_for_condition(
            pilot,
            lambda: controller.membership_state.status == "apply_error",
            message="membership Apply failure was not distinguished",
        )
        assert controller.membership_state.applied_ids == (first["collection_id"],)
        assert controller.membership_state.staged_ids == tuple(
            sorted((first["collection_id"], second["collection_id"]))
        )
        assert controller.membership_state.can_apply is True
        assert apply.disabled is False
        assert screen._library_prompt_dirty is True
        assert screen._library_prompt_status == ""

        apply.press()
        await _wait_for_condition(
            pilot,
            lambda: controller.membership_state.status == "success",
            message="membership Apply Retry did not settle",
        )
        assert replace_calls == 2
        assert (await read_memberships(mode="local", prompt_id=prompt_id))[
            "collection_ids"
        ] == tuple(sorted((first["collection_id"], second["collection_id"])))
        assert screen._library_prompt_dirty is True
        assert screen._library_prompt_status == ""


@pytest.mark.asyncio
async def test_library_screen_collection_manager_selects_exact_browse_scope(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _prompt_uuid, _message = db.add_prompt(
        name="Only here",
        author="A",
        details="Collection browse",
        system_prompt="System",
        user_prompt="User",
    )
    created = await service.create_prompt_collection(
        mode="local",
        name="[bold]",
        prompt_ids=[prompt_id],
    )
    collection_id = created["collection_id"]
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-prompts", Button).press()
        await _wait_for_selector(screen, pilot, "#library-prompts-collection")
        assert (
            str(screen.query_one("#library-prompts-collection", Button).label)
            == "collection: All prompts ▸"
        )

        screen.query_one("#library-prompts-collection", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: host.screen is not screen,
            message="collection manager never opened",
        )
        row = await _wait_for_selector(
            host.screen,
            pilot,
            f"#prompt-collection-manager-row-{collection_id}",
        )
        assert str(row.label) == "[bold]"
        row.press()
        host.screen.query_one("#prompt-collection-manager-done", Button).press()

        await _wait_for_condition(
            pilot,
            lambda: (
                _active_library_screen(host) is screen
                and screen._library_prompt_browse_controller.scope.collection_id
                == collection_id
            ),
            message="collection result never updated the exact Prompt browse scope",
        )
        await _wait_for_selector(screen, pilot, f"#library-prompt-row-{prompt_id}")
        assert screen._library_prompt_browse_controller.scope.page == 1
        assert (
            str(screen.query_one("#library-prompts-collection", Button).label)
            == "collection: [bold] ▸"
        )


@pytest.mark.asyncio
async def test_library_screen_manager_create_search_rename_and_explicit_all(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(
        name="Manager flow prompt",
        author="A",
        details="Manager flow",
        system_prompt="System",
        user_prompt="User",
    )
    created_name = "[bold] 新規"
    renamed_name = "[italic] renamed 集合"
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-prompts", Button).press()
        await _wait_for_selector(screen, pilot, "#library-prompts-collection")
        screen.query_one("#library-prompts-collection", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: host.screen is not screen,
            message="collection manager never opened",
        )
        modal = host.screen
        assert len(modal.query("#prompt-collection-manager-delete")) == 0
        assert len(modal.query("#prompt-collection-manager-source")) == 0
        assert len(modal.query("#prompt-collection-manager-server")) == 0
        await _wait_for_selector(modal, pilot, "#prompt-collection-manager-create")

        modal.query_one(
            "#prompt-collection-manager-new-name", Input
        ).value = created_name
        modal.query_one("#prompt-collection-manager-create", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: (
                str(
                    host.screen.query_one(
                        "#prompt-collection-manager-outcome", Static
                    ).render()
                )
                == "Collection created."
            ),
            message="collection create outcome never settled",
        )
        created_page = await service.list_prompt_collections(
            mode="local", query=created_name, limit=100, offset=0
        )
        assert created_page["total"] == 1
        collection_id = created_page["collections"][0]["collection_id"]

        search = host.screen.query_one("#prompt-collection-manager-search", Input)
        search.value = created_name
        search.focus()
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: (
                host.screen._catalog.query == created_name
                and len(
                    host.screen.query(f"#prompt-collection-manager-row-{collection_id}")
                )
                == 1
            ),
            message="literal collection search did not return the exact row",
        )
        row = host.screen.query_one(
            f"#prompt-collection-manager-row-{collection_id}", Button
        )
        assert str(row.label) == created_name
        row.press()
        await pilot.pause()
        assert host.screen._selected_id == collection_id
        await _wait_for_selector(
            host.screen, pilot, "#prompt-collection-manager-new-name"
        )
        host.screen.query_one(
            "#prompt-collection-manager-new-name", Input
        ).value = renamed_name
        host.screen.query_one("#prompt-collection-manager-rename", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: (
                str(
                    host.screen.query_one(
                        "#prompt-collection-manager-outcome", Static
                    ).render()
                )
                == "Collection renamed."
            ),
            message="collection rename outcome never settled",
        )
        renamed = await service.get_prompt_collection(
            mode="local", collection_id=collection_id
        )
        assert renamed["display_name"] == renamed_name
        assert (
            str(
                host.screen.query_one(
                    f"#prompt-collection-manager-row-{collection_id}", Button
                ).label
            )
            == renamed_name
        )
        host.screen.query_one("#prompt-collection-manager-done", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: (
                _active_library_screen(host) is screen
                and screen._library_prompt_browse_controller.scope.collection_id
                == collection_id
            ),
            message="renamed collection never became the active filter",
        )
        assert (
            str(screen.query_one("#library-prompts-collection", Button).label)
            == f"collection: {renamed_name} ▸"
        )

        screen.query_one("#library-prompts-collection", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: host.screen is not screen,
            message="collection manager did not reopen",
        )
        search = host.screen.query_one("#prompt-collection-manager-search", Input)
        search.value = "does-not-match"
        search.focus()
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: host.screen._catalog.status == "empty",
            message="no-match search did not settle empty",
        )
        host.screen.query_one("#prompt-collection-manager-done", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: _active_library_screen(host) is screen,
            message="filtered manager did not close",
        )
        assert (
            screen._library_prompt_browse_controller.scope.collection_id
            == collection_id
        )
        await _wait_for_selector(screen, pilot, "#library-prompts-collection")
        assert (
            str(screen.query_one("#library-prompts-collection", Button).label)
            == f"collection: {renamed_name} ▸"
        )

        screen.query_one("#library-prompts-collection", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: host.screen is not screen,
            message="collection manager did not reopen for All prompts",
        )
        host.screen.query_one("#prompt-collection-manager-all", Button).press()
        await pilot.pause()
        host.screen.query_one("#prompt-collection-manager-done", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: (
                _active_library_screen(host) is screen
                and screen._library_prompt_browse_controller.scope.collection_id is None
            ),
            message="All prompts was not explicitly restored",
        )
        await _wait_for_selector(screen, pilot, "#library-prompts-collection")
        await _wait_for_condition(
            pilot,
            lambda: (
                str(screen.query_one("#library-prompts-collection", Button).label)
                == "collection: All prompts ▸"
            ),
            message="All prompts label did not refresh",
        )


@pytest.mark.asyncio
async def test_library_screen_membership_apply_is_independent_from_dirty_prompt_save(
    tmp_path,
):
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _prompt_uuid, _message = db.add_prompt(
        name="Dirty prompt",
        author="A",
        details="Before",
        system_prompt="System",
        user_prompt="User",
    )
    first = await service.create_prompt_collection(
        mode="local", name="First", prompt_ids=[prompt_id]
    )
    second = await service.create_prompt_collection(mode="local", name="Second")
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        await _wait_for_condition(
            pilot,
            lambda: (
                getattr(screen, "_library_prompt_collections_controller", None)
                is not None
                and screen._library_prompt_collections_controller.membership_state.status
                == "ready"
            ),
            message="Prompt memberships never loaded",
        )
        name_input = screen.query_one("#library-prompt-name", Input)
        name_input.value = "Dirty prompt edited"
        await pilot.pause()
        assert screen._library_prompt_dirty is True
        refreshes: list[str] = []
        screen._refresh_local_source_snapshot = lambda: refreshes.append("refresh")
        browse_token = screen._library_prompt_browse_controller.result.request_token

        screen.query_one("#library-prompt-memberships-manage", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: host.screen is not screen,
            message="membership manager never opened",
        )
        await _wait_for_selector(
            host.screen,
            pilot,
            f"#prompt-collection-manager-member-{second['collection_id']}",
        )
        checkbox = host.screen.query_one(
            f"#prompt-collection-manager-member-{second['collection_id']}", Checkbox
        )
        checkbox.toggle()
        host.screen.query_one("#prompt-collection-manager-done", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: (
                _active_library_screen(host) is screen
                and screen._library_prompt_collections_controller.membership_state.staged_ids
                == tuple(sorted((first["collection_id"], second["collection_id"])))
            ),
            message="manager result never staged membership choices",
        )
        before_apply = await service.list_prompt_collection_memberships(
            mode="local", prompt_id=prompt_id
        )
        assert before_apply["collection_ids"] == (first["collection_id"],)

        screen.query_one("#library-prompt-memberships-apply", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_prompt_collections_controller.membership_state.status
                == "success"
            ),
            message="membership Apply never completed",
        )
        after_apply = await service.list_prompt_collection_memberships(
            mode="local", prompt_id=prompt_id
        )
        assert after_apply["collection_ids"] == tuple(
            sorted((first["collection_id"], second["collection_id"]))
        )
        assert screen._library_prompt_dirty is True
        assert screen._library_prompt_status == ""
        assert refreshes == ["refresh"]
        assert (
            screen._library_prompt_browse_controller.result.request_token > browse_token
        )
        assert (
            screen._library_prompt_collections_controller.membership_state.outcome
            == "Memberships applied."
        )
