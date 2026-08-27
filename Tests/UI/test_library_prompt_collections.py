"""Focused UI/controller contracts for TASK-198 Prompt collections."""

from __future__ import annotations

import asyncio
import inspect
import threading
from datetime import datetime, timezone
from typing import Any

import pytest

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.containers import VerticalScroll
from textual.geometry import Region
from textual.widgets import Button, Checkbox, Input, Static

import tldw_chatbook.Library.library_prompts_state as prompts_state_module
from tldw_chatbook.Prompt_Management.prompt_scope_service import PromptScopeService
from tldw_chatbook.UI.Library_Modules.prompt_collections import (
    LibraryPromptCollectionsController,
)
from tldw_chatbook.Widgets.Library.library_prompts_canvas import (
    LibraryPromptsListCanvas,
)

from Tests.UI.app_factory import _build_test_app
from Tests.UI.background_signals import wait_for_background_signal, wait_for_signal
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
    _wait_for_widget_state,
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


def _direct_controller(service, **kwargs) -> LibraryPromptCollectionsController:
    """Build a production-shaped controller with an in-loop test dispatcher."""

    async def run(call, *args, **call_kwargs):
        call_kwargs.pop("isolate_in_worker", None)
        return await call(*args, **call_kwargs)

    prompt_id = kwargs.pop("prompt_id", 41)
    return LibraryPromptCollectionsController(
        run_service_call=lambda: run,
        prompt_service=lambda: service,
        sync_memberships=kwargs.pop("sync_memberships", lambda: lambda _state: None),
        current_prompt_id=(prompt_id if callable(prompt_id) else lambda: prompt_id),
        current_prompt_detail=kwargs.pop(
            "current_prompt_detail", lambda: {"backend": "local"}
        ),
        prompt_editor_active=kwargs.pop("prompt_editor_active", lambda: True),
        **kwargs,
    )


def _modal_outcome(modal) -> str:
    for outcome in modal.query("#prompt-collection-manager-outcome").results(Static):
        return str(outcome.render())
    return "<outcome unavailable>"


def test_collection_controller_public_apis_use_google_style_docs():
    required_sections = {
        "__init__": ("Args:",),
        "begin_manager": ("Args:", "Returns:", "Raises:"),
        "manager_is_active": ("Args:", "Returns:"),
        "manager_context_is_active": ("Args:", "Returns:"),
        "end_manager": ("Args:",),
        "invalidate": ("Args:",),
        "identity_for": ("Args:", "Returns:", "Raises:"),
        "open_manager": ("Args:", "Returns:"),
        "load_catalog": ("Args:", "Returns:"),
        "create_collection": ("Args:", "Returns:", "Raises:"),
        "rename_collection": ("Args:", "Returns:", "Raises:"),
        "collection_label": ("Args:", "Returns:"),
        "load_memberships": ("Returns:",),
        "disable_memberships": ("Args:",),
        "stage_memberships": ("Args:", "Raises:"),
        "apply_memberships": ("Returns:",),
    }
    for method_name, sections in required_sections.items():
        doc = inspect.getdoc(getattr(LibraryPromptCollectionsController, method_name))
        assert doc is not None, method_name
        for section in sections:
            assert section in doc, f"{method_name} lacks {section}"


def test_controller_normalizes_catalog_query_and_dispatches_exact_local_kwargs():
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


@pytest.mark.asyncio
async def test_real_library_service_runner_isolates_every_collection_call_and_keeps_ui_responsive():
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    ui_thread = threading.get_ident()
    started = threading.Event()
    release = threading.Event()
    service_threads: list[tuple[str, int]] = []
    dispatched: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    class Service:
        block_first_list = True

        def _record(self, operation: str) -> None:
            service_threads.append((operation, threading.get_ident()))

        def list_prompt_collections(self, **kwargs):
            self._record("list_prompt_collections")
            if self.block_first_list:
                self.block_first_list = False
                started.set()
                assert release.wait(timeout=5)
            return _catalog_page(
                offset=kwargs["offset"], total=207, query=kwargs["query"]
            )

        def create_prompt_collection(self, **_kwargs):
            self._record("create_prompt_collection")
            return {"collection_id": 1}

        def update_prompt_collection(self, **kwargs):
            self._record("update_prompt_collection")
            return {
                "backend": "local",
                "collection_id": kwargs["collection_id"],
                "name": kwargs["name"],
                "display_name": kwargs["name"],
            }

        def list_prompt_collection_memberships(self, **kwargs):
            self._record("list_prompt_collection_memberships")
            return {
                "prompt_id": kwargs["prompt_id"],
                "collection_ids": (1, 101, 207),
                "changed": False,
            }

        def replace_prompt_collection_memberships(self, **kwargs):
            self._record("replace_prompt_collection_memberships")
            return {
                "prompt_id": kwargs["prompt_id"],
                "collection_ids": tuple(kwargs["collection_ids"]),
                "changed": True,
            }

    class ResponsiveHost(ConsolidatedCSSApp):
        def __init__(self) -> None:
            super().__init__()
            self.pings = 0

        def compose(self):
            yield Button("Ping", id="ping")

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "ping":
                self.pings += 1

    service = Service()

    async def run(call, *args, **kwargs):
        dispatched.append((call.__name__, args, dict(kwargs)))
        return await LibraryScreen._run_library_service_call(call, *args, **kwargs)

    controller = LibraryPromptCollectionsController(
        run_service_call=lambda: run,
        prompt_service=lambda: service,
        sync_memberships=lambda: lambda _state: None,
        current_prompt_id=lambda: 41,
        current_prompt_detail=lambda: {"backend": "local"},
        prompt_editor_active=lambda: True,
    )
    host = ResponsiveHost()

    async with host.run_test(size=(40, 12)) as pilot:
        token = controller.begin_manager()
        catalog_task = asyncio.create_task(
            controller.load_catalog(manager_token=token, query="", offset=0)
        )
        await _wait_for_condition(
            pilot, started.is_set, message="blocking catalog call never started"
        )
        await pilot.click("#ping")
        assert host.pings == 1
        assert catalog_task.done() is False
        release.set()
        assert await catalog_task is not None

        await controller.create_collection(manager_token=token, name="Created")
        await controller.rename_collection(
            manager_token=token, collection_id=1, name="Renamed"
        )
        await controller.load_memberships()
        controller.stage_memberships((1, 2))
        await controller.apply_memberships()

    assert service_threads
    assert all(thread_id != ui_thread for _operation, thread_id in service_threads)
    dispatched_by_name = {name: kwargs for name, _args, kwargs in dispatched}
    list_dispatches = [
        kwargs
        for name, _args, kwargs in dispatched
        if name == "list_prompt_collections"
    ]
    assert list_dispatches[-3:] == [
        {
            "mode": "local",
            "query": "",
            "limit": 100,
            "offset": offset,
            "isolate_in_worker": True,
        }
        for offset in (0, 100, 200)
    ]
    assert dispatched_by_name["create_prompt_collection"] == {
        "mode": "local",
        "name": "Created",
        "isolate_in_worker": True,
    }
    assert dispatched_by_name["update_prompt_collection"] == {
        "mode": "local",
        "collection_id": 1,
        "name": "Renamed",
        "isolate_in_worker": True,
    }
    assert dispatched_by_name["list_prompt_collection_memberships"] == {
        "mode": "local",
        "prompt_id": 41,
        "isolate_in_worker": True,
    }
    assert dispatched_by_name["replace_prompt_collection_memberships"] == {
        "mode": "local",
        "prompt_id": 41,
        "collection_ids": (1, 2),
        "isolate_in_worker": True,
    }


def test_controller_rejects_late_catalog_and_membership_results_after_identity_switch():
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

    controller = _direct_controller(
        Service(),
        sync_memberships=lambda: synced.append,
        prompt_id=lambda: prompt_id[0] if prompt_id else None,
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


@pytest.mark.parametrize(
    "response",
    (
        None,
        {"collection_ids": (1,)},
        {"prompt_id": 0, "collection_ids": (1,)},
        {"prompt_id": 42, "collection_ids": (1,)},
        {"prompt_id": 41, "collection_ids": "1"},
    ),
)
def test_controller_membership_load_reply_fails_closed_on_malformed_identity(response):
    class Service:
        async def list_prompt_collection_memberships(self, **_kwargs):
            return response

    controller = _direct_controller(Service())

    asyncio.run(controller.load_memberships())

    assert controller.membership_state.status == "load_error"
    assert controller.membership_state.applied_ids == ()
    assert controller.membership_state.staged_ids == ()
    assert controller.membership_state.outcome == "Couldn't load memberships. Retry."


@pytest.mark.parametrize(
    "response",
    (
        None,
        {"collection_ids": (1, 2)},
        {"prompt_id": 0, "collection_ids": (1, 2)},
        {"prompt_id": 42, "collection_ids": (1, 2)},
        {"prompt_id": 41, "collection_ids": (1,)},
    ),
)
def test_controller_membership_apply_reply_fails_closed_on_identity_or_set_drift(
    response,
):
    class Service:
        async def list_prompt_collection_memberships(self, **_kwargs):
            return {"prompt_id": 41, "collection_ids": (1,), "changed": False}

        async def replace_prompt_collection_memberships(self, **_kwargs):
            return response

    controller = _direct_controller(Service())

    async def exercise() -> None:
        await controller.load_memberships()
        controller.stage_memberships((1, 2))
        await controller.apply_memberships()

    asyncio.run(exercise())

    assert controller.membership_state.status == "apply_error"
    assert controller.membership_state.applied_ids == (1,)
    assert controller.membership_state.staged_ids == (1, 2)
    assert controller.membership_state.outcome == "Couldn't apply memberships. Retry."
    assert controller.membership_state.can_apply is True


def test_controller_invalidate_rejects_old_modal_across_same_prompt_reopen():
    controller = _direct_controller(None)
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
    refreshes: list[str] = []

    class Service:
        async def list_prompt_collection_memberships(self, **kwargs):
            return {
                "prompt_id": kwargs["prompt_id"],
                "collection_ids": (1,),
                "changed": False,
            }

        async def replace_prompt_collection_memberships(self, **kwargs):
            return {
                "prompt_id": kwargs["prompt_id"],
                "collection_ids": kwargs["collection_ids"],
                "changed": True,
            }

    controller = _direct_controller(
        Service(),
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


@pytest.mark.asyncio
async def test_controller_hydrates_207_memberships_with_three_catalog_pages():
    calls: list[tuple[str, dict[str, Any]]] = []

    class Service:
        async def list_prompt_collection_memberships(self, **kwargs):
            calls.append(("memberships", kwargs))
            return {
                "prompt_id": kwargs["prompt_id"],
                "collection_ids": (*range(1, 207), 999),
                "changed": False,
            }

        async def list_prompt_collections(self, **kwargs):
            calls.append(("catalog", kwargs))
            return _catalog_page(offset=kwargs["offset"])

    controller = _direct_controller(Service())

    await controller.load_memberships()

    labels = dict(controller.membership_state.labels)
    assert len(controller.membership_state.applied_ids) == 207
    assert len(labels) == 206
    assert labels[2] == "[bold] · #2"
    assert labels[206] == "集合 206"
    assert 999 not in labels
    assert calls == [
        ("memberships", {"mode": "local", "prompt_id": 41}),
        (
            "catalog",
            {"mode": "local", "query": "", "limit": 100, "offset": 0},
        ),
        (
            "catalog",
            {"mode": "local", "query": "", "limit": 100, "offset": 100},
        ),
        (
            "catalog",
            {"mode": "local", "query": "", "limit": 100, "offset": 200},
        ),
    ]

    app = _CollectionCanvasHost(
        mode="editor", membership_state=controller.membership_state
    )
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        summary = str(
            app.screen.query_one("#library-prompt-memberships-summary", Static).render()
        )
        assert "Collection #999" in summary


@pytest.mark.asyncio
async def test_controller_stops_membership_label_hydration_when_page_cannot_advance():
    catalog_calls = 0

    class Service:
        async def list_prompt_collection_memberships(self, **kwargs):
            return {
                "prompt_id": kwargs["prompt_id"],
                "collection_ids": (999,),
                "changed": False,
            }

        async def list_prompt_collections(self, **kwargs):
            nonlocal catalog_calls
            catalog_calls += 1
            await asyncio.sleep(0)
            return {
                "collections": [],
                "limit": 100,
                "offset": kwargs["offset"],
                "total": 207,
                "query": kwargs["query"],
            }

    controller = _direct_controller(Service())

    await asyncio.wait_for(controller.load_memberships(), timeout=0.5)

    assert catalog_calls == 1
    assert controller.membership_state.status == "ready"
    assert controller.membership_state.applied_ids == (999,)
    assert controller.membership_state.labels == ()


def test_controller_rejects_late_membership_label_hydration_after_identity_switch():
    detail_started = asyncio.Event()
    detail_release = asyncio.Event()
    prompt_id = [41]

    class Service:
        async def list_prompt_collection_memberships(self, **kwargs):
            return {
                "prompt_id": kwargs["prompt_id"],
                "collection_ids": (2,),
                "changed": False,
            }

        async def list_prompt_collections(self, **kwargs):
            detail_started.set()
            await detail_release.wait()
            return _catalog_page(offset=kwargs["offset"])

    controller = _direct_controller(Service(), prompt_id=lambda: prompt_id[0])

    async def exercise() -> None:
        task = asyncio.create_task(controller.load_memberships())
        await wait_for_background_signal(
            detail_started, task, what="the membership load"
        )
        prompt_id[0] = 42
        detail_release.set()
        await task

    asyncio.run(exercise())

    assert controller.membership_state.status == "loading"
    assert controller.membership_state.prompt_id == 41
    assert controller.membership_state.labels == ()


def test_controller_rename_refreshes_off_page_label_from_validated_response():
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

    controller = _direct_controller(
        Service(),
        prompt_id=None,
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


class _ManagerHost(ConsolidatedCSSApp):
    def __init__(
        self,
        *,
        mode: str,
        total: int = 207,
        staged_ids: tuple[int, ...] = (1, 2),
        selected_id: int | None = None,
        display_name: str | None = None,
        create_refresh_error: bool = False,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.total = total
        self.staged_ids = staged_ids
        self.selected_id = selected_id
        self.display_name = display_name
        self.create_refresh_error = create_refresh_error
        self.calls: list[tuple[str, Any]] = []
        self.catalog_state = None

    def on_mount(self) -> None:
        self.push_screen(self._modal())

    def _modal(self):
        from tldw_chatbook.UI.Library_Modules.prompt_collection_manager_modal import (
            PromptCollectionManagerModal,
        )

        async def load(*, query: str, offset: int):
            self.calls.append(("load", (query, offset)))
            append = offset > 0
            loading = prompts_state_module.begin_prompt_collection_catalog(
                query=query,
                request_token=len(self.calls),
                previous=self.catalog_state if append else None,
                append=append,
            )
            page = _catalog_page(offset=offset, total=self.total, query=query)
            if self.display_name is not None and page["collections"]:
                page["collections"][-1]["name"] = self.display_name
                page["collections"][-1]["display_name"] = self.display_name
            self.catalog_state = (
                prompts_state_module.apply_prompt_collection_catalog_page(
                    loading,
                    page,
                    request_token=len(self.calls),
                    append=append,
                )
            )
            return self.catalog_state

        async def create(name: str):
            self.calls.append(("create", name))
            if self.create_refresh_error:
                loading = prompts_state_module.begin_prompt_collection_catalog(
                    query="", request_token=len(self.calls)
                )
                return prompts_state_module.fail_prompt_collection_catalog(
                    loading,
                    request_token=len(self.calls),
                    error="Couldn't load collections. Retry.",
                )
            return await load(query="", offset=0)

        async def rename(collection_id: int, name: str):
            self.calls.append(("rename", (collection_id, name)))
            return await load(query="", offset=0)

        return PromptCollectionManagerModal(
            mode=self.mode,
            selected_collection_id=self.selected_id,
            staged_collection_ids=self.staged_ids,
            load_catalog=load,
            create_collection=create,
            rename_collection=rename,
        )


class _StyledManagerHost(_ManagerHost):
    CSS_PATH = LibraryHarness.CSS_PATH


class _MutationManagerHost(ConsolidatedCSSApp):
    def __init__(
        self, *, failure: BaseException | None = None, action: str = "create"
    ) -> None:
        super().__init__()
        self.failure = failure
        self.action = action
        self.started = asyncio.Event()
        self.finished = asyncio.Event()
        self.release = asyncio.Event()
        self.create_calls = 0
        self.results: list[object] = []

    def on_mount(self) -> None:
        self.push_screen(self._modal(), callback=self.results.append)

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
            try:
                if self.failure is not None:
                    raise self.failure
                await self.release.wait()
                return await load(query="", offset=0)
            finally:
                self.finished.set()

        async def rename(_collection_id: int, _name: str):
            self.started.set()
            try:
                if self.failure is not None:
                    raise self.failure
                await self.release.wait()
                return await load(query="", offset=0)
            finally:
                self.finished.set()

        return PromptCollectionManagerModal(
            mode="browse",
            selected_collection_id=1 if self.action == "rename" else None,
            staged_collection_ids=(),
            load_catalog=load,
            create_collection=create,
            rename_collection=rename,
        )


class _GuardedMutationManagerHost(ConsolidatedCSSApp):
    def __init__(self, *, action: str) -> None:
        super().__init__()
        self.action = action
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.results: list[object] = []
        self.create_calls = 0
        self.rename_calls = 0

    def on_mount(self) -> None:
        self.push_screen(self._modal(), callback=self.results.append)

    @staticmethod
    def _catalog():
        current = prompts_state_module.begin_prompt_collection_catalog(
            query="", request_token=1
        )
        return prompts_state_module.apply_prompt_collection_catalog_page(
            current,
            _catalog_page(offset=0, total=207),
            request_token=1,
        )

    def _modal(self):
        from tldw_chatbook.UI.Library_Modules.prompt_collection_manager_modal import (
            PromptCollectionManagerModal,
        )

        async def load(*, query: str, offset: int):
            assert query == "" and offset == 0
            return self._catalog()

        async def create(_name: str):
            self.create_calls += 1
            if self.action == "retry" and self.create_calls == 1:
                raise RuntimeError("force mutation retry")
            self.started.set()
            await self.release.wait()
            return self._catalog()

        async def rename(_collection_id: int, _name: str):
            self.rename_calls += 1
            self.started.set()
            await self.release.wait()
            return self._catalog()

        return PromptCollectionManagerModal(
            mode="browse",
            selected_collection_id=1,
            staged_collection_ids=(),
            load_catalog=load,
            create_collection=create,
            rename_collection=rename,
        )


class _RemountMutationManagerHost(ConsolidatedCSSApp):
    def __init__(self, *, stale_result: str) -> None:
        super().__init__()
        self.stale_result = stale_result
        self.stale_started = asyncio.Event()
        self.stale_cancelled = asyncio.Event()
        self.stale_release = asyncio.Event()
        self.stale_finished = asyncio.Event()
        self.current_started = asyncio.Event()
        self.current_release = asyncio.Event()
        self.create_calls = 0
        self.results: list[object] = []
        self.modal = self._modal()

    def on_mount(self) -> None:
        self.push_screen(self.modal, callback=self.results.append)

    @staticmethod
    def _catalog():
        current = prompts_state_module.begin_prompt_collection_catalog(
            query="", request_token=1
        )
        return prompts_state_module.apply_prompt_collection_catalog_page(
            current,
            _catalog_page(offset=0, total=2),
            request_token=1,
        )

    def _modal(self):
        from tldw_chatbook.UI.Library_Modules.prompt_collection_manager_modal import (
            PromptCollectionManagerModal,
        )

        async def load(*, query: str, offset: int):
            assert query == "" and offset == 0
            return self._catalog()

        async def create(_name: str):
            self.create_calls += 1
            if self.create_calls == 1:
                self.stale_started.set()
                try:
                    await self.stale_release.wait()
                except asyncio.CancelledError:
                    task = asyncio.current_task()
                    if task is not None:
                        task.uncancel()
                    self.stale_cancelled.set()
                    await self.stale_release.wait()
                self.stale_finished.set()
                if self.stale_result == "failure":
                    raise RuntimeError("stale mutation failure")
                if self.stale_result == "cancelled":
                    raise asyncio.CancelledError
                return self._catalog()

            self.current_started.set()
            await self.current_release.wait()
            return self._catalog()

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


class _PagingRetryManagerHost(ConsolidatedCSSApp):
    def __init__(
        self,
        *,
        fail_offset: int,
        fail_once: bool = True,
        callback_failure: str | None = None,
    ) -> None:
        super().__init__()
        self.fail_offset = fail_offset
        self.fail_once = fail_once
        self.callback_failure = callback_failure
        self.failed = False
        self.offsets: list[int] = []
        self.catalog_state = None

    def on_mount(self) -> None:
        from tldw_chatbook.UI.Library_Modules.prompt_collection_manager_modal import (
            PromptCollectionManagerModal,
        )

        async def load(*, query: str, offset: int):
            self.offsets.append(offset)
            append = offset > 0
            loading = prompts_state_module.begin_prompt_collection_catalog(
                query=query,
                request_token=len(self.offsets),
                previous=self.catalog_state if append else None,
                append=append,
            )
            if offset == self.fail_offset and (not self.failed or not self.fail_once):
                self.failed = True
                if self.callback_failure == "exception":
                    raise RuntimeError("private catalog failure")
                if self.callback_failure == "none":
                    return None
                self.catalog_state = (
                    prompts_state_module.fail_prompt_collection_catalog(
                        loading,
                        request_token=len(self.offsets),
                        error="Couldn't load collections. Retry.",
                    )
                )
            else:
                self.catalog_state = (
                    prompts_state_module.apply_prompt_collection_catalog_page(
                        loading,
                        _catalog_page(offset=offset),
                        request_token=len(self.offsets),
                        append=append,
                    )
                )
            return self.catalog_state

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


class _CatalogMutationRaceHost(ConsolidatedCSSApp):
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


class _CollectionCanvasHost(ConsolidatedCSSApp):
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
                            "id": "local:prompt:1",
                            "local_id": 1,
                            "name": "Literal",
                            "artifact_type": "prompt",
                            "backend": "local",
                            "version": 1,
                        }
                    ],
                    "total_items": 1,
                    "total_pages": 1,
                    "current_page": 1,
                    "page": 1,
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
async def test_shared_manager_restores_exact_focus_after_each_recompose_action():
    app = _ManagerHost(mode="browse")
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        modal = app.screen
        assert modal.query_one("#prompt-collection-manager-search", Input).has_focus

        search = modal.query_one("#prompt-collection-manager-search", Input)
        search.value = "集合"
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: modal._catalog.query == "集合" and modal._catalog.status == "ready",
            message="search did not settle",
        )
        assert modal.query_one("#prompt-collection-manager-search", Input).has_focus
        assert _modal_outcome(modal) == ""

        load_more = modal.query_one("#prompt-collection-manager-load-more", Button)
        load_more.focus()
        load_more.press()
        await _wait_for_condition(
            pilot,
            lambda: len(modal._catalog.items) == 200,
            message="second catalog page did not settle",
        )
        await _wait_for_selector(modal, pilot, "#prompt-collection-manager-load-more")
        await _wait_for_condition(
            pilot,
            lambda: (
                modal.query_one(
                    "#prompt-collection-manager-load-more", Button
                ).has_focus
            ),
            message="Load more focus was not restored",
        )
        assert _modal_outcome(modal) == ""

        row = modal.query_one("#prompt-collection-manager-row-1", Button)
        row.focus()
        row.press()
        await _wait_for_condition(
            pilot,
            lambda: (
                modal.query_one("#prompt-collection-manager-row-1", Button).has_focus
            ),
            message="selected row focus was not restored",
        )

        all_rows = modal.query_one("#prompt-collection-manager-all", Button)
        all_rows.focus()
        all_rows.press()
        await pilot.pause()
        await _wait_for_condition(
            pilot,
            lambda: modal.query_one("#prompt-collection-manager-all", Button).has_focus,
            message="All prompts focus was not restored",
        )

        name = modal.query_one("#prompt-collection-manager-new-name", Input)
        name.value = "Created focus"
        modal.query_one("#prompt-collection-manager-create", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: _modal_outcome(modal) == "Collection created.",
            message="create did not settle",
        )
        await _wait_for_condition(
            pilot,
            lambda: (
                modal.query_one("#prompt-collection-manager-new-name", Input).has_focus
            ),
            message="create settlement focus was not restored",
        )

        row = modal.query_one("#prompt-collection-manager-row-1", Button)
        row.focus()
        row.press()
        await pilot.pause()
        modal.query_one(
            "#prompt-collection-manager-new-name", Input
        ).value = "Renamed focus"
        modal.query_one("#prompt-collection-manager-rename", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: _modal_outcome(modal) == "Collection renamed.",
            message="rename did not settle",
        )
        await _wait_for_condition(
            pilot,
            lambda: (
                modal.query_one("#prompt-collection-manager-row-1", Button).has_focus
            ),
            message="rename settlement focus was not restored",
        )


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
        await _wait_for_condition(
            pilot,
            lambda: len(browse_app.screen._catalog.items) == 200,
            message="second catalog page did not settle",
        )
        assert len(browse_app.screen.query(".prompt-collection-manager-row")) == 201
        assert browse_app.screen.query_one(
            "#prompt-collection-manager-load-more", Button
        ).display
        await pilot.click("#prompt-collection-manager-load-more")
        await _wait_for_condition(
            pilot,
            lambda: len(browse_app.screen._catalog.items) == 207,
            message="third catalog page did not settle",
        )
        assert browse_app.screen.query_one("#prompt-collection-manager-row-207", Button)
        assert browse_app.screen._catalog.total == 207
        assert browse_app.screen._catalog.has_more is False
        load_more = browse_app.screen.query_one(
            "#prompt-collection-manager-load-more", Button
        )
        assert load_more.display is False
        assert load_more.disabled is True
        await _wait_for_condition(
            pilot,
            lambda: (
                browse_app.screen.query_one(
                    "#prompt-collection-manager-row-201", Button
                ).has_focus
            ),
            message="final page did not focus its first surviving appended row",
        )

    membership_app = _ManagerHost(mode="membership", staged_ids=(1, 2, 207))
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
        await pilot.click("#prompt-collection-manager-load-more")
        await _wait_for_condition(
            pilot,
            lambda: len(membership_app.screen._catalog.items) == 200,
            message="membership second page did not settle",
        )
        await pilot.click("#prompt-collection-manager-load-more")
        await _wait_for_condition(
            pilot,
            lambda: len(membership_app.screen._catalog.items) == 207,
            message="membership third page did not settle",
        )
        assert (
            membership_app.screen.query_one(
                "#prompt-collection-manager-member-207", Checkbox
            ).value
            is True
        )
        assert membership_app.screen._staged_ids == {2, 207}
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
        assert "required" in _modal_outcome(app.screen).casefold()
        assert not any(call[0] == "create" for call in app.calls)


@pytest.mark.asyncio
async def test_membership_manager_renames_focused_row_without_changing_staged_set():
    long_name = "[bold]研究🙂[/bold]" * 70
    assert len(long_name) > 1000
    app = _ManagerHost(
        mode="membership",
        total=2,
        staged_ids=(1, 2),
        display_name=long_name,
    )
    async with app.run_test(size=(64, 24)) as pilot:
        await pilot.pause()
        modal = app.screen
        first = modal.query_one("#prompt-collection-manager-member-1", Checkbox)
        second = modal.query_one("#prompt-collection-manager-member-2", Checkbox)
        assert first.value is True and second.value is True
        assert modal._staged_ids == {1, 2}

        second.focus()
        await pilot.pause()
        rename = modal.query_one("#prompt-collection-manager-rename", Button)
        assert rename.disabled is False
        assert modal._staged_ids == {1, 2}

        modal.query_one("#prompt-collection-manager-new-name", Input).focus()
        await pilot.pause()
        target = modal.query_one("#prompt-collection-manager-rename-target", Static)
        assert str(target.render()) == f"Rename target: {long_name}"
        assert target.region.height == 1
        assert target.styles.text_wrap == "nowrap"
        assert target.styles.text_overflow == "ellipsis"
        viewport = Region(0, 0, pilot.app.size.width, pilot.app.size.height)
        for selector in (
            "#prompt-collection-manager-rename",
            "#prompt-collection-manager-done",
        ):
            visible = modal.query_one(selector, Button).region.intersection(viewport)
            assert visible.width > 0 and visible.height > 0
        assert modal._staged_ids == {1, 2}

        modal.query_one(
            "#prompt-collection-manager-new-name", Input
        ).value = "Focused rename"
        rename.press()
        await _wait_for_condition(
            pilot,
            lambda: _modal_outcome(modal) == "Collection renamed.",
            message="focused membership rename did not settle",
        )
        assert ("rename", (2, "Focused rename")) in app.calls
        assert modal._staged_ids == {1, 2}
        assert (
            modal.query_one("#prompt-collection-manager-member-1", Checkbox).value
            is True
        )
        assert (
            modal.query_one("#prompt-collection-manager-member-2", Checkbox).value
            is True
        )
        await _wait_for_condition(
            pilot,
            lambda: (
                modal.query_one(
                    "#prompt-collection-manager-member-2", Checkbox
                ).has_focus
            ),
            message="renamed membership row focus was not restored",
        )


@pytest.mark.asyncio
async def test_off_page_rename_falls_back_to_visible_search_focus():
    app = _ManagerHost(mode="browse", total=150, selected_id=150)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one("#prompt-collection-manager-load-more", Button).press()
        await _wait_for_selector(modal, pilot, "#prompt-collection-manager-row-150")
        modal.query_one("#prompt-collection-manager-row-150", Button).press()
        await _wait_for_selector(modal, pilot, "#prompt-collection-manager-new-name")
        modal.query_one(
            "#prompt-collection-manager-new-name", Input
        ).value = "Off-page renamed"
        modal.query_one("#prompt-collection-manager-rename", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: _modal_outcome(modal) == "Collection renamed.",
            message="off-page rename did not settle",
        )
        await _wait_for_condition(
            pilot,
            lambda: (
                modal.query_one("#prompt-collection-manager-search", Input).has_focus
            ),
            message="off-page rename did not fall back to visible Search focus",
        )


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
            lambda: "Retry" in _modal_outcome(app.screen),
            message="create failure never settled",
        )
        outcome = _modal_outcome(app.screen)
        assert outcome == "Couldn't create collection. Retry."
        assert secret not in outcome
        assert app.screen.query_one("#prompt-collection-manager-retry", Button).display


@pytest.mark.asyncio
async def test_manager_catalog_error_state_always_exposes_exact_retry():
    app = _PagingRetryManagerHost(fail_offset=0, fail_once=False)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        retry = app.screen.query_one("#prompt-collection-manager-retry", Button)
        assert retry.display
        retry.press()
        await _wait_for_condition(
            pilot,
            lambda: app.offsets == [0, 0],
            message="catalog Retry did not repeat the exact root load",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failed_offset", "callback_failure"),
    ((100, None), (200, None), (200, "exception"), (200, "none")),
)
async def test_manager_catalog_retry_repeats_exact_failed_page_offset(
    failed_offset, callback_failure
):
    app = _PagingRetryManagerHost(
        fail_offset=failed_offset, callback_failure=callback_failure
    )
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        modal = app.screen
        if failed_offset == 200:
            modal.query_one("#prompt-collection-manager-load-more", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: len(modal._catalog.items) == 200,
                message="page two did not settle before page-three failure",
            )
            await _wait_for_selector(
                modal, pilot, "#prompt-collection-manager-load-more"
            )
        modal.query_one("#prompt-collection-manager-load-more", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: modal._catalog.status == "error",
            message="catalog page failure did not settle",
        )
        assert modal._catalog.offset == failed_offset
        assert app.offsets[-1] == failed_offset
        await _wait_for_selector(modal, pilot, "#prompt-collection-manager-retry")
        retry = modal.query_one("#prompt-collection-manager-retry", Button)
        assert retry.display is True
        retry.press()
        expected_loaded = min(207, failed_offset + 100)
        await _wait_for_condition(
            pilot,
            lambda: len(modal._catalog.items) == expected_loaded,
            message="catalog page Retry did not settle",
        )
        await pilot.pause()
        assert app.offsets[-2:] == [failed_offset, failed_offset]
        assert (
            modal.query_one("#prompt-collection-manager-retry", Button).display is False
        )
        await _wait_for_condition(
            pilot,
            lambda: _modal_outcome(modal) == "",
            message="catalog page Retry outcome did not clear",
        )


@pytest.mark.asyncio
async def test_successful_mutation_with_refresh_failure_retries_catalog_only():
    app = _ManagerHost(mode="browse", total=1, create_refresh_error=True)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one("#prompt-collection-manager-new-name", Input).value = "Created"
        modal.query_one("#prompt-collection-manager-create", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: "Retry catalog" in _modal_outcome(modal),
            message="partial mutation success was not reported",
        )
        assert [call for call in app.calls if call[0] == "create"] == [
            ("create", "Created")
        ]
        retry = modal.query_one("#prompt-collection-manager-retry", Button)
        assert retry.display is True
        retry.press()
        await _wait_for_condition(
            pilot,
            lambda: (
                modal._catalog.status == "ready"
                and [call for call in app.calls if call[0] == "load"]
                == [("load", ("", 0)), ("load", ("", 0))]
            ),
            message="catalog-only Retry did not settle",
        )
        assert [call for call in app.calls if call[0] == "create"] == [
            ("create", "Created")
        ]
        assert (
            modal.query_one("#prompt-collection-manager-retry", Button).display is False
        )
        assert _modal_outcome(modal) == ""


async def _start_guarded_collection_mutation(
    app: _GuardedMutationManagerHost, pilot
) -> None:
    modal = app.screen
    if app.action == "rename":
        await pilot.click("#prompt-collection-manager-row-1")
        assert modal._selected_id == 1
    modal.query_one("#prompt-collection-manager-new-name", Input).value = "One"
    if app.action == "rename":
        modal.query_one("#prompt-collection-manager-rename", Button).press()
    else:
        modal.query_one("#prompt-collection-manager-create", Button).press()
    if app.action == "retry":
        retry = await _wait_for_widget_state(
            modal,
            pilot,
            "#prompt-collection-manager-retry",
            lambda widget: widget.display,
            what="failed create did not expose mutation retry",
        )
        retry.press()
    await asyncio.wait_for(app.started.wait(), timeout=1.0)
    await pilot.pause()


async def _dispatch_collection_close(
    pilot, source: str, release: asyncio.Event
) -> None:
    if source == "escape":
        dispatch = pilot.press("escape")
    elif source == "backdrop":
        dispatch = pilot.click(offset=(0, 0))
    else:
        dispatch = pilot.click("#prompt-collection-manager-cancel")
    dispatch_task = asyncio.create_task(dispatch)
    done, _pending = await asyncio.wait({dispatch_task}, timeout=1.0)
    if not done:
        release.set()
        pytest.fail("close input was queued behind the collection mutation")
    await dispatch_task


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["create", "rename", "retry"])
@pytest.mark.parametrize(
    "close_sources",
    [
        ("escape", "backdrop", "visible"),
        ("backdrop", "visible", "escape"),
        ("visible", "escape", "backdrop"),
    ],
)
async def test_collection_mutation_close_requests_dispatch_without_queuing(
    action: str,
    close_sources: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _GuardedMutationManagerHost(action=action)
    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        modal = app.screen
        load_more = await _wait_for_widget_state(
            modal,
            pilot,
            "#prompt-collection-manager-load-more",
            lambda widget: widget.display and not widget.disabled,
            what="Load more was not enabled before the collection mutation",
        )
        assert modal._catalog.has_more
        assert load_more.display and not load_more.disabled
        await _start_guarded_collection_mutation(app, pilot)
        content = modal.query_one("#prompt-collection-manager")
        assert not content.region.contains(0, 0)

        disabled_selectors = (
            "#prompt-collection-manager-search",
            "#prompt-collection-manager-load-more",
            "#prompt-collection-manager-all",
            "#prompt-collection-manager-row-1",
            "#prompt-collection-manager-new-name",
            "#prompt-collection-manager-create",
            "#prompt-collection-manager-rename",
            "#prompt-collection-manager-retry",
            "#prompt-collection-manager-done",
        )
        assert all(
            modal.query_one(selector).disabled for selector in disabled_selectors
        )
        assert not modal.query_one("#prompt-collection-manager-cancel", Button).disabled
        outcome = modal.query_one("#prompt-collection-manager-outcome", Static)
        original_update = outcome.update
        status_updates: list[object] = []

        def record_status_update(content: object = "") -> None:
            status_updates.append(content)
            original_update(content)

        monkeypatch.setattr(outcome, "update", record_status_update)

        try:
            for source in close_sources:
                await _dispatch_collection_close(pilot, source, app.release)
                await pilot.pause()
                assert app.screen is modal
                assert app.results == []
                assert (
                    _modal_outcome(modal)
                    == "Finish the current collection change before closing."
                )
            assert status_updates == [
                "Finish the current collection change before closing."
            ]
        finally:
            app.release.set()
        await _wait_for_condition(
            pilot,
            lambda: not modal._mutation_in_flight,
            message="collection mutation did not settle",
        )
        assert app.screen is modal
        assert app.results == []
        assert _modal_outcome(modal) in {
            "Collection created.",
            "Collection renamed.",
        }
        assert app.create_calls == (2 if action == "retry" else int(action == "create"))
        assert app.rename_calls == int(action == "rename")
        restored_load_more = await _wait_for_widget_state(
            modal,
            pilot,
            "#prompt-collection-manager-load-more",
            lambda widget: widget.display and not widget.disabled,
            what="Load more was not restored after the collection mutation",
        )
        assert modal._catalog.has_more
        assert restored_load_more.display and not restored_load_more.disabled


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "expected_outcome", "expected_retry"),
    [
        ("create", "Couldn't create collection. Retry.", ("create", None, "Cancelled")),
        ("rename", "Couldn't rename collection. Retry.", ("rename", 1, "Cancelled")),
    ],
)
async def test_collection_mutation_cancelled_callback_restores_current_modal_controls(
    action: str,
    expected_outcome: str,
    expected_retry: tuple[str, int | None, str],
):
    app = _MutationManagerHost(failure=asyncio.CancelledError(), action=action)
    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one(
            "#prompt-collection-manager-new-name", Input
        ).value = "Cancelled"
        modal.query_one(f"#prompt-collection-manager-{action}", Button).press()
        await asyncio.wait_for(app.started.wait(), timeout=1.0)
        await asyncio.wait_for(app.finished.wait(), timeout=1.0)
        retry = await _wait_for_widget_state(
            modal,
            pilot,
            "#prompt-collection-manager-retry",
            lambda widget: widget.display and not widget.disabled and widget.has_focus,
            what=f"cancelled {action} did not expose focused Retry",
            attempts=50,
        )

        assert app.screen is modal
        assert app.results == []
        assert not modal._mutation_in_flight
        assert _modal_outcome(modal) == expected_outcome
        assert modal._retry_action == expected_retry
        assert retry.display and not retry.disabled and retry.has_focus
        for selector in (
            "#prompt-collection-manager-search",
            "#prompt-collection-manager-new-name",
            "#prompt-collection-manager-create",
            "#prompt-collection-manager-done",
            "#prompt-collection-manager-cancel",
        ):
            assert not modal.query_one(selector).disabled


@pytest.mark.asyncio
@pytest.mark.parametrize("stale_result", ["success", "failure", "cancelled"])
async def test_collection_mutation_remount_rejects_cancellation_resistant_completion(
    stale_result: str,
) -> None:
    app = _RemountMutationManagerHost(stale_result=stale_result)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        modal = app.modal
        try:
            modal.query_one(
                "#prompt-collection-manager-new-name", Input
            ).value = "Stale"
            modal.query_one("#prompt-collection-manager-create", Button).press()
            await asyncio.wait_for(app.stale_started.wait(), timeout=1.0)
            stale_generation = modal._safe_mount_generation

            modal.dismiss(None)
            await asyncio.wait_for(app.stale_cancelled.wait(), timeout=1.0)
            await pilot.pause()
            await app.push_screen(modal, callback=app.results.append)
            await pilot.pause()
            await pilot.pause()
            assert modal._safe_mount_generation > stale_generation

            modal.query_one(
                "#prompt-collection-manager-new-name", Input
            ).value = "Current"
            modal.query_one("#prompt-collection-manager-create", Button).press()
            await asyncio.wait_for(app.current_started.wait(), timeout=1.0)
            await pilot.pause()
            cancel = modal.query_one("#prompt-collection-manager-cancel", Button)
            cancel.focus()
            await pilot.pause()
            assert modal._mutation_in_flight
            assert _modal_outcome(modal) == "Creating collection…"

            app.stale_release.set()
            await asyncio.wait_for(app.stale_finished.wait(), timeout=1.0)
            await pilot.pause()
            assert app.screen is modal
            assert app.results == [None]
            assert modal._mutation_in_flight
            assert _modal_outcome(modal) == "Creating collection…"
            assert modal.focused is cancel

            app.current_release.set()
            await _wait_for_condition(
                pilot,
                lambda: _modal_outcome(modal) == "Collection created.",
                message="current remounted mutation did not settle",
            )
            assert app.create_calls == 2
            assert app.screen is modal
            assert app.results == [None]
        finally:
            app.stale_release.set()
            app.current_release.set()


@pytest.mark.asyncio
async def test_older_catalog_load_cannot_overwrite_completed_mutation():
    app = _CatalogMutationRaceHost()
    async with app.run_test(size=(80, 24)) as pilot:
        await wait_for_signal(
            app.load_started, what="the modal's on-mount catalog load"
        )
        modal = app.screen
        modal.query_one(
            "#prompt-collection-manager-new-name", Input
        ).value = "Created authoritative"
        await pilot.click("#prompt-collection-manager-create")
        await _wait_for_condition(
            pilot,
            lambda: _modal_outcome(modal) == "Collection created.",
            message="authoritative create did not settle",
        )
        assert app.create_calls == 1
        assert len(modal.query("#prompt-collection-manager-row-99")) == 1

        app.release_load.set()
        await pilot.pause()
        assert len(modal.query("#prompt-collection-manager-row-99")) == 1
        assert len(modal.query("#prompt-collection-manager-row-1")) == 0
        assert _modal_outcome(modal) == "Collection created."


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
        assert str(collection.label) == "collection: [bold] · #7"
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
            == "collection: All prompts"
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
            == "collection: [bold]"
        )


@pytest.mark.asyncio
async def test_real_library_screen_collection_manager_crosses_sqlite_page_100(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(
        name="Collection paging prompt",
        author="A",
        details="Keeps the populated-list collection control mounted.",
        system_prompt="System",
        user_prompt="User",
    )
    created = []
    for index in range(1, 108):
        created.append(
            await service.create_prompt_collection(
                mode="local", name=f"Boundary {index:03d}"
            )
        )
    final_id = created[-1]["collection_id"]
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
            message="real collection manager never opened",
        )
        modal = host.screen
        await _wait_for_condition(
            pilot,
            lambda: len(modal._catalog.items) == 100,
            message="real catalog first page did not settle",
        )
        assert len(modal.query(f"#prompt-collection-manager-row-{final_id}")) == 0
        modal.query_one("#prompt-collection-manager-load-more", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: len(modal._catalog.items) == 107,
            message="real catalog second page did not settle",
        )
        await _wait_for_selector(
            modal, pilot, f"#prompt-collection-manager-row-{final_id}"
        )
        row = modal.query_one(f"#prompt-collection-manager-row-{final_id}", Button)
        assert str(row.label) == "Boundary 107"
        assert modal._catalog.total == 107
        assert modal._catalog.has_more is False
        load_more = modal.query_one("#prompt-collection-manager-load-more", Button)
        assert load_more.display is False
        assert load_more.disabled is True


@pytest.mark.asyncio
async def test_library_screen_manager_create_search_rename_and_explicit_all(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _prompt_uuid, _message = db.add_prompt(
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
            lambda: _modal_outcome(host.screen) == "Collection created.",
            message="collection create outcome never settled",
        )
        created_page = await service.list_prompt_collections(
            mode="local", query=created_name, limit=100, offset=0
        )
        assert created_page["total"] == 1
        collection_id = created_page["collections"][0]["collection_id"]
        await service.replace_prompt_collection_memberships(
            mode="local",
            prompt_id=prompt_id,
            collection_ids=[collection_id],
        )

        host.screen.query_one(
            "#prompt-collection-manager-new-name", Input
        ).value = created_name.swapcase()
        host.screen.query_one("#prompt-collection-manager-create", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: (
                _modal_outcome(host.screen) == "Name already exists — choose another."
            ),
            message="case-collision outcome never settled",
        )
        assert (
            host.screen.query_one("#prompt-collection-manager-retry", Button).display
            is False
        )
        await _wait_for_condition(
            pilot,
            lambda: (
                host.screen.query_one(
                    "#prompt-collection-manager-new-name", Input
                ).has_focus
            ),
            message="collision recovery focus did not settle",
        )
        assert (
            await service.list_prompt_collections(
                mode="local", query=created_name, limit=100, offset=0
            )
        )["total"] == 1

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
            lambda: _modal_outcome(host.screen) == "Collection renamed.",
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
        await _wait_for_selector(
            screen, pilot, "#library-prompts-empty-collection-label"
        )
        assert len(screen.query("#library-prompts-collection")) == 0
        assert renamed_name in str(
            screen.query_one(
                "#library-prompts-empty-collection-label", Static
            ).renderable
        )

        screen.query_one("#library-prompts-empty-all-prompts", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_prompt_browse_controller.scope.collection_id is None
                and screen._library_prompt_browse_controller.applied_result is not None
                and screen._library_prompt_browse_controller.applied_result.scope.collection_id
                is None
                and len(screen.query("#library-prompts-collection")) == 1
            ),
            message="empty collection recovery did not restore All prompts",
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
        assert screen._library_prompt_browse_controller.scope.collection_id is None
        await _wait_for_selector(screen, pilot, "#library-prompts-collection")
        assert (
            str(screen.query_one("#library-prompts-collection", Button).label)
            == "collection: All prompts"
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
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_prompt_browse_controller.applied_result is not None
                and screen._library_prompt_browse_controller.applied_result.scope.collection_id
                is None
                and len(screen.query("#library-prompts-collection")) == 1
                and str(
                    screen.query_one("#library-prompts-collection", Button).label
                )
                == "collection: All prompts"
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
        membership_summary = str(
            screen.query_one("#library-prompt-memberships-summary", Static).render()
        )
        assert "First" in membership_summary
        assert "Collection #" not in membership_summary
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


@pytest.mark.asyncio
async def test_membership_apply_refreshes_retained_items_before_clean_back_to_list(
    tmp_path, monkeypatch
):
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _prompt_uuid, _message = db.add_prompt(
        name="Deferred refresh prompt",
        author="A",
        details="Before",
        system_prompt="System",
        user_prompt="User",
    )
    first = await service.create_prompt_collection(
        mode="local", name="First", prompt_ids=[prompt_id]
    )
    second = await service.create_prompt_collection(mode="local", name="Second")
    browse_prompt = service.browse_prompts
    browse_calls: list[dict[str, Any]] = []

    async def recording_browse(**kwargs):
        browse_calls.append(dict(kwargs))
        return await browse_prompt(**kwargs)

    monkeypatch.setattr(service, "browse_prompts", recording_browse)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-prompts", Button).press()
        await _wait_for_selector(screen, pilot, f"#library-prompt-row-{prompt_id}")
        screen._apply_library_prompt_collection(first["collection_id"])
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_prompt_browse_controller.scope.collection_id
                == first["collection_id"]
                and screen._library_prompt_browse_controller.result.status == "ready"
            ),
            message="exact starting collection scope did not settle",
        )
        await _wait_for_selector(screen, pilot, f"#library-prompt-row-{prompt_id}")
        screen.query_one(f"#library-prompt-row-{prompt_id}", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_prompt_collections_controller.membership_state.status
                == "ready"
            ),
            message="memberships did not load before retained Items refresh test",
        )
        scope = screen._library_prompt_browse_controller.scope
        browse_before_apply = len(browse_calls)
        count_refreshes: list[str] = []
        screen._refresh_local_source_snapshot = lambda: count_refreshes.append("count")

        screen._library_prompt_collections_controller.stage_memberships(
            (second["collection_id"],)
        )
        screen.query_one("#library-prompt-memberships-apply", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_prompt_collections_controller.membership_state.status
                == "success"
                and len(browse_calls) == browse_before_apply + 1
                and screen._library_prompt_browse_controller.result.status
                == "empty_collection"
            ),
            message="membership Apply did not refresh retained Items",
        )
        assert screen._library_prompts_view == "editor"
        assert screen._selected_prompt_id == prompt_id
        assert screen.query_one("#library-prompt-name", Input).value == (
            "Deferred refresh prompt"
        )
        assert browse_calls[-1] == {
            "mode": "local",
            "query": scope.query,
            "collection_id": scope.collection_id,
            "sort_by": scope.sort_by,
            "sort_order": scope.sort_order,
            "page": scope.page,
            "page_size": scope.page_size,
        }
        assert count_refreshes == ["count"]
        assert screen._library_prompt_status == ""

        screen.query_one("#library-prompt-back", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_prompts_view == "list"
                and len(browse_calls) == browse_before_apply + 2
                and screen._library_prompt_browse_controller.result.status
                == "empty_collection"
            ),
            message=lambda: (
                "Back to list did not dispatch its exact list-entry browse: "
                f"view={screen._library_prompts_view!r}, "
                f"calls={browse_calls[browse_before_apply:]!r}, "
                f"scope={screen._library_prompt_browse_controller.scope!r}, "
                f"result={screen._library_prompt_browse_controller.result!r}"
            ),
        )
        assert browse_calls[-1] == {
            "mode": "local",
            "query": scope.query,
            "collection_id": scope.collection_id,
            "sort_by": scope.sort_by,
            "sort_order": scope.sort_order,
            "page": scope.page,
            "page_size": scope.page_size,
        }
        assert screen._library_prompt_browse_controller.result.total_items == 0
        assert len(screen.query(f"#library-prompt-row-{prompt_id}")) == 0
        assert (
            await service.list_prompt_collection_memberships(
                mode="local", prompt_id=prompt_id
            )
        )["collection_ids"] == (second["collection_id"],)
        assert count_refreshes == ["count", "count"]
        assert screen._library_prompt_status == ""
