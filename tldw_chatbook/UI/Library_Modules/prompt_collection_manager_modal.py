"""Presentation-only local Prompt collection manager modal."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Literal

from loguru import logger
from rich.markup import escape as escape_markup
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import DescendantFocus
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Button, Checkbox, Input, Static

from ...Library.library_prompts_state import (
    PromptCollectionCatalogState,
    begin_prompt_collection_catalog,
    fail_prompt_collection_catalog,
)
from .prompt_collections import (
    PromptCollectionManagerMode,
    PromptCollectionManagerResult,
    PromptCollectionNameConflictError,
    PromptMembershipIdentity,
)
from ...Widgets.modal_dismissal import SafeModalDismissMixin

CatalogLoader = Callable[..., Awaitable[PromptCollectionCatalogState | None]]
CollectionCreator = Callable[[str], Awaitable[PromptCollectionCatalogState | None]]
CollectionRenamer = Callable[[int, str], Awaitable[PromptCollectionCatalogState | None]]

_CATALOG_ERROR = "Couldn't load collections. Retry."
_CREATE_ERROR = "Couldn't create collection. Retry."
_RENAME_ERROR = "Couldn't rename collection. Retry."


class PromptCollectionManagerModal(
    SafeModalDismissMixin, ModalScreen[PromptCollectionManagerResult | None]
):
    """One reusable presentation surface for browse and membership selection."""

    BINDINGS = [("escape", "request_safe_cancel", "Cancel")]
    SAFE_MODAL_CONTENT = "#prompt-collection-manager"

    DEFAULT_CSS = """
    PromptCollectionManagerModal {
        align: center middle;
        background: $background 75%;
    }
    #prompt-collection-manager {
        width: 96;
        max-width: 100%;
        height: 36;
        max-height: 100%;
        min-height: 20;
        background: $panel;
        border: round $accent;
        padding: 0 1;
    }
    #prompt-collection-manager-title,
    #prompt-collection-manager-authority,
    #prompt-collection-manager-outcome {
        height: auto;
    }
    #prompt-collection-manager-rename-target {
        height: 1;
        min-height: 1;
        overflow: hidden;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }
    #prompt-collection-manager-rows {
        height: 1fr;
        min-height: 2;
        background: $surface-darken-1;
    }
    .prompt-collection-manager-row,
    #prompt-collection-manager-all {
        width: 100%;
        height: 1;
        min-height: 1;
    }
    #prompt-collection-manager-load-more,
    #prompt-collection-manager-new-name {
        height: 3;
    }
    .prompt-collection-manager-actions {
        height: 3;
    }
    .prompt-collection-manager-actions Button {
        width: auto;
        min-width: 8;
        height: 3;
        margin-right: 1;
    }
    """

    def __init__(
        self,
        *,
        mode: PromptCollectionManagerMode,
        selected_collection_id: int | None,
        staged_collection_ids: Sequence[int],
        load_catalog: CatalogLoader,
        create_collection: CollectionCreator,
        rename_collection: CollectionRenamer,
        manager_token: int = 1,
        prompt_identity: PromptMembershipIdentity | None = None,
    ) -> None:
        if mode not in {"browse", "membership"}:
            raise ValueError("Unsupported Prompt collection manager mode.")
        super().__init__()
        self._mode = mode
        self._manager_token = manager_token
        self._prompt_identity = prompt_identity
        self._selected_id = selected_collection_id
        self._staged_ids = set(staged_collection_ids)
        self._rename_selected_id = selected_collection_id if mode == "browse" else None
        self._load_catalog_callback = load_catalog
        self._create_collection_callback = create_collection
        self._rename_collection_callback = rename_collection
        self._request_token = 0
        self._catalog = begin_prompt_collection_catalog(query="", request_token=1)
        self._outcome = ""
        self._retry_action: (
            tuple[Literal["load", "create", "rename"], int | None, str] | None
        ) = None
        self._mutation_in_flight = False
        self._mutation_epoch = 0
        self._mutation_close_rejected = False

    def compose(self) -> ComposeResult:
        with Vertical(id="prompt-collection-manager"):
            yield Static(
                "Manage Prompt collections",
                id="prompt-collection-manager-title",
                markup=False,
            )
            yield Static(
                "Local only · Prompt Save is separate",
                id="prompt-collection-manager-authority",
                markup=False,
            )
            yield Input(
                value=self._catalog.query,
                placeholder="Search collections… (Enter)",
                id="prompt-collection-manager-search",
                disabled=self._mutation_in_flight,
            )
            with VerticalScroll(id="prompt-collection-manager-rows"):
                if self._mode == "browse":
                    all_button = Button(
                        "All prompts",
                        id="prompt-collection-manager-all",
                        classes="prompt-collection-manager-row",
                        compact=True,
                        disabled=self._mutation_in_flight,
                    )
                    all_button.set_class(
                        self._selected_id is None,
                        "prompt-collection-manager-selected",
                    )
                    yield all_button
                if self._catalog.status == "loading" and not self._catalog.items:
                    yield Static(
                        "Loading collections…",
                        id="prompt-collection-manager-loading",
                        markup=False,
                    )
                elif self._catalog.status == "error":
                    yield Static(
                        self._catalog.error,
                        id="prompt-collection-manager-error",
                        markup=False,
                    )
                elif self._catalog.status == "empty":
                    yield Static(
                        "No collections match this search.",
                        id="prompt-collection-manager-empty",
                        markup=False,
                    )
                for item in self._catalog.items:
                    label = escape_markup(item.display_name)
                    if self._mode == "membership":
                        checkbox = Checkbox(
                            label,
                            value=item.collection_id in self._staged_ids,
                            id=f"prompt-collection-manager-member-{item.collection_id}",
                            classes="prompt-collection-manager-row",
                            disabled=self._mutation_in_flight,
                            compact=True,
                        )
                        checkbox.collection_id = item.collection_id
                        yield checkbox
                    else:
                        button = Button(
                            label,
                            id=f"prompt-collection-manager-row-{item.collection_id}",
                            classes="prompt-collection-manager-row",
                            compact=True,
                            disabled=self._mutation_in_flight,
                        )
                        button.collection_id = item.collection_id
                        button.set_class(
                            item.collection_id == self._selected_id,
                            "prompt-collection-manager-selected",
                        )
                        yield button
            load_more = Button(
                "Load more",
                id="prompt-collection-manager-load-more",
                compact=True,
                disabled=(
                    self._mutation_in_flight
                    or self._catalog.status == "loading"
                    or not self._catalog.has_more
                ),
            )
            load_more.display = self._catalog.has_more
            yield load_more
            yield Input(
                placeholder="Collection name",
                id="prompt-collection-manager-new-name",
                disabled=self._mutation_in_flight,
            )
            if self._mode == "membership":
                yield Static(
                    self._rename_target_copy(),
                    id="prompt-collection-manager-rename-target",
                    markup=False,
                )
            with Horizontal(classes="prompt-collection-manager-actions"):
                yield Button(
                    "New collection",
                    id="prompt-collection-manager-create",
                    compact=True,
                    disabled=self._mutation_in_flight,
                )
                yield Button(
                    "Rename selected",
                    id="prompt-collection-manager-rename",
                    compact=True,
                    disabled=(
                        self._mutation_in_flight or self._rename_target_id() is None
                    ),
                )
                retry = Button(
                    "Retry",
                    id="prompt-collection-manager-retry",
                    compact=True,
                    disabled=self._mutation_in_flight,
                )
                retry.display = self._retry_action is not None
                yield retry
            yield Static(
                self._outcome,
                id="prompt-collection-manager-outcome",
                markup=False,
            )
            with Horizontal(classes="prompt-collection-manager-actions"):
                yield Button(
                    "Done",
                    id="prompt-collection-manager-done",
                    compact=True,
                    disabled=self._mutation_in_flight,
                )
                yield Button(
                    "Cancel",
                    id="prompt-collection-manager-cancel",
                    compact=True,
                )

    def on_mount(self) -> None:
        self._mutation_epoch += 1
        self._mutation_in_flight = False
        self._mutation_close_rejected = False
        self._start_catalog_load(query="", offset=0)

    def on_unmount(self) -> None:
        self._mutation_epoch += 1
        self._mutation_in_flight = False
        self._mutation_close_rejected = False

    def _focus_control(self, control_id: str) -> None:
        if not self.is_mounted:
            return
        for control in self.query(f"#{control_id}").results(Widget):
            if not control.disabled:
                control.focus()
            break

    def _refresh(self, *, focus_id: str | None = None) -> None:
        self.refresh(recompose=True)
        if focus_id is not None:
            self.call_after_refresh(self._focus_control, focus_id)

    def _collection_control_id(self, collection_id: int) -> str:
        lane = "row" if self._mode == "browse" else "member"
        return f"prompt-collection-manager-{lane}-{collection_id}"

    def _rename_target_id(self) -> int | None:
        if self._mode == "browse":
            return self._selected_id
        return self._rename_selected_id

    def _rename_target_copy(self) -> str:
        collection_id = self._rename_target_id()
        if collection_id is None:
            return "Rename target: choose a collection row"
        label = next(
            (
                item.display_name
                for item in self._catalog.items
                if item.collection_id == collection_id
            ),
            f"Collection #{collection_id}",
        )
        return f"Rename target: {label}"

    def _catalog_focus_id(self, *, offset: int) -> str:
        if offset <= 0:
            return "prompt-collection-manager-search"
        if self._catalog.has_more:
            return "prompt-collection-manager-load-more"
        if offset < len(self._catalog.items):
            return self._collection_control_id(
                self._catalog.items[offset].collection_id
            )
        if self._catalog.items:
            return self._collection_control_id(self._catalog.items[-1].collection_id)
        return "prompt-collection-manager-search"

    def _collection_focus_id(self, collection_id: int) -> str:
        if any(item.collection_id == collection_id for item in self._catalog.items):
            return self._collection_control_id(collection_id)
        return "prompt-collection-manager-search"

    def on_descendant_focus(self, event: DescendantFocus) -> None:
        if self._mode != "membership":
            return
        collection_id = getattr(event.widget, "collection_id", None)
        if type(collection_id) is not int or collection_id <= 0:
            return
        if collection_id == self._rename_selected_id:
            return
        self._rename_selected_id = collection_id
        for target in self.query("#prompt-collection-manager-rename-target").results(
            Static
        ):
            target.update(self._rename_target_copy())
            break
        for rename in self.query("#prompt-collection-manager-rename").results(Button):
            rename.disabled = self._mutation_in_flight
            break

    def _start_catalog_load(self, *, query: str, offset: int) -> None:
        self._request_token += 1
        request_token = self._request_token
        self.run_worker(
            self._load_catalog(query=query, offset=offset, request_token=request_token),
            exclusive=True,
            group=f"prompt-collection-manager-{self._manager_token}-catalog",
        )

    async def _load_catalog(
        self, *, query: str, offset: int, request_token: int
    ) -> None:
        try:
            catalog = await self._load_catalog_callback(query=query, offset=offset)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Library Prompt collection manager failed; operation=load "
                "exception_type={}",
                type(exc).__name__,
            )
            catalog = None
        if request_token != self._request_token or not self.is_mounted:
            return
        if catalog is None:
            current = begin_prompt_collection_catalog(
                query=query,
                request_token=request_token,
                previous=self._catalog if offset > 0 else None,
                append=offset > 0,
            )
            self._catalog = fail_prompt_collection_catalog(
                current, request_token=request_token, error=_CATALOG_ERROR
            )
            self._retry_action = ("load", offset, query)
            self._outcome = _CATALOG_ERROR
        else:
            self._catalog = catalog
            if catalog.status == "error":
                self._retry_action = ("load", catalog.offset, catalog.query)
                self._outcome = catalog.error
            else:
                self._retry_action = None
                self._outcome = ""
            if (
                self._selected_id is not None
                and not catalog.query
                and not any(
                    item.collection_id == self._selected_id for item in catalog.items
                )
                and not catalog.has_more
            ):
                self._selected_id = None
        self._refresh(focus_id=self._catalog_focus_id(offset=offset))

    @on(Input.Submitted, "#prompt-collection-manager-search")
    def _search_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        if self._mutation_in_flight:
            return
        self._outcome = "Searching collections…"
        self._start_catalog_load(query=event.value.strip(), offset=0)

    @on(Button.Pressed, "#prompt-collection-manager-load-more")
    def _load_more(self, event: Button.Pressed) -> None:
        event.stop()
        if self._mutation_in_flight:
            return
        self._outcome = "Loading more collections…"
        self._start_catalog_load(
            query=self._catalog.query, offset=self._catalog.next_offset
        )

    @on(Button.Pressed, "#prompt-collection-manager-all")
    def _select_all(self, event: Button.Pressed) -> None:
        event.stop()
        if self._mutation_in_flight:
            return
        self._selected_id = None
        self._refresh(focus_id="prompt-collection-manager-all")

    @on(Button.Pressed, ".prompt-collection-manager-row")
    def _select_row(self, event: Button.Pressed) -> None:
        if (
            self._mutation_in_flight
            or self._mode != "browse"
            or event.button.id == "prompt-collection-manager-all"
        ):
            return
        event.stop()
        collection_id = getattr(event.button, "collection_id", None)
        if type(collection_id) is int:
            self._selected_id = collection_id
            self._refresh(focus_id=self._collection_control_id(collection_id))

    @on(Checkbox.Changed, ".prompt-collection-manager-row")
    def _membership_changed(self, event: Checkbox.Changed) -> None:
        if self._mutation_in_flight or self._mode != "membership":
            return
        event.stop()
        collection_id = getattr(event.checkbox, "collection_id", None)
        if type(collection_id) is not int:
            return
        if event.value:
            self._staged_ids.add(collection_id)
        else:
            self._staged_ids.discard(collection_id)
        try:
            rename = self.query_one("#prompt-collection-manager-rename", Button)
        except Exception:
            return
        rename.disabled = self._rename_target_id() is None

    def _name_value(self) -> str:
        return self.query_one(
            "#prompt-collection-manager-new-name", Input
        ).value.strip()

    @on(Button.Pressed, "#prompt-collection-manager-create")
    def _create(self, event: Button.Pressed) -> None:
        event.stop()
        self._start_mutation("create", None, self._name_value())

    @on(Button.Pressed, "#prompt-collection-manager-rename")
    def _rename(self, event: Button.Pressed) -> None:
        event.stop()
        self._start_mutation("rename", self._rename_target_id(), self._name_value())

    def _start_mutation(
        self, action: Literal["create", "rename"], collection_id: int | None, name: str
    ) -> None:
        if self._mutation_in_flight:
            return
        if not name:
            self._outcome = "Collection name is required."
            self._retry_action = None
            self._refresh(focus_id="prompt-collection-manager-new-name")
            return
        if action == "rename" and collection_id is None:
            self._outcome = "Choose exactly one collection to rename."
            self._retry_action = None
            self._refresh(focus_id="prompt-collection-manager-new-name")
            return
        self._request_token += 1
        self._mutation_epoch += 1
        self._mutation_in_flight = True
        self._mutation_close_rejected = False
        self._outcome = (
            "Creating collection…" if action == "create" else "Renaming collection…"
        )
        self._retry_action = None
        self._refresh()
        mutation_epoch = self._mutation_epoch
        mount_generation = self._safe_mount_generation
        self.run_worker(
            self._run_mutation(
                action,
                collection_id,
                name,
                mutation_epoch=mutation_epoch,
                mount_generation=mount_generation,
            ),
            group=f"prompt-collection-manager-{self._manager_token}-mutation",
        )

    def _mutation_is_current(
        self, *, mutation_epoch: int, mount_generation: int
    ) -> bool:
        return (
            self.is_mounted
            and self._mutation_epoch == mutation_epoch
            and self._safe_mount_generation == mount_generation
        )

    async def _run_mutation(
        self,
        action: Literal["create", "rename"],
        collection_id: int | None,
        name: str,
        *,
        mutation_epoch: int,
        mount_generation: int,
    ) -> None:
        cancelled_error: asyncio.CancelledError | None = None
        try:
            if action == "create":
                catalog = await self._create_collection_callback(name)
            else:
                catalog = await self._rename_collection_callback(collection_id, name)  # type: ignore[arg-type]
        except asyncio.CancelledError as exc:
            if not self._mutation_is_current(
                mutation_epoch=mutation_epoch, mount_generation=mount_generation
            ):
                raise
            cancelled_error = exc
            catalog = None
        except PromptCollectionNameConflictError:
            if not self._mutation_is_current(
                mutation_epoch=mutation_epoch, mount_generation=mount_generation
            ):
                return
            self._mutation_in_flight = False
            self._mutation_close_rejected = False
            self._outcome = "Name already exists — choose another."
            self._retry_action = None
            self._refresh(focus_id="prompt-collection-manager-new-name")
            return
        except Exception as exc:
            logger.warning(
                "Library Prompt collection manager failed; operation={} "
                "exception_type={}",
                action,
                type(exc).__name__,
            )
            catalog = None
        if not self._mutation_is_current(
            mutation_epoch=mutation_epoch, mount_generation=mount_generation
        ):
            return
        self._mutation_in_flight = False
        self._mutation_close_rejected = False
        if catalog is None:
            self._outcome = _CREATE_ERROR if action == "create" else _RENAME_ERROR
            self._retry_action = (action, collection_id, name)
            self._refresh(focus_id="prompt-collection-manager-retry")
            if cancelled_error is not None:
                raise cancelled_error
            return
        self._catalog = catalog
        success = "Collection created." if action == "create" else "Collection renamed."
        if catalog.status == "error":
            self._outcome = f"{success} Couldn't refresh collections — Retry catalog."
            self._retry_action = ("load", catalog.offset, catalog.query)
            self._refresh(focus_id="prompt-collection-manager-retry")
            return
        self._outcome = success
        self._retry_action = None
        self._refresh(
            focus_id=(
                "prompt-collection-manager-new-name"
                if action == "create"
                else self._collection_focus_id(collection_id)  # type: ignore[arg-type]
            )
        )

    @on(Button.Pressed, "#prompt-collection-manager-retry")
    def _retry(self, event: Button.Pressed) -> None:
        event.stop()
        if self._mutation_in_flight:
            return
        retry = self._retry_action
        if retry is None:
            return
        action, collection_id, value = retry
        if action == "load":
            self._start_catalog_load(query=value, offset=collection_id or 0)
            return
        self._start_mutation(action, collection_id, value)

    @on(Button.Pressed, "#prompt-collection-manager-done")
    def _done(self, event: Button.Pressed) -> None:
        event.stop()
        if self._mutation_in_flight:
            return
        self._request_token += 1
        self.dismiss(
            PromptCollectionManagerResult(
                mode=self._mode,
                manager_token=self._manager_token,
                selected_collection_id=(
                    self._selected_id if self._mode == "browse" else None
                ),
                staged_collection_ids=tuple(sorted(self._staged_ids)),
                prompt_identity=self._prompt_identity,
            )
        )

    @on(Button.Pressed, "#prompt-collection-manager-cancel")
    async def _cancel_button(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="visible")

    async def _perform_safe_cancel(self, *, source: str) -> None:
        del source
        if self._mutation_in_flight:
            if not self._mutation_close_rejected:
                self._mutation_close_rejected = True
                self._outcome = "Finish the current collection change before closing."
                for outcome in self.query("#prompt-collection-manager-outcome").results(
                    Static
                ):
                    outcome.update(self._outcome)
                    break
            return
        self._request_token += 1
        self.dismiss_safe_once(None)
