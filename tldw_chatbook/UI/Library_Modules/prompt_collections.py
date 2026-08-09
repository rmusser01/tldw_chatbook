"""Local Prompt collection state orchestration and manager coordination."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from loguru import logger

from ...Library.library_prompts_state import (
    PromptCollectionCatalogState,
    PromptCollectionOption,
    PromptMembershipState,
    apply_prompt_collection_catalog_page,
    apply_prompt_memberships_loaded,
    apply_prompt_memberships_saved,
    begin_prompt_collection_catalog,
    begin_prompt_memberships,
    begin_prompt_memberships_apply,
    disable_prompt_memberships,
    fail_prompt_collection_catalog,
    fail_prompt_memberships,
    stage_prompt_memberships,
)

PromptCollectionManagerMode = Literal["browse", "membership"]

_CATALOG_ERROR = "Couldn't load collections. Retry."
_MEMBERSHIP_LOAD_ERROR = "Couldn't load memberships. Retry."
_MEMBERSHIP_APPLY_ERROR = "Couldn't apply memberships. Retry."
_UNSAVED_REASON = "Save this prompt before managing collections."
_RESERVED_NAME_ERROR = "This prompt collection name is reserved."


class PromptCollectionNameConflictError(ValueError):
    """Fixed local classification for a deterministic reserved name."""


def _membership_response_ids(
    response: Any, *, identity: PromptMembershipIdentity
) -> tuple[int, ...]:
    if not isinstance(response, Mapping):
        raise TypeError("Prompt membership response must be a mapping.")
    prompt_id = response.get("prompt_id")
    if type(prompt_id) is not int or prompt_id <= 0 or prompt_id != identity.prompt_id:
        raise ValueError("Prompt membership response identity does not match.")
    collection_ids = response.get("collection_ids")
    if not isinstance(collection_ids, Sequence) or isinstance(
        collection_ids, (str, bytes)
    ):
        raise TypeError("Prompt membership collection_ids must be a sequence.")
    ids = tuple(collection_ids)
    if any(
        type(collection_id) is not int or collection_id <= 0 for collection_id in ids
    ):
        raise ValueError("Prompt membership collection IDs must be positive integers.")
    if tuple(sorted(set(ids))) != ids:
        raise ValueError("Prompt membership collection IDs must be sorted and unique.")
    return ids


@dataclass(frozen=True)
class PromptMembershipIdentity:
    """Exact persisted local Prompt identity guarding membership requests."""

    prompt_id: int
    fingerprint: str
    backend: Literal["local"] = "local"

    def __post_init__(self) -> None:
        if type(self.prompt_id) is not int or self.prompt_id <= 0:
            raise ValueError("prompt_id must be a positive integer.")
        if not isinstance(self.fingerprint, str) or not self.fingerprint:
            raise ValueError("fingerprint is required.")
        if self.backend != "local":
            raise ValueError("Prompt collection membership is local-only.")


@dataclass(frozen=True)
class PromptCollectionManagerResult:
    """Presentation-only result from one immutable manager lane."""

    mode: PromptCollectionManagerMode
    manager_token: int
    selected_collection_id: int | None
    staged_collection_ids: tuple[int, ...]
    prompt_identity: PromptMembershipIdentity | None = None


@dataclass(frozen=True)
class PromptCollectionManagerSession:
    """One controller-owned modal session protected from lane and ABA drift."""

    mode: PromptCollectionManagerMode
    manager_token: int
    prompt_identity: PromptMembershipIdentity | None


class LibraryPromptCollectionsController:
    """Own local collection services, stale guards, and shared manager flow."""

    def __init__(
        self,
        *,
        run_service_call: Callable[[], Callable[..., Awaitable[Any]]],
        prompt_service: Callable[[], Any],
        sync_memberships: Callable[[], Callable[[PromptMembershipState], None]],
        current_prompt_id: Callable[[], int | None],
        current_prompt_detail: Callable[[], Mapping[str, Any] | None],
        prompt_editor_active: Callable[[], bool],
        push_modal: Callable[[], Callable[[Any, Callable[..., Any]], Any]]
        | None = None,
        current_browse_collection_id: Callable[[], int | None] | None = None,
        apply_browse_collection: Callable[[], Callable[[int | None], None]]
        | None = None,
        refresh_browse_projection: Callable[[], Callable[[], None]] | None = None,
        membership_applied: Callable[[], Callable[[], None]] | None = None,
    ) -> None:
        """Initialize the local collection coordinator.

        Args:
            run_service_call: Late-bound off-loop service dispatcher.
            prompt_service: Late-bound Prompt scope service provider.
            sync_memberships: Late-bound membership projection callback.
            current_prompt_id: Current editor Prompt ID provider.
            current_prompt_detail: Current editor record provider.
            prompt_editor_active: Whether the Prompt editor is current.
            push_modal: Optional late-bound modal presenter.
            current_browse_collection_id: Active browse collection provider.
            apply_browse_collection: Browse selection callback provider.
            refresh_browse_projection: Targeted collection-label refresh provider.
            membership_applied: Successful membership-apply callback provider.
        """
        self._run_service_call = run_service_call
        self._prompt_service = prompt_service
        self._sync_memberships = sync_memberships
        self._current_prompt_id = current_prompt_id
        self._current_prompt_detail = current_prompt_detail
        self._prompt_editor_active = prompt_editor_active
        self._push_modal = push_modal
        self._current_browse_collection_id = current_browse_collection_id
        self._apply_browse_collection = apply_browse_collection
        self._refresh_browse_projection = refresh_browse_projection
        self._membership_applied = membership_applied
        self._request_counter = 0
        self._manager_counter = 0
        self._identity_generation = 1
        self._active_manager: tuple[int, PromptCollectionManagerMode] | None = None
        self._label_cache: dict[int, str] = {}
        self.catalog_state = begin_prompt_collection_catalog(
            query="", request_token=self._next_request_token()
        )
        self.membership_state = disable_prompt_memberships(_UNSAVED_REASON)

    def _next_request_token(self) -> int:
        self._request_counter += 1
        return self._request_counter

    def begin_manager(self, mode: PromptCollectionManagerMode = "browse") -> int:
        """Open one manager lane, invalidating an older dialog immediately.

        Args:
            mode: Browse or membership manager lane.

        Returns:
            Monotonic modal session token.

        Raises:
            ValueError: If ``mode`` is unsupported.
        """
        if mode not in {"browse", "membership"}:
            raise ValueError("Unsupported Prompt collection manager mode.")
        self._manager_counter += 1
        self._active_manager = (self._manager_counter, mode)
        return self._manager_counter

    def manager_is_active(
        self, manager_token: int, mode: PromptCollectionManagerMode | None = None
    ) -> bool:
        """Check whether one manager session remains authoritative.

        Args:
            manager_token: Session token to check.
            mode: Optional expected manager lane.

        Returns:
            ``True`` only for the current matching session.
        """
        active = self._active_manager
        return (
            active is not None
            and active[0] == manager_token
            and (mode is None or active[1] == mode)
        )

    def manager_context_is_active(
        self,
        manager_token: int,
        *,
        mode: PromptCollectionManagerMode,
        prompt_identity: PromptMembershipIdentity | None,
    ) -> bool:
        """Validate one dialog lane and, for membership, its Prompt owner.

        Args:
            manager_token: Session token to check.
            mode: Expected manager lane.
            prompt_identity: Expected membership Prompt identity, if any.

        Returns:
            ``True`` only while the lane and identity remain current.
        """
        return self.manager_is_active(manager_token, mode) and (
            mode == "browse" or self._current_identity() == prompt_identity
        )

    def end_manager(
        self, manager_token: int, mode: PromptCollectionManagerMode | None = None
    ) -> None:
        """End the matching manager session.

        Args:
            manager_token: Session token to close.
            mode: Optional expected manager lane.
        """
        if self.manager_is_active(manager_token, mode):
            self._active_manager = None

    def invalidate(self, reason: str = _UNSAVED_REASON) -> None:
        """Invalidate requests and modal authority after navigation.

        Args:
            reason: Truthful disabled-membership explanation.
        """
        self._next_request_token()
        self._identity_generation += 1
        self._active_manager = None
        self.membership_state = disable_prompt_memberships(reason)
        self._sync_memberships()(self.membership_state)

    def identity_for(self, prompt_id: int) -> PromptMembershipIdentity:
        """Bind one local Prompt ID to the current editor session.

        Args:
            prompt_id: Persisted positive local Prompt ID.

        Returns:
            Session-scoped membership identity.

        Raises:
            ValueError: If ``prompt_id`` is not positive.
        """
        return PromptMembershipIdentity(
            prompt_id,
            f"local:prompt:{prompt_id}:session:{self._identity_generation}",
        )

    def _current_identity(self) -> PromptMembershipIdentity | None:
        detail = self._current_prompt_detail()
        prompt_id = self._current_prompt_id()
        if (
            not self._prompt_editor_active()
            or type(prompt_id) is not int
            or prompt_id <= 0
            or not isinstance(detail, Mapping)
            or str(detail.get("backend") or "local").strip().lower() != "local"
            or bool(detail.get("deleted"))
        ):
            return None
        return self.identity_for(prompt_id)

    def open_manager(self, mode: PromptCollectionManagerMode) -> int | None:
        """Build and present the shared manager with an immutable session.

        Args:
            mode: Browse or membership manager lane.

        Returns:
            Session token, or ``None`` when the lane cannot open.
        """
        prompt_identity = self._current_identity() if mode == "membership" else None
        if mode == "membership":
            if prompt_identity is None:
                self.invalidate()
                return None
            state = self.membership_state
            if (
                not state.can_manage
                or state.prompt_id != prompt_identity.prompt_id
                or state.identity_fingerprint != prompt_identity.fingerprint
            ):
                return None
        manager_token = self.begin_manager(mode)
        session = PromptCollectionManagerSession(
            mode=mode,
            manager_token=manager_token,
            prompt_identity=prompt_identity,
        )

        async def load_catalog(*, query: str, offset: int):
            return await self.load_catalog(
                manager_token=manager_token,
                manager_mode=mode,
                prompt_identity=prompt_identity,
                query=query,
                offset=offset,
            )

        async def create_collection(name: str):
            return await self.create_collection(
                manager_token=manager_token,
                manager_mode=mode,
                prompt_identity=prompt_identity,
                name=name,
            )

        async def rename_collection(collection_id: int, name: str):
            return await self.rename_collection(
                manager_token=manager_token,
                manager_mode=mode,
                prompt_identity=prompt_identity,
                collection_id=collection_id,
                name=name,
            )

        from .prompt_collection_manager_modal import PromptCollectionManagerModal

        modal = PromptCollectionManagerModal(
            mode=mode,
            manager_token=manager_token,
            prompt_identity=prompt_identity,
            selected_collection_id=(
                self._current_browse_collection_id()
                if mode == "browse" and self._current_browse_collection_id is not None
                else None
            ),
            staged_collection_ids=(
                self.membership_state.staged_ids if mode == "membership" else ()
            ),
            load_catalog=load_catalog,
            create_collection=create_collection,
            rename_collection=rename_collection,
        )
        if self._push_modal is None:
            self.end_manager(manager_token, mode)
            return None
        self._push_modal()(
            modal,
            lambda result: self._complete_manager(result, session=session),
        )
        return manager_token

    def _complete_manager(
        self,
        result: PromptCollectionManagerResult | None,
        *,
        session: PromptCollectionManagerSession,
    ) -> None:
        if not self.manager_context_is_active(
            session.manager_token,
            mode=session.mode,
            prompt_identity=session.prompt_identity,
        ):
            return
        if result is not None and (
            result.manager_token != session.manager_token
            or result.mode != session.mode
            or result.prompt_identity != session.prompt_identity
        ):
            self.end_manager(session.manager_token, session.mode)
            return
        if result is None:
            if session.mode == "browse" and self._refresh_browse_projection is not None:
                self._refresh_browse_projection()()
        elif session.mode == "membership":
            self.stage_memberships(result.staged_collection_ids)
        elif self._apply_browse_collection is not None:
            self._apply_browse_collection()(result.selected_collection_id)
        self.end_manager(session.manager_token, session.mode)

    def _manager_request_is_active(
        self,
        manager_token: int,
        manager_mode: PromptCollectionManagerMode | None,
        prompt_identity: PromptMembershipIdentity | None,
    ) -> bool:
        if manager_mode is None:
            return self.manager_is_active(manager_token)
        return self.manager_context_is_active(
            manager_token,
            mode=manager_mode,
            prompt_identity=prompt_identity,
        )

    def _remember_catalog(self, state: PromptCollectionCatalogState) -> None:
        self.catalog_state = state
        self._label_cache.update(
            (item.collection_id, item.display_name) for item in state.items
        )
        self._refresh_membership_labels()

    def _refresh_membership_labels(self) -> None:
        if self.membership_state.prompt_id is None:
            return
        ids = set(self.membership_state.applied_ids) | set(
            self.membership_state.staged_ids
        )
        labels = tuple(
            (collection_id, self._label_cache[collection_id])
            for collection_id in sorted(ids)
            if collection_id in self._label_cache
        )
        updated = replace(self.membership_state, labels=labels)
        if updated != self.membership_state:
            self.membership_state = updated
            self._sync_memberships()(updated)

    async def load_catalog(
        self,
        *,
        manager_token: int,
        query: str,
        offset: int,
        manager_mode: PromptCollectionManagerMode | None = None,
        prompt_identity: PromptMembershipIdentity | None = None,
    ) -> PromptCollectionCatalogState | None:
        """Load one bounded local page, rejecting stale completions.

        Args:
            manager_token: Authoritative manager session token.
            query: Collection name search.
            offset: Exact page offset to request.
            manager_mode: Optional expected manager lane.
            prompt_identity: Expected membership Prompt identity, if any.

        Returns:
            Settled catalog state, or ``None`` when stale.
        """
        if not self._manager_request_is_active(
            manager_token, manager_mode, prompt_identity
        ):
            return None
        query = query.strip()
        append = offset > 0
        request_token = self._next_request_token()
        try:
            loading = begin_prompt_collection_catalog(
                query=query,
                request_token=request_token,
                previous=self.catalog_state if append else None,
                append=append,
            )
        except ValueError:
            return None
        self.catalog_state = loading
        service = self._prompt_service()
        list_collections = getattr(service, "list_prompt_collections", None)
        if not callable(list_collections):
            failed = fail_prompt_collection_catalog(
                loading, request_token=request_token, error=_CATALOG_ERROR
            )
            self._remember_catalog(failed)
            return failed
        try:
            response = await self._run_service_call()(
                list_collections,
                mode="local",
                query=query,
                limit=100,
                offset=offset,
                isolate_in_worker=True,
            )
            settled = apply_prompt_collection_catalog_page(
                loading,
                response,
                request_token=request_token,
                append=append,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Library Prompt collections failed; operation=list exception_type={}",
                type(exc).__name__,
            )
            settled = fail_prompt_collection_catalog(
                loading, request_token=request_token, error=_CATALOG_ERROR
            )
        if (
            not self._manager_request_is_active(
                manager_token, manager_mode, prompt_identity
            )
            or self.catalog_state.request_token != request_token
        ):
            return None
        self._remember_catalog(settled)
        return settled

    async def create_collection(
        self,
        *,
        manager_token: int,
        name: str,
        manager_mode: PromptCollectionManagerMode | None = None,
        prompt_identity: PromptMembershipIdentity | None = None,
    ) -> PromptCollectionCatalogState | None:
        """Create one local collection and refresh the catalog.

        Args:
            manager_token: Authoritative manager session token.
            name: Collection name after trimming.
            manager_mode: Optional expected manager lane.
            prompt_identity: Expected membership Prompt identity, if any.

        Returns:
            Refreshed catalog state, or ``None`` when stale.

        Raises:
            ValueError: If input or the service response is invalid.
            PromptCollectionNameConflictError: If the name already exists.
        """
        name = name.strip()
        if not name:
            raise ValueError("Collection name is required.")
        if not self._manager_request_is_active(
            manager_token, manager_mode, prompt_identity
        ):
            return None
        service = self._prompt_service()
        create = getattr(service, "create_prompt_collection", None)
        if not callable(create):
            raise ValueError("Prompt collection creation is unavailable.")
        try:
            await self._run_service_call()(
                create,
                mode="local",
                name=name,
                isolate_in_worker=True,
            )
        except ValueError as exc:
            if exc.args == (_RESERVED_NAME_ERROR,):
                raise PromptCollectionNameConflictError from None
            raise
        if not self._manager_request_is_active(
            manager_token, manager_mode, prompt_identity
        ):
            return None
        return await self.load_catalog(
            manager_token=manager_token,
            query="",
            offset=0,
            manager_mode=manager_mode,
            prompt_identity=prompt_identity,
        )

    async def rename_collection(
        self,
        *,
        manager_token: int,
        collection_id: int,
        name: str,
        manager_mode: PromptCollectionManagerMode | None = None,
        prompt_identity: PromptMembershipIdentity | None = None,
    ) -> PromptCollectionCatalogState | None:
        """Rename one local collection and refresh cached labels.

        Args:
            manager_token: Authoritative manager session token.
            collection_id: Positive local collection ID.
            name: Replacement name after trimming.
            manager_mode: Optional expected manager lane.
            prompt_identity: Expected membership Prompt identity, if any.

        Returns:
            Refreshed catalog state, or ``None`` when stale.

        Raises:
            ValueError: If input or the service response is invalid.
            PromptCollectionNameConflictError: If the name already exists.
        """
        if type(collection_id) is not int or collection_id <= 0:
            raise ValueError("Choose one collection to rename.")
        name = name.strip()
        if not name:
            raise ValueError("Collection name is required.")
        if not self._manager_request_is_active(
            manager_token, manager_mode, prompt_identity
        ):
            return None
        service = self._prompt_service()
        rename = getattr(service, "update_prompt_collection", None)
        if not callable(rename):
            raise ValueError("Prompt collection rename is unavailable.")
        try:
            response = await self._run_service_call()(
                rename,
                mode="local",
                collection_id=collection_id,
                name=name,
                isolate_in_worker=True,
            )
        except ValueError as exc:
            if exc.args == (_RESERVED_NAME_ERROR,):
                raise PromptCollectionNameConflictError from None
            raise
        if not self._manager_request_is_active(
            manager_token, manager_mode, prompt_identity
        ):
            return None
        if not isinstance(response, Mapping) or response.get("backend") != "local":
            raise ValueError("Prompt collection rename returned an invalid record.")
        option = PromptCollectionOption(
            collection_id=response.get("collection_id"),
            name=response.get("name"),
            display_name=response.get("display_name") or response.get("name"),
        )
        if option.collection_id != collection_id:
            raise ValueError("Prompt collection rename returned the wrong collection.")
        self._label_cache[collection_id] = option.display_name
        self._refresh_membership_labels()
        if (
            self._current_browse_collection_id is not None
            and self._current_browse_collection_id() == collection_id
            and self._refresh_browse_projection is not None
        ):
            self._refresh_browse_projection()()
        return await self.load_catalog(
            manager_token=manager_token,
            query="",
            offset=0,
            manager_mode=manager_mode,
            prompt_identity=prompt_identity,
        )

    def collection_label(self, collection_id: int | None) -> str:
        """Return the current literal label for one browse selection.

        Args:
            collection_id: Local collection ID, or ``None`` for all Prompts.

        Returns:
            Cached display label or a stable fallback.
        """
        if collection_id is None:
            return "All prompts"
        return self._label_cache.get(collection_id, f"Collection #{collection_id}")

    async def _hydrate_membership_labels(
        self,
        service: Any,
        *,
        identity: PromptMembershipIdentity,
        request_token: int,
        collection_ids: Sequence[int],
    ) -> Mapping[int, str] | None:
        list_collections = getattr(service, "list_prompt_collections", None)
        if not callable(list_collections):
            return self._label_cache
        missing = set(collection_ids).difference(self._label_cache)
        catalog: PromptCollectionCatalogState | None = None
        offset = 0
        while missing:
            if (
                self._current_identity() != identity
                or self.membership_state.request_token != request_token
            ):
                return None
            try:
                response = await self._run_service_call()(
                    list_collections,
                    mode="local",
                    query="",
                    limit=100,
                    offset=offset,
                    isolate_in_worker=True,
                )
                if (
                    self._current_identity() != identity
                    or self.membership_state.request_token != request_token
                ):
                    return None
                loading = begin_prompt_collection_catalog(
                    query="",
                    request_token=request_token,
                    previous=catalog,
                    append=catalog is not None,
                )
                catalog = apply_prompt_collection_catalog_page(
                    loading,
                    response,
                    request_token=request_token,
                    append=catalog is not None,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                break
            self._label_cache.update(
                (item.collection_id, item.display_name) for item in catalog.items
            )
            missing.difference_update(self._label_cache)
            if not catalog.has_more:
                break
            next_offset = catalog.next_offset
            if next_offset <= offset:
                break
            offset = next_offset
        return self._label_cache

    async def load_memberships(self) -> None:
        """Load memberships for the exact current persisted local Prompt.

        Returns:
            None. Settled state is delivered through ``sync_memberships``.
        """
        identity = self._current_identity()
        if identity is None:
            self.invalidate()
            return
        request_token = self._next_request_token()
        loading = begin_prompt_memberships(
            prompt_id=identity.prompt_id,
            identity_fingerprint=identity.fingerprint,
            request_token=request_token,
        )
        self.membership_state = loading
        self._sync_memberships()(loading)
        service = self._prompt_service()
        list_memberships = getattr(service, "list_prompt_collection_memberships", None)
        if not callable(list_memberships):
            failed = fail_prompt_memberships(
                loading,
                request_token=request_token,
                error=_MEMBERSHIP_LOAD_ERROR,
                phase="load",
            )
            self.membership_state = failed
            self._sync_memberships()(failed)
            return
        try:
            response = await self._run_service_call()(
                list_memberships,
                mode="local",
                prompt_id=identity.prompt_id,
                isolate_in_worker=True,
            )
            collection_ids = _membership_response_ids(response, identity=identity)
            labels = await self._hydrate_membership_labels(
                service,
                identity=identity,
                request_token=request_token,
                collection_ids=collection_ids,
            )
            if labels is None:
                return
            settled = apply_prompt_memberships_loaded(
                loading,
                collection_ids=collection_ids,
                labels=labels,
                request_token=request_token,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Library Prompt collections failed; operation=list_memberships "
                "exception_type={}",
                type(exc).__name__,
            )
            settled = fail_prompt_memberships(
                loading,
                request_token=request_token,
                error=_MEMBERSHIP_LOAD_ERROR,
                phase="load",
            )
        if (
            self._current_identity() != identity
            or self.membership_state.request_token != request_token
        ):
            return
        self.membership_state = settled
        self._sync_memberships()(settled)

    def disable_memberships(self, reason: str = _UNSAVED_REASON) -> None:
        """Disable membership management and invalidate pending work.

        Args:
            reason: Truthful disabled-state explanation.
        """
        self.invalidate(reason)

    def stage_memberships(self, collection_ids: Sequence[int]) -> None:
        """Stage a complete membership set without persisting it.

        Args:
            collection_ids: Positive collection IDs to stage.

        Raises:
            ValueError: If IDs or the current membership state are invalid.
        """
        self.membership_state = stage_prompt_memberships(
            self.membership_state, collection_ids
        )
        self._sync_memberships()(self.membership_state)

    async def apply_memberships(self) -> None:
        """Apply the staged set only for the current persisted Prompt.

        Returns:
            None. Settled state is delivered through ``sync_memberships``.
        """
        identity = self._current_identity()
        state = self.membership_state
        if (
            identity is None
            or state.prompt_id != identity.prompt_id
            or state.identity_fingerprint != identity.fingerprint
        ):
            self.invalidate()
            return
        request_token = self._next_request_token()
        applying = begin_prompt_memberships_apply(state, request_token=request_token)
        if applying is state:
            return
        self.membership_state = applying
        self._sync_memberships()(applying)
        service = self._prompt_service()
        replace_memberships = getattr(
            service, "replace_prompt_collection_memberships", None
        )
        if not callable(replace_memberships):
            failed = fail_prompt_memberships(
                applying,
                request_token=request_token,
                error=_MEMBERSHIP_APPLY_ERROR,
                phase="apply",
            )
            self.membership_state = failed
            self._sync_memberships()(failed)
            return
        try:
            response = await self._run_service_call()(
                replace_memberships,
                mode="local",
                prompt_id=identity.prompt_id,
                collection_ids=applying.staged_ids,
                isolate_in_worker=True,
            )
            settled = apply_prompt_memberships_saved(
                applying,
                collection_ids=_membership_response_ids(response, identity=identity),
                request_token=request_token,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Library Prompt collections failed; operation=replace_memberships "
                "exception_type={}",
                type(exc).__name__,
            )
            settled = fail_prompt_memberships(
                applying,
                request_token=request_token,
                error=_MEMBERSHIP_APPLY_ERROR,
                phase="apply",
            )
        if (
            self._current_identity() != identity
            or self.membership_state.request_token != request_token
        ):
            return
        self.membership_state = settled
        self._sync_memberships()(settled)
        if settled.status == "success" and self._membership_applied is not None:
            self._membership_applied()()
