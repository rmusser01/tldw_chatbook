"""Source-aware watchlists routing for local subscriptions and server sources."""

from __future__ import annotations

import asyncio
import inspect
from enum import Enum
from typing import Any, Mapping

from loguru import logger

from ..runtime_policy.types import PolicyDeniedError
from .watchlist_item_page import WatchlistItemCursor, WatchlistItemPage
from .watchlist_opml_service import WatchlistOpmlService
from .watchlist_preview_service import WatchlistPreviewService

# Generous upper bound for the number of sources included in an OPML export.
WC_EXPORT_OPML_MAX_SOURCES = 10000


class WatchlistBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


_LOCAL_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "watchlists.groups.local",
        "source": "local",
        "supported": False,
        "reason_code": "local_contract_missing",
        "user_message": "Local watchlist group editing is deferred; local sources remain ungrouped/read-only with respect to groups.",
        "affected_action_ids": [
            "watchlists.create.local",
            "watchlists.update.local",
        ],
    },
    {
        "operation_id": "watchlists.runs.execution.local",
        "source": "local",
        "supported": False,
        "reason_code": "local_contract_missing",
        "user_message": "Local watchlist runs are queued and observable locally, but actual scraper execution is not implemented in this scope yet.",
        "affected_action_ids": [
            "watchlists.runs.detail.local",
            "watchlists.runs.launch.local",
            "watchlists.runs.list.local",
            "watchlists.runs.observe.local",
        ],
    },
]

_SERVER_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "watchlists.groups.server",
        "source": "server",
        "supported": False,
        "reason_code": "server_contract_missing",
        "user_message": "Server watchlist group editing is deferred in Chatbook; group membership is treated as read-only.",
        "affected_action_ids": [
            "watchlists.create.server",
            "watchlists.update.server",
        ],
    },
]


class WatchlistScopeService:
    """Route watchlist operations to the active local/server authority."""

    def __init__(
        self, *, local_service: Any, server_service: Any, policy_enforcer: Any = None
    ):
        self.local_service = local_service
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_backend(
        self, runtime_backend: WatchlistBackend | str | None
    ) -> WatchlistBackend:
        if runtime_backend is None:
            return WatchlistBackend.LOCAL
        if isinstance(runtime_backend, WatchlistBackend):
            return runtime_backend
        try:
            return WatchlistBackend(str(runtime_backend))
        except ValueError as exc:
            raise ValueError(f"Invalid watchlists backend: {runtime_backend}") from exc

    @staticmethod
    def _action_id(backend: WatchlistBackend, action: str) -> str:
        return f"watchlists.{action}.{backend.value}"

    def _enforce_policy(self, backend: WatchlistBackend, action: str) -> None:
        if self.policy_enforcer is None:
            return
        action_id = self._action_id(backend, action)
        require_allowed = getattr(self.policy_enforcer, "require_allowed", None)
        require_ui_action_allowed = getattr(
            self.policy_enforcer, "require_ui_action_allowed", None
        )
        if callable(require_allowed):
            require_allowed(action_id=action_id)
        elif callable(require_ui_action_allowed):
            decision = require_ui_action_allowed(action_id=action_id)
            if decision is not None and getattr(decision, "allowed", True) is False:
                raise PolicyDeniedError(
                    action_id=action_id,
                    reason_code=getattr(decision, "reason_code", None)
                    or "authority_denied",
                    user_message=getattr(decision, "user_message", None)
                    or f"{action_id} is not allowed.",
                    effective_source=getattr(decision, "effective_source", None)
                    or backend.value,
                    authority_owner=getattr(decision, "authority_owner", None)
                    or backend.value,
                )

    def _service_for_backend(self, backend: WatchlistBackend) -> Any:
        if backend == WatchlistBackend.LOCAL:
            if self.local_service is None:
                raise ValueError("Local watchlists backend is unavailable.")
            return self.local_service
        if self.server_service is None:
            raise ValueError("Server watchlists backend is unavailable.")
        return self.server_service

    def create_form_source_types(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
    ) -> tuple[str, ...]:
        """Return the active backend's ordered create-form source types.

        Args:
            runtime_backend: Backend contract to inspect. ``None`` selects
                the local backend.

        Returns:
            Ordered source-type identifiers supported by that backend's
            create form.

        Raises:
            ValueError: If the backend is invalid or unavailable.
        """
        backend = self._normalize_backend(runtime_backend)
        service = self._service_for_backend(backend)
        return service.CREATE_FORM_SOURCE_TYPES

    def _get_run_executor(self) -> Any:
        local_service = self.local_service
        if local_service is None:
            return None
        return getattr(local_service, "run_executor", None)

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    @staticmethod
    def _source_id_from_item_id(item_id: Any) -> Any:
        """Resolve a source id that may arrive namespaced.

        `LocalWatchlistsService` returns rows carrying both
        ``"id": "local:subscription:1"`` and ``"source_id": 1``, and the screen
        passes the display id. `launch_run` was the one caller that did not go
        through here, so `check_now` handed the namespaced form to
        `local.launch_run`, which does ``int(source_id)`` -- raising
        `ValueError` into a swallowed debug log and leaving "Check now" doing
        nothing at all (TASK-1100).

        Non-namespaced values are returned **unchanged rather than
        stringified**, so a caller already holding the integer keeps passing an
        integer downstream. The previous version stringified everything, which
        `test_scope_service_routes_run_actions_with_watchlists_run_action_ids`
        caught when `launch_run` started routing through here.

        Args:
            item_id: Either ``"local:subscription:1"`` or a bare id.

        Returns:
            The trailing id when namespaced, otherwise ``item_id`` untouched.
        """
        if isinstance(item_id, str) and ":" in item_id:
            return item_id.rsplit(":", 1)[-1]
        return item_id

    @staticmethod
    def _run_id_from_item_id(item_id: Any) -> str:
        item_id_text = str(item_id)
        if ":" in item_id_text:
            return item_id_text.rsplit(":", 1)[-1]
        return item_id_text

    @staticmethod
    def _rule_id_from_item_id(item_id: Any) -> str:
        item_id_text = str(item_id)
        if ":" in item_id_text:
            return item_id_text.rsplit(":", 1)[-1]
        return item_id_text

    @staticmethod
    def _reject_deferred_group_editing(payload: Mapping[str, Any]) -> None:
        group_keys = {"group", "group_id", "group_ids", "groups"}
        if any(key in payload for key in group_keys):
            raise ValueError("Watchlist group editing is deferred in this slice.")

    def list_unsupported_capabilities(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        backend = self._normalize_backend(runtime_backend)
        if backend == WatchlistBackend.LOCAL:
            reports = [dict(_LOCAL_UNSUPPORTED_CAPABILITIES[0])]
            if not callable(getattr(self.local_service, "execute_run", None)):
                reports.append(dict(_LOCAL_UNSUPPORTED_CAPABILITIES[1]))
            return reports
        return [dict(item) for item in _SERVER_UNSUPPORTED_CAPABILITIES]

    async def list_watch_items(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        limit: int = 100,
        offset: int = 0,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "list")
        service = self._service_for_backend(backend)
        return await self._maybe_await(
            service.list_sources(limit=limit, offset=offset, **filters)
        )

    async def list_items(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        source_id: Any = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
        run_id: Any = None,
        watchlist_id: Any = None,
        unassigned_only: bool = False,
        statuses: list[str] | None = None,
        is_flagged: bool | None = None,
        search: str | None = None,
        since: str | None = None,
    ) -> list[dict[str, Any]]:
        """List content items for watchlist sources.

        Args:
            runtime_backend: Target backend (``local`` or ``server``).
            source_id: Optional source identifier to filter by.
            status: Item status filter (``new``, ``reviewed``, ``ingested``,
                ``ignored``, ``error``).
            limit: Maximum items to return.
            offset: Pagination offset.
            run_id: Optional run whose items to return (TASK-2306). Routed as
                ``items.list`` like every other item read: it narrows which
                items come back, it does not read the run record.
            watchlist_id: Optional watchlist whose sources' items to return
                (TASK-2513).
            unassigned_only: When true, return only items of sources in no
                watchlist (TASK-2513).
            statuses: Optional list of statuses to match any of (TASK-2513);
                combine only with a falsey ``status``.
            is_flagged: Restrict to starred rows (``True``) or unstarred
                rows (``False``), or ``None`` to not filter by the flag
                (TASK-3072 -- the Starred feed's scope).
            search: Full-text terms over title/content/author (TASK-3791 --
                the reader's `/`), or ``None`` for no search predicate.
            since: Effective-date floor (TASK-3791 -- the Today feed's
                scope), or ``None`` for no floor.

        Returns:
            List of normalized watchlist item dicts.

        Raises:
            ValueError: If the server backend is requested; item listing is
                local-only in this slice.
        """
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "items.list")
        if backend == WatchlistBackend.SERVER:
            raise ValueError("Item listing is only supported for the local backend in this slice.")
        service = self._service_for_backend(backend)
        return await self._maybe_await(
            service.list_items(
                source_id=source_id,
                status=status,
                limit=limit,
                offset=offset,
                run_id=run_id,
                watchlist_id=watchlist_id,
                unassigned_only=unassigned_only,
                statuses=statuses,
                is_flagged=is_flagged,
                search=search,
                since=since,
            )
        )

    async def list_reader_items_page(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        source_id: Any = None,
        status: str | None = None,
        limit: int = 50,
        run_id: Any = None,
        watchlist_id: Any = None,
        unassigned_only: bool = False,
        statuses: list[str] | None = None,
        is_flagged: bool | None = None,
        search: str | None = None,
        since: str | None = None,
        snapshot_max_item_id: int | None = None,
        after: WatchlistItemCursor | None = None,
    ) -> WatchlistItemPage:
        """Return one stable Reader page from the local item authority.

        Args:
            runtime_backend: Target backend; Reader paging is local-only.
            source_id: Optional subscription scope.
            status: Optional single status scope.
            limit: Maximum rows in the page.
            run_id: Optional producing-run scope.
            watchlist_id: Optional watchlist scope.
            unassigned_only: Whether to include only unassigned sources.
            statuses: Optional multiple-status scope.
            is_flagged: Optional starred-state scope.
            search: Optional full-text terms.
            since: Optional effective-date floor.
            snapshot_max_item_id: Existing snapshot high-water.
            after: Optional continuation cursor.

        Returns:
            The typed page returned by the local service.

        Raises:
            ValueError: If the Server backend is requested.
        """
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "items.list")
        if backend == WatchlistBackend.SERVER:
            raise ValueError(
                "Item listing is only supported for the local backend in this slice."
            )
        service = self._service_for_backend(backend)
        return await self._maybe_await(
            service.list_reader_items_page(
                source_id=source_id,
                status=status,
                limit=limit,
                run_id=run_id,
                watchlist_id=watchlist_id,
                unassigned_only=unassigned_only,
                statuses=statuses,
                is_flagged=is_flagged,
                search=search,
                since=since,
                snapshot_max_item_id=snapshot_max_item_id,
                after=after,
            )
        )

    async def count_reader_item_arrivals(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        snapshot_max_item_id: int,
        source_id: Any = None,
        status: str | None = None,
        run_id: Any = None,
        watchlist_id: Any = None,
        unassigned_only: bool = False,
        statuses: list[str] | None = None,
        is_flagged: bool | None = None,
        search: str | None = None,
        since: str | None = None,
    ) -> int:
        """Count local Reader arrivals without replacing its snapshot.

        Args:
            runtime_backend: Target backend; Reader counts are local-only.
            snapshot_max_item_id: Snapshot high-water that arrivals exceed.
            source_id: Optional subscription scope.
            status: Optional single status scope.
            run_id: Optional producing-run scope.
            watchlist_id: Optional watchlist scope.
            unassigned_only: Whether to include only unassigned sources.
            statuses: Optional multiple-status scope.
            is_flagged: Optional starred-state scope.
            search: Optional full-text terms.
            since: Optional effective-date floor.

        Returns:
            Number of matching rows created after the high-water.

        Raises:
            ValueError: If the Server backend is requested.
        """
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "items.list")
        if backend == WatchlistBackend.SERVER:
            raise ValueError(
                "Item listing is only supported for the local backend in this slice."
            )
        service = self._service_for_backend(backend)
        return int(
            await self._maybe_await(
                service.count_reader_item_arrivals(
                    snapshot_max_item_id=snapshot_max_item_id,
                    source_id=source_id,
                    status=status,
                    run_id=run_id,
                    watchlist_id=watchlist_id,
                    unassigned_only=unassigned_only,
                    statuses=statuses,
                    is_flagged=is_flagged,
                    search=search,
                    since=since,
                )
            )
        )

    async def get_item_status(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        item_id: Any,
    ) -> str:
        """Read one content item's current status.

        Routed as `items.detail` rather than `items.list`: this is a
        single-item read, and `watchlists.items` already registers DETAIL
        (see `runtime_policy/registry.py`).

        Args:
            runtime_backend: Target backend (``local`` or ``server``).
            item_id: Item identifier, namespaced (``local:watchlist_item:2``)
                or bare.

        Returns:
            The item's current status.

        Raises:
            ValueError: If the server backend is requested; item status is
                local-only, exactly as `list_items` and `update_item` are --
                the server API carries no item-status route.
            KeyError: If no item has that id.
        """
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "items.detail")
        if backend == WatchlistBackend.SERVER:
            raise ValueError(
                "Item status reads are only supported for the local backend "
                "in this slice."
            )
        service = self._service_for_backend(backend)
        return str(
            await self._maybe_await(
                service.get_item_status(self._source_id_from_item_id(item_id))
            )
        )

    async def get_item_content(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        item_id: Any,
    ) -> str | None:
        """Read one content item's full body text -- the reader's DETAIL fetch.

        TASK-15464 counterpart to `get_item_status` above -- same routing
        (`items.detail`, both single-item reads under the DETAIL policy
        `watchlists.items` already registers) and the same local-only
        restriction.

        Args:
            runtime_backend: Target backend (``local`` or ``server``).
            item_id: Item identifier, namespaced (``local:watchlist_item:2``)
                or bare.

        Returns:
            The stored content, or `None` if no row has this id, or the row
            has one but its content is itself NULL. Unlike `get_item_status`
            this does not raise `KeyError` on a miss -- see
            `SubscriptionsDB.get_item_content`'s docstring for why a missing
            row and a present-but-empty one are not distinguished for this
            column.

        Raises:
            ValueError: If the server backend is requested; item content
                reads are local-only, exactly as `list_items` and
                `get_item_status` are.
        """
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "items.detail")
        if backend == WatchlistBackend.SERVER:
            raise ValueError(
                "Item content reads are only supported for the local backend "
                "in this slice."
            )
        service = self._service_for_backend(backend)
        return await self._maybe_await(
            service.get_item_content(self._source_id_from_item_id(item_id))
        )

    async def update_item(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        item_id: Any,
        status: str,
    ) -> dict[str, Any]:
        """Move a watchlist content item to a new status.

        TASK-1120 AC#3. This method did not exist, so
        `WatchlistsBackendController.update_item_status` -- which probes for
        `update_item`, then `update_item_status`, then `mark_item_status` --
        found none of them and raised `NotImplementedError`. The screen caught
        that as a plain `Exception`, logged it at debug and toasted "Failed to
        mark item reviewed", so `Mark reviewed`, `Ingest` and `Ignore` were
        inert with no durable trace (the swallow TASK-1090 is about).

        Args:
            runtime_backend: Target backend (``local`` or ``server``).
            item_id: Item identifier, namespaced (``local:watchlist_item:2``)
                or bare.
            status: One of `LocalWatchlistsService.ITEM_STATUSES`.

        Returns:
            The backend's normalized result for the updated item.

        Raises:
            ValueError: If the server backend is requested; item status is
                local-only, exactly as `list_items` already is -- the server
                API carries no item-status route.
        """
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "items.update")
        if backend == WatchlistBackend.SERVER:
            raise ValueError(
                "Item status updates are only supported for the local backend "
                "in this slice."
            )
        service = self._service_for_backend(backend)
        return await self._maybe_await(
            service.update_item(
                item_id=self._source_id_from_item_id(item_id), status=status
            )
        )

    async def mark_all_read(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        source_id: Any = None,
        watchlist_id: Any = None,
        unassigned_only: bool = False,
    ) -> list[int]:
        """Mark every ``new`` item in scope ``reviewed``; return the affected ids.

        The reader's mark-all-read affordance (TASK-2513). Routed as
        ``items.update`` like `update_item`: it is the same write, batched.
        The returned ids are the undo batch for `restore_items_new`.

        Args:
            runtime_backend: Target backend (``local`` or ``server``).
            source_id: Restrict to one source, or `None` for all. Forwarded
                bare, exactly as `list_items` forwards its `source_id`.
            watchlist_id: Restrict to items of the sources in one watchlist.
            unassigned_only: Restrict to items of sources in no watchlist.

        Returns:
            The local row ids moved to ``reviewed``.

        Raises:
            ValueError: If the server backend is requested; item writes are
                local-only, mirroring `update_item` -- the server API carries
                no item-status route.
        """
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "items.update")
        if backend == WatchlistBackend.SERVER:
            raise ValueError(
                "Item status updates are only supported for the local backend "
                "in this slice."
            )
        service = self._service_for_backend(backend)
        result = await self._maybe_await(
            service.mark_all_read(
                source_id=source_id,
                watchlist_id=watchlist_id,
                unassigned_only=unassigned_only,
            )
        )
        return [int(item_id) for item_id in list(result or [])]

    async def restore_items_new(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        item_ids: list[Any],
    ) -> int:
        """Move the given ids back to ``new`` — the undo half of `mark_all_read`.

        Only rows still ``reviewed`` are restored, so an item the user has
        since ingested or ignored keeps its status.

        Args:
            runtime_backend: Target backend (``local`` or ``server``).
            item_ids: Item identifiers, namespaced
                (``local:watchlist_item:2``) or bare — the batch
                `mark_all_read` returned.

        Returns:
            How many rows were actually restored.

        Raises:
            ValueError: If the server backend is requested; item writes are
                local-only, mirroring `update_item` -- the server API carries
                no item-status route.
        """
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "items.update")
        if backend == WatchlistBackend.SERVER:
            raise ValueError(
                "Item status updates are only supported for the local backend "
                "in this slice."
            )
        service = self._service_for_backend(backend)
        row_ids = [
            self._source_id_from_item_id(item_id) for item_id in item_ids or []
        ]
        return int(
            await self._maybe_await(service.restore_items_new(item_ids=row_ids))
        )

    async def set_item_flagged(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        item_id: Any,
        flagged: bool,
    ) -> None:
        """Star or unstar one item (TASK-3072 plan task 7).

        Routed as ``items.update`` like `update_item`/`mark_all_read`: it is
        the same kind of write -- one flag on one `subscription_items` row.

        Args:
            runtime_backend: Target backend (``local`` or ``server``).
            item_id: Item identifier, namespaced
                (``local:watchlist_item:7``) or bare -- denamespaced exactly
                as `update_item` does.
            flagged: `True` to star the item, `False` to unstar it.

        Raises:
            ValueError: If the server backend is requested; item writes are
                local-only, mirroring `update_item` -- the server API carries
                no item-flag route.
        """
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "items.update")
        if backend == WatchlistBackend.SERVER:
            raise ValueError(
                "Item star updates are only supported for the local backend "
                "in this slice."
            )
        service = self._service_for_backend(backend)
        await self._maybe_await(
            service.set_item_flagged(
                item_id=self._source_id_from_item_id(item_id),
                flagged=bool(flagged),
            )
        )

    async def get_watch_item_detail(
        self,
        item_id: Any,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
    ) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "detail")
        service = self._service_for_backend(backend)
        return await self._maybe_await(
            service.get_source(self._source_id_from_item_id(item_id))
        )

    async def create_watch_item(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "create")
        self._reject_deferred_group_editing(payload)
        service = self._service_for_backend(backend)
        if backend == WatchlistBackend.LOCAL:
            return await self._maybe_await(service.create_source(payload))
        return await self._maybe_await(service.create_source(**dict(payload)))

    async def update_watch_item(
        self,
        item_id: Any,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "update")
        self._reject_deferred_group_editing(payload)
        service = self._service_for_backend(backend)
        source_id = self._source_id_from_item_id(item_id)
        if backend == WatchlistBackend.LOCAL:
            return await self._maybe_await(service.update_source(source_id, payload))
        return await self._maybe_await(
            service.update_source(source_id, **dict(payload))
        )

    async def delete_watch_item(
        self,
        item_id: Any,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
    ) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "delete")
        service = self._service_for_backend(backend)
        return await self._maybe_await(
            service.delete_source(self._source_id_from_item_id(item_id))
        )

    async def launch_run(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        job_id: Any = None,
        source_id: Any = None,
    ) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "runs.launch")
        service = self._service_for_backend(backend)
        launched = await self._maybe_await(
            service.launch_run(
                job_id=job_id,
                source_id=self._source_id_from_item_id(source_id),
            )
        )
        if backend == WatchlistBackend.LOCAL:
            execute_run = getattr(service, "execute_run", None)
            if callable(execute_run):
                run_id = (
                    launched.get("run_id") if isinstance(launched, Mapping) else None
                )
                if run_id is None and isinstance(launched, Mapping):
                    run_id = launched.get("id")
                if run_id is None:
                    raise ValueError(
                        "Local watchlist run launch did not return a run identifier."
                    )
                resolved_run_id = self._run_id_from_item_id(run_id)
                if (
                    isinstance(launched, Mapping)
                    and launched.get("_claim_acquired") is False
                ):
                    return await self._maybe_await(
                        service.wait_for_terminal_run(resolved_run_id)
                    )
                try:
                    return await self._maybe_await(execute_run(resolved_run_id))
                except asyncio.CancelledError:
                    # Batch-4 review, C1 (CRITICAL) -- the second-layer
                    # counterpart of `LocalWatchlistsService.execute_run`'s own
                    # `except asyncio.CancelledError` branch, kept for the
                    # identical reason TASK-1090's `except Exception` below
                    # already is: `execute_run` records its own cancellation
                    # (a screen/widget teardown, e.g. the user switching tabs
                    # mid-check) and re-raises, so ordinarily this branch never
                    # actually has anything left to WRITE -- the run row is
                    # already durably `failed` by the time the re-raise gets
                    # here (`db.transaction()` commits synchronously with no
                    # interruptible inner await, so there is no race to read
                    # stale).
                    #
                    # Batch-4 review ROUND 2, N1 (Important, introduced by the
                    # C1 fix). The run ROW update alone being idempotent
                    # (`record_run_result`'s UPDATE is by id) is not enough:
                    # calling `record_run_failure` a second time for the SAME
                    # cancellation also re-evaluates every alert rule against
                    # the run's stats and re-DISPATCHES a notification for
                    # each match, and `_dispatch_alert_notification`/
                    # `ClientNotificationsDB.insert_notification` have no
                    # deduplicating INSERT -- `dedupe_key` is computed into the
                    # payload and never read back by anything. So a
                    # status-agnostic rule (e.g. "no_items", which fires on the
                    # stats regardless of pass/fail) produced two identical
                    # notification rows for one cancelled check: one from
                    # `execute_run`'s write, one from this one.
                    #
                    # Checked here instead: whether the run already reached a
                    # terminal state before writing again. This is the correct
                    # mechanism, not a dedupe-key lookup -- a dedupe-key check
                    # would mean teaching the notification store a new
                    # "does a row with this key already exist" query for a
                    # value that, today, exists for exactly this one caller,
                    # and it would still let this layer perform a pointless
                    # second run-row UPDATE. Checking status reuses a field
                    # the run row already carries and this method already has
                    # a service handle to read, and it generalizes correctly
                    # to the genuine defense-in-depth case this branch exists
                    # for: if the cancellation struck OUTSIDE `execute_run`'s
                    # own `try` block (the synchronous section before it, or a
                    # future local-like service with no `CancelledError`
                    # handling of its own), the run is still `queued` or
                    # `running` here, and THIS layer is the only one that will
                    # ever record it -- exactly the case the write must still
                    # happen for. "queued"/"running" are the only two
                    # non-terminal statuses a local run row can ever hold
                    # before `execute_run` reaches its own `try` block (see
                    # `launch_run`'s insert and `_mark_run_started`).
                    already_recorded = False
                    get_run = getattr(service, "get_run", None)
                    if callable(get_run):
                        try:
                            current_run = await self._maybe_await(
                                get_run(resolved_run_id)
                            )
                        except Exception:
                            current_run = None
                        if isinstance(current_run, Mapping):
                            already_recorded = (
                                str(current_run.get("status") or "").lower()
                                not in {"queued", "running"}
                            )
                    if not already_recorded:
                        record_failure = getattr(service, "record_run_failure", None)
                        if callable(record_failure):
                            try:
                                await self._maybe_await(
                                    record_failure(
                                        resolved_run_id,
                                        error=(
                                            "Check cancelled: navigated away "
                                            "before it finished."
                                        ),
                                    )
                                )
                            except Exception:
                                logger.opt(exception=True).warning(
                                    "Watchlists: could not record the cancellation "
                                    f"of run {resolved_run_id}."
                                )
                    raise
                except Exception as exc:
                    # TASK-1090. `execute_run` records its own fetch failures,
                    # but anything that escapes it -- a subscription deleted
                    # between launch and execution, a service fault, the
                    # namespaced-id `ValueError` of TASK-1100 -- used to leave
                    # the row inserted a moment ago sitting at `queued`
                    # forever with nothing recorded anywhere. The run is the
                    # user's only durable evidence that a check was attempted
                    # and failed, so it is written before the error is
                    # re-raised for the screen to report.
                    record_failure = getattr(service, "record_run_failure", None)
                    if callable(record_failure):
                        try:
                            await self._maybe_await(
                                record_failure(resolved_run_id, error=exc)
                            )
                        except Exception:
                            logger.opt(exception=True).warning(
                                "Watchlists: could not record the failure of run "
                                f"{resolved_run_id}; the original error is "
                                "re-raised below."
                            )
                    raise
        return launched

    async def list_runs(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        job_id: Any = None,
        limit: int = 100,
        offset: int = 0,
        q: str | None = None,
    ) -> list[dict[str, Any]]:
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "runs.list")
        service = self._service_for_backend(backend)
        return await self._maybe_await(
            service.list_runs(job_id=job_id, limit=limit, offset=offset, q=q)
        )

    async def get_run(
        self,
        run_id: Any,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
    ) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "runs.detail")
        service = self._service_for_backend(backend)
        return await self._maybe_await(
            service.get_run(self._run_id_from_item_id(run_id))
        )

    async def observe_run(
        self,
        run_id: Any,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        include_tallies: bool = False,
    ) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "runs.observe")
        service = self._service_for_backend(backend)
        return await self._maybe_await(
            service.get_run_detail(
                self._run_id_from_item_id(run_id), include_tallies=include_tallies
            )
        )

    async def cancel_run(
        self,
        *,
        run_id: Any,
        runtime_backend: WatchlistBackend | str | None = None,
    ) -> dict[str, Any]:
        """Cancel an in-progress watchlist run.

        Args:
            run_id: Identifier of the run to cancel.
            runtime_backend: Target backend (``local`` or ``server``).

        Returns:
            Cancellation result metadata.
        """
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "runs.cancel")
        service = self._service_for_backend(backend)
        return await self._maybe_await(service.cancel_run(self._run_id_from_item_id(run_id)))

    async def list_alert_rules(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        job_id: Any = None,
    ) -> list[dict[str, Any]]:
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "alert_rules.list")
        service = self._service_for_backend(backend)
        return await self._maybe_await(service.list_alert_rules(job_id=job_id))

    async def get_alert_rule(
        self,
        rule_id: Any,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
    ) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "alert_rules.detail")
        service = self._service_for_backend(backend)
        return await self._maybe_await(
            service.get_alert_rule(self._rule_id_from_item_id(rule_id))
        )

    async def create_alert_rule(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "alert_rules.create")
        service = self._service_for_backend(backend)
        return await self._maybe_await(service.create_alert_rule(**dict(payload)))

    async def update_alert_rule(
        self,
        rule_id: Any,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "alert_rules.update")
        service = self._service_for_backend(backend)
        return await self._maybe_await(
            service.update_alert_rule(
                self._rule_id_from_item_id(rule_id), **dict(payload)
            )
        )

    async def save_alert_rule(
        self,
        *,
        payload: Mapping[str, Any],
        runtime_backend: WatchlistBackend | str | None = None,
    ) -> dict[str, Any]:
        """Create or update an alert rule based on the payload.

        Args:
            payload: Alert rule fields. Presence of ``id`` or ``rule_id``
                selects the update path.
            runtime_backend: Target backend (``local`` or ``server``).

        Returns:
            Created or updated alert rule record.
        """
        backend = self._normalize_backend(runtime_backend)
        service = self._service_for_backend(backend)
        rule_id = None
        if "id" in payload:
            rule_id = payload["id"]
        elif "rule_id" in payload:
            rule_id = payload["rule_id"]
        clean_payload = {k: v for k, v in payload.items() if k not in ("id", "rule_id")}
        if rule_id is not None:
            self._enforce_policy(backend, "alert_rules.update")
            return await self._maybe_await(
                service.update_alert_rule(self._rule_id_from_item_id(rule_id), **clean_payload)
            )
        self._enforce_policy(backend, "alert_rules.create")
        return await self._maybe_await(service.create_alert_rule(**clean_payload))

    async def delete_alert_rule(
        self,
        rule_id: Any,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
    ) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "alert_rules.delete")
        service = self._service_for_backend(backend)
        return await self._maybe_await(
            service.delete_alert_rule(self._rule_id_from_item_id(rule_id))
        )

    async def preview_source(
        self,
        *,
        source_config: Mapping[str, Any],
        runtime_backend: WatchlistBackend | str | None = None,
    ) -> dict[str, Any]:
        """Preview a watchlist source by fetching and parsing its feed.

        Args:
            source_config: Source configuration (URL, parser, etc.).
            runtime_backend: Target backend (``local`` or ``server``).

        Returns:
            Preview result containing items and log text.

        Raises:
            ValueError: If the server backend is requested; preview is local-only.
        """
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "preview")
        if backend == WatchlistBackend.SERVER:
            raise ValueError("Preview is only supported for the local backend in this slice.")
        preview_service = WatchlistPreviewService(run_executor=self._get_run_executor())
        return await self._maybe_await(preview_service.preview(source_config))

    async def check_now(
        self,
        *,
        source_id: Any,
        runtime_backend: WatchlistBackend | str | None = None,
    ) -> dict[str, Any]:
        """Trigger an immediate check for a watchlist source.

        Args:
            source_id: Identifier of the source to check.
            runtime_backend: Target backend (``local`` or ``server``).

        Returns:
            Run metadata for the launched check.
        """
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "runs.launch")
        return await self.launch_run(runtime_backend=backend, source_id=source_id)

    async def import_opml(
        self,
        *,
        xml_text: str,
        runtime_backend: WatchlistBackend | str | None = None,
    ) -> dict[str, Any]:
        """Import watchlist sources from an OPML document.

        ADR-043: folder outlines map to watchlists (resolved or created by
        case-insensitive name), their feeds join as members, top-level
        feeds stay Unassigned, and a feed URL already in the roster is
        reused rather than duplicated -- import is additive only.

        Args:
            xml_text: Raw OPML XML string.
            runtime_backend: Target backend (``local`` or ``server``).

        Returns:
            Summary dict: ``created``/``existing`` source counts, the new
            ``sources`` records, ``watchlists_created`` /
            ``watchlists_reused`` name lists, and the membership
            ``assignments`` and unique top-level ``unassigned`` counts.

        Raises:
            ValueError: If the server backend is requested; OPML import is local-only.
        """
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "import")
        if backend == WatchlistBackend.SERVER:
            raise ValueError("OPML import is only supported for the local backend in this slice.")
        payloads = WatchlistOpmlService().parse(xml_text)
        service = self._service_for_backend(backend)
        created: list[dict[str, Any]] = []
        existing_count = 0
        assignments = 0
        watchlists_created: list[str] = []
        watchlists_reused: list[str] = []
        source_ids_by_url: dict[str, int] = {}
        seen_source_keys: set[str] = set()
        assigned_source_keys: set[str] = set()
        assigned_edges: set[tuple[int, int]] = set()
        # Per-folder memo: one resolve per unique folder name (normalized),
        # so a 40-feed folder costs one lookup, and the summary's
        # created/reused lists name each watchlist once.
        resolved_folders: dict[str, dict[str, Any]] = {}
        for payload_index, payload in enumerate(payloads):
            # ADR-043 rule 6 (additive only): a feed URL already in the
            # roster is reused, never duplicated.
            url = str(payload.get("url") or "")
            source_key = f"url:{url}" if url else f"payload:{payload_index}"
            seen_source_keys.add(source_key)
            if url in source_ids_by_url:
                source_id = source_ids_by_url[url]
            else:
                existing_id = None
                if url:
                    existing_id = await self._maybe_await(
                        service.find_source_id_by_url(url)
                    )
                if existing_id is not None:
                    source_id = int(existing_id)
                    existing_count += 1
                else:
                    source = await self._maybe_await(service.create_source(payload))
                    created.append(dict(source))
                    source_id = source.get("source_id")
                if url and source_id is not None:
                    source_ids_by_url[url] = int(source_id)
            folder = str(payload.get("folder") or "").strip()
            if not folder or source_id is None:
                continue
            key = folder.lower()
            if key not in resolved_folders:
                watchlist, was_created = await self._maybe_await(
                    service.resolve_or_create_watchlist(folder)
                )
                resolved_folders[key] = dict(watchlist)
                name = str(watchlist.get("name") or folder)
                if was_created:
                    watchlists_created.append(name)
                else:
                    watchlists_reused.append(name)
            watchlist_id = int(resolved_folders[key]["id"])
            source_id = int(source_id)
            edge = (watchlist_id, source_id)
            if edge not in assigned_edges:
                await self._maybe_await(
                    service.add_source_to_watchlist(
                        watchlist_id=watchlist_id, source_id=source_id
                    )
                )
                assigned_edges.add(edge)
                assignments += 1
            assigned_source_keys.add(source_key)
        return {
            "created": len(created),
            "existing": existing_count,
            "sources": created,
            "watchlists_created": watchlists_created,
            "watchlists_reused": watchlists_reused,
            "assignments": assignments,
            "unassigned": len(seen_source_keys - assigned_source_keys),
        }

    async def export_opml(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
    ) -> str:
        """Export watchlist sources as an OPML document.

        Args:
            runtime_backend: Target backend (``local`` or ``server``).

        Returns:
            OPML XML string for the retrieved sources.
        """
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "export")
        if backend == WatchlistBackend.SERVER:
            # The server backend's source model carries no local watchlist
            # membership for this seam; keep the pre-ADR-043 flat export
            # there rather than fail a previously-working path.
            sources = await self.list_watch_items(
                runtime_backend=backend, limit=WC_EXPORT_OPML_MAX_SOURCES, offset=0
            )
            return WatchlistOpmlService().export([], sources)
        service = self._service_for_backend(backend)
        watchlists = await self._maybe_await(service.list_watchlists())
        structured: list[dict[str, Any]] = []
        for watchlist in watchlists:
            rows = await self._maybe_await(
                service.list_watchlist_source_rows(watchlist_id=watchlist["id"])
            )
            structured.append({"name": watchlist.get("name"), "sources": rows})
        unassigned = await self._maybe_await(service.list_unassigned_source_rows())
        return WatchlistOpmlService().export(structured, unassigned)
