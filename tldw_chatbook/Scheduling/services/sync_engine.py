"""Sync engine for reconciling local scheduled tasks with the server."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.services.server_client import (
    SchedulingServerClient,
    ServerClientError,
    ServerClientNotFoundError,
    ServerClientPolicyError,
)


_REMINDER_PRIMITIVE = "reminder_task"
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

#: Reserved for the PR-5 definition-push primitive (create/update/lifecycle
#: mutation replay) -- unused here since PR-3 ships only the definitions
#: pull mirror, not a push path (schedules-handoff plan, task 4 deviation
#: 1). Named now so the pending-mutation primitive namespace is settled.
_DEFINITION_PRIMITIVE = "automation_definition"

#: Matches ScheduledTasksDB._RESULT_REVIEW_PRIMITIVE -- pending mutations
#: recorded when a local review is made on a server-mirrored result
#: (Task 5) are replayed to the server here.
_RESULT_REVIEW_PRIMITIVE = "automation_result_review"

#: Page size for the results/definitions pulls, and the bounded
#: newest-pages walk both phases use (spec §5.2 limitation: the server's
#: /results endpoint exposes no updated_at filter, so incremental sync is
#: a bounded newest-N-pages walk rather than a true delta; the
#: definitions pull shares the same cap so an unbounded `while True` can
#: never spin forever against a misbehaving server).
_RESULTS_PAGE_SIZE = 50
_SYNC_MAX_PAGES = 4


@dataclass(frozen=True)
class SyncOutcome:
    """What a sync attempt actually did (task-23105 review F3).

    The engine swallows server errors internally (it records them as
    persisted sync errors), so callers previously could not distinguish
    a FAILED sync from a no-op and reported both as success. Statuses:

    - ``ok``: the attempt ran to completion. ``pulled`` is the number of
      reminder items the server listed (its full set for this owner, not
      only changed rows); ``pushed`` counts the local mutations and
      tombstones pushed.
    - ``not_applicable``: sync cannot run in this configuration (no
      server client, or a runtime-mode policy refusal). Nothing was
      pulled or pushed; not an error.
    - ``error``: the attempt failed. ``error`` carries the message that
      was also recorded as a persisted sync error.
    """

    status: str
    pulled: int = 0
    pushed: int = 0
    error: str | None = None


def now_utc_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def parse_iso(value: Any) -> datetime:
    """Parse an ISO-8601 string or return the epoch for missing/invalid values."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not value or not isinstance(value, str):
        return _EPOCH
    try:
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return _EPOCH


class SyncEngine:
    """Pull, push, and reconcile scheduled-task state with tldw_server."""

    def __init__(
        self,
        db: ScheduledTasksDB,
        server_client: SchedulingServerClient | None,
        owner_id: str,
    ) -> None:
        self.db = db
        self.server_client = server_client
        self.owner_id = owner_id

    async def pull(self, owner_id: str | None = None) -> None:
        """Public entry point to pull server reminders for the given owner."""
        target_owner = owner_id if owner_id is not None else self.owner_id
        if self.server_client is None:
            return

        await self._pull_reminders(target_owner)

        # Automation mirrors run regardless of reminder-phase health (review
        # round 1 #1): a reminder network/transaction failure says nothing
        # about the automation endpoints, and each phase below is already
        # independently error-contained via `_run_phase`.
        _, definitions_counts = await self._run_phase(
            target_owner, "Automation definitions pull", self._pull_definitions
        )
        if definitions_counts:
            logger.info(
                f"Automation definitions pull for {target_owner}: {definitions_counts}"
            )
        _, results_counts = await self._run_phase(
            target_owner, "Automation results pull", self._pull_results
        )
        if results_counts:
            logger.info(f"Automation results pull for {target_owner}: {results_counts}")

    async def _pull_reminders(self, target_owner: str) -> None:
        """The reminder-only half of `pull()` (original body, unchanged)."""
        client = self.server_client
        assert client is not None

        try:
            response = await client.list_reminders()
            if not isinstance(response, dict):
                response = {}
            pulled_items = response.get("items", [])
        except ServerClientPolicyError as exc:
            # A runtime-mode refusal ("requires server mode") means sync is
            # not applicable right now — recording it as a sync error put a
            # standing error badge on local-only profiles (task-2722).
            logger.info(f"Sync pull not applicable for {target_owner}: {exc}")
            return
        except ServerClientError as exc:
            self._record_sync_error(str(exc), target_owner)
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"Sync pull failed for {target_owner}: {exc}")
            self._record_sync_error(str(exc), target_owner)
            return

        try:
            with self.db.transaction() as conn:
                for item in pulled_items:
                    if not item.get("title"):
                        item["title"] = "Untitled reminder"

                pull_conflicts = self.db._apply_pulled_reminders(
                    conn, target_owner, pulled_items, set()
                )
                for conflict in pull_conflicts:
                    self.db._record_conflict_conn(
                        conn,
                        local_id=conflict["local_id"],
                        primitive=_REMINDER_PRIMITIVE,
                        owner_id=target_owner,
                        server_state=conflict["server_state"],
                        local_state=conflict["local_state"],
                    )
                seen_server_ids = {
                    item["id"] for item in pulled_items if item.get("id")
                }
                self.db._detect_server_deletions_conn(
                    conn, target_owner, seen_server_ids
                )
                self.db._update_sync_state_conn(
                    conn,
                    target_owner,
                    last_pull_at=now_utc_iso(),
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"Sync pull transaction failed for {target_owner}: {exc}")
            self._record_sync_error(str(exc), target_owner)

    async def sync_now(self, owner_id: str | None = None) -> SyncOutcome:
        target_owner = owner_id if owner_id is not None else self.owner_id
        if self.server_client is None:
            return SyncOutcome("not_applicable")

        reminder_outcome = await self._sync_reminders(target_owner)

        # Automation mirrors, appended after the reminder phase (task 4) and
        # run REGARDLESS of the reminder phase's own outcome (review round 1
        # #1): a reminder network/transaction failure says nothing about the
        # automation endpoints, and each phase below is already
        # independently error-contained via `_run_phase`. Pushback runs
        # first so a fresh results mirror can't clobber a review the user
        # just made locally (its own pending mutation is cleared before the
        # pull below re-reads that same row).
        phase_errors: list[str] = []

        error, counts = await self._run_phase(
            target_owner, "Automation review pushback", self._replay_review_mutations
        )
        if error:
            phase_errors.append(error)
        if counts:
            logger.info(f"Automation review pushback for {target_owner}: {counts}")

        error, counts = await self._run_phase(
            target_owner, "Automation definitions pull", self._pull_definitions
        )
        if error:
            phase_errors.append(error)
        if counts:
            logger.info(f"Automation definitions pull for {target_owner}: {counts}")

        error, counts = await self._run_phase(
            target_owner, "Automation results pull", self._pull_results
        )
        if error:
            phase_errors.append(error)
        if counts:
            logger.info(f"Automation results pull for {target_owner}: {counts}")

        # The reminder phase's own outcome/status semantics are preserved
        # unchanged (`_sync_reminders` is the original method body) when it
        # already failed or was not applicable. But an "ok" reminder phase
        # sitting beside a failed automation phase used to report clean --
        # the exact dishonesty task-23105 fixed for reminders (F2): surface
        # the first automation-phase error so the caller doesn't toast
        # success next to a fresh error badge. Results/definitions already
        # pulled by an earlier phase this round are not rolled back.
        if reminder_outcome.status == "ok" and phase_errors:
            return replace(reminder_outcome, status="error", error=phase_errors[0])

        return reminder_outcome

    async def _sync_reminders(self, target_owner: str) -> SyncOutcome:
        """The reminder-only half of `sync_now()` (original body, unchanged)."""
        try:
            (
                pulled_items,
                staged_outcomes,
                conflicts,
                tombstone_ids,
                pending_local_ids,
                mutations,
            ) = await self._network_phase(target_owner)
        except ServerClientPolicyError as exc:
            # Same rule as `pull`: a runtime-mode refusal is "not applicable",
            # never a persisted sync error (task-2722).
            logger.info(f"Sync not applicable for {target_owner}: {exc}")
            return SyncOutcome("not_applicable")
        except ServerClientError as exc:
            self._record_sync_error(str(exc), target_owner)
            return SyncOutcome("error", error=str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"Sync network phase failed for {target_owner}: {exc}")
            self._record_sync_error(str(exc), target_owner)
            return SyncOutcome("error", error=str(exc))

        try:
            with self.db.transaction() as conn:
                # Ensure pulled items have safe defaults before the DB helper
                # copies fields verbatim.
                for item in pulled_items:
                    if not item.get("title"):
                        item["title"] = "Untitled reminder"

                pull_conflicts = self.db._apply_pulled_reminders(
                    conn, target_owner, pulled_items, pending_local_ids
                )
                # The DB helper records the local record only; enrich with the
                # pending mutation so conflict resolution can re-queue it.
                mutation_payloads = {
                    m["local_id"]: m.get("payload") or {} for m in mutations
                }
                for conflict in pull_conflicts:
                    conflict.setdefault("local_state", {})
                    conflict["local_state"]["pending_mutation"] = mutation_payloads.get(
                        conflict["local_id"], {}
                    )

                all_conflicts = conflicts + pull_conflicts
                for conflict in all_conflicts:
                    self.db._record_conflict_conn(
                        conn,
                        local_id=conflict["local_id"],
                        primitive=_REMINDER_PRIMITIVE,
                        owner_id=target_owner,
                        server_state=conflict["server_state"],
                        local_state=conflict["local_state"],
                    )
                for outcome in staged_outcomes:
                    local_id = outcome["local_id"]
                    server_id = outcome.get("server_id")
                    if server_id:
                        self.db._set_sync_mapping_conn(
                            conn, local_id, server_id, _REMINDER_PRIMITIVE, target_owner
                        )
                        self.db._update_reminder_task_conn(
                            conn, local_id, server_id=server_id
                        )
                    if outcome.get("delete_local"):
                        self.db._delete_reminder_task_conn(conn, local_id)
                        self.db._delete_sync_mapping_conn(
                            conn, local_id, _REMINDER_PRIMITIVE, target_owner
                        )
                mutation_ids = [o["mutation_id"] for o in staged_outcomes if o.get("mutation_id")]
                self.db._purge_pending_mutations(conn, target_owner, mutation_ids)
                for local_id in tombstone_ids:
                    self.db._delete_tombstone_conn(
                        conn, local_id, _REMINDER_PRIMITIVE, target_owner
                    )
                seen_server_ids = {
                    item["id"] for item in pulled_items if item.get("id")
                }
                self.db._detect_server_deletions_conn(
                    conn, target_owner, seen_server_ids
                )
                self.db._update_sync_state_conn(
                    conn,
                    target_owner,
                    last_pull_at=now_utc_iso(),
                    last_push_at=now_utc_iso() if staged_outcomes or tombstone_ids else None,
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"Sync transaction failed for {target_owner}: {exc}")
            self._record_sync_error(str(exc), target_owner)
            return SyncOutcome("error", error=str(exc))

        # staged_outcomes already includes the tombstone outcomes (see
        # _network_phase), so it IS the pushed count. Automation
        # definitions/results are mirrors, not reminder push/pull, so they
        # are intentionally not folded into these counts (SyncOutcome's
        # contract is documented as reminder-scoped).
        return SyncOutcome(
            "ok",
            pulled=len(pulled_items),
            pushed=len(staged_outcomes),
        )

    async def _run_phase(
        self, owner_id: str, label: str, phase: Any,
    ) -> tuple[str | None, dict[str, int] | None]:
        """Run one self-contained sync phase, containing its own failure.

        Mirrors the top-level pull()/sync_now() error discipline (task-2722):
        a runtime-mode policy refusal is logged and treated as not
        applicable, never persisted; any other error is recorded via
        `_record_sync_error` and swallowed so the phases after this one
        still run.

        Returns:
            A ``(error, counts)`` tuple. ``error`` is ``None`` on success
            or a policy refusal (not an error); otherwise the message that
            was also recorded via `_record_sync_error`. ``counts`` is the
            phase's own return value (an upsert-count dict) on success, or
            ``None`` when the phase raised or returned nothing -- callers
            use it so a truncated/failed phase never silently discards
            what did land (F2/F8).
        """
        try:
            counts = await phase(owner_id)
            return None, counts
        except ServerClientPolicyError as exc:
            logger.info(f"{label} not applicable for {owner_id}: {exc}")
            return None, None
        except ServerClientError as exc:
            self._record_sync_error(str(exc), owner_id)
            return str(exc), None
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"{label} failed for {owner_id}: {exc}")
            self._record_sync_error(str(exc), owner_id)
            return str(exc), None

    async def _replay_review_mutations(self, owner_id: str) -> None:
        """Replay pending `automation_result_review` mutations to the server.

        Success and a 404 (the result was retired server-side) both clear
        the mutation. Any other error is left in place -- and re-raised so
        `_run_phase` records one sync error for the phase and stops
        attempting further mutations this round, mirroring
        `_push_mutation`'s "abort the whole push phase on a retryable
        server error" discipline for reminders.
        """
        assert self.server_client is not None
        mutations = self.db.get_pending_mutations(
            owner_id, primitive=_RESULT_REVIEW_PRIMITIVE
        )
        for mutation in mutations:
            payload = mutation.get("payload") or {}
            server_result_id = payload.get("server_result_id")
            if not server_result_id:
                logger.warning(
                    f"Pending automation_result_review mutation {mutation.get('id')} "
                    "has no server_result_id; dropping (nothing to replay)"
                )
                self.db.delete_pending_mutation(mutation["id"])
                continue
            try:
                await self.server_client.review_automation_result(
                    server_result_id,
                    payload.get("review_state"),
                    review_note=payload.get("review_note"),
                )
            except ServerClientNotFoundError as exc:
                logger.info(
                    f"Automation result {server_result_id} retired server-side "
                    f"({exc}); dropping its pending review mutation"
                )
            # Success or a confirmed retirement: the mutation is settled.
            self.db.delete_pending_mutation(mutation["id"])

    async def _pull_definitions(self, owner_id: str) -> dict[str, int]:
        """Page up to `_SYNC_MAX_PAGES` of the server's automation definitions.

        Upserted per page (not batched to the end) so a later page's
        failure still leaves earlier pages' rows mirrored -- the same
        partial-progress-survives-a-failure shape as `_pull_results`.
        Stops early on an empty page or `has_more=False`; logs (info) when
        the cap was hit with more remaining, so a truncated pull is never
        silent (F4: this was an unbounded `while True` against the server).
        """
        assert self.server_client is not None
        totals: dict[str, int] = {}
        offset = 0
        for _page_num in range(_SYNC_MAX_PAGES):
            response = await self.server_client.list_automation_definitions(
                limit=_RESULTS_PAGE_SIZE, offset=offset
            )
            if not isinstance(response, dict):
                response = {}
            page = list(response.get("items") or [])
            counts = self.db.upsert_automation_definitions_from_server(owner_id, page)
            for key, value in counts.items():
                totals[key] = totals.get(key, 0) + value
            offset += len(page)
            if not page or not response.get("has_more"):
                return totals
        logger.info(
            f"Automation definitions pull hit the {_SYNC_MAX_PAGES}-page cap for "
            f"{owner_id} with more definitions remaining server-side"
        )
        return totals

    async def _pull_results(self, owner_id: str) -> dict[str, int]:
        """Walk up to `_SYNC_MAX_PAGES` newest pages of server results.

        The server's /results endpoint exposes no `updated_at` filter
        (verified at origin/dev), so this is a bounded newest-pages walk
        rather than a true incremental pull (spec §5.2 limitation): review
        drift older than the window waits for a later sync. Stops early
        on a short page or `has_more=False`; logs (info) when the cap was
        hit with more remaining, so a truncated pull is never silent.
        """
        assert self.server_client is not None
        totals: dict[str, int] = {}
        offset = 0
        for _page_num in range(_SYNC_MAX_PAGES):
            response = await self.server_client.list_automation_results(
                limit=_RESULTS_PAGE_SIZE, offset=offset
            )
            if not isinstance(response, dict):
                response = {}
            page = list(response.get("items") or [])
            counts = self.db.upsert_automation_results_from_server(owner_id, page)
            for key, value in counts.items():
                totals[key] = totals.get(key, 0) + value
            offset += len(page)
            if len(page) < _RESULTS_PAGE_SIZE or not response.get("has_more"):
                return totals
        logger.info(
            f"Automation results pull hit the {_SYNC_MAX_PAGES}-page cap for "
            f"{owner_id} with more results remaining server-side"
        )
        return totals

    async def _network_phase(
        self, owner_id: str
    ) -> tuple[list[dict], list[dict], list[dict], list[str], set[str], list[dict]]:
        """Return (pulled_items, staged_outcomes, conflicts, tombstone_ids_to_delete,
        pending_local_ids, mutations).

        On a retryable server error, the whole phase aborts and the caller records
        a single sync error. Non-retryable 404s are converted to conflicts and the
        pending mutation is staged for removal.
        """
        pulled_items: list[dict] = []
        staged_outcomes: list[dict] = []
        conflicts: list[dict] = []
        tombstone_ids_to_delete: list[str] = []

        assert self.server_client is not None
        response = await self.server_client.list_reminders()
        if not isinstance(response, dict):
            response = {}
        pulled_items = response.get("items", [])

        mutations = self.db.get_pending_mutations(owner_id, primitive=_REMINDER_PRIMITIVE)
        pending_local_ids = {m["local_id"] for m in mutations}
        for mutation in mutations:
            outcome = await self._push_mutation(mutation, owner_id)
            if outcome.get("conflict"):
                conflicts.append(outcome["conflict"])
                # The mutation that caused a 404 is staged for deletion.
                staged_outcomes.append({
                    "local_id": outcome["conflict"]["local_id"],
                    "mutation_id": mutation["id"],
                })
            else:
                staged_outcomes.append(outcome)

        tombstones = self.db.get_tombstones(owner_id, primitive=_REMINDER_PRIMITIVE)
        for tombstone in tombstones:
            tombstone_outcome = await self._push_tombstone(tombstone, owner_id)
            if tombstone_outcome is None:
                raise ServerClientError("tombstone phase aborted")
            staged_outcomes.append(tombstone_outcome)
            tombstone_ids_to_delete.append(tombstone["local_id"])

        return (
            pulled_items,
            staged_outcomes,
            conflicts,
            tombstone_ids_to_delete,
            pending_local_ids,
            mutations,
        )

    async def _push_mutation(
        self, mutation: dict, owner_id: str
    ) -> dict[str, Any]:
        local_id = mutation["local_id"]
        payload = mutation.get("payload") or {}
        action = payload.get("action", "update")
        fields = payload.get("fields", {})
        idempotency_key = payload.get("idempotency_key")

        try:
            assert self.server_client is not None
            if action == "create":
                response = await self.server_client.create_reminder(
                    idempotency_key=idempotency_key, **fields
                )
                response = response if isinstance(response, dict) else {}
                return {
                    "local_id": local_id,
                    "server_id": response.get("id"),
                    "mutation_id": mutation["id"],
                }
            if action == "update":
                server_id = self._server_id_for_local(local_id, owner_id=owner_id)
                if server_id is None:
                    # The local task was created offline and has never been synced.
                    # Convert this update into a create so the data is not lost.
                    response = await self.server_client.create_reminder(
                        idempotency_key=idempotency_key, **fields
                    )
                    response = response if isinstance(response, dict) else {}
                    return {
                        "local_id": local_id,
                        "server_id": response.get("id"),
                        "mutation_id": mutation["id"],
                    }
                response = await self.server_client.update_reminder(
                    server_id, idempotency_key=idempotency_key, **fields
                )
                response = response if isinstance(response, dict) else {}
                return {
                    "local_id": local_id,
                    "server_id": response.get("id", server_id),
                    "mutation_id": mutation["id"],
                }
            if action == "delete":
                server_id = self._server_id_for_local(
                    local_id, owner_id=owner_id, from_mapping_only=True
                )
                if server_id is None:
                    return {"local_id": local_id, "mutation_id": mutation["id"]}
                await self.server_client.delete_reminder(server_id)
                return {
                    "local_id": local_id,
                    "mutation_id": mutation["id"],
                    "delete_local": True,
                }
            logger.warning(f"Unknown pending mutation action {action!r}")
            return {"local_id": local_id, "mutation_id": mutation["id"]}
        except ServerClientNotFoundError:
            local_row = self.db.get_reminder_task(local_id)
            return {
                "conflict": {
                    "local_id": local_id,
                    "server_state": {},
                    "local_state": {
                        "record": dict(local_row) if local_row else {},
                        "pending_mutation": payload,
                    },
                }
            }
        except ServerClientError:
            # Abort the whole push phase; caller records one sync error.
            raise

    async def _push_tombstone(
        self, tombstone: dict, owner_id: str
    ) -> dict[str, Any] | None:
        local_id = tombstone["local_id"]
        server_id = self._server_id_for_local(
            local_id, owner_id=owner_id, from_mapping_only=True
        )
        if server_id is None:
            return {"local_id": local_id, "delete_tombstone": True}
        try:
            assert self.server_client is not None
            await self.server_client.delete_reminder(server_id)
            return {"local_id": local_id, "delete_tombstone": True}
        except ServerClientNotFoundError:
            return {"local_id": local_id, "delete_tombstone": True}
        except ServerClientError:
            return None

    def resolve_conflict(self, conflict_id: str, resolution: str = "server") -> bool:
        conflict = self.db.get_conflict_by_id(conflict_id)
        if conflict is None:
            return False

        local_id = conflict["local_id"]
        owner_id = conflict["owner_id"]
        server_state = conflict.get("server_state") or {}
        local_state = conflict.get("local_state") or {}
        pending_mutation = (
            local_state.get("pending_mutation")
            if isinstance(local_state, dict)
            else None
        )

        if resolution == "server":
            if not server_state:
                self.db.delete_reminder_task(local_id)
                self.db.delete_sync_mapping(local_id, _REMINDER_PRIMITIVE, owner_id)
                self.db.delete_tombstone(local_id, _REMINDER_PRIMITIVE, owner_id)
            else:
                self.db.update_reminder_task(
                    local_id, **self._whitelist_reminder_fields(server_state)
                )
        elif resolution == "local":
            if not server_state and pending_mutation:
                self.db.update_reminder_task(local_id, server_id=None)
                self.db.delete_sync_mapping(local_id, _REMINDER_PRIMITIVE, owner_id)
                self.db.record_pending_mutation(
                    local_id,
                    _REMINDER_PRIMITIVE,
                    owner_id,
                    pending_mutation,
                )
            elif not server_state:
                row = self.db.get_reminder_task(local_id) or {}
                self.db.update_reminder_task(local_id, server_id=None)
                self.db.delete_sync_mapping(local_id, _REMINDER_PRIMITIVE, owner_id)
                fields = {
                    key: row.get(key)
                    for key in self._REMINDER_MUTABLE_FIELDS
                    if row.get(key) is not None
                }
                self.db.record_pending_mutation(
                    local_id,
                    _REMINDER_PRIMITIVE,
                    owner_id,
                    {"action": "create", "fields": fields},
                )
            elif pending_mutation:
                self.db.record_pending_mutation(
                    local_id,
                    _REMINDER_PRIMITIVE,
                    owner_id,
                    pending_mutation,
                )
            else:
                fields = {
                    key: value
                    for key, value in (local_state.get("record") or local_state).items()
                    if key in self._REMINDER_MUTABLE_FIELDS
                }
                self.db.record_pending_mutation(
                    local_id,
                    _REMINDER_PRIMITIVE,
                    owner_id,
                    {"action": "update", "fields": fields},
                )
            self.db.increment_conflict_retry_count(conflict_id)

        self.db.resolve_conflict(conflict_id, resolution)
        return True

    def _find_local_row(self, server_id: str) -> dict[str, Any] | None:
        """Find a local reminder row by server id, using mapping or direct lookup."""
        mapping = self.db.get_sync_mapping_by_server_id(
            server_id, _REMINDER_PRIMITIVE, self.owner_id
        )
        if mapping:
            return self.db.get_reminder_task(mapping["local_id"])

        return self.db.get_reminder_task_by_server_id(self.owner_id, server_id)

    def _server_id_for_local(
        self,
        local_id: str,
        owner_id: str | None = None,
        from_mapping_only: bool = False,
    ) -> str | None:
        """Return the server id mapped to ``local_id`` if any."""
        target_owner = owner_id if owner_id is not None else self.owner_id
        if not from_mapping_only:
            row = self.db.get_reminder_task(local_id)
            if row and row.get("server_id"):
                return row["server_id"]

        mapping = self.db.get_sync_mapping_by_local_id(
            local_id, _REMINDER_PRIMITIVE, target_owner
        )
        return mapping.get("server_id") if mapping else None

    def _record_sync_error(
        self, message: str, owner_id: str | None = None
    ) -> None:
        target_owner = owner_id if owner_id is not None else self.owner_id
        self.db._append_sync_error(target_owner, message)

    _REMINDER_MUTABLE_FIELDS = {
        "title",
        "body",
        "schedule_kind",
        "run_at",
        "cron",
        "timezone",
        "enabled",
        "next_run_at",
        "last_run_at",
        "last_status",
        "link_type",
        "link_id",
        "link_url",
    }

    def _whitelist_reminder_fields(self, server_item: dict[str, Any]) -> dict[str, Any]:
        """Return a dict of local reminder-task fields from a server payload.

        Only known fields are copied; unknown fields are dropped. Missing
        required local fields receive safe defaults so that partially-populated
        server records can still be cached.
        """
        local_fields = (
            "title",
            "body",
            "schedule_kind",
            "run_at",
            "cron",
            "timezone",
            "enabled",
            "last_run_at",
            "next_run_at",
            "last_status",
            "link_type",
            "link_id",
            "link_url",
            "created_at",
            "updated_at",
        )

        result: dict[str, Any] = {
            local_key: server_item[local_key]
            for local_key in local_fields
            if local_key in server_item
        }

        if not result.get("title"):
            result["title"] = "Untitled reminder"
        if "schedule_kind" not in result:
            result["schedule_kind"] = "one_time"

        return result
