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

#: Pending mutations recorded when a local automation definition is
#: created or updated (schedules-handoff PR-4, task 3 -- the authoring
#: facade landing in task 4 records these) are replayed to the server by
#: `SyncEngine._replay_definition_mutations`, mirroring
#: `_RESULT_REVIEW_PRIMITIVE`'s review-pushback shape.
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

        error, pushed_review_ids = await self._run_phase(
            target_owner, "Automation review pushback", self._replay_review_mutations
        )
        if error:
            phase_errors.append(error)
        if pushed_review_ids:
            logger.info(
                f"Automation review pushback for {target_owner}: "
                f"pushed {len(pushed_review_ids)} review(s)"
            )
        # Task 5 same-cycle echo (Qodo finding): thread the server_result_ids
        # just replayed above into this SAME results pull as
        # `skip_review_server_ids` -- their pending mutation is already gone
        # by now, so the pull's own pending-mutation guard can't stop a
        # server payload that still echoes the pre-review state (write/
        # read-path lag) from reverting the review that was just pushed.
        skip_review_server_ids = frozenset(pushed_review_ids or ())

        # Definition create/update push replay (Task 3), also before the
        # pulls: a successful create's server_id lands on the local row
        # inside this phase's own transaction, so the definitions pull
        # right after matches it by (owner_id, server_id) instead of
        # inserting a duplicate mirror row (Task 3 pull-ordering note).
        error, definition_push_counts = await self._run_phase(
            target_owner,
            "Automation definition push",
            self._replay_definition_mutations,
        )
        if error:
            phase_errors.append(error)
        if definition_push_counts:
            logger.info(
                f"Automation definition push for {target_owner}: "
                f"{definition_push_counts}"
            )

        error, counts = await self._run_phase(
            target_owner, "Automation definitions pull", self._pull_definitions
        )
        if error:
            phase_errors.append(error)
        if counts:
            logger.info(f"Automation definitions pull for {target_owner}: {counts}")

        error, counts = await self._run_phase(
            target_owner,
            "Automation results pull",
            self._pull_results,
            skip_review_server_ids=skip_review_server_ids,
        )
        if error:
            phase_errors.append(error)
        if counts:
            logger.info(f"Automation results pull for {target_owner}: {counts}")

        # The reminder phase's own outcome/status semantics are preserved
        # unchanged (`_sync_reminders` is the original method body) when it
        # already failed. But an "ok" OR "not_applicable" reminder phase
        # sitting beside a failed automation phase used to report clean --
        # the exact dishonesty task-23105 fixed for reminders (F2): surface
        # the first automation-phase error so the caller doesn't toast
        # success (or silent no-op) next to a fresh error badge.
        # "not_applicable" belongs here too (review round 2): it means the
        # REMINDER phase's own policy action was refused, which says
        # nothing about the automation phases' (different policy actions)
        # own genuinely-failed attempt -- that error must not be masked by
        # the reminder side's non-error status. An "error" reminder phase
        # already carries its own message and is left alone. Results/
        # definitions already pulled by an earlier phase this round are
        # not rolled back.
        if reminder_outcome.status in ("ok", "not_applicable") and phase_errors:
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
        self, owner_id: str, label: str, phase: Any, **phase_kwargs: Any,
    ) -> tuple[str | None, Any | None]:
        """Run one self-contained sync phase, containing its own failure.

        Mirrors the top-level pull()/sync_now() error discipline (task-2722):
        a runtime-mode policy refusal is logged and treated as not
        applicable, never persisted; any other error is recorded via
        `_record_sync_error` and swallowed so the phases after this one
        still run. ``phase_kwargs`` are forwarded to ``phase`` after the
        positional ``owner_id`` (e.g. `_pull_results`'s
        `skip_review_server_ids`).

        Returns:
            A ``(error, result)`` tuple. ``error`` is ``None`` on success
            or a policy refusal (not an error); otherwise the message that
            was also recorded via `_record_sync_error`. ``result`` is the
            phase's own return value (an upsert-count dict, or the
            pushed-review-ids set for the pushback phase) on success, or
            ``None`` when the phase raised or returned nothing -- callers
            use it so a truncated/failed phase never silently discards
            what did land (F2/F8).
        """
        try:
            result = await phase(owner_id, **phase_kwargs)
            return None, result
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

    async def _replay_review_mutations(self, owner_id: str) -> frozenset[str]:
        """Replay pending `automation_result_review` mutations to the server.

        Success and a 404 (the result was retired server-side) both clear
        the mutation. Any other error is left in place -- and re-raised so
        `_run_phase` records one sync error for the phase and stops
        attempting further mutations this round, mirroring
        `_push_mutation`'s "abort the whole push phase on a retryable
        server error" discipline for reminders.

        Returns:
            The ``server_result_id``s settled this cycle (pushed
            successfully or confirmed retired). `sync_now` feeds this set
            into the results pull as `skip_review_server_ids` so a stale
            same-cycle echo of these rows' pre-review state can't revert
            what was just pushed (Task 5 same-cycle echo, Qodo finding) --
            see the design comment on
            `ScheduledTasksDB.upsert_automation_results_from_server`.
        """
        assert self.server_client is not None
        mutations = self.db.get_pending_mutations(
            owner_id, primitive=_RESULT_REVIEW_PRIMITIVE
        )
        pushed_server_ids: set[str] = set()
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
            pushed_server_ids.add(server_result_id)
        return frozenset(pushed_server_ids)

    async def _replay_definition_mutations(self, owner_id: str) -> dict[str, int]:
        """Replay pending `automation_definition` mutations to the server.

        `create` -> preview(mode="create") -> a `"valid"` preview is
        consumed via `create_automation_definition`, and the response's
        server identity plus every server-wins field is adopted onto the
        local row in one transaction (`ScheduledTasksDB.
        adopt_server_definition_identity`). An `"invalid"` preview can
        never succeed by retrying the same payload, so the mutation is
        cleared and the validation errors are recorded
        (`_reject_definition_mutation`).

        `update` -> preview(mode="update", definition_id=the server id)
        -> PATCH via `update_automation_definition`. A missing
        `server_definition_id` (authored offline, never synced) or a
        `ServerClientNotFoundError` from either call (the server
        definition was deleted) converts the mutation into a create,
        mirroring `_push_mutation`'s reminder precedent.

        Any other `ServerClientError` (network/retryable) propagates so
        `_run_phase` records one sync error for the phase and leaves this
        and any later mutation queued for the next cycle -- same
        "abort the whole push phase" discipline as `_push_mutation`.

        Returns:
            Counts of what happened this cycle (``created``/``updated``/
            ``invalid``), for the caller's info log.
        """
        assert self.server_client is not None
        mutations = self.db.get_pending_mutations(
            owner_id, primitive=_DEFINITION_PRIMITIVE
        )
        counts: dict[str, int] = {}
        for mutation in mutations:
            outcome = await self._push_definition_mutation(mutation, owner_id)
            counts[outcome] = counts.get(outcome, 0) + 1
        return counts

    async def _push_definition_mutation(self, mutation: dict, owner_id: str) -> str:
        """Replay one pending `automation_definition` mutation. Returns what happened."""
        local_id = mutation["local_id"]
        mutation_id = mutation["id"]
        payload = mutation.get("payload") or {}
        action = payload.get("action")
        definition_payload = payload.get("definition_payload") or {}
        server_definition_id = payload.get("server_definition_id")

        if action == "update" and server_definition_id:
            return await self._push_definition_update(
                local_id, mutation_id, owner_id, server_definition_id, definition_payload
            )
        if action in ("create", "update"):
            # A `create` action, or an `update` authored offline and never
            # synced (no server_definition_id): both are pushed as a create.
            return await self._push_definition_create(
                local_id, mutation_id, owner_id, definition_payload
            )

        logger.warning(
            f"Unknown pending automation_definition mutation action {action!r} "
            f"for local {local_id}; dropping"
        )
        self.db.delete_pending_mutation(mutation_id)
        return "unknown"

    async def _push_definition_create(
        self,
        local_id: str,
        mutation_id: int,
        owner_id: str,
        definition_payload: dict[str, Any],
    ) -> str:
        """Preview(mode=create) then create; shared by `create` mutations and
        `update` mutations converted to create (offline-authored or 404'd)."""
        request = dict(definition_payload)
        request["mode"] = "create"
        # The server's create-mode validator (mirrored locally by
        # automation_preview.py) rejects definition_id/definition_version
        # as "not_allowed_for_create" -- strip any left over from an
        # update-shaped payload before this offline/404 conversion.
        request.pop("definition_id", None)
        request.pop("definition_version", None)

        assert self.server_client is not None
        preview = await self.server_client.preview_automation_definition(request)
        preview = preview if isinstance(preview, dict) else {}
        if preview.get("status") != "valid":
            self._reject_definition_mutation(local_id, mutation_id, owner_id, preview)
            return "invalid"

        initial_lifecycle = definition_payload.get("initial_lifecycle") or "configured"
        created = await self.server_client.create_automation_definition(
            preview.get("id"), initial_lifecycle=initial_lifecycle
        )
        created = created if isinstance(created, dict) else {}
        self.db.adopt_server_definition_identity(local_id, created)
        self.db.delete_pending_mutation(mutation_id)
        return "created"

    async def _push_definition_update(
        self,
        local_id: str,
        mutation_id: int,
        owner_id: str,
        server_definition_id: str,
        definition_payload: dict[str, Any],
    ) -> str:
        """Preview(mode=update) then PATCH; converts to create on a 404."""
        request = dict(definition_payload)
        request["mode"] = "update"
        request["definition_id"] = server_definition_id

        assert self.server_client is not None
        try:
            preview = await self.server_client.preview_automation_definition(request)
            preview = preview if isinstance(preview, dict) else {}
            if preview.get("status") != "valid":
                self._reject_definition_mutation(
                    local_id, mutation_id, owner_id, preview
                )
                return "invalid"
            updated = await self.server_client.update_automation_definition(
                server_definition_id, preview.get("id")
            )
        except ServerClientNotFoundError:
            return await self._push_definition_create(
                local_id, mutation_id, owner_id, definition_payload
            )
        updated = updated if isinstance(updated, dict) else {}
        self.db.adopt_server_definition_identity(local_id, updated)
        self.db.delete_pending_mutation(mutation_id)
        return "updated"

    def _reject_definition_mutation(
        self,
        local_id: str,
        mutation_id: int,
        owner_id: str,
        preview: dict[str, Any],
    ) -> None:
        """Clear an invalid-preview definition mutation and record why.

        A payload the server has already rejected will never succeed by
        retrying, so the mutation is dropped rather than left queued.
        """
        errors = preview.get("validation_errors") or []
        codes = ", ".join(
            f"{error.get('field')}:{error.get('code')}"
            for error in errors
            if isinstance(error, dict)
        )
        logger.warning(
            f"Automation definition mutation for local {local_id} rejected by "
            f"server preview: {codes or 'no field codes reported'}"
        )
        self._record_sync_error(
            f"Automation definition {local_id} rejected by server: "
            f"{codes or 'invalid preview'}",
            owner_id,
        )
        self.db.delete_pending_mutation(mutation_id)

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

    async def _pull_results(
        self,
        owner_id: str,
        skip_review_server_ids: frozenset[str] = frozenset(),
    ) -> dict[str, int]:
        """Walk up to `_SYNC_MAX_PAGES` newest pages of server results.

        The server's /results endpoint exposes no `updated_at` filter
        (verified at origin/dev), so this is a bounded newest-pages walk
        rather than a true incremental pull (spec §5.2 limitation): review
        drift older than the window waits for a later sync. Stops early
        on a short page or `has_more=False`; logs (info) when the cap was
        hit with more remaining, so a truncated pull is never silent.

        ``skip_review_server_ids`` (Task 5 same-cycle echo) is forwarded
        unchanged to every page's upsert call -- see the design comment on
        `ScheduledTasksDB.upsert_automation_results_from_server`.
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
            counts = self.db.upsert_automation_results_from_server(
                owner_id, page, skip_review_server_ids=skip_review_server_ids
            )
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
