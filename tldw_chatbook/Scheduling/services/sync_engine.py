"""Sync engine for reconciling local scheduled tasks with the server."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
# ADR-097: schedule_vocabulary is imported function-level in
# _server_vocab_definition_payload (boot-resident module; census).
from tldw_chatbook.Scheduling.services.server_client import (
    SchedulingServerClient,
    ServerClientError,
    ServerClientNotFoundError,
    ServerClientPolicyError,
    ServerClientValidationError,
)


_REMINDER_PRIMITIVE = "reminder_task"
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

#: Pending mutations recorded when a local automation definition is
#: created or updated (schedules-handoff PR-4, task 3 -- the authoring
#: facade landing in task 4 records these) are replayed to the server by
#: `SyncEngine._replay_definition_mutations`, mirroring
#: `_RESULT_REVIEW_PRIMITIVE`'s review-pushback shape.
_DEFINITION_PRIMITIVE = "automation_definition"

#: Pending pause/resume/archive mutations, in their OWN queue slot --
#: matches `ScheduledTasksDB._LIFECYCLE_PRIMITIVE`/`SchedulingService.
#: _LIFECYCLE_PRIMITIVE` (Qodo findings 5+6; see the DB constant's comment
#: for why a lifecycle change cannot share the definition primitive's
#: `UNIQUE(local_id, primitive, owner_id)` slot with an edit).
_LIFECYCLE_PRIMITIVE = "automation_lifecycle"

#: The `payload["action"]` verbs `_push_definition_lifecycle` replays --
#: matches `SchedulingService._LIFECYCLE_ACTIONS`'s keys. Still an action
#: set (not just "everything in `_LIFECYCLE_PRIMITIVE`") because the push
#: loop dispatches on `action` and reports per-action outcomes.
_DEFINITION_LIFECYCLE_ACTIONS = frozenset({"pause", "resume", "archive"})

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

#: Mutation actions the transfer machine (spec §6) files. They are NOT
#: content edits of the row they name, so the pull's "a local mutation is
#: pending, therefore the server's version is a conflict" rule must skip
#: them (final review I4).
_TRANSFER_ACTIONS = ("transfer_to_server", "release_from_server")

#: `_push_definition_mutation`'s outcome vocabulary that means "a mutation
#: actually reached the server this cycle" (UAT finding 3c) -- everything
#: else (`invalid`/`orphaned`/`unsynced`/`transfer_skipped`/
#: `transfer_cas_skipped`/`transfer_failed`/`transfer_orphaned`/
#: `unknown`/`{action}_not_found`) settled without a real push. Read by
#: `_replay_definition_mutations`, the only other writer of
#: `last_push_at` besides `_sync_reminders`'s own reminder-scoped one.
_DEFINITION_PUSH_SUCCESS_OUTCOMES = frozenset(
    {"created", "updated", "released", "transferred", "pause", "resume", "archive"}
)

#: Review round 1 finding 2: the complement of the set above, so a
#: drift-guard test can assert every outcome `_push_definition_mutation`
#: (+ its five `_push_definition_*` helpers, + `_replay_definition_
#: mutations`'s own `"transfer_skipped"`) can actually return is
#: CLASSIFIED as one or the other -- an unclassified new outcome fails
#: that test instead of silently never moving `last_push_at`.
#: `{action}_not_found` is `_push_definition_lifecycle`'s dynamic
#: NotFoundError outcome (`action` is always one of the three lifecycle
#: actions below).
_DEFINITION_PUSH_NON_SUCCESS_OUTCOMES = frozenset(
    {
        "unknown",
        "invalid",
        "orphaned",
        "unsynced",
        "transfer_cas_skipped",
        "transfer_orphaned",
        "transfer_failed",
        "transfer_skipped",
        "pause_not_found",
        "resume_not_found",
        "archive_not_found",
    }
)


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

    ``status``/``error`` describe ONLY the reminder phase (`_sync_
    reminders`) -- the automation phases (review pushback, definition
    push, definitions pull, results pull) are independently contained by
    `_run_phase` and run regardless of the reminder phase's own outcome.
    UAT finding 3c: `sync_now` used to collapse any later phase's error
    into `status="error"` + `error=phase_errors[0]`, so a cycle whose
    definition push succeeded still toasted "Sync failed" over an
    unrelated results-pull 404. ``phase_errors`` carries those phases'
    own LABELED failures (e.g. ``"Automation results pull: ..."``)
    instead, so a caller can report the reminder outcome honestly *and*
    surface the rest as its own notice -- never silently dropped, never
    misreported as the whole cycle failing.
    """

    status: str
    pulled: int = 0
    pushed: int = 0
    error: str | None = None
    phase_errors: tuple[str, ...] = ()


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

    def _settle_orphaned_transfer_mutations(self, target_owner: str) -> None:
        """Settle a `transfer_to_server` mutation whose scope no longer
        matches the active server (task-3, root-causes.md #5 / ruling 4).

        `_network_phase`'s mutation query is deliberately scoped to
        `target_owner` (`get_pending_mutations(target_owner, ...)`), so a
        mutation recorded under a PRIOR server scope -- the configured
        server's address changed underneath it -- is never selected,
        never attempted, and the row hangs `to_server_pending` forever
        with no route to Retry/Cancel (the UAT's permanent "Moving to
        server..."). This is the one place that looks ACROSS every owner
        scope (`get_pending_mutations(owner_id=None, ...)`, the same
        all-owners mode `cancel_transfer`'s own lookup already relies on)
        to find one -- mirrors `recover_inflight_transfers`'s per-row,
        per-primitive exception-guarded shape, run once per sync cycle
        (not startup-only) so a mid-session reconfiguration settles on
        the very next sync rather than waiting for a restart.

        Only a still-unattempted mutation (`to_server_pending`) is
        touched -- an ambiguous `to_server_sent` row belongs to
        `recover_inflight_transfers`, not this sweep (same division of
        labor that method's own docstring draws for other stuck states).
        The CAS's own `expected=` guard makes the read-and-write atomic;
        no separate row fetch is needed first.
        """
        for primitive in (_REMINDER_PRIMITIVE, _DEFINITION_PRIMITIVE):
            try:
                mutations = self.db.get_pending_mutations(
                    owner_id=None, primitive=primitive
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    f"Orphaned-transfer sweep failed to list mutations "
                    f"for {primitive}"
                )
                continue
            for mutation in mutations:
                payload = mutation.get("payload") or {}
                if payload.get("action") != "transfer_to_server":
                    continue
                if payload.get("transfer_errors"):
                    continue  # already settled
                mutation_owner = mutation.get("owner_id")
                if mutation_owner == target_owner:
                    continue  # still valid for the active server
                local_id = mutation.get("local_id")
                try:
                    settled = self.db.set_transfer_state(
                        primitive,
                        local_id,
                        "to_server_failed",
                        expected=("to_server_pending",),
                        pending_mutation={
                            "primitive": primitive,
                            "owner_id": mutation_owner,
                            "payload": {
                                **payload,
                                "transfer_errors": [
                                    "The server this move was queued for "
                                    "is no longer configured."
                                ],
                            },
                        },
                    )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        f"Failed to settle orphaned transfer mutation for "
                        f"{primitive} {local_id}"
                    )
                    continue
                if settled:
                    logger.warning(
                        f"Orphaned transfer_to_server mutation settled to "
                        f"to_server_failed for {primitive} {local_id} "
                        f"(queued for {mutation_owner!r}; active server "
                        f"is {target_owner!r})"
                    )

    async def pull(self, owner_id: str | None = None) -> None:
        """Public entry point to pull server reminders for the given owner."""
        target_owner = owner_id if owner_id is not None else self.owner_id
        if self.server_client is None:
            return

        self._settle_orphaned_transfer_mutations(target_owner)
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

        self._settle_orphaned_transfer_mutations(target_owner)
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
            phase_errors.append(f"Automation review pushback: {error}")
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
        error, definition_push_result = await self._run_phase(
            target_owner,
            "Automation definition push",
            self._replay_definition_mutations,
        )
        if error:
            phase_errors.append(f"Automation definition push: {error}")
        definition_push_counts, pushed_lifecycle_server_ids = (
            definition_push_result
            if definition_push_result is not None
            else ({}, frozenset())
        )
        if definition_push_counts:
            logger.info(
                f"Automation definition push for {target_owner}: "
                f"{definition_push_counts}"
            )
        # Task 2 same-cycle echo (mirrors `skip_review_server_ids` above):
        # thread the server_definition_ids just pushed above into this SAME
        # definitions pull so a stale same-cycle echo of the pre-transition
        # lifecycle can't revert what was just pushed -- see the design
        # comment on `ScheduledTasksDB.upsert_automation_definitions_from_server`.
        error, counts = await self._run_phase(
            target_owner,
            "Automation definitions pull",
            self._pull_definitions,
            skip_lifecycle_server_ids=pushed_lifecycle_server_ids,
        )
        if error:
            phase_errors.append(f"Automation definitions pull: {error}")
        if counts:
            logger.info(f"Automation definitions pull for {target_owner}: {counts}")

        error, counts = await self._run_phase(
            target_owner,
            "Automation results pull",
            self._pull_results,
            skip_review_server_ids=skip_review_server_ids,
        )
        if error:
            phase_errors.append(f"Automation results pull: {error}")
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
        #
        # UAT finding 3c: this used to collapse ANY later-phase error into
        # `status="error"` (masking a genuinely successful reminder/
        # definition-push phase behind "Sync failed"). `status`/`error`
        # now describe ONLY the reminder phase; `phase_errors` carries
        # the labeled rest so a caller can report both honestly instead
        # of picking one truth to tell.
        if phase_errors:
            return replace(reminder_outcome, phase_errors=tuple(phase_errors))

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
                # UAT finding 4 / root-causes.md #4 (ghost row): `_network_
                # phase` pulls BEFORE it pushes, so a reminder released THIS
                # cycle is still listed in `pulled_items` from the
                # pre-release snapshot. Applying that stale item would
                # re-insert the mirror `_push_reminder_release` just tore
                # down, with a brand-new local id no tombstone can remove --
                # it then reads as "deleted on server" forever. Exact twin
                # of the `adopted_server_id` seen-set guard below (this
                # method, further down): filter what this cycle just
                # released out of the stale payload before applying it.
                released_server_ids = {
                    outcome["released_server_id"]
                    for outcome in staged_outcomes
                    if outcome.get("released_server_id")
                }
                if released_server_ids:
                    pulled_items = [
                        item
                        for item in pulled_items
                        if item.get("id") not in released_server_ids
                    ]

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
                # `_network_phase` pulls BEFORE it pushes, so a reminder
                # this cycle transferred to the server carries a brand-new
                # server_id the pull could not possibly have listed -- and
                # `convert_row_to_server_mirror` also flipped its owner_id
                # to the destination scope, which is exactly the
                # (owner_id, server_id) pair the deletion scan reads. Left
                # unmerged, a SUCCESSFUL move recorded a "the server
                # deleted this row" conflict on the same cycle (final
                # review C1). Same precedent as the results pushback
                # phase's `skip_review_server_ids`: what this cycle just
                # wrote counts as seen.
                seen_server_ids.update(
                    outcome["adopted_server_id"]
                    for outcome in staged_outcomes
                    if outcome.get("adopted_server_id")
                )
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
            phase's own return value (an upsert-count dict; the pushed-
            review-ids set for the review pushback phase; or the
            ``(counts, pushed_lifecycle_server_ids)`` tuple for the
            definition push phase) on success, or ``None`` when the phase
            raised or returned nothing -- callers use it so a truncated/
            failed phase never silently discards what did land (F2/F8).
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

    async def _replay_definition_mutations(
        self, owner_id: str
    ) -> tuple[dict[str, int], frozenset[str]]:
        """Replay pending automation-definition mutations to the server.

        Reads BOTH definition queue slots -- `_DEFINITION_PRIMITIVE`
        (create/update/transfer) and `_LIFECYCLE_PRIMITIVE`
        (pause/resume/archive, its own slot since Qodo findings 5+6) --
        and settles them in ONE phase rather than two. The dispatch is
        `payload["action"]`-keyed either way, and one phase keeps the
        existing containment discipline intact unchanged: one
        `_run_phase` error entry, one abort-on-retryable-error boundary,
        one `pushed_lifecycle_server_ids` set threaded into the pull
        below it.

        Definition mutations are replayed FIRST so that, for a row
        carrying both, the edit's server echo is adopted while the
        lifecycle mutation is still queued -- which is exactly when
        `adopt_server_definition_identity`'s pull-guard withholds the
        echoed (pre-transition) `lifecycle`. The reverse order converges
        too: the lifecycle push lands server-side before the edit's
        round-trip, so the echo already carries the new value. Both
        orders are pinned (`test_sync_now_replays_a_queued_edit_and_
        pause_together`).

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

        A retryable `ServerClientError` (timeout/5xx/unavailable)
        propagates so `_run_phase` records one sync error for the phase
        and leaves this and any later mutation queued for the next cycle
        -- same "abort the whole push phase" discipline as
        `_push_mutation`. A non-retryable 4xx does NOT abort: it settles
        that one mutation and the loop continues with the rest (see
        `_push_definition_mutation`).

        Returns:
            A ``(counts, pushed_lifecycle_server_ids)`` tuple. ``counts``
            is what happened this cycle (``created``/``updated``/
            ``invalid``/``orphaned``/...), for the caller's info log.
            ``pushed_lifecycle_server_ids`` is the `server_definition_id`s
            whose `pause`/`resume`/`archive` mutation settled successfully
            THIS cycle -- `sync_now` feeds this into the definitions pull
            as `skip_lifecycle_server_ids` (Task 2 same-cycle echo,
            mirroring `_replay_review_mutations`'s
            `skip_review_server_ids`) so a stale same-cycle echo of the
            pre-transition lifecycle can't revert what was just pushed --
            see the design comment on
            `ScheduledTasksDB.upsert_automation_definitions_from_server`.
        """
        assert self.server_client is not None
        mutations = self.db.get_pending_mutations(
            owner_id, primitive=_DEFINITION_PRIMITIVE
        ) + self.db.get_pending_mutations(owner_id, primitive=_LIFECYCLE_PRIMITIVE)
        counts: dict[str, int] = {}
        pushed_lifecycle_server_ids: set[str] = set()
        for mutation in mutations:
            payload = mutation.get("payload") or {}
            if payload.get("transfer_errors"):
                # A `transfer_to_server` mutation that already settled as a
                # definitive failure (spec §6.1.5, ruling 3): never
                # auto-retried by this replay loop -- recovery is a user
                # retry/cancel via Task 6's facade, not another sync cycle.
                counts["transfer_skipped"] = counts.get("transfer_skipped", 0) + 1
                continue
            action = payload.get("action")
            server_definition_id = payload.get("server_definition_id")
            outcome = await self._push_definition_mutation(mutation, owner_id)
            counts[outcome] = counts.get(outcome, 0) + 1
            if (
                action in _DEFINITION_LIFECYCLE_ACTIONS
                and outcome == action
                and server_definition_id
            ):
                pushed_lifecycle_server_ids.add(server_definition_id)
        if any(outcome in _DEFINITION_PUSH_SUCCESS_OUTCOMES for outcome in counts):
            # UAT finding 3c: `_sync_reminders` is the ONLY existing
            # `last_push_at` writer, and only for its own reminder
            # pushes -- a definition edit/lifecycle push (with nothing
            # on the reminder side this cycle) left the header showing
            # "Last push: -" forever. Direct call, not
            # `asyncio.to_thread`, matching every other `self.db.*` call
            # in this method. Only stamped on a genuine push (not
            # `invalid`/`unsynced`/`transfer_skipped`/etc.) so an
            # all-noop cycle can never move it.
            self.db.update_sync_state(owner_id, last_push_at=now_utc_iso())
        return counts, frozenset(pushed_lifecycle_server_ids)

    async def _push_definition_mutation(self, mutation: dict, owner_id: str) -> str:
        """Replay one pending `automation_definition` mutation. Returns what happened.

        A server-side 4xx (`ServerClientValidationError`, e.g. a 409
        `definition_version_conflict` or a 422 `schedule_invalid`) is
        non-retryable by construction -- `_call_with_retry` raises it
        immediately without retrying -- so it is settled here exactly like
        an invalid preview: the mutation is cleared and the reason
        recorded as a sync error (`_reject_definition_mutation`). Letting
        it raise instead aborted the whole push phase, so ONE poisoned
        mutation blocked every other definition mutation for that owner
        forever (final review I3). Only genuinely retryable errors
        (timeout/5xx/unavailable) still abort the phase, and a
        `ServerClientPolicyError` still propagates untouched -- that one is
        an account-level refusal that may be granted later, so the user's
        queued work must survive it rather than be discarded.
        """
        local_id = mutation["local_id"]
        mutation_id = mutation["id"]
        payload = mutation.get("payload") or {}
        action = payload.get("action")
        definition_payload = payload.get("definition_payload") or {}
        server_definition_id = payload.get("server_definition_id")

        try:
            if action == "update" and server_definition_id:
                return await self._push_definition_update(
                    local_id,
                    mutation_id,
                    owner_id,
                    server_definition_id,
                    definition_payload,
                )
            if action in ("create", "update"):
                # A `create` action, or an `update` authored offline and never
                # synced (no server_definition_id): both are pushed as a create.
                return await self._push_definition_create(
                    local_id, mutation_id, owner_id, definition_payload
                )
            if action in ("pause", "resume", "archive"):
                return await self._push_definition_lifecycle(
                    local_id, mutation_id, owner_id, action, server_definition_id
                )
            if action == "transfer_to_server":
                # Owns its own ServerClientValidationError/definitive-failure
                # handling internally (spec §6.1.5) -- never raises it back
                # out, so the generic reject-and-clear except clause below
                # never sees a transfer mutation.
                return await self._push_definition_transfer(
                    local_id, mutation_id, owner_id, payload
                )
            if action == "release_from_server":
                # Owns its own ServerClientNotFoundError handling internally
                # (spec §6.2.3: a 404 on release is an ack, not a conflict
                # or a definitive failure) -- never raises it back out. A
                # genuine ServerClientValidationError still falls through
                # to the generic reject-and-clear clause below, same as
                # every other definition action.
                return await self._push_definition_release(
                    local_id, mutation_id, owner_id, payload
                )
        except ServerClientPolicyError:
            raise
        except ServerClientValidationError as exc:
            self._reject_definition_mutation(
                local_id,
                mutation_id,
                owner_id,
                # The server's own error code IS the useful text here, and
                # `_reject_definition_mutation` records `field:code` pairs
                # (not messages) -- so put it where the user will see it.
                {
                    "validation_errors": [
                        {"field": "_server", "code": str(exc), "message": str(exc)}
                    ]
                },
            )
            return "invalid"

        logger.warning(
            f"Unknown pending {mutation.get('primitive')} mutation action "
            f"{action!r} for local {local_id}; dropping"
        )
        self.db.delete_pending_mutation(mutation_id)
        return "unknown"

    @staticmethod
    def _server_vocab_definition_payload(
        definition_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Copy `definition_payload` with its `schedule` in server vocabulary.

        The ONE shared call site for `to_server_schedule` (Task 3 review
        note) -- locally authored/previewed schedules are in CLIENT
        vocabulary (schedule_compute.py); the server's own preview only
        validates `kind`, so an untranslated payload would pass preview
        and then silently never arm server-side (schedule_vocabulary.py).
        Used by `_push_definition_create`, `_push_definition_update`, and
        `_push_definition_transfer` -- each still applies its own
        mode/definition_id handling on top of the returned copy.
        """
        request = dict(definition_payload)
        schedule = request.get("schedule")
        if isinstance(schedule, dict):
            from tldw_chatbook.Scheduling.schedule_vocabulary import to_server_schedule

            request["schedule"] = to_server_schedule(schedule)
        return request

    async def _push_definition_create(
        self,
        local_id: str,
        mutation_id: int,
        owner_id: str,
        definition_payload: dict[str, Any],
    ) -> str:
        """Preview(mode=create) then create; shared by `create` mutations and
        `update` mutations converted to create (offline-authored or 404'd).

        If the local row is gone by the time the server echoes back (deleted
        between queueing and replay), the adopt finds nothing to write and the
        new server definition has no local home. The mutation is still cleared
        -- replaying it would create a SECOND server definition, not recover
        the first -- and a sync error naming both ids is recorded so the
        orphan is discoverable. Deleting it server-side is not this method's
        call; definition lifecycle actions are a later phase.
        """
        request = self._server_vocab_definition_payload(definition_payload)
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
        adopted = self.db.adopt_server_definition_identity(local_id, created)
        self.db.delete_pending_mutation(mutation_id)
        if not adopted:
            server_definition_id = created.get("id") or "unknown"
            logger.warning(
                f"Automation definition {local_id} vanished locally before its "
                f"create push landed; server definition {server_definition_id} "
                f"has no local row"
            )
            self._record_sync_error(
                f"Automation definition {local_id} was removed while it was "
                f"being created on the server; the server copy "
                f"({server_definition_id}) is still there and is not linked "
                f"to any local automation",
                owner_id,
            )
            return "orphaned"
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
        request = self._server_vocab_definition_payload(definition_payload)
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

    async def _push_definition_lifecycle(
        self,
        local_id: str,
        mutation_id: int,
        owner_id: str,
        action: str,
        server_definition_id: str | None,
    ) -> str:
        """Replay one pending pause/resume/archive lifecycle mutation.

        A direct endpoint call -- no preview, unlike create/update, since a
        lifecycle transition is not a payload edit. A missing
        `server_definition_id` means the definition was never synced, so a
        lifecycle action can't apply to it; unlike `update`, there is no
        create to convert this into, so the mutation is just dropped.

        `ServerClientNotFoundError` (the server definition was deleted
        between queueing and replay) clears the mutation with an info log
        rather than a sync error -- there is nothing left server-side to
        transition, and no local edit to preserve by converting to a
        create the way the update leg does.

        A `ServerClientValidationError` (e.g. a 409
        `scheduled_task_lifecycle_transition_invalid`) is left to
        propagate: the caller (`_push_definition_mutation`) already
        settles it via the same per-mutation rejection path create/update
        use, so one poisoned lifecycle mutation never blocks the rest of
        the queue either.
        """
        if not server_definition_id:
            logger.warning(
                f"Pending {action!r} automation_definition mutation for local "
                f"{local_id} has no server_definition_id (never synced); dropping"
            )
            self.db.delete_pending_mutation(mutation_id)
            return "unsynced"

        assert self.server_client is not None
        method = getattr(self.server_client, f"{action}_automation_definition")
        try:
            response = await method(server_definition_id)
        except ServerClientNotFoundError:
            logger.info(
                f"Automation definition {server_definition_id} ({action}) not "
                f"found server-side; dropping its pending lifecycle mutation"
            )
            self.db.delete_pending_mutation(mutation_id)
            return f"{action}_not_found"

        response = response if isinstance(response, dict) else {}
        # Delete BEFORE mirroring the echo (Task 2 lifecycle pull-guard):
        # this mutation is what the guard's own pending-mutation check
        # would key off (same local_id/owner_id in `_LIFECYCLE_
        # PRIMITIVE`'s slot) -- if it were still present
        # when `upsert_automation_definitions_from_server` runs, the guard
        # would (correctly, in general) withhold `lifecycle` from THIS
        # write too, and the just-confirmed transition would never reach
        # the local row. Clearing it first means the guard sees nothing
        # pending for this row and the echo's `lifecycle` writes through
        # normally, exactly as it did before the guard existed.
        self.db.delete_pending_mutation(mutation_id)
        self.db.upsert_automation_definitions_from_server(owner_id, [response])
        return action

    async def _push_definition_release(
        self,
        local_id: str,
        mutation_id: int,
        owner_id: str,
        payload: dict[str, Any],
    ) -> str:
        """Replay a pending `release_from_server` automation_definition mutation (spec §6.2).

        ``local_id`` is the SERVER-OWNED MIRROR row this release targets
        (the row the release was queued against); ``payload["local_copy_
        id"]`` is the DORMANT local-owner copy `create_local_copy_from_
        mirror` already created (spec §6.2.1) -- a DIFFERENT row, which is
        what actually arms once this call acks. There is no disarm-before-
        send step here the way `_push_definition_transfer` has one: the
        copy was already dormant from the moment it was created, so there
        is no double-dispatch window to close by transitioning it first --
        it only ever moves once, straight to armed, on ack.

        A release IS an archive, just triggered by a transfer rather than
        a direct user action -- this reuses `_push_definition_lifecycle`'s
        exact endpoint (`archive_automation_definition`) and its "mirror
        the echoed response back onto the row" step
        (`upsert_automation_definitions_from_server`). `ServerClientNot
        FoundError` (the server definition is already gone) is treated
        exactly as an ack (spec §6.2.3: nothing left server-side to
        mirror, but the local copy still arms) -- caught here, not left to
        the outer `_push_definition_mutation` try/except, which would
        otherwise have nothing that turns a definition 404 into anything
        but a definitive-failure reject.

        A `ServerClientValidationError` is NOT caught here -- it
        propagates to `_push_definition_mutation`'s own generic reject-
        and-clear clause (`_reject_definition_mutation`), the same
        settlement every other definitively-failed definition mutation
        gets; the local copy is left dormant, matching the spec's
        offline/unacked behavior (there is nothing to arm without a real
        release). A retryable `ServerClientError` (timeout/5xx/
        unavailable) is left to propagate so `_run_phase` records one sync
        error and the mutation stays queued for the next cycle -- the copy
        stays dormant until an actual ack, pinned by
        `test_sync_now_definition_release_retryable_error_keeps_copy_
        dormant`.
        """
        server_definition_id = payload.get("server_definition_id")
        local_copy_id = payload.get("local_copy_id")
        if not server_definition_id:
            logger.warning(
                f"Pending release_from_server automation_definition mutation "
                f"for local {local_id} has no server_definition_id; dropping"
            )
            self.db.delete_pending_mutation(mutation_id)
            return "unsynced"

        assert self.server_client is not None
        try:
            response = await self.server_client.archive_automation_definition(
                server_definition_id
            )
        except ServerClientNotFoundError:
            logger.info(
                f"Automation definition {server_definition_id} release: "
                "already gone server-side; treating as acked"
            )
        else:
            response = response if isinstance(response, dict) else {}
            self.db.upsert_automation_definitions_from_server(owner_id, [response])

        if local_copy_id:
            self.db.clear_transfer_state(
                _DEFINITION_PRIMITIVE,
                local_copy_id,
                expected=("from_server_pending",),
            )
        self.db.delete_pending_mutation(mutation_id)
        return "released"

    async def _push_definition_transfer(
        self,
        local_id: str,
        mutation_id: int,
        owner_id: str,
        payload: dict[str, Any],
    ) -> str:
        """Replay a pending `transfer_to_server` automation_definition mutation.

        Disarm-before-send (spec §6.1.2): CAS `to_server_pending` ->
        `to_server_sent` (Task 1's `set_transfer_state`) BEFORE the create
        request goes out, so a crash or a failed send never leaves the row
        armed both locally and (about to be) on the server. A failed CAS
        means the row is no longer `to_server_pending` -- a concurrent
        cancel (which clears the state entirely) or a mutation that is
        still `to_server_sent` from a PRIOR ambiguous-timeout attempt
        (deliberately left alone here; un-sticking a stuck `to_server_sent`
        row is Task 6's startup `recover_inflight_transfers`, not this
        replay loop) -- either way, this replay is a silent no-op that
        touches neither the server nor the mutation.

        A definitive failure (an invalid preview, or a
        `ServerClientValidationError` from the create call itself -- e.g.
        a 409/422) settles via `_fail_transfer_mutation`: CAS to
        `to_server_failed` (re-arms the row locally, Task 1) and the
        mutation is RETAINED with `transfer_errors` embedded, never
        auto-retried (`_replay_definition_mutations`'s skip check). Any
        other `ServerClientError` (timeout/5xx/unavailable) is left to
        propagate so `_run_phase` records one sync error and aborts the
        phase -- `to_server_sent` and the mutation both stay in place, and
        the next replay (after Task 6's recovery re-arms it) re-runs
        preview->create, which is safe because the server's create is
        hash-idempotent (ruling 4).
        """
        disarmed = self.db.set_transfer_state(
            _DEFINITION_PRIMITIVE,
            local_id,
            "to_server_sent",
            expected=("to_server_pending",),
        )
        if not disarmed:
            logger.info(
                f"Automation definition {local_id} transfer_to_server mutation "
                "skipped: not in to_server_pending (concurrent cancel, or "
                "still to_server_sent from a prior attempt awaiting recovery)"
            )
            return "transfer_cas_skipped"

        definition_payload = payload.get("definition_payload") or {}
        # `_server_vocab_definition_payload` (Task 3 review note): a
        # transfer payload is stored client-vocab and never pre-translated
        # at queue time -- this shared helper translates it exactly once,
        # the same call `_push_definition_create` makes, without calling
        # into that method itself, whose ack/failure handling (adopt +
        # clear-on-invalid) differs from a transfer's (convert-or-merge +
        # retain-with-errors-on-invalid).
        request = self._server_vocab_definition_payload(definition_payload)
        request["mode"] = "create"
        request.pop("definition_id", None)
        request.pop("definition_version", None)

        assert self.server_client is not None
        try:
            preview = await self.server_client.preview_automation_definition(request)
            preview = preview if isinstance(preview, dict) else {}
            if preview.get("status") != "valid":
                errors = preview.get("validation_errors") or []
                error_texts = [
                    f"{error.get('field')}:{error.get('code')}"
                    for error in errors
                    if isinstance(error, dict)
                ] or ["invalid preview"]
                self._fail_transfer_mutation(
                    _DEFINITION_PRIMITIVE, local_id, owner_id, payload, error_texts
                )
                return "transfer_failed"

            # The source row's OWN current lifecycle, not the payload's --
            # a transfer's source definition already exists locally with a
            # real lifecycle (configured or paused; Task 6's refusal gate
            # keeps archived/solved rows from ever queuing a transfer).
            source_row = self.db.get_automation_definition(local_id) or {}
            initial_lifecycle = source_row.get("lifecycle") or "configured"
            created = await self.server_client.create_automation_definition(
                preview.get("id"), initial_lifecycle=initial_lifecycle
            )
        except ServerClientPolicyError:
            raise
        except ServerClientValidationError as exc:
            self._fail_transfer_mutation(
                _DEFINITION_PRIMITIVE, local_id, owner_id, payload, [str(exc)]
            )
            return "transfer_failed"

        created = created if isinstance(created, dict) else {}
        result = self.db.convert_row_to_server_mirror(
            _DEFINITION_PRIMITIVE, local_id, created, owner_id
        )
        self.db.delete_pending_mutation(mutation_id)
        if result == "vanished":
            server_id = created.get("id") or "unknown"
            logger.warning(
                f"Automation definition {local_id} vanished locally before "
                f"its transfer_to_server push landed; server definition "
                f"({server_id}) is not linked to any local automation"
            )
            self._record_sync_error(
                f"Automation definition {local_id} was removed locally while "
                f"it was being transferred to the server; the server copy "
                f"({server_id}) is still there and is not linked to any "
                "local automation",
                owner_id,
            )
            return "transfer_orphaned"
        return "transferred"

    def _fail_transfer_mutation(
        self,
        table_kind: str,
        local_id: str,
        owner_id: str,
        payload: dict[str, Any],
        errors: list[str],
    ) -> None:
        """Settle a definitively-failed `transfer_to_server` mutation (spec §6.1.5).

        CAS `to_server_sent` -> `to_server_failed` -- NOT a dormant state
        (Task 1), so the row re-arms and keeps executing locally. The
        mutation is RETAINED, not cleared, with `transfer_errors` embedded
        in its payload (ruling 3) via `record_pending_mutation`'s existing
        "INSERT OR REPLACE keyed by (local_id, primitive, owner_id)"
        upsert -- which also preserves the original `idempotency_key`
        already present in ``payload``. `_replay_definition_mutations`'s
        and `_network_phase`'s skip checks (a TRUTHY
        `payload["transfer_errors"]` -- an empty list would re-arm
        auto-retry, which is why this always writes a non-empty one) then
        stop this mutation from being retried automatically
        -- recovery is a user retry/cancel action via Task 6's facade, not
        another sync cycle.
        """
        re_armed = self.db.set_transfer_state(
            table_kind, local_id, "to_server_failed", expected=("to_server_sent",)
        )
        if not re_armed:
            # Only plausible under multi-process concurrency -- nothing
            # else touches this row's transfer_state within a single
            # sync_now() call. The mutation is still retained with
            # transfer_errors below regardless (that half of the contract
            # always holds); this just makes the state-column half's
            # silent no-op visible instead of swallowed.
            logger.warning(
                f"Transfer failure CAS to_server_sent->to_server_failed did "
                f"not land for {table_kind} {local_id} (row no longer "
                "to_server_sent -- likely a concurrent process)"
            )
        self.db.record_pending_mutation(
            local_id, table_kind, owner_id, {**payload, "transfer_errors": errors}
        )
        message = "; ".join(errors) or "invalid preview"
        logger.warning(
            f"Transfer to server failed for {table_kind} {local_id}: {message}"
        )
        self._record_sync_error(
            f"Transfer to server failed for {table_kind} {local_id}: {message}",
            owner_id,
        )

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

    async def _automation_capabilities_available(self) -> bool:
        """Whether the server still exposes Scheduled Tasks automation at all.

        task-3 (schedules UAT remediation ruling 5) capabilities
        handshake: `SchedulingServerClient.get_capabilities` returns
        ``None`` only for a definitive "the capabilities route itself
        does not exist" answer (a server old enough to predate Scheduled
        Tasks automation entirely) -- that is the ONE case this returns
        ``False`` for, so `_pull_definitions`/`_pull_results` can skip
        outright rather than let every page 404 (root-causes.md #7).
        Fails OPEN on any other outcome (a transient probe failure must
        not silently stop pulling automations that otherwise work fine --
        the per-call `ServerClientNotFoundError` handling below is what
        catches the narrower "capabilities exist, this ONE route doesn't
        yet" case a probe alone cannot distinguish).
        """
        assert self.server_client is not None
        try:
            return await self.server_client.get_capabilities() is not None
        except ServerClientError:
            return True
        except Exception:  # noqa: BLE001
            return True

    async def _pull_definitions(
        self,
        owner_id: str,
        skip_lifecycle_server_ids: frozenset[str] = frozenset(),
    ) -> dict[str, int]:
        """Page up to `_SYNC_MAX_PAGES` of the server's automation definitions.

        Upserted per page (not batched to the end) so a later page's
        failure still leaves earlier pages' rows mirrored -- the same
        partial-progress-survives-a-failure shape as `_pull_results`.
        Stops early on an empty page or `has_more=False`; logs (info) when
        the cap was hit with more remaining, so a truncated pull is never
        silent (F4: this was an unbounded `while True` against the server).

        ``skip_lifecycle_server_ids`` (Task 2 same-cycle echo) is
        forwarded unchanged to every page's upsert call -- see the design
        comment on
        `ScheduledTasksDB.upsert_automation_definitions_from_server`.

        task-3 capabilities handshake: returns an empty (no-op) result
        outright when the server does not expose Scheduled Tasks
        automation at all, rather than paging into a guaranteed 404.
        """
        assert self.server_client is not None
        if not await self._automation_capabilities_available():
            return {}
        totals: dict[str, int] = {}
        offset = 0
        for _page_num in range(_SYNC_MAX_PAGES):
            response = await self.server_client.list_automation_definitions(
                limit=_RESULTS_PAGE_SIZE, offset=offset
            )
            if not isinstance(response, dict):
                response = {}
            page = list(response.get("items") or [])
            counts = self.db.upsert_automation_definitions_from_server(
                owner_id, page, skip_lifecycle_server_ids=skip_lifecycle_server_ids
            )
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

        task-3 capabilities handshake: returns an empty (no-op) result
        outright when the server does not expose Scheduled Tasks
        automation at all (rather than paging into a guaranteed 404). A
        server whose capabilities probe DOES succeed but whose `/results`
        route specifically is missing (root-causes.md #7's actual UAT
        repro -- a mid-rollout server new enough for capabilities, too
        old for the results-inbox surface) cannot be told apart by that
        probe alone; the `ServerClientNotFoundError` below is what turns
        THAT case into the same honest copy instead of a raw
        ``scheduled_task_not_found`` poisoning the sync verdict (UAT
        Minor 24 / Major 7).
        """
        assert self.server_client is not None
        if not await self._automation_capabilities_available():
            return {}
        totals: dict[str, int] = {}
        offset = 0
        for _page_num in range(_SYNC_MAX_PAGES):
            try:
                response = await self.server_client.list_automation_results(
                    limit=_RESULTS_PAGE_SIZE, offset=offset
                )
            except ServerClientNotFoundError as exc:
                raise ServerClientNotFoundError(
                    "This server does not provide the results inbox "
                    "(server too old)."
                ) from exc
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
        # `_apply_pulled_reminders`' rule is "a local EDIT is pending, so
        # the server's version must not silently win" -- a transfer action
        # is not an edit of the pulled row's content, and a
        # `release_from_server` mutation is deliberately keyed on the
        # MIRROR the pull is listing, so leaving them in made EVERY
        # server -> local move raise a bogus conflict on the very cycle
        # that performed it (final review I4).
        pending_local_ids = {
            m["local_id"]
            for m in mutations
            if (m.get("payload") or {}).get("action") not in _TRANSFER_ACTIONS
        }
        for mutation in mutations:
            mutation_payload = mutation.get("payload") or {}
            if mutation_payload.get("transfer_errors"):
                # A `transfer_to_server` mutation that already settled as a
                # definitive failure (spec §6.1.5, ruling 3): never
                # auto-retried by this replay loop -- recovery is a user
                # retry/cancel via Task 6's facade, not another sync cycle.
                continue
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
            if action == "transfer_to_server":
                # Owns its own ServerClientValidationError/definitive-failure
                # handling internally (spec §6.1.5) -- never raises it back
                # out, so the `except ServerClientError: raise` below never
                # sees a transfer mutation's own definitive failure (a
                # genuinely retryable error still propagates through it
                # untouched, same as every other action here).
                return await self._push_reminder_transfer(
                    local_id, mutation["id"], owner_id, payload
                )
            if action == "release_from_server":
                # Owns its own ServerClientNotFoundError handling internally
                # (spec §6.2.3: a 404 on release is an ack, not the
                # conflict the `except ServerClientNotFoundError` clause
                # below turns every other action's 404 into) -- never
                # raises it back out. A genuine retryable ServerClientError
                # still propagates through it untouched, same as every
                # other action here.
                return await self._push_reminder_release(
                    local_id, mutation["id"], owner_id, payload
                )
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

    async def _push_reminder_transfer(
        self,
        local_id: str,
        mutation_id: int,
        owner_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Replay a pending `transfer_to_server` reminder mutation (spec §6.1).

        Same disarm-before-send / definitive-failure-vs-retryable split as
        `_push_definition_transfer` (see its docstring for the full
        reasoning); the differences are reminder-specific:

        - No preview step -- the reminder create call itself is the only
          request, so a definitive failure is a `ServerClientValidation
          Error` straight from `create_reminder`.
        - The create carries `link_type="chatbook_transfer"` +
          `link_id=<local id>` so an ambiguous timeout is recoverable by
          Task 6's list-and-match on the next pull (spec §6.1.3) -- the
          row and its mutation are simply left in place
          (`to_server_sent`, mutation retained) for that recovery, not
          retried by this replay loop itself. A reminder that ALREADY
          carries a link (a `watchlist_run`, say) keeps it: overwriting
          it destroyed the original link both server-side and, via the
          next pull, locally too (final review L12). The cost is that
          such a row is not list-and-matchable by the transfer marker, so
          an ambiguous timeout on it waits for a user retry/cancel
          instead of self-healing -- losing the user's own link is the
          worse of the two.

        Returns ``{"local_id": ..., "adopted_server_id": ...}`` (no
        ``server_id``/``delete_local``/``mutation_id`` keys): unlike
        create/update/delete, this action does its own DB write
        immediately (`convert_row_to_server_mirror` +
        `delete_pending_mutation`, both self-contained transactions)
        rather than deferring to `_sync_reminders`'s batched apply.
        ``adopted_server_id`` exists only so that batched apply can count
        this cycle's brand-new server id as "seen" by the deletion scan
        (final review C1) -- it deliberately is NOT ``server_id``, which
        that loop would re-apply as a mapping/update write.
        """
        disarmed = self.db.set_transfer_state(
            _REMINDER_PRIMITIVE,
            local_id,
            "to_server_sent",
            expected=("to_server_pending",),
        )
        if not disarmed:
            logger.info(
                f"Reminder {local_id} transfer_to_server mutation skipped: "
                "not in to_server_pending (concurrent cancel, or still "
                "to_server_sent from a prior attempt awaiting recovery)"
            )
            return {"local_id": local_id}

        task_payload = dict(payload.get("task_payload") or {})
        if not task_payload.get("link_type") and not task_payload.get("link_id"):
            task_payload["link_type"] = "chatbook_transfer"
            task_payload["link_id"] = local_id

        assert self.server_client is not None
        try:
            response = await self.server_client.create_reminder(**task_payload)
        except ServerClientPolicyError:
            raise
        except ServerClientValidationError as exc:
            self._fail_transfer_mutation(
                _REMINDER_PRIMITIVE, local_id, owner_id, payload, [str(exc)]
            )
            return {"local_id": local_id}

        response = response if isinstance(response, dict) else {}
        result = self.db.convert_row_to_server_mirror(
            _REMINDER_PRIMITIVE, local_id, response, owner_id
        )
        self.db.delete_pending_mutation(mutation_id)
        outcome = {"local_id": local_id, "adopted_server_id": response.get("id")}
        if result == "vanished":
            server_id = response.get("id") or "unknown"
            logger.warning(
                f"Reminder {local_id} vanished locally before its "
                f"transfer_to_server push landed; server reminder "
                f"({server_id}) is not linked to any local reminder"
            )
            self._record_sync_error(
                f"Reminder {local_id} was removed locally while it was "
                f"being transferred to the server; the server copy "
                f"({server_id}) is still there and is not linked to any "
                "local reminder",
                owner_id,
            )
        return outcome

    async def _push_reminder_release(
        self,
        local_id: str,
        mutation_id: int,
        owner_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Replay a pending `release_from_server` reminder mutation (spec §6.2).

        ``local_id`` is the server-owned MIRROR reminder this release
        targets; ``payload["local_copy_id"]`` is the dormant local-owner
        copy `create_local_copy_from_mirror` already created, which is
        what actually arms once this call acks. No disarm-before-send
        step here -- same reasoning as `_push_definition_release`'s
        docstring: the copy was already dormant from creation, so there is
        no double-dispatch window to close.

        A release IS a delete, just triggered by a transfer rather than a
        tombstone -- reuses `_push_tombstone`'s own call
        (`delete_reminder`). `ServerClientNotFoundError` (the server task
        is already gone) is treated exactly as an ack (spec §6.2.3),
        caught here rather than left to the outer `_push_mutation` try/
        except, which would otherwise turn every reminder action's 404
        into a conflict.

        Unlike a definition release, there is no upsert-echo to mirror the
        outcome onto the mirror row here: the server task is now GONE (a
        delete, not an archive). The mirror row is torn down HERE, on the
        ack, together with its sync mapping. Task 5's original brief
        deferred that to "the next pull's full-set reconciliation", but
        `_detect_server_deletions_conn` only DELETES a row that has a
        local tombstone -- a released mirror has none, so what it actually
        did was record a "the server deleted this row" conflict, on the
        very cycle that performed the release, and then skip the row
        forever after because an unresolved conflict already existed
        (final review I4). Resolving that conflict "local" re-created the
        reminder server-side while the local copy was already armed --
        genuine double execution. Deleting the mirror on the ack is what
        makes the ADR's "the mirror is torn down" claim true.

        Returns ``{"local_id": ..., "released_server_id": ...}`` once the
        mirror is actually torn down (no ``server_id``/``delete_local``/
        ``mutation_id`` keys, same shape family as `_push_reminder_transfer`'s
        `adopted_server_id`): this action does its own DB writes
        immediately, so `_sync_reminders`'s batched apply has nothing left
        to do with it. ``released_server_id`` exists so `_sync_reminders`
        can filter this cycle's already-stale `pulled_items` (`_network_
        phase` pulls BEFORE it pushes) -- otherwise the pre-release pull
        payload re-inserts the mirror this call just deleted, with a new
        local id no tombstone can remove (root-causes.md #4 / UAT finding
        4). Exact twin of `adopted_server_id`'s seen-set guard 12 lines
        below `_sync_reminders`'s own apply call.
        """
        server_task_id = payload.get("server_task_id")
        local_copy_id = payload.get("local_copy_id")
        if not server_task_id:
            logger.warning(
                f"Pending release_from_server reminder mutation for local "
                f"{local_id} has no server_task_id; dropping"
            )
            self.db.delete_pending_mutation(mutation_id)
            return {"local_id": local_id}

        assert self.server_client is not None
        try:
            await self.server_client.delete_reminder(server_task_id)
        except ServerClientNotFoundError:
            logger.info(
                f"Reminder {server_task_id} release: already gone "
                "server-side; treating as acked"
            )
        except ServerClientPolicyError:
            raise
        except ServerClientValidationError as exc:
            # A definitively rejected release settles PER MUTATION, the
            # same containment `_reject_definition_mutation` gives the
            # definitions leg: left to propagate it aborted the entire
            # reminder push phase, every cycle, forever (final review
            # L15). The dormant copy is deliberately left
            # `from_server_pending` -- nothing was released, so nothing
            # may arm; `cancel_transfer` is its recovery (Task 5
            # adjudication: cancel is state-keyed, not mutation-keyed).
            logger.warning(
                f"Reminder {server_task_id} release rejected by the server: {exc}"
            )
            self._record_sync_error(
                f"Moving reminder {server_task_id} to this device was "
                f"refused by the server: {exc}",
                owner_id,
            )
            self.db.delete_pending_mutation(mutation_id)
            return {"local_id": local_id}

        if local_copy_id:
            self.db.clear_transfer_state(
                _REMINDER_PRIMITIVE, local_copy_id, expected=("from_server_pending",)
            )
        # The mirror is now a row pointing at a server task that no longer
        # exists -- tear it down with its mapping (see the docstring: the
        # pull's reconciliation does NOT delete it, it conflicts on it).
        self.db.delete_reminder_task(local_id)
        self.db.delete_sync_mapping(local_id, _REMINDER_PRIMITIVE, owner_id)
        self.db.delete_pending_mutation(mutation_id)
        return {"local_id": local_id, "released_server_id": server_task_id}

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
