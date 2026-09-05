"""Property test: the transfer machine never arms a row both locally and
on the server at the same time (spec §10).

Drives randomized interleavings of {begin, push-attempt-start, ack,
definitive-fail, cancel, release, release-ack} against a REAL tmp-dir
``ScheduledTasksDB`` + the real ``SchedulingService``/``SyncEngine`` --
never a reimplementation of the state machine. A stateful fake server
client stands in for the network: its ``created``/``deleted`` sets model
whether the server counterpart is live, the same way
``test_automation_sync_end_to_end.py``'s own stateful fake models a round
trip across two ``sync_now()`` calls.

Two of the requested steps collapse into one rule each here, because the
fake's calls are synchronous (no real network latency to pause a step
at): "push-attempt-start" and its resolution ("ack" or "definitive-fail")
happen inside the SAME ``sync_now()`` call (disarm-before-send is
unconditional and already verified by Task 4/5's own red-checks), and
likewise "release"/"release-ack" resolve inside one ``sync_now()`` call
once a release mutation is queued. ``begin_transfer_to_server``/
``begin_transfer_to_local``/``cancel_transfer`` remain their own discrete
rules, and ``recover_inflight_transfers`` is included as an extra rule
(startup recovery must never double-arm a row either).

Scoped to ``reminder_task``: it exercises 100% of the SHARED CAS/DB
transfer machinery the invariant is actually about (``set_transfer_
state``/``clear_transfer_state``/``convert_row_to_server_mirror``/
``create_local_copy_from_mirror`` -- identical code paths for both
primitives). The definition-specific network shape (preview-then-create,
plus the `recurring_question` health gate) is already covered by Task
4/5/6's own directed unit tests; adding it here would mean stubbing
``compute_local_health`` and a preview response without changing what
this property actually checks.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from tldw_chatbook.Scheduling.db.scheduled_tasks_db import (
    DORMANT_TRANSFER_STATES,
    ScheduledTasksDB,
)
from tldw_chatbook.Scheduling.services.scheduling_service import SchedulingService
from tldw_chatbook.Scheduling.services.server_client import (
    ServerClientNotFoundError,
    ServerClientValidationError,
    ServerUnavailableError,
)


class _FakeApp:
    """Only ``active_server_id`` is read by the transfer facade."""

    active_server_id = "1"


class _FakeReminderServerClient:
    """Stateful fake modeling the server side of ONE reminder's transfer.

    ``created``/``deleted`` are the server-side "live" set the invariant
    checks against. ``next_outcome`` ("success"/"fail"/"error") is set by
    the state machine before each network-touching rule and governs both
    the create (transfer) and delete (release) calls uniformly.
    """

    def __init__(self) -> None:
        self.notifications_service = object()
        self.created: dict[str, dict] = {}
        self.deleted: set[str] = set()
        self._seq = 0
        self.next_outcome = "success"

    def _new_id(self) -> str:
        self._seq += 1
        return f"srv-{self._seq}"

    async def create_reminder(self, **payload):
        if self.next_outcome == "fail":
            raise ServerClientValidationError("rejected")
        if self.next_outcome == "error":
            raise ServerUnavailableError("offline")
        if self.next_outcome == "ambiguous":
            # The real "ambiguous timeout" spec §6.1.3 exists for: the
            # create actually LANDS server-side, but the response never
            # reaches the client. `recover`'s list-and-match is the only
            # thing that resolves this.
            server_id = self._new_id()
            self.created[server_id] = {"id": server_id, **payload}
            raise ServerUnavailableError("timeout after create")
        server_id = self._new_id()
        self.created[server_id] = {"id": server_id, **payload}
        return dict(self.created[server_id])

    async def delete_reminder(self, server_id):
        if self.next_outcome == "error":
            raise ServerUnavailableError("offline")
        if server_id not in self.created or server_id in self.deleted:
            raise ServerClientNotFoundError("gone")
        if self.next_outcome == "fail":
            raise ServerClientValidationError("rejected")
        self.deleted.add(server_id)
        return {}

    async def list_reminders(self):
        return {
            "items": [
                dict(item)
                for sid, item in self.created.items()
                if sid not in self.deleted
            ]
        }

    # Automation-side surface: never exercised (reminder-only scope
    # above), but SyncEngine's pull phases call these unconditionally
    # every sync_now() cycle regardless of whether any mutations are
    # queued for them -- stub them so that always happens cleanly instead
    # of relying on _run_phase's broad exception-swallowing.
    async def list_automation_definitions(self, *, limit: int = 50, offset: int = 0):
        return {"items": [], "total": 0, "has_more": False}

    async def list_automation_results(self, *, limit: int = 50, offset: int = 0):
        return {"items": [], "total": 0, "has_more": False}

    def is_live_for_local_id(self, local_id) -> bool:
        """Whether a live (created, undeleted) server item traces back to
        ``local_id`` via ``link_id`` -- independent of whether the LOCAL
        row's own ``server_id`` column got linked correctly, so a
        recovery/conversion bug that forgets to reassign ownership is
        still detectable (the fake's own created/archived sets are the
        model of the server side, per the brief -- not the local DB's
        say-so about them)."""
        return any(
            sid not in self.deleted
            for sid, item in self.created.items()
            if item.get("link_id") == local_id
        )


class TransferInvariantMachine(RuleBasedStateMachine):
    """Drives one reminder through begin/push/cancel/release/recover
    interleavings, asserting "never both armed" after every step."""

    def __init__(self) -> None:
        super().__init__()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db = ScheduledTasksDB(Path(self._tmpdir.name) / "transfer.db")
        self.server = _FakeReminderServerClient()
        self.service = SchedulingService(
            db=self.db,
            server_client=self.server,
            runtime_source="local",
            app_getter=lambda: _FakeApp(),
        )
        # task-3 (ruling 4): `transfer_refusal` now also gates on a real
        # `refresh_server_reachability` probe (default `False`) -- this
        # fake models an always-connected server, so pre-seed the same
        # verdict a probe would reach, or every `begin_to_server`/
        # `begin_to_local` rule below silently refuses and the invariant
        # stops exercising the transfer machinery it exists to stress.
        self.service._server_reachable = True
        self.row_id = self.db.create_reminder_task(
            owner_id="local",
            title="t",
            schedule_kind="one_time",
            run_at="2030-01-01T00:00:00+00:00",
        )
        self.copy_id: str | None = None

    def teardown(self) -> None:
        self.db.close()
        self._tmpdir.cleanup()

    @staticmethod
    def _run(coro):
        return asyncio.run(coro)

    @rule()
    def begin_to_server(self):
        self._run(self.service.begin_transfer_to_server("reminder_task", self.row_id))

    @rule(outcome=st.sampled_from(["success", "fail", "error", "ambiguous"]))
    def sync_push(self, outcome):
        """Replays whatever transfer_to_server/release_from_server
        mutation is currently queued under the (single) connected
        server's owner scope -- covers push-attempt-start -> ack /
        definitive-fail, and release -> release-ack, in one call."""
        self.server.next_outcome = outcome
        self._run(self.service.sync_now("server:1"))

    @rule(use_copy=st.booleans())
    def cancel(self, use_copy):
        target = self.copy_id if (use_copy and self.copy_id) else self.row_id
        outcome = self._run(self.service.cancel_transfer("reminder_task", target))
        if outcome.status == "cancelled" and target == self.copy_id:
            self.copy_id = None

    @rule()
    def begin_to_local(self):
        outcome = self._run(
            self.service.begin_transfer_to_local("reminder_task", self.row_id)
        )
        if outcome.status == "pending" and outcome.row_id:
            self.copy_id = outcome.row_id

    @rule()
    def recover(self):
        self._run(self.service.recover_inflight_transfers())

    @invariant()
    def never_both_armed(self):
        # The server counterpart's identity is `self.row_id` in EITHER
        # direction: a to-server transfer creates it carrying
        # `link_id=self.row_id`, and a release deletes that same item --
        # `self.copy_id` (the dormant-then-armed local copy a release
        # spawns) never itself corresponds 1:1 to a server item, so the
        # cross-primitive check has to key server-liveness by the ONE
        # conceptual identity and compare it against EVERY local row that
        # could be armed for it (the original row, or its released copy).
        server_armed = self.server.is_live_for_local_id(self.row_id)
        armed_row_ids = []
        for row_id in {self.row_id, self.copy_id}:
            if row_id is None:
                continue
            row = self.db.get_reminder_task(row_id)
            if row is None:
                continue
            armed_locally = (
                not str(row.get("owner_id") or "").startswith("server:")
                and row.get("transfer_state") not in DORMANT_TRANSFER_STATES
            )
            if armed_locally:
                armed_row_ids.append(row_id)

        assert not (armed_row_ids and server_armed), (
            f"BOTH armed: locally-armed rows={armed_row_ids} while the "
            f"server counterpart (link_id={self.row_id!r}) is live -- "
            f"created={self.server.created!r} deleted={self.server.deleted!r}"
        )


TestTransferInvariant = TransferInvariantMachine.TestCase
