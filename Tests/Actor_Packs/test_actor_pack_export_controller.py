"""App-owned Actor Pack export operation and cancellation boundaries."""

from __future__ import annotations

import asyncio
import hashlib
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from tldw_chatbook.Actor_Packs.controller import (
    ActorPackExportController,
    ActorPackExportControllerError,
)
from tldw_chatbook.Actor_Packs.export import ActorPackExportSnapshot
from tldw_chatbook.Actor_Packs.publication import capture_actor_pack_destination

from .conftest import PNG_1X1, PORTABLE_UUID, canonical_json


def _snapshot(*, revision: int = 1) -> ActorPackExportSnapshot:
    return ActorPackExportSnapshot(
        actor_kind="character",
        actor_revision=revision,
        portable_uuid=PORTABLE_UUID,
        identity_version=1,
        portrait_name="portrait.png",
        portrait_sha256=hashlib.sha256(PNG_1X1).hexdigest(),
        local_actor_id="private-character-id",
        actor_payload=canonical_json(
            {
                "schema": "tldw.actor/v1",
                "actor_kind": "character",
                "portable_uuid": PORTABLE_UUID,
                "data": {"name": "Controlled"},
            }
        ),
        portrait_bytes=PNG_1X1,
    )


class _ExportService:
    def __init__(
        self,
        snapshots: tuple[ActorPackExportSnapshot, ...] | None = None,
    ) -> None:
        self.snapshots = snapshots or (_snapshot(),)
        self.calls = 0
        self.thread_ids: list[int] = []

    def capture_snapshot(
        self,
        actor_kind: str,
        actor_id: str,
        *,
        source: str,
        phase_hook=None,
    ) -> ActorPackExportSnapshot:
        assert (actor_kind, actor_id, source) == (
            "character",
            "private-character-id",
            "local",
        )
        self.thread_ids.append(threading.get_ident())
        index = min(self.calls, len(self.snapshots) - 1)
        self.calls += 1
        if phase_hook is not None:
            phase_hook("visuals_loaded")
        return self.snapshots[index]


@pytest.mark.asyncio
async def test_controller_owns_off_loop_capture_revalidation_and_publication(
    tmp_path: Path,
) -> None:
    service = _ExportService()
    phase_threads: list[int] = []
    controller = ActorPackExportController(
        service, phase_hook=lambda _phase: phase_threads.append(threading.get_ident())
    )
    request = controller.create_request(
        actor_kind="character",
        local_actor_id="private-character-id",
        source="local",
        destination=capture_actor_pack_destination(
            tmp_path / "controlled.tldw-actor-pack"
        ),
    )
    event_loop_thread = threading.get_ident()

    operation = controller.start_export(request)
    outcome = await controller.wait(operation)

    assert outcome.operation_id == operation
    assert outcome.error_category is None
    assert outcome.result is not None and outcome.result.committed is True
    assert service.calls == 2
    assert service.thread_ids and set(service.thread_ids) != {event_loop_thread}
    assert phase_threads and event_loop_thread not in phase_threads
    assert (tmp_path / "controlled.tldw-actor-pack").is_file()


@pytest.mark.asyncio
async def test_duplicate_submit_is_refused_until_owned_operation_settles(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    def barrier(phase: str) -> None:
        if phase == "visuals_loaded":
            entered.set()
            release.wait()

    controller = ActorPackExportController(_ExportService(), phase_hook=barrier)
    request = controller.create_request(
        actor_kind="character",
        local_actor_id="private-character-id",
        source="local",
        destination=capture_actor_pack_destination(tmp_path / "one.tldw-actor-pack"),
    )
    operation = controller.start_export(request)
    assert await asyncio.to_thread(entered.wait, 2)

    with pytest.raises(ActorPackExportControllerError, match="actor_pack_export_busy"):
        controller.start_export(request)

    release.set()
    assert (await controller.wait(operation)).result is not None


@pytest.mark.asyncio
async def test_final_snapshot_reread_rejects_actor_revision_aba(
    tmp_path: Path,
) -> None:
    service = _ExportService((_snapshot(), replace(_snapshot(), actor_revision=2)))
    controller = ActorPackExportController(service)
    destination = tmp_path / "stale.tldw-actor-pack"
    request = controller.create_request(
        actor_kind="character",
        local_actor_id="private-character-id",
        source="local",
        destination=capture_actor_pack_destination(destination),
    )

    outcome = await controller.wait(controller.start_export(request))

    assert outcome.result is None
    assert outcome.error_category == "actor_pack_export_authority_changed"
    assert not destination.exists()


@pytest.mark.asyncio
async def test_cancel_after_archive_fsync_cleans_only_owned_temp(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    def barrier(phase: str) -> None:
        if phase == "archive_fsynced":
            entered.set()
            release.wait()

    controller = ActorPackExportController(_ExportService(), phase_hook=barrier)
    destination = tmp_path / "cancelled.tldw-actor-pack"
    request = controller.create_request(
        actor_kind="character",
        local_actor_id="private-character-id",
        source="local",
        destination=capture_actor_pack_destination(destination),
    )
    operation = controller.start_export(request)
    assert await asyncio.to_thread(entered.wait, 2)

    assert controller.cancel(operation) is True
    release.set()
    outcome = await controller.wait(operation)

    assert outcome.error_category == "actor_pack_export_cancelled"
    assert not destination.exists()
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.asyncio
async def test_repeated_waiter_cancellation_drains_before_releasing_slot(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    def barrier(phase: str) -> None:
        if phase == "visuals_loaded":
            entered.set()
            release.wait()

    controller = ActorPackExportController(_ExportService(), phase_hook=barrier)
    request = controller.create_request(
        actor_kind="character",
        local_actor_id="private-character-id",
        source="local",
        destination=capture_actor_pack_destination(tmp_path / "drain.tldw-actor-pack"),
    )
    operation = controller.start_export(request)
    waiter = asyncio.create_task(controller.wait(operation))
    assert await asyncio.to_thread(entered.wait, 2)

    waiter.cancel()
    waiter.cancel()
    await asyncio.sleep(0)
    with pytest.raises(ActorPackExportControllerError, match="actor_pack_export_busy"):
        controller.start_export(request)
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await waiter
    replacement = controller.create_request(
        actor_kind="character",
        local_actor_id="private-character-id",
        source="local",
        destination=capture_actor_pack_destination(
            tmp_path / "replacement.tldw-actor-pack"
        ),
    )
    assert (await controller.wait(controller.start_export(replacement))).result


@pytest.mark.asyncio
async def test_profile_invalidation_cancels_exact_active_generation(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    def barrier(phase: str) -> None:
        if phase == "visuals_loaded":
            entered.set()
            release.wait()

    controller = ActorPackExportController(_ExportService(), phase_hook=barrier)
    request = controller.create_request(
        actor_kind="character",
        local_actor_id="private-character-id",
        source="local",
        destination=capture_actor_pack_destination(
            tmp_path / "profile.tldw-actor-pack"
        ),
    )
    operation = controller.start_export(request)
    assert await asyncio.to_thread(entered.wait, 2)

    controller.invalidate_profile()
    release.set()
    outcome = await controller.wait(operation)

    assert outcome.error_category in {
        "actor_pack_export_cancelled",
        "actor_pack_export_authority_changed",
    }
    assert not (tmp_path / "profile.tldw-actor-pack").exists()


@pytest.mark.asyncio
async def test_shutdown_closes_admission_and_drains_before_return(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    def barrier(phase: str) -> None:
        if phase == "visuals_loaded":
            entered.set()
            release.wait()

    controller = ActorPackExportController(_ExportService(), phase_hook=barrier)
    request = controller.create_request(
        actor_kind="character",
        local_actor_id="private-character-id",
        source="local",
        destination=capture_actor_pack_destination(
            tmp_path / "shutdown.tldw-actor-pack"
        ),
    )
    controller.start_export(request)
    assert await asyncio.to_thread(entered.wait, 2)

    shutdown = asyncio.create_task(controller.shutdown())
    await asyncio.sleep(0)
    assert shutdown.done() is False
    with pytest.raises(
        ActorPackExportControllerError, match="actor_pack_export_shutdown"
    ):
        controller.start_export(request)
    release.set()
    await shutdown


def test_request_and_outcome_repr_hide_actor_and_destination(tmp_path: Path) -> None:
    controller = ActorPackExportController(_ExportService())
    request = controller.create_request(
        actor_kind="character",
        local_actor_id="private-character-id",
        source="local",
        destination=capture_actor_pack_destination(
            tmp_path / "private-name.tldw-actor-pack"
        ),
    )

    assert "private-character-id" not in repr(request)
    assert str(tmp_path) not in repr(request)
