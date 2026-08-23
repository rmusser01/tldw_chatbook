from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from pathlib import Path

import pytest

from tldw_chatbook.Actor_Packs.import_controller import (
    ActorPackImportController,
    ActorPackImportControllerError,
)


@dataclass(frozen=True)
class _Review:
    allowed_actions: tuple[str, ...] = ("create_new",)


@dataclass(frozen=True)
class _Result:
    actor_kind: str = "character"
    local_actor_id: str = "42"
    portable_uuid: str = "123e4567-e89b-42d3-a456-426614174000"


class _Importer:
    def __init__(self) -> None:
        self.cleaned: list[object] = []

    def inspect_archive(self, _path: Path, *, cancel_requested):
        if cancel_requested():
            raise ValueError("actor_pack_import_cancelled")
        return _Review()

    def cleanup_review(self, review: object) -> bool:
        self.cleaned.append(review)
        return True


class _Activation:
    def activate(self, review: object, action: str, *, cancel_requested):
        assert review == _Review()
        assert action == "create_new"
        if cancel_requested():
            raise ValueError("actor_pack_import_cancelled")
        return _Result()


@pytest.mark.asyncio
async def test_inspect_then_activate_one_leased_review(tmp_path: Path) -> None:
    importer = _Importer()
    refreshed: list[str] = []
    controller = ActorPackImportController(
        importer,
        _Activation(),
        refresh_callbacks=(lambda _result: refreshed.append("library"),),
    )
    request = controller.create_request(
        (tmp_path / "incoming.tldw-actor-pack").resolve()
    )

    inspected = await controller.wait(controller.start_inspection(request))
    assert inspected.review == _Review()
    activated = await controller.wait(
        controller.start_activation(inspected.review, "create_new")
    )

    assert activated.result == _Result()
    assert activated.error_category is None
    assert refreshed == ["library"]


@pytest.mark.asyncio
async def test_refresh_failures_are_isolated_after_commit(tmp_path: Path) -> None:
    called: list[str] = []

    def broken(_result: object) -> None:
        called.append("broken")
        raise RuntimeError

    controller = ActorPackImportController(
        _Importer(),
        _Activation(),
        refresh_callbacks=(broken, lambda _result: called.append("later")),
    )
    inspected = await controller.wait(
        controller.start_inspection(
            controller.create_request((tmp_path / "incoming.tldw-actor-pack").resolve())
        )
    )

    activated = await controller.wait(
        controller.start_activation(inspected.review, "create_new")
    )

    assert activated.result == _Result()
    assert activated.refresh_errors == ("actor_pack_import_refresh_failed",)
    assert called == ["broken", "later"]


@pytest.mark.asyncio
async def test_profile_invalidation_cancels_and_cleans_review(tmp_path: Path) -> None:
    importer = _Importer()
    controller = ActorPackImportController(importer, _Activation())
    inspected = await controller.wait(
        controller.start_inspection(
            controller.create_request((tmp_path / "incoming.tldw-actor-pack").resolve())
        )
    )

    controller.invalidate_profile()
    await asyncio.sleep(0)

    assert importer.cleaned == [inspected.review]
    with pytest.raises(ActorPackImportControllerError) as raised:
        controller.start_activation(inspected.review, "create_new")
    assert raised.value.category == "actor_pack_import_operation_unknown"


@pytest.mark.asyncio
async def test_one_active_operation_and_shutdown_drain(tmp_path: Path) -> None:
    entered = threading.Event()
    released = threading.Event()

    class _BlockingImporter(_Importer):
        def inspect_archive(self, path: Path, *, cancel_requested):
            entered.set()
            released.wait(2)
            return super().inspect_archive(path, cancel_requested=cancel_requested)

    controller = ActorPackImportController(_BlockingImporter(), _Activation())
    operation = controller.start_inspection(
        controller.create_request((tmp_path / "incoming.tldw-actor-pack").resolve())
    )
    await asyncio.to_thread(entered.wait, 2)
    with pytest.raises(ActorPackImportControllerError) as raised:
        controller.start_inspection(
            controller.create_request((tmp_path / "other.tldw-actor-pack").resolve())
        )
    assert raised.value.category == "actor_pack_import_busy"

    shutdown = asyncio.create_task(controller.shutdown())
    await asyncio.sleep(0)
    assert not shutdown.done()
    released.set()
    await shutdown
    outcome = controller.last_outcome(operation)
    assert outcome is not None
    assert outcome.error_category == "actor_pack_import_cancelled"


def test_request_repr_does_not_expose_archive_path(tmp_path: Path) -> None:
    controller = ActorPackImportController(_Importer(), _Activation())
    archive = (tmp_path / "private-name.tldw-actor-pack").resolve()

    request = controller.create_request(archive)

    assert str(archive) not in repr(request)
    assert "private-name" not in repr(request)
