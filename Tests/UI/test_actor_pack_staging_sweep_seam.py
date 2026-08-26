# Tests/UI/test_actor_pack_staging_sweep_seam.py
"""task-22216: the Actor Pack staging sweep moved out of ``TldwCli.__init__``.

PR #1998 ran ``ActorPackImportService.__init__``'s ``sweep_staging()`` on the
construction path — a ``secure_private_directory`` privacy walk plus a scandir
over staging candidates, every boot, before the event loop exists (the same
class task-21106 removed for recovery). These pins cover the new app seam;
the construct-time zero-I/O proof itself lives in
``Tests/App/test_boot_construct_fs_side_effects.py`` (subprocess boot with
counters at the importer seam) because the factory harness here builds without
a ChaChaNotes DB and therefore never constructs the import service at all.

- ``ensure_actor_pack_staging_sweep`` skips cleanly without an import service
  (the factory harness) and delegates to the service's once-gate otherwise;
- a sweep failure is absorbed and logged, never raised — the app stays up and
  the service gate stays open for a first-use retry;
- the deferred-startup worker actually kicks the sweep on a real mounted app.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.Actor_Packs.importer import ActorPackImportError


def test_ensure_staging_sweep_skips_without_an_import_service():
    """No import service (the factory harness has no ChaChaNotes DB) — no-op."""
    app = _build_test_app()
    assert app.actor_pack_import_service is None

    app.ensure_actor_pack_staging_sweep()  # must not raise


def test_ensure_staging_sweep_delegates_to_the_service_once_gate():
    """The app method delegates every call; once-semantics are the service's."""
    app = _build_test_app()
    calls: list[str] = []
    app.actor_pack_import_service = SimpleNamespace(
        ensure_staging_swept=lambda: calls.append("sweep")
    )

    app.ensure_actor_pack_staging_sweep()
    app.ensure_actor_pack_staging_sweep()

    assert calls == ["sweep", "sweep"]


def test_ensure_staging_sweep_absorbs_sweep_failure():
    """A failing sweep is logged, not raised — the app must stay up.

    The pre-move behavior (sweep inside the service constructor, reached
    from ``TldwCli.__init__``) aborted app construction on failure; the
    deferred seam deliberately softens that to the task-21106 shape. The
    service's gate does not latch on failure, so the first import use
    retries and surfaces the categorized error to the user instead.
    """
    app = _build_test_app()

    def failing_sweep() -> None:
        raise ActorPackImportError("actor_pack_import_cleanup_denied")

    app.actor_pack_import_service = SimpleNamespace(
        ensure_staging_swept=failing_sweep
    )

    app.ensure_actor_pack_staging_sweep()  # must not raise


@pytest.mark.asyncio
async def test_deferred_startup_kicks_the_staging_sweep_worker():
    """A real mounted app runs the sweep via the deferred-startup worker.

    Mirrors the task-21106 recovery-worker pin: swap the app-level ensure
    for a recorder and mount for real — this is the only automatic runner,
    so it must be proven on the genuine app, not a harness double.
    """
    app = _build_test_app()
    ran = threading.Event()
    app.ensure_actor_pack_staging_sweep = ran.set  # instance attr shadows
    async with app.run_test() as pilot:
        for _ in range(80):
            if ran.is_set():
                break
            await pilot.pause(0.05)
    assert ran.is_set(), (
        "deferred startup never invoked ensure_actor_pack_staging_sweep — "
        "the task-22216 background kick is unwired"
    )
