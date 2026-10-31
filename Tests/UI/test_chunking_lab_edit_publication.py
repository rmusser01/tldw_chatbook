"""The edit consumer must publish after competing same-session status renders."""

import asyncio
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from types import SimpleNamespace

import pytest
from textual.widgets import Input, Static, TextArea

from Tests.private_profile import private_profile_test
from Tests.UI.test_chunking_lab_sample_import_drafts import (
    entered_read,
    mounted_lab,
    sample,
)
from Tests.UI.test_chunking_lab_screen import settle_lab


@contextmanager
def held_final_refresh(monkeypatch, screen, ordering=None):
    """Order real off-loop preparation without changing its publication gates."""
    from tldw_chatbook.UI.Screens import chunking_lab_screen as module

    gate = SimpleNamespace(
        entered=threading.Event(),
        release=threading.Event(),
        worker_entered=threading.Event(),
        worker_release=threading.Event(),
        worker_completed=asyncio.Event(),
    )
    owned = ContextVar("lab_owned_refresh", default=False)
    refresh, validate = screen._refresh_session, module._validation

    def validation(draft):
        if owned.get() and not gate.entered.is_set():
            gate.entered.set()
            assert gate.release.wait(10), "final preparation was not released"
        elif (
            not owned.get()
            and gate.entered.is_set()
            and (ordering == "pending-worker" or not gate.release.is_set())
        ):
            # Hold every later status preparation too; none may rescue the echo.
            gate.worker_entered.set()
            if ordering == "pending-worker":
                gate.release.set()
                assert gate.worker_release.wait(
                    10
                ), "status preparation was not released"
        return validate(draft)

    async def ordered_refresh(*, edit_complete=False):
        token = owned.set(edit_complete)
        try:
            await refresh(edit_complete=edit_complete)
            if not edit_complete and gate.worker_entered.is_set():
                gate.worker_completed.set()
                if ordering == "worker-first":
                    gate.release.set()
        finally:
            owned.reset(token)

    with monkeypatch.context() as patch:
        patch.setattr(module, "_validation", validation)
        patch.setattr(screen, "_refresh_session", ordered_refresh)
        try:
            yield gate
        finally:
            gate.release.set()
            gate.worker_release.set()


async def type_initial_sample(screen, pilot):
    editor = screen.query_one("#lab-sample-text", TextArea)
    editor.focus()
    await pilot.press("x")
    await pilot.pause()
    await screen.drain_edits()
    assert editor.text == sample(screen)["text"] == "x"
    assert screen.coordinator.session.undo[-1]["sample"]["text"] == ""
    return editor


@pytest.mark.asyncio
@pytest.mark.parametrize("ordering", ["pending-worker", "worker-first"])
@private_profile_test
async def test_undo_publishes_after_competing_status_preparation(
    request, monkeypatch, ordering
):
    from tldw_chatbook.Chunking import lab_state

    async with mounted_lab() as (host, screen, pilot):
        editor = await type_initial_sample(screen, pilot)
        await settle_lab(host, screen, pilot)
        with held_final_refresh(monkeypatch, screen, ordering) as gate:
            screen.queue_edit(lab_state.undo_edit)
            await entered_read(gate.entered)
            await entered_read(gate.worker_entered)
            await screen.drain_edits()
            # Canonical Undo must remain correct independently of its visible echo.
            assert sample(screen)["text"] == ""
            assert screen.coordinator.session.undo == ()
            assert editor.text == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery", ["held", "delivered"])
@private_profile_test
async def test_final_preparation_preserves_newer_keyboard_input(
    request, monkeypatch, delivery
):
    from tldw_chatbook.Chunking import lab_state

    async with mounted_lab() as (host, screen, pilot):
        editor = await type_initial_sample(screen, pilot)
        parent_entered, parent_release = asyncio.Event(), asyncio.Event()

        async def hold_parent():
            parent_entered.set()
            await asyncio.wait_for(parent_release.wait(), 10)

        with held_final_refresh(monkeypatch, screen) as gate:
            screen.queue_edit(lab_state.undo_edit)
            await entered_read(gate.entered)
            assert sample(screen)["text"] == ""
            await pilot.press("home")
            key = None
            try:
                if delivery == "held":
                    screen.call_next(hold_parent)
                    await asyncio.wait_for(parent_entered.wait(), 3)
                key = asyncio.create_task(pilot.press("y"))
                async with asyncio.timeout(3):
                    while editor.text != "yx":
                        await asyncio.sleep(0.01)
                if delivery == "held":
                    assert editor.sample_revision != screen._sample_input_revision
                else:
                    await key
                    assert screen._edits
                gate.release.set()
                await screen.drain_edits()
                assert editor.text == "yx"
                if delivery == "held":
                    assert sample(screen)["text"] == ""
                parent_release.set()
                await key
                await pilot.pause()
                await screen.drain_edits()
                assert editor.text == sample(screen)["text"] == "yx"
            finally:
                parent_release.set()
                gate.release.set()
                if key is not None:
                    await key


@pytest.mark.asyncio
@private_profile_test
async def test_status_refresh_stays_live_during_final_preparation(request, monkeypatch):
    from tldw_chatbook.Chunking import lab_state
    from tldw_chatbook.UI.Chunking_Lab_Modules.sample_region import SampleRegion

    async with mounted_lab() as (host, screen, pilot):
        await type_initial_sample(screen, pilot)
        entered, release = asyncio.Event(), asyncio.Event()

        async def guarded_operation():
            entered.set()
            await asyncio.wait_for(release.wait(), 10)

        with held_final_refresh(monkeypatch, screen) as gate:
            screen.queue_edit(lab_state.undo_edit)
            await entered_read(gate.entered)
            guarded = asyncio.create_task(screen.coordinator._guard(guarded_operation))
            try:
                await asyncio.wait_for(entered.wait(), 3)
                async with asyncio.timeout(3):
                    while not screen.query_one(SampleRegion).disabled:
                        await asyncio.sleep(0.01)
                assert not screen._edit_task.done()
                # Real autosave also completes while the final authoring echo waits.
                await screen.coordinator._writer.flush()
                async with asyncio.timeout(3):
                    while not str(
                        screen.query_one("#lab-status", Static).content
                    ).startswith("Saved locally"):
                        await asyncio.sleep(0.01)
                assert screen.coordinator.guarded
            finally:
                release.set()
                gate.release.set()
                await guarded
            await screen.drain_edits()
            await settle_lab(host, screen, pilot)
            assert not screen.query_one(SampleRegion).disabled


@pytest.mark.asyncio
@private_profile_test
async def test_final_sample_echo_preserves_queued_configuration_keys(
    request, monkeypatch
):
    from tldw_chatbook.Chunking import lab_state

    async with mounted_lab() as (host, screen, pilot):
        await type_initial_sample(screen, pilot)
        screen.query_one("#lab-show-configure").press()
        await pilot.pause()
        await screen.drain_edits()
        await settle_lab(host, screen, pilot)
        name = screen.query_one("#lab-name", Input)
        name.focus()
        await pilot.press("end")
        await pilot.pause()
        initial = name.value
        parent_entered, parent_release = asyncio.Event(), asyncio.Event()

        async def hold_parent():
            parent_entered.set()
            await asyncio.wait_for(parent_release.wait(), 10)

        with held_final_refresh(monkeypatch, screen) as gate:
            keys = []
            try:
                screen.queue_edit(lab_state.undo_edit)
                await entered_read(gate.entered)
                await asyncio.wait_for(gate.worker_completed.wait(), 5)
                screen.call_next(hold_parent)
                await asyncio.wait_for(parent_entered.wait(), 3)
                inputs = screen._input_generation
                keys.append(asyncio.create_task(pilot.press("y")))
                async with asyncio.timeout(3):
                    while name.value != initial + "y":
                        await asyncio.sleep(0.01)
                assert screen._input_generation == inputs
                assert not screen._edits  # The accepted key's parent delivery is held.
                gate.release.set()
                await screen.drain_edits()
                second_expected = name.value + "z"
                keys.append(asyncio.create_task(pilot.press("z")))
                async with asyncio.timeout(3):
                    while name.value != second_expected:
                        await asyncio.sleep(0.01)
                parent_release.set()
                await asyncio.gather(*keys)
                await pilot.pause()
                await screen.drain_edits()
                await settle_lab(host, screen, pilot)
                canonical = screen.coordinator.session.candidates[screen._b_id()][
                    "draft"
                ]["record_fields"]["name"]
                assert name.value == canonical == initial + "yz"
            finally:
                parent_release.set()
                gate.release.set()
                await asyncio.gather(*keys, return_exceptions=True)
