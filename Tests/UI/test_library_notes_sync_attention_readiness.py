"""The sync-attention journey must tolerate an unfinished root-folder load."""

from __future__ import annotations

import asyncio

import pytest
from textual.widgets import Button

from Tests.UI import test_library_notes_sync_attention as journey
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]


async def test_attention_journey_waits_for_a_pending_root_folder_page(
    tmp_path, monkeypatch
):
    release = asyncio.Event()
    requested = asyncio.Event()
    real_page = NotesScopeService.page_note_folder_children
    real_until = journey._until

    async def held_root_page(self, **kwargs):
        if kwargs.get("parent_id") is None:
            requested.set()
            await release.wait()
        return await real_page(self, **kwargs)

    async def observe_pending_folder(pilot, predicate, what, **kwargs):
        if what == "the held folder's tree row to say it needs attention":
            try:
                await real_until(
                    pilot, requested.is_set, "the root folder request to start"
                )
                labels = [
                    str(button.label)
                    for button in pilot.app.screen.query(
                        ".library-notes-folder-row"
                    ).results(Button)
                ]
                assert not any("VSync" in label for label in labels), labels
                assert not predicate(), "an unpublished folder is not ready"
            finally:
                release.set()
        await real_until(pilot, predicate, what, **kwargs)

    monkeypatch.setattr(NotesScopeService, "page_note_folder_children", held_root_page)
    monkeypatch.setattr(journey, "_until", observe_pending_folder)
    try:
        await journey.test_a_held_sync_folder_reads_needs_attention_until_recovery_heals_it(
            tmp_path, monkeypatch
        )
    finally:
        release.set()
