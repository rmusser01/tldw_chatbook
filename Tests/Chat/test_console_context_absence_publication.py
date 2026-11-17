"""Durable absence is independent of transient transcript revisions."""

import sys
import threading
from contextlib import contextmanager

import pytest

from Tests.Chat.test_console_first_send_atomicity import _controller
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_context_repository import (
    ConsoleContextRepository,
    ConsoleMemorySelectionRecord,
    MemorySelectionKind,
    read_context_presentation_inputs,
)
from tldw_chatbook.Chat.console_preparation_reads import run_preparation_read
from tldw_chatbook.Chat.console_context_policy import ConsoleContextPolicyOverrides
from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
    ConsoleContextReadSnapshot,
)

pytestmark = pytest.mark.bootstrap_profile


@contextmanager
def _reads():
    counts = {"probe": 0, "worker": 0}
    codes = {
        ConsoleContextRepository.has_active_memory_selection.__code__: "probe",
        run_preparation_read.__code__: "worker",
    }
    previous_thread, previous_main = threading.getprofile(), sys.getprofile()

    def observe(frame, event, _argument):
        if event == "call" and frame.f_code in codes:
            counts[codes[frame.f_code]] += 1

    threading.setprofile_all_threads(observe)
    try:
        yield counts
    finally:
        threading.setprofile_all_threads(previous_thread)
        sys.setprofile(previous_main)


@pytest.mark.asyncio
async def test_memory_absence_survives_payload_but_reconciles_external_and_local_writes(
    tmp_path,
):
    database, store, controller, _ = _controller(tmp_path)
    try:
        store.active_session_id = "session-1"
        message = store.append_message(
            "session-1", role=ConsoleMessageRole.USER, content="hello", persist=True
        )
        snapshot = ConsoleContextReadSnapshot()
        with _reads() as first:
            assert await snapshot.warm(controller, "session-1")
        assert first["probe"] == 1
        # Updating only local display/payload must not issue another native read.
        store.append_message(
            "session-1",
            role=ConsoleMessageRole.ASSISTANT,
            content="stream",
            persist=False,
        )
        with _reads() as streaming:
            assert await snapshot.warm(controller, "session-1")
        assert streaming == {"probe": 0, "worker": 0}
        repository = controller._context_repository
        selection = ConsoleMemorySelectionRecord(
            sequence=1,
            selection_id="reset",
            conversation_id=store.sessions()[0].persisted_conversation_id,
            activation_message_id=message.persisted_message_id,
            selected_memory_id=None,
            event_kind=MemorySelectionKind.RESET,
            suppresses_legacy=True,
            created_at="2026-11-13T00:00:00Z",
        )
        repository.insert_memory_selection(selection)
        with _reads() as local:
            assert await snapshot.warm(controller, "session-1")
        assert local["probe"] == 1
        snapshot.at -= 1.01
        with _reads() as external:
            assert await snapshot.warm(controller, "session-1")
        assert external["probe"] == 1
    finally:
        database.close()


@pytest.mark.asyncio
async def test_unsaved_raw_context_with_known_policy_needs_no_native_worker(tmp_path):
    database, _store, controller, _ = _controller(tmp_path)
    try:
        with _reads() as calls:
            result = await read_context_presentation_inputs(
                controller,
                "session-1",
                global_overrides=ConsoleContextPolicyOverrides(),
            )
        assert result[2].kind.value == "raw"
        assert calls == {"probe": 0, "worker": 0}
    finally:
        database.close()
