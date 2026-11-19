"""A send's service-owned turn authority, read off the UI pump (TASK-33620.15).

``turn_admission.precapture`` reads the MCP maximum, skill catalog, project
and review roots, scratch space, RAG depth and the other service-owned
values on a worker thread; the synchronous snapshot builder then uses them
only while the session inputs they were read for still match. These pin
that the snapshot is the one the builder would have captured inline, that
the worker never runs the reads on the calling thread, and that a session
change in between makes the builder read again.
"""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleProviderSelection,
    ConsoleWorkspaceContext,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
)
from tldw_chatbook.Chat.console_scratch_space import ConsoleScratchSpaceManager
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.UI.Console_Modules import turn_admission
from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController

pytestmark = pytest.mark.bootstrap_profile


class _Consent:
    """Counts review-root admissions and the threads that made them."""

    def __init__(self) -> None:
        self.threads: list[int] = []

    def admit_turn(self, workspace_id):
        self.threads.append(threading.get_ident())
        return SimpleNamespace(
            ready_roots=[str(Path("C:/workspace") / str(workspace_id))],
            ready_aliases=["folder-ready"],
            skipped_roots=[],
        )


def _controller(tmp_path):
    store = ConsoleChatStore()
    session = store.create_session(
        workspace_id="workspace-a",
        ephemeral=True,
        settings=ConsoleSessionSettings(
            provider="openai", model="model-a", system_prompt="system-a"
        ),
    )
    selection = ConsoleProviderSelection(
        provider="openai",
        explicit_model="model-a",
        system_prompt="system-a",
        workspace_context=ConsoleWorkspaceContext(active_workspace_id="workspace-a"),
    )
    consent = _Consent()
    controller = ConsoleSessionController.__new__(ConsoleSessionController)
    controller.app_instance = SimpleNamespace(change_review_consent_service=consent)
    controller._provider_readiness_app_config_fn = lambda: {
        "console": {"agent_runtime": "true"}
    }
    controller._build_provider_selection_fn = lambda _session_id: selection
    controller._current_chat_store_accessor = lambda: store
    controller._chat_store_accessor = lambda: store
    controller._rag_source_types_accessor = lambda: ["notes"]
    controller._rag_top_k_accessor = lambda: 6
    scratch = ConsoleScratchSpaceManager(temp_parent=tmp_path)
    controller._scratch_snapshot_provider = scratch.snapshot
    return controller, store, session, consent, scratch


@pytest.mark.asyncio
async def test_the_precaptured_snapshot_is_the_inline_one_read_on_a_worker(tmp_path):
    """Same snapshot as an inline build; the reads ran off the calling thread."""
    controller, _store, session, consent, scratch = _controller(tmp_path)
    try:
        authority = await turn_admission.precapture(controller, session.id)
        assert authority is not None
        assert consent.threads and consent.threads[0] != threading.get_ident()

        with turn_admission.applied(authority):
            precaptured = controller._build_console_turn_execution_context(session.id)
        assert len(consent.threads) == 1, "the builder read the authority again"

        inline = controller._build_console_turn_execution_context(session.id)
        assert len(consent.threads) == 2
        assert precaptured == inline
        assert precaptured.workspace_roots == (
            str(Path("C:/workspace") / "workspace-a"),
        )
        assert precaptured.rag_defaults == {"source_types": ("notes",), "top_k": 6}
    finally:
        assert scratch.dispose()


@pytest.mark.asyncio
async def test_a_session_change_after_the_precapture_is_read_inline(tmp_path):
    """Stale authority is never frozen into the turn: the builder re-reads it."""
    controller, store, session, consent, scratch = _controller(tmp_path)
    try:
        authority = await turn_admission.precapture(controller, session.id)
        assert authority is not None
        live = next(item for item in store.sessions() if item.id == session.id)
        live.project_instruction_state = replace(
            ProjectInstructionControlState.new_session(),
            working_folder_binding_id="binding-b",
        )

        with turn_admission.applied(authority):
            snapshot = controller._build_console_turn_execution_context(session.id)

        assert len(consent.threads) == 2
        assert consent.threads[1] == threading.get_ident()
        assert snapshot.project_authority.working_folder_binding_id == "binding-b"
    finally:
        assert scratch.dispose()


@pytest.mark.asyncio
async def test_a_failed_precapture_leaves_the_error_to_the_builder(tmp_path):
    """A read that raises on the worker is dropped, so admission raises it."""
    controller, _store, session, _consent, scratch = _controller(tmp_path)
    try:

        def unavailable(_session_id):
            raise RuntimeError("scratch space unavailable")

        controller._scratch_snapshot_provider = unavailable
        assert await turn_admission.precapture(controller, session.id) is None
        with pytest.raises(RuntimeError, match="scratch space unavailable"):
            controller._build_console_turn_execution_context(session.id)
    finally:
        assert scratch.dispose()
