"""Application-owner recovery when packaged Canvas profiles are unavailable."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest


def _conversation_with_v2_history(database, snapshot):
    """Seed history through the shipping service before simulating package damage."""
    from tldw_chatbook.Canvas.models import CanvasScope
    from tldw_chatbook.Canvas.service import CanvasService

    conversation_id = database.add_conversation({"title": "Recovery history"})
    assert conversation_id is not None
    message_id = database.add_message(
        {
            "conversation_id": conversation_id,
            "parent_message_id": None,
            "sender": "assistant",
            "role": "assistant",
            "content": "Canvas recovery owner",
        }
    )
    assert message_id is not None
    scope = CanvasScope(
        session_id="recovery-session",
        conversation_id=conversation_id,
        active_message_ids=(message_id,),
        selected_canvas_id=None,
        selected_revision_id=None,
        run_id="seed-run",
    )
    service = CanvasService(database, profile_snapshot=snapshot)
    created = service.create_canvas(
        scope,
        title="Recovery diagram",
        source="<!doctype html><p>historical plain source</p>",
    )
    updated = service.update_canvas(
        scope,
        created.revision.canvas_id,
        expected_parent_revision_id=created.revision.revision_id,
        source=(
            '<!doctype html><pre data-canvas-diagram="mermaid">'
            "flowchart TD\nA[Stored] --> B[History]</pre>"
        ),
    )
    return scope, message_id, created, updated


@pytest.mark.parametrize(
    "package_damage",
    [
        "missing-v2-library",
        "tampered-v2-worker",
        "missing-catalog",
        "malformed-catalog",
    ],
)
def test_native_console_owner_keeps_v2_source_history_when_packaged_snapshot_fails(
    tmp_path, candidate_snapshot, damage_canvas_package, package_damage
) -> None:
    """Package damage must not stop chat or rewrite stored Canvas authority."""
    from tldw_chatbook.Canvas.service import CanvasServiceError
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    database = CharactersRAGDB(
        tmp_path / "unavailable-native.sqlite", "unavailable-native"
    )
    runtime = None
    try:
        scope, message_id, created, updated = _conversation_with_v2_history(
            database, candidate_snapshot
        )
        revision_count = (
            database.get_connection()
            .execute("SELECT COUNT(*) FROM canvas_revisions")
            .fetchone()[0]
        )

        damage_canvas_package(package_damage)
        runtime = ConsoleRuntime(SimpleNamespace(chachanotes_db=database))

        store = runtime.ensure_chat_store()
        controller = runtime.canvas_controller
        assert store is not None
        assert controller is not None
        service = controller.durable_service
        assert service is not None

        current_scope = replace(
            scope,
            run_id="recovery-current",
            selected_canvas_id=updated.revision.canvas_id,
            selected_revision_id=updated.revision.revision_id,
        )
        historical_scope = replace(
            scope,
            run_id="recovery-historical",
            selected_canvas_id=created.revision.canvas_id,
            selected_revision_id=created.revision.revision_id,
        )
        assert service.list_canvases(current_scope)[0].runtime_profile == (
            "canvas-v2-mermaid-1"
        )
        current = service.read_canvas(current_scope, updated.revision.canvas_id)
        assert current.source == (
            '<!doctype html><pre data-canvas-diagram="mermaid">'
            "flowchart TD\nA[Stored] --> B[History]</pre>"
        )
        historical = service.read_canvas(historical_scope, created.revision.canvas_id)
        assert historical.source == "<!doctype html><p>historical plain source</p>"

        with pytest.raises(CanvasServiceError) as create_error:
            service.create_canvas(
                replace(scope, run_id="refused-create"),
                title="Must remain unavailable",
                source="<!doctype html><p>new source</p>",
            )
        assert create_error.value.code == "document_incompatible"
        with pytest.raises(CanvasServiceError) as update_error:
            service.update_canvas(
                replace(scope, run_id="refused-update"),
                updated.revision.canvas_id,
                expected_parent_revision_id=updated.revision.revision_id,
                source="<!doctype html><p>replacement source</p>",
            )
        assert update_error.value.code == "document_incompatible"

        assert (
            database.get_connection()
            .execute("SELECT COUNT(*) FROM canvas_revisions")
            .fetchone()[0]
            == revision_count
        )
        assert message_id in current_scope.active_message_ids
    finally:
        if runtime is not None:
            asyncio.run(runtime.dispose())
        database.close_connection()
