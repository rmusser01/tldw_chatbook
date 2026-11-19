"""Transcript memory reads retain lineage checks without discarded policy I/O."""

import pytest

from Tests.Chat.test_console_context_compaction import (
    _insert_real_prefix_memory,
    _real_selection_controller,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_context_compaction import (
    EffectiveMemoryKind,
    prefix_digest,
)

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]


@pytest.mark.parametrize("lineage", ["current", "unpersisted", "corrupt"])
def test_memory_projection_validates_real_lineage_without_reading_policy(
    tmp_path, monkeypatch, lineage
):
    db, repo, controller, store, session, conversation, snapshots = (
        _real_selection_controller(tmp_path, "memory-only")
    )
    try:
        _insert_real_prefix_memory(
            repo,
            conversation_id=conversation,
            memory_id="memory",
            selection_id="selection",
            boundary_message_id=snapshots[0].message_id,
            activation_message_id=snapshots[-1].message_id,
            summarized_prefix_digest=prefix_digest(snapshots[:1]),
            created_at="2026-01-01T00:00:00Z",
        )
        if lineage == "unpersisted":
            store.append_message(
                session.id, role=ConsoleMessageRole.USER, content="pending"
            )
        elif lineage == "corrupt":
            with db.transaction() as cursor:
                cursor.execute("DELETE FROM console_conversation_memory_scopes")
        original = controller._global_context_policy_overrides
        policy_reads = []

        def observed():
            policy_reads.append(True)
            return original()

        monkeypatch.setattr(controller, "_global_context_policy_overrides", observed)
        memory = controller.context_control_inputs(session.id, include_global=False)[2]
        assert policy_reads == []
        assert memory.kind is (
            EffectiveMemoryKind.GENERATED_PREFIX
            if lineage == "current"
            else EffectiveMemoryKind.RAW
        )
        assert controller.context_control_inputs(session.id)[2] == memory
        assert policy_reads == [True]
        with pytest.raises(KeyError, match="missing"):
            controller.context_control_inputs("missing", include_global=False)
    finally:
        db.close()


def test_memory_projection_does_not_hide_repository_errors(tmp_path, monkeypatch):
    db, repo, controller, _store, session, conversation, snapshots = (
        _real_selection_controller(tmp_path, "memory-failure")
    )
    error = RuntimeError("repository unavailable")

    def failed(*_args):
        raise error

    try:
        _insert_real_prefix_memory(
            repo,
            conversation_id=conversation,
            memory_id="memory",
            selection_id="selection",
            boundary_message_id=snapshots[0].message_id,
            activation_message_id=snapshots[-1].message_id,
            summarized_prefix_digest=prefix_digest(snapshots[:1]),
            created_at="2026-01-01T00:00:00Z",
        )
        monkeypatch.setattr(controller, "_select_session_effective_memory", failed)
        with pytest.raises(RuntimeError) as caught:
            controller.context_control_inputs(session.id, include_global=False)
        assert caught.value is error
    finally:
        db.close()
