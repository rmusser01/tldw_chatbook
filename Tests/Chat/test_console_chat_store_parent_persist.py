"""Tests that ``_persist_new_message`` writes a real ``parent_message_id``.

Task 4 of the Console conversation-branching Phase A plan: persistence must
carry the persisted id of the in-memory tree parent (Task 3's
``_native_parent_by_message`` / ``_nodes_by_session``) instead of the
previously hardcoded ``None``.
"""

from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole


class _RecordingPersistence:
    db = None

    def __init__(self):
        self.created = []

    def create_conversation(self, **kw):
        return "conv-1"

    def create_message(
        self,
        *,
        conversation_id,
        sender,
        content,
        image_data,
        image_mime_type,
        message_id=None,
        parent_message_id=None,
        feedback=None,
        attachments=None,
    ):
        pid = f"p{len(self.created) + 1}"
        self.created.append(
            {
                "id": pid,
                "parent_message_id": parent_message_id,
                "content": content,
                "sender": sender,
            }
        )
        return pid

    def update_message_content(self, **kw):
        return True


def test_linear_persist_sets_parent_chain():
    p = _RecordingPersistence()
    store = ConsoleChatStore(persistence=p)
    s = store.create_session(title="t")
    store.active_session_id = s.id
    store.append_message(s.id, role=ConsoleMessageRole.USER, content="hi", persist=True)
    store.append_message(s.id, role=ConsoleMessageRole.ASSISTANT, content="yo", persist=True)
    # First message (user echo) persists as the root: no parent yet.
    assert p.created[0]["parent_message_id"] is None
    # Second message (assistant) is parented at the first message's
    # persisted id -- proving the parent (user echo) persisted before the
    # child (assistant) in this linear flow.
    assert p.created[1]["parent_message_id"] == p.created[0]["id"]
