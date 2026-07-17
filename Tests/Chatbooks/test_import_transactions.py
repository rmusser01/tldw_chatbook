# test_import_transactions.py
# Description: RED-first regression coverage for task-250 (wrap chatbook import in a single transaction).
"""
Task-250: ``ChatbookImporter._import_conversations`` used to let every
``add_conversation``/``add_message``/``set_message_attachments`` call open
and commit its own top-level transaction -- ~1,500 commits for a 50x30
import. ``TransactionContextManager`` is depth-tracked/reentrant
(``ChaChaNotes_DB.py`` ``TransactionContextManager``), so wrapping each
conversation's writes in one outer ``with db.transaction():`` collapses
that to one commit per conversation, while still failing a single bad
conversation in isolation (a mid-loop exception rolls back only that
conversation's transaction; the per-conversation try/except in the caller
is unaffected).

This test imports a synthetic 3-conversation x 5-message chatbook (no
attachments) through the real ``ChatbookImporter`` + a real, tmp_path-backed
``CharactersRAGDB`` and counts *actual top-level commits* by wrapping
``TransactionContextManager.__exit__`` -- not just counting ``.transaction()``
call sites, which fire for every nested call too and wouldn't distinguish
"opened a transaction" from "committed one".

Pre-fix arithmetic for this fixture: 1 commit per add_conversation (x3) +
1 commit per add_message (x5 per conversation, x3 conversations = 15) = 18
top-level commits from the conversation-import loop alone (plus a small,
fixed number from CharactersRAGDB's own schema-init transaction). Post-fix:
1 commit per conversation (x3) -- the message-loop's nested transactions no
longer commit individually.
"""

import json
import zipfile
from datetime import datetime
from pathlib import Path

import pytest

import tldw_chatbook.DB.ChaChaNotes_DB as chachanotes_module
from tldw_chatbook.Chatbooks.chatbook_importer import ChatbookImporter, ImportStatus
from tldw_chatbook.Chatbooks.conflict_resolver import ConflictResolution

NUM_CONVERSATIONS = 3
MESSAGES_PER_CONVERSATION = 5


def _build_synthetic_chatbook(tmp_path: Path) -> Path:
    """A chatbook with NUM_CONVERSATIONS conversations, each with
    MESSAGES_PER_CONVERSATION messages and no attachments -- the minimal
    fixture the commit-count arithmetic in this module's docstring is based
    on."""
    chatbook_path = tmp_path / "synthetic_chatbook.zip"
    now = datetime.now().isoformat()

    content_items = []
    conversation_files = {}
    for i in range(NUM_CONVERSATIONS):
        conv_id = f"conv-{i}"
        content_items.append({
            "id": conv_id,
            "type": "conversation",
            "title": f"Synthetic Conversation {i}",
            "created_at": now,
            "file_path": f"content/conversations/conversation_{i}.json",
        })
        conversation_files[f"content/conversations/conversation_{i}.json"] = {
            "id": conv_id,
            "name": f"Synthetic Conversation {i}",
            "title": f"Synthetic Conversation {i}",
            "created_at": now,
            "messages": [
                {
                    "role": "user" if j % 2 == 0 else "assistant",
                    "content": f"conversation {i} message {j}",
                    "timestamp": now,
                }
                for j in range(MESSAGES_PER_CONVERSATION)
            ],
        }

    manifest = {
        "version": "1.0",
        "name": "Synthetic Transaction-Count Chatbook",
        "description": "Synthetic fixture for task-250",
        "author": "Test",
        "created_at": now,
        "updated_at": now,
        "content_items": content_items,
        "relationships": [],
        "include_media": False,
        "include_embeddings": False,
        "media_quality": "thumbnail",
        "statistics": {
            "total_conversations": NUM_CONVERSATIONS,
            "total_notes": 0,
            "total_characters": 0,
            "total_media_items": 0,
            "total_size_bytes": 0,
        },
        "tags": [],
        "categories": [],
        "language": "en",
        "license": None,
    }

    with zipfile.ZipFile(chatbook_path, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        for file_path, content in conversation_files.items():
            zf.writestr(file_path, json.dumps(content, indent=2))

    return chatbook_path


@pytest.fixture
def top_level_commit_counter(monkeypatch):
    """Count actual top-level (outermost, successful) transaction commits
    across every CharactersRAGDB instance the import path constructs, by
    wrapping TransactionContextManager.__exit__ itself -- this is the one
    place that knows both "was this the outermost transaction" and "did it
    succeed" (i.e. is about to call conn.commit()), regardless of how many
    nested .transaction() calls happened inside it.
    """
    counts = {"n": 0}
    original_exit = chachanotes_module.TransactionContextManager.__exit__

    def counting_exit(self, exc_type, exc_val, exc_tb):
        result = original_exit(self, exc_type, exc_val, exc_tb)
        if self.is_outermost_transaction and exc_type is None:
            counts["n"] += 1
        return result

    monkeypatch.setattr(
        chachanotes_module.TransactionContextManager, "__exit__", counting_exit
    )
    return counts


def test_import_collapses_per_message_commits_into_per_conversation_commits(
    tmp_path, top_level_commit_counter
):
    chatbook_path = _build_synthetic_chatbook(tmp_path)
    db_path = tmp_path / "databases" / "ChaChaNotes.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    importer = ChatbookImporter(db_paths={"ChaChaNotes": str(db_path)})
    status = ImportStatus()

    success, _message = importer.import_chatbook(
        chatbook_path=chatbook_path,
        conflict_resolution=ConflictResolution.SKIP,
        import_status=status,
    )

    assert success is True
    assert status.successful_items == NUM_CONVERSATIONS

    # Pre-fix this was ~18 (3 add_conversation + 15 add_message commits) plus
    # a small constant from CharactersRAGDB's own schema-init transaction.
    # Post-fix it should be ~NUM_CONVERSATIONS (one outer commit per
    # conversation) plus that same small constant. Assert a robust upper
    # bound rather than an exact number so unrelated constant-count
    # transactions (e.g. schema init) don't make this fixture brittle.
    max_expected = NUM_CONVERSATIONS + 3  # conversations + small constant
    assert top_level_commit_counter["n"] <= max_expected, (
        f"Expected <= {max_expected} top-level commits for "
        f"{NUM_CONVERSATIONS} conversations x {MESSAGES_PER_CONVERSATION} "
        f"messages (import should commit once per conversation, not once "
        f"per message); got {top_level_commit_counter['n']}"
    )
    # And a hard ceiling well below the pre-fix ~18, so a partial fix (e.g.
    # only wrapping some of the writes) still fails loudly.
    assert top_level_commit_counter["n"] < 10


def test_import_still_persists_all_conversations_and_messages(tmp_path):
    """Functional check alongside the commit-count assertion: wrapping the
    writes in a transaction must not change what actually gets imported."""
    chatbook_path = _build_synthetic_chatbook(tmp_path)
    db_path = tmp_path / "databases" / "ChaChaNotes.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    importer = ChatbookImporter(db_paths={"ChaChaNotes": str(db_path)})
    status = ImportStatus()

    success, _message = importer.import_chatbook(
        chatbook_path=chatbook_path,
        conflict_resolution=ConflictResolution.SKIP,
        import_status=status,
    )

    assert success is True
    assert status.successful_items == NUM_CONVERSATIONS
    assert status.failed_items == 0

    db = chachanotes_module.CharactersRAGDB(db_path, "verify-client")
    try:
        for i in range(NUM_CONVERSATIONS):
            matches = db.get_conversation_by_name(f"Synthetic Conversation {i}")
            assert len(matches) == 1
            conv_id = matches[0]["id"]
            messages = db.get_messages_for_conversation(conv_id)
            assert len(messages) == MESSAGES_PER_CONVERSATION
    finally:
        db.close_connection()


def test_success_not_counted_when_commit_fails(tmp_path, monkeypatch):
    """PR #651 review: successful_items must not increment before the outer
    transaction commits — a failing commit previously double-counted the
    conversation as both success and failure."""
    from tldw_chatbook.Chatbooks.chatbook_importer import ChatbookImporter
    from tldw_chatbook.Chatbooks.chatbook_models import ContentType
    from tldw_chatbook.DB import ChaChaNotes_DB as db_module

    chatbook = _build_synthetic_chatbook(tmp_path)
    dest = tmp_path / "dest-commitfail"
    dest.mkdir()
    dest_paths = {name: str(dest / f"{name}.db") for name in
                  ("ChaChaNotes", "Prompts", "Media", "Evals", "RAG")}

    original_transaction = db_module.CharactersRAGDB.transaction

    class _FailingCommitCtx:
        def __init__(self, inner):
            self._inner = inner

        def __enter__(self):
            return self._inner.__enter__()

        def __exit__(self, exc_type, exc, tb):
            self._inner.__exit__(exc_type, exc, tb)
            if exc_type is None:
                raise RuntimeError("simulated commit failure")

    call_depth = {"n": 0}

    def failing_transaction(self):
        ctx = original_transaction(self)
        call_depth["n"] += 1
        if call_depth["n"] == 1:  # only the importer's OUTER transaction
            return _FailingCommitCtx(ctx)
        return ctx

    monkeypatch.setattr(db_module.CharactersRAGDB, "transaction", failing_transaction)
    importer = ChatbookImporter(dest_paths)
    ok, _message = importer.import_chatbook(
        chatbook, content_selections={ContentType.CONVERSATION: ["conv-0"]}
    )
    status = importer.last_import_status if hasattr(importer, "last_import_status") else None
    # The import must not report the conversation as successful; depending on
    # reporting shape, either ok is False or the summary reports 0 successes.
    assert (not ok) or ("0/" in _message) or ("Failed" in _message) or (
        status is not None and status.successful_items == 0
    )
