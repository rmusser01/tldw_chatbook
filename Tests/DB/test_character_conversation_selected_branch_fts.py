from pathlib import Path

import pytest

from tldw_chatbook.Character_Chat.character_conversation_navigation import (
    CharacterConversationNavigationService,
)
from tldw_chatbook.Chat.thinking_blocks import (
    DisplayableThinkingBlock,
    ThinkingEnvelope,
    dump_thinking_blocks_json,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.character_conversation_search import (
    SelectedBranchEligibilityProjector,
)


def _conversation(db: CharactersRAGDB, conversation_id: str = "conversation-1") -> str:
    authority = db.get_local_authority_id()
    created = db.add_conversation(
        {
            "id": conversation_id,
            "character_id": 1,
            "assistant_kind": "character",
            "assistant_id": "1",
            "assistant_authority_id": authority,
            "title": "Selected branch title",
        }
    )
    assert created == conversation_id
    return conversation_id


def _message(
    db: CharactersRAGDB,
    message_id: str,
    conversation_id: str,
    parent_id: str | None,
    role: str,
    content: str,
) -> None:
    assert db.add_message(
        {
            "id": message_id,
            "conversation_id": conversation_id,
            "parent_message_id": parent_id,
            "sender": role,
            "role": role,
            "content": content,
        }
    ) == message_id


def _thinking_canary() -> str:
    value = dump_thinking_blocks_json(
        ThinkingEnvelope(
            blocks=(
                DisplayableThinkingBlock(
                    block_id="round-0",
                    round_ordinal=0,
                    provider="local",
                    model="test",
                    protocol="openai_chat",
                    source_format="start_anchored_think",
                    status="complete",
                    text="THINKING_CANARY",
                ),
            )
        )
    )
    assert value is not None
    return value


def test_projector_emits_only_selected_visible_user_and_assistant_path(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "eligible.sqlite", client_id="projector")
    conversation_id = _conversation(db)
    _message(db, "m1", conversation_id, None, "user", "SELECTED_USER_CANARY")
    _message(db, "m2", conversation_id, "m1", "assistant", "SELECTED_ASSISTANT_CANARY")
    _message(db, "m3", conversation_id, "m2", "system", "SYSTEM_CANARY")
    _message(db, "m4", conversation_id, "m3", "tool", "TOOL_CANARY")
    _message(db, "m5", conversation_id, "m4", "user", "FINAL_USER_CANARY")
    _message(db, "m6", conversation_id, "m5", "assistant", "FINAL_ASSISTANT_CANARY")
    _message(db, "branch", conversation_id, "m1", "assistant", "NON_SELECTED_CANARY")
    _message(db, "deleted", conversation_id, "m1", "assistant", "DELETED_CANARY")
    assert db.soft_delete_message("deleted", expected_version=1)
    assert db.update_message_with_attachments(
        "m2",
        {"thinking_blocks_json": _thinking_canary()},
        expected_version=1,
        attachments=(
            {
                "position": 1,
                "data": b"ATTACHMENT_CANARY",
                "mime_type": "text/plain",
                "display_name": "ATTACHMENT_CANARY",
            },
        ),
        preserve_descendants=True,
    )
    db.set_conversation_active_leaf(conversation_id, "m6")

    document = SelectedBranchEligibilityProjector(db).project(conversation_id)

    assert document is not None
    assert document.title == "Selected branch title"
    assert "SELECTED_USER_CANARY" in document.body
    assert "SELECTED_ASSISTANT_CANARY" in document.body
    assert "FINAL_USER_CANARY" in document.body
    assert "FINAL_ASSISTANT_CANARY" in document.body
    for excluded in (
        "SYSTEM_CANARY",
        "TOOL_CANARY",
        "THINKING_CANARY",
        "ATTACHMENT_CANARY",
        "DELETED_CANARY",
        "NON_SELECTED_CANARY",
    ):
        assert excluded not in document.body

    service = CharacterConversationNavigationService(db)
    assert service.ensure_keyword_index().value == "ready"
    assert [row.title for row in service.keyword_search("SELECTED_USER_CANARY").rows] == [
        "Selected branch title"
    ]
    for excluded in (
        "SYSTEM_CANARY",
        "TOOL_CANARY",
        "THINKING_CANARY",
        "ATTACHMENT_CANARY",
        "DELETED_CANARY",
        "NON_SELECTED_CANARY",
    ):
        assert service.keyword_search(excluded).rows == ()
    assert db.get_connection().execute(
        "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'SYSTEM_CANARY'"
    ).fetchone()[0] == 1


@pytest.mark.parametrize("defect", ["cycle", "dangling", "cross_conversation"])
def test_projector_fails_closed_for_cycle_dangling_parent_and_cross_conversation_parent(
    tmp_path: Path,
    defect: str,
) -> None:
    db = CharactersRAGDB(tmp_path / f"invalid-{defect}.sqlite", client_id="projector")
    conversation_id = _conversation(db)
    _message(db, "m1", conversation_id, None, "user", "root")
    _message(db, "m2", conversation_id, "m1", "assistant", "leaf")
    leaf = "m2"
    if defect == "cycle":
        assert db.update_message(
            "m1",
            {"parent_message_id": "m2"},
            expected_version=1,
            preserve_descendants=True,
        )
    elif defect == "dangling":
        connection = db.get_connection()
        connection.execute("PRAGMA foreign_keys = OFF")
        with db.transaction() as cursor:
            authorization = db._semantic_mutation_authorization_for_coordinator(
                connection
            )
            with authorization._authorize(
                message_id="m1", operations={"message_update"}
            ):
                cursor.execute(
                    "UPDATE messages SET parent_message_id = 'missing' WHERE id = 'm1'"
                )
        connection.execute("PRAGMA foreign_keys = ON")
    else:
        _conversation(db, "other-conversation")
        _message(db, "other-root", "other-conversation", None, "user", "other")
        assert db.update_message(
            "m1",
            {"parent_message_id": "other-root"},
            expected_version=1,
            preserve_descendants=True,
        )
    db.set_conversation_active_leaf(conversation_id, leaf)

    assert SelectedBranchEligibilityProjector(db).project(conversation_id) is None


def test_projector_uses_unique_leaf_only_for_legacy_linear_conversation(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "legacy.sqlite", client_id="projector")
    conversation_id = _conversation(db)
    _message(db, "m1", conversation_id, None, "user", "root")
    _message(db, "m2", conversation_id, "m1", "assistant", "unique leaf")
    projector = SelectedBranchEligibilityProjector(db)

    assert projector.project(conversation_id) is not None

    _message(db, "m3", conversation_id, "m1", "assistant", "ambiguous leaf")
    assert projector.project(conversation_id) is None


def test_projector_digest_changes_when_selected_eligible_content_changes(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "digest.sqlite", client_id="projector")
    conversation_id = _conversation(db)
    _message(db, "m1", conversation_id, None, "user", "before")
    db.set_conversation_active_leaf(conversation_id, "m1")
    projector = SelectedBranchEligibilityProjector(db)

    before = projector.project(conversation_id)
    assert db.update_message(
        "m1", {"content": "after"}, expected_version=1, preserve_descendants=True
    )
    after = projector.project(conversation_id)

    assert before is not None and after is not None
    assert before.body == "before"
    assert after.body == "after"
    assert before.eligibility_digest != after.eligibility_digest


def test_projector_fails_closed_when_variant_group_has_multiple_live_selections(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "variant-selection.sqlite", client_id="projector")
    conversation_id = _conversation(db)
    _message(db, "root", conversation_id, None, "user", "root")
    _message(db, "base", conversation_id, "root", "assistant", "base")
    _message(db, "variant", conversation_id, "root", "assistant", "variant")
    with db.transaction() as connection:
        connection.execute(
            "UPDATE messages SET variant_of = 'base', variant_number = 2, "
            "is_selected_variant = 1, total_variants = 2 WHERE id = 'variant'"
        )
        connection.execute(
            "UPDATE messages SET variant_number = 1, is_selected_variant = 1, "
            "total_variants = 2 WHERE id = 'base'"
        )
    db.set_conversation_active_leaf(conversation_id, "base")

    assert SelectedBranchEligibilityProjector(db).project(conversation_id) is None


def test_projector_excludes_conversation_with_deleted_character_card(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "deleted-card.sqlite", client_id="projector")
    conversation_id = _conversation(db)
    _message(db, "root", conversation_id, None, "user", "visible")
    db.set_conversation_active_leaf(conversation_id, "root")
    card = db.get_character_card_by_id(1)
    assert card is not None
    assert db.soft_delete_character_card(1, expected_version=int(card["version"]))

    assert SelectedBranchEligibilityProjector(db).project(conversation_id) is None
