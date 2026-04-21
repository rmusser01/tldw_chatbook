"""
Tests for chat conversation request/response schemas.
"""

from datetime import datetime, timezone

import pytest

from tldw_chatbook.tldw_api.chat_conversation_schemas import (
    ALLOWED_CONVERSATION_STATES,
    ConversationListItem,
    ConversationListPagination,
    ConversationListResponse,
    ConversationMetadata,
    ConversationScopeParams,
    ConversationTreeNode,
    ConversationTreePagination,
    ConversationTreeResponse,
    ConversationUpdateRequest,
)


class TestChatConversationSchemas:
    """Validate conversation schema defaults, validation, and nested responses."""

    def test_scope_params_requires_workspace_id_for_workspace_scope(self):
        with pytest.raises(ValueError, match="workspace_id is required"):
            ConversationScopeParams(scope_type="workspace")

    def test_scope_params_clears_workspace_id_for_global_scope(self):
        params = ConversationScopeParams(scope_type="global", workspace_id="ws-1")

        assert params.scope_type == "global"
        assert params.workspace_id is None

    def test_update_request_normalizes_state_and_keywords(self):
        request = ConversationUpdateRequest(
            version=3,
            state=" Resolved ",
            keywords=[" Alpha ", "beta", "alpha", "", None, "Beta"],
        )

        assert request.state == "resolved"
        assert request.keywords == ["Alpha", "beta"]

    def test_update_request_rejects_invalid_state(self):
        with pytest.raises(ValueError, match="Allowed: in-progress, resolved, backlog, non-viable"):
            ConversationUpdateRequest(version=1, state="unknown")

    def test_update_request_preserves_runtime_and_discovery_fields_in_model_dump(self):
        request = ConversationUpdateRequest(
            version=9,
            runtime_backend=" Server ",
            discovery_owner=" CCP_CHARACTER ",
            discovery_entity_id=" char.local.alice ",
        )

        dumped = request.model_dump(exclude_none=True, exclude_unset=True)
        assert dumped["runtime_backend"] == "server"
        assert dumped["discovery_owner"] == "ccp_character"
        assert dumped["discovery_entity_id"] == "char.local.alice"

    def test_list_and_tree_models_parse_nested_payloads(self):
        timestamp = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

        list_payload = {
            "id": "conv-1",
            "scope_type": "workspace",
            "workspace_id": "ws-1",
            "character_id": 12,
            "assistant_kind": "persona",
            "assistant_id": "helper",
            "persona_memory_mode": "read_only",
            "title": "Thread",
            "state": "resolved",
            "topic_label": "triage",
            "topic_label_source": "manual",
            "topic_last_tagged_at": timestamp.isoformat(),
            "topic_last_tagged_message_id": "msg-9",
            "bm25_norm": 0.75,
            "last_modified": timestamp.isoformat(),
            "created_at": timestamp.isoformat(),
            "message_count": 4,
            "keywords": ["alpha", "beta"],
            "cluster_id": "cluster-1",
            "source": "api",
            "external_ref": "ref-1",
            "version": 7,
        }
        list_response = ConversationListResponse.model_validate(
            {
                "items": [list_payload],
                "pagination": {"limit": 10, "offset": 20, "total": 1, "has_more": False},
            }
        )

        metadata_payload = {
            "id": "conv-1",
            "scope_type": "workspace",
            "workspace_id": "ws-1",
            "character_id": 12,
            "assistant_kind": "persona",
            "assistant_id": "helper",
            "persona_memory_mode": "read_only",
            "title": "Thread",
            "state": "resolved",
            "topic_label": "triage",
            "topic_label_source": "manual",
            "topic_last_tagged_at": timestamp.isoformat(),
            "topic_last_tagged_message_id": "msg-9",
            "created_at": timestamp.isoformat(),
            "cluster_id": "cluster-1",
            "source": "api",
            "external_ref": "ref-1",
            "version": 7,
            "last_modified": timestamp.isoformat(),
        }
        tree_response = ConversationTreeResponse.model_validate(
            {
                "conversation": metadata_payload,
                "root_threads": [
                    {
                        "id": "msg-1",
                        "role": "user",
                        "content": "Hello",
                        "created_at": timestamp.isoformat(),
                        "children": [
                            {
                                "id": "msg-2",
                                "role": "assistant",
                                "content": "Hi",
                                "created_at": timestamp.isoformat(),
                                "children": [],
                                "truncated": False,
                            }
                        ],
                        "truncated": True,
                    }
                ],
                "pagination": {"limit": 10, "offset": 20, "total_root_threads": 1, "has_more": False},
                "depth_cap": 4,
            }
        )

        assert list_response.items[0].topic_label_source == "manual"
        assert list_response.items[0].topic_last_tagged_message_id == "msg-9"
        assert list_response.items[0].topic_last_tagged_at == timestamp
        assert list_response.pagination.total == 1
        assert tree_response.conversation.assistant_id == "helper"
        assert tree_response.conversation.topic_label_source == "manual"
        assert tree_response.conversation.topic_last_tagged_message_id == "msg-9"
        assert tree_response.conversation.created_at == timestamp
        assert tree_response.root_threads[0].children[0].role == "assistant"
        assert tree_response.depth_cap == 4

    def test_allowed_states_cover_server_contract(self):
        assert ALLOWED_CONVERSATION_STATES == (
            "in-progress",
            "resolved",
            "backlog",
            "non-viable",
        )
