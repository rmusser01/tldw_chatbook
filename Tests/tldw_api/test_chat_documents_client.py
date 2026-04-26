"""Tests for chat document generation endpoint wiring on the shared TLDW API client."""

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api.chat_documents_schemas import (
    AsyncGenerationResponse,
    BulkGenerateRequest,
    BulkGenerateResponse,
    DocumentListResponse,
    GenerateDocumentRequest,
    GenerateDocumentResponse,
    GenerationStatistics,
    JobStatusResponse,
    PromptConfigResponse,
    SavePromptConfigRequest,
)
from tldw_chatbook.tldw_api.client import TLDWAPIClient


@pytest.mark.asyncio
async def test_chat_document_generation_methods_route_current_server_contract(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "document_id": 11,
                "conversation_id": "conv-1",
                "document_type": "study_guide",
                "title": "Study Guide",
                "content": "Guide content",
                "provider": "openai",
                "model": "gpt-test",
                "generation_time_ms": 1234,
                "created_at": "2026-04-23T18:00:00Z",
            },
            {
                "job_id": "job-1",
                "status": "pending",
                "conversation_id": "conv-1",
                "document_type": "summary",
                "created_at": "2026-04-23T18:00:00Z",
                "message": "Document generation job created",
            },
            {
                "job_id": "job-1",
                "conversation_id": "conv-1",
                "document_type": "summary",
                "status": "completed",
                "provider": "openai",
                "model": "gpt-test",
                "result_content": "Summary",
                "created_at": "2026-04-23T18:00:00Z",
                "progress_percentage": 100,
            },
            {"message": "Job job-1 cancelled successfully"},
            {
                "documents": [
                    {
                        "id": 11,
                        "conversation_id": "conv-1",
                        "document_type": "study_guide",
                        "title": "Study Guide",
                        "content": "Guide content",
                        "provider": "openai",
                        "model": "gpt-test",
                        "generation_time_ms": 1234,
                        "created_at": "2026-04-23T18:00:00Z",
                    }
                ],
                "total": 1,
                "conversation_id": "conv-1",
                "document_type": "study_guide",
            },
            {
                "id": 11,
                "conversation_id": "conv-1",
                "document_type": "study_guide",
                "title": "Study Guide",
                "content": "Guide content",
                "provider": "openai",
                "model": "gpt-test",
                "generation_time_ms": 1234,
                "created_at": "2026-04-23T18:00:00Z",
            },
            {"message": "Document 11 deleted successfully"},
            {
                "document_type": "study_guide",
                "system_prompt": "System",
                "user_prompt": "User",
                "temperature": 0.5,
                "max_tokens": 2000,
                "is_custom": True,
                "created_at": "2026-04-23T18:00:00Z",
                "updated_at": "2026-04-23T18:00:00Z",
            },
            {
                "document_type": "study_guide",
                "system_prompt": "System",
                "user_prompt": "User",
                "temperature": 0.5,
                "max_tokens": 2000,
                "is_custom": True,
            },
            {
                "total_jobs": 2,
                "job_ids": ["job-1", "job-2"],
                "estimated_time_seconds": 20,
                "message": "Created 2 generation jobs",
            },
            {
                "total_documents": 1,
                "by_type": {"study_guide": 1},
                "by_provider": {"openai": 1},
                "average_generation_time_ms": 1234.0,
                "total_tokens_used": 500,
                "last_generated": "2026-04-23T18:00:00Z",
                "most_used_model": "gpt-test",
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    sync_response = await client.generate_chat_document(
        GenerateDocumentRequest(
            conversation_id="conv-1",
            document_type="study_guide",
            provider="openai",
            model="gpt-test",
        )
    )
    async_response = await client.generate_chat_document(
        GenerateDocumentRequest(
            conversation_id="conv-1",
            document_type="summary",
            provider="openai",
            model="gpt-test",
            async_generation=True,
        )
    )
    job_status = await client.get_chat_document_job_status("job-1")
    cancel_result = await client.cancel_chat_document_job("job-1")
    documents = await client.list_chat_generated_documents(
        conversation_id="conv-1",
        document_type="study_guide",
        limit=25,
    )
    document = await client.get_chat_generated_document(11)
    delete_result = await client.delete_chat_generated_document(11)
    saved_prompt = await client.save_chat_document_prompt_config(
        SavePromptConfigRequest(
            document_type="study_guide",
            system_prompt="System",
            user_prompt="User",
            temperature=0.5,
            max_tokens=2000,
        )
    )
    prompt_config = await client.get_chat_document_prompt_config("study_guide")
    bulk = await client.bulk_generate_chat_documents(
        BulkGenerateRequest(
            conversation_ids=["conv-1"],
            document_types=["study_guide", "summary"],
            provider="openai",
            model="gpt-test",
            api_key="ignored-by-client",
        )
    )
    stats = await client.get_chat_document_generation_statistics()

    assert isinstance(sync_response, GenerateDocumentResponse)
    assert isinstance(async_response, AsyncGenerationResponse)
    assert isinstance(job_status, JobStatusResponse)
    assert cancel_result == {"message": "Job job-1 cancelled successfully"}
    assert isinstance(documents, DocumentListResponse)
    assert document.id == 11
    assert delete_result == {"message": "Document 11 deleted successfully"}
    assert isinstance(saved_prompt, PromptConfigResponse)
    assert isinstance(prompt_config, PromptConfigResponse)
    assert isinstance(bulk, BulkGenerateResponse)
    assert isinstance(stats, GenerationStatistics)
    assert [call.args[:2] for call in mocked.await_args_list] == [
        ("POST", "/api/v1/chat/documents/generate"),
        ("POST", "/api/v1/chat/documents/generate"),
        ("GET", "/api/v1/chat/documents/jobs/job-1"),
        ("DELETE", "/api/v1/chat/documents/jobs/job-1"),
        ("GET", "/api/v1/chat/documents"),
        ("GET", "/api/v1/chat/documents/11"),
        ("DELETE", "/api/v1/chat/documents/11"),
        ("POST", "/api/v1/chat/documents/prompts"),
        ("GET", "/api/v1/chat/documents/prompts/study_guide"),
        ("POST", "/api/v1/chat/documents/bulk"),
        ("GET", "/api/v1/chat/documents/statistics"),
    ]
    assert mocked.await_args_list[0].kwargs["json_data"] == {
        "conversation_id": "conv-1",
        "document_type": "study_guide",
        "provider": "openai",
        "model": "gpt-test",
        "stream": False,
        "async_generation": False,
    }
    assert mocked.await_args_list[4].kwargs["params"] == {
        "conversation_id": "conv-1",
        "document_type": "study_guide",
        "limit": 25,
    }
    assert mocked.await_args_list[7].kwargs["json_data"]["temperature"] == 0.5
    assert mocked.await_args_list[9].kwargs["json_data"]["document_types"] == ["study_guide", "summary"]


@pytest.mark.asyncio
async def test_chat_document_stream_generation_is_rejected_before_json_request(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(return_value={})
    monkeypatch.setattr(client, "_request", mocked)

    with pytest.raises(ValueError, match="Streaming chat document generation"):
        await client.generate_chat_document(
            GenerateDocumentRequest(
                conversation_id="conv-1",
                document_type="study_guide",
                provider="openai",
                model="gpt-test",
                stream=True,
            )
        )

    assert mocked.await_count == 0
