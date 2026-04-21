from tldw_chatbook.RAG_Admin.rag_admin_normalizers import (
    normalize_collection_record,
    normalize_template_record,
)


def test_normalize_local_template_maps_is_system_to_is_builtin():
    record = normalize_template_record(
        backend="local",
        payload={
            "id": 7,
            "name": "general",
            "description": "General",
            "template_json": '{"chunking": {"method": "words", "config": {"max_size": 400}}}',
            "is_system": 1,
            "created_at": "2026-04-20T00:00:00Z",
            "updated_at": "2026-04-20T00:00:00Z",
        },
    )

    assert record["record_id"] == "local:chunking_template:general"
    assert record["backend"] == "local"
    assert record["is_builtin"] is True
    assert record["tags"] == []
    assert record["template"]["chunking"]["method"] == "words"


def test_normalize_server_collection_uses_stats_and_metadata_defaults():
    record = normalize_collection_record(
        backend="server",
        payload={
            "name": "demo",
            "count": 3,
            "embedding_dimension": 1536,
            "metadata": {"provider": "openai"},
        },
    )

    assert record["record_id"] == "server:embedding_collection:demo"
    assert record["count"] == 3
    assert record["embedding_dimension"] == 1536
    assert record["metadata"]["provider"] == "openai"
