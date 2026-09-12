from tldw_chatbook.RAG_Admin.rag_admin_normalizers import (
    normalize_collection_record,
    normalize_template_record,
)


def test_normalize_local_template_uses_v7_columns():
    """task-8 (AC 26): local records are v7 rows — ``is_builtin`` straight
    from the column (no ``is_system`` fallback), ``uuid``/``version``
    sourced from the DB row, never fabricated."""
    record = normalize_template_record(
        backend="local",
        payload={
            "id": 7,
            "uuid": "0d2f4a26-5d8f-4a0b-9df3-7e1a83c1f001",
            "name": "general",
            "description": "General",
            "template_json": '{"chunking": {"method": "words", "config": {"max_size": 400}}}',
            "is_builtin": 1,
            "version": 4,
            "tags": ["custom"],
            "created_at": "2026-04-20T00:00:00Z",
            "updated_at": "2026-04-20T00:00:00Z",
        },
    )

    assert record["record_id"] == "local:chunking_template:general"
    assert record["backend"] == "local"
    assert record["is_builtin"] is True
    assert record["tags"] == ["custom"]
    assert record["template"]["chunking"]["method"] == "words"
    assert record["uuid"] == "0d2f4a26-5d8f-4a0b-9df3-7e1a83c1f001"
    assert record["version"] == 4


def test_normalize_template_does_not_fabricate_version():
    """A payload without a version normalizes to None — the v7 column is
    NOT NULL, so a live row always carries one; a fabricated ``1`` would
    mask a caller that stopped forwarding the DB value (AC 26)."""
    record = normalize_template_record(
        backend="server",
        payload={
            "name": "no-version",
            "description": "d",
            "template_json": '{"chunking": {"method": "words"}}',
        },
    )

    assert record["version"] is None
    assert record["uuid"] is None
    assert record["is_builtin"] is False


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
