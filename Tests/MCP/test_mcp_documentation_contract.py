"""Public-documentation contract for the standalone MCP security boundary."""

from __future__ import annotations

from pathlib import Path
import re

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS = (
    REPO_ROOT / "Docs" / "Design" / "MCP.md",
    REPO_ROOT / "Docs" / "User_Guide" / "mcp.md",
    REPO_ROOT / "Docs" / "Development" / "release-recovery-setup.md",
)
BUILTIN_TOOLS = (
    "chat_with_llm",
    "chat_with_character",
    "search_rag",
    "search_conversations",
    "create_note",
    "search_notes",
    "list_characters",
    "get_conversation_history",
    "export_conversation",
    "ingest_media",
)
RESOURCE_TEMPLATES = (
    "conversation://{conversation_id}",
    "note://{note_id}",
    "character://{character_id}",
    "media://{media_id}",
    "rag-chunk://{chunk_uuid}",
)
PROMPTS = (
    "summarize_conversation",
    "generate_document",
    "analyze_media",
    "search_and_synthesize",
    "character_writing",
)
PRIVATE_LIBRARY_TOOLS = (
    "library_list_media",
    "library_get_media",
    "library_search_media",
    "library_list_notes",
    "library_get_note",
    "library_search_notes",
    "library_list_prompts",
    "library_get_prompt",
    "library_search_prompts",
    "library_list_skills",
    "library_get_skill",
    "library_search_skills",
    "library_list_conversations",
    "library_get_conversation",
    "library_search_conversations",
    "library_list_collections",
    "library_get_collection",
    "library_search_collections",
)
FORBIDDEN_LIVE_COMMANDS = (
    re.compile(r"\bmcp\s+(?:dev|install|run)\b", re.IGNORECASE),
    re.compile(r"pip\s+install\s+[\"']?mcp(?:\[cli\])?\b", re.IGNORECASE),
    re.compile(r"mcp\.server\.fastmcp", re.IGNORECASE),
    re.compile(r"\bFastMCP\b"),
)


@pytest.fixture(params=DOCUMENTS, ids=lambda path: path.name)
def document(request: pytest.FixtureRequest) -> tuple[Path, str]:
    path = request.param
    return path, path.read_text(encoding="utf-8")


def test_documents_publish_exact_install_and_launch_commands(
    document: tuple[Path, str],
) -> None:
    path, text = document
    assert 'pip install "tldw_chatbook[mcp]"' in text, path
    assert "python -m tldw_chatbook.MCP" in text, path
    for forbidden in FORBIDDEN_LIVE_COMMANDS:
        assert forbidden.search(text) is None, (path, forbidden.pattern)


def test_documents_pin_revision_and_batch_behavior(
    document: tuple[Path, str],
) -> None:
    path, text = document
    assert all(
        version in text for version in ("2025-03-26", "2025-11-25", "2026-07-28")
    ), path
    normalized = " ".join(text.split())
    assert (
        "Batch requests are accepted only with `2025-03-26`; "
        "`2025-11-25` and `2026-07-28` reject them."
    ) in normalized, path


def test_documents_list_the_exact_standalone_and_private_inventories(
    document: tuple[Path, str],
) -> None:
    path, text = document
    normalized = " ".join(text.split())
    assert "exactly 10 built-in tools" in normalized, path
    assert "exactly 5 resource templates" in normalized, path
    assert "exactly 5 prompts" in normalized, path
    assert all(f"`{name}`" in text for name in BUILTIN_TOOLS), path
    assert all(f"`{uri}`" in text for uri in RESOURCE_TEMPLATES), path
    assert all(f"`{name}`" in text for name in PROMPTS), path
    assert "All 18 are excluded from the standalone stdio catalog" in normalized, path
    assert all(f"`{name}`" in text for name in PRIVATE_LIBRARY_TOOLS), path


def test_documents_explain_continuation_and_local_tool_policy(
    document: tuple[Path, str],
) -> None:
    path, text = document
    normalized = " ".join(text.split())
    assert "256 KiB" in text, path
    assert '`_meta["tldw.chatbook/continuation"]`' in text, path
    assert '`_meta["tldw.chatbook/resource"]`' in text, path
    assert "`[mcp] expose_local_tools = false`" in text, path
    assert "`mcp_permissions.json`" in text, path
    assert "kill switch" in text.lower(), path
    assert "workspace confinement" in normalized.lower(), path
    assert "external `ask` state is refused" in normalized, path


def test_documents_warn_about_external_local_data_and_cloud_egress(
    document: tuple[Path, str],
) -> None:
    path, text = document
    normalized = " ".join(text.split())
    assert "> [!WARNING]" in text, path
    assert "user's OS access" in normalized, path
    assert "private local Library" in normalized, path
    assert "tools, resources, and prompts" in normalized, path
    assert "off-device" in normalized, path
    assert "cloud model" in normalized, path
