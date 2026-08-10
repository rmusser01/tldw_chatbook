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
TASK_2511 = (
    REPO_ROOT
    / "backlog"
    / "tasks"
    / "task-2511 - Smoke-test-FastMCP-local-tool-binding-with-the-mcp-extra.md"
)
DESIGN_DOCUMENT = DOCUMENTS[0]
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
INVENTORY_CONTRACT = {
    "Built-in tools": BUILTIN_TOOLS,
    "Resource templates": RESOURCE_TEMPLATES,
    "Prompts": PROMPTS,
    "Library tools excluded from standalone": PRIVATE_LIBRARY_TOOLS,
}
UNEXPECTED_INVENTORY_ITEMS = {
    "Built-in tools": "unexpected_tool",
    "Resource templates": "unexpected://{id}",
    "Prompts": "unexpected_prompt",
    "Library tools excluded from standalone": "library_unexpected",
}
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


def _standalone_inventory_block(text: str) -> str:
    headings = list(re.finditer(r"^### Standalone inventory\s*$", text, re.MULTILINE))
    assert len(headings) == 1
    start = headings[0].end()
    following_heading = re.search(r"^#{2,3} ", text[start:], re.MULTILINE)
    end = start + following_heading.start() if following_heading else len(text)
    return text[start:end].strip()


def _assert_inventory_contract(path: Path, text: str) -> None:
    block = _standalone_inventory_block(text)
    category_pattern = re.compile(
        r"^- \*\*(?P<label>[^\n]+?) \((?P<count>\d+)\):\*\*\s*"
        r"(?P<body>.*?)(?=^- \*\*|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    categories = list(category_pattern.finditer(block))
    assert len(categories) == len(INVENTORY_CONTRACT), path
    assert len(re.findall(r"^- \*\*", block, re.MULTILINE)) == len(
        INVENTORY_CONTRACT
    ), path

    documented = {match.group("label"): match for match in categories}
    assert set(documented) == set(INVENTORY_CONTRACT), path
    for label, expected in INVENTORY_CONTRACT.items():
        match = documented[label]
        items = re.findall(r"`([^`\n]+)`", match.group("body"))
        assert int(match.group("count")) == len(expected), (path, label)
        assert len(items) == len(expected), (path, label)
        assert set(items) == set(expected), (path, label)


def _mutate_inventory(
    text: str, *, label: str, mutation: str, expected: tuple[str, ...]
) -> str:
    block = _standalone_inventory_block(text)
    token = f"`{expected[0]}`"
    assert block.count(token) == 1
    if mutation == "extra":
        replacement = f"{token}, `{UNEXPECTED_INVENTORY_ITEMS[label]}`"
    elif mutation == "missing":
        replacement = ""
    else:
        replacement = f"{token}, {token}"
    mutated_block = block.replace(token, replacement, 1)
    return text.replace(block, mutated_block, 1)


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
    _assert_inventory_contract(path, text)


@pytest.mark.parametrize("path", DOCUMENTS, ids=lambda path: path.name)
@pytest.mark.parametrize("mutation", ("extra", "missing", "duplicate"))
@pytest.mark.parametrize(
    ("label", "expected"),
    INVENTORY_CONTRACT.items(),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_exact_inventory_contract_rejects_mutations(
    path: Path,
    mutation: str,
    label: str,
    expected: tuple[str, ...],
) -> None:
    text = path.read_text(encoding="utf-8")
    _assert_inventory_contract(path, text)
    mutated = _mutate_inventory(text, label=label, mutation=mutation, expected=expected)
    with pytest.raises(AssertionError):
        _assert_inventory_contract(path, mutated)


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


def test_design_does_not_advertise_unimplemented_standalone_controls() -> None:
    text = DESIGN_DOCUMENT.read_text(encoding="utf-8")
    stale_claims = (
        'transport = "stdio"',
        "http_port = ",
        "allowed_clients = ",
        "require_auth = ",
        "rate_limit = 100",
        "max_concurrent_requests = 10",
        "Client allowlisting via",
    )
    assert all(claim not in text for claim in stale_claims)
    normalized = " ".join(text.split())
    assert "standalone gateway is stdio-only" in normalized
    assert "600 requests per minute" in normalized
    assert "16 in-flight requests" in normalized
    assert "`[mcp] expose_local_tools = false`" in text


def test_task_2511_records_truthful_supersession_instead_of_fastmcp_completion():
    text = TASK_2511.read_text(encoding="utf-8")
    assert "status: Done" in text
    assert "## Implementation Plan" in text
    assert "- [ ]" not in text
    assert (
        "- [x] #1 TASK-2512 independently smoke-tests the installed wheel and "
        "sdist with the `mcp` extra"
    ) in text
    assert (
        "- [x] #2 The original FastMCP smoke was not performed because its "
        "runtime boundary is obsolete"
    ) in text
    assert "Original FastMCP acceptance criterion: superseded, not completed." in text
