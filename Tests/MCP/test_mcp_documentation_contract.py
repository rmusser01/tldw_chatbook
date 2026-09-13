"""Public-documentation contract for the standalone MCP security boundary."""

from __future__ import annotations

from pathlib import Path
import re

import pytest

from tldw_chatbook.Library.library_tool_contract import (
    LIBRARY_TOOL_DESCRIPTORS,
)


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
USER_GUIDE_DOCUMENT = DOCUMENTS[1]
CONSOLE_AGENT_TOOLS_DOCUMENT = (
    REPO_ROOT / "Docs" / "User_Guide" / "console" / "agent-runs-and-tools.md"
)
SKILLS_EXAMPLES_DOCUMENT = REPO_ROOT / "Docs" / "Examples" / "skills" / "README.md"
CONFIG_TEMPLATE = REPO_ROOT / "tldw_chatbook" / "config.py"
BUILTIN_TOOL_GATE = REPO_ROOT / "tldw_chatbook" / "Agents" / "builtin_tool_gate.py"
MCP_TOOLS_MODE = (
    REPO_ROOT / "tldw_chatbook" / "UI" / "MCP_Modules" / "mcp_tools_mode.py"
)
MCP_SERVERS_MODE = (
    REPO_ROOT / "tldw_chatbook" / "UI" / "MCP_Modules" / "mcp_servers_mode.py"
)
LOCAL_TOOL_PROVIDER = REPO_ROOT / "tldw_chatbook" / "Agents" / "local_tool_provider.py"
MCP_SERVER = REPO_ROOT / "tldw_chatbook" / "MCP" / "server.py"
MCP_WORKBENCH = REPO_ROOT / "tldw_chatbook" / "UI" / "MCP_Modules" / "mcp_workbench.py"
WATCHLISTS_TOOL_DOCUMENTS = (CONSOLE_AGENT_TOOLS_DOCUMENT, USER_GUIDE_DOCUMENT)
WATCHLISTS_REMEDIATION_DESIGN = (
    REPO_ROOT
    / "Docs"
    / "superpowers"
    / "specs"
    / "2026-08-26-console-driven-watchlists-workflow-uat-remediation-design.md"
)
LOCAL_TOOL_COPY_SURFACES = (
    CONFIG_TEMPLATE,
    BUILTIN_TOOL_GATE,
    MCP_TOOLS_MODE,
    MCP_SERVERS_MODE,
    CONSOLE_AGENT_TOOLS_DOCUMENT,
    USER_GUIDE_DOCUMENT,
    SKILLS_EXAMPLES_DOCUMENT,
)
LOCAL_LIBRARY_TOOLS_DOCUMENT = (
    REPO_ROOT / "Docs" / "Development" / "Agent-Tools" / "local-library-tools.md"
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
#: Derived, never transcribed. ``LIBRARY_TOOL_DESCRIPTORS`` is the contract the
#: local MCP manifest is itself built from -- ``MCP/server.py``'s
#: ``_describe_local_library_tools`` iterates it unfiltered, and its docstring
#: says outright that it is "never hand-maintained here". A second hand-kept
#: copy in this test is what let the surface and the documents drift apart in
#: four directions at once (TASK-21501): the code carried 24 tools while
#: `Docs/Design/MCP.md` said 18, `Docs/User_Guide/mcp.md` said 23, and this
#: tuple said 18. Deriving removes the drift class rather than resetting it.
#:
#: A tuple rather than a view: ``_mutate_inventory`` indexes element 0 to pick
#: the token it perturbs.
PRIVATE_LIBRARY_TOOLS = tuple(LIBRARY_TOOL_DESCRIPTORS)
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


def _admonition_block(text: str, kind: str) -> str:
    match = re.search(
        rf"^> \[!{re.escape(kind)}\]\s*$\n(?P<body>(?:^>.*(?:\n|$))+)",
        text,
        re.MULTILINE,
    )
    assert match is not None
    return " ".join(
        line.removeprefix("> ").strip() for line in match.group("body").splitlines()
    )


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


def _assert_hub_task_boundary(text: str) -> None:
    normalized = " ".join(text.split())
    assert "workspace file, read-only Git, web, and Watchlists tools" in normalized
    assert all(
        tool_name in normalized
        for tool_name in ("todo_create", "todo_update", "todo_get", "todo_list")
    )
    assert "require Console session state and are not Hub tools" in normalized
    assert (
        re.search(
            r"(?:includes?|alongside|provides?|offers?)\s+"
            r"(?:[^.]{0,80}\s)?session[- ](?:todo|task)\s+tools",
            normalized,
            re.IGNORECASE,
        )
        is None
    )


def _tool_parameter_names(text: str, tool_name: str) -> list[str]:
    section = re.search(
        rf"^#### `{re.escape(tool_name)}`\s*$\n(?P<body>.*?)(?=^#{{3,4}} |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert section is not None, tool_name
    return re.findall(r"^\| `([^`]+)` \|", section.group("body"), re.MULTILINE)


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


def test_documents_explain_retired_ingest_media_replacement(
    document: tuple[Path, str],
) -> None:
    path, text = document
    normalized = " ".join(text.split())
    assert "retired `ingest_media`" in normalized, path
    assert "Library Import" in normalized, path


def test_mcp_documents_explain_profile_driven_rag_search_compatibility() -> None:
    for path in (DESIGN_DOCUMENT, USER_GUIDE_DOCUMENT):
        normalized = " ".join(path.read_text(encoding="utf-8").split())
        assert "`false` forces media keyword search" in normalized, path
        assert "`true` or omission follows the active RAG profile" in normalized, path
        assert "`plain`, `semantic`, or `hybrid` search mode" in normalized, path


def test_user_guide_explains_weak_match_similarity_provenance() -> None:
    normalized = " ".join(USER_GUIDE_DOCUMENT.read_text(encoding="utf-8").split())
    assert "every similarity-bearing row" in normalized
    assert "every scored row" not in normalized
    for contract in (
        "ordinary semantic rows use their score",
        "hybrid rows use the preserved vector leg when present",
        "FTS-only hybrid, reranker, and unscored keyword rows do not trigger a cosine-similarity claim",
    ):
        assert contract in normalized


def test_console_guide_documents_the_virtual_cli_security_boundary() -> None:
    normalized = " ".join(
        CONSOLE_AGENT_TOOLS_DOCUMENT.read_text(encoding="utf-8").split()
    )
    assert "one model tool named `virtual_cli`" in normalized
    assert "does not accept a command-line string" in normalized
    assert "discoverability is not authorization" in normalized
    assert "Every command has its own Allow, Ask, or Off setting" in normalized
    assert "allowing `fs_read` does not allow virtual `cat`" in normalized
    for command in (
        "ls",
        "cat",
        "grep",
        "find",
        "stat",
        "git_status",
        "git_diff",
        "git_log",
        "git_blame",
        "git_branches",
    ):
        assert f"`{command}`" in normalized


def test_local_library_tools_documentation_uses_current_standalone_inventory() -> None:
    text = LOCAL_LIBRARY_TOOLS_DOCUMENT.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    assert (
        "The standalone server exposes exactly nine implemented legacy tools; "
        "retired `ingest_media` is absent, and persistent URL/file ingestion "
        "uses Library Import."
    ) in normalized


def test_user_guide_keeps_session_task_tools_out_of_hub_inventory() -> None:
    _assert_hub_task_boundary(USER_GUIDE_DOCUMENT.read_text(encoding="utf-8"))


def test_user_guide_hub_inventory_contract_rejects_session_task_synonym() -> None:
    text = USER_GUIDE_DOCUMENT.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    mutated = normalized.replace(
        "workspace file, read-only Git, web, and Watchlists tools",
        "workspace file, read-only Git, web, and Watchlists tools. "
        "It also includes session task tools",
        1,
    )

    assert mutated != normalized
    with pytest.raises(AssertionError):
        _assert_hub_task_boundary(mutated)


def test_watchlists_tools_document_every_public_parameter_and_bound() -> None:
    for path in WATCHLISTS_TOOL_DOCUMENTS:
        text = path.read_text(encoding="utf-8")
        normalized = " ".join(text.split())
        assert _tool_parameter_names(text, "watchlists_search_items") == [
            "query",
            "collection",
            "source",
            "statuses",
            "since",
            "limit",
            "cursor",
        ], path
        assert _tool_parameter_names(text, "watchlists_get_item") == ["item_id"], path
        assert _tool_parameter_names(text, "watchlists_list_sources") == [
            "name",
            "type",
            "state",
            "collection",
            "limit",
            "cursor",
        ], path
        assert _tool_parameter_names(text, "watchlists_list_collections") == [
            "name",
            "limit",
            "cursor",
        ], path
        assert _tool_parameter_names(text, "watchlists_list_briefings") == [
            "collection",
            "statuses",
            "since",
            "limit",
            "cursor",
        ], path
        assert _tool_parameter_names(text, "watchlists_get_briefing") == [
            "briefing_id",
            "selected_cursor",
            "cited_cursor",
        ], path
        assert _tool_parameter_names(text, "watchlists_get_operations_status") == [
            "source",
            "collection",
            "limit",
            "cursor",
        ], path
        assert _tool_parameter_names(text, "watchlists_get_operation_status") == [
            "operation_id"
        ], path
        for contract in (
            "512 characters and 32 whitespace-delimited terms",
            "collection names are limited to 256 characters",
            "source names or configured URLs are limited to 2,048 characters",
            "positive local row ID from 1 through 2^63-1",
            "source integer IDs use the same 1 through 2^63-1 range",
            "non-empty, unique array of at most five values",
            "`new`, `reviewed`, `ingested`, `ignored`, or `error`",
            "inclusive effective-date floor in `YYYY-MM-DD` or RFC 3339 form, normalized to UTC",
            "defaults to 10 and accepts 1 through 50",
            "non-blank opaque string of at most 2,048 characters",
            "required canonical `local:watchlist_item:<positive integer>` ID",
            "item integer is limited to 1 through 2^63-1",
            "maximum 40 characters",
            "Unknown parameters are rejected",
            "Booleans are not accepted as integer IDs or limits",
            "Numeric strings remain names",
        ):
            assert contract in normalized, (path, contract)


def test_watchlists_tools_document_search_evidence_and_cursor_semantics() -> None:
    for path in WATCHLISTS_TOOL_DOCUMENTS:
        text = path.read_text(encoding="utf-8")
        normalized = " ".join(text.split())
        for contract in (
            "literal full-text over title, body, and author; it is not semantic search",
            "Results are local-first",
            "local Watchlists database",
            "newest-first",
            "source-linked",
            "30 KiB",
            "untrusted evidence",
            "`effective_date` is the normalized publication date, falling back to item creation time",
            "`published_date`, `created_at`, and `updated_at` remain separate",
            "`last_checked` and `last_successful_check` remain separate",
            "For “all,” follow `next_cursor` until `has_more` is `false`",
            "Continuation excludes later inserts but is not snapshot isolation",
            "casefolded_name_prefix_asc_name_prefix_asc_id_asc",
            "first 96 Unicode characters",
            "server Watchlists search is not yet supported",
            "`status` is `unsupported`",
            "`retryable` is `false`",
            "`message` is exactly `server Watchlists search is not supported; switch Watchlists to Local before retrying`",
        ):
            assert contract in normalized, (path, contract)


def test_watchlists_design_pins_bounded_name_cursor_ordering() -> None:
    normalized = " ".join(
        WATCHLISTS_REMEDIATION_DESIGN.read_text(encoding="utf-8").split()
    )
    assert "casefolded_name_prefix_asc_name_prefix_asc_id_asc" in normalized
    assert "first 96 Unicode characters" in normalized


def test_watchlists_tools_document_privacy_and_external_mcp_permission() -> None:
    for path in WATCHLISTS_TOOL_DOCUMENTS:
        text = path.read_text(encoding="utf-8")
        normalized = " ".join(text.split())
        for contract in (
            "URL paths are authorized Watchlists metadata",
            "userinfo, query, and fragment are removed",
            "Only absolute HTTP(S) URLs with a host are returned",
            "`[mcp] expose_local_tools`",
            "per-tool permission must be Allow",
            "Ask is refused",
            "send approved metadata and receipts to its client or model",
            "article and briefing content remains Console-only",
        ):
            assert contract in normalized, (path, contract)


def test_watchlists_operations_limit_is_documented_as_one_combined_page() -> None:
    for path in WATCHLISTS_TOOL_DOCUMENTS:
        normalized = " ".join(path.read_text(encoding="utf-8").split())
        start = normalized.index("#### `watchlists_get_operations_status`")
        end = normalized.index("#### `watchlists_get_operation_status`", start)
        section = normalized[start:end]
        assert "combined operation page" in section, path
        assert "per receipt kind" not in section, path


def test_watchlists_documents_pin_external_receipts_and_console_content() -> None:
    for path in WATCHLISTS_TOOL_DOCUMENTS:
        normalized = " ".join(path.read_text(encoding="utf-8").split())
        for name in (
            "watchlists_list_sources",
            "watchlists_list_collections",
            "watchlists_list_briefings",
            "watchlists_get_operations_status",
            "watchlists_get_operation_status",
        ):
            assert name in normalized, (path, name)
        for name in (
            "watchlists_search_items",
            "watchlists_get_item",
            "watchlists_get_briefing",
        ):
            assert name in normalized, (path, name)
        assert "never article snippets, article bodies, briefing Markdown" in normalized or (
            "Console-only" in normalized and "never registers or resolves" in normalized
        ), path


def test_expanded_local_tool_group_copy_names_watchlists_everywhere() -> None:
    """The master-switch label names all three tool groups on every surface.

    Whitespace-normalized, like every other prose assertion in this module, and
    for two reasons rather than one. The obvious one is that these are wrapped
    Markdown paragraphs: the label spans a line break in
    `Docs/User_Guide/console/agent-runs-and-tools.md` ("Local workspace,\nweb,
    and Watchlists tools"), so a raw substring check reported the copy missing
    when it was present and correct.

    The reason that actually matters is the *stale* half. Checking raw text for
    the retired label means a retired label that happens to wrap escapes the
    check -- the guard would pass on exactly the content it exists to catch.
    Reflowing a paragraph must not be able to switch this guard off.

    Raises:
        AssertionError: If a surface still carries the retired two-group label,
            or does not carry the three-group one.
    """
    stale = "Local workspace + web tools"
    expected = "workspace, web, and Watchlists"
    for path in LOCAL_TOOL_COPY_SURFACES:
        normalized = " ".join(path.read_text(encoding="utf-8").split())
        assert stale not in normalized, path
        assert expected in normalized, path


def test_builtin_gate_rejects_two_category_master_switch_copy() -> None:
    text = BUILTIN_TOOL_GATE.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    assert "standard web research" not in normalized
    assert "local/web master switch" not in normalized


def test_skill_author_inventory_links_watchlists_task_and_permission_decisions() -> (
    None
):
    text = SKILLS_EXAMPLES_DOCUMENT.read_text(encoding="utf-8")
    assert (
        "task-16222%20-%20Expose-local-Watchlists-search-tools-to-Console-and-MCP.md"
        in text
    )
    assert "030-local-library-agent-tool-boundary.md" in text
    assert "032-local-agent-tool-permission-boundary.md" in text


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
    warning = _admonition_block(text, "WARNING")
    assert "user's OS access" in warning, path
    assert "private local Library" in warning, path
    assert "tools, resources, and prompts" in warning, path
    assert "off-device" in warning, path
    assert "cloud model" in warning, path


def test_user_guide_warning_names_private_watchlists_egress_and_trust_boundary() -> (
    None
):
    warning = _admonition_block(
        USER_GUIDE_DOCUMENT.read_text(encoding="utf-8"), "WARNING"
    )
    assert "private Watchlists source, collection, briefing-receipt, and operation metadata" in warning
    assert "does not expose Watchlists article snippets or bodies, or briefing Markdown/provenance" in warning
    assert "external MCP client may send" in warning
    assert "off-device to a cloud model" in warning
    assert "trust both the client and the model provider" in warning


def test_internal_watchlists_provider_inventories_reject_verified_stale_copy() -> None:
    stale_by_path = {
        LOCAL_TOOL_PROVIDER: ("workspace-local fs_/web_/todo_ tools",),
        MCP_SERVER: (
            "workspace-local agent tools (`fs_*`",
            "workspace-local agent tools (fs_*/git_*/web_*)",
        ),
        MCP_WORKBENCH: (
            "workspace-local agent tool set (fs_*/git_*/web_*)",
            "local/web master switch",
        ),
        MCP_TOOLS_MODE: ("local/web provider master switch",),
    }
    for path, stale_phrases in stale_by_path.items():
        text = path.read_text(encoding="utf-8")
        assert "workspace, web, and Watchlists" in text, path
        for stale in stale_phrases:
            assert stale not in text, (path, stale)


def test_design_distinguishes_payload_free_diagnostics_from_authorized_egress() -> None:
    text = DESIGN_DOCUMENT.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    assert (
        re.search(
            r"sensitive data (?:is|are) never exposed through MCP",
            normalized,
            re.IGNORECASE,
        )
        is None
    )
    assert "Internal diagnostics and refusals are payload-free." in normalized
    assert "Authorized external MCP clients can read private Library data" in normalized
    assert "may send that data onward" in normalized


def test_design_create_note_documents_exact_parameter_inventory() -> None:
    text = DESIGN_DOCUMENT.read_text(encoding="utf-8")
    section_match = re.search(
        r"^#### `create_note`\s*$\n(?P<body>.*?)(?=^#### |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert section_match is not None
    parameters = re.findall(
        r"^  - `([^`]+)`: ", section_match.group("body"), re.MULTILINE
    )
    assert parameters == ["title", "content"]


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
        "mcp.enabled",
        "enabled = true",
    )
    assert all(claim not in text for claim in stale_claims)
    normalized = " ".join(text.split())
    assert "standalone gateway is stdio-only" in normalized
    assert "600 requests per minute" in normalized
    assert "16 in-flight requests" in normalized
    assert "`[mcp] expose_local_tools = false`" in text
    assert (
        "The only `[mcp]` configuration key consumed by the standalone gateway "
        "is `expose_local_tools`."
    ) in normalized


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
