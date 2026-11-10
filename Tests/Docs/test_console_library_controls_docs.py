"""Documentation contract for Console's per-conversation Library controls."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_CONTRACT: dict[str, tuple[str, ...]] = {
    # TASK-38974: the README (d01daac8b4) names the four controls in prose,
    # not by their labels; pinned at that wording.
    "README.md": (
        "manual Library search, automatic retrieval policy, assistant tool "
        "permission, and direct/RAG retrieval mode",
        "New conversations do not silently grant every tool or source",
    ),
    "Docs/User_Guide/console.md": (
        "Manual Search Library",
        "Auto: Never / Automatic",
        "Assistant: Blocked / Allowed",
        "Direct / RAG",
    ),
    "Docs/User_Guide/console/context-and-rag.md": (
        "Manual Search Library is always available",
        "Auto: Never / Automatic",
        "Assistant: Blocked / Allowed",
        "Direct / RAG is a selector",
        "Notes, Media, and Conversations",
        "Retry",
        "Send once",
        "Cancel",
        "Zero results",
    ),
    "Docs/User_Guide/settings/rag.md": (
        "does not grant automatic retrieval or assistant Library access",
        "per-conversation Library controls",
        "Conversation defaults",
        "Automatic retrieval",
        "Agent Library access",
        "Allowed Library access",
    ),
    "Docs/User_Guide/library/search-and-rag.md": (
        "user-initiated Library search",
        "does not change the current conversation's Library controls",
    ),
    "Docs/User_Guide/library/import-and-export.md": (
        "device-local Library policy",
        "assistant generation state",
        "unresolved imported or remote state remains inert",
    ),
    # TASK-38974: pinned at the wording f09a7786d4 shipped for "Use as source".
    "Docs/User_Guide/library/media-and-conversations.md": (
        "does not send automatically or change the conversation's Auto or "
        "Assistant setting",
        "Use as source stages a reference to the selected conversation in the "
        "current Console session",
    ),
    "Docs/Development/Agent-Tools/local-library-tools.md": (
        "selector, not an enable switch",
        "18 direct Library tools",
        "search_library_rag",
        "statically reserved",
    ),
    "Docs/Design/MCP.md": (
        "independent of Console's per-conversation Library policy",
        "Direct / RAG selector",
    ),
    "Docs/User_Guide/mcp.md": (
        "not governed by Console's per-conversation Library controls",
        "Direct / RAG selector",
    ),
}
MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
# Any URL scheme (http:, mailto:, media:, ...) names no file in the tree.
URL_SCHEME_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")
FORBIDDEN_CONTEXT_AND_RAG_CLAIMS = (
    "a send is never blocked on it",
    "the send goes out without evidence",
    "zero-result outcome currently clears the in-flight placeholder with no further notice",
    "writes [chat_defaults] rag_auto_retrieve_on_send = true at toggle time",
    "before Esc, and Esc leaves it set",
)


@pytest.mark.parametrize(("relative_path", "required_text"), DOC_CONTRACT.items())
def test_console_library_control_docs_state_the_contract(
    relative_path: str,
    required_text: tuple[str, ...],
) -> None:
    """Each governed page states the part of the control contract it owns."""
    text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    prose = re.sub(r"[*`]", "", text)
    prose = " ".join(prose.split())

    missing = [fragment for fragment in required_text if fragment not in prose]

    assert not missing, f"{relative_path} is missing: {missing}"


@pytest.mark.parametrize("relative_path", DOC_CONTRACT)
def test_console_library_control_docs_have_valid_local_links(
    relative_path: str,
) -> None:
    """Every local Markdown link in the governed pages resolves in the tree."""
    missing = missing_local_links(REPO_ROOT / relative_path)

    assert not missing, f"{relative_path} has missing local links: {missing}"


def missing_local_links(
    document: Path, link_re: re.Pattern[str] = MARKDOWN_LINK_RE
) -> list[str]:
    """Return the link targets in ``document`` that name no file in the tree."""
    missing: list[str] = []
    for raw_target in link_re.findall(document.read_text(encoding="utf-8")):
        target = raw_target.strip()
        if target.startswith("<") and target.endswith(">"):
            target = target[1:-1]
        else:
            target = target.split(maxsplit=1)[0]
        if URL_SCHEME_RE.match(target):
            continue
        local_path = unquote(target.partition("#")[0])
        resolved = (
            document if not local_path else (document.parent / local_path).resolve()
        )
        if not resolved.exists():
            missing.append(target)
    return missing


def test_context_and_rag_has_no_superseded_automatic_retrieval_claims() -> None:
    """The current guide cannot retain behavior contradicted by ADR-079."""
    text = (REPO_ROOT / "Docs/User_Guide/console/context-and-rag.md").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(text.split())

    stale = [claim for claim in FORBIDDEN_CONTEXT_AND_RAG_CLAIMS if claim in normalized]

    assert not stale, f"context-and-rag.md retains superseded claims: {stale}"
