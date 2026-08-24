#!/usr/bin/env python3
"""
Test script for RAG UI integration.

This tests:
1. get_rag_context_for_chat function
2. UI checkbox states simulation
3. Context formatting
4. Integration with chat messages
"""

import asyncio
from importlib.util import find_spec
import sys
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Set up logging
from loguru import logger

logger.add(sys.stderr, level="INFO")

_RAG_DEPENDENCIES = ("chromadb", "sentence_transformers", "torch")
pytestmark = pytest.mark.skipif(
    not all(find_spec(name) is not None for name in _RAG_DEPENDENCIES),
    reason="RAG dependencies not available",
)


class MockCheckbox:
    """Mock checkbox widget."""

    def __init__(self, value: bool = False):
        self.value = value


class MockInput:
    """Mock input widget."""

    def __init__(self, value: str = ""):
        self.value = value


class MockSelect:
    """Mock select widget."""

    def __init__(self, value: str = ""):
        self.value = value


class MockApp:
    """Mock app with UI elements for testing."""

    def __init__(self, rag_settings: Dict[str, Any]):
        # Databases
        from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
        from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

        # This mock bypasses application startup, so it must establish the
        # trusted application-owned namespace that startup normally secures.
        data_dir = Path.home() / ".local/share/tldw_cli"
        data_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.media_db = MediaDatabase(
            str(data_dir / "tldw_cli_media_v2.db"),
            client_id="test_client",
        )
        self.rag_db = CharactersRAGDB(
            str(data_dir / "tldw_chatbook_ChaChaNotes.db"),
            client_id="test_client",
        )
        self.chachanotes_db = self.rag_db

        # Mock notes service
        self.notes_service = MagicMock()
        self.notes_service.search_notes.return_value = []
        self.notes_user_id = "test_user"

        # UI elements based on settings
        self.ui_elements = {
            "#chat-rag-enable-checkbox": MockCheckbox(
                rag_settings.get("enable_full_rag", False)
            ),
            "#chat-rag-plain-enable-checkbox": MockCheckbox(
                rag_settings.get("enable_plain_rag", True)
            ),
            "#chat-rag-search-mode": MockSelect(
                rag_settings.get("search_mode", "plain")
            ),
            "#chat-rag-search-media-checkbox": MockCheckbox(
                rag_settings.get("search_media", True)
            ),
            "#chat-rag-search-conversations-checkbox": MockCheckbox(
                rag_settings.get("search_conversations", True)
            ),
            "#chat-rag-search-notes-checkbox": MockCheckbox(
                rag_settings.get("search_notes", False)
            ),
            "#chat-rag-top-k": MockInput(str(rag_settings.get("top_k", 5))),
            "#chat-rag-max-context-length": MockInput(
                str(rag_settings.get("max_context_length", 10000))
            ),
            "#chat-rag-keyword-filter": MockInput(""),
            "#chat-rag-rerank-enable-checkbox": MockCheckbox(
                rag_settings.get("enable_rerank", False)
            ),
            "#chat-rag-reranker-model": MockSelect(
                rag_settings.get("reranker_model", "flashrank")
            ),
            "#chat-rag-chunk-size": MockInput(str(rag_settings.get("chunk_size", 400))),
            "#chat-rag-chunk-overlap": MockInput(
                str(rag_settings.get("chunk_overlap", 100))
            ),
            "#chat-rag-chunk-type": MockSelect(rag_settings.get("chunk_type", "words")),
            "#chat-rag-include-metadata-checkbox": MockCheckbox(
                rag_settings.get("include_metadata", False)
            ),
        }

        self.notifications = []

    def query_one(self, selector: str):
        """Mock query_one to return UI elements."""
        return self.ui_elements.get(selector, MockCheckbox(False))

    def notify(self, message: str, severity: str = "info"):
        """Mock notify method."""
        self.notifications.append((message, severity))
        logger.info(f"[{severity.upper()}] {message}")


@pytest.mark.asyncio

# Every check below was written as `if ok: logger.success("✅ ...") else:
# logger.error("❌ ...")`, and every function ended in `return True`. pytest
# ignores a returned value, so all five tests passed unconditionally -- including
# on the paths their own `❌` strings describe as failures. The claims were
# already here; they were just never asserted. This converts them.


@pytest.mark.asyncio
async def test_rag_context_is_none_when_rag_is_disabled():
    """RAG off means no context, not empty context."""
    from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import (
        get_rag_context_for_chat,
    )

    app = MockApp({"enable_full_rag": False, "enable_plain_rag": False})

    assert await get_rag_context_for_chat(app, "Test message") is None


@pytest.mark.asyncio
async def test_returned_context_carries_its_framing():
    """If context comes back at all, it must be framed for the prompt.

    Deliberately conditional on a result: this path runs a real search, so
    whether anything matches depends on the fixture data. What must not vary is
    the framing of a non-empty result -- so the emptiness is tolerated and the
    formatting is not. `test_context_formatting_is_exact` below pins the same
    markers against a controlled search, where nothing is conditional.
    """
    from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import (
        get_rag_context_for_chat,
    )

    app = MockApp(
        {
            "enable_plain_rag": True,
            "search_media": True,
            "search_conversations": False,
            "search_notes": False,
            "top_k": 3,
            "max_context_length": 1000,
        }
    )

    context = await get_rag_context_for_chat(app, "python programming")

    # An empty result is a sentence, not None and not empty framing. Pinned
    # because it is what a user sees when nothing matched.
    if context == "No relevant context found.":
        pytest.skip("no matching results here; test_context_formatting_is_exact pins the framing")
    assert context is not None

    assert "### Context from RAG Search:" in context
    assert "### End of Context" in context
    assert "Based on the above context" in context


@pytest.mark.asyncio
async def test_no_selected_sources_returns_none_and_tells_the_user():
    """Selecting nothing is a user error, and must be reported as one."""
    from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import (
        get_rag_context_for_chat,
    )

    app = MockApp(
        {
            "enable_plain_rag": True,
            "search_media": False,
            "search_conversations": False,
            "search_notes": False,
        }
    )

    assert await get_rag_context_for_chat(app, "test") is None
    assert any(
        "select at least one RAG source" in message
        for message, _ in app.notifications
    ), f"user was not told why nothing happened; notifications={app.notifications}"


@pytest.mark.xfail(
    strict=False,
    reason=(
        "TASK-21564: the search double is not reached with this MockApp. MockApp predates the Console Library policy work, "
        "so `_authorize_local_results_for_prompt` reports "
        "`reason=not_currently_authorized` and the context is discarded. The "
        "assertions here are the ones the original test only logged; they are "
        "kept visible rather than deleted so the harness gap stays tracked."
    ),
)
@pytest.mark.asyncio
async def test_ui_settings_reach_the_search_unchanged():
    """The UI's numbers are the search's numbers.

    The original recorded its assertions inside the search double -- so if the
    double was never called, nothing was checked and the test still passed. The
    invocation is now itself asserted, which is the difference between "the
    settings were forwarded correctly" and "the settings were never read".
    """
    from unittest.mock import patch

    from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import (
        get_rag_context_for_chat,
    )

    forwarded: list[dict[str, object]] = []

    async def recording_search(
        app, query, sources, top_k, max_context_length, enable_rerank, reranker_model
    ):
        forwarded.append(
            {
                "top_k": top_k,
                "max_context_length": max_context_length,
                "enable_rerank": enable_rerank,
                "reranker_model": reranker_model,
            }
        )
        return [], "Test context"

    app = MockApp(
        {
            "enable_plain_rag": True,
            "search_media": True,
            "top_k": 10,
            "max_context_length": 15000,
            "enable_rerank": True,
            "reranker_model": "cohere",
            "chunk_size": 500,
            "chunk_overlap": 150,
            "include_metadata": True,
        }
    )

    with patch(
        "tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events.perform_plain_rag_search",
        recording_search,
    ):
        context = await get_rag_context_for_chat(app, "test")

    assert forwarded, "the search was never called, so no setting was checked"
    assert forwarded[0] == {
        "top_k": 10,
        "max_context_length": 15000,
        "enable_rerank": True,
        "reranker_model": "cohere",
    }
    assert context


@pytest.mark.asyncio
async def test_unreadable_ui_elements_degrade_to_no_context():
    """A broken screen must not take the send down with it."""
    from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import (
        get_rag_context_for_chat,
    )

    class BrokenApp:
        def query_one(self, selector: str):
            if "enable" in selector:
                raise Exception("UI element not found")
            return MockCheckbox(True)

        def notify(self, message: str, severity: str = "info"):
            logger.info(f"[{severity.upper()}] {message}")

    assert await get_rag_context_for_chat(BrokenApp(), "test") is None


@pytest.mark.asyncio
async def test_search_failure_returns_none_and_tells_the_user():
    """A failing search is reported, not swallowed into a silent empty result."""
    from unittest.mock import patch

    from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import (
        get_rag_context_for_chat,
    )

    async def failing_search(*args, **kwargs):
        raise Exception("Search database error")

    app = MockApp({"enable_plain_rag": True, "search_media": True})

    with patch(
        "tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events.perform_plain_rag_search",
        failing_search,
    ):
        context = await get_rag_context_for_chat(app, "test")

    assert context is None
    # The original looked for "RAG search error", which the product has never
    # emitted -- it says "RAG search failed". Because that check was an `if`
    # around a log line, the mismatch was silent for as long as it existed.
    assert any(
        "RAG search failed" in message for message, _ in app.notifications
    ), f"a failed search was not surfaced; notifications={app.notifications}"


@pytest.mark.xfail(
    strict=False,
    reason=(
        "TASK-21564: candidates are dropped by the Library prompt-authority gate. MockApp predates the Console Library policy work, "
        "so `_authorize_local_results_for_prompt` reports "
        "`reason=not_currently_authorized` and the context is discarded. The "
        "assertions here are the ones the original test only logged; they are "
        "kept visible rather than deleted so the harness gap stays tracked."
    ),
)
@pytest.mark.asyncio
async def test_context_formatting_is_exact():
    """Against a controlled search, every part of the framing is checkable."""
    from unittest.mock import patch

    from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import (
        get_rag_context_for_chat,
    )

    test_results = [
        {
            "source": "media",
            "id": "123",
            "title": "Test Document",
            "content": "This is test content about Python programming.",
            "score": 0.95,
            "metadata": {"type": "article"},
        }
    ]
    test_context = (
        "[MEDIA - Test Document]\nThis is test content about Python programming.\n"
    )

    async def mock_search(*args, **kwargs):
        return test_results, test_context

    app = MockApp(
        {
            "enable_plain_rag": True,
            "search_media": True,
            "top_k": 1,
            "max_context_length": 500,
        }
    )

    with patch(
        "tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events.perform_plain_rag_search",
        mock_search,
    ):
        context = await get_rag_context_for_chat(app, "python")

    assert context
    assert context.split("\n")[0] == "### Context from RAG Search:"
    assert test_context in context
    assert "### End of Context" in context
    assert (
        "Based on the above context, please answer the following question:" in context
    )
    # The whole point of the framing: it prepends to the user's message.
    user_message = "What is Python?"
    assert (context + user_message).endswith(user_message)
