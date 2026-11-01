"""Slower arms of `test_library_rag_answer_names_model.py`, outside the PR lane
(TASK-34000.21, Review Focus 5: a very long model name).

The quiet line is a single reserved row (`h-1`, `markup=False`) so the
query region never shifts; a 60-character model name must not break that
contract at 80 columns, and the painted row must still begin with the
recipient AND show the start of the model. Before the `#library-rag-query-
quiet-line` nowrap/ellipsis rule, Textual word-wrapped the long name onto
a clipped second row and the user saw only "To openai ·" -- a real
`claude-sonnet-4-5-20250929` at 80 columns did the same.
"""

from __future__ import annotations

import pytest
from textual.widgets import Input, Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_rag_answer_names_model import (
    _painted,
    _persist_provider,
)
from Tests.UI.test_library_rag_result_focus import _assert_painted
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _seed_conversations,
    _StaticLibraryRagSearchService,
    _wait_for_library_rag_query_ready,
    _wait_for_library_shell,
)
from Tests.UI.test_product_maturity_gate16_library_search_rag import (
    _rag_result_fixture,
    _switch_to_rag_mode,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

LONG_MODEL = "gpt-4.1-mini-" + "x" * 47
assert len(LONG_MODEL) == 60


async def test_long_model_name_keeps_the_quiet_line_on_one_row_at_80_columns(
    monkeypatch,
):
    _persist_provider(
        monkeypatch, provider="OpenAI", model=LONG_MODEL, key_field="openai"
    )
    app = _build_test_app()
    _seed_conversations(app, [], notes=[{"id": "note-42", "title": "Incident"}])
    app.library_rag_search_service = _StaticLibraryRagSearchService(
        _rag_result_fixture()
    )
    app.library_rag_answer_chat = lambda **kwargs: ""
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=(80, 24)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-search").press()
        await _switch_to_rag_mode(screen, pilot)
        field = screen.query_one("#library-rag-query-input", Input)
        field.value = "Why did the incident happen?"
        await _wait_for_library_rag_query_ready(screen, pilot, field.value)
        await pilot.pause()

        quiet = screen.query_one("#library-rag-query-quiet-line", Static)
        _assert_painted(screen, quiet)
        assert quiet.region.height == 1
        assert str(quiet.renderable) == f"To openai · {LONG_MODEL}: question + evidence"
        rows = [
            strip.text
            for strip in screen._compositor.render_strips()
            if "To openai ·" in strip.text
        ]
        assert len(rows) == 1, _painted(screen)
        row = rows[0]
        # The recipient and the start of the model are on the one row, and
        # the clip is announced rather than silent.
        assert "To openai · gpt-4.1-mini-xxxx" in row, row
        assert "…" in row, row
        # Nothing of the sentence spilled onto another row.
        assert "question + evidence" not in _painted(screen).replace(row, "")
