"""LibraryScreen rail-level UI tests."""

import pytest
import pytest_asyncio
from textual.widgets import Button

from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _seed_conversations,
    _two_conversations,
    _wait_for_library_shell,
)
from Tests.UI.test_screen_navigation import _build_test_app


@pytest_asyncio.fixture
async def library_screen():
    """Provide a mounted LibraryScreen with its rail fully loaded."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        await _wait_for_library_shell(host.screen, pilot)
        yield host.screen


@pytest.mark.asyncio
async def test_ingest_button_present(library_screen):
    """The rail-top Ingest button is rendered with the expected label."""
    button = library_screen.query_one("#library-ingest-top-button", Button)
    assert str(button.label) == "Ingest content…"
