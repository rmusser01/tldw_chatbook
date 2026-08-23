"""A factory-built test app is a returning Library user, not a new profile.

`app.py` sets `library_new_profile_admission` from
`first_profile_created_this_session()`. The per-test config sandbox creates a
profile for *every* test, so the flag was True for every app the shared factory
built -- each one claiming to be a brand-new profile.

The Library rail answers that claim correctly, with progressive disclosure: it
composes a compact starter rail (two rows plus "Explore all tools") and returns
before the search input, before every Browse/Create section, and before the
Details disclosure (`Widgets/Library/library_rail.py`). Rows such as
`#library-row-browse-media` are therefore not in the DOM at all -- which is why
~143 Library tests written before progressive disclosure failed with
`NoMatches ... on LibraryScreen()`.

Three Library modules had already hand-rolled this same clearing in local
`_build_test_app` wrappers. TASK-21280 hoisted it to the one factory they all
go through. These tests pin the default, and pin that a test which is *about*
new-profile admission can still get one.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Library.library_rail_state import LibraryLifecycle

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _two_conversations,
    _two_media_items,
    _seed_conversations,
    _wait_for_library_shell,
)

#: Every selector the failing cluster queried, so a future change that
#: re-strands them names them rather than reporting one arbitrary miss.
BROWSE_ROWS = (
    "#library-row-browse-media",
    "#library-row-browse-notes",
    "#library-row-browse-conversations",
    "#library-row-browse-collections",
    "#library-row-browse-prompts",
    "#library-row-browse-search",
    "#library-search-input",
)


def test_the_factory_clears_new_profile_admission() -> None:
    assert _build_test_app().library_new_profile_admission is False


def test_a_new_profile_can_still_be_requested() -> None:
    """The flag is what `first_profile_created_this_session()` produced, not a
    value this factory invented, so a new-profile test must be able to keep it."""
    app = _build_test_app(preserve_profile_admission=True)
    assert app.library_new_profile_admission is True


@pytest.mark.asyncio
async def test_the_rail_composes_its_full_sections_for_a_factory_app() -> None:
    """The behaviour the default exists for, asserted on the DOM rather than on
    the flag -- the flag is the mechanism, this is the contract.

    Note this asserts no lifecycle *value*: the factory deliberately does not
    pin one, so the screen still derives its own. An existing profile with no
    persisted lifecycle settling to Expanded is the product's contract, pinned
    by `test_library_real_existing_config_without_lifecycle_defaults_expanded`.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        assert screen._library_lifecycle is not LibraryLifecycle.UNKNOWN
        assert screen._library_lifecycle is not LibraryLifecycle.STARTER
        missing = [sel for sel in BROWSE_ROWS if not screen.query(sel)]
        assert not missing, f"rail did not compose: {missing}"
