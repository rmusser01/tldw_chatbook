"""
Test suite for TabLinks navigation widget.

Tests that each tab link is clickable and navigates to the proper window.
"""

import pytest
from pathlib import Path
import sys
from types import SimpleNamespace

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tldw_chatbook.app import TldwCli
from tldw_chatbook.UI.Tab_Links import TabLinks
from tldw_chatbook.Constants import (
    TAB_CHAT, TAB_CCP, TAB_NOTES, TAB_MEDIA, TAB_SEARCH,
    TAB_INGEST, TAB_EVALS, TAB_LLM, TAB_TOOLS_SETTINGS,
    TAB_STATS, TAB_LOGS, TAB_CODING, TAB_STTS, TAB_STUDY,
    TAB_CHATBOOKS, ALL_TABS, TAB_GROUPS
)


def _tab_label_text(link) -> str:
    """Return plain label text for a Static tab link."""
    rendered = link.render()
    return getattr(rendered, "plain", str(rendered)).strip()


def _separator_counts() -> tuple[int, int]:
    """Return expected thin and group separator counts for the grouped nav."""
    visible_groups = []
    grouped_tabs = set()

    for group_tab_ids in TAB_GROUPS.values():
        visible_tabs = [tab_id for tab_id in group_tab_ids if tab_id in ALL_TABS]
        if visible_tabs:
            visible_groups.append(visible_tabs)
            grouped_tabs.update(visible_tabs)

    ungrouped_tabs = [tab_id for tab_id in ALL_TABS if tab_id not in grouped_tabs]

    thin_separator_count = sum(max(len(group) - 1, 0) for group in visible_groups)
    if ungrouped_tabs:
        thin_separator_count += max(len(ungrouped_tabs) - 1, 0)

    group_separator_count = max(len(visible_groups) + (1 if ungrouped_tabs else 0) - 1, 0)
    return thin_separator_count, group_separator_count


def _expected_current_tab(tab_id: str) -> str:
    """Return the app-level current_tab value for a clicked tab link."""
    return tab_id


async def _click_tab_link(tab_links: TabLinks, pilot, tab_id: str, pause: float = 1.0) -> None:
    """Dispatch a tab-link click through the TabLinks container."""
    link = tab_links.query_one(f"#tab-link-{tab_id}")
    original_get_widget_at = tab_links.app.get_widget_at
    tab_links.app.get_widget_at = lambda _x, _y: (link, None)
    try:
        await tab_links.on_click(SimpleNamespace(screen_x=0, screen_y=0))
        await pilot.pause(pause)
    finally:
        tab_links.app.get_widget_at = original_get_widget_at


async def _wait_for_current_tab(app: TldwCli, pilot, expected_tab: str, attempts: int = 30) -> None:
    """Wait for app.current_tab to settle on the expected value."""
    for _ in range(attempts):
        if app.current_tab == expected_tab:
            return
        await pilot.pause(0.1)


@pytest.mark.asyncio
class TestTabLinksNavigation:
    """Test suite for TabLinks navigation functionality."""
    
    async def test_all_tab_links_clickable_and_navigate(self):
        """Test that each top-level tab link is clickable and navigates to the proper window."""
        # Create the full app
        app = TldwCli()
        
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(7)  # Wait for splash screen and UI initialization
            
            # Verify TabLinks is present
            tab_links = app.query_one(TabLinks)
            assert tab_links is not None, "TabLinks widget should be present"
            
            # Test each tab link
            for tab_id in ALL_TABS:
                tab_links = app.query_one(TabLinks)
                link_id = f"#tab-link-{tab_id}"
                link = tab_links.query_one(link_id)
                assert link is not None, f"Tab link for {tab_id} should exist"

                await _click_tab_link(tab_links, pilot, tab_id, pause=1.0)
                
                # Verify navigation happened
                expected_tab = _expected_current_tab(tab_id)
                await _wait_for_current_tab(app, pilot, expected_tab)
                assert app.current_tab == expected_tab, f"Should have navigated to {tab_id}"
                
                # Verify the link is marked as active
                tab_links = app.query_one(TabLinks)
                link = tab_links.query_one(link_id)
                assert "-active" in link.classes, f"Tab link {tab_id} should be marked as active"
                
                # Verify other links are not active
                for other_tab_id in ALL_TABS:
                    if other_tab_id != tab_id:
                        other_link = tab_links.query_one(f"#tab-link-{other_tab_id}")
                        assert "-active" not in other_link.classes, \
                            f"Tab {other_tab_id} should not be active when {tab_id} is selected"
    
    async def test_tab_links_initial_state(self):
        """Test that TabLinks initializes with correct active tab."""
        app = TldwCli()
        
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(7)  # Wait for splash screen and UI initialization
            
            tab_links = app.query_one(TabLinks)

            # Check initial active tab (should be TAB_CHAT by default)
            chat_link = tab_links.query_one(f"#tab-link-{TAB_CHAT}")
            assert "-active" in chat_link.classes, "Chat tab should be initially active"
            
            # Check other tabs are not active
            for tab_id in ALL_TABS:
                if tab_id != TAB_CHAT:
                    link = tab_links.query_one(f"#tab-link-{tab_id}")
                    assert "-active" not in link.classes, f"{tab_id} should not be initially active"
    
    async def test_tab_labels_correct(self):
        """Test that each tab has the correct label text."""
        expected_labels = {
            TAB_CHAT: "Chat",
            TAB_CCP: "CCP",
            TAB_NOTES: "Notes",
            TAB_MEDIA: "Media",
            TAB_SEARCH: "Search",
            TAB_INGEST: "Ingest",
            TAB_EVALS: "Evals",
            TAB_LLM: "LLM",
            TAB_TOOLS_SETTINGS: "Settings",
            TAB_STATS: "Stats",
            TAB_LOGS: "Logs",
            TAB_CODING: "Coding",
            TAB_STTS: "S/TT/S",
            TAB_STUDY: "Study",
            TAB_CHATBOOKS: "Chatbooks"
        }
        
        app = TldwCli()
        
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(7)  # Wait for splash screen and UI initialization
            
            tab_links = app.query_one(TabLinks)

            for tab_id in ALL_TABS:
                link = tab_links.query_one(f"#tab-link-{tab_id}")
                actual_label = _tab_label_text(link)
                
                # Handle special cases where tab_id doesn't match expected labels
                if tab_id == "conversations_characters_prompts":
                    expected = "CCP"
                elif tab_id == "llm_management":
                    expected = "LLM"
                elif tab_id == "subscriptions":
                    expected = "Subscriptions"
                elif tab_id == "stts":
                    expected = "S/TT/S"
                else:
                    expected = expected_labels.get(tab_id, tab_id.replace('_', ' ').title())
                
                assert actual_label == expected, \
                    f"Tab {tab_id} should have label '{expected}', got '{actual_label}'"
    
    async def test_separators_present(self):
        """Test that grouped navigation separators are rendered correctly."""
        app = TldwCli()
        
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(7)  # Wait for splash screen and UI initialization
            
            tab_links = app.query_one(TabLinks)
            thin_expected, group_expected = _separator_counts()

            thin_separators = tab_links.query(".tab-separator")
            group_separators = tab_links.query(".tab-group-separator")

            assert len(thin_separators) == thin_expected, \
                f"Should have {thin_expected} thin separators, found {len(thin_separators)}"
            assert len(group_separators) == group_expected, \
                f"Should have {group_expected} group separators, found {len(group_separators)}"
    
    async def test_rapid_tab_switching(self):
        """Test that rapid tab switching works correctly."""
        app = TldwCli()
        
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(7)  # Wait for splash screen and UI initialization
            
            # Rapidly switch between multiple tabs
            test_sequence = [TAB_CHAT, TAB_NOTES, TAB_MEDIA, TAB_CHAT, TAB_CODING]
            
            for tab_id in test_sequence:
                tab_links = app.query_one(TabLinks)
                await _click_tab_link(tab_links, pilot, tab_id, pause=1.0)
            
            # Final tab should be active
            final_tab = test_sequence[-1]
            assert app.current_tab == _expected_current_tab(final_tab), f"Should end on {final_tab}"
            
            # Check active state is correct
            final_link = app.query_one(TabLinks).query_one(f"#tab-link-{final_tab}")
            assert "-active" in final_link.classes, f"{final_tab} should be active"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
