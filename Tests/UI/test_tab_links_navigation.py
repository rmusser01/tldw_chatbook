"""
Test suite for TabLinks navigation widget.

Tests that each tab link is clickable and navigates to the proper window.
"""

import pytest
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tldw_chatbook.app import TldwCli
from tldw_chatbook.UI.Tab_Links import TabLinks
from tldw_chatbook.Constants import (
    TAB_CHAT, TAB_CCP, TAB_NOTES, TAB_MEDIA, TAB_SEARCH,
    TAB_INGEST, TAB_EVALS, TAB_LLM, TAB_TOOLS_SETTINGS,
    TAB_STATS, TAB_LOGS, TAB_CODING, TAB_STTS, TAB_STUDY,
    TAB_CHATBOOKS, ALL_TABS
)


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
                # Get the tab link
                link_id = f"#tab-link-{tab_id}"
                link = app.query_one(link_id)
                assert link is not None, f"Tab link for {tab_id} should exist"
                
                # Check if link is visible in viewport, if not scroll to it
                container = app.query_one("#tab-links-container")
                
                # Scroll the widget into view if needed
                container.scroll_to_widget(link, animate=False)
                await pilot.pause(0.2)  # Let scroll complete
                
                # Click the tab link
                await pilot.click(link_id)
                await pilot.pause(1.0)  # Let the navigation complete - some tabs are heavy
                
                # Verify navigation happened
                assert app.current_tab == tab_id, f"Should have navigated to {tab_id}"
                
                # Verify the link is marked as active
                link = app.query_one(link_id)  # Re-query after state change
                assert "-active" in link.classes, f"Tab link {tab_id} should be marked as active"
                
                # Verify other links are not active
                for other_tab_id in ALL_TABS:
                    if other_tab_id != tab_id:
                        other_link = app.query_one(f"#tab-link-{other_tab_id}")
                        assert "-active" not in other_link.classes, \
                            f"Tab {other_tab_id} should not be active when {tab_id} is selected"
    
    async def test_tab_links_initial_state(self):
        """Test that TabLinks initializes with correct active tab."""
        app = TldwCli()
        
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(7)  # Wait for splash screen and UI initialization
            
            # Check initial active tab (should be TAB_CHAT by default)
            chat_link = app.query_one(f"#tab-link-{TAB_CHAT}")
            assert "-active" in chat_link.classes, "Chat tab should be initially active"
            
            # Check other tabs are not active
            for tab_id in ALL_TABS:
                if tab_id != TAB_CHAT:
                    link = app.query_one(f"#tab-link-{tab_id}")
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
            
            for tab_id in ALL_TABS:
                link = app.query_one(f"#tab-link-{tab_id}")
                actual_label = str(link.renderable).strip()
                
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
        """Test that separators are present between tab links."""
        app = TldwCli()
        
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(7)  # Wait for splash screen and UI initialization
            
            # Check that separators exist
            separators = app.query(".tab-separator")
            # Should be one less separator than tabs
            assert len(separators) == len(ALL_TABS) - 1, \
                f"Should have {len(ALL_TABS) - 1} separators, found {len(separators)}"
    
    async def test_rapid_tab_switching(self):
        """Test that rapid tab switching works correctly."""
        app = TldwCli()
        
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(7)  # Wait for splash screen and UI initialization
            
            # Rapidly switch between multiple tabs
            test_sequence = [TAB_CHAT, TAB_NOTES, TAB_MEDIA, TAB_CHAT, TAB_CODING]
            
            for tab_id in test_sequence:
                await pilot.click(f"#tab-link-{tab_id}")
                await pilot.pause(0.05)  # Small pause between clicks
            
            # Final tab should be active
            final_tab = test_sequence[-1]
            assert app.current_tab == final_tab, f"Should end on {final_tab}"
            
            # Check active state is correct
            final_link = app.query_one(f"#tab-link-{final_tab}")
            assert "-active" in final_link.classes, f"{final_tab} should be active"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])