#!/usr/bin/env python3
"""Simple test to check if TabLinks is working."""

import asyncio
from unittest.mock import patch
from tldw_chatbook.app import TldwCli
from tldw_chatbook.UI.Tab_Links import TabLinks
from tldw_chatbook.Constants import TAB_CHAT

async def test_tab_links():
    """Test TabLinks directly."""
    with patch('tldw_chatbook.app.get_cli_setting') as mock_settings:
        def settings_side_effect(*args, **kwargs):
            # Handle get_cli_setting(section, key, default) format
            if len(args) >= 2:
                section = args[0]
                key = args[1]
                default = args[2] if len(args) > 2 else kwargs.get('default', None)
                
                # Check section + key combination
                if section == "general" and key == "use_link_navigation":
                    return True
                elif section == "general" and key == "use_dropdown_navigation":
                    return False
                elif section == "general" and key == "default_tab":
                    return TAB_CHAT
                elif section == "splash_screen" and key == "enabled":
                    return False  # Disable splash screen
            elif len(args) == 1:
                # Single arg format
                key = args[0]
                default = kwargs.get('default', None)
                if key == "splash_screen":
                    return False
                    
            return default
        
        mock_settings.side_effect = settings_side_effect
        
        app = TldwCli()
        
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.5)  # Let app initialize
            
            # Check what's actually present
            print("Checking for TabLinks...")
            tab_links = app.query("TabLinks")
            print(f"Found {len(tab_links)} TabLinks widgets")
            
            if tab_links:
                tab_links_widget = tab_links.first()
                print(f"TabLinks widget: {tab_links_widget}")
                
                # Check for tab links
                links = app.query(".tab-link")
                print(f"Found {len(links)} tab links")
                for link in links:
                    print(f"  - {link.id}: {link.renderable}")
                
                # Check for separators
                separators = app.query(".tab-separator")
                print(f"Found {len(separators)} separators")
            else:
                # Check what navigation is actually being used
                print("\nChecking actual navigation...")
                tab_bar = app.query("TabBar")
                print(f"Found {len(tab_bar)} TabBar widgets")
                
                tab_dropdown = app.query("TabDropdown")
                print(f"Found {len(tab_dropdown)} TabDropdown widgets")
                
                # Check the header
                header = app.query("Header")
                print(f"Found {len(header)} Header widgets")
                if header:
                    print(f"Header children: {list(header.first().children)}")

if __name__ == "__main__":
    asyncio.run(test_tab_links())