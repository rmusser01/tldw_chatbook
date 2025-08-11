#!/usr/bin/env python3
"""Debug why tab links are outside visible region."""

import asyncio
from tldw_chatbook.app import TldwCli
from tldw_chatbook.UI.Tab_Links import TabLinks

async def debug_positions():
    """Debug tab link positions."""
    app = TldwCli()
    
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(7)  # Wait for app to be ready
        
        print("\n=== Screen Info ===")
        print(f"App size: {app.size}")
        print(f"Screen size: {app.screen.size}")
        
        # Get TabLinks container
        tab_links = app.query_one(TabLinks)
        print(f"\n=== TabLinks Container ===")
        print(f"Region: {tab_links.region}")
        print(f"Size: {tab_links.size}")
        print(f"Offset: {tab_links.offset}")
        print(f"Visible: {tab_links.visible}")
        print(f"Display: {tab_links.display}")
        
        # Check each tab link
        print(f"\n=== Individual Tab Links ===")
        for tab_id in ['chat', 'conversations_characters_prompts', 'notes', 'media']:
            try:
                link = app.query_one(f"#tab-link-{tab_id}")
                print(f"\n{tab_id}:")
                print(f"  Region: {link.region}")
                print(f"  Visible: {link.visible}")
                print(f"  Display: {link.display}")
                
                # Check if it's within screen bounds
                if link.region.x >= app.size.width:
                    print(f"  ⚠️ OUTSIDE SCREEN: x={link.region.x} >= width={app.size.width}")
                elif link.region.x + link.region.width > app.size.width:
                    print(f"  ⚠️ PARTIALLY OUTSIDE: ends at {link.region.x + link.region.width} > width={app.size.width}")
            except Exception as e:
                print(f"  Error: {e}")

if __name__ == "__main__":
    asyncio.run(debug_positions())