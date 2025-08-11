#!/usr/bin/env python3
"""Check all tab link positions."""

import asyncio
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Constants import ALL_TABS

async def check_all():
    """Check all tab link positions."""
    app = TldwCli()
    
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(7)
        
        print(f"\nScreen width: {app.size.width}")
        print("\n=== All Tab Link Positions ===")
        
        for tab_id in ALL_TABS:
            try:
                link = app.query_one(f"#tab-link-{tab_id}")
                status = "✅ OK" if (link.region.x + link.region.width) <= app.size.width else "❌ OUTSIDE"
                print(f"{tab_id:30} x={link.region.x:3} width={link.region.width:3} end={link.region.x + link.region.width:3} {status}")
            except Exception as e:
                print(f"{tab_id:30} Error: {e}")

if __name__ == "__main__":
    asyncio.run(check_all())