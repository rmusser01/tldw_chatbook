"""
Entry point for running the MCP server directly.

Usage:
    python -m tldw_chatbook.MCP
"""

import asyncio
import sys

if __name__ == "__main__":
    try:
        from tldw_chatbook.MCP.server import main

        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("MCP server interrupted.", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception:
        print("MCP server failed.", file=sys.stderr)
        raise SystemExit(1) from None
