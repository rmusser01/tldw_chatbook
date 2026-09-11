#!/usr/bin/env python3
"""Regenerate Docs/User_Guide/images/console/approval-card.svg (task-32290).

Mounts the real ``ChatApprovalCard`` with the real app stylesheets and one
pending MCP row, then exports the screenshot Rich/Textual produces.

Run it against the worktree holding the card you want pictured::

    PYTHONPATH=<worktree> .venv/bin/python scripts/regen_approval_card_svg.py \
        Docs/User_Guide/images/console/approval-card.svg
"""

from __future__ import annotations

import sys
from pathlib import Path

from textual.app import ComposeResult

from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard

CALL = {
    "llm_name": "search_notes",
    "tool_name": "search_notes",
    "server_key": "mcp:tldw_chatbook",
    "server_label": "tldw_chatbook",
    "arguments": {"query": "demo"},
    "call_id": "call_1",
    "reason": "ask",
}


class _CardApp(ConsolidatedCSSApp):
    TITLE = "tldw chatbook"
    CSS_PATH = [str(path) for path in APP_STYLESHEETS]

    def compose(self) -> ComposeResult:
        yield ChatApprovalCard(id="chat-approval-card")



def main() -> int:
    out = Path(sys.argv[1]).resolve()

    async def _run() -> None:
        app = _CardApp()
        async with app.run_test(size=(120, 20)) as pilot:
            card = app.query_one(ChatApprovalCard)
            card.display = True
            card.set_batch([CALL], timeout_seconds=120, round_id="round-1")
            await pilot.pause()
            await pilot.pause()
            app.save_screenshot(str(out))

    import asyncio

    asyncio.run(_run())
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
