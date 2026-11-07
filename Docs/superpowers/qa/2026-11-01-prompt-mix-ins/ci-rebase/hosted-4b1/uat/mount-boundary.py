import asyncio
import json
from textual.app import App
from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard

class Harness(App):
    def compose(self):
        yield ChatApprovalCard(id="chat-approval-card")

async def main():
    app = Harness()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch([{ "llm_name": "list_directory", "display_name": "list_directory", "server_name": "Built-in", "arguments": {"path": "."}}], timeout_seconds=0, round_id="readiness-probe")
        rows = list(card.query(".approval-row"))
        before = {"rows": len(rows), "headers": len(card.query(".approval-row-header")), "row_mounted": rows[0].is_mounted if rows else None}
        await pilot.pause()
        after = {"rows": len(card.query(".approval-row")), "headers": len(card.query(".approval-row-header")), "header": [str(x.renderable) for x in card.query(".approval-row-header")]}
        print(json.dumps({"before": before, "after": after}, indent=2))
        assert before["rows"] == 1 and before["headers"] == 0
        assert after["headers"] == 1

asyncio.run(main())
