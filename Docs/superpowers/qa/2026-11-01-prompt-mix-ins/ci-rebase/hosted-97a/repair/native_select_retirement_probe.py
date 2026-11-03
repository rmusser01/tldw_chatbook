import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))
import Tests.UI.conftest
from textual.app import App
from textual.events import Unmount
from textual.widgets import Select
from Tests.UI.test_console_archive_recovery_cards import confirmation
from tldw_chatbook.Widgets.Chat_Widgets.archive_recovery_cards import ArchiveRecoveryConfirmCard

rows = []
first = replace(confirmation(), selected_scope='chat_only')
latest = replace(first, request_id='replacement')
card = ArchiveRecoveryConfirmCard(first)

class ProbeApp(App):
    def compose(self):
        yield card

def observe(message):
    sender = message._sender
    if isinstance(message, Unmount) and isinstance(sender, Select):
        row = {'phase': 'scope_unmount', 'scope_mounted': sender.is_mounted, 'scope_children': len(sender.children), 'card_children': len(card.children)}
        try:
            row['result'] = card.set_confirmation(latest)
        except Exception as error:
            row.update(error=type(error).__name__, message=str(error))
        rows.append(row)

async def main():
    app = ProbeApp()
    async with app.run_test(message_hook=observe):
        await card.query_one('#recovery-scope').remove()
    print(json.dumps(rows, indent=2))

asyncio.run(main())
