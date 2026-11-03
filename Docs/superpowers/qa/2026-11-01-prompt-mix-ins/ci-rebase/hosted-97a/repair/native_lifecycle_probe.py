import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))
import Tests.UI.conftest  # Apply the existing private bootstrap before imports.
from textual.app import App
from textual.containers import Vertical
from tldw_chatbook.UI.Console_Modules.archive_recovery import ConsoleArchiveRecoveryView
from tldw_chatbook.Widgets.Chat_Widgets.archive_recovery_cards import ArchiveRecoveryConfirmCard

observations = []

class ProbeCard(ArchiveRecoveryConfirmCard):
    def compose(self):
        try:
            self.set_confirmation(None)
        except Exception as error:
            observations.append({'phase': 'compose', 'is_mounted': self.is_mounted, 'children': len(self.children), 'error': type(error).__name__, 'message': str(error)})
        yield from super().compose()

    def on_unmount(self):
        row = {'phase': 'unmount', 'is_mounted': self.is_mounted, 'children': len(self.children), 'queryable': self.app.view._confirmation() is self}
        try:
            row['result'] = self.app.view.show_confirmation(None)
        except Exception as error:
            row.update(error=type(error).__name__, message=str(error))
        observations.append(row)

class ProbeApp(App):
    def compose(self):
        self.view = ConsoleArchiveRecoveryView(self.screen)
        with Vertical(id='console-archive-recovery-controls'):
            yield ProbeCard()

async def main():
    app = ProbeApp()
    async with app.run_test():
        await app.query_one(ProbeCard).remove()
    print(json.dumps(observations, indent=2))

asyncio.run(main())
