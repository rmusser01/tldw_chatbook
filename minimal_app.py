from textual.app import App, ComposeResult
from textual.widgets import Label

class MinimalApp(App):
    def compose(self) -> ComposeResult:
        yield Label("Hello, World!")

app = MinimalApp()
