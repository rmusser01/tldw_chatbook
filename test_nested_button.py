"""Test button inside nested containers."""

import asyncio
from textual.app import App
from textual.widgets import Button
from textual.containers import Container, Horizontal
from textual import on

class TestWidget(Container):
    def compose(self):
        with Horizontal(classes="message-header"):
            with Container(classes="message-actions"):
                yield Button("Copy", id="copy-btn", classes="action-button")
    
    @on(Button.Pressed, "#copy-btn")
    def handle_copy(self):
        print("Copy button pressed in widget!")

class TestApp(App):
    def compose(self):
        yield TestWidget()

async def main():
    app = TestApp()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        
        widget = pilot.app.query_one(TestWidget)
        button = widget.query_one("#copy-btn")
        print(f"Found button: {button}")
        
        print("Clicking button...")
        await pilot.click(button)
        await pilot.pause(0.2)
        
        print("Test complete")

if __name__ == "__main__":
    asyncio.run(main())