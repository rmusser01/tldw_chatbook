"""Test simple button click in Textual."""

import asyncio
from textual.app import App
from textual.widgets import Button
from textual import on

class TestApp(App):
    def compose(self):
        yield Button("Test Button", id="test-btn")
    
    @on(Button.Pressed, "#test-btn")
    def handle_button(self):
        print("Button was pressed!")
        self.exit()

async def main():
    app = TestApp()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        
        button = pilot.app.query_one("#test-btn")
        print(f"Found button: {button}")
        
        print("Clicking button...")
        await pilot.click(button)
        await pilot.pause(0.2)
        
        print("Test complete")

if __name__ == "__main__":
    asyncio.run(main())