#!/usr/bin/env python3
"""
Simple Tamagotchi Example

Demonstrates basic usage of the Tamagotchi widget in a Textual app.
Run this file directly to see a working tamagotchi.
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Button, Static, Label
from textual.binding import Binding
from textual import on

# Import from parent directory
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from Tamagotchi import Tamagotchi, CompactTamagotchi, TamagotchiInteraction, TamagotchiDeath


class SimpleTamagotchiApp(App):
    """Simple app demonstrating tamagotchi functionality."""
    
    CSS = """
    #main-container {
        align: center middle;
        padding: 1;
    }
    
    #pet-container {
        height: 7;
        width: 40;
        border: round $primary;
        padding: 1;
        margin: 1;
    }
    
    #actions {
        height: 3;
        align: center middle;
        margin: 1;
    }
    
    #actions Button {
        margin: 0 1;
    }
    
    #status {
        height: 1;
        text-align: center;
        color: $text-muted;
    }
    
    #footer-pet {
        dock: bottom;
        height: 1;
        background: $surface;
        padding: 0 1;
    }
    """
    
    BINDINGS = [
        Binding("f", "feed", "Feed"),
        Binding("p", "play", "Play"),
        Binding("s", "sleep", "Sleep"),
        Binding("c", "clean", "Clean"),
        Binding("q", "quit", "Quit"),
    ]
    
    def compose(self) -> ComposeResult:
        """Build the app layout."""
        yield Header(show_clock=True)
        
        with Container(id="main-container"):
            yield Label("🎮 Tamagotchi Demo", id="title")
            
            # Main pet display
            with Container(id="pet-container"):
                yield Tamagotchi(
                    name="Pixel",
                    personality="balanced",
                    update_interval=10,  # Fast updates for demo
                    id="main-pet"
                )
            
            # Action buttons
            with Horizontal(id="actions"):
                yield Button("Feed 🍽️", id="feed-btn", variant="primary")
                yield Button("Play 🎮", id="play-btn", variant="success")
                yield Button("Sleep 😴", id="sleep-btn", variant="warning")
                yield Button("Clean 🧼", id="clean-btn")
            
            # Status display
            yield Static("Click buttons or use keyboard shortcuts", id="status")
        
        # Footer with compact tamagotchi
        with Horizontal(id="footer-pet"):
            yield Static("Footer Pet: ")
            yield CompactTamagotchi(
                name="Bit",
                personality="energetic",
                update_interval=15,
                id="footer-tamagotchi"
            )
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Set up the app when mounted."""
        self.title = "Tamagotchi Demo"
        self.sub_title = "Keep your pet happy!"
    
    @on(Button.Pressed, "#feed-btn")
    def feed_pet(self) -> None:
        """Feed button pressed."""
        pet = self.query_one("#main-pet", Tamagotchi)
        pet.interact("feed")
    
    @on(Button.Pressed, "#play-btn")
    def play_with_pet(self) -> None:
        """Play button pressed."""
        pet = self.query_one("#main-pet", Tamagotchi)
        pet.interact("play")
    
    @on(Button.Pressed, "#sleep-btn")
    def put_pet_to_sleep(self) -> None:
        """Sleep button pressed."""
        pet = self.query_one("#main-pet", Tamagotchi)
        pet.interact("sleep")
    
    @on(Button.Pressed, "#clean-btn")
    def clean_pet(self) -> None:
        """Clean button pressed."""
        pet = self.query_one("#main-pet", Tamagotchi)
        pet.interact("clean")
    
    def action_feed(self) -> None:
        """Keyboard shortcut: feed."""
        pet = self.query_one("#main-pet", Tamagotchi)
        pet.interact("feed")
    
    def action_play(self) -> None:
        """Keyboard shortcut: play."""
        pet = self.query_one("#main-pet", Tamagotchi)
        pet.interact("play")
    
    def action_sleep(self) -> None:
        """Keyboard shortcut: sleep."""
        pet = self.query_one("#main-pet", Tamagotchi)
        pet.interact("sleep")
    
    def action_clean(self) -> None:
        """Keyboard shortcut: clean."""
        pet = self.query_one("#main-pet", Tamagotchi)
        pet.interact("clean")
    
    def on_tamagotchi_interaction(self, event: TamagotchiInteraction) -> None:
        """Handle tamagotchi interaction events."""
        status = self.query_one("#status", Static)
        
        if event.success:
            message = f"{event.pet_name}: {event.message}"
            # Show stat changes
            changes = []
            for stat, change in event.changes.items():
                if change != 0:
                    symbol = "+" if change > 0 else ""
                    changes.append(f"{stat} {symbol}{change:.0f}")
            if changes:
                message += f" ({', '.join(changes)})"
        else:
            message = f"{event.pet_name}: {event.message}"
        
        status.update(message)
        
        # Also show notification
        self.notify(message, timeout=2)
    
    def on_tamagotchi_death(self, event: TamagotchiDeath) -> None:
        """Handle tamagotchi death."""
        message = f"😢 {event.pet_name} has died! Cause: {event.cause}, Age: {event.age:.1f} hours"
        
        status = self.query_one("#status", Static)
        status.update(message)
        
        self.notify(message, severity="error", timeout=5)


def main():
    """Run the demo app."""
    app = SimpleTamagotchiApp()
    app.run()


if __name__ == "__main__":
    main()