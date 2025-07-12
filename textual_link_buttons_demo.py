#!/usr/bin/env python3
"""
Textual Link-Like Buttons Demo
Demonstrates different ways to create hyperlink-style buttons in Textual
"""

import webbrowser
from textual.app import App, ComposeResult
from textual.containers import Vertical, Horizontal, Container
from textual.widgets import Button, Label, Static, Header, Footer
from textual.widget import Widget
from textual.reactive import reactive
from textual.message import Message
from textual import events
from rich.text import Text


class LinkButton(Button):
    """A button styled to look like a hyperlink"""
    
    DEFAULT_CSS = """
    LinkButton {
        background: transparent;
        border: none;
        padding: 0 1;
        height: 1;
        min-width: 1;
        color: $primary;
        text-style: none;
    }
    
    LinkButton:hover {
        background: transparent;
        color: $primary-lighten-2;
        text-style: underline;
    }
    
    LinkButton:focus {
        text-style: underline;
        background: transparent;
    }
    
    LinkButton.-active {
        background: transparent;
        color: $primary-darken-1;
        text-style: underline;
    }
    """


class ClickableLabel(Label):
    """A label that acts like a clickable link"""
    
    can_focus = True
    
    class Clicked(Message):
        """Message sent when label is clicked"""
        def __init__(self, label: "ClickableLabel") -> None:
            self.label = label
            super().__init__()
    
    DEFAULT_CSS = """
    ClickableLabel {
        color: $primary;
        width: auto;
        height: 1;
    }
    
    ClickableLabel:hover {
        color: $primary-lighten-2;
        text-style: underline;
    }
    
    ClickableLabel:focus {
        text-style: underline bold;
    }
    """
    
    def on_click(self) -> None:
        """Handle click events"""
        self.post_message(self.Clicked(self))


class HyperlinkWidget(Widget):
    """Custom widget that behaves like a hyperlink"""
    
    DEFAULT_CSS = """
    HyperlinkWidget {
        height: 1;
        width: auto;
        color: $primary;
        background: transparent;
    }
    
    HyperlinkWidget:hover {
        color: $primary-lighten-2;
        text-style: underline;
    }
    
    HyperlinkWidget:focus {
        text-style: underline bold;
        background: $surface;
    }
    
    HyperlinkWidget.-clicked {
        color: $primary-darken-2;
    }
    """
    
    text = reactive("")
    url = reactive("")
    clicked = reactive(False)
    
    def __init__(self, text: str = "", url: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.text = text
        self.url = url
        self.can_focus = True
    
    def render(self) -> Text:
        """Render the hyperlink text"""
        return Text(self.text)
    
    def on_click(self) -> None:
        """Handle click events"""
        self.clicked = True
        if self.url:
            webbrowser.open(self.url)
        self.set_timer(0.2, lambda: setattr(self, 'clicked', False))
    
    def on_key(self, event: events.Key) -> None:
        """Handle keyboard events"""
        if event.key in ("enter", "space"):
            self.on_click()
    
    def watch_clicked(self, clicked: bool) -> None:
        """React to clicked state changes"""
        if clicked:
            self.add_class("-clicked")
        else:
            self.remove_class("-clicked")


class LinkButtonsDemo(App):
    """Demonstration app for link-like buttons in Textual"""
    
    CSS = """
    Screen {
        align: center middle;
        background: $surface;
    }
    
    Container {
        width: 80;
        height: auto;
        padding: 2;
        background: $panel;
        border: solid $primary;
    }
    
    .demo-section {
        margin: 1 0;
        padding: 1;
        border: dashed $surface-lighten-2;
    }
    
    .section-title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }
    
    .demo-item {
        margin: 1 0;
        height: auto;
    }
    
    .description {
        color: $text-muted;
        margin-left: 2;
    }
    
    /* Additional styling for URL buttons */
    .url-button {
        color: $primary;
        background: transparent;
        border: none;
        padding: 0 1;
        height: 1;
    }
    
    .url-button:hover {
        color: $primary-lighten-2;
        text-style: underline;
        background: transparent;
    }
    
    .url-button:focus {
        text-style: underline;
        background: $surface;
    }
    
    #status {
        dock: bottom;
        height: 3;
        background: $panel-darken-1;
        padding: 1;
        color: $text-muted;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        
        with Container():
            yield Static("Link-Like Buttons in Textual", classes="section-title")
            
            # Section 1: Styled LinkButton
            with Vertical(classes="demo-section"):
                yield Static("1. Styled Button (LinkButton class)", classes="section-title")
                with Horizontal(classes="demo-item"):
                    yield LinkButton("Click me - I look like a link!", id="link-button-1")
                yield Static("A button with custom CSS to look like a hyperlink", classes="description")
            
            # Section 2: Button that opens URL
            with Vertical(classes="demo-section"):
                yield Static("2. URL Opening Button", classes="section-title")
                with Horizontal(classes="demo-item"):
                    yield Button("Open Textual Documentation", id="url-button", classes="url-button")
                yield Static("Opens https://textual.textualize.io in your browser", classes="description")
            
            # Section 3: Clickable Label
            with Vertical(classes="demo-section"):
                yield Static("3. Clickable Label", classes="section-title")
                with Horizontal(classes="demo-item"):
                    yield ClickableLabel("I'm a label that acts like a link", id="clickable-label")
                yield Static("A Label widget with click handling and link styling", classes="description")
            
            # Section 4: Custom Hyperlink Widget
            with Vertical(classes="demo-section"):
                yield Static("4. Custom HyperlinkWidget", classes="section-title")
                with Horizontal(classes="demo-item"):
                    yield HyperlinkWidget(
                        "Visit Python.org", 
                        url="https://python.org",
                        id="hyperlink-widget"
                    )
                yield Static("Custom widget that opens URLs and has full link behavior", classes="description")
            
            # Additional examples
            with Vertical(classes="demo-section"):
                yield Static("More Examples", classes="section-title")
                with Vertical(classes="demo-item"):
                    yield HyperlinkWidget("Textual GitHub", url="https://github.com/Textualize/textual")
                    yield LinkButton("Internal action link", id="internal-link")
                    yield ClickableLabel("Another clickable label", id="another-label")
        
        yield Static("Status: Ready", id="status")
        yield Footer()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events"""
        status = self.query_one("#status", Static)
        
        if event.button.id == "url-button":
            webbrowser.open("https://textual.textualize.io")
            status.update("Opened Textual documentation in browser")
        elif event.button.id == "link-button-1":
            status.update("LinkButton clicked!")
        elif event.button.id == "internal-link":
            status.update("Internal action triggered")
    
    def on_clickable_label_clicked(self, event: ClickableLabel.Clicked) -> None:
        """Handle clickable label events"""
        status = self.query_one("#status", Static)
        if event.label.id == "clickable-label":
            status.update("Clickable label was clicked!")
        elif event.label.id == "another-label":
            status.update("Another label clicked!")
    
    def on_hyperlink_widget_click(self, event: events.Click) -> None:
        """Handle hyperlink widget clicks"""
        if isinstance(event.widget, HyperlinkWidget):
            status = self.query_one("#status", Static)
            if event.widget.url:
                status.update(f"Opening {event.widget.url}")
            else:
                status.update(f"Hyperlink '{event.widget.text}' clicked")


if __name__ == "__main__":
    app = LinkButtonsDemo()
    app.run()