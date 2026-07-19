from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header,
    Footer,
    Button,
    Static,
    Label,
    Input,
    Checkbox,
    RadioButton,
    Switch,
    DataTable,
    Markdown,
    Pretty,
    Tree,
)
# textual.theme.Theme is used in themes.py
# No need to import Color or ColorSystem here for basic theme switching.

from themes import ALL_THEMES  # This now imports a dictionary of {name: ThemeObject}


class ThemeDemoApp(App):
    TITLE = "Textual Theme Demo"
    CSS_PATH = (
        "theme_tester.tcss"  # Ensure this file exists, can be empty or for layout
    )

    BINDINGS = [
        ("t", "next_theme", "Next Theme"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.custom_theme_names = list(MY_THEMES.keys())
        self.custom_theme_names = [theme.name for theme in ALL_THEMES]
        # Textual's built-in themes that can be set via self.theme directly
        self.builtin_theme_names = ["dark", "light"]
        # "monokai" might require specific CSS or registration if not a standard built-in name.
        # We'll include it and let the try-except catch if it's not found by default.
        self.available_theme_names = (
            self.custom_theme_names + self.builtin_theme_names + ["monokai"]
        )

        self.current_theme_index = 0
        if self.available_theme_names:
            # Start with the first custom theme if available, else 'dark'
            self.initial_theme_name = self.available_theme_names[0]
        else:
            self.initial_theme_name = "dark"  # Fallback

        self.theme_status_label = Label()  # For displaying the current theme

    def on_mount(self) -> None:
        """Register custom themes and set the initial theme."""
        # Register all custom themes from MY_THEMES
        # for theme_name, theme_object in MY_THEMES.items():
        for theme_object in ALL_THEMES:
            try:
                theme_name = theme_object.name
                self.register_theme(theme_object)
                # print(f"Successfully registered theme: {theme_name}")
            except Exception as e:
                print(f"Error registering theme '{theme_name}': {e}")
                self.notify(
                    f"Error registering theme '{theme_name}': {e}", severity="error"
                )

        # Set the initial theme
        self.apply_theme(self.initial_theme_name, is_initial=True)

    def apply_theme(self, theme_name: str, is_initial: bool = False) -> None:
        """Applies the specified theme by name."""
        try:
            self.theme = theme_name  # This is the Textual way to set a theme
            message = (
                f"{'Initial theme' if is_initial else 'Theme changed to'}: {theme_name}"
            )
            self.update_theme_status_label(message)
            self.notify(message)
        except Exception as e:
            error_message = f"Failed to apply theme '{theme_name}': {e}"
            print(error_message)
            self.notify(error_message, severity="error")
            # If initial theme fails, try a fallback
            if is_initial and theme_name != "dark":
                self.notify(
                    "Attempting to apply fallback 'dark' theme.", severity="warning"
                )
                self.apply_theme(
                    "dark", is_initial=True
                )  # Try to apply 'dark' as a last resort
            elif is_initial:  # Failed to apply 'dark' as initial
                self.notify(
                    "CRITICAL: Failed to apply any initial theme.", severity="error"
                )

    def action_next_theme(self) -> None:
        """Cycle to the next available theme."""
        if not self.available_theme_names:
            self.notify("No themes available to cycle.", severity="warning")
            return

        self.current_theme_index = (self.current_theme_index + 1) % len(
            self.available_theme_names
        )
        theme_to_apply = self.available_theme_names[self.current_theme_index]
        self.apply_theme(theme_to_apply)

    def update_theme_status_label(self, text: str) -> None:
        """Updates the text of the theme status label."""
        self.theme_status_label.update(text)

    def compose(self) -> ComposeResult:
        yield Header()
        with ScrollableContainer(id="main_container"):
            # The label will be updated in on_mount and action_next_theme
            yield self.theme_status_label

            # --- UI Elements to test themes ---
            with Horizontal(classes="row"):
                with Vertical(classes="column"):
                    yield Static("Buttons:", classes="group_label")
                    yield Button("Default Button")
                    yield Button("Primary Button", variant="primary")
                    yield Button("Success Button", variant="success")
                    yield Button("Warning Button", variant="warning")
                    yield Button("Danger Button", variant="error")
                    yield Button("Disabled Button", disabled=True)

                with Vertical(classes="column"):
                    yield Static("Inputs & Selection:", classes="group_label")
                    yield Input(placeholder="Enter text here...")
                    yield Input("Disabled Input", disabled=True)
                    yield Checkbox("A checkbox option")
                    yield Checkbox("Checked by default", value=True)
                    yield Switch()
                    yield Switch(value=True)
                    with Vertical():
                        yield Static("Radio Buttons:", classes="sub_group_label")
                        yield RadioButton("Option A")
                        yield RadioButton("Option B")

            with Horizontal(classes="row"):
                with Vertical(classes="column"):
                    yield Static("Static & Pretty:", classes="group_label")
                    yield Static("This is a [b]Static[/b] widget with [i]markup[/i].")
                    yield Pretty(
                        {"name": "Textual", "version": "Latest", "status": "Theming!"}
                    )

                with Vertical(classes="column"):
                    yield Static("DataTable:", classes="group_label")
                    dt = DataTable()
                    dt.add_columns("ID", "Name", "Score")
                    dt.add_rows([(1, "Alice", 85), (2, "Bob", 92), (3, "Charlie", 78)])
                    yield dt

            with Horizontal(classes="row"):
                with Vertical(classes="column"):
                    yield Static("Markdown:", classes="group_label")
                    md_content = """
# Markdown Header
This is some *italic* and **bold** text.
- Item 1
- Item 2
  - Sub-item
[A Link](https://textualize.io)
                    """
                    yield Markdown(md_content)
                with Vertical(classes="column", id="tree-column"):
                    yield Static("Tree Control:", classes="group_label")
                    tree = Tree("Root")
                    node1 = tree.root.add("Node 1")
                    node1.add_leaf("Leaf 1.1")
                    node1.add_leaf("Leaf 1.2")
                    tree.root.add("Node 2").add_leaf("Leaf 2.1")
                    yield tree
        yield Footer()


if __name__ == "__main__":
    app = ThemeDemoApp()
    app.run()
