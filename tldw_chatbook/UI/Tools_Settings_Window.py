# tldw_chatbook/UI/Tools_Settings_Window.py
#
#
# Imports
from typing import TYPE_CHECKING
#
# 3rd-Party Imports
import toml
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal
from textual.widgets import Static, Button, TextArea, Label, Input, Select, Checkbox
# Local Imports
from tldw_chatbook.config import load_cli_config_and_ensure_existence, DEFAULT_CONFIG_PATH
#
# Local Imports
#
if TYPE_CHECKING:
    from ..app import TldwCli
#
#######################################################################################################################
#
# Functions:

class ToolsSettingsWindow(Container):
    """
    Container for the Tools & Settings Tab's UI.
    """
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.config_data = load_cli_config_and_ensure_existence()

    def _compose_core_settings(self) -> ComposeResult:
        """Compose the Core Settings UI with RAG settings and other core configurations."""
        with VerticalScroll(classes="core-settings-container"):
            yield Static("Core Settings", classes="section-title")
            
            # RAG Settings Section
            yield Static("RAG Search Settings", classes="subsection-title")
            
            with Container(classes="settings-group"):
                # Default Search Mode
                yield Label("Default Search Mode:")
                search_modes = [("Q&A Mode", "qa"), ("Chat Mode", "chat")]
                default_mode = self.config_data.get("rag_search", {}).get("default_mode", "qa")
                yield Select(
                    options=search_modes,
                    value=default_mode,
                    id="core-rag-search-mode",
                    classes="settings-select"
                )
                
                # Default Sources
                yield Label("Default Search Sources:", classes="settings-label")
                with Horizontal(classes="checkbox-group"):
                    sources_config = self.config_data.get("rag_search", {}).get("default_sources", {})
                    yield Checkbox(
                        "Media Files",
                        value=sources_config.get("media", True),
                        id="core-rag-source-media"
                    )
                    yield Checkbox(
                        "Conversations",
                        value=sources_config.get("conversations", True),
                        id="core-rag-source-conversations"
                    )
                    yield Checkbox(
                        "Notes",
                        value=sources_config.get("notes", True),
                        id="core-rag-source-notes"
                    )
                
                # Default Top-K
                yield Label("Default Top-K Results:", classes="settings-label")
                default_top_k = str(self.config_data.get("rag_search", {}).get("default_top_k", 10))
                yield Input(
                    value=default_top_k,
                    placeholder="10",
                    id="core-rag-top-k",
                    classes="settings-input"
                )
                
                # Re-ranking Settings
                yield Label("Re-ranking Settings:", classes="settings-label")
                rerank_config = self.config_data.get("rag_search", {}).get("reranking", {})
                yield Checkbox(
                    "Enable Re-ranking by Default",
                    value=rerank_config.get("enabled", False),
                    id="core-rag-rerank-enabled"
                )
                
                yield Label("Re-ranker Model:", classes="settings-label")
                yield Input(
                    value=rerank_config.get("model", "cross-encoder/ms-marco-MiniLM-L-12-v2"),
                    placeholder="cross-encoder/ms-marco-MiniLM-L-12-v2",
                    id="core-rag-rerank-model",
                    classes="settings-input"
                )
            
            # Chunking Settings Section
            yield Static("Chunking Settings", classes="subsection-title")
            
            with Container(classes="settings-group"):
                chunking_config = self.config_data.get("rag_search", {}).get("chunking", {})
                
                yield Label("Default Chunk Size:", classes="settings-label")
                yield Input(
                    value=str(chunking_config.get("size", 512)),
                    placeholder="512",
                    id="core-chunk-size",
                    classes="settings-input"
                )
                
                yield Label("Default Chunk Overlap:", classes="settings-label")
                yield Input(
                    value=str(chunking_config.get("overlap", 128)),
                    placeholder="128",
                    id="core-chunk-overlap",
                    classes="settings-input"
                )
                
                yield Label("Chunking Method:", classes="settings-label")
                chunking_methods = [
                    ("Fixed Size", "fixed"),
                    ("Semantic", "semantic"),
                    ("Sentence-based", "sentence")
                ]
                yield Select(
                    options=chunking_methods,
                    value=chunking_config.get("method", "fixed"),
                    id="core-chunk-method",
                    classes="settings-select"
                )
            
            # Memory Management Settings
            yield Static("Memory Management", classes="subsection-title")
            
            with Container(classes="settings-group"):
                memory_config = self.config_data.get("rag_search", {}).get("memory", {})
                
                yield Label("Max Memory Usage (MB):", classes="settings-label")
                yield Input(
                    value=str(memory_config.get("max_memory_mb", 1024)),
                    placeholder="1024",
                    id="core-memory-max",
                    classes="settings-input"
                )
                
                yield Checkbox(
                    "Enable Memory Monitoring",
                    value=memory_config.get("monitoring_enabled", True),
                    id="core-memory-monitoring"
                )
            
            # Save Button
            with Container(classes="settings-actions"):
                yield Button("Save Core Settings", id="save-core-settings", variant="primary")
                yield Button("Reset to Defaults", id="reset-core-settings", variant="default")

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="tools-settings-nav-pane", classes="tools-nav-pane"):
            yield Static("Navigation", classes="sidebar-title")
            yield Button("General Settings", id="ts-nav-general-settings", classes="ts-nav-button")
            yield Button("Configuration File Settings", id="ts-nav-config-file-settings", classes="ts-nav-button")
            yield Button("Database Tools", id="ts-nav-db-tools", classes="ts-nav-button")
            yield Button("Appearance", id="ts-nav-appearance", classes="ts-nav-button")

        with Container(id="tools-settings-content-pane", classes="tools-content-pane"):
            yield Container(
                *self._compose_core_settings(),
                id="ts-view-general-settings",
                classes="ts-view-area",
            )
            yield Container(
                TextArea(
                    text=toml.dumps(load_cli_config_and_ensure_existence()),
                    language="toml",
                    read_only=False, # Made editable
                    id="config-text-area"
                ),
                Container(
                    Button("Save", id="save-config-button", variant="primary"),
                    Button("Reload", id="reload-config-button"),
                    classes="config-button-container"
                ),
                id="ts-view-config-file-settings",
                classes="ts-view-area",
            )
            yield Container(
                Static("Database Tools Area - Content Coming Soon!"),
                id="ts-view-db-tools",
                classes="ts-view-area",
            )
            yield Container(
                Static("Appearance Settings Area - Content Coming Soon!"),
                id="ts-view-appearance",
                classes="ts-view-area",
            )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Event handler called when a button is pressed."""
        button_id = event.button.id

        if button_id == "save-core-settings":
            await self._save_core_settings()
        elif button_id == "reset-core-settings":
            await self._reset_core_settings()
        elif button_id == "save-config-button":
            try:
                config_text_area = self.query_one("#config-text-area", TextArea)
                config_data = toml.loads(config_text_area.text)
                with open(DEFAULT_CONFIG_PATH, "w") as f:
                    toml.dump(config_data, f)
                self.app_instance.notify("Configuration saved successfully.")
            except toml.TOMLDecodeError:
                self.app_instance.notify("Error: Invalid TOML format.", severity="error")
            except IOError:
                self.app_instance.notify("Error: Could not write to configuration file.", severity="error")

        elif button_id == "reload-config-button":
            try:
                config_text_area = self.query_one("#config-text-area", TextArea)
                config_data = load_cli_config_and_ensure_existence(force_reload=True)
                config_text_area.text = toml.dumps(config_data)
                self.app_instance.notify("Configuration reloaded.")
            except Exception as e:
                self.app_instance.notify(f"Error reloading configuration: {e}", severity="error")

    async def _save_core_settings(self) -> None:
        """Save Core Settings to the configuration file."""
        try:
            # Get current config
            config_data = load_cli_config_and_ensure_existence()
            
            # Ensure rag_search section exists
            if "rag_search" not in config_data:
                config_data["rag_search"] = {}
            
            # Get values from UI
            search_mode = self.query_one("#core-rag-search-mode", Select).value
            
            # Update search mode
            config_data["rag_search"]["default_mode"] = search_mode
            
            # Update default sources
            if "default_sources" not in config_data["rag_search"]:
                config_data["rag_search"]["default_sources"] = {}
            
            config_data["rag_search"]["default_sources"]["media"] = self.query_one("#core-rag-source-media", Checkbox).value
            config_data["rag_search"]["default_sources"]["conversations"] = self.query_one("#core-rag-source-conversations", Checkbox).value
            config_data["rag_search"]["default_sources"]["notes"] = self.query_one("#core-rag-source-notes", Checkbox).value
            
            # Update top-k
            try:
                top_k = int(self.query_one("#core-rag-top-k", Input).value)
                config_data["rag_search"]["default_top_k"] = top_k
            except ValueError:
                self.app_instance.notify("Invalid Top-K value. Using default.", severity="warning")
                config_data["rag_search"]["default_top_k"] = 10
            
            # Update re-ranking settings
            if "reranking" not in config_data["rag_search"]:
                config_data["rag_search"]["reranking"] = {}
            
            config_data["rag_search"]["reranking"]["enabled"] = self.query_one("#core-rag-rerank-enabled", Checkbox).value
            config_data["rag_search"]["reranking"]["model"] = self.query_one("#core-rag-rerank-model", Input).value
            
            # Update chunking settings
            if "chunking" not in config_data["rag_search"]:
                config_data["rag_search"]["chunking"] = {}
            
            try:
                chunk_size = int(self.query_one("#core-chunk-size", Input).value)
                config_data["rag_search"]["chunking"]["size"] = chunk_size
            except ValueError:
                self.app_instance.notify("Invalid chunk size. Using default.", severity="warning")
                config_data["rag_search"]["chunking"]["size"] = 512
            
            try:
                chunk_overlap = int(self.query_one("#core-chunk-overlap", Input).value)
                config_data["rag_search"]["chunking"]["overlap"] = chunk_overlap
            except ValueError:
                self.app_instance.notify("Invalid chunk overlap. Using default.", severity="warning")
                config_data["rag_search"]["chunking"]["overlap"] = 128
            
            config_data["rag_search"]["chunking"]["method"] = self.query_one("#core-chunk-method", Select).value
            
            # Update memory settings
            if "memory" not in config_data["rag_search"]:
                config_data["rag_search"]["memory"] = {}
            
            try:
                max_memory = int(self.query_one("#core-memory-max", Input).value)
                config_data["rag_search"]["memory"]["max_memory_mb"] = max_memory
            except ValueError:
                self.app_instance.notify("Invalid memory limit. Using default.", severity="warning")
                config_data["rag_search"]["memory"]["max_memory_mb"] = 1024
            
            config_data["rag_search"]["memory"]["monitoring_enabled"] = self.query_one("#core-memory-monitoring", Checkbox).value
            
            # Save to file
            with open(DEFAULT_CONFIG_PATH, "w") as f:
                toml.dump(config_data, f)
            
            # Update internal config
            self.config_data = config_data
            
            self.app_instance.notify("Core Settings saved successfully!")
            
        except Exception as e:
            self.app_instance.notify(f"Error saving Core Settings: {e}", severity="error")

    async def _reset_core_settings(self) -> None:
        """Reset Core Settings to default values."""
        try:
            # Default values
            defaults = {
                "rag_search": {
                    "default_mode": "qa",
                    "default_sources": {
                        "media": True,
                        "conversations": True,
                        "notes": True
                    },
                    "default_top_k": 10,
                    "reranking": {
                        "enabled": False,
                        "model": "cross-encoder/ms-marco-MiniLM-L-12-v2"
                    },
                    "chunking": {
                        "size": 512,
                        "overlap": 128,
                        "method": "fixed"
                    },
                    "memory": {
                        "max_memory_mb": 1024,
                        "monitoring_enabled": True
                    }
                }
            }
            
            # Update UI elements
            self.query_one("#core-rag-search-mode", Select).value = defaults["rag_search"]["default_mode"]
            self.query_one("#core-rag-source-media", Checkbox).value = defaults["rag_search"]["default_sources"]["media"]
            self.query_one("#core-rag-source-conversations", Checkbox).value = defaults["rag_search"]["default_sources"]["conversations"]
            self.query_one("#core-rag-source-notes", Checkbox).value = defaults["rag_search"]["default_sources"]["notes"]
            self.query_one("#core-rag-top-k", Input).value = str(defaults["rag_search"]["default_top_k"])
            self.query_one("#core-rag-rerank-enabled", Checkbox).value = defaults["rag_search"]["reranking"]["enabled"]
            self.query_one("#core-rag-rerank-model", Input).value = defaults["rag_search"]["reranking"]["model"]
            self.query_one("#core-chunk-size", Input).value = str(defaults["rag_search"]["chunking"]["size"])
            self.query_one("#core-chunk-overlap", Input).value = str(defaults["rag_search"]["chunking"]["overlap"])
            self.query_one("#core-chunk-method", Select).value = defaults["rag_search"]["chunking"]["method"]
            self.query_one("#core-memory-max", Input).value = str(defaults["rag_search"]["memory"]["max_memory_mb"])
            self.query_one("#core-memory-monitoring", Checkbox).value = defaults["rag_search"]["memory"]["monitoring_enabled"]
            
            self.app_instance.notify("Core Settings reset to defaults!")
            
        except Exception as e:
            self.app_instance.notify(f"Error resetting Core Settings: {e}", severity="error")

#
# End of Tools_Settings_Window.py
#######################################################################################################################
