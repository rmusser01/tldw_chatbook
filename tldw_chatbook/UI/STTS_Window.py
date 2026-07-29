# STTS_Window.py
# Description: S/TT/S (Speech/Text-to-Speech) tab with TTS Playground, Settings, and AudioBook/Podcast Generation
#
# Imports
import asyncio
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Optional, Dict, Any, List, Literal
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer, Container
from textual.widgets import (
    Label,
    Button,
    TextArea,
    Select,
    Input,
    Static,
    RichLog,
    Switch,
    Collapsible,
    Rule,
)
from textual.css.query import QueryError
from textual.widget import Widget
from textual.reactive import reactive
from textual.binding import Binding
from textual import on, work
from loguru import logger
from rich.text import Text

# Local imports
from tldw_chatbook.config import get_cli_setting
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSPlaygroundGenerateEvent,
    STTSSettingsSaveEvent,
    STTSAudioBookGenerateEvent,
)
from tldw_chatbook.TTS import (
    STTSGeneratedAudio,
    STTSPlaygroundRequest,
    TTSPreferencesSnapshot,
    get_tts_service,
)
from tldw_chatbook.TTS.adapter_types import (
    TTSOperationError,
    TTSProviderCatalog,
    TTSProviderReconfiguringError,
    TTSRegistryClosedError,
)
from tldw_chatbook.TTS.audio_cpp_config import AudioCppConfig
from tldw_chatbook.TTS.legacy_catalogs import (
    LEGACY_DEFAULT_MODELS,
    LEGACY_DEFAULT_VOICES,
    LEGACY_VOICE_OPTIONS,
)
from tldw_chatbook.UI.stts_playground_catalog import (
    AUDIO_CPP_PROVIDER_ID,
    CatalogRequestToken,
    FIRST_AVAILABLE_MODEL_ID,
    LOADING_SELECT_VALUE,
    PlaygroundControls,
    SERVER_DEFAULT_VOICE_ID,
    UNAVAILABLE_SELECT_VALUE,
    SelectSentinel,
    SelectValue,
    controls_from_catalog,
    provider_options,
    voice_id_for_request,
)
from tldw_chatbook.UI.Speech.speech_playground_pane import SpeechPlaygroundPane
from tldw_chatbook.UI.Speech.speech_settings_mixin import SpeechSettingsMixin
from tldw_chatbook.UI.Speech.speech_effects_pane import SpeechEffectsPane
from tldw_chatbook.UI.Speech.speech_settings_pane import SpeechSettingsPane
from tldw_chatbook.UI.destination_recovery import optional_dependency_recovery_state
from tldw_chatbook.Widgets.voice_blend_dialog import VoiceBlendDialog
from tldw_chatbook.Widgets.enhanced_file_picker import (
    EnhancedFileOpen as FileOpen,
    EnhancedFileSave as FileSave,
)
from tldw_chatbook.Third_Party.textual_fspicker import Filters
from tldw_chatbook.UI.Dictation_Window_Improved import (
    ImprovedDictationWindow as DictationWindow,
)
from tldw_chatbook.Utils.optional_deps import (
    DEPENDENCIES_AVAILABLE,
    check_stt_deps,
    check_tts_deps,
)
# Note: Not using form_components due to generator/widget incompatibility

import json

#######################################################################################################################
#
# Classes:







































































class AudioBookGenerationWidget(Widget):
    """AudioBook/Podcast Generation widget"""

    DEFAULT_CSS = """
    AudioBookGenerationWidget {
        height: 100%;
        width: 100%;
    }
    
    .audiobook-container {
        padding: 1;
        height: 100%;
    }
    
    .chapter-list {
        height: 20;
        border: solid $primary;
        margin: 1 0;
        overflow-y: auto;
    }
    
    #audiobook-generation-log {
        height: 15;
        border: solid $secondary;
    }
    
    .cost-estimate {
        color: $warning;
        margin: 1 0;
    }
    """

    def __init__(self):
        super().__init__()
        self.content_text = ""
        self.detected_chapters = []
        self.generated_audiobook_path = None

    def compose(self) -> ComposeResult:
        """Compose the AudioBook/Podcast UI"""
        with ScrollableContainer(classes="audiobook-container"):
            yield Label("📚 AudioBook/Podcast Generation", classes="section-title")

            # Import section
            with Collapsible(title="Import Content", classes="settings-section"):
                with Horizontal(classes="form-row"):
                    yield Label("Import From:", classes="form-label")
                    yield Select(
                        options=[
                            ("file", "Text File"),
                            ("notes", "Notes"),
                            ("conversation", "Conversation"),
                            ("paste", "Paste Text"),
                        ],
                        id="import-source-select",
                    )

                yield Button(
                    "📁 Import Content", id="import-content-btn", variant="default"
                )

            # Content preview
            yield Label("Content Preview:")
            yield TextArea(id="content-preview", disabled=True)

            # Chapter Editor - Enhanced visual chapter editing
            with Collapsible(
                title="📖 Chapter Editor", classes="settings-section", collapsed=False
            ):
                from tldw_chatbook.Widgets.TTS.chapter_editor_widget import (
                    ChapterEditorWidget,
                )

                yield ChapterEditorWidget(id="chapter-editor-widget")

            # Voice assignment - Enhanced character voice management
            with Collapsible(title="🎭 Voice Assignment", classes="settings-section"):
                with Horizontal(classes="form-row"):
                    yield Label("Narrator Voice:", classes="form-label")
                    yield Select(
                        options=[
                            ("alloy", "Alloy"),
                            ("echo", "Echo"),
                            ("fable", "Fable"),
                            ("onyx", "Onyx"),
                            ("nova", "Nova"),
                            ("shimmer", "Shimmer"),
                        ],
                        id="narrator-voice-select",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Enable Multi-voice:", classes="form-label")
                    yield Switch(id="multi-voice-switch", value=False)

                # Character voice widget
                from tldw_chatbook.Widgets.TTS.character_voice_widget import (
                    CharacterVoiceWidget,
                )

                yield CharacterVoiceWidget(id="character-voice-widget")

            # Generation settings
            with Collapsible(title="Generation Settings", classes="settings-section"):
                with Horizontal(classes="form-row"):
                    yield Label("Provider:", classes="form-label")
                    yield Select(
                        options=[
                            ("openai", "OpenAI"),
                            ("elevenlabs", "ElevenLabs"),
                            ("kokoro", "Kokoro (Local)"),
                            ("chatterbox", "Chatterbox (Local)"),
                        ],
                        id="audiobook-provider-select",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Audio Format:", classes="form-label")
                    yield Select(
                        options=[
                            ("mp3", "MP3"),
                            ("m4b", "M4B (AudioBook)"),
                            ("opus", "Opus"),
                            ("aac", "AAC"),
                            ("wav", "WAV"),
                        ],
                        id="audiobook-format-select",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Include Chapter Markers:", classes="form-label")
                    yield Switch(id="chapter-markers-switch", value=True)

                with Horizontal(classes="form-row"):
                    yield Label("Background Music:", classes="form-label")
                    yield Switch(id="background-music-switch", value=False)

            # Cost estimate
            yield Static("", id="cost-estimate", classes="cost-estimate")

            # Generate button
            yield Button(
                "🎙️ Generate AudioBook", id="generate-audiobook-btn", variant="primary"
            )

            # Export button (initially disabled)
            yield Button(
                "💾 Export AudioBook",
                id="audiobook-export-btn",
                variant="success",
                disabled=True,
            )

            # Progress section
            yield Rule()
            yield Label("Generation Progress:")
            yield RichLog(id="audiobook-generation-log", highlight=True, markup=True)

    def on_mount(self) -> None:
        """Set initial values from config after mount"""
        # Delay initialization to ensure widgets are ready
        self.set_timer(0.1, self._initialize_audiobook_defaults)

    def _initialize_audiobook_defaults(self) -> None:
        """Initialize default values after widgets are ready"""
        try:
            # Set audiobook provider
            provider_select = self.query_one("#audiobook-provider-select", Select)
            default_provider = get_cli_setting("app_tts", "default_provider", "openai")
            if default_provider in ["openai", "elevenlabs", "kokoro", "chatterbox"]:
                try:
                    provider_select.value = default_provider
                except Exception as e:
                    logger.debug(f"Could not set audiobook provider: {e}")

            # Set default format to m4b
            format_select = self.query_one("#audiobook-format-select", Select)
            try:
                format_select.value = "m4b"
            except Exception as e:
                logger.debug(f"Could not set audiobook format: {e}")
        except Exception as e:
            logger.warning(f"Failed to set audiobook defaults: {e}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "import-content-btn":
            self._import_content()
            event.stop()  # Prevent event from bubbling up
        elif event.button.id == "generate-audiobook-btn":
            self._generate_audiobook()
            event.stop()
        elif event.button.id == "audiobook-export-btn":
            self._export_audiobook()
            event.stop()

    @on(Select.Changed)
    def on_audiobook_provider_select_for_voice_widget_changed(
        self, event: Select.Changed
    ) -> None:
        """Handle select widget changes"""
        if event.select.id == "audiobook-provider-select":
            # Update character voice widget provider
            try:
                from tldw_chatbook.Widgets.TTS.character_voice_widget import (
                    CharacterVoiceWidget,
                )

                voice_widget = self.query_one(
                    "#character-voice-widget", CharacterVoiceWidget
                )
                voice_widget.provider = event.value
                logger.info(f"Updated voice widget provider to: {event.value}")
            except Exception as e:
                logger.debug(f"Could not update voice widget provider: {e}")

    def _import_content(self) -> None:
        """Import content for audiobook generation"""
        import_source = self.query_one("#import-source-select", Select).value

        if import_source == "file":
            self._import_from_file()
        elif import_source == "notes":
            self._import_from_notes()
        elif import_source == "conversation":
            self._import_from_conversation()
        elif import_source == "paste":
            self._import_from_paste()

    def _import_from_file(self) -> None:
        """Import content from a text file"""
        try:
            # Create file picker for text files using pre-imported FileOpen
            filters = Filters(
                ("Text Files", lambda p: p.suffix.lower() in [".txt", ".md", ".rst"]),
                ("eBook Files", lambda p: p.suffix.lower() in [".epub", ".mobi"]),
                ("All Files", lambda p: True),
            )

            file_picker = FileOpen(
                title="Select Text File for AudioBook",
                filters=filters,
                context="audiobook_text",
            )

            # Mount the file picker
            self.app.push_screen(file_picker, self._handle_file_selection)
        except ImportError:
            # Fallback to simple file input
            self.app.notify(
                "File picker not available. Please paste your text instead.",
                severity="warning",
            )

    def _handle_file_selection(self, path: Optional[str]) -> None:
        """Handle file selection for audiobook content"""
        if not path:
            return

        try:
            # Read the file content
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            # Update content preview
            self.content_text = content
            content_preview = self.query_one("#content-preview", TextArea)
            content_preview.load_text(
                content[:1000] + "..." if len(content) > 1000 else content
            )
            content_preview.disabled = False

            # Detect chapters if enabled
            if self.query_one("#auto-chapters-switch", Switch).value:
                self._detect_chapters()

            self.app.notify(
                f"Imported {len(content)} characters from {Path(path).name}",
                severity="information",
            )

        except Exception as e:
            logger.error(f"Failed to import file: {e}")
            self.app.notify(f"Failed to import file: {e}", severity="error")

    def _import_from_notes(self) -> None:
        """Import content from notes"""
        from tldw_chatbook.Widgets.Note_Widgets.note_selection_dialog import (
            NoteSelectionDialog,
        )
        from tldw_chatbook.DB.ChaChaNotes_DB import fetch_all_notes

        try:
            # Fetch all notes from database
            notes = fetch_all_notes()
            if not notes:
                self.app.notify("No notes found in database", severity="warning")
                return

            # Show note selection dialog
            def handle_note_selection(selected_ids: Optional[List[int]]) -> None:
                if selected_ids:
                    # Fetch full content for selected notes
                    from tldw_chatbook.DB.ChaChaNotes_DB import fetch_note_by_id

                    combined_content = []

                    for note_id in selected_ids:
                        note = fetch_note_by_id(note_id)
                        if note:
                            # Add note title as chapter if it exists
                            if note.get("title"):
                                combined_content.append(f"# {note['title']}\n")
                            combined_content.append(note.get("content", ""))
                            combined_content.append("\n\n")  # Separator between notes

                    # Load combined content
                    self.content_text = "\n".join(combined_content)
                    content_preview = self.query_one("#content-preview", TextArea)
                    preview_text = (
                        self.content_text[:1000] + "..."
                        if len(self.content_text) > 1000
                        else self.content_text
                    )
                    content_preview.load_text(preview_text)
                    content_preview.disabled = False

                    # Detect chapters if enabled
                    if self.query_one("#auto-chapters-switch", Switch).value:
                        self._detect_chapters()

                    self.app.notify(
                        f"Imported {len(selected_ids)} note(s)", severity="information"
                    )

            self.app.push_screen(NoteSelectionDialog(notes), handle_note_selection)

        except Exception as e:
            logger.error(f"Failed to import from notes: {e}")
            self.app.notify(f"Failed to import notes: {e}", severity="error")

    def _import_from_conversation(self) -> None:
        """Import content from conversation"""
        from tldw_chatbook.Widgets.conversation_selection_dialog import (
            ConversationSelectionDialog,
        )
        from tldw_chatbook.DB.ChaChaNotes_DB import fetch_all_conversations

        try:
            # Fetch all conversations from database
            conversations = fetch_all_conversations()
            if not conversations:
                self.app.notify(
                    "No conversations found in database", severity="warning"
                )
                return

            # Show conversation selection dialog
            def handle_conversation_selection(
                selection: Optional[Dict[str, Any]],
            ) -> None:
                if selection:
                    # Fetch messages for selected conversation
                    from tldw_chatbook.DB.ChaChaNotes_DB import (
                        fetch_messages_by_conversation_id,
                    )

                    messages = fetch_messages_by_conversation_id(
                        selection["conversation_id"]
                    )

                    if not messages:
                        self.app.notify(
                            "No messages found in conversation", severity="warning"
                        )
                        return

                    # Build content based on options
                    content_parts = []
                    for msg in messages:
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "")

                        # Filter based on inclusion options
                        if selection.get("include_all"):
                            pass  # Include all messages
                        elif selection.get("include_user") and role != "user":
                            continue
                        elif selection.get("include_assistant") and role != "assistant":
                            continue

                        # Format based on speaker option
                        if selection.get("include_speakers"):
                            speaker_name = "User" if role == "user" else "Assistant"
                            content_parts.append(f"{speaker_name}: {content}")
                        else:
                            content_parts.append(content)

                        content_parts.append("")  # Empty line between messages

                    # Load combined content
                    self.content_text = "\n".join(content_parts)
                    content_preview = self.query_one("#content-preview", TextArea)
                    preview_text = (
                        self.content_text[:1000] + "..."
                        if len(self.content_text) > 1000
                        else self.content_text
                    )
                    content_preview.load_text(preview_text)
                    content_preview.disabled = False

                    # Auto-detect chapters might not be suitable for conversations
                    # but run it if enabled
                    if self.query_one("#auto-chapters-switch", Switch).value:
                        self._detect_chapters()

                    self.app.notify(
                        f"Imported conversation with {len(messages)} messages",
                        severity="information",
                    )

            self.app.push_screen(
                ConversationSelectionDialog(conversations),
                handle_conversation_selection,
            )

        except Exception as e:
            logger.error(f"Failed to import from conversation: {e}")
            self.app.notify(f"Failed to import conversation: {e}", severity="error")

    def _import_from_paste(self) -> None:
        """Import content from clipboard paste"""
        # Enable the content preview for editing
        content_preview = self.query_one("#content-preview", TextArea)
        content_preview.disabled = False
        content_preview.focus()
        self.app.notify(
            "Paste your text into the content preview area", severity="information"
        )

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Handle text area content changes"""
        if event.text_area.id == "content-preview":
            self.content_text = event.text_area.text
            # Detect chapters if auto-detect is enabled
            if (
                self.query_one("#auto-chapters-switch", Switch).value
                and self.content_text
            ):
                self._detect_chapters()

    def on_chapter_edit_event(self, event) -> None:
        """Handle chapter edit events from the chapter editor"""
        from tldw_chatbook.Widgets.TTS.chapter_editor_widget import ChapterEditEvent

        if isinstance(event, ChapterEditEvent):
            # Update our internal chapter list
            try:
                from tldw_chatbook.Widgets.TTS.chapter_editor_widget import (
                    ChapterEditorWidget,
                )

                chapter_editor = self.query_one(
                    "#chapter-editor-widget", ChapterEditorWidget
                )
                self.detected_chapters = chapter_editor.get_chapters()
                logger.info(f"Chapter {event.action}: {event.chapter.title}")
            except Exception as e:
                logger.error(f"Failed to handle chapter edit: {e}")

    def on_chapter_preview_event(self, event) -> None:
        """Handle chapter preview requests"""
        from tldw_chatbook.Widgets.TTS.chapter_editor_widget import ChapterPreviewEvent

        if isinstance(event, ChapterPreviewEvent):
            if event.preview_type == "audio":
                self._preview_chapter_audio(event.chapter)

    def on_character_detection_event(self, event) -> None:
        """Handle character detection requests"""
        from tldw_chatbook.Widgets.TTS.character_voice_widget import (
            CharacterDetectionEvent,
            CharacterVoiceWidget,
        )

        if isinstance(event, CharacterDetectionEvent):
            # Detect characters from current content
            if self.content_text:
                try:
                    voice_widget = self.query_one(
                        "#character-voice-widget", CharacterVoiceWidget
                    )
                    characters = voice_widget.detect_characters_from_text(
                        self.content_text, event.auto_assign
                    )
                    self.app.notify(
                        f"Detected {len(characters)} characters", severity="information"
                    )
                except Exception as e:
                    logger.error(f"Failed to detect characters: {e}")
                    self.app.notify(
                        f"Failed to detect characters: {e}", severity="error"
                    )
            else:
                self.app.notify("Please import content first", severity="warning")

    def on_character_voice_assign_event(self, event) -> None:
        """Handle character voice assignments"""
        from tldw_chatbook.Widgets.TTS.character_voice_widget import (
            CharacterVoiceAssignEvent,
        )

        if isinstance(event, CharacterVoiceAssignEvent):
            logger.info(f"Voice assigned: {event.character_name} → {event.voice_id}")

    def _detect_chapters(self) -> None:
        """Detect chapters in the content"""
        if not self.content_text:
            return

        try:
            from tldw_chatbook.TTS.audiobook_generator import ChapterDetector
            from tldw_chatbook.Widgets.TTS.chapter_editor_widget import (
                ChapterEditorWidget,
            )

            # Detect chapters
            self.detected_chapters = ChapterDetector.detect_chapters(self.content_text)

            # Update the chapter editor widget
            try:
                chapter_editor = self.query_one(
                    "#chapter-editor-widget", ChapterEditorWidget
                )
                chapter_editor.set_chapters(self.detected_chapters)
                self.app.notify(
                    f"Detected {len(self.detected_chapters)} chapters",
                    severity="information",
                )
            except Exception as e:
                logger.warning(f"Could not update chapter editor: {e}")
                # Fall back to old display method if chapter editor not found
                chapter_list = self.query_one("#chapter-list", Static)
                if self.detected_chapters:
                    chapter_display = []
                    for i, chapter in enumerate(self.detected_chapters):
                        chapter_display.append(
                            f"{i + 1}. {chapter.title} ({len(chapter.content.split())} words)"
                        )

                    chapter_list.update("\n".join(chapter_display))
                    self.app.notify(
                        f"Detected {len(self.detected_chapters)} chapters",
                        severity="information",
                    )
                else:
                    chapter_list.update("No chapters detected")

        except Exception as e:
            logger.error(f"Failed to detect chapters: {e}")
            self.app.notify(f"Failed to detect chapters: {e}", severity="error")

    def _generate_audiobook(self) -> None:
        """Generate the audiobook"""
        # Validate content
        if not self.content_text:
            self.app.notify("Please import content first", severity="warning")
            return

        # Get settings from UI
        provider = self.query_one("#audiobook-provider-select", Select).value
        audio_format = self.query_one("#audiobook-format-select", Select).value
        narrator_voice = self.query_one("#narrator-voice-select", Select).value

        # Validate voice selection
        if not narrator_voice or narrator_voice == Select.BLANK:
            self.app.notify("Please select a valid narrator voice", severity="warning")
            return

        multi_voice = self.query_one("#multi-voice-switch", Switch).value
        include_chapters = self.query_one("#chapter-markers-switch", Switch).value
        background_music = self.query_one("#background-music-switch", Switch).value

        # Get chapters from the chapter editor widget
        try:
            from tldw_chatbook.Widgets.TTS.chapter_editor_widget import (
                ChapterEditorWidget,
            )

            chapter_editor = self.query_one(
                "#chapter-editor-widget", ChapterEditorWidget
            )
            chapters = chapter_editor.get_chapters()
        except Exception as e:
            logger.warning(f"Could not get chapters from editor: {e}")
            chapters = self.detected_chapters

        # Get title from first chapter or use default
        title = "Untitled AudioBook"
        if chapters:
            # Use book title if detected, otherwise use first chapter
            for chapter in chapters:
                if "title" in chapter.title.lower() or chapter.number == 1:
                    title = chapter.title
                    break

        # Get character voice assignments if multi-voice is enabled
        character_voices = {}
        if multi_voice:
            try:
                from tldw_chatbook.Widgets.TTS.character_voice_widget import (
                    CharacterVoiceWidget,
                )

                voice_widget = self.query_one(
                    "#character-voice-widget", CharacterVoiceWidget
                )
                character_voices = voice_widget.get_voice_assignments()
                logger.info(f"Using character voices: {character_voices}")
            except Exception as e:
                logger.warning(f"Could not get character voices: {e}")

        # Prepare options
        options = {
            "title": title,
            "author": "Unknown",
            "provider": provider,
            "model": self._get_model_for_provider(provider),
            "chapter_detection": include_chapters,
            "multi_voice": multi_voice,
            "character_voices": character_voices,
            "background_music": None if not background_music else True,
            "enable_ssml": provider in ["elevenlabs"],
            "normalize_audio": True,
        }

        # Log start
        log = self.query_one("#audiobook-generation-log", RichLog)
        log.clear()
        log.write("[bold yellow]Starting audiobook generation...[/bold yellow]")
        log.write(f"Provider: {provider}")
        log.write(f"Format: {audio_format}")
        log.write(f"Content length: {len(self.content_text)} characters")

        # Estimate cost
        self._estimate_cost(provider, len(self.content_text))

        # Disable generate button
        self.query_one("#generate-audiobook-btn", Button).disabled = True

        # Post event to generate audiobook
        self.app.post_message(
            STTSAudioBookGenerateEvent(
                content=self.content_text,
                chapters=self.detected_chapters if include_chapters else [],
                narrator_voice=narrator_voice,
                output_format=audio_format,
                options=options,
            )
        )

    def _get_model_for_provider(self, provider: str) -> str:
        """Get default model for provider"""
        models = {
            "openai": "tts-1",
            "elevenlabs": "eleven_multilingual_v2",
            "kokoro": "kokoro-v0_19",
            "chatterbox": "chatterbox-v1",
        }
        return models.get(provider, "tts-1")

    def _estimate_cost(self, provider: str, char_count: int) -> None:
        """Estimate and display cost"""
        # Simple cost estimation (prices per 1K characters)
        costs_per_1k = {
            "openai": 0.015,  # TTS-1 pricing
            "elevenlabs": 0.13,  # Starter pricing
            "kokoro": 0.0,  # Local
            "chatterbox": 0.0,  # Local
        }

        cost_per_1k = costs_per_1k.get(provider, 0.0)
        estimated_cost = (char_count / 1000) * cost_per_1k

        cost_display = self.query_one("#cost-estimate", Static)
        if estimated_cost > 0:
            cost_display.update(f"Estimated cost: ${estimated_cost:.2f}")
        else:
            cost_display.update("Free (using local model)")

    def _is_valid_voice(self, voice: str) -> bool:
        """Check if a voice value is valid (not a separator)"""
        return bool(voice) and not str(voice).startswith("_separator")

    @on(Select.Changed)
    def on_audiobook_selects_changed(self, event: Select.Changed) -> None:
        """Handle select changes"""
        if event.select.id == "audiobook-provider-select":
            # Update narrator voice options based on provider
            self._update_voice_options(event.value)
            # Update cost estimate
            if self.content_text:
                self._estimate_cost(event.value, len(self.content_text))
        elif event.select.id == "narrator-voice-select":
            # Validate voice selection (prevent selecting separators)
            if not self._is_valid_voice(event.value):
                # Find and select the first valid voice
                voice_select = event.select
                for value, _ in voice_select._options:
                    if self._is_valid_voice(value):
                        voice_select.value = value
                        break

    def _update_voice_options(self, provider: str) -> None:
        """Update voice options based on provider"""
        voice_select = self.query_one("#narrator-voice-select", Select)

        if provider == "openai":
            voice_select.set_options(
                [
                    ("alloy", "Alloy"),
                    ("echo", "Echo"),
                    ("fable", "Fable"),
                    ("onyx", "Onyx"),
                    ("nova", "Nova"),
                    ("shimmer", "Shimmer"),
                ]
            )
        elif provider == "elevenlabs":
            voice_select.set_options(
                [
                    ("21m00Tcm4TlvDq8ikWAM", "Rachel"),
                    ("AZnzlk1XvdvUeBnXmlld", "Domi"),
                    ("EXAVITQu4vr4xnSDxMaL", "Bella"),
                    ("ErXwobaYiN019PkySvjV", "Antoni"),
                    ("MF3mGyEYCl7XYWbV9V6O", "Elli"),
                ]
            )
        elif provider == "kokoro":
            logger.info(f"Setting up Kokoro voices for provider: {provider}")
            voice_options = [
                ("af_bella", "Bella (US Female)"),
                ("af_nicole", "Nicole (US Female)"),
                ("af_sarah", "Sarah (US Female)"),
                ("am_adam", "Adam (US Male)"),
                ("am_michael", "Michael (US Male)"),
                ("bf_emma", "Emma (UK Female)"),
                ("bm_george", "George (UK Male)"),
            ]

            # Add saved voice blends
            blend_file = (
                Path.home() / ".config" / "tldw_cli" / "kokoro_voice_blends.json"
            )
            if blend_file.exists():
                try:
                    import json

                    with open(blend_file, "r") as f:
                        blends = json.load(f)
                        if blends:
                            # Add separator
                            voice_options.append(
                                ("_separator", "──── Voice Blends ────")
                            )
                            # Add each blend
                            for blend_name, blend_data in blends.items():
                                display_name = f"🎭 {blend_name}"
                                if blend_data.get("description"):
                                    display_name += (
                                        f" - {blend_data['description'][:30]}"
                                    )
                                voice_options.append(
                                    (f"blend:{blend_name}", display_name)
                                )
                except Exception as e:
                    logger.error(f"Failed to load voice blends: {e}")

            voice_select.set_options(voice_options)

            # Find first valid voice option (skip separators)
            valid_voice = None
            for value, _ in voice_options:
                if self._is_valid_voice(value):
                    valid_voice = value
                    break

            if valid_voice:
                voice_select.value = valid_voice

        elif provider == "chatterbox":
            voice_select.set_options(
                [
                    ("default", "Default"),
                    ("custom", "Custom Voice"),
                ]
            )

    def _export_audiobook(self) -> None:
        """Export the generated audiobook"""
        if not self.generated_audiobook_path:
            self.app.notify("No audiobook to export", severity="warning")
            return

        try:
            # Create file picker for save location using pre-imported FileSave
            filters = Filters(
                ("AudioBook Files", lambda p: p.suffix.lower() in [".m4b", ".mp3"]),
                ("All Files", lambda p: True),
            )

            file_picker = FileSave(
                title="Save AudioBook As",
                filters=filters,
                default_filename=self.generated_audiobook_path.name,
                context="audiobook_save",
            )

            # Mount the file picker
            self.app.push_screen(file_picker, self._handle_export_location)
        except ImportError:
            # Fallback
            self.app.notify(
                f"AudioBook saved to: {self.generated_audiobook_path}",
                severity="information",
            )

    def _handle_export_location(self, path: Optional[str]) -> None:
        """Handle export location selection"""
        if not path or not self.generated_audiobook_path:
            return

        try:
            import shutil

            shutil.copy2(self.generated_audiobook_path, path)
            self.app.notify(
                f"AudioBook exported to: {Path(path).name}", severity="information"
            )
        except Exception as e:
            logger.error(f"Failed to export audiobook: {e}")
            self.app.notify(f"Failed to export audiobook: {e}", severity="error")

    def audiobook_generation_complete(
        self, success: bool, path: Optional[Path] = None
    ) -> None:
        """Handle audiobook generation completion"""
        # Re-enable generate button
        self.query_one("#generate-audiobook-btn", Button).disabled = False

        if success and path:
            self.generated_audiobook_path = path
            # Enable export button
            self.query_one("#audiobook-export-btn", Button).disabled = False

            # Update log
            log = self.query_one("#audiobook-generation-log", RichLog)
            log.write("[bold green]✓ AudioBook generation complete![/bold green]")
            log.write(f"Output file: {path.name}")
        else:
            # Update log
            log = self.query_one("#audiobook-generation-log", RichLog)
            log.write("[bold red]✗ AudioBook generation failed![/bold red]")

    def _preview_chapter_audio(self, chapter) -> None:
        """Generate audio preview for a single chapter"""
        try:
            # Get current settings
            provider = self.query_one("#audiobook-provider-select", Select).value
            narrator_voice = self.query_one("#narrator-voice-select", Select).value

            if not narrator_voice or narrator_voice == Select.BLANK:
                self.app.notify(
                    "Please select a valid narrator voice", severity="warning"
                )
                return

            # Limit preview to first 500 characters
            preview_text = (
                chapter.content[:500] + "..."
                if len(chapter.content) > 500
                else chapter.content
            )

            # Log preview generation
            log = self.query_one("#audiobook-generation-log", RichLog)
            log.write(f"[yellow]Generating preview for: {chapter.title}[/yellow]")

            # Create TTS request event
            from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
                STTSPlaygroundGenerateEvent,
            )

            # Post event to generate preview
            self.post_message(
                STTSPlaygroundGenerateEvent(
                    STTSPlaygroundRequest(
                        operation_id=str(uuid4()),
                        provider_id=provider,
                        model_id=self._get_model_for_provider(provider),
                        text=preview_text,
                        voice_id=narrator_voice,
                        response_format="mp3",
                        speed=1.0,
                        options={"preview_chapter": chapter.title},
                    )
                )
            )

            log.write("[green]Preview generation started...[/green]")

        except Exception as e:
            logger.error(f"Failed to preview chapter audio: {e}")
            self.app.notify(f"Failed to generate preview: {e}", severity="error")

    def _get_model_for_provider(self, provider: str) -> str:
        """Get the default model for a given provider"""
        model_map = {
            "openai": "tts-1",
            "elevenlabs": "eleven_multilingual_v2",
            "kokoro": "kokoro",
            "chatterbox": "chatterbox",
            "alltalk": "alltalk",
        }
        return model_map.get(provider, "default")


class STTSWindow(Container):
    """Main S/TT/S window containing all sub-windows"""

    DEFAULT_CSS = """
    STTSWindow {
        layout: horizontal;
        height: 100%;
    }
    
    .stts-content {
        width: 1fr;
    }
    
    .section-title {
        text-style: bold;
        margin-bottom: 1;
    }
    
    .sidebar-button {
        width: 100%;
        margin-bottom: 1;
    }

    .speech-capability-status {
        margin-top: 1;
        padding: 1;
        border: round $surface;
        color: $text-muted;
    }
    """

    current_view = reactive("playground")

    def __init__(self, app_instance, **kwargs):
        """Initialize the S/TT/S window."""
        super().__init__(**kwargs)
        self.app_instance = app_instance

    def compose(self) -> ComposeResult:
        """Compose the S/TT/S window: content only.

        The sidebar that used to lead this method -- six view buttons and the
        capability status line -- moved into the Lab frame's rail and status
        chip (``UI/Screens/stts_screen.py``), so that Speech has the same
        chrome as Models and Evals instead of a second, differently-styled
        navigation column inside the body.

        The window keeps ownership of ``current_view`` and of mounting the
        matching content widget; the screen only points it at a view.
        """
        with Container(classes="stts-content"):
            # Seeds the default view, and records it: `watch_current_view`
            # would otherwise mount a SECOND playground on the first change,
            # because its `remove_children()` is deferred and the new widget
            # lands while the old one is still there. Legacy got away with
            # the same shape only because its widget carried no id to
            # collide on.
            yield SpeechPlaygroundPane(id="speech-playground-pane")
        self._mounted_view = "playground"

    def _speech_capability_status_text(self) -> str:
        """Return a concise local speech dependency status for the sidebar."""
        check_tts_deps()
        check_stt_deps()

        if self._speech_dependencies_available():
            return "Local speech: ready"

        return self._speech_dependency_recovery_state().visible_copy

    def _speech_capability_status_tooltip(self) -> str:
        """Return install guidance for local speech dependencies."""
        if self._speech_dependencies_available():
            return "Local TTS and STT dependencies are available."
        return self._speech_dependency_recovery_state().disabled_tooltip

    def _speech_dependencies_available(self) -> bool:
        return bool(DEPENDENCIES_AVAILABLE.get("tts_processing", False)) and bool(
            DEPENDENCIES_AVAILABLE.get("stt_processing", False)
        )

    def _speech_dependency_recovery_state(self):
        missing_dependencies = []
        if not DEPENDENCIES_AVAILABLE.get("tts_processing", False):
            missing_dependencies.append("local_tts")
        if not DEPENDENCIES_AVAILABLE.get("stt_processing", False):
            missing_dependencies.extend(
                ("transcription_faster_whisper", "speech_recording")
            )

        return optional_dependency_recovery_state(
            unavailable_what="Local speech providers",
            missing_dependencies=tuple(missing_dependencies),
            install_target='pip install "tldw_chatbook[local_tts,transcription_faster_whisper,speech_recording]"',
            stable_selector="speech-capability-status",
            recovery_action="Settings > Speech",
        )

    def watch_current_view(self, old_view: str, new_view: str) -> None:
        """Handle view changes.

        Returns early when the content container is not mounted yet. The
        window is now the Lab frame's deferred body, so it is mounted after
        first paint rather than composed inline -- and a reactive watcher can
        fire against a window whose own children have not been composed. The
        unguarded `query_one` raised NoMatches out of the frame's body mount,
        which took down the whole screen. Mirrors the same QueryError
        tolerance `LLMManagementWindow.watch_active_view` carries.
        """
        if new_view == getattr(self, "_mounted_view", None):
            return

        try:
            content_container = self.query_one(".stts-content", Container)
        except QueryError:
            logger.debug(
                "STTS content container not mounted yet; deferring view "
                f"change to '{new_view}' until compose completes."
            )
            return

        # Give widgets a chance to clean up before removal
        for child in content_container.children:
            if hasattr(child, "cleanup") and callable(child.cleanup):
                try:
                    child.cleanup()
                except Exception as e:
                    logger.debug(f"Error during widget cleanup: {e}")

        # Remove all children from the container
        content_container.remove_children()

        # Add new content based on view
        self._mounted_view = new_view
        if new_view == "playground":
            content_container.mount(
                SpeechPlaygroundPane(id="speech-playground-pane")
            )
        elif new_view == "settings":
            content_container.mount(SpeechSettingsPane(id="speech-settings-pane"))
        elif new_view == "audiobook":
            content_container.mount(AudioBookGenerationWidget())
        elif new_view == "effects":
            content_container.mount(SpeechEffectsPane(id="speech-effects-pane"))
        elif new_view == "dictation":
            content_container.mount(DictationWindow())

        # Selection styling is the rail's job now. These lines used to
        # `query_one("#view-*-btn")` for the four view buttons; those live on
        # STTSScreen since the sidebar moved, so every one of them would raise
        # NoMatches on the first view change. The screen watches
        # `current_view` and applies `is-active` itself.

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle sidebar button presses and delegate to content widgets"""
        # Handle sidebar buttons
        if event.button.id == "view-playground-btn":
            self.current_view = "playground"
        elif event.button.id == "view-settings-btn":
            self.current_view = "settings"
        elif event.button.id == "view-audiobook-btn":
            self.current_view = "audiobook"
        elif event.button.id == "view-voice-cloning-btn":
            # Import and push the Voice Cloning window
            from tldw_chatbook.UI.Voice_Cloning_Window import VoiceCloningWindow

            self.app.push_screen(VoiceCloningWindow())
        elif event.button.id == "view-stt-btn":
            self.current_view = "dictation"
        elif event.button.id == "view-effects-btn":
            self.app.notify("Audio Effects coming soon!", severity="information")
        else:
            # Try to delegate to the active content widget
            try:
                content_container = self.query_one(".stts-content", Container)
                if content_container.children:
                    # Get the active widget (should be only one)
                    active_widget = content_container.children[0]
                    if hasattr(active_widget, "on_button_pressed"):
                        active_widget.on_button_pressed(event)
            except Exception as e:
                logger.debug(f"Could not delegate button event: {e}")


#
# End of STTS_Window.py
#######################################################################################################################
