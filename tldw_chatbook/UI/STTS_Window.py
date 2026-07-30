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
    ProfileAvailabilityState,
    STTSGeneratedAudio,
    STTSPlaygroundRequest,
    TTSPlaygroundSelectionPreset,
    TTSPreferencesSnapshot,
    TTSProfileService,
    get_tts_service,
)
from tldw_chatbook.TTS.adapter_types import (
    TTSOperationError,
    TTSProviderCatalog,
    TTSProviderReconfiguringError,
    TTSRegistryClosedError,
    TTSVoiceDiscoveryResult,
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
    SERVER_DEFAULT_VOICE_LABEL,
    UNAVAILABLE_SELECT_VALUE,
    SelectSentinel,
    SelectValue,
    controls_from_catalog,
    controls_from_profile_preset,
    profile_availability_from_catalog,
    provider_options,
    voice_id_for_request,
)
from tldw_chatbook.UI.Speech.speech_effects_pane import SpeechEffectsPane
from tldw_chatbook.UI.Speech.speech_playground_pane import SpeechPlaygroundPane
from tldw_chatbook.UI.Speech.speech_settings_pane import SpeechSettingsPane
from tldw_chatbook.UI.stts_profile_library import (
    PROFILE_ACTION_FAILED_COPY,
    PROFILE_STORE_UNAVAILABLE_COPY,
    ProfilePreviewRequested,
    STTSProfileLibrary,
    TTSProfileNameModal,
    profile_action_error_copy,
)
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

_PROFILE_RESULT_STALE_COPY = (
    "TTS settings changed after this audio was generated. Generate a new "
    "result before saving it as a profile."
)






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
        """Compose the AudioBook/Podcast UI.

        The actions sit ABOVE the scroll region. Inside it, after eight
        collapsible groups, `generate-audiobook-btn` measured at y=40 in a
        26-row viewport -- the reason the view exists, four screens down.
        The grouping itself is unchanged: the spec keeps it, as the closest
        thing here to the Console grammar already.
        """
        yield Label("📚 AudioBook/Podcast Generation", classes="section-title")
        with Horizontal(id="audiobook-actions", classes="workbench-command-strip"):
            yield Button(
                "🎙️ Generate AudioBook",
                id="generate-audiobook-btn",
                variant="primary",
            )
            yield Button(
                "💾 Export AudioBook",
                id="audiobook-export-btn",
                variant="success",
                disabled=True,
            )

        with ScrollableContainer(classes="audiobook-container"):

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


class TTSPlaygroundWidget(Widget):
    """TTS Playground for testing different providers and settings"""

    BINDINGS = [
        Binding("ctrl+g", "generate_tts", "Generate Speech"),
        Binding("ctrl+r", "random_text", "Random Text"),
        Binding("ctrl+l", "clear_text", "Clear Text"),
        Binding("ctrl+p", "play_audio", "Play Audio"),
        Binding("ctrl+s", "stop_audio", "Stop Audio"),
    ]

    DEFAULT_CSS = """
    TTSPlaygroundWidget {
        height: 100%;
        width: 100%;
    }
    
    .tts-playground-container {
        padding: 1;
        height: 100%;
    }
    
    .form-row {
        height: 3;
        margin-bottom: 1;
    }

    Select.profile-exact-select {
        height: 3;
        min-height: 3;
        max-height: 3;
    }

    Select.profile-exact-select > SelectCurrent {
        height: 3;
        min-height: 3;
        max-height: 3;
    }

    Select.profile-exact-select > SelectCurrent > Static#label {
        height: 1;
        max-height: 1;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    #kokoro-language-row {
        display: none;
    }
    
    #kokoro-language-row.visible {
        display: block;
    }
    
    .provider-settings {
        display: none;
    }
    
    #kokoro-settings.visible {
        display: block;
    }
    
    #elevenlabs-settings.visible {
        display: block;
    }
    
    #chatterbox-settings.visible {
        display: block;
    }
    
    #higgs-settings.visible {
        display: block;
    }
    
    .watermark-notice {
        color: $warning;
        margin-top: 1;
    }
    
    .status-text {
        margin: 1 0;
        text-style: italic;
    }

    #tts-profile-preview-status {
        width: 100%;
        height: auto;
        max-height: 4;
        padding: 0 1;
        margin-bottom: 1;
        background: $boost;
    }

    #tts-profile-preview-status.profile-preview-available {
        color: $success;
    }

    #tts-profile-preview-status.profile-preview-loading {
        color: $text-muted;
    }

    #tts-profile-preview-status.profile-preview-unverified {
        color: $warning;
    }

    #tts-profile-preview-status.profile-preview-unavailable {
        color: $error;
    }
    
    .audio-player {
        height: 11;
        border: solid $primary;
        padding: 1;
        margin-top: 1;
    }
    
    .generation-log {
        height: 20;
        border: solid $secondary;
        margin-top: 1;
    }
    
    .audio-progress {
        width: 100%;
        margin: 0 1;
    }
    
    .audio-time {
        width: auto;
        margin: 0 1;
    }
    
    .hidden {
        display: none;
    }
    
    .generation-status {
        height: 4;
        margin: 1 0;
        border: solid $primary;
        padding: 0 1;
    }
    
    #generation-status-text {
        margin-bottom: 0;
    }
    
    #generation-progress {
        margin-top: 0;
    }
    
    .tts-text-input, #tts-text-input {
        height: 10;
        min-height: 5;
        max-height: 20;
        border: solid $primary;
        padding: 0 1;
        margin-bottom: 1;
    }
    
    .text-input-container {
        height: auto;
        min-height: 12;
        margin-bottom: 1;
    }
    
    .example-text {
        color: $text-muted;
        margin-bottom: 1;
    }
    
    .quick-tips {
        border: solid $secondary;
        padding: 1;
        margin: 1 0;
        background: $boost;
    }
    
    .tip-text {
        color: $text-muted;
    }
    """

    def __init__(
        self,
        profile_preset: TTSPlaygroundSelectionPreset | None = None,
    ) -> None:
        super().__init__()
        self._profile_preset = profile_preset
        self._profile_effective_availability: ProfileAvailabilityState | None = (
            profile_preset.availability if profile_preset is not None else None
        )
        self._profile_preview_loading = profile_preset is not None
        self._profile_configuration_revision: int | None = None
        self._profile_voice_validation_token: CatalogRequestToken | None = None
        self.current_audio_file = None
        self.current_audio_artifact: STTSGeneratedAudio | None = None
        self.reference_audio_path = None
        self.higgs_reference_audio_path = None
        self._progress_timer_task = None
        self._play_worker_task = None
        self._active_playback_release: Callable[[], None] | None = None
        self._tts_service = None
        self._provider_ids: frozenset[str] = frozenset()
        self._provider_display_names: dict[str, str] = {}
        self._displayed_provider_id: str | None = None
        self._selected_provider_id: str | None = None
        self._catalogs: dict[str, TTSProviderCatalog] = {}
        self._catalog_configuration_revisions: dict[str, int] = {}
        self._catalog_request_generations: dict[str, int] = {}
        self._voice_request_generations: dict[tuple[str, str], int] = {}
        self._discovered_voices: dict[tuple[str, str], tuple[str, ...]] = {}
        self._pending_voice_selections: dict[str, str] = {}
        self._provider_control_snapshots: dict[str, dict[str, Any]] = {}
        self._stale_providers: set[str] = set()
        self._catalog_generation_allowed = False
        self._applying_catalog_controls = False
        self._applied_model_id: str | None = None
        self._applied_voice_id: SelectValue | None = None
        self._applied_format: str | None = None
        self._generation_operation_id: str | None = None
        self._profile_save_suppressed = False
        self._profile_controls_applied = profile_preset is None
        self._active_profile_name_modal: TTSProfileNameModal | None = None
        self.example_texts = [
            "Welcome to the Text-to-Speech playground! This is where you can experiment with different voices, providers, and settings to create natural-sounding speech.",
            "The quick brown fox jumps over the lazy dog. This pangram contains all letters of the alphabet.",
            "In a world of artificial intelligence, the ability to convert text into natural speech opens countless possibilities.",
            "Testing, one, two, three. Can you hear the difference between various voice models?",
            "Good morning! Today's weather is sunny with a high of 75 degrees. Perfect for a walk in the park.",
        ]

    def compose(self) -> ComposeResult:
        """Compose the TTS Playground UI"""
        with ScrollableContainer(classes="tts-playground-container"):
            yield Label("🎤 TTS Playground", classes="section-title")
            yield Static(
                "",
                id="tts-profile-preview-status",
                classes="hidden",
            )

            # Text input area
            with Vertical(classes="text-input-container"):
                yield Label("Text to Synthesize:")
                yield Static(
                    "Example: Hello! Welcome to the TTS Playground. Try different voices and settings.",
                    classes="example-text",
                )
                yield TextArea(
                    "Welcome to the Text-to-Speech playground! This is where you can experiment with different voices, providers, and settings to create natural-sounding speech.",
                    id="tts-text-input",
                    classes="tts-text-input",
                )

            # Provider selection
            with Horizontal(classes="form-row"):
                yield Label("Provider:", classes="form-label")
                yield Select(
                    options=[("Loading providers…", LOADING_SELECT_VALUE)],
                    id="tts-provider-select",
                    allow_blank=False,
                    disabled=True,
                )
                yield Button(
                    "Refresh Models",
                    id="tts-refresh-catalog-btn",
                    disabled=True,
                )

            yield Static(
                "Loading TTS providers…",
                id="tts-provider-status",
                classes="status-text",
            )

            # Voice selection (will be populated based on provider)
            with Horizontal(classes="form-row"):
                yield Label("Voice:", classes="form-label")
                yield Select(
                    options=[("Waiting for provider…", LOADING_SELECT_VALUE)],
                    id="tts-voice-select",
                    allow_blank=False,
                    disabled=True,
                )

            # Model selection
            with Horizontal(classes="form-row"):
                yield Label("Model:", classes="form-label")
                yield Select(
                    options=[("Waiting for provider…", LOADING_SELECT_VALUE)],
                    id="tts-model-select",
                    allow_blank=False,
                    disabled=True,
                )

            # Language selection (for Kokoro)
            with Horizontal(classes="form-row", id="kokoro-language-row"):
                yield Label("Language:", classes="form-label")
                yield Select(
                    options=[
                        ("en-us", "American English"),
                        ("en-gb", "British English"),
                        ("ja", "Japanese"),
                        ("zh", "Mandarin Chinese"),
                        ("es", "Spanish"),
                        ("fr", "French"),
                        ("hi", "Hindi"),
                        ("it", "Italian"),
                        ("pt-br", "Brazilian Portuguese"),
                    ],
                    id="tts-language-select",
                )

            # Kokoro-specific settings
            with Vertical(id="kokoro-settings", classes="provider-settings"):
                # ONNX/PyTorch toggle
                with Horizontal(classes="form-row"):
                    yield Label("Use ONNX:", classes="form-label")
                    yield Switch(
                        id="tts-kokoro-use-onnx",
                        value=get_cli_setting("app_tts", "KOKORO_USE_ONNX", True),
                    )

            # Speed control
            with Horizontal(classes="form-row"):
                yield Label("Speed:", classes="form-label")
                yield Input(
                    id="tts-speed-input",
                    value="1.0",
                    placeholder="0.25-4.0",
                    type="number",
                    disabled=True,
                )

            # ElevenLabs-specific settings
            with Vertical(id="elevenlabs-settings", classes="provider-settings"):
                with Horizontal(classes="form-row"):
                    yield Label("Voice Stability:", classes="form-label")
                    yield Input(
                        id="tts-stability-input",
                        value="0.5",
                        placeholder="0.0-1.0",
                        type="number",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Similarity Boost:", classes="form-label")
                    yield Input(
                        id="tts-similarity-input",
                        value="0.8",
                        placeholder="0.0-1.0",
                        type="number",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Style:", classes="form-label")
                    yield Input(
                        id="tts-style-input",
                        value="0.0",
                        placeholder="0.0-1.0",
                        type="number",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Speaker Boost:", classes="form-label")
                    yield Switch(id="tts-speaker-boost-switch", value=True)

            # Chatterbox-specific settings
            with Vertical(id="chatterbox-settings", classes="provider-settings"):
                with Horizontal(classes="form-row"):
                    yield Label("Exaggeration:", classes="form-label")
                    yield Input(
                        id="tts-exaggeration-input",
                        value="0.5",
                        placeholder="0.0-1.0",
                        type="number",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("CFG Weight:", classes="form-label")
                    yield Input(
                        id="tts-cfg-weight-input",
                        value="0.5",
                        placeholder="0.0-1.0",
                        type="number",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Temperature:", classes="form-label")
                    yield Input(
                        id="tts-temperature-input",
                        value="0.5",
                        placeholder="0.0-2.0",
                        type="number",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Candidates:", classes="form-label")
                    yield Input(
                        id="tts-num-candidates-input",
                        value="1",
                        placeholder="1-5",
                        type="number",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Whisper Validation:", classes="form-label")
                    yield Switch(id="tts-validate-whisper-switch", value=False)

                with Horizontal(classes="form-row"):
                    yield Label("Text Preprocessing:", classes="form-label")
                    yield Switch(id="tts-preprocess-text-switch", value=True)

                with Horizontal(classes="form-row"):
                    yield Label("Audio Normalization:", classes="form-label")
                    yield Switch(id="tts-normalize-audio-switch", value=True)

                with Horizontal(classes="form-row"):
                    yield Label("Target dB:", classes="form-label")
                    yield Input(
                        id="tts-target-db-input",
                        value="-20.0",
                        placeholder="-30 to -10",
                        type="number",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Random Seed:", classes="form-label")
                    yield Input(
                        id="tts-random-seed-input",
                        value="",
                        placeholder="Optional (e.g., 42)",
                        type="number",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Reference Audio:", classes="form-label")
                    yield Button(
                        "📁 Upload Audio", id="reference-audio-btn", variant="default"
                    )
                    yield Button(
                        "❌ Clear",
                        id="clear-reference-audio-btn",
                        variant="default",
                        disabled=True,
                    )

                yield Static(
                    "No reference audio selected",
                    id="reference-audio-status",
                    classes="status-text",
                )

                # Watermark notice
                yield Static(
                    "⚠️ Note: Generated audio includes watermarking for responsible AI use",
                    classes="watermark-notice",
                )

            # Higgs-specific settings
            with Vertical(id="higgs-settings", classes="provider-settings"):
                with Horizontal(classes="form-row"):
                    yield Label("Temperature:", classes="form-label")
                    yield Input(
                        id="tts-higgs-temperature-input",
                        value="0.7",
                        placeholder="0.0-2.0",
                        type="number",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Top P:", classes="form-label")
                    yield Input(
                        id="tts-higgs-top-p-input",
                        value="0.9",
                        placeholder="0.0-1.0",
                        type="number",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Repetition Penalty:", classes="form-label")
                    yield Input(
                        id="tts-higgs-repetition-penalty-input",
                        value="1.1",
                        placeholder="1.0+",
                        type="number",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Enable Voice Cloning:", classes="form-label")
                    yield Switch(id="tts-higgs-voice-cloning-switch", value=True)

                with Horizontal(classes="form-row"):
                    yield Label("Multi-speaker Mode:", classes="form-label")
                    yield Switch(id="tts-higgs-multi-speaker-switch", value=True)

                with Horizontal(classes="form-row", id="higgs-voice-upload-row"):
                    yield Label("Voice Reference:", classes="form-label")
                    yield Button(
                        "📁 Upload Voice",
                        id="higgs-voice-upload-btn",
                        variant="default",
                    )
                    yield Button(
                        "❌ Clear",
                        id="higgs-clear-voice-btn",
                        variant="default",
                        disabled=True,
                    )

                yield Static(
                    "No voice reference selected",
                    id="higgs-voice-status",
                    classes="status-text",
                )

                with Horizontal(classes="form-row"):
                    yield Label("Speaker Delimiter:", classes="form-label")
                    yield Input(
                        id="tts-higgs-delimiter-input",
                        value="|||",
                        placeholder="Default: |||",
                    )

                yield Static(
                    "💡 For multi-speaker: Use format 'Speaker|||Text' in your input",
                    classes="help-text",
                )

            # Format selection
            with Horizontal(classes="form-row"):
                yield Label("Format:", classes="form-label")
                yield Select(
                    options=[("Waiting for provider…", LOADING_SELECT_VALUE)],
                    id="tts-format-select",
                    allow_blank=False,
                    disabled=True,
                )
            yield Static(
                "audio.cpp returns one complete WAV and currently uses speed 1.0.",
                id="tts-audio-cpp-restrictions",
                classes="status-text hidden",
            )

            # Generate button and quick actions
            with Horizontal(classes="form-row"):
                yield Button(
                    "🔊 Generate Speech",
                    id="tts-generate-btn",
                    variant="primary",
                    disabled=True,
                )
                yield Button(
                    "🎲 Random Text", id="tts-random-text-btn", variant="default"
                )
                yield Button("🗑️ Clear", id="tts-clear-text-btn", variant="default")

            # Audio player placeholder
            with Container(id="audio-player-container", classes="audio-player"):
                yield Static(
                    "Audio player will appear here after generation",
                    id="audio-player-status",
                )

                # Progress bar for playback
                from textual.widgets import ProgressBar

                yield ProgressBar(
                    total=100,
                    show_eta=False,
                    show_percentage=False,
                    id="audio-progress-bar",
                    classes="audio-progress hidden",
                )
                yield Static(
                    "0:00 / 0:00", id="audio-time-display", classes="audio-time hidden"
                )

                with Horizontal():
                    yield Button("▶️ Play", id="audio-play-btn", disabled=True)
                    yield Button("⏸️ Pause", id="pause-audio-btn", disabled=True)
                    yield Button("⏹️ Stop", id="stop-audio-btn", disabled=True)
                    yield Button("💾 Export", id="audio-export-btn", disabled=True)
                yield Button(
                    "Save result as profile",
                    id="audio-save-profile-btn",
                    classes="hidden",
                    disabled=True,
                )

            # Generation status and progress
            with Container(
                id="generation-status-container", classes="generation-status hidden"
            ):
                yield Static("Ready to generate", id="generation-status-text")
                yield ProgressBar(
                    id="generation-progress", show_eta=True, show_percentage=True
                )

            # Generation log
            yield Label("Generation Log:")
            yield RichLog(
                id="tts-generation-log",
                classes="generation-log",
                highlight=True,
                markup=True,
            )

            # Keyboard shortcuts info
            yield Rule()
            yield Static(
                "Shortcuts: Ctrl+G=Generate | Ctrl+R=Random | Ctrl+L=Clear | Ctrl+P=Play | Ctrl+S=Stop",
                classes="tip-text",
            )

    def on_mount(self) -> None:
        """Load provider descriptors and only the selected provider catalog."""
        self._rehydrate_handler_state()
        if self._profile_preset is not None:
            self._prime_profile_preset_controls()
            self.query_one("#tts-text-input", TextArea).focus()
        else:
            self._sync_profile_preview_status()
        self._load_provider_catalog(initialize=True)

    async def on_unmount(self) -> None:
        """Clean up resources when widget is unmounted"""
        modal = self._active_profile_name_modal
        if modal is not None:
            self._dismiss_profile_name_modal(modal)
        self._active_profile_name_modal = None
        try:
            self.app.workers.cancel_group(self, "stts-catalog-discovery")
            self.app.workers.cancel_group(self, "stts-voice-discovery")
            self.app.workers.cancel_group(self, "stts-playback")
            # Cancel any active progress timer
            if self._progress_timer_task and not self._progress_timer_task.done():
                self._progress_timer_task.cancel()
                await asyncio.sleep(0.05)

            # Cancel any active play worker
            if (
                hasattr(self, "_play_worker_task")
                and self._play_worker_task
                and not self._play_worker_task.is_finished
            ):
                self._play_worker_task.cancel()
                await asyncio.sleep(0.05)

            # Stop audio playback if active
            if hasattr(self.app, "audio_player"):
                await self.app.audio_player.stop()
                self._release_playback_artifact()
            else:
                self._release_playback_artifact()

            logger.debug("TTSPlaygroundWidget cleanup completed")
        except Exception as e:
            logger.error(f"Error during TTSPlaygroundWidget cleanup: {e}")

    def _is_valid_voice(self, voice: object) -> bool:
        """Check if a voice value is valid (not a separator)."""
        return bool(voice) and not str(voice).startswith("_separator")

    def _load_provider_catalog(
        self,
        provider_id: str | None = None,
        *,
        refresh: bool = False,
        initialize: bool = False,
    ) -> None:
        """Reserve request identity before starting exclusive catalog work."""
        target = provider_id or self._selected_provider_id
        preset = self._profile_preset
        if preset is not None and target == preset.provider_id:
            self._profile_preview_loading = True
            self._sync_profile_preview_status()
        request_generation = (
            self._reserve_catalog_request(target) if isinstance(target, str) else None
        )
        self._load_provider_catalog_worker(
            provider_id,
            refresh=refresh,
            initialize=initialize,
            request_generation=request_generation,
        )

    def _reserve_catalog_request(self, provider_id: str) -> int:
        """Reserve and return the next catalog request generation."""
        generation = self._catalog_request_generations.get(provider_id, 0) + 1
        self._catalog_request_generations[provider_id] = generation
        return generation

    @work(
        exclusive=True,
        group="stts-catalog-discovery",
        exit_on_error=False,
    )
    async def _load_provider_catalog_worker(
        self,
        provider_id: str | None = None,
        *,
        refresh: bool = False,
        initialize: bool = False,
        request_generation: int | None = None,
    ) -> None:
        """Load descriptors and one selected provider catalog."""
        token: CatalogRequestToken | None = None
        profile_voice_token: CatalogRequestToken | None = None
        try:
            if self._tts_service is None:
                self._tts_service = await get_tts_service()

            service = self._tts_service
            if initialize:
                descriptors = service.provider_descriptors()
                options = provider_options(descriptors)
                if not options:
                    self._profile_preview_loading = False
                    self._set_provider_status("No TTS providers are registered")
                    return
                self._provider_ids = frozenset(value for _label, value in options)
                self._provider_display_names = {
                    value: label for label, value in options
                }
                provider_select = self.query_one("#tts-provider-select", Select)
                provider_select.set_options(self._safe_select_options(options))
                provider_select.disabled = False
                configured_default = get_cli_setting(
                    "app_tts",
                    "default_provider",
                    options[0][1],
                )
                preset_provider = (
                    self._profile_preset.provider_id
                    if self._profile_preset is not None
                    else None
                )
                selected = options[0][1]
                if configured_default in self._provider_ids:
                    selected = configured_default
                if preset_provider in self._provider_ids:
                    selected = preset_provider
                self._selected_provider_id = selected
                self._applying_catalog_controls = True
                try:
                    provider_select.value = selected
                finally:
                    self._applying_catalog_controls = False
                self.query_one("#tts-refresh-catalog-btn", Button).disabled = False
                self._show_provider_specific_controls(selected)
                provider_id = selected

            if provider_id is None:
                provider_id = self._selected_provider_id
            if provider_id is None or provider_id not in getattr(
                self, "_provider_ids", ()
            ):
                self._profile_preview_loading = False
                self._sync_profile_preview_status()
                return

            configuration_revision = service.configuration_revision(provider_id)
            if request_generation is None:
                request_generation = self._reserve_catalog_request(provider_id)
            token = CatalogRequestToken(
                provider_id=provider_id,
                configuration_revision=configuration_revision,
                request_generation=request_generation,
            )
            preset = self._profile_preset
            if preset is not None and preset.provider_id == provider_id:
                self._profile_configuration_revision = configuration_revision
            self._set_provider_status("Loading selected provider models…")
            catalog = await service.get_catalog(provider_id, refresh=refresh)
            if not self._catalog_token_is_current(token):
                if self._catalog_request_is_latest(token):
                    self._mark_stale_catalog_result(token)
                return
            if catalog.provider_id != provider_id:
                self._catalog_failure(
                    provider_id,
                    "The selected provider returned an incompatible catalog",
                )
                return

            self._profile_preview_loading = False
            previous_catalog = self._catalogs.get(provider_id)
            if (
                previous_catalog is not None
                and previous_catalog.revision != catalog.revision
            ):
                self._discovered_voices = {
                    key: value
                    for key, value in self._discovered_voices.items()
                    if key[0] != provider_id
                }
            self._catalogs[provider_id] = catalog
            self._catalog_configuration_revisions[provider_id] = configuration_revision
            self._stale_providers.discard(provider_id)
            preset = self._profile_preset
            if preset is not None and preset.provider_id == provider_id:
                self._profile_effective_availability = (
                    profile_availability_from_catalog(preset, catalog)
                )
            if (
                preset is not None
                and preset.provider_id == provider_id
                and preset.voice_id is not None
                and self._profile_effective_availability != "unavailable"
            ):
                profile_voice_token = self._reserve_voice_request_token(
                    provider_id,
                    preset.model_id,
                    catalog.revision,
                )
                self._profile_voice_validation_token = profile_voice_token
            self._apply_catalog(provider_id, catalog)
            if (
                preset is not None
                and preset.provider_id == provider_id
                and catalog.health.state == "closed"
            ):
                self._stale_providers.add(provider_id)
                self._catalog_generation_allowed = False
                if self._profile_effective_availability != "unavailable":
                    self._set_provider_status("The TTS service is unavailable")
                self._sync_generate_enabled()
                if profile_voice_token is not None:
                    self._clear_profile_voice_validation(profile_voice_token)
                return
            if (
                preset is not None
                and preset.provider_id == provider_id
                and (
                    self._profile_effective_availability == "unavailable"
                    or preset.voice_id is None
                )
            ):
                return

            model_id = self._current_select_value("#tts-model-select")
            if isinstance(model_id, str):
                self._load_provider_voices(
                    provider_id,
                    model_id,
                    catalog.revision,
                    refresh=refresh,
                    request_token=profile_voice_token,
                )
            elif profile_voice_token is not None:
                self._clear_profile_voice_validation(profile_voice_token)
        except asyncio.CancelledError:
            if profile_voice_token is not None:
                self._clear_profile_voice_validation(profile_voice_token)
            raise
        except Exception as error:
            if profile_voice_token is not None:
                self._clear_profile_voice_validation(profile_voice_token)
            target = provider_id or self._selected_provider_id
            if token is not None and not self._catalog_token_is_current(token):
                if self._catalog_request_is_latest(token):
                    self._mark_stale_catalog_result(token)
                return
            if target is not None:
                exact_attempt_allowed = (
                    self._tts_service is not None
                    and not isinstance(
                        error,
                        TTSRegistryClosedError,
                    )
                    and not (
                        isinstance(error, TTSOperationError)
                        and error.code in {"configuration_invalid", "not_configured"}
                    )
                )
                self._catalog_failure(
                    target,
                    self._catalog_error_copy(error, target),
                    exact_attempt_allowed=exact_attempt_allowed,
                )

    def _load_provider_voices(
        self,
        provider_id: str,
        model_id: str,
        catalog_revision: int,
        *,
        refresh: bool = False,
        request_token: CatalogRequestToken | None = None,
    ) -> None:
        """Reserve request identity before starting exclusive voice work."""
        token = request_token or self._reserve_voice_request_token(
            provider_id,
            model_id,
            catalog_revision,
        )
        preset = self._profile_preset
        if (
            preset is not None
            and preset.provider_id == provider_id
            and preset.model_id == model_id
            and preset.voice_id is not None
        ):
            self._profile_voice_validation_token = token
            self._sync_profile_preview_status()
            self._sync_generate_enabled()
        self._load_provider_voices_worker(
            provider_id,
            model_id,
            catalog_revision,
            refresh=refresh,
            request_token=token,
        )

    def _reserve_voice_request_token(
        self,
        provider_id: str,
        model_id: str,
        catalog_revision: int,
    ) -> CatalogRequestToken:
        """Reserve one voice request and capture its catalog authority."""
        request_key = (provider_id, model_id)
        request_generation = self._voice_request_generations.get(request_key, 0) + 1
        self._voice_request_generations[request_key] = request_generation
        configuration_revision = self._catalog_configuration_revisions.get(provider_id)
        if configuration_revision is None:
            service = self._tts_service
            if service is None:
                raise TTSRegistryClosedError("The TTS service is unavailable")
            configuration_revision = service.configuration_revision(provider_id)
        return CatalogRequestToken(
            provider_id=provider_id,
            configuration_revision=configuration_revision,
            catalog_revision=catalog_revision,
            model_id=model_id,
            request_generation=request_generation,
        )

    @work(
        exclusive=True,
        group="stts-voice-discovery",
        exit_on_error=False,
    )
    async def _load_provider_voices_worker(
        self,
        provider_id: str,
        model_id: str,
        catalog_revision: int,
        *,
        refresh: bool = False,
        request_token: CatalogRequestToken,
    ) -> None:
        """Load voices for only the selected provider model."""
        try:
            service = self._tts_service
            if service is None:
                self._clear_profile_voice_validation(request_token)
                return
            observation: TTSVoiceDiscoveryResult | None = None
            preset = self._profile_preset
            observe_voices = getattr(service, "observe_voices", None)
            if (
                preset is not None
                and preset.provider_id == provider_id
                and provider_id == AUDIO_CPP_PROVIDER_ID
                and callable(observe_voices)
            ):
                observation = await observe_voices(
                    provider_id,
                    model_id,
                    refresh=refresh,
                )
                if (
                    type(observation) is not TTSVoiceDiscoveryResult
                    or observation.provider_id != provider_id
                    or observation.model_id != model_id
                    or observation.catalog_revision != catalog_revision
                ):
                    raise ValueError(
                        "The selected provider returned incompatible voice metadata"
                    )
                voices = observation.voices if observation.state == "complete" else ()
            else:
                voices = await service.get_voices(
                    provider_id,
                    model_id,
                    refresh=refresh,
                )
        except asyncio.CancelledError:
            self._clear_profile_voice_validation(request_token)
            raise
        except Exception as error:
            self._clear_profile_voice_validation(request_token)
            if not self._voice_token_is_current(request_token):
                return
            if isinstance(
                error,
                (TTSProviderReconfiguringError, TTSRegistryClosedError),
            ):
                if provider_id == self._selected_provider_id:
                    preset = self._profile_preset
                    if preset is not None and preset.provider_id == provider_id:
                        if self._profile_effective_availability != "unavailable":
                            self._profile_effective_availability = "unverified"
                        if (
                            isinstance(error, TTSProviderReconfiguringError)
                            and self._profile_effective_availability != "unavailable"
                        ):
                            self._stale_providers.add(provider_id)
                            self._catalog_generation_allowed = True
                            self._set_provider_status(
                                "Profile availability is unverified. Generate makes "
                                "one exact attempt without fallback and shows a warning."
                            )
                            self._sync_generate_enabled()
                            return
                        self._stale_providers.add(provider_id)
                        self._catalog_generation_allowed = False
                        if self._profile_effective_availability == "unavailable":
                            self._set_provider_status(
                                "The exact profile selection is unavailable. Return "
                                "to Voice profiles and choose Edit."
                            )
                        else:
                            self._set_provider_status(
                                self._catalog_error_copy(error, provider_id)
                            )
                        self._sync_generate_enabled()
                        return
                    self._stale_providers.add(provider_id)
                    self._catalog_generation_allowed = False
                    self._set_provider_status(
                        self._catalog_error_copy(error, provider_id)
                    )
                    self._sync_generate_enabled()
                return
            logger.warning(
                "TTS voice discovery failed ({})",
                type(error).__name__,
            )
            self._discovered_voices[(provider_id, model_id)] = ()
            self._pending_voice_selections.pop(provider_id, None)
            self._provider_control_snapshots.setdefault(provider_id, {})["voice_id"] = (
                SERVER_DEFAULT_VOICE_ID
            )
            catalog = self._catalogs.get(provider_id)
            preset = self._profile_preset
            if (
                preset is not None
                and preset.provider_id == provider_id
                and self._profile_effective_availability != "unavailable"
            ):
                self._profile_effective_availability = "unverified"
            if catalog is not None:
                self._apply_catalog(provider_id, catalog)
            if preset is not None and preset.provider_id == provider_id:
                if self._profile_effective_availability != "unavailable":
                    self._set_provider_status(
                        "Exact profile voice discovery is unverified; "
                        "the exact selection remains selected without fallback."
                    )
            else:
                self._set_provider_status(
                    "Voices are unavailable; the provider default remains available"
                )
            return

        if not self._voice_token_is_current(request_token):
            self._clear_profile_voice_validation(request_token)
            return
        self._discovered_voices[(provider_id, model_id)] = tuple(voices)
        catalog = self._catalogs.get(provider_id)
        preset = self._profile_preset
        if preset is not None and preset.provider_id == provider_id:
            if observation is not None:
                if observation.state == "unverified":
                    if self._profile_effective_availability != "unavailable":
                        self._profile_effective_availability = "unverified"
                elif observation.state == "model_missing":
                    self._profile_effective_availability = "unavailable"
                elif (
                    preset.voice_id is not None
                    and preset.voice_id not in observation.voices
                ):
                    self._profile_effective_availability = "unavailable"
                elif catalog is not None:
                    self._profile_effective_availability = (
                        profile_availability_from_catalog(preset, catalog)
                    )
            elif catalog is not None:
                self._profile_effective_availability = (
                    profile_availability_from_catalog(preset, catalog)
                )
        if catalog is not None:
            self._apply_catalog(provider_id, catalog)
        self._clear_profile_voice_validation(request_token)

    def _clear_profile_voice_validation(
        self,
        request_token: CatalogRequestToken,
    ) -> None:
        """Clear only the pending exact-profile observation owned by a token."""
        if self._profile_voice_validation_token != request_token:
            return
        self._profile_voice_validation_token = None
        if self.is_mounted:
            self._sync_profile_preview_status()
            self._sync_generate_enabled()

    def _voice_token_is_current(self, token: CatalogRequestToken) -> bool:
        """Return whether a voice result still targets the displayed model."""
        service = self._tts_service
        if service is None or not self.is_mounted:
            return False
        catalog = self._catalogs.get(token.provider_id)
        current_revision = catalog.revision if catalog is not None else None
        selected_model = self._current_select_value("#tts-model-select")
        current_model = selected_model if isinstance(selected_model, str) else None
        try:
            configuration_revision = service.configuration_revision(token.provider_id)
        except (KeyError, TTSRegistryClosedError):
            return False
        return token.matches(
            provider_id=self._selected_provider_id or "",
            configuration_revision=configuration_revision,
            catalog_revision=current_revision,
            model_id=current_model,
            request_generation=self._voice_request_generations.get(
                (token.provider_id, token.model_id or "")
            ),
        )

    def _catalog_token_is_current(self, token: CatalogRequestToken) -> bool:
        service = self._tts_service
        if service is None:
            return False
        try:
            configuration_revision = service.configuration_revision(token.provider_id)
        except (KeyError, TTSRegistryClosedError):
            return False
        return token.matches(
            provider_id=self._selected_provider_id or "",
            configuration_revision=configuration_revision,
            catalog_revision=None,
            model_id=None,
            request_generation=self._catalog_request_generations.get(token.provider_id),
        )

    def _catalog_request_is_latest(self, token: CatalogRequestToken) -> bool:
        """Return whether a catalog token is still its provider's newest request."""
        return token.request_generation == self._catalog_request_generations.get(
            token.provider_id
        )

    def _mark_stale_catalog_result(self, token: CatalogRequestToken) -> None:
        if token.provider_id != self._selected_provider_id:
            return
        self._profile_preview_loading = False
        self._stale_providers.add(token.provider_id)
        self._catalog_generation_allowed = False
        preset = self._profile_preset
        if preset is not None and preset.provider_id == token.provider_id:
            if preset.availability != "unavailable":
                self._profile_effective_availability = "unverified"
            self._project_profile_preset_controls(
                token.provider_id,
                generation_allowed=False,
            )
        display_name = self._provider_display_name(token.provider_id)
        self._set_provider_status(f"{display_name} settings changed; refresh models")
        self._sync_generate_enabled()

    def _project_profile_preset_controls(
        self,
        provider_id: str,
        *,
        generation_allowed: bool,
    ) -> bool:
        """Project exact preset controls even when no catalog was acquired."""
        preset = self._profile_preset
        if preset is None or preset.provider_id != provider_id:
            return False
        controls = controls_from_profile_preset(
            self._catalogs.get(provider_id),
            preset=preset,
            discovered_voices=self._discovered_voices.get(
                (provider_id, preset.model_id)
            ),
        )
        self._apply_controls(replace(controls, generation_allowed=generation_allowed))
        return True

    def _prime_profile_preset_controls(self) -> None:
        """Show one exact preset disabled before service discovery completes."""
        preset = self._profile_preset
        if preset is None:
            return
        provider_id = preset.provider_id
        display_name = (
            "audio.cpp" if provider_id == AUDIO_CPP_PROVIDER_ID else provider_id
        )
        self._selected_provider_id = provider_id
        self._provider_ids = frozenset((provider_id,))
        self._provider_display_names = {provider_id: display_name}
        provider_select = self.query_one("#tts-provider-select", Select)
        provider_select.set_options(
            self._safe_select_options(((display_name, provider_id),))
        )
        self._applying_catalog_controls = True
        try:
            provider_select.value = provider_id
        finally:
            self._applying_catalog_controls = False
        provider_select.disabled = True
        self.query_one("#tts-refresh-catalog-btn", Button).disabled = True
        self._show_provider_specific_controls(provider_id)
        self._project_profile_preset_controls(
            provider_id,
            generation_allowed=False,
        )

    def _apply_catalog(
        self,
        provider_id: str,
        catalog: TTSProviderCatalog,
    ) -> None:
        if provider_id != self._selected_provider_id:
            return
        snapshot = self._control_snapshot_for(provider_id)
        preset = self._profile_preset
        if preset is not None and preset.provider_id != provider_id:
            preset = None
        if preset is not None:
            selected_model: object = preset.model_id
            selected_voice: object = preset.voice_id
            selected_format: object = preset.response_format
            speed = preset.speed
        else:
            selected_model = snapshot.get("model_id")
            if selected_model is None:
                if provider_id == AUDIO_CPP_PROVIDER_ID:
                    configured_model = get_cli_setting(
                        "app_tts",
                        "default_model",
                        None,
                    )
                    selected_model = (
                        configured_model
                        if isinstance(configured_model, str) and configured_model
                        else None
                    )
                else:
                    selected_model = LEGACY_DEFAULT_MODELS.get(provider_id)
            selected_voice = snapshot.get("voice_id")
            if selected_voice is None:
                if provider_id == AUDIO_CPP_PROVIDER_ID:
                    configured_voice = get_cli_setting(
                        "app_tts",
                        "default_voice",
                        None,
                    )
                    selected_voice = (
                        configured_voice
                        if isinstance(configured_voice, str) and configured_voice
                        else None
                    )
                else:
                    selected_voice = LEGACY_DEFAULT_VOICES.get(provider_id)
            selected_format = snapshot.get("response_format")
            if selected_format is None:
                selected_format = get_cli_setting(
                    "app_tts",
                    "default_format",
                    None,
                )
            speed = self._snapshot_speed(snapshot)
        pending_voice = self._pending_voice_selections.get(provider_id)
        if pending_voice is not None and preset is None:
            selected_voice = pending_voice

        voice_choices: tuple[tuple[str, SelectValue], ...] | None = None
        discovered_voices: tuple[str, ...] | None
        if provider_id == AUDIO_CPP_PROVIDER_ID:
            model_for_voices = (
                preset.model_id
                if preset is not None
                else self._catalog_model_id(catalog, selected_model)
            )
            discovered_voices = (
                self._discovered_voices.get((provider_id, model_for_voices))
                if model_for_voices is not None
                else None
            )
            voice_discovery_pending = discovered_voices is None
            if (
                voice_discovery_pending
                and preset is None
                and isinstance(selected_voice, str)
                and selected_voice
            ):
                pending_voice = selected_voice
                self._pending_voice_selections[provider_id] = selected_voice
        else:
            model_for_voices = self._catalog_model_id(catalog, selected_model)
            base_voices = self._catalog_model_voices(catalog, model_for_voices)
            voice_choices = self._legacy_voice_choices(provider_id, base_voices)
            discovered_voices = tuple(value for _label, value in voice_choices)
            voice_discovery_pending = False

        if preset is not None:
            controls = controls_from_profile_preset(
                catalog,
                preset=preset,
                discovered_voices=discovered_voices,
            )
        else:
            controls = controls_from_catalog(
                catalog,
                selected_model_id=(
                    selected_model if isinstance(selected_model, str) else None
                ),
                selected_voice_id=(
                    selected_voice
                    if isinstance(selected_voice, (str, SelectSentinel))
                    else None
                ),
                discovered_voices=discovered_voices,
                selected_format=(
                    selected_format if isinstance(selected_format, str) else None
                ),
                speed=speed,
            )
        if voice_choices is not None:
            controls = replace(controls, voice_options=voice_choices)
        if preset is None and voice_discovery_pending and pending_voice is not None:
            model_changed = (
                selected_model is not None
                and selected_model != controls.selected_model_id
            )
            controls = replace(controls, selection_changed=model_changed)
        self._apply_controls(controls)
        if preset is None and voice_discovery_pending and pending_voice is not None:
            self._provider_control_snapshots.setdefault(provider_id, {})["voice_id"] = (
                pending_voice
            )
            self._catalog_generation_allowed = False
            self._sync_generate_enabled()
        elif provider_id == AUDIO_CPP_PROVIDER_ID and discovered_voices is not None:
            self._pending_voice_selections.pop(provider_id, None)

    def _apply_controls(self, controls: PlaygroundControls) -> None:
        model_select = self.query_one("#tts-model-select", Select)
        voice_select = self.query_one("#tts-voice-select", Select)
        format_select = self.query_one("#tts-format-select", Select)
        speed_input = self.query_one("#tts-speed-input", Input)
        self._applied_model_id = controls.selected_model_id
        self._applied_voice_id = controls.selected_voice_id
        self._applied_format = controls.selected_format
        self._applying_catalog_controls = True
        try:
            self._set_select_state(
                model_select,
                controls.model_options,
                controls.selected_model_id,
                "No models available",
            )
            self._set_select_state(
                voice_select,
                controls.voice_options,
                controls.selected_voice_id,
                "No voices available",
            )
            format_options = tuple(
                (audio_format.upper(), audio_format)
                for audio_format in controls.format_options
            )
            self._set_select_state(
                format_select,
                format_options,
                controls.selected_format,
                "No formats available",
            )
            format_select.disabled = controls.format_locked
            speed_input.value = str(controls.speed)
            speed_input.disabled = controls.speed_locked
        finally:
            self._applying_catalog_controls = False

        restriction = self.query_one("#tts-audio-cpp-restrictions", Static)
        if controls.provider_id == AUDIO_CPP_PROVIDER_ID:
            restriction.remove_class("hidden")
            format_select.tooltip = "audio.cpp returns one complete WAV response"
            speed_input.tooltip = "audio.cpp currently supports speed 1.0"
        else:
            restriction.add_class("hidden")
            format_select.tooltip = None
            speed_input.tooltip = None

        catalog = self._catalogs.get(controls.provider_id)
        self._displayed_provider_id = controls.provider_id
        preset = self._profile_preset
        if preset is not None and preset.provider_id != controls.provider_id:
            preset = None
        if preset is not None:
            model_select.add_class("profile-exact-select")
            voice_select.add_class("profile-exact-select")
            model_select.tooltip = Text(preset.model_id)
            voice_select.tooltip = Text(
                preset.voice_id
                if preset.voice_id is not None
                else SERVER_DEFAULT_VOICE_LABEL
            )
            availability = self._profile_effective_availability
            self._catalog_generation_allowed = bool(
                controls.generation_allowed and availability != "unavailable"
            )
            if availability == "unavailable":
                self._set_provider_status(
                    "The exact profile selection is unavailable. Return to Voice "
                    "profiles and choose Edit."
                )
            elif availability == "unverified":
                self._set_provider_status(
                    "Profile availability is unverified. Generate makes one exact "
                    "attempt without fallback and shows a warning."
                )
            else:
                self._set_provider_status(
                    "Profile preview loaded with its exact persisted selection."
                )
        else:
            model_select.remove_class("profile-exact-select")
            voice_select.remove_class("profile-exact-select")
            model_select.tooltip = None
            voice_select.tooltip = None
            service = self._tts_service
            self._catalog_generation_allowed = (
                controls.generation_allowed
                and service is not None
                and catalog is not None
                and controls.provider_id not in self._stale_providers
                and self._catalog_configuration_revisions.get(controls.provider_id)
                == service.configuration_revision(controls.provider_id)
            )
            if catalog is not None:
                self._set_provider_status(self._catalog_health_copy(catalog))
        self._remember_current_controls(controls.provider_id)
        if preset is not None:
            self._profile_controls_applied = True
        self._sync_generate_enabled()
        if controls.selection_changed:
            self.app.notify(
                "Available models or voices changed; a valid selection was chosen",
                severity="warning",
            )

    @staticmethod
    def _safe_select_options(
        options: tuple[tuple[str, SelectValue], ...],
    ) -> list[tuple[Text, SelectValue]]:
        return [(Text(label, no_wrap=True), value) for label, value in options]

    def _set_select_state(
        self,
        select: Select,
        options: tuple[tuple[str, SelectValue], ...],
        selected: SelectValue | None,
        empty_label: str,
    ) -> None:
        if not options:
            select.set_options([(empty_label, UNAVAILABLE_SELECT_VALUE)])
            select.value = UNAVAILABLE_SELECT_VALUE
            select.disabled = True
            return
        select.set_options(self._safe_select_options(options))
        select.disabled = False
        select.value = selected or options[0][1]

    def _control_snapshot_for(self, provider_id: str) -> dict[str, Any]:
        if getattr(self, "_displayed_provider_id", None) == provider_id:
            self._remember_current_controls(provider_id)
        return dict(self._provider_control_snapshots.get(provider_id, {}))

    def _remember_current_controls(self, provider_id: str) -> None:
        if getattr(self, "_displayed_provider_id", None) != provider_id:
            return
        speed_value = self.query_one("#tts-speed-input", Input).value
        try:
            speed = float(speed_value)
        except ValueError:
            speed = 1.0
        self._provider_control_snapshots[provider_id] = {
            "model_id": self._current_select_value("#tts-model-select"),
            "voice_id": self._current_select_value("#tts-voice-select"),
            "response_format": self._current_select_value("#tts-format-select"),
            "speed": speed,
        }

    @staticmethod
    def _snapshot_speed(snapshot: Mapping[str, Any]) -> float:
        speed = snapshot.get("speed", 1.0)
        try:
            return float(speed)
        except (TypeError, ValueError):
            return 1.0

    def _current_select_value(self, selector: str) -> SelectValue | None:
        value = self.query_one(selector, Select).value
        if value is LOADING_SELECT_VALUE or value is UNAVAILABLE_SELECT_VALUE:
            return None
        return value if isinstance(value, (str, SelectSentinel)) else None

    @staticmethod
    def _catalog_model_id(
        catalog: TTSProviderCatalog,
        selected_model_id: object,
    ) -> str | None:
        if isinstance(selected_model_id, str) and any(
            model.model_id == selected_model_id for model in catalog.models
        ):
            return selected_model_id
        return catalog.models[0].model_id if catalog.models else None

    @staticmethod
    def _catalog_model_voices(
        catalog: TTSProviderCatalog,
        model_id: str | None,
    ) -> tuple[str, ...]:
        for model in catalog.models:
            if model.model_id == model_id:
                return model.voices
        return ()

    def _legacy_voice_choices(
        self,
        provider_id: str,
        base_voices: tuple[str, ...],
    ) -> tuple[tuple[str, str], ...]:
        configured_choices = LEGACY_VOICE_OPTIONS.get(provider_id)
        choices = (
            list(configured_choices)
            if configured_choices is not None
            else [(voice.replace("_", " ").title(), voice) for voice in base_voices]
        )
        if provider_id == "chatterbox":
            choices.extend(self._chatterbox_profile_choices())
        elif provider_id == "higgs":
            choices.extend(self._higgs_profile_choices())
        elif provider_id == "kokoro":
            choices.extend(self._kokoro_blend_choices())
        return tuple(choices)

    @staticmethod
    def _kokoro_blend_choices() -> list[tuple[str, str]]:
        blend_file = Path.home() / ".config" / "tldw_cli" / "kokoro_voice_blends.json"
        if not blend_file.is_file():
            return []
        try:
            payload = json.loads(blend_file.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            logger.warning("Saved Kokoro voice blends could not be loaded")
            return []
        if not isinstance(payload, Mapping):
            return []
        return [
            (f"Voice blend: {name}", f"blend:{name}")
            for name in payload
            if isinstance(name, str) and name
        ]

    @staticmethod
    def _chatterbox_profile_choices() -> list[tuple[str, str]]:
        try:
            from tldw_chatbook.TTS.backends.chatterbox_voice_manager import (
                ChatterboxVoiceManager,
            )

            voice_dir = Path.home() / ".config" / "tldw_cli" / "chatterbox_voices"
            if not voice_dir.is_dir():
                return []
            profiles = ChatterboxVoiceManager(voice_dir).list_profiles()
            return [
                (str(profile.get("display_name") or profile["name"]), profile["name"])
                for profile in profiles
                if isinstance(profile, Mapping)
                and isinstance(profile.get("name"), str)
                and profile["name"]
            ]
        except Exception:
            logger.warning("Saved Chatterbox voice profiles could not be loaded")
            return []

    @staticmethod
    def _higgs_profile_choices() -> list[tuple[str, str]]:
        try:
            from tldw_chatbook.TTS.backends.higgs_voice_manager import (
                HiggsVoiceProfileManager,
            )

            voice_dir = Path.home() / ".config" / "tldw_cli" / "higgs_voices"
            if not voice_dir.is_dir():
                return []
            profiles = HiggsVoiceProfileManager(voice_dir).list_profiles()
            return [
                (
                    str(profile.get("display_name") or profile["name"]),
                    f"profile:{profile['name']}",
                )
                for profile in profiles
                if isinstance(profile, Mapping)
                and isinstance(profile.get("name"), str)
                and profile["name"]
            ]
        except Exception:
            logger.warning("Saved Higgs voice profiles could not be loaded")
            return []

    def _catalog_health_copy(self, catalog: TTSProviderCatalog) -> str:
        display_name = self._provider_display_name(catalog.provider_id)
        if catalog.provider_id in self._stale_providers:
            return f"{display_name} settings changed; refresh models"
        health = catalog.health
        if health.state == "available" and health.fresh:
            return f"{display_name} is ready"
        if health.state == "available":
            return f"{display_name} catalog is stale; refresh models"
        if health.state == "not_configured":
            return f"{display_name} is not configured; open STTS Settings"
        if health.state == "reconfiguring":
            return f"{display_name} settings are being applied; retry shortly"
        if health.state == "closed":
            return "The TTS service is unavailable"
        return f"{display_name} is unavailable; check STTS Settings"

    def _provider_display_name(self, provider_id: str) -> str:
        return self._provider_display_names.get(provider_id, "TTS provider")

    def _catalog_error_copy(self, error: Exception, provider_id: str) -> str:
        display_name = self._provider_display_name(provider_id)
        if isinstance(error, TTSProviderReconfiguringError):
            return f"{display_name} settings are being applied; retry shortly"
        if isinstance(error, TTSRegistryClosedError):
            return "The TTS service is unavailable"
        if isinstance(error, TTSOperationError):
            if error.code in {"configuration_invalid", "not_configured"}:
                return f"{display_name} is not configured; open STTS Settings"
            if error.code == "contract_incompatible":
                return f"The configured {display_name} service is incompatible"
            return f"{display_name} is unavailable; check STTS Settings"
        if isinstance(error, ValueError):
            return f"{display_name} is not configured; open STTS Settings"
        return f"{display_name} is unavailable; check STTS Settings"

    def _catalog_failure(
        self,
        provider_id: str,
        copy: str,
        *,
        exact_attempt_allowed: bool = False,
    ) -> None:
        logger.warning("TTS catalog discovery failed for {}", provider_id)
        if provider_id != self._selected_provider_id:
            return
        self._profile_preview_loading = False
        preset = self._profile_preset
        if preset is not None and preset.provider_id == provider_id:
            if preset.availability != "unavailable":
                self._profile_effective_availability = "unverified"
            generation_allowed = bool(
                exact_attempt_allowed
                and self._profile_effective_availability != "unavailable"
            )
            self._stale_providers.add(provider_id)
            self._project_profile_preset_controls(
                provider_id,
                generation_allowed=generation_allowed,
            )
            if (
                not generation_allowed
                and self._profile_effective_availability != "unavailable"
            ):
                self._set_provider_status(copy)
            self._sync_generate_enabled()
            return
        self._stale_providers.add(provider_id)
        self._catalog_generation_allowed = False
        self._set_provider_status(copy)
        self._sync_generate_enabled()

    def _set_provider_status(self, copy: str) -> None:
        self.query_one("#tts-provider-status", Static).update(Text(copy))
        self._sync_profile_preview_status()

    def _sync_profile_preview_status(self) -> None:
        banner = self.query_one("#tts-profile-preview-status", Static)
        preset = self._profile_preset
        availability = self._profile_effective_availability
        if preset is None or availability is None:
            banner.add_class("hidden")
            banner.update("")
            return
        style_state = availability
        if availability == "unavailable":
            copy = (
                "Profile preview unavailable — return to Voice profiles and "
                "choose Edit."
            )
        elif (
            self._profile_preview_loading
            or self._profile_voice_validation_token is not None
        ):
            copy = "Profile preview loading — checking the exact saved selection."
            style_state = "loading"
        elif (
            blocked := self._profile_preview_blocked_presentation(preset)
        ) is not None:
            copy, style_state = blocked
        elif availability == "unverified":
            copy = (
                "Profile preview unverified — Generate makes one exact attempt "
                "without fallback."
            )
        else:
            copy = "Profile preview — exact saved selection."
        for state in ("loading", "available", "unverified", "unavailable"):
            banner.set_class(
                style_state == state,
                f"profile-preview-{state}",
            )
        banner.update(Text(copy))
        banner.remove_class("hidden")

    def _profile_preview_blocked_presentation(
        self,
        preset: TTSPlaygroundSelectionPreset,
    ) -> tuple[str, ProfileAvailabilityState] | None:
        """Return bounded recovery copy when the exact preset cannot generate."""
        service = self._tts_service
        if service is None:
            return (
                "Profile preview blocked — the TTS service is unavailable.",
                "unavailable",
            )
        catalog = self._catalogs.get(preset.provider_id)
        if catalog is not None and catalog.health.state == "closed":
            return (
                "Profile preview blocked — the TTS service is unavailable.",
                "unavailable",
            )
        try:
            current_revision = service.configuration_revision(preset.provider_id)
        except (KeyError, TTSRegistryClosedError):
            return (
                "Profile preview blocked — the TTS service is unavailable.",
                "unavailable",
            )
        expected_revision = self._profile_configuration_revision
        if expected_revision is None:
            return (
                "Profile preview blocked — refresh or retry from Voice profiles.",
                "unverified",
            )
        if current_revision != expected_revision:
            return (
                "Profile preview blocked — TTS settings changed; refresh models.",
                "unverified",
            )
        if not self._catalog_generation_allowed:
            return (
                "Profile preview blocked — refresh or retry from Voice profiles.",
                "unverified",
            )
        return None

    def _sync_generate_enabled(self) -> None:
        text_present = bool(self.query_one("#tts-text-input", TextArea).text.strip())
        provider_id = self._selected_provider_id
        revision_matches = False
        service = self._tts_service
        if provider_id is not None and service is not None:
            preset = self._profile_preset
            expected_revision = (
                self._profile_configuration_revision
                if preset is not None and preset.provider_id == provider_id
                else self._catalog_configuration_revisions.get(provider_id)
            )
            try:
                revision_matches = (
                    expected_revision is not None
                    and expected_revision == service.configuration_revision(provider_id)
                )
            except (KeyError, TTSRegistryClosedError):
                revision_matches = False
        self.query_one("#tts-generate-btn", Button).disabled = not (
            text_present
            and self._catalog_generation_allowed
            and revision_matches
            and (
                provider_id not in self._stale_providers
                or (
                    self._profile_preset is not None
                    and self._profile_preset.provider_id == provider_id
                )
            )
            and self._generation_operation_id is None
            and self._profile_voice_validation_token is None
            and not getattr(self.app, "_is_generating", False)
        )

    def _generation_readiness_error(
        self,
        provider_id: object,
        model_id: object,
    ) -> str | None:
        """Return fixed UI copy when a generation snapshot is not authoritative."""
        if self._generation_operation_id is not None:
            return "TTS generation is already in progress"

        handler = getattr(self.app, "_stts_handler", None)
        state_getter = getattr(handler, "playground_state", None)
        if callable(state_getter):
            try:
                if getattr(state_getter(), "generation_active", False):
                    return "TTS generation is already in progress"
            except Exception:
                return "The TTS service is unavailable"

        preset = self._profile_preset
        if preset is not None:
            if self._profile_voice_validation_token is not None:
                return (
                    "The exact profile voice is still being checked; "
                    "wait before generating"
                )
            if self._profile_effective_availability == "unavailable":
                return (
                    "The exact profile selection is unavailable; return to Voice "
                    "profiles and choose Edit"
                )
            if provider_id != preset.provider_id or model_id != preset.model_id:
                return "The exact profile selection changed; choose Preview again"
            service = self._tts_service
            if service is None:
                return "The TTS service is unavailable"
            try:
                current_revision = service.configuration_revision(preset.provider_id)
            except (KeyError, TTSRegistryClosedError):
                return "The TTS service is unavailable"
            if (
                self._profile_configuration_revision is None
                or current_revision != self._profile_configuration_revision
            ):
                return "TTS provider settings changed; refresh models"
            if not self._catalog_generation_allowed:
                return "The exact profile selection is not ready; retry from Voice profiles"
            return None

        if (
            not isinstance(provider_id, str)
            or provider_id != self._selected_provider_id
            or provider_id not in self._provider_ids
        ):
            return "Please select a valid TTS provider"
        if not isinstance(model_id, str):
            return "Please select a valid TTS model"

        service = self._tts_service
        catalog = self._catalogs.get(provider_id)
        if service is None or catalog is None:
            return "The selected provider catalog is not ready; refresh models"
        revision_matches = self._catalog_configuration_revisions.get(
            provider_id
        ) == service.configuration_revision(provider_id)
        if (
            provider_id in self._pending_voice_selections
            and provider_id not in self._stale_providers
            and catalog.health.state == "available"
            and catalog.health.fresh
            and revision_matches
        ):
            return "Voices are still loading; wait before generating"
        if (
            provider_id in self._stale_providers
            or not self._catalog_generation_allowed
            or catalog.health.state != "available"
            or not catalog.health.fresh
            or not revision_matches
        ):
            return "The selected provider catalog is stale; refresh models"
        if not any(model.model_id == model_id for model in catalog.models):
            return "The selected model is no longer available; refresh models"
        return None

    def _show_provider_specific_controls(self, provider_id: str) -> None:
        language_row = self.query_one("#kokoro-language-row", Horizontal)
        kokoro_settings = self.query_one("#kokoro-settings", Vertical)
        elevenlabs_settings = self.query_one("#elevenlabs-settings", Vertical)
        chatterbox_settings = self.query_one("#chatterbox-settings", Vertical)
        higgs_settings = self.query_one("#higgs-settings", Vertical)
        language_row.set_class(provider_id == "kokoro", "visible")
        kokoro_settings.set_class(provider_id == "kokoro", "visible")
        elevenlabs_settings.set_class(provider_id == "elevenlabs", "visible")
        chatterbox_settings.set_class(provider_id == "chatterbox", "visible")
        higgs_settings.set_class(provider_id == "higgs", "visible")
        if provider_id == "higgs":
            self._check_higgs_installation()

    def mark_provider_configuration_changed(
        self,
        provider_id: str,
        configuration_revision: int,
    ) -> None:
        """Invalidate cached controls after a changed provider configuration."""
        del configuration_revision
        self._stale_providers.add(provider_id)
        self._discovered_voices = {
            key: value
            for key, value in self._discovered_voices.items()
            if key[0] != provider_id
        }
        pending_voice_token = self._profile_voice_validation_token
        if (
            pending_voice_token is not None
            and pending_voice_token.provider_id == provider_id
        ):
            self._profile_voice_validation_token = None
        if provider_id != self._selected_provider_id:
            return
        self.app.workers.cancel_group(self, "stts-catalog-discovery")
        self.app.workers.cancel_group(self, "stts-voice-discovery")
        self._profile_preview_loading = False
        self._catalog_generation_allowed = False
        display_name = self._provider_display_name(provider_id)
        self._set_provider_status(f"{display_name} settings changed; refresh models")
        self._sync_generate_enabled()

    def _end_profile_preset(self, *, before_controls: bool = False) -> bool:
        """Detach exact profile semantics after a user selection edit."""
        if self._profile_preset is None:
            return False
        if not before_controls and not self._profile_controls_applied:
            return False
        self._profile_preset = None
        self._profile_effective_availability = None
        self._profile_preview_loading = False
        self._profile_configuration_revision = None
        self._profile_voice_validation_token = None
        self._profile_controls_applied = True
        self._sync_profile_preview_status()
        self._sync_generate_enabled()
        return True

    def _reproject_current_catalog(self) -> None:
        provider_id = self._selected_provider_id
        if provider_id is None:
            return
        catalog = self._catalogs.get(provider_id)
        if catalog is not None:
            self._apply_catalog(provider_id, catalog)

    @on(Select.Changed)
    def on_tts_provider_select_changed(self, event: Select.Changed) -> None:
        """Handle canonical provider/model/voice/format selections."""
        if self._applying_catalog_controls:
            return
        if event.value != event.select.value:
            return
        if event.select.id == "tts-provider-select":
            if not isinstance(event.value, str) or event.value not in getattr(
                self, "_provider_ids", ()
            ):
                return
            if event.value == self._selected_provider_id:
                return
            self._end_profile_preset(before_controls=True)
            if self._selected_provider_id is not None:
                self._remember_current_controls(self._selected_provider_id)
            self._selected_provider_id = event.value
            self._show_provider_specific_controls(event.value)
            self._catalog_generation_allowed = False
            self._sync_generate_enabled()
            self._load_provider_catalog(event.value)
            return
        if event.select.id == "tts-model-select":
            provider_id = self._selected_provider_id
            if provider_id is None or not isinstance(event.value, str):
                return
            if event.value == self._applied_model_id:
                return
            self._end_profile_preset()
            self._remember_current_controls(provider_id)
            catalog = self._catalogs.get(provider_id)
            if catalog is not None:
                self._apply_catalog(provider_id, catalog)
                model_id = self._current_select_value("#tts-model-select")
                if isinstance(model_id, str):
                    self._load_provider_voices(
                        provider_id,
                        model_id,
                        catalog.revision,
                    )
            return
        if event.select.id in {"tts-voice-select", "tts-format-select"}:
            if (
                event.select.id == "tts-voice-select"
                and event.value == self._applied_voice_id
            ) or (
                event.select.id == "tts-format-select"
                and event.value == self._applied_format
            ):
                return
            preset_ended = self._end_profile_preset()
            if event.select.id == "tts-voice-select":
                self._applied_voice_id = (
                    event.value
                    if isinstance(event.value, (str, SelectSentinel))
                    else None
                )
            else:
                self._applied_format = (
                    event.value if isinstance(event.value, str) else None
                )
            if self._selected_provider_id is not None:
                self._remember_current_controls(self._selected_provider_id)
            if preset_ended:
                self._reproject_current_catalog()
            else:
                self._sync_generate_enabled()
            return
        if event.select.has_focus and self._end_profile_preset():
            self._reproject_current_catalog()

    @on(Input.Changed)
    def on_tts_speed_changed(self, event: Input.Changed) -> None:
        if self._applying_catalog_controls:
            return
        if event.value != event.input.value:
            return
        if self._selected_provider_id is not None:
            if event.input.id == "tts-speed-input":
                self._remember_current_controls(self._selected_provider_id)
                preset = self._profile_preset
                try:
                    unchanged = (
                        preset is not None and float(event.value) == preset.speed
                    )
                except ValueError:
                    unchanged = False
                if unchanged:
                    return
            elif not event.input.has_focus:
                return
            if self._end_profile_preset():
                self._reproject_current_catalog()

    @on(Switch.Changed)
    def on_tts_option_switch_changed(self, event: Switch.Changed) -> None:
        if (
            not self._applying_catalog_controls
            and event.switch.has_focus
            and self._end_profile_preset()
        ):
            self._reproject_current_catalog()

    @on(TextArea.Changed)
    def on_tts_text_changed(self, _event: TextArea.Changed) -> None:
        self._sync_generate_enabled()

    def _get_select_key(self, select_widget: Select) -> SelectValue | None:
        """Return exact canonical values for catalog-driven controls."""
        current = select_widget.value
        if current is LOADING_SELECT_VALUE or current is UNAVAILABLE_SELECT_VALUE:
            return None
        if current is SERVER_DEFAULT_VOICE_ID:
            return current
        if not isinstance(current, str):
            return None
        if select_widget.id == "tts-language-select":
            for language_id, display_name in select_widget._options:
                if display_name == current:
                    return str(language_id)
        return current

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        logger.debug(f"TTSPlaygroundWidget received button press: {event.button.id}")
        if event.button.id == "tts-generate-btn":
            self._generate_tts()
            event.stop()  # Prevent event from bubbling up
        elif event.button.id == "tts-refresh-catalog-btn":
            if self._selected_provider_id is not None:
                self._load_provider_catalog(
                    self._selected_provider_id,
                    refresh=True,
                )
            event.stop()
        elif event.button.id == "tts-random-text-btn":
            self._insert_random_text()
            event.stop()
        elif event.button.id == "tts-clear-text-btn":
            self._clear_text()
            event.stop()
        elif event.button.id == "audio-play-btn":
            self._play_audio()
            event.stop()
        elif event.button.id == "pause-audio-btn":
            logger.debug("Pause button clicked")
            self._pause_audio()
            event.stop()
        elif event.button.id == "stop-audio-btn":
            logger.debug("Stop button clicked")
            self._stop_audio()
            event.stop()
        elif event.button.id == "audio-export-btn":
            self._export_audio()
            event.stop()
        elif event.button.id == "audio-save-profile-btn":
            event.button.disabled = True
            self.run_worker(
                self._save_current_result_as_profile(),
                name="save_tts_result_as_profile",
                group="save_tts_result_as_profile",
                exclusive=True,
                exit_on_error=False,
            )
            event.stop()
        elif event.button.id == "reference-audio-btn":
            self._select_reference_audio()
            event.stop()
        elif event.button.id == "clear-reference-audio-btn":
            self._clear_reference_audio()
            event.stop()
        elif event.button.id == "higgs-voice-upload-btn":
            self._upload_higgs_voice()
            event.stop()
        elif event.button.id == "higgs-clear-voice-btn":
            self._clear_higgs_voice()
            event.stop()

    def _generate_tts(self) -> None:
        """Generate TTS audio"""
        if self._generation_operation_id is not None:
            self.app.notify(
                "TTS generation is already in progress",
                severity="warning",
            )
            return

        # Get form values
        text_area = self.query_one("#tts-text-input", TextArea)
        text = text_area.text.strip()

        if not text:
            self.app.notify("Please enter text to synthesize", severity="warning")
            return

        provider_select = self.query_one("#tts-provider-select", Select)
        voice_select = self.query_one("#tts-voice-select", Select)
        model_select = self.query_one("#tts-model-select", Select)

        # Get the actual keys, not display text
        provider = self._get_select_key(provider_select) or provider_select.value
        voice = self._get_select_key(voice_select) or voice_select.value
        model = self._get_select_key(model_select) or model_select.value
        preset = self._profile_preset
        if preset is not None:
            provider = preset.provider_id
            model = preset.model_id
            voice = (
                SERVER_DEFAULT_VOICE_ID if preset.voice_id is None else preset.voice_id
            )

        readiness_error = self._generation_readiness_error(provider, model)
        if readiness_error is not None:
            self._sync_generate_enabled()
            self.app.notify(readiness_error, severity="warning")
            return
        if preset is not None and self._profile_effective_availability == "unverified":
            self.app.notify(
                "Profile availability is unverified; attempting the exact "
                "selection once without fallback.",
                severity="warning",
            )

        # Validate voice selection
        if not self._is_valid_voice(voice):
            self.app.notify("Please select a valid voice", severity="warning")
            return
        speed = float(self.query_one("#tts-speed-input", Input).value or "1.0")
        format_select = self.query_one("#tts-format-select", Select)
        format = format_select.value
        if preset is not None:
            speed = preset.speed
            format = preset.response_format

        # Ensure format has a valid value
        if not format or format == Select.BLANK or str(format) == "Select.BLANK":
            format = "mp3"
            logger.warning("No format selected, defaulting to mp3")
        elif isinstance(format, tuple):
            # If it's a tuple, take the first element
            format = format[0]

        # Additional validation - also handle uppercase
        valid_formats = ["mp3", "opus", "aac", "flac", "wav", "pcm"]
        format_lower = format.lower() if isinstance(format, str) else format
        if format_lower in valid_formats:
            format = format_lower
        else:
            logger.warning("Invalid Playground audio format; using mp3")
            format = "mp3"

        # Collect provider-specific settings
        extra_params = {}
        if provider == "kokoro":
            language_select = self.query_one("#tts-language-select", Select)
            language = self._get_select_key(language_select) or language_select.value
            extra_params["language"] = language
            # Add ONNX setting
            use_onnx = self.query_one("#tts-kokoro-use-onnx", Switch).value
            extra_params["use_onnx"] = use_onnx
        elif provider == "elevenlabs":
            stability = float(
                self.query_one("#tts-stability-input", Input).value or "0.5"
            )
            similarity = float(
                self.query_one("#tts-similarity-input", Input).value or "0.8"
            )
            style = float(self.query_one("#tts-style-input", Input).value or "0.0")
            speaker_boost = self.query_one("#tts-speaker-boost-switch", Switch).value
            extra_params["stability"] = stability
            extra_params["similarity_boost"] = similarity
            extra_params["style"] = style
            extra_params["use_speaker_boost"] = speaker_boost
        elif provider == "chatterbox":
            exaggeration = float(
                self.query_one("#tts-exaggeration-input", Input).value or "0.5"
            )
            cfg_weight = float(
                self.query_one("#tts-cfg-weight-input", Input).value or "0.5"
            )
            temperature = float(
                self.query_one("#tts-temperature-input", Input).value or "0.5"
            )
            num_candidates = int(
                self.query_one("#tts-num-candidates-input", Input).value or "1"
            )
            validate_whisper = self.query_one(
                "#tts-validate-whisper-switch", Switch
            ).value
            preprocess_text = self.query_one(
                "#tts-preprocess-text-switch", Switch
            ).value
            normalize_audio = self.query_one(
                "#tts-normalize-audio-switch", Switch
            ).value
            target_db = float(
                self.query_one("#tts-target-db-input", Input).value or "-20.0"
            )
            random_seed_input = self.query_one(
                "#tts-random-seed-input", Input
            ).value.strip()

            extra_params["exaggeration"] = exaggeration
            extra_params["cfg_weight"] = cfg_weight
            extra_params["temperature"] = temperature
            extra_params["num_candidates"] = num_candidates
            extra_params["validate_with_whisper"] = validate_whisper
            extra_params["preprocess_text"] = preprocess_text
            extra_params["normalize_audio"] = normalize_audio
            extra_params["target_db"] = target_db
            if random_seed_input:
                extra_params["random_seed"] = int(random_seed_input)

            # Handle voice selection
            if voice == "custom" and self.reference_audio_path:
                # Use custom voice with reference audio
                voice = f"custom:{self.reference_audio_path}"
            elif voice == "custom":
                self.app.notify(
                    "Please select reference audio for custom voice", severity="warning"
                )
                self.query_one("#tts-generate-btn", Button).disabled = False
                return
            elif voice not in [
                "default",
                "custom",
                "_separator",
                "_separator2",
            ] and not voice.startswith(("custom:", "profile:")):
                # This is a saved profile - format it as profile:name
                voice = f"profile:{voice}"
        elif provider == "higgs":
            # Collect Higgs-specific parameters
            temperature = float(
                self.query_one("#tts-higgs-temperature-input", Input).value
            )
            top_p = float(self.query_one("#tts-higgs-top-p-input", Input).value)
            repetition_penalty = float(
                self.query_one("#tts-higgs-repetition-penalty-input", Input).value
            )
            enable_voice_cloning = self.query_one(
                "#tts-higgs-voice-cloning-switch", Switch
            ).value
            enable_multi_speaker = self.query_one(
                "#tts-higgs-multi-speaker-switch", Switch
            ).value
            speaker_delimiter = self.query_one(
                "#tts-higgs-delimiter-input", Input
            ).value

            extra_params["temperature"] = temperature
            extra_params["top_p"] = top_p
            extra_params["repetition_penalty"] = repetition_penalty
            extra_params["enable_voice_cloning"] = enable_voice_cloning
            extra_params["enable_multi_speaker"] = enable_multi_speaker
            extra_params["speaker_delimiter"] = speaker_delimiter

            # Handle voice selection for custom upload
            if (
                voice == "custom"
                and hasattr(self, "higgs_reference_audio_path")
                and self.higgs_reference_audio_path
            ):
                # Use custom voice with reference audio
                voice = f"custom:{self.higgs_reference_audio_path}"
            elif voice == "custom":
                self.app.notify(
                    "Please upload reference audio for custom voice", severity="warning"
                )
                self.query_one("#tts-generate-btn", Button).disabled = False
                return
            elif voice not in [
                "professional_female",
                "warm_female",
                "storyteller_male",
                "deep_male",
                "energetic_female",
                "soft_female",
                "custom",
                "_separator",
                "_separator2",
            ] and not voice.startswith(("custom:", "profile:")):
                # This is a saved profile - format it as profile:name
                voice = f"profile:{voice}"

        # Log the request
        log = self.query_one("#tts-generation-log", RichLog)
        log.write("[bold blue]Generating TTS...[/bold blue]")
        log.write(f"Speed: {speed}")
        log.write(f"Format: {format}")
        log.write(f"Text length: {len(text)} characters")

        if not isinstance(provider, str) or provider not in self._provider_ids:
            self.app.notify("Please select a valid TTS provider", severity="warning")
            return
        if not isinstance(model, str):
            self.app.notify("Please select a valid TTS model", severity="warning")
            return
        if not isinstance(format, str):
            self.app.notify("Please select a valid audio format", severity="warning")
            return
        voice_id = voice_id_for_request(voice)
        if provider == AUDIO_CPP_PROVIDER_ID:
            format = "wav"
            speed = 1.0
            extra_params = {}

        # Disable generate button
        self.query_one("#tts-generate-btn", Button).disabled = True

        request = STTSPlaygroundRequest(
            operation_id=str(uuid4()),
            provider_id=provider,
            model_id=model,
            text=text,
            voice_id=voice_id,
            response_format=format,
            speed=speed,
            options=extra_params,
        )
        self._generation_operation_id = request.operation_id
        self._profile_save_suppressed = True
        self._sync_save_profile_action()
        self.app.post_message(STTSPlaygroundGenerateEvent(request))

    def _generation_complete(
        self,
        artifact: STTSGeneratedAudio | None,
    ) -> None:
        """Store one delivered artifact independently of current selectors."""
        if (
            artifact is not None
            and self._generation_operation_id is not None
            and artifact.operation_id != self._generation_operation_id
        ):
            return
        self._generation_operation_id = None
        self._sync_generate_enabled()

        if artifact is not None:
            self._store_delivered_artifact(artifact, announce=True)
        else:
            self._profile_save_suppressed = True
            self._sync_save_profile_action()
            log = self.query_one("#tts-generation-log", RichLog)
            log.write("[bold red]✗ TTS generation failed![/bold red]")

    def _store_delivered_artifact(
        self,
        artifact: STTSGeneratedAudio,
        *,
        announce: bool,
    ) -> None:
        self.current_audio_artifact = artifact
        self.current_audio_file = artifact.path
        self._profile_save_suppressed = False
        if announce:
            self.query_one("#tts-generation-log", RichLog).write(
                "[bold green]✓ TTS generation complete![/bold green]"
            )
        self.query_one("#audio-play-btn", Button).disabled = False
        self.query_one("#pause-audio-btn", Button).disabled = True
        self.query_one("#stop-audio-btn", Button).disabled = True
        self.query_one("#audio-export-btn", Button).disabled = False
        self._sync_save_profile_action()
        self.query_one("#audio-player-status", Static).update(
            f"{artifact.audio_format.upper()} audio ready to play"
        )

    def _sync_save_profile_action(self) -> None:
        """Expose save only for an idle artifact with native provenance."""
        button = self.query_one("#audio-save-profile-btn", Button)
        artifact = self.current_audio_artifact
        eligible = bool(
            artifact is not None
            and artifact.profile_save_eligible
            and self._generation_operation_id is None
            and not self._profile_save_suppressed
        )
        button.set_class(not eligible, "hidden")
        button.disabled = not eligible

    @staticmethod
    def _dismiss_profile_name_modal(modal: TTSProfileNameModal) -> None:
        if modal.is_mounted and modal.is_current:
            modal.dismiss(None)

    async def _save_current_result_as_profile(self) -> None:
        """Save a captured eligible artifact without rereading selectors."""
        artifact = self.current_audio_artifact
        if (
            artifact is None
            or not artifact.profile_save_eligible
            or self._generation_operation_id is not None
            or self._profile_save_suppressed
        ):
            self._sync_save_profile_action()
            return

        modal = TTSProfileNameModal()
        active = self._active_profile_name_modal
        if active is not None:
            self._dismiss_profile_name_modal(active)
        self._active_profile_name_modal = modal
        try:
            display_name = await self.app.push_screen_wait(modal)
        except asyncio.CancelledError:
            self._dismiss_profile_name_modal(modal)
            if self.is_mounted:
                self._sync_save_profile_action()
            raise
        except Exception:  # noqa: BLE001 - isolate modal lifecycle failure
            self._dismiss_profile_name_modal(modal)
            if self.is_mounted:
                self.query_one("#audio-player-status", Static).update(
                    PROFILE_ACTION_FAILED_COPY
                )
                self._sync_save_profile_action()
            return
        finally:
            if self._active_profile_name_modal is modal:
                self._active_profile_name_modal = None
        if not isinstance(display_name, str) or not display_name.strip():
            self._sync_save_profile_action()
            return

        ensure_service = getattr(self.app, "_ensure_tts_profile_service", None)
        if not callable(ensure_service):
            self.query_one("#audio-player-status", Static).update(
                PROFILE_STORE_UNAVAILABLE_COPY
            )
            self._sync_save_profile_action()
            return
        try:
            service = await ensure_service()
            if service is None:
                self.query_one("#audio-player-status", Static).update(
                    PROFILE_STORE_UNAVAILABLE_COPY
                )
                return
            await service.create_from_artifact(display_name, artifact)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - map to bounded UI copy
            copy = (
                _PROFILE_RESULT_STALE_COPY
                if getattr(error, "code", None) == "stale_configuration"
                else profile_action_error_copy(error)
            )
            self.query_one("#audio-player-status", Static).update(copy)
            return
        finally:
            if self.is_mounted:
                self._sync_save_profile_action()
        self.query_one("#audio-player-status", Static).update("Voice profile saved.")

    def _rehydrate_handler_state(self) -> None:
        handler = getattr(self.app, "_stts_handler", None)
        snapshot_getter = getattr(handler, "playground_state", None)
        if not callable(snapshot_getter):
            return
        try:
            state = snapshot_getter()
        except Exception as error:
            logger.debug(
                "Could not rehydrate TTS Playground state ({})",
                type(error).__name__,
            )
            return
        artifact = getattr(state, "artifact", None)
        if isinstance(artifact, STTSGeneratedAudio) and artifact.path.exists():
            self._store_delivered_artifact(artifact, announce=False)
        active_operation_id = getattr(state, "active_operation_id", None)
        if getattr(state, "generation_active", False) and isinstance(
            active_operation_id,
            str,
        ):
            self._generation_operation_id = active_operation_id
            self._profile_save_suppressed = True
            self.query_one("#generation-status-container").remove_class("hidden")
            self.query_one("#generation-status-text", Static).update(
                "Generation in progress…"
            )
            self.query_one("#tts-generate-btn", Button).disabled = True
            self._sync_save_profile_action()

    def _current_generated_audio_path(self) -> Path | None:
        """Return the delivered artifact path, with legacy path fallback."""
        if self.current_audio_artifact is not None:
            return self.current_audio_artifact.path
        if self.current_audio_file is None:
            return None
        return Path(self.current_audio_file)

    def _capture_audio_for_action(
        self,
    ) -> tuple[Path, Callable[[], None]] | None:
        """Capture and, when handler-owned, lease the current artifact."""
        artifact = self.current_audio_artifact
        audio_path = self._current_generated_audio_path()
        if audio_path is None:
            return None
        if artifact is None or artifact.path != audio_path:
            return audio_path, lambda: None

        handler = getattr(self.app, "_stts_handler", None)
        acquire = getattr(handler, "lease_playground_artifact", None)
        release = getattr(handler, "release_playground_artifact", None)
        if not callable(acquire) or not callable(release):
            return audio_path, lambda: None
        try:
            if not acquire(artifact):
                return None
        except Exception as error:
            logger.debug(
                "Could not lease Playground artifact ({})",
                type(error).__name__,
            )
            return None

        released = False

        def release_once() -> None:
            nonlocal released
            if released:
                return
            released = True
            try:
                release(artifact)
            except Exception as error:
                logger.debug(
                    "Could not release Playground artifact ({})",
                    type(error).__name__,
                )

        return audio_path, release_once

    def _release_playback_artifact(self) -> None:
        """Release the artifact retained by the active playback, if any."""
        release = self._active_playback_release
        self._active_playback_release = None
        if release is not None:
            release()

    def _play_audio(self) -> None:
        """Play the generated audio"""
        logger.debug(
            f"_play_audio called, current_audio_file: {self.current_audio_file}"
        )

        # Check if we're already playing
        if (
            hasattr(self, "_play_worker_task")
            and self._play_worker_task
            and not self._play_worker_task.is_finished
        ):
            logger.debug("Play already in progress, ignoring request")
            return

        captured = self._capture_audio_for_action()
        if captured is None:
            self.app.notify("No audio file to play", severity="warning")
            return
        audio_path, release_artifact = captured

        logger.debug(
            f"Audio path: {audio_path}, exists: {audio_path.exists() if audio_path else False}"
        )

        if not audio_path.exists():
            release_artifact()
            self.app.notify(
                f"Audio file not found: {audio_path.name}", severity="warning"
            )
            return

        if self._ensure_audio_player():
            # Cancel any existing progress timer first
            if self._progress_timer_task and not self._progress_timer_task.done():
                self._progress_timer_task.cancel()
                self._progress_timer_task = None
                logger.debug("Cancelled existing progress timer")

            # Enable pause and stop buttons
            self.query_one("#pause-audio-btn", Button).disabled = False
            self.query_one("#stop-audio-btn", Button).disabled = False
            self.query_one("#audio-player-status", Static).update("Playing...")

            # Use the new audio player method
            # Store the worker task so we can check if it's running
            playback = self._play_audio_async(audio_path, release_artifact)
            try:
                self._play_worker_task = self.run_worker(
                    playback,
                    group="stts-playback",
                    exclusive=True,
                )
            except Exception:
                playback.close()
                release_artifact()
                raise
        else:
            release_artifact()
            self.app.notify("Audio playback not available", severity="warning")

    async def _play_audio_async(
        self,
        audio_path: Path | None = None,
        release_artifact: Callable[[], None] | None = None,
    ) -> None:
        """Play audio asynchronously using the audio player"""
        try:
            if audio_path is None:
                audio_path = self._current_generated_audio_path()
            if audio_path is not None:
                if audio_path.exists():
                    # Get current player state before stopping
                    current_state = await self.app.audio_player.get_state()
                    logger.debug(f"Current player state before play: {current_state}")

                    # Always force stop any existing playback first
                    stop_result = await self.app.audio_player.stop()
                    logger.debug(f"Stop result: {stop_result}")
                    self._release_playback_artifact()

                    # Small delay to ensure clean state
                    import asyncio

                    await asyncio.sleep(0.2)

                    # Check state after stop
                    state_after_stop = await self.app.audio_player.get_state()
                    logger.debug(f"Player state after stop: {state_after_stop}")

                    # Attempt to play the audio file
                    logger.info(f"Attempting to play audio file: {audio_path}")
                    success = await self.app.audio_player.play(audio_path)
                    logger.debug(f"Play result: {success}")

                    if success:
                        self._active_playback_release = release_artifact
                        release_artifact = None

                        # Cancel any existing progress timer
                        if (
                            self._progress_timer_task
                            and not self._progress_timer_task.done()
                        ):
                            self._progress_timer_task.cancel()
                            await asyncio.sleep(
                                0.05
                            )  # Small delay to ensure cancellation

                        # Start new progress timer
                        self._progress_timer_task = asyncio.create_task(
                            self._update_progress_timer()
                        )
                        logger.debug("Started new progress timer")

                        # Wait a tiny bit to ensure the player has started
                        await asyncio.sleep(0.1)

                        # Double-check the player is actually playing
                        is_playing = await self.app.audio_player.is_playing()
                        logger.debug(
                            f"Player is_playing check after start: {is_playing}"
                        )

                        # Clear the worker task reference as it's now running
                        self._play_worker_task = None
                    else:
                        logger.error("Failed to start playback - play() returned False")
                        self.app.notify("Failed to start playback", severity="error")
                        # Reset button states on failure
                        self.query_one("#audio-play-btn", Button).disabled = False
                        self.query_one("#pause-audio-btn", Button).disabled = True
                        self.query_one("#stop-audio-btn", Button).disabled = True
                        self.query_one("#audio-player-status", Static).update(
                            "Playback failed"
                        )
                else:
                    logger.warning(f"Audio file not found: {audio_path}")
                    self.app.notify(
                        f"Audio file not found: {audio_path.name}", severity="warning"
                    )
            else:
                logger.warning("No audio file to play")
                self.app.notify("No audio file to play", severity="warning")
        except Exception as e:
            logger.opt(exception=True).error(f"Error playing audio: {e}")
            self.app.notify(f"Playback error: {str(e)}", severity="error")
            # Reset button states on error
            self.query_one("#audio-play-btn", Button).disabled = False
            self.query_one("#pause-audio-btn", Button).disabled = True
            self.query_one("#stop-audio-btn", Button).disabled = True
            self.query_one("#audio-player-status", Static).update("Playback error")
        finally:
            if release_artifact is not None:
                release_artifact()

    def _ensure_audio_player(self) -> bool:
        """Ensure audio player is initialized (lazy loading)"""
        if not hasattr(self.app, "audio_player"):
            try:
                from tldw_chatbook.TTS.audio_player import AsyncAudioPlayer

                self.app.audio_player = AsyncAudioPlayer()
                logger.info("Audio player initialized on first use")
                return True
            except Exception as e:
                logger.error(f"Failed to initialize audio player: {e}")
                self.app.notify("Failed to initialize audio player", severity="error")
                return False
        return True

    def _pause_audio(self) -> None:
        """Pause audio playback"""
        logger.debug("_pause_audio called")
        if self._ensure_audio_player():
            logger.debug("Audio player available, running pause worker")
            self.run_worker(
                self._pause_audio_async,
                group="stts-playback",
                exclusive=True,
            )
        else:
            logger.debug("Audio player not available")
            self.app.notify("Audio player not available", severity="warning")

    async def _pause_audio_async(self) -> None:
        """Pause audio playback asynchronously"""
        try:
            from tldw_chatbook.TTS.audio_player import PlaybackState
            import asyncio

            logger.debug("_pause_audio_async called")
            # Small delay to ensure UI is ready
            await asyncio.sleep(0.1)

            state = await self.app.audio_player.get_state()
            logger.debug(f"Current playback state: {state}")
            if state == PlaybackState.PLAYING:
                success = await self.app.audio_player.pause()
                if success:
                    # Update button states
                    self.query_one("#pause-audio-btn", Button).label = "▶️ Resume"
                    self.app.notify("Playback paused", severity="information")
                else:
                    self.app.notify("Failed to pause playback", severity="warning")
            elif state == PlaybackState.PAUSED:
                success = await self.app.audio_player.resume()
                if success:
                    # Update button states
                    self.query_one("#pause-audio-btn", Button).label = "⏸️ Pause"
                    self.app.notify("Playback resumed", severity="information")
                    # Cancel any existing timer and restart
                    if (
                        self._progress_timer_task
                        and not self._progress_timer_task.done()
                    ):
                        self._progress_timer_task.cancel()
                    import asyncio

                    self._progress_timer_task = asyncio.create_task(
                        self._update_progress_timer()
                    )
                else:
                    self.app.notify("Failed to resume playback", severity="warning")
        except Exception as e:
            logger.error(f"Error toggling pause: {e}")
            from rich.markup import escape

            self.app.notify(f"Error: {escape(str(e))}", severity="error")

    def _stop_audio(self) -> None:
        """Stop audio playback"""
        logger.debug("_stop_audio called")
        if self._ensure_audio_player():
            logger.debug("Audio player available, running stop worker")
            self.run_worker(
                self._stop_audio_async,
                group="stts-playback",
                exclusive=True,
            )
        else:
            logger.debug("Audio player not available")
            self.app.notify("Audio player not available", severity="warning")

    async def _stop_audio_async(self) -> None:
        """Stop audio playback asynchronously"""
        try:
            logger.debug("_stop_audio_async called")
            # Cancel progress timer if running
            if self._progress_timer_task and not self._progress_timer_task.done():
                self._progress_timer_task.cancel()
                self._progress_timer_task = None

            # Force stop any playback
            success = await self.app.audio_player.stop()
            logger.debug(f"Stop result: {success}")
            self._release_playback_artifact()

            # Also ensure progress timer is cancelled
            if self._progress_timer_task and not self._progress_timer_task.done():
                self._progress_timer_task.cancel()
                await asyncio.sleep(0.1)  # Give it time to cancel

            # Always reset button states regardless of success
            # (audio may have already finished playing)
            self.query_one(
                "#audio-play-btn", Button
            ).disabled = False  # Re-enable play button
            self.query_one("#pause-audio-btn", Button).label = "⏸️ Pause"
            self.query_one("#pause-audio-btn", Button).disabled = True
            self.query_one("#stop-audio-btn", Button).disabled = True

            if success:
                self.query_one("#audio-player-status", Static).update(
                    "Playback stopped"
                )
                self.app.notify("Playback stopped", severity="information")
            else:
                # Audio already finished or wasn't playing
                self.query_one("#audio-player-status", Static).update(
                    "Audio ready to play"
                )
                logger.debug("Audio may have already finished playing")
        except Exception as e:
            logger.error(f"Error stopping playback: {e}")
            from rich.markup import escape

            self.app.notify(f"Error: {escape(str(e))}", severity="error")

    def _export_audio(self) -> None:
        """Export the generated audio"""
        captured = self._capture_audio_for_action()
        if captured is None:
            self.app.notify("No audio file to export", severity="warning")
            return
        original_path, release_artifact = captured
        if not original_path.exists():
            release_artifact()
            self.app.notify("No audio file to export", severity="warning")
            return

        # Create file save dialog
        filters = Filters(
            (
                "Audio Files",
                lambda p: (
                    p.suffix.lower() in [".mp3", ".wav", ".aac", ".flac", ".opus"]
                ),
            ),
            ("All Files", lambda p: True),
        )

        # Get original filename and extension
        default_name = f"tts_export_{original_path.stem}{original_path.suffix}"

        file_picker = FileSave(
            title="Export Audio File",
            filters=filters,
            default_filename=default_name,
            context="audio_export",
        )

        def handle_export(path: Optional[str]) -> None:
            try:
                self._handle_audio_export(path, source_path=original_path)
            finally:
                release_artifact()

        try:
            self.app.push_screen(file_picker, handle_export)
        except Exception:
            release_artifact()
            raise

    def _handle_audio_export(
        self,
        path: Optional[str],
        *,
        source_path: Path | None = None,
    ) -> None:
        """Handle audio file export"""
        if source_path is None:
            source_path = self._current_generated_audio_path()
        if not path or source_path is None:
            return

        try:
            import shutil
            from tldw_chatbook.Utils.path_validation import (
                validate_filename,
                validate_path_simple,
            )

            dest_path = Path(path)

            # If different format requested, we need conversion
            if source_path.suffix.lower() != dest_path.suffix.lower():
                # For now, just copy - format conversion would require audio service
                self.app.notify(
                    f"Format conversion not yet implemented. Exporting as {source_path.suffix}",
                    severity="warning",
                )
                dest_path = dest_path.with_suffix(source_path.suffix)

            validate_path_simple(dest_path, require_exists=False)
            validated_parent = validate_path_simple(
                dest_path.parent,
                require_exists=True,
            ).resolve()
            validated_filename = validate_filename(dest_path.name)
            dest_path = validated_parent / validated_filename

            # Copy the file
            shutil.copy2(source_path, dest_path)
            self.app.notify(f"Audio exported to: {dest_path.name}", severity="success")

        except Exception as e:
            logger.error(f"Failed to export audio: {e}")
            self.app.notify(f"Export failed: {str(e)}", severity="error")

    def _select_reference_audio(self) -> None:
        """Select reference audio file for voice cloning"""
        # Create file picker for audio files using pre-imported FileOpen
        filters = Filters(
            (
                "Audio Files",
                lambda p: p.suffix.lower() in [".wav", ".mp3", ".m4a", ".flac", ".aac"],
            ),
            ("All Files", lambda p: True),
        )

        file_picker = FileOpen(
            title="Select Reference Audio", filters=filters, context="reference_audio"
        )

        # Mount the file picker
        self.app.push_screen(file_picker, self._handle_reference_audio_selection)

    def _handle_reference_audio_selection(self, path: Optional[str]) -> None:
        """Handle reference audio file selection"""
        if path:
            self.reference_audio_path = path
            # Update status
            status = self.query_one("#reference-audio-status", Static)
            filename = Path(path).name
            status.update(f"Selected: {filename}")
            # Enable clear button
            self.query_one("#clear-reference-audio-btn", Button).disabled = False
            logger.info(f"Reference audio selected: {path}")
        else:
            logger.info("Reference audio selection cancelled")

    def _clear_reference_audio(self) -> None:
        """Clear the selected reference audio"""
        self.reference_audio_path = None
        # Update status
        status = self.query_one("#reference-audio-status", Static)
        status.update("No reference audio selected")
        # Disable clear button
        self.query_one("#clear-reference-audio-btn", Button).disabled = True
        logger.info("Reference audio cleared")

    def _upload_higgs_voice(self) -> None:
        """Open file dialog to select reference audio for Higgs voice cloning"""

        def handle_selection(path: Optional[Path]) -> None:
            if path:
                self.higgs_reference_audio_path = str(path)
                # Update status
                status = self.query_one("#higgs-voice-status", Static)
                status.update(f"Selected: {path.name}")
                # Enable clear button
                self.query_one("#higgs-clear-voice-btn", Button).disabled = False
                logger.info(f"Higgs reference audio selected: {path}")
            else:
                logger.info("Higgs reference audio selection cancelled")

        file_open = FileOpen(
            title="Select Reference Audio for Voice Cloning",
            filters=Filters(
                (
                    "Audio Files",
                    lambda p: (
                        p.suffix.lower()
                        in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
                    ),
                ),
                ("All Files", lambda p: True),
            ),
            must_exist=True,
        )
        self.app.push_screen(file_open, handle_selection)

    def _clear_higgs_voice(self) -> None:
        """Clear the selected Higgs reference audio"""
        self.higgs_reference_audio_path = None
        # Update status
        status = self.query_one("#higgs-voice-status", Static)
        status.update("No voice reference selected")
        # Disable clear button
        self.query_one("#higgs-clear-voice-btn", Button).disabled = True
        logger.info("Higgs reference audio cleared")

    def _check_higgs_installation(self) -> None:
        """Check if Higgs Audio is properly installed"""
        try:
            import boson_multimodal  # noqa: F401

            logger.info("Higgs Audio is installed and available")
        except ImportError:
            self.app.notify(
                "⚠️ Higgs Audio not installed! Run: ./scripts/install_higgs.sh",
                severity="warning",
                timeout=10,
            )
            logger.warning("Higgs Audio (boson_multimodal) is not installed")

    def _insert_random_text(self) -> None:
        """Insert a random example text"""
        import random

        text_area = self.query_one("#tts-text-input", TextArea)
        text_area.text = random.choice(self.example_texts)
        text_area.focus()
        self.app.notify("Random example text inserted", severity="information")

    def _clear_text(self) -> None:
        """Clear the text input"""
        text_area = self.query_one("#tts-text-input", TextArea)
        text_area.clear()
        text_area.focus()
        self.app.notify("Text cleared", severity="information")

    def action_generate_tts(self) -> None:
        """Keyboard shortcut action for generate"""
        self._generate_tts()

    def action_random_text(self) -> None:
        """Keyboard shortcut action for random text"""
        self._insert_random_text()

    def action_clear_text(self) -> None:
        """Keyboard shortcut action for clear text"""
        self._clear_text()

    def action_play_audio(self) -> None:
        """Keyboard shortcut action for play audio"""
        if not self.query_one("#audio-play-btn", Button).disabled:
            self._play_audio()

    def action_stop_audio(self) -> None:
        """Keyboard shortcut action for stop audio"""
        if not self.query_one("#stop-audio-btn", Button).disabled:
            self._stop_audio()

    async def _update_progress_timer(self) -> None:
        """Update progress bar during playback"""
        import asyncio
        from tldw_chatbook.TTS.audio_player import PlaybackState
        from textual.widgets import ProgressBar

        # Ensure audio player exists
        if not hasattr(self.app, "audio_player"):
            return

        while True:
            try:
                state = await self.app.audio_player.get_state()
                if state == PlaybackState.PLAYING:
                    position = await self.app.audio_player.get_position()
                    duration = await self.app.audio_player.get_duration()

                    if duration and duration > 0:
                        # Update progress bar
                        progress_bar = self.query_one(
                            "#audio-progress-bar", ProgressBar
                        )
                        progress_bar.update(progress=position, total=duration)

                        # Update time display
                        time_display = self.query_one("#audio-time-display")
                        current_time = self._format_time(position)
                        total_time = self._format_time(duration)
                        time_display.update(f"{current_time} / {total_time}")

                        # Show progress elements
                        progress_bar.remove_class("hidden")
                        time_display.remove_class("hidden")
                elif state in [PlaybackState.IDLE, PlaybackState.FINISHED]:
                    self._release_playback_artifact()

                    # Hide progress elements
                    self.query_one("#audio-progress-bar").add_class("hidden")
                    self.query_one("#audio-time-display").add_class("hidden")

                    # Reset button states when playback finishes
                    self.query_one("#audio-play-btn", Button).disabled = False
                    self.query_one("#pause-audio-btn", Button).disabled = True
                    self.query_one("#pause-audio-btn", Button).label = "⏸️ Pause"
                    self.query_one("#stop-audio-btn", Button).disabled = True
                    self.query_one("#audio-player-status", Static).update(
                        "Playback complete"
                    )

                    # Notify that playback is complete
                    if state == PlaybackState.FINISHED:
                        self.app.notify("Playback complete", severity="information")

                    break

                await asyncio.sleep(0.1)  # Update every 100ms
            except asyncio.CancelledError:
                logger.debug("Progress timer cancelled")
                break
            except Exception as e:
                logger.error(f"Error updating progress: {e}")
                break

        # Ensure UI is reset on exit
        try:
            self.query_one("#audio-play-btn", Button).disabled = False
            self.query_one("#pause-audio-btn", Button).disabled = True
            self.query_one("#stop-audio-btn", Button).disabled = True
            self.query_one("#audio-player-status", Static).update("Ready to play")
        except Exception as e:
            logger.debug(f"Could not reset UI on progress timer exit: {e}")

    def _format_time(self, seconds: float) -> str:
        """Format seconds to MM:SS format"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"


def _seed_axis_defaults() -> dict[str, str]:
    """Seed `SpeechPlaygroundPane.axis_defaults` from persisted preferences.

    `SpeechPlaygroundPane.axis_values`/`axis_defaults` are the model of
    record for the axis row's override markers
    (`Docs/superpowers/specs/2026-07-30-speech-preset-axis-ownership.md`).
    Reads the same `get_cli_setting` values `SpeechSettingsMixin._set_initial_values`
    already reads, so a first-run pane and Settings agree on what "default"
    means, and builds one `TTSPreferencesSnapshot` -- its `model_mode`/
    `voice_mode` are what "only when its mode is exact" needs to check.

    A missing preference is OMITTED rather than stored as a sentinel: the
    axis row's `is_override` treats a missing default as "not an override",
    which is the correct first-run behaviour.

    Returns:
        A ``{control_id: value}`` mapping, or ``{}`` if preferences cannot
        be read for any reason -- this seeds `compose()`, which must never
        raise (an escaping exception there exits the whole app).
    """
    try:
        default_provider = get_cli_setting("app_tts", "default_provider", "openai")
        is_audio_cpp = default_provider == AUDIO_CPP_PROVIDER_ID
        preference_values: dict[str, object] = {
            "default_provider": default_provider,
            "default_model": get_cli_setting(
                "app_tts",
                "default_model",
                "" if is_audio_cpp else "tts-1",
            ),
            "default_voice": get_cli_setting(
                "app_tts",
                "default_voice",
                "" if is_audio_cpp else "alloy",
            ),
            "default_format": get_cli_setting(
                "app_tts",
                "default_format",
                "wav" if is_audio_cpp else "mp3",
            ),
            "default_speed": get_cli_setting("app_tts", "default_speed", 1.0),
        }
        missing_mode = object()
        for mode_key in ("default_model_mode", "default_voice_mode"):
            mode = get_cli_setting("app_tts", mode_key, missing_mode)
            if mode is not missing_mode:
                preference_values[mode_key] = mode
        preferences = TTSPreferencesSnapshot.from_settings(
            {"app_tts": preference_values}
        )
    except Exception:  # noqa: BLE001 - compose() must never raise
        logger.debug("Could not seed Playground axis defaults from preferences")
        return {}

    defaults: dict[str, str] = {
        "tts-provider-select": preferences.provider_id,
        "tts-format-select": preferences.response_format,
        "tts-speed-input": str(preferences.speed),
    }
    if preferences.model_mode == "exact" and preferences.model_id:
        defaults["tts-model-select"] = preferences.model_id
    if preferences.voice_mode == "exact" and preferences.voice_id:
        defaults["tts-voice-select"] = preferences.voice_id
    return defaults


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
        self._pending_playground_preset: TTSPlaygroundSelectionPreset | None = None

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
            # Show playground by default. The rebuilt pane, not the legacy
            # widget -- see the takeover ruling above the module-level
            # `_seed_axis_defaults` helper.
            yield SpeechPlaygroundPane(
                id="speech-playground-pane",
                axis_defaults=_seed_axis_defaults(),
            )
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
            preset = self._pending_playground_preset
            self._pending_playground_preset = None
            content_container.mount(
                SpeechPlaygroundPane(
                    id="speech-playground-pane",
                    profile_preset=preset,
                    axis_defaults=_seed_axis_defaults(),
                )
            )
        elif new_view == "profiles":
            content_container.mount(STTSProfileLibrary(self._load_profile_service))
        elif new_view == "settings":
            content_container.mount(SpeechSettingsPane(id="speech-settings-pane"))
        elif new_view == "voice-cloning":
            from tldw_chatbook.UI.Voice_Cloning_Window import VoiceCloningWindow

            content_container.mount(VoiceCloningWindow())
        elif new_view == "effects":
            content_container.mount(SpeechEffectsPane(id="speech-effects-pane"))
        elif new_view == "audiobook":
            content_container.mount(AudioBookGenerationWidget())
        elif new_view == "dictation":
            content_container.mount(DictationWindow())

        # Selection styling is the rail's job now. These lines used to
        # `query_one("#view-*-btn")` for the four view buttons; those live on
        # STTSScreen since the sidebar moved, so every one of them would raise
        # NoMatches on the first view change. The screen watches
        # `current_view` and applies `is-active` itself.

    @on(ProfilePreviewRequested)
    def on_profile_preview_requested(
        self,
        message: ProfilePreviewRequested,
    ) -> None:
        """Hand one exact preset to the next Playground mount."""
        if type(message.preset) is not TTSPlaygroundSelectionPreset:
            return
        self._pending_playground_preset = message.preset
        self.current_view = "playground"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle sidebar button presses and delegate to content widgets"""
        # Handle sidebar buttons
        if event.button.id == "view-playground-btn":
            self.current_view = "playground"
        elif event.button.id == "view-profiles-btn":
            self.current_view = "profiles"
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

    async def _load_profile_service(self) -> TTSProfileService | None:
        """Resolve the app-owned profile service without affecting speech."""
        ensure_service = getattr(
            self.app_instance,
            "_ensure_tts_profile_service",
            None,
        )
        if not callable(ensure_service):
            return None
        try:
            return await ensure_service()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("TTS profile storage is unavailable")
            return None


#
# End of STTS_Window.py
#######################################################################################################################
