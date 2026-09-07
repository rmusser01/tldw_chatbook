# STTS_Window.py
# Description: S/TT/S (Speech/Text-to-Speech) tab with TTS Playground, Settings, and AudioBook/Podcast Generation
#
# Imports
import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pathlib import Path
from uuid import UUID, uuid4
from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Container, Vertical
from textual.widgets import (
    Label,
    Button,
    TextArea,
    Select,
    Static,
    RichLog,
    Switch,
    Collapsible,
    Rule,
)
from textual.css.query import QueryError
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widget import Widget
from textual.reactive import reactive
from textual import on, work
from loguru import logger

# Local imports
from tldw_chatbook.config import get_cli_setting, get_runtime_config_snapshot
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSAudioBookGenerateEvent,
    STTSProviderConfigurationChanged,
)
from tldw_chatbook.TTS import (
    STTSPlaygroundRequest,
    TTSPlaygroundSelectionPreset,
    TTSPreferencesSnapshot,
    TTSProfileService,
)
from tldw_chatbook.TTS.studio_preferences import (
    StudioTTSLoadResult,
    StudioTTSLoadState,
    StudioTTSPreferenceStore,
    StudioTTSPreferencesSnapshot,
)
from tldw_chatbook.TTS.voice_blend_paths import kokoro_ui_blend_file
from tldw_chatbook.UI.Speech.speech_effects_pane import SpeechEffectsPane
from tldw_chatbook.UI.Speech.speech_playground_pane import (
    OpenStudioPreferencesRequested,
    OpenVoiceProfilesRequested,
    SpeechPlaygroundPane,
)
from tldw_chatbook.UI.Speech.speech_playground_model import AXIS_CONTROLS
from tldw_chatbook.UI.Speech.speech_profile_mixin import (
    AdoptStudioPreferencesRequested,
)
from tldw_chatbook.UI.Speech.speech_runtime_status import (
    SpeechLocalDependencyAvailability,
    speech_tts_runtime_status_store,
)
from tldw_chatbook.UI.Speech.speech_settings_contracts import (
    SpeechTTSConfigurationState,
    SpeechTTSNavigationTarget,
)
from tldw_chatbook.UI.Screens.settings_speech_tts import (
    BUILT_IN_TTS_PROVIDER_ORDER,
    global_speech_tts_provider_configuration_state,
    load_global_speech_tts_state,
)
from tldw_chatbook.UI.Speech.speech_settings_pane import (
    SpeechDestinationBackRequested,
    SpeechDestinationRequested,
    SpeechSettingsPane,
    StudioPreferencesSaved,
    VoiceBlendsPane,
)
from tldw_chatbook.UI.Speech.speech_settings_mixin import (
    normalize_provider_voice_selection,
)
from tldw_chatbook.UI.stts_profile_library import (
    ProfileLibraryContinuity,
    ProfileLibraryRestoreReady,
    ProfilePreviewRequested,
    ProfileTestVerified,
    ProfileVerificationReconciled,
    ProfileVerificationResult,
    STTSProfileLibrary,
    _retire_profile_test_context,
)
from tldw_chatbook.UI.Lab_Modules.lab_speech_status import (
    speech_capability_text,
    speech_capability_tooltip,
    speech_dependencies_available,
    speech_local_dependency_availability,
)
from tldw_chatbook.Widgets.enhanced_file_picker import (
    EnhancedFileOpen as FileOpen,
    EnhancedFileSave as FileSave,
)
from tldw_chatbook.Third_Party.textual_fspicker import Filters
from tldw_chatbook.UI.Dictation_Window_Improved import (
    ImprovedDictationWindow as DictationWindow,
)
# Note: Not using form_components due to generator/widget incompatibility


#######################################################################################################################
#
# Classes:

STTS_VIEW_KEYS = frozenset(
    {
        "playground",
        "profiles",
        "blends",
        "settings",
        "voice-cloning",
        "effects",
        "audiobook",
        "dictation",
    }
)


@dataclass(frozen=True, slots=True)
class SpeechNavigationDestination:
    """Exact identity for a user-facing Speech voice tool."""

    destination_id: str
    label: str
    view: str
    provider_id: str | None


_SPEECH_NAVIGATION_DESTINATIONS = {
    "voice-profiles": SpeechNavigationDestination(
        "voice-profiles",
        "Voice Profiles",
        "profiles",
        None,
    ),
    "voice-blends": SpeechNavigationDestination(
        "voice-blends",
        "Voice Blends",
        "blends",
        "kokoro",
    ),
}


def resolve_speech_navigation(destination_id: str) -> SpeechNavigationDestination:
    """Resolve one unambiguous Voice Profiles or Voice Blends destination."""

    if type(destination_id) is not str:
        raise TypeError("Speech destination ID must be a string")
    try:
        return _SPEECH_NAVIGATION_DESTINATIONS[destination_id]
    except KeyError:
        raise ValueError("unknown Speech destination") from None


class VoiceProfilePickerModal(
    ModalScreen[tuple[TTSPlaygroundSelectionPreset, UUID] | None]
):
    """Reuse the Voice Profiles library without unmounting clone setup."""

    DEFAULT_CSS = """
    VoiceProfilePickerModal {
        align: center middle;
        background: $background 70%;
    }

    #speech-voice-profile-picker {
        width: 92%;
        height: 90%;
        max-width: 140;
        background: $surface;
        border: round $accent;
        padding: 1;
    }

    #speech-voice-profile-picker-actions {
        height: 3;
        align-horizontal: right;
    }

    #speech-voice-profile-picker-cancel {
        width: auto;
        min-width: 12;
    }
    """

    def __init__(
        self,
        service_loader: Callable[[], Awaitable[TTSProfileService | None]],
        *,
        default_profile_id_reader: Callable[[], object | None] | None = None,
        voice_bundle_service_loader: Callable[[], Awaitable[object | None]]
        | None = None,
    ) -> None:
        super().__init__()
        if not callable(service_loader):
            raise TypeError("Voice Profile service loader must be callable")
        if default_profile_id_reader is not None and not callable(
            default_profile_id_reader
        ):
            raise TypeError("default profile reader must be callable")
        self._service_loader = service_loader
        self._default_profile_id_reader = default_profile_id_reader
        self._voice_bundle_service_loader = voice_bundle_service_loader

    def compose(self) -> ComposeResult:
        with Vertical(id="speech-voice-profile-picker"):
            yield STTSProfileLibrary(
                self._service_loader,
                default_profile_id_reader=self._default_profile_id_reader,
                voice_bundle_service_loader=self._voice_bundle_service_loader,
            )
            with Horizontal(id="speech-voice-profile-picker-actions"):
                yield Button("Back to setup", id="speech-voice-profile-picker-cancel")

    @on(ProfilePreviewRequested)
    def _preview_profile(self, message: ProfilePreviewRequested) -> None:
        """Return the exact existing-library preview preset to the Playground."""

        message.stop()
        if (
            type(message.preset) is TTSPlaygroundSelectionPreset
            and type(message.context_token) is UUID
        ):
            self.dismiss((message.preset, message.context_token))

    @on(Button.Pressed, "#speech-voice-profile-picker-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)


class AudioBookGenerationWidget(Widget):
    """AudioBook/Podcast Generation widget"""

    # How long to wait after the last keystroke in the content-preview paste
    # box before running chapter detection (task-15478). Detection walks the
    # full pasted text with ChapterDetector.detect_chapters and pops a notify
    # toast, so it must not run on every TextArea.Changed message.
    _CHAPTER_DETECT_DEBOUNCE_SECONDS = 1.0

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
        self._chapter_detect_debounce_timer: Optional[Timer] = None
        # Last chapter count a "Detected N chapters" toast was shown for
        # (task-15478 review): a debounced re-paste can re-run detection
        # several times as the user keeps typing, and every settle used to
        # pop its own toast even when the count hadn't moved. Reset in
        # `_import_content` -- see that method's comment for why that's the
        # right seam.
        self._last_notified_chapter_count: Optional[int] = None
        # Monotonically-increasing dispatch id (task-15478 review round 2):
        # `exclusive=True` on the detection worker cancels QUEUED workers in
        # its group, but cannot interrupt one already running on an OS
        # thread -- a slower, superseded dispatch can still finish after a
        # faster, newer one and overwrite its result. `_apply_detected_
        # chapters` only applies a result whose generation matches the
        # latest dispatched one.
        self._chapter_detect_generation: int = 0
        self._last_valid_narrator_voice: str | None = None

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
                            ("Text File", "file"),
                            ("Notes", "notes"),
                            ("Conversation", "conversation"),
                            ("Paste Text", "paste"),
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
                            ("Alloy", "alloy"),
                            ("Echo", "echo"),
                            ("Fable", "fable"),
                            ("Onyx", "onyx"),
                            ("Nova", "nova"),
                            ("Shimmer", "shimmer"),
                        ],
                        id="narrator-voice-select",
                    )
                yield Static(
                    "Voice Blends",
                    id="audiobook-voice-blends-label",
                    classes="hidden",
                    markup=False,
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
                            ("OpenAI", "openai"),
                            ("ElevenLabs", "elevenlabs"),
                            ("Kokoro (Local)", "kokoro"),
                            ("Chatterbox (Local)", "chatterbox"),
                        ],
                        id="audiobook-provider-select",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Audio Format:", classes="form-label")
                    yield Select(
                        options=[
                            ("MP3", "mp3"),
                            ("M4B (AudioBook)", "m4b"),
                            ("Opus", "opus"),
                            ("AAC", "aac"),
                            ("WAV", "wav"),
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

    def on_unmount(self) -> None:
        """Cancel any pending debounced chapter detection."""
        if self._chapter_detect_debounce_timer is not None:
            self._chapter_detect_debounce_timer.stop()
            self._chapter_detect_debounce_timer = None

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

        # Every source (including "paste": this is the app's own signal
        # that the user is about to start pasting a new document, since
        # `_import_from_paste` is what enables and focuses the previously
        # disabled content-preview box) is a deliberate, one-shot "bring in
        # new content" action. Reset the notify-dedup memory here, not
        # inside `_detect_chapters`/`_apply_detected_chapters`, so a freshly
        # imported document that happens to detect the same chapter count
        # as whatever the PREVIOUS session last toasted still gets its own
        # toast (task-15478 review: this used to never reset, so a
        # genuinely new import could go silently un-toasted). Detections
        # re-run later within the SAME session -- e.g. the paste box's
        # debounced re-detection as the user keeps typing after this point,
        # with no further `_import_content` call in between -- still dedupe
        # against each other, since this is the only reset point.
        self._last_notified_chapter_count = None

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

            # Auto-detect chapters. This used to be gated by an
            # "#auto-chapters-switch" that no longer exists in the composed UI
            # (task-15478) -- it defaulted to on and there is currently no
            # control to turn it off, so a one-shot import always detects.
            self._detect_chapters()

            self.app.notify(
                f"Imported {len(content)} characters from {Path(path).name}",
                severity="information",
            )

        except Exception as e:
            logger.error(f"Failed to import file: {e}")
            self.app.notify(f"Failed to import file: {e}", severity="error")

    def _import_from_notes(self) -> None:
        """Import content from notes.

        Task-19576: this used to import `fetch_all_notes`/`fetch_note_by_id`
        from `tldw_chatbook.DB.ChaChaNotes_DB` -- neither function exists,
        and the imports sat outside the `try:` block, so every use of this
        action crashed with an uncaught `ImportError`. This now routes
        through the shared `notes_scope_service` seam (the same one
        Library/Home use) instead of resurrecting module-level DB
        functions. The service is async, so the actual work happens in
        `_import_from_notes_worker`.
        """
        self._import_from_notes_worker()

    @work(exclusive=True, group="audiobook-import-notes")
    async def _import_from_notes_worker(self) -> None:
        """Worker half of `_import_from_notes` (see its docstring)."""
        from tldw_chatbook.Notes.notes_scope_service import ScopeType
        from tldw_chatbook.Widgets.Note_Widgets.note_selection_dialog import (
            NoteSelectionDialog,
        )

        notes_scope_service = getattr(self.app, "notes_scope_service", None)
        if notes_scope_service is None:
            self.app.notify(
                "Notes are unavailable in this session.", severity="error"
            )
            return

        user_id = getattr(self.app, "notes_user_id", None) or "default_user"

        try:
            notes = await notes_scope_service.list_notes(
                scope=ScopeType.LOCAL_NOTE,
                user_id=user_id,
            )
        except Exception as e:
            logger.error(f"Failed to import from notes: {e}")
            self.app.notify(f"Failed to import notes: {e}", severity="error")
            return

        if not notes:
            self.app.notify("No notes found in database", severity="warning")
            return

        # `list_notes` already returns full note rows (title + content, not
        # a preview), so the notes selected in the dialog below can be
        # combined straight from this same page -- no second per-note fetch
        # needed.
        notes_by_id: Dict[str, Dict[str, Any]] = {}
        dialog_notes: List[Dict[str, Any]] = []
        for note in notes:
            note_id = note.get("id")
            if note_id is None:
                continue
            note_id = str(note_id)
            notes_by_id[note_id] = note
            dialog_notes.append(
                {
                    "note_id": note_id,
                    "title": note.get("title", ""),
                    "content": note.get("content", ""),
                    "created_at": note.get("created_at", "Unknown"),
                }
            )

        try:
            selected_ids = await self.app.push_screen(
                NoteSelectionDialog(dialog_notes), wait_for_dismiss=True
            )
        except Exception as e:
            logger.error(f"Failed to import from notes: {e}")
            self.app.notify(f"Failed to import notes: {e}", severity="error")
            return

        if not selected_ids:
            return

        combined_content = []
        for note_id in selected_ids:
            note = notes_by_id.get(str(note_id))
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

        # Auto-detect chapters (see task-15478: the switch that
        # used to gate this is gone from the composed UI).
        self._detect_chapters()

        self.app.notify(
            f"Imported {len(selected_ids)} note(s)", severity="information"
        )

    def _import_from_conversation(self) -> None:
        """Import content from a conversation.

        Task-19576: this used to import `fetch_all_conversations`/
        `fetch_messages_by_conversation_id` from
        `tldw_chatbook.DB.ChaChaNotes_DB` -- neither function exists, and
        the imports sat outside the `try:` block, so every use of this
        action crashed with an uncaught `ImportError` (same defect and
        same fix shape as `_import_from_notes`). This now routes through
        the shared `chat_conversation_scope_service` seam. The service is
        async, so the actual work happens in
        `_import_from_conversation_worker`.
        """
        self._import_from_conversation_worker()

    # Bounded page walk for message loading (task-19576): the removed
    # `fetch_messages_by_conversation_id` had no limit at all.
    # `get_messages_with_context` pages in bounded chunks; this caps the
    # total collected messages generously above what any audiobook-worthy
    # conversation should reach, rather than reintroducing an unbounded read.
    _CONVERSATION_IMPORT_PAGE_SIZE = 200
    _CONVERSATION_IMPORT_MAX_MESSAGES = 5000

    @work(exclusive=True, group="audiobook-import-conversation")
    async def _import_from_conversation_worker(self) -> None:
        """Worker half of `_import_from_conversation` (see its docstring)."""
        from tldw_chatbook.Widgets.conversation_selection_dialog import (
            ConversationSelectionDialog,
        )

        conversation_service = getattr(
            self.app, "chat_conversation_scope_service", None
        )
        if conversation_service is None:
            self.app.notify(
                "Conversations are unavailable in this session.", severity="error"
            )
            return

        try:
            payload = await conversation_service.list_conversations(
                mode="local",
                # "all" spans global- and workspace-scoped conversations, so
                # a Console chat saved inside a workspace session is still
                # importable here.
                scope_type="all",
                limit=100,
                offset=0,
            )
        except Exception as e:
            logger.error(f"Failed to import from conversation: {e}")
            self.app.notify(f"Failed to import conversation: {e}", severity="error")
            return

        items = payload.get("items") if isinstance(payload, dict) else None
        conversations: List[Dict[str, Any]] = []
        for item in items or []:
            conversation_id = item.get("id")
            if conversation_id is None:
                continue
            conversations.append(
                {
                    "conversation_id": str(conversation_id),
                    "title": item.get("title", ""),
                    "model_name": item.get("runtime_backend") or "Unknown",
                    "message_count": item.get("message_count", 0),
                    "created_at": item.get("created_at", "Unknown"),
                    "updated_at": item.get("last_modified", "Unknown"),
                }
            )

        if not conversations:
            self.app.notify(
                "No conversations found in database", severity="warning"
            )
            return

        try:
            selection = await self.app.push_screen(
                ConversationSelectionDialog(conversations), wait_for_dismiss=True
            )
        except Exception as e:
            logger.error(f"Failed to import from conversation: {e}")
            self.app.notify(f"Failed to import conversation: {e}", severity="error")
            return

        if not selection:
            return

        conversation_id = str(selection["conversation_id"])

        messages: List[Dict[str, Any]] = []
        offset = 0
        try:
            while len(messages) < self._CONVERSATION_IMPORT_MAX_MESSAGES:
                page = await conversation_service.get_messages_with_context(
                    conversation_id,
                    mode="local",
                    limit=self._CONVERSATION_IMPORT_PAGE_SIZE,
                    offset=offset,
                    include_rag_context=False,
                )
                if not page:
                    break
                messages.extend(page)
                if len(page) < self._CONVERSATION_IMPORT_PAGE_SIZE:
                    break
                offset += self._CONVERSATION_IMPORT_PAGE_SIZE
        except Exception as e:
            logger.error(f"Failed to import from conversation: {e}")
            self.app.notify(f"Failed to import conversation: {e}", severity="error")
            return

        if not messages:
            self.app.notify(
                "No messages found in conversation", severity="warning"
            )
            return

        # Build content based on options
        content_parts = []
        for msg in messages:
            sender = str(msg.get("role") or msg.get("sender") or "")
            is_user_message = sender.strip().lower() == "user"

            # Filter based on inclusion options
            if selection.get("include_all"):
                pass  # Include all messages
            elif selection.get("include_user") and not is_user_message:
                continue
            elif selection.get("include_assistant") and is_user_message:
                continue

            content = msg.get("content", "")

            # Format based on speaker option
            if selection.get("include_speakers"):
                speaker_name = "User" if is_user_message else "Assistant"
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

        # Auto-detect chapters might not be suitable for
        # conversations, but run it anyway (see task-15478: the
        # switch that used to gate this is gone from the UI).
        self._detect_chapters()

        self.app.notify(
            f"Imported conversation with {len(messages)} messages",
            severity="information",
        )

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
            self._queue_debounced_chapter_detection()

    def _queue_debounced_chapter_detection(self) -> None:
        """Debounce chapter detection while the user is still typing/pasting.

        `_detect_chapters()` walks the entire `content_text` with
        `ChapterDetector.detect_chapters` and pops a notify toast, so running
        it synchronously from `TextArea.Changed` -- which fires once per
        keystroke -- is unacceptable (task-15478; this replaced a since
        removed "#auto-chapters-switch" guard). Instead, (re)arm a timer on
        every change and only run detection once the input goes quiet.
        """
        if self._chapter_detect_debounce_timer is not None:
            self._chapter_detect_debounce_timer.stop()
            self._chapter_detect_debounce_timer = None

        if not self.content_text:
            return

        self._chapter_detect_debounce_timer = self.set_timer(
            self._CHAPTER_DETECT_DEBOUNCE_SECONDS,
            self._run_debounced_chapter_detection,
        )

    def _run_debounced_chapter_detection(self) -> None:
        """Timer callback: run detection once, then clear the timer handle."""
        self._chapter_detect_debounce_timer = None
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
        """Detect chapters in the content, off the event loop.

        `ChapterDetector.detect_chapters` is O(len(content)) regex scanning
        over every line -- benchmarked (task-15478 review) at ~19ms for 90k
        words, ~60ms for 300k, and ~200ms on a 6MB paste. That is well past
        the repo's 100ms worker budget, for exactly the large pastes an
        audiobook feature invites, and it used to run synchronously on the
        event loop from all four call sites (three one-shot imports, plus
        the debounced paste-box timer). None of the three import paths have
        a genuinely bounded size either -- a book imported from a file, a
        note, or a long conversation can all be just as large as a paste --
        so all four now route through this one threaded path rather than
        special-casing which callers are "small enough".

        The CPU-bound detection itself runs in `_detect_chapters_worker` (a
        `@work(thread=True)` method); this method only snapshots
        `content_text` and dispatches it.

        `exclusive=True` on that worker only cancels a prior *queued* run in
        the same group -- it cannot interrupt one already executing on an OS
        thread. A stale, slower dispatch can therefore still finish and
        marshal its result back AFTER a newer, faster one (task-15478
        review round 2; reproduced 3/3 with two back-to-back dispatches, no
        debounce between them, which is exactly what happens across the
        three one-shot import paths). A monotonically-increasing generation
        id is captured here and threaded through; `_apply_detected_chapters`
        only applies a result whose generation still matches the latest one
        dispatched.
        """
        if not self.content_text:
            return
        self._chapter_detect_generation += 1
        generation = self._chapter_detect_generation
        self._detect_chapters_worker(self.content_text, generation)

    @work(thread=True, exclusive=True, group="audiobook-chapter-detection")
    def _detect_chapters_worker(self, content: str, generation: int) -> None:
        """Thread: run the CPU-bound detector, then marshal the result back.

        `exclusive=True` cancels any prior *queued* worker in this same
        group; it does not stop one already mid-execution on its OS thread
        (`Worker.cancel()` cannot interrupt a running thread). Deliberately
        NOT also checking `get_current_worker().is_cancelled` here as a
        "skip the wasted marshal" shortcut, tempting as that is: it would
        make the one-marshal-always-arrives, `_apply_detected_chapters`
        never-overwrites-a-newer-result contract depend on a second,
        cross-thread-timing-sensitive mechanism whose semantics are a
        Textual implementation detail, for a saving that only matters on
        the rare superseded path anyway. The `generation` stamp -- checked
        in `_apply_detected_chapters`, on the main thread, with no
        cross-thread race since only that thread ever reads or writes it --
        is the one real correctness guard.
        """
        try:
            from tldw_chatbook.TTS.audiobook_generator import ChapterDetector

            chapters = ChapterDetector.detect_chapters(content)
        except Exception as e:
            logger.error("Failed to detect chapters in worker")
            self.app.call_from_thread(
                self.app.notify,
                f"Failed to detect chapters: {e}",
                severity="error",
            )
            return

        self.app.call_from_thread(
            self._apply_detected_chapters, chapters, generation
        )

    def _apply_detected_chapters(self, chapters: List, generation: int) -> None:
        """Main-thread half of chapter detection: apply results to the UI.

        Called via `call_from_thread` from `_detect_chapters_worker`, so it
        must stay main-thread-only (widget queries/mutation, notify). Only
        applies a result if `generation` still matches the most recently
        dispatched detection -- a superseded (stale) result is dropped
        rather than overwriting a newer, already-applied one.
        """
        if generation != self._chapter_detect_generation:
            logger.debug("Dropping a stale chapter-detection result")
            return

        self.detected_chapters = chapters

        try:
            from tldw_chatbook.Widgets.TTS.chapter_editor_widget import (
                ChapterEditorWidget,
            )

            # Update the chapter editor widget
            try:
                chapter_editor = self.query_one(
                    "#chapter-editor-widget", ChapterEditorWidget
                )
                chapter_editor.set_chapters(chapters)
                self._notify_chapter_count(len(chapters))
            except Exception as e:
                logger.warning(f"Could not update chapter editor: {e}")
                # Fall back to old display method if chapter editor not found
                chapter_list = self.query_one("#chapter-list", Static)
                if chapters:
                    chapter_display = []
                    for i, chapter in enumerate(chapters):
                        chapter_display.append(
                            f"{i + 1}. {chapter.title} ({len(chapter.content.split())} words)"
                        )

                    chapter_list.update("\n".join(chapter_display))
                    self._notify_chapter_count(len(chapters))
                else:
                    chapter_list.update("No chapters detected")

        except Exception as e:
            logger.error("Failed to apply detected chapters")
            self.app.notify(f"Failed to detect chapters: {e}", severity="error")

    def _notify_chapter_count(self, count: int) -> None:
        """Toast only when the detected chapter count changed since the last
        toast (task-15478 review Minor).

        A debounced re-paste can re-run detection several times as the user
        keeps typing; before this, every settle popped its own
        "Detected N chapters" toast even when N hadn't moved.
        """
        if count == self._last_notified_chapter_count:
            return
        self._last_notified_chapter_count = count
        self.app.notify(f"Detected {count} chapters", severity="information")

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
            voice_select = event.select
            available_voice_ids = tuple(
                value
                for _label, value in voice_select._options
                if self._is_valid_voice(value)
            )
            if event.value in available_voice_ids:
                self._last_valid_narrator_voice = event.value
                return
            prior = self._last_valid_narrator_voice
            voice_select.value = (
                prior if prior in available_voice_ids else Select.BLANK
            )

    def _update_voice_options(self, provider: str) -> None:
        """Update voice options based on provider"""
        voice_select = self.query_one("#narrator-voice-select", Select)
        blend_group = self.query_one("#audiobook-voice-blends-label", Static)
        blend_group.add_class("hidden")

        if provider == "openai":
            voice_select.set_options(
                [
                    ("Alloy", "alloy"),
                    ("Echo", "echo"),
                    ("Fable", "fable"),
                    ("Onyx", "onyx"),
                    ("Nova", "nova"),
                    ("Shimmer", "shimmer"),
                ]
            )
        elif provider == "elevenlabs":
            voice_select.set_options(
                [
                    ("Rachel", "21m00Tcm4TlvDq8ikWAM"),
                    ("Domi", "AZnzlk1XvdvUeBnXmlld"),
                    ("Bella", "EXAVITQu4vr4xnSDxMaL"),
                    ("Antoni", "ErXwobaYiN019PkySvjV"),
                    ("Elli", "MF3mGyEYCl7XYWbV9V6O"),
                ]
            )
        elif provider == "kokoro":
            logger.info(f"Setting up Kokoro voices for provider: {provider}")
            voice_options = [
                ("Bella (US Female)", "af_bella"),
                ("Nicole (US Female)", "af_nicole"),
                ("Sarah (US Female)", "af_sarah"),
                ("Adam (US Male)", "am_adam"),
                ("Michael (US Male)", "am_michael"),
                ("Emma (UK Female)", "bf_emma"),
                ("George (UK Male)", "bm_george"),
            ]

            # Add saved voice blends
            blend_file = kokoro_ui_blend_file()
            if blend_file.exists():
                try:
                    import json

                    with open(blend_file, "r") as f:
                        blends = json.load(f)
                        if blends:
                            blend_group.remove_class("hidden")
                            for blend_name, blend_data in blends.items():
                                display_name = f"🎭 {blend_name}"
                                if blend_data.get("description"):
                                    display_name += (
                                        f" - {blend_data['description'][:30]}"
                                    )
                                voice_options.append(
                                    (display_name, f"blend:{blend_name}")
                                )
                except Exception as e:
                    logger.error(f"Failed to load voice blends: {e}")

            voice_select.set_options(voice_options)

            # Find first valid voice option (skip separators)
            valid_voice = None
            for _, value in voice_options:
                if self._is_valid_voice(value):
                    valid_voice = value
                    break

            if valid_voice:
                voice_select.value = valid_voice

        elif provider == "chatterbox":
            voice_select.set_options(
                [
                    ("Default", "default"),
                    ("Custom Voice", "custom"),
                ]
            )

        available_voice_ids = tuple(
            value
            for _label, value in voice_select._options
            if type(value) is str and value != Select.BLANK
        )
        normalized = normalize_provider_voice_selection(
            provider,
            voice_select.value,
            available_voice_ids,
        )
        voice_select.value = normalized if normalized is not None else Select.BLANK
        self._last_valid_narrator_voice = normalized

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

    def _get_model_for_provider(self, provider: str) -> str:  # type: ignore[no-redef]
        """Get the default model for a given provider"""
        model_map = {
            "openai": "tts-1",
            "elevenlabs": "eleven_multilingual_v2",
            "kokoro": "kokoro",
            "chatterbox": "chatterbox",
            "alltalk": "alltalk",
        }
        return model_map.get(provider, "default")


def _seed_axis_defaults(
    studio_preferences: StudioTTSPreferencesSnapshot | None = None,
    global_preferences: TTSPreferencesSnapshot | None = None,
) -> dict[str, str]:
    """Seed `SpeechPlaygroundPane.axis_defaults` from GENUINELY persisted preferences.

    `SpeechPlaygroundPane.axis_values`/`axis_defaults` are the model of
    record for the axis row's override markers
    (`Docs/superpowers/specs/2026-07-30-speech-preset-axis-ownership.md`,
    contract 5). Contract 5 is explicit: a missing preference must leave the
    axis absent from `defaults`, not substituted -- `SpeechAxisRow.is_override`
    already treats an absent key as "not an override", which is the correct
    first-run behaviour.

    Deliberately NOT `SpeechSettingsMixin._set_initial_values`'s block, and
    NOT `TTSPreferencesSnapshot.from_settings`, though both were tried:

    - `_set_initial_values` substitutes hardcoded fallbacks (`"openai"`,
      `"tts-1"`, `"alloy"`, `"mp3"`) for anything unset, because it exists to
      populate a form that must always show something. Reusing it here
      fabricated four "saved defaults" on a fresh install that were never
      saved, and marked four axes overridden the first time the pane was
      ever opened.
    - `TTSPreferencesSnapshot.from_settings` has the same problem one level
      down: for a non-`audio_cpp` provider, its own resolution treats an
      unset model/voice as *mode "exact" with the legacy default id*
      (`tts-1-hd`/`shimmer`), not as "unconfigured" -- so routing through it
      would silently reintroduce the same fabrication under a different
      name.

    So each preference is read directly, each with its OWN absence
    sentinel, and included only when it was actually set.

    Returns:
        A ``{control_id: value}`` mapping containing only genuinely
        configured axes, or ``{}`` if preferences cannot be read for any
        reason -- this seeds `compose()`, which must never raise (an
        escaping exception there exits the whole app).
    """
    missing = object()
    try:
        defaults: dict[str, str] = {}

        provider_id = get_cli_setting("app_tts", "default_provider", missing)
        if isinstance(provider_id, str) and provider_id:
            defaults["tts-provider-select"] = provider_id

        response_format = get_cli_setting("app_tts", "default_format", missing)
        if isinstance(response_format, str) and response_format:
            defaults["tts-format-select"] = response_format

        speed = get_cli_setting("app_tts", "default_speed", missing)
        if speed is not missing:
            try:
                defaults["tts-speed-input"] = str(float(speed))
            except (TypeError, ValueError):
                pass

        # Model/voice defaults are keyed to "exact" mode only (file map:
        # "only when its mode is exact"). A missing mode is NOT treated as
        # exact-with-a-legacy-default here, unlike the runtime dispatch path
        # (`TTS/preferences.py`'s `_resolved_selection`) -- that fallback
        # exists so synthesis always has *something* to request, which is
        # not evidence the user configured a default at all.
        model_mode = get_cli_setting("app_tts", "default_model_mode", missing)
        model_id = get_cli_setting("app_tts", "default_model", missing)
        if model_mode == "exact" and isinstance(model_id, str) and model_id:
            defaults["tts-model-select"] = model_id

        voice_mode = get_cli_setting("app_tts", "default_voice_mode", missing)
        voice_id = get_cli_setting("app_tts", "default_voice", missing)
        if voice_mode == "exact" and isinstance(voice_id, str) and voice_id:
            defaults["tts-voice-select"] = voice_id

        if global_preferences is not None:
            return SpeechPlaygroundPane._project_axis_defaults(
                studio_preferences,
                global_preferences,
            )

        if studio_preferences is not None:
            selection = studio_preferences.selection
            if selection.provider_id is not None:
                if (
                    global_preferences is not None
                    and selection.provider_id != global_preferences.provider_id
                ):
                    # Global model/voice/format/speed defaults are scoped to
                    # the global provider. A Studio provider override inherits
                    # that provider's fallback for absent axes, not OpenAI
                    # values mislabeled as Chatterbox/audio.cpp defaults.
                    defaults = {}
                defaults["tts-provider-select"] = selection.provider_id
            if selection.model_mode == "exact" and selection.model_id is not None:
                defaults["tts-model-select"] = selection.model_id
            elif selection.model_mode == "first_available":
                defaults.pop("tts-model-select", None)
            if selection.voice_mode == "exact" and selection.voice_id is not None:
                defaults["tts-voice-select"] = selection.voice_id
            elif selection.voice_mode == "server_default":
                defaults.pop("tts-voice-select", None)
            if selection.response_format is not None:
                defaults["tts-format-select"] = selection.response_format
            if selection.speed is not None:
                defaults["tts-speed-input"] = str(selection.speed)

        return defaults
    except Exception:  # noqa: BLE001 - compose() must never raise
        logger.debug("Could not seed Playground axis defaults from preferences")
        return {}


def _rebase_inherited_axis_values(
    values: Mapping[str, str],
    *,
    old_defaults: Mapping[str, str],
    new_defaults: Mapping[str, str],
) -> dict[str, str]:
    """Move inherited retained axes to fresh defaults and keep real overrides."""

    rebased = dict(values)
    missing = object()
    for axis in set(old_defaults) | set(new_defaults):
        current = rebased.get(axis, missing)
        old_default = old_defaults.get(axis, missing)
        new_default = new_defaults.get(axis, missing)
        if old_default == new_default:
            continue
        is_session_override = current is not missing and (
            old_default is missing or current != old_default
        )
        if is_session_override:
            continue
        if new_default is missing:
            rebased.pop(axis, None)
        else:
            rebased[axis] = new_default
    return rebased


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

    def __init__(
        self,
        app_instance,
        *,
        playground_axis_values: Mapping[str, str] | None = None,
        local_dependencies: SpeechLocalDependencyAvailability | None = None,
        **kwargs,
    ):
        """Initialize the S/TT/S window."""
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self._pending_playground_preset: TTSPlaygroundSelectionPreset | None = None
        self._pending_profile_context_token: UUID | None = None
        self._pending_playground_navigation: SpeechTTSNavigationTarget | None = None
        self._pending_adopted_preset: TTSPlaygroundSelectionPreset | None = None
        self._profile_library_continuity: ProfileLibraryContinuity | None = None
        self._pending_profile_verification: ProfileVerificationResult | None = None
        self._profile_focus_sequence = 0
        self._profile_focus_restore_token: int | None = None
        self._profile_focus_restore_baseline = None
        self._voice_tool_origin: tuple[str, str | None] | None = None
        self._voice_tool_back_in_progress = False
        self._voice_tool_navigation_token = 0
        self._view_mount_lock = asyncio.Lock()
        # Bounded, process-local Playground axes survive only internal Lab
        # view switches. They are never written to global or Studio settings.
        self._playground_axis_values: dict[str, str] = dict(
            playground_axis_values or {}
        )
        if local_dependencies is None:
            local_dependencies = speech_local_dependency_availability(refresh=True)
        elif type(local_dependencies) is not SpeechLocalDependencyAvailability:
            raise TypeError("local_dependencies must be a Speech dependency snapshot")
        self._speech_local_dependencies = local_dependencies
        self._studio_store = StudioTTSPreferenceStore()
        self._global_preferences = SpeechSettingsPane._read_global_preferences()
        self._last_global_preferences_revision: int | None = None
        self._studio_load_result: StudioTTSLoadResult | None = None

    def receive_provider_configuration_changed(
        self,
        message: STTSProviderConfigurationChanged,
    ) -> None:
        """Refresh retained Lab panes once for each newer global revision."""

        revision = message.global_preferences_revision
        if type(revision) is not int or revision < 0:
            return
        previous = self._last_global_preferences_revision
        if previous is not None and revision <= previous:
            return
        snapshot = SpeechSettingsPane._read_global_preferences()
        playgrounds = list(self.query(SpeechPlaygroundPane))
        if not playgrounds:
            load_result = self._studio_load_result
            studio_snapshot = None if load_result is None else load_result.snapshot
            old_defaults = _seed_axis_defaults(
                studio_snapshot,
                self._global_preferences,
            )
            new_defaults = _seed_axis_defaults(studio_snapshot, snapshot)
            self._playground_axis_values = _rebase_inherited_axis_values(
                self._playground_axis_values,
                old_defaults=old_defaults,
                new_defaults=new_defaults,
            )
        self._global_preferences = snapshot
        self._last_global_preferences_revision = revision
        for pane in self.query(SpeechSettingsPane):
            callback = getattr(pane, "refresh_global_preferences", None)
            if callable(callback):
                callback(snapshot)
        for pane in playgrounds:
            callback = getattr(pane, "refresh_global_preferences", None)
            if callable(callback):
                callback(snapshot)

    def compose(self) -> ComposeResult:
        """Compose a non-interactive shell until Studio preferences are loaded."""

        with Container(classes="stts-content"):
            yield Static(
                "Loading Studio TTS preferences…",
                id="speech-studio-loading",
                classes="speech-status-line",
                markup=False,
            )
        self._mounted_view: str | None = None

    def on_mount(self) -> None:
        """Load and migrate Studio preferences away from the UI message pump."""

        self.run_worker(
            self._load_studio_preferences(),
            group="speech-studio-preferences-load",
            exclusive=True,
            exit_on_error=False,
        )

    async def _load_studio_preferences(self) -> None:
        """Publish one exact Studio snapshot before mounting an editable view."""

        try:
            result = await asyncio.to_thread(self._studio_store.load)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Studio TTS preferences could not be loaded for Speech Lab")
            result = StudioTTSLoadResult(
                StudioTTSPreferencesSnapshot(),
                StudioTTSLoadState.CORRUPT,
                ("speech_studio",),
            )
        self._studio_load_result = result
        if self.is_mounted:
            await self._mount_view(self.current_view, force=True)

    def _speech_capability_status_text(self) -> str:
        """Return a concise local speech dependency status for the sidebar."""
        return speech_capability_text(self._speech_local_dependencies)

    def _speech_capability_status_tooltip(self) -> str:
        """Return install guidance for local speech dependencies."""
        return speech_capability_tooltip(self._speech_local_dependencies)

    def _speech_dependencies_available(self) -> bool:
        return speech_dependencies_available(self._speech_local_dependencies)

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
        self.run_worker(
            self._mount_view(new_view),
            group="speech-view-mount",
            exit_on_error=False,
        )

    def select_view(
        self,
        view: str,
        *,
        profile_preset: TTSPlaygroundSelectionPreset | None = None,
        profile_context_token: UUID | None = None,
        navigation_target: SpeechTTSNavigationTarget | None = None,
    ) -> None:
        """Select an existing view and apply an exact one-shot preset."""

        if type(view) is not str or view not in STTS_VIEW_KEYS:
            raise ValueError("invalid Speech view")
        if profile_preset is not None and (
            view != "playground"
            or type(profile_preset) is not TTSPlaygroundSelectionPreset
        ):
            raise ValueError("invalid Speech profile preset")
        if profile_context_token is not None and (
            profile_preset is None or type(profile_context_token) is not UUID
        ):
            raise ValueError("invalid Speech profile context token")
        if navigation_target is not None and (
            view != "playground"
            or type(navigation_target) is not SpeechTTSNavigationTarget
        ):
            raise ValueError("invalid Speech navigation target")
        if (
            self.current_view in {"profiles", "blends"}
            and view != self.current_view
            and self._voice_tool_origin is not None
        ):
            self._invalidate_voice_tool_navigation()
        if view == "playground":
            if self._pending_profile_context_token != profile_context_token:
                _retire_profile_test_context(self._pending_profile_context_token)
            self._pending_playground_preset = profile_preset
            self._pending_profile_context_token = profile_context_token
            self._pending_playground_navigation = navigation_target
        else:
            _retire_profile_test_context(self._pending_profile_context_token)
            self._pending_playground_preset = None
            self._pending_profile_context_token = None
            self._pending_playground_navigation = None
        if self.current_view != view:
            self.current_view = view
            return
        if profile_preset is not None:
            self.run_worker(
                self._mount_view(view, force=True),
                group="speech-view-mount",
                exit_on_error=False,
            )
            return
        if navigation_target is not None:
            self.call_after_refresh(self._apply_pending_playground_navigation)

    async def request_view(
        self,
        view: str,
        *,
        profile_preset: TTSPlaygroundSelectionPreset | None = None,
        profile_context_token: UUID | None = None,
        navigation_target: SpeechTTSNavigationTarget | None = None,
        voice_tool_back_token: int | None = None,
    ) -> bool:
        """Select a view after resolving any dirty Studio preference draft."""

        if view != "settings" and not await self.confirm_studio_preferences_leave():
            return False
        if (
            voice_tool_back_token is not None
            and voice_tool_back_token != self._voice_tool_navigation_token
        ):
            return False
        current_view = self.current_view
        reset_current_tool = (
            voice_tool_back_token is None
            and current_view in {"profiles", "blends"}
        )
        if reset_current_tool:
            self._invalidate_voice_tool_navigation()
        if current_view == "profiles" and (
            view != "profiles" or reset_current_tool
        ):
            try:
                content = self.query_one(".stts-content", Container)
            except QueryError:
                content = None
            if content is not None:
                await content.remove_children()
        self.select_view(
            view,
            profile_preset=profile_preset,
            profile_context_token=profile_context_token,
            navigation_target=navigation_target,
        )
        if reset_current_tool and view == current_view:
            await self._mount_view(view, force=True)
        return True

    async def confirm_studio_preferences_leave(self) -> bool:
        """Delegate leave protection to the mounted Studio editor, if any."""

        if self.current_view != "settings":
            return True
        try:
            pane = self.query_one(SpeechSettingsPane)
        except QueryError:
            return True
        return await pane.confirm_leave()

    @staticmethod
    def _global_provider_configuration_states() -> dict[
        str, SpeechTTSConfigurationState
    ]:
        """Return safe provider setup states without contacting a provider."""

        try:
            values = get_runtime_config_snapshot().values
            state = load_global_speech_tts_state(
                values if isinstance(values, Mapping) else {}
            )
        except (OSError, TypeError, ValueError):
            state = load_global_speech_tts_state({})
        return {
            provider_id: global_speech_tts_provider_configuration_state(
                state,
                provider_id=provider_id,
            )
            for provider_id in BUILT_IN_TTS_PROVIDER_ORDER
        }

    async def _mount_view(self, new_view: str, *, force: bool = False) -> None:
        """Replace the mounted content when a view change requires it."""

        if type(new_view) is not str or new_view not in STTS_VIEW_KEYS:
            return
        async with self._view_mount_lock:
            if not force and new_view != self.current_view:
                return
            await self._mount_view_unlocked(new_view, force=force)

    async def _mount_view_unlocked(
        self,
        new_view: str,
        *,
        force: bool,
    ) -> None:
        """Replace one view while the caller holds the mount lock."""

        load_result = self._studio_load_result
        if load_result is None:
            return
        if not force and new_view == getattr(self, "_mounted_view", None):
            return
        try:
            content_container = self.query_one(".stts-content", Container)
        except QueryError:
            logger.debug(
                "STTS content container not mounted yet; deferring view "
                f"change to '{new_view}' until compose completes."
            )
            return
        if (
            force
            and new_view == "playground"
            and getattr(self, "_mounted_view", None) == "playground"
        ):
            self.call_after_refresh(self._apply_pending_playground_preset)
            return

        # Give widgets a chance to clean up before removal
        for child in content_container.children:
            if isinstance(child, SpeechPlaygroundPane):
                child.invalidate_profile_mount_callbacks()
                self._playground_axis_values = dict(child.axis_values)
            if hasattr(child, "cleanup") and callable(child.cleanup):
                try:
                    child.cleanup()
                except Exception as e:
                    logger.debug(f"Error during widget cleanup: {e}")

        # Await both sides of the replacement so a rapid rail action cannot
        # prune nested controls while their Mount events are still queued.
        await content_container.remove_children()

        # Add new content based on view
        self._mounted_view = new_view
        if new_view == "playground":
            preset = self._pending_playground_preset
            profile_context_token = self._pending_profile_context_token
            navigation_target = self._pending_playground_navigation
            await content_container.mount(
                SpeechPlaygroundPane(
                    id="speech-playground-pane",
                    profile_preset=preset,
                    profile_context_token=profile_context_token,
                    axis_values=self._playground_axis_values,
                    axis_defaults=_seed_axis_defaults(
                        load_result.snapshot,
                        self._global_preferences,
                    ),
                    studio_preferences=load_result.snapshot,
                    global_preferences=self._global_preferences,
                    navigation_target=navigation_target,
                    provider_configuration_states=(
                        self._global_provider_configuration_states()
                    ),
                    runtime_status_store=speech_tts_runtime_status_store(
                        self.app_instance
                    ),
                    local_dependencies=self._speech_local_dependencies,
                )
            )
            if self._pending_playground_preset is preset:
                self._pending_playground_preset = None
            if self._pending_profile_context_token == profile_context_token:
                self._pending_profile_context_token = None
            if self._pending_playground_navigation is navigation_target:
                self._pending_playground_navigation = None
        elif new_view == "profiles":
            focus_restore_token = self._begin_profile_focus_restore()
            if self._voice_tool_origin is not None:
                await content_container.mount(
                    Button(
                        "Back to previous Speech view",
                        id="speech-destination-back",
                        disabled=True,
                    )
                )
            await content_container.mount(
                STTSProfileLibrary(
                    self._load_profile_service,
                    default_profile_id_reader=(
                        lambda: get_cli_setting("app_tts", "default_profile_id", None)
                    ),
                    voice_bundle_service_loader=self._load_voice_bundle_service,
                    continuity=self._profile_library_continuity,
                    pending_verification=self._pending_profile_verification,
                    focus_restore_token=focus_restore_token,
                )
            )
            if self._voice_tool_origin is not None:
                self.call_after_refresh(self._enable_profile_destination_back)
        elif new_view == "blends":
            await content_container.mount(
                VoiceBlendsPane(id="speech-voice-blends-pane")
            )
        elif new_view == "settings":
            adopted = self._pending_adopted_preset
            await content_container.mount(
                SpeechSettingsPane(
                    id="speech-settings-pane",
                    store=self._studio_store,
                    global_preferences=self._global_preferences,
                    load_result=load_result,
                    adopted_preset=adopted,
                )
            )
            if self._pending_adopted_preset is adopted:
                self._pending_adopted_preset = None
        elif new_view == "voice-cloning":
            from tldw_chatbook.UI.Voice_Cloning_Window import VoiceCloningWindow

            await content_container.mount(VoiceCloningWindow())
        elif new_view == "effects":
            await content_container.mount(SpeechEffectsPane(id="speech-effects-pane"))
        elif new_view == "audiobook":
            await content_container.mount(AudioBookGenerationWidget())
        elif new_view == "dictation":
            await content_container.mount(DictationWindow())

        if new_view == "playground":
            # Top-level mount completion does not include composed descendants.
            # Hold the replacement lock until every strict axis Select and the
            # children its Mount handler queries have completed their mounts.
            select_ids = tuple(
                axis for axis in AXIS_CONTROLS if axis.endswith("-select")
            )
            while True:
                try:
                    select_nodes = tuple(
                        content_container.query_one(f"#{select_id}", Select)
                        for select_id in select_ids
                    )
                    select_children = tuple(
                        child
                        for select in select_nodes
                        for child in (
                            select.query_one("#label", Static),
                            select.query_one("SelectOverlay"),
                        )
                    )
                except QueryError:
                    await asyncio.sleep(0)
                    continue
                await asyncio.gather(
                    *(child._mounted_event.wait() for child in select_children)
                )
                await asyncio.gather(
                    *(select._mounted_event.wait() for select in select_nodes)
                )
                break

        # Selection styling is the rail's job now. These lines used to
        # `query_one("#view-*-btn")` for the four view buttons; those live on
        # STTSScreen since the sidebar moved, so every one of them would raise
        # NoMatches on the first view change. The screen watches
        # `current_view` and applies `is-active` itself.

    def playground_axis_snapshot(self) -> dict[str, str]:
        """Return detached process-local axes for a fresh Speech screen."""

        try:
            pane = self.query_one(SpeechPlaygroundPane)
        except QueryError:
            pass
        else:
            self._playground_axis_values = dict(pane.axis_values)
        return dict(self._playground_axis_values)

    def _apply_pending_playground_preset(
        self,
        retries_remaining: int = 3,
    ) -> None:
        """Apply a same-view exact preset after nested controls mount."""

        if not self.is_mounted or self.current_view != "playground":
            return
        try:
            playground = self.query_one(SpeechPlaygroundPane)
            playground.query_one("#tts-provider-select", Select).query_one(
                "SelectOverlay"
            )
        except QueryError:
            if retries_remaining > 0:
                self.call_after_refresh(
                    self._apply_pending_playground_preset,
                    retries_remaining - 1,
                )
            return
        preset = self._pending_playground_preset
        profile_context_token = self._pending_profile_context_token
        if preset is None:
            return
        playground.apply_profile_preset(
            preset,
            context_token=profile_context_token,
        )
        if self._pending_playground_preset is preset:
            self._pending_playground_preset = None
        if self._pending_profile_context_token == profile_context_token:
            self._pending_profile_context_token = None

    def _apply_pending_playground_navigation(self) -> None:
        """Apply a same-view provider/intent target without invoking it."""

        if not self.is_mounted or self.current_view != "playground":
            return
        target = self._pending_playground_navigation
        if target is None:
            return
        try:
            playground = self.query_one(SpeechPlaygroundPane)
        except QueryError:
            return
        playground.apply_navigation_target(target)
        if self._pending_playground_navigation is target:
            self._pending_playground_navigation = None

    @on(ProfilePreviewRequested)
    def on_profile_preview_requested(
        self,
        message: ProfilePreviewRequested,
    ) -> None:
        """Hand one exact preset to the next Playground mount."""
        if type(message.preset) is not TTSPlaygroundSelectionPreset:
            return
        if type(message.continuity) is not ProfileLibraryContinuity:
            return
        self._profile_library_continuity = message.continuity
        self.select_view(
            "playground",
            profile_preset=message.preset,
            profile_context_token=message.context_token,
        )

    @on(ProfileTestVerified)
    def on_profile_test_verified(self, message: ProfileTestVerified) -> None:
        """Retain one exact result until a fresh library mount rechecks it."""

        if type(message.result) is not ProfileVerificationResult:
            return
        self._pending_profile_verification = message.result

    @on(ProfileVerificationReconciled)
    def on_profile_verification_reconciled(
        self,
        message: ProfileVerificationReconciled,
    ) -> None:
        """Retire only the result consumed by the current library mount."""

        if self._pending_profile_verification == message.result:
            self._pending_profile_verification = None

    def _begin_profile_focus_restore(self) -> int | None:
        """Issue one owner token only for a library return with focus intent."""

        continuity = self._profile_library_continuity
        if continuity is None or continuity.focus_target is None:
            self._profile_focus_restore_token = None
            self._profile_focus_restore_baseline = None
            return None
        self._profile_focus_sequence += 1
        self._profile_focus_restore_token = self._profile_focus_sequence
        self._profile_focus_restore_baseline = None
        return self._profile_focus_sequence

    def cancel_profile_focus_restore(self) -> None:
        """Yield focus ownership after user input during a pending return."""

        if self._profile_focus_restore_token is None:
            return
        self._profile_focus_sequence += 1
        self._profile_focus_restore_token = None
        self._profile_focus_restore_baseline = None

    @on(ProfileLibraryRestoreReady)
    def on_profile_library_restore_ready(
        self,
        message: ProfileLibraryRestoreReady,
    ) -> None:
        """Wait beyond mount fallback before restoring bounded focus intent."""

        if message.ownership_token != self._profile_focus_restore_token:
            return
        self.call_after_refresh(
            self._arm_profile_library_focus_restore,
            message.ownership_token,
        )

    def _arm_profile_library_focus_restore(self, ownership_token: int) -> None:
        if ownership_token != self._profile_focus_restore_token:
            return
        self._profile_focus_restore_baseline = self.screen.focused
        self.call_after_refresh(
            self._complete_profile_library_focus_restore,
            ownership_token,
        )

    def _complete_profile_library_focus_restore(self, ownership_token: int) -> None:
        if ownership_token != self._profile_focus_restore_token:
            return
        if (
            self.current_view != "profiles"
            or self.screen.focused is not self._profile_focus_restore_baseline
        ):
            self.cancel_profile_focus_restore()
            return
        continuity = self._profile_library_continuity
        if continuity is None or continuity.focus_target is None:
            self.cancel_profile_focus_restore()
            return
        try:
            library = self.query_one(STTSProfileLibrary)
            target = library.query_one(f"#{continuity.focus_target}")
        except QueryError:
            self.cancel_profile_focus_restore()
            return
        target.focus()
        self._profile_focus_restore_token = None
        self._profile_focus_restore_baseline = None

    @on(AdoptStudioPreferencesRequested)
    def on_adopt_studio_preferences_requested(
        self,
        message: AdoptStudioPreferencesRequested,
    ) -> None:
        """Open the Studio editor with one explicit, still-unsaved adoption."""

        if type(message.preset) is not TTSPlaygroundSelectionPreset:
            return
        self._pending_adopted_preset = message.preset
        self.select_view("settings")

    @on(OpenStudioPreferencesRequested)
    def on_open_studio_preferences_requested(
        self,
        message: OpenStudioPreferencesRequested,
    ) -> None:
        """Open the Studio-only editor from the Playground action strip."""

        message.stop()
        self.select_view("settings")

    @on(OpenVoiceProfilesRequested)
    def on_open_voice_profiles_requested(
        self,
        message: OpenVoiceProfilesRequested,
    ) -> None:
        """Open the existing Voice Profiles library without dropping setup."""

        message.stop()
        self.app.push_screen(
            VoiceProfilePickerModal(
                self._load_profile_service,
                default_profile_id_reader=(
                    lambda: get_cli_setting("app_tts", "default_profile_id", None)
                ),
                voice_bundle_service_loader=self._load_voice_bundle_service,
            ),
            self._apply_voice_profile_picker_result,
        )

    def _apply_voice_profile_picker_result(
        self,
        result: tuple[TTSPlaygroundSelectionPreset, UUID] | None,
    ) -> None:
        """Apply one exact Preview result to the still-mounted Playground."""

        if (
            type(result) is not tuple
            or len(result) != 2
            or type(result[0]) is not TTSPlaygroundSelectionPreset
            or type(result[1]) is not UUID
        ):
            return
        preset, context_token = result
        if self.current_view != "playground":
            _retire_profile_test_context(context_token)
            return
        try:
            playground = self.query_one(SpeechPlaygroundPane)
        except QueryError:
            _retire_profile_test_context(context_token)
            return
        playground.apply_profile_preset(preset, context_token=context_token)

    @on(SpeechDestinationRequested)
    def on_speech_destination_requested(
        self,
        message: SpeechDestinationRequested,
    ) -> None:
        """Open an exact voice tool and retain its return destination."""

        message.stop()
        destination = resolve_speech_navigation(message.destination_id)
        focused_id = getattr(self.screen.focused, "id", None)
        origin = (
            self.current_view,
            focused_id if isinstance(focused_id, str) else None,
        )
        self.run_worker(
            self._open_speech_destination(destination, origin),
            group="speech-voice-tool-navigation",
            exclusive=True,
            exit_on_error=False,
        )

    async def _open_speech_destination(
        self,
        destination: SpeechNavigationDestination,
        origin: tuple[str, str | None],
    ) -> None:
        prior_origin = self._voice_tool_origin
        self._voice_tool_navigation_token += 1
        token = self._voice_tool_navigation_token
        self._voice_tool_back_in_progress = False
        self._voice_tool_origin = origin
        if not await self.request_view(destination.view):
            if (
                token == self._voice_tool_navigation_token
                and self._voice_tool_origin == origin
            ):
                self._voice_tool_origin = prior_origin

    @on(SpeechDestinationBackRequested)
    def on_speech_destination_back_requested(
        self,
        message: SpeechDestinationBackRequested,
    ) -> None:
        """Return to the originating view and restore its action focus."""

        message.stop()
        if self._voice_tool_back_in_progress:
            return
        origin = self._voice_tool_origin
        if origin is None:
            return
        self._voice_tool_back_in_progress = True
        self._voice_tool_navigation_token += 1
        token = self._voice_tool_navigation_token
        self._set_voice_tool_back_disabled(True)
        view, focus_id = origin
        if view in {"profiles", "blends"} or view not in STTS_VIEW_KEYS:
            view, focus_id = "playground", None
        self.run_worker(
            self._return_to_speech_origin(view, focus_id, origin, token),
            group="speech-voice-tool-navigation",
            exclusive=True,
            exit_on_error=False,
        )

    async def _return_to_speech_origin(
        self,
        view: str,
        focus_id: str | None,
        origin: tuple[str, str | None],
        token: int,
    ) -> None:
        self._voice_tool_origin = None
        try:
            navigated = await self.request_view(
                view,
                voice_tool_back_token=token,
            )
        except asyncio.CancelledError:
            if token == self._voice_tool_navigation_token:
                self._voice_tool_origin = origin
                self._voice_tool_back_in_progress = False
                self._set_voice_tool_back_disabled(False)
            raise
        except Exception as exc:
            if token == self._voice_tool_navigation_token:
                self._voice_tool_origin = origin
                self._voice_tool_back_in_progress = False
                self._set_voice_tool_back_disabled(False)
            logger.error(
                "Could not return to the originating Speech view "
                "(exception_type={})",
                type(exc).__name__,
            )
            return
        if token != self._voice_tool_navigation_token:
            return
        self._voice_tool_back_in_progress = False
        if not navigated:
            self._voice_tool_origin = origin
            self._set_voice_tool_back_disabled(False)
            return
        if focus_id is not None:
            self.call_after_refresh(self._restore_destination_focus, focus_id)

    def _invalidate_voice_tool_navigation(self) -> None:
        """Discard origin and any in-flight Back ownership."""

        self._voice_tool_navigation_token += 1
        self._voice_tool_origin = None
        self._voice_tool_back_in_progress = False

    def _set_voice_tool_back_disabled(self, disabled: bool) -> None:
        try:
            self.query_one("#speech-destination-back", Button).disabled = disabled
        except QueryError:
            return

    def _restore_destination_focus(self, focus_id: str, attempts: int = 20) -> None:
        try:
            self.query_one(f"#{focus_id}").focus()
        except QueryError:
            if attempts > 0:
                self.call_after_refresh(
                    self._restore_destination_focus,
                    focus_id,
                    attempts - 1,
                )
            return

    def _enable_profile_destination_back(self) -> None:
        """Expose Back only after the profile library is safe to leave."""

        if self.current_view != "profiles" or self._voice_tool_origin is None:
            return
        try:
            back = self.query_one("#speech-destination-back", Button)
            self.query_one("#stts-profile-edit-btn", Button)
        except QueryError:
            self.call_after_refresh(self._enable_profile_destination_back)
            return
        back.disabled = False

    def on_unmount(self) -> None:
        """Release any profile authority not yet transferred to a pane."""

        _retire_profile_test_context(self._pending_profile_context_token)
        self._pending_profile_context_token = None

    @on(StudioPreferencesSaved)
    def on_studio_preferences_saved(self, message: StudioPreferencesSaved) -> None:
        """Publish a Studio-only save to later Playground mounts."""

        self._studio_load_result = StudioTTSLoadResult(
            message.snapshot,
            StudioTTSLoadState.LOADED,
        )
        if message.reset_to_global:
            # Reset removes the Studio preference layer, so an exact axis that
            # was merely seeded from that layer must not survive as a bounded
            # Playground draft and continue outranking the inherited global.
            self._playground_axis_values.clear()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle sidebar button presses and delegate to content widgets"""
        # Handle sidebar buttons
        if event.button.id == "speech-destination-back":
            event.stop()
            self.post_message(SpeechDestinationBackRequested())
        elif event.button.id == "view-playground-btn":
            self.run_worker(
                self.request_view("playground"),
                exclusive=True,
                group="stts-request-view",
            )
        elif event.button.id == "view-profiles-btn":
            self.run_worker(
                self.request_view("profiles"),
                exclusive=True,
                group="stts-request-view",
            )
        elif event.button.id == "view-settings-btn":
            self.run_worker(
                self.request_view("settings"),
                exclusive=True,
                group="stts-request-view",
            )
        elif event.button.id == "view-audiobook-btn":
            self.run_worker(
                self.request_view("audiobook"),
                exclusive=True,
                group="stts-request-view",
            )
        elif event.button.id == "view-voice-cloning-btn":
            # Import and push the Voice Cloning window
            from tldw_chatbook.UI.Voice_Cloning_Window import VoiceCloningWindow

            self.app.push_screen(VoiceCloningWindow())
        elif event.button.id == "view-stt-btn":
            self.run_worker(
                self.request_view("dictation"),
                exclusive=True,
                group="stts-request-view",
            )
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

    async def _load_voice_bundle_service(self) -> object | None:
        """Resolve the app-owned portability service only when first used."""

        ensure_service = getattr(
            self.app_instance,
            "_ensure_tts_voice_bundle_service",
            None,
        )
        if not callable(ensure_service):
            return None
        try:
            return await ensure_service()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("TTS voice bundle portability is unavailable")
            return None


#
# End of STTS_Window.py
#######################################################################################################################
