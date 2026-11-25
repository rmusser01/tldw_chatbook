"""Configure an output and expose actions for its exact owner."""

import asyncio
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from rich.text import Text
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    DataTable,
    Input,
    Select,
    Static,
    TextArea,
)

from ...Research_Workspace.chat_models import ResearchRetrievalSettings
from ...Research_Workspace.studio_mapping import SERVER_TABLE_KINDS
from ...Research_Workspace.studio_models import (
    ResearchGenerationConfig,
    ResearchOutputKind,
    StudioOwnerKind,
    StudioOutputKind,
    StudioSpeechChoice,
)
from ...Research_Workspace.work_product_templates import TEXT_WORK_PRODUCT_TEMPLATES
from ...Widgets.modal_dismissal import SafeModalDismissMixin
from .studio_history import StudioHistoryRow
from .studio_output_card import output_label


async def play_voice_preview(content: bytes, audio_format: str) -> None:
    from ...TTS.audio_player import PlaybackState, get_audio_player

    player = get_audio_player()
    with tempfile.TemporaryDirectory(prefix="research-voice-") as directory:
        path = Path(directory) / f"sample.{audio_format}"
        path.write_bytes(content)
        started = asyncio.create_task(asyncio.to_thread(player.play, path))
        try:
            if not await asyncio.shield(started):
                raise ValueError("No audio player could play this voice sample.")
            if player.get_state() is PlaybackState.HANDED_OFF:
                await asyncio.sleep(8)
            while player.get_current_file() == path and player.is_playing():
                await asyncio.sleep(0.15)  # noqa: ASYNC110 -- The player exposes a status reader.
        finally:
            await asyncio.gather(started, return_exceptions=True)
            if player.get_current_file() == path and player.is_playing():
                await asyncio.to_thread(player.stop)


class ResearchStudioOptions(
    SafeModalDismissMixin, ModalScreen[ResearchGenerationConfig | None]
):
    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel", show=False)]
    SAFE_MODAL_CONTENT = "#research-studio-dialog"

    def __init__(
        self,
        kind: StudioOutputKind,
        owner: str,
        route: str,
        source_count: int,
        *,
        action: str = "Generate",
        slide_styles: tuple[tuple[str, tuple[str, str]], ...] = (),
        speech_choices: tuple[StudioSpeechChoice, ...] = (),
        speech_preview: Callable[[StudioSpeechChoice, float], Awaitable[bytes]] | None = None,
    ) -> None:
        super().__init__()
        self.kind = kind
        self.owner = owner
        self.route = route
        self.source_count = source_count
        self.action_label = action
        self.slide_styles = slide_styles
        self.speech_choices = speech_choices
        self.speech_preview = speech_preview
        self._previewing_voice = False

    @property
    def is_study(self) -> bool:
        return self.kind in {ResearchOutputKind.FLASHCARDS, ResearchOutputKind.QUIZ}

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="research-studio-dialog"):
            yield Static(output_label(self.kind), classes="research-pane-title")
            yield Static(
                f"{self.source_count} selected sources · Save to {self.owner}\n{self.route}",
                markup=False,
            )
            yield Static("Title")
            yield Input(placeholder=output_label(self.kind), id="research-studio-title")
            if self.kind in SERVER_TABLE_KINDS:
                yield Static("Table focus (optional)")
                yield Input(placeholder="What should the table extract?", id="research-studio-focus")
                if self.kind is ResearchOutputKind.DATA_TABLE:
                    yield Static("Column names (optional)")
                    yield Input(placeholder="Comma-separated column names", id="research-studio-columns")
                yield Static("Maximum rows (optional)")
                yield Input(placeholder="Maximum rows", type="integer", id="research-studio-max-rows")
            elif self.kind is ResearchOutputKind.SLIDES:
                yield Static("Visual style")
                yield Select(
                    [("Server default", "default"), *((Text(name), i) for i, (name, _) in enumerate(self.slide_styles))],
                    value="default", allow_blank=False, id="research-studio-visual-style",
                )
            elif self.kind is ResearchOutputKind.AUDIO_SUMMARY:
                yield Static("Summary focus (optional)")
                yield Input(placeholder="What should the summary cover?", id="research-studio-focus")
                yield Static("Speech provider, model, voice and format")
                yield Select(
                    [
                        (Text(f"{route.provider} · {route.model} · {route.voice} · {route.format.upper()}"), i)
                        for i, route in enumerate(self.speech_choices)
                    ],
                    value=0, allow_blank=False, id="research-studio-speech",
                )
                yield Static("Speech speed")
                yield Input(
                    "1.0", type="number", id="research-studio-speech-speed",
                    disabled=not self.speech_choices[0].supports_speed,
                )
                if self.speech_preview is not None:
                    yield Button("Preview voice", id="research-studio-voice-preview", compact=True)
            else:
                yield Static("Retrieval focus (optional for FTS)")
                yield Input(id="research-studio-focus")
                if self.is_study:
                    yield Static("Study items")
                    yield Input("8", type="integer", id="research-studio-items")
                yield Static("Detail")
                yield Select(
                    [
                        ("Brief", "brief"),
                        ("Standard", "standard"),
                        ("Detailed", "detailed"),
                    ],
                    value="standard",
                    allow_blank=False,
                    id="research-studio-detail",
                )
                with Collapsible(title="Advanced options"):
                    yield Static("Retrieval method")
                    yield Select(
                        [("FTS", "fts"), ("Vector", "semantic"), ("Hybrid", "hybrid")],
                        value="fts",
                        allow_blank=False,
                        id="research-studio-retrieval",
                    )
                    yield Static("Top K")
                    yield Input("8", type="integer", id="research-studio-top-k")
                    yield Static("Score threshold (optional)")
                    yield Input(type="number", id="research-studio-threshold")
                    yield Checkbox("Rerank evidence", id="research-studio-rerank")
                    yield Checkbox(
                        "Source citations required"
                        if self.kind in TEXT_WORK_PRODUCT_TEMPLATES
                        else "Include source citations",
                        value=True,
                        id="research-studio-citations",
                        disabled=self.kind in TEXT_WORK_PRODUCT_TEMPLATES,
                    )
                    yield Static("Temperature")
                    yield Input("0.3", type="number", id="research-studio-temperature")
                    yield Static("Top P")
                    yield Input("1.0", type="number", id="research-studio-top-p")
                    yield Static("Max output tokens")
                    yield Input("4096", type="integer", id="research-studio-tokens")
            yield Static("", id="research-studio-options-status", markup=False)
            yield Button(
                self.action_label,
                id="research-studio-generate",
                variant="primary",
                compact=True,
            )
            yield Button("Cancel", id="research-studio-close", compact=True)

    def _input(self, name: str) -> str:
        return self.query_one(f"#research-studio-{name}", Input).value

    @on(Select.Changed, "#research-studio-speech")
    def speech_changed(self, event: Select.Changed) -> None:
        if type(event.value) is int:
            speed = self.query_one("#research-studio-speech-speed", Input)
            speed.disabled = not self.speech_choices[event.value].supports_speed
            if speed.disabled:
                speed.value = "1.0"

    @on(Button.Pressed, "#research-studio-voice-preview")
    def preview_voice(self) -> None:
        selection = self.query_one("#research-studio-speech", Select).value
        if type(selection) is int and self.speech_preview is not None and not self._previewing_voice:
            self._previewing_voice = True
            self.run_worker(
                self._preview_voice(self.speech_choices[selection], self._input("speech-speed")),
                group="research-voice-preview", exclusive=True, exit_on_error=False,
            )

    async def _preview_voice(self, choice: StudioSpeechChoice, speed: str) -> None:
        preview = self.speech_preview
        if preview is None:
            self._previewing_voice = False
            return
        status = self.query_one("#research-studio-options-status", Static)
        status.update("Preparing voice preview…")
        try:
            async with asyncio.timeout(45):
                sample = await preview(choice, float(speed))
                status.update("Playing voice preview…")
                await play_voice_preview(sample, choice.format)
            status.update("Voice preview finished.")
        except ValueError as exc:
            status.update(str(exc))
        except Exception:
            status.update("The voice preview could not play. Check the selected Speech owner and retry.")
        finally:
            self._previewing_voice = False

    @on(Button.Pressed, "#research-studio-generate")
    def submit(self) -> None:
        try:
            if self.kind in SERVER_TABLE_KINDS:
                columns = self._input("columns") if self.kind is ResearchOutputKind.DATA_TABLE else ""
                rows = self._input("max-rows")
                options = ResearchGenerationConfig(
                    title=self._input("title"), focus=self._input("focus"),
                    table_columns=tuple(columns.split(",")) if columns.strip() else (),
                    table_max_rows=int(rows) if rows else None,
                )
            elif self.kind is ResearchOutputKind.SLIDES:
                selected = self.query_one("#research-studio-visual-style", Select).value
                options = ResearchGenerationConfig(
                    title=self._input("title"),
                    slides_visual_style=self.slide_styles[selected][1] if type(selected) is int else None,
                )
            elif self.kind is ResearchOutputKind.AUDIO_SUMMARY:
                choice = self.query_one("#research-studio-speech", Select).value
                options = ResearchGenerationConfig(
                    title=self._input("title"), focus=self._input("focus"),
                    speech=self.speech_choices[choice], speech_speed=float(self._input("speech-speed")),
                )
            else:
                threshold = self._input("threshold")
                retrieval = ResearchRetrievalSettings(
                    mode=self.query_one("#research-studio-retrieval", Select).value,
                    top_k=int(self._input("top-k")),
                    threshold=float(threshold) if threshold else None,
                    rerank=self.query_one("#research-studio-rerank", Checkbox).value,
                    citations=self.query_one("#research-studio-citations", Checkbox).value,
                    history_limit=0,
                )
                options = ResearchGenerationConfig(
                    title=self._input("title"),
                    focus=self._input("focus"),
                    detail=self.query_one("#research-studio-detail", Select).value,
                    item_count=int(self._input("items")) if self.is_study else 8,
                    temperature=float(self._input("temperature")),
                    top_p=float(self._input("top-p")),
                    max_tokens=int(self._input("tokens")),
                    retrieval=retrieval,
                )
        except ValueError as exc:
            self.query_one("#research-studio-options-status", Static).update(str(exc))
            return
        self.dismiss(options)

    @on(Button.Pressed, "#research-studio-close")
    def close(self) -> None:
        self.dismiss(None)


class ResearchStudioActions(SafeModalDismissMixin, ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "request_safe_cancel", "Close", show=False)]
    SAFE_MODAL_CONTENT = "#research-studio-actions-dialog"

    def __init__(
        self, row: StudioHistoryRow, actions: dict[str, str], limits: str, owner: str
    ) -> None:
        super().__init__()
        self.row = row
        self.actions = actions
        self.limits = limits
        self.owner = owner

    def compose(self) -> ComposeResult:
        with Vertical(id="research-studio-actions-dialog"):
            with VerticalScroll(classes="research-studio-action-scroll"):
                yield Static(
                    self.row.title, classes="research-pane-title", markup=False
                )
                yield Static(self.owner, markup=False)
                if self.row.sources_changed:
                    yield Static("Sources changed since generation")
                if self.row.output.owner_kind is StudioOwnerKind.SERVER_DATA_TABLE and self.row.record:
                    record = self.row.record
                    preview = DataTable(id="research-studio-native-table", cursor_type="row")
                    columns = record.get("columns", ())
                    for column in columns:
                        preview.add_column(Text(str(column["name"])[:80]), key=column["column_id"])
                    for row in record.get("rows", ()):
                        values = row["data"]
                        cells = (
                            [values.get(column["column_id"], "") for column in columns]
                            if isinstance(values, dict) else values
                        )
                        preview.add_row(*(Text(str(cell)[:120]) for cell in cells[:len(columns)]))
                    yield Static(f"Showing {preview.row_count} of {record['table']['row_count']} rows")
                    yield preview
                if self.row.output.owner_kind is StudioOwnerKind.SERVER_SLIDES and self.row.record:
                    slides = self.row.record["slides"]
                    yield Static(f"Showing {min(5, len(slides))} of {len(slides)} slides")
                    for index, slide in enumerate(slides[:5], 1):
                        yield Static(
                            f"{slide.get('title') or f'Slide {index}'}\n{slide.get('content', '')[:700]}",
                            markup=False,
                        )
                if self.row.output.owner_kind in {StudioOwnerKind.SERVER_AUDIO, StudioOwnerKind.LOCAL_AUDIO} and self.row.record:
                    yield Static(f"Saved audio · {self.row.record['format'].upper()}")
                if self.row.content:
                    yield Static(self.row.content, markup=False)
                if self.limits:
                    yield Static(self.limits, markup=False)
                for action, label in self.actions.items():
                    yield Button(
                        label, id=f"research-studio-action-{action}", compact=True
                    )
            yield Button("Close", id="research-studio-actions-close", compact=True)

    @on(Button.Pressed)
    def pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "research-studio-actions-close":
            self.dismiss(None)
        elif button_id.startswith("research-studio-action-"):
            self.dismiss(button_id.removeprefix("research-studio-action-"))


class ResearchStudioEdit(SafeModalDismissMixin, ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel", show=False)]
    SAFE_MODAL_CONTENT = "#research-studio-edit-dialog"

    def __init__(self, row: StudioHistoryRow) -> None:
        super().__init__()
        self.row = row

    def compose(self) -> ComposeResult:
        with Vertical(id="research-studio-edit-dialog"):
            yield Static(self.row.title, classes="research-pane-title", markup=False)
            yield TextArea(self.row.content, id="research-studio-edit-body")
            yield Button(
                "Save edit",
                id="research-studio-edit-save",
                compact=True,
                variant="primary",
            )
            yield Button("Cancel", id="research-studio-edit-cancel", compact=True)

    @on(Button.Pressed, "#research-studio-edit-save")
    def save(self) -> None:
        self.dismiss(self.query_one(TextArea).text)

    @on(Button.Pressed, "#research-studio-edit-cancel")
    def close(self) -> None:
        self.dismiss(None)
