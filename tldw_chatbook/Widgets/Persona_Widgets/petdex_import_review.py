"""Scrollable source, terms and animation mapping review for Petdex imports."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Label, Select, Static, TextArea

from ...Petdex.review import (
    PetdexPreviewCodecUnavailable,
    PetdexReviewedArchive,
    drain_thread,
    preview_state,
)
from ..modal_dismissal import SafeModalDismissMixin

_REQUIRED = ("idle", "thinking", "error", "listening", "speaking")


class PetdexImportReviewDialog(
    SafeModalDismissMixin, ModalScreen[PetdexReviewedArchive | None]
):
    """Review immutable Petdex bytes before accepting an unpublished native draft."""

    BINDINGS = (Binding("escape", "request_safe_cancel", "Cancel", show=False),)
    SAFE_MODAL_CONTENT = "#petdex-review"
    BUNDLED_CSS = """
    PetdexImportReviewDialog { align: center middle; }
    PetdexImportReviewDialog #petdex-review {
        width: 94%; max-width: 96; height: 94%;
        border: round $accent; background: $panel; padding: 0 1;
    }
    PetdexImportReviewDialog #petdex-scroll { height: 1fr; }
    PetdexImportReviewDialog .petdex-copy { height: auto; }
    PetdexImportReviewDialog #petdex-states { height: 10; }
    PetdexImportReviewDialog #petdex-preview-host { height: 10; }
    PetdexImportReviewDialog .petdex-actions { height: 3; }
    PetdexImportReviewDialog .petdex-actions Button {
        width: 1fr; min-width: 0; border: none;
    }
    """

    def __init__(
        self,
        *,
        authority_guard: Callable[[], bool],
        config: dict,
        independent: bool = False,
    ) -> None:
        super().__init__()
        self._authority_guard = authority_guard
        self._config = config
        self._independent = independent
        self.source = None
        self.inspection = None
        self._prepared = None
        self._prepared_key = None
        self._prepared_states = ()
        self._busy = False
        self._review_closed = False
        self._cancel_event = threading.Event()
        self._preview_generation = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="petdex-review"):
            yield Static(
                "Import Petdex as an independent Buddy"
                if self._independent
                else "Import Petdex into this Persona’s visual draft",
                classes="petdex-copy",
            )
            with VerticalScroll(id="petdex-scroll"):
                yield Label(
                    "Public pet URL or slug; or local folder, ZIP or pet.json path"
                )
                yield Input(id="petdex-source", max_length=4096)
                with Horizontal(classes="petdex-actions"):
                    yield Button("Fetch URL / slug", id="petdex-fetch")
                    yield Button("Read local path", id="petdex-local")
                    yield Button("Choose package…", id="petdex-choose")
                yield Static(
                    "No source loaded.",
                    id="petdex-credits",
                    markup=False,
                    classes="petdex-copy",
                )
                yield Static(
                    "", id="petdex-layout", markup=False, classes="petdex-copy"
                )
                yield Label("Source states — editable JSON array")
                yield Static(
                    'Each row needs {"name":"idle","row":0,"frames":4,"duration_ms":1000,"loop":true}. '
                    "Rows start at zero. Unknown v2 layouts require explicit rows, counts and timing.",
                    classes="petdex-copy",
                    markup=False,
                )
                yield TextArea("[]", id="petdex-states")
                yield Button("Apply state list", id="petdex-apply-states")
                yield Static(
                    "Required native mappings (all must be selected)",
                    classes="petdex-copy",
                )
                for name in _REQUIRED:
                    yield Label(name.capitalize())
                    yield Select([], id=f"petdex-map-{name}")
                yield Label("Preview source state")
                yield Select([], id="petdex-preview-state")
                with Container(id="petdex-preview-host"):
                    yield Static(
                        "Prepare preview after reviewing states and mappings.",
                        classes="petdex-copy",
                    )
                yield Static(
                    "", id="petdex-warnings", markup=False, classes="petdex-copy"
                )
                yield Checkbox(
                    "I reviewed the source terms and mapping warnings",
                    id="petdex-reviewed",
                )
                yield Static(
                    "Apply in Buddy management installs this reviewed artwork."
                    if self._independent
                    else "Active visuals stay unchanged until Save Pack.",
                    classes="petdex-copy",
                )
                yield Static(
                    "Load a source to begin.",
                    id="petdex-status",
                    markup=False,
                    classes="petdex-copy",
                )
            with Horizontal(classes="petdex-actions"):
                yield Button("Prepare preview", id="petdex-prepare", disabled=True)
                yield Button("Use draft", id="petdex-accept", disabled=True)
                yield Button("Cancel", id="petdex-cancel")

    def _current(self) -> bool:
        return self.is_mounted and not self._review_closed and self._authority_guard()

    def _status(self, message: str) -> None:
        if self.is_mounted and not self._review_closed:
            self.query_one("#petdex-status", Static).update(message)

    def collect_states(self) -> tuple:
        """Decode the explicit schema; conversion remains the validation boundary."""
        data = json.loads(self.query_one("#petdex-states", TextArea).text)
        if not isinstance(data, list) or not data:
            raise ValueError("Provide at least one source state.")
        fields = {"name", "row", "frames", "duration_ms", "loop"}
        if any(not isinstance(item, dict) or set(item) != fields for item in data):
            raise ValueError(
                "Each state needs name, row, frames, duration_ms and loop."
            )
        from ...Petdex.conversion import PetdexState

        if any(
            type(item["name"]) is not str
            or not item["name"]
            or any(
                type(item[key]) is not int for key in ("row", "frames", "duration_ms")
            )
            or type(item["loop"]) is not bool
            for item in data
        ):
            raise ValueError(
                "State names must be text, row/count/duration integers and loop a boolean."
            )
        return tuple(PetdexState(**item) for item in data)

    def _key(self) -> tuple:
        return (
            self.query_one("#petdex-states", TextArea).text,
            tuple(
                (name, self.query_one(f"#petdex-map-{name}", Select).value)
                for name in _REQUIRED
            ),
        )

    def _sync(self) -> None:
        if not self.is_mounted or self._review_closed:
            return
        for action in ("fetch", "local", "choose", "apply-states"):
            self.query_one(f"#petdex-{action}", Button).disabled = self._busy
        self.query_one("#petdex-prepare", Button).disabled = (
            self._busy or self.source is None
        )
        self.query_one("#petdex-accept", Button).disabled = bool(
            self._busy
            or self._prepared is None
            or self._prepared_key != self._key()
            or not self.query_one("#petdex-reviewed", Checkbox).value
        )

    @on(TextArea.Changed)
    @on(Checkbox.Changed)
    def _edited(self, event: Any) -> None:
        self._sync()

    @on(Select.Changed)
    def _selected(self, event: Select.Changed) -> None:
        self._sync()
        if event.select.id == "petdex-preview-state" and self._prepared is not None:
            self.run_worker(self._show_preview, group="petdex-preview", exclusive=True)

    @on(Button.Pressed, "#petdex-fetch")
    @on(Button.Pressed, "#petdex-local")
    @on(Button.Pressed, "#petdex-choose")
    def _load_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if not self._busy:
            self.run_worker(
                self._load(event.button.id), group="petdex-load", exclusive=True
            )

    async def _load(self, action: str) -> None:
        from ...Petdex.conversion import inspect_petdex
        from ...Petdex.registry import fetch_petdex_source
        from ...Petdex.sources import read_local_package
        from ..enhanced_file_picker import EnhancedFileOpen, Filters

        self._busy = True
        self._sync()
        try:
            value = self.query_one("#petdex-source", Input).value.strip()
            if action == "petdex-choose":
                value = await self.app.push_screen_wait(
                    EnhancedFileOpen(
                        title="Choose Petdex ZIP or pet.json (or enter a folder path)",
                        filters=Filters(
                            (
                                "Petdex package",
                                lambda path: (
                                    path.suffix.lower() == ".zip"
                                    or path.name == "pet.json"
                                ),
                            )
                        ),
                        context="petdex_package_import",
                    )
                )
                if not value or not self._current():
                    return
            self.source = None
            self._prepared = None
            self._preview_generation += 1
            self._status("Reading bounded Petdex source…")
            source = (
                await drain_thread(
                    fetch_petdex_source,
                    str(value),
                    cancel_requested=self._cancel_event.is_set,
                )
                if action == "petdex-fetch"
                else await drain_thread(read_local_package, value)
            )
            if not self._current():
                return
            inspection = await drain_thread(inspect_petdex, source)
            if not self._current() or not await drain_thread(source.is_current):
                return
            if not self._current():
                return
            self.source, self.inspection = source, inspection
            self._prepared = None
            self.query_one("#petdex-reviewed", Checkbox).value = False
            artwork = source.artwork or {}
            credits = (
                f"{source.title}\nCreator: {artwork.get('creator') or 'unspecified'}\n"
                f"Source: {artwork.get('source_url') or 'local package'}\n"
                f"Artwork terms: {artwork.get('license') or 'unspecified'}\n"
                f"{artwork.get('notices') or ''}\nSource SHA-256: {source.source_sha256}"
            )
            self.query_one("#petdex-credits", Static).update(credits)
            self.query_one("#petdex-layout", Static).update(
                f"{inspection.rows} rows · {inspection.cell_width}×{inspection.cell_height} cells · {inspection.mapping_source}"
            )
            self.query_one("#petdex-states", TextArea).load_text(
                json.dumps([asdict(state) for state in inspection.states], indent=2)
            )
            self._set_state_options(inspection.states)
            self.query_one("#petdex-warnings", Static).update(
                "\n".join(inspection.warnings)
            )
            self._status(
                "Review source states and mappings, then prepare preview."
                if inspection.states
                else "This layout needs a manual state list before preview."
            )
        except (ValueError, OSError, RuntimeError) as exc:
            self._status(f"Could not load source: {exc}")
        finally:
            self._busy = False
            self._sync()

    def _set_state_options(self, states: tuple) -> None:
        choices = [(state.name, state.name) for state in states]
        names = {state.name for state in states}
        defaults = {
            "idle": "idle",
            "thinking": "review",
            "error": "failed",
            "listening": "waiting",
            "speaking": "idle",
        }
        for name in _REQUIRED:
            select = self.query_one(f"#petdex-map-{name}", Select)
            previous = select.value
            select.set_options(choices)
            value = previous if previous in names else defaults[name]
            if value not in names:
                value = "idle" if "idle" in names else Select.NULL
            select.value = value
        preview = self.query_one("#petdex-preview-state", Select)
        preview.set_options(choices)
        preview.value = (
            "idle" if "idle" in names else (states[0].name if states else Select.NULL)
        )

    @on(Button.Pressed, "#petdex-apply-states")
    def _apply_states(self, event: Button.Pressed) -> None:
        event.stop()
        try:
            self._set_state_options(self.collect_states())
            self._prepared = None
            self._status(
                "State list applied. Select all required mappings and prepare preview."
            )
        except (ValueError, TypeError) as exc:
            self._status(f"Invalid state list: {exc}")
        self._sync()

    @on(Button.Pressed, "#petdex-prepare")
    def _prepare_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if not self._busy:
            self.run_worker(self._prepare, group="petdex-prepare", exclusive=True)

    async def _prepare(self) -> None:
        from ...Petdex.conversion import build_petdex_archive

        source = self.source
        if source is None:
            return
        self._busy = True
        self._prepared = None
        self._sync()
        try:
            key = self._key()
            states = self.collect_states()
            mappings = dict(key[1])
            if any(type(value) is not str for value in mappings.values()):
                raise ValueError("Select every required native mapping.")
            if not await drain_thread(source.is_current) or not self._current():
                raise ValueError("Source or destination changed. Start a fresh review.")
            unchanged = states == self.inspection.states
            archive = await drain_thread(
                build_petdex_archive,
                source,
                states=None if unchanged else states,
                mappings=mappings,
            )
            if not self._current() or self.source is not source or key != self._key():
                return
            if not await drain_thread(source.is_current) or not self._current():
                raise ValueError("Source changed. Load it again.")
            if key != self._key():
                return
            self._prepared = archive
            self._prepared_key = key
            self._prepared_states = states
            warnings = list(self.inspection.warnings)
            warnings.extend(
                f"{name} uses {target}"
                + (" (idle fallback)" if name != "idle" and target == "idle" else "")
                for name, target in mappings.items()
            )
            warnings.append(
                "Source states are retained as custom animations. Import remains unsaved until "
                + ("Apply." if self._independent else "Save Pack.")
            )
            self.query_one("#petdex-warnings", Static).update("\n".join(warnings))
            self.query_one("#petdex-reviewed", Checkbox).value = False
            self._status(
                "Preview ready. Review each source state, terms and mappings before Use draft."
            )
            await self._show_preview()
        except (ValueError, OSError, RuntimeError, TypeError) as exc:
            self._status(f"Could not prepare: {exc}")
        finally:
            self._busy = False
            self._sync()

    async def _show_preview(self) -> None:
        from ...Chat.character_expression_playback import expression_motion_enabled
        from ..Console.character_expression_avatar import CharacterExpressionAvatar

        if not self._current() or self._prepared is None:
            return
        state = next(
            (
                item
                for item in self._prepared_states
                if item.name == self.query_one("#petdex-preview-state", Select).value
            ),
            None,
        )
        if state is None:
            return
        self._preview_generation += 1
        generation = self._preview_generation
        animated = True
        try:
            data = await drain_thread(
                preview_state, self.source, self.inspection, state
            )
        except PetdexPreviewCodecUnavailable:
            animated = False
            data = await drain_thread(
                preview_state,
                self.source,
                self.inspection,
                state,
                animate=False,
            )
            if self._current() and generation == self._preview_generation:
                self._status(
                    "Showing the first frame: animated preview needs Pillow with WebP support. "
                    "The native pack keeps the full animation."
                )
        if not self._current() or generation != self._preview_generation:
            return
        host = self.query_one("#petdex-preview-host", Container)
        await host.remove_children()
        if not self._current() or generation != self._preview_generation:
            return
        await host.mount(
            CharacterExpressionAvatar(
                data,
                box=(24, 10),
                animate=animated
                and expression_motion_enabled(self._config, react=True, manual=False),
                is_current=lambda: (
                    self._current() and generation == self._preview_generation
                ),
                monochrome=False,
                mode="pixels",
                id="petdex-preview",
            )
        )

    @on(Button.Pressed, "#petdex-accept")
    def _accept_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if not event.button.disabled:
            self.run_worker(self._accept, group="petdex-accept", exclusive=True)

    async def _accept(self) -> None:
        source, archive, key = self.source, self._prepared, self._prepared_key
        if source is None or archive is None:
            return
        self._busy = True
        self._sync()
        try:
            if (
                not await drain_thread(source.is_current)
                or not self._current()
                or self._key() != key
                or not self.query_one("#petdex-reviewed", Checkbox).value
            ):
                self._status(
                    "Source, destination or mappings changed. Prepare a fresh preview."
                )
                return
            self._review_closed = True
            self.dismiss_safe_once(PetdexReviewedArchive(archive, source))
        finally:
            self._busy = False
            self._sync()

    @on(Button.Pressed, "#petdex-cancel")
    async def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="button")

    async def _perform_safe_cancel(self, *, source: str) -> None:
        self._review_closed = True
        self._cancel_event.set()
        self.dismiss_safe_once(None)

    def on_unmount(self) -> None:
        super().on_unmount()
        self._review_closed = True
        self._cancel_event.set()
        self._preview_generation += 1
