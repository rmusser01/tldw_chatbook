"""Disposable review of a Buddy snapshot before independent character creation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Label, Select, Static, TextArea

from ...Chat.character_expression_playback import expression_motion_enabled
from ..Console.character_expression_avatar import CharacterExpressionAvatar
from ..modal_dismissal import SafeModalDismissMixin


@dataclass(frozen=True)
class BuddyCharacterCreated:
    """Committed identity plus the user's explicit navigation choice."""

    result: Any
    name: str
    open_console: bool


class BuddyCharacterReviewDialog(
    SafeModalDismissMixin, ModalScreen[BuddyCharacterCreated | None]
):
    """Keep conversion errors editable and publish only the reviewed bytes."""

    BINDINGS = (Binding("escape", "request_safe_cancel", "Cancel", show=False),)
    SAFE_MODAL_CONTENT = "#buddy-review"
    BUNDLED_CSS = """
    BuddyCharacterReviewDialog { align: center middle; }
    BuddyCharacterReviewDialog #buddy-review {
        width: 94%; max-width: 96; height: 94%;
        border: round $accent; background: $panel; padding: 0 1;
    }
    BuddyCharacterReviewDialog #buddy-review-scroll { height: 1fr; }
    BuddyCharacterReviewDialog .buddy-copy { height: auto; }
    BuddyCharacterReviewDialog .buddy-text { height: 4; }
    BuddyCharacterReviewDialog #buddy-preview-host,
    BuddyCharacterReviewDialog #buddy-portrait-host { height: 8; }
    BuddyCharacterReviewDialog #buddy-actions { height: 3; }
    BuddyCharacterReviewDialog #buddy-actions Button {
        width: 1fr; min-width: 0; border: none;
    }
    BuddyCharacterReviewDialog #buddy-status { height: auto; max-height: 4; }
    """

    def __init__(
        self,
        snapshot: Any,
        rows: Any,
        *,
        db: Any,
        local_service: Any,
        profile_root: Any,
        authority_guard: Callable[[], bool],
        config: dict,
        allow_open_console: bool = True,
    ) -> None:
        super().__init__()
        self.snapshot = snapshot
        self.rows = tuple(rows)
        self._db = db
        self._local_service = local_service
        self._profile_root = profile_root
        self._authority_guard = authority_guard
        self._config = config
        self._allow_open_console = allow_open_console
        self._conversion = None
        self._conversion_key = None
        self._result = None
        self._created_name = ""
        self._busy = False
        self._publishing = False
        self._review_closed = False
        self._preview_generation = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="buddy-review"):
            yield Static("Create character from Buddy", classes="buddy-copy")
            with VerticalScroll(id="buddy-review-scroll"):
                yield Static(
                    f"{self.snapshot.title}\nSource SHA-256: {self.snapshot.source_sha256}",
                    markup=False,
                    classes="buddy-copy",
                )
                artwork = self.snapshot.artwork
                artwork = artwork or {}
                notice = (
                    f"Original creator: {artwork.get('creator') or 'unspecified'}\n"
                    f"Artwork terms: {artwork.get('license') or 'unspecified'}"
                )
                for key in ("source_url", "notices"):
                    if artwork.get(key):
                        notice += "\n" + str(artwork[key])
                yield Static(notice, markup=False, classes="buddy-copy")
                yield Label("New character name")
                yield Input(self.snapshot.title, id="buddy-name", max_length=256)
                yield Label("Personality (optional)")
                yield TextArea(id="buddy-personality", classes="buddy-text")
                yield Label("Greeting (optional)")
                yield TextArea(id="buddy-greeting", classes="buddy-text")
                yield Static(
                    "Expression mappings — leave a key blank to exclude that state. A neutral expression is required.",
                    classes="buddy-copy",
                )
                for index, row in enumerate(self.rows):
                    suffix = " · fallback imagery" if row.fallback else ""
                    yield Static(
                        f"{row.source_state} · {row.frame_count} frames{suffix}",
                        markup=False,
                        classes="buddy-copy",
                    )
                    yield Input(
                        row.expression_key, id=f"buddy-mapping-{index}", max_length=100
                    )
                yield Checkbox(
                    "Preserve animation in created expressions",
                    value=True,
                    id="buddy-animate",
                )
                yield Label("Portrait source (independent of expression preview)")
                choices = [(row.source_state, row.source_state) for row in self.rows]
                default = (
                    "idle"
                    if any(row.source_state == "idle" for row in self.rows)
                    else self.rows[0].source_state
                )
                yield Select(
                    choices, value=default, allow_blank=False, id="buddy-portrait-state"
                )
                yield Label("Portrait frame (0 = first; blank = pack preview pose)")
                yield Input("", type="integer", id="buddy-portrait-frame")
                with Container(id="buddy-portrait-host"):
                    yield Static(
                        "Portrait preview appears after preparation.",
                        classes="buddy-copy",
                    )
                yield Label("Preview expression")
                yield Select(
                    choices, value=default, allow_blank=False, id="buddy-preview-state"
                )
                yield Select(
                    [("Dynamic", "dynamic"), ("Static — encoded frame zero", "static")],
                    value="dynamic",
                    allow_blank=False,
                    id="buddy-preview-mode",
                )
                with Container(id="buddy-preview-host"):
                    yield Static(
                        "Prepare preview to review the converted expressions.",
                        classes="buddy-copy",
                    )
                yield Static(
                    "", id="buddy-warnings", markup=False, classes="buddy-copy"
                )
                yield Checkbox(
                    "I reviewed the conversion warnings above",
                    id="buddy-warnings-accepted",
                )
            yield Static(
                "Edit mappings, then prepare preview before creating.",
                id="buddy-status",
                markup=False,
            )
            with Horizontal(id="buddy-actions"):
                yield Button("Prepare preview", id="buddy-prepare")
                yield Button("Create character", id="buddy-create", disabled=True)
                yield Button("Open in Console", id="buddy-open", disabled=True)
                yield Button("Cancel", id="buddy-cancel")

    def on_mount(self) -> None:
        super().on_mount()
        self._sync_controls()

    def collect_mappings(self) -> dict[str, str | None]:
        """Return every row, including explicit exclusions."""
        return {
            row.source_state: self.query_one(
                f"#buddy-mapping-{index}", Input
            ).value.strip()
            or None
            for index, row in enumerate(self.rows)
        }

    def _key(self) -> tuple:
        return (
            tuple(self.collect_mappings().items()),
            self.query_one("#buddy-animate", Checkbox).value,
            str(self.query_one("#buddy-portrait-state", Select).value),
            self.query_one("#buddy-portrait-frame", Input).value,
        )

    def _current(self) -> bool:
        return not self._review_closed and self.is_mounted and self._authority_guard()

    async def _source_current(self) -> bool:
        valid = await asyncio.to_thread(self.snapshot.is_current)
        return valid and self._current()

    def _set_status(self, text: str) -> None:
        if self.is_mounted and not self._review_closed:
            self.query_one("#buddy-status", Static).update(text)

    def _sync_controls(self) -> None:
        if not self.is_mounted or self._review_closed:
            return
        created = self._result is not None
        ready = self._conversion is not None and self._conversion_key == self._key()
        warnings_ok = (
            not self._conversion
            or not self._conversion.warnings
            or self.query_one("#buddy-warnings-accepted", Checkbox).value
        )
        self.query_one("#buddy-create", Button).disabled = bool(
            self._busy
            or created
            or not ready
            or not warnings_ok
            or not self.query_one("#buddy-name", Input).value.strip()
        )
        self.query_one("#buddy-prepare", Button).disabled = self._busy or created
        self.query_one("#buddy-open", Button).disabled = not created
        self.query_one("#buddy-open", Button).display = (
            created and self._allow_open_console
        )
        self.query_one("#buddy-prepare", Button).display = not created
        self.query_one("#buddy-create", Button).display = not created
        self.query_one("#buddy-cancel", Button).disabled = self._publishing
        self.query_one("#buddy-cancel", Button).label = "Done" if created else "Cancel"
        self.query_one("#buddy-review-scroll").disabled = self._publishing or created

    @on(Input.Changed)
    @on(TextArea.Changed)
    @on(Checkbox.Changed)
    def _edited(self, event: Any) -> None:
        if self.is_mounted:
            self._sync_controls()

    @on(Select.Changed)
    def _selection_changed(self, event: Select.Changed) -> None:
        if not self.is_mounted:
            return
        self._sync_controls()
        if self._conversion is not None and event.select.id in {
            "buddy-preview-state",
            "buddy-preview-mode",
        }:
            self.run_worker(self._show_preview, group="buddy-preview", exclusive=True)

    @on(Button.Pressed, "#buddy-prepare")
    def _prepare_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if not self._busy and self._result is None:
            self.run_worker(self._prepare, group="buddy-conversion", exclusive=True)

    async def _prepare(self) -> None:
        from ...Character_Chat.buddy_conversion import convert_buddy

        self._busy = True
        self._conversion = None
        self._preview_generation += 1
        await self.query_one("#buddy-preview-host", Container).remove_children()
        await self.query_one("#buddy-portrait-host", Container).remove_children()
        self.query_one("#buddy-warnings-accepted", Checkbox).value = False
        self._sync_controls()
        self._set_status("Preparing converted preview…")
        try:
            key = self._key()
            frame = int(key[3]) if key[3].strip() else None
            if frame is not None and frame < 0:
                raise ValueError("Portrait frame must be zero or greater.")
            if not await self._source_current():
                raise ValueError(
                    "Buddy or destination changed. Close and start a fresh review."
                )
            conversion = await asyncio.to_thread(
                convert_buddy,
                self.snapshot,
                dict(key[0]),
                animate=key[1],
                portrait_state=key[2],
                portrait_frame=frame,
            )
            if not await self._source_current():
                raise ValueError(
                    "Buddy or destination changed. Close and start a fresh review."
                )
            if self._key() != key:
                self._set_status(
                    "Edits changed during preparation. Prepare preview again."
                )
                return
            self._conversion = conversion
            self._conversion_key = key
            self.query_one("#buddy-warnings", Static).update(
                "\n".join(conversion.warnings)
            )
            self._set_status(
                "Review the conversion warnings before creating."
                if conversion.warnings
                else "Preview ready. Review each expression before creating."
            )
            await self._show_preview()
            host = self.query_one("#buddy-portrait-host", Container)
            await host.remove_children()
            if self._current():
                await host.mount(
                    CharacterExpressionAvatar(
                        conversion.portrait,
                        box=(24, 8),
                        animate=False,
                        is_current=lambda: (
                            self._current() and self._conversion is conversion
                        ),
                        monochrome=False,
                        mode="pixels",
                        id="buddy-portrait-preview",
                    )
                )
        except (ValueError, OSError, RuntimeError) as exc:
            self._set_status(f"Could not prepare: {exc}")
        finally:
            self._busy = False
            self._sync_controls()

    async def _show_preview(self) -> None:
        if not self._current() or self._conversion is None:
            return
        host = self.query_one("#buddy-preview-host", Container)
        self._preview_generation += 1
        generation = self._preview_generation
        await host.remove_children()
        if not self._current() or generation != self._preview_generation:
            return
        state = str(self.query_one("#buddy-preview-state", Select).value)
        expression = next(
            (
                item
                for item in self._conversion.expressions
                if item.source_state == state
            ),
            None,
        )
        if expression is None:
            await host.mount(Static("This state is excluded.", markup=False))
            return
        dynamic = self.query_one("#buddy-preview-mode", Select).value == "dynamic"
        animate = dynamic and expression_motion_enabled(
            self._config, react=True, manual=False
        )
        await host.mount(
            CharacterExpressionAvatar(
                expression.data,
                box=(24, 8),
                animate=animate,
                is_current=lambda: (
                    self._current()
                    and self._conversion is not None
                    and generation == self._preview_generation
                ),
                monochrome=False,
                mode="pixels",
                id="buddy-expression-preview",
            )
        )
        if dynamic and not animate:
            self._set_status("Motion preferences are showing a static preview.")

    @on(Button.Pressed, "#buddy-create")
    def _create_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if not self.query_one("#buddy-create", Button).disabled:
            self.run_worker(self._publish, group="buddy-publication", exclusive=True)

    async def _publish(self) -> None:
        from ...Character_Chat.buddy_conversion import publish_buddy_character

        if self._conversion is None or self._key() != self._conversion_key:
            return
        self._busy = self._publishing = True
        self._sync_controls()
        self._set_status("Creating independent character…")
        try:
            if not self._current():
                raise ValueError("Destination changed. Start a fresh review.")
            name = self.query_one("#buddy-name", Input).value.strip()
            task = asyncio.create_task(
                asyncio.to_thread(
                    publish_buddy_character,
                    self._conversion,
                    name=name,
                    personality=self.query_one("#buddy-personality", TextArea).text,
                    first_message=self.query_one("#buddy-greeting", TextArea).text,
                    db=self._db,
                    local_service=self._local_service,
                    profile_root=self._profile_root,
                    authority_guard=lambda: self.app.call_from_thread(self._current),
                )
            )
            try:
                result = await asyncio.shield(task)
            except asyncio.CancelledError:
                await task
                raise
            self._result = result
            self._created_name = name
            self._set_status(
                f"Created {name}."
                + (
                    " Private-file cleanup is pending."
                    if result.cleanup_pending
                    else " Choose Open in Console when ready."
                    if self._allow_open_console
                    else " Find it in Characters."
                )
            )
        except (ValueError, OSError, RuntimeError) as exc:
            self._set_status(f"Could not create: {exc}")
        finally:
            self._busy = self._publishing = False
            self._sync_controls()

    @on(Button.Pressed, "#buddy-open")
    def _open_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self._result is not None:
            self.dismiss_safe_once(
                BuddyCharacterCreated(self._result, self._created_name, True)
            )

    @on(Button.Pressed, "#buddy-cancel")
    async def _cancel_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="button")

    async def _perform_safe_cancel(self, *, source: str) -> None:
        """Cancel preparation freely; wait for atomic publication once admitted."""
        if not self._publishing:
            self._review_closed = True
            self.dismiss_safe_once(
                BuddyCharacterCreated(self._result, self._created_name, False)
                if self._result
                else None
            )

    def on_unmount(self) -> None:
        super().on_unmount()
        self._review_closed = True
        self._preview_generation += 1
        self._conversion = None
