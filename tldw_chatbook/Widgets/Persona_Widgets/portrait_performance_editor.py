"""Inline disposable portrait review; the screen owns every external operation."""

from __future__ import annotations

from collections.abc import Callable

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Collapsible, Input, Select, Static

from ...Character_Chat.portrait_compositing import ComponentRole
from ...Character_Chat.portrait_performance import (
    FRAME_ROLES,
    PortraitPerformanceRecord,
)
from ...Character_Chat.portrait_performance_authoring import PortraitPerformanceDraft
from ..Console.character_expression_avatar import CharacterExpressionAvatar


class CharacterPerformanceRequested(Message):
    """One explicit authoring action, without database or source authority."""

    def __init__(
        self,
        action: str,
        *,
        source: str = "idle",
        role: str = "rest",
        component: ComponentRole | None = None,
    ) -> None:
        super().__init__()
        self.action = action
        self.source = source
        self.role = role
        self.component = component


class PortraitPerformanceEditor(Vertical):
    """Keyboard-accessible six-pose import and review surface."""

    BUNDLED_CSS = """
    PortraitPerformanceEditor { height: $ds-height-auto; width: $ds-width-fill; }
    PortraitPerformanceEditor .portrait-review-row { height: $ds-height-auto; width: $ds-width-fill; }
    PortraitPerformanceEditor Button { width: $ds-width-auto; margin-right: $ds-space-1; }
    PortraitPerformanceEditor .portrait-review-images { height: $ds-height-auto; width: $ds-width-fill; }
    PortraitPerformanceEditor .portrait-review-image { width: $ds-portrait-review-cols; height: $ds-portrait-review-lines; margin-right: $ds-space-2; }
    PortraitPerformanceEditor .portrait-review-copy { height: $ds-height-auto; color: $ds-text-muted; }
    PortraitPerformanceEditor.portrait-review-compact .portrait-review-row,
    PortraitPerformanceEditor.portrait-review-compact .portrait-review-images { layout: vertical; }
    """

    def __init__(self) -> None:
        super().__init__(id="portrait-performance-editor")
        self._epoch = 0
        self._draft: PortraitPerformanceDraft | None = None
        self._busy = False
        self._allowed = False
        self.applied = False
        self.sample_running = False

    def compose(self) -> ComposeResult:
        with Collapsible(title="Living portrait", collapsed=True):
            yield Static(
                "Six matching stills: rest, small/open mouth and their blink variants. "
                "Saved local artwork only. Save changed artwork or Apply reaction artwork first; text edits stay unsaved.",
                classes="portrait-review-copy",
                markup=False,
            )
            yield Input(
                value="idle",
                placeholder="Saved expression: idle or expression key",
                id="portrait-performance-source",
            )
            yield Button(
                "Prepare saved artwork",
                id="portrait-performance-prepare",
                disabled=True,
            )
            yield Static(
                "Prepare artwork to import poses.",
                id="portrait-performance-status",
                markup=False,
            )
            yield Button(
                "Edit regions…", id="portrait-performance-regions", disabled=True
            )
            yield Static("", id="portrait-performance-slots", markup=False)
            with Horizontal(classes="portrait-review-row"):
                yield Select(
                    [(role.replace("_", " "), role) for role in FRAME_ROLES],
                    value="rest",
                    allow_blank=False,
                    id="portrait-performance-role",
                )
                yield Button(
                    "Import / replace…", id="portrait-performance-import", disabled=True
                )
                yield Button(
                    "Clear slot", id="portrait-performance-clear", disabled=True
                )
            with Horizontal(classes="portrait-review-images"):
                with Vertical(classes="portrait-review-image"):
                    yield Static("Saved source", markup=False)
                    yield Container(id="portrait-performance-source-image")
                with Vertical(classes="portrait-review-image"):
                    yield Static("Prepared poses", markup=False)
                    yield Container(id="portrait-performance-preview")
            yield Select(
                [
                    ("Cycle six poses", "cycle"),
                    ("Idle + blink", "idle"),
                    ("Speech sample", "speech"),
                    *[(r.replace("_", " "), r) for r in FRAME_ROLES],
                ],
                value="cycle",
                allow_blank=False,
                id="portrait-performance-preview-mode",
            )
            yield Button(
                "Review pairs…", id="portrait-performance-review", disabled=True
            )
            with Horizontal(classes="portrait-review-row"):
                yield Button(
                    "Apply performance", id="portrait-performance-apply", disabled=True
                )
                yield Button(
                    "Remove applied", id="portrait-performance-remove", disabled=True
                )
                yield Button(
                    "Cancel draft", id="portrait-performance-cancel", disabled=True
                )
            yield Select(
                [
                    ("Eyelids", "eyelids"),
                    ("Small mouth", "small"),
                    ("Open mouth", "open"),
                ],
                value="eyelids",
                allow_blank=False,
                id="portrait-performance-component",
            )
            with Horizontal(classes="portrait-review-row"):
                yield Button(
                    "Generate components (3)…",
                    id="portrait-performance-generate",
                    disabled=True,
                )
                yield Button(
                    "Retry component (1)…",
                    id="portrait-performance-regenerate",
                    disabled=True,
                )
                yield Button(
                    "Play speech sample…",
                    id="portrait-performance-sample",
                    disabled=True,
                )
            yield Static(
                "Generation sends the saved reference to your configured image-edit backend. Three requests per batch; one per selected component retry. Review destination before sending.",
                classes="portrait-review-copy",
                markup=False,
            )
            yield Static(
                "Profile-local: full profile backups include performances. Character cards, Actor Packs, sync and server exports do not.",
                classes="portrait-review-copy",
                markup=False,
            )

    def on_resize(self) -> None:
        self.call_after_refresh(self._sync_layout)

    @on(Collapsible.Expanded)
    def _expanded(self) -> None:
        self.call_after_refresh(self._sync_layout)

    def _sync_layout(self) -> None:
        if self.is_mounted:
            required = sum(
                child.outer_size.width for child in self.query(".portrait-review-image")
            )
            self.set_class(self.size.width < required, "portrait-review-compact")

    def reset(self) -> None:
        """Invalidate pending preview work and release disposable artwork."""
        self._epoch += 1
        if self._draft is not None:
            self._draft.cancel()
        self._draft = None
        self.applied = False
        if self.is_mounted:
            for selector in (
                "#portrait-performance-source-image",
                "#portrait-performance-preview",
            ):
                self.query_one(selector).remove_children()
            self.query_one("#portrait-performance-slots", Static).update("")
            self.status("Draft discarded; current artwork kept.")
            self.set_busy(False)

    def set_allowed(self, allowed: bool) -> None:
        """Expose the saved-character prerequisite; operation handlers revalidate."""
        self._allowed = allowed
        if self.is_mounted:
            self.set_busy(self._busy)

    def set_busy(self, busy: bool, *, saving: bool = False) -> None:
        """Guard duplicate work while allowing non-publishing work to be discarded."""
        self._busy = busy
        draft = self._draft
        for button in self.query(Button):
            action = button.id.removeprefix("portrait-performance-")
            button.disabled = (
                busy or not self._allowed or (action != "prepare" and draft is None)
            )
            if action == "apply" and draft is not None:
                button.disabled |= not draft.can_apply
            if action == "sample":
                button.label = (
                    "Stop sample" if self.sample_running else "Play speech sample…"
                )
                button.disabled = not self.sample_running and (
                    busy or draft is None or not draft.can_apply
                )
            if action == "regenerate":
                button.disabled |= draft is None or draft.composition is None
            if action == "review":
                button.disabled |= (
                    draft is None or not draft.can_apply or draft.composition is None
                )
            if action == "remove":
                button.disabled |= not self.applied
            if action == "cancel":
                button.disabled = saving or (draft is None and not busy)
        self.query_one("#portrait-performance-source", Input).disabled = (
            busy or draft is not None
        )
        for selector in (
            "#portrait-performance-role",
            "#portrait-performance-component",
            "#portrait-performance-preview-mode",
        ):
            self.query_one(selector, Select).disabled = busy

    def status(self, text: str) -> None:
        self.query_one("#portrait-performance-status", Static).update(text)

    async def show_draft(
        self,
        draft: PortraitPerformanceDraft,
        is_current: Callable[[], bool],
        *,
        motion: bool = True,
        read_speech: Callable | None = None,
    ) -> None:
        """Compare source and staged artwork with the Console's fitted renderer."""
        self._draft = draft
        self._epoch += 1
        epoch = self._epoch
        current = lambda: self._epoch == epoch and self._draft is draft and is_current()
        self.query_one("#portrait-performance-slots", Static).update(
            "  ·  ".join(
                f"{role.replace('_', ' ')}: {'ready' if role in draft.staged_roles else 'missing'}"
                for role in FRAME_ROLES
            )
        )
        source = self.query_one("#portrait-performance-source-image", Container)
        preview = self.query_one("#portrait-performance-preview", Container)
        await source.remove_children()
        await preview.remove_children()
        if not current():
            return
        # ds-runtime: Fit both images to the same measured token-sized review box.
        box = (max(1, source.parent.size.width), max(1, source.parent.size.height - 1))
        await source.mount(
            CharacterExpressionAvatar(
                draft.source_bytes,
                box=box,
                animate=False,
                is_current=current,
                monochrome=self.app.console.no_color,
                mode="pixels",
                id="portrait-performance-original",
            )
        )
        if not current():
            return
        complete = draft.can_apply
        selected = self.query_one("#portrait-performance-role", Select).value
        frame = draft.frame(selected)
        if not complete and frame is None:
            self.set_busy(self._busy)
            return
        record = (
            PortraitPerformanceRecord(
                draft.source, draft.expected_revision, True, draft.frames()
            )
            if complete
            else None
        )
        mode = str(self.query_one("#portrait-performance-preview-mode", Select).value)
        index = FRAME_ROLES.index(mode) if mode in FRAME_ROLES else 0
        self.query_one("#portrait-performance-preview-mode", Select).tooltip = (
            None
            if motion
            else "Motion is disabled in Appearance. Select individual poses for a still preview."
        )
        avatar = CharacterExpressionAvatar(
            record.frames[index].png if complete else frame.png,
            box=box,
            animate=complete and motion,
            is_current=current,
            monochrome=self.app.console.no_color,
            mode="pixels",
            id="portrait-performance-staged",
            performance=record if motion else None,
            read_speech=read_speech,
        )
        avatar.set_preview_mode(
            str(self.query_one("#portrait-performance-preview-mode", Select).value)
        )
        await preview.mount(avatar)
        self.set_busy(self._busy)

    @on(Button.Pressed)
    def _pressed(self, event: Button.Pressed) -> None:
        if not event.button.id or not event.button.id.startswith(
            "portrait-performance-"
        ):
            return
        event.stop()
        self.post_message(
            CharacterPerformanceRequested(
                event.button.id.removeprefix("portrait-performance-"),
                source=self.query_one(
                    "#portrait-performance-source", Input
                ).value.strip(),
                role=str(self.query_one("#portrait-performance-role", Select).value),
                component=str(
                    self.query_one("#portrait-performance-component", Select).value
                )
                if event.button.id == "portrait-performance-regenerate"
                else None,
            )
        )

    @on(Select.Changed, "#portrait-performance-preview-mode")
    def _preview_mode(self, event: Select.Changed) -> None:
        event.stop()
        if self._draft is not None:
            self.post_message(CharacterPerformanceRequested("preview"))

    @on(Select.Changed, "#portrait-performance-role")
    def _role_changed(self, event: Select.Changed) -> None:
        event.stop()
        if self._draft is not None:
            self.post_message(CharacterPerformanceRequested("preview"))
