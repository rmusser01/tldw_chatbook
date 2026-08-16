"""Metadata-only Visual Identity pack browser for the Personas workbench."""

from __future__ import annotations

from typing import Any, Literal

from textual import events, on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Input, OptionList, Static
from textual.widgets.option_list import Option

from .personas_pane_messages import (
    VisualIdentityAssetMetadata,
    VisualIdentityPackClearRequested,
    VisualIdentityPackGenerateRequested,
    VisualIdentityPackMetadata,
    VisualIdentityPackPreviewRequested,
    VisualIdentityPackReplaceRequested,
    VisualIdentityPackSaveRequested,
)


class VisualIdentityPackGenerateAllRequested(Message):
    """Ask the screen to stage all expressions through image generation."""


class VisualIdentityPackCancelRequested(Message):
    """Ask the screen to cancel and discard the unpublished candidate."""


class PersonasVisualIdentityPackWidget(Vertical):
    """Browse path-free pack metadata and request one selected preview."""

    BUNDLED_CSS = """
    PersonasVisualIdentityPackWidget {
        width: 100%;
        height: 1fr;
        min-height: 16;
    }

    PersonasVisualIdentityPackWidget #personas-visual-identity-title,
    PersonasVisualIdentityPackWidget #personas-visual-identity-notice,
    PersonasVisualIdentityPackWidget #personas-visual-identity-count,
    PersonasVisualIdentityPackWidget #personas-visual-identity-label,
    PersonasVisualIdentityPackWidget #personas-visual-identity-key,
    PersonasVisualIdentityPackWidget #personas-visual-identity-dirty {
        width: 100%;
        height: auto;
    }

    PersonasVisualIdentityPackWidget #personas-visual-identity-notice,
    PersonasVisualIdentityPackWidget #personas-visual-identity-count,
    PersonasVisualIdentityPackWidget #personas-visual-identity-key,
    PersonasVisualIdentityPackWidget #personas-visual-identity-dirty {
        color: $text-muted;
    }

    PersonasVisualIdentityPackWidget #personas-visual-identity-filter {
        width: 100%;
        height: 3;
    }

    PersonasVisualIdentityPackWidget #personas-visual-identity-body {
        width: 100%;
        height: 1fr;
        min-height: 5;
    }

    PersonasVisualIdentityPackWidget #personas-visual-identity-results {
        width: 2fr;
        height: 100%;
        min-width: 0;
    }

    PersonasVisualIdentityPackWidget #personas-visual-identity-preview {
        width: 1fr;
        height: 100%;
        min-width: 0;
        margin-left: 1;
        padding: 1;
        background: $surface;
    }

    PersonasVisualIdentityPackWidget #personas-visual-identity-preview-image {
        width: 100%;
        height: 1fr;
        content-align: center middle;
    }

    PersonasVisualIdentityPackWidget #personas-visual-identity-actions {
        width: 100%;
        height: 3;
        overflow-x: auto;
    }

    PersonasVisualIdentityPackWidget #personas-visual-identity-actions Button {
        width: 1fr;
        height: 3;
        min-width: 0;
        border: none;
        margin-right: 1;
    }

    PersonasVisualIdentityPackWidget #personas-visual-identity-actions Button:focus {
        outline: heavy $accent;
    }

    PersonasVisualIdentityPackWidget.-narrow #personas-visual-identity-preview-image {
        display: none;
    }"""

    def __init__(
        self, pack: VisualIdentityPackMetadata | None = None, **kwargs: Any
    ) -> None:
        kwargs.setdefault("id", "personas-visual-identity-pack")
        super().__init__(**kwargs)
        self.pack = pack
        self._filtered: tuple[VisualIdentityAssetMetadata, ...] = ()
        self._selected: VisualIdentityAssetMetadata | None = None
        self._staged: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        yield Static("Visual Identity", id="personas-visual-identity-title")
        yield Static("", id="personas-visual-identity-notice", markup=False)
        yield Input(
            placeholder="Filter expressions",
            id="personas-visual-identity-filter",
        )
        yield Static("0 / 0", id="personas-visual-identity-count")
        with Horizontal(id="personas-visual-identity-body"):
            yield OptionList(id="personas-visual-identity-results")
            with Vertical(id="personas-visual-identity-preview"):
                yield Static("", id="personas-visual-identity-label", markup=False)
                yield Static("", id="personas-visual-identity-key", markup=False)
                yield Container(id="personas-visual-identity-preview-image")
        yield Static("No staged changes", id="personas-visual-identity-dirty")
        with Horizontal(id="personas-visual-identity-actions"):
            yield Button(
                "Replace…",
                id="personas-visual-identity-replace",
                classes="console-action-secondary",
            )
            yield Button(
                "Generate",
                id="personas-visual-identity-generate",
                classes="console-action-secondary",
            )
            yield Button(
                "Generate All",
                id="personas-visual-identity-generate-all",
                classes="console-action-secondary",
            )
            yield Button(
                "Clear",
                id="personas-visual-identity-clear",
                classes="console-action-subdued",
            )
            yield Button(
                "Save",
                id="personas-visual-identity-save",
                classes="console-action-primary",
                disabled=True,
            )
            yield Button(
                "Cancel",
                id="personas-visual-identity-cancel",
                classes="console-action-subdued",
            )

    def on_mount(self) -> None:
        self._sync_pack_copy()
        self.apply_filter("")
        self._sync_narrow()
        self.query_one("#personas-visual-identity-cancel", Button).display = False

    def on_resize(self, event: events.Resize) -> None:
        self._sync_narrow(event.size.width)

    def _sync_narrow(self, width: int | None = None) -> None:
        width = self.size.width if width is None else width
        self.set_class(width < 96, "-narrow")

    def _sync_pack_copy(self) -> None:
        pack = self.pack
        title = pack.title if pack is not None else "Visual Identity"
        self.query_one("#personas-visual-identity-title", Static).update(title)
        notice = ""
        if pack is not None and pack.source_kind == "builtin":
            notice = "Built-in pack — the first edit creates a private copy."
        self.query_one("#personas-visual-identity-notice", Static).update(notice)

    def apply_filter(self, query: str) -> None:
        """Filter path-free labels/keys and select the first visible asset."""

        needle = str(query or "").strip().casefold()
        assets = self.pack.assets if self.pack is not None else ()
        self._filtered = tuple(
            asset
            for asset in assets
            if not needle
            or needle in asset.display_label.casefold()
            or needle in asset.original_label.casefold()
            or needle in asset.expression_key.casefold()
        )
        options = self.query_one("#personas-visual-identity-results", OptionList)
        options.clear_options()
        options.add_options(
            Option(asset.display_label, id=f"asset-{asset.asset_id}")
            for asset in self._filtered
        )
        if self._filtered:
            options.highlighted = 0
            self._select_index(0)
        else:
            self._selected = None
            self._sync_selection_copy()
            self._replace_preview("Unavailable")

    def _select_index(self, index: int) -> None:
        if not (0 <= index < len(self._filtered)):
            return
        selected = self._filtered[index]
        changed = selected != self._selected
        self._selected = selected
        self._sync_selection_copy()
        if changed:
            self._replace_preview("Loading…")
            self.post_message(VisualIdentityPackPreviewRequested(selected))

    def _sync_selection_copy(self) -> None:
        selected = self._selected
        total = len(self._filtered)
        index = self._filtered.index(selected) + 1 if selected in self._filtered else 0
        self.query_one("#personas-visual-identity-count", Static).update(
            f"{index} / {total}"
        )
        self.query_one("#personas-visual-identity-label", Static).update(
            selected.display_label if selected is not None else "No matches"
        )
        self.query_one("#personas-visual-identity-key", Static).update(
            selected.expression_key if selected is not None else ""
        )
        disabled = selected is None
        for action in ("replace", "generate", "clear"):
            self.query_one(
                f"#personas-visual-identity-{action}", Button
            ).disabled = disabled

    def _replace_preview(self, renderable: object) -> None:
        """Replace the preview holder with one renderable or status."""

        holder = self.query_one("#personas-visual-identity-preview-image", Container)
        holder.remove_children()
        holder.mount(
            renderable if isinstance(renderable, Widget) else Static(renderable)
        )

    def set_preview(self, renderable: object, *, asset_id: int) -> None:
        """Mount one already-decoded preview when it still matches selection."""

        if self._selected is None or self._selected.asset_id != asset_id:
            return
        self._replace_preview(renderable)

    def set_preview_unavailable(self, *, asset_id: int) -> None:
        """Show failure only when the failed asset remains selected."""

        if self._selected is not None and self._selected.asset_id == asset_id:
            self._replace_preview("Unavailable")

    @property
    def selected_asset(self) -> VisualIdentityAssetMetadata | None:
        """Return the selected path-free asset metadata."""

        return self._selected

    def set_staged_change(
        self,
        expression_key: str,
        action: Literal["replace", "generate", "clear"],
    ) -> None:
        """Reflect a screen-owned staged operation without performing it."""

        if self.pack is None or expression_key not in {
            asset.expression_key for asset in self.pack.assets
        }:
            return
        self._staged[expression_key] = action
        count = len(self._staged)
        suffix = "change" if count == 1 else "changes"
        self.query_one("#personas-visual-identity-dirty", Static).update(
            f"{count} staged {suffix}"
        )
        self.query_one("#personas-visual-identity-save", Button).disabled = False

    def set_generating(self, generating: bool) -> None:
        """Expose one honest generation/cancellation state."""

        for action in ("replace", "generate", "generate-all", "clear", "save"):
            self.query_one(
                f"#personas-visual-identity-{action}", Button
            ).disabled = generating
        self.query_one("#personas-visual-identity-cancel", Button).display = generating
        self.query_one("#personas-visual-identity-dirty", Static).update(
            "Generating reactions…" if generating else self._dirty_copy()
        )

    def reset_staged(self) -> None:
        """Discard only unpublished widget state."""

        self._staged.clear()
        self.set_generating(False)
        self.query_one("#personas-visual-identity-save", Button).disabled = True

    def _dirty_copy(self) -> str:
        count = len(self._staged)
        if not count:
            return "No staged changes"
        return f"{count} staged {'change' if count == 1 else 'changes'}"

    @on(Input.Changed, "#personas-visual-identity-filter")
    def _filter_changed(self, event: Input.Changed) -> None:
        self.apply_filter(event.value)

    @on(OptionList.OptionHighlighted, "#personas-visual-identity-results")
    def _option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_index is not None:
            self._select_index(event.option_index)

    def _post_selected(self, message_type: type) -> None:
        if self._selected is not None:
            self.post_message(message_type(self._selected))

    @on(Button.Pressed, "#personas-visual-identity-replace")
    def _replace_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._post_selected(VisualIdentityPackReplaceRequested)

    @on(Button.Pressed, "#personas-visual-identity-generate")
    def _generate_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._post_selected(VisualIdentityPackGenerateRequested)

    @on(Button.Pressed, "#personas-visual-identity-generate-all")
    def _generate_all_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self.pack is not None:
            self.post_message(VisualIdentityPackGenerateAllRequested())

    @on(Button.Pressed, "#personas-visual-identity-clear")
    def _clear_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._post_selected(VisualIdentityPackClearRequested)

    @on(Button.Pressed, "#personas-visual-identity-save")
    def _save_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self.pack is not None and self._staged:
            self.post_message(
                VisualIdentityPackSaveRequested(
                    self.pack.pack_id, self.pack.pack_version_id
                )
            )

    @on(Button.Pressed, "#personas-visual-identity-cancel")
    def _cancel_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(VisualIdentityPackCancelRequested())


__all__ = [
    "PersonasVisualIdentityPackWidget",
    "VisualIdentityPackCancelRequested",
    "VisualIdentityPackGenerateAllRequested",
]
