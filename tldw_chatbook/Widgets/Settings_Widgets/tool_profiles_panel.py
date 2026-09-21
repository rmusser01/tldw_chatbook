"""Presentation-only Settings panel for portable Tool policy profiles."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Static

if TYPE_CHECKING:
    from tldw_chatbook.Tool_Packs.service import (
        ToolProfileListing,
        ToolProfilePresentation,
    )


_ORIGIN_LABELS = {
    "local": "Local",
    "workspace-managed": "Workspace-managed",
    "imported": "Imported Tool Pack",
}
_RECEIPT_LABELS = {
    "not_applicable": "Local provenance",
    "available": "Import receipt available",
    "unavailable": "Receipt unavailable",
}


def _plain_text(value: object, *, limit: int = 256) -> str:
    """Return bounded terminal-safe text without interpreting Rich markup."""
    raw = str(value)
    cleaned = "".join(
        character
        if character >= " " and character != "\x7f"
        else f"\\u{ord(character):04x}"
        for character in raw
    )
    return cleaned if len(cleaned) <= limit else f"{cleaned[: limit - 1]}…"


@dataclass(frozen=True, slots=True)
class ToolProfileRow:
    """Plain presentation data exposed for focused UI assertions."""

    profile_id: str
    origin: str
    lifecycle_valid: bool
    binding_state: str
    first_bind_confirmation_required: bool
    reference_counts: tuple[int, int]
    posture_counts: tuple[int, int, int]
    receipt_health: str
    removal_eligible: bool
    removal_blocker: str | None
    revision: int | None
    policy_digest: str | None


class _ToolProfileResult(Static):
    """Recheck outcome visibility after its text has wrapped."""

    profile_id: str | None = None

    def on_resize(self) -> None:
        for ancestor in self.ancestors:
            if isinstance(ancestor, ToolProfilesPanel):
                ancestor._reveal_result()
                break


class ToolProfilesPanel(Vertical):
    """Render immutable profile facts and emit explicit management requests."""

    BUNDLED_CSS = """
    ToolProfilesPanel {
        width: 100%;
        height: auto;
    }

    ToolProfilesPanel .tool-profiles-toolbar,
    ToolProfilesPanel .tool-profile-actions {
        width: 100%;
        height: auto;
        layout: horizontal;
    }

    ToolProfilesPanel .tool-profile-row {
        width: 100%;
        height: auto;
        margin-bottom: 1;
        padding: 0 1;
        border: round $panel;
    }

    ToolProfilesPanel .tool-profile-title {
        width: 100%;
        text-style: bold;
    }

    ToolProfilesPanel .tool-profile-detail {
        width: 100%;
        color: $text-muted;
    }

    ToolProfilesPanel Button.tool-profiles-panel-button {
        width: auto;
        min-width: 8;
        height: 1;
        min-height: 1;
        padding: 0 1;
        border: none;
        margin-right: 1;
    }

    ToolProfilesPanel .tool-profile-actions Button.tool-profiles-panel-button {
        margin-right: 0;
    }
    """

    has_policy_editor = False

    class ImportRequested(Message):
        """Request selection and inspection of one Tool Pack archive."""

    class _ProfileRequested(Message):
        """Base message carrying exact profile authority captured by the row."""

        def __init__(
            self,
            profile_id: str,
            revision: int | None,
            policy_digest: str | None,
        ) -> None:
            super().__init__()
            self.profile_id = profile_id
            self.revision = revision
            self.policy_digest = policy_digest

    class ExportRequested(_ProfileRequested):
        """Request export review for one exact profile revision."""

    class EditPolicyRequested(_ProfileRequested):
        """Request MCP Permissions for one exact profile revision."""

    class BindRequested(_ProfileRequested):
        """Request the workspace binding flow for one exact profile revision."""

    class RemoveRequested(_ProfileRequested):
        """Request removal review for one exact profile revision."""

    def __init__(
        self,
        listing: ToolProfileListing,
        *,
        result: str = "",
        result_profile_id: str | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._listing = listing
        self._result = result
        self._result_profile_id = result_profile_id
        self._result_anchor: Widget | None = None
        self._profiles = listing.profiles
        self._rows = {
            profile.profile_id: self._present(profile) for profile in self._profiles
        }
        self._button_actions: dict[Button, tuple[str, ToolProfilePresentation]] = {}
        self._listing_lock = asyncio.Lock()

    @staticmethod
    def _present(profile: ToolProfilePresentation) -> ToolProfileRow:
        return ToolProfileRow(
            profile_id=profile.profile_id,
            origin=_ORIGIN_LABELS[profile.origin],
            lifecycle_valid=profile.lifecycle_valid,
            binding_state=profile.binding_state,
            first_bind_confirmation_required=(profile.first_bind_confirmation_required),
            reference_counts=profile.reference_counts,
            posture_counts=profile.posture_counts,
            receipt_health=_RECEIPT_LABELS[profile.receipt_health],
            removal_eligible=profile.removal_eligible,
            removal_blocker=profile.removal_blocker,
            revision=profile.revision,
            policy_digest=profile.policy_digest,
        )

    @property
    def profile_ids(self) -> tuple[str, ...]:
        """Return visible profile ids in service-defined order."""
        return tuple(self._rows)

    def row(self, profile_id: str) -> ToolProfileRow:
        """Return immutable presentation facts for one visible profile."""
        return self._rows[profile_id]

    async def apply_listing(self, listing: ToolProfileListing) -> None:
        """Refresh changed facts while retaining the user's current action."""
        async with self._listing_lock:
            if listing == self._listing or not self.is_attached:
                return
            screen = self.screen
            focused = screen.focused
            reveal_result = focused is not None and focused is self._result_anchor
            context = self._button_actions.get(focused)
            focus_key = (context[0], context[1].profile_id) if context else None
            # TASK-32800.4's guard: this resumes after the lock await, so the
            # panel's children are not guaranteed to be the ones it entered
            # with. A missing import button just means no import focus key.
            imports = self.query("#tool-profiles-import")
            if imports and focused is imports.first(Button):
                focus_key = ("import", None)
            if focus_key is not None:
                # Prevent teardown's automatic fallback from being mistaken for
                # deliberate user focus. Any newer focus during the await wins.
                screen.set_focus(None)
            self._listing = listing
            self._profiles = listing.profiles
            self._rows = {
                profile.profile_id: self._present(profile) for profile in self._profiles
            }
            self._button_actions.clear()
            await self.recompose()
            if focus_key is None or not self.is_attached or screen.focused is not None:
                return
            action, profile_id = focus_key
            enabled = [
                (button, kind, profile.profile_id)
                for button, (kind, profile) in self._button_actions.items()
                if not button.disabled
            ]
            target = next(
                (button for button, kind, key in enabled if (kind, key) == focus_key),
                None,
            )
            if target is None and action != "import":
                target = next(
                    (button for button, _, key in enabled if key == profile_id), None
                )
            if target is None:
                # TASK-32800.4's guard: this resumes after the recompose
                # above, so the import button is a hope, not a guarantee.
                # Falling through to the focusable ancestor is what this
                # branch already does for a disabled one.
                imports = self.query("#tool-profiles-import")
                target = imports.first(Button) if imports else None
                if target is None or target.disabled:
                    target = next(
                        (ancestor for ancestor in self.ancestors if ancestor.focusable),
                        None,
                    )
            screen.set_focus(target)
            self._result_anchor = target if reveal_result else None

            def reveal() -> None:
                if (
                    target is not None
                    and target.is_attached
                    and self.app.screen is screen
                    and screen.focused is target
                ):
                    target.scroll_visible(animate=False, immediate=True)
                    self._reveal_result()

            self.call_after_refresh(reveal)

    def set_result(self, result: str, *, profile_id: str | None = None) -> None:
        """Keep removal feedback beside its profile, or the Import continuation."""
        self._result = result[:512]
        self._result_profile_id = profile_id
        focused = self.screen.focused
        context = self._button_actions.get(focused)
        self._result_anchor = (
            focused
            if context and (context[0], context[1].profile_id) == ("remove", profile_id)
            else None
        )
        self._sync_results()
        self.call_after_refresh(self._reveal_result)

    def _result_widget(self, profile_id: str | None, widget_id: str) -> Static:
        text = self._result_for(profile_id)
        widget = _ToolProfileResult(
            text,
            id=widget_id,
            classes="settings-detail-row"
            if profile_id is None
            else "settings-status-row",
            markup=False,
        )
        widget.profile_id = profile_id
        widget.display = bool(text) or (
            profile_id is None and self._result_profile_id is None
        )
        return widget

    def _result_for(self, profile_id: str | None) -> str:
        destination = (
            self._result_profile_id if self._result_profile_id in self._rows else None
        )
        return self._result if profile_id == destination else ""

    def _sync_results(self) -> None:
        for widget in self.query(_ToolProfileResult):
            text = self._result_for(widget.profile_id)
            widget.update(text)
            widget.display = bool(text) or (
                widget.profile_id is None and self._result_profile_id is None
            )

    def _reveal_result(self) -> None:
        """Reveal only while the originating action still owns visible focus."""
        anchor = self._result_anchor
        if (
            not self.is_attached
            or not self._result
            or anchor is None
            or not anchor.is_attached
            or self.app.screen is not self.screen
            or self.screen.focused is not anchor
        ):
            return
        for widget in self.query(_ToolProfileResult):
            if widget.display and widget.size:
                widget.scroll_visible(animate=False, immediate=True)
                anchor.scroll_visible(animate=False, immediate=True)
                break

    @staticmethod
    def _reference_label(counts: tuple[int, int]) -> str:
        active, archived = counts
        return f"{active} active · {archived} archived"

    @staticmethod
    def _posture_label(counts: tuple[int, int, int]) -> str:
        allow, ask, deny = counts
        return f"Allow {allow} · Ask {ask} · Deny {deny}"

    def _action_button(
        self,
        label: str,
        action: str,
        index: int,
        profile: ToolProfilePresentation,
        *,
        disabled: bool,
        tooltip: str,
    ) -> Button:
        button_id = f"tool-profile-{action}-{index}"
        button = Button(
            label,
            id=button_id,
            classes="console-action-subdued tool-profile-action tool-profiles-panel-button",
            compact=True,
            disabled=disabled,
            tooltip=tooltip,
        )
        # A queued event belongs to this control, even if a refresh reuses its ID.
        self._button_actions[button] = (action, profile)
        return button

    def compose(self) -> ComposeResult:
        yield Static(
            "Tool Profiles", classes="destination-section settings-column-title"
        )
        yield Static(
            "Portable permission profiles change tool policy only. Importing a "
            "profile never installs tools or binds it to a workspace.",
            classes="settings-detail-row",
        )
        yield self._result_widget(None, "tool-profiles-result")
        with Horizontal(classes="tool-profiles-toolbar"):
            yield Button(
                "Import Tool Pack",
                id="tool-profiles-import",
                classes="console-action-subdued tool-profiles-panel-button",
                compact=True,
                disabled=self._listing.unavailable_category is not None,
                tooltip="Inspect a Tool Pack before importing an unbound profile.",
            )

        if self._listing.unavailable_category is not None:
            yield Static(
                f"Profiles unavailable · {self._listing.unavailable_category}",
                id="tool-profiles-unavailable",
                classes="settings-detail-row",
                markup=False,
            )
            return
        if not self._profiles:
            yield Static(
                "No visible Tool profiles. Import a Tool Pack to create an unbound profile.",
                id="tool-profiles-empty",
                classes="settings-detail-row",
            )
            return

        for index, profile in enumerate(self._profiles):
            row = self._rows[profile.profile_id]
            lifecycle_label = (
                "Policy lifecycle valid"
                if row.lifecycle_valid
                else "Invalid policy lifecycle"
            )
            with Vertical(
                id=f"tool-profile-row-{index}",
                classes="tool-profile-row settings-focus-card",
            ):
                yield Static(
                    _plain_text(row.profile_id),
                    classes="tool-profile-title",
                    markup=False,
                )
                yield Static(
                    f"{row.origin} · {row.binding_state.title()} · {lifecycle_label}",
                    classes="tool-profile-detail",
                )
                if row.origin == "Imported Tool Pack":
                    yield Static(
                        (
                            "First bind review required"
                            if row.first_bind_confirmation_required
                            else "First bind already reviewed"
                        ),
                        classes="tool-profile-detail",
                    )
                if row.policy_digest is None:
                    identity = "Policy identity unavailable"
                else:
                    revision = (
                        f"Revision {row.revision}"
                        if row.revision is not None
                        else "Unversioned local profile"
                    )
                    identity = f"{revision} · Policy digest {row.policy_digest}"
                yield Static(
                    identity,
                    classes="tool-profile-detail",
                    markup=False,
                )
                yield Static(
                    f"{row.receipt_health} · References: "
                    f"{self._reference_label(row.reference_counts)}",
                    classes="tool-profile-detail",
                )
                yield Static(
                    self._posture_label(row.posture_counts),
                    classes="tool-profile-detail",
                )
                yield self._result_widget(
                    profile.profile_id, f"tool-profile-result-{index}"
                )
                with Horizontal(classes="tool-profile-actions"):
                    invalid = not row.lifecycle_valid
                    yield self._action_button(
                        "Export",
                        "export",
                        index,
                        profile,
                        disabled=invalid,
                        tooltip="Review and export this profile without tool binaries.",
                    )
                    yield self._action_button(
                        "Edit",
                        "edit",
                        index,
                        profile,
                        disabled=invalid,
                        tooltip="Open this profile in MCP Permissions.",
                    )
                    yield self._action_button(
                        "Bind",
                        "bind",
                        index,
                        profile,
                        disabled=invalid,
                        tooltip="Choose a workspace and review this profile before binding.",
                    )
                    yield self._action_button(
                        "Remove",
                        "remove",
                        index,
                        profile,
                        disabled=invalid or not row.removal_eligible,
                        tooltip=(
                            "Remove this unreferenced profile."
                            if row.removal_eligible
                            else f"Cannot remove · {row.removal_blocker or 'referenced'}"
                        ),
                    )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if not self.is_attached or not event.button.is_attached:
            event.stop()
            return
        button_id = event.button.id or ""
        if button_id == "tool-profiles-import":
            event.stop()
            self.post_message(self.ImportRequested())
            return
        action_context = self._button_actions.get(event.button)
        if action_context is None:
            return
        event.stop()
        action, profile = action_context
        message_type = {
            "export": self.ExportRequested,
            "edit": self.EditPolicyRequested,
            "bind": self.BindRequested,
            "remove": self.RemoveRequested,
        }[action]
        self.post_message(
            message_type(profile.profile_id, profile.revision, profile.policy_digest)
        )
