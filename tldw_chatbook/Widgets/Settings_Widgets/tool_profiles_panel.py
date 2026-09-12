"""Presentation-only Settings panel for portable Tool policy profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
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


class ToolProfilesPanel(Vertical):
    """Render immutable profile facts and emit explicit management requests."""

    DEFAULT_CSS = """
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

    ToolProfilesPanel Button {
        width: auto;
        min-width: 8;
        height: 1;
        min-height: 1;
        padding: 0 1;
        border: none;
        margin-right: 1;
    }

    ToolProfilesPanel .tool-profile-actions Button {
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
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._listing = listing
        self._result = result
        self._profiles = listing.profiles
        self._rows = {
            profile.profile_id: self._present(profile) for profile in self._profiles
        }
        self._button_actions: dict[str, tuple[str, ToolProfilePresentation]] = {}

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
        """Replace presentation state with one complete service snapshot."""
        self._listing = listing
        self._profiles = listing.profiles
        self._rows = {
            profile.profile_id: self._present(profile) for profile in self._profiles
        }
        self._button_actions.clear()
        await self.recompose()

    def set_result(self, result: str) -> None:
        """Show one bounded path-free workflow outcome."""
        self._result = result[:512]
        try:
            self.query_one("#tool-profiles-result", Static).update(self._result)
        except Exception:
            return

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
        self._button_actions[button_id] = (action, profile)
        return Button(
            label,
            id=button_id,
            classes="console-action-subdued",
            compact=True,
            disabled=disabled,
            tooltip=tooltip,
        )

    def compose(self) -> ComposeResult:
        yield Static(
            "Tool Profiles", classes="destination-section settings-column-title"
        )
        yield Static(
            "Portable permission profiles change tool policy only. Importing a "
            "profile never installs tools or binds it to a workspace.",
            classes="settings-detail-row",
        )
        yield Static(
            self._result,
            id="tool-profiles-result",
            classes="settings-detail-row",
            markup=False,
        )
        with Horizontal(classes="tool-profiles-toolbar"):
            yield Button(
                "Import Tool Pack",
                id="tool-profiles-import",
                classes="console-action-subdued",
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
        button_id = event.button.id or ""
        if button_id == "tool-profiles-import":
            event.stop()
            self.post_message(self.ImportRequested())
            return
        action_context = self._button_actions.get(button_id)
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
