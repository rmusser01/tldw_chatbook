"""Path-free review modals for portable Tool profile operations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Collapsible, Input, Static, TextArea

from tldw_chatbook.Tool_Packs.contracts import PortableToolRule
from tldw_chatbook.Tool_Packs.binding import ToolProfileBindingReview
from tldw_chatbook.Tool_Packs.export import ToolPackExportReview
from tldw_chatbook.Tool_Packs.importer import (
    ServerMapping,
    ToolPackImportReview,
)
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin
from tldw_chatbook.Workspaces.models import WorkspaceAssistantDefaults


def _plain_text(value: object, *, limit: int = 512) -> str:
    """Return bounded terminal-safe text without interpreting Rich markup."""
    raw = str(value)
    cleaned = "".join(
        character
        if character >= " " and character != "\x7f"
        else f"\\u{ord(character):04x}"
        for character in raw
    )
    return cleaned if len(cleaned) <= limit else f"{cleaned[: limit - 1]}…"


def _source_rules(review: ToolPackImportReview) -> tuple[PortableToolRule, ...]:
    candidates: Iterable[PortableToolRule] = (
        *(mapped.source_rule for mapped in review.matched),
        *review.changed,
        *review.missing,
        *review.pending_denies,
        *review.omitted_allow_ask,
    )
    unique: dict[tuple[object, ...], PortableToolRule] = {}
    for rule in candidates:
        key = (
            rule.authority,
            rule.server_key,
            rule.tool_name,
            rule.state,
            rule.contract_sha256,
        )
        unique[key] = rule
    return tuple(unique.values())


def _source_counts(review: ToolPackImportReview) -> tuple[int, int, int]:
    states = [rule.state for rule in _source_rules(review)]
    return states.count("allow"), states.count("ask"), states.count("deny")


@dataclass(frozen=True, slots=True)
class ToolPackImportOptions:
    """User-selected destination id and explicit server mappings."""

    destination_id: str
    mappings: tuple[ServerMapping, ...] = ()


class ToolPackImportOptionsModal(
    SafeModalDismissMixin,
    ModalScreen[ToolPackImportOptions | None],
):
    """Capture import identity and mapping choices before inspection."""

    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel", show=False)]
    SAFE_MODAL_CONTENT = "#tool-pack-import-options"
    MAX_MAPPINGS = 256

    BUNDLED_CSS = """
    ToolPackImportOptionsModal {
        align: center middle;
    }

    ToolPackImportOptionsModal #tool-pack-import-options {
        width: 82%;
        max-width: 86;
        height: auto;
        max-height: 34;
        background: $surface;
        border: tall $accent;
        padding: 1 2;
    }

    ToolPackImportOptionsModal .tool-pack-options-copy {
        width: 100%;
        height: auto;
        text-wrap: wrap;
        margin-bottom: 1;
    }

    ToolPackImportOptionsModal #tool-pack-import-server-mappings {
        height: 8;
        margin-bottom: 1;
    }

    ToolPackImportOptionsModal #tool-pack-import-options-error {
        color: $error;
        min-height: 1;
    }

    ToolPackImportOptionsModal #tool-pack-import-options-actions {
        height: auto;
        align: right middle;
    }

    ToolPackImportOptionsModal #tool-pack-import-options-actions Button {
        width: auto;
        min-width: 12;
        margin-left: 1;
    }
    """

    def __init__(self, options: ToolPackImportOptions) -> None:
        if type(options) is not ToolPackImportOptions:
            raise ValueError("tool_pack_import_options_invalid")
        super().__init__()
        self.options = options

    @staticmethod
    def _mapping_text(mappings: tuple[ServerMapping, ...]) -> str:
        return "\n".join(
            f"{item.source_server_key} -> {item.destination_server_key}"
            for item in mappings
        )

    def compose(self) -> ComposeResult:
        with Container(id="tool-pack-import-options"):
            yield Static(
                "Choose Tool Pack identity", classes="tool-pack-review-heading"
            )
            yield Static(
                "The profile is imported unbound. Add mappings only when a source "
                "MCP server should target a different local server.",
                classes="tool-pack-options-copy",
                markup=False,
            )
            yield Input(
                value=self.options.destination_id,
                placeholder="Profile id",
                id="tool-pack-import-profile-id",
                max_length=128,
            )
            yield Static(
                "Server mappings · one `source -> destination` pair per line",
                classes="tool-pack-options-copy",
                markup=False,
            )
            yield TextArea(
                self._mapping_text(self.options.mappings),
                id="tool-pack-import-server-mappings",
            )
            yield Static("", id="tool-pack-import-options-error", markup=False)
            with Horizontal(id="tool-pack-import-options-actions"):
                yield Button(
                    "Cancel",
                    id="tool-pack-import-options-cancel",
                    classes="console-action-secondary",
                )
                yield Button(
                    "Inspect pack",
                    id="tool-pack-import-options-review",
                    variant="primary",
                )

    def on_mount(self) -> None:
        super().on_mount()
        self.query_one("#tool-pack-import-profile-id", Input).focus()

    def _parse(self) -> ToolPackImportOptions | None:
        destination_id = self.query_one(
            "#tool-pack-import-profile-id", Input
        ).value.strip()
        if not destination_id:
            self.query_one("#tool-pack-import-options-error", Static).update(
                "Enter a profile id."
            )
            return None
        mappings: list[ServerMapping] = []
        for raw_line in self.query_one(
            "#tool-pack-import-server-mappings", TextArea
        ).text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            separator = "->" if "->" in line else "=" if "=" in line else None
            if separator is None:
                self.query_one("#tool-pack-import-options-error", Static).update(
                    "Use `source -> destination`, one mapping per line."
                )
                return None
            source, destination = (part.strip() for part in line.split(separator, 1))
            if not source or not destination:
                self.query_one("#tool-pack-import-options-error", Static).update(
                    "Every mapping needs both source and destination."
                )
                return None
            mappings.append(ServerMapping(source, destination))
            if len(mappings) > self.MAX_MAPPINGS:
                self.query_one("#tool-pack-import-options-error", Static).update(
                    "Too many server mappings."
                )
                return None
        return ToolPackImportOptions(destination_id, tuple(mappings))

    @on(Button.Pressed, "#tool-pack-import-options-cancel")
    async def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="button")

    @on(Button.Pressed, "#tool-pack-import-options-review")
    def _review(self, event: Button.Pressed) -> None:
        event.stop()
        options = self._parse()
        if options is not None:
            self.dismiss_safe_once(options)


class ToolPackImportReviewModal(
    SafeModalDismissMixin,
    ModalScreen[ToolPackImportReview | Literal["revise"] | None],
):
    """Present an immutable import review and return one unbound action."""

    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel", show=False)]
    SAFE_MODAL_CONTENT = "#tool-pack-import-review"

    BUNDLED_CSS = """
    ToolPackImportReviewModal {
        align: center middle;
    }

    ToolPackImportReviewModal #tool-pack-import-review {
        width: 88%;
        max-width: 96;
        height: 90%;
        max-height: 42;
        background: $surface;
        border: tall $accent;
        padding: 1 2;
    }

    ToolPackImportReviewModal #tool-pack-import-title {
        height: 1;
        text-style: bold;
        margin-bottom: 1;
    }

    ToolPackImportReviewModal #tool-pack-import-scroll {
        height: 1fr;
        scrollbar-size: 1 1;
    }

    ToolPackImportReviewModal .tool-pack-review-heading {
        height: 1;
        text-style: bold;
        margin-top: 1;
    }

    ToolPackImportReviewModal .tool-pack-review-copy {
        width: 100%;
        height: auto;
        text-wrap: wrap;
    }

    ToolPackImportReviewModal #tool-pack-import-actions {
        height: auto;
        min-height: 3;
        margin-top: 1;
        align: right middle;
    }

    ToolPackImportReviewModal #tool-pack-import-actions Button {
        width: auto;
        min-width: 12;
        margin-left: 1;
    }
    """

    def __init__(self, review: ToolPackImportReview) -> None:
        if type(review) is not ToolPackImportReview:
            raise ValueError("tool_pack_import_review_invalid")
        super().__init__()
        self.review = review

    def _identity_copy(self) -> str:
        producer = " ".join(_plain_text(part) for part in self.review.producer)
        return "\n".join(
            (
                f"Name: {_plain_text(self.review.display_name)}",
                f"Producer: {producer or 'Not provided'}",
                f"Content digest: {self.review.content_digest}",
                f"Proposed profile id: {_plain_text(self.review.destination_id)}",
            )
        )

    def _policy_copy(self) -> str:
        allow, ask, deny = _source_counts(self.review)
        lines = [
            f"Fallback · {item.authority}/{_plain_text(item.server_key)}: "
            f"{item.state.title()}"
            for item in self.review.fallbacks
        ]
        lines.append(f"Source rules · Allow {allow} · Ask {ask} · Deny {deny}")
        lines.extend(
            (
                f"Exact matches: {len(self.review.matched)}",
                f"Changed contracts: {len(self.review.changed)}",
                f"Missing tools: {len(self.review.missing)}",
                f"Pending Denies: {len(self.review.pending_denies)}",
                f"Omitted Ask/Allow: {len(self.review.omitted_allow_ask)}",
            )
        )
        return "\n".join(lines)

    def _destination_copy(self) -> str:
        if not self.review.matched:
            return "No exact destination matches."
        return "\n".join(
            f"{_plain_text(item.server_key)}/{_plain_text(item.tool_name)} · "
            f"{'connected' if item.destination_connected else 'disconnected (cached definition)'}"
            for item in self.review.matched
        )

    def _mapping_copy(self) -> str:
        if not self.review.mappings:
            return "No explicit server mappings."
        return "\n".join(
            f"{_plain_text(item.source_server_key)} → "
            f"{_plain_text(item.destination_server_key)}"
            for item in self.review.mappings
        )

    def compose(self) -> ComposeResult:
        with Container(id="tool-pack-import-review"):
            yield Static("Review Tool Pack", id="tool-pack-import-title")
            with VerticalScroll(id="tool-pack-import-scroll"):
                yield Static("Identity", classes="tool-pack-review-heading")
                yield Static(
                    self._identity_copy(),
                    id="tool-pack-import-identity",
                    classes="tool-pack-review-copy",
                    markup=False,
                )
                yield Static("Policy", classes="tool-pack-review-heading")
                yield Static(
                    self._policy_copy(),
                    id="tool-pack-import-policy",
                    classes="tool-pack-review-copy",
                    markup=False,
                )
                yield Static("Destination tools", classes="tool-pack-review-heading")
                yield Static(
                    self._destination_copy(),
                    id="tool-pack-import-destinations",
                    classes="tool-pack-review-copy",
                    markup=False,
                )
                yield Static("Server mappings", classes="tool-pack-review-heading")
                yield Static(
                    self._mapping_copy(),
                    id="tool-pack-import-mappings",
                    classes="tool-pack-review-copy",
                    markup=False,
                )
                yield Static("Import boundary", classes="tool-pack-review-heading")
                yield Static(
                    "Imported unbound. This creates a permission profile only; "
                    "it does not install tools and does not bind a workspace.",
                    id="tool-pack-import-boundary",
                    classes="tool-pack-review-copy",
                    markup=False,
                )
            with Horizontal(id="tool-pack-import-actions"):
                yield Button(
                    "Cancel",
                    id="tool-pack-import-cancel",
                    classes="console-action-secondary",
                )
                yield Button(
                    "Change id or mappings",
                    id="tool-pack-import-revise",
                    classes="console-action-secondary",
                )
                yield Button(
                    "Import unbound profile",
                    id="tool-pack-import-unbound",
                    variant="primary",
                )

    def on_mount(self) -> None:
        super().on_mount()
        self.query_one("#tool-pack-import-unbound", Button).focus()

    @on(Button.Pressed, "#tool-pack-import-cancel")
    async def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="button")

    @on(Button.Pressed, "#tool-pack-import-revise")
    def _revise(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss_safe_once("revise")

    @on(Button.Pressed, "#tool-pack-import-unbound")
    def _confirm(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss_safe_once(self.review)


class ToolPackExportReviewModal(
    SafeModalDismissMixin,
    ModalScreen[ToolPackExportReview | None],
):
    """Review one immutable policy snapshot before choosing a destination."""

    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel", show=False)]
    SAFE_MODAL_CONTENT = "#tool-pack-export-review"

    BUNDLED_CSS = """
    ToolPackExportReviewModal {
        align: center middle;
    }

    ToolPackExportReviewModal #tool-pack-export-review {
        width: 86%;
        max-width: 92;
        height: 86%;
        max-height: 38;
        background: $surface;
        border: tall $accent;
        padding: 1 2;
    }

    ToolPackExportReviewModal #tool-pack-export-scroll {
        height: 1fr;
        scrollbar-size: 1 1;
    }

    ToolPackExportReviewModal .tool-pack-export-heading {
        height: 1;
        text-style: bold;
        margin-top: 1;
    }

    ToolPackExportReviewModal .tool-pack-export-copy {
        width: 100%;
        height: auto;
        text-wrap: wrap;
    }

    ToolPackExportReviewModal #tool-pack-export-actions {
        height: auto;
        min-height: 3;
        align: right middle;
    }

    ToolPackExportReviewModal #tool-pack-export-actions Button {
        width: auto;
        min-width: 12;
        margin-left: 1;
    }
    """

    def __init__(
        self,
        review: ToolPackExportReview,
        *,
        profile_id: str,
        revision: int | None,
        policy_digest: str | None,
    ) -> None:
        if type(review) is not ToolPackExportReview:
            raise ValueError("tool_pack_export_review_invalid")
        if type(profile_id) is not str or not profile_id:
            raise ValueError("tool_pack_export_profile_invalid")
        super().__init__()
        self.review = review
        self.profile_id = profile_id
        self.revision = revision
        self.policy_digest = policy_digest

    def _identity_copy(self) -> str:
        manifest = self.review.snapshot.manifest
        revision = str(self.revision) if self.revision is not None else "local snapshot"
        digest = self.policy_digest or "Captured with the export snapshot"
        return "\n".join(
            (
                f"Profile: {_plain_text(self.profile_id)} · revision {revision}",
                f"Policy digest: {digest}",
                f"Name: {_plain_text(manifest.display_name)}",
                f"Suggested id: {_plain_text(manifest.suggested_id)}",
                "Producer: "
                f"{_plain_text(manifest.producer_name)} "
                f"{_plain_text(manifest.producer_version)}",
                f"Content digest: {manifest.content_digest}",
            )
        )

    def _policy_copy(self) -> str:
        payload = self.review.snapshot.payload
        states = [rule.state for rule in payload.rules]
        lines = [
            f"Fallback · {item.authority}/{_plain_text(item.server_key)}: "
            f"{item.state.title()}"
            for item in payload.fallbacks
        ]
        lines.extend(
            (
                f"Allow {states.count('allow')} · Ask {states.count('ask')} · "
                f"Deny {states.count('deny')}",
                f"Omitted Ask/Allow: {len(self.review.omitted_allow_ask)}",
                f"Pending Denies: {len(self.review.pending_denies)}",
                f"Inventory digest: {self.review.inventory_digest}",
            )
        )
        return "\n".join(lines)

    def _excluded_copy(self) -> str:
        if not self.review.excluded_counts:
            return "No excluded tool namespaces."
        return "\n".join(
            f"{_plain_text(reason)}: {count}"
            for reason, count in self.review.excluded_counts
        )

    def compose(self) -> ComposeResult:
        with Container(id="tool-pack-export-review"):
            yield Static("Review Tool Pack export", classes="tool-pack-export-heading")
            with VerticalScroll(id="tool-pack-export-scroll"):
                yield Static("Identity", classes="tool-pack-export-heading")
                yield Static(
                    self._identity_copy(),
                    classes="tool-pack-export-copy",
                    markup=False,
                )
                yield Static("Portable policy", classes="tool-pack-export-heading")
                yield Static(
                    self._policy_copy(),
                    classes="tool-pack-export-copy",
                    markup=False,
                )
                yield Static("Excluded", classes="tool-pack-export-heading")
                yield Static(
                    self._excluded_copy(),
                    classes="tool-pack-export-copy",
                    markup=False,
                )
                yield Static("Export boundary", classes="tool-pack-export-heading")
                yield Static(
                    "This archive contains permission policy only. It does not "
                    "include or install tools, skills, plugins, credentials, "
                    "workspace bindings, or runtime history.",
                    classes="tool-pack-export-copy",
                    markup=False,
                )
            with Horizontal(id="tool-pack-export-actions"):
                yield Button(
                    "Cancel",
                    id="tool-pack-export-cancel",
                    classes="console-action-secondary",
                )
                yield Button(
                    "Choose destination",
                    id="tool-pack-export-continue",
                    variant="primary",
                )

    def on_mount(self) -> None:
        super().on_mount()
        self.query_one("#tool-pack-export-continue", Button).focus()

    @on(Button.Pressed, "#tool-pack-export-cancel")
    async def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="button")

    @on(Button.Pressed, "#tool-pack-export-continue")
    def _confirm(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss_safe_once(self.review)


def _authority_rows(rows: tuple[tuple[str, str], ...]) -> str:
    """Format bounded authority/name pairs without Rich interpretation."""
    if not rows:
        return "None"
    return ", ".join(
        f"{_plain_text(authority)}/{_plain_text(tool_name)}"
        for authority, tool_name in rows
    )


class ToolProfileFirstBindReviewModal(
    SafeModalDismissMixin,
    ModalScreen[ToolProfileBindingReview | None],
):
    """Review the exact authority accepted by one first-bind token."""

    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel", show=False)]
    SAFE_MODAL_CONTENT = "#tool-profile-bind-review"

    BUNDLED_CSS = """
    ToolProfileFirstBindReviewModal {
        align: center middle;
    }

    ToolProfileFirstBindReviewModal #tool-profile-bind-review {
        width: 88%;
        max-width: 96;
        height: 90%;
        max-height: 42;
        background: $surface;
        border: tall $warning;
        padding: 1 2;
    }

    ToolProfileFirstBindReviewModal #tool-profile-bind-title {
        height: 1;
        text-style: bold;
        margin-bottom: 1;
    }

    ToolProfileFirstBindReviewModal #tool-profile-bind-scroll {
        height: 1fr;
        scrollbar-size: 1 1;
    }

    ToolProfileFirstBindReviewModal .tool-profile-bind-heading {
        height: 1;
        text-style: bold;
        margin-top: 1;
    }

    ToolProfileFirstBindReviewModal .tool-profile-bind-copy {
        width: 100%;
        height: auto;
        text-wrap: wrap;
    }

    ToolProfileFirstBindReviewModal #tool-profile-bind-actions {
        height: auto;
        min-height: 3;
        margin-top: 1;
        align: right middle;
    }

    ToolProfileFirstBindReviewModal #tool-profile-bind-actions Button {
        width: auto;
        min-width: 12;
        margin-left: 1;
    }
    """

    def __init__(
        self,
        review: ToolProfileBindingReview,
        intended_defaults: WorkspaceAssistantDefaults,
    ) -> None:
        if type(review) is not ToolProfileBindingReview:
            raise ValueError("tool_profile_binding_review_invalid")
        if type(intended_defaults) is not WorkspaceAssistantDefaults:
            raise ValueError("tool_profile_binding_defaults_invalid")
        if intended_defaults.tool_policy_profile_id != review.profile_id:
            raise ValueError("tool_profile_binding_profile_mismatch")
        super().__init__()
        self.review = review
        self.intended_defaults = intended_defaults

    def _target_copy(self) -> str:
        return "\n".join(
            (
                f"Workspace: {_plain_text(self.review.workspace_id)}",
                f"Action: {self.review.action.title()}",
                f"Profile: {_plain_text(self.review.profile_id)} · "
                f"revision {self.review.revision}",
                f"Policy digest: {self.review.policy_digest}",
                f"Assistant kind: {self.intended_defaults.assistant_kind}",
                f"Persona: {_plain_text(self.intended_defaults.assistant_id)}",
                f"Memory: {self.intended_defaults.persona_memory_mode}",
                f"Voice: {_plain_text(self.intended_defaults.voice or 'Default')}",
                f"Style: {_plain_text(self.intended_defaults.style or 'Default')}",
            )
        )

    def _posture_copy(self) -> str:
        summary = self.review.summary
        return "\n".join(
            (
                f"Global fallback: {summary.global_fallback.title()}",
                f"Built-in fallback: {summary.builtin_fallback.title()}",
                "Allow-server fallbacks: "
                + (
                    ", ".join(map(_plain_text, summary.allow_server_fallbacks))
                    or "None"
                ),
                f"Allow {summary.allow_count} · Ask {summary.ask_count} · "
                f"Deny {summary.deny_count}",
                f"Stored exact allows: {_authority_rows(summary.stored_exact_allows)}",
                f"Effective allows: {_authority_rows(summary.effective_allows)}",
                f"Unavailable allows: {_authority_rows(summary.unavailable_allows)}",
                f"Downgraded allows: {_authority_rows(summary.downgraded_allows)}",
                f"High-risk allows: {_authority_rows(summary.high_risk_allows)}",
                f"Inventory digest: {summary.inventory_digest}",
            )
        )

    def compose(self) -> ComposeResult:
        with Container(id="tool-profile-bind-review"):
            yield Static("Review first Tool Profile bind", id="tool-profile-bind-title")
            with VerticalScroll(id="tool-profile-bind-scroll"):
                yield Static("Target", classes="tool-profile-bind-heading")
                yield Static(
                    self._target_copy(),
                    classes="tool-profile-bind-copy",
                    markup=False,
                )
                yield Static("Effective policy", classes="tool-profile-bind-heading")
                yield Static(
                    self._posture_copy(),
                    classes="tool-profile-bind-copy",
                    markup=False,
                )
                with Collapsible(
                    title=f"Ask detail ({self.review.summary.ask_count})",
                    collapsed=True,
                ):
                    yield Static(
                        _authority_rows(self.review.summary.effective_asks),
                        classes="tool-profile-bind-copy",
                        markup=False,
                    )
                with Collapsible(
                    title=f"Deny detail ({self.review.summary.deny_count})",
                    collapsed=True,
                ):
                    yield Static(
                        _authority_rows(self.review.summary.effective_denies),
                        classes="tool-profile-bind-copy",
                        markup=False,
                    )
                yield Static(
                    "Confirmation boundary", classes="tool-profile-bind-heading"
                )
                yield Static(
                    "Confirming authorizes this exact workspace, assistant defaults, "
                    "profile revision, policy digest, and tool inventory once. Any change "
                    "requires a new review. This is separate from the read_write "
                    "memory acknowledgement.",
                    classes="tool-profile-bind-copy",
                    markup=False,
                )
            with Horizontal(id="tool-profile-bind-actions"):
                yield Button(
                    "Cancel",
                    id="tool-profile-bind-cancel",
                    classes="console-action-secondary",
                )
                yield Button(
                    "Confirm exact bind",
                    id="tool-profile-bind-confirm",
                    variant="warning",
                )

    def on_mount(self) -> None:
        super().on_mount()
        self.query_one("#tool-profile-bind-confirm", Button).focus()

    @on(Button.Pressed, "#tool-profile-bind-cancel")
    async def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="button")

    @on(Button.Pressed, "#tool-profile-bind-confirm")
    def _confirm(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss_safe_once(self.review)


__all__ = [
    "ToolPackExportReviewModal",
    "ToolPackImportOptions",
    "ToolPackImportOptionsModal",
    "ToolPackImportReviewModal",
    "ToolProfileFirstBindReviewModal",
]
