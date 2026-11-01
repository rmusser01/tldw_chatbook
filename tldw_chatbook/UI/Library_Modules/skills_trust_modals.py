"""Skill-trust modals shared by the Library Skills surfaces.

Extracted from the retired standalone ``UI/Screens/skills_screen.py``
(task-34893): skill management lives entirely inside Library, so the three
trust modals now live next to their only consumers.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

from ...Widgets.modal_dismissal import SafeModalDismissMixin

if TYPE_CHECKING:
    from ...Skills_Interop.recovery_activation import RecoveryReview

#: TASK-35028: the stale-selection copy `SkillRecoveryReviewModal` writes when
#: its 0.2 s poll finds the reviewed selection no longer current. A module
#: constant (not an inline literal) so the poll's equality gate compares the
#: exact string it wrote.
STALE_SELECTION_COPY = "Selection changed. Cancel and capture a fresh review."


class SkillTrustPassphraseModal(SafeModalDismissMixin, ModalScreen[str | None]):
    """Prompt for the local skill trust passphrase without logging it."""

    BINDINGS = [("escape", "request_safe_cancel", "Cancel")]
    SAFE_MODAL_CONTENT = "#skill-trust-passphrase-modal"

    def __init__(
        self,
        *,
        confirm_bootstrap: bool,
        title: str | None = None,
        message: str | None = None,
    ) -> None:
        """Prompt for the local skill trust passphrase.

        Args:
            confirm_bootstrap: Whether this prompt confirms a first-run
                trust bootstrap (drives the default title/message copy).
            title: Optional title override so a caller can state its actual
                purpose (task-418: e.g. the Library approve flow) instead of
                every purpose presenting as "Unlock Local Skill Trust".
            message: Optional message-body override, paired with ``title``.
        """
        super().__init__()
        self._confirm_bootstrap = confirm_bootstrap
        self._title_override = title
        self._message_override = message

    def compose(self) -> ComposeResult:
        title = self._title_override or (
            "Bootstrap Local Skill Trust"
            if self._confirm_bootstrap
            else "Unlock Local Skill Trust"
        )
        message = self._message_override or (
            "Current local skill files will become the trusted baseline. "
            "Enter the local skill trust passphrase to continue."
            if self._confirm_bootstrap
            else "Enter the local skill trust passphrase to unlock trust checks for this session."
        )
        with Vertical(id="skill-trust-passphrase-modal"):
            yield Static(
                title,
                id="skill-trust-passphrase-title",
                classes="destination-section",
            )
            yield Static(message, id="skill-trust-passphrase-message")
            yield Input(
                password=True,
                id="skill-trust-passphrase-input",
                placeholder="Trust passphrase",
            )
            yield Static("", id="skill-trust-passphrase-error", markup=False)
            with Horizontal(id="skill-trust-passphrase-actions"):
                yield Button("Cancel", id="skill-trust-passphrase-cancel")
                yield Button(
                    "Submit", id="skill-trust-passphrase-submit", variant="primary"
                )

    def on_mount(self) -> None:
        self.query_one("#skill-trust-passphrase-input", Input).focus()

    @on(Button.Pressed, "#skill-trust-passphrase-cancel")
    async def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="visible")

    @on(Button.Pressed, "#skill-trust-passphrase-submit")
    def _submit_button(self, event: Button.Pressed) -> None:
        event.stop()
        self._submit()

    @on(Input.Submitted, "#skill-trust-passphrase-input")
    def _submit_input(self, event: Input.Submitted) -> None:
        event.stop()
        self._submit()

    def _submit(self) -> None:
        passphrase = self.query_one("#skill-trust-passphrase-input", Input).value
        if not passphrase:
            self.query_one("#skill-trust-passphrase-error", Static).update(
                "Passphrase cannot be blank."
            )
            return
        self.dismiss(passphrase)


class SkillRecoveryReviewModal(SafeModalDismissMixin, ModalScreen[bool]):
    """Confirm the actual immutable current bundles before fresh trust setup."""

    BUNDLED_CSS = """
    SkillRecoveryReviewModal { align: center middle; }
    #skills-recovery-review { width: 90%; max-width: 110; height: 90%; border: tall $accent; padding: 1 2; background: $surface; }
    #skills-recovery-scroll { height: 1fr; }
    #skills-recovery-scroll Static.skill-recovery-review-modal-static { height: auto; margin-bottom: 1; }
    #skills-recovery-actions { height: 3; align-horizontal: right; }
    #skills-recovery-actions Button.skill-recovery-review-modal-button { width: auto; margin-left: 1; }
    #skills-recovery-status { height: auto; color: $warning; }
    """
    BINDINGS: ClassVar = [("escape", "request_safe_cancel", "Cancel")]
    SAFE_MODAL_CONTENT = "#skills-recovery-review"

    def __init__(
        self, review: RecoveryReview, root: Path, is_current: Callable[[], bool]
    ) -> None:
        super().__init__()
        self._review = review
        self._root = root
        self._is_current = is_current

    def compose(self) -> ComposeResult:
        review = self._review
        files = []
        for snapshot in review.skills:
            files.append(snapshot.skill_name)
            for item in snapshot.fingerprints:
                files.append(
                    f"{item.relative_path} — {item.byte_length} bytes — SHA256 {item.sha256}"
                )
            for name, content in snapshot.text_files.items():
                files.append(f"\n{snapshot.skill_name}/{name}\n{content}")
        with Vertical(id="skills-recovery-review"):
            yield Static(
                "Review restored skills",
                markup=False,
                classes="skill-recovery-review-modal-static",
            )
            with VerticalScroll(id="skills-recovery-scroll"):
                yield Static(
                    "Review these current local files. Setup preserves the imported trust files and creates fresh local trust. If finishing an interrupted setup, use the same passphrase you chose then. Script execution still requires a separate explicit grant.",
                    markup=False,
                    classes="skill-recovery-review-modal-static",
                )
                yield Static(
                    "Local trust folder: "
                    + str(self._root)
                    + (
                        "\nCreate these missing parent folders:\n"
                        + "\n".join(review.missing_trust_parents)
                        if review.missing_trust_parents
                        else ""
                    ),
                    id="skills-recovery-root",
                    markup=False,
                    classes="skill-recovery-review-modal-static",
                )
                yield Static(
                    "Retained script grants (inactive):\n"
                    + ("\n".join(review.historical_script_grants) or "None"),
                    id="skills-recovery-history",
                    markup=False,
                    classes="skill-recovery-review-modal-static",
                )
                yield Static(
                    "This setup uses a local file marker. Protection against restoring an older copy of the entire trust folder is reduced.",
                    markup=False,
                    classes="skill-recovery-review-modal-static",
                )
                yield Static(
                    "\n\n".join(files) or "No current Skills bundles.",
                    id="skills-recovery-files",
                    markup=False,
                    classes="skill-recovery-review-modal-static",
                )
            yield Static(
                "",
                id="skills-recovery-status",
                markup=False,
                classes="skill-recovery-review-modal-static",
            )
            with Horizontal(id="skills-recovery-actions"):
                yield Button(
                    "Cancel",
                    id="skills-recovery-cancel",
                    classes="skill-recovery-review-modal-button",
                )
                yield Button(
                    "Confirm review",
                    id="skills-recovery-continue",
                    variant="primary",
                    classes="skill-recovery-review-modal-button",
                )

    def on_mount(self) -> None:
        self.set_interval(0.2, self._check_selection)
        self._check_selection()

    def _check_selection(self) -> bool:
        current = self._is_current()
        if not current:
            self.query_one("#skills-recovery-continue", Button).disabled = True
            status = self.query_one("#skills-recovery-status", Static)
            # TASK-35028: the stale-selection copy is a constant; a tick that
            # finds it already shown must not repaint (and re-layout) the
            # height:auto row 5x/s.
            if status.content != STALE_SELECTION_COPY:
                status.update(STALE_SELECTION_COPY)
        return current

    @on(Button.Pressed, "#skills-recovery-continue")
    def _confirm(self, event: Button.Pressed) -> None:
        event.stop()
        if self._check_selection():
            self.dismiss(True)

    @on(Button.Pressed, "#skills-recovery-cancel")
    async def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="visible")


class SkillTrustBootstrapModal(SafeModalDismissMixin, ModalScreen[str | None]):
    """Prompt for a brand-new local skill trust passphrase, twice, before bootstrap.

    Structural twin of ``SkillTrustPassphraseModal`` above (same CSS id/
    layout shape, same Cancel/Submit actions and ``escape`` binding) --
    this is the ONLY trust modal that creates a passphrase rather than
    unlocking an existing one, so it asks for it twice and refuses to
    dismiss on a mismatch instead of silently proceeding with a possibly-
    mistyped passphrase nobody could recover afterward. Used by the Library
    Skills editor's first-run "Set up skill trust" state (the Phase-1 gate
    fix: a brand-new install previously had no live-UI path to bootstrap
    trust at all).
    """

    BINDINGS = [("escape", "request_safe_cancel", "Cancel")]
    SAFE_MODAL_CONTENT = "#skill-trust-bootstrap-modal"

    def compose(self) -> ComposeResult:
        with Vertical(id="skill-trust-bootstrap-modal"):
            yield Static("Set Up Local Skill Trust", classes="destination-section")
            yield Static(
                "Choose a passphrase to protect local skill trust. Current local "
                "skill files become the trusted baseline. Enter it twice to confirm.",
                id="skill-trust-bootstrap-message",
            )
            yield Input(
                password=True,
                id="skill-trust-bootstrap-input",
                placeholder="New trust passphrase",
            )
            yield Input(
                password=True,
                id="skill-trust-bootstrap-confirm-input",
                placeholder="Confirm trust passphrase",
            )
            yield Static("", id="skill-trust-bootstrap-error", markup=False)
            with Horizontal(id="skill-trust-bootstrap-actions"):
                yield Button("Cancel", id="skill-trust-bootstrap-cancel")
                yield Button(
                    "Submit", id="skill-trust-bootstrap-submit", variant="primary"
                )

    def on_mount(self) -> None:
        self.query_one("#skill-trust-bootstrap-input", Input).focus()

    @on(Button.Pressed, "#skill-trust-bootstrap-cancel")
    async def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="visible")

    @on(Button.Pressed, "#skill-trust-bootstrap-submit")
    def _submit_button(self, event: Button.Pressed) -> None:
        event.stop()
        self._submit()

    @on(Input.Submitted, "#skill-trust-bootstrap-input")
    @on(Input.Submitted, "#skill-trust-bootstrap-confirm-input")
    def _submit_input(self, event: Input.Submitted) -> None:
        event.stop()
        self._submit()

    def _submit(self) -> None:
        passphrase = self.query_one("#skill-trust-bootstrap-input", Input).value
        confirm = self.query_one("#skill-trust-bootstrap-confirm-input", Input).value
        error = self.query_one("#skill-trust-bootstrap-error", Static)
        if not passphrase:
            error.update("Passphrase cannot be blank.")
            return
        if passphrase != confirm:
            error.update("Passphrases do not match.")
            return
        self.dismiss(passphrase)
