"""Serializable native Console task-resume state."""

from dataclasses import dataclass
from typing import Any


@dataclass
class TaskResumeState:
    """Serializable summary of the current agentic task state."""

    summary: str = ""
    last_step: str = ""
    pending_approval: dict[str, Any] | None = None
    pending_skill_install: dict[str, Any] | None = None
    # TASK-1051: this field is fully live within one screen instance -- the
    # normal path is `ChatScreen._set_console_pending_skill_script` mutating
    # it directly while a `ConsoleChatController` round is actually armed --
    # but `from_dict` below deliberately never repopulates it from a
    # snapshot. See `from_dict`'s docstring for why that asymmetry with
    # `pending_skill_install` (restored below) is intentional, not an
    # oversight to "fix" by making the two match.
    pending_skill_script: dict[str, Any] | None = None
    diff_summary: str = ""
    next_action: str = ""

    def has_resume_content(self) -> bool:
        """Return whether the resume panel should be visible.

        Returns:
            ``True`` when at least one valid text field contains content.
        """
        return any(
            value.strip()
            for value in (
                self.summary,
                self.last_step,
                self.diff_summary,
                self.next_action,
            )
            if isinstance(value, str)
        )

    def has_pending_approval(self) -> bool:
        """Return whether an approval prompt should be shown."""
        return bool(self.pending_approval)

    def has_pending_skill_install(self) -> bool:
        """Return whether a skill-install confirmation should be shown."""
        return bool(self.pending_skill_install)

    def has_pending_skill_script(self) -> bool:
        """Return whether a skill-script confirmation should be shown."""
        return bool(self.pending_skill_script)

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable task-resume payload."""
        return {
            "summary": self.summary,
            "last_step": self.last_step,
            "pending_approval": self.pending_approval,
            "pending_skill_install": self.pending_skill_install,
            "pending_skill_script": self.pending_skill_script,
            "diff_summary": self.diff_summary,
            "next_action": self.next_action,
        }

    @classmethod
    def from_dict(cls, data: object | None) -> "TaskResumeState":
        """Restore validated state while dropping an unresolvable script round.

        TASK-1051 (call-chain investigated, not guessed): this snapshot
        round-trips through ``ChatScreen.save_state``/``restore_state``,
        which ``app.py``'s ``handle_screen_navigation`` calls on every TAB
        SWITCH -- never on an app restart. ``ScreenStateStore`` (the thing
        holding the dict this method receives) is explicitly "memory-only
        ownership for cross-visit screen snapshots" (its own module
        docstring); it never touches disk. That still doesn't make a
        restored round resumable: ``ChatScreen._create_navigation_screen``'s
        docstring is explicit that screens are "never cached and
        re-mounted" -- every navigation builds a brand-new ``ChatScreen``
        whose ``_console_chat_controller`` starts life as ``None`` and is
        lazily rebuilt from scratch (``_ensure_console_chat_controller``).
        A skill-script confirm round only exists as an entry in that
        controller's own ``_pending_skill_script_rounds`` dict, keyed by
        ``request_id`` and guarding a worker thread blocked on a
        ``threading.Event`` -- state that lives on the OLD controller
        instance and is gone once a new one is built. Restoring the
        ``pending_skill_script`` payload here would still mount an
        apparently-live ``SkillScriptConfirmCard``, but any decision on it
        reaches ``ConsoleChatController.resolve_pending_skill_script``,
        which silently drops a resolve whose ``request_id`` doesn't match a
        currently-armed round (fail-closed by design) -- i.e. a real-looking
        card whose buttons do nothing forever. Dropping the payload here
        instead is what keeps the card from mounting at all, which is the
        strictly better failure mode (see
        ``Tests/UI/test_skill_script_confirm_card.py::
        test_restored_state_drops_the_pending_script_so_no_dead_card_appears``,
        which pins this exact contract and predates this task).

        ``pending_skill_install`` is restored below despite going through
        the IDENTICAL architecture (``_pending_skill_install_rounds``,
        ``resolve_pending_skill_install``'s identical strict ``request_id``
        match) -- it is exposed to the exact same dead-card failure mode.
        That is a pre-existing asymmetry from TASK-910 (which added
        install's round-keyed restore before this dead-UI hazard was
        identified for script), not a principled distinction, and its own
        round-trip is pinned by
        ``Tests/UI/test_console_skill_install_confirm.py::
        test_task_resume_state_pending_skill_install_roundtrip``. Changing
        that is out of TASK-1051's scope (only ``pending_skill_script`` was
        in the acceptance criteria) and is left as a follow-up rather than
        silently "fixed" here alongside an unrelated contract.

        Args:
            data: Untrusted value read from a persisted Console snapshot.

        Returns:
            A resume state containing only correctly typed snapshot fields.
        """
        if not isinstance(data, dict):
            return cls()

        def _text(key: str) -> str:
            value = data.get(key)
            return value if isinstance(value, str) else ""

        def _payload(key: str) -> dict[str, Any] | None:
            value = data.get(key)
            return dict(value) if isinstance(value, dict) else None

        return cls(
            summary=_text("summary"),
            last_step=_text("last_step"),
            pending_approval=_payload("pending_approval"),
            pending_skill_install=_payload("pending_skill_install"),
            # Deliberately NOT `_payload("pending_skill_script")` -- see the
            # docstring above for why restoring it would only ever produce
            # a dead card, never a functional one.
            pending_skill_script=None,
            diff_summary=_text("diff_summary"),
            next_action=_text("next_action"),
        )
