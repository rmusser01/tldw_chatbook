"""Serializable native Console task-resume state."""

from dataclasses import dataclass
from typing import Any


@dataclass
class TaskResumeState:
    """Serializable summary of the current agentic task state."""

    summary: str = ""
    last_step: str = ""
    pending_approval: dict[str, Any] | None = None
    # TASK-1051 / TASK-1130: this field is fully live within one screen
    # instance -- the normal path is
    # `ChatScreen._set_console_pending_skill_install` mutating it directly
    # while a `ConsoleChatController` round is actually armed -- but
    # `from_dict` below deliberately never repopulates it from a snapshot.
    # See `from_dict`'s docstring for why that's true of both skill-confirm
    # fields, not an oversight to "fix" by restoring one of them.
    pending_skill_install: dict[str, Any] | None = None
    # TASK-1051 / TASK-1130: this field is fully live within one screen instance -- the
    # normal path is `ChatScreen._set_console_pending_skill_script` mutating
    # it directly while a `ConsoleChatController` round is actually armed --
    # but `from_dict` below deliberately never repopulates it from a
    # snapshot. See `from_dict`'s docstring -- `pending_skill_install`
    # (also dropped below, as of TASK-1130) goes through the identical
    # architecture and is no longer an asymmetric exception.
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
        """Restore validated state while dropping unresolvable skill-confirm rounds.

        TASK-1051 (call-chain investigated, not guessed) and TASK-1130
        (which closed the asymmetry this docstring used to describe): this
        snapshot round-trips through ``ChatScreen.save_state``/
        ``restore_state``, which ``app.py``'s ``handle_screen_navigation``
        calls on every TAB SWITCH -- never on an app restart.
        ``ScreenStateStore`` (the thing holding the dict this method
        receives) is explicitly "memory-only ownership for cross-visit
        screen snapshots" (its own module docstring); it never touches
        disk. That still doesn't make a restored round resumable:
        ``TldwCli._create_navigation_screen``'s (app.py) docstring is
        explicit that screens are "never cached and re-mounted" -- every
        navigation builds a brand-new ``ChatScreen`` whose
        ``_console_chat_controller`` starts life as ``None`` and is lazily
        rebuilt from scratch (``_ensure_console_chat_controller``). A
        skill-script or skill-install confirm round only exists as an entry
        in that controller's own ``_pending_skill_script_rounds``/
        ``_pending_skill_install_rounds`` dict, keyed by ``request_id`` and
        guarding a worker thread blocked on a ``threading.Event`` -- state
        that lives on the OLD controller instance and is gone once a new
        one is built. Restoring either payload here would still mount an
        apparently-live confirm card, but any decision on it reaches
        ``ConsoleChatController.resolve_pending_skill_script``/
        ``resolve_pending_skill_install``, which silently drops a resolve
        whose ``request_id`` doesn't match a currently-armed round
        (fail-closed by design) -- i.e. a real-looking card whose buttons
        do nothing forever. Dropping both payloads here instead is what
        keeps either card from mounting at all, which is the strictly
        better failure mode (see
        ``Tests/UI/test_skill_script_confirm_card.py::
        test_restored_state_drops_the_pending_script_so_no_dead_card_appears``
        and
        ``Tests/UI/test_console_skill_install_confirm.py::
        test_restored_state_drops_the_pending_install_so_no_dead_card_appears``,
        which pin this exact contract for each field).

        TASK-1130 also re-verified TASK-1051's premise still held before
        applying it to ``pending_skill_install``: no reconnection seam has
        appeared since -- ``ChatScreen`` still gets a fresh
        ``ConsoleChatController`` per navigation (unchanged), and TASK-1143
        added a navigation guard that DENIES every in-flight/parked round
        on teardown (``ConsoleChatController.busy_fleet_session_count``
        gates a confirm-before-leaving dialog, and the outgoing controller's
        ``shutdown()`` still denies whatever is left) -- so a round captured
        in a snapshot is now not merely orphaned but actively torn down
        before the snapshot can ever be restored. A real reconnection path
        (making a restored round resumable) would require rounds to survive
        controller teardown, which contradicts that deny-on-teardown
        architecture -- not attempted here. Restoring
        ``pending_skill_install`` was originally added by TASK-910 for
        round-trip data fidelity; live serialization stays pinned by
        ``test_task_resume_state_pending_skill_install_serializes_while_live``
        and the restore-side drop by
        ``test_restored_state_drops_the_pending_install_so_no_dead_card_appears``.

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
            # Deliberately NOT `_payload("pending_skill_install")` (as of
            # TASK-1130) / NOT `_payload("pending_skill_script")` (as of
            # TASK-1051) -- see the docstring above for why restoring
            # either would only ever produce a dead card, never a
            # functional one.
            pending_skill_install=None,
            pending_skill_script=None,
            diff_summary=_text("diff_summary"),
            next_action=_text("next_action"),
        )
