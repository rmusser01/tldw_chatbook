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
            pending_skill_script=None,
            diff_summary=_text("diff_summary"),
            next_action=_text("next_action"),
        )
