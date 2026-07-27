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
        """Return whether the resume panel should be visible."""
        return any(
            (
                self.summary.strip(),
                self.last_step.strip(),
                self.diff_summary.strip(),
                self.next_action.strip(),
            )
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
    def from_dict(cls, data: dict[str, Any] | None) -> "TaskResumeState":
        """Restore state while dropping an unresolvable skill-script round."""
        if not data:
            return cls()
        return cls(
            summary=data.get("summary", ""),
            last_step=data.get("last_step", ""),
            pending_approval=data.get("pending_approval"),
            pending_skill_install=data.get("pending_skill_install"),
            pending_skill_script=None,
            diff_summary=data.get("diff_summary", ""),
            next_action=data.get("next_action", ""),
        )
