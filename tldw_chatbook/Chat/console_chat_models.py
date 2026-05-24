"""Pure Console-native chat state contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal
from uuid import uuid4


class ConsoleMessageRole(str, Enum):
    """Roles used by the native Console transcript."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ConsoleRunStatus(str, Enum):
    """Lifecycle states for a Console send or recovery run."""

    IDLE = "idle"
    VALIDATING = "validating"
    STREAMING = "streaming"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    STOPPED = "stopped"
    FAILED = "failed"
    RETRYING = "retrying"


ConsoleMessageStatus = Literal["complete", "pending", "streaming", "stopped", "failed"]


@dataclass(frozen=True)
class ConsoleStagedSource:
    """A source currently staged for use by Console."""

    source_id: str
    label: str
    source_type: str
    workspace_id: str | None = None


@dataclass(frozen=True)
class ConsoleWorkspaceContext:
    """Workspace and source policy state used before sending to a provider."""

    active_workspace_id: str = "global"
    staged_sources: tuple[ConsoleStagedSource, ...] = ()
    active_run_id: str | None = None
    handoff_id: str | None = None

    @property
    def blocked_sources(self) -> list[ConsoleStagedSource]:
        """Return staged sources that cannot be used in the active workspace."""
        return [
            source
            for source in self.staged_sources
            if source.workspace_id not in (None, self.active_workspace_id)
        ]

    @property
    def allowed_sources(self) -> list[ConsoleStagedSource]:
        """Return staged sources available to the active workspace."""
        blocked = {source.source_id for source in self.blocked_sources}
        return [source for source in self.staged_sources if source.source_id not in blocked]

    @property
    def has_policy_blocks(self) -> bool:
        """Return whether any staged source is blocked by workspace policy."""
        return bool(self.blocked_sources)

    @property
    def recovery_copy(self) -> str:
        """Human-readable recovery text for workspace policy blocks."""
        labels = ", ".join(source.label for source in self.blocked_sources)
        return f"Workspace policy blocked sources outside {self.active_workspace_id}: {labels}"


@dataclass(frozen=True)
class ConsoleProviderSelection:
    """Effective provider/model/base URL selected for a Console send."""

    provider: str
    base_url: str | None = None
    explicit_model: str | None = None
    configured_model: str | None = None
    workspace_context: ConsoleWorkspaceContext = field(default_factory=ConsoleWorkspaceContext)


@dataclass(frozen=True)
class ConsoleRunState:
    """Visible run state surfaced in Console controls and inspector."""

    status: ConsoleRunStatus = ConsoleRunStatus.IDLE
    visible_copy: str = ""

    @classmethod
    def blocked(cls, visible_copy: str) -> "ConsoleRunState":
        """Build a blocked run state with visible recovery copy."""
        return cls(ConsoleRunStatus.BLOCKED, visible_copy)

    @classmethod
    def retrying(cls, visible_copy: str = "Retrying failed response") -> "ConsoleRunState":
        """Build a retrying run state."""
        return cls(ConsoleRunStatus.RETRYING, visible_copy)

    @property
    def is_send_allowed(self) -> bool:
        """Return whether Console can accept a new send from this state."""
        return self.status in {
            ConsoleRunStatus.IDLE,
            ConsoleRunStatus.BLOCKED,
            ConsoleRunStatus.COMPLETED,
            ConsoleRunStatus.FAILED,
            ConsoleRunStatus.STOPPED,
        }

    @property
    def is_stop_allowed(self) -> bool:
        """Return whether Console can stop an active stream from this state."""
        return self.status is ConsoleRunStatus.STREAMING


@dataclass
class ConsoleChatMessage:
    """A native Console transcript message."""

    role: ConsoleMessageRole
    content: str
    id: str = field(default_factory=lambda: str(uuid4()))
    turn_id: str | None = None
    status: ConsoleMessageStatus = "complete"
    persisted_message_id: str | None = None
    variants: "ConsoleVariantSet | None" = None


@dataclass(frozen=True)
class ConsoleVariant:
    """One regenerated variant for a turn."""

    content: str
    id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class ConsoleVariantSet:
    """Regenerated variants for one turn with current selection state."""

    turn_id: str
    variants: list[ConsoleVariant]
    selected_index: int = 0

    @classmethod
    def from_contents(
        cls,
        *,
        turn_id: str,
        contents: list[str],
        selected_index: int = 0,
    ) -> "ConsoleVariantSet":
        """Build a variant set from raw message contents."""
        if not contents:
            raise ValueError("ConsoleVariantSet requires at least one variant")
        if selected_index < 0 or selected_index >= len(contents):
            raise ValueError("selected_index must reference an existing variant")
        return cls(
            turn_id=turn_id,
            variants=[ConsoleVariant(content) for content in contents],
            selected_index=selected_index,
        )

    @property
    def current(self) -> ConsoleVariant:
        """Return the currently selected variant."""
        return self.variants[self.selected_index]

    @property
    def can_go_previous(self) -> bool:
        """Return whether a previous variant exists."""
        return self.selected_index > 0

    @property
    def can_go_next(self) -> bool:
        """Return whether a next variant exists."""
        return self.selected_index < len(self.variants) - 1
