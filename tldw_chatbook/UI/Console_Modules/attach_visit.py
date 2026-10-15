"""Identity of one visible Console attachment visit across awaited refreshes."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConsoleAttachVisit:
    """An observation, not authority to finish a later visit or replacement view."""

    runtime: Any
    view: object
    generation: int | None
    visit: int
    active: bool

    def same_visit(self, current: "ConsoleAttachVisit") -> bool:
        """Compare identities without invoking runtime or view equality."""
        return (
            self.runtime is current.runtime
            and self.view is current.view
            and self.generation == current.generation
            and self.visit == current.visit
        )

    def can_complete(self, current: "ConsoleAttachVisit") -> bool:
        """Require the same currently visible visit and original runtime claim."""
        return (
            self.active
            and current.active
            and self.same_visit(current)
            and self.generation is not None
            and self.runtime.view is self.view
            and self.runtime._attached_generation == self.generation
        )
