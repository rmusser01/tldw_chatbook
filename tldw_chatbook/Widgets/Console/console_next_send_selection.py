"""Inspector-only transport for disposable Personal Context explanations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from tldw_chatbook.Chat.console_chat_models import ConsoleContextSnapshot

if TYPE_CHECKING:
    from tldw_chatbook.Personal_Context.context_service import (
        ProfileContextSelectionExplanation,
    )


class ConsoleNextSendSelectionResult:
    """Keep diagnostic metadata outside generic snapshot serialization."""

    __slots__ = ("_is_current", "explanation", "snapshot")

    def __init__(
        self,
        snapshot: ConsoleContextSnapshot,
        explanation: ProfileContextSelectionExplanation | None,
        is_current: Callable[[], Awaitable[bool]],
    ) -> None:
        self.snapshot = snapshot
        self.explanation = explanation
        self._is_current = is_current

    async def is_current(self) -> bool:
        """Validate the captured request immediately before publication."""

        return self.explanation is not None and await self._is_current()
