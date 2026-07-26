# tldw_chatbook/Tools/base.py
"""The `Tool` ABC.

Lives apart from ``tool_executor`` because the executor is being removed
(it has no callers left) while the ABC is load-bearing: ``risk_tags`` is
what the permission gate resolves against, and every built-in pack tool
subclasses this.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class Tool(ABC):
    """Base class for all tools that can be called by LLMs."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the tool as it will be called by the LLM."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the tool does."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON Schema for the tool's parameters."""
        pass

    @property
    def risk_tags(self) -> tuple[str, ...]:
        """Risk classes for the permission gate, e.g. ``("mutates",)``.

        Concrete with an empty default so every existing subclass keeps
        working unchanged. The vocabulary is the permission store's
        ``HIGH_RISK_TAGS`` (``mutates``/``process``/``network``) -- a tool
        tagged with one of those has an INHERITED ``allow`` floored to ``ask`` by
        ``resolve_builtin_state``. Read-only tools leave this empty.

        Returns:
            A tuple of risk tag strings drawn from ``HIGH_RISK_TAGS``;
            empty for a tool with no elevated risk.
        """
        return ()

    @property
    def timeout_seconds(self) -> float:
        """Per-call wall-clock ceiling, or 0 to use the run's default.

        Concrete with a 0 default so every existing subclass is unchanged.
        A tool whose real work legitimately outlasts
        ``RunBudget.max_tool_call_seconds`` (ingestion, transcription)
        raises this; a tool that must be cut short sooner (``run_command``)
        lowers it. Note the timeout ABANDONS the worker thread rather than
        killing it, so a tool raising this must be idempotent or must say
        so in its timeout message.

        Returns:
            Seconds, or 0.0 to defer to the run budget.
        """
        return 0.0

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool with the given parameters.

        Args:
            **kwargs: Parameters for the tool

        Returns:
            Dictionary with the result or error
        """
        pass

    def to_openai_format(self) -> dict:
        """Convert tool to OpenAI function format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
