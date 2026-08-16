"""Console ``/research`` command intent parsing (task-16793).

Pure parser shared by the Console dispatch: extracts an optional policy
token (``--policy web_only|academic_only|web_first|academic_first|balanced``)
and an optional providers/categories list (``--providers biomedical,zenodo``)
from the command args, rejecting anything malformed before any run launches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "RESEARCH_POLICIES",
    "ResearchCommandIntent",
    "parse_research_command",
]

RESEARCH_POLICIES = (
    "web_only",
    "academic_only",
    "web_first",
    "academic_first",
    "balanced",
)

def _extract_flags(args: str) -> tuple[dict[str, str], str]:
    """Split --flag value tokens out of the args.

    Args:
        args: The raw /research argument text.

    Returns:
        (flags dict, remaining question text). Flags may appear anywhere;
        a flag consumes exactly the next whitespace-delimited token.
    """
    flags: dict[str, str] = {}
    kept: list[str] = []
    tokens = args.split()
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in ("--policy", "--providers") and index + 1 < len(tokens):
            flags[token.lstrip("-")] = tokens[index + 1]
            index += 2
            continue
        kept.append(token)
        index += 1
    return flags, " ".join(kept)


@dataclass(frozen=True)
class ResearchCommandIntent:
    """A parsed ``/research`` invocation.

    Attributes:
        question: The research question (flags stripped).
        source_policy: The lane policy for the launched run.
        providers: Optional source-id/category tokens for the academic lane.
    """

    question: str
    source_policy: str = "balanced"
    providers: list[str] | None = None

    def provider_overrides(self) -> dict[str, Any] | None:
        """The run's provider_overrides payload (None when unfiltered).

        Returns:
            ``{"academic_providers": [...]}`` when providers were given,
            else None.
        """
        if self.providers:
            return {"academic_providers": list(self.providers)}
        return None


def parse_research_command(args: str) -> ResearchCommandIntent:
    """Parse ``/research`` arguments into a launch intent.

    Args:
        args: The raw text after the command word.

    Returns:
        The parsed intent (question, policy, providers).

    Raises:
        ValueError: For an unknown policy, an empty question, or empty
            provider lists.
    """
    from tldw_chatbook.Utils.input_validation import validate_text_input

    raw_args = str(args or "")
    if not validate_text_input(raw_args, max_length=2000):
        raise ValueError("arguments too long or contain invalid content")

    policy = "balanced"
    providers: list[str] | None = None

    flags, question = _extract_flags(raw_args)
    if "--policy" in raw_args.split() and "policy" not in flags:
        raise ValueError("--policy needs a value")
    if "--providers" in raw_args.split() and "providers" not in flags:
        raise ValueError("--providers needs a value")

    if "policy" in flags:
        value = flags["policy"].lower()
        if value not in RESEARCH_POLICIES:
            raise ValueError(
                f"unknown policy {value!r}; expected one of {RESEARCH_POLICIES}"
            )
        policy = value

    if "providers" in flags:
        tokens = [t.strip().lower() for t in flags["providers"].split(",") if t.strip()]
        if not tokens:
            raise ValueError("--providers needs at least one name or category")
        providers = tokens

    question = question.strip()
    if not question:
        raise ValueError("--policy/--providers given but no question remains")

    return ResearchCommandIntent(
        question=question, source_policy=policy, providers=providers
    )
