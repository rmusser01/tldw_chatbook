"""Scope-aware models for Study screen navigation and persistence."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Optional

from ...Utils.input_validation import (
    escape_markup,
    sanitize_string,
    validate_text_input,
)


MATERIAL_SOURCE_LIBRARY = "library"
MATERIAL_TITLE_LIBRARY_SOURCES = "Local Library Sources"
#: How many carried source titles Study KEEPS (``StudyScreen._clean_material_
#: titles`` truncates the hand-off to this many).
STUDY_MATERIAL_TITLES_LIMIT = 10
#: How many of those titles a summary NAMES before counting the rest.
STUDY_MATERIAL_TITLES_NAMED_LIMIT = 3

_HTML_TAG_RE = re.compile(r"<[^>]*>")
_DANGEROUS_TEXT_RE = re.compile(
    r"javascript\s*:|\bon(?:click|error)\s*=", re.IGNORECASE
)


def clean_material_text(value: Any, *, max_length: int) -> str:
    """Sanitize one piece of carried material text, or return ``""``.

    Hoisted from ``StudyScreen._clean_material_text`` (TASK-34000.6 fix round
    1) so the Library hand-off and Study clean titles with the SAME rule.

    Args:
        value: Raw text (coerced with ``str``; ``None`` is empty).
        max_length: Length bound applied before and after stripping.

    Returns:
        The cleaned text, or ``""`` when nothing safe remains.
    """
    text = sanitize_string(str(value or ""), max_length=max_length).strip()
    if not text:
        return ""
    text = _HTML_TAG_RE.sub("", text)
    text = _DANGEROUS_TEXT_RE.sub("", text).strip()
    if not validate_text_input(text, max_length=max_length, allow_html=False):
        return ""
    return text


STUDY_SOURCE_ITEMS_LIMIT = 25
STUDY_MATERIAL_TITLE_LENGTH_LIMIT = 160
STUDY_MATERIAL_SUMMARY_LENGTH_LIMIT = 1000
STUDY_SOURCE_ID_LENGTH_LIMIT = 128
STUDY_INITIAL_SECTIONS = frozenset(
    {
        "dashboard",
        "paths",
        "flashcards",
        "quizzes",
        "mindmaps",
        "course",
        "guides",
        "learning_map",
    }
)

#: task-4011: the screens that hand off INTO Study, threaded through
#: ``HandoffChannel.STUDY_ORIGIN`` so the breadcrumb and Escape target can
#: name where the user actually came from. "library" is the default when no
#: origin is staged (every pre-task-4011 entry path was Library's staging
#: canvas, task-2854).
STUDY_ORIGIN_HOME = "home"
STUDY_ORIGIN_LIBRARY = "library"
STUDY_ORIGINS = frozenset({STUDY_ORIGIN_HOME, STUDY_ORIGIN_LIBRARY})


class StudyScopeType(str, Enum):
    GLOBAL = "global"
    WORKSPACE = "workspace"


@dataclass(frozen=True)
class CarriedTitlesSummary:
    """The carried-scope description both Library and Study render.

    Attributes:
        named: The titles the line names, in order.
        remaining: How many MORE titles Study keeps beyond the named ones.
    """

    named: tuple[str, ...]
    remaining: int

    @property
    def text(self) -> str:
        """``"a, b, c"`` or ``"a, b, c and N more"``."""
        joined = ", ".join(self.named)
        if self.remaining > 0:
            return f"{joined} and {self.remaining} more"
        return joined


def summarize_carried_titles(titles: Iterable[str]) -> CarriedTitlesSummary:
    """Describe the carried titles under the ONE rule Library and Study share.

    TASK-34000.6 (S-05): the Library hand-off counted every sampled title
    ("… and 134 more") while Study, which truncates the hand-off to
    ``STUDY_MATERIAL_TITLES_LIMIT`` on receipt, counted what it kept ("+7
    more"). Both now name the first ``STUDY_MATERIAL_TITLES_NAMED_LIMIT``
    titles and count the rest of what Study keeps, so the two cannot disagree.
    Fix round 1: the describer also CLEANS (``clean_material_text``, the rule
    Study applies on receipt) and markup-escapes the names itself, so a caller
    cannot feed it a different title set -- a title that cleans to empty (for
    example ``<draft>``) drops out on both surfaces alike.

    Args:
        titles: The carried titles in order, raw; entries that clean to empty
            are ignored.

    Returns:
        The named titles (markup-escaped, display-ready) and the remaining
        count under Study's cap.
    """
    kept: list[str] = []
    for title in titles:
        clean = clean_material_text(title, max_length=STUDY_MATERIAL_TITLE_LENGTH_LIMIT)
        if clean:
            kept.append(clean)
        if len(kept) >= STUDY_MATERIAL_TITLES_LIMIT:
            break
    named = tuple(
        escape_markup(title) for title in kept[:STUDY_MATERIAL_TITLES_NAMED_LIMIT]
    )
    return CarriedTitlesSummary(named=named, remaining=len(kept) - len(named))


@dataclass(frozen=True)
class StudySourceItem:
    """Concrete source item that can back server-side study generation.

    Attributes:
        source_type: Stable source category, such as ``note`` or ``media``.
        source_id: Stable identifier for the selected source record.
        label: Optional user-facing source label.
        excerpt_text: Optional source excerpt for generation context.
        locator: Optional structured locator metadata for the source.
    """

    source_type: str
    source_id: str
    label: Optional[str] = None
    excerpt_text: Optional[str] = None
    locator: dict[str, Any] = field(default_factory=dict)

    def as_payload(self) -> dict[str, Any]:
        """Return the server study-pack generation payload.

        Returns:
            Dictionary containing the source identity plus optional label,
            excerpt, and locator metadata.
        """
        payload: dict[str, Any] = {
            "source_type": self.source_type,
            "source_id": self.source_id,
        }
        if self.label:
            payload["label"] = self.label
        if self.excerpt_text:
            payload["excerpt_text"] = self.excerpt_text
        payload["locator"] = dict(self.locator or {})
        return payload


@dataclass(frozen=True)
class StudyScopeContext:
    """Durable Study scope inputs passed across screen navigation."""

    scope_type: StudyScopeType = StudyScopeType.GLOBAL
    workspace_id: Optional[str] = None
    workspace_name: Optional[str] = None
    return_hint: Optional[str] = None
    material_source: Optional[str] = None
    material_title: Optional[str] = None
    material_summary: Optional[str] = None
    material_titles: tuple[str, ...] = field(default_factory=tuple)
    source_items: tuple[StudySourceItem, ...] = field(default_factory=tuple)


@dataclass
class StudyScopeState:
    """Effective Study scope, including runtime-derived fields."""

    scope_type: StudyScopeType = StudyScopeType.GLOBAL
    workspace_id: Optional[str] = None
    workspace_name: Optional[str] = None
    return_hint: Optional[str] = None
    backend: str = "local"
    workspace_scope_available: bool = False
    error_message: Optional[str] = None
    material_source: Optional[str] = None
    material_title: Optional[str] = None
    material_summary: Optional[str] = None
    material_titles: tuple[str, ...] = field(default_factory=tuple)
    source_items: tuple[StudySourceItem, ...] = field(default_factory=tuple)

    def as_context(self) -> StudyScopeContext:
        return StudyScopeContext(
            scope_type=self.scope_type,
            workspace_id=self.workspace_id,
            workspace_name=self.workspace_name,
            return_hint=self.return_hint,
            material_source=self.material_source,
            material_title=self.material_title,
            material_summary=self.material_summary,
            material_titles=self.material_titles,
            source_items=self.source_items,
        )
