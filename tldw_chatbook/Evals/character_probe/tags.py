"""The character-probe review vocabulary.

Every tag carries a kind so no view can imply "fewer tags is better": a
``positive`` tag and a ``failure`` tag are both observations, and the summary
groups by kind rather than counting them together. Creating a tag therefore
REQUIRES a kind -- the design spec rules out guessing one, and rules out
``notable`` as a fallback specifically because it would hide genuine failures
in the view meant to surface them.

Stdlib only, deliberately: this module is imported by the engine, the review
UI, and the summary, and must never drag a database or Textual behind it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: The only kinds a tag may carry. Order is display order, worst first.
TAG_KINDS: tuple[str, str, str] = ("failure", "notable", "positive")

#: A canonical slug: lowercase, digits, and single interior hyphens.
_SLUG_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


@dataclass(frozen=True)
class Tag:
    """One review tag.

    Args:
        slug: The canonical stored form (lowercase, hyphen-separated).
        label: What a reviewer reads. Never empty.
        kind: One of ``TAG_KINDS``.

    Raises:
        ValueError: If the slug is not canonical, the label is blank, or the
            kind is not one of ``TAG_KINDS`` -- naming the offending value.
    """

    slug: str
    label: str
    kind: str

    def __post_init__(self) -> None:
        if not isinstance(self.slug, str) or not _SLUG_RE.match(self.slug):
            raise ValueError(
                f"Tag slug must be lowercase-hyphenated, got {self.slug!r}."
            )
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError(f"Tag {self.slug!r} needs a non-empty label.")
        if self.kind not in TAG_KINDS:
            raise ValueError(
                f"Tag {self.slug!r} has kind {self.kind!r}; must be one of "
                f"{', '.join(TAG_KINDS)}. A kind is never guessed -- the wrong "
                f"one mis-groups the observation in the summary."
            )


#: The vocabulary every bench starts with (spec, Tags section). A bench
#: extends this through ``CharacterProbeConfig.extra_tags``; it never
#: replaces it.
BUILTIN_TAGS: tuple[Tag, ...] = (
    Tag("broke-character", "Broke character", "failure"),
    Tag("refused", "Refused", "failure"),
    Tag("leaked-prompt", "Leaked the card's prompt", "failure"),
    Tag("generic-assistant-voice", "Generic assistant voice", "failure"),
    Tag("contradicted-card", "Contradicted the card", "failure"),
    Tag("ignored-the-question", "Ignored the question", "failure"),
    Tag("notable", "Notable", "notable"),
    Tag("surprising", "Surprising", "notable"),
    Tag("in-character", "In character", "positive"),
    Tag("handled-well", "Handled well", "positive"),
)
