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
from typing import Any, Mapping, Sequence

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

#: Matches every run of characters that cannot appear in a canonical slug.
_NON_SLUG_RE = re.compile(r"[^a-z0-9]+")


def canonical_slug(text: str) -> str:
    """The single stored form of a tag name.

    One canonical form is what makes "does this tag already exist?" a
    decidable question, which is what limits the ``broke-character`` /
    ``OOC`` / ``out-of-character`` fragmentation per-bench extension invites.

    Args:
        text: A human-typed tag name.

    Returns:
        str: Lowercase, with every run of non-alphanumerics collapsed to a
        single hyphen and leading/trailing hyphens removed.

    Raises:
        ValueError: If nothing usable survives -- an empty slug would collide
            with every other empty slug and silently merge unrelated tags.
    """
    slug = _NON_SLUG_RE.sub("-", str(text).strip().lower()).strip("-")
    if not slug:
        raise ValueError(f"{text!r} has no characters usable in a tag slug.")
    return slug


def coerce_tag(raw: Any) -> Tag:
    """One extra-tag entry as a validated ``Tag``.

    Args:
        raw: A ``Tag``, or a mapping with ``slug`` and ``kind`` and an
            optional ``label``.

    Returns:
        Tag: The validated tag, its slug canonicalised.

    Raises:
        ValueError: If the entry is not a mapping, has no slug, or omits the
            kind -- naming the slug, since a guessed kind mis-groups the
            observation in the summary.
    """
    if isinstance(raw, Tag):
        return raw
    if not isinstance(raw, Mapping):
        raise ValueError(f"An extra tag must be a mapping or Tag, got {raw!r}.")
    raw_slug = raw.get("slug")
    if not raw_slug:
        raise ValueError(f"An extra tag needs a slug: {dict(raw)!r}.")
    slug = canonical_slug(str(raw_slug))
    kind = raw.get("kind")
    if not kind:
        raise ValueError(
            f"Extra tag {slug!r} has no kind. Every tag states one of "
            f"{', '.join(TAG_KINDS)} -- it is never guessed."
        )
    label = str(raw.get("label") or slug)
    return Tag(slug=slug, label=label, kind=str(kind))


def resolve_vocabulary(extra_tags: Sequence[Any] = ()) -> tuple[Tag, ...]:
    """The full tag vocabulary for one bench: built-ins plus its extras.

    An extra whose slug matches a built-in relabels it in place rather than
    appending a duplicate. It may NOT change a built-in's kind: two benches
    whose ``failure`` sets mean different things cannot be read in one
    summary.

    Args:
        extra_tags: The bench's ``extra_tags``, as ``Tag`` objects or as the
            raw mappings older rows and run snapshots store.

    Returns:
        tuple[Tag, ...]: Built-ins in their declared order, with overrides
        applied in place, then each new extra in the order supplied.

    Raises:
        ValueError: If an extra is malformed, omits its kind, or tries to
            change a built-in's kind -- naming the slug in every case.
    """
    builtin_kinds = {tag.slug: tag.kind for tag in BUILTIN_TAGS}
    resolved: dict[str, Tag] = {tag.slug: tag for tag in BUILTIN_TAGS}
    for raw in extra_tags or ():
        tag = coerce_tag(raw)
        builtin_kind = builtin_kinds.get(tag.slug)
        if builtin_kind is not None and tag.kind != builtin_kind:
            raise ValueError(
                f"Extra tag {tag.slug!r} would change the built-in kind "
                f"{builtin_kind!r} to {tag.kind!r}. Built-in kinds are fixed "
                f"so one summary can read every bench."
            )
        resolved[tag.slug] = tag
    return tuple(resolved.values())


def tag_by_slug(vocabulary: Sequence[Tag], slug: str) -> Tag:
    """One tag from a vocabulary.

    Args:
        vocabulary: The bench's resolved vocabulary.
        slug: The canonical slug to find.

    Returns:
        Tag: The matching tag.

    Raises:
        KeyError: If no tag matches -- naming the slug and the vocabulary, so
            a stored annotation referencing a retired tag says which one.
    """
    for tag in vocabulary:
        if tag.slug == slug:
            return tag
    known = ", ".join(t.slug for t in vocabulary)
    raise KeyError(f"No tag {slug!r} in this bench's vocabulary ({known}).")
