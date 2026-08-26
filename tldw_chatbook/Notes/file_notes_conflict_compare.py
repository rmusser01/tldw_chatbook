"""Bounded, presentation-neutral File Notes conflict comparison."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import unified_diff
from hashlib import sha256
from typing import Literal

CONFLICT_DIFF_MAX_INPUT_CHARS = 200_000
CONFLICT_DIFF_MAX_INPUT_LINES = 10_000
CONFLICT_DIFF_MAX_OUTPUT_CHARS = 120_000
CONFLICT_DIFF_MAX_OUTPUT_LINES = 2_000

_SIDE_ROLES = {
    "Base": "editor baseline",
    "Draft": "current editor",
    "Disk": "latest readable snapshot",
}


@dataclass(frozen=True, slots=True)
class ConflictSide:
    """One immutable side of a File Notes conflict."""

    label: Literal["Base", "Draft", "Disk"]
    state: Literal["readable", "absent", "unreadable"]
    text: str | None
    character_count: int
    byte_count: int
    line_count: int
    sha256: str
    detail: str = ""

    @classmethod
    def from_text(
        cls,
        label: Literal["Base", "Draft", "Disk"],
        text: str,
    ) -> ConflictSide:
        """Create one exact readable body identity.

        Args:
            label: Stable conflict-side name.
            text: Exact editor-body text captured for the side.

        Returns:
            An immutable side with exact size and SHA-256 metadata.
        """
        encoded = text.encode("utf-8")
        return cls(
            label=label,
            state="readable",
            text=text,
            character_count=len(text),
            byte_count=len(encoded),
            line_count=len(text.splitlines()),
            sha256=sha256(encoded).hexdigest(),
        )

    @classmethod
    def absent(cls, label: Literal["Disk"] = "Disk") -> ConflictSide:
        """Represent an explicitly absent disk side.

        Args:
            label: Stable disk-side name.

        Returns:
            An immutable absent side.
        """
        return cls(label, "absent", None, 0, 0, 0, "")

    @classmethod
    def unreadable(
        cls,
        detail: str,
        label: Literal["Disk"] = "Disk",
    ) -> ConflictSide:
        """Represent a disk side that could not be read.

        Args:
            detail: Bounded user-facing read failure detail.
            label: Stable disk-side name.

        Returns:
            An immutable unreadable side.
        """
        return cls(label, "unreadable", None, 0, 0, 0, "", detail[:500])

    @property
    def summary(self) -> str:
        """Return a complete literal identity line for this side."""
        if self.state == "absent":
            return f"{self.label} · absent"
        if self.state == "unreadable":
            suffix = f": {self.detail}" if self.detail else ""
            return f"{self.label} · unreadable{suffix}"
        role = _SIDE_ROLES[self.label]
        return (
            f"{self.label} · {role} · {self.character_count:,} chars · "
            f"{self.byte_count:,} UTF-8 bytes · {self.line_count:,} lines · "
            f"SHA-256 {self.sha256}"
        )


@dataclass(frozen=True, slots=True)
class ConflictComparison:
    """Bounded display payload for all three conflict sides."""

    sides: tuple[ConflictSide, ConflictSide, ConflictSide]
    summary_text: str
    diff_text: str
    output_elided: bool


def _input_exceeds_diff_budget(side: ConflictSide) -> bool:
    return side.state == "readable" and (
        side.character_count > CONFLICT_DIFF_MAX_INPUT_CHARS
        or side.line_count > CONFLICT_DIFF_MAX_INPUT_LINES
    )


def _diff_section(base: ConflictSide, target: ConflictSide) -> list[str]:
    heading = f"{base.label} → {target.label}"
    if target.state == "absent":
        return [heading, f"{target.label} is absent; no textual diff is available."]
    if target.state == "unreadable":
        return [
            heading,
            f"{target.label} is unreadable; no textual diff is available.",
        ]
    assert base.text is not None
    assert target.text is not None
    lines = list(
        unified_diff(
            base.text.splitlines(),
            target.text.splitlines(),
            fromfile=base.label,
            tofile=target.label,
            lineterm="",
        )
    )
    if not lines:
        lines = ["No body changes."]
    return [heading, *lines]


def _bounded_output(lines: list[str]) -> tuple[str, bool]:
    retained: list[str] = []
    retained_chars = 0
    elided = False
    marker = "… comparison output elided at the bounded display limit."
    for line in lines:
        addition = len(line) + (1 if retained else 0)
        if (
            len(retained) >= CONFLICT_DIFF_MAX_OUTPUT_LINES
            or retained_chars + addition + len(marker) + 1
            > CONFLICT_DIFF_MAX_OUTPUT_CHARS
        ):
            elided = True
            break
        retained.append(line)
        retained_chars += addition
    if elided:
        retained.append(marker)
    return "\n".join(retained), elided


def build_conflict_comparison(
    base: ConflictSide,
    draft: ConflictSide,
    disk: ConflictSide,
) -> ConflictComparison:
    """Build two bounded unified comparisons around the editor baseline.

    Args:
        base: Body loaded into the editor or last saved successfully.
        draft: Exact body currently retained in the editor.
        disk: Latest readable disk body, or an explicit unavailable state.

    Returns:
        Summary and bounded Base-to-Draft/Base-to-Disk diff output.
    """
    sides = (base, draft, disk)
    summary_text = "\n".join(side.summary for side in sides)
    if any(_input_exceeds_diff_budget(side) for side in sides):
        diff_text = (
            "Base → Draft\nBase → Disk\n\n"
            "Diff omitted because one or more sides exceed the bounded "
            f"comparison input limit ({CONFLICT_DIFF_MAX_INPUT_CHARS:,} "
            f"characters or {CONFLICT_DIFF_MAX_INPUT_LINES:,} lines per side). "
            "Use the exact sizes and SHA-256 identities above before choosing a "
            "recovery action."
        )
        return ConflictComparison(sides, summary_text, diff_text, True)

    lines = [
        *_diff_section(base, draft),
        "",
        *_diff_section(base, disk),
    ]
    diff_text, output_elided = _bounded_output(lines)
    return ConflictComparison(sides, summary_text, diff_text, output_elided)
