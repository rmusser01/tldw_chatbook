"""Conservative decomposition of legacy compatibility text into editable blocks."""

from __future__ import annotations

from hashlib import sha256
import re

from .prompt_artifact_models import (
    BlockArtifactDefinition,
    LegacyDecomposition,
    LegacyLaneOrigin,
    PromptBlock,
    PromptLane,
)


_MARKDOWN_HEADING = re.compile(r"(?m)^# ([^\n]+)\n\n")
_XML_OPEN = re.compile(r"<([A-Za-z_][A-Za-z0-9_.:-]*)>")
_XML_TOKEN = re.compile(r"</?([A-Za-z_][A-Za-z0-9_.:-]*)\s*/?>")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})([^\n]*)$")


def _in_fence(text: str, position: int) -> bool:
    active: tuple[str, int] | None = None
    for line in text[:position].splitlines():
        match = _FENCE.fullmatch(line)
        if match is None:
            continue
        marker, suffix = match.groups()
        marker_character = marker[0]
        if active is None:
            if marker_character == "`" and "`" in suffix:
                continue
            active = (marker_character, len(marker))
            continue
        active_character, active_length = active
        if (
            marker_character == active_character
            and len(marker) >= active_length
            and not suffix.strip()
        ):
            active = None
    return active is not None


def _xml_span(text: str, start: int) -> tuple[str, int, int, int] | None:
    opening = _XML_OPEN.match(text, start)
    if opening is None or _in_fence(text, start):
        return None
    tag = opening.group(1)
    depth = 0
    for token in _XML_TOKEN.finditer(text, start):
        if token.start() != start and _in_fence(text, token.start()):
            continue
        if token.group(1) != tag:
            continue
        raw_token = token.group(0)
        if raw_token.startswith("</"):
            depth -= 1
            if depth == 0:
                return tag, opening.end(), token.start(), token.end()
            if depth < 0:
                return None
        elif raw_token.rstrip().endswith("/>"):
            continue
        else:
            depth += 1
    return None


def _candidates(text: str) -> list[tuple[int, int, str, str, int | None, int | None]]:
    candidates: list[tuple[int, int, str, str, int | None, int | None]] = []
    for heading in _MARKDOWN_HEADING.finditer(text):
        if not _in_fence(text, heading.start()):
            candidates.append(
                (heading.start(), heading.end(), "markdown", heading.group(1), None, None)
            )
    for opening in _XML_OPEN.finditer(text):
        if opening.start() and text[opening.start() - 1] != "\n":
            continue
        span = _xml_span(text, opening.start())
        if span is not None:
            tag, content_start, content_end, end = span
            candidates.append(
                (opening.start(), end, "xml", tag, content_start, content_end)
            )
    candidates.sort(key=lambda candidate: candidate[0])

    accepted: list[tuple[int, int, str, str, int | None, int | None]] = []
    covered_until = -1
    for candidate in candidates:
        start, end, syntax, _title, _content_start, _content_end = candidate
        if start < covered_until:
            continue
        accepted.append(candidate)
        if syntax == "xml":
            covered_until = end
    return accepted


def _lane_blocks(lane_id: str, text: str) -> tuple[PromptBlock, ...]:
    blocks: list[PromptBlock] = []
    cursor = 0
    block_number = 1
    candidates = _candidates(text)
    for index, (start, end, syntax, title, content_start, content_end) in enumerate(
        candidates
    ):
        if start > cursor:
            blocks.append(
                PromptBlock(
                    id=f"legacy-{lane_id}-{block_number}",
                    title="Legacy text",
                    syntax="freeform",
                    content=text[cursor:start],
                )
            )
            block_number += 1
        if syntax == "xml":
            assert content_start is not None and content_end is not None
            blocks.append(
                PromptBlock(
                    id=f"legacy-{lane_id}-{block_number}",
                    title=title,
                    syntax="xml",
                    xml_tag=title,
                    content=text[content_start:content_end],
                )
            )
            cursor = end
        else:
            next_start = candidates[index + 1][0] if index + 1 < len(candidates) else len(text)
            blocks.append(
                PromptBlock(
                    id=f"legacy-{lane_id}-{block_number}",
                    title=title,
                    syntax="markdown",
                    content=text[end:next_start],
                )
            )
            cursor = next_start
        block_number += 1
    if cursor < len(text):
        blocks.append(
            PromptBlock(
                id=f"legacy-{lane_id}-{block_number}",
                title="Legacy text",
                syntax="freeform",
                content=text[cursor:],
            )
        )
    return tuple(blocks)


def _origin(text: str) -> LegacyLaneOrigin:
    return LegacyLaneOrigin(text=text, fingerprint=sha256(text.encode("utf-8")).hexdigest())


def decompose_legacy_lanes(system_prompt: str, user_prompt: str) -> LegacyDecomposition:
    """Create an editable view while retaining each legacy lane's exact origin."""
    system_text = str(system_prompt)
    user_text = str(user_prompt)
    definition = BlockArtifactDefinition(
        kind="block_prompt",
        schema_version=2,
        lanes=(
            PromptLane(id="system", blocks=_lane_blocks("system", system_text)),
            PromptLane(id="user", blocks=_lane_blocks("user", user_text)),
        ),
    )
    return LegacyDecomposition(
        definition=definition,
        system_origin=_origin(system_text),
        user_origin=_origin(user_text),
    )
