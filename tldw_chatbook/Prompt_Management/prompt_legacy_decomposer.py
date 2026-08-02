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


def _in_fence(text: str, position: int) -> bool:
    return len(re.findall(r"(?m)^```[^\n]*$", text[:position])) % 2 == 1


def _xml_span(text: str, start: int) -> tuple[str, int] | None:
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
                return tag, token.end()
            if depth < 0:
                return None
        elif raw_token.rstrip().endswith("/>"):
            continue
        else:
            depth += 1
    return None


def _candidates(text: str) -> list[tuple[int, int | None, str, str]]:
    candidates: list[tuple[int, int | None, str, str]] = []
    for heading in _MARKDOWN_HEADING.finditer(text):
        if not _in_fence(text, heading.start()):
            candidates.append((heading.start(), heading.end(), "markdown", heading.group(1)))
    for opening in _XML_OPEN.finditer(text):
        if opening.start() and text[opening.start() - 1] != "\n":
            continue
        span = _xml_span(text, opening.start())
        if span is not None:
            tag, end = span
            candidates.append((opening.start(), end, "xml", tag))
    candidates.sort(key=lambda candidate: candidate[0])

    accepted: list[tuple[int, int | None, str, str]] = []
    covered_until = -1
    for candidate in candidates:
        start, end, syntax, title = candidate
        if start < covered_until:
            continue
        accepted.append(candidate)
        if syntax == "xml" and end is not None:
            covered_until = end
    return accepted


def _lane_blocks(lane_id: str, text: str) -> tuple[PromptBlock, ...]:
    blocks: list[PromptBlock] = []
    cursor = 0
    block_number = 1
    candidates = _candidates(text)
    for index, (start, marker_end, syntax, title) in enumerate(candidates):
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
            assert marker_end is not None
            opening_end = text.find(">", start) + 1
            content = text[opening_end : marker_end - len(title) - 3]
            blocks.append(
                PromptBlock(
                    id=f"legacy-{lane_id}-{block_number}",
                    title=title,
                    syntax="xml",
                    xml_tag=title,
                    content=content,
                )
            )
            cursor = marker_end
        else:
            assert marker_end is not None
            next_start = candidates[index + 1][0] if index + 1 < len(candidates) else len(text)
            blocks.append(
                PromptBlock(
                    id=f"legacy-{lane_id}-{block_number}",
                    title=title,
                    syntax="markdown",
                    content=text[marker_end:next_start],
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
