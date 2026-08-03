"""Deterministic compatibility rendering for Console block artifacts."""

from __future__ import annotations

import re

from .prompt_artifact_models import BlockArtifactDefinition, PromptBlock, PromptLane


_XML_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]*$")


def validate_xml_wrapper(xml_tag: str | None, content: str) -> None:
    """Validate the wrapper tag without modifying user-provided content."""
    if not isinstance(xml_tag, str) or not _XML_NAME.fullmatch(xml_tag):
        raise ValueError(f"Invalid XML wrapper name: {xml_tag!r}")
    if not isinstance(content, str):
        raise ValueError(f"XML wrapper content must be text: {content!r}")

    collision = re.compile(rf"<\s*/?\s*{re.escape(xml_tag)}(?:\s+[^>]*)?\s*/?\s*>")
    if collision.search(content):
        raise ValueError(
            f"XML wrapper collision for {xml_tag!r} in content: {content!r}"
        )


def compile_block(block: PromptBlock) -> str:
    """Render one block to canonical compatibility text."""
    if block.content == "":
        return ""
    if block.syntax == "freeform":
        return block.content
    if block.syntax == "markdown":
        return f"# {block.title}\n\n{block.content}"
    validate_xml_wrapper(block.xml_tag, block.content)
    return f"<{block.xml_tag}>{block.content}</{block.xml_tag}>"


def compile_lane(lane: PromptLane) -> str:
    """Render the non-empty blocks in a lane with canonical separation."""
    return "\n\n".join(
        rendered for block in lane.blocks if (rendered := compile_block(block))
    )


def compile_block_artifact(definition: BlockArtifactDefinition) -> tuple[str, str]:
    """Compile a strict v2 definition into System and User compatibility text."""
    system_lane, user_lane = definition.lanes
    return compile_lane(system_lane), compile_lane(user_lane)
