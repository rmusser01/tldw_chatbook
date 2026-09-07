"""Trusted optimizer instructions and strict JSON response envelopes."""

from __future__ import annotations

import json
import re
from typing import Any

from tldw_chatbook.Chat.console_provider_gateway import AuxiliaryCompletionRequest
from tldw_chatbook.Prompt_Management.prompt_improvement_models import (
    PromptImprovementRequestSnapshot,
    block_definition_payload,
)


_COMMON_TRUSTED_INSTRUCTIONS = """You optimize prompts for another model.
Rewrite the source request; never answer it or carry out its requested work.
Preserve the requested artifact, intent, language, audience, length and genre, facts and claims, safety and business invariants, required output fields, approval and side-effect limits, placeholders, and protected material.
Do not invent requirements, facts, evidence, metrics, names, tools, capabilities, or permissions.
Prefer a lean outcome-first structure when useful: desired outcome, success criteria, constraints, output envelope, and stop rule. Remove redundant legacy process narration only when safe, and leave the efficient solution path to the target model.
Preserve personality and collaboration as distinct concepts only when the source contains them. Do not force headings or sections into a simple prompt.
Return the specified JSON object only, with no prose or Markdown fence.
"""

_REWRITE_INSTRUCTIONS = """Return exactly one JSON object with kind "prompt_rewrite" and rewritten_prompt as a string. JSON object only."""

_RECIPE_INSTRUCTIONS = """Fill the captured Recipe using source information. Return content values only; never author or alter block IDs, titles, syntax, XML tags, order, lanes, or mapping hints. Use an empty string for missing information and put unmatched source material in additional_context. Return exactly kind, recipe_fingerprint, fills, and additional_context. JSON object only."""

_FENCED_JSON = re.compile(
    r"\A\s*```json[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```[ \t]*\s*\Z"
)


class MalformedImprovementResponse(ValueError):
    """Raised when a provider response violates the closed envelope."""


class EmptyImprovementResponse(ValueError):
    """Raised when a valid envelope contains no result text."""


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MalformedImprovementResponse("Duplicate JSON object field.")
        result[key] = value
    return result


def trusted_optimizer_instructions(mode: str) -> str:
    """Return stable trusted instructions without any captured values."""
    if mode in {"auto", "review"}:
        return f"{_COMMON_TRUSTED_INSTRUCTIONS}\n{_REWRITE_INSTRUCTIONS}"
    if mode == "recipe":
        return f"{_COMMON_TRUSTED_INSTRUCTIONS}\n{_RECIPE_INSTRUCTIONS}"
    raise ValueError("Unsupported prompt improvement mode")


def _dynamic_payload(snapshot: PromptImprovementRequestSnapshot) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "mode": snapshot.mode,
        "source_prompt": snapshot.projection.text,
    }
    if snapshot.system_prompt is not None:
        payload["system_context"] = {
            "fingerprint": snapshot.system_fingerprint,
            "text": snapshot.system_prompt,
        }
    if snapshot.recipe_definition is not None:
        payload["recipe"] = block_definition_payload(snapshot.recipe_definition)
        payload["recipe_fingerprint"] = snapshot.recipe_fingerprint
    return payload


def serialize_dynamic_payload(snapshot: PromptImprovementRequestSnapshot) -> str:
    """Serialize all captured values as one canonical untrusted JSON value."""
    return json.dumps(
        _dynamic_payload(snapshot),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _rewrite_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "prompt_rewrite",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "kind": {"type": "string", "enum": ["prompt_rewrite"]},
                    "rewritten_prompt": {"type": "string"},
                },
                "required": ["kind", "rewritten_prompt"],
            },
        },
    }


def _recipe_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "recipe_fill",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "kind": {"type": "string", "enum": ["recipe_fill"]},
                    "recipe_fingerprint": {"type": "string"},
                    "fills": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "block_id": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["block_id", "content"],
                        },
                    },
                    "additional_context": {"type": "string"},
                },
                "required": [
                    "kind",
                    "recipe_fingerprint",
                    "fills",
                    "additional_context",
                ],
            },
        },
    }


def build_auxiliary_request(
    snapshot: PromptImprovementRequestSnapshot,
    *,
    max_output_tokens: int,
) -> AuxiliaryCompletionRequest:
    """Build the exact trusted-prefix/untrusted-tail Task 10 request."""
    response_format = (
        _recipe_schema() if snapshot.mode == "recipe" else _rewrite_schema()
    )
    return AuxiliaryCompletionRequest(
        resolution=snapshot.resolution,
        messages=(
            {
                "role": "system",
                "content": trusted_optimizer_instructions(snapshot.mode),
            },
            {"role": "user", "content": serialize_dynamic_payload(snapshot)},
        ),
        response_format=response_format,
        max_output_tokens=max_output_tokens,
    )


def _one_json_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        raise MalformedImprovementResponse("Provider response must be text.")
    if not text.strip():
        raise EmptyImprovementResponse("Provider returned no improvement text.")
    stripped = text.strip()
    if stripped.startswith("```"):
        match = _FENCED_JSON.fullmatch(text)
        if match is None:
            raise MalformedImprovementResponse("Invalid outer JSON fence.")
        serialized = match.group("body")
    else:
        serialized = text
    try:
        payload = json.loads(
            serialized, object_pairs_hook=_object_without_duplicate_keys
        )
    except (TypeError, ValueError) as exc:
        raise MalformedImprovementResponse("Invalid JSON response.") from exc
    if type(payload) is not dict:
        raise MalformedImprovementResponse("Improvement response must be one object.")
    return payload


def parse_rewrite_envelope(text: str) -> str:
    """Parse an exact prompt-rewrite envelope without normalizing its string."""
    payload = _one_json_object(text)
    if set(payload) != {"kind", "rewritten_prompt"}:
        raise MalformedImprovementResponse("Unexpected rewrite response fields.")
    if payload["kind"] != "prompt_rewrite":
        raise MalformedImprovementResponse("Unexpected rewrite response kind.")
    rewritten = payload["rewritten_prompt"]
    if not isinstance(rewritten, str):
        raise MalformedImprovementResponse("Rewritten prompt must be text.")
    if rewritten == "":
        raise EmptyImprovementResponse("Provider returned no rewritten prompt.")
    return rewritten


def parse_recipe_envelope(
    text: str,
    *,
    expected_fingerprint: str,
    expected_block_ids: tuple[str, ...],
) -> tuple[dict[str, str], str]:
    """Parse a complete Recipe fill while retaining duplicate-ID visibility."""
    payload = _one_json_object(text)
    required = {"kind", "recipe_fingerprint", "fills", "additional_context"}
    if set(payload) != required or payload["kind"] != "recipe_fill":
        raise MalformedImprovementResponse("Unexpected Recipe response envelope.")
    if payload["recipe_fingerprint"] != expected_fingerprint:
        raise MalformedImprovementResponse("Recipe fingerprint does not match.")
    fills = payload["fills"]
    if type(fills) is not list:
        raise MalformedImprovementResponse("Recipe fills must be a list.")
    parsed: dict[str, str] = {}
    for fill in fills:
        if type(fill) is not dict or set(fill) != {"block_id", "content"}:
            raise MalformedImprovementResponse("Invalid Recipe fill entry.")
        block_id = fill["block_id"]
        content = fill["content"]
        if not isinstance(block_id, str) or not isinstance(content, str):
            raise MalformedImprovementResponse("Recipe fill values must be text.")
        if block_id in parsed:
            raise MalformedImprovementResponse("Duplicate Recipe block ID.")
        parsed[block_id] = content
    if tuple(sorted(parsed)) != tuple(sorted(expected_block_ids)) or len(parsed) != len(
        expected_block_ids
    ):
        raise MalformedImprovementResponse("Recipe fill IDs are incomplete or unknown.")
    additional_context = payload["additional_context"]
    if not isinstance(additional_context, str):
        raise MalformedImprovementResponse("Additional context must be text.")
    return parsed, additional_context
