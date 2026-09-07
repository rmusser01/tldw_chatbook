"""Headless one-shot prompt-improvement orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import replace
from hashlib import sha256
import hmac
import json
from time import monotonic
from typing import Any, Protocol

from loguru import logger

from tldw_chatbook.Chat.console_provider_gateway import (
    MAX_AUXILIARY_OUTPUT_TOKENS,
    AuxiliaryCompletionRequest,
    AuxiliaryCompletionResult,
)
from tldw_chatbook.Prompt_Management.prompt_artifact_models import (
    BlockArtifactDefinition,
    PromptBlock,
    PromptLane,
)
from tldw_chatbook.Prompt_Management.prompt_block_compiler import (
    compile_block_artifact,
    validate_xml_wrapper,
)
from tldw_chatbook.Prompt_Management.prompt_improvement_models import (
    PromptImprovementOutcome,
    PromptImprovementRequestSnapshot,
    fingerprint_block_definition,
    fingerprint_text,
)
from tldw_chatbook.Prompt_Management.prompt_improvement_prompts import (
    EmptyImprovementResponse,
    MalformedImprovementResponse,
    build_auxiliary_request,
    parse_recipe_envelope,
    parse_rewrite_envelope,
)
from tldw_chatbook.Prompt_Management.prompt_preservation import (
    preservation_violations,
)
from tldw_chatbook.Utils.token_counter import estimate_tokens


UNKNOWN_MODEL_CONTEXT_CAP_TOKENS = 32_768
_OUTPUT_ENVELOPE_ALLOWANCE_TOKENS = 1_024
_ADDITIONAL_CONTEXT_PREFIX = "additional-context"


class _AuxiliaryGateway(Protocol):
    async def complete_auxiliary(
        self, request: AuxiliaryCompletionRequest
    ) -> AuxiliaryCompletionResult: ...


TokenEstimator = Callable[[str, str, str], int]
LimitResolver = Callable[[str, str], int | None]
TelemetrySink = Callable[[dict[str, Any]], None]


_OUTCOME_MESSAGES: Mapping[str, str] = {
    "success": "Prompt improvement is ready for review.",
    "no_change": "The provider returned the original prompt unchanged.",
    "empty": "The provider returned no prompt improvement.",
    "unsupported": "This captured prompt cannot be improved in the selected mode.",
    "cancelled": "Prompt improvement was cancelled.",
    "provider_error": "The provider could not complete prompt improvement.",
    "malformed": "The provider returned an invalid prompt improvement.",
    "preservation_veto": "The result changed protected prompt material.",
    "context_limit": "The captured prompt exceeds the selected model context limit.",
    "stale": "The captured prompt improvement request is stale.",
}


def _known_context_limit(provider: str, model: str) -> int | None:
    try:
        from tldw_chatbook.model_capabilities import get_context_window

        return get_context_window(provider, model)
    except Exception:
        return None


def _known_output_limit(provider: str, model: str) -> int | None:
    try:
        from tldw_chatbook.model_capabilities import get_model_capabilities

        capabilities = get_model_capabilities().get_model_capabilities(provider, model)
    except Exception:
        return None
    for key in ("max_output_tokens", "output_token_limit"):
        value = capabilities.get(key)
        if type(value) is int and value > 0:
            return value
    return None


def _valid_limit(value: int | None) -> int | None:
    return value if type(value) is int and value > 0 else None


def _snapshot_fingerprint(snapshot: Any) -> str:
    payload = {
        "cursor_index": snapshot.cursor_index,
        "edit_serial": snapshot.edit_serial,
        "generation": snapshot.generation,
        "segments": [
            {
                "collapse_state": segment.collapse_state,
                "generated_boundary": segment.generated_boundary,
                "label": segment.label,
                "origin": segment.origin,
                "paste_block": segment.paste_block,
                "text": segment.text,
            }
            for segment in snapshot.segments
        ],
        "selection": (
            list(snapshot.selection)
            if isinstance(snapshot.selection, tuple)
            else snapshot.selection
        ),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return sha256(b"tldw.console.composer.snapshot.v1\0" + encoded).hexdigest()


def _projection_fingerprint(projection: Any) -> str:
    encoded = json.dumps(
        {
            "placeholder_ids": list(projection.placeholder_ids),
            "placeholder_nonce": projection.placeholder_nonce,
            "text": projection.text,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(b"tldw.console.composer.projection.v1\0" + encoded).hexdigest()


def _projection_matches_snapshot(snapshot: PromptImprovementRequestSnapshot) -> bool:
    placeholder_ids = snapshot.projection.placeholder_ids
    if type(placeholder_ids) is not tuple:
        return False
    parts: list[str] = []
    protected_index = 0
    for segment in snapshot.composer_snapshot.segments:
        if segment.origin != "inline_file":
            parts.append(segment.text)
            continue
        if protected_index >= len(placeholder_ids):
            return False
        parts.append(placeholder_ids[protected_index])
        protected_index += 1
    return (
        protected_index == len(placeholder_ids)
        and "".join(parts) == snapshot.projection.text
    )


def _validate_block_definition(
    definition: BlockArtifactDefinition,
    *,
    expected_kind: str,
) -> None:
    """Apply canonical structural and rendering validation to every block."""
    if (
        not isinstance(definition, BlockArtifactDefinition)
        or definition.kind != expected_kind
        or definition.schema_version != 2
    ):
        raise ValueError("Unexpected block artifact definition")
    for lane in definition.lanes:
        for block in lane.blocks:
            if block.syntax == "xml":
                validate_xml_wrapper(block.xml_tag, block.content)
    compile_block_artifact(definition)


def _captured_state_is_consistent(snapshot: PromptImprovementRequestSnapshot) -> bool:
    try:
        model = str(snapshot.resolution.model)
        checks = (
            snapshot.request_id == snapshot.projection.placeholder_nonce,
            hmac.compare_digest(
                snapshot.composer_snapshot.fingerprint,
                _snapshot_fingerprint(snapshot.composer_snapshot),
            ),
            hmac.compare_digest(
                snapshot.projection.fingerprint,
                _projection_fingerprint(snapshot.projection),
            ),
            _projection_matches_snapshot(snapshot),
            snapshot.provider_label == snapshot.resolution.provider,
            snapshot.model_label == model,
        )
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(checks):
        return False
    try:
        if snapshot.system_prompt is not None and not hmac.compare_digest(
            str(snapshot.system_fingerprint), fingerprint_text(snapshot.system_prompt)
        ):
            return False
        if snapshot.recipe_definition is not None:
            _validate_block_definition(
                snapshot.recipe_definition,
                expected_kind="block_recipe",
            )
        return not (
            snapshot.recipe_definition is not None
            and not hmac.compare_digest(
                str(snapshot.recipe_fingerprint),
                fingerprint_block_definition(snapshot.recipe_definition),
            )
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _reserved_additional_context_id(block_id: str) -> bool:
    normalized = block_id.casefold().replace("_", "-")
    return normalized == _ADDITIONAL_CONTEXT_PREFIX or normalized.startswith(
        f"{_ADDITIONAL_CONTEXT_PREFIX}-"
    )


def _supported_snapshot(snapshot: PromptImprovementRequestSnapshot) -> bool:
    if snapshot.mode in {"auto", "review"}:
        improvable = snapshot.projection.text
        for placeholder in snapshot.projection.placeholder_ids:
            improvable = improvable.replace(placeholder, "")
        return bool(improvable.strip())
    definition = snapshot.recipe_definition
    if (
        definition is None
        or definition.kind != "block_recipe"
        or definition.schema_version != 2
    ):
        return False
    return not any(
        _reserved_additional_context_id(block.id)
        for lane in definition.lanes
        for block in lane.blocks
    )


def _recipe_block_ids(definition: BlockArtifactDefinition) -> tuple[str, ...]:
    return tuple(block.id for lane in definition.lanes for block in lane.blocks)


def _merge_recipe(
    definition: BlockArtifactDefinition,
    fills: Mapping[str, str],
    additional_context: str,
) -> BlockArtifactDefinition:
    lanes = tuple(
        PromptLane(
            id=lane.id,
            blocks=tuple(
                replace(block, content=fills[block.id]) for block in lane.blocks
            ),
        )
        for lane in definition.lanes
    )
    if additional_context != "":
        system_lane, user_lane = lanes
        user_lane = replace(
            user_lane,
            blocks=user_lane.blocks
            + (
                PromptBlock(
                    id=_ADDITIONAL_CONTEXT_PREFIX,
                    title="Additional context",
                    syntax="markdown",
                    content=additional_context,
                ),
            ),
        )
        lanes = (system_lane, user_lane)
    merged = BlockArtifactDefinition(kind="block_prompt", schema_version=2, lanes=lanes)
    _validate_block_definition(merged, expected_kind="block_prompt")
    return merged


def _recipe_preservation_source(snapshot: PromptImprovementRequestSnapshot) -> str:
    parts = []
    if snapshot.system_prompt is not None:
        parts.append(snapshot.system_prompt)
    parts.append(snapshot.projection.text)
    if snapshot.recipe_definition is not None:
        parts.extend(
            block.content
            for lane in snapshot.recipe_definition.lanes
            for block in lane.blocks
        )
    return "\n".join(parts)


def _recipe_preservation_result(
    definition: BlockArtifactDefinition,
) -> str:
    return "\n".join(
        block.content for lane in definition.lanes for block in lane.blocks
    )


class PromptImprovementService:
    """Validate, budget, call, and parse one captured improvement request."""

    def __init__(
        self,
        *,
        gateway: _AuxiliaryGateway,
        token_estimator: TokenEstimator = estimate_tokens,
        context_limit_resolver: LimitResolver = _known_context_limit,
        output_limit_resolver: LimitResolver = _known_output_limit,
        telemetry_sink: TelemetrySink | None = None,
    ) -> None:
        self._gateway = gateway
        self._token_estimator = token_estimator
        self._context_limit_resolver = context_limit_resolver
        self._output_limit_resolver = output_limit_resolver
        self._telemetry_sink = telemetry_sink or self._log_telemetry

    @staticmethod
    def _log_telemetry(event: dict[str, Any]) -> None:
        logger.bind(**event).info("Prompt improvement completed")

    @staticmethod
    def _outcome(
        snapshot: PromptImprovementRequestSnapshot,
        kind: str,
        *,
        rewritten_prompt: str | None = None,
        filled_definition: BlockArtifactDefinition | None = None,
    ) -> PromptImprovementOutcome:
        return PromptImprovementOutcome(
            request_id=snapshot.request_id,
            kind=kind,  # type: ignore[arg-type]
            rewritten_prompt=rewritten_prompt,
            filled_definition=filled_definition,
            provider=snapshot.resolution.provider,
            model=str(snapshot.resolution.model),
            user_message=_OUTCOME_MESSAGES[kind],
        )

    def _emit(
        self,
        snapshot: PromptImprovementRequestSnapshot,
        outcome: PromptImprovementOutcome,
        *,
        started: float,
        input_bytes: int,
        output_bytes: int,
        input_tokens: int,
        requested_output_tokens: int,
    ) -> PromptImprovementOutcome:
        event = {
            "request_id": snapshot.request_id,
            "provider": snapshot.resolution.provider,
            "model": str(snapshot.resolution.model),
            "mode": snapshot.mode,
            "duration_ms": max(0, int((monotonic() - started) * 1_000)),
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "estimated_input_tokens": input_tokens,
            "requested_output_tokens": requested_output_tokens,
            "outcome": outcome.kind,
        }
        self._telemetry_sink(event)
        return outcome

    async def improve(
        self, snapshot: PromptImprovementRequestSnapshot
    ) -> PromptImprovementOutcome:
        """Run at most one auxiliary call for one immutable captured request."""
        if not isinstance(snapshot, PromptImprovementRequestSnapshot):
            raise TypeError("snapshot must be a PromptImprovementRequestSnapshot")
        started = monotonic()
        if not _captured_state_is_consistent(snapshot):
            return self._emit(
                snapshot,
                self._outcome(snapshot, "stale"),
                started=started,
                input_bytes=0,
                output_bytes=0,
                input_tokens=0,
                requested_output_tokens=0,
            )
        if not _supported_snapshot(snapshot):
            return self._emit(
                snapshot,
                self._outcome(snapshot, "unsupported"),
                started=started,
                input_bytes=0,
                output_bytes=0,
                input_tokens=0,
                requested_output_tokens=0,
            )

        provisional = build_auxiliary_request(snapshot, max_output_tokens=1)
        provider = snapshot.resolution.provider
        model = str(snapshot.resolution.model)
        message_texts = tuple(
            str(message["content"]) for message in provisional.messages
        )
        input_tokens = sum(
            self._token_estimator(text, model, provider) for text in message_texts
        )
        output_estimate = (
            self._token_estimator(message_texts[-1], model, provider)
            + _OUTPUT_ENVELOPE_ALLOWANCE_TOKENS
        )
        advertised_output = _valid_limit(self._output_limit_resolver(provider, model))
        output_allowance = min(
            output_estimate,
            advertised_output or MAX_AUXILIARY_OUTPUT_TOKENS,
            MAX_AUXILIARY_OUTPUT_TOKENS,
        )
        context_limit = (
            _valid_limit(self._context_limit_resolver(provider, model))
            or UNKNOWN_MODEL_CONTEXT_CAP_TOKENS
        )
        input_bytes = sum(len(text.encode("utf-8")) for text in message_texts)
        if input_tokens + output_allowance > context_limit:
            return self._emit(
                snapshot,
                self._outcome(snapshot, "context_limit"),
                started=started,
                input_bytes=input_bytes,
                output_bytes=0,
                input_tokens=input_tokens,
                requested_output_tokens=output_allowance,
            )

        request = build_auxiliary_request(snapshot, max_output_tokens=output_allowance)
        try:
            result = await self._gateway.complete_auxiliary(request, route=None)
        except asyncio.CancelledError:
            return self._emit(
                snapshot,
                self._outcome(snapshot, "cancelled"),
                started=started,
                input_bytes=input_bytes,
                output_bytes=0,
                input_tokens=input_tokens,
                requested_output_tokens=output_allowance,
            )
        except Exception:
            return self._emit(
                snapshot,
                self._outcome(snapshot, "provider_error"),
                started=started,
                input_bytes=input_bytes,
                output_bytes=0,
                input_tokens=input_tokens,
                requested_output_tokens=output_allowance,
            )

        if not isinstance(result, AuxiliaryCompletionResult):
            outcome = self._outcome(snapshot, "provider_error")
            output_bytes = 0
        elif not isinstance(result.text, str):
            outcome = self._outcome(snapshot, "malformed")
            output_bytes = 0
        else:
            output_bytes = len(result.text.encode("utf-8"))
            if result.provider != provider or result.model != model:
                outcome = self._outcome(snapshot, "stale")
            else:
                outcome = self._parse_result(snapshot, result.text)
        return self._emit(
            snapshot,
            outcome,
            started=started,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            input_tokens=input_tokens,
            requested_output_tokens=output_allowance,
        )

    def _parse_result(
        self, snapshot: PromptImprovementRequestSnapshot, text: str
    ) -> PromptImprovementOutcome:
        try:
            if snapshot.mode in {"auto", "review"}:
                rewritten = parse_rewrite_envelope(text)
                if rewritten == snapshot.projection.text:
                    return self._outcome(snapshot, "no_change")
                if preservation_violations(snapshot.projection.text, rewritten):
                    return self._outcome(
                        snapshot,
                        "preservation_veto",
                        rewritten_prompt=rewritten,
                    )
                return self._outcome(snapshot, "success", rewritten_prompt=rewritten)

            definition = snapshot.recipe_definition
            if definition is None:
                return self._outcome(snapshot, "unsupported")
            fills, additional_context = parse_recipe_envelope(
                text,
                expected_fingerprint=str(snapshot.recipe_fingerprint),
                expected_block_ids=_recipe_block_ids(definition),
            )
            merged = _merge_recipe(definition, fills, additional_context)
            preservation_result = _recipe_preservation_result(merged)
            if preservation_violations(
                _recipe_preservation_source(snapshot), preservation_result
            ):
                return self._outcome(snapshot, "preservation_veto")
            return self._outcome(snapshot, "success", filled_definition=merged)
        except EmptyImprovementResponse:
            return self._outcome(snapshot, "empty")
        except (MalformedImprovementResponse, TypeError, ValueError):
            return self._outcome(snapshot, "malformed")
