"""Validated configuration boundary for device-local reasoning replay."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

REASONING_HISTORY_ENV = "TLDW_CONSOLE_REASONING_HISTORY"
REASONING_HISTORY_OVERRIDES_ENV = "TLDW_CONSOLE_REASONING_HISTORY_OVERRIDES"
REASONING_NATIVE_TOOL_OVERRIDES_ENV = (
    "TLDW_CONSOLE_REASONING_NATIVE_TOOL_OVERRIDES"
)
REASONING_HISTORY_MODES = frozenset({"auto", "current", "all", "off"})


class ConsoleReasoningConfig(BaseModel):
    """Strict effective configuration for local reasoning replay.

    Attributes:
        replay_thinking: Legacy global replay preference.
        reasoning_history: Default local replay mode for Conversation Auto.
        reasoning_history_overrides: Replay modes keyed by normalized target digest.
        reasoning_native_tool_overrides: Native-tool support keyed by target digest.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    replay_thinking: bool = True
    reasoning_history: str = "auto"
    reasoning_history_overrides: dict[str, str] = Field(default_factory=dict)
    reasoning_native_tool_overrides: dict[str, bool] = Field(default_factory=dict)

    @field_validator("reasoning_history")
    @classmethod
    def _validate_mode(cls, value: str) -> str:
        if value not in REASONING_HISTORY_MODES:
            raise ValueError("unsupported reasoning history mode")
        return value

    @field_validator("reasoning_history_overrides")
    @classmethod
    def _validate_modes(cls, values: dict[str, str]) -> dict[str, str]:
        if any(value not in REASONING_HISTORY_MODES for value in values.values()):
            raise ValueError("unsupported reasoning history override")
        return values


@dataclass(frozen=True, slots=True)
class ConsoleReasoningConfigResolution:
    """Validated effective settings and value-free fallback diagnostics.

    Attributes:
        settings: Strict effective reasoning configuration.
        diagnostics: Safe messages naming fields that used fallback defaults.
    """

    settings: ConsoleReasoningConfig
    diagnostics: tuple[str, ...] = ()


def migrate_legacy_reasoning_history(
    config: MutableMapping[str, object],
) -> None:
    """Preserve an explicit legacy replay opt-out before defaults are merged.

    Args:
        config: Raw user configuration, before programmatic defaults are merged.
    """

    console = config.get("console")
    if not isinstance(console, MutableMapping):
        return
    if (
        "reasoning_history" not in console
        and console.get("replay_thinking") is False
    ):
        console["reasoning_history"] = "off"


def _environment_json_value(raw: str) -> object:
    """Decode one non-empty JSON environment value for model validation."""

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def resolve_console_reasoning_config(
    console: object,
    *,
    environ: Mapping[str, str] | None = None,
) -> ConsoleReasoningConfigResolution:
    """Resolve environment-first replay settings through the strict model.

    Args:
        console: Merged Console configuration mapping.
        environ: Optional environment mapping. Defaults to ``os.environ``.

    Returns:
        Validated settings plus sanitized diagnostics for rejected fields.
    """

    values = console if isinstance(console, Mapping) else {}
    environment = os.environ if environ is None else environ
    candidate: dict[str, object] = {
        "replay_thinking": values.get("replay_thinking", True),
        "reasoning_history": values.get("reasoning_history", "auto"),
        "reasoning_history_overrides": values.get(
            "reasoning_history_overrides", {}
        ),
        "reasoning_native_tool_overrides": values.get(
            "reasoning_native_tool_overrides", {}
        ),
    }
    environment_fields = {
        "reasoning_history": REASONING_HISTORY_ENV,
        "reasoning_history_overrides": REASONING_HISTORY_OVERRIDES_ENV,
        "reasoning_native_tool_overrides": REASONING_NATIVE_TOOL_OVERRIDES_ENV,
    }
    for field_name, environment_name in environment_fields.items():
        override = environment.get(environment_name)
        if override in (None, ""):
            continue
        candidate[field_name] = (
            override
            if field_name == "reasoning_history"
            else _environment_json_value(override)
        )

    try:
        settings = ConsoleReasoningConfig.model_validate(candidate)
        return ConsoleReasoningConfigResolution(settings)
    except ValidationError as exc:
        invalid_fields = {
            str(error["loc"][0])
            for error in exc.errors(
                include_url=False,
                include_context=False,
                include_input=False,
            )
            if error.get("loc")
        }

    safe_defaults: dict[str, object] = {
        "replay_thinking": True,
        "reasoning_history": "auto",
        "reasoning_history_overrides": {},
        "reasoning_native_tool_overrides": {},
    }
    for field_name in invalid_fields:
        candidate[field_name] = safe_defaults[field_name]
    settings = ConsoleReasoningConfig.model_validate(candidate)
    diagnostics = tuple(
        f"console.{field_name} is invalid; using its safe default"
        for field_name in sorted(invalid_fields)
    )
    return ConsoleReasoningConfigResolution(settings, diagnostics)
