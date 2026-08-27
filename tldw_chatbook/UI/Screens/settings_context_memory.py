"""Settings contracts for global Console memory and model context capacity."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from ...Chat.console_context_policy import (
    CompactionFailureBehavior,
    ConsoleContextPolicyDefaults,
    ContextBudgetMode,
    ContextCarryForwardMode,
    ContextCompactionMode,
    ContextCompactionRepresentation,
    ContextPolicyError,
    application_context_policy_defaults,
    context_policy_overrides_from_console_config,
    merge_context_policy,
)
from ...model_capabilities import ModelCapabilities
from ...config import coerce_bool_setting
from ...Chat.thinking_blocks import (
    ThinkingHistoryPolicy,
    normalize_thinking_history_policy,
)
from ...Utils.token_counter import get_table_model_token_limit


SUMMARY_PROMPT_ID = "console.rewind_summarize"

CONTEXT_MEMORY_CONFIG_KEYS = (
    "conversation_budget_mode",
    "conversation_budget_tokens",
    "compaction_mode",
    "compaction_representation",
    "compaction_trigger_ratio",
    "compaction_target_ratio",
    "compaction_summary_max_tokens",
    "compaction_failure_behavior",
    "compaction_carry_forward_mode",
)


def load_show_model_thinking(
    console_config: Mapping[str, object] | None,
) -> bool:
    """Resolve the device-local presentation toggle, defaulting safely on."""

    raw = (console_config or {}).get("show_model_thinking", True)
    if type(raw) is not bool:
        return True
    return coerce_bool_setting(raw, True)


def load_thinking_history_policy_default(
    console_config: Mapping[str, object] | None,
) -> ThinkingHistoryPolicy:
    """Resolve the optional replay policy copied into new conversations."""

    return normalize_thinking_history_policy(
        (console_config or {}).get("thinking_history_policy_default")
    )


@dataclass(frozen=True, slots=True)
class SettingsContextMemoryValues:
    """Concrete values rendered by Settings > Console Behavior."""

    conversation_budget_mode: str
    conversation_budget_tokens: int | str
    compaction_mode: str
    compaction_representation: str
    compaction_trigger_ratio: float
    compaction_target_ratio: float
    compaction_summary_max_tokens: int
    compaction_failure_behavior: str
    compaction_carry_forward_mode: str

    def to_mapping(self) -> dict[str, object]:
        return {key: getattr(self, key) for key in CONTEXT_MEMORY_CONFIG_KEYS}


@dataclass(frozen=True, slots=True)
class ModelContextWindowState:
    """Detected and configured context-window provenance for one model."""

    effective_tokens: int | None
    detected_tokens: int | None
    configured_override_tokens: int | None

    @property
    def has_configured_override(self) -> bool:
        return self.configured_override_tokens is not None


def load_context_memory_values(
    console_config: Mapping[str, object] | None,
) -> SettingsContextMemoryValues:
    """Resolve saved global overrides over ADR-052 application defaults.

    Invalid hand-edited values never prevent Settings from mounting. They fall
    back as one unit to the documented application defaults and can then be
    repaired through the validated form.
    """

    try:
        policy = merge_context_policy(
            global_overrides=context_policy_overrides_from_console_config(
                console_config
            )
        )
    except ContextPolicyError:
        policy = application_context_policy_defaults()
    return _values_from_policy(policy)


def normalize_context_memory_values(
    values: Mapping[str, object],
) -> SettingsContextMemoryValues:
    """Validate staged Settings values through the canonical policy contract."""

    budget_mode = _enum_value(
        values.get("conversation_budget_mode"),
        ContextBudgetMode,
        "Budget strategy",
    )
    custom_budget = _optional_positive_int(
        values.get("conversation_budget_tokens"),
        "Custom conversation budget",
    )
    if budget_mode is ContextBudgetMode.CUSTOM and custom_budget is None:
        raise ValueError("Custom conversation budget requires a positive token value.")

    compaction_mode = _enum_value(
        values.get("compaction_mode"),
        ContextCompactionMode,
        "Compaction mode",
    )
    compaction_representation = _enum_value(
        values.get("compaction_representation"),
        ContextCompactionRepresentation,
        "Compaction representation",
    )
    failure_behavior = _enum_value(
        values.get("compaction_failure_behavior"),
        CompactionFailureBehavior,
        "Failure behavior",
    )
    carry_forward_mode = _enum_value(
        values.get("compaction_carry_forward_mode"),
        ContextCarryForwardMode,
        "Carry-forward mode",
    )
    trigger = _ratio(values.get("compaction_trigger_ratio"), "Trigger")
    target = _ratio(values.get("compaction_target_ratio"), "Target")
    summary_max = _positive_int(
        values.get("compaction_summary_max_tokens"),
        "Summary max tokens",
    )

    try:
        policy = ConsoleContextPolicyDefaults(
            budget_mode=budget_mode,
            custom_budget_tokens=custom_budget,
            compaction_mode=compaction_mode,
            compaction_representation=compaction_representation,
            trigger_ratio=trigger,
            target_ratio=target,
            summary_max_tokens=summary_max,
            failure_behavior=failure_behavior,
            carry_forward_mode=carry_forward_mode,
        )
    except ContextPolicyError as exc:
        raise ValueError(_friendly_policy_error(str(exc))) from exc
    return _values_from_policy(policy)


def format_ratio_percent(value: object) -> str:
    """Format a stored 0..1 ratio as an editable percentage."""

    try:
        percent = float(value) * 100
    except (TypeError, ValueError):
        return ""
    return f"{percent:g}"


def ratio_from_percent(value: object) -> float | str:
    """Convert a percentage input to a ratio, retaining invalid draft text."""

    text = str(value or "").strip()
    try:
        percent = float(text)
    except (TypeError, ValueError):
        return text
    if not 0 < percent < 100:
        return text
    return percent / 100


def resolve_model_context_window(
    app_config: Mapping[str, object] | None,
    provider: str,
    model: str,
) -> int | None:
    """Read capacity from TASK-320's model-capability authority."""

    return model_context_window_state(app_config, provider, model).effective_tokens


def model_context_window_state(
    app_config: Mapping[str, object] | None,
    provider: str,
    model: str,
) -> ModelContextWindowState:
    """Resolve configured override, detected value, and effective value."""

    model_id = str(model or "").strip()
    if not model_id:
        return ModelContextWindowState(None, None, None)
    section = _model_capabilities_section(app_config)
    models = section.get("models")
    configured_override = None
    if isinstance(models, Mapping):
        entry = models.get(model_id)
        if isinstance(entry, Mapping):
            configured_override = _valid_positive_int(entry.get("context_window"))

    detected_section = deepcopy(section)
    detected_models = detected_section.get("models")
    if isinstance(detected_models, dict):
        detected_entry = detected_models.get(model_id)
        if isinstance(detected_entry, Mapping):
            next_entry = dict(detected_entry)
            next_entry.pop("context_window", None)
            if next_entry:
                detected_models[model_id] = next_entry
            else:
                detected_models.pop(model_id, None)
    detected = _capability_or_table_window(detected_section, provider, model_id)
    return ModelContextWindowState(
        configured_override or detected,
        detected,
        configured_override,
    )


def model_context_window_save_entry(
    app_config: Mapping[str, object] | None,
    provider: str,
    model: str,
    context_window: object,
) -> dict[str, Any]:
    """Build one direct capability entry without dropping other capabilities."""

    model_id = str(model or "").strip()
    if not model_id:
        raise ValueError("Model is required before saving its context window.")
    window = _positive_int(context_window, "Model context window")
    section = _model_capabilities_section(app_config)
    capabilities = deepcopy(
        ModelCapabilities(section).get_model_capabilities(provider, model_id)
    )
    capabilities["context_window"] = window
    return capabilities


def model_context_window_reset_entry(
    app_config: Mapping[str, object] | None,
    model: str,
) -> dict[str, Any] | None:
    """Return the existing direct entry without its context-window override."""

    section = _model_capabilities_section(app_config)
    models = section.get("models")
    if not isinstance(models, Mapping):
        return None
    entry = models.get(str(model or "").strip())
    if not isinstance(entry, Mapping):
        return None
    next_entry = deepcopy(dict(entry))
    next_entry.pop("context_window", None)
    return next_entry or None


def _model_capabilities_section(
    app_config: Mapping[str, object] | None,
) -> dict[str, Any]:
    raw = (app_config or {}).get("model_capabilities", {})
    return deepcopy(dict(raw)) if isinstance(raw, Mapping) else {}


def _capability_or_table_window(
    section: Mapping[str, object], provider: str, model: str
) -> int | None:
    value = ModelCapabilities(dict(section)).get_context_window(provider, model)
    valid = _valid_positive_int(value)
    return valid or get_table_model_token_limit(model, provider)


def _valid_positive_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _values_from_policy(
    policy: ConsoleContextPolicyDefaults,
) -> SettingsContextMemoryValues:
    return SettingsContextMemoryValues(
        conversation_budget_mode=policy.budget_mode.value,
        conversation_budget_tokens=policy.custom_budget_tokens or "",
        compaction_mode=policy.compaction_mode.value,
        compaction_representation=policy.compaction_representation.value,
        compaction_trigger_ratio=policy.trigger_ratio,
        compaction_target_ratio=policy.target_ratio,
        compaction_summary_max_tokens=policy.summary_max_tokens,
        compaction_failure_behavior=policy.failure_behavior.value,
        compaction_carry_forward_mode=policy.carry_forward_mode.value,
    )


def _enum_value(value: object, enum_type, label: str):
    try:
        return enum_type(str(value or "").strip().lower())
    except ValueError as exc:
        choices = ", ".join(item.value for item in enum_type)
        raise ValueError(f"{label} must be one of: {choices}.") from exc


def _positive_int(value: object, label: str) -> int:
    parsed = _optional_positive_int(value, label)
    if parsed is None:
        raise ValueError(f"{label} must be a positive whole number.")
    return parsed


def _optional_positive_int(value: object, label: str) -> int | None:
    text = str(value if value is not None else "").strip()
    if not text:
        return None
    if not text.isdecimal() or int(text) <= 0:
        raise ValueError(f"{label} must be a positive whole number.")
    return int(text)


def _ratio(value: object, label: str) -> float:
    try:
        ratio = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a percentage between 1 and 95.") from exc
    if not 0 < ratio < 1:
        raise ValueError(f"{label} must be a percentage between 1 and 95.")
    return ratio


def _friendly_policy_error(message: str) -> str:
    replacements = {
        "trigger_ratio must be no greater than 0.95.": (
            "Compaction trigger must be no greater than 95%."
        ),
        "target_ratio must be lower than trigger_ratio.": (
            "Compact-toward target must be lower than the trigger."
        ),
        "trigger_ratio and target_ratio must differ by at least 0.15.": (
            "Trigger and compact-toward target must differ by at least 15 percentage points."
        ),
    }
    return replacements.get(message, message)
