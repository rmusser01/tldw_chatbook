"""Read/write authoring helpers for the Settings "Internal Prompts" page.

Pure functions over CATALOG + config helpers. No Textual imports. Config
helpers are imported lazily (call-time) so the package stays off the
cold-start import chain (import-hygiene test).

Override storage: sparse ``[internal_prompts.<subsystem>.<key>]`` tables of
``{text, baseline}``. ``baseline`` is a fingerprint of the shipped default at
save time; the resolver ignores it, the UI uses it to flag "default changed".
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .catalog import CATALOG, PromptSpec
from .resolver import get_internal_prompt


@dataclass(frozen=True)
class OverrideState:
    customized: bool          # resolved text != shipped default (override OR legacy)
    default_changed: bool     # override table exists AND its baseline != current default hash
    has_override_table: bool  # a [internal_prompts.<sub>.<key>] table is present
    active_text: str          # currently resolved text (editor prefill)


def baseline_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def iter_specs_by_subsystem() -> list[tuple[str, list[PromptSpec]]]:
    order: list[str] = list(dict.fromkeys(s.subsystem for s in CATALOG.values()))
    grouped: dict[str, list[PromptSpec]] = {name: [] for name in order}
    for spec in CATALOG.values():
        grouped[spec.subsystem].append(spec)
    return [
        (name, sorted(grouped[name], key=lambda s: s.title)) for name in order
    ]


def _split(prompt_id: str) -> tuple[str, str]:
    subsystem, _, key = prompt_id.partition(".")
    return subsystem, key


def _override_table(prompt_id: str) -> dict | None:
    from tldw_chatbook.config import get_cli_setting  # lazy
    subsystem, key = _split(prompt_id)
    raw = get_cli_setting("internal_prompts." + subsystem, key, None)
    return raw if isinstance(raw, dict) else None


def override_state(prompt_id: str) -> OverrideState:
    spec = CATALOG[prompt_id]
    active = get_internal_prompt(prompt_id)
    customized = active != spec.default
    table = _override_table(prompt_id)
    has_table = table is not None
    default_changed = bool(
        has_table and table.get("baseline") != baseline_hash(spec.default)
    )
    return OverrideState(
        customized=customized,
        default_changed=default_changed,
        has_override_table=has_table,
        active_text=active,
    )


def save_override(prompt_id: str, text: str) -> bool:
    from tldw_chatbook.config import save_settings_to_cli_config  # lazy
    spec = CATALOG[prompt_id]
    subsystem, key = _split(prompt_id)
    return save_settings_to_cli_config(
        {
            "internal_prompts." + subsystem: {
                key: {"text": text, "baseline": baseline_hash(spec.default)}
            }
        }
    )


def _legacy_differs_from_shipped(spec: PromptSpec) -> tuple[str, str] | None:
    """Return (section, key) of the legacy config path IF the user's current
    value there differs from the shipped default (i.e. a real customization).
    None when no legacy path, or the value is absent/equal-to-shipped."""
    if not spec.legacy_config_path:
        return None
    from tldw_chatbook.config import (  # lazy
        get_cli_setting,
        DEFAULT_CONFIG_FROM_TOML,
    )
    section, _, key = spec.legacy_config_path.rpartition(".")
    if not section:
        return None
    current = get_cli_setting(section, key, None)
    if current is None:
        return None
    node: object = DEFAULT_CONFIG_FROM_TOML
    for part in spec.legacy_config_path.split("."):
        if not isinstance(node, dict) or part not in node:
            node = None
            break
        node = node[part]
    shipped = node
    if current == shipped:
        return None
    return section, key


def reset_override(prompt_id: str) -> bool:
    from tldw_chatbook.config import delete_settings_from_cli_config  # lazy
    spec = CATALOG[prompt_id]
    subsystem, key = _split(prompt_id)
    ok = delete_settings_from_cli_config("internal_prompts." + subsystem, [key])
    legacy = _legacy_differs_from_shipped(spec)
    if legacy is not None:
        section, legacy_key = legacy
        ok = delete_settings_from_cli_config(section, [legacy_key]) and ok
    return ok


def customized_count() -> int:
    return sum(1 for pid in CATALOG if override_state(pid).customized)
