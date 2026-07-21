"""Declarative catalog of internal/system prompts.

Pure data. This module (and the per-subsystem prompt modules that call
``register``) must not import ``tldw_chatbook.config`` directly or
transitively — the resolver does config lookups lazily. Prompt ids are
public config API and frozen once shipped.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptSpec:
    """One internal prompt: identity, shipped default, and edit contract."""

    id: str
    subsystem: str
    title: str
    description: str
    used_in: str
    default: str
    required_placeholders: tuple[str, ...] = ()
    optional_placeholders: tuple[str, ...] = ()
    contract_note: str | None = None
    legacy_config_path: str | None = None
    applies: str = "live"


CATALOG: dict[str, PromptSpec] = {}


def register(spec: PromptSpec) -> PromptSpec:
    """Add a prompt spec to the global ``CATALOG``.

    Args:
        spec: The spec to register. Its ``id`` must be globally unique and
            must start with ``spec.subsystem + "."``.

    Returns:
        The registered spec, unchanged (convenient for module-level use).

    Raises:
        ValueError: If the id is already registered or does not match the
            spec's subsystem prefix.
    """
    if spec.id in CATALOG:
        raise ValueError(f"Duplicate internal prompt id: {spec.id!r}")
    if not spec.id.startswith(spec.subsystem + "."):
        raise ValueError(
            f"Prompt id {spec.id!r} must start with its subsystem {spec.subsystem!r} + '.'"
        )
    CATALOG[spec.id] = spec
    return spec
