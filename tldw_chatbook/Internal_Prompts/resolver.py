"""Resolution + rendering for internal prompts.

Precedence: user override table -> customized legacy config key -> shipped
default. Never raises for user-caused problems (bad override text falls back
to the default with a once-per-prompt warning). Unknown prompt ids raise
KeyError — that is a programmer error the test suite catches.

Config helpers are imported lazily to keep this package off the cold-start
import chain and out of config.py import cycles.
"""

from __future__ import annotations

import re

from loguru import logger

from .catalog import CATALOG, PromptSpec

_warned_ids: set[str] = set()


def safe_substitute(text: str, **values: object) -> str:
    """Replace only the exact ``{name}`` tokens named in ``values``.

    Single-pass: tokens introduced by substituted values are NOT re-expanded,
    and all other braces (JSON examples, Ollama ``{{ .Prompt }}`` cruft,
    stray user typos) pass through untouched. Cannot raise.

    Args:
        text: Template text containing ``{name}`` tokens.
        **values: Token values; each is substituted as ``str(value)``. Tokens
            not named here are left untouched (a full token includes its
            closing brace, so overlapping names like ``query``/``query_list``
            cannot corrupt each other).

    Returns:
        The text with the named tokens substituted.
    """
    if not values:
        return text
    pattern = re.compile(
        "|".join(re.escape("{" + name + "}") for name in values)
    )
    return pattern.sub(lambda m: str(values[m.group(0)[1:-1]]), text)


def get_internal_prompt(prompt_id: str) -> str:
    """Resolve the internal prompt, prioritizing override → legacy → default.

    Never raises for user-caused problems (bad override text falls back to
    default with a once-per-prompt warning).

    Args:
        prompt_id: Prompt identifier (e.g., "chat.system_prompt").

    Returns:
        Resolved prompt text with placeholders intact.

    Raises:
        KeyError: If prompt_id is not registered in CATALOG.
    """
    spec = CATALOG[prompt_id]

    raw_override = _config_value("internal_prompts." + prompt_id)
    override = _extract_text(raw_override)
    if override is not None:
        if _has_required_placeholders(override, spec):
            return override
        _warn_once(
            prompt_id,
            f"override for {prompt_id} is missing a required placeholder; "
            "falling back",
        )
    elif raw_override is not None and not isinstance(raw_override, (str, dict)):
        # A present override that is neither text nor a {text, ...} table
        # (e.g. a hand-edited int or array) is silently unusable; surface it
        # once so the misconfiguration is visible instead of vanishing.
        _warn_once(
            prompt_id + ":type",
            f"override for {prompt_id} is not text or a table "
            f"(got {type(raw_override).__name__}); ignoring it and "
            "falling back",
        )

    if spec.legacy_config_path:
        legacy = _extract_text(_config_value(spec.legacy_config_path))
        if legacy is not None and legacy != _shipped_default_for(
            spec.legacy_config_path
        ):
            if _has_required_placeholders(legacy, spec):
                return legacy
            _warn_once(
                prompt_id,
                f"legacy value at [{spec.legacy_config_path}] for {prompt_id} "
                "is missing a required placeholder; falling back",
            )

    return spec.default


def render_internal_prompt(prompt_id: str, **values: object) -> str:
    """Resolve and substitute placeholders in an internal prompt.

    Args:
        prompt_id: Prompt identifier (e.g., "chat.system_prompt").
        **values: Placeholder values, e.g., name="Ada", context="...".

    Returns:
        Prompt text with placeholders replaced.

    Raises:
        KeyError: If prompt_id is not registered in CATALOG.
    """
    spec = CATALOG[prompt_id]
    rendered = safe_substitute(get_internal_prompt(prompt_id), **values)
    missing = [
        name
        for name in spec.required_placeholders
        if name not in values and "{" + name + "}" in rendered
    ]
    if missing:
        _warn_once(
            prompt_id + ":render",
            f"render_internal_prompt({prompt_id!r}) called without required "
            f"placeholder value(s) {missing}; tokens left unsubstituted",
        )
    return rendered


def _has_required_placeholders(text: str, spec: PromptSpec) -> bool:
    return all("{" + name + "}" in text for name in spec.required_placeholders)


def _extract_text(raw: object) -> str | None:
    """Normalize a config value: {text, baseline} table or plain string."""
    if isinstance(raw, dict):
        raw = raw.get("text")
    if isinstance(raw, str) and raw.strip():
        return raw
    return None


def _config_value(dotted_path: str) -> object:
    from tldw_chatbook.config import get_cli_setting  # lazy on purpose

    section, _, key = dotted_path.rpartition(".")
    if not section:
        return None
    return get_cli_setting(section, key, None)


def _shipped_default_for(dotted_path: str) -> object:
    from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML  # lazy on purpose

    node: object = DEFAULT_CONFIG_FROM_TOML
    for part in dotted_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _warn_once(key: str, message: str) -> None:
    """Log ``message`` once per ``key`` (typically a prompt id) per process."""
    if key in _warned_ids:
        return
    _warned_ids.add(key)
    logger.warning(f"internal_prompts: {message}")
