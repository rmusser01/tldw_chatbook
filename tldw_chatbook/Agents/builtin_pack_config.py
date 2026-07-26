# tldw_chatbook/Agents/builtin_pack_config.py
"""Which built-in tool packs the user has enabled.

Reads ``[agent_tools] enabled_packs``. Uses ``get_cli_setting``'s 3-arg
form deliberately: the 2-arg form's second positional slot carries the
DEFAULT, not a key, and mis-slotting it there is exactly the defect
TASK-547 records in ``get_tool_executor()``.

Back-compat: TASK-584 shipped per-tool ``[tools] read_file_enabled`` /
``list_directory_enabled`` flags. Those keep working as a deprecated
fallback so an existing user is never silently switched off, but an
explicit ``enabled_packs`` list always wins -- including an explicitly
empty one, which means "no packs" rather than "fall back".

``BuiltinToolProvider`` -- and therefore this module's ``enabled_packs()``
-- is rebuilt fresh every agent turn (see
``console_agent_bridge``'s per-run builder docstring), so the legacy-flag
deprecation warning uses the same once-per-process log idiom as
``Internal_Prompts.resolver`` (see ``_warn_once`` there): without it, a
user still on the legacy flags would get the warning on every single turn
for the life of a conversation.
"""

from __future__ import annotations

from loguru import logger

from tldw_chatbook.config import get_cli_setting

#: `[tools]` flags that used to enable individual file tools, and the pack
#: that now owns them.
_LEGACY_FILE_FLAGS = ("read_file_enabled", "list_directory_enabled")

_MISSING = object()

#: Dedup keys already warned about (see `_warn_once`). Mirrors
#: `Internal_Prompts.resolver._warned_ids`; tests clear this directly the
#: same way that module's tests clear its set (see `cli_setting` in
#: `Tests/Agents/test_builtin_pack_config.py`), so one test's fallback
#: warning can never suppress another test's assertion that it fired.
_warned_ids: set[str] = set()


def _warn_once(key: str, message: str, **kwargs: object) -> None:
    """Log ``message`` once per ``key`` per process.

    Same idiom as ``Internal_Prompts.resolver._warn_once``: a per-run
    provider that calls the guarded path every turn must not re-log a
    warning that already ran the first time.

    Args:
        key: Dedup key; ``message`` logs at most once per key per process.
        message: The loguru-style message, with ``{name}`` placeholders.
        **kwargs: Values for ``message``'s placeholders.
    """
    if key in _warned_ids:
        return
    _warned_ids.add(key)
    logger.warning(message, **kwargs)


def enabled_packs() -> frozenset[str]:
    """Pack names the user has switched on.

    Returns:
        The configured pack names; an empty set when nothing is enabled.
        Defaults to empty -- built-in file tools ship disabled (TASK-584)
        and this function must not change that posture.
    """
    configured = get_cli_setting("agent_tools", "enabled_packs", _MISSING)
    if configured is not _MISSING:
        if not isinstance(configured, list):
            logger.warning(
                "[agent_tools] enabled_packs must be a list; ignoring {value!r}",
                value=configured,
            )
            return frozenset()
        return frozenset(str(name) for name in configured)

    if any(get_cli_setting("tools", flag, False) for flag in _LEGACY_FILE_FLAGS):
        _warn_once(
            "legacy_tools_flags",
            "[tools] {flags} are deprecated; set [agent_tools] enabled_packs = "
            '["files"] instead. Enabling the files pack for now.',
            flags=", ".join(_LEGACY_FILE_FLAGS),
        )
        return frozenset({"files"})

    return frozenset()
