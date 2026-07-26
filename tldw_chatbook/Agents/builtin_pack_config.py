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
"""

from __future__ import annotations

from loguru import logger

from tldw_chatbook.config import get_cli_setting

#: `[tools]` flags that used to enable individual file tools, and the pack
#: that now owns them.
_LEGACY_FILE_FLAGS = ("read_file_enabled", "list_directory_enabled")

_MISSING = object()


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
        logger.warning(
            "[tools] {flags} are deprecated; set [agent_tools] enabled_packs = "
            '["files"] instead. Enabling the files pack for now.',
            flags=", ".join(_LEGACY_FILE_FLAGS),
        )
        return frozenset({"files"})

    return frozenset()
