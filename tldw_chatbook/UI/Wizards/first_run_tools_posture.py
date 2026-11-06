"""The Tools step's own pieces beyond the shared tool-surface reading (TASK-34100.13).

fulltrack-02: the Tools step said "Everything is off by default" while plain
chat was handed the whole agent surface; the Summary and the Quick-track note
repeated it. Since TASK-34100.5.3 ("Light surface by default", owner ruling
2026-10-04) every one of those sentences -- the step's posture sentence, the
Summary's Tools row and the Quick note's tools part -- comes from
``Agents.plain_chat_tool_surface``, the one reading of config the send uses.
This module no longer derives its own posture (the rebase onto that work
retired the second derivation); it keeps only what the step adds:

* ``WORKSPACE_HINT`` / ``FILE_TOOL_NAMES`` -- fulltrack-06's one static hint
  under the file rows;
* ``gate_config_view`` -- the ``[tools]``/``[console]`` sections the step's
  switches, posture and commit baseline are read from (the config as it is
  now, never ``load_settings()``'s copy, which has no ``[tools]`` table).

The local group's ``[console]`` master switch is described by the light-surface
copy, never offered: a reflexive "turn everything off" in setup would silently
remove what Full agent tools adds.
"""

from __future__ import annotations

from typing import Any

#: The built-in file tools (`file_operation_tools`), which the Workspace hint
#: sits under.
FILE_TOOL_NAMES = frozenset(
    {"read_file", "list_directory", "write_file", "glob_files", "grep_files"}
)

#: Where file tools can reach, and where a folder is added.
WORKSPACE_HINT = (
    "File tools only reach folders you add as a Workspace (Settings ▸ "
    "Workspaces) and the chat's scratch area."
)


def gate_config_view(wizard: Any) -> dict[str, dict[str, Any]]:
    """The ``[tools]``/``[console]`` sections the step reads gates from.

    Args:
        wizard: The Tools step's wizard.

    Returns:
        ``{"tools": {...}, "console": {...}}`` for ``all_tool_gates(config=...)``
        and ``summarize_tool_surface``, read from the config as it is now
        (``wizard_config_sections``).
    """
    from tldw_chatbook.UI.Wizards.first_run_setup_state import wizard_config_sections

    return wizard_config_sections(wizard, "tools", "console")
