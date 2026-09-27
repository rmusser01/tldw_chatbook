"""Census of code that reads an admitted workspace root's ``.root``.

``RunAdmittedWorkspaceRoot.root`` (and a project selection's ``root``) is
``Path | AdmittedRoot``: it can be a :class:`RemoteRoot` describing a folder
on an SSH host. Code that treats it as a laptop ``Path`` would read the
LAPTOP's copy of that path -- the wrong-file hazard ADR-181 exists to close.
``Path(remote_root)`` raises, but ``Path(str(root))`` or a string join does
not, so every read site is pinned here and a new one must be reviewed.

A failure here means a function started reading ``authority.root`` /
``selection.root`` (or reads it more often). Handle remote roots with
``remote_root_types.is_remote`` / ``local_root_path`` (or prove the value
can only be local), then update the row below.
"""

from __future__ import annotations

import ast
import collections
import warnings
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parents[2] / "tldw_chatbook"

#: Receiver names that hold an admitted root (authority / selection objects).
_RECEIVER_MARKERS = ("authority", "selection", "admitted")

#: "path::function" -> number of ``.root`` reads, as of TASK-33009 follow-ups.
_PINNED: dict[str, int] = {
    "tldw_chatbook/Agents/agent_worktree_recovery.py::_authority": 2,
    "tldw_chatbook/Agents/agent_worktree_recovery.py::_metadata": 1,
    "tldw_chatbook/Agents/agent_worktree_recovery.py::_snapshot": 1,
    "tldw_chatbook/Agents/agent_worktree_recovery.py::recover_agent_worktree": 1,
    "tldw_chatbook/Agents/local_tool_provider.py::__init__": 3,
    "tldw_chatbook/Agents/local_tool_provider.py::_invoke_allowed": 4,
    "tldw_chatbook/Agents/local_tool_provider.py::admit_run_workspace_root": 3,
    "tldw_chatbook/Agents/local_tool_provider.py::path_targets": 1,
    "tldw_chatbook/Agents/virtual_cli_provider.py::__init__": 3,
    "tldw_chatbook/Agents/virtual_cli_provider.py::_error_redaction_root": 3,
    "tldw_chatbook/Agents/virtual_cli_provider.py::execute": 3,
    "tldw_chatbook/Chat/console_agent_bridge.py::_read_run_log_page": 1,
    "tldw_chatbook/Chat/console_agent_bridge.py::load_run_log_text": 1,
    "tldw_chatbook/Chat/console_chat_controller.py::_build_personal_context_snapshot": 2,
    "tldw_chatbook/Chat/console_chat_controller.py::_build_project_instruction_preview_for_session": 5,
    "tldw_chatbook/Chat/console_chat_controller.py::_compose_agent_request_providers": 1,
    "tldw_chatbook/Chat/console_chat_controller.py::_default_remote_instruction_executor": 1,
    "tldw_chatbook/Chat/console_chat_controller.py::_project_binding_snapshot": 3,
    "tldw_chatbook/Chat/console_chat_controller.py::_project_instruction_excluded_dirs": 1,
    "tldw_chatbook/Chat/console_chat_controller.py::_remote_instruction_io_for_selection": 2,
    "tldw_chatbook/Chat/console_chat_controller.py::_resolve_remote_project_instruction_startup": 1,
    "tldw_chatbook/Chat/console_chat_controller.py::_run_agent_reply": 4,
    "tldw_chatbook/Chat/console_chat_controller.py::_workspace_binding_authority_is_current": 1,
    "tldw_chatbook/Chat/console_chat_controller.py::capture_run_admitted_workspace_roots": 6,
    "tldw_chatbook/Chat/console_chat_controller.py::resolve_project_instruction_binding": 1,
    "tldw_chatbook/Chat/console_worktree_recovery.py::read": 1,
}


def _census() -> collections.Counter[str]:
    hits: collections.Counter[str] = collections.Counter()
    root = _PACKAGE.parent
    for source in sorted(_PACKAGE.rglob("*.py")):
        if source.name == "remote_worker_bundle.py":
            continue  # generated copy of modules scanned at their source
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source.read_text(encoding="utf-8"))
        relative = source.relative_to(root).as_posix()
        functions: list[str] = []

        class _Visitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.AST) -> None:
                functions.append(node.name)  # type: ignore[attr-defined]
                self.generic_visit(node)
                functions.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Attribute(self, node: ast.Attribute) -> None:
                if node.attr == "root":
                    value = node.value
                    base = (
                        value.id
                        if isinstance(value, ast.Name)
                        else value.attr
                        if isinstance(value, ast.Attribute)
                        else None
                    )
                    if base and any(marker in base for marker in _RECEIVER_MARKERS):
                        where = functions[-1] if functions else "<module>"
                        hits[f"{relative}::{where}"] += 1
                self.generic_visit(node)

        _Visitor().visit(tree)
    return hits


def test_admitted_root_read_sites_match_the_reviewed_census() -> None:
    current = _census()
    grown = {
        site: count
        for site, count in current.items()
        if count > _PINNED.get(site, 0)
    }
    assert not grown, (
        "new or increased reads of an admitted root's .root -- handle "
        "RemoteRoot (is_remote / local_root_path) and update _PINNED: "
        f"{grown}"
    )
    shrunk = {site: count for site, count in _PINNED.items() if current[site] < count}
    assert not shrunk, f"sites shrank or moved; lower their _PINNED rows: {shrunk}"
