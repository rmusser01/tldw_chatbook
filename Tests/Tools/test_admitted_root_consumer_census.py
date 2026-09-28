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
    # Agent worktrees are local git checkouts (git_* is refused on remote
    # bindings), read through the ``source = self._worktree_repo_authority`` alias.
    "tldw_chatbook/Agents/agent_service.py::_admit_agent_worktree": 3,
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


def _terminal_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _holds_root(name: str | None) -> bool:
    return bool(name) and any(marker in name for marker in _RECEIVER_MARKERS)


def _function_aliases(function: ast.AST) -> set[str]:
    """Local names bound straight from a root holder (``source = authority``)."""
    aliases: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if _holds_root(_terminal_name(value)):
            aliases.update(t.id for t in targets if isinstance(t, ast.Name))
    return aliases


def _census_of_source(text: str, relative: str) -> collections.Counter[str]:
    hits: collections.Counter[str] = collections.Counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(text)
    scopes: list[tuple[str, set[str]]] = []

    class _Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.AST) -> None:
            scopes.append((node.name, _function_aliases(node)))  # type: ignore[attr-defined]
            self.generic_visit(node)
            scopes.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Attribute(self, node: ast.Attribute) -> None:
            if node.attr == "root":
                base = _terminal_name(node.value)
                aliases = scopes[-1][1] if scopes else set()
                if _holds_root(base) or (
                    isinstance(node.value, ast.Name) and node.value.id in aliases
                ):
                    where = scopes[-1][0] if scopes else "<module>"
                    hits[f"{relative}::{where}"] += 1
            self.generic_visit(node)

    _Visitor().visit(tree)
    return hits


def _census() -> collections.Counter[str]:
    hits: collections.Counter[str] = collections.Counter()
    root = _PACKAGE.parent
    for source in sorted(_PACKAGE.rglob("*.py")):
        if source.name == "remote_worker_bundle.py":
            continue  # generated copy of modules scanned at their source
        relative = source.relative_to(root).as_posix()
        hits.update(_census_of_source(source.read_text(encoding="utf-8"), relative))
    return hits


def test_census_follows_simple_aliases() -> None:
    """``source = self._authority`` then ``source.root`` counts as a read."""
    text = (
        "def admit(self):\n"
        "    source = self._worktree_repo_authority\n"
        "    return source.root, other.root\n"
    )
    assert _census_of_source(text, "m.py") == {"m.py::admit": 1}


def test_admitted_root_read_sites_match_the_reviewed_census() -> None:
    """Every function reading an admitted root's ``.root`` is pinned.

    A new or increased site fails with guidance: handle ``RemoteRoot``
    (``is_remote`` / ``local_root_path``) or prove the value is local, then
    update ``_PINNED``. A shrunk or moved site fails so the pin stays exact.
    """
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
