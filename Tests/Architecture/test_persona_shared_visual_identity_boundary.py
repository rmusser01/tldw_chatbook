"""Architecture guardrails for Persona Shared Visual Identity reactions."""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PERSONA_ADAPTER = (
    REPO_ROOT / "tldw_chatbook" / "Character_Chat" / "persona_visual_identity.py"
)
SHARED_RUNTIME = REPO_ROOT / "tldw_chatbook" / "Character_Chat" / "visual_identity.py"
CONSOLE_SESSION = REPO_ROOT / "tldw_chatbook" / "UI" / "Console_Modules" / "session.py"
CONSOLE_CHARACTER = (
    REPO_ROOT / "tldw_chatbook" / "UI" / "Console_Modules" / "character.py"
)
PRODUCTION_MODULES = (
    PERSONA_ADAPTER,
    SHARED_RUNTIME,
    CONSOLE_SESSION,
    CONSOLE_CHARACTER,
)
FORBIDDEN_IMPORT_PARTS = frozenset(
    {
        "Actor_Pack",
        "Persona_Buddy",
        "Persona_Visual",
        "tldw_api",
        "tldw_server",
    }
)
CONSOLE_REACTION_FUNCTIONS = frozenset(
    {
        "_resolve_visual_identity_for_db",
        "_visual_identity_options_for_db",
        "_set_manual_reaction",
        "_select_console_reaction",
        "_preview_console_reaction",
    }
)


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imports(path: Path) -> set[str]:
    imported: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(("." * node.level) + (node.module or ""))
    return imported


def test_persona_shared_identity_modules_do_not_cross_other_runtime_boundaries() -> (
    None
):
    for path in PRODUCTION_MODULES:
        for imported in _imports(path):
            assert not FORBIDDEN_IMPORT_PARTS.intersection(imported.split(".")), (
                f"{path.relative_to(REPO_ROOT)} imports forbidden boundary {imported}"
            )


def test_persona_adapter_reuses_shared_operational_state_mapping() -> None:
    assigned_names = {
        target.id
        for node in ast.walk(_tree(PERSONA_ADAPTER))
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
        if isinstance(target, ast.Name)
    }

    assert "_OPERATIONAL_EXPRESSION_KEYS" not in assigned_names


def test_console_reaction_functions_do_not_reference_buddy_or_persona_visual() -> None:
    functions = {
        node.name: ast.unparse(node).lower()
        for node in ast.walk(_tree(CONSOLE_SESSION))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in CONSOLE_REACTION_FUNCTIONS
    }

    assert functions.keys() == CONSOLE_REACTION_FUNCTIONS
    for name, source in functions.items():
        assert "persona_buddy" not in source, name
        assert "server" not in source, name
