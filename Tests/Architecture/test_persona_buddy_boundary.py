"""Architecture and privacy guardrails for the app-owned Persona Buddy."""

from __future__ import annotations

import ast
import dataclasses
import importlib
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BUDDY_ROOT = REPO_ROOT / "tldw_chatbook" / "Persona_Buddy"
HEADLESS_MODULES = (
    BUDDY_ROOT / "controller.py",
    BUDDY_ROOT / "preferences.py",
    BUDDY_ROOT / "rendering.py",
)
EXPORT_MODULES = (
    REPO_ROOT / "tldw_chatbook" / "Character_Chat" / "Character_Chat_Lib.py",
    REPO_ROOT / "tldw_chatbook" / "UI" / "Screens" / "personas_screen.py",
)
FORBIDDEN_IMPORT_PARTS = frozenset({"textual", "UI", "Widgets"})
PRIVATE_SNAPSHOT_FIELD_TOKENS = (
    "bytes",
    "credential",
    "path",
    "prompt",
    "provider",
    "secret",
    "token",
)
FIXED_CATEGORY = re.compile(r"persona_buddy_[a-z0-9_]+\Z")


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


def test_headless_buddy_modules_have_no_textual_or_ui_imports() -> None:
    for path in HEADLESS_MODULES:
        for imported in _imports(path):
            assert not FORBIDDEN_IMPORT_PARTS.intersection(imported.split(".")), (
                f"{path.relative_to(REPO_ROOT)} imports forbidden boundary {imported}"
            )


def test_controller_has_no_model_or_emote_parser_boundary() -> None:
    path = BUDDY_ROOT / "controller.py"
    tree = _tree(path)
    imported = _imports(path)
    assert not any(
        "llm_calls" in name.lower() or "emote" in name.lower() for name in imported
    )

    string_literals = {
        node.value.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert not any("emote:" in value for value in string_literals)
    assert {"prompt", "model_text", "assistant_text"}.isdisjoint(string_literals)


def test_buddy_preferences_do_not_enter_persona_exporters() -> None:
    for path in EXPORT_MODULES:
        for node in ast.walk(_tree(path)):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if "export" not in node.name.lower():
                continue
            strings = {
                child.value.lower()
                for child in ast.walk(node)
                if isinstance(child, ast.Constant) and isinstance(child.value, str)
            }
            assert "persona_buddy" not in strings, (
                f"{path.relative_to(REPO_ROOT)}:{node.name} exports Buddy preferences"
            )


def test_public_snapshots_and_logs_exclude_private_payload_fields() -> None:
    for module_name in ("controller", "rendering"):
        module = importlib.import_module(f"tldw_chatbook.Persona_Buddy.{module_name}")
        for name, value in vars(module).items():
            if "Snapshot" not in name or not dataclasses.is_dataclass(value):
                continue
            fields = {field.name.lower() for field in dataclasses.fields(value)}
            leaked = {
                field
                for field in fields
                if any(token in field for token in PRIVATE_SNAPSHOT_FIELD_TOKENS)
            }
            assert not leaked, f"{module.__name__}.{name} exposes {sorted(leaked)}"

    for path in HEADLESS_MODULES:
        for node in ast.walk(_tree(path)):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
                continue
            if (
                not isinstance(node.func.value, ast.Name)
                or node.func.value.id != "logger"
            ):
                continue
            assert len(node.args) == 1 and not node.keywords
            message = node.args[0]
            assert isinstance(message, ast.Constant) and isinstance(message.value, str)
            assert FIXED_CATEGORY.fullmatch(message.value), (
                f"{path.relative_to(REPO_ROOT)} logs non-category Buddy detail"
            )
