"""Architecture guardrails for the separate local Persona Visual runtime."""

from __future__ import annotations

import ast
import dataclasses
import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "tldw_chatbook" / "Persona_Visual"
MIGRATIONS_ROOT = REPO_ROOT / "tldw_chatbook" / "DB" / "migrations"
PRODUCTION_MODULES = tuple(sorted(PACKAGE_ROOT.glob("*.py")))
FORBIDDEN_IMPORT_PARTS = frozenset(
    {
        "Character_Chat",
        "Image_Generation",
        "LLM_Calls",
        "UI",
        "Video_Generation",
        "Widgets",
        "tldw_server",
    }
)
SAFE_EXTERNAL_IMPORT_ROOTS = frozenset({"PIL"})
#: TASK-38990 reviewed exceptions, keyed by (module, imported name). The
#: create-only built-in Buddy installer (f4c1ad1a7a) writes a Character_Chat
#: persona and its Persona Visual pack in one coordinated step, so it reads
#: the character store by design. No other Persona_Visual module may.
#: Production has not called it since 11150b849d (Persona_Buddy.library seeds
#: the built-in Buddy); moving or deleting it is an owner decision.
REVIEWED_BOUNDARY_EXCEPTIONS = {
    (
        "builtin_pixel_migu.py",
        "tldw_chatbook.Character_Chat.local_character_persona_service",
    ): "installer writes the persona record it binds a visual pack to",
    (
        "builtin_pixel_migu.py",
        "tldw_chatbook.Character_Chat.builtin_pixel_migu",
    ): "installer finds or creates the bundled character it binds to",
}
PRIVATE_FIELD_TOKENS = ("path", "relpath", "exception", "traceback")


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(("." * node.level) + (node.module or ""))
    return imported


def _created_tables(path: Path) -> set[str]:
    sql = path.read_text(encoding="utf-8")
    return {
        line.split()[2].strip('"`[').rstrip("]")
        for line in sql.splitlines()
        if line.upper().startswith("CREATE TABLE ")
    }


def test_persona_visual_modules_stay_outside_other_runtime_and_ui_boundaries() -> None:
    external_roots: set[str] = set()
    for path in PRODUCTION_MODULES:
        imports = _imports(path)
        for imported in imports:
            if (path.name, imported) in REVIEWED_BOUNDARY_EXCEPTIONS:
                continue
            assert not FORBIDDEN_IMPORT_PARTS.intersection(imported.split(".")), (
                f"{path.relative_to(REPO_ROOT)} imports forbidden boundary {imported}"
            )
            if imported.startswith(".") or imported.startswith("tldw_chatbook"):
                continue
            root = imported.split(".", 1)[0]
            if root not in sys.stdlib_module_names and root != "__future__":
                external_roots.add(root)

    assert external_roots <= SAFE_EXTERNAL_IMPORT_ROOTS


def test_reviewed_boundary_exceptions_are_live() -> None:
    """TASK-38990: an exception outlives neither its module nor its import."""
    for module, imported in REVIEWED_BOUNDARY_EXCEPTIONS:
        assert imported in _imports(PACKAGE_ROOT / module), (module, imported)


def test_persona_visual_tables_are_separate_from_shared_visual_identity() -> None:
    persona_tables = _created_tables(
        MIGRATIONS_ROOT / "chachanotes_v40_to_v41_persona_visual.sql"
    )
    shared_tables = _created_tables(
        MIGRATIONS_ROOT / "chachanotes_v38_to_v39_visual_identity.sql"
    )

    assert persona_tables == {
        "persona_visual_packs",
        "persona_visual_pack_versions",
        "persona_visual_assets",
        "persona_visual_bindings",
    }
    assert persona_tables.isdisjoint(shared_tables)
    assert all(name.startswith("visual_identity_") for name in shared_tables)


def test_public_persona_visual_dataclasses_do_not_expose_private_details() -> None:
    for path in PRODUCTION_MODULES:
        if path.name == "__init__.py":
            continue
        module = importlib.import_module(f"tldw_chatbook.Persona_Visual.{path.stem}")
        for name, value in vars(module).items():
            if not name.startswith("PersonaVisual") or not dataclasses.is_dataclass(
                value
            ):
                continue
            fields = {field.name.lower() for field in dataclasses.fields(value)}
            leaked = {
                field
                for field in fields
                if any(token in field for token in PRIVATE_FIELD_TOKENS)
            }
            assert not leaked, f"{module.__name__}.{name} exposes {sorted(leaked)}"

        for name, value in vars(module).items():
            if name.startswith("PersonaVisual") and name.endswith("Error"):
                assert not dataclasses.is_dataclass(value)
