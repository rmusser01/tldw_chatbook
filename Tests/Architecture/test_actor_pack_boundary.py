"""Architecture and scope guardrails for portable Actor Pack foundations."""

from __future__ import annotations

import ast
import dataclasses
import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ACTOR_PACK_ROOT = REPO_ROOT / "tldw_chatbook" / "Actor_Packs"
PURE_MODULES = (ACTOR_PACK_ROOT / "contracts.py",)
PRODUCTION_MODULES = tuple(sorted(ACTOR_PACK_ROOT.glob("*.py")))
FOUNDATION_MODULES = tuple(
    ACTOR_PACK_ROOT / name for name in ("contracts.py", "creation.py", "repository.py")
)
EXPORT_MODULES = tuple(
    ACTOR_PACK_ROOT / name for name in ("export.py", "publication.py", "controller.py")
)
CHACHANOTES_DB = REPO_ROOT / "tldw_chatbook" / "DB" / "ChaChaNotes_DB.py"
FORBIDDEN_IMPORT_PARTS = frozenset(
    {
        "Agents",
        "Image_Generation",
        "LLM_Calls",
        "Persona_Buddy",
        "Persona_Visual",
        "UI",
        "Video_Generation",
        "Widgets",
        "aiohttp",
        "httpx",
        "requests",
        "socket",
        "tldw_api",
        "tldw_server",
        "textual",
        "urllib",
    }
)
SAFE_EXTERNAL_IMPORT_ROOTS = frozenset({"PIL"})
FORBIDDEN_ARCHIVE_CALLS = frozenset(
    {
        "extract",
        "extractall",
        "open",
        "write",
        "writestr",
    }
)
FORBIDDEN_SURFACE_WORDS = frozenset(
    {"activate", "activation", "extract", "import", "review"}
)
PRIVATE_FIELD_TOKENS = (
    "absolute",
    "credential",
    "exception",
    "host_path",
    "prompt",
    "provider",
    "secret",
    "traceback",
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


def test_pure_contracts_do_not_cross_runtime_ui_network_or_provider_boundaries() -> (
    None
):
    external_roots: set[str] = set()
    for path in PURE_MODULES:
        for imported in _imports(path):
            parts = imported.split(".")
            assert not FORBIDDEN_IMPORT_PARTS.intersection(parts), (
                f"{path.relative_to(REPO_ROOT)} imports forbidden boundary {imported}"
            )
            if imported.startswith(".") or imported.startswith("tldw_chatbook"):
                continue
            root = imported.split(".", 1)[0]
            if root not in sys.stdlib_module_names and root != "__future__":
                external_roots.add(root)

    assert external_roots <= SAFE_EXTERNAL_IMPORT_ROOTS


def test_foundation_defines_no_archive_io_or_import_activation_review_surface() -> None:
    for path in FOUNDATION_MODULES:
        tree = _tree(path)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                words = frozenset(node.name.lower().split("_"))
                assert words.isdisjoint(FORBIDDEN_SURFACE_WORDS), (
                    f"{path.relative_to(REPO_ROOT)} exposes out-of-scope {node.name}"
                )
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
                continue
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "zipfile"
            ):
                assert node.func.attr not in FORBIDDEN_ARCHIVE_CALLS, (
                    f"{path.relative_to(REPO_ROOT)} performs archive I/O"
                )


def test_actor_pack_modules_do_not_merge_visual_runtime_ownership() -> None:
    for path in FOUNDATION_MODULES:
        imported = _imports(path)
        assert not any(
            "Persona_Visual" in name or "visual_identity" in name for name in imported
        ), f"{path.relative_to(REPO_ROOT)} imports a visual runtime owner"


def test_export_modules_remain_screen_free_and_define_no_import_activation() -> None:
    forbidden = {
        "Agents",
        "LLM_Calls",
        "UI",
        "Widgets",
        "httpx",
        "requests",
        "socket",
        "textual",
        "tldw_api",
        "tldw_server",
        "urllib",
    }
    for path in EXPORT_MODULES:
        for imported in _imports(path):
            assert forbidden.isdisjoint(imported.split(".")), (
                f"{path.relative_to(REPO_ROOT)} imports forbidden boundary {imported}"
            )
        for node in ast.walk(_tree(path)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                words = frozenset(node.name.lower().split("_"))
                assert "import" not in words and "activation" not in words, (
                    f"{path.relative_to(REPO_ROOT)} exposes import activation {node.name}"
                )


def test_export_public_records_hide_paths_ids_bytes_and_graph_authority() -> None:
    sensitive = {
        "actor_payload",
        "data",
        "destination",
        "graph_identity",
        "local_actor_id",
        "portrait_bytes",
        "source_identity",
    }
    for module_name in ("controller", "export", "publication"):
        module = importlib.import_module(f"tldw_chatbook.Actor_Packs.{module_name}")
        for name, value in vars(module).items():
            if not name.startswith("ActorPack") or not dataclasses.is_dataclass(value):
                continue
            for field in dataclasses.fields(value):
                if field.name in sensitive:
                    assert field.repr is False, (
                        f"{module.__name__}.{name}.{field.name} leaks through repr"
                    )


def test_public_actor_pack_records_and_errors_are_path_free() -> None:
    for module_name in ("contracts", "creation", "repository"):
        module = importlib.import_module(f"tldw_chatbook.Actor_Packs.{module_name}")
        for name, value in vars(module).items():
            if not name.startswith("ActorPack") or not dataclasses.is_dataclass(value):
                continue
            fields = {field.name.lower() for field in dataclasses.fields(value)}
            leaked = {
                field
                for field in fields
                if any(token in field for token in PRIVATE_FIELD_TOKENS)
            }
            assert not leaked, f"{module.__name__}.{name} exposes {sorted(leaked)}"


def test_actor_pack_migration_diagnostics_are_fixed_and_path_free() -> None:
    migration = next(
        node
        for node in _tree(CHACHANOTES_DB).body
        if isinstance(node, ast.ClassDef)
        for node in node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_migrate_from_v44_to_v45"
    )
    diagnostic_calls = [
        node
        for node in ast.walk(migration)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"debug", "error", "info", "warning"}
    ]

    assert len(diagnostic_calls) == 3
    for call in diagnostic_calls:
        source = ast.unparse(call)
        assert "db_path_str" not in source
        assert "exception=True" not in source
        assert "{exc}" not in source
