"""Agents and LLM_Calls stay below Console, and no private name crosses the line.

TASK-36944.2. The provider contract moved to ``Provider_Contract`` (TASK-36944.1),
so what is left of Agents -> Chat is a set of reaches into ``Chat.console_*``
modules. ``FROZEN_CONSOLE_IMPORTS`` freezes the ones that existed when this test
landed, one ``(importer, imported, reason)`` row each. It may only shrink: a new
pair fails the first test, a pair that is no longer imported fails the second.
Imports through the ``Chat.*`` aliases of the moved Provider_Contract modules
are not ``console_*`` modules and are not flagged.

The third test is the underscore rule: no module in Agents, LLM_Calls or Chat
imports (``from m import _x``), reads (``mod._x``) or ``getattr``s an
underscore-prefixed name of a module on the other side of the Agents/Chat line.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "tldw_chatbook"
UPSTREAM = ("Agents", "LLM_Calls")
CONSOLE_PREFIX = "tldw_chatbook.Chat.console_"

_P = "tldw_chatbook.Chat."
FROZEN_CONSOLE_IMPORTS: frozenset[tuple[str, str, str]] = frozenset(
    ("tldw_chatbook." + importer, _P + imported, reason)
    for importer, imported, reason in (
        ("Agents.agent_models", "console_provider_support", "readiness keys table"),
        ("Agents.agent_request_execution", "console_prepared_request", "PreparedProviderRequest type"),
        ("Agents.agent_request_execution", "console_thinking_capture", "CALL_THINKING_KEY constant"),
        ("Agents.agent_routing", "console_provider_endpoints", "effective endpoint lookup"),
        ("Agents.agent_routing", "console_provider_gateway", "custom endpoint engine swap"),
        ("Agents.agent_routing", "console_provider_support", "provider identity resolution"),
        ("Agents.agent_routing", "console_session_settings", "default settings and custom-endpoint key readiness"),
        ("Agents.agent_service", "console_history_budget", "history pruning and token counting"),
        ("Agents.agent_service", "console_project_instructions", "EPHEMERAL_ORIGIN_KEY re-export"),
        ("Agents.agent_service", "console_provider_support", "generation request keys"),
        ("Agents.agent_service", "console_trace_service", "TraceCallPersistenceError"),
        ("Agents.automatic_work_runtime", "console_goal_runs", "GoalIterationAuthorization"),
        ("Agents.code_review_history", "console_prepared_request", "resolve_request_capacity"),
        ("Agents.code_review_store", "console_scratch_space", "scratch space manager"),
        ("Agents.first_request_window", "console_prepared_request", "request_safety_margin_tokens"),
        ("Agents.library_tool_provider", "console_library_policy", "ConsoleAssistantLibraryAccess"),
        ("Agents.local_tool_provider", "console_project_instructions", "fingerprint_canonical_locator"),
        ("Agents.run_log_eviction", "console_history_budget", "window bounding and token counting"),
        ("Agents.tool_catalog", "console_ephemeral", "tool blocked-reason helpers"),
        ("Agents.tool_catalog", "console_library_policy", "ConsoleAssistantLibraryAccess"),
        ("LLM_Calls.deepseek", "console_provider_endpoints", "builtin_provider_endpoint"),
        ("LLM_Calls.groq", "console_provider_endpoints", "builtin_provider_endpoint"),
        ("LLM_Calls.mistral", "console_provider_endpoints", "builtin_provider_endpoint"),
        ("LLM_Calls.openrouter", "console_provider_endpoints", "builtin_provider_endpoint"),
        ("LLM_Calls.LLM_API_Calls", "console_prepared_request", "ConsoleCacheBoundary"),
        ("LLM_Calls.LLM_API_Calls", "console_project_instructions", "EPHEMERAL_ORIGIN_KEY re-export"),
        ("LLM_Calls.LLM_API_Calls", "console_provider_endpoints", "builtin_provider_endpoint"),
        ("LLM_Calls.LLM_API_Calls_Local", "console_provider_gateway", "normalize_llamacpp_base_url"),
        ("LLM_Calls.LLM_API_Calls_Local", "console_provider_support", "local thinking payload fields"),
    )
)
FROZEN_PAIRS = frozenset((a, b) for a, b, _ in FROZEN_CONSOLE_IMPORTS)


def _modules(*areas: str):
    for area in areas:
        for path in sorted((PACKAGE_ROOT / area).rglob("*.py")):
            parts = list(path.relative_to(REPO_ROOT).with_suffix("").parts)
            is_init = parts[-1] == "__init__"
            if is_init:
                parts = parts[:-1]
            yield path, ".".join(parts), is_init


def _resolve(current: str, is_init: bool, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    base = current.split(".") if is_init else current.split(".")[:-1]
    base = base[: len(base) - (node.level - 1)]
    return ".".join(base + ([node.module] if node.module else []))


def _is_module(name: str) -> bool:
    rel = Path(*name.split("."))
    return (REPO_ROOT / rel).with_suffix(".py").exists() or (
        REPO_ROOT / rel / "__init__.py"
    ).exists()


def _area(module: str) -> str:
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else ""


def _imported_modules(tree: ast.AST, current: str, is_init: bool):
    """Yield (module imported, names bound from it or None, local binding) per import."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, None, (alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            target = _resolve(current, is_init, node)
            for alias in node.names:
                yield target, alias.name, (alias.asname or alias.name)


def _console_pairs() -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for path, current, is_init in _modules(*UPSTREAM):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for target, name, _binding in _imported_modules(tree, current, is_init):
            if target.startswith(CONSOLE_PREFIX):
                found.add((current, ".".join(target.split(".")[:3])))
            elif target == "tldw_chatbook.Chat" and name and name.startswith("console_"):
                found.add((current, target + "." + name))
    return found


def test_agents_and_llm_calls_import_no_new_console_module() -> None:
    new = sorted(_console_pairs() - FROZEN_PAIRS)
    assert not new, (
        "Agents/LLM_Calls must not import Chat.console_* modules; move the "
        f"shared piece below Chat instead of adding a row: {new}"
    )


def test_frozen_console_imports_have_no_stale_rows() -> None:
    stale = sorted(FROZEN_PAIRS - _console_pairs())
    assert not stale, f"remove these rows, the imports are gone (list only shrinks): {stale}"


def _literal_module(node: ast.AST) -> str | None:
    """The ``tldw_chatbook.*`` literal of ``import_module(..)``/``sys.modules[..]``/``.get(..)``."""
    if isinstance(node, ast.Subscript) and ast.unparse(node.value) == "sys.modules":
        arg = node.slice
    elif isinstance(node, ast.Call) and node.args and ast.unparse(node.func) in {
        "importlib.import_module",
        "import_module",
        "sys.modules.get",
    }:
        arg = node.args[0]
    else:
        return None
    if isinstance(arg, ast.Constant) and str(arg.value).startswith("tldw_chatbook."):
        return str(arg.value)
    return None


def _private_crossings() -> list[str]:
    crossings: list[str] = []
    other = {"Agents": {"Chat"}, "LLM_Calls": {"Chat"}, "Chat": {"Agents", "LLM_Calls"}}
    for path, current, is_init in _modules("Agents", "LLM_Calls", "Chat"):
        targets = other[_area(current)]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        bound: set[str] = set()
        for target, name, binding in _imported_modules(tree, current, is_init):
            if name is None:
                if _area(target) in targets:
                    bound.add(binding)
                continue
            if _area(target) in targets and name.startswith("_") and not name.startswith("__"):
                crossings.append(f"{current}: from {target} import {name}")
            if _area(target) in targets and _is_module(f"{target}.{name}"):
                bound.add(binding)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                literal = _literal_module(node.value)
                if literal and _area(literal) in targets:
                    bound.update(
                        ast.unparse(t) for t in node.targets if isinstance(t, ast.Name)
                    )
        for node in ast.walk(tree):
            literal = _literal_module(node.value) if isinstance(node, ast.Attribute) else None
            if (
                isinstance(node, ast.Attribute)
                and literal
                and _area(literal) in targets
                and node.attr.startswith("_")
                and not node.attr.startswith("__")
            ):
                crossings.append(f"{current}:{node.lineno}: {ast.unparse(node)}")
            elif (
                isinstance(node, ast.Attribute)
                and node.attr.startswith("_")
                and not node.attr.startswith("__")
                and ast.unparse(node.value) in bound
            ):
                crossings.append(f"{current}:{node.lineno}: {ast.unparse(node)}")
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"getattr", "setattr", "hasattr"}
                and len(node.args) > 1
                and ast.unparse(node.args[0]) in bound
                and isinstance(node.args[1], ast.Constant)
                and str(node.args[1].value).startswith("_")
            ):
                crossings.append(f"{current}:{node.lineno}: {node.func.id} {node.args[1].value}")
    return crossings


def test_no_underscore_name_crosses_the_agents_chat_boundary() -> None:
    assert _private_crossings() == []
