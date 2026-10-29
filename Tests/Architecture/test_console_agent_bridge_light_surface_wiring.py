"""Every production ``ConsoleAgentBridge`` is built with a light-surface reader.

TASK-34100.5.3 (owner ruling 2026-10-04, "Light surface by default"): plain
Console chat gets a small, safe tool set until the user turns on Full agent
tools. The bridge asks its ``light_tool_surface`` reader once per send, and a
bridge built WITHOUT one keeps the full agent surface -- the constructor
default stays ``None`` on purpose, because the many test harnesses that build
a bridge predate the light surface and test the agent surface.

That default is what let the served mobile companion slip through. Its
``MobileChatService`` built its own bridge with no reader, so every phone
send got ``spawn_subagent`` and the rest of the agent surface whatever Full
agent tools said (pre-merge review, 2026-10-29). Nothing failed: a missing
keyword argument is silent.

So this guard reads the source. Every ``ConsoleAgentBridge(...)`` call under
``tldw_chatbook/`` must pass ``light_tool_surface=`` explicitly. Test
harnesses live under ``Tests/`` and are not scanned: they build the bridge
on purpose. A production subclass is refused too, because its constructions
would not be spelled ``ConsoleAgentBridge(``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "tldw_chatbook"
BRIDGE_CLASS = "ConsoleAgentBridge"
REQUIRED_KEYWORD = "light_tool_surface"


def _bridge_names(tree: ast.AST) -> frozenset[str]:
    """The names this module may call the bridge class by (aliases included)."""
    names = {BRIDGE_CLASS}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == BRIDGE_CLASS and alias.asname:
                    names.add(alias.asname)
    return frozenset(names)


def _is_bridge_reference(node: ast.AST, names: frozenset[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id in names
    if isinstance(node, ast.Attribute):
        return node.attr == BRIDGE_CLASS
    return False


def scan_source(source: str, label: str) -> tuple[list[str], list[str]]:
    """Find bridge constructions in one module's source.

    Returns:
        ``(constructions, violations)``: a ``label:line`` for every
        ``ConsoleAgentBridge(...)`` call, and a description of every call
        without an explicit ``light_tool_surface=`` and every subclass.
    """
    tree = ast.parse(source, filename=label)
    names = _bridge_names(tree)
    constructions: list[str] = []
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_bridge_reference(node.func, names):
            site = f"{label}:{node.lineno}"
            constructions.append(site)
            if not any(keyword.arg == REQUIRED_KEYWORD for keyword in node.keywords):
                violations.append(
                    f"{site} builds a {BRIDGE_CLASS} without {REQUIRED_KEYWORD}="
                )
        elif isinstance(node, ast.ClassDef) and any(
            _is_bridge_reference(base, names) for base in node.bases
        ):
            violations.append(
                f"{label}:{node.lineno} subclasses {BRIDGE_CLASS} ({node.name})"
            )
    return constructions, violations


def _scan_package() -> tuple[list[str], list[str]]:
    constructions: list[str] = []
    violations: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if BRIDGE_CLASS not in source:
            continue
        label = path.relative_to(REPO_ROOT).as_posix()
        found, refused = scan_source(source, label)
        constructions.extend(found)
        violations.extend(refused)
    return constructions, violations


@pytest.mark.unit
def test_every_production_bridge_is_built_with_a_light_surface_reader() -> None:
    constructions, violations = _scan_package()

    # Non-vacuous: the two production builders are found -- the desktop
    # Console's runtime and the served mobile companion.
    built_in = {site.rsplit(":", 1)[0] for site in constructions}
    assert {
        "tldw_chatbook/Chat/console_runtime.py",
        "tldw_chatbook/Web_Server/mobile_service.py",
    } <= built_in, constructions
    assert not violations, (
        "A production ConsoleAgentBridge without a light_tool_surface reader "
        "hands plain Console chat the full agent surface whatever Full agent "
        "tools says. Pass light_surface_configured (and the kill-switch "
        "reader) as ConsoleRuntime.ensure_console_agent_bridge does:\n"
        + "\n".join(violations)
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("ConsoleAgentBridge(store=s)\n", 1),
        ("from m import ConsoleAgentBridge as Bridge\nBridge(store=s)\n", 1),
        ("bridge_module.ConsoleAgentBridge(store=s)\n", 1),
        ("class Mine(ConsoleAgentBridge):\n    pass\n", 1),
        ("ConsoleAgentBridge(store=s, light_tool_surface=reader)\n", 0),
        ("ConsoleAgentBridge.cancel_all_subagents(bridge)\n", 0),
    ],
    ids=["bare", "aliased", "attribute", "subclass", "wired", "not-a-construction"],
)
def test_the_scanner_flags_a_bridge_built_without_a_reader(source, expected) -> None:
    """Negative control: the scan sees every spelling it claims to."""
    _constructions, violations = scan_source(source, "probe.py")

    assert len(violations) == expected, violations
