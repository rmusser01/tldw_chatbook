"""Chat must not grow code-identity checks or reach UI-private names (TASK-36944.9)."""

from __future__ import annotations

import ast
from functools import partial
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
CHAT = ROOT / "tldw_chatbook" / "Chat"
PREPARATION = CHAT / "console_configuration_preparation.py"

# backlog/decisions/285-native-phased-agent-workflows.md amends the 56/27/43 ceilings.
# Counts cannot prove count-preserving substitutions or introspection outside this predicate.
# Lower the test and ADR together on reductions; retire a row only at all zeros.
# Increases and new listed owners require a fresh policy decision.
PINS = {"__code__": 56, "__defaults__": 27, "__kwdefaults__": 43}
OWNER_COUNTS = {
    "archive_recovery/runtime.py": (11, 1, 1),
    "chat_conversation_service.py": (2, 2, 4),
    "console_activity_receipts.py": (1, 1, 1),
    "console_agent_bridge.py": (9, 7, 9),
    "console_agent_sources.py": (3, 3, 2),
    "console_attention_preparation.py": (3, 3, 3),
    "console_chat_controller.py": (23, 12, 18),
    "console_configuration_preparation.py": (12, 4, 4),
    "console_first_send_persistence.py": (1, 1, 1),
    "console_fleet_attention.py": (1, 1, 1),
    "console_fleet_wake.py": (1, 1, 1),
    "console_hook_preparation.py": (3, 3, 3),
    "console_ordinary_tree.py": (24, 11, 14),
    "console_preparation_reads.py": (1, 1, 2),
    "console_provider_gateway.py": (6, 3, 5),
    "console_raw_cli_custody.py": (1, 1, 1),
    "console_reference_preparation.py": (1, 1, 1),
    "console_references.py": (1, 0, 0),
    "console_runtime.py": (17, 13, 16),
    "console_ssh_custody.py": (1, 1, 1),
    "console_streaming_request.py": (1, 1, 1),
    "console_workflow_preparation.py": (54, 24, 27),
    "console_workflow_runs.py": (17, 8, 14),
    "console_workflow_wave.py": (32, 26, 26),
}
ORDINARY_COUNTS = dict(zip(PINS, (19, 10, 17)))


def _identity_attr(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute) and node.attr in PINS:
        return node.attr
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) > 1
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value in PINS
    ):
        return node.args[1].value
    return None


def _owner_counts(chat: Path) -> dict[str, dict[str, int]]:
    owners = {}
    for path in chat.rglob("*.py"):
        counts = dict.fromkeys(PINS, 0)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            name = _identity_attr(node)
            if name:
                counts[name] += 1
        owners[path.relative_to(chat).as_posix()] = counts
    return owners


def _counts() -> dict[str, int]:
    owners = _owner_counts(CHAT)
    return {attr: sum(counts[attr] for counts in owners.values()) for attr in PINS}


def _check_census(chat, listed, ordinary, attr=None):
    actual = _owner_counts(chat)
    for name in [attr] if attr else PINS:
        for owner, expected in listed.items():
            count = actual.get(owner, {}).get(name, 0)
            assert count == expected[name], (
                f"{owner}: {name} is {count}, expected {expected[name]}"
            )
        count = sum(
            counts[name] for owner, counts in actual.items() if owner not in listed
        )
        assert count == ordinary[name], (
            f"ordinary: {name} is {count}, expected {ordinary[name]}"
        )


@pytest.mark.parametrize("attr", sorted(PINS))
def test_chat_code_identity_references_do_not_rise(attr):
    listed = {owner: dict(zip(PINS, counts)) for owner, counts in OWNER_COUNTS.items()}
    _check_census(CHAT, listed, ORDINARY_COUNTS, attr)


def _is_private(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def ui_private_reaches(source: str) -> list[str]:
    """Private names imported from, or read off, tldw_chatbook.UI in ``source``."""
    tree = ast.parse(source)
    package = "tldw_chatbook.Chat"
    ui_names: set[str] = set()
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".")[: len(package.split(".")) - (node.level - 1)]
                base = ".".join(parts + ([node.module] if node.module else []))
            for alias in node.names:
                full = f"{base}.{alias.name}"
                if full.startswith("tldw_chatbook.UI") or base.startswith("tldw_chatbook.UI"):
                    ui_names.add(alias.asname or alias.name)
                    if _is_private(alias.name):
                        hits.append(f"import {full}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("tldw_chatbook.UI") and alias.asname:
                    ui_names.add(alias.asname)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and _is_private(node.attr):
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in ui_names:
                hits.append(f"attribute {root.id}...{node.attr}")
            elif isinstance(root, ast.Name) and root.id == "tldw_chatbook":
                dotted = ast.unparse(node.value)  # bare `import tldw_chatbook.UI.x` form
                if dotted.startswith("tldw_chatbook.UI"):
                    hits.append(f"attribute {dotted}.{node.attr}")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) > 1
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in ui_names
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
            and _is_private(node.args[1].value)
        ):
            hits.append(f"getattr {node.args[0].id}, {node.args[1].value!r}")
    return hits


def test_console_configuration_preparation_reaches_no_ui_private_names():
    assert ui_private_reaches(PREPARATION.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(
    "line",
    [
        "from tldw_chatbook.UI.Console_Modules import turn_admission as ta\nz = ta._X\n",
        "from tldw_chatbook.UI.Console_Modules.session import _X\n",
        "from ..UI.Console_Modules import wiring\nz = wiring.a._X\n",
        "import tldw_chatbook.UI.Console_Modules.wiring as w\nz = getattr(w, '_X')\n",
        "import tldw_chatbook.UI.Console_Modules.wiring\nz = tldw_chatbook.UI.Console_Modules.wiring._X\n",
    ],
)
def test_ui_private_reach_detector_catches_each_shape(line):
    assert ui_private_reaches(line)


def test_ui_private_reach_detector_ignores_public_and_dunder():
    src = "from tldw_chatbook.UI.Console_Modules import turn_admission as ta\nz = ta.pub; y = ta.__name__\n"
    assert ui_private_reaches(src) == []


# --- the UI-owned stock probes Chat now asks (replaces Chat's own code-identity checks) ---


def _stock_owner():
    from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController

    owner = SimpleNamespace()
    owner._build_console_turn_capture_selection = MethodType(
        ConsoleSessionController._build_console_turn_capture_selection, owner
    )
    return owner


def _stock_adapter():
    from tldw_chatbook.UI.Console_Modules.wiring import _stock_console_scratch_snapshot

    return partial(_stock_console_scratch_snapshot, SimpleNamespace())


def test_stock_mounted_capture_accepts_stock_wiring():
    from tldw_chatbook.UI.Console_Modules import turn_admission

    probe = turn_admission.stock_mounted_capture(_stock_owner())
    assert probe is not None
    assert probe[1](_stock_adapter())


def test_stock_mounted_capture_refuses_swapped_selector():
    from tldw_chatbook.UI.Console_Modules import turn_admission

    owner = SimpleNamespace()
    owner._build_console_turn_capture_selection = MethodType(lambda self, *a, **k: None, owner)
    assert turn_admission.stock_mounted_capture(owner) is None
    owner._build_console_turn_capture_selection = lambda *a, **k: None
    assert turn_admission.stock_mounted_capture(owner) is None


def test_stock_mounted_capture_refuses_patched_selection_source(monkeypatch):
    from tldw_chatbook.UI.Console_Modules import turn_admission

    monkeypatch.setattr(turn_admission, "policy_producers_current", lambda: True)
    assert turn_admission.stock_mounted_capture(_stock_owner()) is None


def test_stock_mounted_capture_refuses_source_code_replaced_in_place():
    from tldw_chatbook.UI.Console_Modules import turn_admission

    callback = turn_admission.console_live_runtime_enabled
    original = callback.__code__
    probe = turn_admission.stock_mounted_capture(_stock_owner())
    assert probe is not None
    callback.__code__ = (lambda resident_config, fallback_console: True).__code__
    try:
        assert turn_admission.stock_mounted_capture(_stock_owner()) is None
        assert not probe[1](_stock_adapter())  # also refused mid-capture
    finally:
        callback.__code__ = original


def test_stock_mounted_capture_refuses_source_defaults_replaced(monkeypatch):
    from tldw_chatbook.UI.Console_Modules import turn_admission

    callback = turn_admission.console_live_runtime_enabled
    probe = turn_admission.stock_mounted_capture(_stock_owner())
    monkeypatch.setattr(callback, "__defaults__", (1,))
    assert turn_admission.stock_mounted_capture(_stock_owner()) is None
    assert not probe[1](_stock_adapter())


def test_stock_mounted_capture_refuses_source_kwdefaults_replaced(monkeypatch):
    from tldw_chatbook.UI.Console_Modules import turn_admission

    callback = turn_admission.console_live_runtime_enabled
    probe = turn_admission.stock_mounted_capture(_stock_owner())
    monkeypatch.setattr(callback, "__kwdefaults__", {"x": 1})
    assert turn_admission.stock_mounted_capture(_stock_owner()) is None
    assert not probe[1](_stock_adapter())


def test_stock_mounted_capture_refuses_rebound_sources_mid_capture(monkeypatch):
    from tldw_chatbook.UI.Console_Modules import turn_admission

    probe = turn_admission.stock_mounted_capture(_stock_owner())
    monkeypatch.setattr(turn_admission, "console_live_runtime_enabled", lambda *a: True)
    monkeypatch.setattr(turn_admission, "_CONSOLE_SELECTION_SOURCES", ())
    assert not probe[1](_stock_adapter())


def test_stock_mounted_capture_refuses_swapped_scratch_adapter():
    from tldw_chatbook.UI.Console_Modules import turn_admission

    probe = turn_admission.stock_mounted_capture(_stock_owner())
    assert not probe[1](partial(lambda screen, session_id: None, SimpleNamespace()))


def test_is_stock_session_provider_accepts_only_the_stock_builder():
    from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController
    from tldw_chatbook.UI.Console_Modules.turn_admission import is_stock_session_provider

    controller = ConsoleSessionController.__new__(ConsoleSessionController)
    assert is_stock_session_provider(controller._build_console_turn_execution_context)
    assert not is_stock_session_provider(MethodType(lambda self, *a, **k: None, controller))
    assert not is_stock_session_provider(
        MethodType(
            ConsoleSessionController._build_console_turn_execution_context,
            SimpleNamespace(),
        )
    )
    assert not is_stock_session_provider(None)


@pytest.mark.parametrize("attr", sorted(PINS))
def test_identity_census_enforces_each_partition_and_reductions(tmp_path, attr):
    listed = tmp_path / "listed.py"
    listed.write_text(f"x.{attr}\n", encoding="utf-8")
    ordinary = tmp_path / "ordinary.py"
    ordinary.write_text(f"getattr(x, '{attr}')\n", encoding="utf-8")
    owners = {"listed.py": {name: int(name == attr) for name in PINS}}
    budget = {name: int(name == attr) for name in PINS}
    _check_census(tmp_path, owners, budget)

    listed.write_text(f"x.{attr}\nx.{attr}\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        _check_census(tmp_path, owners, budget)
    listed.write_text(f"x.{attr}\n", encoding="utf-8")
    (tmp_path / "new.py").write_text(f"x.{attr}\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        _check_census(tmp_path, owners, budget)
    (tmp_path / "new.py").unlink()

    listed.unlink()
    with pytest.raises(AssertionError):
        _check_census(tmp_path, owners, budget)
    owners["listed.py"][attr] = 0
    _check_census(tmp_path, owners, budget)
    del owners["listed.py"]
    _check_census(tmp_path, owners, budget)
    ordinary.unlink()
    with pytest.raises(AssertionError):
        _check_census(tmp_path, owners, budget)
    budget[attr] = 0
    _check_census(tmp_path, owners, budget)


@pytest.mark.parametrize("attr", sorted(PINS))
def test_identity_census_rejects_growth_in_zero_valued_owner_dimension(tmp_path, attr):
    (tmp_path / "listed.py").write_text(f"x.{attr}\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        _check_census(
            tmp_path, {"listed.py": dict.fromkeys(PINS, 0)}, dict.fromkeys(PINS, 0)
        )
