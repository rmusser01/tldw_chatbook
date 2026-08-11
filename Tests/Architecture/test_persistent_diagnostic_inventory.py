from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

from Tests.LLM_Calls.summarization_diagnostic_guard import (
    DiagnosticCall,
    discover_diagnostic_calls,
)
from scripts import check_persistent_diagnostic_inventory as diagnostic_inventory


REPO_ROOT = Path(__file__).resolve().parents[2]


REVIEWED_METADATA_ONLY_DIAGNOSTICS = {
    "tldw_chatbook/Chat/console_chat_controller.py": {
        "Console global user display-name accessor failed": ("type(exc).__name__",),
    },
    "tldw_chatbook/Chat/console_chat_store.py": {
        "Failed to persist seeded Console roleplay context": (),
        "Failed to persist planned Console roleplay system prompt projection": (
            "type(exc).__name__",
        ),
        "Failed to persist planned Console roleplay message projection": (
            "type(exc).__name__",
        ),
        "Failed to enqueue roleplay Sync v2 chat message": ("type(exc).__name__",),
        "Failed to persist Console roleplay message projection": (
            "type(exc).__name__",
        ),
        "Failed to persist Console roleplay system prompt projection": (
            "type(exc).__name__",
        ),
        "Failed to persist Console roleplay identity context": ("type(exc).__name__",),
        "Failed to flush Console roleplay context on first persist": (),
    },
    "tldw_chatbook/Local_Ingestion/audio_processing.py": {
        "Time-range trim could not be converted": (),
    },
    "tldw_chatbook/Media_Playback/frame_source.py": {
        "decode stopped early": ("type(exc).__name__",),
    },
    "tldw_chatbook/RAG_Search/eval/regression.py": {
        "Saved metric baseline": ("len(metrics)",),
    },
    "tldw_chatbook/RAG_Search/simplified/rag_service.py": {
        "has no FTS5-searchable": (),
        "Keyword search found": ("len(results)", "len(rankings)"),
        "Media keyword sub-leg found": ("len(results)",),
        "Media keyword sub-leg failed": ("type(e).__name__",),
        "ChaChaNotes keyword sub-legs failed": ("type(e).__name__",),
        "Could not resolve the ChaChaNotes database path": ("type(e).__name__",),
        "Rejected chachanotes_db_path from config": ("type(e).__name__",),
        "ChaChaNotes database not found": (),
        "Could not open the ChaChaNotes database": ("type(e).__name__",),
        "Notes keyword sub-leg failed": ("type(e).__name__",),
        "Conversations keyword sub-leg failed": ("type(e).__name__",),
        "Query truncated": ("len(query)", "MAX_QUERY_LENGTH"),
    },
    "tldw_chatbook/UI/Screens/chat_screen.py": {
        "retention sweep failed": (),
        "Video generation raised": ("type(exc).__name__",),
        "video play failed": ("type(exc).__name__",),
        "save-video copy failed": (),
        "Video regeneration raised": ("type(exc).__name__",),
        "stream resolution failed": ("type(exc).__name__",),
    },
    "tldw_chatbook/UI/Screens/library_screen.py": {
        "in bulk delete": (),
        "Failed to restore a Library media item in bulk-delete undo": (
            "type(exc).__name__",
        ),
    },
    "tldw_chatbook/UI/Console_Modules/session.py": {
        "Character swap: roleplay template seed failed": ("type(exc).__name__",),
        "Start Chat: roleplay template seed/persist failed": ("type(exc).__name__",),
    },
    "tldw_chatbook/UI/MCP_Modules/mcp_workbench.py": {
        "MCP Tools-mode local master save failed": ("type(exc).__name__",),
        "MCP Tools-mode workspace root save failed": ("type(exc).__name__",),
    },
    "tldw_chatbook/UI/Screens/settings_screen.py": {
        "Console identity refresh hook failed after settings save": (
            "type(screen).__name__",
            "generation",
            "type(exc).__name__",
        ),
    },
    "tldw_chatbook/UI/Screens/settings_video_gen_defaults.py": {
        "could not resolve config path": ("type(exc).__name__",),
        "could not parse video-generation config": ("type(exc).__name__",),
    },
    "tldw_chatbook/UI/Screens/video_player_screen.py": {
        "frame render skipped": ("type(exc).__name__",),
    },
    "tldw_chatbook/Video_Generation/adapter_registry.py": {
        "Failed to initialize video adapter": ("name", "type(exc).__name__"),
        "Failed to resolve video adapter class": (
            "name",
            "type(exc).__name__",
        ),
    },
    "tldw_chatbook/Video_Generation/adapters/minimax_video_adapter.py": {
        "remote task cancel failed": ("type(exc).__name__",),
    },
    "tldw_chatbook/Video_Generation/config.py": {
        "unknown-key scan failed": ("type(e).__name__",),
        "keyring lookup failed": ("backend", "type(e).__name__"),
    },
    "tldw_chatbook/Video_Generation/video_store.py": {
        "VideoStore: saved": ("len(content)",),
        "VideoStore: failed to remove": ("type(exc).__name__",),
    },
    "tldw_chatbook/Video_Generation/video_templates.py": {
        "is not a table": (),
        "has unknown keys": ("len(unknown)",),
        "has no prompt_suffix": (),
    },
    "tldw_chatbook/Widgets/Console/console_video_preview.py": {
        "peer pause failed": (),
        "decode loop ended early": ("type(exc).__name__",),
        "frame render skipped": ("type(exc).__name__",),
    },
    "tldw_chatbook/Widgets/settings_agents_panel.py": {
        "could not open agent runs database": ("type(exc).__name__",),
    },
    "tldw_chatbook/app.py": {
        "Config load failure warning failed": ("type(e).__name__",),
    },
    "tldw_chatbook/config.py": {
        "Invalid chat display name in [chat_defaults]": (),
        "Refusing to write CLI config": ("type(exc).__name__",),
    },
}


def test_production_diagnostic_inventory_and_sink_topology_are_unchanged() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_persistent_diagnostic_inventory.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_reviewed_diagnostic_changes_are_metadata_only() -> None:
    """TASK-14651: reviewed drift cannot persist private values or tracebacks."""
    failures: list[str] = []
    for relative, expected_by_label in REVIEWED_METADATA_ONLY_DIAGNOSTICS.items():
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative)
        logger_symbols = diagnostic_inventory._logger_symbols(tree)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and diagnostic_inventory._is_diagnostic_call(node, logger_symbols)
            and node.args
        ]
        for label, expected_fields in expected_by_label.items():
            matches = [call for call in calls if label in ast.unparse(call.args[0])]
            if len(matches) != 1:
                failures.append(
                    f"{relative}: expected one diagnostic containing {label!r}, "
                    f"found {len(matches)}"
                )
                continue
            call = matches[0]
            fields = [
                ast.unparse(node.value)
                for node in ast.walk(call.args[0])
                if isinstance(node, ast.FormattedValue)
            ]
            fields.extend(ast.unparse(argument) for argument in call.args[1:])
            fields.extend(
                ast.unparse(node.value)
                for node in ast.walk(call.func)
                if isinstance(node, ast.keyword) and node.arg != "exception"
            )
            fields.extend(
                ast.unparse(keyword.value)
                for keyword in call.keywords
                if keyword.arg not in {"exc_info", "stack_info", "stacklevel"}
            )
            captures_exception = (
                (isinstance(call.func, ast.Attribute) and call.func.attr == "exception")
                or any(
                    isinstance(node, ast.keyword)
                    and node.arg == "exception"
                    and not (
                        isinstance(node.value, ast.Constant)
                        and node.value.value is False
                    )
                    for node in ast.walk(call.func)
                )
                or any(
                    keyword.arg in {"exc_info", "stack_info"}
                    and not (
                        isinstance(keyword.value, ast.Constant)
                        and keyword.value.value in {False, None}
                    )
                    for keyword in call.keywords
                )
            )
            if sorted(fields) != sorted(expected_fields):
                failures.append(
                    f"{relative}: {label!r} fields {fields!r}, "
                    f"expected {list(expected_fields)!r}"
                )
            if captures_exception:
                failures.append(
                    f"{relative}: {label!r} captures exception or stack details"
                )

    assert failures == []


def _run_metadata_guard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: str
) -> None:
    relative = "tldw_chatbook/reviewed_diagnostic.py"
    reviewed_module = tmp_path / relative
    reviewed_module.parent.mkdir(parents=True)
    reviewed_module.write_text(source, encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        sys.modules[__name__],
        "REVIEWED_METADATA_ONLY_DIAGNOSTICS",
        {relative: {"reviewed diagnostic": ()}},
    )
    test_reviewed_diagnostic_changes_are_metadata_only()


def test_metadata_guard_rejects_dynamic_exception_capture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = (
        "from loguru import logger\n"
        "def emit(exc: Exception) -> None:\n"
        "    logger.opt(exception=exc).warning('reviewed diagnostic')\n"
    )

    with pytest.raises(AssertionError, match="captures exception or stack details"):
        _run_metadata_guard(monkeypatch, tmp_path, source)


def test_metadata_guard_allows_explicitly_disabled_exception_capture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = (
        "from loguru import logger\n"
        "def emit() -> None:\n"
        "    logger.opt(exception=False).warning('reviewed diagnostic')\n"
    )

    _run_metadata_guard(monkeypatch, tmp_path, source)


def test_metadata_guard_rejects_bound_private_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = (
        "from loguru import logger\n"
        "def emit(session_id: str) -> None:\n"
        "    logger.bind(session_id=session_id).warning('reviewed diagnostic')\n"
    )

    with pytest.raises(AssertionError, match="fields.*session_id"):
        _run_metadata_guard(monkeypatch, tmp_path, source)


def test_metadata_guard_rejects_keyword_private_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = (
        "from loguru import logger\n"
        "def emit(secret: str) -> None:\n"
        "    logger.warning('reviewed diagnostic: {secret}', secret=secret)\n"
    )

    with pytest.raises(AssertionError, match="fields.*secret"):
        _run_metadata_guard(monkeypatch, tmp_path, source)


def test_metadata_guard_rejects_stdlib_exception_capture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = (
        "import logging\n"
        "def emit(exc: Exception) -> None:\n"
        "    logging.getLogger(__name__).warning(\n"
        "        'reviewed diagnostic', exc_info=exc\n"
        "    )\n"
    )

    with pytest.raises(AssertionError, match="captures exception or stack details"):
        _run_metadata_guard(monkeypatch, tmp_path, source)


def test_persistent_metadata_marker_cannot_be_forged_outside_its_owner() -> None:
    marker = "_tldw_metadata_only_record"
    allowed = {
        "tldw_chatbook/Utils/persistent_diagnostics.py",
    }
    offenders = []
    for path in (REPO_ROOT / "tldw_chatbook").rglob("*.py"):
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative in allowed:
            continue
        if marker in path.read_text(encoding="utf-8"):
            offenders.append(relative)
    assert offenders == [], (
        "persistent metadata admission marker used outside its sole owner: "
        + ", ".join(offenders)
    )


def _digest(source: str) -> str:
    diagnostics, _sinks = diagnostic_inventory.scan_source(source)
    return diagnostic_inventory.diagnostic_digest(diagnostics)


BASE_MODULE = (
    "from loguru import logger\n"
    "\n"
    "\n"
    "def alpha() -> None:\n"
    "    value = 1\n"
    "    logger.info('alpha ran')\n"
    "    return value\n"
    "\n"
    "\n"
    "def beta() -> None:\n"
    "    logger.error('beta failed')\n"
)


def test_moving_a_logger_call_within_a_file_does_not_change_the_digest() -> None:
    """AC1: pure movement is not a review event."""
    moved_down = (
        "from loguru import logger\n"
        "\n"
        "\n"
        "def alpha() -> None:\n"
        "    value = 1\n"
        "    # a comment inserted above the call shifts every line below it\n"
        "\n"
        "    logger.info('alpha ran')\n"
        "    return value\n"
        "\n"
        "\n"
        "def beta() -> None:\n"
        "    logger.error('beta failed')\n"
    )
    # Relocating a call into the *other* function, and swapping the order of
    # the two diagnostics, is still pure movement: the statements themselves
    # are untouched.
    relocated = (
        "from loguru import logger\n"
        "\n"
        "\n"
        "def alpha() -> None:\n"
        "    value = 1\n"
        "    return value\n"
        "\n"
        "\n"
        "def beta() -> None:\n"
        "    logger.error('beta failed')\n"
        "    logger.info('alpha ran')\n"
    )

    assert _digest(moved_down) == _digest(BASE_MODULE)
    assert _digest(relocated) == _digest(BASE_MODULE)


def test_editing_a_diagnostic_changes_the_digest() -> None:
    """AC2: reword, re-level, add, and remove all remain review events."""
    reworded = BASE_MODULE.replace("'alpha ran'", "'alpha ran with 3 items'")
    downgraded = BASE_MODULE.replace("logger.error(", "logger.debug(")
    added = BASE_MODULE + "    logger.warning('beta warned')\n"
    removed = BASE_MODULE.replace("    logger.error('beta failed')\n", "    pass\n")

    baseline = _digest(BASE_MODULE)
    for label, mutated in (
        ("reworded message", reworded),
        ("level downgraded to debug", downgraded),
        ("diagnostic added", added),
        ("diagnostic removed", removed),
    ):
        assert _digest(mutated) != baseline, f"{label} did not change the digest"


def test_duplicate_diagnostics_are_digested_with_multiplicity() -> None:
    """Two identical calls are not one: deleting a twin must still be caught."""
    twice = (
        "from loguru import logger\n"
        "\n"
        "\n"
        "def alpha() -> None:\n"
        "    logger.info('same text')\n"
        "    logger.info('same text')\n"
    )
    once = (
        "from loguru import logger\n"
        "\n"
        "\n"
        "def alpha() -> None:\n"
        "    logger.info('same text')\n"
    )

    assert _digest(twice) != _digest(once)


def test_sink_entries_are_position_independent_but_still_content_sensitive() -> None:
    """Sinks carry a scope, not a line: they move quietly, change loudly."""
    base = (
        "import logging\n"
        "\n"
        "\n"
        "def configure() -> None:\n"
        "    root = logging.getLogger()\n"
        "    root.addHandler(logging.FileHandler('/var/log/app.log'))\n"
    )
    moved = base.replace(
        "    root = logging.getLogger()\n",
        "    root = logging.getLogger()\n    root.setLevel(logging.INFO)\n",
    )
    retargeted = base.replace("/var/log/app.log", "/tmp/app.log")

    _diagnostics, base_sinks = diagnostic_inventory.scan_source(base)
    _diagnostics, moved_sinks = diagnostic_inventory.scan_source(moved)
    _diagnostics, retargeted_sinks = diagnostic_inventory.scan_source(retargeted)

    assert base_sinks == moved_sinks
    assert base_sinks != retargeted_sinks
    assert all("line" not in entry for entry in base_sinks)
    assert {entry["scope"] for entry in base_sinks} == {"configure"}


def test_shifting_a_real_inventory_file_leaves_its_digest_unchanged() -> None:
    """End-to-end on shipped source, not just a synthetic fixture."""
    inventory = json.loads(
        (REPO_ROOT / "Docs/security/production-diagnostic-inventory.json").read_text(
            encoding="utf-8"
        )
    )
    entry = next(item for item in inventory["owners"] if item["call_count"] >= 3)
    source = (REPO_ROOT / entry["path"]).read_text(encoding="utf-8")
    diagnostics, _sinks = diagnostic_inventory.scan_source(source)

    assert len(diagnostics) == entry["call_count"]
    assert (
        diagnostic_inventory.diagnostic_digest(diagnostics)
        == (entry["diagnostic_digest"])
    )
    # Prepending blank lines shifts every line number in the file.
    assert _digest("\n\n\n" + source) == entry["diagnostic_digest"]


def test_inventory_counts_chained_logger_diagnostic_calls() -> None:
    tree = ast.parse(
        "import logging\n"
        "from loguru import logger\n"
        "logger.opt(exception=True).warning('persisted diagnostic')\n"
        "logger.bind(component='test').error('bound diagnostic')\n"
        "logging.getLogger(__name__).info('standard diagnostic')\n"
    )
    logger_symbols = diagnostic_inventory._logger_symbols(tree)
    diagnostic_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and diagnostic_inventory._is_diagnostic_call(node, logger_symbols)
    ]

    assert len(diagnostic_calls) == 3


TASK_15103_REVIEW_PATH = REPO_ROOT / "Docs/security/task-15103-diagnostic-review.json"
TASK_15103_RECORDED_BASE = "6d72f15f8332b6469a5d644d409b80914634a8dd"
TASK_15103_PLANNING_BASE = "85863257dd7a30b16451f8f32e0c7142dd1d5273"
TASK_15103_OWNER_STARTING = {
    "tldw_chatbook/Agents/agent_service.py": (9, "578de6bb91649fc9fc87"),
    "tldw_chatbook/Chat/console_agent_bridge.py": (
        12,
        "7caa9d8c2694081e94e9",
    ),
    "tldw_chatbook/Chat/console_chat_controller.py": (
        35,
        "5361a9926d2d6bede509",
    ),
    "tldw_chatbook/Chat/console_chat_store.py": (
        40,
        "354cf52f8e1d76bbb9b8",
    ),
    "tldw_chatbook/Chat/console_context_compaction.py": (
        2,
        "ad596e6fde321c7720f4",
    ),
    "tldw_chatbook/Chat/console_provider_gateway.py": (
        2,
        "747b675587167a1bec60",
    ),
    "tldw_chatbook/MCP/client.py": (48, "7110f1c19a6a982f290a"),
    "tldw_chatbook/MCP/local_server_tools.py": (
        1,
        "5a6a84d5e177534348d2",
    ),
    "tldw_chatbook/MCP/prompts.py": (1, "af163025ed5780b49a46"),
    "tldw_chatbook/MCP/server.py": (8, "86cbb8127d9b610be073"),
    "tldw_chatbook/RAG_Search/fusion.py": (8, "97d06c55271a5f01c549"),
    "tldw_chatbook/RAG_Search/simplified/rag_service.py": (
        57,
        "283cb4ecf295fff754bc",
    ),
    "tldw_chatbook/RAG_Search/simplified/search_service.py": (
        5,
        "57ac009dd80f8d7028b8",
    ),
    "tldw_chatbook/UI/Console_Modules/session.py": (
        9,
        "a748c6b9e50e24ec30f5",
    ),
    "tldw_chatbook/UI/Screens/chat_screen.py": (
        158,
        "eef7b929c039940e51f5",
    ),
    "tldw_chatbook/UI/Screens/library_screen.py": (
        84,
        "c14a8222d35aec3a6e34",
    ),
    "tldw_chatbook/app.py": (305, "dec5c30c1ad1b1b1c8fc"),
    "tldw_chatbook/UI/Screens/settings_screen.py": (
        31,
        "62ea61e3ba363d516a6e",
    ),
}
_TASK_15103_DISPOSITIONS = {
    "reviewed-safe",
    "metadata-repair",
    "justified-deletion",
}
TASK_15103_EXPECTED_DISPOSITION_COUNTS = {
    "reviewed-safe": 45,
    "metadata-repair": 40,
    "justified-deletion": 12,
}
TASK_15103_EXPECTED_ATOM_MULTIPLICITY = {
    "tldw_chatbook/Agents/agent_service.py": (9, 9),
    "tldw_chatbook/Chat/console_agent_bridge.py": (0, 1),
    "tldw_chatbook/Chat/console_chat_controller.py": (10, 8),
    "tldw_chatbook/Chat/console_chat_store.py": (4, 4),
    "tldw_chatbook/Chat/console_context_compaction.py": (2, 2),
    "tldw_chatbook/Chat/console_provider_gateway.py": (1, 1),
    "tldw_chatbook/MCP/client.py": (10, 36),
    "tldw_chatbook/MCP/local_server_tools.py": (1, 1),
    "tldw_chatbook/MCP/prompts.py": (5, 1),
    "tldw_chatbook/MCP/server.py": (4, 2),
    "tldw_chatbook/RAG_Search/fusion.py": (6, 3),
    "tldw_chatbook/RAG_Search/simplified/rag_service.py": (4, 5),
    "tldw_chatbook/RAG_Search/simplified/search_service.py": (1, 1),
    "tldw_chatbook/UI/Console_Modules/session.py": (1, 1),
    "tldw_chatbook/UI/Screens/chat_screen.py": (0, 0),
    "tldw_chatbook/UI/Screens/library_screen.py": (2, 2),
    "tldw_chatbook/app.py": (12, 9),
    "tldw_chatbook/UI/Screens/settings_screen.py": (0, 1),
}
_TASK_15103_SEVERITIES = {
    "critical",
    "debug",
    "error",
    "info",
    "log",
    "success",
    "trace",
    "warning",
}


def _task_15103_exact_keys(
    value: Any, expected: set[str], *, location: str
) -> dict[str, Any]:
    assert isinstance(value, dict), f"{location} must be an object"
    actual = set(value)
    assert actual == expected, (
        f"{location} fields differ: missing={sorted(expected - actual)!r}, "
        f"unknown={sorted(actual - expected)!r}"
    )
    return value


def _task_15103_is_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and re.fullmatch(r"[0-9a-f]+", value) is not None
    )


def _task_15103_validate_owner_pair(value: Any, *, location: str) -> None:
    pair = _task_15103_exact_keys(
        value, {"call_count", "diagnostic_digest"}, location=location
    )
    assert type(pair["call_count"]) is int and pair["call_count"] >= 0, (
        f"{location}.call_count must be a non-negative integer"
    )
    assert _task_15103_is_hex(pair["diagnostic_digest"], 20), (
        f"{location}.diagnostic_digest must be 20 lowercase hex characters"
    )


def _task_15103_validate_semantic_pair(value: Any, *, location: str) -> None:
    pair = _task_15103_exact_keys(
        value, {"call_count", "semantic_digest"}, location=location
    )
    assert type(pair["call_count"]) is int and pair["call_count"] >= 0, (
        f"{location}.call_count must be a non-negative integer"
    )
    assert _task_15103_is_hex(pair["semantic_digest"], 64), (
        f"{location}.semantic_digest must be a full lowercase SHA-256"
    )


def _task_15103_validate_provenance(value: Any, *, location: str) -> None:
    provenance = _task_15103_exact_keys(
        value, set(value) if isinstance(value, dict) else set(), location=location
    )
    assert len(provenance) == 1, (
        f"{location} must contain exactly one exact_commit or verified_range"
    )
    if "exact_commit" in provenance:
        assert _task_15103_is_hex(provenance["exact_commit"], 40), (
            f"{location}.exact_commit must be a full lowercase commit SHA"
        )
        return
    assert set(provenance) == {"verified_range"}, (
        f"{location} must contain exact_commit or verified_range"
    )
    verified_range = _task_15103_exact_keys(
        provenance["verified_range"],
        {"start_exclusive", "end_inclusive"},
        location=f"{location}.verified_range",
    )
    assert _task_15103_is_hex(verified_range["start_exclusive"], 40)
    assert _task_15103_is_hex(verified_range["end_inclusive"], 40)
    assert verified_range["start_exclusive"] != verified_range["end_inclusive"], (
        f"{location}.verified_range cannot have identical endpoints"
    )


def _task_15103_validate_atom(value: Any, *, location: str) -> None:
    atom = _task_15103_exact_keys(
        value,
        {"method", "semantic_digest", "multiplicity_delta"}
        | (
            {"qualified_scope"}
            if isinstance(value, dict) and "qualified_scope" in value
            else set()
        ),
        location=location,
    )
    assert isinstance(atom["method"], str) and atom["method"], (
        f"{location}.method must be non-empty"
    )
    assert _task_15103_is_hex(atom["semantic_digest"], 64), (
        f"{location}.semantic_digest must be a full lowercase SHA-256"
    )
    assert type(atom["multiplicity_delta"]) is int and atom["multiplicity_delta"] > 0, (
        f"{location}.multiplicity_delta must be a positive integer"
    )
    if "qualified_scope" in atom:
        assert isinstance(atom["qualified_scope"], str) and atom["qualified_scope"], (
            f"{location}.qualified_scope must be non-empty"
        )


def _task_15103_validate_checkpoint_evidence(value: Any, *, location: str) -> None:
    evidence = _task_15103_exact_keys(
        value,
        {"base", "head", "aggregate", "owners"},
        location=location,
    )
    assert _task_15103_is_hex(evidence["base"], 40)
    assert _task_15103_is_hex(evidence["head"], 40)
    _task_15103_validate_semantic_pair(
        evidence["aggregate"], location=f"{location}.aggregate"
    )
    assert isinstance(evidence["owners"], list), f"{location}.owners must be a list"
    checkpoint_paths: list[str] = []
    for index, owner in enumerate(evidence["owners"]):
        owner_record = _task_15103_exact_keys(
            owner,
            {"path", "evidence"},
            location=f"{location}.owners[{index}]",
        )
        checkpoint_paths.append(owner_record["path"])
        _task_15103_validate_semantic_pair(
            owner_record["evidence"],
            location=f"{location}.owners[{index}].evidence",
        )
    assert set(checkpoint_paths) == set(TASK_15103_OWNER_STARTING), (
        f"{location}.owners must contain the exact 18-owner path set"
    )
    assert len(checkpoint_paths) == len(set(checkpoint_paths)), (
        f"{location}.owners contains duplicate paths"
    )


def _task_15103_validate_review_ledger(ledger: Any) -> None:
    assert isinstance(ledger, dict), "TASK-15103 ledger must be an object"
    status = ledger.get("review_status")
    assert status in {"planned", "reviewed"}, (
        "review_status must be planned or reviewed"
    )
    top_level = {
        "schema_version",
        "review_status",
        "incident",
        "owners",
        "change_groups",
    }
    if "integration_checkpoint" in ledger:
        top_level.add("integration_checkpoint")
    _task_15103_exact_keys(ledger, top_level, location="ledger")
    assert type(ledger["schema_version"]) is int and ledger["schema_version"] == 1, (
        "schema_version must be integer 1"
    )

    incident_fields = {"recorded_base", "planning_base"}
    if status == "reviewed":
        incident_fields.add("final_base")
    incident = _task_15103_exact_keys(
        ledger["incident"], incident_fields, location="incident"
    )
    assert incident["recorded_base"] == TASK_15103_RECORDED_BASE
    assert incident["planning_base"] == TASK_15103_PLANNING_BASE
    if status == "reviewed":
        assert _task_15103_is_hex(incident["final_base"], 40)

    assert isinstance(ledger["owners"], list), "owners must be a list"
    owner_paths: list[str] = []
    for index, owner in enumerate(ledger["owners"]):
        owner_fields = {"path", "starting"}
        if status == "reviewed":
            owner_fields.add("reviewed_final")
        owner_record = _task_15103_exact_keys(
            owner, owner_fields, location=f"owners[{index}]"
        )
        path = owner_record["path"]
        assert isinstance(path, str), f"owners[{index}].path must be a string"
        owner_paths.append(path)
        _task_15103_validate_owner_pair(
            owner_record["starting"], location=f"owners[{index}].starting"
        )
        if status == "reviewed":
            _task_15103_validate_owner_pair(
                owner_record["reviewed_final"],
                location=f"owners[{index}].reviewed_final",
            )
    assert set(owner_paths) == set(TASK_15103_OWNER_STARTING), (
        "owners must contain the exact 18-owner path set"
    )
    assert len(owner_paths) == len(set(owner_paths)), "owners contains duplicate paths"

    assert isinstance(ledger["change_groups"], list) and ledger["change_groups"], (
        "change_groups must be a non-empty list"
    )
    group_ids: list[str] = []
    for index, group in enumerate(ledger["change_groups"]):
        location = f"change_groups[{index}]"
        group_record = _task_15103_exact_keys(
            group,
            {
                "id",
                "owner_path",
                "provenance",
                "disposition",
                "rationale",
                "fixed_event",
                "severity",
                "permitted_fields",
                "captures_exception",
                "removed",
                "proposed_surviving",
            },
            location=location,
        )
        group_id = group_record["id"]
        assert isinstance(group_id, str) and re.fullmatch(
            r"TASK-15103-G[0-9]{3}", group_id
        ), f"{location}.id must be a stable TASK-15103-GNNN identifier"
        group_ids.append(group_id)
        assert group_record["owner_path"] in TASK_15103_OWNER_STARTING, (
            f"{location}.owner_path is outside the approved 18-owner set"
        )
        _task_15103_validate_provenance(
            group_record["provenance"], location=f"{location}.provenance"
        )
        assert group_record["disposition"] in _TASK_15103_DISPOSITIONS, (
            f"{location}.disposition is not approved"
        )
        assert (
            isinstance(group_record["rationale"], str)
            and group_record["rationale"].strip()
        )
        assert (
            isinstance(group_record["fixed_event"], str) and group_record["fixed_event"]
        )
        assert group_record["severity"] in _TASK_15103_SEVERITIES
        assert type(group_record["captures_exception"]) is bool
        assert isinstance(group_record["permitted_fields"], list)
        expressions: list[str] = []
        for field_index, field in enumerate(group_record["permitted_fields"]):
            permitted = _task_15103_exact_keys(
                field,
                {"expression", "provenance"},
                location=f"{location}.permitted_fields[{field_index}]",
            )
            assert isinstance(permitted["expression"], str) and permitted["expression"]
            assert (
                isinstance(permitted["provenance"], str)
                and permitted["provenance"].strip()
            )
            expressions.append(permitted["expression"])
        assert len(expressions) == len(set(expressions)), (
            f"{location}.permitted_fields contains duplicate expressions"
        )
        assert isinstance(group_record["removed"], list)
        assert isinstance(group_record["proposed_surviving"], list)
        assert group_record["removed"] or group_record["proposed_surviving"], (
            f"{location} must remove or propose at least one atom"
        )
        for side in ("removed", "proposed_surviving"):
            digests: list[str] = []
            for atom_index, atom in enumerate(group_record[side]):
                _task_15103_validate_atom(
                    atom, location=f"{location}.{side}[{atom_index}]"
                )
                digests.append(atom["semantic_digest"])
            assert len(digests) == len(set(digests)), (
                f"{location}.{side} must aggregate identical atoms by multiplicity"
            )
    assert len(group_ids) == len(set(group_ids)), "change_groups contains duplicate ids"

    if "integration_checkpoint" in ledger:
        assert status == "reviewed", (
            "integration_checkpoint is forbidden before reviewed Task-8 integration"
        )
        checkpoint = ledger["integration_checkpoint"]
        assert isinstance(checkpoint, dict)
        checkpoint_state = checkpoint.get("state")
        assert checkpoint_state in {"pre_rebase", "post_rebase"}
        checkpoint_fields = {"state", "pre_rebase"}
        if checkpoint_state == "post_rebase":
            checkpoint_fields.add("post_rebase")
        _task_15103_exact_keys(
            checkpoint, checkpoint_fields, location="integration_checkpoint"
        )
        _task_15103_validate_checkpoint_evidence(
            checkpoint["pre_rebase"],
            location="integration_checkpoint.pre_rebase",
        )
        if checkpoint_state == "post_rebase":
            _task_15103_validate_checkpoint_evidence(
                checkpoint["post_rebase"],
                location="integration_checkpoint.post_rebase",
            )


def _task_15103_semantic_contract(call: DiagnosticCall) -> dict[str, Any]:
    return {
        "method": call.method,
        "event": call.event,
        "message_shape": call.message_shape,
        "expressions": list(call.expressions),
        "captures_exception": call.captures_exception,
        "level_expression": call.level_expression,
    }


def _task_15103_semantic_digest(contract: dict[str, Any]) -> str:
    compact = json.dumps(contract, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(compact.encode("utf-8")).hexdigest()


_TASK_15103_SPEC_BLOBS: dict[tuple[str, str], str | None] = {}
_TASK_15103_BLOB_SOURCES: dict[str, str] = {}
_TASK_15103_BLOB_CALLS: dict[str, tuple[DiagnosticCall, ...]] = {}
_TASK_15103_BLOB_MANIFEST_PAIRS: dict[str, tuple[int, str]] = {}


def _task_15103_preload_git_sources(revision_paths: list[tuple[str, str]]) -> None:
    pending = list(dict.fromkeys(revision_paths))
    pending = [pair for pair in pending if pair not in _TASK_15103_SPEC_BLOBS]
    if not pending:
        return
    specs = [f"{revision}:{path}" for revision, path in pending]
    resolved = subprocess.run(
        ["git", "cat-file", "--batch-check"],
        cwd=REPO_ROOT,
        input="\n".join(specs) + "\n",
        capture_output=True,
        check=True,
        text=True,
    ).stdout.splitlines()
    assert len(resolved) == len(pending), "Git batch-check result count mismatch"
    for pair, line in zip(pending, resolved, strict=True):
        fields = line.split()
        _TASK_15103_SPEC_BLOBS[pair] = (
            fields[0] if len(fields) == 3 and fields[1] == "blob" else None
        )

    missing_blobs = list(
        dict.fromkeys(
            blob
            for pair in pending
            if (blob := _TASK_15103_SPEC_BLOBS[pair]) is not None
            and blob not in _TASK_15103_BLOB_SOURCES
        )
    )
    if not missing_blobs:
        return
    output = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=REPO_ROOT,
        input=("\n".join(missing_blobs) + "\n").encode(),
        capture_output=True,
        check=True,
    ).stdout
    offset = 0
    for expected_blob in missing_blobs:
        newline = output.index(b"\n", offset)
        header = output[offset:newline].decode().split()
        assert len(header) == 3 and header[0] == expected_blob and header[1] == "blob"
        size = int(header[2])
        start = newline + 1
        end = start + size
        assert output[end : end + 1] == b"\n"
        _TASK_15103_BLOB_SOURCES[expected_blob] = output[start:end].decode("utf-8")
        offset = end + 1
    assert offset == len(output), "Git batch output contains trailing data"


def _task_15103_blob_at(revision: str, path: str) -> str | None:
    pair = (revision, path)
    _task_15103_preload_git_sources([pair])
    return _TASK_15103_SPEC_BLOBS[pair]


def _task_15103_git_source(revision: str, path: str) -> str | None:
    blob = _task_15103_blob_at(revision, path)
    return _TASK_15103_BLOB_SOURCES[blob] if blob is not None else None


def _task_15103_calls_at(revision: str, path: str) -> tuple[DiagnosticCall, ...]:
    blob = _task_15103_blob_at(revision, path)
    if blob is None:
        return ()
    if blob not in _TASK_15103_BLOB_CALLS:
        _TASK_15103_BLOB_CALLS[blob] = tuple(
            discover_diagnostic_calls(_TASK_15103_BLOB_SOURCES[blob], module=path)
        )
    return _TASK_15103_BLOB_CALLS[blob]


def _task_15103_population(
    calls: tuple[DiagnosticCall, ...],
) -> tuple[Counter[str], dict[str, DiagnosticCall]]:
    population: Counter[str] = Counter()
    details: dict[str, DiagnosticCall] = {}
    for call in calls:
        digest = _task_15103_semantic_digest(_task_15103_semantic_contract(call))
        population[digest] += 1
        previous = details.setdefault(digest, call)
        assert _task_15103_semantic_contract(previous) == (
            _task_15103_semantic_contract(call)
        ), "semantic SHA-256 collision"
    return population, details


@lru_cache(maxsize=None)
def _task_15103_population_at(
    revision: str, path: str
) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[str, DiagnosticCall], ...]]:
    population, details = _task_15103_population(_task_15103_calls_at(revision, path))
    return tuple(sorted(population.items())), tuple(sorted(details.items()))


def _task_15103_population_and_details_at(
    revision: str, path: str
) -> tuple[Counter[str], dict[str, DiagnosticCall]]:
    population, details = _task_15103_population_at(revision, path)
    return Counter(dict(population)), dict(details)


def _task_15103_manifest_pair_at(revision: str, path: str) -> tuple[int, str]:
    blob = _task_15103_blob_at(revision, path)
    if blob is None:
        return 0, diagnostic_inventory.diagnostic_digest([])
    if blob not in _TASK_15103_BLOB_MANIFEST_PAIRS:
        source = _TASK_15103_BLOB_SOURCES[blob]
        tree = ast.parse(source, filename=f"{revision}:{path}")
        logger_symbols = diagnostic_inventory._logger_symbols(tree)
        source_lines = ast._splitlines_no_ff(source)
        diagnostics: list[dict[str, str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not (
                diagnostic_inventory._is_diagnostic_call(node, logger_symbols)
            ):
                continue
            start_line = node.lineno - 1
            end_line = node.end_lineno - 1
            if start_line == end_line:
                segment = (
                    source_lines[start_line]
                    .encode()[node.col_offset : node.end_col_offset]
                    .decode()
                )
            else:
                first = source_lines[start_line].encode()[node.col_offset :].decode()
                last = source_lines[end_line].encode()[: node.end_col_offset].decode()
                middle = source_lines[start_line + 1 : end_line]
                segment = "".join((first, *middle, last))
            diagnostics.append(
                {
                    "method": (
                        node.func.attr
                        if isinstance(node.func, ast.Attribute)
                        else node.func.id
                        if isinstance(node.func, ast.Name)
                        else "call"
                    ),
                    "digest": hashlib.sha256(segment.encode("utf-8")).hexdigest()[:16],
                }
            )
        _TASK_15103_BLOB_MANIFEST_PAIRS[blob] = (
            len(diagnostics),
            diagnostic_inventory.diagnostic_digest(diagnostics),
        )
    return _TASK_15103_BLOB_MANIFEST_PAIRS[blob]


@lru_cache(maxsize=1)
def _task_15103_complete_history(planning_base: str) -> dict[str, Any]:
    """Reconstruct every owner transition independently of ledger provenance."""
    stored_inventory = json.loads(
        diagnostic_inventory.INVENTORY_PATH.read_text(encoding="utf-8")
    )
    stored_rows = {row["path"]: row for row in stored_inventory["owners"]}
    histories: dict[str, list[tuple[str, str]]] = {}
    for path in TASK_15103_OWNER_STARTING:
        result = subprocess.run(
            [
                "git",
                "log",
                "--follow",
                "--format=%H%x00%P",
                planning_base,
                "--",
                path,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
            text=True,
        )
        entries: list[tuple[str, str]] = []
        for line in result.stdout.splitlines():
            commit, parents = line.split("\0", 1)
            entries.append((commit, parents.split()[0] if parents else ""))
        assert entries, f"no Git history found for approved owner {path}"
        histories[path] = entries

    stored_matches: dict[str, tuple[str, int]] = {}
    chunk_size = 16
    max_history = max(len(history) for history in histories.values())
    empty_pair = (0, diagnostic_inventory.diagnostic_digest([]))
    for offset in range(0, max_history, chunk_size):
        unresolved = [
            path for path in TASK_15103_OWNER_STARTING if path not in stored_matches
        ]
        if not unresolved:
            break
        pending = [
            (revision, path)
            for path in unresolved
            for revision, _parent in histories[path][offset : offset + chunk_size]
        ]
        _task_15103_preload_git_sources(pending)
        for path in unresolved:
            row = stored_rows.get(path)
            expected = (
                (row["call_count"], row["diagnostic_digest"])
                if row is not None
                else empty_pair
            )
            upper = min(offset + chunk_size, len(histories[path]))
            for index in range(offset, upper):
                revision = histories[path][index][0]
                if _task_15103_manifest_pair_at(revision, path) == expected:
                    stored_matches[path] = (revision, index)
                    break

    for path, history in histories.items():
        if path in stored_matches:
            continue
        oldest_parent = history[-1][1]
        assert oldest_parent, f"could not reconstruct stored population for {path}"
        _task_15103_preload_git_sources([(oldest_parent, path)])
        row = stored_rows.get(path)
        expected = (
            (row["call_count"], row["diagnostic_digest"])
            if row is not None
            else empty_pair
        )
        assert _task_15103_manifest_pair_at(oldest_parent, path) == expected, (
            f"could not reconstruct stored diagnostic population for {path}"
        )
        stored_matches[path] = (oldest_parent, len(history))

    transition_specs = [
        (revision, path)
        for path, history in histories.items()
        for revision, _parent in history[: stored_matches[path][1]]
    ] + [
        (parent, path)
        for path, history in histories.items()
        for _revision, parent in history[: stored_matches[path][1]]
        if parent
    ]
    transition_specs.extend((planning_base, path) for path in TASK_15103_OWNER_STARTING)
    _task_15103_preload_git_sources(transition_specs)

    transitions: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    denominator: Counter[tuple[str, str, str, str]] = Counter()
    for path, history in histories.items():
        for revision, parent in reversed(history[: stored_matches[path][1]]):
            assert parent, f"path transition {revision} for {path} has no parent"
            before, before_details = _task_15103_population_and_details_at(parent, path)
            after, after_details = _task_15103_population_and_details_at(revision, path)
            introduced = after - before
            removed = before - after
            if not introduced and not removed:
                continue
            transition = {
                "commit": revision,
                "parent": parent,
                "introduced": introduced,
                "removed": removed,
                "before_details": before_details,
                "after_details": after_details,
            }
            transitions[path].append(transition)
            for direction, population in (
                ("introduced", introduced),
                ("removed", removed),
            ):
                for digest, multiplicity in population.items():
                    denominator[(path, revision, direction, digest)] += multiplicity
    return {
        "recorded_revisions": {
            path: revision for path, (revision, _index) in stored_matches.items()
        },
        "transitions": dict(transitions),
        "denominator": denominator,
    }


@lru_cache(maxsize=1)
def _task_15103_current_inventory() -> dict[str, Any]:
    return diagnostic_inventory.build_inventory()


def _task_15103_atom_counter(atoms: list[dict[str, Any]]) -> Counter[str]:
    return Counter(
        {atom["semantic_digest"]: atom["multiplicity_delta"] for atom in atoms}
    )


def _task_15103_group_atom_counter(
    groups: list[dict[str, Any]], side: str
) -> Counter[str]:
    population: Counter[str] = Counter()
    for group in groups:
        population.update(_task_15103_atom_counter(group[side]))
    return population


def _task_15103_synthetic_contract(group: dict[str, Any]) -> dict[str, Any]:
    assert len(group["proposed_surviving"]) == 1, (
        f"{group['id']} metadata repair must have one aggregated target atom"
    )
    method = group["proposed_surviving"][0]["method"]
    expressions = [field["expression"] for field in group["permitted_fields"]]
    suffix = f", {', '.join(expressions)}" if expressions else ""
    source = (
        "from loguru import logger\n\n"
        "def emit():\n"
        f"    logger.{method}({group['fixed_event']!r}{suffix})\n"
    )
    calls = discover_diagnostic_calls(source, module="task_15103_synthetic")
    assert len(calls) == 1
    return _task_15103_semantic_contract(calls[0])


def _task_15103_severity(method: str) -> str:
    return "error" if method == "exception" else method


def _task_15103_assignment_values(
    source: str, name: str, *, scope: str
) -> list[ast.AST]:
    tree = ast.parse(source)
    scopes = diagnostic_inventory._scope_names(tree)
    local_values: list[ast.AST] = []
    module_values: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value = node.value
        else:
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == name for target in targets
        ):
            continue
        node_scope = scopes.get(id(node), "")
        if node_scope == scope:
            local_values.append(value)
        elif node_scope == "":
            module_values.append(value)
    return local_values or module_values


def _task_15103_contains_int_conversion(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == "int"
        for child in ast.walk(node)
    )


def _task_15103_call_returns_int(source: str, node: ast.AST) -> bool:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return False
    for candidate in ast.walk(ast.parse(source)):
        if not isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if candidate.name != node.func.id or candidate.returns is None:
            continue
        return ast.unparse(candidate.returns) == "int"
    return False


def _task_15103_expected_field_provenance(
    expression: str, *, source: str, scope: str
) -> str:
    value_text = expression
    keyword = re.fullmatch(r"[A-Za-z_]\w*=(.+)", expression)
    if keyword is not None:
        value_text = keyword.group(1)
    node = ast.parse(value_text, mode="eval").body
    if isinstance(node, ast.Constant):
        return (
            "The value is a fixed code literal, not provider-, user-, or "
            "model-controlled data."
        )
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "len"
        and len(node.args) == 1
        and not node.keywords
    ):
        return (
            "A local len(...) over in-memory capability collections yields only "
            "an integer count."
        )
    if (
        isinstance(node, ast.Attribute)
        and node.attr == "__name__"
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "type"
        and len(node.value.args) == 1
    ):
        subject = node.value.args[0]
        if isinstance(subject, ast.Name) and subject.id in {"e", "exc", "error"}:
            return (
                "The exception class name is ADR-029 metadata and excludes the "
                "exception message and traceback."
            )
        return "The runtime class name is code metadata and excludes object contents."
    if isinstance(node, ast.Name):
        values = _task_15103_assignment_values(source, node.id, scope=scope)
        assert values, f"no source assignment proves permitted field {expression!r}"
        if any(_task_15103_contains_int_conversion(value) for value in values):
            return (
                "The value has crossed the production integer conversion branch "
                "before this diagnostic."
            )
        if any(
            isinstance(value, ast.Constant) and type(value.value) in {int, float}
            for value in values
        ) or any(_task_15103_call_returns_int(source, value) for value in values):
            return (
                "The value is a code-owned numeric constant reviewed at the "
                "recorded provenance revision."
            )
    raise AssertionError(
        f"no code/source evidence proves permitted field {expression!r}"
    )


def _task_15103_assert_group_contract(
    group: dict[str, Any],
    *,
    introduced: Counter[str],
    removed: Counter[str],
    before_details: dict[str, DiagnosticCall],
    after_details: dict[str, DiagnosticCall],
    source: str,
) -> None:
    disposition = group["disposition"]
    contract: dict[str, Any]
    scope: str
    if disposition == "metadata-repair":
        contract = _task_15103_synthetic_contract(group)
        actual_removed = _task_15103_atom_counter(group["removed"]) & (
            introduced + removed
        )
        assert actual_removed, f"{group['id']} has no Git-backed atom to repair"
        digest = next(iter(actual_removed))
        scope = (after_details.get(digest) or before_details[digest]).qualname
    elif disposition == "reviewed-safe":
        assert len(group["proposed_surviving"]) == 1
        digest = group["proposed_surviving"][0]["semantic_digest"]
        assert digest in after_details, (
            f"{group['id']} semantic multiset reconciliation missing introduced atom"
        )
        call = after_details[digest]
        contract = _task_15103_semantic_contract(call)
        scope = call.qualname
    else:
        assert not group["proposed_surviving"]
        assert not group["permitted_fields"]
        digest = group["removed"][0]["semantic_digest"]
        call = before_details.get(digest) or after_details[digest]
        assert group["fixed_event"] == call.event
        assert group["severity"] == _task_15103_severity(call.method)
        return

    target_digest = _task_15103_semantic_digest(contract)
    for atom in group["proposed_surviving"]:
        assert atom["semantic_digest"] == target_digest, (
            f"{group['id']} semantic contract digest mismatch"
        )
        assert atom["method"] == contract["method"]
    assert group["fixed_event"] == contract["event"], (
        f"{group['id']} semantic contract digest mismatch"
    )
    assert group["severity"] == _task_15103_severity(contract["method"])
    assert group["captures_exception"] == contract["captures_exception"]
    expressions = list(dict.fromkeys(contract["expressions"]))
    permitted = [field["expression"] for field in group["permitted_fields"]]
    assert permitted == expressions, f"{group['id']} permitted fields mismatch"
    for field in group["permitted_fields"]:
        expected = _task_15103_expected_field_provenance(
            field["expression"], source=source, scope=scope
        )
        assert field["provenance"] == expected, (
            f"{group['id']} permitted-field provenance mismatch"
        )


def _task_15103_counter_is_subset(expected: Counter[str], actual: Counter[str]) -> bool:
    return not (expected - actual)


def _task_15103_assert_group_provenance(
    group: dict[str, Any],
) -> tuple[
    Counter[str],
    Counter[str],
    dict[str, DiagnosticCall],
    dict[str, DiagnosticCall],
    str,
    tuple[str, ...],
]:
    provenance = group["provenance"]
    path = group["owner_path"]
    if "exact_commit" in provenance:
        end = provenance["exact_commit"]
        parent = subprocess.run(
            ["git", "rev-parse", f"{end}^"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
        start = parent
    else:
        verified_range = provenance["verified_range"]
        start = verified_range["start_exclusive"]
        end = verified_range["end_inclusive"]
    before, before_details = _task_15103_population_and_details_at(start, path)
    after, after_details = _task_15103_population_and_details_at(end, path)
    introduced = after - before
    removed = before - after
    actual_positive: Counter[str] = Counter()
    actual_negative: Counter[str] = Counter()
    disposition = group["disposition"]
    for atom in group["removed"]:
        digest = atom["semantic_digest"]
        multiplicity = atom["multiplicity_delta"]
        introduced_count = introduced[digest]
        removed_count = removed[digest]
        if disposition in {"metadata-repair", "justified-deletion"} and (
            introduced_count >= multiplicity
        ):
            assert atom["method"] == after_details[digest].method, (
                f"{group['id']} removed atom method mismatch for {digest}"
            )
            actual_positive[digest] += multiplicity
        elif removed_count >= multiplicity:
            assert atom["method"] == before_details[digest].method, (
                f"{group['id']} removed atom method mismatch for {digest}"
            )
            actual_negative[digest] += multiplicity
        elif introduced_count or removed_count:
            raise AssertionError(
                f"{group['id']} semantic multiset reconciliation has wrong "
                f"multiplicity for {digest}"
            )
        else:
            raise AssertionError(
                f"{group['id']} removed atom {digest} is absent from claimed Git delta"
            )
    for atom in group["proposed_surviving"]:
        digest = atom["semantic_digest"]
        multiplicity = atom["multiplicity_delta"]
        if introduced[digest] >= multiplicity:
            assert atom["method"] == after_details[digest].method, (
                f"{group['id']} proposed atom method mismatch for {digest}"
            )
            actual_positive[digest] += multiplicity
        elif introduced[digest]:
            raise AssertionError(
                f"{group['id']} semantic multiset reconciliation has wrong "
                f"multiplicity for {digest}"
            )
        elif disposition != "metadata-repair":
            raise AssertionError(
                f"{group['id']} proposed atom {digest} is absent from claimed Git delta"
            )
    assert actual_positive or actual_negative, (
        f"{group['id']} provenance does not introduce or remove its owned atom(s)"
    )
    if "verified_range" not in provenance:
        return (
            actual_positive,
            actual_negative,
            before_details,
            after_details,
            end,
            (end,),
        )
    revisions = subprocess.run(
        ["git", "rev-list", "--ancestry-path", "--reverse", f"{start}..{end}"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.splitlines()
    relevant: list[str] = []
    for revision in revisions:
        parent = subprocess.run(
            ["git", "rev-parse", f"{revision}^"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
        step_before, _ = _task_15103_population_and_details_at(parent, path)
        step_after, _ = _task_15103_population_and_details_at(revision, path)
        if (actual_positive & (step_after - step_before)) or (
            actual_negative & (step_before - step_after)
        ):
            relevant.append(revision)
    assert relevant and relevant[-1] == end
    assert len(relevant) > 1, f"{group['id']} verified range is non-minimal"
    return (
        actual_positive,
        actual_negative,
        before_details,
        after_details,
        end,
        tuple(revisions),
    )


def _task_15103_claim_group_transitions(
    group: dict[str, Any],
    positive: Counter[str],
    negative: Counter[str],
    revisions: tuple[str, ...],
    transitions: list[dict[str, Any]],
) -> Counter[tuple[str, str, str, str]]:
    claims: Counter[tuple[str, str, str, str]] = Counter()
    allowed = set(revisions)
    for direction, population in (("introduced", positive), ("removed", negative)):
        for digest, multiplicity in population.items():
            remaining = multiplicity
            for transition in transitions:
                if transition["commit"] not in allowed:
                    continue
                available = transition[direction][digest]
                consumed = min(remaining, available)
                if not consumed:
                    continue
                claims[
                    (
                        group["owner_path"],
                        transition["commit"],
                        direction,
                        digest,
                    )
                ] += consumed
                remaining -= consumed
                if not remaining:
                    break
            assert not remaining, (
                f"{group['id']} cannot bind atom {digest} to complete Git history"
            )
    return claims


def _task_15103_validate_canonical_reconciliation(ledger: Any) -> None:
    """Tie the planned ledger to canonical AST populations and Git changes."""
    _task_15103_validate_review_ledger(ledger)
    planning_base = ledger["incident"]["planning_base"]
    stored_inventory = json.loads(
        diagnostic_inventory.INVENTORY_PATH.read_text(encoding="utf-8")
    )
    current_inventory = _task_15103_current_inventory()
    assert (
        stored_inventory["persistent_sink_topology"]
        == current_inventory["persistent_sink_topology"]
    )
    stored_rows = {row["path"]: row for row in stored_inventory["owners"]}
    current_rows = {row["path"]: row for row in current_inventory["owners"]}
    changed_paths = {
        path
        for path in set(stored_rows) | set(current_rows)
        if stored_rows.get(path) != current_rows.get(path)
    }
    assert changed_paths == set(TASK_15103_OWNER_STARTING), (
        "canonical inventory delta must contain the exact 18-owner path set"
    )
    history = _task_15103_complete_history(planning_base)

    groups_by_owner: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in ledger["change_groups"]:
        groups_by_owner[group["owner_path"]].append(group)

    claimed_transitions: Counter[tuple[str, str, str, str]] = Counter()
    for owner in ledger["owners"]:
        path = owner["path"]
        source = _task_15103_git_source(planning_base, path)
        assert source is not None
        current, _current_details = _task_15103_population_and_details_at(
            planning_base, path
        )
        assert _task_15103_manifest_pair_at(planning_base, path) == (
            owner["starting"]["call_count"],
            owner["starting"]["diagnostic_digest"],
        ), f"{path} starting evidence must match immutable planning-base source"
        recorded_revision = history["recorded_revisions"][path]
        recorded, _recorded_details = _task_15103_population_and_details_at(
            recorded_revision, path
        )
        added = current - recorded
        removed = recorded - current
        groups = groups_by_owner[path]
        actual_positive: Counter[str] = Counter()
        actual_negative: Counter[str] = Counter()
        for group in groups:
            (
                group_positive,
                group_negative,
                before_details,
                after_details,
                provenance_end,
                provenance_revisions,
            ) = _task_15103_assert_group_provenance(group)
            provenance_source = _task_15103_git_source(provenance_end, path)
            assert provenance_source is not None
            _task_15103_assert_group_contract(
                group,
                introduced=group_positive,
                removed=group_negative,
                before_details=before_details,
                after_details=after_details,
                source=provenance_source,
            )
            claimed_transitions.update(
                _task_15103_claim_group_transitions(
                    group,
                    group_positive,
                    group_negative,
                    provenance_revisions,
                    history["transitions"].get(path, []),
                )
            )
            actual_positive.update(group_positive)
            actual_negative.update(group_negative)
        assert (actual_positive - actual_negative) == added and (
            actual_negative - actual_positive
        ) == removed, f"{path} semantic multiset reconciliation failed"
    denominator = history["denominator"]
    assert claimed_transitions == denominator, (
        "complete Git history transition multiset mismatch: "
        f"missing={denominator - claimed_transitions!r}, "
        f"extra={claimed_transitions - denominator!r}"
    )


def _task_15103_synthetic_planned_ledger() -> dict[str, Any]:
    owners = [
        {
            "path": path,
            "starting": {
                "call_count": pair[0],
                "diagnostic_digest": pair[1],
            },
        }
        for path, pair in TASK_15103_OWNER_STARTING.items()
    ]
    return {
        "schema_version": 1,
        "review_status": "planned",
        "incident": {
            "recorded_base": TASK_15103_RECORDED_BASE,
            "planning_base": TASK_15103_PLANNING_BASE,
        },
        "owners": owners,
        "change_groups": [
            {
                "id": "TASK-15103-G001",
                "owner_path": owners[0]["path"],
                "provenance": {"exact_commit": "1" * 40},
                "disposition": "reviewed-safe",
                "rationale": "Synthetic schema fixture.",
                "fixed_event": "synthetic event",
                "severity": "info",
                "permitted_fields": [],
                "captures_exception": False,
                "removed": [],
                "proposed_surviving": [
                    {
                        "method": "info",
                        "semantic_digest": "2" * 64,
                        "multiplicity_delta": 1,
                    }
                ],
            }
        ],
    }


def _task_15103_synthetic_reviewed_ledger() -> dict[str, Any]:
    ledger = _task_15103_synthetic_planned_ledger()
    ledger["review_status"] = "reviewed"
    ledger["incident"]["final_base"] = "3" * 40
    for owner in ledger["owners"]:
        owner["reviewed_final"] = copy.deepcopy(owner["starting"])
    return ledger


def _task_15103_checkpoint_evidence(seed: str) -> dict[str, Any]:
    return {
        "base": seed * 40,
        "head": str((int(seed) + 1) % 10) * 40,
        "aggregate": {"call_count": 735, "semantic_digest": seed * 64},
        "owners": [
            {
                "path": path,
                "evidence": {
                    "call_count": pair[0],
                    "semantic_digest": hashlib.sha256(path.encode("utf-8")).hexdigest(),
                },
            }
            for path, pair in TASK_15103_OWNER_STARTING.items()
        ],
    }


def test_task_15103_review_ledger_canonical_planned_schema_and_arithmetic(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    ledger = json.loads(TASK_15103_REVIEW_PATH.read_text(encoding="utf-8"))
    repaired_candidate = copy.deepcopy(diagnostic_inventory.build_inventory())
    repaired_owner = next(
        owner
        for owner in repaired_candidate["owners"]
        if owner["path"] == "tldw_chatbook/Agents/agent_service.py"
    )
    repaired_owner["call_count"] -= 1
    repaired_owner["diagnostic_digest"] = "f" * 20
    monkeypatch.setattr(
        diagnostic_inventory,
        "build_inventory",
        lambda: copy.deepcopy(repaired_candidate),
    )
    _task_15103_current_inventory.cache_clear()
    request.addfinalizer(_task_15103_current_inventory.cache_clear)

    _task_15103_validate_canonical_reconciliation(ledger)

    assert ledger["review_status"] == "planned"
    starting = {
        owner["path"]: (
            owner["starting"]["call_count"],
            owner["starting"]["diagnostic_digest"],
        )
        for owner in ledger["owners"]
    }
    assert starting == TASK_15103_OWNER_STARTING
    disposition_counts = Counter(
        group["disposition"] for group in ledger["change_groups"]
    )
    assert disposition_counts == TASK_15103_EXPECTED_DISPOSITION_COUNTS
    assert len(ledger["change_groups"]) == 97
    atom_multiplicity: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for group in ledger["change_groups"]:
        owner_path = group["owner_path"]
        for side in ("removed", "proposed_surviving"):
            atom_multiplicity[owner_path][side] += sum(
                atom["multiplicity_delta"] for atom in group[side]
            )
    reconciled = {
        path: (
            atom_multiplicity[path]["removed"],
            atom_multiplicity[path]["proposed_surviving"],
        )
        for path in TASK_15103_OWNER_STARTING
    }
    assert reconciled == TASK_15103_EXPECTED_ATOM_MULTIPLICITY
    assert sum(pair[0] for pair in reconciled.values()) == 72
    assert sum(pair[1] for pair in reconciled.values()) == 87


def test_task_15103_review_ledger_canonical_provenance_revisions_exist() -> None:
    ledger = json.loads(TASK_15103_REVIEW_PATH.read_text(encoding="utf-8"))
    planning_base = ledger["incident"]["planning_base"]
    failures: list[str] = []

    for group in ledger["change_groups"]:
        provenance = group["provenance"]
        if "exact_commit" in provenance:
            revisions = [provenance["exact_commit"]]
            ancestry_pairs = [(provenance["exact_commit"], planning_base)]
        else:
            verified_range = provenance["verified_range"]
            revisions = [
                verified_range["start_exclusive"],
                verified_range["end_inclusive"],
            ]
            ancestry_pairs = [
                (
                    verified_range["start_exclusive"],
                    verified_range["end_inclusive"],
                ),
                (verified_range["end_inclusive"], planning_base),
            ]
        for revision in revisions:
            exists = subprocess.run(
                ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
                cwd=REPO_ROOT,
                capture_output=True,
                check=False,
                text=True,
            )
            if exists.returncode:
                failures.append(f"{group['id']}: missing commit {revision}")
        for ancestor, descendant in ancestry_pairs:
            ancestry = subprocess.run(
                ["git", "merge-base", "--is-ancestor", ancestor, descendant],
                cwd=REPO_ROOT,
                capture_output=True,
                check=False,
                text=True,
            )
            if ancestry.returncode:
                failures.append(
                    f"{group['id']}: {ancestor} is not an ancestor of {descendant}"
                )

    assert failures == []


def test_task_15103_review_ledger_synthetic_reviewed_state_requires_final_evidence() -> (
    None
):
    reviewed = _task_15103_synthetic_reviewed_ledger()
    _task_15103_validate_review_ledger(reviewed)
    starting_and_final = {
        owner["path"]: (
            (
                owner["starting"]["call_count"],
                owner["starting"]["diagnostic_digest"],
            ),
            (
                owner["reviewed_final"]["call_count"],
                owner["reviewed_final"]["diagnostic_digest"],
            ),
        )
        for owner in reviewed["owners"]
    }
    assert starting_and_final == {
        path: (pair, pair) for path, pair in TASK_15103_OWNER_STARTING.items()
    }

    missing_final_base = copy.deepcopy(reviewed)
    del missing_final_base["incident"]["final_base"]
    with pytest.raises(AssertionError, match="incident fields differ"):
        _task_15103_validate_review_ledger(missing_final_base)

    missing_owner_final = copy.deepcopy(reviewed)
    del missing_owner_final["owners"][0]["reviewed_final"]
    with pytest.raises(AssertionError, match=r"owners\[0\] fields differ"):
        _task_15103_validate_review_ledger(missing_owner_final)


def test_task_15103_review_ledger_rejects_early_lifecycle_fields() -> None:
    planned = _task_15103_synthetic_planned_ledger()
    planned["incident"]["final_base"] = "3" * 40
    with pytest.raises(AssertionError, match="incident fields differ"):
        _task_15103_validate_review_ledger(planned)

    planned = _task_15103_synthetic_planned_ledger()
    planned["owners"][0]["reviewed_final"] = copy.deepcopy(
        planned["owners"][0]["starting"]
    )
    with pytest.raises(AssertionError, match=r"owners\[0\] fields differ"):
        _task_15103_validate_review_ledger(planned)

    planned = _task_15103_synthetic_planned_ledger()
    planned["integration_checkpoint"] = {
        "state": "pre_rebase",
        "pre_rebase": _task_15103_checkpoint_evidence("4"),
    }
    with pytest.raises(AssertionError, match="integration_checkpoint is forbidden"):
        _task_15103_validate_review_ledger(planned)


def test_task_15103_review_ledger_recognizes_exact_task_8_checkpoint_lifecycle() -> (
    None
):
    reviewed = _task_15103_synthetic_reviewed_ledger()
    reviewed["integration_checkpoint"] = {
        "state": "pre_rebase",
        "pre_rebase": _task_15103_checkpoint_evidence("4"),
    }
    _task_15103_validate_review_ledger(reviewed)

    reviewed["integration_checkpoint"] = {
        "state": "post_rebase",
        "pre_rebase": _task_15103_checkpoint_evidence("4"),
        "post_rebase": _task_15103_checkpoint_evidence("6"),
    }
    _task_15103_validate_review_ledger(reviewed)

    invalid = copy.deepcopy(reviewed)
    invalid["integration_checkpoint"]["state"] = "pre_rebase"
    with pytest.raises(AssertionError, match="integration_checkpoint fields differ"):
        _task_15103_validate_review_ledger(invalid)


@pytest.mark.parametrize("extra_path", [False, True])
def test_task_15103_review_ledger_requires_exact_owner_path_set(
    extra_path: bool,
) -> None:
    ledger = _task_15103_synthetic_planned_ledger()
    if extra_path:
        ledger["owners"].append(
            {
                "path": "tldw_chatbook/nineteenth_owner.py",
                "starting": {
                    "call_count": 1,
                    "diagnostic_digest": "a" * 20,
                },
            }
        )
    else:
        ledger["owners"].pop()

    with pytest.raises(AssertionError, match="exact 18-owner path set"):
        _task_15103_validate_review_ledger(ledger)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda ledger: ledger["owners"][0]["reviewed_final"].update(
                {"unknown": True}
            ),
            r"reviewed_final fields differ",
        ),
        (
            lambda ledger: ledger["integration_checkpoint"].update({"unknown": True}),
            "integration_checkpoint fields differ",
        ),
        (
            lambda ledger: ledger["integration_checkpoint"]["pre_rebase"].update(
                {"unknown": True}
            ),
            "pre_rebase fields differ",
        ),
        (
            lambda ledger: ledger["integration_checkpoint"]["pre_rebase"][
                "aggregate"
            ].update({"unknown": True}),
            "aggregate fields differ",
        ),
        (
            lambda ledger: ledger["integration_checkpoint"]["pre_rebase"]["owners"][
                0
            ].update({"unknown": True}),
            r"owners\[0\] fields differ",
        ),
        (
            lambda ledger: ledger["integration_checkpoint"]["pre_rebase"]["owners"][0][
                "evidence"
            ].update({"unknown": True}),
            r"evidence fields differ",
        ),
    ],
)
def test_task_15103_review_ledger_rejects_unknown_fields_at_reviewed_checkpoint_levels(
    mutate: Any, match: str
) -> None:
    ledger = _task_15103_synthetic_reviewed_ledger()
    ledger["integration_checkpoint"] = {
        "state": "post_rebase",
        "pre_rebase": _task_15103_checkpoint_evidence("4"),
        "post_rebase": _task_15103_checkpoint_evidence("6"),
    }
    mutate(ledger)

    with pytest.raises(AssertionError, match=match):
        _task_15103_validate_review_ledger(ledger)


def test_task_15103_review_ledger_rejects_unknown_permitted_field_and_range_fields() -> (
    None
):
    ledger = _task_15103_synthetic_planned_ledger()
    ledger["change_groups"][0]["permitted_fields"] = [
        {
            "expression": "len(items)",
            "provenance": "Synthetic bounded-count evidence.",
            "unknown": True,
        }
    ]
    with pytest.raises(AssertionError, match=r"permitted_fields\[0\] fields differ"):
        _task_15103_validate_review_ledger(ledger)

    ledger = _task_15103_synthetic_planned_ledger()
    ledger["change_groups"][0]["provenance"] = {
        "verified_range": {
            "start_exclusive": "1" * 40,
            "end_inclusive": "2" * 40,
            "unknown": True,
        }
    }
    with pytest.raises(AssertionError, match="verified_range fields differ"):
        _task_15103_validate_review_ledger(ledger)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda ledger: ledger.update({"unknown": True}), "ledger fields differ"),
        (
            lambda ledger: ledger["incident"].update({"unknown": True}),
            "incident fields differ",
        ),
        (
            lambda ledger: ledger["owners"][0].update({"unknown": True}),
            r"owners\[0\] fields differ",
        ),
        (
            lambda ledger: ledger["owners"][0]["starting"].update({"unknown": True}),
            r"owners\[0\]\.starting fields differ",
        ),
        (
            lambda ledger: ledger["change_groups"][0].update({"unknown": True}),
            r"change_groups\[0\] fields differ",
        ),
        (
            lambda ledger: ledger["change_groups"][0]["provenance"].update(
                {"unknown": "1" * 40}
            ),
            "exactly one exact_commit or verified_range",
        ),
        (
            lambda ledger: ledger["change_groups"][0]["proposed_surviving"][0].update(
                {"unknown": True}
            ),
            r"proposed_surviving\[0\] fields differ",
        ),
    ],
)
def test_task_15103_review_ledger_rejects_unknown_fields_at_every_planned_level(
    mutate: Any, match: str
) -> None:
    ledger = _task_15103_synthetic_planned_ledger()
    mutate(ledger)
    with pytest.raises(AssertionError, match=match):
        _task_15103_validate_review_ledger(ledger)


@pytest.mark.parametrize(
    "provenance",
    [
        {},
        {"exact_commit": "1" * 40, "verified_range": {}},
        {"verified_range": {"start_exclusive": "1" * 40}},
        {
            "verified_range": {
                "start_exclusive": "1" * 40,
                "end_inclusive": "1" * 40,
            }
        },
    ],
)
def test_task_15103_review_ledger_rejects_missing_or_ambiguous_provenance(
    provenance: dict[str, Any],
) -> None:
    ledger = _task_15103_synthetic_planned_ledger()
    ledger["change_groups"][0]["provenance"] = provenance
    with pytest.raises(AssertionError):
        _task_15103_validate_review_ledger(ledger)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("semantic_digest", "ABC", "full lowercase SHA-256"),
        ("semantic_digest", "a" * 63, "full lowercase SHA-256"),
        ("multiplicity_delta", 0, "positive integer"),
        ("multiplicity_delta", -1, "positive integer"),
        ("multiplicity_delta", True, "positive integer"),
    ],
)
def test_task_15103_review_ledger_rejects_invalid_digest_or_multiplicity(
    field: str, value: Any, match: str
) -> None:
    ledger = _task_15103_synthetic_planned_ledger()
    ledger["change_groups"][0]["proposed_surviving"][0][field] = value
    with pytest.raises(AssertionError, match=match):
        _task_15103_validate_review_ledger(ledger)


def test_task_15103_review_ledger_rejects_valid_but_wrong_semantic_digest() -> None:
    ledger = json.loads(TASK_15103_REVIEW_PATH.read_text(encoding="utf-8"))
    ledger["change_groups"][0]["proposed_surviving"][0]["semantic_digest"] = "f" * 64

    with pytest.raises(AssertionError, match="semantic contract digest mismatch"):
        _task_15103_validate_canonical_reconciliation(ledger)


def test_task_15103_review_ledger_rejects_omitted_transient_history_atom() -> None:
    ledger = json.loads(TASK_15103_REVIEW_PATH.read_text(encoding="utf-8"))
    intermediate = next(
        group for group in ledger["change_groups"] if group["id"] == "TASK-15103-G091"
    )
    successor = next(
        group for group in ledger["change_groups"] if group["id"] == "TASK-15103-G062"
    )
    split = next(
        group for group in ledger["change_groups"] if group["id"] == "TASK-15103-G089"
    )
    intermediate_parent = subprocess.run(
        ["git", "rev-parse", f"{intermediate['provenance']['exact_commit']}^"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    intermediate["provenance"] = {
        "verified_range": {
            "start_exclusive": intermediate_parent,
            "end_inclusive": successor["provenance"]["exact_commit"],
        }
    }
    for field in (
        "fixed_event",
        "severity",
        "permitted_fields",
        "captures_exception",
        "proposed_surviving",
        "rationale",
    ):
        intermediate[field] = copy.deepcopy(successor[field])
    intermediate["disposition"] = "reviewed-safe"
    ledger["change_groups"].remove(successor)

    second_removed = split["removed"].pop()
    split["proposed_surviving"][0]["multiplicity_delta"] = 1
    replacement = copy.deepcopy(split)
    replacement["id"] = "TASK-15103-G062"
    replacement["removed"] = [second_removed]
    ledger["change_groups"].append(replacement)
    assert len(ledger["change_groups"]) == 97
    assert Counter(group["disposition"] for group in ledger["change_groups"]) == (
        TASK_15103_EXPECTED_DISPOSITION_COUNTS
    )

    with pytest.raises(AssertionError, match="complete Git history transition"):
        _task_15103_validate_canonical_reconciliation(ledger)


def test_task_15103_review_ledger_binds_removed_atom_method_to_history() -> None:
    ledger = json.loads(TASK_15103_REVIEW_PATH.read_text(encoding="utf-8"))
    group = next(
        group for group in ledger["change_groups"] if group["id"] == "TASK-15103-G001"
    )
    group["removed"][0]["method"] = "critical"

    with pytest.raises(AssertionError, match="removed atom method mismatch"):
        _task_15103_validate_canonical_reconciliation(ledger)


def test_task_15103_review_ledger_rejects_unrelated_existing_ancestor() -> None:
    ledger = json.loads(TASK_15103_REVIEW_PATH.read_text(encoding="utf-8"))
    ledger["change_groups"][0]["provenance"] = {
        "exact_commit": TASK_15103_RECORDED_BASE
    }

    with pytest.raises(AssertionError, match="absent from claimed Git delta"):
        _task_15103_validate_canonical_reconciliation(ledger)


def test_task_15103_review_ledger_rejects_false_permitted_field_provenance() -> None:
    ledger = json.loads(TASK_15103_REVIEW_PATH.read_text(encoding="utf-8"))
    group = next(
        group for group in ledger["change_groups"] if group["permitted_fields"]
    )
    group["permitted_fields"][0]["provenance"] = "Well-formed but false evidence."

    with pytest.raises(AssertionError, match="permitted-field provenance mismatch"):
        _task_15103_validate_canonical_reconciliation(ledger)


def test_task_15103_review_ledger_rejects_rebalanced_atom_multiplicity() -> None:
    ledger = json.loads(TASK_15103_REVIEW_PATH.read_text(encoding="utf-8"))
    first = next(
        group for group in ledger["change_groups"] if group["id"] == "TASK-15103-G024"
    )
    duplicate = next(
        group for group in ledger["change_groups"] if group["id"] == "TASK-15103-G055"
    )
    first["proposed_surviving"][0]["multiplicity_delta"] += 1
    duplicate["proposed_surviving"][0]["multiplicity_delta"] -= 1

    with pytest.raises(AssertionError, match="semantic multiset reconciliation"):
        _task_15103_validate_canonical_reconciliation(ledger)


def test_task_15103_review_ledger_rejects_contract_digest_mismatch() -> None:
    ledger = json.loads(TASK_15103_REVIEW_PATH.read_text(encoding="utf-8"))
    ledger["change_groups"][0]["fixed_event"] += " altered"

    with pytest.raises(AssertionError, match="semantic contract digest mismatch"):
        _task_15103_validate_canonical_reconciliation(ledger)


def test_task_15103_review_ledger_rejects_fabricated_intermediate_chain() -> None:
    ledger = json.loads(TASK_15103_REVIEW_PATH.read_text(encoding="utf-8"))
    intermediate = next(
        group for group in ledger["change_groups"] if group["id"] == "TASK-15103-G091"
    )
    successor = next(
        group for group in ledger["change_groups"] if group["id"] == "TASK-15103-G062"
    )
    intermediate["fixed_event"] = "Fabricated intermediate database failure."
    fabricated_digest = _task_15103_semantic_digest(
        _task_15103_synthetic_contract(intermediate)
    )
    intermediate["proposed_surviving"][0]["semantic_digest"] = fabricated_digest
    successor["removed"][0]["semantic_digest"] = fabricated_digest

    with pytest.raises(AssertionError, match="absent from claimed Git delta"):
        _task_15103_validate_canonical_reconciliation(ledger)


@pytest.mark.parametrize("schema_version", [True, 1.0])
def test_task_15103_review_ledger_requires_exact_integer_schema_version(
    schema_version: Any,
) -> None:
    ledger = _task_15103_synthetic_planned_ledger()
    ledger["schema_version"] = schema_version

    with pytest.raises(AssertionError, match="schema_version must be integer 1"):
        _task_15103_validate_review_ledger(ledger)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("call_count", -1, "non-negative integer"),
        ("call_count", True, "non-negative integer"),
        ("diagnostic_digest", "A" * 20, "20 lowercase hex"),
        ("diagnostic_digest", "a" * 19, "20 lowercase hex"),
    ],
)
def test_task_15103_review_ledger_rejects_invalid_owner_starting_pair(
    field: str, value: Any, match: str
) -> None:
    ledger = _task_15103_synthetic_planned_ledger()
    ledger["owners"][0]["starting"][field] = value

    with pytest.raises(AssertionError, match=match):
        _task_15103_validate_review_ledger(ledger)


def test_task_15103_review_ledger_semantic_atom_digest_is_scope_independent() -> None:
    semantic_projection = {
        "method": "warning",
        "event": "fixed event",
        "message_shape": "Constant(value='fixed event')",
        "expressions": [],
        "captures_exception": False,
        "level_expression": None,
    }
    compact = json.dumps(semantic_projection, separators=(",", ":"), sort_keys=True)
    expected = hashlib.sha256(compact.encode("utf-8")).hexdigest()
    atom = {
        "method": "warning",
        "semantic_digest": expected,
        "multiplicity_delta": 2,
        "qualified_scope": "One.place",
    }
    _task_15103_validate_atom(atom, location="atom")
    moved = {**atom, "qualified_scope": "Another.place"}

    assert moved["semantic_digest"] == atom["semantic_digest"]
