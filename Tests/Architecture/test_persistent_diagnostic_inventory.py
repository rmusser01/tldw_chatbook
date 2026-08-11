from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pytest

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
TASK_15103_PLANNING_BASE = "6754133f51ce31ebd40ddfcc4a59c0ccc628371b"
TASK_15103_OWNER_STARTING = {
    "tldw_chatbook/Agents/agent_service.py": (9, "578de6bb91649fc9fc87"),
    "tldw_chatbook/Chat/console_agent_bridge.py": (
        12,
        "7caa9d8c2694081e94e9",
    ),
    "tldw_chatbook/Chat/console_chat_controller.py": (
        32,
        "491f364f638ff7ddc093",
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
}
_TASK_15103_DISPOSITIONS = {
    "reviewed-safe",
    "metadata-repair",
    "justified-deletion",
}
TASK_15103_EXPECTED_DISPOSITION_COUNTS = {
    "reviewed-safe": 44,
    "metadata-repair": 37,
    "justified-deletion": 6,
}
TASK_15103_EXPECTED_ATOM_MULTIPLICITY = {
    "tldw_chatbook/Agents/agent_service.py": (9, 9),
    "tldw_chatbook/Chat/console_agent_bridge.py": (0, 1),
    "tldw_chatbook/Chat/console_chat_controller.py": (5, 5),
    "tldw_chatbook/Chat/console_chat_store.py": (4, 4),
    "tldw_chatbook/Chat/console_context_compaction.py": (2, 2),
    "tldw_chatbook/Chat/console_provider_gateway.py": (1, 1),
    "tldw_chatbook/MCP/client.py": (10, 36),
    "tldw_chatbook/MCP/local_server_tools.py": (1, 1),
    "tldw_chatbook/MCP/prompts.py": (5, 1),
    "tldw_chatbook/MCP/server.py": (3, 1),
    "tldw_chatbook/RAG_Search/fusion.py": (4, 3),
    "tldw_chatbook/RAG_Search/simplified/rag_service.py": (2, 5),
    "tldw_chatbook/RAG_Search/simplified/search_service.py": (1, 1),
    "tldw_chatbook/UI/Console_Modules/session.py": (1, 1),
    "tldw_chatbook/UI/Screens/chat_screen.py": (0, 0),
    "tldw_chatbook/UI/Screens/library_screen.py": (2, 2),
    "tldw_chatbook/app.py": (12, 9),
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
        f"{location}.owners must contain the exact 17-owner path set"
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
    assert ledger["schema_version"] == 1, "schema_version must be 1"

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
        "owners must contain the exact 17-owner path set"
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
            f"{location}.owner_path is outside the approved 17-owner set"
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


def test_task_15103_review_ledger_canonical_planned_schema_and_arithmetic() -> None:
    ledger = json.loads(TASK_15103_REVIEW_PATH.read_text(encoding="utf-8"))

    _task_15103_validate_review_ledger(ledger)

    assert ledger["review_status"] == "planned"
    starting = {
        owner["path"]: (
            owner["starting"]["call_count"],
            owner["starting"]["diagnostic_digest"],
        )
        for owner in ledger["owners"]
    }
    assert starting == TASK_15103_OWNER_STARTING
    current_inventory = diagnostic_inventory.build_inventory()
    current_by_path = {owner["path"]: owner for owner in current_inventory["owners"]}
    current_starting = {
        path: (
            current_by_path[path]["call_count"],
            current_by_path[path]["diagnostic_digest"],
        )
        for path in TASK_15103_OWNER_STARTING
    }
    assert current_starting == TASK_15103_OWNER_STARTING
    disposition_counts = Counter(
        group["disposition"] for group in ledger["change_groups"]
    )
    assert disposition_counts == TASK_15103_EXPECTED_DISPOSITION_COUNTS
    assert len(ledger["change_groups"]) == 87
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
    assert sum(pair[0] for pair in reconciled.values()) == 62
    assert sum(pair[1] for pair in reconciled.values()) == 82


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
                "path": "tldw_chatbook/eighteenth_owner.py",
                "starting": {
                    "call_count": 1,
                    "diagnostic_digest": "a" * 20,
                },
            }
        )
    else:
        ledger["owners"].pop()

    with pytest.raises(AssertionError, match="exact 17-owner path set"):
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
