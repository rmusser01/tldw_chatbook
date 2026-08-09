from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

from scripts import check_persistent_diagnostic_inventory as diagnostic_inventory


REPO_ROOT = Path(__file__).resolve().parents[2]


REVIEWED_METADATA_ONLY_DIAGNOSTICS = {
    "tldw_chatbook/Media_Playback/frame_source.py": {
        "decode stopped early": ("type(exc).__name__",),
    },
    "tldw_chatbook/RAG_Search/eval/regression.py": {
        "Saved metric baseline": ("len(metrics)",),
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
            captures_exception = (
                isinstance(call.func, ast.Attribute) and call.func.attr == "exception"
            ) or any(
                isinstance(node, ast.keyword)
                and node.arg == "exception"
                and isinstance(node.value, ast.Constant)
                and node.value.value is True
                for node in ast.walk(call.func)
            )
            if sorted(fields) != sorted(expected_fields):
                failures.append(
                    f"{relative}: {label!r} fields {fields!r}, "
                    f"expected {list(expected_fields)!r}"
                )
            if captures_exception:
                failures.append(f"{relative}: {label!r} captures exception details")

    assert failures == []


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
    entry = next(
        item for item in inventory["owners"] if item["call_count"] >= 3
    )
    source = (REPO_ROOT / entry["path"]).read_text(encoding="utf-8")
    diagnostics, _sinks = diagnostic_inventory.scan_source(source)

    assert len(diagnostics) == entry["call_count"]
    assert diagnostic_inventory.diagnostic_digest(diagnostics) == (
        entry["diagnostic_digest"]
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
