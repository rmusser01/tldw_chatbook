"""Behavioral coverage for the executable profile-owned-path inventory."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.check_profile_owned_path_inventory import (
    Disposition,
    ExceptionRule,
    Occurrence,
    reconcile_inventory,
    scan_source,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_scanner_reports_embedded_and_multiline_physical_lines() -> None:
    """Embedded paths retain the physical line on which they appear."""
    source = '''COPY = "edit ~/.config/tldw_cli/config.toml now"
DEFAULTS = """[database]
description = "a deliberately long preceding line makes token offsets exceed the final line column"
media = "~/.local/share/tldw_cli/media.db"""
'''

    found = scan_source(source, "tldw_chatbook/example.py")

    assert [(item.line, item.context, item.expression) for item in found] == [
        (1, "module:COPY", "literal:~/.config/tldw_cli/config.toml"),
        (
            4,
            "module:DEFAULTS",
            "literal:~/.local/share/tldw_cli/media.db",
        ),
    ]


def test_scanner_detects_direct_indirect_and_join_function_roots() -> None:
    """Path joins are detected even when their base is not ``Path.home()``."""
    source = '''
direct = Path.home() / ".config" / "tldw_cli" / "models"
indirect = base / ".config" / "tldw_cli" / "themes"
data = os.path.join(home, ".local", "share", "tldw_cli", "cache")
'''

    found = scan_source(source, "tldw_chatbook/example.py")

    assert [item.expression for item in found] == [
        "join:.config/tldw_cli",
        "join:.config/tldw_cli",
        "join:.local/share/tldw_cli",
    ]


def test_scanner_detects_join_components_that_contain_a_complete_root() -> None:
    """Joined path components may contain more than one root segment."""
    source = '''
config = base / ".config/tldw_cli" / "models"
data = base.joinpath(".local/share/tldw_cli", "cache")
'''

    found = scan_source(source, "tldw_chatbook/example.py")

    assert [item.expression for item in found] == [
        "join:.config/tldw_cli",
        "join:.local/share/tldw_cli",
    ]


def test_scanner_distinguishes_matching_join_shapes_by_owner_context() -> None:
    """A module compatibility seed cannot mask a resolver's identical join."""
    source = '''BASE_DATA_DIR_CLI = Path.home() / ".local" / "share" / "tldw_cli"
def _default_base_data_dir():
    return Path.home() / ".local" / "share" / "tldw_cli"
'''

    found = scan_source(source, "tldw_chatbook/config.py")

    assert [(item.context, item.expression) for item in found] == [
        ("module:BASE_DATA_DIR_CLI", "join:.local/share/tldw_cli"),
        ("function:_default_base_data_dir", "join:.local/share/tldw_cli"),
    ]


def test_scanner_ignores_comments_and_actual_docstrings() -> None:
    """Only executable string tokens are ownership candidates."""
    source = '''# ~/.config/tldw_cli/comment
def sample():
    """~/.local/share/tldw_cli/docstring"""
    return "safe"
'''

    assert scan_source(source, "tldw_chatbook/example.py") == ()


def test_reconcile_rejects_duplicates_new_counts_and_stale_rules() -> None:
    """Rules are exact counts, never a file-level allowlist."""
    occurrence = Occurrence(
        "tldw_chatbook/example.py",
        4,
        "module:MODEL_DIR",
        "literal:~/.config/tldw_cli/models",
    )
    rule = ExceptionRule(
        "tldw_chatbook/example.py",
        "module:MODEL_DIR",
        "literal:~/.config/tldw_cli/models",
        1,
        Disposition.SHARED_ARTIFACT,
        "reusable model weights",
    )

    assert reconcile_inventory((occurrence,), (rule,)) == ()
    assert reconcile_inventory((occurrence, occurrence), (rule,))
    assert reconcile_inventory((), (rule,))
    assert reconcile_inventory((occurrence,), ())


def test_reconcile_describes_duplicate_and_empty_exception_rules() -> None:
    """Invalid exception records cannot silently act as an allowlist."""
    rule = ExceptionRule(
        "tldw_chatbook/example.py",
        "module:MODEL_DIR",
        "literal:~/.config/tldw_cli/models",
        1,
        Disposition.SHARED_ARTIFACT,
        "",
    )

    problems = reconcile_inventory((), (rule, rule))

    assert [problem.reason for problem in problems] == [
        "duplicate exception rule",
        "empty exception reason",
        "stale exception rule",
    ]


def test_cli_prints_the_real_source_census_and_fails_unclassified_inventory() -> None:
    """The developer CLI uses the scanner and reconciliation contract together."""
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/check_profile_owned_path_inventory.py",
            "--print-occurrences",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "tldw_chatbook/" in completed.stdout
    assert "unapproved occurrence" in completed.stderr
