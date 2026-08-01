"""Behavioral coverage for the executable profile-owned-path inventory."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from scripts.check_profile_owned_path_inventory import (
    APPROVED_EXCEPTIONS,
    Disposition,
    ExceptionRule,
    Occurrence,
    reconcile_inventory,
    scan_source,
    scan_tree,
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


def test_reconcile_rejects_a_scanner_detected_duplicate_literal() -> None:
    """A second literal in one owner context breaks the exact census."""
    occurrences = scan_source(
        '''def profile_path():
    return "~/.config/tldw_cli/chatterbox_voices"
    return "~/.config/tldw_cli/chatterbox_voices"
''',
        "tldw_chatbook/example.py",
    )
    rule = ExceptionRule(
        "tldw_chatbook/example.py",
        "function:profile_path",
        "literal:~/.config/tldw_cli/chatterbox_voices",
        1,
        Disposition.SHARED_ARTIFACT,
        "reusable Chatterbox voice profiles",
    )

    assert len(occurrences) == 2
    problems = reconcile_inventory(occurrences, (rule,))

    assert [problem.reason for problem in problems] == [
        "expected 1 occurrence(s), found 2"
    ]


def test_production_profile_owned_path_inventory_is_exact() -> None:
    """Production source contains only the frozen ADR-040 census."""
    occurrences = scan_tree(REPO_ROOT / "tldw_chatbook")

    assert reconcile_inventory(occurrences, APPROVED_EXCEPTIONS) == ()


def test_exception_rules_are_sorted_by_inventory_identity() -> None:
    """Reviewers can compare the frozen census in scanner output order."""
    identities = tuple(
        (rule.relative_path, rule.context, rule.expression)
        for rule in APPROVED_EXCEPTIONS
    )

    assert identities == tuple(sorted(identities))


def test_shared_asset_exceptions_are_explicit() -> None:
    """Reusable TTS/tokenizer assets cannot be mistaken for profile defaults."""
    shared_paths = (
        "tldw_chatbook/TTS/TTS_Backends.py",
        "tldw_chatbook/TTS/backends/kokoro.py",
        "tldw_chatbook/TTS/kokoro_pytorch.py",
        "tldw_chatbook/TTS/utils/download_models.py",
        "tldw_chatbook/TTS/backends/chatterbox.py",
        "tldw_chatbook/TTS/backends/higgs.py",
        "tldw_chatbook/TTS/backends/higgs_voice_manager.py",
        "tldw_chatbook/UI/STTS_Window.py",
        "tldw_chatbook/UI/Speech/speech_catalog_mixin.py",
        "tldw_chatbook/UI/Speech/speech_settings_mixin.py",
        "tldw_chatbook/UI/Speech/speech_settings_model.py",
        "tldw_chatbook/UI/Voice_Cloning_Window.py",
        "tldw_chatbook/Utils/custom_tokenizers.py",
    )
    shared_rules = tuple(
        rule
        for rule in APPROVED_EXCEPTIONS
        if rule.relative_path in shared_paths
    )
    embedding_rule = next(
        rule
        for rule in APPROVED_EXCEPTIONS
        if (
            rule.relative_path == "tldw_chatbook/config.py"
            and rule.expression
            == "literal:~/.local/share/tldw_cli/models/embeddings"
        )
    )

    assert shared_rules
    assert all(
        rule.disposition is Disposition.SHARED_ARTIFACT for rule in shared_rules
    )
    assert embedding_rule.disposition is Disposition.PERSISTED_DEFAULT


def _dotted_name(node: ast.AST) -> str | None:
    """Return a dotted source name for a simple attribute expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _base_data_dir_cli_consumer_lines(tree: ast.AST) -> tuple[int, ...]:
    """Return source lines that consume the compatibility data-dir constant."""
    direct_names = {"BASE_DATA_DIR_CLI"}
    config_module_names: set[str] = set()
    package_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "tldw_chatbook.config":
                    if alias.asname:
                        config_module_names.add(alias.asname)
                    else:
                        package_names.add("tldw_chatbook")
                elif alias.name == "tldw_chatbook":
                    package_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "tldw_chatbook.config":
                direct_names.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "BASE_DATA_DIR_CLI"
                )
            elif node.module == "tldw_chatbook":
                config_module_names.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "config"
                )

    config_module_names.update(
        f"{package_name}.config" for package_name in package_names
    )
    consumers: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in direct_names:
                consumers.append(node.lineno)
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Load)
            and node.attr == "BASE_DATA_DIR_CLI"
            and _dotted_name(node.value) in config_module_names
        ):
            consumers.append(node.lineno)
    return tuple(sorted(consumers))


def test_base_data_dir_census_detects_qualified_and_aliased_consumers() -> None:
    """Compatibility-constant consumers include qualified import access."""
    tree = ast.parse(
        '''import tldw_chatbook.config as config
from tldw_chatbook.config import BASE_DATA_DIR_CLI as data_dir

config.BASE_DATA_DIR_CLI / "export"
data_dir / "export"
'''
    )

    assert _base_data_dir_cli_consumer_lines(tree) == (4, 5)


def test_compatibility_and_runtime_policy_constants_have_no_new_runtime_owners() -> None:
    """Legacy constants remain isolated from normal profile resolution."""
    base_data_consumers: list[tuple[str, int]] = []
    runtime_constant_definitions: list[str] = []
    for source_path in sorted(REPO_ROOT.rglob("*.py")):
        relative_path = source_path.relative_to(REPO_ROOT).as_posix()
        tree = ast.parse(source_path.read_text(encoding="utf-8"), relative_path)
        base_data_consumers.extend(
            (relative_path, line)
            for line in _base_data_dir_cli_consumer_lines(tree)
        )
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
                if any(
                    isinstance(target, ast.Name)
                    and target.id == "DEFAULT_RUNTIME_POLICY_PATH"
                    for target in targets
                ):
                    runtime_constant_definitions.append(relative_path)

    assert base_data_consumers == [
        ("Helper_Scripts/Prompts/Prompts_Dump.py", 96)
    ]
    assert runtime_constant_definitions == [
        "tldw_chatbook/runtime_policy/bootstrap.py"
    ]

    from tldw_chatbook.runtime_policy.bootstrap import default_runtime_policy_path

    assert isinstance(default_runtime_policy_path(), Path)


def test_cli_prints_the_real_source_census_and_enforces_it() -> None:
    """The developer CLI shares the exact production reconciliation contract."""
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

    assert completed.returncode == 0
    assert "tldw_chatbook/" in completed.stdout
    assert completed.stderr == ""
