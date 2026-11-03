"""Behavioral coverage for the executable profile-owned-path inventory."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest

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


def test_inventory_scanner_class_names_follow_public_style_contract() -> None:
    """Scanner implementation classes use conventional PascalCase names."""
    source_path = REPO_ROOT / "scripts" / "check_profile_owned_path_inventory.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), source_path.as_posix())

    invalid_names = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and re.fullmatch(r"[A-Z][A-Za-z0-9]*", node.name) is None
    ]

    assert invalid_names == []


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
    source = """
direct = Path.home() / ".config" / "tldw_cli" / "models"
indirect = base / ".config" / "tldw_cli" / "themes"
data = os.path.join(home, ".local", "share", "tldw_cli", "cache")
logs = base.joinpath(".local", "share", "tldw_cli", "logs")
"""

    found = scan_source(source, "tldw_chatbook/example.py")

    assert [(item.line, item.context, item.expression) for item in found] == [
        (2, "module:direct", "join:.config/tldw_cli"),
        (3, "module:indirect", "join:.config/tldw_cli"),
        (4, "module:data", "join:.local/share/tldw_cli"),
        (5, "module:logs", "join:.local/share/tldw_cli"),
    ]


def test_scanner_detects_join_components_that_contain_a_complete_root() -> None:
    """Joined path components may contain more than one root segment."""
    source = """
config = base / ".config/tldw_cli" / "models"
data = base.joinpath(".local/share/tldw_cli", "cache")
"""

    found = scan_source(source, "tldw_chatbook/example.py")

    assert [(item.line, item.context, item.expression) for item in found] == [
        (2, "module:config", "join:.config/tldw_cli"),
        (3, "module:data", "join:.local/share/tldw_cli"),
    ]


def test_scanner_distinguishes_matching_join_shapes_by_owner_context() -> None:
    """A module compatibility seed cannot mask a resolver's identical join."""
    source = """BASE_DATA_DIR_CLI = Path.home() / ".local" / "share" / "tldw_cli"
def _default_base_data_dir():
    return Path.home() / ".local" / "share" / "tldw_cli"
"""

    found = scan_source(source, "tldw_chatbook/config.py")

    assert [(item.line, item.context, item.expression) for item in found] == [
        (1, "module:BASE_DATA_DIR_CLI", "join:.local/share/tldw_cli"),
        (3, "function:_default_base_data_dir", "join:.local/share/tldw_cli"),
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
        """def profile_path():
    return "~/.config/tldw_cli/chatterbox_voices"
    return "~/.config/tldw_cli/chatterbox_voices"
""",
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
        "tldw_chatbook/TTS/backends/chatterbox.py",
        "tldw_chatbook/TTS/backends/higgs.py",
        "tldw_chatbook/TTS/backends/higgs_voice_manager.py",
        "tldw_chatbook/UI/STTS_Window.py",
        "tldw_chatbook/UI/Speech/speech_catalog_mixin.py",
        "tldw_chatbook/UI/Speech/speech_settings_mixin.py",
        "tldw_chatbook/UI/Speech/speech_settings_model.py",
        "tldw_chatbook/UI/Voice_Cloning_Window.py",
        "tldw_chatbook/Chat/custom_tokenizers.py",
    )
    shared_rules = tuple(
        rule for rule in APPROVED_EXCEPTIONS if rule.relative_path in shared_paths
    )
    assert shared_rules
    assert all(rule.disposition is Disposition.SHARED_ARTIFACT for rule in shared_rules)
    # The persisted embedding-cache default is TOML data now (TASK-36945.1): it ships once
    # in the packaged template and no shared-asset exception covers it. The full template
    # path pin is test_default_config_template_profile_paths_are_pinned.
    embedding_default = "~/.local/share/tldw_cli/models/embeddings"
    template = (REPO_ROOT / "tldw_chatbook/Config_Files/default_config.toml").read_text(
        encoding="utf-8"
    )
    assert template.count(embedding_default) == 1
    assert not any(embedding_default in rule.expression for rule in shared_rules)


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


#: TASK-39126: a file can only hold a runtime-owner finding if it spells one of
#: these, so only such files are parsed (parsing all ~9,050 took up to 113 s
#: of the node's 180 s CI timeout).
RUNTIME_OWNER_NAMES = ("BASE_DATA_DIR_CLI", "DEFAULT_RUNTIME_POLICY_PATH")


def _may_name_runtime_owner(source: str) -> bool:
    """Whether ``source`` can spell a guarded name as the parser reads it.

    The parser NFKC-normalizes identifiers, so a non-ASCII spelling such as a
    full-width first letter counts too.
    """
    if not source.isascii():
        source = unicodedata.normalize("NFKC", source)
    return any(name in source for name in RUNTIME_OWNER_NAMES)


def test_base_data_dir_census_detects_qualified_and_aliased_consumers() -> None:
    """Compatibility-constant consumers include qualified import access."""
    source = """import tldw_chatbook.config as config
from tldw_chatbook.config import BASE_DATA_DIR_CLI as data_dir

config.BASE_DATA_DIR_CLI / "export"
data_dir / "export"
"""
    tree = ast.parse(source)

    assert _base_data_dir_cli_consumer_lines(tree) == (4, 5)
    assert _may_name_runtime_owner(source)
    # A full-width first letter is the same identifier to the parser, so the
    # prefilter must not skip the file (TASK-39126).
    full_width = (
        "from tldw_chatbook.config import \uff22ASE_DATA_DIR_CLI\n"
        "\uff22ASE_DATA_DIR_CLI / 'x'\n"
    )
    assert _base_data_dir_cli_consumer_lines(ast.parse(full_width)) == (2,)
    assert _may_name_runtime_owner(full_width)
    assert not _may_name_runtime_owner("data_dir / 'export'\n")


#: TASK-38989: Python under Docs/ is evidence and snippets, never runtime
#: (bffa2bffed added a QA excerpt that is not even parseable on its own).
NON_RUNTIME_PYTHON_ROOTS = ("Docs/",)

def _runtime_python_source_paths(root: Path) -> tuple[Path, ...]:
    """Enumerate project code, including tracked tooling and new runtime files.

    ADR-040 excludes test examples and documentation from production ownership.
    Git excludes ignored local evidence, environments, and build copies; tracked
    vendored production modules remain in the population.
    """
    completed = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "*.py",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(
        root / relative
        for relative in sorted(set(completed.stdout.split("\0")) - {""})
        if Path(relative).parts[0] not in {"Docs", "Tests"}
    )


def test_compatibility_and_runtime_policy_constants_have_no_new_runtime_owners() -> (
    None
):
    """Legacy constants remain isolated from normal profile resolution."""
    base_data_consumers: list[tuple[str, int]] = []
    runtime_constant_definitions: list[str] = []
    for source_path in _runtime_python_source_paths(REPO_ROOT):
        relative_path = source_path.relative_to(REPO_ROOT).as_posix()
        if relative_path.startswith(NON_RUNTIME_PYTHON_ROOTS):
            continue
        source = source_path.read_text(encoding="utf-8")
        if not _may_name_runtime_owner(source):
            continue
        tree = ast.parse(source, relative_path)
        base_data_consumers.extend(
            (relative_path, line) for line in _base_data_dir_cli_consumer_lines(tree)
        )
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else (node.target,)
                )
                if any(
                    isinstance(target, ast.Name)
                    and target.id == "DEFAULT_RUNTIME_POLICY_PATH"
                    for target in targets
                ):
                    runtime_constant_definitions.append(relative_path)

    assert base_data_consumers == []
    assert runtime_constant_definitions == ["tldw_chatbook/runtime_policy/bootstrap.py"]

    from tldw_chatbook.runtime_policy.bootstrap import default_runtime_policy_path

    assert isinstance(default_runtime_policy_path(), Path)


#: Roots this node skips: the package (preflight's profile census parses it),
#: Tests/ (pytest collection parses each file a lane runs) and Docs/ (not
#: runtime). Only git-tracked files count: a local build/, venv/ or
#: node_modules/ is not repository source.
PARSED_ELSEWHERE_ROOTS = ("tldw_chatbook/", "Tests/", "Docs/")


def test_python_outside_the_package_and_tests_parses() -> None:
    """Every other tracked ``.py`` file parses (TASK-39126).

    The runtime-policy node above used to parse every file, so an unparseable
    script failed it as a side effect. It now parses only files that name its
    constants; this node keeps that side effect for the roots no other check
    parses (scripts/, Helper_Scripts/, Packaging/, packages/, .github/,
    native/, top-level files, ...). ``git ls-files`` reads the index, so it
    works in CI's shallow checkout. Bytes are parsed, so a BOM or a PEP 263
    cookie is honoured as Python honours it.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.py"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.decode("utf-8", "surrogateescape")
    unparseable: list[str] = []
    for relative_path in tracked.split("\0"):
        if not relative_path or relative_path.startswith(PARSED_ELSEWHERE_ROOTS):
            continue
        path = REPO_ROOT / relative_path
        if not path.is_file():  # tracked, but deleted on disk and not yet staged
            continue
        try:
            ast.parse(path.read_bytes(), filename=relative_path)
        except SyntaxError as error:
            unparseable.append(f"{relative_path}:{error.lineno}: {error.msg}")

    assert not unparseable, "\n".join(unparseable)


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


@pytest.fixture
def runtime_source_tree(tmp_path: Path) -> Path:
    """Create a tiny repository with tracked code and historical evidence."""
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    files = {
        ".gitignore": ".superpowers/\nbuild/\n.venv/\n",
        "tldw_chatbook/runtime_policy/bootstrap.py": "DEFAULT_RUNTIME_POLICY_PATH = None\n",
        "Helper_Scripts/tracked.py": "pass\n",
        "Docs/proposed.py": "    def historical_fragment(): pass\n",
        "Tests/fixture.py": "    def historical_fragment(): pass\n",
    }
    for relative, source in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    subprocess.run(
        [
            "git",
            "add",
            "--",
            "Helper_Scripts/tracked.py",
            "Docs/proposed.py",
            "Tests/fixture.py",
        ],
        cwd=tmp_path,
        check=True,
    )
    return tmp_path


def test_runtime_source_population_keeps_tooling_and_ignores_nonruntime_evidence(
    runtime_source_tree: Path, monkeypatch
) -> None:
    root = runtime_source_tree
    for relative in ("scripts/new.py", "entrypoint.py", ".github/probe.py"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("pass\n", encoding="utf-8")
    for directory in (".superpowers", "build", ".venv"):
        path = root / directory / "historical.py"
        path.parent.mkdir(parents=True)
        path.write_text("    def historical_fragment(): pass\n", encoding="utf-8")

    paths = _runtime_python_source_paths(root)

    assert [path.relative_to(root).as_posix() for path in paths] == [
        ".github/probe.py",
        "Helper_Scripts/tracked.py",
        "entrypoint.py",
        "scripts/new.py",
        "tldw_chatbook/runtime_policy/bootstrap.py",
    ]
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8"), path.as_posix())
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", root)
    test_compatibility_and_runtime_policy_constants_have_no_new_runtime_owners()


@pytest.mark.parametrize(
    "relative, source",
    (
        (
            "tldw_chatbook/new.py",
            "from tldw_chatbook.config import BASE_DATA_DIR_CLI as old\nold / 'new'\n",
        ),
        (
            "scripts/new.py",
            "import tldw_chatbook.config as cfg\ncfg.BASE_DATA_DIR_CLI / 'new'\n",
        ),
        (
            "Helper_Scripts/tracked.py",
            "from tldw_chatbook import config\nconfig.BASE_DATA_DIR_CLI / 'new'\n",
        ),
        (
            "entrypoint.py",
            "import tldw_chatbook.config\ntldw_chatbook.config.BASE_DATA_DIR_CLI / 'new'\n",
        ),
        ("Helper_Scripts/tracked.py", "DEFAULT_RUNTIME_POLICY_PATH = None\n"),
    ),
)
def test_runtime_constant_guard_rejects_new_runtime_owners(
    runtime_source_tree: Path, monkeypatch, relative: str, source: str
) -> None:
    path = runtime_source_tree / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", runtime_source_tree)

    with pytest.raises(AssertionError):
        test_compatibility_and_runtime_policy_constants_have_no_new_runtime_owners()


def test_runtime_constant_guard_does_not_swallow_runtime_parse_failure(
    runtime_source_tree: Path, monkeypatch
) -> None:
    path = runtime_source_tree / "scripts" / "broken.py"
    path.parent.mkdir(parents=True)
    path.write_text("    def broken(): pass\n", encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", runtime_source_tree)

    with pytest.raises(IndentationError):
        test_compatibility_and_runtime_policy_constants_have_no_new_runtime_owners()
