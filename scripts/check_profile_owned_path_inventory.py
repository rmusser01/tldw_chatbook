#!/usr/bin/env python3
"""Inventory executable uses of historical profile-owned path roots."""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
import re
import sys
import tokenize
from typing import Iterable
import warnings


ROOTS = ("~/.config/tldw_cli", "~/.local/share/tldw_cli")
JOIN_SUFFIXES = (
    (".config", "tldw_cli"),
    (".local", "share", "tldw_cli"),
)


class Disposition(StrEnum):
    """ADR-040 classes for retained executable path occurrences."""

    PERSISTED_DEFAULT = "persisted_default"
    RESOLVER_SEED = "resolver_seed"
    COMPATIBILITY_CONSTANT = "compatibility_constant"
    SHARED_ARTIFACT = "shared_artifact"
    READ_ONLY_LEGACY_PROBE = "read_only_legacy_probe"


@dataclass(frozen=True, order=True)
class Occurrence:
    """One executable path-root occurrence, normalized for inventory matching."""

    relative_path: str
    line: int
    context: str
    expression: str


@dataclass(frozen=True)
class ExceptionRule:
    """The exact expected census for one retained ownership exception."""

    relative_path: str
    context: str
    expression: str
    expected_count: int
    disposition: Disposition
    reason: str


@dataclass(frozen=True)
class _Problem:
    relative_path: str
    line: int
    context: str
    expression: str
    reason: str


@dataclass(frozen=True)
class _Span:
    start_line: int
    start_column: int
    end_line: int
    end_column: int
    context: str

    def contains(self, line: int, column: int) -> bool:
        """Return whether a source coordinate is within this AST span."""
        if (line, column) < (self.start_line, self.start_column):
            return False
        return (line, column) <= (self.end_line, self.end_column)

    @property
    def size(self) -> tuple[int, int]:
        """Return a sortable approximation of span size."""
        return (self.end_line - self.start_line, self.end_column - self.start_column)


def _physical_line(token_text: str, token_line: int, offset: int) -> int:
    return token_line + token_text[:offset].count("\n")


def _physical_column(token_text: str, token_column: int, offset: int) -> int:
    """Return the physical source column for a token-relative offset."""
    final_newline = token_text.rfind("\n", 0, offset)
    if final_newline < 0:
        return token_column + offset
    return offset - final_newline - 1


def _literal_expression(value: str, root: str, offset: int) -> str:
    match = re.match(re.escape(root) + r"[A-Za-z0-9_./<>-]*", value[offset:])
    assert match is not None
    return f"literal:{match.group(0)}"


def _node_span(node: ast.AST, context: str) -> _Span:
    """Build a source span for an AST node with location information."""
    return _Span(
        node.lineno,
        node.col_offset,
        node.end_lineno,
        node.end_col_offset,
        context,
    )


def _assignment_target(node: ast.AST) -> str | None:
    """Return the first simple module assignment target, if one exists."""
    targets: list[ast.expr]
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    elif isinstance(node, ast.AugAssign):
        targets = [node.target]
    else:
        return None

    for target in targets:
        if isinstance(target, ast.Name):
            return target.id
        if isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                if isinstance(item, ast.Name):
                    return item.id
    return None


def _contains_span(outer: _Span, inner: _Span) -> bool:
    """Return whether ``outer`` wholly contains ``inner``."""
    return (
        (outer.start_line, outer.start_column)
        <= (inner.start_line, inner.start_column)
        and (inner.end_line, inner.end_column)
        <= (outer.end_line, outer.end_column)
    )


def _source_contexts(tree: ast.Module) -> tuple[list[_Span], list[_Span], list[_Span]]:
    """Collect docstring, scope, and module-assignment spans from source AST."""
    docstrings: list[_Span] = []
    scopes: list[_Span] = []
    assignments: list[_Span] = []

    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            docstrings.append(_node_span(body[0].value, "docstring"))

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scopes.append(_node_span(node, f"function:{node.name}"))
        elif isinstance(node, ast.ClassDef):
            scopes.append(_node_span(node, f"class:{node.name}"))

    scope_spans = tuple(scopes)
    for node in ast.walk(tree):
        target = _assignment_target(node)
        if target is None:
            continue
        assignment = _node_span(node, f"module:{target}")
        if not any(_contains_span(scope, assignment) for scope in scope_spans):
            assignments.append(assignment)

    return docstrings, scopes, assignments


def _context_at(
    line: int,
    column: int,
    scopes: Iterable[_Span],
    assignments: Iterable[_Span],
) -> str:
    """Return the smallest owning scope, or its module assignment target."""
    containing_scopes = [span for span in scopes if span.contains(line, column)]
    if containing_scopes:
        return min(containing_scopes, key=lambda span: span.size).context

    containing_assignments = [
        span for span in assignments if span.contains(line, column)
    ]
    if containing_assignments:
        return min(containing_assignments, key=lambda span: span.size).context
    return "module"


def _is_docstring(
    line: int, column: int, docstrings: Iterable[_Span]
) -> bool:
    """Return whether a string token begins at an actual AST docstring span."""
    return any(span.contains(line, column) for span in docstrings)


def _is_os_path_join(node: ast.Call) -> bool:
    """Return whether a call is exactly ``os.path.join(...)``."""
    function = node.func
    return (
        isinstance(function, ast.Attribute)
        and function.attr == "join"
        and isinstance(function.value, ast.Attribute)
        and function.value.attr == "path"
        and isinstance(function.value.value, ast.Name)
        and function.value.value.id == "os"
    )


def _string_components(node: ast.AST) -> list[str | None]:
    """Flatten recognized path joins into literal and opaque components."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.split("/")
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _string_components(node.left) + _string_components(node.right)
    if isinstance(node, ast.Call) and _is_os_path_join(node):
        components: list[str | None] = []
        for argument in node.args:
            components.extend(_string_components(argument))
        return components
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "joinpath"
    ):
        components = _string_components(node.func.value)
        for argument in node.args:
            components.extend(_string_components(argument))
        return components
    return [None]


def _join_expression(node: ast.AST) -> str | None:
    """Normalize a recognized join expression when it contains a root suffix."""
    components = _string_components(node)
    for suffix in JOIN_SUFFIXES:
        width = len(suffix)
        if any(tuple(components[index : index + width]) == suffix for index in range(len(components) - width + 1)):
            return f"join:{'/'.join(suffix)}"
    return None


def scan_source(source: str, relative_path: str) -> tuple[Occurrence, ...]:
    """Return all executable historical-root occurrences in Python source text."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(source, filename=relative_path)
    docstrings, scopes, assignments = _source_contexts(tree)
    found: list[Occurrence] = []

    for token in tokenize.generate_tokens(iter(source.splitlines(keepends=True)).__next__):
        if token.type != tokenize.STRING or _is_docstring(
            token.start[0], token.start[1], docstrings
        ):
            continue
        for root in ROOTS:
            offset = token.string.find(root)
            while offset >= 0:
                line = _physical_line(token.string, token.start[0], offset)
                column = _physical_column(token.string, token.start[1], offset)
                found.append(
                    Occurrence(
                        relative_path,
                        line,
                        _context_at(line, column, scopes, assignments),
                        _literal_expression(token.string, root, offset),
                    )
                )
                offset = token.string.find(root, offset + len(root))

    seen_joins: set[tuple[str, int, int, str, str]] = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Div)
            or isinstance(node, ast.Call)
            and (
                _is_os_path_join(node)
                or isinstance(node.func, ast.Attribute)
                and node.func.attr == "joinpath"
            )
        ):
            continue
        expression = _join_expression(node)
        if expression is None:
            continue
        context = _context_at(node.lineno, node.col_offset, scopes, assignments)
        key = (relative_path, node.lineno, node.col_offset, context, expression)
        if key in seen_joins:
            continue
        seen_joins.add(key)
        found.append(Occurrence(relative_path, node.lineno, context, expression))

    return tuple(sorted(found))


def scan_tree(root: Path) -> tuple[Occurrence, ...]:
    """Scan every Python source file under ``root`` in deterministic order."""
    root = root.resolve()
    occurrences: list[Occurrence] = []
    for source_path in sorted(root.rglob("*.py")):
        with tokenize.open(source_path) as source_file:
            source = source_file.read()
        relative_path = source_path.relative_to(root.parent).as_posix()
        occurrences.extend(scan_source(source, relative_path))
    return tuple(sorted(occurrences))


def reconcile_inventory(
    occurrences: Iterable[Occurrence], rules: Iterable[ExceptionRule]
) -> tuple[_Problem, ...]:
    """Report new, count-changed, duplicate, invalid, and stale exceptions."""
    observed: dict[tuple[str, str, str], list[Occurrence]] = defaultdict(list)
    for occurrence in sorted(occurrences):
        observed[
            (occurrence.relative_path, occurrence.context, occurrence.expression)
        ].append(occurrence)

    grouped_rules: dict[tuple[str, str, str], list[ExceptionRule]] = defaultdict(list)
    for rule in rules:
        grouped_rules[(rule.relative_path, rule.context, rule.expression)].append(rule)

    problems: list[_Problem] = []
    for key, matching_rules in sorted(grouped_rules.items()):
        relative_path, context, expression = key
        line = observed[key][0].line if key in observed else 0
        if len(matching_rules) != 1:
            problems.append(
                _Problem(relative_path, line, context, expression, "duplicate exception rule")
            )
        if any(not rule.reason.strip() for rule in matching_rules):
            problems.append(
                _Problem(relative_path, line, context, expression, "empty exception reason")
            )

    for key, matching_occurrences in sorted(observed.items()):
        relative_path, context, expression = key
        matching_rules = grouped_rules.get(key, [])
        if not matching_rules:
            problems.extend(
                _Problem(
                    occurrence.relative_path,
                    occurrence.line,
                    occurrence.context,
                    occurrence.expression,
                    "unapproved occurrence",
                )
                for occurrence in matching_occurrences
            )
            continue
        if len(matching_rules) == 1 and len(matching_occurrences) != matching_rules[0].expected_count:
            problems.append(
                _Problem(
                    relative_path,
                    matching_occurrences[0].line,
                    context,
                    expression,
                    f"expected {matching_rules[0].expected_count} occurrence(s), found {len(matching_occurrences)}",
                )
            )

    for key, matching_rules in sorted(grouped_rules.items()):
        if key in observed:
            continue
        relative_path, context, expression = key
        problems.append(
            _Problem(relative_path, 0, context, expression, "stale exception rule")
        )

    return tuple(sorted(problems, key=lambda item: (item.relative_path, item.line, item.context, item.expression, item.reason)))


APPROVED_EXCEPTIONS: tuple[ExceptionRule, ...] = (
    ExceptionRule(
        "tldw_chatbook/Evals/eval_orchestrator.py",
        "function:_warn_if_legacy_data_exists",
        "literal:~/.local/share/tldw_cli",
        1,
        Disposition.READ_ONLY_LEGACY_PROBE,
        "read-only warning probe for stranded legacy Evals data",
    ),
    ExceptionRule(
        "tldw_chatbook/RAG_Search/simplified/config.py",
        "module:EXAMPLE_TOML_CONFIG",
        "literal:~/.local/share/tldw_cli/chromadb",
        1,
        Disposition.PERSISTED_DEFAULT,
        "shipped RAG example default for the persisted ChromaDB directory",
    ),
    ExceptionRule(
        "tldw_chatbook/TTS/TTS_Backends.py",
        "function:_prepare_backend_config",
        "literal:~/.config/tldw_cli/higgs_voices",
        1,
        Disposition.SHARED_ARTIFACT,
        "reusable Higgs voice artifact directory",
    ),
    ExceptionRule(
        "tldw_chatbook/TTS/backends/chatterbox.py",
        "function:__init__",
        "literal:~/.config/tldw_cli/chatterbox_voices",
        1,
        Disposition.SHARED_ARTIFACT,
        "reusable Chatterbox voice artifact directory",
    ),
    ExceptionRule(
        "tldw_chatbook/TTS/backends/higgs.py",
        "function:__init__",
        "literal:~/.config/tldw_cli/higgs_voices",
        1,
        Disposition.SHARED_ARTIFACT,
        "reusable Higgs voice artifact directory",
    ),
    ExceptionRule(
        "tldw_chatbook/TTS/backends/higgs_voice_manager.py",
        "function:main",
        "literal:~/.config/tldw_cli/higgs_voices",
        1,
        Disposition.SHARED_ARTIFACT,
        "reusable Higgs voice-manager artifact directory",
    ),
    ExceptionRule(
        "tldw_chatbook/TTS/backends/kokoro.py",
        "function:initialize",
        "join:.config/tldw_cli",
        3,
        Disposition.SHARED_ARTIFACT,
        "shared Kokoro model and voice asset root",
    ),
    ExceptionRule(
        "tldw_chatbook/TTS/backends/kokoro.py",
        "function:load_model",
        "join:.config/tldw_cli",
        1,
        Disposition.SHARED_ARTIFACT,
        "shared Kokoro model artifact root",
    ),
    ExceptionRule(
        "tldw_chatbook/TTS/kokoro_pytorch.py",
        "module:DEFAULT_MODEL_PATH",
        "join:.config/tldw_cli",
        1,
        Disposition.SHARED_ARTIFACT,
        "shared Kokoro default model artifact root",
    ),
    ExceptionRule(
        "tldw_chatbook/TTS/utils/download_models.py",
        "function:__init__",
        "join:.config/tldw_cli",
        1,
        Disposition.SHARED_ARTIFACT,
        "shared downloadable TTS model artifact root",
    ),
    ExceptionRule(
        "tldw_chatbook/UI/STTS_Window.py",
        "function:_chatterbox_profile_choices",
        "join:.config/tldw_cli",
        1,
        Disposition.SHARED_ARTIFACT,
        "reusable Chatterbox voice profile artifact root",
    ),
    ExceptionRule(
        "tldw_chatbook/UI/STTS_Window.py",
        "function:_higgs_profile_choices",
        "join:.config/tldw_cli",
        1,
        Disposition.SHARED_ARTIFACT,
        "reusable Higgs voice profile artifact root",
    ),
    ExceptionRule(
        "tldw_chatbook/UI/Screens/settings_storage_defaults.py",
        "module:DEFAULT_CHACHANOTES_DB_PATH",
        "literal:~/.local/share/tldw_cli/tldw_chatbook_ChaChaNotes.db",
        1,
        Disposition.PERSISTED_DEFAULT,
        "shipped Settings storage default for the ChaChaNotes database",
    ),
    ExceptionRule(
        "tldw_chatbook/UI/Screens/settings_storage_defaults.py",
        "module:DEFAULT_LIBRARY_COLLECTIONS_DB_PATH",
        "literal:~/.local/share/tldw_cli/tldw_chatbook_library_collections.db",
        1,
        Disposition.PERSISTED_DEFAULT,
        "shipped Settings storage default for the library collections database",
    ),
    ExceptionRule(
        "tldw_chatbook/UI/Screens/settings_storage_defaults.py",
        "module:DEFAULT_MEDIA_DB_PATH",
        "literal:~/.local/share/tldw_cli/tldw_cli_media_v2.db",
        1,
        Disposition.PERSISTED_DEFAULT,
        "shipped Settings storage default for the media database",
    ),
    ExceptionRule(
        "tldw_chatbook/UI/Screens/settings_storage_defaults.py",
        "module:DEFAULT_PROMPTS_DB_PATH",
        "literal:~/.local/share/tldw_cli/tldw_cli_prompts.db",
        1,
        Disposition.PERSISTED_DEFAULT,
        "shipped Settings storage default for the prompts database",
    ),
    ExceptionRule(
        "tldw_chatbook/UI/Screens/settings_storage_defaults.py",
        "module:DEFAULT_RESEARCH_DB_PATH",
        "literal:~/.local/share/tldw_cli/tldw_chatbook_research.db",
        1,
        Disposition.PERSISTED_DEFAULT,
        "shipped Settings storage default for the research database",
    ),
    ExceptionRule(
        "tldw_chatbook/UI/Screens/settings_storage_defaults.py",
        "module:DEFAULT_USER_DB_BASE_DIR",
        "literal:~/.local/share/tldw_cli/",
        1,
        Disposition.PERSISTED_DEFAULT,
        "shipped Settings storage default for the user database root",
    ),
    ExceptionRule(
        "tldw_chatbook/UI/Screens/settings_storage_defaults.py",
        "module:DEFAULT_WORKSPACES_DB_PATH",
        "literal:~/.local/share/tldw_cli/tldw_chatbook_workspaces.db",
        1,
        Disposition.PERSISTED_DEFAULT,
        "shipped Settings storage default for the workspaces database",
    ),
    ExceptionRule(
        "tldw_chatbook/UI/Screens/settings_storage_defaults.py",
        "module:DEFAULT_WRITING_DB_PATH",
        "literal:~/.local/share/tldw_cli/tldw_chatbook_writing.db",
        1,
        Disposition.PERSISTED_DEFAULT,
        "shipped Settings storage default for the writing database",
    ),
    ExceptionRule(
        "tldw_chatbook/UI/Speech/speech_catalog_mixin.py",
        "function:_chatterbox_profile_choices",
        "join:.config/tldw_cli",
        1,
        Disposition.SHARED_ARTIFACT,
        "reusable Chatterbox voice profile artifact root",
    ),
    ExceptionRule(
        "tldw_chatbook/UI/Speech/speech_catalog_mixin.py",
        "function:_higgs_profile_choices",
        "join:.config/tldw_cli",
        1,
        Disposition.SHARED_ARTIFACT,
        "reusable Higgs voice profile artifact root",
    ),
    ExceptionRule(
        "tldw_chatbook/UI/Speech/speech_settings_mixin.py",
        "function:_set_initial_values",
        "literal:~/.config/tldw_cli/chatterbox_voices",
        1,
        Disposition.SHARED_ARTIFACT,
        "reusable Chatterbox voice artifact directory",
    ),
    ExceptionRule(
        "tldw_chatbook/UI/Speech/speech_settings_model.py",
        "module:SETTING_CONFIG_SOURCES",
        "literal:~/.config/tldw_cli/higgs_voices",
        1,
        Disposition.SHARED_ARTIFACT,
        "reusable Higgs voice configuration artifact directory",
    ),
    ExceptionRule(
        "tldw_chatbook/UI/Tools_Settings_Window.py",
        "function:_compose_database_config_form",
        "literal:~/.local/share/tldw_cli/",
        1,
        Disposition.PERSISTED_DEFAULT,
        "shipped Settings storage default displayed in the database form",
    ),
    ExceptionRule(
        "tldw_chatbook/UI/Tools_Settings_Window.py",
        "function:_reset_database_config_form",
        "literal:~/.local/share/tldw_cli/",
        1,
        Disposition.PERSISTED_DEFAULT,
        "shipped Settings storage default restored by the database form",
    ),
    ExceptionRule(
        "tldw_chatbook/UI/Voice_Cloning_Window.py",
        "function:_initialize_backends",
        "literal:~/.config/tldw_cli/chatterbox_voices",
        1,
        Disposition.SHARED_ARTIFACT,
        "reusable Chatterbox voice artifact directory",
    ),
    ExceptionRule(
        "tldw_chatbook/UI/Voice_Cloning_Window.py",
        "function:_initialize_backends",
        "literal:~/.config/tldw_cli/higgs_voices",
        1,
        Disposition.SHARED_ARTIFACT,
        "reusable Higgs voice artifact directory",
    ),
    ExceptionRule(
        "tldw_chatbook/Utils/custom_tokenizers.py",
        "function:__init__",
        "literal:~/.config/tldw_cli/tokenizers",
        1,
        Disposition.SHARED_ARTIFACT,
        "shared custom tokenizer artifact directory",
    ),
    ExceptionRule(
        "tldw_chatbook/config.py",
        "function:_default_base_data_dir",
        "join:.local/share/tldw_cli",
        1,
        Disposition.RESOLVER_SEED,
        "call-time resolver seed for the default user data directory",
    ),
    ExceptionRule(
        "tldw_chatbook/config.py",
        "module:BASE_DATA_DIR_CLI",
        "join:.local/share/tldw_cli",
        1,
        Disposition.COMPATIBILITY_CONSTANT,
        "import-time compatibility constant retained for Prompts_Dump",
    ),
    ExceptionRule(
        "tldw_chatbook/config.py",
        "module:CONFIG_TOML_CONTENT",
        "literal:~/.config/tldw_cli/config.toml",
        1,
        Disposition.PERSISTED_DEFAULT,
        "shipped config template default for the active config file",
    ),
    ExceptionRule(
        "tldw_chatbook/config.py",
        "module:CONFIG_TOML_CONTENT",
        "literal:~/.config/tldw_cli/github_profiles",
        1,
        Disposition.PERSISTED_DEFAULT,
        "shipped config template default for GitHub profiles",
    ),
    ExceptionRule(
        "tldw_chatbook/config.py",
        "module:CONFIG_TOML_CONTENT",
        "literal:~/.local/share/tldw_cli/",
        1,
        Disposition.PERSISTED_DEFAULT,
        "shipped config template base data-directory default",
    ),
    ExceptionRule(
        "tldw_chatbook/config.py",
        "module:CONFIG_TOML_CONTENT",
        "literal:~/.local/share/tldw_cli/evals.db",
        1,
        Disposition.PERSISTED_DEFAULT,
        "shipped config template default for the Evals database",
    ),
    ExceptionRule(
        "tldw_chatbook/config.py",
        "module:CONFIG_TOML_CONTENT",
        "literal:~/.local/share/tldw_cli/models/embeddings",
        1,
        Disposition.PERSISTED_DEFAULT,
        "shipped config template persisted embedding-cache default",
    ),
    ExceptionRule(
        "tldw_chatbook/config.py",
        "module:CONFIG_TOML_CONTENT",
        "literal:~/.local/share/tldw_cli/rag_indexing.db",
        1,
        Disposition.PERSISTED_DEFAULT,
        "shipped config template default for the RAG indexing database",
    ),
    ExceptionRule(
        "tldw_chatbook/config.py",
        "module:CONFIG_TOML_CONTENT",
        "literal:~/.local/share/tldw_cli/tldw_chatbook_ChaChaNotes.db",
        1,
        Disposition.PERSISTED_DEFAULT,
        "shipped config template default for the ChaChaNotes database",
    ),
    ExceptionRule(
        "tldw_chatbook/config.py",
        "module:CONFIG_TOML_CONTENT",
        "literal:~/.local/share/tldw_cli/tldw_chatbook_library_collections.db",
        1,
        Disposition.PERSISTED_DEFAULT,
        "shipped config template default for the library collections database",
    ),
    ExceptionRule(
        "tldw_chatbook/config.py",
        "module:CONFIG_TOML_CONTENT",
        "literal:~/.local/share/tldw_cli/tldw_chatbook_research.db",
        1,
        Disposition.PERSISTED_DEFAULT,
        "shipped config template default for the research database",
    ),
    ExceptionRule(
        "tldw_chatbook/config.py",
        "module:CONFIG_TOML_CONTENT",
        "literal:~/.local/share/tldw_cli/tldw_chatbook_subscriptions.db",
        1,
        Disposition.PERSISTED_DEFAULT,
        "shipped config template default for the subscriptions database",
    ),
    ExceptionRule(
        "tldw_chatbook/config.py",
        "module:CONFIG_TOML_CONTENT",
        "literal:~/.local/share/tldw_cli/tldw_chatbook_workspaces.db",
        1,
        Disposition.PERSISTED_DEFAULT,
        "shipped config template default for the workspaces database",
    ),
    ExceptionRule(
        "tldw_chatbook/config.py",
        "module:CONFIG_TOML_CONTENT",
        "literal:~/.local/share/tldw_cli/tldw_chatbook_writing.db",
        1,
        Disposition.PERSISTED_DEFAULT,
        "shipped config template default for the writing database",
    ),
    ExceptionRule(
        "tldw_chatbook/config.py",
        "module:CONFIG_TOML_CONTENT",
        "literal:~/.local/share/tldw_cli/tldw_cli_media_v2.db",
        1,
        Disposition.PERSISTED_DEFAULT,
        "shipped config template default for the media database",
    ),
    ExceptionRule(
        "tldw_chatbook/config.py",
        "module:CONFIG_TOML_CONTENT",
        "literal:~/.local/share/tldw_cli/tldw_cli_prompts.db",
        1,
        Disposition.PERSISTED_DEFAULT,
        "shipped config template default for the prompts database",
    ),
    ExceptionRule(
        "tldw_chatbook/config.py",
        "module:DEFAULT_CONFIG_PATH",
        "join:.config/tldw_cli",
        1,
        Disposition.RESOLVER_SEED,
        "resolver seed for the default config path",
    ),
)
REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    """Print the source census and enforce the approved exception registry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--print-occurrences",
        action="store_true",
        help="print the complete sorted source census",
    )
    arguments = parser.parse_args(argv)

    occurrences = scan_tree(REPO_ROOT / "tldw_chatbook")
    if arguments.print_occurrences:
        for occurrence in occurrences:
            print(
                f"{occurrence.relative_path}:{occurrence.line}: "
                f"{occurrence.context}: {occurrence.expression}"
            )

    problems = reconcile_inventory(occurrences, APPROVED_EXCEPTIONS)
    for problem in problems:
        print(
            f"{problem.relative_path}:{problem.line}: {problem.context}: "
            f"{problem.expression}: {problem.reason}",
            file=sys.stderr,
        )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
