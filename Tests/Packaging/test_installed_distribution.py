from __future__ import annotations

import configparser
from email.parser import Parser
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from typing import NamedTuple
import zipfile

import pytest

from Tests.reactive_ownership_contract import (
    RETAINED_TLDW_REACTIVES,
    RETIRED_TLDW_REACTIVES,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_NAMES = {
    "academic_paper",
    "code_documentation",
    "conversation",
    "ebook_chapters",
    "json",
    "legal_document",
    "paragraphs",
    "rolling_summarize",
    "semantic",
    "sentences",
    "tokens",
    "words",
    "xml",
}
CITATION_MIGRATION_PATH = (
    "tldw_chatbook/DB/migrations/chachanotes_v26_to_v27_citation_provenance.sql"
)
CHARACTER_AUTHORITY_MIGRATION_PATH = (
    "tldw_chatbook/DB/migrations/chachanotes_v27_to_v28_character_authority.sql"
)
RUNTIME_MIGRATION_PATHS = {
    CITATION_MIGRATION_PATH,
    CHARACTER_AUTHORITY_MIGRATION_PATH,
}
INSTALLED_PROBE = r"""
from pathlib import Path
import ast
import asyncio
from collections import Counter
import json
import math
import os
import sys
import time
import tomllib

expected_target = Path(os.environ["EXPECTED_TARGET"]).resolve(strict=True)
excluded_source_roots = (
    Path(os.environ["CHECKOUT_ROOT"]).resolve(strict=True),
    Path(os.environ["BUILD_SOURCE_ROOT"]).resolve(strict=True),
)
expected_reactives = frozenset(json.loads(os.environ["EXPECTED_REACTIVES"]))
retired_reactives = frozenset(json.loads(os.environ["RETIRED_REACTIVES"]))
assert expected_reactives == {"current_tab", "splash_screen_active"}
assert len(retired_reactives) == 59
assert expected_reactives.isdisjoint(retired_reactives)
default_screen_wait_seconds = 30.0
screen_wait_seconds_env = "TLDW_TEST_SCREEN_WAIT_SECONDS"


def get_screen_wait_seconds():
    raw_value = os.environ.get(
        screen_wait_seconds_env,
        str(default_screen_wait_seconds),
    )
    try:
        seconds = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{screen_wait_seconds_env} must be a positive finite number"
        ) from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(
            f"{screen_wait_seconds_env} must be a positive finite number"
        )
    return seconds


def is_under(path, root):
    return path == root or path.is_relative_to(root)


for entry in sys.path:
    try:
        resolved_entry = Path(entry or os.getcwd()).resolve(strict=True)
    except (FileNotFoundError, OSError):
        continue
    assert not any(is_under(resolved_entry, root) for root in excluded_source_roots), (
        resolved_entry,
        excluded_source_roots,
    )

import tldw_chatbook
from tldw_chatbook.Chunking.chunking_templates import ChunkingTemplateManager
from tldw_chatbook.Constants import TAB_CHAT, TAB_HOME
from tldw_chatbook.Evals.config_loader import EvalConfigLoader
from tldw_chatbook.RAG_Search.pipeline_loader import PipelineLoader
from tldw_chatbook.app import TldwCli, get_app
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.home_screen import HomeScreen

package_root = Path(tldw_chatbook.__file__).resolve().parent
expected_templates = set(json.loads(os.environ["EXPECTED_TEMPLATES"]))
assert package_root.is_relative_to(expected_target)
assert (package_root / "css" / "tldw_cli_modular.tcss").is_file()

with (package_root / "Config_Files" / "rag_pipelines.toml").open("rb") as stream:
    assert "plain" in tomllib.load(stream)["pipelines"]

loader = PipelineLoader(config_dir=package_root / "Config_Files")
loader.load_pipeline_config()
assert "plain" in loader.pipelines
assert set(ChunkingTemplateManager().get_available_templates()) == expected_templates
assert "code_execution" in EvalConfigLoader().get_task_types()
assert (package_root / "Third_Party" / "aider" / "LICENSE.txt").is_file()
assert (
    package_root / "Third_Party" / "textual_fspicker" / "LICENSE"
).is_file()


def chain(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = chain(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def bound_names(node):
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(name for item in node.elts for name in bound_names(item))
    return ()


def class_body_reactives(class_node):
    names = set()
    for statement in class_node.body:
        if isinstance(statement, ast.Assign):
            targets = statement.targets
            value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            targets = (statement.target,)
            value = statement.value
        else:
            continue
        if not (
            isinstance(value, ast.Call)
            and chain(value.func).rsplit(".", 1)[-1] == "reactive"
        ):
            continue
        for target in targets:
            names.update(bound_names(target))
    return frozenset(names)


def is_root_app(node):
    expression = chain(node)
    return bool(expression) and expression.rsplit(".", 1)[-1] in {
        "app",
        "app_instance",
    }


def is_root_mapping(node, root_predicate):
    if root_predicate(node):
        return True
    if (
        isinstance(node, ast.Attribute)
        and node.attr == "__dict__"
        and root_predicate(node.value)
    ):
        return True
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "vars"
        and len(node.args) == 1
        and root_predicate(node.args[0])
    )


def constant_name(node):
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"getattr", "setattr", "delattr", "hasattr"}
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    ):
        return node.args[1].value
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ):
        return node.args[0].value
    return None


def root_accesses(tree, relative_path):
    found = []
    for node in ast.walk(tree):
        target = None
        if (
            isinstance(node, ast.Attribute)
            and node.attr in retired_reactives
            and is_root_app(node.value)
        ):
            target = node.attr
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"getattr", "setattr", "delattr", "hasattr"}
            and len(node.args) >= 2
            and is_root_app(node.args[0])
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in retired_reactives
        ):
            target = node.args[1].value
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and constant_name(node) in retired_reactives
            and is_root_mapping(node.func.value, is_root_app)
            and not is_root_app(node.func.value)
        ):
            target = constant_name(node)
        elif (
            isinstance(node, ast.Subscript)
            and is_root_mapping(node.value, is_root_app)
            and isinstance(node.slice, ast.Constant)
            and node.slice.value in retired_reactives
        ):
            target = node.slice.value
        if target is not None:
            found.append((relative_path, node.lineno, target))
    return found


class TldwCliRetiredAccesses(ast.NodeVisitor):
    def __init__(self):
        self.nested_class_depth = 0
        self.found = []

    def root_receiver(self, node):
        return is_root_app(node) or (
            self.nested_class_depth == 0 and chain(node) == "self"
        )

    def visit_ClassDef(self, node):
        self.nested_class_depth += 1
        self.generic_visit(node)
        self.nested_class_depth -= 1

    def visit_Attribute(self, node):
        if node.attr in retired_reactives and self.root_receiver(node.value):
            self.found.append((node.lineno, node.attr))
        self.generic_visit(node)

    def visit_Call(self, node):
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in {"getattr", "setattr", "delattr", "hasattr"}
            and len(node.args) >= 2
            and self.root_receiver(node.args[0])
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in retired_reactives
        ):
            self.found.append((node.lineno, node.args[1].value))
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and constant_name(node) in retired_reactives
            and is_root_mapping(node.func.value, self.root_receiver)
            and not self.root_receiver(node.func.value)
        ):
            self.found.append((node.lineno, constant_name(node)))
        self.generic_visit(node)

    def visit_Subscript(self, node):
        if (
            is_root_mapping(node.value, self.root_receiver)
            and isinstance(node.slice, ast.Constant)
            and node.slice.value in retired_reactives
        ):
            self.found.append((node.lineno, node.slice.value))
        self.generic_visit(node)

    def visit_keyword(self, node):
        if (
            self.nested_class_depth == 0
            and node.arg == "reactive_attr"
            and isinstance(node.value, ast.Constant)
            and node.value.value in retired_reactives
        ):
            self.found.append((node.lineno, node.value.value))
        self.generic_visit(node)


app_path = package_root / "app.py"
app_tree = ast.parse(app_path.read_text(encoding="utf-8"), filename=str(app_path))
app_class = next(
    node
    for node in app_tree.body
    if isinstance(node, ast.ClassDef) and node.name == "TldwCli"
)
local_classes = {
    node.name: node for node in app_tree.body if isinstance(node, ast.ClassDef)
}
root_owner_classes = []
seen_root_classes = set()


def add_root_owner_class(class_node):
    if class_node.name in seen_root_classes:
        return
    seen_root_classes.add(class_node.name)
    root_owner_classes.append(class_node)
    for base in class_node.bases:
        base_class = local_classes.get(base.id) if isinstance(base, ast.Name) else None
        if base_class is not None:
            add_root_owner_class(base_class)


add_root_owner_class(app_class)
assert (
    frozenset().union(
        *(class_body_reactives(node) for node in root_owner_classes)
    )
    == expected_reactives
)
root_methods = {
    node.name
    for owner in root_owner_classes
    for node in owner.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
}
assert "watch_current_tab" not in root_methods
assert retired_reactives.isdisjoint(root_methods)
assert {f"watch_{name}" for name in retired_reactives}.isdisjoint(root_methods)
class_level_retired = []
for owner in root_owner_classes:
    for statement in owner.body:
        if isinstance(statement, ast.Assign):
            targets = statement.targets
        elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
            targets = (statement.target,)
        else:
            continue
        for target in targets:
            class_level_retired.extend(
                name for name in bound_names(target) if name in retired_reactives
            )
assert class_level_retired == []

tldw_accesses = []
for owner in root_owner_classes:
    owner_accesses = TldwCliRetiredAccesses()
    for statement in owner.body:
        owner_accesses.visit(statement)
    tldw_accesses.extend(
        (owner.name, line, name) for line, name in owner_accesses.found
    )
assert tldw_accesses == []

installed_root_accesses = []
for source_path in sorted(package_root.rglob("*.py")):
    source_tree = ast.parse(
        source_path.read_text(encoding="utf-8"),
        filename=str(source_path),
    )
    installed_root_accesses.extend(
        root_accesses(source_tree, source_path.relative_to(package_root).as_posix())
    )
assert installed_root_accesses == []

wiring_methods = (
    "_wire_writing_services",
    "_wire_chat_conversation_services",
)
expected_wiring_calls = Counter({name: 1 for name in wiring_methods})
wiring_calls = Counter()
for method_name in wiring_methods:
    original = getattr(TldwCli, method_name)

    def counted(
        self,
        _original=original,
        _method_name=method_name,
    ):
        wiring_calls[_method_name] += 1
        _original(self)

    setattr(TldwCli, method_name, counted)

sync_consumer_classes = (
    sys.modules[TldwCli.__module__].ChatConversationScopeService,
    sys.modules[TldwCli.__module__].MediaReadingScopeService,
)
initial_sync_arguments = {
    consumer.__name__: [] for consumer in sync_consumer_classes
}
for consumer in sync_consumer_classes:
    original_init = consumer.__init__

    def captured_init(
        self,
        *args,
        _original=original_init,
        _consumer_name=consumer.__name__,
        **kwargs,
    ):
        initial_sync_arguments[_consumer_name].append(
            kwargs.get("sync_scope_service")
        )
        _original(self, *args, **kwargs)

    consumer.__init__ = captured_init


def service_identities(app):
    return tuple(
        getattr(app, name)
        for name in (
            "local_writing_service",
            "server_writing_service",
            "writing_scope_service",
            "local_chat_conversation_service",
            "conversation_local_marks_service",
            "server_chat_conversation_service",
            "chat_conversation_scope_service",
            "citation_trace_repository",
            "citation_legacy_migration_service",
            "citation_artifact_ownership_coordinator",
            "media_reading_scope_service",
            "sync_scope_service",
        )
    )


def assert_service_identities(app, expected):
    current = service_identities(app)
    assert len(current) == len(expected)
    assert all(
        actual is original
        for actual, original in zip(current, expected, strict=True)
    )


def assert_service_graph(app):
    assert app.writing_scope_service.local_service is app.local_writing_service
    assert app.writing_scope_service.server_service is app.server_writing_service
    assert app.server_writing_service.client_provider is app.server_context_provider
    assert (
        app.chat_conversation_scope_service.local_service
        is app.local_chat_conversation_service
    )
    assert (
        app.chat_conversation_scope_service.server_service
        is app.server_chat_conversation_service
    )
    assert (
        app.chat_conversation_scope_service.sync_scope_service
        is app.sync_scope_service
    )
    assert app.media_reading_scope_service.sync_scope_service is app.sync_scope_service
    assert (
        app.local_chat_conversation_service.citation_legacy_migration
        is app.citation_legacy_migration_service
    )
    assert (
        app.citation_artifact_ownership_coordinator.trace_repository
        is app.citation_trace_repository
    )
    assert (
        app.citation_artifact_ownership_coordinator.artifact_store
        is app.local_chatbook_service
    )


app = get_app()
assert isinstance(app, TldwCli)
assert wiring_calls == expected_wiring_calls
assert initial_sync_arguments == {
    consumer.__name__: [app.sync_scope_service]
    for consumer in sync_consumer_classes
}
assert_service_graph(app)
initial_service_identities = service_identities(app)
assert all(not hasattr(app, name) for name in retired_reactives)


async def wait_for(pilot, predicate, failure):
    deadline = time.monotonic() + get_screen_wait_seconds()
    while time.monotonic() < deadline:
        if predicate():
            return
        await pilot.pause(0.01)
    raise AssertionError(failure)


async def exercise_production_app():
    async with app.run_test(size=(120, 40)) as pilot:
        await wait_for(
            pilot,
            lambda: (
                type(app.screen) is HomeScreen
                and app.current_tab == TAB_HOME
                and app.screen.is_mounted
            ),
            "installed production app did not mount registered Home",
        )
        await app.handle_screen_navigation(NavigateToScreen(TAB_CHAT))
        await wait_for(
            pilot,
            lambda: (
                type(app.screen) is ChatScreen
                and app.current_tab == TAB_CHAT
                and app.screen.is_mounted
            ),
            "installed production app did not navigate to registered Chat",
        )
        assert all(not hasattr(app, name) for name in retired_reactives)
        assert wiring_calls == expected_wiring_calls
        assert_service_identities(app, initial_service_identities)
        assert_service_graph(app)


asyncio.run(exercise_production_app())
assert wiring_calls == expected_wiring_calls
assert_service_identities(app, initial_service_identities)
assert_service_graph(app)

loaded_package_paths = []
for module_name, module in tuple(sys.modules.items()):
    if module_name != "tldw_chatbook" and not module_name.startswith(
        "tldw_chatbook."
    ):
        continue
    module_file = getattr(module, "__file__", None)
    if module_file:
        loaded_package_paths.append((module_name, Path(module_file).resolve(strict=True)))
    module_path = getattr(module, "__path__", None)
    if module_path:
        loaded_package_paths.extend(
            (module_name, Path(path).resolve(strict=True)) for path in module_path
        )

assert loaded_package_paths
for module_name, loaded_path in loaded_package_paths:
    assert is_under(loaded_path, expected_target), (module_name, loaded_path)
    assert not any(
        is_under(loaded_path, source_root) for source_root in excluded_source_roots
    ), (module_name, loaded_path, excluded_source_roots)

print(package_root)
"""


class BuiltDistributions(NamedTuple):
    source_root: Path
    dist_dir: Path
    sdist: Path
    wheel: Path


def _copy_build_inputs(destination: Path) -> None:
    ignored = shutil.ignore_patterns(
        "__pycache__",
        "*.pyc",
        "*.pyo",
        ".DS_Store",
        "build",
        "dist",
        "*.egg-info",
    )
    for name in ("tldw_chatbook", "Packaging"):
        shutil.copytree(REPO_ROOT / name, destination / name, ignore=ignored)

    seen_test_trees: set[tuple[int, int]] = set()
    for name in ("Tests", "tests", "STests"):
        source = REPO_ROOT / name
        if not source.is_dir():
            continue
        stat = source.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in seen_test_trees:
            continue
        seen_test_trees.add(identity)
        shutil.copytree(source, destination / name, ignore=ignored)

    for name in (
        "pyproject.toml",
        "MANIFEST.in",
        "README.md",
        "LICENSE",
        "CLAUDE.md",
        "CHANGELOG.md",
        "requirements.txt",
    ):
        source = REPO_ROOT / name
        if source.is_file():
            shutil.copy2(source, destination / name)


@pytest.fixture(scope="module")
def built_distributions(tmp_path_factory: pytest.TempPathFactory) -> BuiltDistributions:
    source_root = tmp_path_factory.mktemp("distribution-source")
    _copy_build_inputs(source_root)
    dist_dir = source_root / "dist"
    command = [
        sys.executable,
        "-m",
        "build",
        "--sdist",
        "--wheel",
        "--no-isolation",
        "--outdir",
        str(dist_dir),
    ]
    completed = subprocess.run(
        command,
        cwd=source_root,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, (
        f"command: {command}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert "`project.license` as a TOML table is deprecated" not in (
        completed.stdout + completed.stderr
    )
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    wheels = sorted(dist_dir.glob("*.whl"))
    assert len(sdists) == 1
    assert len(wheels) == 1
    return BuiltDistributions(source_root, dist_dir, sdists[0], wheels[0])


def _sdist_members(path: Path) -> set[str]:
    with tarfile.open(path, "r:gz") as archive:
        files = [member.name for member in archive.getmembers() if member.isfile()]
    roots = {name.split("/", 1)[0] for name in files}
    assert len(roots) == 1
    return {name.split("/", 1)[1] for name in files if "/" in name}


def _wheel_members(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        return {name for name in archive.namelist() if not name.endswith("/")}


def _run_manifest_checker(
    built: BuiltDistributions,
    dist_dir: Path,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(built.source_root / "Packaging" / "check_manifest.py"),
            str(dist_dir),
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _install_wheel(
    built: BuiltDistributions,
    target: Path,
) -> None:
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--target",
        str(target),
        str(built.wheel),
    ]
    completed = subprocess.run(
        command,
        cwd=target.parent,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, (
        f"command: {command}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


def _target_hashes(target: Path) -> dict[str, str]:
    return {
        path.relative_to(target).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(target.rglob("*"))
        if path.is_file()
    }


def _private_child_env(
    state_root: Path,
    target: Path,
    build_source_root: Path,
) -> dict[str, str]:
    state_root = state_root.resolve(strict=True)
    target = target.resolve(strict=True)
    checkout_root = REPO_ROOT.resolve(strict=True)
    build_source_root = build_source_root.resolve(strict=True)
    config_root = state_root / "config"
    data_root = state_root / "data"
    temp_root = state_root / "tmp"
    for path in (config_root, data_root, temp_root):
        path.mkdir(parents=True, mode=0o700, exist_ok=True)
    config_path = config_root / "config.toml"
    config_path.write_text(
        '[general]\ndefault_tab = "home"\n\n'
        "[first_run]\nsetup_completed = true\n\n"
        "[splash_screen]\nenabled = false\n",
        encoding="utf-8",
    )
    config_path.chmod(0o600)

    env = os.environ.copy()
    for name in ("TLDW_TEST_CONFIG_ROOT", "TLDW_TEST_CONFIG_ROOT_OWNER"):
        env.pop(name, None)
    env.update(
        {
            "HOME": str(state_root),
            "USERPROFILE": str(state_root),
            "APPDATA": str(data_root),
            "LOCALAPPDATA": str(data_root),
            "XDG_CONFIG_HOME": str(config_root),
            "XDG_DATA_HOME": str(data_root),
            "TLDW_CONFIG_PATH": str(config_path),
            "TMPDIR": str(temp_root),
            "TEMP": str(temp_root),
            "TMP": str(temp_root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(target),
            "EXPECTED_TARGET": str(target),
            "CHECKOUT_ROOT": str(checkout_root),
            "BUILD_SOURCE_ROOT": str(build_source_root),
            "EXPECTED_REACTIVES": json.dumps(sorted(RETAINED_TLDW_REACTIVES)),
            "RETIRED_REACTIVES": json.dumps(sorted(RETIRED_TLDW_REACTIVES)),
            "EXPECTED_TEMPLATES": json.dumps(sorted(TEMPLATE_NAMES)),
        }
    )
    return env


def _run_child(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, (
        f"command: {command}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return completed


def test_built_artifacts_match_distribution_contract(
    built_distributions: BuiltDistributions,
) -> None:
    sdist_members = _sdist_members(built_distributions.sdist)
    wheel_members = _wheel_members(built_distributions.wheel)

    required_sdist = {
        "LICENSE",
        "README.md",
        "CLAUDE.md",
        "CHANGELOG.md",
        "MANIFEST.in",
        "pyproject.toml",
        "requirements.txt",
        "tldw_chatbook/css/tldw_cli_modular.tcss",
        "tldw_chatbook/css/components/stats_screen.css",
        "tldw_chatbook/Config_Files/rag_pipelines.toml",
        "tldw_chatbook/Evals/config/eval_config.yaml",
        "tldw_chatbook/Third_Party/aider/LICENSE.txt",
        "tldw_chatbook/Third_Party/textual_fspicker/LICENSE",
    } | RUNTIME_MIGRATION_PATHS
    required_wheel = {
        "tldw_chatbook/css/tldw_cli_modular.tcss",
        "tldw_chatbook/Config_Files/rag_pipelines.toml",
        "tldw_chatbook/Evals/config/eval_config.yaml",
        "tldw_chatbook/Third_Party/aider/LICENSE.txt",
        "tldw_chatbook/Third_Party/textual_fspicker/LICENSE",
    } | RUNTIME_MIGRATION_PATHS
    assert not required_sdist - sdist_members
    assert not required_wheel - wheel_members

    wheel_templates = {
        Path(name).stem
        for name in wheel_members
        if name.startswith("tldw_chatbook/Chunking/templates/")
        and name.endswith(".json")
    }
    assert wheel_templates == TEMPLATE_NAMES

    forbidden_wheel = {
        "tldw_chatbook/css/components/stats_screen.css",
        "tldw_chatbook/Config_Files/embedding_configs_examples.toml",
        "tldw_chatbook/Config_Files/pipeline_configs/custom_pipelines_example.toml",
        "tldw_chatbook/Chunking/templates/README.md",
        "tldw_chatbook/Chunking/templates/example_usage.py",
        "tldw_chatbook/Evals/DEVELOPER_GUIDE.md",
    }
    assert forbidden_wheel.isdisjoint(wheel_members)
    for members in (sdist_members, wheel_members):
        assert not any(
            name.startswith(("Tests/", "tests/", "STests/"))
            or "/__pycache__/" in name
            or name.endswith((".pyc", ".pyo", ".DS_Store"))
            for name in members
        )

    with zipfile.ZipFile(built_distributions.wheel) as archive:
        metadata_name = next(
            name for name in wheel_members if name.endswith(".dist-info/METADATA")
        )
        entry_points_name = next(
            name
            for name in wheel_members
            if name.endswith(".dist-info/entry_points.txt")
        )
        metadata = Parser().parsestr(archive.read(metadata_name).decode("utf-8"))
        entry_points = configparser.ConfigParser()
        entry_points.read_string(archive.read(entry_points_name).decode("utf-8"))

    with tarfile.open(built_distributions.sdist, "r:gz") as archive:
        pkg_info = next(
            member
            for member in archive.getmembers()
            if member.isfile() and member.name.endswith("/PKG-INFO")
        )
        pkg_info_stream = archive.extractfile(pkg_info)
        assert pkg_info_stream is not None
        sdist_metadata = Parser().parsestr(pkg_info_stream.read().decode("utf-8"))

    assert metadata["Metadata-Version"] == "2.4"
    assert metadata["License-Expression"] == "AGPL-3.0-or-later"
    assert "LICENSE" in (metadata.get_all("License-File") or [])
    assert sdist_metadata["Metadata-Version"] == "2.4"
    assert sdist_metadata["License-Expression"] == "AGPL-3.0-or-later"
    assert "LICENSE" in (sdist_metadata.get_all("License-File") or [])
    assert any(name.endswith(".dist-info/licenses/LICENSE") for name in wheel_members)
    assert dict(entry_points["console_scripts"]) == {
        "tldw-cli": "tldw_chatbook.cli:main_cli_runner",
        "tldw-serve": "tldw_chatbook.Web_Server.serve:main",
    }


def test_release_checker_accepts_fresh_artifacts(
    built_distributions: BuiltDistributions,
    tmp_path: Path,
) -> None:
    result = _run_manifest_checker(
        built_distributions,
        built_distributions.dist_dir,
        tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_release_checker_rejects_multiple_wheels(
    built_distributions: BuiltDistributions,
    tmp_path: Path,
) -> None:
    dist_dir = tmp_path / "dist"
    shutil.copytree(built_distributions.dist_dir, dist_dir)
    shutil.copy2(
        built_distributions.wheel,
        dist_dir / f"duplicate-{built_distributions.wheel.name}",
    )

    result = _run_manifest_checker(built_distributions, dist_dir, tmp_path)

    assert result.returncode == 1
    assert "exactly one wheel" in (result.stdout + result.stderr).lower()


def test_release_checker_rejects_sdist_only_css_in_wheel(
    built_distributions: BuiltDistributions,
    tmp_path: Path,
) -> None:
    dist_dir = tmp_path / "dist"
    shutil.copytree(built_distributions.dist_dir, dist_dir)
    wheel = next(dist_dir.glob("*.whl"))
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr(
            "tldw_chatbook/css/components/stats_screen.css",
            "forbidden",
        )

    result = _run_manifest_checker(built_distributions, dist_dir, tmp_path)

    assert result.returncode == 1
    assert "stats_screen.css" in result.stdout + result.stderr


def test_release_checker_rejects_missing_runtime_data(
    built_distributions: BuiltDistributions,
    tmp_path: Path,
) -> None:
    dist_dir = tmp_path / "dist"
    shutil.copytree(built_distributions.dist_dir, dist_dir)
    wheel = next(dist_dir.glob("*.whl"))
    rewritten = wheel.with_suffix(".rewritten")
    missing = "tldw_chatbook/Evals/config/eval_config.yaml"
    with (
        zipfile.ZipFile(wheel) as source,
        zipfile.ZipFile(rewritten, "w") as destination,
    ):
        for member in source.infolist():
            if member.filename != missing:
                destination.writestr(member, source.read(member.filename))
    rewritten.replace(wheel)

    result = _run_manifest_checker(built_distributions, dist_dir, tmp_path)

    assert result.returncode == 1
    assert missing in result.stdout + result.stderr


@pytest.mark.parametrize("missing", sorted(RUNTIME_MIGRATION_PATHS))
def test_release_checker_rejects_missing_database_migration(
    built_distributions: BuiltDistributions,
    tmp_path: Path,
    missing: str,
) -> None:
    dist_dir = tmp_path / "dist"
    shutil.copytree(built_distributions.dist_dir, dist_dir)
    wheel = next(dist_dir.glob("*.whl"))
    rewritten = wheel.with_suffix(".rewritten")
    with (
        zipfile.ZipFile(wheel) as source,
        zipfile.ZipFile(rewritten, "w") as destination,
    ):
        for member in source.infolist():
            if member.filename != missing:
                destination.writestr(member, source.read(member.filename))
    rewritten.replace(wheel)

    result = _run_manifest_checker(built_distributions, dist_dir, tmp_path)

    assert result.returncode == 1
    assert missing in result.stdout + result.stderr


def test_installed_wheel_loaders_entry_points_and_assets_are_immutable(
    built_distributions: BuiltDistributions,
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    state_root = tmp_path / "state"
    run_root = tmp_path / "run"
    state_root.mkdir(mode=0o700)
    run_root.mkdir()
    _install_wheel(built_distributions, target)
    env = _private_child_env(
        state_root,
        target,
        built_distributions.source_root,
    )
    before = _target_hashes(target)
    results = [_run_child([sys.executable, "-c", INSTALLED_PROBE], run_root, env)]

    script_path = os.pathsep.join(
        str(path) for path in (target / "bin", target / "Scripts")
    )
    for name in ("tldw-cli", "tldw-serve"):
        script = shutil.which(name, path=script_path)
        assert script is not None, (
            f"missing installed script {name!r}; "
            f"target files: {sorted(_target_hashes(target))}"
        )
        results.append(_run_child([script, "--help"], run_root, env))

    after = _target_hashes(target)
    process_text = "\n".join(result.stdout + "\n" + result.stderr for result in results)
    log_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in state_root.rglob("*.log*")
        if path.is_file()
    )
    observed_text = process_text + "\n" + log_text
    for forbidden in (
        "Building modular CSS",
        "Failed to build modular CSS",
        "Error handling CSS file",
    ):
        assert forbidden not in observed_text
    assert after == before
