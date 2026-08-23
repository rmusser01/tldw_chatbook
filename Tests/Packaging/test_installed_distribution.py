from __future__ import annotations

import configparser
from contextlib import contextmanager
from email.parser import Parser
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tarfile
from typing import Iterable, Iterator, NamedTuple
import venv
import zipfile

import pytest

from Tests.reactive_ownership_contract import (
    RETAINED_TLDW_REACTIVES,
    RETIRED_TLDW_REACTIVES,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
# The file template store (13 JSON + README.md + example_usage.py) was
# deleted (spec §8.1.2): no tldw_chatbook/Chunking/templates/ path may ship
# in either artifact, and the installed tree must not carry the directory.
CHUNKING_TEMPLATES_PREFIX = "tldw_chatbook/Chunking/templates/"
# Migration expectations are DERIVED, never listed (task-19860). The
# fifteen hand-written constants this replaced had drifted to thirteen files
# (one was even defined twice), and the app cannot start without two of the
# ones nobody added: a wheel built from that list died at V40->V41 with a
# SchemaError, and the two later gaps were invisible because the chain
# aborts at the first.
MIGRATIONS_PREFIX = "tldw_chatbook/DB/migrations/"
CHACHANOTES_DB_MODULE_PATH = "tldw_chatbook/DB/ChaChaNotes_DB.py"
# Matches ``Path(__file__).parent / "migrations" / "<name>.sql"``, the form
# every file-backed migration step uses to locate its script.
RUNTIME_MIGRATION_READ = re.compile(r'"migrations"\s*/\s*"([^"\n]+\.sql)"')


def _source_migration_paths(repo_root: Path) -> frozenset[str]:
    """Return every migration script present in a checkout."""
    directory = repo_root / "tldw_chatbook" / "DB" / "migrations"
    return frozenset(
        f"{MIGRATIONS_PREFIX}{path.name}" for path in directory.glob("*.sql")
    )


def _runtime_migration_paths(module_source: str) -> frozenset[str]:
    """Return the migrations a schema-runner source text opens at runtime."""
    return frozenset(
        f"{MIGRATIONS_PREFIX}{name}"
        for name in RUNTIME_MIGRATION_READ.findall(module_source)
    )


SOURCE_MIGRATION_PATHS = _source_migration_paths(REPO_ROOT)
RUNTIME_MIGRATION_PATHS = _runtime_migration_paths(
    (REPO_ROOT / CHACHANOTES_DB_MODULE_PATH).read_text(encoding="utf-8")
)
SAMIRA_RESOURCE_ROOT = "tldw_chatbook/assets/characters/samira"
SAMIRA_REACTION_LABELS = (
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "neutral",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "thinking",
    "speaking",
    "error",
)
SAMIRA_RESOURCE_PATHS = {
    f"{SAMIRA_RESOURCE_ROOT}/{name}"
    for name in (
        "ASSET_LICENSE.md",
        "Samira.character.json",
        "Sammy.png",
        "visual_identity_pack.json",
    )
} | {
    f"{SAMIRA_RESOURCE_ROOT}/expressions/{label}.webp"
    for label in SAMIRA_REACTION_LABELS
}
AUDIO_CPP_ARTIFACT_MANIFEST_PATH = "tldw_chatbook/TTS/audio_cpp_artifact_manifest.json"
AUDIO_CPP_ARTIFACT_REPOSITORY = "audio-cpp/audio.cpp-gguf"
AUDIO_CPP_ARTIFACT_COMMIT = "597048d9a920592808d7d4e2acd7b9c4596a143a"
_PRIVATE_CHILD_BASELINE_ENV_KEYS = (
    "PATH",
    "LANG",
    "LC_ALL",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
)
_BUILD_TOOL_DISTRIBUTIONS = (
    "build",
    "setuptools",
    "wheel",
    "packaging",
    "pyproject_hooks",
)
_BUILD_ENV_KEYS = frozenset(_PRIVATE_CHILD_BASELINE_ENV_KEYS) | {
    "HOME",
    "USERPROFILE",
    "TMPDIR",
    "TEMP",
    "TMP",
    "PYTHONDONTWRITEBYTECODE",
    "PIP_CONFIG_FILE",
    "PIP_DISABLE_PIP_VERSION_CHECK",
    "PIP_NO_INDEX",
}
BUILD_TOOL_PROBE = r"""
import importlib.util
import os

assert "PYTHONPATH" not in os.environ
assert not any(
    "PROXY" in name.upper()
    or any(
        marker in name.upper()
        for marker in (
            "API_KEY",
            "APIKEY",
            "TOKEN",
            "SECRET",
            "PASSWORD",
            "CREDENTIAL",
        )
    )
    for name in os.environ
)
for module_name in ("build", "setuptools", "wheel", "packaging", "pyproject_hooks"):
    __import__(module_name)
assert importlib.util.find_spec("PIL") is None
assert importlib.util.find_spec("tldw_chatbook") is None
print("curated-build-tools-ok")
"""
INSTALLED_PROBE = r"""
from pathlib import Path
import ast
import asyncio
from collections import Counter
import importlib.util
import json
import math
import os
import sys
import time
import tomllib


def is_sensitive_environment_name(name):
    normalized = name.upper()
    return "PROXY" in normalized or any(
        marker in normalized
        for marker in (
            "API_KEY",
            "APIKEY",
            "TOKEN",
            "SECRET",
            "PASSWORD",
            "CREDENTIAL",
        )
    )


assert not any(is_sensitive_environment_name(name) for name in os.environ)

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
for retired_module in (
    "tldw_chatbook.Audio.transcription_history",
    "tldw_chatbook.Widgets.transcription_history_viewer",
    "tldw_chatbook.UI.Dictation_Window",
    "tldw_chatbook.Chunking.chunking_templates",
):
    assert importlib.util.find_spec(retired_module) is None

from tldw_chatbook.config import get_cli_config_path, get_user_data_dir

assert get_cli_config_path().is_relative_to(Path(os.environ["HOME"]))
assert get_user_data_dir().is_relative_to(Path(os.environ["HOME"]))

# The file template store is deleted (spec §8.1.1): the module is gone AND
# the package root no longer re-exports its names (the vendored engine's
# ChunkingTemplate -- same public name, different class -- is deliberately
# NOT re-exported either; nothing outside the service layer resolves
# templates, spec §8.2).
import tldw_chatbook.Chunking as _installed_chunking

for _retired_chunking_export in (
    "ChunkingTemplateManager",
    "ChunkingPipeline",
    "ChunkingStage",
    "ChunkingOperation",
    "ChunkingTemplate",
):
    assert not hasattr(_installed_chunking, _retired_chunking_export), (
        f"installed tldw_chatbook.Chunking still exports {_retired_chunking_export!r}"
    )
from tldw_chatbook.Constants import TAB_CHAT, TAB_HOME
from tldw_chatbook.Evals.config_loader import EvalConfigLoader
from tldw_chatbook.RAG_Search.pipeline_loader import PipelineLoader
from tldw_chatbook.runtime_policy.server_context import RuntimeServerContextProvider
from tldw_chatbook.Utils.log_sanitizer import sanitize_dict, sanitize_string

assert sanitize_string("claude-opus-4-20250514") == "claude-opus-4-20250514"
assert sanitize_dict({"x-api-key": "PRIVATE_INSTALLED_SENTINEL"}) == {
    "x-api-key": "***REDACTED***"
}
assert "PRIVATE_INSTALLED_SENTINEL" not in sanitize_string(
    'x-api-key="PRIVATE_INSTALLED_SENTINEL"'
)


def deny_server_client_construction(_self):
    raise AssertionError("installed probe attempted server client construction")


RuntimeServerContextProvider.build_client = deny_server_client_construction

from tldw_chatbook.app import TldwCli, get_app
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.home_screen import HomeScreen

package_root = Path(tldw_chatbook.__file__).resolve().parent
assert package_root.is_relative_to(expected_target)
assert (package_root / "css" / "tldw_cli_modular.tcss").is_file()

with (package_root / "Config_Files" / "rag_pipelines.toml").open("rb") as stream:
    assert "plain" in tomllib.load(stream)["pipelines"]

loader = PipelineLoader(config_dir=package_root / "Config_Files")
loader.load_pipeline_config()
assert "plain" in loader.pipelines
assert not (package_root / "Chunking" / "templates").exists(), (
    "the deleted file template store must not be installed"
)
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
            "server_sync_service",
            "local_first_sync_service",
            "manual_sync_control_service",
            "sync_v2_dataset_keys",
            "sync_state_repository",
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
    assert app.server_sync_service.client is None
    assert app.server_sync_service.client_provider is app.server_context_provider
    assert app.server_sync_service.state_repository is app.sync_state_repository
    assert app.sync_scope_service.server_service is app.server_sync_service
    assert app.sync_scope_service.state_repository is app.sync_state_repository
    assert app.local_first_sync_service.server_service is app.server_sync_service
    assert app.local_first_sync_service.state_repository is app.sync_state_repository
    assert app.local_first_sync_service.local_store is None
    assert app.local_first_sync_service.dataset_keys is app.sync_v2_dataset_keys
    assert app.sync_v2_dataset_keys == {}
    assert (
        app.manual_sync_control_service.local_first_sync_service
        is app.local_first_sync_service
    )
    assert app.manual_sync_control_service.state_repository is app.sync_state_repository
    assert app.manual_sync_control_service.dataset_keys is app.sync_v2_dataset_keys


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
assert app.server_context_provider._cached_client is None

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

INSTALLED_MIGRATION_PROBE = r"""
from pathlib import Path
import os

expected_target = Path(os.environ["EXPECTED_TARGET"]).resolve(strict=True)

import tldw_chatbook
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Utils.path_validation import validate_path

package_file = Path(tldw_chatbook.__file__).resolve(strict=True)
assert package_file.is_relative_to(expected_target), (package_file, expected_target)

home_path = Path(os.environ["HOME"]).resolve(strict=True)
migration_path = validate_path("installed-migration-probe.sqlite", home_path)
# The migration target is read from the installed distribution inside this
# child process -- never a hand-maintained literal (task-19044; the pinned
# number went stale on two consecutive schema bumps). The only guard needed
# is that the fixed v35 baseline below remains a genuine downgrade.
current_schema_version = CharactersRAGDB._CURRENT_SCHEMA_VERSION
assert current_schema_version > 35

# A from-scratch initialization first: this is precisely what a user gets
# after `pip install tldw_chatbook`, and it walks the WHOLE v4->current
# chain, reading every file-backed migration off the installed tree. A wheel
# short of one script dies here with a SchemaError (task-19860). The reached
# version is read back out of the database, never asserted from the constant.
fresh_path = validate_path("installed-fresh-probe.sqlite", home_path)
assert not fresh_path.exists()
fresh_db = CharactersRAGDB(fresh_path, client_id="installed-probe-fresh")
fresh_version = fresh_db.get_connection().execute(
    "SELECT version FROM db_schema_version WHERE schema_name = ?",
    (CharactersRAGDB._SCHEMA_NAME,),
).fetchone()[0]
fresh_db.close_connection()
assert fresh_version == current_schema_version, (fresh_version, current_schema_version)
print(f"installed-wheel-fresh-init-ok v{fresh_version}")
CharactersRAGDB._CURRENT_SCHEMA_VERSION = 35
try:
    legacy_db = CharactersRAGDB(migration_path, client_id="installed-probe-v35")
    assert legacy_db._get_db_version(legacy_db.get_connection()) == 35
    legacy_db.close_connection()
finally:
    CharactersRAGDB._CURRENT_SCHEMA_VERSION = current_schema_version

upgraded_db = CharactersRAGDB(migration_path, client_id="installed-probe-current")
upgraded_connection = upgraded_db.get_connection()
assert upgraded_db._get_db_version(upgraded_connection) == current_schema_version
installed_tables = {
    row[0]
    for row in upgraded_connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    )
}
assert {"note_folders", "note_folder_memberships"} <= installed_tables
assert {
    "visual_identity_packs",
    "visual_identity_pack_versions",
    "visual_identity_assets",
    "visual_identity_bindings",
} <= installed_tables
assert "transcript_annotations" in installed_tables
upgraded_db.close_connection()
print(f"installed-wheel-v35-to-current-ok v{current_schema_version}")
"""

INSTALLED_SAMIRA_PROBE = r"""
from importlib import resources
import json
import os
from pathlib import Path

from tldw_chatbook.Character_Chat.visual_identity import (
    SAMIRA_REACTION_LABELS,
    ensure_builtin_samira,
    parse_visual_identity_manifest_json,
    validate_visual_identity_assets,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository

expected_target = Path(os.environ["EXPECTED_TARGET"]).resolve(strict=True)
package_root = Path(str(resources.files("tldw_chatbook"))).resolve(strict=True)
assert package_root.is_relative_to(expected_target), (package_root, expected_target)
samira_root = package_root / "assets" / "characters" / "samira"
assert {path.name for path in samira_root.iterdir()} == {
    "ASSET_LICENSE.md",
    "Samira.character.json",
    "Sammy.png",
    "visual_identity_pack.json",
    "expressions",
}
assert {path.name for path in (samira_root / "expressions").iterdir()} == {
    f"{label}.webp" for label in SAMIRA_REACTION_LABELS
}

directory_bytes = sum(path.stat().st_size for path in samira_root.rglob("*") if path.is_file())
manifest = parse_visual_identity_manifest_json(
    (samira_root / "visual_identity_pack.json").read_bytes(),
    require_samira_bundle=True,
    directory_bytes=directory_bytes,
)
loaded = validate_visual_identity_assets(
    manifest,
    source_kind="builtin",
    directory_bytes=directory_bytes,
)
assert len(loaded) == 31
assert all(len(asset.data) <= 1024 * 1024 for asset in loaded)
assert sum(len(asset.data) for asset in loaded) <= 16 * 1024 * 1024
assert directory_bytes <= 20 * 1024 * 1024

database_path = Path(os.environ["HOME"]) / "private-profile.sqlite"
db = CharactersRAGDB(database_path, client_id="installed-samira-probe")
try:
    assert db.get_character_card_by_id(1)["name"] == "Default Assistant"
    ensure_builtin_samira(db)
    cards = [
        dict(row)
        for row in db.execute_query(
            "SELECT * FROM character_cards WHERE deleted = 0 ORDER BY id"
        ).fetchall()
        if json.loads(row["extensions"] or "{}").get("tldw/builtin_id") == "samira"
    ]
    assert len(cards) == 1
    card = cards[0]
    graph = VisualIdentityRepository(db).get_active_actor_pack(
        "character", card["id"]
    )
    assert graph is not None
    assert graph["pack"]["source_kind"] == "builtin"
    assert len(graph["assets"]) == 31
    assert db.execute_query("SELECT COUNT(*) FROM visual_identity_packs").fetchone()[0] == 1
    assert db.execute_query("SELECT COUNT(*) FROM visual_identity_pack_versions").fetchone()[0] == 1
    assert db.execute_query("SELECT COUNT(*) FROM visual_identity_bindings").fetchone()[0] == 1
    assert db.execute_query("SELECT COUNT(*) FROM visual_identity_assets").fetchone()[0] == 31

    assert db.update_character_card(
        card["id"],
        {"description": "Installed distribution first edit marker"},
        expected_version=card["version"],
    )
    edited = db.get_character_card_by_id(card["id"])
    assert edited["description"] == "Installed distribution first edit marker"
    assert [row["id"] for row in db.search_character_cards("distribution")] == [card["id"]]
finally:
    db.close_connection()

print("installed-samira-distribution-ok")
"""


class BuiltDistributions(NamedTuple):
    source_root: Path
    dist_dir: Path
    sdist: Path
    wheel: Path


class SdistWheel(NamedTuple):
    source_root: Path
    wheel: Path


class InstalledPathState(NamedTuple):
    mode: int
    size: int
    mtime_ns: int
    digest: str | None


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


@pytest.fixture(scope="module")
def sdist_wheel(
    built_distributions: BuiltDistributions,
    tmp_path_factory: pytest.TempPathFactory,
) -> SdistWheel:
    extract_root = tmp_path_factory.mktemp("sdist-source")
    with tarfile.open(built_distributions.sdist, "r:gz") as archive:
        archive.extractall(extract_root, filter="data")
    source_roots = [path for path in extract_root.iterdir() if path.is_dir()]
    assert len(source_roots) == 1
    source_root = source_roots[0]
    dist_dir = extract_root / "dist"
    build_env = extract_root / "build-env"
    venv.EnvBuilder(symlinks=True).create(build_env)
    build_python = build_env / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    tool_layer = build_env / (
        "Lib/site-packages"
        if os.name == "nt"
        else f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    )
    tool_layer.mkdir(parents=True, exist_ok=True)
    _copy_build_tool_layer(tool_layer)
    build_state = extract_root / "build-state"
    build_env_vars = _sanitized_build_env(build_state)
    probe_root = extract_root / "build-probe"
    probe_root.mkdir()
    probe = subprocess.run(
        [str(build_python), "-I", "-c", BUILD_TOOL_PROBE],
        cwd=probe_root,
        env=build_env_vars,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert "curated-build-tools-ok" in probe.stdout
    command = [
        str(build_python),
        "-I",
        "-m",
        "build",
        "--wheel",
        "--no-isolation",
        "--outdir",
        str(dist_dir),
    ]
    completed = subprocess.run(
        command,
        cwd=source_root,
        env=build_env_vars,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, (
        f"command: {command}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    wheels = sorted(dist_dir.glob("*.whl"))
    assert len(wheels) == 1
    return SdistWheel(source_root, wheels[0])


def _copy_build_tool_layer(destination: Path) -> None:
    copied: set[Path] = set()
    for distribution_name in _BUILD_TOOL_DISTRIBUTIONS:
        distribution = metadata.distribution(distribution_name)
        files = distribution.files
        assert files is not None, distribution_name
        for relative in files:
            relative_path = Path(str(relative))
            if relative_path.is_absolute() or ".." in relative_path.parts:
                continue
            source = Path(distribution.locate_file(relative))
            if not source.is_file():
                continue
            target = destination / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.add(relative_path)
    assert copied
    assert {
        distribution.metadata["Name"].lower().replace("-", "_")
        for distribution in metadata.distributions(path=[str(destination)])
    } == set(_BUILD_TOOL_DISTRIBUTIONS)
    assert {
        path.relative_to(destination)
        for path in destination.rglob("*")
        if path.is_file()
    } == copied


def _sanitized_build_env(state_root: Path) -> dict[str, str]:
    state_root.mkdir(mode=0o700)
    temp_root = state_root / "tmp"
    temp_root.mkdir(mode=0o700)
    env = {
        name: value
        for name in _PRIVATE_CHILD_BASELINE_ENV_KEYS
        if (value := os.environ.get(name)) and not _is_sensitive_environment_name(name)
    }
    env.update(
        {
            "HOME": str(state_root),
            "USERPROFILE": str(state_root),
            "TMPDIR": str(temp_root),
            "TEMP": str(temp_root),
            "TMP": str(temp_root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
        }
    )
    assert set(env) <= _BUILD_ENV_KEYS
    assert "PYTHONPATH" not in env
    assert not any(_is_sensitive_environment_name(name) for name in env)
    return env


def _sdist_members(path: Path) -> set[str]:
    with tarfile.open(path, "r:gz") as archive:
        files = [member.name for member in archive.getmembers() if member.isfile()]
    roots = {name.split("/", 1)[0] for name in files}
    assert len(roots) == 1
    return {name.split("/", 1)[1] for name in files if "/" in name}


def _wheel_members(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        return {name for name in archive.namelist() if not name.endswith("/")}


def _sdist_member_text(path: Path, member: str) -> str:
    with tarfile.open(path, "r:gz") as archive:
        item = next(
            entry
            for entry in archive.getmembers()
            if entry.isfile() and entry.name.split("/", 1)[-1] == member
        )
        stream = archive.extractfile(item)
        assert stream is not None, member
        return stream.read().decode("utf-8")


def _wheel_member_text(path: Path, member: str) -> str:
    with zipfile.ZipFile(path) as archive:
        return archive.read(member).decode("utf-8")


def _link_or_copy(source: Path, destination: Path) -> None:
    """Hard link an unmodified archive, falling back to a copy across devices."""
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _dist_dir_without(
    built: BuiltDistributions,
    tmp_path: Path,
    *,
    drop_from_wheel: Iterable[str] = (),
    drop_from_sdist: Iterable[str] = (),
) -> Path:
    """Copy the built dist directory, omitting the named archive members.

    Only the archive that is actually mutated is rewritten; the other is hard
    linked, so a parametrized mutation sweep does not re-copy tens of
    megabytes per case.

    Args:
        built: The module-scoped build under test.
        tmp_path: Per-test temporary directory.
        drop_from_wheel: Wheel member names to omit.
        drop_from_sdist: Sdist member names (archive-relative, without the
            top-level directory) to omit.

    Returns:
        Path to the new distribution directory.
    """
    dropped_wheel = set(drop_from_wheel)
    dropped_sdist = set(drop_from_sdist)
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()

    wheel = dist_dir / built.wheel.name
    if dropped_wheel:
        with (
            zipfile.ZipFile(built.wheel) as source,
            zipfile.ZipFile(wheel, "w") as destination,
        ):
            present = set(source.namelist())
            assert dropped_wheel <= present, sorted(dropped_wheel - present)
            for member in source.infolist():
                if member.filename not in dropped_wheel:
                    destination.writestr(member, source.read(member.filename))
    else:
        _link_or_copy(built.wheel, wheel)

    sdist = dist_dir / built.sdist.name
    if dropped_sdist:
        seen: set[str] = set()
        with (
            tarfile.open(built.sdist, "r:gz") as source,
            tarfile.open(sdist, "w:gz") as destination,
        ):
            for member in source.getmembers():
                relative = member.name.split("/", 1)[-1]
                if member.isfile() and relative in dropped_sdist:
                    seen.add(relative)
                    continue
                stream = source.extractfile(member) if member.isfile() else None
                destination.addfile(member, stream)
        assert seen == dropped_sdist, sorted(dropped_sdist - seen)
    else:
        _link_or_copy(built.sdist, sdist)

    return dist_dir


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
    _install_wheel_path(built.wheel, target)


def _install_wheel_path(wheel: Path, target: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--target",
        str(target),
        str(wheel),
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


def _target_snapshot(target: Path) -> dict[str, InstalledPathState]:
    snapshot = {}
    for path in (target, *sorted(target.rglob("*"))):
        path_stat = path.lstat()
        snapshot[path.relative_to(target).as_posix()] = InstalledPathState(
            stat.S_IMODE(path_stat.st_mode),
            path_stat.st_size,
            path_stat.st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
        )
    return snapshot


@contextmanager
def _read_only_installed_tree(
    target: Path,
) -> Iterator[dict[str, InstalledPathState]]:
    paths = (target, *sorted(target.rglob("*")))
    assert not any(path.is_symlink() for path in paths)
    original_modes = {path: stat.S_IMODE(path.lstat().st_mode) for path in paths}
    for path in paths:
        if path.is_file():
            path.chmod(original_modes[path] & ~0o222)
    for path in reversed(paths):
        if path.is_dir():
            path.chmod(original_modes[path] & ~0o222)
    before = _target_snapshot(target)
    try:
        yield before
        assert _target_snapshot(target) == before, (
            "installed package tree content or metadata changed"
        )
    finally:
        current_paths = (target, *sorted(target.rglob("*")))
        for path in current_paths:
            if path.is_dir():
                path.chmod(original_modes.get(path, 0o755))
        for path in current_paths:
            if path.is_file():
                path.chmod(original_modes.get(path, 0o644))
        assert {
            path: stat.S_IMODE(path.lstat().st_mode) for path in original_modes
        } == original_modes


def _is_sensitive_environment_name(name: str) -> bool:
    """Return whether an environment name can carry credentials or proxy data.

    Args:
        name: Environment-variable name to classify.

    Returns:
        True when the name belongs to a credential or proxy category.
    """
    normalized = name.upper()
    return "PROXY" in normalized or any(
        marker in normalized
        for marker in (
            "API_KEY",
            "APIKEY",
            "TOKEN",
            "SECRET",
            "PASSWORD",
            "CREDENTIAL",
        )
    )


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
        "[splash_screen]\nenabled = false\n\n"
        "[model_catalog]\nauto_refresh_enabled = false\n",
        encoding="utf-8",
    )
    config_path.chmod(0o600)

    env = {
        name: value
        for name in _PRIVATE_CHILD_BASELINE_ENV_KEYS
        if (value := os.environ.get(name)) and not _is_sensitive_environment_name(name)
    }
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
            "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring",
            "PYTHONPATH": str(target),
            "EXPECTED_TARGET": str(target),
            "CHECKOUT_ROOT": str(checkout_root),
            "BUILD_SOURCE_ROOT": str(build_source_root),
            "EXPECTED_REACTIVES": json.dumps(sorted(RETAINED_TLDW_REACTIVES)),
            "RETIRED_REACTIVES": json.dumps(sorted(RETIRED_TLDW_REACTIVES)),
        }
    )
    return env


def test_private_child_env_excludes_host_credentials_and_proxy_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Retain safe process baselines while excluding host secrets and proxies.

    Args:
        monkeypatch: Pytest environment-isolation fixture.
        tmp_path: Private filesystem root for the child-process fixture.
    """
    state_root = tmp_path / "state"
    target = tmp_path / "target"
    build_source_root = tmp_path / "build-source"
    for path in (state_root, target, build_source_root):
        path.mkdir()
    credential_name = "TASK1601_TEST_API_KEY"
    proxy_name = "HTTPS_PROXY"
    monkeypatch.setenv(credential_name, "test-only-value")
    monkeypatch.setenv(proxy_name, "http://127.0.0.1:9")
    safe_baseline = {
        "PATH": "/task1601/bin",
        "LANG": "en_US.UTF-8",
        "LC_ALL": "C.UTF-8",
        "SYSTEMROOT": "/task1601/windows",
        "WINDIR": "/task1601/windows",
        "COMSPEC": "/task1601/windows/cmd.exe",
        "PATHEXT": ".COM;.EXE;.BAT;.CMD",
    }
    for name, value in safe_baseline.items():
        monkeypatch.setenv(name, value)

    env = _private_child_env(state_root, target, build_source_root)

    assert credential_name not in env
    assert proxy_name not in env
    assert {name: env.get(name) for name in safe_baseline} == safe_baseline
    assert env["PYTHON_KEYRING_BACKEND"] == "keyring.backends.null.Keyring"


def test_sdist_build_env_excludes_host_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TASK16319_BUILD_API_KEY", "test-only-value")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("PYTHONPATH", "/host/purelib")

    env = _sanitized_build_env(tmp_path / "build-state")

    assert set(env) <= _BUILD_ENV_KEYS
    assert "TASK16319_BUILD_API_KEY" not in env
    assert "HTTPS_PROXY" not in env
    assert "PYTHONPATH" not in env
    assert env["PIP_NO_INDEX"] == "1"


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

    required_sdist = (
        {
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
            AUDIO_CPP_ARTIFACT_MANIFEST_PATH,
        }
        | SAMIRA_RESOURCE_PATHS
    )
    required_wheel = (
        {
            "tldw_chatbook/css/tldw_cli_modular.tcss",
            "tldw_chatbook/Config_Files/rag_pipelines.toml",
            "tldw_chatbook/Evals/config/eval_config.yaml",
            "tldw_chatbook/Third_Party/aider/LICENSE.txt",
            "tldw_chatbook/Third_Party/textual_fspicker/LICENSE",
            AUDIO_CPP_ARTIFACT_MANIFEST_PATH,
        }
        | SAMIRA_RESOURCE_PATHS
    )
    assert not required_sdist - sdist_members
    assert not required_wheel - wheel_members
    for members in (sdist_members, wheel_members):
        assert {
            name for name in members if name.startswith(f"{SAMIRA_RESOURCE_ROOT}/")
        } == SAMIRA_RESOURCE_PATHS

    retired_modules = {
        "tldw_chatbook/Audio/transcription_history.py",
        "tldw_chatbook/Widgets/transcription_history_viewer.py",
        "tldw_chatbook/UI/Dictation_Window.py",
    }
    assert retired_modules.isdisjoint(sdist_members)
    assert retired_modules.isdisjoint(wheel_members)

    # The file template store is deleted (spec §8.1.2): neither artifact may
    # carry any tldw_chatbook/Chunking/templates/ path -- the JSONs, the
    # README, and example_usage.py all die with the store.
    shipped_template_store = {
        name
        for name in sdist_members | wheel_members
        if name.startswith(CHUNKING_TEMPLATES_PREFIX)
    }
    assert shipped_template_store == set(), (
        "the deleted file template store must not ship: "
        f"{sorted(shipped_template_store)}"
    )

    forbidden_wheel = {
        "tldw_chatbook/css/components/stats_screen.css",
        "tldw_chatbook/Config_Files/embedding_configs_examples.toml",
        "tldw_chatbook/Config_Files/pipeline_configs/custom_pipelines_example.toml",
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


@pytest.mark.parametrize("wheel_source", ["source", "sdist"])
def test_installed_distribution_migrates_v35_database_to_current(
    built_distributions: BuiltDistributions,
    sdist_wheel: SdistWheel,
    tmp_path: Path,
    wheel_source: str,
) -> None:
    """Install the wheel into an empty tree and drive the schema for real.

    Two databases, both created inside the installed distribution: one from
    scratch -- the fresh-install path, which walks the entire v4->current
    chain -- and one pinned back to v35 to prove the upgrade path. Each reads
    its reached version out of ``db_schema_version`` rather than asserting the
    constant back at itself (task-19044, task-19860).
    """
    wheel, build_source_root = (
        (built_distributions.wheel, built_distributions.source_root)
        if wheel_source == "source"
        else (sdist_wheel.wheel, sdist_wheel.source_root)
    )
    target = tmp_path / "target"
    state_root = tmp_path / "state"
    run_root = tmp_path / "run"
    state_root.mkdir(mode=0o700)
    run_root.mkdir()
    _install_wheel_path(wheel, target)
    env = _private_child_env(
        state_root,
        target,
        build_source_root,
    )

    with _read_only_installed_tree(target):
        result = _run_child(
            [sys.executable, "-c", INSTALLED_MIGRATION_PROBE],
            run_root,
            env,
        )

    assert "installed-wheel-fresh-init-ok" in result.stdout
    assert "installed-wheel-v35-to-current-ok" in result.stdout


def test_installed_wheel_loads_pinned_audio_cpp_artifact_manifest(
    built_distributions: BuiltDistributions,
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    state_root = tmp_path / "state"
    run_root = tmp_path / "run"
    state_root.mkdir(mode=0o700)
    run_root.mkdir()
    _install_wheel(built_distributions, target)
    env = _private_child_env(state_root, target, built_distributions.source_root)
    probe = f"""
from pathlib import Path
import os
import urllib.request
from tldw_chatbook.TTS.audio_cpp_artifact_catalog import load_audio_cpp_artifact_source_manifest
import tldw_chatbook.TTS.audio_cpp_artifact_catalog as catalog_module

expected_target = Path(os.environ["EXPECTED_TARGET"]).resolve(strict=True)
assert Path(catalog_module.__file__).resolve(strict=True).is_relative_to(expected_target)
def fail_network(*_args, **_kwargs):
    raise AssertionError("installed manifest loader touched the network")
urllib.request.urlopen = fail_network
manifest = load_audio_cpp_artifact_source_manifest()
assert manifest.repository == {AUDIO_CPP_ARTIFACT_REPOSITORY!r}
assert manifest.commit == {AUDIO_CPP_ARTIFACT_COMMIT!r}
assert len(manifest.packages) == 45
print("installed-audio-cpp-manifest-ok")
"""

    with _read_only_installed_tree(target):
        result = _run_child([sys.executable, "-c", probe], run_root, env)

    assert "installed-audio-cpp-manifest-ok" in result.stdout


def test_installed_migration_probe_validates_environment_derived_path() -> None:
    assert (
        "from tldw_chatbook.Utils.path_validation import validate_path"
        in INSTALLED_MIGRATION_PROBE
    )
    assert "migration_path = validate_path(" in INSTALLED_MIGRATION_PROBE


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


@pytest.mark.parametrize("archive_kind", ["wheel", "sdist"])
def test_release_checker_rejects_missing_samira_reaction(
    built_distributions: BuiltDistributions,
    tmp_path: Path,
    archive_kind: str,
) -> None:
    dist_dir = tmp_path / "dist"
    shutil.copytree(built_distributions.dist_dir, dist_dir)
    missing = f"{SAMIRA_RESOURCE_ROOT}/expressions/anger.webp"
    if archive_kind == "wheel":
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
    else:
        sdist = next(dist_dir.glob("*.tar.gz"))
        rewritten = sdist.with_name(f"{sdist.name}.rewritten")
        with (
            tarfile.open(sdist, "r:gz") as source,
            tarfile.open(rewritten, "w:gz") as destination,
        ):
            for member in source.getmembers():
                if member.name.endswith(f"/{missing}"):
                    continue
                stream = source.extractfile(member) if member.isfile() else None
                destination.addfile(member, stream)
        rewritten.replace(sdist)

    result = _run_manifest_checker(built_distributions, dist_dir, tmp_path)

    assert result.returncode == 1
    assert missing in result.stdout + result.stderr


def test_migration_expectations_are_derived_not_enumerated() -> None:
    """The expectations must come from reality, or they cannot catch drift."""
    assert SOURCE_MIGRATION_PATHS, "no migration scripts found in the checkout"
    assert RUNTIME_MIGRATION_PATHS, (
        "no migration reads detected in ChaChaNotes_DB.py; "
        "RUNTIME_MIGRATION_READ no longer matches the schema runner"
    )
    orphans = sorted(RUNTIME_MIGRATION_PATHS - SOURCE_MIGRATION_PATHS)
    assert not orphans, f"schema runner opens missing scripts: {orphans}"


@pytest.mark.parametrize("archive_kind", ["wheel", "sdist"])
def test_built_artifact_contains_every_migration_script(
    built_distributions: BuiltDistributions,
    archive_kind: str,
) -> None:
    """Every ``.sql`` in the tree must be inside the artifact -- all reported.

    The assertion reads the ARCHIVE's members, never the text of
    ``pyproject.toml`` or ``MANIFEST.in``: swapping the build backend cannot
    make it pass vacuously. Every missing file is named at once; the enumerated
    lists this replaced were 19 files behind and the runtime symptom showed
    only the first (task-19860).
    """
    members = (
        _wheel_members(built_distributions.wheel)
        if archive_kind == "wheel"
        else _sdist_members(built_distributions.sdist)
    )

    missing = sorted(SOURCE_MIGRATION_PATHS - members)

    assert not missing, (
        f"{len(missing)} migration script(s) present in the source tree are "
        f"absent from the {archive_kind}:\n  " + "\n  ".join(missing)
    )


@pytest.mark.parametrize("archive_kind", ["wheel", "sdist"])
def test_built_artifact_ships_the_migrations_its_own_code_opens(
    built_distributions: BuiltDistributions,
    archive_kind: str,
) -> None:
    """Self-consistency: the packaged schema runner's reads must resolve.

    Derived from the artifact's own ``ChaChaNotes_DB.py``, so this holds for
    any artifact from anywhere -- no checkout required. This is the property a
    user's install actually depends on.
    """
    archive, members = (
        (built_distributions.wheel, _wheel_members(built_distributions.wheel))
        if archive_kind == "wheel"
        else (built_distributions.sdist, _sdist_members(built_distributions.sdist))
    )
    read_text = _wheel_member_text if archive_kind == "wheel" else _sdist_member_text
    required = _runtime_migration_paths(read_text(archive, CHACHANOTES_DB_MODULE_PATH))

    assert required, "packaged schema runner exposed no migration reads"
    missing = sorted(required - members)

    assert not missing, (
        f"the {archive_kind}'s own ChaChaNotes_DB.py opens "
        f"{len(missing)} script(s) the {archive_kind} does not carry:\n  "
        + "\n  ".join(missing)
    )


@pytest.mark.parametrize("archive_kind", ["wheel", "sdist"])
def test_release_checker_reports_every_missing_database_migration(
    built_distributions: BuiltDistributions,
    tmp_path: Path,
    archive_kind: str,
) -> None:
    """Removing several migrations must name all of them, not just the first.

    Aborting at the first gap is exactly how 19 missing files stayed invisible
    behind one reported symptom (task-19860).
    """
    dropped = set(sorted(SOURCE_MIGRATION_PATHS)[:3]) | {
        max(SOURCE_MIGRATION_PATHS)
    }
    dist_dir = _dist_dir_without(
        built_distributions,
        tmp_path,
        drop_from_wheel=dropped if archive_kind == "wheel" else (),
        drop_from_sdist=dropped if archive_kind == "sdist" else (),
    )

    result = _run_manifest_checker(built_distributions, dist_dir, tmp_path)

    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    unreported = sorted(name for name in dropped if name not in output)
    assert not unreported, f"checker stayed silent about {unreported}\n{output}"


@pytest.mark.parametrize("archive_kind", ["wheel", "sdist"])
@pytest.mark.parametrize("missing", sorted(RUNTIME_MIGRATION_PATHS))
def test_release_checker_rejects_missing_database_migration(
    built_distributions: BuiltDistributions,
    tmp_path: Path,
    archive_kind: str,
    missing: str,
) -> None:
    dist_dir = _dist_dir_without(
        built_distributions,
        tmp_path,
        drop_from_wheel=[missing] if archive_kind == "wheel" else (),
        drop_from_sdist=[missing] if archive_kind == "sdist" else (),
    )

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
    with _read_only_installed_tree(target):
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


@pytest.mark.skipif(
    os.name != "posix",
    reason="POSIX mode bits are required to enforce read-only installed trees",
)
def test_read_only_installed_tree_rejects_rewrite_and_catches_touch(
    built_distributions: BuiltDistributions,
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    _install_wheel(built_distributions, target)
    package_file = target / "tldw_chatbook" / "__init__.py"
    original_modes = {
        path: stat.S_IMODE(path.lstat().st_mode)
        for path in (target, *sorted(target.rglob("*")))
    }

    with _read_only_installed_tree(target):
        assert {
            path: stat.S_IMODE(path.lstat().st_mode) for path in original_modes
        } == {path: mode & ~0o222 for path, mode in original_modes.items()}
        script_path = os.pathsep.join(
            str(path) for path in (target / "bin", target / "Scripts")
        )
        assert all(
            shutil.which(name, path=script_path) is not None
            for name in ("tldw-cli", "tldw-serve")
        )
        original = package_file.read_bytes()
        with pytest.raises(PermissionError):
            package_file.write_bytes(original)
    assert {
        path: stat.S_IMODE(path.lstat().st_mode) for path in original_modes
    } == original_modes

    with pytest.raises(
        AssertionError,
        match="installed package tree content or metadata changed",
    ):
        with _read_only_installed_tree(target):
            package_file.touch()


@pytest.mark.parametrize("wheel_source", ["source", "sdist"])
def test_installed_distribution_validates_and_seeds_samira_without_package_writes(
    built_distributions: BuiltDistributions,
    sdist_wheel: SdistWheel,
    tmp_path: Path,
    wheel_source: str,
) -> None:
    wheel, build_source_root = (
        (built_distributions.wheel, built_distributions.source_root)
        if wheel_source == "source"
        else (sdist_wheel.wheel, sdist_wheel.source_root)
    )
    target = tmp_path / "target"
    state_root = tmp_path / "state"
    run_root = tmp_path / "run"
    state_root.mkdir(mode=0o700)
    run_root.mkdir()
    _install_wheel_path(wheel, target)
    env = _private_child_env(state_root, target, build_source_root)
    with _read_only_installed_tree(target):
        result = _run_child(
            [sys.executable, "-c", INSTALLED_SAMIRA_PROBE],
            run_root,
            env,
        )

    assert "installed-samira-distribution-ok" in result.stdout
