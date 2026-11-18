"""Shared RAG path policy must stay live without loading optional services."""

import importlib
import inspect
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]

_COLD_PROTECTED_GETTER = r"""
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys

repo, private = (Path(value).resolve() for value in sys.argv[1:])
core = repo / "packages/tldw_profile_core/src"
sys.path.insert(0, str(repo))
sys.path.insert(1, str(core))
for name, expected in {
    "Tests": repo / "Tests/__init__.py",
    "tldw_chatbook": repo / "tldw_chatbook/__init__.py",
    "tldw_profile_core": core / "tldw_profile_core/__init__.py",
}.items():
    assert Path(importlib.util.find_spec(name).origin).resolve() == expected.resolve()
from Tests import real_profile_guard
real_profile_guard.install()
original_home = real_profile_guard._real_home()
assert original_home is not None and original_home.resolve() != private / "home"
keep = {"SYSTEMROOT", "WINDIR", "SYSTEMDRIVE", "COMSPEC", "PATH", "PATHEXT",
        "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMW6432", "PROGRAMDATA",
        "PROCESSOR_ARCHITECTURE", "TLDW_TEST_REAL_HOME"}
for name in list(os.environ):
    if name.upper() not in keep:
        os.environ.pop(name)
for name, relative in {
    "HOME": "home", "USERPROFILE": "home", "XDG_CONFIG_HOME": "config",
    "XDG_DATA_HOME": "data", "TEMP": "temp", "TMP": "temp",
    "APPDATA": "appdata", "LOCALAPPDATA": "localappdata",
}.items():
    location = private / relative
    location.mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(location)
os.environ.update(TLDW_CONFIG_PATH=str(private / "config/config.toml"),
                  TLDW_TEST_CONFIG_ROOT=str(private), PYTHONUTF8="1",
                  PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
                  HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
network = []
def audit(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.bind", "socket.sendto"}:
        network.append(event)
        raise RuntimeError("test_network_refused")
sys.addaudithook(audit)
from tldw_chatbook.Backup_Recovery import bootstrap
assert bootstrap.startup_permission(bootstrap.effective_config_path(), bootstrap.default_bootstrap_root())[0]
from tldw_chatbook import config
from tldw_chatbook.Utils import sensitive_paths
blocked = ("numpy", "torch", "transformers", "sentence_transformers", "chromadb",
           "tldw_chatbook.RAG_Search.simplified", "tldw_chatbook.RAG_Search.config_profiles",
           "tldw_chatbook.RAG_Search.reranker", "tldw_chatbook.RAG_Search.parallel_processor",
           "tldw_chatbook.RAG_Search.chunking_service")
assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules for prefix in blocked)
class RefuseOptionalRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == prefix or fullname.startswith(prefix + ".") for prefix in blocked):
            raise RuntimeError("optional_runtime_blocked:" + fullname)
sys.meta_path.insert(0, RefuseOptionalRuntime())
vector = private / "vectors"
os.environ["RAG_PERSIST_DIR"] = str(vector)
print("REAL_PROTECTED_GETTER_ENTER", flush=True)
containers = sensitive_paths._direct_child_rule_container_dirs()
assert containers == (config.get_user_data_dir(), config._get_effective_config_path().parent,
                      vector, config.get_user_data_dir() / "rag_profiles")
assert sensitive_paths.find_root_binding_conflict(vector) == vector
moved = private / "next-vectors"
os.environ["RAG_PERSIST_DIR"] = str(moved)
assert moved in sensitive_paths.resolve_sensitive_context().direct_child_denied_dirs
assert sensitive_paths.find_root_binding_conflict(moved) == moved
assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules for prefix in blocked)
assert not network
assert not real_profile_guard.take_violations()
print("PROTECTED_RAG_PATHS_WITHOUT_RUNTIME_OK")
"""


@pytest.mark.parametrize("block_rag_package", [False, True])
def test_real_protected_getter_resolves_paths_with_optional_runtime_blocked(
    tmp_path, block_rag_package
):
    code = _COLD_PROTECTED_GETTER
    if block_rag_package:
        code = code.replace(
            'blocked = ("numpy",',
            'blocked = ("tldw_chatbook.RAG_Search", "numpy",',
            1,
        )
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            "-B",
            "-c",
            code,
            str(ROOT),
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    assert "REAL_PROTECTED_GETTER_ENTER" in result.stdout, result.stdout + result.stderr
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PROTECTED_RAG_PATHS_WITHOUT_RUNTIME_OK" in result.stdout


def _chroma_api():
    module = importlib.import_module("tldw_chatbook.RAG_Search.state_paths")
    return module.default_chroma_persist_directory, importlib.import_module(
        "tldw_chatbook.config"
    )


@pytest.mark.parametrize(
    ("environment", "rag", "expected"),
    [
        (None, {}, None),
        (
            " env-vectors ",
            {"vector_store": {"persist_directory": "current"}},
            "env-vectors",
        ),
        (
            "   ",
            {
                "vector_store": {"persist_directory": " current "},
                "chroma": {"persist_directory": "legacy"},
            },
            "current",
        ),
        (
            None,
            {
                "vector_store": {"persist_directory": " "},
                "chroma": {"persist_directory": "legacy"},
            },
            "legacy",
        ),
        (
            None,
            {"vector_store": [], "chroma": {"persist_directory": "legacy"}},
            "legacy",
        ),
        (None, {"vector_store": {"persist_directory": 0}}, "0"),
        (None, {"vector_store": {"persist_directory": False}}, "False"),
        (
            None,
            {"vector_store": {"persist_directory": Path("relative-vectors")}},
            "relative-vectors",
        ),
        (None, [], None),
        (" ~/vectors ", {}, "~/vectors"),
    ],
)
def test_chroma_path_precedence_coercion_and_relative_paths(
    environment, rag, expected, monkeypatch, tmp_path
):
    getter, runtime = _chroma_api()
    data = tmp_path / "data"
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(
        runtime, "get_cli_setting", lambda section, key, default=None: rag
    )
    monkeypatch.setattr(runtime, "get_user_data_dir", lambda: data)
    if environment is None:
        monkeypatch.delenv("RAG_PERSIST_DIR", raising=False)
    else:
        monkeypatch.setenv("RAG_PERSIST_DIR", environment)
    wanted = data / "chromadb" if expected is None else Path(expected).expanduser()
    assert getter() == wanted
    assert not inspect.signature(getter).parameters


def test_setting_read_failure_is_local_to_one_precedence_step(monkeypatch, tmp_path):
    getter, runtime = _chroma_api()
    calls = []

    def read(section, key, default=None):
        calls.append((section, key))
        if len(calls) == 1:
            raise RuntimeError("private config read refused")
        return {"chroma": {"persist_directory": "legacy-after-failure"}}

    monkeypatch.delenv("RAG_PERSIST_DIR", raising=False)
    monkeypatch.setattr(runtime, "get_cli_setting", read)
    monkeypatch.setattr(runtime, "get_user_data_dir", lambda: tmp_path)
    assert getter() == Path("legacy-after-failure")
    assert calls == [("AppRAGSearchConfig", "rag"), ("AppRAGSearchConfig", "rag")]


def test_env_selection_does_not_read_config_or_userdata(monkeypatch):
    getter, runtime = _chroma_api()

    def unexpected(*args, **kwargs):
        pytest.fail("a selected environment path must short-circuit later accessors")

    monkeypatch.setenv("RAG_PERSIST_DIR", "relative-env")
    monkeypatch.setattr(runtime, "get_cli_setting", unexpected)
    monkeypatch.setattr(runtime, "get_user_data_dir", unexpected)
    assert getter() == Path("relative-env")


def test_current_config_and_userdata_are_read_again(monkeypatch, tmp_path):
    getter, runtime = _chroma_api()
    state = {"rag": {}, "data": tmp_path / "first"}
    monkeypatch.delenv("RAG_PERSIST_DIR", raising=False)
    monkeypatch.setattr(runtime, "get_cli_setting", lambda *args: state["rag"])
    monkeypatch.setattr(runtime, "get_user_data_dir", lambda: state["data"])
    assert getter() == state["data"] / "chromadb"
    state["data"] = tmp_path / "second"
    assert getter() == state["data"] / "chromadb"
    state["rag"] = {"vector_store": {"persist_directory": "changed"}}
    assert getter() == Path("changed")


def test_profile_directory_keeps_live_userdata_getter_seam(monkeypatch, tmp_path):
    module = importlib.import_module("tldw_chatbook.RAG_Search.state_paths")
    runtime = importlib.import_module("tldw_chatbook.config")
    state = {"data": tmp_path / "first"}
    monkeypatch.setattr(runtime, "get_user_data_dir", lambda: state["data"])
    getter = module.default_rag_profiles_dir
    assert getter() == state["data"] / "rag_profiles"
    state["data"] = tmp_path / "second"
    assert getter() == state["data"] / "rag_profiles"
    failure = RuntimeError("private userdata read refused")

    def refuse():
        raise failure

    monkeypatch.setattr(runtime, "get_user_data_dir", refuse)
    with pytest.raises(RuntimeError) as raised:
        getter()
    assert raised.value is failure
    assert not inspect.signature(getter).parameters


def test_config_failure_falls_back_without_logging_exception_values(
    monkeypatch, tmp_path
):
    """Path discovery must not put a config reader's private values in a sink."""
    from loguru import logger

    getter, runtime = _chroma_api()
    canary = "PRIVATE_CONFIG_VALUE_CANARY_355"

    def refuse(*args):
        raise ValueError(canary)

    monkeypatch.delenv("RAG_PERSIST_DIR", raising=False)
    monkeypatch.setattr(runtime, "get_cli_setting", refuse)
    monkeypatch.setattr(runtime, "get_user_data_dir", lambda: tmp_path)
    messages = []
    sink = logger.add(messages.append, level="DEBUG")
    try:
        assert getter() == tmp_path / "chromadb"
    finally:
        logger.remove(sink)
    assert len(messages) == 2
    assert canary not in "\n".join(str(message) for message in messages)
    assert canary not in repr([message.record for message in messages])
    assert all(
        message.record["message"] == "Could not read RAG state path setting"
        for message in messages
    )
    assert all(
        message.record["extra"]["exception_type"] == "ValueError"
        for message in messages
    )
