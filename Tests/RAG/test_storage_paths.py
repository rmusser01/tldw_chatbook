"""Canonical RAG directory policy and legacy live collaborators."""

import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture(params=("storage_paths", "simplified.config"))
def paths(request, monkeypatch, tmp_path):
    module = importlib.import_module(f"tldw_chatbook.RAG_Search.{request.param}")
    monkeypatch.delenv("RAG_PERSIST_DIR", raising=False)
    monkeypatch.setattr(module, "get_user_data_dir", lambda: tmp_path)
    monkeypatch.setattr(module, "get_cli_setting", lambda *args: {})
    return module


@pytest.mark.parametrize(
    "env,vector,legacy,expected,reads",
    [
        (" /env ", "/vector", "/legacy", "/env", []),
        (" ", " /vector ", "/legacy", "/vector", ["vector_store"]),
        (None, " ", " /legacy ", "/legacy", ["vector_store", "chroma"]),
        (None, None, None, None, ["vector_store", "chroma"]),
        (" ~/rag ", None, None, "~/rag", []),
    ],
)
def test_directory_precedence_and_short_circuit(
    paths, monkeypatch, tmp_path, env, vector, legacy, expected, reads
):
    calls = []

    def explicit(section, key):
        assert key == "persist_directory"
        calls.append(section)
        return {"vector_store": vector, "chroma": legacy}[section]

    monkeypatch.setattr(paths, "_explicit_rag_setting", explicit)
    if env is not None:
        monkeypatch.setenv("RAG_PERSIST_DIR", env)
    assert paths.default_chroma_persist_directory() == (
        Path(expected).expanduser() if expected else tmp_path / "chromadb"
    )
    assert calls == reads


def test_live_getter_and_environment_changes_are_not_cached(paths, monkeypatch):
    settings = {"vector_store": {"persist_directory": "/first"}}
    calls = []

    def getter(*args):
        assert args == ("AppRAGSearchConfig", "rag", {})
        calls.append(args)
        return settings

    monkeypatch.setattr(paths, "get_cli_setting", getter)
    assert paths.default_chroma_persist_directory() == Path("/first")
    settings["vector_store"]["persist_directory"] = "/second"
    assert paths.default_chroma_persist_directory() == Path("/second")
    monkeypatch.setenv("RAG_PERSIST_DIR", "/third")
    assert paths.default_chroma_persist_directory() == Path("/third")
    assert len(calls) == 2


@pytest.mark.parametrize("settings", (None, [], {"vector_store": []}, {}))
def test_malformed_or_missing_explicit_settings_fall_back(
    paths, monkeypatch, tmp_path, settings
):
    monkeypatch.setattr(paths, "get_cli_setting", lambda *args: settings)
    assert paths.default_chroma_persist_directory() == tmp_path / "chromadb"


def test_getter_failure_is_caught_but_collaborator_failures_propagate(
    paths, monkeypatch, tmp_path
):
    def broken(*args):
        raise RuntimeError("sentinel")

    monkeypatch.setattr(paths, "get_cli_setting", broken)
    assert paths.default_chroma_persist_directory() == tmp_path / "chromadb"
    monkeypatch.setattr(paths, "_explicit_rag_setting", broken)
    with pytest.raises(RuntimeError, match="sentinel"):
        paths.default_chroma_persist_directory()
    monkeypatch.setenv("RAG_PERSIST_DIR", "/env")
    assert paths.default_chroma_persist_directory() == Path("/env")
    monkeypatch.delenv("RAG_PERSIST_DIR")
    monkeypatch.setattr(paths, "_explicit_rag_setting", lambda *args: None)
    monkeypatch.setattr(paths, "get_user_data_dir", broken)
    with pytest.raises(RuntimeError, match="sentinel"):
        paths.default_chroma_persist_directory()


def test_cleaner_and_path_errors_propagate(paths, monkeypatch):
    class BrokenPath:
        def __str__(self):
            raise ValueError("path sentinel")

    monkeypatch.setattr(paths, "_explicit_rag_setting", lambda *args: BrokenPath())
    with pytest.raises(ValueError, match="path sentinel"):
        paths.default_chroma_persist_directory()

    def broken(value):
        raise RuntimeError("cleaner sentinel")

    monkeypatch.setattr(paths, "_cleaned_path_setting", broken)
    with pytest.raises(RuntimeError, match="cleaner sentinel"):
        paths.default_chroma_persist_directory()


@pytest.mark.parametrize("name", ("storage_paths", "config_profiles"))
def test_profile_directory_uses_live_user_data_accessor(name, monkeypatch, tmp_path):
    module = importlib.import_module(f"tldw_chatbook.RAG_Search.{name}")
    for child in ("first", "second"):
        monkeypatch.setattr(module, "get_user_data_dir", lambda: tmp_path / child)
        assert module.default_rag_profiles_dir() == tmp_path / child / "rag_profiles"
