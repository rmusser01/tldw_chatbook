"""Cold real profile metadata/capture regressions; no package stubs or App."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _cold_probe(tmp_path, mode):
    from Tests import real_profile_guard

    private = tmp_path / (mode + "-private")
    command = [
        sys.executable,
        "-I",
        str(Path(__file__).with_name("_metadata_import_probe.py")),
        "--source-root",
        str(ROOT),
        "--private-root",
        str(private),
        "--original-home",
        str(real_profile_guard._real_home()),
        "--mode",
        mode,
    ]
    completed = subprocess.run(
        command, cwd=ROOT, text=True, encoding="utf-8", capture_output=True, timeout=120
    )
    (tmp_path / (mode + ".stdout.txt")).write_text(completed.stdout, encoding="utf-8")
    (tmp_path / (mode + ".stderr.txt")).write_text(completed.stderr, encoding="utf-8")
    result = (
        json.loads((private / "probe-results.json").read_text(encoding="utf-8"))
        if (private / "probe-results.json").exists()
        else None
    )
    assert completed.returncode == 0, (
        completed.stdout[-3500:],
        completed.stderr[-3500:],
        result,
    )
    assert result is not None and result["startup_admitted"]
    assert result["import_attempts"] == []
    assert result["network_denials"] == []
    assert result["real_profile_refusals"] == []


def test_real_library_depth_read_is_cold_and_engine_free(tmp_path):
    _cold_probe(tmp_path, "library")


def test_real_received_configuration_depth_capture_is_cold_and_engine_free(tmp_path):
    _cold_probe(tmp_path, "received")


def test_socket_guard_requires_exact_selfpipe_code_globals_and_denies_dns():
    """Exercise the actual audit function with synthetic frames; no sockets."""
    import ast
    from asyncio import proactor_events
    from types import SimpleNamespace

    probe = Path(__file__).with_name("_metadata_import_probe.py")
    tree = ast.parse(probe.read_text(encoding="utf-8"))
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    audit = next(
        node
        for node in main.body
        if isinstance(node, ast.FunctionDef) and node.name == "audit"
    )
    function = proactor_events.BaseProactorEventLoop._make_self_pipe
    path = Path(function.__code__.co_filename).resolve()
    code = function.__code__
    globals_owner = function.__globals__
    cloned_code = code.replace()
    assert cloned_code is not code
    cases = [
        ("genuine", code, globals_owner, "socket.bind", "127.0.0.1", True),
        ("copied_code", cloned_code, globals_owner, "socket.bind", "127.0.0.1", False),
        ("wrong_globals", code, {}, "socket.connect", "127.0.0.1", False),
        ("external", code, globals_owner, "socket.connect", "192.0.2.1", False),
        (
            "no_selfpipe",
            compile("pass", str(path), "exec"),
            globals_owner,
            "socket.bind",
            "127.0.0.1",
            False,
        ),
    ]
    cases += [
        (event, code, globals_owner, event, "127.0.0.1", False)
        for event in (
            "socket.getaddrinfo",
            "socket.gethostbyname",
            "socket.gethostbyaddr",
            "socket.getnameinfo",
            "socket.sendto",
            "socket.sendmsg",
        )
    ]
    for label, candidate, namespace, event, host, expected in cases:
        frame = SimpleNamespace(f_code=candidate, f_globals=namespace, f_back=None)
        denied, allowed = [], []
        values = {
            "Path": Path,
            "sys": SimpleNamespace(_getframe=lambda n: frame, modules={}),
            "selfpipe_bindings": [(code, globals_owner, path)],
            "network_denials": denied,
            "network_refusal_frames": [],
            "selfpipes": allowed,
        }
        exec(
            compile(
                ast.Module(body=[audit], type_ignores=[]), "actual-cold-audit", "exec"
            ),
            values,
        )
        accepted = True
        try:
            values["audit"](event, (object(), (host, 0)))
        except AssertionError:
            accepted = False
        assert accepted is expected, label
        assert bool(allowed) is expected and bool(denied) is not expected, label


def test_local_hostname_metadata_is_not_a_dns_or_traffic_allowance():
    """The pure audit ignores a hostname read; it does not perform that read."""
    import ast

    tree = ast.parse(
        Path(__file__)
        .with_name("_metadata_import_probe.py")
        .read_text(encoding="utf-8")
    )
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    audit = next(
        node
        for node in main.body
        if isinstance(node, ast.FunctionDef) and node.name == "audit"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[audit], type_ignores=[]), "actual-cold-audit", "exec"),
        namespace,
    )
    namespace["audit"]("socket.gethostname", ())


def test_known_ipv6_import_probe_remains_refused_by_original_oserror_path():
    """Pure partial-module identity checks; no urllib3 import or bind occurs."""
    import ast
    from types import ModuleType, SimpleNamespace

    tree = ast.parse(
        Path(__file__)
        .with_name("_metadata_import_probe.py")
        .read_text(encoding="utf-8")
    )
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    audit = next(
        node
        for node in main.body
        if isinstance(node, ast.FunctionDef) and node.name == "audit"
    )
    expected = ROOT / "synthetic-urllib3-source.py"
    partial = ModuleType("controlled-partial")
    partial.__file__ = str(expected)
    exec(compile("def _has_ipv6(): pass", str(expected), "exec"), vars(partial))
    function = partial._has_ipv6
    for valid in (True, False):
        frame = SimpleNamespace(
            f_code=function.__code__ if valid else function.__code__.replace(),
            f_globals=vars(partial),
            f_back=None,
        )
        refused, denied = [], []
        namespace = {
            "Path": Path,
            "sys": SimpleNamespace(
                _getframe=lambda n: frame, modules={"urllib3.util.connection": partial}
            ),
            "selfpipe_bindings": [],
            "selfpipes": [],
            "network_denials": denied,
            "network_refusal_frames": [],
            "refused_import_probes": refused,
            "expected_ipv6_source": expected.resolve(),
        }
        exec(
            compile(
                ast.Module(body=[audit], type_ignores=[]), "actual-cold-audit", "exec"
            ),
            namespace,
        )
        caught = None
        try:
            namespace["audit"]("socket.bind", (object(), ("::1", 0)))
        except (OSError, AssertionError) as error:
            caught = error
        assert isinstance(caught, OSError if valid else AssertionError)
        assert bool(refused) is valid and bool(denied) is not valid


# Public export owners from the pre-repair facade (7cf51a), not implementation internals.
_LEGACY_EXPORTS = [
    ("Citation", "citations", "Citation"),
    ("CitationType", "citations", "CitationType"),
    ("SearchResultWithCitations", "citations", "SearchResultWithCitations"),
    ("merge_citations", "citations", "merge_citations"),
    ("group_citations_by_document", "citations", "group_citations_by_document"),
    ("filter_overlapping_citations", "citations", "filter_overlapping_citations"),
    ("EmbeddingsServiceWrapper", "embeddings_wrapper", "EmbeddingsServiceWrapper"),
    ("EmbeddingsService", "embeddings_wrapper", "EmbeddingsServiceWrapper"),
    ("create_embeddings_service", "embeddings_wrapper", "create_embeddings_service"),
    ("SearchResult", "vector_store", "SearchResult"),
    ("VectorStore", "vector_store", "VectorStore"),
    ("VectorStoreSearchError", "vector_store", "VectorStoreSearchError"),
    ("ChromaVectorStore", "vector_store", "ChromaVectorStore"),
    ("InMemoryVectorStore", "vector_store", "InMemoryVectorStore"),
    ("create_vector_store", "vector_store", "create_vector_store"),
    ("RAGService", "rag_service", "RAGService"),
    ("RAGConfig", "rag_service", "RAGConfig"),
    ("create_and_index", "rag_service", "create_and_index"),
    ("EnhancedRAGService", "enhanced_rag_service", "EnhancedRAGService"),
    (
        "create_enhanced_rag_service",
        "enhanced_rag_service",
        "create_enhanced_rag_service",
    ),
    ("EnhancedRAGServiceV2", "enhanced_rag_service_v2", "EnhancedRAGServiceV2"),
    ("IndexingResult", "data_models", "IndexingResult"),
    ("SimpleRAGCache", "simple_cache", "SimpleRAGCache"),
    ("get_rag_cache", "simple_cache", "get_rag_cache"),
    ("EmbeddingConfig", "config", "EmbeddingConfig"),
    ("VectorStoreConfig", "config", "VectorStoreConfig"),
    ("ChunkingConfig", "config", "ChunkingConfig"),
    ("SearchConfig", "config", "SearchConfig"),
    ("create_config_for_collection", "config", "create_config_for_collection"),
    ("create_config_for_testing", "config", "create_config_for_testing"),
    ("create_rag_service", "rag_factory", "create_rag_service"),
    ("create_rag_service_from_config", "rag_factory", "create_rag_service_from_config"),
    ("create_auto_rag_service", "rag_factory", "create_auto_rag_service"),
    ("get_available_profiles", "rag_factory", "get_available_profiles"),
]


@pytest.mark.parametrize("public,module,attribute", _LEGACY_EXPORTS)
def test_lazy_facade_retains_actual_public_alias_or_original_import_failure(
    public, module, attribute
):
    import importlib
    import tldw_chatbook.RAG_Search.simplified as facade

    # RAGConfig was already imported by rag_service from this exact metadata owner.
    owner_module = "config" if public == "RAGConfig" else module
    try:
        owner = importlib.import_module(
            "tldw_chatbook.RAG_Search.simplified." + owner_module
        )
    except ImportError:
        with pytest.raises(ImportError):
            getattr(facade, public)
    else:
        assert getattr(facade, public) is getattr(owner, attribute)


def test_metadata_classes_retain_legacy_type_and_pickle_aliases():
    from dataclasses import asdict
    import pickle
    from tldw_chatbook.RAG_Search.config_metadata import (
        ProcessingConfig,
        RerankingConfig,
    )
    from tldw_chatbook.RAG_Search.parallel_processor import (
        ProcessingConfig as LegacyProcessing,
    )
    from tldw_chatbook.RAG_Search.reranker import RerankingConfig as LegacyReranking

    for cls, legacy, module in (
        (RerankingConfig, LegacyReranking, "reranker"),
        (ProcessingConfig, LegacyProcessing, "parallel_processor"),
    ):
        assert cls is legacy
        value = cls()
        assert isinstance(value, legacy)
        assert asdict(pickle.loads(pickle.dumps(value))) == asdict(value)
        historical_global = (
            "ctldw_chatbook.RAG_Search." + module + "\n" + cls.__name__ + "\n."
        ).encode()
        assert pickle.loads(historical_global) is cls
    assert RerankingConfig().strategy == "pointwise"
    assert RerankingConfig().score_scale == (0.0, 1.0)
    assert ProcessingConfig().num_workers is None
    assert ProcessingConfig().max_workers == 16


def test_facade_metadata_patch_seams_and_unknown_attribute_remain_live(monkeypatch):
    import tldw_chatbook.RAG_Search.simplified as facade
    from tldw_chatbook.RAG_Search.simplified import config

    actual = config.RAGConfig
    assert facade.RAGConfig is actual
    replacement = object()
    monkeypatch.setattr(facade, "RAGConfig", replacement)
    assert facade.RAGConfig is replacement and config.RAGConfig is actual
    monkeypatch.undo()
    monkeypatch.delattr(facade, "RAGConfig")
    monkeypatch.setattr(config, "RAGConfig", replacement)
    assert facade.RAGConfig is replacement
    with pytest.raises(AttributeError):
        getattr(facade, "unknown_metadata_export")
    assert set(public for public, _module, _attribute in _LEGACY_EXPORTS).issubset(
        dir(facade)
    )
