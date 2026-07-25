"""Uniform Chroma ``persist_directory`` validation (task-482).

Both ``ChromaVectorStore`` (vector_store.py) and
``collection_indexes._client()`` construct a ``chromadb.PersistentClient``
against a config-sourced ``persist_directory``. chromadb's
``SharedSystemClient`` caches one client per persist-directory *string* per
process and raises ``ValueError`` on a ``Settings`` mismatch for a second
construction at the same on-disk path -- so the two call sites must never
normalize that path differently, and must reject invalid input identically.

These tests exercise the real construction seams (``ChromaVectorStore.client``
and ``collection_indexes._client``), not copies of the normalization logic,
so a future edit that reintroduces divergence between the two sites would be
caught here.
"""
from unittest.mock import patch

import pytest

from tldw_chatbook.RAG_Search.simplified.vector_store import ChromaVectorStore
from tldw_chatbook.RAG_Search.simplified import collection_indexes
from tldw_chatbook.RAG_Search.simplified.config import (
    validate_chroma_persist_directory,
)


@pytest.mark.requires_chromadb
def test_both_sites_construct_client_with_identical_path(tmp_path):
    """Same messy-but-legit input -> identical ``path=`` at both real seams.

    A double slash + trailing slash is legal on POSIX but would previously
    normalize differently depending on whether the receiving site wrapped
    the value in ``Path(...)`` (as ``ChromaVectorStore.__init__`` did) or
    used the raw string as-is (as ``collection_indexes._client`` did).
    """
    raw_input = str(tmp_path) + "/nested_index//store/"

    with patch("chromadb.PersistentClient") as mock_client_cls:
        store = ChromaVectorStore(persist_directory=raw_input)
        _ = store.client  # triggers the first PersistentClient(...) construction

        collection_indexes._client(raw_input)  # second, independent construction site

    assert mock_client_cls.call_count == 2
    first_path = mock_client_cls.call_args_list[0].kwargs["path"]
    second_path = mock_client_cls.call_args_list[1].kwargs["path"]
    assert first_path == second_path


@pytest.mark.requires_chromadb
def test_both_sites_reject_dangerous_path_identically(tmp_path):
    """An invalid (null-byte-containing) path is rejected the same way by both."""
    dangerous = str(tmp_path) + "/evil\x00dir"

    with pytest.raises(ValueError):
        ChromaVectorStore(persist_directory=dangerous)

    with pytest.raises(ValueError):
        collection_indexes._client(dangerous)


def test_expanduser_and_absolute_paths_normalize_stably(monkeypatch, tmp_path):
    """Legit absolute and ``~``-containing paths still work and are stable."""
    monkeypatch.setenv("HOME", str(tmp_path))

    from_tilde = validate_chroma_persist_directory("~/chromadb")
    from_absolute = validate_chroma_persist_directory(str(tmp_path / "chromadb"))

    assert from_tilde == from_absolute == tmp_path / "chromadb"
    # Idempotent: re-validating an already-validated path is a no-op.
    assert validate_chroma_persist_directory(from_tilde) == from_tilde


# === Producer/consumer agreement (task-482 review follow-up) ===
#
# validate_chroma_persist_directory closes the divergence gap between the two
# *consumer* sites (ChromaVectorStore, collection_indexes._client), but a
# persist_directory that reaches either consumer with a literal, unexpanded
# ``~`` would silently normalize to a DIFFERENT string than one that was
# already expanded upstream -- reintroducing the exact SharedSystemClient
# collision this task exists to prevent, just one hop earlier. The two
# *producer* sites below must therefore expand (and validate) a
# ``~``-containing persist_directory the SAME way the consumer sites do.


def test_env_var_persist_directory_expands_tilde_like_client_sites(monkeypatch, tmp_path):
    """RAG_PERSIST_DIR=~/x must resolve to the same path validate_chroma_persist_directory
    computes -- the active-profile env-override layer (active_config.py) is a
    persist_directory PRODUCER and must agree with the consumer sites.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RAG_PERSIST_DIR", "~/x")

    from tldw_chatbook.RAG_Search.simplified.active_config import _apply_env_overrides
    from tldw_chatbook.RAG_Search.simplified.config import RAGConfig

    config = _apply_env_overrides(RAGConfig())

    assert config.vector_store.persist_directory == validate_chroma_persist_directory("~/x")


def test_from_dict_persist_directory_expands_tilde_like_client_sites(monkeypatch, tmp_path):
    """A saved/legacy profile JSON's persist_directory ('~/x') must resolve to
    the same path validate_chroma_persist_directory computes --
    RAGConfig.from_dict() is a persist_directory PRODUCER (profile load path)
    and must agree with the consumer sites.
    """
    monkeypatch.setenv("HOME", str(tmp_path))

    from tldw_chatbook.RAG_Search.simplified.config import RAGConfig

    config = RAGConfig.from_dict(
        {"vector_store": {"type": "chroma", "persist_directory": "~/x"}}
    )

    assert config.vector_store.persist_directory == validate_chroma_persist_directory("~/x")
