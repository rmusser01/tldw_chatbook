"""One-shot collection operations release only their acquired client."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tldw_chatbook.RAG_Search.simplified import collection_indexes as indexes


def _run_operation(operation: str, path: Path):
    if operation == "list":
        return indexes.list_indexes(path)
    if operation == "adopt":
        return indexes.adopt_legacy_collection(path, "legacy", "adopted", {})
    return indexes.delete_index(path, "legacy")


@pytest.mark.parametrize("operation", ["list", "adopt", "delete"])
@pytest.mark.parametrize(
    "outcome",
    [
        "success",
        "empty",
        "error",
        "close_error",
        "both_errors",
        "no_close",
        "noncallable_close",
    ],
)
def test_collection_operation_releases_exact_client_without_changing_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, operation: str, outcome: str
) -> None:
    """Owned disposal covers success, refusal, errors and older clients.

    Args:
        monkeypatch: Replace only this operation's client constructor.
        tmp_path: Test-owned client destination.
        operation: Collection operation under test.
        outcome: Operation result or client-close capability to exercise.
    """
    collection = SimpleNamespace(
        name="legacy", metadata={}, count=Mock(return_value=2), modify=Mock()
    )
    client = SimpleNamespace(
        list_collections=Mock(return_value=[] if outcome == "empty" else [collection]),
        get_collection=Mock(return_value=collection),
        delete_collection=Mock(),
    )
    close = Mock()
    if outcome != "no_close":
        client.close = None if outcome == "noncallable_close" else close
    if outcome in {"error", "both_errors"}:
        client.list_collections.side_effect = RuntimeError("operation failed")
    if outcome in {"close_error", "both_errors"}:
        close.side_effect = RuntimeError("close failed")
    monkeypatch.setattr(indexes, "_client", lambda _path: client)

    result = _run_operation(operation, tmp_path)

    if operation == "list":
        assert result == (
            []
            if outcome in {"empty", "error", "both_errors"}
            else [{"name": "legacy", "fp": None, "provenance": {}, "count": 2}]
        )
    else:
        assert result is (outcome not in {"empty", "error", "both_errors"})
    if outcome in {"no_close", "noncallable_close"}:
        close.assert_not_called()
    else:
        close.assert_called_once_with()


def test_adoption_existing_target_releases_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An existing adoption target keeps both collections and releases its client.

    Args:
        monkeypatch: Supply the operation-owned client.
        tmp_path: Test-owned client destination.
    """
    client = SimpleNamespace(
        list_collections=lambda: [
            SimpleNamespace(name="legacy"),
            SimpleNamespace(name="adopted"),
        ],
        get_collection=Mock(),
        close=Mock(),
    )
    monkeypatch.setattr(indexes, "_client", lambda _path: client)
    assert _run_operation("adopt", tmp_path) is False
    client.get_collection.assert_not_called()
    client.close.assert_called_once_with()


@pytest.mark.parametrize("operation", ["list", "adopt", "delete"])
def test_collection_client_construction_failure_preserves_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, operation: str
) -> None:
    """A failed constructor creates no cleanup target and retains the fallback.

    Args:
        monkeypatch: Fail client construction before an owner is acquired.
        tmp_path: Test-owned client destination.
        operation: Collection operation under test.
    """
    monkeypatch.setattr(
        indexes, "_client", Mock(side_effect=RuntimeError("construction failed"))
    )
    result = _run_operation(operation, tmp_path)
    assert result == ([] if operation == "list" else False)


@pytest.mark.requires_chromadb
def test_repeated_real_operations_preserve_foreign_same_path_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Real close retires each local reference without stopping a shared owner.

    Args:
        monkeypatch: Record the exact clients acquired by collection operations.
        tmp_path: Unique persistent database, never a process-global Chroma path.
    """
    import chromadb
    from chromadb.api.client import Client
    from chromadb.api.shared_system_client import SharedSystemClient
    from chromadb.config import Settings

    if not callable(getattr(Client, "close", None)):
        pytest.skip("Installed Chroma has no exact-client close API")
    path = tmp_path / "chroma"
    foreign = chromadb.PersistentClient(
        path=str(path), settings=Settings(anonymized_telemetry=False, allow_reset=True)
    )
    owned = []
    try:
        identifier = foreign._identifier
        system = foreign._system
        baseline = SharedSystemClient._identifier_to_refcount[identifier]
        original_client = indexes._client

        def record_client(persist_directory):
            client = original_client(persist_directory)
            owned.append(client)
            return client

        monkeypatch.setattr(indexes, "_client", record_client)
        for iteration in range(3):
            legacy = f"legacy-{iteration}"
            adopted = f"adopted-{iteration}"
            collection = foreign.create_collection(legacy)
            collection.add(
                ids=["row"], embeddings=[[0.1, 0.2]], documents=["local test"]
            )
            assert (
                indexes.adopt_legacy_collection(
                    path, legacy, adopted, {"source": "lifecycle-test"}
                )
                is True
            )
            assert foreign.get_collection(adopted).count() == 1
            assert any(entry["name"] == adopted for entry in indexes.list_indexes(path))
            assert indexes.delete_index(path, adopted) is True
            assert indexes.delete_index(path, adopted) is False
            assert indexes.adopt_legacy_collection(path, legacy, adopted, {}) is False
            assert all(client._closed for client in owned)
            assert SharedSystemClient._identifier_to_refcount[identifier] == baseline
            assert foreign._system is system
            assert foreign.list_collections() == []
    finally:
        # Safety cleanup also runs on RED; never clear the shared cache or stop
        # a system directly. Chroma's idempotent close owns reference release.
        for client in reversed([foreign, *owned]):
            close = getattr(client, "close", None)
            if callable(close):
                close()
