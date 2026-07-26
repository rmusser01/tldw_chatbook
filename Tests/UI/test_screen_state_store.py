from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import logging

import pytest

from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.UI.Navigation.screen_state_store import (
    RuntimeIdentity,
    ScreenStateStore,
)


def _local_identity() -> RuntimeIdentity:
    return RuntimeIdentity(active_source="local", active_server_id=None)


def test_runtime_identity_derives_only_runtime_scope() -> None:
    assert RuntimeIdentity.from_state(
        RuntimeSourceState(
            active_source="server",
            active_server_id="server-a",
            last_known_server_label="private-label",
            server_reachability="reachable",
        )
    ) == RuntimeIdentity("server", "server-a")
    assert RuntimeIdentity.from_state(
        RuntimeSourceState(
            active_source="local",
            active_server_id="stale-server",
        )
    ) == RuntimeIdentity("local", None)
    assert RuntimeIdentity("local", "stale-server") == RuntimeIdentity(
        "local",
        None,
    )


@pytest.mark.parametrize("source", ["", "remote", " local "])
def test_runtime_identity_rejects_noncanonical_sources(source: str) -> None:
    with pytest.raises(ValueError, match="runtime source"):
        RuntimeIdentity(source)


def test_save_and_restore_copy_only_the_outer_mapping() -> None:
    nested = {"history": ["large", "payload"]}
    original = {"selected": "row-1", "nested": nested}
    store = ScreenStateStore()
    identity = _local_identity()

    store.save("chat", original, identity)
    original["selected"] = "changed-after-save"
    restored = store.restore("chat", identity)

    assert restored == {"selected": "row-1", "nested": nested}
    assert restored is not original
    assert restored["nested"] is nested
    restored["selected"] = "consumer-change"
    assert store.restore("chat", identity)["selected"] == "row-1"


def test_save_and_restore_never_deep_copy_nested_payloads() -> None:
    class DeepCopySentinel:
        def __deepcopy__(self, _memo):
            raise AssertionError("screen state store must not deep-copy payloads")

    nested = DeepCopySentinel()
    store = ScreenStateStore()
    identity = _local_identity()

    store.save("console", {"history": nested}, identity)

    assert store.restore("console", identity)["history"] is nested


@pytest.mark.parametrize("route", ["", "   "])
def test_empty_canonical_key_is_rejected_without_mutation(route: str) -> None:
    store = ScreenStateStore()
    identity = _local_identity()
    store.save("chat", {"selected": "row-1"}, identity)

    with pytest.raises(ValueError, match="canonical route"):
        store.save(route, {"selected": "row-2"}, identity)

    assert store.restore("chat", identity) == {"selected": "row-1"}


def test_non_string_canonical_key_is_rejected_without_mutation() -> None:
    store = ScreenStateStore()
    identity = _local_identity()
    store.save("chat", {"selected": "row-1"}, identity)

    with pytest.raises(TypeError, match="canonical route"):
        store.save(17, {"selected": "row-2"}, identity)  # type: ignore[arg-type]

    assert store.restore("chat", identity) == {"selected": "row-1"}


def test_non_mapping_snapshot_is_rejected_without_mutation() -> None:
    store = ScreenStateStore()
    identity = _local_identity()
    store.save("chat", {"selected": "row-1"}, identity)

    with pytest.raises(TypeError, match="screen snapshot"):
        store.save("chat", ["not", "a", "mapping"], identity)  # type: ignore[arg-type]

    assert store.restore("chat", identity) == {"selected": "row-1"}


def test_runtime_identity_argument_is_required_without_mutation() -> None:
    store = ScreenStateStore()
    identity = _local_identity()
    store.save("chat", {"selected": "row-1"}, identity)

    with pytest.raises(TypeError, match="runtime identity"):
        store.save("chat", {"selected": "row-2"}, object())  # type: ignore[arg-type]

    assert store.restore("chat", identity) == {"selected": "row-1"}


def test_server_identity_mismatch_discards_snapshot() -> None:
    store = ScreenStateStore()
    store.save(
        "library",
        {"selected": "n-1"},
        RuntimeIdentity("server", "server-a"),
    )

    assert (
        store.restore(
            "library",
            RuntimeIdentity("server", "server-b"),
        )
        is None
    )
    assert store.has_snapshots(RuntimeIdentity("server", "server-a")) is False


def test_source_mismatch_discards_snapshot() -> None:
    store = ScreenStateStore()
    store.save(
        "library",
        {"selected": "n-1"},
        RuntimeIdentity("server", "server-a"),
    )

    assert store.restore("library", _local_identity()) is None
    assert store.has_snapshots(_local_identity()) is False


def test_local_identity_ignores_stale_server_metadata() -> None:
    store = ScreenStateStore()
    store.save(
        "library",
        {"selected": "n-1"},
        RuntimeIdentity("local", "stale-server-a"),
    )

    assert store.restore(
        "library",
        RuntimeIdentity("local", "stale-server-b"),
    ) == {"selected": "n-1"}


def test_restore_discards_corrupt_envelope() -> None:
    store = ScreenStateStore()
    identity = _local_identity()
    store.save("library", {"selected": "n-1"}, identity)
    store._entries["library"] = object()  # type: ignore[assignment]

    assert store.restore("library", identity) is None
    assert store.has_snapshots(identity) is False


def test_restore_discards_envelope_stored_under_the_wrong_route() -> None:
    store = ScreenStateStore()
    identity = _local_identity()
    store.save("library", {"selected": "n-1"}, identity)
    store._entries["chat"] = store._entries.pop("library")

    assert store.restore("chat", identity) is None
    assert store.has_snapshots(identity) is False


def test_has_snapshots_lazily_discards_only_incompatible_entries() -> None:
    store = ScreenStateStore()
    local = _local_identity()
    store.save("chat", {"selected": "c-1"}, local)
    store.save(
        "library",
        {"selected": "n-1"},
        RuntimeIdentity("server", "server-a"),
    )

    assert store.has_snapshots(local) is True
    assert store.restore("chat", local) == {"selected": "c-1"}
    assert (
        store.restore(
            "library",
            RuntimeIdentity("server", "server-a"),
        )
        is None
    )


def test_discard_is_idempotent_and_route_scoped() -> None:
    store = ScreenStateStore()
    identity = _local_identity()
    store.save("chat", {"selected": "c-1"}, identity)
    store.save("library", {"selected": "n-1"}, identity)

    store.discard("chat")
    store.discard("chat")

    assert store.restore("chat", identity) is None
    assert store.restore("library", identity) == {"selected": "n-1"}


@pytest.mark.parametrize(
    "operation",
    [
        lambda store, identity: store.save("chat", {}, identity),
        lambda store, identity: store.restore("chat", identity),
        lambda store, _identity: store.discard("chat"),
        lambda store, identity: store.has_snapshots(identity),
    ],
)
def test_all_store_operations_reject_off_owner_thread(
    operation: Callable[[ScreenStateStore, RuntimeIdentity], object],
) -> None:
    store = ScreenStateStore()
    identity = _local_identity()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(operation, store, identity)
        with pytest.raises(RuntimeError, match="owner thread"):
            future.result()


def test_failure_diagnostics_never_log_snapshot_or_sentinel(
    caplog: pytest.LogCaptureFixture,
) -> None:
    payload_sentinel = "SCREEN-SNAPSHOT-PAYLOAD-SENTINEL-6a72"
    store = ScreenStateStore()
    identity = _local_identity()
    caplog.set_level(logging.DEBUG)

    with pytest.raises(TypeError, match="screen snapshot"):
        store.save("chat", [payload_sentinel], identity)  # type: ignore[arg-type]

    assert payload_sentinel not in caplog.text
    assert "SCREEN-SNAPSHOT" not in caplog.text


def test_store_exposes_no_persistence_or_backing_mapping_api() -> None:
    store = ScreenStateStore()

    for name in (
        "entries",
        "mapping",
        "persist",
        "load",
        "save_to_disk",
        "to_dict",
    ):
        assert not hasattr(store, name)
