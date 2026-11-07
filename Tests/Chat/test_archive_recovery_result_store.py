"""Real session-registry evidence for ADR-166."""

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from tldw_chatbook.Chat.archive_recovery.contracts import (
    ArchiveCandidate,
    ArchiveRecoveryError,
    ArchiveSearchPage,
    ArchiveSearchRequest,
    ArchiveTarget,
    ArchiveWorkspaceSnapshot,
    RecoveryCallKey,
)
from tldw_chatbook.Chat.archive_recovery.result_store import (
    ArchiveResultStore,
    argument_digest,
)


def candidate(identity="first"):
    return ArchiveCandidate(
        target=ArchiveTarget(identity, 3, "First", True, None, None),
        conversation_archived_at="2026-09-21T10:00:00+00:00",
        excerpt="A remembered discussion",
        match_basis=("chat",),
        sort_key=(0, "2026-09-21T10:00:00+00:00", identity),
        chronology_incomplete=False,
        match_message_id=None,
        allowed_scopes=("chat_only",),
    )


def page(item=None, anchor=None):
    return ArchiveSearchPage((item or candidate(),), anchor, "digest", False)


def test_immutable_original_survives_later_pages_and_projection_mutation():
    store = ArchiveResultStore()
    first = store.publish("s1", ArchiveSearchRequest(), page(), model_visible=True)
    ref = first["items"][0]["result_ref"]
    first["items"][0]["title"] = "tampered"
    store.publish(
        "s1", ArchiveSearchRequest(), page(candidate("second")), model_visible=True
    )
    retained = store.resolve("s1", ref)
    assert retained.target.conversation_id == "first"
    assert retained.target.title == "First"
    with pytest.raises(FrozenInstanceError):
        retained.target.title = "changed"


def test_expiry_wrong_owner_and_manual_agent_selection():
    clock = [100.0]
    store = ArchiveResultStore(monotonic=lambda: clock[0])
    output = store.publish("s1", ArchiveSearchRequest(), page(), model_visible=False)
    ref = output["items"][0]["result_ref"]
    assert store.resolve("s1", ref).excerpt == "A remembered discussion"
    with pytest.raises(ArchiveRecoveryError):
        store.resolve("s2", ref)
    with pytest.raises(ArchiveRecoveryError, match="manual"):
        store.resolve("s1", ref, for_agent=True)
    clock[0] += 1800
    with pytest.raises(ArchiveRecoveryError, match="expired"):
        store.resolve("s1", ref)


def test_fifo_eviction_cursors_close_restart_and_import_cannot_rehydrate():
    store = ArchiveResultStore()
    request = ArchiveSearchRequest(query="needle", workspace_id=" named ", limit=20)
    first = store.publish(
        "s1", request, page(anchor=candidate().sort_key), model_visible=True
    )
    ref = first["items"][0]["result_ref"]
    cursor = first["cursor"]
    bound, anchor, digest = store.resolve_cursor("s1", cursor)
    assert bound == request and bound.workspace_id == "named"
    assert anchor == candidate().sort_key and digest == "digest"
    imported = json.loads(json.dumps(first))
    imported["query"] = "changed"
    assert store.resolve_cursor("s1", cursor)[0].query == "needle"
    with pytest.raises(ArchiveRecoveryError):
        store.resolve_cursor("s2", cursor)
    for i in range(10):
        store.publish("s1", request, page(candidate(str(i))), model_visible=True)
    with pytest.raises(ArchiveRecoveryError):
        store.resolve("s1", ref)
    with pytest.raises(ArchiveRecoveryError):
        store.resolve_cursor("s1", cursor)
    newest = store.publish("s1", request, page(), model_visible=True)
    new_ref = newest["items"][0]["result_ref"]
    with pytest.raises(ArchiveRecoveryError):
        ArchiveResultStore().resolve("s1", new_ref)
    store.close_session("s1")
    with pytest.raises(ArchiveRecoveryError):
        store.resolve("s1", new_ref)


def test_cursor_expiry_is_from_creation_and_not_extended_by_read():
    clock = [0.0]
    store = ArchiveResultStore(monotonic=lambda: clock[0])
    result = store.publish(
        "s",
        ArchiveSearchRequest(),
        page(anchor=candidate().sort_key),
        model_visible=True,
    )
    clock[0] = 1799
    store.resolve_cursor("s", result["cursor"])
    clock[0] = 1801
    with pytest.raises(ArchiveRecoveryError, match="expired"):
        store.resolve_cursor("s", result["cursor"])


def test_digest_and_call_identity_are_canonical_and_frozen():
    assert argument_digest({"b": "世界", "a": 1}) == argument_digest(
        {"a": 1, "b": "世界"}
    )
    assert len(argument_digest({})) == 64
    key = RecoveryCallKey("s", "r", "c", argument_digest({}))
    with pytest.raises(FrozenInstanceError):
        key.call_id = "other"


def test_public_projection_separates_event_basis_and_message_context():
    item = replace(
        candidate(),
        match_basis=("chat", "title", "keywords", "message", "inactive_branch"),
        match_message_id="m",
    )
    row = ArchiveResultStore().publish(
        "s", ArchiveSearchRequest(), page(item), model_visible=True
    )["items"][0]
    assert row["archive_match_basis"] == ["chat"]
    assert row["text_match_basis"] == ["title", "keywords", "message"]
    assert row["inactive_branch"] is True and row["match_message_id"] == "m"
    assert row["allowed_scopes"] == ["chat_only"]


def test_unicode_final_byte_fitting_anchors_at_last_emitted_row():
    workspace = ArchiveWorkspaceSnapshot(
        "w", "界" * 480, True, "2026-09-21T10:00:00+00:00", 3
    )
    items = tuple(
        replace(
            candidate(str(i)),
            target=ArchiveTarget(str(i), 3, "界" * 1024, True, "w", workspace),
            excerpt="界" * 320,
        )
        for i in range(20)
    )
    store = ArchiveResultStore()
    output = store.publish(
        "s",
        ArchiveSearchRequest(limit=20),
        ArchiveSearchPage(items, None, "digest", True),
        model_visible=True,
    )
    assert (
        len(json.dumps(output, ensure_ascii=False, separators=(",", ":")).encode())
        <= 32 * 1024
    )
    assert output["truncated"]
    assert output["items"]
    if len(output["items"]) < len(items):
        assert (
            store.resolve_cursor("s", output["cursor"])[1]
            == items[len(output["items"]) - 1].sort_key
        )
    for row in output["items"]:
        assert row["workspace"]["name"] == workspace.name
        assert store.resolve("s", row["result_ref"]).excerpt == "界" * 320


def test_oversized_identity_fails_without_publishing_an_empty_cursor():
    huge = replace(
        candidate(), target=replace(candidate().target, conversation_id="界" * 12000)
    )
    with pytest.raises(ArchiveRecoveryError, match="fit"):
        ArchiveResultStore().publish(
            "s", ArchiveSearchRequest(), page(huge), model_visible=True
        )


def test_near_boundary_unicode_lifecycle_and_cursor_overhead_force_row_omission():
    workspace = ArchiveWorkspaceSnapshot("w", "界" * 500, True, None, 19)
    items = tuple(
        replace(
            candidate(str(i)),
            target=ArchiveTarget(str(i), 3, "", True, "w", workspace),
            excerpt="",
        )
        for i in range(20)
    )
    store = ArchiveResultStore()
    output = store.publish(
        "s",
        ArchiveSearchRequest(limit=20),
        ArchiveSearchPage(items, items[-1].sort_key, "d" * 64, True),
        model_visible=True,
    )
    assert 1 < len(output["items"]) < 20
    assert (
        30000
        < len(json.dumps(output, ensure_ascii=False, separators=(",", ":")).encode())
        <= 32768
    )
    request, anchor, digest = store.resolve_cursor("s", output["cursor"])
    assert anchor == items[len(output["items"]) - 1].sort_key
    assert digest == "d" * 64 and request.limit == 20
    assert all(row["workspace"]["archive_revision"] == 19 for row in output["items"])


def test_supplied_mutable_tuple_fields_are_snapshotted():
    scopes = ["chat_only"]
    basis = ["chat"]
    item = replace(candidate(), allowed_scopes=scopes, match_basis=basis)
    store = ArchiveResultStore()
    output = store.publish("s", ArchiveSearchRequest(), page(item), model_visible=True)
    scopes.append("workspace_only")
    basis.append("message")
    retained = store.resolve("s", output["items"][0]["result_ref"])
    assert retained.allowed_scopes == ("chat_only",)
    assert retained.match_basis == ("chat",)


def test_empty_results_and_separate_session_retention():
    store = ArchiveResultStore()
    empty = store.publish(
        "s",
        ArchiveSearchRequest(),
        ArchiveSearchPage((), None, "d", False),
        model_visible=True,
    )
    assert empty["items"] == [] and empty["cursor"] is None
    other = store.publish("other", ArchiveSearchRequest(), page(), model_visible=True)
    for i in range(11):
        store.publish(
            "s", ArchiveSearchRequest(), page(candidate(str(i))), model_visible=True
        )
    assert (
        store.resolve("other", other["items"][0]["result_ref"]).target.conversation_id
        == "first"
    )


def test_supplied_list_cursor_anchor_cannot_mutate_retained_continuation():
    supplied_anchor = [0, "2026-09-21T10:00:00+00:00", "first"]
    store = ArchiveResultStore()
    output = store.publish(
        "s", ArchiveSearchRequest(), page(anchor=supplied_anchor), model_visible=True
    )
    supplied_anchor[2] = "caller-tampered"
    resolved = store.resolve_cursor("s", output["cursor"])[1]
    assert resolved == (0, "2026-09-21T10:00:00+00:00", "first")
    assert isinstance(resolved, tuple)
    with pytest.raises(TypeError):
        resolved[2] = "resolver-tampered"
    assert store.resolve_cursor("s", output["cursor"])[1] == (
        0,
        "2026-09-21T10:00:00+00:00",
        "first",
    )


@pytest.mark.parametrize(
    "anchor",
    [
        [],
        [0, "time"],
        [False, "time", "id"],
        [0, [], "id"],
        [0, "time", []],
        [2, "time", "id"],
    ],
)
def test_invalid_cursor_anchor_is_refused_before_publication(anchor):
    store = ArchiveResultStore()
    with pytest.raises(ArchiveRecoveryError, match="anchor"):
        store.publish(
            "s", ArchiveSearchRequest(), page(anchor=anchor), model_visible=True
        )
