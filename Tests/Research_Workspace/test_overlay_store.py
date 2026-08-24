import json

import pytest

from tldw_chatbook.Research_Workspace.contracts import (
    QualifiedWorkspaceRef,
    WorkspaceDataSource,
)
from tldw_chatbook.Research_Workspace.layout_state import ResearchPanePreferences
from tldw_chatbook.Research_Workspace.overlay_store import (
    MAX_OVERLAY_FILE_BYTES,
    MAX_OVERLAY_RECORDS,
    OverlayConflictError,
    OverlayLimitError,
    ResearchPresentationOverlayStore,
)
from tldw_chatbook.Utils.private_paths import atomic_private_write_text


NOW = "2026-08-24T12:00:00Z"
LATER = "2026-08-24T12:01:00Z"


def local_ref(workspace_id: str = "workspace-local-1") -> QualifiedWorkspaceRef:
    return QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, workspace_id)


def server_ref(
    workspace_id: str,
    *,
    profile: str = "profile-a",
    principal: str = "principal-a",
) -> QualifiedWorkspaceRef:
    return QualifiedWorkspaceRef(
        WorkspaceDataSource.SERVER,
        workspace_id,
        server_profile_id=profile,
        principal_id=principal,
    )


def raw_record(
    ref: QualifiedWorkspaceRef,
    *,
    revision: int = 1,
    preferences: dict[str, object] | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "key": {
            "data_source": ref.data_source.value,
            "workspace_id": ref.workspace_id,
            "server_profile_id": ref.server_profile_id,
            "principal_id": ref.principal_id,
        },
        "revision": revision,
        "pane_preferences": preferences or {"sources_open": True, "studio_open": True},
        "preferred_companion": "sources",
        "created_at": NOW,
        "updated_at": NOW,
    }
    if extra:
        record.update(extra)
    return record


def write_payload(path, payload: object) -> None:
    atomic_private_write_text(
        path,
        json.dumps(payload),
        application_owned_directory=path.parent,
    )


def test_round_trip_persists_only_presentation_preferences_and_qualified_identity(
    tmp_path,
) -> None:
    path = tmp_path / "research" / "overlay-v1.json"
    store = ResearchPresentationOverlayStore(path)
    ref = server_ref("workspace-server-1")
    preferences = ResearchPanePreferences(
        sources_open=False,
        studio_open=True,
        preferred_companion="studio",
    )

    saved = store.save(ref, preferences, expected_revision=0, timestamp=NOW)
    loaded = store.load(ref)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert loaded == saved
    assert saved.revision == 1
    assert saved.preferences == preferences
    assert set(payload) == {"schema_version", "records"}
    assert set(payload["records"][0]) == {
        "key",
        "revision",
        "pane_preferences",
        "preferred_companion",
        "created_at",
        "updated_at",
    }
    serialized = json.dumps(payload).lower()
    for forbidden in ("forced_closed", "source_body", "chat_body", "path", "token"):
        assert forbidden not in serialized


def test_server_overlays_are_isolated_by_canonical_qualified_identity_not_display_name(
    tmp_path,
) -> None:
    store = ResearchPresentationOverlayStore(tmp_path / "research" / "overlay.json")
    first = server_ref("workspace-id-1")
    second = server_ref("workspace-id-2")

    store.save(
        first,
        ResearchPanePreferences(sources_open=False),
        expected_revision=0,
        timestamp=NOW,
    )
    store.save(
        second,
        ResearchPanePreferences(studio_open=False),
        expected_revision=0,
        timestamp=NOW,
    )

    assert store.load(first).preferences.sources_open is False  # type: ignore[union-attr]
    assert store.load(second).preferences.studio_open is False  # type: ignore[union-attr]


def test_compare_before_replace_rejects_a_concurrent_revision_change(tmp_path) -> None:
    path = tmp_path / "research" / "overlay.json"
    first_process = ResearchPresentationOverlayStore(path)
    second_process = ResearchPresentationOverlayStore(path)
    ref = local_ref()
    initial = first_process.save(
        ref,
        ResearchPanePreferences(),
        expected_revision=0,
        timestamp=NOW,
    )

    second_process.save(
        ref,
        ResearchPanePreferences(sources_open=False),
        expected_revision=initial.revision,
        timestamp=LATER,
    )

    with pytest.raises(OverlayConflictError, match="revision"):
        first_process.save(
            ref,
            ResearchPanePreferences(studio_open=False),
            expected_revision=initial.revision,
            timestamp=LATER,
        )

    assert first_process.load(ref).preferences.sources_open is False  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "poison",
    [
        {
            "pane_preferences": {
                "sources_open": True,
                "studio_open": True,
                "source_body": "canonical text",
            }
        },
        {"access_token": "secret-token-value"},
        {"workspace_path": "/private/source.md"},
    ],
)
def test_content_secret_and_path_fields_are_quarantined_per_record(
    tmp_path,
    poison: dict[str, object],
) -> None:
    path = tmp_path / "research" / "overlay.json"
    good_ref = local_ref("good")
    bad_ref = local_ref("bad")
    bad = raw_record(bad_ref)
    if "pane_preferences" in poison:
        bad["pane_preferences"] = poison["pane_preferences"]
    else:
        bad.update(poison)
    write_payload(
        path,
        {"schema_version": 1, "records": [raw_record(good_ref), bad]},
    )

    result = ResearchPresentationOverlayStore(path).load_all()

    assert tuple(result.records) == (good_ref,)
    assert len(result.quarantined) == 1
    assert result.quarantined[0].record_index == 1
    assert json.loads(result.quarantined[0].export_json()) == bad


def test_corrupt_record_does_not_block_canonical_workspace_overlay_loading(
    tmp_path,
) -> None:
    path = tmp_path / "research" / "overlay.json"
    good_ref = local_ref("canonical-workspace")
    write_payload(
        path,
        {
            "schema_version": 1,
            "records": [raw_record(good_ref), {"key": "not-a-qualified-key"}],
        },
    )

    result = ResearchPresentationOverlayStore(path).load_all()

    assert result.records[good_ref].preferences == ResearchPanePreferences()
    assert len(result.quarantined) == 1


def test_effective_forced_collapse_fields_are_never_accepted_for_persistence(
    tmp_path,
) -> None:
    path = tmp_path / "research" / "overlay.json"
    ref = local_ref()
    write_payload(
        path,
        {
            "schema_version": 1,
            "records": [
                raw_record(
                    ref,
                    extra={
                        "sources_forced_closed": True,
                        "studio_forced_closed": True,
                    },
                )
            ],
        },
    )

    result = ResearchPresentationOverlayStore(path).load_all()

    assert result.records == {}
    assert len(result.quarantined) == 1


def test_file_and_record_limits_fail_before_unbounded_decode(tmp_path) -> None:
    oversized_path = tmp_path / "oversized" / "overlay.json"
    atomic_private_write_text(
        oversized_path,
        " " * (MAX_OVERLAY_FILE_BYTES + 1),
        application_owned_directory=oversized_path.parent,
    )
    with pytest.raises(OverlayLimitError, match="file"):
        ResearchPresentationOverlayStore(oversized_path).load_all()

    crowded_path = tmp_path / "crowded" / "overlay.json"
    write_payload(
        crowded_path,
        {
            "schema_version": 1,
            "records": [
                raw_record(local_ref(str(index)))
                for index in range(MAX_OVERLAY_RECORDS + 1)
            ],
        },
    )
    with pytest.raises(OverlayLimitError, match="records"):
        ResearchPresentationOverlayStore(crowded_path).load_all()


def test_overlong_identity_string_is_quarantined_without_hiding_other_records(
    tmp_path,
) -> None:
    path = tmp_path / "research" / "overlay.json"
    good_ref = local_ref("good")
    overlong = raw_record(local_ref("x" * 300))
    write_payload(
        path,
        {"schema_version": 1, "records": [overlong, raw_record(good_ref)]},
    )

    result = ResearchPresentationOverlayStore(path).load_all()

    assert tuple(result.records) == (good_ref,)
    assert len(result.quarantined) == 1
