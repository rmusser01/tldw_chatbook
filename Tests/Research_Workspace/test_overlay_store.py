import json

import pytest

import tldw_chatbook.Research_Workspace.overlay_store as overlay_store_module
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
    ResearchSourceAnnotation,
    ResearchSourceFolder,
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
        "source_folders",
        "source_annotations",
    }
    serialized = json.dumps(payload).lower()
    for forbidden in ("forced_closed", "source_body", "chat_body", "path", "token"):
        assert forbidden not in serialized


def test_v1_overlay_migrates_to_v2_without_inventing_source_organization(
    tmp_path,
) -> None:
    path = tmp_path / "research" / "overlay.json"
    ref = server_ref("workspace-server-1")
    write_payload(path, {"schema_version": 1, "records": [raw_record(ref)]})

    store = ResearchPresentationOverlayStore(path)
    loaded = store.load(ref)

    assert loaded is not None
    assert loaded.source_folders == ()
    assert loaded.source_annotations == ()

    store.save(
        ref,
        loaded.preferences,
        expected_revision=loaded.revision,
        timestamp=LATER,
    )
    migrated = json.loads(path.read_text(encoding="utf-8"))
    assert migrated["schema_version"] == 2
    assert migrated["records"][0]["source_folders"] == []
    assert migrated["records"][0]["source_annotations"] == []


def test_v2_source_organization_is_qualified_bounded_and_private(tmp_path) -> None:
    path = tmp_path / "research" / "overlay.json"
    store = ResearchPresentationOverlayStore(path)
    first_ref = server_ref("same-id", profile="profile-a", principal="one")
    second_ref = server_ref("same-id", profile="profile-a", principal="two")
    folder = ResearchSourceFolder(
        folder_id="folder-reading",
        name="Reading",
        source_ids=("workspace-source-7",),
    )
    annotation = ResearchSourceAnnotation(
        annotation_id="annotation-1",
        source_id="workspace-source-7",
        quote="Bounded excerpt",
        note="Device-only observation",
        created_at=NOW,
        updated_at=NOW,
    )

    saved = store.save(
        first_ref,
        ResearchPanePreferences(),
        expected_revision=0,
        source_folders=(folder,),
        source_annotations=(annotation,),
        timestamp=NOW,
    )

    assert saved.source_folders == (folder,)
    assert saved.source_annotations == (annotation,)
    assert store.load(second_ref) is None
    serialized = path.read_text(encoding="utf-8")
    for forbidden in (
        "source_body",
        "note_body",
        "file_path",
        "https://user:secret@",
        "bearer ",
    ):
        assert forbidden not in serialized.lower()


def test_preference_only_save_preserves_concurrent_source_overlay_fields(
    tmp_path,
) -> None:
    path = tmp_path / "research" / "overlay.json"
    store = ResearchPresentationOverlayStore(path)
    ref = local_ref()
    first = store.save(
        ref,
        ResearchPanePreferences(),
        expected_revision=0,
        source_folders=(
            ResearchSourceFolder(
                folder_id="folder-a", name="Evidence", source_ids=("membership-a",)
            ),
        ),
        timestamp=NOW,
    )

    second = store.save(
        ref,
        ResearchPanePreferences(sources_open=False),
        expected_revision=first.revision,
        timestamp=LATER,
    )

    assert second.preferences.sources_open is False
    assert second.source_folders == first.source_folders


def test_overlays_are_isolated_on_every_qualified_identity_axis(
    tmp_path,
) -> None:
    store = ResearchPresentationOverlayStore(tmp_path / "research" / "overlay.json")
    refs_and_preferences = (
        (
            local_ref("same-workspace-id"),
            ResearchPanePreferences(sources_open=False, studio_open=False),
        ),
        (
            server_ref("same-workspace-id", profile="profile-a", principal="one"),
            ResearchPanePreferences(sources_open=False, studio_open=True),
        ),
        (
            server_ref("same-workspace-id", profile="profile-b", principal="one"),
            ResearchPanePreferences(sources_open=True, studio_open=False),
        ),
        (
            server_ref("same-workspace-id", profile="profile-a", principal="two"),
            ResearchPanePreferences(preferred_companion="studio"),
        ),
    )

    for ref, preferences in refs_and_preferences:
        store.save(
            ref,
            preferences,
            expected_revision=0,
            timestamp=NOW,
        )

    assert {
        ref: store.load(ref).preferences  # type: ignore[union-attr]
        for ref, _ in refs_and_preferences
    } == dict(refs_and_preferences)


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


def test_replace_boundary_rejects_revision_change_after_store_precondition_read(
    tmp_path,
    monkeypatch,
) -> None:
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
    real_atomic_write = overlay_store_module.atomic_private_write_text
    interleaved = False

    def replace_between_compare_and_helper(*args, **kwargs):
        nonlocal interleaved
        if not interleaved:
            interleaved = True
            second_process.save(
                ref,
                ResearchPanePreferences(sources_open=False),
                expected_revision=initial.revision,
                timestamp=LATER,
            )
        return real_atomic_write(*args, **kwargs)

    monkeypatch.setattr(
        overlay_store_module,
        "atomic_private_write_text",
        replace_between_compare_and_helper,
    )

    with pytest.raises(OverlayConflictError, match="replacement boundary"):
        first_process.save(
            ref,
            ResearchPanePreferences(studio_open=False),
            expected_revision=initial.revision,
            timestamp=LATER,
        )

    assert second_process.load(ref).preferences.sources_open is False  # type: ignore[union-attr]


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


def test_valid_save_preserves_corrupt_record_for_quarantine_export(tmp_path) -> None:
    path = tmp_path / "research" / "overlay.json"
    good_ref = local_ref("canonical-workspace")
    corrupt = {"key": "not-a-qualified-key"}
    write_payload(
        path,
        {
            "schema_version": 1,
            "records": [raw_record(good_ref), corrupt],
        },
    )
    store = ResearchPresentationOverlayStore(path)

    store.save(
        good_ref,
        ResearchPanePreferences(sources_open=False),
        expected_revision=1,
        timestamp=LATER,
    )
    reloaded = store.load_all()

    assert reloaded.records[good_ref].revision == 2
    assert len(reloaded.quarantined) == 1
    assert json.loads(reloaded.quarantined[0].export_json()) == corrupt


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
