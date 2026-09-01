"""Phase-39 cutover contracts for the Collections capture reader."""

from __future__ import annotations

import inspect

import pytest

from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
from tldw_chatbook.Library.library_collections_service import (
    LegacyCollectionsReadOnlyError,
    LocalLibraryCollectionsService,
)
from tldw_chatbook.UI.Screens import library_screen


def test_collections_route_has_no_generic_container_controller_or_panel() -> None:
    source = inspect.getsource(library_screen)

    assert "LibraryCollectionsBrowseController" not in source
    assert "LibraryCollectionsPanel" not in source
    assert "create_library_collection" not in source
    assert "rename_library_collection" not in source
    assert "confirm_library_collection_delete" not in source
    assert "LibraryCollectionsCaptureController" in source
    assert "LibraryCollectionsItemsPane" in source
    assert "LibraryCollectionsWorkPane" in source


@pytest.mark.parametrize(
    ("method_name", "args", "kwargs"),
    (
        ("create_collection", ("Old container",), {}),
        ("rename_collection", ("legacy-1", "Renamed"), {}),
        ("add_item_to_collection", ("legacy-1",), {"source_type": "note", "source_id": "1"}),
        ("delete_collection", ("legacy-1",), {}),
        ("restore_collection", ("legacy-1",), {}),
    ),
)
def test_legacy_container_mutations_remain_callable_and_read_only(
    tmp_path,
    method_name: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> None:
    db = LibraryCollectionsDB(tmp_path / "collections.db")
    service = LocalLibraryCollectionsService(db)

    with pytest.raises(LegacyCollectionsReadOnlyError) as caught:
        getattr(service, method_name)(*args, **kwargs)

    assert caught.value.reason == "legacy_read_only"
    assert "recovery" in caught.value.recovery.casefold()
    db.close()
