from dataclasses import FrozenInstanceError

import pytest

from tldw_chatbook.Library.library_artifacts_state import ArtifactKey, ArtifactScope


def test_identity_is_namespaced_immutable_and_validated():
    assert ArtifactKey("live_report", 1) != ArtifactKey("kept_report", 1)
    with pytest.raises(FrozenInstanceError):
        ArtifactKey("live_report", 1).native_id = 2
    for source, value in [("bogus", 1), ("kept_report", True), ("kept_report", 0)]:
        with pytest.raises(ValueError):
            ArtifactKey(source, value)


def test_scope_normalizes_query_without_unicode_casefold():
    assert ArtifactScope(query="  ÉNEWS  ").query == "ÉNEWS"
    with pytest.raises(ValueError):
        ArtifactScope(sort="invalid")
