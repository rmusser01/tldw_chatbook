"""Resolve result document IDs only within their declared local source type."""

import pytest

from tldw_chatbook.Library.library_rag_state import LibraryRagResultRow


def _row(kind, source_id, **kwargs):
    return LibraryRagResultRow.from_result(
        {"source_id": source_id, "provenance": {"source_type": kind}, **kwargs}
    )


@pytest.mark.parametrize("kind", ["media", "prompt"])
@pytest.mark.parametrize("shape", ["17", "{kind}_17", "{kind}-17", "local:{kind}:17"])
def test_numeric_source_spellings_share_the_exact_open_identity(kind, shape):
    row = _row(kind, shape.format(kind=kind))
    assert row.resolve_local_open_id() == (
        "local:media:17" if kind == "media" else "17"
    )
    assert row.source_id == shape.format(kind=kind)


@pytest.mark.parametrize("kind", ["media", "prompt"])
@pytest.mark.parametrize(
    "source_id",
    [
        "",
        "0",
        "-1",
        "+1",
        "1.0",
        "17_chunk_2",
        "other_17",
        "server:media:17",
        "local:notes:17",
        "17:18",
        "1 7",
        "media_media_17",
    ],
)
def test_invalid_or_different_authority_ids_are_never_guessed(kind, source_id):
    with pytest.raises(ValueError, match="Can't open"):
        _row(kind, source_id).resolve_local_open_id()


@pytest.mark.parametrize("kind", ["notes", "conversations"])
@pytest.mark.parametrize(
    "source_id", ["opaque-note_17", "73d691ff-38a1-4187-9e6a-d4e30b086e77"]
)
def test_opaque_source_record_ids_are_not_rewritten(kind, source_id):
    assert _row(kind, source_id).resolve_local_open_id() == source_id


@pytest.mark.parametrize("kind", ["media", "prompt", "notes", "conversations"])
@pytest.mark.parametrize(
    "metadata",
    [
        {"runtime_backend": "server"},
        {"runtime_backend": "rag-server"},
        {"provenance": {"source_authority": "server"}},
        {"provenance": {"backend": "server"}},
    ],
)
def test_server_result_cannot_resolve_to_a_local_record(kind, metadata):
    values = {"source_type": kind, **metadata.get("provenance", {})}
    with pytest.raises(ValueError, match="server"):
        _row(kind, "17", **{**metadata, "provenance": values}).resolve_local_open_id()


def test_unknown_source_type_cannot_open():
    with pytest.raises(ValueError, match="Can't open"):
        _row("unknown", "17").resolve_local_open_id()


@pytest.mark.parametrize("kind", ["media", "prompt", "notes", "conversations"])
@pytest.mark.parametrize("suffix", ["javascript:17", "onclick=17", "1\x007"])
def test_display_sanitization_cannot_grant_a_different_record_identity(kind, suffix):
    row = _row(kind, f"{kind}_{suffix}")
    assert row.source_id == f"{kind}_17"
    with pytest.raises(ValueError, match="Can't open.*source ID is invalid"):
        row.resolve_local_open_id()


def test_excessively_long_numeric_id_has_a_user_facing_refusal():
    with pytest.raises(ValueError, match="Can't open.*source ID is invalid"):
        _row("media", "media_" + "1" * 5000).resolve_local_open_id()
