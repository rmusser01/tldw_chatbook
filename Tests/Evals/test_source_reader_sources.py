"""Source-reader boundaries exercised through disposable SQLite Library items."""

from dataclasses import FrozenInstanceError

import pytest

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Evals.source_reader.sources import (
    Packet,
    Selection,
    Source,
    SourceReaderError,
    assemble_sources,
    pack_sources,
)
from tldw_chatbook.Library.library_tool_contract import make_cursor, make_public_id
from tldw_chatbook.Library.local_library_tool_service import LocalLibraryToolService
from tldw_chatbook.Media.local_media_reading_service import LocalMediaReadingService


@pytest.fixture
def library(tmp_path):
    db = MediaDatabase(db_path=tmp_path / "media.db", client_id="source-reader-test")
    service = LocalLibraryToolService(media_service=LocalMediaReadingService(db))
    yield db, service
    db.close_connection()


def add_source(library, text, title="Fixture source"):
    db, service = library
    _, raw_id, _ = db.add_media_with_keywords(
        title=title, content=text, media_type="article"
    )
    assert raw_id
    source_id = make_public_id("media", raw_id)
    page = service.invoke("library_get_media", {"id": source_id, "max_chars": 1})
    return Selection(source_id, page["content"]["revision"])


class ObservedService:
    """Keep real DB reads, optionally corrupt one response or change the DB."""

    def __init__(self, service, *, before=None, change=None):
        self.service = service
        self.before = before
        self.change = change
        self.calls = []

    def invoke(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        if self.before:
            self.before(len(self.calls), arguments)
        result = self.service.invoke(name, arguments)
        if self.change:
            self.change(len(self.calls), result)
        return result


def test_real_sqlite_pages_preserve_unicode_and_complete_coverage(library):
    text = "A\n雪🙂e\u0301\r\n" * 2500
    selection = add_source(library, text)
    service = ObservedService(library[1])

    sources = assemble_sources(service, (selection,), (selection.source_id,))

    assert sources == (
        Source(selection.source_id, selection.revision, "Fixture source", text),
    )
    assert len(service.calls) == 4  # Three body pages, then one revision check.
    assert all(name == "library_get_media" for name, _ in service.calls)
    assert all(args["max_chars"] == 8000 for _, args in service.calls)
    assert service.calls[0][1]["cursor"] is None
    assert service.calls[-1][1]["cursor"] is None


@pytest.mark.parametrize(
    "bad_id",
    ["not-an-id", "media:L2V0Yy9wYXNzd2Q", make_public_id("note", "note-id"), 42],
)
def test_invalid_last_requested_id_prevents_every_body_query(library, bad_id):
    selection = add_source(library, "fixture body")
    queries = []
    library[0].get_connection().set_trace_callback(queries.append)

    with pytest.raises(SourceReaderError, match="invalid_scope"):
        assemble_sources(library[1], (selection,), (selection.source_id, bad_id))

    assert queries == []


def test_foreign_or_empty_scope_never_reads_a_body(library):
    selection = add_source(library, "fixture body")
    service = ObservedService(library[1])
    for selections, requested in [
        ((), (selection.source_id,)),
        ((selection,), ()),
        ((selection,), (make_public_id("media", "foreign"),)),
        ((selection,), (selection.source_id, selection.source_id)),
    ]:
        with pytest.raises(SourceReaderError, match="invalid_scope"):
            assemble_sources(service, selections, requested)
    assert service.calls == []


def test_disabled_direct_access_never_reads_a_body(library):
    selection = add_source(library, "fixture body")
    service = ObservedService(library[1])
    with pytest.raises(SourceReaderError, match="direct_access_disabled"):
        assemble_sources(
            service, (selection,), (selection.source_id,), direct_enabled=False
        )
    assert service.calls == []


def test_manifest_conflicting_revisions_fail_before_reads(library):
    selection = add_source(library, "fixture body")
    service = ObservedService(library[1])
    with pytest.raises(SourceReaderError, match="invalid_scope"):
        assemble_sources(
            service,
            (selection, Selection(selection.source_id, "different")),
            (selection.source_id,),
        )
    assert service.calls == []


def test_requested_subset_and_order_are_preserved(library):
    first = add_source(library, "first body", "First")
    second = add_source(library, "second body", "Second")
    third = add_source(library, "third body", "Third")
    sources = assemble_sources(
        library[1], (first, second, third, first), (third.source_id, first.source_id)
    )
    assert [(s.title, s.text) for s in sources] == [
        ("Third", "third body"),
        ("First", "first body"),
    ]


def test_inputs_are_frozen_before_reading(library):
    selection = add_source(library, "fixture body")
    selections = [selection]
    requested = [selection.source_id]

    def clear_caller_lists(count, _):
        if count == 1:
            selections.clear()
            requested.clear()

    service = ObservedService(library[1], before=clear_caller_lists)
    assert assemble_sources(service, selections, requested)[0].text == "fixture body"


def test_seventh_source_is_rejected_before_reads(library):
    selections = tuple(
        Selection(make_public_id("media", f"source-{i}"), "1") for i in range(7)
    )
    service = ObservedService(library[1])
    with pytest.raises(SourceReaderError, match="invalid_scope"):
        assemble_sources(service, selections, tuple(s.source_id for s in selections))
    assert service.calls == []


@pytest.mark.parametrize("text", ["", " \n\t\r\u2003"])
def test_empty_or_whitespace_body_is_a_source_error(library, text):
    selection = add_source(library, text)
    with pytest.raises(SourceReaderError, match="empty_source"):
        assemble_sources(library[1], (selection,), (selection.source_id,))


@pytest.mark.parametrize("change_at", [1, 2, 3])
def test_revision_changes_before_during_or_after_paging_fail(library, change_at):
    selection = add_source(library, "x" * 9000)

    def change_revision(count, _):
        if count == change_at:
            with library[0].transaction() as conn:
                conn.execute("UPDATE Media SET version = version + 1")

    service = ObservedService(library[1], before=change_revision)
    with pytest.raises(SourceReaderError, match="content_changed"):
        assemble_sources(service, (selection,), (selection.source_id,))


def test_first_source_is_rechecked_after_all_sources_are_assembled(library):
    first = add_source(library, "first body", "First")
    second = add_source(library, "second body", "Second")

    def change_first(count, _):
        if count == 2:
            with library[0].transaction() as conn:
                conn.execute(
                    "UPDATE Media SET version = version + 1 WHERE title = ?", ("First",)
                )

    service = ObservedService(library[1], before=change_first)
    with pytest.raises(SourceReaderError, match="content_changed"):
        assemble_sources(service, (first, second), (first.source_id, second.source_id))


def test_missing_or_revoked_source_is_unavailable(library):
    selection = add_source(library, "fixture body")
    with library[0].transaction() as conn:
        conn.execute("UPDATE Media SET deleted = 1, version = version + 1")
    with pytest.raises(SourceReaderError, match="source_unavailable"):
        assemble_sources(library[1], (selection,), (selection.source_id,))


@pytest.mark.parametrize(
    "field,value",
    [
        ("start", 1),
        ("start", False),
        ("end", 7999),
        ("returned_chars", 7999),
        ("total_chars", 7999),
        ("total_chars", True),
        ("has_more", False),
        ("next_cursor", None),
        ("next_cursor", "invalid-cursor"),
        ("text", None),
    ],
)
def test_malformed_page_is_rejected_before_a_continuation(library, field, value):
    selection = add_source(library, "x" * 9000)

    def corrupt(_, result):
        result["content"][field] = value

    service = ObservedService(library[1], change=corrupt)
    with pytest.raises(SourceReaderError, match="invalid_source_response"):
        assemble_sources(service, (selection,), (selection.source_id,))
    assert len(service.calls) == 1


def test_nonprogressing_page_fails_without_another_read(library):
    selection = add_source(library, "x" * 9000)

    def corrupt(_, result):
        result["content"].update(text="", end=0, returned_chars=0)
        result["content"]["next_cursor"] = make_cursor(
            item_id=selection.source_id, revision=selection.revision, offset=0
        )

    service = ObservedService(library[1], change=corrupt)
    with pytest.raises(SourceReaderError, match="non_progressing_source"):
        assemble_sources(service, (selection,), (selection.source_id,))
    assert len(service.calls) == 1


@pytest.mark.parametrize("identity", ["item", "revision", "offset"])
def test_continuation_must_match_source_revision_and_end(library, identity):
    selection = add_source(library, "x" * 9000)

    def corrupt(_, result):
        result["content"]["next_cursor"] = make_cursor(
            item_id=make_public_id("media", "foreign")
            if identity == "item"
            else selection.source_id,
            revision="foreign" if identity == "revision" else selection.revision,
            offset=8001 if identity == "offset" else 8000,
        )

    service = ObservedService(library[1], change=corrupt)
    with pytest.raises(SourceReaderError, match="invalid_source_response"):
        assemble_sources(service, (selection,), (selection.source_id,))
    assert len(service.calls) == 1


def test_multibyte_aggregate_limit_accepts_exact_boundary_and_rejects_next_byte(
    library,
):
    selection = add_source(library, "🙂" * 65536)
    second = add_source(library, "x", "Second")
    assert (
        len(assemble_sources(library[1], (selection,), (selection.source_id,))[0].text)
        == 65536
    )
    with pytest.raises(SourceReaderError, match="source_budget_exceeded"):
        assemble_sources(
            library[1], (selection, second), (selection.source_id, second.source_id)
        )


def test_packet_overlap_preserves_original_unicode_offsets_and_line_breaks():
    text = "🙂" * 3800 + "\n雪e\u0301" * 50 + "end"
    source = Source(make_public_id("media", "one"), "1", "secret title", text)
    packets = pack_sources((source,))
    assert [(p.packet_id, p.start, len(p.text)) for p in packets] == [
        ("p1", 0, 4000),
        ("p2", 3800, 203),
    ]
    assert packets[0].text[-200:] == "\n雪e\u0301" * 50
    assert packets[1].text == "\n雪e\u0301" * 50 + "end"
    assert packets[0].text + packets[1].text[200:] == text
    assert all((p.source_id, p.revision) == (source.source_id, "1") for p in packets)


def test_packets_use_global_deterministic_ids_and_accept_short_sources():
    sources = (
        Source(make_public_id("media", "first"), "1", "One", "OK"),
        Source(make_public_id("media", "second"), "2", "Two", "雪"),
    )
    assert [(p.packet_id, p.start, p.text) for p in pack_sources(sources)] == [
        ("p1", 0, "OK"),
        ("p2", 0, "雪"),
    ]
    assert pack_sources(sources) == pack_sources(sources)


def test_packet_limit_fails_instead_of_dropping_source_tail():
    source_id = make_public_id("media", "large")
    assert len(pack_sources((Source(source_id, "1", "Large", "x" * 243400),))) == 64
    with pytest.raises(SourceReaderError, match="packet_budget_exceeded"):
        pack_sources((Source(source_id, "1", "Large", "x" * 243401),))


def test_source_records_are_immutable_and_repr_omits_sensitive_text():
    source_id = make_public_id("media", "one")
    selection = Selection(source_id, "1")
    source = Source(source_id, "1", "TITLE_CANARY", "TEXT_CANARY")
    packet = Packet("p1", source_id, "1", 0, "TEXT_CANARY")
    for record in (selection, source, packet):
        with pytest.raises(FrozenInstanceError):
            record.revision = "2"
        assert "CANARY" not in repr(record)


def test_error_codes_cannot_echo_arbitrary_payloads():
    error = SourceReaderError("PRIVATE_CANARY " * 1000)
    assert len(error.code) <= 64
    assert "PRIVATE_CANARY" not in str(error)
    assert str(error) == error.code
