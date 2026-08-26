"""Every field the Library sends must be one the server actually declares.

task-3309. The ingest-jobs endpoint binds its form fields with explicit
``Form(...)`` declarations (``get_add_media_form``) and never reads
``request.form()``, so a multipart field it does not declare is discarded
without an error and the submission still answers 200. That failure mode is
invisible from the client: the job succeeds, and the settings the user chose
simply did not happen.

Eighteen forwarded option names were in that state when this test was written.
The point of the test is that the next one fails here instead of shipping.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tldw_chatbook.Library.ingest_capabilities import (
    get_capabilities,
    list_type_groups,
)
from tldw_chatbook.Library.server_ingest_request import (
    SERVER_FIELD_ALIASES,
    SERVER_UNSUPPORTED_OPTIONS,
    build_server_ingest_kwargs,
    server_unsupported_options,
)

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "server_ingest_jobs_form_fields.json"
)

#: Kwargs consumed by ``submit_ingest_jobs``' own signature rather than being
#: forwarded as form fields, plus the transport-level ones.
_NOT_FORM_FIELDS = frozenset({"file_paths"})


def _declared_fields() -> frozenset[str]:
    payload = json.loads(_FIXTURE.read_text())
    return frozenset(payload["declared_form_fields"])


def _sample_for(name: str) -> object:
    """A truthy value so the option survives to the request."""
    if name in {"chunk_size", "chunk_overlap", "max_pages", "max_depth"}:
        return 7
    return "x"


def test_fixture_records_where_it_came_from():
    payload = json.loads(_FIXTURE.read_text())
    provenance = payload["_provenance"]
    assert provenance["endpoint"] == "POST /api/v1/media/ingest/jobs"
    assert provenance["captured_from"].startswith("http")
    assert payload["declared_form_fields"], "fixture must list the server's fields"


def test_fixture_captures_each_shared_generic_server_field():
    """The captured endpoint contract accepts every shared Server projection."""
    assert {
        "overwrite_existing",
        "custom_prompt",
        "system_prompt",
        "generate_embeddings",
        "keep_original_file",
    } <= _declared_fields()


@pytest.mark.parametrize(
    "source",
    [
        "/tmp/sample.pdf",
        "/tmp/sample.mp3",
        "/tmp/sample.epub",
        "/tmp/sample.txt",
    ],
)
def test_every_forwarded_field_is_one_the_server_declares(source):
    """No submission may carry a field the server will silently discard."""
    declared = _declared_fields()
    options = {
        name: {
            field_name: _sample_for(field_name)
            for field_name in get_capabilities(name).field_names
        }
        for name in list_type_groups()
    }

    kwargs = build_server_ingest_kwargs(source, options=options)

    undeclared = sorted(
        name
        for name in kwargs
        if name not in declared and name not in _NOT_FORM_FIELDS
    )
    assert not undeclared, (
        f"{source} would send form fields the server does not declare, so they "
        f"are dropped in silence: {undeclared}. Either add an entry to "
        f"SERVER_FIELD_ALIASES (if the server spells it differently) or to "
        f"SERVER_UNSUPPORTED_OPTIONS (if it has no equivalent)."
    )


def test_aliases_all_point_at_fields_the_server_declares():
    """A rename is only correct if the target actually exists."""
    declared = _declared_fields()
    broken = sorted(
        f"{client} -> {server}"
        for client, server in SERVER_FIELD_ALIASES.items()
        if server not in declared
    )
    assert not broken, f"aliases point at undeclared server fields: {broken}"


def test_alias_and_unsupported_sets_do_not_overlap():
    """A field cannot both translate and have no equivalent."""
    overlap = sorted(set(SERVER_FIELD_ALIASES) & set(SERVER_UNSUPPORTED_OPTIONS))
    assert not overlap, f"listed as both aliased and unsupported: {overlap}"


def test_dropped_options_are_reported_rather_than_lost():
    """Silently discarding is the bug; dropping and saying so is the fix."""
    options = {"audio_video": {"translate_to_english": True, "diarization": True}}

    kwargs = build_server_ingest_kwargs("/tmp/sample.mp3", options=options)

    assert "translate_to_english" not in kwargs
    assert kwargs["diarize"] is True
    lost = server_unsupported_options("/tmp/sample.mp3", options)
    assert [name for name, _reason in lost] == ["translate_to_english"]
    # The reason matters: this one is a capability the server HAS and its API
    # does not expose, which is a different thing from "the server cannot".
    assert "does not expose it" in dict(lost)["translate_to_english"]


def test_an_unset_unsupported_option_is_not_reported_as_lost():
    """Nothing is lost when the user never asked for it."""
    options = {"audio_video": {"translate_to_english": False, "cookies_file": ""}}

    assert server_unsupported_options("/tmp/sample.mp3", options) == ()


@pytest.mark.parametrize(
    "source, expected",
    [
        ("/tmp/sample.png", "no handler for 'image'"),
        ("https://example.com/article", "web-clipper endpoint"),
    ],
)
def test_sources_with_no_ingest_jobs_equivalent_are_refused(source, expected):
    """task-3309 AC#2: record what server mode deliberately excludes.

    Images have no server media type at all, and a web page is clipped through
    ``/api/v1/media/ingest-web-content`` rather than the ingest-jobs API -- so
    neither reaches the request builder this module's field contract covers.
    Pinned here so the exclusions are stated rather than discovered.
    """
    from tldw_chatbook.Library.server_ingest_request import ServerIngestUnsupported

    with pytest.raises(ServerIngestUnsupported) as excinfo:
        build_server_ingest_kwargs(source, options={})

    assert expected in str(excinfo.value)


def test_nothing_the_service_adds_reaches_the_wire_undeclared():
    """task-3309: the options loop is not the only way a field reaches the wire.

    ``ServerMediaReadingService.submit_ingest_jobs`` names several fields in its
    own signature and puts them on the request regardless of what the canvas
    set, bypassing ``build_server_ingest_kwargs``' translation entirely. This
    asserts against the REQUEST the service actually builds rather than its
    signature -- a parameter may legitimately exist without being forwarded.
    That distinction is the fix for ``force_regenerate_embeddings``, which was
    sent on every submission and declared by the server on none of them.
    """
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import patch

    from tldw_chatbook.Media.server_media_reading_service import (
        ServerMediaReadingService,
    )

    declared = _declared_fields()
    captured: dict[str, object] = {}

    async def fake_submit(request_data, file_paths=None):
        captured["request"] = request_data
        return SimpleNamespace(batch_id="b", jobs=[], errors=[])

    service = ServerMediaReadingService.__new__(ServerMediaReadingService)
    with (
        patch.object(ServerMediaReadingService, "_enforce", lambda self, _a: None),
        patch.object(
            ServerMediaReadingService,
            "_require_client",
            lambda self: SimpleNamespace(submit_media_ingest_jobs=fake_submit),
        ),
    ):
        asyncio.run(
            service.submit_ingest_jobs(
                media_type="audio",
                file_paths=["/tmp/sample.mp3"],
                generate_embeddings=True,
                force_regenerate_embeddings=True,
            )
        )

    sent = captured["request"].model_dump(exclude_none=True)
    undeclared = sorted(name for name in sent if name not in declared)
    assert not undeclared, (
        "the service puts fields on the request that the server does not "
        f"declare, so they are dropped in silence: {undeclared}"
    )
    # The declared sibling must still travel -- dropping the undeclared one
    # must not take the honoured one with it.
    assert sent["generate_embeddings"] is True
