"""Library export execution: payload building + service call ordering (F4 Task 3).

Covers ``LibraryScreen``'s two pure/static execution helpers directly --
neither needs a mounted screen, a real DB, or a real thread:

- ``_build_library_export_payload``: the spec-critical
  ``include_media`` invariant (F4 plan Global Constraints: "``include_media
  =True`` is ALWAYS passed when media is in scope").
- ``_run_library_export_via_service``: the spec-critical "zip first,
  registry record only on success" ordering, exercised against a fake
  ``local_chatbook_service`` double that records call order.

Both are exposed as ``@staticmethod``s on ``LibraryScreen`` specifically so
these invariants are unit-testable without booting a Textual pilot, a real
worker thread, or ``asyncio.run`` re-entrancy concerns.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Chatbooks.chatbook_models import ContentType
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen


# --- _build_library_export_payload: include_media invariant -----------------


def test_payload_include_media_true_when_media_in_selections():
    payload = LibraryScreen._build_library_export_payload(
        name="Export",
        description="",
        selections={ContentType.MEDIA: ["1", "2"]},
        destination="/tmp/out.zip",
        media_quality="thumbnail",
    )

    assert payload["include_media"] is True
    assert payload["content_selections"] == {ContentType.MEDIA: ["1", "2"]}
    assert payload["output_path"] == "/tmp/out.zip"
    assert payload["media_quality"] == "thumbnail"


def test_payload_include_media_false_when_scope_has_no_media_selections():
    """A conversations-only scope never resolves a MEDIA key at all."""
    payload = LibraryScreen._build_library_export_payload(
        name="Export",
        description="",
        selections={ContentType.CONVERSATION: ["c-1"]},
        destination="/tmp/out.zip",
        media_quality="thumbnail",
    )

    assert payload["include_media"] is False


def test_payload_include_media_false_for_everything_scope_with_zero_media():
    """An "everything" export of a library with no media at all still
    resolves correctly: ``resolve_export_selections`` (Task 1) omits the
    ``ContentType.MEDIA`` key entirely rather than including it with an
    empty list, so ``include_media`` must come out ``False`` here too --
    the spec-critical invariant keys off membership, not scope kind."""
    payload = LibraryScreen._build_library_export_payload(
        name="Export",
        description="",
        selections={
            ContentType.CONVERSATION: ["c-1"],
            ContentType.NOTE: ["n-1"],
        },
        destination="/tmp/out.zip",
        media_quality="thumbnail",
    )

    assert payload["include_media"] is False


def test_payload_include_media_true_combined_with_other_content_types():
    payload = LibraryScreen._build_library_export_payload(
        name="Export",
        description="",
        selections={
            ContentType.MEDIA: ["m-1"],
            ContentType.CONVERSATION: ["c-1"],
            ContentType.NOTE: ["n-1"],
        },
        destination="/tmp/out.zip",
        media_quality="original",
    )

    assert payload["include_media"] is True


# --- _run_library_export_via_service: zip-first, registry-only-on-success ---


class _FakeExportService:
    """Records call order/arguments; async-signature/sync-body like the real service."""

    def __init__(self, *, export_result=None, create_error=None):
        self.calls: list[str] = []
        self.export_payloads: list[dict] = []
        self.create_kwargs: list[dict] = []
        self._export_result = export_result if export_result is not None else {
            "success": True,
            "message": "",
            "path": "/tmp/out.zip",
            "dependency_info": {},
            "name": "Export",
        }
        self._create_error = create_error

    async def export_chatbook(self, request_data):
        self.calls.append("export_chatbook")
        self.export_payloads.append(dict(request_data))
        return dict(self._export_result)

    async def create_chatbook(self, **kwargs):
        self.calls.append("create_chatbook")
        self.create_kwargs.append(kwargs)
        if self._create_error is not None:
            raise self._create_error
        return {"chatbook_id": 1, **kwargs}


def _payload(**overrides):
    base = {
        "name": "Export",
        "description": "",
        "content_selections": {ContentType.MEDIA: ["1"]},
        "output_path": "/tmp/out.zip",
        "media_quality": "thumbnail",
        "include_media": True,
    }
    base.update(overrides)
    return base


def test_export_via_service_calls_export_then_create_in_order_on_success():
    service = _FakeExportService(
        export_result={
            "success": True,
            "message": "ok",
            "path": "/tmp/out.zip",
            "dependency_info": {"auto_included": [1, 2]},
        }
    )

    outcome = LibraryScreen._run_library_export_via_service(
        service, _payload(), name="Export", description="desc"
    )

    assert service.calls == ["export_chatbook", "create_chatbook"]
    assert outcome == {
        "success": True,
        "message": "ok",
        "path": "/tmp/out.zip",
        "dependency_info": {"auto_included": [1, 2]},
        "registry_recorded": True,
    }
    assert service.create_kwargs[0]["file_path"] == "/tmp/out.zip"
    assert service.create_kwargs[0]["tags"] == ["library-export"]


def test_export_via_service_never_calls_create_chatbook_when_export_fails():
    service = _FakeExportService(
        export_result={
            "success": False,
            "message": "Destination is not writable.",
            "path": "",
            "dependency_info": {},
        }
    )

    outcome = LibraryScreen._run_library_export_via_service(
        service, _payload(), name="Export", description="desc"
    )

    assert service.calls == ["export_chatbook"]  # create_chatbook never ran
    assert outcome["success"] is False
    assert outcome["message"] == "Destination is not writable."
    assert outcome["registry_recorded"] is False


def test_export_via_service_reports_success_even_when_registry_recording_raises():
    """The zip succeeded -- a registry-recording failure afterward must not
    flip the overall outcome to failure (the artifact genuinely exists on
    disk); only ``registry_recorded`` reflects the bookkeeping miss."""
    service = _FakeExportService(create_error=RuntimeError("disk full"))

    outcome = LibraryScreen._run_library_export_via_service(
        service, _payload(), name="Export", description="desc"
    )

    assert service.calls == ["export_chatbook", "create_chatbook"]
    assert outcome["success"] is True
    assert outcome["registry_recorded"] is False


def test_export_via_service_wraps_export_chatbook_exception_as_failure():
    class _RaisingService:
        async def export_chatbook(self, request_data):
            raise RuntimeError("boom")

    outcome = LibraryScreen._run_library_export_via_service(
        _RaisingService(), _payload(), name="Export", description="desc"
    )

    assert outcome["success"] is False
    assert "boom" in outcome["message"]
    assert outcome["registry_recorded"] is False


# --- _build_library_export_success_message: task-158 counts surfacing -------


def test_success_message_includes_creators_detail_stripped_of_redundant_prefix():
    """task-158: the creator's own ``outcome["message"]`` (e.g. missing-
    dependency warnings) was previously discarded entirely -- only the
    bare path reached the notification. Its redundant "Chatbook created
    successfully at <path>" prefix (the path is already the primary
    notify line) must be stripped, leaving just the detail."""
    message = LibraryScreen._build_library_export_success_message(
        "/tmp/out.zip",
        {"auto_included": [1, 2, 3]},
        "Chatbook created successfully at /tmp/out.zip. Warning: 2 character "
        "dependencies are missing",
    )

    assert message == (
        "Exported chatbook to /tmp/out.zip: Warning: 2 character "
        "dependencies are missing (3 characters auto-included)"
    )


def test_success_message_keeps_unrecognized_creator_message_verbatim():
    """A creator message that doesn't match the known redundant prefix
    (e.g. a different service implementation) is kept as-is rather than
    guessed at or silently dropped."""
    message = LibraryScreen._build_library_export_success_message(
        "/tmp/out.zip", {}, "ok"
    )

    assert message == "Exported chatbook to /tmp/out.zip: ok"


def test_success_message_omits_detail_segment_when_creator_message_is_empty():
    message = LibraryScreen._build_library_export_success_message(
        "/tmp/out.zip", {"auto_included": [1]}, ""
    )

    assert message == "Exported chatbook to /tmp/out.zip (1 characters auto-included)"
