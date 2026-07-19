"""Local Sync v2 adapter for workspace relationships."""

from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from tldw_chatbook.tldw_api import SyncV2Envelope

from ._helpers import call_if_present


class WorkspacesSyncAdapter:
    """Apply workspace source link and unlink envelopes."""

    def apply(
        self,
        envelope: SyncV2Envelope,
        *,
        dataset_key: bytes,
        local_store: Any,
        record_conflict: Callable[..., dict[str, Any]],
    ) -> dict[str, Any]:
        del dataset_key, record_conflict
        workspace_id = str(envelope.payload_clear.get("workspace_id") or "")
        source_id = str(envelope.payload_clear.get("source_id") or "")
        if not workspace_id or not source_id:
            return {"status": "rejected", "error_code": "missing_workspace_source_ref"}
        if envelope.operation == "link":
            call_if_present(local_store, "link_workspace_source", workspace_id, source_id)
            return {"status": "applied"}
        if envelope.operation == "unlink":
            call_if_present(local_store, "unlink_workspace_source", workspace_id, source_id)
            return {"status": "applied"}
        return {"status": "rejected", "error_code": "unsupported_workspace_operation"}
