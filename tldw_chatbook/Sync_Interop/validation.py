"""Validation helpers for Sync v2 transport boundaries."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from tldw_chatbook.tldw_api import SyncV2Envelope


def validate_response_dataset_identity(
    *,
    dataset_id: str,
    response_dataset_id: Any,
    response_label: str,
) -> None:
    """Reject server responses whose dataset identity does not match the request."""

    if response_dataset_id is not None and str(response_dataset_id) != dataset_id:
        raise ValueError(f"Sync v2 {response_label} dataset_id must match requested dataset_id")


def validate_pulled_response_scope(
    *,
    dataset_id: str,
    response_dataset_id: Any,
    envelopes: Iterable[SyncV2Envelope],
    domains: Iterable[str] | None = None,
) -> None:
    """Reject pulled Sync v2 data outside the requested dataset or domains."""

    validate_response_dataset_identity(
        dataset_id=dataset_id,
        response_dataset_id=response_dataset_id,
        response_label="pull response",
    )
    domain_set = {str(domain) for domain in domains or []}
    for envelope in envelopes:
        if envelope.dataset_id != dataset_id:
            raise ValueError("pulled Sync v2 envelope dataset_id must match requested dataset_id")
        if domain_set and str(envelope.domain) not in domain_set:
            raise ValueError("pulled Sync v2 envelope domain must be included in requested domains")


def validate_push_response_scope(
    *,
    dataset_id: str,
    response_dataset_id: Any,
    submitted_client_envelope_ids: Iterable[str],
    accepted: Iterable[dict[str, Any]],
    rejected: Iterable[dict[str, Any]],
    conflicts: Iterable[dict[str, Any]],
) -> None:
    """Reject Sync v2 push responses that do not correspond to the submitted batch."""

    validate_response_dataset_identity(
        dataset_id=dataset_id,
        response_dataset_id=response_dataset_id,
        response_label="push response",
    )
    submitted_ids = {str(client_envelope_id) for client_envelope_id in submitted_client_envelope_ids}
    seen_ids: set[str] = set()
    for item in [*accepted, *rejected, *conflicts]:
        client_envelope_id = item.get("client_envelope_id")
        if not client_envelope_id or str(client_envelope_id) not in submitted_ids:
            raise ValueError("Sync v2 push response referenced unknown client_envelope_id")
        response_id = str(client_envelope_id)
        if response_id in seen_ids:
            raise ValueError("Sync v2 push response contained duplicate client_envelope_id")
        seen_ids.add(response_id)


def validate_outgoing_envelope_scope(
    *,
    dataset_id: str,
    device_id: str,
    envelopes: Iterable[SyncV2Envelope],
    domains: Iterable[str],
) -> None:
    """Reject outgoing Sync v2 envelopes outside the active profile scope."""

    domain_set = {str(domain) for domain in domains}
    for envelope in envelopes:
        if envelope.dataset_id != dataset_id:
            raise ValueError("outgoing Sync v2 envelope dataset_id must match profile dataset_id")
        if envelope.device_id != device_id:
            raise ValueError("outgoing Sync v2 envelope device_id must match profile device_id")
        if domain_set and str(envelope.domain) not in domain_set:
            raise ValueError("outgoing Sync v2 envelope domain must be included in requested domains")
