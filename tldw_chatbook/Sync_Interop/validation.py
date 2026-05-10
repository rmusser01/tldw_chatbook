"""Validation helpers for Sync v2 transport boundaries."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from tldw_chatbook.tldw_api import SyncV2Envelope


def validate_pulled_response_scope(
    *,
    dataset_id: str,
    response_dataset_id: Any,
    envelopes: Iterable[SyncV2Envelope],
    domains: Iterable[str] | None = None,
) -> None:
    """Reject pulled Sync v2 data outside the requested dataset or domains."""

    if response_dataset_id is not None and str(response_dataset_id) != dataset_id:
        raise ValueError("pulled Sync v2 batch dataset_id must match requested dataset_id")
    domain_set = {str(domain) for domain in domains or []}
    for envelope in envelopes:
        if envelope.dataset_id != dataset_id:
            raise ValueError("pulled Sync v2 envelope dataset_id must match requested dataset_id")
        if domain_set and str(envelope.domain) not in domain_set:
            raise ValueError("pulled Sync v2 envelope domain must be included in requested domains")
