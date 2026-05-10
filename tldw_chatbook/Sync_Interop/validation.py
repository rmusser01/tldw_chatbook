"""Validation helpers for Sync v2 transport boundaries."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from tldw_chatbook.tldw_api import SyncV2Envelope


def validate_pulled_dataset_identity(
    *,
    dataset_id: str,
    response_dataset_id: Any,
    envelopes: Iterable[SyncV2Envelope],
) -> None:
    """Reject pulled Sync v2 data that does not belong to the requested dataset."""

    if response_dataset_id is not None and str(response_dataset_id) != dataset_id:
        raise ValueError("pulled Sync v2 batch dataset_id must match requested dataset_id")
    for envelope in envelopes:
        if envelope.dataset_id != dataset_id:
            raise ValueError("pulled Sync v2 envelope dataset_id must match requested dataset_id")
