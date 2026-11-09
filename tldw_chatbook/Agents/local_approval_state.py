"""Private local approval stamp lifetime and best-effort remembering (ADR235/300)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


@contextmanager
def local_stamp_scope(provider: Any, run_id: str) -> Iterator[None]:
    """Snapshot/restore ``run_id``'s stamps around a nested sub-agent run.

    Clears ``run_id``'s slice on entry -- a deliberate divergence from
    a pure snapshot, preserved from the pre-Task-5 version: a nested
    run that somehow reused this run id would start stamp-less and
    re-check permissions itself. The run's stamps are restored on exit,
    even on exception; other runs' slices are untouched in both
    directions.

    Per-run keying is the real protection now (a child stamps under
    its OWN run id and never sees this one's), and it is the only one
    that holds once children run concurrently -- snapshot/restore is
    sound only for a strictly nested inline child.

    Args:
        run_id: The run whose slice is cleared, then restored.
    """
    from .approval_observation import provider_observation_scope

    with provider_observation_scope(provider, run_id, clear=True):
        with provider._stamps_lock:
            provider._index_call_stamps = {
                key: value
                for key, value in provider._index_call_stamps.items()
                if key[0] != run_id
            }
            saved = {
                key: value
                for key, value in provider._stamps.items()
                if key[0] == run_id
            }
            provider._stamps = {
                key: value
                for key, value in provider._stamps.items()
                if key[0] != run_id
            }
        try:
            yield
        finally:
            with provider._stamps_lock:
                provider._index_call_stamps = {
                    key: value
                    for key, value in provider._index_call_stamps.items()
                    if key[0] != run_id
                }
                provider._stamps = {
                    key: value
                    for key, value in provider._stamps.items()
                    if key[0] != run_id
                }
                provider._stamps.update(saved)


def persist_local_approval(provider: Any, hub: Any, decision: str, *, log: Any) -> None:
    """Observe a real writer while preserving best-effort current-call execution."""
    from .approval_observation import (
        approval_contexts_scope,
        provider_approval_contexts,
        publish_grant_application,
    )

    with approval_contexts_scope(provider_approval_contexts(provider, hub.name)):
        if provider._persist_approval is None:
            publish_grant_application("not_applied", error_code="writer_unavailable")
            return
        try:
            provider._persist_approval(hub, decision)
        except Exception as exc:  # noqa: BLE001 - current call survives optional grant write failure
            publish_grant_application("failed", error_code="writer_failed")
            log.warning(
                f"LocalToolProvider: persist_approval ({decision}) failed for {hub.name}: {exc}"
            )
