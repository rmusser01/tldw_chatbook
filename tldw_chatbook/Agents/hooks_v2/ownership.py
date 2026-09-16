"""Host process custody, independent of plugin storage and authority.

Plugin composition adapts the existing exact-root runtime owner here. Its
reserve_launch must acquire all root custody (including dirty publication)
before returning. Publication receives metadata, never claims terminal proof.
False settlement must retain ownership. The executor holds actual transports
separately; an adapter may capture stronger platform identity at publication.
"""

from __future__ import annotations

from typing import Protocol
from uuid import uuid4

from .models import HookEvent


class HookProcessOwner(Protocol):
    def reserve_launch(self, event: HookEvent) -> str: ...
    def publish_process(self, token: str, provenance: dict) -> None: ...
    def settle_process(self, token: str, confirmed: bool) -> None: ...


class HostProcessOwner:
    """In-memory custody for standalone, application-owned command hooks."""

    def __init__(self) -> None:
        self.records: dict[str, dict] = {}

    def reserve_launch(self, event: HookEvent) -> str:
        token = uuid4().hex
        self.records[token] = {"event_id": event.event_id, "state": "pending"}
        return token

    def publish_process(self, token: str, provenance: dict) -> None:
        self.records[token].update(provenance, state="published")

    def settle_process(self, token: str, confirmed: bool) -> None:
        if confirmed:
            self.records.pop(token, None)
        else:
            self.records[token]["state"] = "unresolved"
