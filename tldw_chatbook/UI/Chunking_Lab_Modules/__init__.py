"""Focused presentation components for the local Chunking Lab."""

from textual.message import Message


class ChunkingTemplatesChanged(Message):
    """Local catalog invalidation with record identity/version only."""

    def __init__(self, record_id: int, version: int):
        super().__init__()
        self.record_id, self.version = record_id, version
