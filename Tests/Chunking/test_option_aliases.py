from __future__ import annotations

from tldw_chatbook.Chunking.engine.chunker import Chunker
from tldw_chatbook.Chunking.engine.base import ChunkMetadata, ChunkResult


class _CaptureStrategy:
    def __init__(self) -> None:
        self.language = "en"
        self.last = None

    def chunk(self, text: str, max_size: int, overlap: int = 0, **options):
        self.last = {"max_size": max_size, "overlap": overlap, "options": options}
        return ["ok"]

    def chunk_with_metadata(self, text: str, max_size: int, overlap: int = 0, **options):
        self.last = {"max_size": max_size, "overlap": overlap, "options": options}
        md = ChunkMetadata(index=0, start_char=0, end_char=min(1, len(text)), word_count=1)
        return [ChunkResult(text=text[:1], metadata=md)]


def test_chunker_maps_semantic_aliases_in_chunk_text():
    chunker = Chunker()
    capture = _CaptureStrategy()
    chunker._strategies["semantic"] = capture

    chunker.chunk_text(
        "alpha beta gamma",
        method="semantic",
        max_size=3,
        overlap=99,
        semantic_overlap_sentences=2,
        semantic_similarity_threshold=0.9,
    )

    assert capture.last["overlap"] == 2
    assert capture.last["options"].get("similarity_threshold") == 0.9


def test_chunker_maps_semantic_aliases_in_metadata_path():
    chunker = Chunker()
    capture = _CaptureStrategy()
    chunker._strategies["semantic"] = capture

    chunker.chunk_text_with_metadata(
        "alpha beta gamma",
        method="semantic",
        max_size=3,
        overlap=99,
        semantic_overlap_sentences=1,
    )

    assert capture.last["overlap"] == 1


def test_chunker_maps_json_chunkable_key_alias():
    chunker = Chunker()
    capture = _CaptureStrategy()
    chunker._strategies["json"] = capture

    chunker.chunk_text(
        "{}",
        method="json",
        max_size=1,
        overlap=0,
        json_chunkable_data_key="items",
    )

    assert capture.last["options"].get("chunkable_key") == "items"


def test_chunker_maps_proposition_min_length_alias():
    chunker = Chunker()
    capture = _CaptureStrategy()
    chunker._strategies["propositions"] = capture

    chunker.chunk_text(
        "Hello world.",
        method="propositions",
        max_size=1,
        overlap=0,
        proposition_min_proposition_length=7,
    )

    assert capture.last["options"].get("min_proposition_length") == 7
