# tldw_chatbook/RAG_Search/simplified/collection_fingerprint.py
"""Deterministic, versioned fingerprint of index-determining RAG config.

The fingerprint keys a vector collection to the config that built it: two
configs that differ only in query-time settings share a collection; two that
differ in embedding model, any chunking field, or distance metric get
separate collections. See
Docs/superpowers/specs/2026-07-21-rag-index-isolation-design.md.

This module is PURE (no IO). The fingerprint is a persistent contract:
changing FINGERPRINT_VERSION or the input set re-points every collection, so
treat any such change as a migration.
"""
from __future__ import annotations

import hashlib
import json
import re
import string
from typing import Any, List, Tuple

from .config import RAGConfig

#: Bump ONLY with a migration mapping old->new collection names.
FINGERPRINT_VERSION: int = 1

_MAX_CHROMA_NAME = 63
_MIN_CHROMA_NAME = 3
_FP_LEN = 12  # hex chars of the digest kept

#: Chroma collection names must be ASCII-only [a-zA-Z0-9._-].
_ALLOWED_NAME_CHARS = set(string.ascii_letters + string.digits + "._-")
_RUN_OF_DOTS = re.compile(r"\.{2,}")


def _index_fields(config: RAGConfig) -> List[Tuple[str, Any]]:
    """Ordered (name, value) pairs of every index-determining field.

    Values are normalized so that equal logical configs hash identically
    regardless of source type (TOML strings vs ints/bools).
    """
    e = config.embedding
    c = config.chunking
    v = config.vector_store
    raw: List[Tuple[str, Any]] = [
        ("embedding.model", e.model),
        ("embedding.max_length", e.max_length),
        ("chunking.chunk_size", c.chunk_size),
        ("chunking.chunk_overlap", c.chunk_overlap),
        ("chunking.chunking_method", c.chunking_method),
        ("chunking.min_chunk_size", c.min_chunk_size),
        ("chunking.max_chunk_size", c.max_chunk_size),
        ("chunking.enable_parent_retrieval", c.enable_parent_retrieval),
        ("chunking.parent_size_multiplier", c.parent_size_multiplier),
        ("chunking.preserve_structure", c.preserve_structure),
        ("chunking.clean_artifacts", c.clean_artifacts),
        ("chunking.preserve_tables", c.preserve_tables),
        ("vector_store.distance_metric", v.distance_metric),
    ]
    return [(k, _normalize(val)) for k, val in raw]


def _normalize(value: Any) -> Any:
    """Coerce a config value to a canonical, hash-stable form."""
    if isinstance(value, bool):
        return bool(value)               # keep bools distinct from ints
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        # ints-as-floats (400.0) normalize to int; real floats keep precision
        return int(value) if float(value).is_integer() else float(value)
    if value is None:
        return None
    s = str(value).strip()
    # numeric strings ("400") normalize to int so they match int configs
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    return s


def fingerprint_collection(config: RAGConfig) -> str:
    """Return the 12-hex-char fingerprint of ``config``'s index-determining fields."""
    payload = {
        "version": FINGERPRINT_VERSION,
        "fields": _index_fields(config),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:_FP_LEN]


def _safe_base(name: str, reserved: int) -> str:
    """A Chroma-safe base name leaving ``reserved`` chars for the suffix."""
    # ASCII-only [a-zA-Z0-9._-]; anything else (including non-ASCII
    # "alnum" chars like 'é' or '北') is replaced so Chroma never rejects it.
    cleaned = "".join(ch if ch in _ALLOWED_NAME_CHARS else "-" for ch in str(name))
    # Collapse runs of dots BEFORE stripping/truncating so slicing can never
    # leave (or re-create) a ".." anywhere in the final name.
    cleaned = _RUN_OF_DOTS.sub(".", cleaned)
    cleaned = cleaned.strip("._-") or "rag"
    budget = max(1, _MAX_CHROMA_NAME - reserved)
    cleaned = cleaned[:budget].strip("._-") or "rag"
    return cleaned


def fingerprinted_collection_name(config: RAGConfig) -> str:
    """Return ``"{base}__{fingerprint}"`` — always a valid Chroma name."""
    fp = fingerprint_collection(config)
    base = _safe_base(config.collection_name, reserved=len(fp) + 2)  # "__" + fp
    name = f"{base}__{fp}"
    if len(name) < _MIN_CHROMA_NAME:  # pragma: no cover - fp alone is 12 chars
        name = f"rag__{fp}"
    return name


def collection_provenance(
    config: RAGConfig, *, source: str = "built", verified: bool = True
) -> dict:
    """Metadata stamped on a collection at creation, read back by the UI (SP3)."""
    return {
        "fp_version": FINGERPRINT_VERSION,
        "fp": fingerprint_collection(config),
        "embedding_model": str(config.embedding.model),
        # Reuse the same tolerant coercion fingerprinting uses so provenance
        # never raises on inputs that fingerprint_collection() accepts.
        "chunk_size": _normalize(config.chunking.chunk_size),
        "chunk_overlap": _normalize(config.chunking.chunk_overlap),
        "chunking_method": str(config.chunking.chunking_method),
        "distance_metric": str(config.vector_store.distance_metric),
        "source": source,
        "verified": bool(verified),
    }
