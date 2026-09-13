"""Independent fixtures for the Actor Pack V1 contract."""

from __future__ import annotations

import copy
import base64
import hashlib
import json
from collections.abc import Mapping

import pytest


PORTABLE_UUID = "123e4567-e89b-42d3-a456-426614174000"
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def canonical_json(value: object) -> bytes:
    """The fixture oracle; deliberately independent of production code."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def file_descriptor(path: str, data: bytes) -> dict[str, object]:
    return {
        "path": path,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def with_content_digest(manifest: Mapping[str, object]) -> dict[str, object]:
    materialized = copy.deepcopy(dict(manifest))
    materialized.pop("content_digest", None)
    materialized["content_digest"] = hashlib.sha256(
        canonical_json(materialized)
    ).hexdigest()
    return materialized


@pytest.fixture
def minimal_character_files() -> dict[str, bytes]:
    actor = canonical_json(
        {
            "schema": "tldw.actor/v1",
            "actor_kind": "character",
            "portable_uuid": PORTABLE_UUID,
            "data": {"name": "Guide"},
        }
    )
    return {
        "actor/actor.json": actor,
        "actor/portrait.png": PNG_1X1,
    }


@pytest.fixture
def minimal_character_manifest(
    minimal_character_files: Mapping[str, bytes],
) -> dict[str, object]:
    manifest = {
        "schema": "tldw.actor-pack/v1",
        "actor": {
            "kind": "character",
            "portable_uuid": PORTABLE_UUID,
            "payload": "actor/actor.json",
            "portrait": "actor/portrait.png",
        },
        "sections": [],
        "producer": {"name": "tldw_chatbook", "version": "0.1.8"},
        "license": {"value": "unspecified"},
        "provenance": {"source": "local"},
        "required_features": [],
        "files": [
            file_descriptor(path, data)
            for path, data in sorted(minimal_character_files.items())
        ],
    }
    return with_content_digest(manifest)
