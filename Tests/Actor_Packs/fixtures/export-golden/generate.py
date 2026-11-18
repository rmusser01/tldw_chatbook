"""Regenerate the minimal Actor Pack export goldens without the product writer.

Stdlib and the fixture oracle in ``Tests/Actor_Packs/conftest.py`` only:
nothing from ``tldw_chatbook`` is imported, so the goldens stay independent of
``write_actor_pack_archive`` (TASK-41112). The ZIP settings below are the V1
byte contract written out by hand, not read from ``Actor_Packs.contracts``.

``producer.version`` is ``GOLDEN_PRODUCER_VERSION``, not the app version: the
golden test pins the writer's version to it, so a release bump does not move
these bytes.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Tests.Actor_Packs.conftest import (  # noqa: E402
    GOLDEN_PRODUCER_VERSION,
    PNG_1X1,
    PORTABLE_UUID,
    canonical_json,
    file_descriptor,
    with_content_digest,
)


def golden_archive(actor_kind: str) -> bytes:
    files = {
        "actor/actor.json": canonical_json(
            {
                "schema": "tldw.actor/v1",
                "actor_kind": actor_kind,
                "portable_uuid": PORTABLE_UUID,
                "data": {"name": "Golden"},
            }
        ),
        "actor/portrait.png": PNG_1X1,
    }
    root = with_content_digest(
        {
            "schema": "tldw.actor-pack/v1",
            "actor": {
                "kind": actor_kind,
                "portable_uuid": PORTABLE_UUID,
                "payload": "actor/actor.json",
                "portrait": "actor/portrait.png",
            },
            "sections": [],
            "producer": {"name": "tldw_chatbook", "version": GOLDEN_PRODUCER_VERSION},
            "license": {"value": "unspecified"},
            "provenance": {"source": "local"},
            "required_features": [],
            "files": [file_descriptor(path, data) for path, data in sorted(files.items())],
        }
    )
    members = {"actor-pack.json": canonical_json(root), **dict(sorted(files.items()))}
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path, data in members.items():
            info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.flag_bits = 0
            info.external_attr = 0o100644 << 16
            with archive.open(info, mode="w", force_zip64=True) as member:
                member.write(data)
    return output.getvalue()


def main() -> None:
    for actor_kind in ("character", "persona"):
        (Path(__file__).parent / f"minimal-{actor_kind}.tldw-actor-pack").write_bytes(
            golden_archive(actor_kind)
        )


if __name__ == "__main__":
    main()
