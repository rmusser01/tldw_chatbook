"""Regenerate the reviewed minimal Actor Pack export goldens."""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import tempfile
from pathlib import Path

_ISOLATED_ROOT = Path(tempfile.mkdtemp(prefix="actor-pack-golden-"))
os.environ.update(
    {
        "HOME": str(_ISOLATED_ROOT / "home"),
        "XDG_CONFIG_HOME": str(_ISOLATED_ROOT / "config"),
        "XDG_DATA_HOME": str(_ISOLATED_ROOT / "data"),
        "XDG_CACHE_HOME": str(_ISOLATED_ROOT / "cache"),
        "TLDW_CONFIG_PATH": str(_ISOLATED_ROOT / "config" / "config.toml"),
        "TLDW_TEST_MODE": "1",
    }
)
for directory in ("home", "config", "data", "cache"):
    (_ISOLATED_ROOT / directory).mkdir(mode=0o700)

from Tests.Actor_Packs.conftest import (  # noqa: E402
    PNG_1X1,
    PORTABLE_UUID,
    canonical_json,
)
from tldw_chatbook.Actor_Packs.export import (  # noqa: E402
    ActorPackExportSnapshot,
    write_actor_pack_archive,
)


def main() -> None:
    try:
        output_dir = Path(__file__).parent
        for actor_kind in ("character", "persona"):
            payload = canonical_json(
                {
                    "schema": "tldw.actor/v1",
                    "actor_kind": actor_kind,
                    "portable_uuid": PORTABLE_UUID,
                    "data": {"name": "Golden"},
                }
            )
            snapshot = ActorPackExportSnapshot(
                actor_kind=actor_kind,
                actor_revision=1,
                portable_uuid=PORTABLE_UUID,
                identity_version=1,
                portrait_name="portrait.png",
                portrait_sha256=hashlib.sha256(PNG_1X1).hexdigest(),
                local_actor_id="private-local-id",
                actor_payload=payload,
                portrait_bytes=PNG_1X1,
            )
            archive = io.BytesIO()
            write_actor_pack_archive(snapshot, archive)
            (output_dir / f"minimal-{actor_kind}.tldw-actor-pack").write_bytes(
                archive.getvalue()
            )
    finally:
        shutil.rmtree(_ISOLATED_ROOT, ignore_errors=True)


if __name__ == "__main__":
    main()
