"""Fixed qualification policies; never imported by production entry points."""

import hashlib
from dataclasses import replace


def release_snapshot(mode):
    from tldw_chatbook.Canvas.profiles import load_profile_snapshot

    if mode not in {"candidate", "revoked"}:
        raise ValueError("unknown qualification policy")
    base = load_profile_snapshot()
    return replace(
        base,
        policy_id=hashlib.sha256(("canvas-release-test:" + mode).encode()).hexdigest(),
        default_diagram_profile="canvas-v2-mermaid-1" if mode == "candidate" else None,
        profiles=tuple(
            replace(
                row,
                executable=mode == "candidate",
                reason=None if mode == "candidate" else "revoked",
            )
            if row.profile_id == "canvas-v2-mermaid-1"
            else row
            for row in base.profiles
        ),
    )
