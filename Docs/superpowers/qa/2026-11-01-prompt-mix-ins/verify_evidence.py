"""Verify published bytes strictly; qualify optional historical Git provenance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[4]
qa = Path(__file__).resolve().parent
revision = sys.argv[1] if len(sys.argv) > 1 else None
count = 0
for manifest in ("evidence-manifest.json", "product-source-manifest.json"):
    records = json.loads((qa / manifest).read_text())["files"]
    assert records, (manifest, "empty manifest")
    for path, expected in records.items():
        data = (root / path).read_bytes()
        assert len(data) == expected["bytes"], path
        assert hashlib.sha256(data).hexdigest() == expected["sha256"], path
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        assert blob == expected["git_blob"], path
        spec = f"{revision}:{path}" if revision else f":{path}"
        stored = subprocess.check_output(["git", "show", spec], cwd=root)
        assert stored == data, (spec, "Git bytes differ")
        count += 1
print(f"Verified {count} current evidence/source records against {revision or 'Git index'}.")
current_live = json.loads((qa / "integration/live/final-source-sha256.json").read_text())
assert current_live, "empty current live manifest"
for path, digest in current_live.items():
    assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest, path
print(f"Verified {len(current_live)} current live source hashes.")
for manifest, commit in (
    ("live/product-source-sha256.json", "247c9eb19ddbf49e49d55c7c1851f72fed2616a2"),
    ("final-fix/live-preview/product-source-sha256.json", "d7744a78e9094ad4df08a6466fab6d4eb8f412ce"),
):
    records = json.loads((qa / manifest).read_text())
    assert records, manifest
    available = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=root,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0
    if not available:
        print(f"Historical source verification UNAVAILABLE: {commit} ({len(records)} hashes); old source history is not part of the focused PR.")
        continue
    for path, digest in records.items():
        data = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=root)
        assert hashlib.sha256(data).hexdigest() == digest, (commit, path)
    print(f"Verified {len(records)} historical source hashes at {commit}; these are not current port evidence.")
