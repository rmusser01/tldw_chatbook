"""Read preserved native ZIPs and verify ordered fixture content after exit.

Usage: python verify_artifacts.py PROFILE
"""

import hashlib
import json
import sys
import zipfile
from pathlib import Path


def main():
    root = Path(sys.argv[1]).resolve()
    evidence = root / "evidence"
    result = json.loads((evidence / "result.json").read_text())
    fixtures = json.loads((evidence / "fixtures.json").read_text())["conversations"]
    assert result["passed"] and result["app_run_returned"]
    assert len(result["artifacts"]) == 8
    receipts = []
    for artifact in result["artifacts"]:
        label = artifact["label"]
        assert label.endswith(("-whole", "-selected"))
        expected = fixtures[:2] if label.endswith("-whole") else fixtures[:1]
        path = evidence / (label + ".zip")
        assert path.resolve().is_relative_to(evidence)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
        assert path.stat().st_size == artifact["size_bytes"]
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
            assert len(archive.namelist()) == len(set(archive.namelist()))
            manifest = json.loads(archive.read("manifest.json"))
            items = manifest["content_items"]
            assert len(items) == len(expected)
            assert {item["id"] for item in items} == {f["id"] for f in expected}
            assert all(item["type"] == "conversation" for item in items)
            assert not any(
                name.startswith("content/notes/") for name in archive.namelist()
            )
            for item in items:
                fixture = next(f for f in expected if f["id"] == item["id"])
                payload = json.loads(archive.read(item["file_path"]))
                messages = payload["messages"]
                assert payload["id"] == fixture["id"]
                assert payload["name"] == fixture["title"]
                assert len(messages) == 2
                assert [m["id"] for m in messages] == [
                    m["id"] for m in fixture["messages"]
                ]
                assert [m["content"] for m in messages] == [
                    m["content"] for m in fixture["messages"]
                ]
                assert [m["role"] for m in messages] == ["user", "assistant"]
                assert [m["order"] for m in messages] == [0, 1]
        if label.endswith("-selected"):
            final_path = Path(artifact["path"]).resolve()
            assert final_path.is_relative_to(root / "exports")
            assert (
                hashlib.sha256(final_path.read_bytes()).hexdigest()
                == artifact["sha256"]
            )
        receipts.append(
            artifact
            | {
                "manifest_length_and_ids_exact": True,
                "ordered_message_ids_bodies_roles_and_lengths_exact": True,
            }
        )
    output = {
        "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "artifact_count": len(receipts),
        "duplicate_manifest_and_zip_entries_absent": True,
        "final_four_destinations_match_selected_exports": True,
        "artifacts": receipts,
    }
    (evidence / "artifact-verification.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    print(
        f"Verified {len(receipts)} ZIPs, exact ordered fixture messages and four final replacements."
    )


if __name__ == "__main__":
    main()
