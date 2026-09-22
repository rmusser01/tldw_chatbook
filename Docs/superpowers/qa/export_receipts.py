"""Copy explicitly selected text QA receipts with host paths normalized.

Keep raw evidence private. Publish through this command, then link its manifest:
    python Docs/superpowers/qa/export_receipts.py --destination QA_DIR RECEIPT...
"""

import argparse
import hashlib
import json
import re
from pathlib import Path


def normalize_receipt(text: str, *, repo: Path, home: Path) -> str:
    """Replace host paths while preserving relative paths and evidence values.

    Args:
        text: UTF-8 receipt contents, including serialized JSON or command output.
        repo: Checkout that produced the receipt, possibly under .worktrees.
        home: User home directory to replace with a stable placeholder.

    Returns:
        Receipt text with repository, home and temporary roots normalized.
    """
    main = str(repo).split("/.worktrees/", 1)[0]
    text = re.sub(
        re.escape(main) + r"(?:/\.worktrees/[^/\s\"']+)?(?=/|$|[\s\"'])",
        "<repo>",
        text,
    )
    text = text.replace(str(home), "<home>")
    text = re.sub(r"(?<![\w:/])/(?:Users|home)/[^/\s\"']+", "<home>", text)
    text = re.sub(r"/(?:private/)?var/folders/[^/]+/[^/]+/T(?=/)", "<tmp>", text)
    text = re.sub(r"(?<![\w:/])/(?:private/)?tmp(?=/)", "<tmp>", text)
    return re.sub(r"(?<=/)pytest-of-[^/\s\"']+", "pytest-of-user", text)


def main() -> None:
    """Copy text receipts and record source/export hashes without editing raw files.

    Raises:
        SystemExit: Two for ambiguous destinations or an input/output collision.
        OSError: A receipt cannot be read or the destination cannot be written.
        UnicodeError: An input is not a UTF-8 text receipt.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("receipts", nargs="+", type=Path)
    args = parser.parse_args()
    names = [p.name for p in args.receipts]
    if len(set(names)) != len(names) or "publication-manifest.json" in names:
        parser.error("receipt names must be unique and cannot name the output manifest")
    if args.destination.exists() or args.destination.is_symlink():
        parser.error("destination must be a new directory")
    repo, home = Path(__file__).resolve().parents[3], Path.home()
    prepared = []
    for path in args.receipts:
        original = path.read_bytes()
        normalized = normalize_receipt(original.decode(), repo=repo, home=home)
        exported = (
            "\n".join(line.rstrip() for line in normalized.splitlines()) + "\n"
        ).encode()
        prepared.append((path, original, exported))
    args.destination.mkdir(parents=True, exist_ok=False)
    manifest = []
    for path, original, exported in prepared:
        (args.destination / path.name).write_bytes(exported)
        manifest.append(
            {
                "source": normalize_receipt(str(path), repo=repo, home=home),
                "export": path.name,
                "source_sha256": hashlib.sha256(original).hexdigest(),
                "export_sha256": hashlib.sha256(exported).hexdigest(),
                "normalization": "host paths and trailing whitespace",
            }
        )
    (args.destination / "publication-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
