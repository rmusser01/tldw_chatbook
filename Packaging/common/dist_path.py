"""Distribution output path validation for release scripts."""

from pathlib import Path

from tldw_chatbook.Utils.path_validation import validate_path


def resolve_dist_dir(dist_dir: str, repo_root: Path) -> Path:
    """Return a validated repository-contained distribution directory."""
    if not dist_dir:
        raise ValueError("DIST_DIR cannot be empty")

    root = repo_root.resolve()
    destination = validate_path(dist_dir, root, redact_paths=True)
    relative = destination.relative_to(root)
    if not relative.parts:
        raise ValueError("DIST_DIR must be a repository subdirectory")

    return destination
