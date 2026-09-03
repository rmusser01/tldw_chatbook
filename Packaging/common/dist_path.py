"""Distribution output path validation for release scripts."""

from inspect import signature
from pathlib import Path

from tldw_chatbook.Utils.path_validation import validate_path


_VALIDATE_PATH_PARAMETERS = signature(validate_path).parameters


def resolve_dist_dir(dist_dir: str, repo_root: Path) -> Path:
    """Return a validated repository-contained distribution directory.

    Args:
        dist_dir: Candidate distribution output directory.
        repo_root: Repository root that must contain the output directory.

    Returns:
        The resolved distribution directory path.

    Raises:
        ValueError: If the output path is empty, points at the repository root,
            or escapes the repository root.
    """
    if not dist_dir:
        raise ValueError("DIST_DIR cannot be empty")

    root = repo_root.resolve()
    kwargs: dict[str, bool] = {}
    if "redact_paths" in _VALIDATE_PATH_PARAMETERS:
        kwargs["redact_paths"] = True
    destination = validate_path(dist_dir, root, **kwargs)
    relative = destination.relative_to(root)
    if not relative.parts:
        raise ValueError("DIST_DIR must be a repository subdirectory")

    return destination
