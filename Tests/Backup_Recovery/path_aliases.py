"""Real directory aliases without requiring Windows symlink privileges."""

import os
import subprocess
from pathlib import Path


def create_directory_alias(alias: Path, target: Path) -> None:
    """Create an NTFS junction on Windows or a directory symlink on POSIX."""
    target = target.resolve(strict=True)
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(alias), str(target)],
            check=True,
            capture_output=True,
        )  # nosec B603 B607 -- fixed native junction command and owned test paths.
        assert alias.is_junction()
    else:
        alias.symlink_to(target, target_is_directory=True)
        assert alias.is_symlink()
    assert alias.samefile(target)
