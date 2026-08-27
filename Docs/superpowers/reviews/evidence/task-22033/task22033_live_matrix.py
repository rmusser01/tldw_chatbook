"""Isolated bootstrap for the TASK-22033 Prompt evidence journeys."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from tldw_chatbook.Utils.input_validation import validate_username
from tldw_chatbook.Utils.path_validation import validate_path


REQUIRED_ENV = (
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_CACHE_HOME",
    "TLDW_CONFIG_PATH",
    "TLDW_TEST_MODE",
    "TASK22033_DATA_DIR",
    "TASK22033_SCRATCH_ROOT",
)
JOURNEYS = frozenset({"geometry", "preservation", "bulk", "import", "detail"})


def _prepare_isolated_environment() -> tuple[Path, Path]:
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"refusing unisolated run; missing: {', '.join(missing)}")

    scratch_root = Path(os.environ["TASK22033_SCRATCH_ROOT"]).resolve()
    try:
        config_path = validate_path(
            os.environ["TLDW_CONFIG_PATH"], scratch_root, allow_hidden=True
        )
        data_dir = validate_path(
            os.environ["TASK22033_DATA_DIR"], scratch_root, allow_hidden=True
        )
        app_data_dir = validate_path(
            scratch_root / "app-data", scratch_root, allow_hidden=True
        )
        xdg_dirs = tuple(
            validate_path(os.environ[name], scratch_root, allow_hidden=True)
            for name in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME")
        )
    except ValueError as exc:
        raise SystemExit(
            "config/data/XDG paths must be contained by TASK22033_SCRATCH_ROOT"
        ) from exc

    for isolated_dir in (
        scratch_root,
        data_dir,
        app_data_dir,
        config_path.parent,
        *xdg_dirs,
    ):
        isolated_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        isolated_dir.chmod(0o700)
    config_path.write_text(
        f'[paths]\ndata_dir = "{app_data_dir.as_posix()}"\n',
        encoding="utf-8",
    )
    config_path.chmod(0o600)
    return config_path, data_dir


def _validated_journeys(raw_selectors: list[str]) -> set[str]:
    selected = set(raw_selectors) or set(JOURNEYS)
    for selector in selected:
        if not validate_username(selector, min_length=1, max_length=20):
            raise SystemExit(f"invalid journey selector: {selector!r}")
        if selector not in JOURNEYS:
            raise SystemExit(f"unsupported journey selector: {selector!r}")
    return selected


def main() -> None:
    """Validate isolation and execute the requested evidence journeys."""
    config_path, data_dir = _prepare_isolated_environment()
    selected = _validated_journeys(sys.argv[1:])

    from task22033_live_matrix_runner import run

    asyncio.run(run(selected, config_path=config_path, data_dir=data_dir))


if __name__ == "__main__":
    main()
