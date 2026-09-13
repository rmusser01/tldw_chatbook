"""Last-used start directory for the Library's plain (vendored) pickers.

``Import once``, ``Keep a folder synced`` and ``Folder files`` all open a
vendored ``FileOpen``/``SelectDirectory`` (``Third_Party.textual_fspicker``),
which -- unlike ``EnhancedFileOpen`` -- has no per-context memory of its own.
Each caller therefore remembers its own directory under its own config key.
The read and the write both live here rather than being copied per caller
(task-32174, PR #2554 review):

* the value read back is *persisted user state*, i.e. a trust boundary, so
  it goes through ``Utils/path_validation.py`` once, in one place, before any
  filesystem probe or picker use; and
* the write is ordered per key, so a second selection made while the first
  config write is still in flight cannot be overwritten by the older one.
"""

from __future__ import annotations

import threading
from pathlib import Path

from loguru import logger

from ..config import get_cli_setting, save_setting_to_cli_config
from ..Utils.path_validation import validate_existing_absolute_directory

#: Serializes the generation check with the config write it guards, so the
#: newest claimed selection is the one that survives for a given key.
_WRITE_ORDER = threading.Lock()
_GENERATIONS: dict[tuple[str, str], int] = {}


def validated_browse_directory(remembered: object) -> Path | None:
    """Validate a picker start directory read back from configuration.

    Args:
        remembered: The raw ``[section] key`` value, as ``get_cli_setting``
            returned it. Anything falsy means "nothing remembered yet".

    Returns:
        The normalized, existing absolute directory, or ``None`` when the
        stored value is absent, malformed, relative, traversing, or no longer
        a directory -- callers fall back to their own default (home).
    """
    if not remembered:
        return None
    try:
        return validate_existing_absolute_directory(
            Path(str(remembered)).expanduser()
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        # Deliberately path-free: the value is user content.
        logger.warning("Ignoring an unusable remembered picker directory")
        return None


def browse_start_directory(remembered: object) -> Path:
    """Return where a folder picker should open, with every fallback.

    task-32251 AC#5: before anything has been remembered for a context,
    every picker opened at ``$HOME`` -- even though ``[notes]
    sync_directory`` already names the folder this user keeps notes in.
    Home stays the last resort.

    Args:
        remembered: The raw ``[section] key`` value for this picker's own
            context, as ``get_cli_setting`` returned it.

    Returns:
        The remembered directory, else the configured notes sync
        directory, else the user's home directory. Every candidate goes
        through :func:`validated_browse_directory`, so a configured value
        that is relative, traversing or gone is skipped like any other.
    """
    for candidate in (
        remembered,
        get_cli_setting("notes", "sync_directory", None),
    ):
        validated = validated_browse_directory(candidate)
        if validated is not None:
            return validated
    return Path.home()


def claim_browse_directory(section: str, key: str) -> int:
    """Reserve the next write slot for one picker context.

    Call this on the event loop, in selection order; hand the result to
    :func:`remember_browse_directory` on the worker.

    Args:
        section: The TOML section holding the key (e.g. ``"library.ingest"``).
        key: The key within that section.

    Returns:
        The generation number this selection owns.
    """
    with _WRITE_ORDER:
        generation = _GENERATIONS.get((section, key), 0) + 1
        _GENERATIONS[(section, key)] = generation
        return generation


def remember_browse_directory(
    section: str, key: str, selected_path: Path, generation: int
) -> None:
    """Persist the directory a picker selection came from. Blocking.

    Reads the filesystem and rewrites ``config.toml``; run it on a worker
    thread, never on the event loop. A selection whose ``generation`` has
    since been superseded is dropped instead of written.

    Args:
        section: The TOML section to write.
        key: The key within that section.
        selected_path: What the picker returned -- a directory is kept as is,
            a file contributes its parent.
        generation: The value :func:`claim_browse_directory` returned for
            this selection.
    """
    try:
        directory = (
            selected_path if selected_path.is_dir() else selected_path.parent
        )
        with _WRITE_ORDER:
            if _GENERATIONS.get((section, key)) != generation:
                logger.debug(
                    f"Superseded remembered directory for [{section}].{key}"
                )
                return
            saved = save_setting_to_cli_config(section, key, str(directory))
    except Exception:
        logger.exception(
            f"Could not remember the last-used directory for [{section}].{key}"
        )
        return
    if not saved:
        logger.error(
            f"[{section}].{key} was not saved -- that picker will reopen at "
            "its previous directory"
        )
