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

from ..config import get_cli_setting, save_settings_to_cli_config
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


def picker_recent_context(section: str) -> str:
    """The recents key for one picker context (task-32643 AC#3).

    The three Notes doors already key their remembered START directory by
    config section (task-32174: each context remembers its own, deliberately
    not shared). The list of roots each door recently RETURNED is keyed the
    same way, off the same string, so a caller cannot pass the picker one
    context and persist under another.

    Args:
        section: The config section this picker writes, e.g. ``"file_notes"``.

    Returns:
        The context name for ``RecentLocations``, i.e. the suffix of the
        ``[filepicker] recent_<context>`` key it stores under.
    """
    return section.replace(".", "_")


def _recent_roots_write(context: str, directory: Path) -> dict[str, object]:
    """Build the ``[filepicker]`` half of a selection's single config write.

    Reuses ``RecentLocations`` -- the store the enhanced picker family has
    always used -- for the dedupe, the newest-first ordering and the trim, so
    there is one recents FORMAT in the app rather than a second one invented
    for the vendored pickers. ``persist=False`` keeps its own synchronous
    config rewrite out of this, because the caller folds both halves into one
    write below.

    Args:
        context: The recents context, from :func:`picker_recent_context`.
        directory: The directory the picker just returned.

    Returns:
        A ``save_settings_to_cli_config`` fragment, or ``{}`` when the recents
        store is unusable -- remembering the start directory must not fail
        because of the nicety beside it.
    """
    try:
        from ..Widgets.enhanced_file_picker import RecentLocations

        recent = RecentLocations(context=context)
        recent.add(directory, "directory", persist=False)
        return {"filepicker": {f"recent_{context}": recent.get_recent()}}
    except Exception:
        logger.warning("Could not update a picker's recent-folder list")
        return {}


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
    section: str,
    key: str,
    selected_path: Path,
    generation: int,
    recent_context: str = "",
) -> None:
    """Persist the directory a picker selection came from. Blocking.

    Reads the filesystem and rewrites ``config.toml``; run it on a worker
    thread, never on the event loop. A selection whose ``generation`` has
    since been superseded is dropped instead of written.

    Also appends the directory to this context's recent-roots list when
    ``recent_context`` is given (task-32643 AC#3), in the SAME config write
    rather than a second one -- the two facts are written by one selection and
    rewriting config.toml twice per pick would be two chances to lose a
    concurrent edit.

    Args:
        section: The TOML section to write.
        key: The key within that section.
        selected_path: What the picker returned -- a directory is kept as is,
            a file contributes its parent.
        generation: The value :func:`claim_browse_directory` returned for
            this selection.
        recent_context: The :func:`picker_recent_context` value of a picker
            that OFFERS its recent roots back (the three Notes folder doors).
            Blank -- the default -- writes no recents list, so a caller whose
            picker never reads one does not accumulate a key nothing shows.
            Only a selection that WAS a folder joins that list; see below.
    """
    try:
        chose_a_folder = selected_path.is_dir()
        directory = selected_path if chose_a_folder else selected_path.parent
        with _WRITE_ORDER:
            if _GENERATIONS.get((section, key)) != generation:
                logger.debug(
                    f"Superseded remembered directory for [{section}].{key}"
                )
                return
            # Inside the lock: the recents update is a read-modify-write of a
            # LIST, so two selections racing here would otherwise each read
            # the pre-existing list and the loser's entry would vanish.
            settings: dict[str, object] = {section: {key: str(directory)}}
            # Only a folder the user actually CHOSE joins the recents list.
            # The start directory happily takes a picked file's parent -- that
            # is where to reopen -- but Ctrl+R offers its rows as answers
            # (`base_dialog._on_recent_selected` dismisses with the row), so a
            # folder the user merely browsed THROUGH on the way to a file
            # would come back as a one-keystroke whole-folder adoption of a
            # folder they never picked. Review round 1, finding 2.
            if recent_context and chose_a_folder:
                settings.update(_recent_roots_write(recent_context, directory))
            saved = save_settings_to_cli_config(settings)
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
