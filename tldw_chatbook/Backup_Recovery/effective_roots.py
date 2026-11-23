"""Omit only absent aliases already covered by enrolled directory ownership."""

import stat
from contextlib import ExitStack
from os.path import ismount
from pathlib import Path

from tldw_chatbook.Utils.platform_files import os

from . import bootstrap
from .native_files import pinned_directory


def file_root_token(folder, root):
    """Registration evidence that ``root`` was a regular file in this folder."""
    return f"file:{folder.st_ino}:{root}"


def file_root_tokens(roots):
    """The folder evidence ``Admission.register`` records for its file roots."""
    return {
        file_root_token(os.stat(root.parent), root)
        for root in roots
        if stat.S_ISREG(os.stat(root, follow_symlinks=False).st_mode)
    }


def deleted_file_roots(roots, entries):
    """Registered files deleted from a folder that is still there (TASK-42115).

    The one deleted-root rule: such a root holds nothing and scopes exactly its
    own path, at every root consumer, with or without a recovery journal. It
    qualifies only when it was registered as a regular file with its folder's
    evidence, and its folder is present, is not a mount point and is still that
    folder (by inode, checked twice). A file inside an enrolled folder root
    qualifies too: without a journal ``effective_roots``' alias proof (with its
    profile co-ownership check) still runs first, and under a journal, which
    skips that proof, this rule is what lets a retired alias be recovered. A
    folder root, a root registered before the evidence existed, or a file whose
    folder is gone, replaced or a mount point (an unmounted drive) stays
    strict. Callers pass every registered root and entry.

    Args:
        roots: Every registered root.
        entries: Every registry entry (``roots`` and ``historical``).

    Returns:
        The deleted file roots, as ``Path`` objects.
    """
    roots = {Path(root) for root in roots}
    entries = tuple(entries)
    result = set()
    for root in roots:
        if os.path.lexists(root):
            continue
        try:
            if ismount(root.parent):
                continue
            with pinned_directory(root.parent) as folder:
                info = os.fstat(folder)
                try:
                    os.stat(root.name, dir_fd=folder, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    continue
            with pinned_directory(root.parent) as folder:
                again = os.fstat(folder)
        except (OSError, ValueError, RuntimeError):
            continue
        token = file_root_token(info, root)
        if (again.st_dev, again.st_ino) == (info.st_dev, info.st_ino) and any(
            str(root) in entry["roots"] and token in entry["historical"]
            for entry in entries
        ):
            result.add(root)
    return result


def registered_deleted_file_roots(bootstrap_root):
    """``deleted_file_roots`` over the settled registry under ``bootstrap_root``.

    A missing, unreadable or pending registry yields nothing, so every root
    stays strict.
    """
    try:
        registry = bootstrap._registry(bootstrap_root)
    except (OSError, ValueError, RuntimeError):
        return set()
    if registry is None:
        return set()
    return deleted_file_roots(
        (root for entry in registry.values() for root in entry["roots"]),
        registry.values(),
    )


def effective_roots(roots, entries):
    """Keep unproved roots strict; redundant absence grants no file authority.

    Both arguments come from the same declared namespace set. Serialized roots
    and historical exclusion tokens remain unchanged. Existing aliases retain
    the ordinary resolver's behavior; only genuine no-follow absence is omitted.
    """
    roots = tuple(dict.fromkeys(Path(root) for root in roots))
    entries = tuple(entries)
    result = []
    for root in roots:
        if os.path.lexists(root) or not any(
            str(root) in entry["roots"] and "path:" + str(root) in entry["historical"]
            for entry in entries
        ):
            result.append(root)
            continue
        covered = False
        for parent in roots:
            parents = [
                entry
                for entry in entries
                if entry["roots"] == [str(parent)]
                and entry["pending"] is None
                and not entry["proposed"]
            ]
            if parent not in root.parents or not parents:
                continue
            try:
                with pinned_directory(parent) as descriptor, ExitStack() as opened:
                    info = os.fstat(descriptor)
                    expected = {
                        "path:" + str(parent),
                        bootstrap.inode_token(info),
                    }
                    if not any(
                        bootstrap.identity_view(entry["historical"]) == expected
                        for entry in parents
                    ):
                        continue
                    cursor, chain = descriptor, []
                    for component in root.relative_to(parent).parts:
                        try:
                            current = os.stat(
                                component, dir_fd=cursor, follow_symlinks=False
                            )
                        except FileNotFoundError:
                            # The absence comes from the original pinned root,
                            # never a new directory reached by reopening its path.
                            with pinned_directory(parent) as verified:
                                final = os.fstat(verified)
                                covered = (final.st_dev, final.st_ino) == (
                                    info.st_dev,
                                    info.st_ino,
                                )
                            for directory, name, identity in chain:
                                final = os.stat(
                                    name, dir_fd=directory, follow_symlinks=False
                                )
                                covered &= (final.st_dev, final.st_ino) == identity
                            try:
                                os.stat(component, dir_fd=cursor, follow_symlinks=False)
                            except FileNotFoundError:
                                pass
                            else:
                                covered = False
                            break
                        if not stat.S_ISDIR(current.st_mode):
                            break
                        child = os.open(
                            component,
                            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=cursor,
                        )
                        opened.callback(os.close, child)
                        held = os.fstat(child)
                        if (
                            (held.st_dev, held.st_ino)
                            != (current.st_dev, current.st_ino)
                            or held.st_uid not in (0, os.geteuid())
                            or held.st_mode & 0o022
                            and not held.st_mode & stat.S_ISVTX
                        ):
                            break
                        chain.append((cursor, component, (held.st_dev, held.st_ino)))
                        cursor = child
            except OSError:
                # bootstrap._binding's strict resolver will refuse this unproved
                # root; containment checks treat it as holding nothing.
                covered = False
                continue
            if covered:
                break
        if not covered:
            result.append(root)
    return tuple(result)


def absent_strict_roots(entries):
    """Name registered roots that admission refuses because they are gone.

    TASK-34100.16.1 review v5r2: an absent root keeps refusing unless an
    enrolled directory already owns it (``effective_roots``). Naming these is
    the only actionable part of that refusal; the result grants nothing. A
    deleted file root (``deleted_file_roots``, TASK-42115) refuses nothing, so
    it is not named.

    Args:
        entries: Registry entry mappings, as ``bootstrap._registry`` returns.

    Returns:
        The absent roots ``effective_roots`` keeps strict, in declared order.
    """
    entries = tuple(entries)
    declared = tuple(
        dict.fromkeys(Path(root) for entry in entries for root in entry["roots"])
    )
    deleted = deleted_file_roots(declared, entries)
    return tuple(
        root
        for root in effective_roots(declared, entries)
        if not os.path.lexists(root) and root not in deleted
    )


def check_redundant_profiles(registry, profiles):
    """Prove exhaustive fixed-profile co-ownership before global alias omission."""
    roots = {Path(path) for entry in registry.values() for path in entry["roots"]}
    omitted = roots - set(effective_roots(roots, registry.values()))
    for name, entry in registry.items():
        absent = omitted.intersection(map(Path, entry["roots"]))
        if not absent:
            continue
        owners = [profile for profile in profiles if name in profile["namespaces"]]
        if not owners:
            raise FileNotFoundError("redundant_root_profile_required")
        for profile in owners:
            entries = [registry[scope] for scope in profile["namespaces"]]
            declared = {Path(path) for value in entries for path in value["roots"]}
            if sorted(map(str, declared)) != profile["roots"] or absent.intersection(
                effective_roots(declared, entries)
            ):
                raise FileNotFoundError("redundant_root_profile_required")
    return omitted
