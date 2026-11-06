"""Omit only absent aliases already covered by enrolled directory ownership."""

import stat
from contextlib import ExitStack
from pathlib import Path

from tldw_chatbook.Utils.platform_files import os

from . import bootstrap
from .native_files import pinned_directory


def effective_roots(roots, entries, *, _read_budget=None, _custody=None):
    """Keep unproved roots strict; redundant absence grants no file authority.

    Both arguments come from the same declared namespace set. Serialized roots
    and historical exclusion tokens remain unchanged. Existing aliases retain
    the ordinary resolver's behavior; only genuine no-follow absence is omitted.
    """
    if _read_budget is not None:
        from .admission import _price_paths

        _price_paths(roots, _read_budget, copies=3)
        _read_budget.container(3 * len(roots))
    roots = tuple(dict.fromkeys(Path(root) for root in roots))
    entries = tuple(entries)
    result = []
    for root in roots:
        if _read_budget is not None:
            _read_budget.reserve(512)
            _price_paths((root,), _read_budget, copies=len(entries) + 2)
        if os.path.lexists(root) or not any(
            str(root) in entry["roots"] and "path:" + str(root) in entry["historical"]
            for entry in entries
        ):
            result.append(root)
            continue
        covered = False
        for parent in roots:
            if _read_budget is not None:
                _price_paths((parent, root), _read_budget, copies=len(entries) + 3)
                _read_budget.container(len(entries))
                for entry in entries:
                    _read_budget.container(len(entry["historical"]))
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
                with (
                    (
                        pinned_directory(parent)
                        if _custody is None
                        else pinned_directory(parent, _custody=_custody)
                    ) as descriptor,
                    ExitStack() as opened,
                ):
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
                            with (
                                pinned_directory(parent)
                                if _custody is None
                                else pinned_directory(parent, _custody=_custody)
                            ) as verified:
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
                        if _custody is None:
                            opened.callback(os.close, child)
                        else:
                            from .native_files import _close_owned

                            _custody.fds.append(child)
                            opened.callback(_close_owned, child, _custody)
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
                # The existing strict resolver will refuse this unproved root.
                covered = False
                continue
            if covered:
                break
        if not covered:
            result.append(root)
    return tuple(result)


def check_redundant_profiles(registry, profiles, *, _read_budget=None, _custody=None):
    """Prove exhaustive fixed-profile co-ownership before global alias omission."""
    if _read_budget is not None:
        from .admission import _price_paths

        _read_budget.value(registry)
        _read_budget.value(profiles)
        for entry in registry.values():
            _price_paths(entry["roots"], _read_budget, copies=3)
    roots = {Path(path) for entry in registry.values() for path in entry["roots"]}
    omitted = roots - set(
        effective_roots(
            roots,
            tuple(registry.values()),
            _read_budget=_read_budget,
            _custody=_custody,
        )
    )
    for name, entry in registry.items():
        if _read_budget is not None:
            _price_paths(entry["roots"], _read_budget, copies=2)
            _read_budget.container(len(profiles))
        absent = omitted.intersection(map(Path, entry["roots"]))
        if not absent:
            continue
        owners = [profile for profile in profiles if name in profile["namespaces"]]
        if not owners:
            raise FileNotFoundError("redundant_root_profile_required")
        for profile in owners:
            if _read_budget is not None:
                _read_budget.container(3 * len(profile["namespaces"]))
                for namespace in profile["namespaces"]:
                    _price_paths(registry[namespace]["roots"], _read_budget, copies=3)
            entries = [registry[scope] for scope in profile["namespaces"]]
            declared = {Path(path) for value in entries for path in value["roots"]}
            if sorted(map(str, declared)) != profile["roots"] or absent.intersection(
                effective_roots(
                    declared, entries, _read_budget=_read_budget, _custody=_custody
                )
            ):
                raise FileNotFoundError("redundant_root_profile_required")
    return omitted
