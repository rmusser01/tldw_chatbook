"""Finite admission for installed loose voice file operations.

A call owns its declared directory through all related publications. Nested calls
reuse that hold only on the same task and thread. Native-close uncertainty and
failed partial publication remain counted so maintenance cannot capture them.
"""

from __future__ import annotations

import inspect
import io
import os
import shutil
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from weakref import WeakKeyDictionary

from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.profile_paths import (
    effective_config_path,
    lexical_path,
)

_current = ContextVar("tts_voice_operation", default=None)
_sources = WeakKeyDictionary()
# Retain the finite selected paths after short-lived catalog/manager objects go
# away. These are refusal evidence, never additional discovery/capture authority.
_observed_sources = {}
_native = ContextVar("tts_voice_native", default=None)


def observed_sources(config_path):
    """Return immutable observations from actual admitted installed voice calls."""
    with storage._lock:
        return tuple(
            (root, *binding)
            for (profile, root), binding in _observed_sources.items()
            if profile == lexical_path(config_path)
        )


def native():
    operation = active()
    return operation if operation is _native.get() else None


def active():
    """Return the current call, never one inherited by another task/thread."""
    operation = _current.get()
    if operation is not None and operation.identity == (
        os.getpid(),
        threading.get_ident(),
        storage._task_identity(),
    ):
        return operation
    return None


def _root(receiver):
    for name in ("voice_blends_dir", "voice_samples_dir", "voice_dir"):
        value = getattr(receiver, name, None)
        if value is not None:
            return lexical_path(value)
    raise bootstrap.RecoveryRequired("tts_voice_root_not_selected")


class _Call:
    def __init__(self, receiver):
        self.receiver = receiver
        self.identity = (os.getpid(), threading.get_ident(), storage._task_identity())
        self.roots = {}
        self.descriptors = set()
        self.changed = 0
        self.uncertain = False
        self.source = None

    def reserve(self, path):
        selected = lexical_path(path)
        if selected not in self.roots:
            self.roots[selected] = storage.acquire_storage(selected)
        return selected

    def check(self, path, *, external=False):
        root = _root(self.receiver)
        if self.source is None:
            self.reserve(root)
            self.source = root
        binding = _sources.get(self.receiver)
        resolved = root.resolve()
        info = root.stat() if root.exists() else None
        identity = (info.st_dev, info.st_ino) if info is not None else None
        if binding is None:
            _sources[self.receiver] = (root, resolved, identity)
        elif binding != (root, resolved, identity):
            if binding == (root, resolved, None) and identity is not None:
                _sources[self.receiver] = (root, resolved, identity)
            else:
                raise bootstrap.RecoveryRequired("tts_voice_root_changed")
        with storage._lock:
            _observed_sources[(effective_config_path(), root)] = (resolved, identity)
        if root != self.source:
            raise bootstrap.RecoveryRequired("tts_voice_root_changed")
        selected = lexical_path(path)
        if not selected.is_relative_to(root):
            if not external:
                raise bootstrap.RecoveryRequired("tts_voice_path_outside_scope")
            if not any(selected.is_relative_to(owned) for owned in self.roots):
                raise bootstrap.RecoveryRequired("tts_voice_path_outside_scope")
        # Reject observed links throughout owned descendants. Parent traversal
        # grants no ownership over siblings and needs no broad parent enrollment.
        relative = (
            selected.relative_to(root) if selected.is_relative_to(root) else Path(".")
        )
        for candidate in (
            root,
            *(
                root / Path(*relative.parts[:i])
                for i in range(1, len(relative.parts) + 1)
            ),
        ):
            if candidate.is_symlink():
                raise bootstrap.RecoveryRequired("tts_voice_path_symlink")
        for owned, lease in self.roots.items():
            lease.execution_context(owned)
        return selected

    @contextmanager
    def native_scope(self):
        token = _native.set(self)
        try:
            yield
        finally:
            _native.reset(token)

    def opened(self, fd):
        self.descriptors.add(fd)

    def close(self, fd):
        try:
            os.close(fd)
        except BaseException:
            self.uncertain = True
            raise
        else:
            self.descriptors.discard(fd)


@contextmanager
def _scope(receiver):
    previous = active()
    if previous is not None and previous.receiver is receiver:
        yield previous
        return
    operation = _Call(receiver)
    with storage._changed:
        if storage._pause is not None:
            raise bootstrap.RecoveryRequired("storage_locally_paused")
        storage._raw_operations.add(operation)
    token = _current.set(operation)
    try:
        yield operation
    except BaseException:
        if operation.changed:
            operation.uncertain = True
        raise
    finally:
        try:
            if operation.source is not None and not operation.uncertain:
                operation.check(operation.source)
        except BaseException:
            operation.uncertain = True
            raise
        finally:
            _current.reset(token)
        if not operation.uncertain and not operation.descriptors:
            for lease in reversed(tuple(operation.roots.values())):
                lease.close()
            with storage._changed:
                storage._raw_operations.discard(operation)
                storage._changed.notify_all()


def call(function):
    """Fence one installed constructor/catalog/publication operation."""
    signature = inspect.signature(function)

    def before(receiver, args, kwargs, operation):
        values = signature.bind(receiver, *args, **kwargs).arguments
        if function.__name__ != "__init__":
            operation.check(_root(receiver))
        # Names form path components in these installed writers.
        for field in ("profile_name", "name"):
            value = values.get(field)
            if value is not None and (
                not isinstance(value, str)
                or value in (".", "..")
                or "/" in value
                or "\\" in value
            ):
                raise ValueError("invalid_voice_name")
        export = values.get("export_path")
        if export is not None:
            operation.reserve(lexical_path(export))

    def after(result, operation, prior_changes):
        failed = result is False or (
            isinstance(result, tuple) and result and result[0] is False
        )
        if failed and operation.changed > prior_changes:
            operation.uncertain = True
        return result

    if inspect.iscoroutinefunction(function):

        @wraps(function)
        async def asynchronous(receiver, *args, **kwargs):
            with _scope(receiver) as operation:
                before(receiver, args, kwargs, operation)
                prior_changes = operation.changed
                return after(
                    await function(receiver, *args, **kwargs), operation, prior_changes
                )

        return asynchronous

    @wraps(function)
    def synchronous(receiver, *args, **kwargs):
        with _scope(receiver) as operation:
            before(receiver, args, kwargs, operation)
            prior_changes = operation.changed
            return after(function(receiver, *args, **kwargs), operation, prior_changes)

    return synchronous


def directory(receiver, path, **kwargs):
    """Prepare the selected owned directory through the native private helper."""
    from tldw_chatbook.Utils.private_paths import (
        secure_private_directory,
        verify_trusted_directory,
    )

    operation = active()
    if operation is None:
        # The abstract base also serves unrelated installed backends.
        with storage.acquire_storage(path):
            return secure_private_directory(path, create=True, application_owned=True)
    operation.check(path, external=True)
    if lexical_path(path).exists() and not kwargs.get("create"):
        return verify_trusted_directory(path, allow_shared_sticky=False)
    return secure_private_directory(path, create=True, application_owned=True)


def mkdir(receiver, path, **kwargs):
    if active() is None:
        return Path(path).mkdir(**kwargs)
    return directory(receiver, path, create=True)


def _operation(receiver, path, *, external=False):
    operation = active()
    if operation is None or operation.receiver is not receiver:
        raise bootstrap.RecoveryRequired("tts_voice_operation_missing")
    selected = operation.check(path, external=external)
    return operation, selected


@contextmanager
def open_text(receiver, path, mode="r", **kwargs):
    """Read a checked regular file or atomically publish a complete text file."""
    from tldw_chatbook.Utils.private_paths import (
        atomic_private_write_text,
        open_private_binary,
    )

    operation = active()
    selected = lexical_path(path)
    if mode == "r":
        # Import input is read-only and may be outside the selected voice root.
        if selected.is_relative_to(_root(receiver)):
            operation.check(selected)
        with open_private_binary(selected) as opened:
            yield io.StringIO(
                opened.stream.read().decode(kwargs.get("encoding") or "utf-8")
            )
        return
    if mode != "w":
        raise ValueError("unsupported_voice_file_mode")
    _operation(
        receiver, selected, external=not selected.is_relative_to(_root(receiver))
    )
    stream = io.StringIO()
    yield stream
    atomic_private_write_text(selected, stream.getvalue())
    operation.changed += 1


def copy(receiver, source, destination):
    """Publish reference/backup bytes atomically under the retained call."""
    from tldw_chatbook.Utils.private_paths import (
        atomic_private_write_bytes,
        open_private_binary,
    )

    operation, selected = _operation(
        receiver,
        destination,
        external=not lexical_path(destination).is_relative_to(_root(receiver)),
    )
    with open_private_binary(source) as opened:
        payload = opened.stream.read()
    atomic_private_write_bytes(selected, payload)
    operation.changed += 1
    return str(selected)


def unlink(receiver, path):
    operation, selected = _operation(receiver, path)
    from tldw_chatbook.Utils.private_paths import _open_verified_parent

    with operation.native_scope():
        parent, leaf = _open_verified_parent(selected, missing_leaf_allowed=False)
        try:
            os.unlink(leaf, dir_fd=parent)
            operation.changed += 1
        finally:
            operation.close(parent)


def remove_tree(receiver, path):
    """Remove only an owned descendant using fd-relative safe tree removal."""
    operation, selected = _operation(receiver, path)
    if selected == _root(receiver) or not shutil.rmtree.avoids_symlink_attacks:
        raise bootstrap.RecoveryRequired("tts_voice_delete_unqualified")
    from tldw_chatbook.Utils.private_paths import _open_verified_parent

    with operation.native_scope():
        parent, leaf = _open_verified_parent(selected, missing_leaf_allowed=False)
        try:
            shutil.rmtree(leaf, dir_fd=parent)
            operation.changed += 1
        except BaseException:
            operation.uncertain = True
            raise
        finally:
            operation.close(parent)


def reference_filename(value):
    """Require imported references to name one file inside their package."""
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
        or value in (".", "..")
        or "\\" in value
    ):
        raise ValueError("invalid_voice_reference_filename")
    return value
