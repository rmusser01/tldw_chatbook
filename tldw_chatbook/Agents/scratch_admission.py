"""Fixed native Console scratch composition; persistent admission stays ordinary."""

import inspect
import os
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from weakref import WeakSet

from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.activation import (
    _source_scope_admitted,
    activation_permission,
)
from tldw_chatbook.Backup_Recovery.admission_runtime import execution_identity
from tldw_chatbook.Backup_Recovery.participants import (
    native_prepared_scratch,
    native_scratch_writer as _native_writer,
)
from tldw_chatbook.Backup_Recovery.storage_admission import (
    StorageLease,
    acquire_storage,
)

_current = ContextVar("native_console_scratch_admission", default=None)
_issued = WeakSet()
_workers = WeakSet()
_incoming = ContextVar("native_console_scratch_worker", default=None)


def _refuse():
    from .activation import AgentActivationRequired

    raise AgentActivationRequired()


@dataclass(eq=False)
class _Frame:
    writer: object
    manager: object
    snapshot: object
    state: object
    identity: tuple
    scratch_sources: tuple
    live: bool = True

    def check(self, *, local=True):
        from .activation import AgentActivationRequired

        if (
            self not in _issued
            or not self.live
            or self.identity[0] != os.getpid()
            or (local and self.identity != execution_identity())
        ):
            _refuse()
        self.state.check(AgentActivationRequired)
        if _native_writer(self.writer) != (self.manager, self.snapshot):
            _refuse()
        with self.manager._condition:
            record = self.manager._matching_record_locked(self.snapshot)
            if (
                record is None
                or record.leases <= 0
                or not self.manager._identity_matches(self.snapshot)
            ):
                _refuse()


@dataclass(eq=False)
class _Worker:
    parent: _Frame


def capture_worker():
    frame = _current.get()
    if frame is None:
        return None
    frame.check()
    handoff = _Worker(frame)
    _workers.add(handoff)
    return handoff


@contextmanager
def worker_entry(handoff):
    try:
        if handoff is not None:
            if type(handoff) is not _Worker or handoff not in _workers:
                _refuse()
            handoff.parent.check(local=False)
    except (OSError, ValueError, TypeError, RuntimeError, AttributeError):
        _refuse()
    current = _current.set(None)
    incoming = _incoming.set(handoff)
    try:
        yield
    finally:
        _incoming.reset(incoming)
        _current.reset(current)


def _scratch_path(writer, snapshot, owner, path):
    if owner != "agents.history":
        return False
    root = snapshot.root
    log_dir = inspect.getattr_static(writer, "log_dir", None)
    if path != root and (log_dir is None or path != log_dir):
        return False
    if not path.resolve().is_relative_to(root):
        _refuse()
    return True


def _selected_scope(guard, state):
    if (
        state is None
        or guard.context.get() is not state
        or state.identity != execution_identity()
    ):
        _refuse()
    selected = bootstrap.effective_config_path()
    lease = state.leases.get(selected)
    if type(lease) is not StorageLease:
        _refuse()
    root, names = lease.execution_context(selected)
    if selected != bootstrap.effective_config_path():
        _refuse()
    return selected, root, names


def _check_activation(selected, root, names):
    if not all(
        activation_permission(
            owner,
            config_selector=selected,
            bootstrap_root=root,
            namespaces=names,
            ordinary_only=names is None,
        )
        for owner in ("config", "agents.history")
    ):
        _refuse()
    if selected != bootstrap.effective_config_path():
        _refuse()


def _check_selected(frame, guard):
    frame.check()
    selected, root, names = _selected_scope(guard, frame.state)
    for owner, path in frame.scratch_sources:
        if not _scratch_path(
            frame.writer, frame.snapshot, owner, path
        ) or not _source_scope_admitted(root, names or (), path):
            _refuse()
    _check_activation(selected, root, names)


def check_prepared_scratch(manager, snapshot, session_id):
    """Verify the already held native scratch before a log writer exists."""
    from .activation import _guard

    if not native_prepared_scratch(manager, snapshot, session_id):
        _refuse()
    state = _guard.context.get()
    if state is None:
        _refuse()
    state.check(_guard.error)
    selected, root, names = _selected_scope(_guard, state)
    if not _source_scope_admitted(root, names or (), snapshot.root):
        _refuse()
    _check_activation(selected, root, names)


@contextmanager
def execution(guard, service, sources):
    from .activation import _explicit_writer_root

    with ExitStack() as stack:
        frame = None
        try:
            active, handoff = _current.get(), _incoming.get()
            if handoff is not None:
                if type(handoff) is not _Worker or handoff not in _workers:
                    _refuse()
                # The parent is checked at entry, not during independent work.
                handoff.parent.check(local=False)
                writer, inherited = (
                    handoff.parent.writer,
                    handoff.parent.scratch_sources,
                )
            elif active is not None:
                active.check()
                writer, inherited = active.writer, active.scratch_sources
            else:
                writer, inherited = None, ()
                if _explicit_writer_root(service) is not None:
                    writer = inspect.getattr_static(
                        service, "_injected_run_log_writer", None
                    )
            native = _native_writer(writer) if writer is not None else None
            if native is None:
                if handoff is not None or active is not None:
                    _refuse()
                stack.enter_context(guard.execution(service, sources=sources))
            else:
                manager, snapshot = native
                stack.enter_context(manager.lease(snapshot))
                observed = tuple(
                    dict.fromkeys(inherited + tuple(sources) + guard.sources(service))
                )
                scratch = tuple(
                    (owner, path)
                    for owner, path in observed
                    if _scratch_path(writer, snapshot, owner, path)
                )
                if not scratch:
                    scratch = (("agents.history", snapshot.root),)
                ordinary = tuple(row for row in observed if row not in scratch)
                stack.enter_context(
                    guard.execution(service, resolve=lambda _service: ordinary)
                )
                frame = _Frame(
                    writer,
                    manager,
                    snapshot,
                    guard.context.get(),
                    execution_identity(),
                    scratch,
                )
                _issued.add(frame)
                _check_selected(frame, guard)
        except (OSError, ValueError, TypeError, RuntimeError, AttributeError):
            if frame is not None:
                frame.live = False
            _refuse()
        current = _current.set(frame)
        incoming = _incoming.set(None)
        try:
            yield
        finally:
            if frame is not None:
                frame.live = False
            _incoming.reset(incoming)
            _current.reset(current)


@contextmanager
def writer_storage(writer, path):
    """Three stock writer effect sites; unknown writers keep ordinary storage."""
    from .activation import _guard
    from .run_log import MANIFEST_NAME

    frame = _current.get()
    if frame is None or writer is not frame.writer:
        with acquire_storage(path):
            yield
        return
    _check_selected(frame, _guard)
    root, log_dir = frame.snapshot.root, writer.log_dir
    allowed = path == root or (
        log_dir is not None
        and path
        in (
            log_dir,
            writer._segment_path(),
            log_dir / MANIFEST_NAME,
        )
    )
    if not allowed or not Path(path).resolve().is_relative_to(root):
        _refuse()
    with frame.manager.lease(frame.snapshot):
        yield
