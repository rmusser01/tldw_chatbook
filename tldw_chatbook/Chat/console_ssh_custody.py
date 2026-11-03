"""Private custody of the original model fs_* SSH operation (ADR-181/285)."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import subprocess
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Event

_current = ContextVar("native_model_ssh", default=None)


def current_ssh_call():
    """Return only the attachment carried by the original model tool worker."""
    return _current.get()


def capture_model_ssh(registry, name):
    from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
    from tldw_chatbook.Agents.virtual_cli_provider import _NATIVE_VIRTUAL_CLASS
    from tldw_chatbook.Tools.remote_root_types import RemoteRoot

    from .console_workflow_runs import native_workflow_qualified

    selection = registry.resolve_owner_for_name(name)
    if (
        not native_workflow_qualified()
        or selection is None
        or not (name.startswith("fs_") or name == "virtual_cli")
    ):
        return None
    provider = selection[1]
    virtual = name == "virtual_cli" and type(provider) is _NATIVE_VIRTUAL_CLASS
    authorities = inspect.getattr_static(provider, "_admitted_roots", None)
    if authorities is not None and type(authorities) is not dict:
        raise RuntimeError("original SSH authority source changed")
    if not (virtual or type(provider) is LocalToolProvider) or not any(
        isinstance(a.root, RemoteRoot) for a in (authorities or {}).values()
    ):
        return None
    if virtual:
        require_virtual_sources(provider)
    return _ModelSshCall(registry, name, selection, virtual=virtual)


def bind_model_ssh(provider, authority, spec):
    call = _current.get()
    if call is not None:
        call.bind(provider, authority, spec)


def virtual_source_records(provider=None):
    """Read only the original finite virtual provider/parser source records."""
    from tldw_chatbook.Agents import virtual_cli_provider as virtual
    from tldw_chatbook.Tools import virtual_cli_impls as commands
    from tldw_chatbook.Tools import workspace_tool_executor as executor

    from .console_ordinary_tree import _function_unchanged

    values, functions = [], []
    native = provider is None or type(provider) is virtual._NATIVE_VIRTUAL_CLASS
    if not native:
        return (), (), False
    for module, target in ((virtual, provider), (commands, None)):
        cls = module._NATIVE_VIRTUAL_CLASS
        values.append(
            (module, cls.__name__, inspect.getattr_static(module, cls.__name__), None)
        )
        native &= values[-1][2] is cls
        for name, original in module._NATIVE_VIRTUAL_METHODS:
            receiver = cls if target is None else target
            actual = inspect.getattr_static(receiver, name, None)
            values.append((receiver, name, actual, None))
            native &= actual is original
        functions.extend(module._NATIVE_VIRTUAL_FUNCTIONS)
    for module, sources in (
        (virtual, virtual._NATIVE_VIRTUAL_IMPORTS),
        (commands, commands._NATIVE_VIRTUAL_HELPERS),
    ):
        for name, original in sources:
            actual = inspect.getattr_static(module, name, None)
            values.append((module, name, actual, None))
            native &= actual is original
    cls = executor._NATIVE_ORDINARY_EXECUTOR_CLASS
    for obj, sources in (
        (
            executor,
            (
                ("WorkspaceToolExecutor", cls),
                *executor._NATIVE_ORDINARY_EXECUTOR_HELPERS,
            ),
        ),
        (cls, executor._NATIVE_ORDINARY_EXECUTOR_METHODS),
    ):
        for name, original in sources:
            actual = inspect.getattr_static(obj, name, None)
            values.append((obj, name, actual, None))
            native &= actual is original
    functions.extend(executor._NATIVE_ORDINARY_EXECUTOR_FUNCTIONS)
    parsers, entries = commands._NATIVE_VIRTUAL_PARSERS
    native &= commands._PARSERS is parsers and tuple(parsers.items()) == entries
    for parser, actions in commands._NATIVE_VIRTUAL_ACTIONS:
        actual = inspect.getattr_static(parser, "_actions", None)
        native &= type(actual) is list and tuple(actual) == actions
    for obj, name, original in commands._NATIVE_VIRTUAL_PARSER_SOURCES:
        actual = inspect.getattr_static(obj, name, None)
        values.append((obj, name, actual, None))
        native &= actual is original
    native &= all(_function_unchanged(record) for record in functions)
    if provider is not None:
        registries = inspect.getattr_static(provider, "_registries_by_alias", None)
        authorities = inspect.getattr_static(provider, "_admitted_roots", None)
        if (
            type(registries) is not dict
            or authorities is not None
            and type(authorities) is not dict
        ):
            return tuple(values), tuple(functions), False
        for registry in registries.values():
            native &= type(registry) is commands._NATIVE_VIRTUAL_CLASS
            for name, original in commands._NATIVE_VIRTUAL_METHODS:
                native &= inspect.getattr_static(registry, name, None) is original
    return tuple(values), tuple(functions), native


def require_virtual_sources(provider):
    if not virtual_source_records(provider)[2]:
        raise RuntimeError("original virtual provider/parser source changed")


def _original_methods(target, methods):
    return all(
        getattr(current, "__func__", current) is method
        for name, method in methods.items()
        for current in (getattr(target, name, None),)
    )


@dataclass
class _Request:
    executor: object
    wire: bytes
    operation_id: str
    issued: bool = False
    local_returned: bool = False
    terminal: object = None

    @property
    def proven(self):
        return not self.issued or (
            self.local_returned
            and self.terminal is not None
            and self.terminal.cleanup_proven
        )


class _RemoteSource:
    """The actual pre-enqueue authority, adapter, executor and transport inputs."""

    def __init__(self, authority, *, executor=None, local_specs=True):
        self.authority = authority
        self.local_specs = local_specs
        self.adapter = authority.workspace_executor if authority is not None else None
        self.executor = (
            getattr(self.adapter, "_inner", None) if authority is not None else executor
        )
        self.cfg = getattr(self.executor, "_ssh", None)
        self.cfg_values = dict(vars(self.cfg)) if self.cfg is not None else {}
        self.registry = None
        if self.cfg is not None and self.cfg.session_key:
            from tldw_chatbook.Tools.remote_session_registry import get_session_registry

            self.registry = get_session_registry()
        self.control_dir = (
            self.cfg.masters._control_dir if self.cfg is not None else None
        )
        self.manager_configuration = (
            _master_configuration(self.cfg.masters) if self.cfg is not None else None
        )
        self.transport_grace = (
            self.cfg.transport.grace_seconds if self.cfg is not None else None
        )
        self.transport_manager = (
            self.cfg.transport._manager if self.cfg is not None else None
        )
        self.executor_values = tuple(
            getattr(self.executor, k, None)
            for k in (
                "_root",
                "_root_locator",
                "_identity_chain_source",
                "_budget_seconds",
                "_sensitive_exclusions",
            )
        )

    def check(self):
        from tldw_chatbook.Agents import local_tool_provider as local
        from tldw_chatbook.Tools import remote_workspace_executor as executor
        from tldw_chatbook.Tools import remote_workspace_transport as transport

        from . import console_chat_controller as host

        if self.cfg is not None and self.control_dir is None:
            # Another original call can finish the manager's lazy resolution
            # after this call was captured. Accept only its actual stock receipt.
            receipt = self.cfg.masters._native_control_directory
            if (
                type(receipt) is transport._ControlDirectorySource
                and receipt.configuration == self.manager_configuration
                and receipt.directory is self.cfg.masters._control_dir
            ):
                self.control_dir = receipt.directory

        if (
            self.authority is not None
            and (
                type(self.adapter) is not host._RemoteExecutorDispatchAdapter
                or self.authority.workspace_executor is not self.adapter
                or self.adapter._inner is not self.executor
                or self.adapter._alias != self.authority.alias
                or not _original_methods(self.adapter, host._NATIVE_SSH_ADAPTER_METHODS)
            )
            or type(self.executor) is not executor.RemoteWorkspaceToolExecutor
            or self.executor._ssh is not self.cfg
            or self.cfg is None
            or vars(self.cfg) != self.cfg_values
            or any(
                getattr(self.cfg, key) is not self.cfg_values[key]
                for key in ("loc", "cache", "masters", "transport", "session_spawn")
            )
            or _master_configuration(self.cfg.masters) != self.manager_configuration
            or self.cfg.masters._control_dir is not self.control_dir
            or self.cfg.transport.grace_seconds != self.transport_grace
            or self.cfg.session_spawn is not None
            or tuple(
                getattr(self.executor, k, None)
                for k in (
                    "_root",
                    "_root_locator",
                    "_identity_chain_source",
                    "_budget_seconds",
                    "_sensitive_exclusions",
                )
            )
            != self.executor_values
            or not _original_methods(self.executor, executor._NATIVE_SSH_METHODS)
            or type(self.cfg.transport) is not transport.RemoteWorkspaceTransport
            or self.cfg.transport._manager is not self.transport_manager
            or self.transport_manager is not self.cfg.masters
            or type(self.cfg.transport).__dict__.get("master_manager")
            is not transport._NATIVE_TRANSPORT_MANAGER
            or not _original_methods(
                self.cfg.transport, transport._NATIVE_TRANSPORT_METHODS
            )
            or type(self.cfg.masters) is not transport.SshMasterManager
            or not _original_methods(self.cfg.masters, transport._NATIVE_MASTER_METHODS)
            or self.authority is not None
            and self.local_specs
            and local._default_specs is not local._NATIVE_DEFAULT_SPECS
        ):
            raise RuntimeError("original SSH executor source changed")
        from tldw_chatbook.Tools import remote_binding_locator as locator
        from tldw_chatbook.Tools import remote_session_registry as sessions

        if locator.canonicalize_locator is not locator._NATIVE_CANONICALIZE:
            raise RuntimeError("original SSH destination helper changed")
        if self.registry is not None and (
            sessions._REGISTRY is not self.registry
            or type(self.registry) is not sessions.RemoteSessionRegistry
            or not _original_methods(self.registry, sessions._NATIVE_REGISTRY_METHODS)
            or sessions.get_session_registry is not sessions._NATIVE_REGISTRY_GET
        ):
            raise RuntimeError("original SSH session registry changed")
        if self.registry is not None:
            from tldw_chatbook.Tools import remote_session_worker as session

            for name, original in session._NATIVE_SESSION_FUNCTIONS.items():
                if getattr(session, name) is not original:
                    raise RuntimeError("original SSH session request callback changed")
        for name, original in executor._NATIVE_SSH_FUNCTIONS.items():
            if getattr(executor, name) is not original:
                raise RuntimeError("original SSH request callback changed")
        for name, original in transport._NATIVE_TRANSPORT_FUNCTIONS.items():
            if getattr(transport, name) is not original:
                raise RuntimeError("original SSH transport callback changed")


class _ModelSshCall:
    def __init__(self, registry, name, selection, *, virtual=False):
        from tldw_chatbook.Tools.remote_root_types import RemoteRoot

        self.registry, self.name, self.selection = registry, name, selection
        self.provider = selection[1]
        self.virtual = virtual
        self.virtual_callbacks = ()
        self.virtual_functions = ()
        self.virtual_partials = ()
        if virtual:
            require_virtual_sources(self.provider)
            from functools import partial
            from types import FunctionType, MethodType

            from tldw_chatbook.Agents.virtual_cli_provider import (
                _NATIVE_VIRTUAL_CALLBACK_FIELDS,
            )

            from .console_ordinary_tree import _function_record

            self.virtual_callbacks = tuple(
                (name, inspect.getattr_static(self.provider, name, None))
                for name in _NATIVE_VIRTUAL_CALLBACK_FIELDS
            )
            for _name, value in self.virtual_callbacks:
                function = value
                if type(value) is partial:
                    self.virtual_partials += (
                        (value, value.func, value.args, tuple(value.keywords.items())),
                    )
                    function = value.func
                if type(function) is MethodType:
                    function = function.__func__
                if type(function) is FunctionType:
                    self.virtual_functions += (_function_record(function),)
        self.generation, self.catalog = (
            registry._catalog_generation,
            registry._catalog_snapshot,
        )
        self.authorities = dict(self.provider._admitted_roots)
        self.specs = (
            {}
            if virtual
            else {
                alias: dict(items)
                for alias, items in self.provider._native_ssh_specs.items()
            }
        )
        self.virtual_registries = (
            dict(self.provider._registries_by_alias) if virtual else {}
        )
        self.sources = {
            alias: _RemoteSource(a, local_specs=not virtual)
            for alias, a in self.authorities.items()
            if isinstance(a.root, RemoteRoot)
        }
        self.source = self.operation = self.request = None
        self.requests = []
        self.master_launches = []
        self.session_starts = []
        self.finished = False
        self.uncertain = False

    def check(self):
        from tldw_chatbook.Agents import local_tool_provider as local
        from tldw_chatbook.Agents.tool_catalog import (
            _NATIVE_SSH_CATALOG_METHODS,
            ToolCatalogRegistry,
        )

        if self.virtual:
            require_virtual_sources(self.provider)
            from .console_ordinary_tree import _function_unchanged

            if (
                any(
                    inspect.getattr_static(self.provider, name, None) is not value
                    for name, value in self.virtual_callbacks
                )
                or any(
                    not _function_unchanged(record) for record in self.virtual_functions
                )
                or any(
                    callback.func is not function
                    or callback.args is not arguments
                    or len(callback.keywords) != len(keywords)
                    or any(
                        callback.keywords.get(key) is not value
                        for key, value in keywords
                    )
                    for callback, function, arguments, keywords in self.virtual_partials
                )
            ):
                raise RuntimeError("original virtual callback source changed")
        if (
            (not self.virtual and type(self.provider) is not local.LocalToolProvider)
            or type(self.registry) is not ToolCatalogRegistry
            or not _original_methods(self.registry, _NATIVE_SSH_CATALOG_METHODS)
            or self.registry._catalog_generation != self.generation
            or self.registry._catalog_snapshot is not self.catalog
            or self.registry.resolve_owner_for_name(self.name) != self.selection
            or self.provider._admitted_roots != self.authorities
            or any(
                self.provider._admitted_roots.get(alias) is not authority
                for alias, authority in self.authorities.items()
            )
            or not self.virtual
            and not _original_methods(self.provider, local._NATIVE_SSH_PROVIDER_METHODS)
        ):
            raise RuntimeError("original SSH tool authority changed")
        if self.virtual and (
            self.provider._registries_by_alias != self.virtual_registries
            or any(
                self.provider._registries_by_alias.get(alias) is not original
                for alias, original in self.virtual_registries.items()
            )
            or any(
                inspect.getattr_static(
                    self.virtual_registries[alias], "_workspace_executor", None
                )
                is not authority.workspace_executor
                for alias, authority in self.authorities.items()
            )
        ):
            raise RuntimeError("original virtual SSH registry changed")
        for source in self.sources.values():
            source.check()

    def control_directory(self, manager, directory):
        for source in self.sources.values():
            if source.cfg.masters is manager:
                if manager._control_dir is not directory or source.control_dir not in (
                    None,
                    directory,
                ):
                    raise RuntimeError("original SSH control directory changed")
                source.control_dir = directory

    def bind(self, provider, authority, spec):
        self.check()
        if provider is not self.provider:
            raise RuntimeError("original SSH provider changed")
        if authority is None or authority.alias not in self.sources:
            return
        if self.authorities.get(authority.alias) is not authority or (
            self.virtual_registries.get(authority.alias) is not spec
            if self.virtual
            else self.specs[authority.alias].get(self.name) is not spec
        ):
            raise RuntimeError("original SSH selected handler changed")
        self.source = self.sources[authority.alias]

    @contextmanager
    def activate(self, operation):
        if operation is None:
            yield
            return
        self.operation = operation
        token = _current.set(self)
        try:
            self.check()
            yield
        finally:
            if self.uncertain or any(
                not r.proven
                for r in (*self.requests, *self.master_launches, *self.session_starts)
            ):
                operation.mark_stopping()
                operation.owner.mark_cleanup_unproven()
                Event().wait()  # Unknown physical state retains the original worker/storage.
            try:
                operation.finish()
                self.finished = True
            finally:
                _current.reset(token)

    @contextmanager
    def exchange(self, executor, wire):
        self.check()
        if self.source is None or executor is not self.source.executor:
            raise RuntimeError("original SSH request executor changed")
        request = _Request(executor, wire, json.loads(wire)["operation_id"])
        self.requests.append(request)
        previous, self.request = self.request, request
        try:
            yield request
        finally:
            self.request = previous

    def response(self, request, result):
        from tldw_chatbook.Tools.remote_workspace_executor import _parse_worker_output
        from tldw_chatbook.Tools.workspace_tool_executor import (
            WorkspaceToolExecutionError,
        )

        self.check()
        if result.response is not None:
            try:
                request.terminal = _parse_worker_output(
                    result.response, expected_operation_id=request.operation_id
                )
            except WorkspaceToolExecutionError:
                pass  # Preserve the executor's existing protocol-error rendering.

    def retire_process(self, proc, threads):
        """Join actual pipes and the original new-session group on this worker."""
        from tldw_chatbook.Notes.git_process_containment import (
            OwnedProcessTree,
            ProcessTreeController,
        )

        try:
            for thread in threads:
                thread.join()
            proc.wait()
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream is not None:
                    stream.close()
            # Popen was created by the pinned stock source with start_new_session.
            tree, controller = OwnedProcessTree(proc, proc.pid), ProcessTreeController()
            with asyncio.Runner() as runner:
                if not runner.run(controller.wait(tree, timeout=float("inf"))):
                    raise OSError("SSH local group proof unavailable")
            controller.close(tree)
        except BaseException:
            self.uncertain = True
            raise

    def finite_helper(self, argv, *, timeout, text=False, errors=None):
        """Preserve ssh helper argv/config/stdin while owning its finite tree."""

        async def run():
            from tldw_chatbook.Notes.git_process_containment import (
                ProcessTreeAdmissionError,
                ProcessTreeController,
            )

            controller, tree, io = ProcessTreeController(), None, None
            try:
                try:
                    tree = await controller.spawn(
                        *argv,
                        cwd=os.getcwd(),
                        environment=os.environ.copy(),
                        stdin=None,
                    )
                except ProcessTreeAdmissionError as error:
                    tree = error.tree
                    raise
                io = asyncio.create_task(tree.process.communicate())
                deadline = asyncio.get_running_loop().time() + timeout
                try:
                    stdout, stderr = await asyncio.wait_for(asyncio.shield(io), timeout)
                except TimeoutError:
                    raise subprocess.TimeoutExpired(argv, timeout) from None
                if not await controller.wait(
                    tree, timeout=max(0, deadline - asyncio.get_running_loop().time())
                ):
                    raise subprocess.TimeoutExpired(argv, timeout)
                controller.close(tree)
                if text:
                    import locale

                    encoding = locale.getencoding()
                    stdout, stderr = (
                        part.decode(encoding, errors or "strict")
                        .replace("\r\n", "\n")
                        .replace("\r", "\n")
                        for part in (stdout, stderr)
                    )
                return subprocess.CompletedProcess(
                    argv, tree.process.returncode, stdout, stderr
                )
            except BaseException:
                if tree is not None:
                    self.operation.mark_stopping()
                    if io is None:
                        io = asyncio.create_task(tree.process.communicate())
                    try:
                        for stop in (controller.terminate, controller.kill):
                            try:
                                stop(tree)
                            except OSError:
                                pass
                            if await controller.wait(tree, timeout=0.25):
                                break
                        else:
                            if not await controller.wait(tree, timeout=float("inf")):
                                raise OSError("SSH helper cleanup unavailable")
                        await asyncio.shield(asyncio.gather(io, return_exceptions=True))
                        controller.close(tree)
                    except BaseException:  # noqa: BLE001 -- failed native proof cannot retire
                        self.uncertain = True
                raise

        self.check()
        with asyncio.Runner() as runner:
            return runner.run(run())


class _PreparationSource(_RemoteSource):
    """Exact executor issued by the originating native Console factory."""

    def __init__(self, tree, executor, binding, factory):
        super().__init__(None, executor=executor)
        self.tree, self.factory = tree, factory
        self.workspace_registry = tree.app.workspace_registry_service
        self.binding_id = binding.binding_id
        self.binding = self.workspace_registry.get_runtime_binding(self.binding_id)
        self.io = None
        self.check()

    def check(self):
        from tldw_chatbook.Tools.remote_binding_locator import locator_string

        from . import console_chat_controller as host

        self.tree.check()
        if (
            self.factory not in host._NATIVE_REMOTE_EXECUTOR_FACTORIES
            or getattr(host, self.factory.__name__, None) is not self.factory
            or self.tree.app.workspace_registry_service is not self.workspace_registry
            or self.workspace_registry.get_runtime_binding(self.binding_id)
            != self.binding
            or self.binding is None
            or self.binding.workspace_id != self.tree.workspace
            or self.cfg is None
            or self.cfg.binding_id != self.binding_id
            or locator_string(self.cfg.loc) != self.binding.locator
            or self.io is not None
            and _original_instruction_executor(self.io) is not self.executor
        ):
            raise RuntimeError("original direct SSH producer changed")
        super().check()


def register_ordinary_ssh_executor(executor, binding, factory):
    """Attach only the actual native factory result, before it can be scheduled."""
    from .console_ordinary_tree import current_ordinary_tree

    tree = current_ordinary_tree()
    if tree is not None:
        tree.borrow(tree.runtime)
        tree.ssh_sources[executor] = _PreparationSource(
            tree, executor, binding, factory
        )
    return executor


def _original_instruction_executor(io):
    from . import console_chat_controller as host

    if type(io) is not host._NativeRemoteInstructionIO or any(
        inspect.getattr_static(io, name, None) is not descriptor
        for name, descriptor in host._NATIVE_REMOTE_INSTRUCTION_IO_DESCRIPTORS.items()
    ):
        raise RuntimeError("original SSH instruction IO source changed")
    return object.__getattribute__(io, "_executor")


def capture_preparation_ssh(executor=None, *, io=None):
    from .console_ordinary_tree import current_ordinary_tree

    tree = current_ordinary_tree()
    if tree is None:
        return None
    if io is not None:
        executor = _original_instruction_executor(io)
    source = tree.ssh_sources.get(executor)
    if source is None:
        raise RuntimeError("original direct SSH issuer unavailable")
    if io is not None:
        if source.io is not None and source.io is not io:
            raise RuntimeError("original SSH instruction IO replaced")
        source.io = io
    source.check()
    return _PreparationSshCall(source)


class _PreparationSshCall(_ModelSshCall):
    """Use the existing request/helper/terminal proof for a direct producer."""

    def __init__(self, source):
        self.source = source
        self.sources = {source.binding_id: source}
        self.operation = self.request = None
        self.requests, self.master_launches, self.session_starts = [], [], []
        self.finished = self.uncertain = False

    def check(self):
        self.source.check()
        if self.source.tree.ssh_sources.get(self.source.executor) is not self.source:
            raise RuntimeError("original direct SSH source replaced")

    def probe_worker(self, callback):
        """Reserve before Thread.start; keep original source through cache return."""
        tree = self.source.tree
        operation = tree.borrow(tree.runtime).reserve_tool()
        self.operation = operation

        def owned():
            with tree.source.activate(), self.activate(operation):
                return callback()

        from tldw_chatbook.Agents.activation import worker_guard

        return worker_guard(tree.controller._agent_bridge)(owned)

    def unstarted(self):
        with self.source.tree.source.activate():
            self.operation.finish()


@dataclass(frozen=True)
class _MasterAcceptance:
    manager: object
    locator: object
    control_dir: object
    configuration: tuple
    check_argv: tuple
    launcher: object


def _master_configuration(manager):
    return (
        manager._ssh_bin,
        manager._control_persist,
        manager._enabled,
        manager._connect_timeout_s,
        manager._state_dir,
    )


@dataclass
class _MasterLaunch:
    manager: object
    locator: object
    argv: tuple
    process: object = None
    returned: bool = False
    accepted: bool = False
    never_started: bool = False

    @property
    def proven(self):
        return self.never_started or (self.returned and self.accepted)


def run_master_launcher(manager, loc, argv, timeout):
    """Observe the original launcher; successful adoption is a separate fact."""
    native = _current.get()
    if native is None:
        return subprocess.run(
            argv,
            capture_output=True,
            timeout=timeout,
            check=False,
            start_new_session=True,
        )
    native.check()
    attempt = _MasterLaunch(manager, loc, tuple(argv))
    native.master_launches.append(attempt)
    try:
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True
        )
    except OSError:
        attempt.never_started = True
        raise
    attempt.process = proc
    with proc:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise
        result = subprocess.CompletedProcess(argv, proc.wait(), stdout, stderr)
    attempt.returned = True
    return result


def accept_master(manager, loc, control_dir):
    """Consume exact finite verification into the existing manager's custody."""
    native = _current.get()
    if native is None:
        return
    from tldw_chatbook.Tools.remote_workspace_transport import _host_key

    native.check()
    key = _host_key(loc)
    config = _master_configuration(manager)
    attempt = next(
        (
            a
            for a in reversed(native.master_launches)
            if a.manager is manager and a.locator is loc
        ),
        None,
    )
    with manager._registry_lock:
        previous = manager._native_acceptances.get(key)
        reusable = (
            type(previous) is _MasterAcceptance
            and previous.manager is manager
            and previous.control_dir == control_dir
            and previous.configuration == config
            and not manager._closed
            and manager._registrations.get(key) == previous.locator
            and (attempt is None or attempt.accepted)
        )
    if reusable:
        with manager._registry_lock:
            manager._native_checks.pop(key, None)
        return
    if attempt is not None and attempt.never_started:
        return
    # An existing path, a successful -MNf or one directory entry cannot adopt.
    # _master_alive uses the original finite native helper and records this call.
    if not manager._master_alive(loc):
        native.uncertain = True
        return
    with manager._registry_lock:
        checked = manager._native_checks.pop(key, None)
        if (
            checked is None
            or checked[0] is not native
            or checked[1] is not loc
            or manager._closed
        ):
            native.uncertain = True
            return
        attempt = next(
            (
                a
                for a in reversed(native.master_launches)
                if a.manager is manager and a.locator is loc
            ),
            None,
        )
        if attempt is not None and not attempt.returned:
            native.uncertain = True
            return
        acceptance = _MasterAcceptance(
            manager, loc, control_dir, config, checked[2], attempt
        )
        manager._native_acceptances[key] = acceptance
        if attempt is not None:
            attempt.accepted = True


@dataclass(frozen=True)
class _SessionReady:
    worker: object
    process: object
    reader: object
    stderr: object
    spawn: object
    stamp: str


@dataclass(frozen=True)
class _SessionAcceptance:
    registry: object
    key: tuple
    ready: _SessionReady


@dataclass
class _SessionStart:
    registry: object
    key: tuple
    worker: object
    proven: bool = False


def record_session_ready(worker, stamp):
    worker._native_ready = _SessionReady(
        worker,
        worker._proc,
        worker._reader_thread,
        worker._stderr_thread,
        worker._spawn,
        stamp,
    )


def session_publication(registry, key, worker):
    """Pure observation at the existing registry's actual publication boundary."""
    from tldw_chatbook.Tools.remote_session_worker import (
        _NATIVE_SESSION_METHODS,
        RemoteSessionWorker,
    )

    ready = getattr(worker, "_native_ready", None)
    if (
        type(worker) is RemoteSessionWorker
        and _original_methods(worker, _NATIVE_SESSION_METHODS)
        and type(ready) is _SessionReady
        and ready.worker is worker
        and ready.process is worker._proc
        and ready.reader is worker._reader_thread
        and ready.stderr is worker._stderr_thread
        and ready.spawn is worker._spawn
        and worker._spawn is worker._native_spawn
    ):
        return _SessionAcceptance(registry, key, ready)
    return None


def starting_session(registry, key, worker):
    native = _current.get()
    if native is None:
        return
    from tldw_chatbook.Tools.remote_session_registry import _NATIVE_REGISTRY_METHODS
    from tldw_chatbook.Tools.remote_session_worker import (
        _NATIVE_SESSION_METHODS,
        RemoteSessionWorker,
    )

    native.check()
    cfg = native.source.cfg
    if (
        type(worker) is not RemoteSessionWorker
        or not _original_methods(worker, _NATIVE_SESSION_METHODS)
        or not _original_methods(registry, _NATIVE_REGISTRY_METHODS)
        or worker._transport is not cfg.transport
        or worker._loc is not cfg.loc
        or worker._python != cfg.python
        or worker._spawn is not worker._native_spawn
        or key != (cfg.session_key, cfg.binding_id)
    ):
        raise RuntimeError("original SSH session start changed")
    native.session_starts.append(_SessionStart(registry, key, worker))


def unpublished_session(registry, key, worker):
    native = _current.get()
    if native is None:
        return
    attempt = next(
        a
        for a in reversed(native.session_starts)
        if a.registry is registry and a.key == key and a.worker is worker
    )
    if worker._proc is not None:
        native.retire_process(
            worker._proc,
            tuple(
                t
                for t in (worker._reader_thread, worker._stderr_thread)
                if t is not None
            ),
        )
    attempt.proven = True


def accept_session(registry, key, worker):
    native = _current.get()
    if native is None:
        return
    from tldw_chatbook.Tools.remote_session_worker import _NATIVE_SESSION_METHODS

    native.check()
    if registry is not native.source.registry:
        raise RuntimeError("original SSH session registry changed")
    acceptance = worker._native_acceptance
    if (
        type(acceptance) is not _SessionAcceptance
        or acceptance.registry is not registry
        or acceptance.key != key
        or acceptance.ready.worker is not worker
        or acceptance.ready.process is not worker._proc
        or acceptance.ready.reader is not worker._reader_thread
        or acceptance.ready.stderr is not worker._stderr_thread
        or acceptance.ready.spawn is not worker._spawn
        or not _original_methods(worker, _NATIVE_SESSION_METHODS)
    ):
        raise RuntimeError("original SSH session acceptance unavailable")
    for attempt in native.session_starts:
        if (
            attempt.registry is registry
            and attempt.key == key
            and attempt.worker is worker
        ):
            attempt.proven = True
