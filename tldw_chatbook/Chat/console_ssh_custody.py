"""Attach native SSH custody to the original Console preparation sources."""

from __future__ import annotations

import inspect

from tldw_chatbook.Tools.workspace_tool_executor import (
    ModelSshCall,
    RemoteSshSource,
    bind_model_ssh as bind_model_ssh,
    capture_model_ssh as capture_model_ssh,
    current_ssh_call as current_ssh_call,
)


class _PreparationSource(RemoteSshSource):
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


class _PreparationSshCall(ModelSshCall):
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
