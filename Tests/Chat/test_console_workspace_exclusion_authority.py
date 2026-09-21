"""Run-admitted exclusion authority: snapshot freeze + high-water semantics."""
from __future__ import annotations

from pathlib import Path

import pytest

from tldw_chatbook.Chat.console_chat_controller import (
    _exclusion_paths_provider,
    capture_run_admitted_workspace_roots,
)


class _FakeBinding:
    def __init__(self, workspace_id, binding_id, locator, exclusions=()):
        from tldw_chatbook.Workspaces.models import RuntimeBindingKind, RuntimeBindingStatus

        self.workspace_id = workspace_id
        self.binding_id = binding_id
        self.binding_kind = RuntimeBindingKind.LOCAL_FILESYSTEM
        self.status = RuntimeBindingStatus.READY
        self.locator = locator
        self.metadata = {
            "access": "rw",
            "exclusions": [{"path": p, "kind": "directory", "added_at": ""} for p in exclusions],
        }


class _FakeRegistry:
    def __init__(self, bindings):
        self._bindings = {b.binding_id: b for b in bindings}

    def list_runtime_bindings(self, workspace_id):
        return tuple(b for b in self._bindings.values() if b.workspace_id == workspace_id)

    def get_runtime_binding(self, binding_id):
        return self._bindings.get(binding_id)


class _FakeSession:
    def __init__(self, workspace_id):
        self.workspace_id = workspace_id


def test_snapshot_frozen_at_admission(tmp_path: Path):
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    registry = _FakeRegistry([_FakeBinding("ws", "folder-1", root, ("secrets",))])
    roots = capture_run_admitted_workspace_roots(session=_FakeSession("ws"), registry=registry)
    assert len(roots) == 1
    paths = roots[0].exclusions_provider()
    assert paths == ((root / "secrets").resolve(strict=False),)


def test_mid_run_addition_applies_immediately(tmp_path: Path):
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    binding = _FakeBinding("ws", "folder-1", root)
    registry = _FakeRegistry([binding])
    roots = capture_run_admitted_workspace_roots(session=_FakeSession("ws"), registry=registry)
    provider = roots[0].exclusions_provider
    assert provider() == ()
    binding.metadata["exclusions"] = [
        {"path": "secrets", "kind": "directory", "added_at": ""}
    ]
    assert (root / "secrets").resolve(strict=False) in provider()


def test_mid_run_removal_stays_excluded_high_water(tmp_path: Path):
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    binding = _FakeBinding("ws", "folder-1", root, ("secrets",))
    registry = _FakeRegistry([binding])
    roots = capture_run_admitted_workspace_roots(session=_FakeSession("ws"), registry=registry)
    provider = roots[0].exclusions_provider
    provider()  # observe live set once
    binding.metadata["exclusions"] = []
    assert (root / "secrets").resolve(strict=False) in provider()


def test_registry_failure_reuses_last_known(tmp_path: Path):
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    binding = _FakeBinding("ws", "folder-1", root, ("secrets",))
    registry = _FakeRegistry([binding])

    def boom(binding_id):
        raise RuntimeError("registry down")

    provider = _exclusion_paths_provider(registry, "folder-1", root, ("secrets",))
    provider()
    registry.get_runtime_binding = boom
    assert (root / "secrets").resolve(strict=False) in provider()


def test_provider_skips_unresolvable_entry_and_keeps_rest(tmp_path: Path):
    """Finding 2a: one unresolvable exclusion must not kill the provider call.

    A self-referential symlink under the binding root makes resolving that
    one relative exclusion raise; the provider must skip it (warning) and
    still return the resolvable rest rather than propagating the error and
    zeroing the effective set for the caller.
    """
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    (root / "secrets").mkdir()
    loop = root / "loop"
    loop.symlink_to(loop)
    registry = _FakeRegistry([_FakeBinding("ws", "folder-1", root, ("secrets", "loop"))])

    provider = _exclusion_paths_provider(
        registry, "folder-1", root, ("secrets", "loop")
    )
    paths = provider()

    assert paths == ((root / "secrets").resolve(strict=False),)


def test_provider_refuses_excluded_path_end_to_end(tmp_path: Path):
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    (root / "secrets").mkdir()
    (root / "secrets" / "key.pem").write_text("k")
    registry = _FakeRegistry([_FakeBinding("ws", "folder-1", root, ("secrets",))])
    roots = capture_run_admitted_workspace_roots(session=_FakeSession("ws"), registry=registry)
    from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
    from tldw_chatbook.MCP.permission_store import EffectiveToolState

    provider = LocalToolProvider(
        workspace_root=tmp_path,
        resolve_state=lambda hub: EffectiveToolState.ALLOW,
        admitted_roots=roots,
    )
    result = provider._invoke_detailed("local:fs_read", {"path": "secrets/key.pem", "root_alias": "folder-1"})
    assert result.result.ok is False
    # Controller ruling: parent-side denials wrap as an opaque executor code,
    # so the refusal text must leak neither the exclusion mechanism nor the
    # excluded path itself.
    assert "exclud" not in result.result.error.lower()
    assert "secrets" not in result.result.error


def test_path_targets_preflight_refuses_excluded_target(tmp_path: Path):
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    (root / "secrets").mkdir()
    (root / "secrets" / "key.pem").write_text("k")
    registry = _FakeRegistry([_FakeBinding("ws", "folder-1", root, ("secrets",))])
    roots = capture_run_admitted_workspace_roots(session=_FakeSession("ws"), registry=registry)
    from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
    from tldw_chatbook.MCP.permission_store import EffectiveToolState
    from tldw_chatbook.Tools.local_tool_impls import LocalToolError

    provider = LocalToolProvider(
        workspace_root=tmp_path,
        resolve_state=lambda hub: EffectiveToolState.ALLOW,
        admitted_roots=roots,
    )
    with pytest.raises(LocalToolError):
        provider.path_targets("local:fs_read", {"path": "secrets/key.pem", "root_alias": "folder-1"})
