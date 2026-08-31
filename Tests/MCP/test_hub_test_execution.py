"""Pure-unit coverage for Hub Test Tool admission primitives."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path
from typing import Any

import pytest

from tldw_chatbook.MCP import hub_test_execution
from tldw_chatbook.MCP.hub_test_execution import (
    RegisteredToolTestPreview,
    ToolTestAdmissionPreview,
    ToolTestPreviewRegistry,
    authority_fingerprint,
    canonicalize_arguments,
)
from tldw_chatbook.Utils.filesystem_identity import DirectoryChain, DirectoryIdentity


def _identity(
    *,
    device: int = 11,
    inode: int = 22,
    mode: int = 0o40700,
    reparse: bool = False,
) -> DirectoryIdentity:
    return DirectoryIdentity(
        device=device,
        inode=inode,
        mode=mode,
        reparse=reparse,
    )


def _chain(
    root: str = "/private/workspace",
    identities: tuple[DirectoryIdentity, ...] | None = None,
) -> DirectoryChain:
    return DirectoryChain(
        canonical_root=Path(root),
        identities=identities or (_identity(), _identity(device=33, inode=44)),
    )


def _issue(
    registry: ToolTestPreviewRegistry,
    *,
    tool_name: str = "fs_read",
    authority: DirectoryChain | None = None,
) -> ToolTestAdmissionPreview:
    return registry.issue(
        server_key="local:__local__",
        tool_name=tool_name,
        definition_hash="definition-v1",
        rendered_gate="ask",
        authority=authority,
        safe_authority_label="Selected workspace" if authority else None,
    )


def test_canonical_arguments_round_trip_nested_json_without_changing_scalar_types():
    arguments = {
        "bool": True,
        "integer": 7,
        "float": 2.5,
        "nested": {"items": [None, False, -4, 1.25, "text"]},
    }

    encoded, dispatch = canonicalize_arguments(arguments)

    assert encoded == (
        b'{"bool":true,"float":2.5,"integer":7,'
        b'"nested":{"items":[null,false,-4,1.25,"text"]}}'
    )
    assert dispatch == arguments
    assert dispatch["bool"] is True
    assert type(dispatch["integer"]) is int
    assert type(dispatch["float"]) is float


def test_canonical_arguments_return_a_dispatch_copy_independent_of_the_caller():
    arguments = {"outer": {"items": ["original"]}}

    encoded, dispatch = canonicalize_arguments(arguments)
    arguments["outer"]["items"].append("caller mutation")
    dispatch["outer"]["items"].append("dispatch mutation")

    assert encoded == b'{"outer":{"items":["original"]}}'
    assert arguments == {"outer": {"items": ["original", "caller mutation"]}}
    assert dispatch == {"outer": {"items": ["original", "dispatch mutation"]}}


@pytest.mark.parametrize("value", [None, [], ["not", "an", "object"], "text", 3])
def test_canonical_arguments_reject_top_level_non_objects(value: Any):
    with pytest.raises(ValueError, match="JSON object"):
        canonicalize_arguments(value)


@pytest.mark.parametrize(
    "arguments",
    [
        {1: "non-string key"},
        {"nested": {2: "non-string key"}},
        {"value": object()},
        {"value": ("tuple",)},
    ],
)
def test_canonical_arguments_reject_non_string_keys_and_non_json_values(arguments):
    with pytest.raises(ValueError, match="JSON"):
        canonicalize_arguments(arguments)


@pytest.mark.parametrize("number", [float("nan"), float("inf"), float("-inf")])
def test_canonical_arguments_reject_non_finite_numbers(number: float):
    with pytest.raises(ValueError, match="finite"):
        canonicalize_arguments({"number": number})


def test_authority_fingerprint_is_deterministic_for_identical_full_chains():
    left = _chain()
    right = _chain(
        identities=(
            _identity(),
            _identity(device=33, inode=44),
        )
    )

    assert authority_fingerprint(left) == authority_fingerprint(right)


@pytest.mark.parametrize(
    "changed",
    [
        _chain("/private/different-workspace"),
        _chain(identities=(_identity(device=12), _identity(device=33, inode=44))),
        _chain(identities=(_identity(inode=23), _identity(device=33, inode=44))),
        _chain(identities=(_identity(mode=0o40500), _identity(device=33, inode=44))),
        _chain(identities=(_identity(reparse=True), _identity(device=33, inode=44))),
        _chain(identities=(_identity(), _identity(device=33, inode=45))),
    ],
)
def test_authority_fingerprint_binds_locator_and_every_identity_field(changed):
    assert authority_fingerprint(_chain()) != authority_fingerprint(changed)


def test_public_preview_is_frozen_slotted_and_never_exposes_raw_authority():
    chain = _chain()
    registry = ToolTestPreviewRegistry(max_entries=4, ttl_seconds=30)

    preview = _issue(registry, authority=chain)

    assert isinstance(preview, ToolTestAdmissionPreview)
    assert not hasattr(preview, "__dict__")
    assert "canonical_root" not in asdict(preview)
    assert "identities" not in asdict(preview)
    assert str(chain.canonical_root) not in repr(preview)
    with pytest.raises(FrozenInstanceError):
        preview.tool_name = "fs_write"  # type: ignore[misc]


def test_registry_mints_opaque_nonce_and_retains_authority_only_privately(monkeypatch):
    monkeypatch.setattr(hub_test_execution.secrets, "token_urlsafe", lambda: "opaque")
    chain = _chain()
    registry = ToolTestPreviewRegistry(max_entries=4, ttl_seconds=30)

    preview = _issue(registry, authority=chain)
    registered = registry.consume(preview.nonce)

    assert preview.nonce == "opaque"
    assert preview.server_key not in preview.nonce
    assert preview.tool_name not in preview.nonce
    assert preview.authority_fingerprint == authority_fingerprint(chain)
    assert isinstance(registered, RegisteredToolTestPreview)
    assert registered.authority is chain
    assert registered.public is preview


def test_successful_consume_is_single_use_and_removes_before_returning():
    registry = ToolTestPreviewRegistry(max_entries=4, ttl_seconds=30)
    preview = _issue(registry)

    first = registry.consume(preview.nonce)

    assert first is not None
    assert registry.consume(preview.nonce) is None


def test_revoke_removes_preview():
    registry = ToolTestPreviewRegistry(max_entries=4, ttl_seconds=30)
    preview = _issue(registry)

    registry.revoke(preview.nonce)

    assert registry.consume(preview.nonce) is None


def test_expired_preview_is_removed_and_unavailable(monkeypatch):
    now = 100.0
    monkeypatch.setattr(hub_test_execution.time, "monotonic", lambda: now)
    registry = ToolTestPreviewRegistry(max_entries=4, ttl_seconds=5)
    preview = _issue(registry)

    now = 105.0

    assert registry.consume(preview.nonce) is None
    assert registry.consume(preview.nonce) is None


def test_capacity_evicts_the_oldest_preview():
    registry = ToolTestPreviewRegistry(max_entries=2, ttl_seconds=30)
    oldest = _issue(registry, tool_name="first")
    middle = _issue(registry, tool_name="second")

    newest = _issue(registry, tool_name="third")

    assert registry.consume(oldest.nonce) is None
    assert registry.consume(middle.nonce) is not None
    assert registry.consume(newest.nonce) is not None


def test_concurrent_consume_has_exactly_one_winner():
    registry = ToolTestPreviewRegistry(max_entries=4, ttl_seconds=30)
    preview = _issue(registry)

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(registry.consume, [preview.nonce] * 64))

    assert sum(result is not None for result in results) == 1


def test_clear_removes_every_preview():
    registry = ToolTestPreviewRegistry(max_entries=4, ttl_seconds=30)
    previews = [_issue(registry, tool_name=f"tool-{index}") for index in range(4)]

    registry.clear()

    assert all(registry.consume(preview.nonce) is None for preview in previews)
