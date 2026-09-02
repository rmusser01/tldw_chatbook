from __future__ import annotations

from dataclasses import replace
import hashlib
import io
from pathlib import Path
import struct
from types import MappingProxyType
import zipfile

import pytest

from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.MCP.permission_store import PermissionStoreSnapshot, definition_hash
from tldw_chatbook.Tool_Packs import export as export_module
from tldw_chatbook.Tool_Packs.catalog_snapshot import (
    PermissionInventoryAdapter,
    PermissionInventoryRegistry,
    capture_v1_inventory,
)
from tldw_chatbook.Tool_Packs.contracts import ToolPackError
from tldw_chatbook.Tool_Packs.export import (
    ToolPackExportService,
    ToolPackExportSnapshot,
    write_tool_pack_archive,
)


def _tool(*, server_key: str = "local:docs", name: str = "search") -> HubTool:
    return HubTool(
        server_key=server_key,
        server_label=server_key,
        source="builtin" if server_key == "agent:builtin" else "local",
        name=name,
        description=f"{name} description",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        tags=(),
        stale=False,
        executable=True,
    )


class _Adapter(PermissionInventoryAdapter):
    def __init__(self, namespace: str, tools: tuple[HubTool, ...]) -> None:
        self.namespace = namespace
        self.complete = True
        self._tools = tools
        self.calls = 0

    def snapshot(self) -> tuple[HubTool, ...]:
        self.calls += 1
        return self._tools


class _Store:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.calls = 0

    def read_snapshot_strict(self) -> PermissionStoreSnapshot:
        self.calls += 1
        return PermissionStoreSnapshot(
            payload=MappingProxyType(self._payload),
            generation="sha256:" + "0" * 64,
            file_exists=True,
        )


@pytest.fixture(autouse=True)
def _capture_unit_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep flattening tests independent from the sealed V1 provider assembly."""
    monkeypatch.setattr(export_module, "capture_v1_inventory", lambda registry: registry.capture())


class _LocalControlService:
    def __init__(self, inventory: object, external_servers: object) -> None:
        self.inventory = inventory
        self.external_servers = external_servers
        self.inventory_calls = 0
        self.external_server_calls = 0

    def get_inventory(self) -> object:
        self.inventory_calls += 1
        return self.inventory

    def get_external_servers(self) -> object:
        self.external_server_calls += 1
        return self.external_servers


def test_generic_registry_cannot_export_even_when_its_inventory_is_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(export_module, "capture_v1_inventory", capture_v1_inventory)
    registry = PermissionInventoryRegistry(
        current_permission_namespaces=lambda: {"local:docs"}
    )
    registry.register(_Adapter("local:docs", (_tool(),)))
    service = ToolPackExportService(
        _Store(
            {
                "schema_version": 1,
                "profiles": {"default": {"global_default": "deny", "servers": {}}},
            }
        ),
        registry,
    )

    with pytest.raises(ToolPackError, match=r"^tool_pack\.export\.inventory_incomplete$"):
        service.capture(
            profile_id="default", display_name="Default", suggested_id="default"
        )


def test_export_rejects_a_subclass_that_overrides_generic_export_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(export_module, "capture_v1_inventory", capture_v1_inventory)
    class SubvertingRegistry(PermissionInventoryRegistry):
        def capture_for_export(self):
            return self.capture()

    registry = SubvertingRegistry(current_permission_namespaces=lambda: set())
    service = ToolPackExportService(
        _Store(
            {
                "schema_version": 1,
                "profiles": {"default": {"global_default": "deny", "servers": {}}},
            }
        ),
        registry,
    )

    with pytest.raises(ToolPackError, match=r"^tool_pack\.export\.inventory_incomplete$"):
        service.capture(
            profile_id="default", display_name="Default", suggested_id="default"
        )


def test_v1_zero_tool_namespace_receives_a_portable_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(export_module, "capture_v1_inventory", capture_v1_inventory)
    local_service = _LocalControlService(
        {"tools": []},
        [
            {
                "profile_id": "empty",
                "is_connected": True,
                "discovery_snapshot": {"tools": []},
            }
        ],
    )
    registry = PermissionInventoryRegistry.v1(
        local_service,
        fallback_root=tmp_path,
    )
    review = ToolPackExportService(
        _Store(
            {
                "schema_version": 1,
                "profiles": {"default": {"global_default": "deny", "servers": {}}},
            }
        ),
        registry,
    ).capture(profile_id="default", display_name="Default", suggested_id="default")

    assert ("mcp", "local:empty", "deny") in [
        (item.authority, item.server_key, item.state)
        for item in review.payload.fallbacks
    ]
    assert (local_service.inventory_calls, local_service.external_server_calls) == (1, 1)


def test_capture_flattens_one_strict_snapshot_into_safe_portable_policy() -> None:
    """Removing the strict snapshot or resolver flattening makes this export unsafe."""
    search = _tool()
    builtin = _tool(server_key="agent:builtin", name="calculator")
    registry = PermissionInventoryRegistry(
        current_permission_namespaces=lambda: {"agent:builtin", "local:docs"}
    )
    builtin_adapter = _Adapter("agent:builtin", (builtin,))
    docs_adapter = _Adapter("local:docs", (search,))
    registry.register(builtin_adapter)
    registry.register(docs_adapter)
    store = _Store(
        {
            "schema_version": 1,
            "kill_switch": True,
            "profiles": {
                "default": {"global_default": "deny", "servers": {}},
                "research": {
                    "servers": {
                        "local:docs": {
                            "default": "allow",
                            "tools": {
                                "search": {
                                    "state": "allow",
                                    "definition_hash": definition_hash(
                                        search.description, search.input_schema
                                    ),
                                }
                            },
                        }
                    }
                },
            },
        }
    )

    review = ToolPackExportService(store, registry).capture(
        profile_id="research", display_name="Research", suggested_id="research"
    )

    assert store.calls == 1
    assert builtin_adapter.calls == docs_adapter.calls == 1
    assert [(item.server_key, item.tool_name, item.state) for item in review.payload.tools] == [
        ("agent:builtin", "calculator", "allow"),
        ("local:docs", "search", "allow"),
    ]
    assert [(item.authority, item.server_key, item.state) for item in review.payload.fallbacks] == [
        ("builtin", "agent:builtin", "ask"),
        ("mcp", "*", "deny"),
        ("mcp", "local:docs", "ask"),
    ]


def test_pending_deny_gets_a_safe_fallback_while_missing_ask_is_reported() -> None:
    registry = PermissionInventoryRegistry(current_permission_namespaces=lambda: set())
    store = _Store(
        {
            "schema_version": 1,
            "profiles": {
                "default": {"global_default": "deny", "servers": {}},
                "research": {
                    "servers": {
                        "local:missing": {
                            "tools": {
                                "gone": {"state": "deny"},
                                "review-me": {"state": "ask"},
                                "trust-me": {
                                    "state": "allow",
                                    "definition_hash": "0" * 64,
                                },
                            }
                        }
                    }
                },
            },
        }
    )

    review = ToolPackExportService(store, registry).capture(
        profile_id="research", display_name="Research", suggested_id="research"
    )

    assert review.pending_denies == (("local:missing", "gone"),)
    assert review.omitted_allow_ask == (
        ("local:missing", "review-me"),
        ("local:missing", "trust-me"),
    )
    assert [(rule.server_key, rule.tool_name, rule.state) for rule in review.payload.tools] == [
        ("local:missing", "gone", "deny")
    ]
    assert ("mcp", "local:missing", "deny") in [
        (item.authority, item.server_key, item.state)
        for item in review.payload.fallbacks
    ]


def test_shadowed_default_deny_is_not_serialized_as_a_named_pending_rule() -> None:
    registry = PermissionInventoryRegistry(current_permission_namespaces=lambda: set())
    review = ToolPackExportService(
        _Store(
            {
                "schema_version": 1,
                "profiles": {
                    "default": {
                        "global_default": "deny",
                        "servers": {
                            "local:missing": {
                                "tools": {"gone": {"state": "deny"}}
                            }
                        },
                    },
                    "research": {
                        "servers": {"local:missing": {"default": "ask"}}
                    },
                },
            }
        ),
        registry,
    ).capture(
        profile_id="research", display_name="Research", suggested_id="research"
    )

    assert review.pending_denies == ()
    assert not review.payload.tools


def test_casefold_collision_between_live_and_pending_rules_is_export_invalid() -> None:
    registry = PermissionInventoryRegistry(
        current_permission_namespaces=lambda: {"local:docs"}
    )
    registry.register(_Adapter("local:docs", (_tool(),)))
    store = _Store(
        {
            "schema_version": 1,
            "profiles": {
                "default": {"global_default": "deny", "servers": {}},
                "research": {
                    "servers": {
                        "local:DOCS": {
                            "tools": {"SEARCH": {"state": "deny"}}
                        }
                    }
                },
            },
        }
    )

    with pytest.raises(ToolPackError, match=r"^tool_pack\.export\.profile_invalid$"):
        ToolPackExportService(store, registry).capture(
            profile_id="research", display_name="Research", suggested_id="research"
        )


def test_fallback_probe_cannot_hit_a_stored_sentinel_named_tool() -> None:
    tool = _tool()
    registry = PermissionInventoryRegistry(
        current_permission_namespaces=lambda: {"local:docs"}
    )
    registry.register(_Adapter("local:docs", (tool,)))
    review = ToolPackExportService(
        _Store(
            {
                "schema_version": 1,
                "profiles": {
                    "default": {"global_default": "deny", "servers": {}},
                    "research": {
                        "servers": {
                            "local:docs": {
                                "default": "deny",
                                "tools": {
                                    "__tool_pack_unseen__": {
                                        "state": "allow",
                                        "definition_hash": "0" * 64,
                                    }
                                },
                            }
                        }
                    },
                },
            }
        ),
        registry,
    ).capture(
        profile_id="research", display_name="Research", suggested_id="research"
    )

    assert ("mcp", "local:docs", "deny") in [
        (item.authority, item.server_key, item.state)
        for item in review.payload.fallbacks
    ]


def test_global_fallback_ignores_a_literal_star_server_entry() -> None:
    review = ToolPackExportService(
        _Store(
            {
                "schema_version": 1,
                "profiles": {
                    "default": {
                        "global_default": "deny",
                        "servers": {"*": {"default": "ask"}},
                    }
                },
            }
        ),
        PermissionInventoryRegistry(current_permission_namespaces=lambda: set()),
    ).capture(profile_id="default", display_name="Default", suggested_id="default")

    assert ("mcp", "*", "deny") in [
        (item.authority, item.server_key, item.state)
        for item in review.payload.fallbacks
    ]


def test_flattening_applies_config_high_risk_and_raw_shell_floors() -> None:
    changed = _tool(name="changed")
    risky = replace(_tool(name="risky"), tags=("process",))
    shell = _tool(server_key="local:__local__", name="shell_exec")
    registry = PermissionInventoryRegistry(
        current_permission_namespaces=lambda: {"local:__local__", "local:docs"}
    )
    registry.register(_Adapter("local:docs", (changed, risky)))
    registry.register(_Adapter("local:__local__", (shell,)))
    review = ToolPackExportService(
        _Store(
            {
                "schema_version": 1,
                "profiles": {
                    "default": {"global_default": "deny", "servers": {}},
                    "research": {
                        "servers": {
                            "local:docs": {
                                "default": "allow",
                                "tools": {
                                    "changed": {
                                        "state": "allow",
                                        "definition_hash": "0" * 64,
                                    }
                                },
                            },
                            "local:__local__": {
                                "tools": {
                                    "shell_exec": {
                                        "state": "allow",
                                        "definition_hash": definition_hash(
                                            shell.description, shell.input_schema
                                        ),
                                    }
                                }
                            },
                        }
                    },
                },
            }
        ),
        registry,
    ).capture(
        profile_id="research", display_name="Research", suggested_id="research"
    )

    assert {
        (item.server_key, item.tool_name): item.state
        for item in review.payload.tools
    } == {
        ("local:__local__", "shell_exec"): "ask",
        ("local:docs", "changed"): "ask",
        ("local:docs", "risky"): "ask",
    }


def _imported_profile() -> dict[str, object]:
    return {
        "servers": {},
        "profile_kind": "tool_pack_imported",
        "tool_pack_lifecycle": {
            "schema": "tldw.tool-pack-lifecycle/v1",
            "origin": "imported",
            "pack_digest": "7" * 64,
            "imported_at": "2026-09-01T00:00:00Z",
            "first_bind_confirmation_required": True,
            "receipt_id": "tp-" + "8" * 32,
            "receipt_digest": "9" * 64,
            "policy_digest": "a" * 64,
            "revision": 1,
            "counts": {"matched": 1, "omitted": 2, "pending_deny": 3},
        },
    }


def _tombstone_profile() -> dict[str, object]:
    profile = _imported_profile()
    lifecycle = dict(profile["tool_pack_lifecycle"])  # type: ignore[arg-type]
    lifecycle.update(
        origin="tombstone",
        removed_at="2026-09-01T01:00:00Z",
        first_bind_confirmation_required=False,
    )
    lifecycle.pop("counts")
    profile["profile_kind"] = "tool_pack_tombstone"
    profile["tool_pack_lifecycle"] = lifecycle
    return profile


@pytest.mark.parametrize(
    "profile",
    [
        _tombstone_profile(),
        {"servers": {}, "profile_kind": "tool_pack_imported"},
    ],
)
def test_tombstone_or_invalid_lifecycle_profile_cannot_export(profile) -> None:
    store = _Store(
        {
            "schema_version": 1,
            "profiles": {
                "default": {"global_default": "deny", "servers": {}},
                "research": profile,
            },
        }
    )

    with pytest.raises(ToolPackError, match=r"profile_invalid$"):
        ToolPackExportService(
            store,
            PermissionInventoryRegistry(current_permission_namespaces=lambda: set()),
        ).capture(
            profile_id="research", display_name="Research", suggested_id="research"
        )


def test_nonstring_suggested_id_uses_the_stable_export_error() -> None:
    service = ToolPackExportService(
        _Store(
            {
                "schema_version": 1,
                "profiles": {
                    "default": {"global_default": "deny", "servers": {}},
                    "research": {"servers": {}},
                },
            }
        ),
        PermissionInventoryRegistry(current_permission_namespaces=lambda: set()),
    )

    with pytest.raises(ToolPackError, match=r"^tool_pack\.export\.profile_invalid$"):
        service.capture(
            profile_id="research",
            display_name="Research",
            suggested_id=None,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("profile_id", ["", " ", "\tinvalid"])
def test_invalid_profile_id_uses_the_stable_export_error(profile_id: str) -> None:
    service = ToolPackExportService(
        _Store(
            {
                "schema_version": 1,
                "profiles": {
                    "default": {"global_default": "deny", "servers": {}},
                    profile_id: {"servers": {}},
                },
            }
        ),
        PermissionInventoryRegistry(current_permission_namespaces=lambda: set()),
    )

    with pytest.raises(ToolPackError, match=r"^tool_pack\.export\.profile_invalid$"):
        service.capture(
            profile_id=profile_id,
            display_name="Research",
            suggested_id="research",
        )


def test_unhashable_profile_id_uses_the_stable_export_error() -> None:
    service = ToolPackExportService(
        _Store(
            {
                "schema_version": 1,
                "profiles": {"default": {"global_default": "deny", "servers": {}}},
            }
        ),
        PermissionInventoryRegistry(current_permission_namespaces=lambda: set()),
    )

    with pytest.raises(ToolPackError, match=r"^tool_pack\.export\.profile_invalid$"):
        service.capture(
            profile_id=[],  # type: ignore[arg-type]
            display_name="Research",
            suggested_id="research",
        )


def test_reserved_workspace_source_uses_generic_id_and_omits_receipt_history() -> None:
    workspace_profile = _imported_profile()
    review = ToolPackExportService(
        _Store(
            {
                "schema_version": 1,
                "kill_switch": True,
                "updated_at": "private-runtime-sentinel",
                "profiles": {
                    "default": {"global_default": "deny", "servers": {}},
                    "ws-private-workspace-sentinel": workspace_profile,
                },
            }
        ),
        PermissionInventoryRegistry(current_permission_namespaces=lambda: set()),
    ).capture(
        profile_id="ws-private-workspace-sentinel",
        display_name="Portable policy",
        suggested_id="private-workspace-sentinel",
    )
    output = io.BytesIO()
    write_tool_pack_archive(review.snapshot, output)

    assert review.snapshot.manifest.suggested_id == "tool-profile"
    for forbidden in (
        b"private-workspace-sentinel",
        b"private-runtime-sentinel",
        b"tp-88888888888888888888888888888888",
        b"2026-09-01T00:00:00Z",
    ):
        assert forbidden not in output.getvalue()


def test_archive_has_pinned_two_member_zip_headers_and_bytes() -> None:
    """Changing ZIP defaults, order, or metadata must change this observable archive."""
    tool = _tool()
    registry = PermissionInventoryRegistry(
        current_permission_namespaces=lambda: {"local:docs"}
    )
    registry.register(_Adapter("local:docs", (tool,)))
    review = ToolPackExportService(
        _Store(
            {
                "schema_version": 1,
                "kill_switch": False,
                "profiles": {"default": {"global_default": "deny", "servers": {}}},
            }
        ),
        registry,
    ).capture(profile_id="default", display_name="Default", suggested_id="default")

    first, second = io.BytesIO(), io.BytesIO()
    first_digest = write_tool_pack_archive(review.snapshot, first)
    second_digest = write_tool_pack_archive(review.snapshot, second)

    assert first.getvalue() == second.getvalue()
    assert first_digest == second_digest == hashlib.sha256(first.getvalue()).hexdigest()
    with zipfile.ZipFile(io.BytesIO(first.getvalue())) as archive:
        assert archive.namelist() == ["tool-pack.json", "profile/profile.json"]
        assert archive.comment == b""
        for member in archive.infolist():
            assert (
                member.compress_type,
                member.date_time,
                member.create_system,
                member.create_version,
                member.extract_version,
                member.flag_bits,
                member.external_attr,
                member.extra,
                member.comment,
            ) == (zipfile.ZIP_STORED, (1980, 1, 1, 0, 0, 0), 3, 20, 20, 0, 0o100644 << 16, b"", b"")
            assert member.compress_size == member.file_size
            local_flags = struct.unpack_from("<H", first.getvalue(), member.header_offset + 6)[0]
            assert local_flags & 0x08 == 0


def test_archive_writer_rejects_mismatched_snapshot_and_short_or_failed_sink() -> None:
    tool = _tool()
    registry = PermissionInventoryRegistry(
        current_permission_namespaces=lambda: {"local:docs"}
    )
    registry.register(_Adapter("local:docs", (tool,)))
    review = ToolPackExportService(
        _Store(
            {
                "schema_version": 1,
                "profiles": {"default": {"global_default": "deny", "servers": {}}},
            }
        ),
        registry,
    ).capture(profile_id="default", display_name="Default", suggested_id="default")
    mismatched = ToolPackExportSnapshot(
        replace(
            review.snapshot.manifest,
            payload_size=review.snapshot.manifest.payload_size + 1,
        ),
        review.snapshot.payload,
    )

    with pytest.raises(ToolPackError, match=r"profile_invalid$"):
        write_tool_pack_archive(mismatched, io.BytesIO())

    class ShortSink:
        def write(self, data: bytes) -> int:
            return len(data) - 1

    class FailedSink:
        def write(self, data: bytes) -> int:
            raise OSError("private destination")

    class PartialSink:
        def __init__(self) -> None:
            self.data = bytearray()

        def write(self, data: bytes) -> int:
            written = max(1, len(data) // 2)
            self.data.extend(data[:written])
            return written

    with pytest.raises(ToolPackError, match=r"publication_failed$"):
        write_tool_pack_archive(review.snapshot, ShortSink())
    with pytest.raises(ToolPackError, match=r"publication_failed$"):
        write_tool_pack_archive(review.snapshot, FailedSink())
    partial = PartialSink()
    digest = write_tool_pack_archive(review.snapshot, partial)
    assert digest == hashlib.sha256(partial.data).hexdigest()
    with zipfile.ZipFile(io.BytesIO(partial.data)) as archive:
        assert archive.namelist() == ["tool-pack.json", "profile/profile.json"]


def test_minimal_archive_matches_the_checked_in_golden_bytes() -> None:
    """A platform ZIP default must not silently replace our canonical archive."""
    fixture_root = Path(__file__).with_name("fixtures")
    expected = (fixture_root / "minimal-tool-pack.bytes").read_bytes()
    expected_digest = (fixture_root / "minimal-tool-pack.sha256").read_text().strip()
    tool = _tool()
    registry = PermissionInventoryRegistry(
        current_permission_namespaces=lambda: {"local:docs"}
    )
    registry.register(_Adapter("local:docs", (tool,)))
    review = ToolPackExportService(
        _Store(
            {
                "schema_version": 1,
                "kill_switch": False,
                "profiles": {"default": {"global_default": "deny", "servers": {}}},
            }
        ),
        registry,
    ).capture(profile_id="default", display_name="Default", suggested_id="default")
    actual = io.BytesIO()

    assert zipfile.is_zipfile(io.BytesIO(expected))
    assert write_tool_pack_archive(review.snapshot, actual) == expected_digest
    assert actual.getvalue() == expected
