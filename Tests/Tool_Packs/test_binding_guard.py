"""First-bind Tool Profile confirmation at the workspace authority boundary."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import threading

import pytest

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.MCP.permission_store import (
    MCPPermissionStore,
    definition_hash,
)
from tldw_chatbook.Tool_Packs import binding as binding_module
from tldw_chatbook.Tool_Packs.catalog_snapshot import (
    PermissionInventorySnapshot,
    PermissionInventoryTool,
)
from tldw_chatbook.Workspaces.models import WorkspaceAssistantDefaults
from tldw_chatbook.Workspaces.registry_service import (
    LocalWorkspaceRegistryService,
    WorkspaceRegistryServiceError,
)


def _tool(
    server_key: str,
    name: str,
    *,
    description: str = "",
    tags: tuple[str, ...] = (),
) -> HubTool:
    return HubTool(
        server_key=server_key,
        server_label=server_key,
        source="local",
        name=name,
        description=description,
        input_schema={"type": "object"},
        tags=tags,
        stale=False,
        executable=True,
    )


def _inventory(*tools: HubTool) -> PermissionInventorySnapshot:
    entries = tuple(
        PermissionInventoryTool(
            "builtin" if tool.server_key == "agent:builtin" else "mcp",
            tool,
            "f" * 64,
        )
        for tool in tools
    )
    namespaces = tuple(
        sorted(
            {
                (
                    "builtin" if tool.server_key == "agent:builtin" else "mcp",
                    tool.server_key,
                )
                for tool in tools
            }
        )
    )
    return PermissionInventorySnapshot(entries, namespaces, (), "i" * 64)


def _imported_profile() -> dict[str, object]:
    exact = _tool("local:docs", "exact", description="exact")
    stale = _tool("local:docs", "stale", description="old")
    profile: dict[str, object] = {
        "global_default": "deny",
        "servers": {
            "agent:builtin": {"default": "ask"},
            "local:docs": {
                "default": "ask",
                "tools": {
                    "exact": {
                        "state": "allow",
                        "definition_hash": definition_hash(
                            exact.description, exact.input_schema
                        ),
                    },
                    "stale": {
                        "state": "allow",
                        "definition_hash": definition_hash(
                            stale.description, stale.input_schema
                        ),
                    },
                    "missing": {
                        "state": "allow",
                        "definition_hash": "a" * 64,
                    },
                },
            },
            "local:fallback": {"default": "allow"},
        },
        "profile_kind": "tool_pack_imported",
    }
    profile["tool_pack_lifecycle"] = {
        "schema": "tldw.tool-pack-lifecycle/v1",
        "origin": "imported",
        "pack_digest": "b" * 64,
        "imported_at": "2026-08-31T00:00:00Z",
        "first_bind_confirmation_required": True,
        "receipt_id": f"tp-{'c' * 32}",
        "receipt_digest": "d" * 64,
        "counts": {"matched": 3, "omitted": 0, "pending_deny": 0},
        "policy_digest": binding_module.profile_policy_digest(profile),
        "revision": 1,
    }
    return profile


def _install(store: MCPPermissionStore, profile_id: str = "research") -> None:
    store.install_profile_if_absent(
        profile_id,
        _imported_profile(),
        expected_generation=store.read_snapshot_strict().generation,
        max_profiles=128,
        max_store_bytes=8 * 1024 * 1024,
    )


def _tombstone_profile() -> dict[str, object]:
    profile: dict[str, object] = {
        "global_default": "deny",
        "servers": {"agent:builtin": {"default": "deny"}},
        "profile_kind": "tool_pack_tombstone",
    }
    profile["tool_pack_lifecycle"] = {
        "schema": "tldw.tool-pack-lifecycle/v1",
        "origin": "tombstone",
        "pack_digest": "b" * 64,
        "imported_at": "2026-08-31T00:00:00Z",
        "removed_at": "2026-09-01T00:00:00Z",
        "first_bind_confirmation_required": False,
        "receipt_id": f"tp-{'e' * 32}",
        "receipt_digest": "f" * 64,
        "policy_digest": binding_module.profile_policy_digest(profile),
        "revision": 2,
    }
    return profile


def _defaults(
    profile_id: str | None = "research",
    *,
    assistant_id: str = "persona-1",
    memory: str = "read_only",
) -> WorkspaceAssistantDefaults:
    return WorkspaceAssistantDefaults(
        assistant_id=assistant_id,
        persona_memory_mode=memory,
        tool_policy_profile_id=profile_id,
    )


class _Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 9, 1, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.value


def _category(error: pytest.ExceptionInfo[BaseException]) -> str | None:
    return getattr(error.value, "category", None)


def _build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    create_workspace: bool = True,
):
    store = MCPPermissionStore(tmp_path / "permissions.json")
    _install(store)
    registry = LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="c1")
    )
    if create_workspace:
        registry.create_workspace(workspace_id="w-1", name="One")
    clock = _Clock()
    inventory = _inventory(
        _tool("local:docs", "exact", description="exact", tags=("mutates",)),
        _tool("local:docs", "stale", description="changed", tags=("mutates",)),
        _tool("local:fallback", "inherited", tags=("mutates",)),
    )
    monkeypatch.setattr(
        binding_module,
        "capture_v1_inventory",
        lambda _registry: inventory,
        raising=False,
    )
    guard_type = getattr(binding_module, "ToolProfileBindingGuard")
    guard = guard_type(
        permission_store=store,
        inventory=object(),
        workspace_defaults_reader=lambda workspace_id: (
            record.assistant_defaults
            if (record := registry.get_workspace(workspace_id)) is not None
            else None
        ),
        now=clock,
    )
    registry.attach_tool_profile_guard(guard)
    return registry, store, guard, clock


def _lifecycle(store: MCPPermissionStore, profile_id: str = "research"):
    return store.read_snapshot_strict().payload["profiles"][profile_id][
        "tool_pack_lifecycle"
    ]


def test_direct_set_requires_confirmation_and_confirmed_token_binds_once(
    tmp_path, monkeypatch
):
    """Removing the registry guard would let direct service calls bind imported policy."""
    registry, store, guard, _clock = _build(tmp_path, monkeypatch)
    intended = _defaults()

    with pytest.raises(ValueError) as missing:
        registry.set_assistant_defaults("w-1", intended)
    assert _category(missing) == "confirmation_required"

    review = guard.review("w-1", intended, action="set")
    token = guard.confirm(review)
    updated = registry.set_assistant_defaults(
        "w-1", intended, tool_profile_confirmation_token=token
    )
    assert updated.assistant_defaults == intended
    assert _lifecycle(store)["first_bind_confirmation_required"] is False

    with pytest.raises(ValueError) as replayed:
        registry.set_assistant_defaults(
            "w-1", intended, tool_profile_confirmation_token=token
        )
    assert _category(replayed) == "confirmation_invalid"


def test_inline_create_requires_a_token_before_any_workspace_row_is_written(
    tmp_path, monkeypatch
):
    """Moving the guard after INSERT would leave an unauthorized workspace binding."""
    registry, _store, guard, _clock = _build(
        tmp_path, monkeypatch, create_workspace=False
    )
    intended = _defaults()

    with pytest.raises(ValueError) as missing:
        registry.create_workspace(
            workspace_id="w-new", name="New", assistant_defaults=intended
        )
    assert _category(missing) == "confirmation_required"
    assert registry.get_workspace("w-new") is None

    token = guard.confirm(guard.review("w-new", intended, action="create"))
    created = registry.create_workspace(
        workspace_id="w-new",
        name="New",
        assistant_defaults=intended,
        tool_profile_confirmation_token=token,
    )
    assert created.assistant_defaults == intended


@pytest.mark.parametrize(
    ("memory_ack", "tool_ack", "succeeds"),
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (True, True, True),
    ],
)
def test_inline_create_keeps_memory_and_tool_acknowledgements_independent(
    tmp_path, monkeypatch, memory_ack, tool_ack, succeeds
):
    """Inline create must not bypass the memory gate that set already enforces."""
    registry, _store, guard, _clock = _build(
        tmp_path, monkeypatch, create_workspace=False
    )
    intended = _defaults(memory="read_write")
    token = (
        guard.confirm(guard.review("w-new", intended, action="create"))
        if tool_ack
        else None
    )

    def call():
        return registry.create_workspace(
            workspace_id="w-new",
            name="New",
            assistant_defaults=intended,
            confirm_read_write=memory_ack,
            tool_profile_confirmation_token=token,
        )

    if succeeds:
        assert call().assistant_defaults == intended
    else:
        expected_error = WorkspaceRegistryServiceError if not memory_ack else ValueError
        with pytest.raises(expected_error) as rejected:
            call()
        if memory_ack:
            assert _category(rejected) == "confirmation_required"
        assert registry.get_workspace("w-new") is None


def test_review_recomputes_current_fallback_allow_and_risk_posture(
    tmp_path, monkeypatch
):
    """Trusting a receipt would hide current missing, changed, and high-risk Allows."""
    _registry, _store, guard, _clock = _build(tmp_path, monkeypatch)

    summary = guard.review("w-1", _defaults(), action="set").summary

    assert summary.global_fallback == "deny"
    assert summary.builtin_fallback == "ask"
    assert summary.allow_server_fallbacks == ("local:fallback",)
    assert ("local:docs", "missing") in summary.unavailable_allows
    assert ("local:docs", "stale") in summary.downgraded_allows
    assert ("local:docs", "exact") in summary.effective_allows
    assert ("local:docs", "exact") in summary.high_risk_allows
    assert ("local:docs", "stale") in summary.high_risk_allows
    assert ("local:fallback", "inherited") in summary.high_risk_allows
    assert len(summary.effective_asks) == summary.ask_count
    assert len(summary.effective_denies) == summary.deny_count
    assert {
        *summary.effective_allows,
        *summary.effective_asks,
        *summary.effective_denies,
    } == {
        ("local:docs", "exact"),
        ("local:docs", "stale"),
        ("local:fallback", "inherited"),
    }


def test_abandoned_confirmation_state_is_pruned_and_bounded(tmp_path, monkeypatch):
    """Cancelled reviews and unspent tokens must not grow for process lifetime."""
    monkeypatch.setattr(binding_module, "_MAX_PENDING_CONFIRMATIONS", 2)
    registry, _store, guard, clock = _build(tmp_path, monkeypatch)
    intended = _defaults()

    first = guard.review("w-1", intended, action="set")
    second = guard.review("w-1", intended, action="set")
    third = guard.review("w-1", intended, action="set")
    assert len(guard._reviews) == 2
    with pytest.raises(ValueError) as evicted_review:
        guard.confirm(first)
    assert _category(evicted_review) == "confirmation_invalid"

    first_token = guard.confirm(second)
    guard.confirm(third)
    fourth = guard.review("w-1", intended, action="set")
    guard.confirm(fourth)
    assert len(guard._tokens) == 2
    with pytest.raises(ValueError) as evicted_token:
        registry.set_assistant_defaults(
            "w-1", intended, tool_profile_confirmation_token=first_token
        )
    assert _category(evicted_token) == "confirmation_invalid"

    abandoned = guard.review("w-1", intended, action="set")
    abandoned_token = guard.confirm(abandoned)
    unconfirmed = guard.review("w-1", intended, action="set")
    clock.value += timedelta(minutes=11)
    fresh = guard.review("w-1", intended, action="set")
    assert abandoned_token not in guard._tokens
    assert id(unconfirmed) not in guard._reviews
    assert set(guard._reviews) == {id(fresh)}


def test_token_ttl_starts_when_confirmation_is_issued(tmp_path, monkeypatch):
    """A delayed confirmation still grants the full ten-minute token lifetime."""
    registry, _store, guard, clock = _build(tmp_path, monkeypatch)
    intended = _defaults()
    review = guard.review("w-1", intended, action="set")
    clock.value += timedelta(minutes=9)
    token = guard.confirm(review)
    clock.value += timedelta(minutes=2)

    assert (
        registry.set_assistant_defaults(
            "w-1", intended, tool_profile_confirmation_token=token
        ).assistant_defaults
        == intended
    )


@pytest.mark.parametrize(
    "mutation",
    ["defaults", "workspace", "action", "policy", "expiry", "removal"],
)
def test_confirmation_is_bound_to_every_review_axis(tmp_path, monkeypatch, mutation):
    """Dropping any token axis would let a materially different bind reuse approval."""
    registry, store, guard, clock = _build(tmp_path, monkeypatch)
    registry.create_workspace(workspace_id="w-2", name="Two")
    intended = _defaults()
    token = guard.confirm(guard.review("w-1", intended, action="set"))
    workspace_id = "w-1"
    submitted = intended

    if mutation == "defaults":
        submitted = _defaults(assistant_id="persona-2")
    elif mutation == "workspace":
        workspace_id = "w-2"
    elif mutation == "action":
        registry.set_assistant_defaults("w-1", _defaults("local-profile"))
    elif mutation == "policy":
        store.set_server_default("local:docs", "deny", profile_id="research")
    elif mutation == "expiry":
        clock.value += timedelta(minutes=11)
    elif mutation == "removal":
        store.replace_profile_with_tombstone(
            "research",
            _tombstone_profile(),
            expected_revision=1,
            expected_generation=store.read_snapshot_strict().generation,
            max_store_bytes=8 * 1024 * 1024,
        )

    with pytest.raises(ValueError) as stale:
        registry.set_assistant_defaults(
            workspace_id,
            submitted,
            tool_profile_confirmation_token=token,
        )
    assert _category(stale) in {
        "confirmation_stale",
        "confirmation_expired",
        "confirmation_invalid",
        "lifecycle_invalid",
    }
    assert registry.get_workspace(workspace_id).assistant_defaults != submitted


def test_stale_commit_attempt_consumes_the_one_use_token(tmp_path, monkeypatch):
    """A stale token must not become valid again after workspace state is restored."""
    registry, _store, guard, _clock = _build(tmp_path, monkeypatch)
    intended = _defaults()
    token = guard.confirm(guard.review("w-1", intended, action="set"))
    registry.set_assistant_defaults("w-1", _defaults("local-profile"))

    with pytest.raises(ValueError) as stale:
        registry.set_assistant_defaults(
            "w-1", intended, tool_profile_confirmation_token=token
        )
    assert _category(stale) == "confirmation_stale"
    registry.clear_assistant_defaults("w-1")
    with pytest.raises(ValueError) as consumed:
        registry.set_assistant_defaults(
            "w-1", intended, tool_profile_confirmation_token=token
        )
    assert _category(consumed) == "confirmation_invalid"


def test_workspace_race_consumes_the_one_use_token(tmp_path, monkeypatch):
    """A state change before the final fence check must still burn the token."""
    registry, _store, guard, _clock = _build(tmp_path, monkeypatch)
    intended = _defaults()
    token = guard.confirm(guard.review("w-1", intended, action="set"))
    raced = _defaults("local-profile")
    reads = 0

    def raced_reader(workspace_id):
        nonlocal reads
        reads += 1
        if reads == 1:
            return raced
        record = registry.get_workspace(workspace_id)
        return record.assistant_defaults if record is not None else None

    guard._workspace_defaults_reader = raced_reader
    with pytest.raises(ValueError) as stale:
        registry.set_assistant_defaults(
            "w-1", intended, tool_profile_confirmation_token=token
        )
    assert _category(stale) == "confirmation_stale"

    with pytest.raises(ValueError) as consumed:
        registry.set_assistant_defaults(
            "w-1", intended, tool_profile_confirmation_token=token
        )
    assert _category(consumed) == "confirmation_invalid"


def test_replace_of_existing_local_defaults_accepts_exact_imported_review(
    tmp_path, monkeypatch
):
    """Treating every bind as set would reject the existing-default replacement path."""
    registry, _store, guard, _clock = _build(tmp_path, monkeypatch)
    registry.set_assistant_defaults("w-1", _defaults("local-profile"))
    intended = _defaults()
    token = guard.confirm(guard.review("w-1", intended, action="replace"))

    assert (
        registry.set_assistant_defaults(
            "w-1", intended, tool_profile_confirmation_token=token
        ).assistant_defaults
        == intended
    )


@pytest.mark.parametrize(
    ("memory_ack", "tool_ack", "succeeds"),
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (True, True, True),
    ],
)
def test_memory_and_tool_profile_acknowledgements_are_independent(
    tmp_path, monkeypatch, memory_ack, tool_ack, succeeds
):
    """Either acknowledgement satisfying the other would merge two trust gates."""
    registry, _store, guard, _clock = _build(tmp_path, monkeypatch)
    intended = _defaults(memory="read_write")
    token = (
        guard.confirm(guard.review("w-1", intended, action="set")) if tool_ack else None
    )

    if succeeds:
        result = registry.set_assistant_defaults(
            "w-1",
            intended,
            confirm_read_write=memory_ack,
            tool_profile_confirmation_token=token,
        )
        assert result.assistant_defaults == intended
        return

    expected_error = WorkspaceRegistryServiceError if not memory_ack else ValueError
    with pytest.raises(expected_error) as rejected:
        registry.set_assistant_defaults(
            "w-1",
            intended,
            confirm_read_write=memory_ack,
            tool_profile_confirmation_token=token,
        )
    if memory_ack:
        assert _category(rejected) == "confirmation_required"
    if tool_ack and not memory_ack:
        # The independent memory gate runs before and does not consume the Tool token.
        assert (
            registry.set_assistant_defaults(
                "w-1",
                intended,
                confirm_read_write=True,
                tool_profile_confirmation_token=token,
            ).assistant_defaults
            == intended
        )


def test_uncertain_commit_reconciles_exact_binding_before_marker_clear(
    tmp_path, monkeypatch
):
    """Clearing on an unverified post-commit error could authorize no workspace."""
    registry, store, guard, _clock = _build(tmp_path, monkeypatch)
    intended = _defaults()
    token = guard.confirm(guard.review("w-1", intended, action="set"))
    original = registry.db.transaction

    @contextmanager
    def committed_then_uncertain(*, immediate: bool = False):
        with original(immediate=immediate) as conn:
            yield conn
        raise sqlite3.OperationalError("commit acknowledgement lost")

    monkeypatch.setattr(registry.db, "transaction", committed_then_uncertain)
    with pytest.raises(ValueError) as uncertain:
        registry.set_assistant_defaults(
            "w-1", intended, tool_profile_confirmation_token=token
        )
    assert _category(uncertain) == "binding_uncertain"
    assert registry.get_workspace("w-1").assistant_defaults == intended
    assert _lifecycle(store)["first_bind_confirmation_required"] is False


def test_unproven_failed_commit_never_clears_marker(tmp_path, monkeypatch):
    """Treating every transaction exception as committed would clear too early."""
    registry, store, guard, _clock = _build(tmp_path, monkeypatch)
    intended = _defaults()
    token = guard.confirm(guard.review("w-1", intended, action="set"))
    original = registry.db.transaction

    @contextmanager
    def rolled_back(*, immediate: bool = False):
        with original(immediate=immediate) as conn:
            yield conn
            raise sqlite3.OperationalError("write failed before commit")

    monkeypatch.setattr(registry.db, "transaction", rolled_back)
    with pytest.raises(ValueError) as uncertain:
        registry.set_assistant_defaults(
            "w-1", intended, tool_profile_confirmation_token=token
        )
    assert _category(uncertain) == "binding_uncertain"
    assert registry.get_workspace("w-1").assistant_defaults is None
    assert _lifecycle(store)["first_bind_confirmation_required"] is True


def test_process_control_exception_propagates_without_marker_clear(
    tmp_path, monkeypatch
):
    """Binding reconciliation must not convert interrupts into domain errors."""
    registry, store, guard, _clock = _build(tmp_path, monkeypatch)
    intended = _defaults()
    token = guard.confirm(guard.review("w-1", intended, action="set"))
    original = registry.db.transaction

    @contextmanager
    def interrupted(*, immediate: bool = False):
        with original(immediate=immediate) as conn:
            yield conn
            raise KeyboardInterrupt

    monkeypatch.setattr(registry.db, "transaction", interrupted)
    with pytest.raises(KeyboardInterrupt):
        registry.set_assistant_defaults(
            "w-1", intended, tool_profile_confirmation_token=token
        )
    assert registry.get_workspace("w-1").assistant_defaults == intended
    assert _lifecycle(store)["first_bind_confirmation_required"] is True


def test_uncertain_reconciliation_read_failure_reports_binding_uncertain(
    tmp_path, monkeypatch
):
    """A failed exact reread must not leak a storage error or clear the marker."""
    registry, store, guard, _clock = _build(tmp_path, monkeypatch)
    intended = _defaults()
    token = guard.confirm(guard.review("w-1", intended, action="set"))
    original = registry.db.transaction

    @contextmanager
    def committed_then_uncertain(*, immediate: bool = False):
        with original(immediate=immediate) as conn:
            yield conn
        raise sqlite3.OperationalError("commit acknowledgement lost")

    calls = 0

    def reader(_workspace_id):
        nonlocal calls
        calls += 1
        if calls == 1:
            return None
        raise RuntimeError("reconciliation unavailable")

    guard._workspace_defaults_reader = reader
    monkeypatch.setattr(registry.db, "transaction", committed_then_uncertain)
    with pytest.raises(ValueError) as uncertain:
        registry.set_assistant_defaults(
            "w-1", intended, tool_profile_confirmation_token=token
        )
    assert _category(uncertain) == "binding_uncertain"
    assert _lifecycle(store)["first_bind_confirmation_required"] is True


def test_known_binding_survives_marker_clear_failure_and_prompts_again(
    tmp_path, monkeypatch
):
    """Marker persistence failure must not roll back or pretend the marker cleared."""
    registry, store, guard, _clock = _build(tmp_path, monkeypatch)
    intended = _defaults()
    token = guard.confirm(guard.review("w-1", intended, action="set"))

    monkeypatch.setattr(
        store, "save", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk"))
    )
    assert (
        registry.set_assistant_defaults(
            "w-1", intended, tool_profile_confirmation_token=token
        ).assistant_defaults
        == intended
    )
    assert _lifecycle(store)["first_bind_confirmation_required"] is True

    registry.create_workspace(workspace_id="w-2", name="Two")
    with pytest.raises(ValueError) as required:
        registry.set_assistant_defaults("w-2", intended)
    assert _category(required) == "confirmation_required"


def test_binding_holds_lifecycle_serialization_through_sqlite_commit(
    tmp_path, monkeypatch
):
    """Releasing before commit would let removal pass a stale reference check."""
    registry, _store, guard, _clock = _build(tmp_path, monkeypatch)
    intended = _defaults()
    token = guard.confirm(guard.review("w-1", intended, action="set"))
    original = registry.db.transaction
    write_reached = threading.Event()
    release_commit = threading.Event()
    competing_mutation_entered = threading.Event()
    errors: list[BaseException] = []

    @contextmanager
    def blocked_commit(*, immediate: bool = False):
        with original(immediate=immediate) as conn:
            yield conn
            write_reached.set()
            assert release_commit.wait(2)

    monkeypatch.setattr(registry.db, "transaction", blocked_commit)

    def bind() -> None:
        try:
            registry.set_assistant_defaults(
                "w-1", intended, tool_profile_confirmation_token=token
            )
        except BaseException as exc:  # pragma: no cover - assertion reports it
            errors.append(exc)

    def competing_mutation() -> None:
        with guard.lifecycle.mutation():
            competing_mutation_entered.set()

    bind_thread = threading.Thread(target=bind)
    bind_thread.start()
    assert write_reached.wait(2)
    removal_thread = threading.Thread(target=competing_mutation)
    removal_thread.start()
    assert not competing_mutation_entered.wait(0.1)
    release_commit.set()
    bind_thread.join(2)
    removal_thread.join(2)
    assert errors == []
    assert competing_mutation_entered.is_set()


def test_local_and_workspace_profiles_need_no_tool_pack_token(tmp_path, monkeypatch):
    """Treating all named profiles as imported would regress existing defaults."""
    registry, _store, _guard, _clock = _build(tmp_path, monkeypatch)

    assert (
        registry.set_assistant_defaults(
            "w-1", _defaults("local-profile")
        ).assistant_defaults.tool_policy_profile_id
        == "local-profile"
    )
    assert (
        registry.set_assistant_defaults(
            "w-1", _defaults("ws-w-1")
        ).assistant_defaults.tool_policy_profile_id
        == "ws-w-1"
    )


def test_profile_free_create_and_clear_survive_unreadable_tool_authority(
    tmp_path, monkeypatch
):
    """Safe profile-free writes must not depend on unrelated policy JSON health."""
    registry, store, _guard, _clock = _build(tmp_path, monkeypatch)
    store.path.write_text("{", encoding="utf-8")

    created = registry.create_workspace(workspace_id="w-plain", name="Plain")
    assert created.assistant_defaults is None
    assert registry.clear_assistant_defaults("w-1").assistant_defaults is None
