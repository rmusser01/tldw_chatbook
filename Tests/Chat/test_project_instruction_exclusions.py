"""AGENTS.md under an excluded path never activates (spec section 2)."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents.agent_models import ToolCall, ToolCatalogEntry, ToolResult, ToolSchema
from tldw_chatbook.Agents.project_instruction_resolver import (
    InstructionChainDelivery,
    InstructionSnapshot,
    InstructionSource,
    ProjectInstructionResolver,
)
from tldw_chatbook.Agents.project_instruction_runtime import (
    InstructionActivationLedger,
    InstructionChainPayloadState,
    path_is_excluded,
)
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry, ToolPathTarget
from tldw_chatbook.Chat.console_chat_controller import (
    _project_instruction_excluded_dirs,
    resolve_project_instruction_binding,
)
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
)
from tldw_chatbook.Workspaces.models import WorkspaceRuntimeBinding

SENTINEL = "EXCLUDED_PRIVATE_GUIDANCE_SENTINEL_9c31"


def test_path_is_excluded_matches_dir_and_children(tmp_path: Path):
    excluded = frozenset({tmp_path / "private"})
    assert path_is_excluded(tmp_path / "private", excluded)
    assert path_is_excluded(tmp_path / "private" / "AGENTS.md", excluded)
    assert not path_is_excluded(tmp_path / "public" / "AGENTS.md", excluded)


def test_path_is_excluded_folds_case(tmp_path: Path):
    """A Settings-typed ``Private`` exclusion still excludes on-disk ``private/``.

    macOS/Windows filesystems are case-insensitive by default and
    ``Path.resolve()`` preserves the typed spelling, so an exact-case compare
    lets ``private/AGENTS.md`` through a ``Private`` exclusion -- the same
    hole the denylist's own ``_compare_key`` discipline (TASK-19800) closes
    for the fs-tool families. Deny-side only: folding must never touch a
    confinement check.
    """
    excluded = frozenset({tmp_path / "Private"})
    assert path_is_excluded(tmp_path / "private", excluded)
    assert path_is_excluded(tmp_path / "private" / "AGENTS.md", excluded)
    assert not path_is_excluded(tmp_path / "privatefoo" / "AGENTS.md", excluded)
    assert not path_is_excluded(tmp_path / "other" / "AGENTS.md", excluded)


class _BindingRegistry:
    def __init__(self, bindings):
        self.bindings = {binding.binding_id: binding for binding in bindings}

    def list_runtime_bindings(self, workspace_id):
        return tuple(
            binding
            for binding in self.bindings.values()
            if binding.workspace_id == workspace_id
        )

    def get_runtime_binding(self, binding_id):
        return self.bindings.get(binding_id)


def _binding(root: Path, exclusions: tuple[str, ...] = ("private",)) -> WorkspaceRuntimeBinding:
    return WorkspaceRuntimeBinding(
        workspace_id="w1",
        binding_id="b1",
        binding_kind="local-filesystem",
        label="b1",
        locator=str(root),
        status="ready",
        metadata={
            "access": "rw",
            "exclusions": [
                {
                    "path": path,
                    "kind": "directory",
                    "added_at": "2026-09-20T00:00:00Z",
                }
                for path in exclusions
            ],
        },
    )


def _selection(root: Path):
    session = SimpleNamespace(
        workspace_id="w1",
        project_instruction_state=ProjectInstructionControlState.new_session(),
    )
    return resolve_project_instruction_binding(session, _BindingRegistry([_binding(root)]))


def _source(root: Path, relative_path: str, body: str) -> InstructionSource:
    raw = body.encode()
    parent = Path(relative_path).parent
    return InstructionSource(
        canonical_path=root / relative_path,
        relative_path=relative_path,
        scope="." if str(parent) == "." else parent.as_posix(),
        kind="standard",
        body=body,
        byte_count=len(raw),
        digest=hashlib.sha256(raw).hexdigest(),
    )


def _snapshot(
    root: Path,
    *,
    excluded_dirs: frozenset[Path] = frozenset(),
) -> InstructionSnapshot:
    source = _source(root, "AGENTS.md", "root guidance")
    return InstructionSnapshot(
        binding_id="b1",
        binding_root=root,
        locator_fingerprint="fingerprint",
        dispatch_started_wall_ns=time.time_ns() + 1_000_000_000,
        startup_source=source,
        global_outcomes=(),
        primary_delivery=InstructionChainDelivery(
            source_digests=(source.digest,), outcomes=()
        ),
        warning_codes=(),
        excluded_dirs=excluded_dirs,
    )


class _PathProvider:
    def __init__(self, target: Path) -> None:
        self.target = target

    def list_catalog(self):
        return [ToolCatalogEntry("fake:read", "read", "read", "local")]

    def load_schema(self, tool_id):
        return ToolSchema(tool_id, "read", "read", {})

    def invoke(self, tool_id, args):
        return ToolResult(ok=True)

    def path_targets(self, tool_id, args):
        return (ToolPathTarget(self.target, "exact"),)


def _registry(target: Path) -> ToolCatalogRegistry:
    registry = ToolCatalogRegistry()
    registry.register_provider(_PathProvider(target))
    return registry


def _payload(*, allowance: int = 10_000):
    def request_builder(messages, active_schemas):
        return (list(messages), tuple(active_schemas))

    state = InstructionChainPayloadState(
        request_builder=request_builder,
        safe_token_allowance=lambda _request, _rows: allowance,
        count_tokens=lambda rows: len(rows),
    )
    state.capture(messages=[], active_schemas=(), calls=[])
    return state


def test_binding_exclusions_flow_to_ledger_and_skip_activation(tmp_path: Path):
    (tmp_path / "AGENTS.md").write_text("root guidance")
    private = tmp_path / "private"
    private.mkdir()
    (private / "AGENTS.md").write_text(SENTINEL)

    selection = _selection(tmp_path)
    excluded = _project_instruction_excluded_dirs(selection)
    assert excluded == frozenset({(tmp_path / "private").resolve(strict=False)})

    ledger = InstructionActivationLedger(
        _snapshot(tmp_path, excluded_dirs=excluded), nested_max_bytes=10_000
    )
    call = ToolCall("read", {"path": str(private / "AGENTS.md")}, "call-1")
    preparation = ledger.prepare(
        (call,), "primary", _registry(private / "AGENTS.md"), _payload()
    )

    assert preparation.status == "proceed"
    assert preparation.rows == ()
    assert SENTINEL not in repr(preparation)
    assert ledger.warning_keys == ()

    from tldw_chatbook.Agents.project_instruction_resolver import (
        InstructionPromotionSnapshotError,
    )

    with pytest.raises(InstructionPromotionSnapshotError) as error:
        ledger.snapshot_promotion_target("private/AGENTS.md")
    assert error.value.code == "ineligible_target"


def test_included_nested_instructions_still_activate(tmp_path: Path):
    (tmp_path / "AGENTS.md").write_text("root guidance")
    public = tmp_path / "public"
    public.mkdir()
    (public / "AGENTS.md").write_text(SENTINEL)

    ledger = InstructionActivationLedger(
        _snapshot(tmp_path, excluded_dirs=frozenset()), nested_max_bytes=10_000
    )
    call = ToolCall("read", {"path": str(public / "AGENTS.md")}, "call-1")
    preparation = ledger.prepare(
        (call,), "primary", _registry(public / "AGENTS.md"), _payload()
    )

    assert preparation.status == "retry_with_context"
    assert any(SENTINEL in str(row.get("content", "")) for row in preparation.rows)


def test_resolver_treats_excluded_candidates_as_nonexistent(tmp_path: Path):
    root = tmp_path.resolve()
    private = root / "private"
    (private / "sub").mkdir(parents=True)
    # Oversized so any actual read would surface as omitted_byte_budget.
    (private / "AGENTS.md").write_text("x" * 100)
    (private / "sub" / "AGENTS.md").write_text("y" * 100)
    (root / "AGENTS.md").write_text("root guidance")
    excluded = frozenset({private})
    resolver = ProjectInstructionResolver()

    batch = resolver.resolve_targets(
        root,
        (private / "sub",),
        max_bytes=10,
        dispatch_started_wall_ns=time.time_ns() + 1_000_000_000,
        pinned_by_canonical_path={},
        excluded_dirs=excluded,
    )
    assert batch.sources == ()
    assert batch.outcomes == ()

    startup = resolver.resolve_startup(
        binding_id="b1",
        binding_root=root,
        locator_fingerprint="fingerprint",
        max_bytes=10,
        dispatch_started_wall_ns=time.time_ns() + 1_000_000_000,
        excluded_dirs=frozenset({root / "AGENTS.md"}),
    )
    assert startup.source is None
    assert startup.outcomes == ()
    assert startup.excluded_dirs == frozenset({root / "AGENTS.md"})


def test_resolver_skips_case_variant_excluded_directory(tmp_path: Path):
    """End-to-end Finding 1 regression: exclusion ``Private`` vs on-disk ``private/``.

    Before the casefold fix the differently-cased exclusion missed, the
    nested directory was admitted, and its AGENTS.md body was read into
    context (the byte-budget pass below would have returned the sentinel).
    """
    root = tmp_path.resolve()
    private = root / "private"
    private.mkdir()
    (private / "AGENTS.md").write_text(SENTINEL)
    resolver = ProjectInstructionResolver()

    batch = resolver.resolve_targets(
        root,
        (private,),
        max_bytes=10_000,
        dispatch_started_wall_ns=time.time_ns() + 1_000_000_000,
        pinned_by_canonical_path={},
        excluded_dirs=frozenset({root / "Private"}),
    )
    assert batch.sources == ()
    assert batch.outcomes == ()


def test_nested_excluded_file_treated_as_absent(tmp_path: Path):
    """Finding 1: excluding exactly the FILE ``docs/AGENTS.md`` must hide it.

    The ``resolve_targets`` walk only prunes excluded directories, so
    ``docs/`` stays traversable; the excluded standard candidate must be
    skipped exactly like a nonexistent file (no source, no outcome) while
    an ``AGENTS.md`` in a non-excluded directory still activates.
    """
    root = tmp_path.resolve()
    docs = root / "docs"
    docs.mkdir()
    (docs / "AGENTS.md").write_text(SENTINEL)
    public = root / "public"
    public.mkdir()
    (public / "AGENTS.md").write_text("public guidance")
    resolver = ProjectInstructionResolver()

    batch = resolver.resolve_targets(
        root,
        (docs, public),
        max_bytes=10_000,
        dispatch_started_wall_ns=time.time_ns() + 1_000_000_000,
        pinned_by_canonical_path={},
        excluded_dirs=frozenset({docs / "AGENTS.md"}),
    )

    assert [source.relative_path for source in batch.sources] == ["public/AGENTS.md"]
    assert all(SENTINEL not in source.body for source in batch.sources)
    assert batch.outcomes == ()


def test_nested_excluded_standard_sibling_override_still_activates(tmp_path: Path):
    """The non-excluded sibling candidate in the same directory still reads."""
    root = tmp_path.resolve()
    docs = root / "docs"
    docs.mkdir()
    (docs / "AGENTS.md").write_text(SENTINEL)
    (docs / "AGENTS.override.md").write_text("docs override guidance")
    resolver = ProjectInstructionResolver()

    batch = resolver.resolve_targets(
        root,
        (docs,),
        max_bytes=10_000,
        dispatch_started_wall_ns=time.time_ns() + 1_000_000_000,
        pinned_by_canonical_path={},
        excluded_dirs=frozenset({docs / "AGENTS.md"}),
    )

    assert [source.relative_path for source in batch.sources] == [
        "docs/AGENTS.override.md"
    ]
    assert all(SENTINEL not in source.body for source in batch.sources)
    assert batch.outcomes == ()


def test_nested_excluded_override_falls_back_to_standard(tmp_path: Path):
    """An excluded override behaves exactly like an absent one: fallback.

    Before the fix the excluded ``AGENTS.override.md`` was read and won
    over the standard file, leaking its body into the batch.
    """
    root = tmp_path.resolve()
    docs = root / "docs"
    docs.mkdir()
    (docs / "AGENTS.override.md").write_text(SENTINEL)
    (docs / "AGENTS.md").write_text("docs standard guidance")
    resolver = ProjectInstructionResolver()

    batch = resolver.resolve_targets(
        root,
        (docs,),
        max_bytes=10_000,
        dispatch_started_wall_ns=time.time_ns() + 1_000_000_000,
        pinned_by_canonical_path={},
        excluded_dirs=frozenset({docs / "AGENTS.override.md"}),
    )

    assert [source.relative_path for source in batch.sources] == ["docs/AGENTS.md"]
    assert batch.sources[0].body == "docs standard guidance"
    assert all(SENTINEL not in source.body for source in batch.sources)
    assert batch.outcomes == ()


def test_promotion_chain_skips_excluded_ancestor_instruction(tmp_path: Path):
    """Finding 2: an excluded ancestor instruction stays out of the chain.

    Excluding exactly the root ``AGENTS.md`` must not block promotion of
    ``docs/AGENTS.md`` (the target gate only rejects the target itself),
    but the excluded root file must not be read into ``effective_chain``.
    """
    root = tmp_path.resolve()
    docs = root / "docs"
    docs.mkdir()
    (root / "AGENTS.md").write_text("root guidance")
    (docs / "AGENTS.md").write_text("docs guidance")
    resolver = ProjectInstructionResolver()

    snapshot = resolver.snapshot_promotion_target(
        binding_id="b1",
        binding_root=root,
        locator_fingerprint="fingerprint",
        target_path=docs / "AGENTS.md",
        activation_revision=0,
        max_bytes=10_000,
        excluded_dirs=frozenset({root / "AGENTS.md"}),
    )

    assert snapshot.target_relative_path == "docs/AGENTS.md"
    chain_relative_paths = [entry[0] for entry in snapshot.effective_chain]
    assert "AGENTS.md" not in chain_relative_paths
    assert chain_relative_paths == ["docs/AGENTS.md"]


def test_excluded_dirs_survive_one_unresolvable_entry(tmp_path: Path):
    """Finding 2b: one unresolvable exclusion entry must not zero the set.

    A self-referential symlink under the binding root makes resolving that
    one entry raise; the effective set must keep every other exclusion
    instead of collapsing to ``frozenset()`` (fail-open).
    """
    root = tmp_path.resolve()
    (root / "private").mkdir()
    loop = root / "loop"
    loop.symlink_to(loop)
    session = SimpleNamespace(
        workspace_id="w1",
        project_instruction_state=ProjectInstructionControlState.new_session(),
    )
    selection = resolve_project_instruction_binding(
        session, _BindingRegistry([_binding(tmp_path, exclusions=("private", "loop"))])
    )
    assert selection is not None

    excluded = _project_instruction_excluded_dirs(selection)

    assert excluded == frozenset({(root / "private").resolve(strict=False)})
