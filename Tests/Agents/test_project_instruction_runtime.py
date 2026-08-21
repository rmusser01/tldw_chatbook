from __future__ import annotations

import hashlib
import time
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tldw_chatbook.Agents import project_instruction_resolver as resolver_module
from tldw_chatbook.Agents.agent_models import ToolCall, ToolCatalogEntry, ToolResult, ToolSchema
from tldw_chatbook.Agents.project_instruction_resolver import (
    InstructionChainDelivery,
    InstructionOutcome,
    InstructionSnapshot,
    InstructionSource,
    NestedResolutionBatch,
    ProjectInstructionResolver,
)
from tldw_chatbook.Agents.project_instruction_runtime import (
    PROJECT_INSTRUCTION_ROW_KEY,
    InstructionActivationLedger,
    InstructionChainPayloadState,
    InstructionDeliveryReceipt,
    build_project_instruction_deferral_rows,
)
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry, ToolPathTarget
from tldw_chatbook.Chat.console_project_instructions import EPHEMERAL_ORIGIN_KEY


def _source(root: Path, relative_path: str, body: str = "guidance") -> InstructionSource:
    raw = body.encode()
    return InstructionSource(
        canonical_path=root / relative_path,
        relative_path=relative_path,
        scope=str(Path(relative_path).parent).replace(".", ".", 1),
        kind="standard",
        body=body,
        byte_count=len(raw),
        digest=hashlib.sha256(raw).hexdigest(),
    )


def _snapshot(root: Path, *, root_delivered: bool = True) -> InstructionSnapshot:
    source = _source(root, "AGENTS.md", "root-secret")
    delivery = InstructionChainDelivery(
        source_digests=(source.digest,) if root_delivered else (),
        outcomes=(() if root_delivered else (
            InstructionOutcome("AGENTS.md", ".", "omitted_token_budget"),
        )),
    )
    return InstructionSnapshot(
        binding_id="binding",
        binding_root=root,
        locator_fingerprint="fingerprint",
        dispatch_started_wall_ns=time.time_ns() + 1_000_000_000,
        startup_source=source,
        global_outcomes=(),
        primary_delivery=delivery,
        warning_codes=(),
    )


def _payload(*, allowance: int | Exception = 10_000, row_cost: int = 1):
    built: list[tuple[list[dict], tuple]] = []

    def request_builder(messages, active_schemas):
        request = (list(messages), tuple(active_schemas))
        built.append(request)
        return request

    def safe_tokens(_request, _candidate_rows):
        if isinstance(allowance, Exception):
            raise allowance
        return allowance

    state = InstructionChainPayloadState(
        request_builder=request_builder,
        safe_token_allowance=safe_tokens,
        count_tokens=lambda rows: len(rows) * row_cost,
    )
    state.capture(messages=[], active_schemas=(), calls=[])
    return state, built


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


def test_receipt_is_frozen_content_free_and_has_no_persistence_api(tmp_path: Path):
    ledger = InstructionActivationLedger(_snapshot(tmp_path), nested_max_bytes=100)
    payload, _ = _payload()

    delivery = ledger.initial_context_for_chain("child", payload)

    assert isinstance(delivery.receipt, InstructionDeliveryReceipt)
    assert "root-secret" not in repr(delivery.receipt)
    assert str(tmp_path) not in repr(delivery.receipt)
    assert not hasattr(delivery.receipt, "to_json")
    with pytest.raises(FrozenInstanceError):
        delivery.receipt.chain_id = "other"


@pytest.mark.parametrize(
    "relative_path", ["/private/AGENTS.md", r"C:\private\AGENTS.md", "../AGENTS.md"]
)
def test_receipt_rejects_absolute_or_escaping_outcome_paths(relative_path):
    with pytest.raises(ValueError, match="outcome key"):
        InstructionDeliveryReceipt(
            receipt_id="receipt",
            chain_id="chain",
            through_revision=0,
            source_digests=(),
            outcome_keys=(f"invalid\x1f{relative_path}\x1f.",),
            row_keys=("row",),
        )


def test_primary_cursor_starts_at_snapshot_delivery_but_child_gets_active_root(tmp_path):
    ledger = InstructionActivationLedger(_snapshot(tmp_path), nested_max_bytes=100)
    payload, _ = _payload()

    assert ledger.initial_context_for_chain("primary", payload).rows == ()
    child = ledger.initial_context_for_chain("child", payload)

    assert len(child.rows) == 1
    assert "root-secret" in child.rows[0]["content"]
    assert child.rows[0][EPHEMERAL_ORIGIN_KEY] == "project_instructions"
    assert child.rows[0][PROJECT_INSTRUCTION_ROW_KEY] in child.receipt.row_keys


def test_cursor_advances_only_after_exact_issued_receipt_is_marked(tmp_path):
    ledger = InstructionActivationLedger(_snapshot(tmp_path), nested_max_bytes=100)
    payload, _ = _payload()
    first = ledger.initial_context_for_chain("child", payload)

    assert ledger.initial_context_for_chain("child", payload).receipt == first.receipt
    forged = InstructionDeliveryReceipt(
        receipt_id=first.receipt.receipt_id,
        chain_id="child",
        through_revision=first.receipt.through_revision,
        source_digests=first.receipt.source_digests,
        outcome_keys=first.receipt.outcome_keys,
        row_keys=("forged",),
    )
    with pytest.raises(ValueError, match="receipt"):
        ledger.mark_payload_sent(forged)

    ledger.mark_payload_sent(first.receipt)
    assert ledger.initial_context_for_chain("child", payload).rows == ()
    with pytest.raises(ValueError, match="receipt"):
        ledger.mark_payload_sent(first.receipt)


def test_payload_capture_counts_canonical_deferral_rows_and_current_schema():
    payload, built = _payload(allowance=123)
    calls = [ToolCall("read", {"path": "src/a.py"}, "call-1")]
    schemas = (ToolSchema("fake:read", "read", "read", {}),)
    payload.capture(
        messages=[{"role": "user", "content": "current"}],
        active_schemas=schemas,
        calls=calls,
    )

    assert payload.safe_input_tokens([{"role": "user", "content": "candidate"}]) == 123
    messages, captured_schemas = built[-1]
    assert captured_schemas == schemas
    assert messages[0]["content"] == "current"
    assert messages[1:] == list(build_project_instruction_deferral_rows(calls))
    assert calls[0].call_id in messages[1]["tool_call_id"]


def test_payload_capture_is_not_changed_by_later_nested_message_mutation():
    payload, built = _payload(allowance=123)
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "captured"}]}
    ]
    payload.capture(messages=messages, active_schemas=(), calls=[])
    messages[0]["content"][0]["text"] = "mutated"

    payload.safe_input_tokens([])

    assert built[-1][0][0]["content"][0]["text"] == "captured"


def test_nested_source_pins_once_shares_budget_and_defers_each_chain(tmp_path: Path):
    nested = tmp_path / "src"
    nested.mkdir()
    instruction = nested / "AGENTS.md"
    instruction.write_text("nested-secret")
    ledger = InstructionActivationLedger(_snapshot(tmp_path), nested_max_bytes=100)
    registry = _registry(nested / "file.py")
    payload, _ = _payload()
    calls = [ToolCall("read", {"path": "src/file.py"}, "call-1")]

    parent = ledger.prepare(calls, "primary", registry, payload)
    assert parent.receipt is not None
    assert "nested-secret" in parent.rows[0]["content"]
    assert ledger.remaining_nested_bytes == 100 - len(b"nested-secret")
    instruction.write_text("edited-after-pin")

    child = ledger.prepare(calls, "child", registry, payload)
    assert child.receipt is not None
    assert any("nested-secret" in row["content"] for row in child.rows)
    assert all("edited-after-pin" not in row["content"] for row in child.rows)
    assert ledger.remaining_nested_bytes == 100 - len(b"nested-secret")


def test_successful_nested_delivery_makes_identical_retry_proceed(tmp_path: Path):
    nested = tmp_path / "src"
    nested.mkdir()
    (nested / "AGENTS.md").write_text("nested")
    ledger = InstructionActivationLedger(_snapshot(tmp_path), nested_max_bytes=100)
    registry = _registry(nested / "file.py")
    payload, _ = _payload()
    calls = [ToolCall("read", {}, "call-1")]

    first = ledger.prepare(calls, "primary", registry, payload)
    ledger.mark_payload_sent(first.receipt)

    assert ledger.prepare(calls, "primary", registry, payload).rows == ()


def test_child_first_context_racing_later_activation_gets_next_revision(tmp_path: Path):
    nested = tmp_path / "src"
    nested.mkdir()
    (nested / "AGENTS.md").write_text("nested")
    ledger = InstructionActivationLedger(_snapshot(tmp_path), nested_max_bytes=100)
    payload, _ = _payload()
    root_delivery = ledger.initial_context_for_chain("child", payload)

    nested_delivery = ledger.prepare(
        [ToolCall("read", {}, "call-1")],
        "primary",
        _registry(nested / "file.py"),
        payload,
    )
    assert nested_delivery.receipt.through_revision > root_delivery.receipt.through_revision
    ledger.mark_payload_sent(root_delivery.receipt)

    child_update = ledger.initial_context_for_chain("child", payload)
    assert len(child_update.rows) == 1
    assert "nested" in child_update.rows[0]["content"]


@pytest.mark.parametrize("allowance", [0, -1, RuntimeError("unknown model")])
def test_missing_or_nonpositive_headroom_becomes_chain_token_omission_once(
    tmp_path: Path, allowance
):
    ledger = InstructionActivationLedger(_snapshot(tmp_path), nested_max_bytes=100)
    payload, _ = _payload(allowance=allowance)

    first = ledger.initial_context_for_chain("child", payload)
    assert first.receipt.source_digests == ()
    assert any("omitted_token_budget" in row["content"] for row in first.rows)
    ledger.mark_payload_sent(first.receipt)

    assert ledger.initial_context_for_chain("child", payload).rows == ()


def test_two_chains_have_independent_token_outcomes(tmp_path: Path):
    ledger = InstructionActivationLedger(_snapshot(tmp_path), nested_max_bytes=100)
    roomy, _ = _payload(allowance=10)
    cramped, _ = _payload(allowance=0)

    delivered = ledger.initial_context_for_chain("roomy", roomy)
    omitted = ledger.initial_context_for_chain("cramped", cramped)

    assert delivered.receipt.source_digests
    assert omitted.receipt.source_digests == ()
    assert any("omitted_token_budget" in row["content"] for row in omitted.rows)


def test_global_terminal_outcome_is_warned_once_per_chain(tmp_path: Path):
    snapshot = _snapshot(tmp_path)
    snapshot = InstructionSnapshot(
        binding_id=snapshot.binding_id,
        binding_root=snapshot.binding_root,
        locator_fingerprint=snapshot.locator_fingerprint,
        dispatch_started_wall_ns=snapshot.dispatch_started_wall_ns,
        startup_source=snapshot.startup_source,
        global_outcomes=(InstructionOutcome("bad/AGENTS.md", "bad", "invalid"),),
        primary_delivery=snapshot.primary_delivery,
        warning_codes=("invalid",),
    )
    ledger = InstructionActivationLedger(snapshot, nested_max_bytes=100)
    payload, _ = _payload()

    first = ledger.initial_context_for_chain("child", payload)
    assert any("bad/AGENTS.md" in row["content"] for row in first.rows)
    ledger.mark_payload_sent(first.receipt)
    assert ledger.initial_context_for_chain("child", payload).rows == ()


def test_nested_invalid_outcome_remains_terminal_after_file_becomes_valid(
    tmp_path, monkeypatch
):
    nested = tmp_path / "src"
    nested.mkdir()
    path = nested / "AGENTS.override.md"
    path.write_bytes(b"\xff")
    ledger = InstructionActivationLedger(_snapshot(tmp_path), nested_max_bytes=100)
    registry = _registry(nested / "file.py")
    payload, _ = _payload()
    calls = [ToolCall("read", {}, "call-1")]

    first = ledger.prepare(calls, "primary", registry, payload)
    assert any("invalid" in row["content"] for row in first.rows)
    ledger.mark_payload_sent(first.receipt)
    path.write_text("must wait until next dispatch")
    real_open = resolver_module.os.open

    def refuse_terminal_reread(candidate, flags):
        if Path(candidate) == path:
            raise AssertionError("terminal source must not be reread")
        return real_open(candidate, flags)

    monkeypatch.setattr(resolver_module.os, "open", refuse_terminal_reread)

    retry = ledger.prepare(calls, "primary", registry, payload)
    assert retry.status == "proceed"
    assert all("must wait until next dispatch" not in row["content"] for row in retry.rows)


def test_outside_binding_warning_defers_once_without_exposing_path(tmp_path: Path):
    class OutsideProvider(_PathProvider):
        def path_targets(self, tool_id, args):
            return (ToolPathTarget(tmp_path.parent / "private", "outside"),)

    registry = ToolCatalogRegistry()
    registry.register_provider(OutsideProvider(tmp_path))
    ledger = InstructionActivationLedger(_snapshot(tmp_path), nested_max_bytes=100)
    payload, _ = _payload()
    calls = [ToolCall("read", {}, "call-1")]

    first = ledger.prepare(calls, "primary", registry, payload)
    rendered = repr(first.rows)
    assert "outside_instruction_scope" in rendered
    assert str(tmp_path.parent / "private") not in rendered
    ledger.mark_payload_sent(first.receipt)
    assert ledger.prepare(calls, "primary", registry, payload).rows == ()


def test_resolver_walks_only_target_chain_and_renders_broad_to_specific(tmp_path):
    sibling = tmp_path / "sibling"
    target = tmp_path / "src" / "pkg"
    sibling.mkdir()
    target.mkdir(parents=True)
    (sibling / "AGENTS.md").write_text("sibling")
    (tmp_path / "src" / "AGENTS.md").write_text("broad")
    (target / "AGENTS.override.md").write_text("specific")

    batch = ProjectInstructionResolver().resolve_targets(
        tmp_path,
        [target],
        max_bytes=100,
        dispatch_started_wall_ns=time.time_ns() + 1_000_000_000,
        pinned_by_canonical_path={},
    )

    assert isinstance(batch, NestedResolutionBatch)
    assert [source.relative_path for source in batch.sources] == [
        "src/AGENTS.md",
        "src/pkg/AGENTS.override.md",
    ]
    assert all("sibling" not in source.relative_path for source in batch.sources)


def test_nested_byte_admission_is_deepest_first_but_output_is_broad_first(tmp_path):
    target = tmp_path / "src" / "pkg"
    target.mkdir(parents=True)
    (tmp_path / "src" / "AGENTS.md").write_text("broad!")
    (target / "AGENTS.md").write_text("deep!!")

    batch = ProjectInstructionResolver().resolve_targets(
        tmp_path,
        [target],
        max_bytes=6,
        dispatch_started_wall_ns=time.time_ns() + 1_000_000_000,
        pinned_by_canonical_path={},
    )

    assert [source.scope for source in batch.sources] == ["src/pkg"]
    assert [(outcome.scope, outcome.code) for outcome in batch.outcomes] == [
        ("src", "omitted_byte_budget")
    ]


def test_exhausted_global_budget_does_not_change_empty_override_precedence(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    (first_dir / "AGENTS.md").write_text("111111")
    (second_dir / "AGENTS.override.md").write_text(" ")
    (second_dir / "AGENTS.md").write_text("222222")
    ledger = InstructionActivationLedger(_snapshot(tmp_path), nested_max_bytes=6)
    payload, _ = _payload()

    first = ledger.prepare(
        [ToolCall("read", {}, "first")],
        "primary",
        _registry(first_dir / "file.py"),
        payload,
    )
    ledger.mark_payload_sent(first.receipt)
    second = ledger.prepare(
        [ToolCall("read", {}, "second")],
        "primary",
        _registry(second_dir / "file.py"),
        payload,
    )

    rendered = "\n".join(row["content"] for row in second.rows)
    assert "second/AGENTS.md" in rendered
    assert "second/AGENTS.override.md" not in rendered
    assert "omitted_byte_budget" in rendered


def test_nested_pinned_source_is_reused_by_identity_after_delete(tmp_path):
    target = tmp_path / "src"
    target.mkdir()
    path = target / "AGENTS.md"
    path.write_text("pinned")
    resolver = ProjectInstructionResolver()
    first = resolver.resolve_targets(
        tmp_path,
        [target],
        max_bytes=100,
        dispatch_started_wall_ns=time.time_ns() + 1_000_000_000,
        pinned_by_canonical_path={},
    )
    pinned = {first.sources[0].canonical_path: first.sources[0]}
    path.unlink()

    second = resolver.resolve_targets(
        tmp_path,
        [target],
        max_bytes=0,
        dispatch_started_wall_ns=0,
        pinned_by_canonical_path=pinned,
    )

    assert second.sources[0] is first.sources[0]
    assert second.outcomes == ()


def test_nested_created_after_dispatch_is_stale(tmp_path):
    target = tmp_path / "src"
    target.mkdir()
    path = target / "AGENTS.md"
    path.write_text("too new")
    old_cutoff = path.stat().st_mtime_ns - 1

    batch = ProjectInstructionResolver().resolve_targets(
        tmp_path,
        [target],
        max_bytes=100,
        dispatch_started_wall_ns=old_cutoff,
        pinned_by_canonical_path={},
    )

    assert batch.sources == ()
    assert [(outcome.relative_path, outcome.code) for outcome in batch.outcomes] == [
        ("src/AGENTS.md", "stale")
    ]


def test_nested_resolution_rejects_binding_root_swap_before_candidate_read(
    tmp_path, monkeypatch
):
    root = tmp_path / "selected"
    nested = root / "src"
    nested.mkdir(parents=True)
    (nested / "AGENTS.md").write_text("original")
    displaced = tmp_path / "displaced"
    real_lstat = resolver_module.os.lstat
    nested_lstats = 0

    def swap_on_nested_revalidation(candidate):
        nonlocal nested_lstats
        if Path(candidate) == nested:
            nested_lstats += 1
            if nested_lstats == 2:
                root.rename(displaced)
                (root / "src").mkdir(parents=True)
                (root / "src" / "AGENTS.md").write_text("replacement")
        return real_lstat(candidate)

    monkeypatch.setattr(resolver_module.os, "lstat", swap_on_nested_revalidation)

    batch = ProjectInstructionResolver().resolve_targets(
        root,
        [nested],
        max_bytes=100,
        dispatch_started_wall_ns=time.time_ns() + 1_000_000_000,
        pinned_by_canonical_path={},
    )

    assert nested_lstats >= 2
    assert batch.sources == ()
    assert [outcome.code for outcome in batch.outcomes] == ["resolution_failed"]
