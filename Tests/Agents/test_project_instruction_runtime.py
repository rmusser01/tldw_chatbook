from __future__ import annotations

import hashlib
import time
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tldw_chatbook.Agents import project_instruction_resolver as resolver_module
from tldw_chatbook.Agents.agent_models import (
    ToolCall,
    ToolCatalogEntry,
    ToolResult,
    ToolSchema,
)
from tldw_chatbook.Agents.project_instruction_resolver import (
    InstructionChainDelivery,
    InstructionOutcome,
    InstructionSnapshot,
    InstructionSource,
    NestedResolutionBatch,
    ProjectInstructionResolver,
    capture_binding_root_identity,
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


def _source(
    root: Path, relative_path: str, body: str = "guidance"
) -> InstructionSource:
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
        outcomes=(
            ()
            if root_delivered
            else (InstructionOutcome("AGENTS.md", ".", "omitted_token_budget"),)
        ),
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


def _payload(
    *,
    allowance: int | Exception = 10_000,
    row_cost: int = 1,
    count_tokens=None,
):
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
        count_tokens=count_tokens or (lambda rows: len(rows) * row_cost),
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


def test_primary_cursor_starts_at_snapshot_delivery_but_child_gets_active_root(
    tmp_path,
):
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
        ledger.mark_payload_sent(forged, first.rows)

    ledger.mark_payload_sent(first.receipt, first.rows)
    assert ledger.initial_context_for_chain("child", payload).rows == ()
    with pytest.raises(ValueError, match="receipt"):
        ledger.mark_payload_sent(first.receipt, first.rows)


def test_mark_rejects_dropped_partial_or_mutated_staged_rows(tmp_path):
    ledger = InstructionActivationLedger(_snapshot(tmp_path), nested_max_bytes=100)
    payload, _ = _payload()
    delivery = ledger.initial_context_for_chain("child", payload)

    with pytest.raises(ValueError, match="payload rows"):
        ledger.mark_payload_sent(delivery.receipt, [])
    mutated = [dict(row) for row in delivery.rows]
    mutated[0]["content"] += " changed"
    with pytest.raises(ValueError, match="payload rows"):
        ledger.mark_payload_sent(delivery.receipt, mutated)
    stripped = [dict(row) for row in delivery.rows]
    stripped[0].pop(PROJECT_INSTRUCTION_ROW_KEY)
    with pytest.raises(ValueError, match="payload rows"):
        ledger.mark_payload_sent(delivery.receipt, stripped)

    outgoing = [{"role": "system", "content": "ordinary"}, *delivery.rows]
    ledger.mark_payload_sent(delivery.receipt, outgoing)
    assert ledger.initial_context_for_chain("child", payload).status == "proceed"


@pytest.mark.parametrize(
    "unsafe",
    ["../escape", "/absolute", r"C:\absolute", "has space", "line\nbreak", "x" * 129],
)
@pytest.mark.parametrize("field_name", ["receipt_id", "chain_id", "row_key"])
def test_receipt_rejects_unsafe_or_unbounded_identifiers(unsafe, field_name):
    values = {
        "receipt_id": "receipt-1",
        "chain_id": "child:1",
        "row_keys": ("receipt-1-row-1",),
    }
    if field_name == "row_key":
        values["row_keys"] = (unsafe,)
    else:
        values[field_name] = unsafe
    with pytest.raises(ValueError, match="receipt"):
        InstructionDeliveryReceipt(
            **values,
            through_revision=0,
            source_digests=(),
            outcome_keys=("invalid\x1fsrc/AGENTS.md\x1fsrc",),
        )


def test_receipt_copies_mutable_sequences_before_validation_and_mark(tmp_path):
    ledger = InstructionActivationLedger(_snapshot(tmp_path), nested_max_bytes=100)
    payload, _ = _payload()
    delivery = ledger.initial_context_for_chain("child", payload)
    digests = list(delivery.receipt.source_digests)
    outcomes = list(delivery.receipt.outcome_keys)
    row_keys = list(delivery.receipt.row_keys)
    receipt = InstructionDeliveryReceipt(
        receipt_id=delivery.receipt.receipt_id,
        chain_id=delivery.receipt.chain_id,
        through_revision=delivery.receipt.through_revision,
        source_digests=digests,  # type: ignore[arg-type]
        outcome_keys=outcomes,  # type: ignore[arg-type]
        row_keys=row_keys,  # type: ignore[arg-type]
    )
    original_hash = hash(receipt)

    digests.append("f" * 64)
    outcomes.append("invalid\x1fforged/AGENTS.md\x1fforged")
    row_keys.append("forged-row")

    assert hash(receipt) == original_hash
    assert isinstance(receipt.source_digests, tuple)
    assert isinstance(receipt.outcome_keys, tuple)
    assert isinstance(receipt.row_keys, tuple)
    assert "forged" not in repr(receipt)
    ledger.mark_payload_sent(receipt, delivery.rows)


def test_receipt_rejects_nested_mutable_digest():
    mutable_digest = list("0" * 64)

    with pytest.raises(ValueError, match="source digest"):
        InstructionDeliveryReceipt(
            receipt_id="receipt",
            chain_id="chain",
            through_revision=0,
            source_digests=(mutable_digest,),  # type: ignore[arg-type]
            outcome_keys=(),
            row_keys=("row",),
        )


def test_receipt_rejects_nested_mutable_outcome_key():
    with pytest.raises(ValueError, match="outcome key"):
        InstructionDeliveryReceipt(
            receipt_id="receipt",
            chain_id="chain",
            through_revision=0,
            source_digests=(),
            outcome_keys=(["invalid", "src/AGENTS.md", "src"],),  # type: ignore[arg-type]
            row_keys=("row",),
        )


def test_receipt_rejects_nested_mutable_row_key():
    with pytest.raises(ValueError, match="row keys"):
        InstructionDeliveryReceipt(
            receipt_id="receipt",
            chain_id="chain",
            through_revision=0,
            source_digests=(),
            outcome_keys=(),
            row_keys=(["row"],),  # type: ignore[arg-type]
        )


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
    messages = [{"role": "user", "content": [{"type": "text", "text": "captured"}]}]
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
    ledger.mark_payload_sent(first.receipt, first.rows)

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
    assert (
        nested_delivery.receipt.through_revision
        > root_delivery.receipt.through_revision
    )
    ledger.mark_payload_sent(root_delivery.receipt, root_delivery.rows)

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
    assert first.status == "retry_with_context"
    assert any("omitted_token_budget" in row["content"] for row in first.rows)
    assert "omitted_token_budget" in ledger.warning_keys
    assert ledger.initial_context_for_chain("child", payload).receipt == first.receipt
    with pytest.raises(ValueError, match="payload rows"):
        ledger.mark_payload_sent(first.receipt, [])

    ledger.mark_payload_sent(first.receipt, first.rows)
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
    assert "omitted_token_budget" in ledger.warning_keys


def test_unmeasurable_warning_only_delivery_remains_pending_until_exact_mark(
    tmp_path,
):
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

    def fail_estimator(_rows):
        raise RuntimeError("estimator unavailable")

    ledger = InstructionActivationLedger(snapshot, nested_max_bytes=100)
    payload, _ = _payload(allowance=10, count_tokens=fail_estimator)

    first = ledger.initial_context_for_chain("primary", payload)
    assert first.status == "retry_with_context"
    assert any("invalid" in row["content"] for row in first.rows)
    assert ledger.initial_context_for_chain("primary", payload).receipt == first.receipt
    # Task 10 must terminal-error rather than mark/send when bounding drops this row.
    with pytest.raises(ValueError, match="payload rows"):
        ledger.mark_payload_sent(first.receipt, [])

    ledger.mark_payload_sent(first.receipt, first.rows)
    assert ledger.initial_context_for_chain("primary", payload).status == "proceed"


def _wrapped_row_tokens(rows):
    if not rows:
        return 0
    return 2 + sum(
        5 if row["content"].startswith("Project instructions (") else 2 for row in rows
    )


def test_token_admission_charges_global_and_new_omission_rows_with_wrapper(tmp_path):
    target = tmp_path / "src" / "pkg"
    target.mkdir(parents=True)
    (tmp_path / "src" / "AGENTS.md").write_text("broad")
    (target / "AGENTS.md").write_text("deep")
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
    payload, _ = _payload(allowance=9, count_tokens=_wrapped_row_tokens)

    delivery = ledger.prepare(
        [ToolCall("read", {}, "call")],
        "primary",
        _registry(target / "file.py"),
        payload,
    )

    assert _wrapped_row_tokens(delivery.rows) <= 9
    assert all("deep" not in row["content"] for row in delivery.rows)
    assert sum("omitted_token_budget" in row["content"] for row in delivery.rows) == 2
    assert any("invalid" in row["content"] for row in delivery.rows)
    assert "omitted_token_budget" in ledger.warning_keys


def test_token_admission_charges_outside_warning_before_staging_source(tmp_path):
    nested = tmp_path / "src"
    nested.mkdir()
    (nested / "AGENTS.md").write_text("nested")

    class ExactAndOutsideProvider(_PathProvider):
        def path_targets(self, tool_id, args):
            return (
                ToolPathTarget(nested / "file.py", "exact"),
                ToolPathTarget(tmp_path.parent / "other", "outside"),
            )

    registry = ToolCatalogRegistry()
    registry.register_provider(ExactAndOutsideProvider(nested / "file.py"))
    ledger = InstructionActivationLedger(_snapshot(tmp_path), nested_max_bytes=100)
    payload, _ = _payload(allowance=8, count_tokens=_wrapped_row_tokens)

    delivery = ledger.prepare(
        [ToolCall("read", {}, "call")], "primary", registry, payload
    )

    assert _wrapped_row_tokens(delivery.rows) <= 8
    assert any("outside_instruction_scope" in row["content"] for row in delivery.rows)
    assert any("omitted_token_budget" in row["content"] for row in delivery.rows)
    assert all("nested\n" not in row["content"] for row in delivery.rows)


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
    ledger.mark_payload_sent(first.receipt, first.rows)
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
    ledger.mark_payload_sent(first.receipt, first.rows)
    path.write_text("must wait until next dispatch")
    real_open = resolver_module.os.open

    def refuse_terminal_reread(candidate, flags):
        if Path(candidate) == path:
            raise AssertionError("terminal source must not be reread")
        return real_open(candidate, flags)

    monkeypatch.setattr(resolver_module.os, "open", refuse_terminal_reread)

    retry = ledger.prepare(calls, "primary", registry, payload)
    assert retry.status == "proceed"
    assert all(
        "must wait until next dispatch" not in row["content"] for row in retry.rows
    )


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
    ledger.mark_payload_sent(first.receipt, first.rows)
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
    ledger.mark_payload_sent(first.receipt, first.rows)
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


def test_later_batch_runs_one_deepest_first_greedy_pass_on_remaining_budget(tmp_path):
    used = tmp_path / "used"
    target = tmp_path / "src" / "pkg"
    used.mkdir()
    target.mkdir(parents=True)
    (used / "AGENTS.md").write_text("12345")
    (tmp_path / "src" / "AGENTS.md").write_text("1234")
    (target / "AGENTS.md").write_text("1234567")
    ledger = InstructionActivationLedger(_snapshot(tmp_path), nested_max_bytes=10)
    payload, _ = _payload()

    first = ledger.prepare(
        [ToolCall("read", {}, "used")],
        "primary",
        _registry(used / "file.py"),
        payload,
    )
    ledger.mark_payload_sent(first.receipt, first.rows)
    later = ledger.prepare(
        [ToolCall("read", {}, "later")],
        "primary",
        _registry(target / "file.py"),
        payload,
    )

    rendered = "\n".join(row["content"] for row in later.rows)
    assert "src/AGENTS.md" in rendered
    assert "1234" in rendered
    assert "src/pkg/AGENTS.md" in rendered
    assert "omitted_byte_budget" in rendered
    assert "1234567" not in rendered
    assert ledger.remaining_nested_bytes == 1


def test_ledger_rejects_old_pin_after_binding_root_replacement(tmp_path):
    root = tmp_path / "selected"
    nested = root / "src"
    nested.mkdir(parents=True)
    (nested / "AGENTS.md").write_text("OLD_PIN_SECRET")
    ledger = InstructionActivationLedger(_snapshot(root), nested_max_bytes=100)
    payload, _ = _payload()
    calls = [ToolCall("read", {}, "call")]
    registry = _registry(nested / "file.py")

    first = ledger.prepare(calls, "primary", registry, payload)
    assert any("OLD_PIN_SECRET" in row["content"] for row in first.rows)
    ledger.mark_payload_sent(first.receipt, first.rows)

    displaced = tmp_path / "displaced"
    root.rename(displaced)
    (root / "src").mkdir(parents=True)
    (root / "src" / "AGENTS.md").write_text("replacement")

    second = ledger.prepare(calls, "primary", registry, payload)
    rendered = repr(second.rows)
    assert "resolution_failed" in rendered
    assert "OLD_PIN_SECRET" not in rendered
    assert "replacement" not in rendered


def test_nested_pinned_source_is_reused_by_identity_after_delete(tmp_path):
    target = tmp_path / "src"
    target.mkdir()
    path = target / "AGENTS.md"
    path.write_text("pinned")
    resolver = ProjectInstructionResolver()
    identity = capture_binding_root_identity(tmp_path)
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
        expected_binding_identity=identity,
    )

    assert second.sources[0] is first.sources[0]
    assert second.outcomes == ()


@pytest.mark.parametrize("identity_kind", ["absent", "mismatched"])
def test_direct_pinned_resolution_requires_matching_dispatch_authority(
    tmp_path, identity_kind
):
    target = tmp_path / "src"
    target.mkdir()
    source = _source(tmp_path, "src/AGENTS.md", "PIN_WITHOUT_AUTHORITY")
    identity = None
    if identity_kind == "mismatched":
        other = tmp_path / "other"
        other.mkdir()
        identity = capture_binding_root_identity(other)

    batch = ProjectInstructionResolver().resolve_targets(
        tmp_path,
        [target],
        max_bytes=100,
        dispatch_started_wall_ns=time.time_ns() + 1_000_000_000,
        pinned_by_canonical_path={source.canonical_path: source},
        expected_binding_identity=identity,
    )

    assert batch.sources == ()
    assert [outcome.code for outcome in batch.outcomes] == ["resolution_failed"]
    assert "PIN_WITHOUT_AUTHORITY" not in repr(batch)


@pytest.mark.parametrize(
    ("relative_path", "scope"),
    [("../outside/AGENTS.md", "src"), ("src/AGENTS.md", "../outside")],
)
def test_nested_rejects_injected_pinned_source_metadata(tmp_path, relative_path, scope):
    target = tmp_path / "src"
    target.mkdir()
    expected_path = target / "AGENTS.md"
    identity = capture_binding_root_identity(tmp_path)
    injected = InstructionSource(
        canonical_path=tmp_path.parent / "outside" / "AGENTS.md",
        relative_path=relative_path,
        scope=scope,
        kind="standard",
        body="PINNED_SECRET_MUST_NOT_ESCAPE",
        byte_count=29,
        digest="0" * 64,
    )

    batch = ProjectInstructionResolver().resolve_targets(
        tmp_path,
        [target],
        max_bytes=100,
        dispatch_started_wall_ns=time.time_ns() + 1_000_000_000,
        pinned_by_canonical_path={expected_path: injected},
        expected_binding_identity=identity,
    )

    assert batch.sources == ()
    assert [outcome.code for outcome in batch.outcomes] == ["resolution_failed"]
    assert "PINNED_SECRET_MUST_NOT_ESCAPE" not in repr(batch)


@pytest.mark.parametrize(
    ("body", "byte_count", "digest"),
    [
        ("forged", len(b"trusted"), hashlib.sha256(b"trusted").hexdigest()),
        ("trusted", len(b"trusted") + 1, hashlib.sha256(b"trusted").hexdigest()),
        ("trusted", len(b"trusted"), "0" * 64),
        ("x", True, hashlib.sha256(b"x").hexdigest()),
    ],
)
def test_nested_rejects_pinned_source_with_forged_content_identity(
    tmp_path, body, byte_count, digest
):
    target = tmp_path / "src"
    target.mkdir()
    expected_path = target / "AGENTS.md"
    identity = capture_binding_root_identity(tmp_path)
    forged = InstructionSource(
        canonical_path=expected_path,
        relative_path="src/AGENTS.md",
        scope="src",
        kind="standard",
        body=body,
        byte_count=byte_count,
        digest=digest,
    )

    batch = ProjectInstructionResolver().resolve_targets(
        tmp_path,
        [target],
        max_bytes=100,
        dispatch_started_wall_ns=time.time_ns() + 1_000_000_000,
        pinned_by_canonical_path={expected_path: forged},
        expected_binding_identity=identity,
    )

    assert batch.sources == ()
    assert [outcome.code for outcome in batch.outcomes] == ["resolution_failed"]
    assert "forged" not in repr(batch)


def test_nested_pinned_reuse_rejects_binding_root_swap(tmp_path, monkeypatch):
    root = tmp_path / "selected"
    nested = root / "src"
    nested.mkdir(parents=True)
    source = _source(root, "src/AGENTS.md", "pinned")
    identity = capture_binding_root_identity(root)
    displaced = tmp_path / "displaced"
    real_lstat = resolver_module.os.lstat
    nested_lstats = 0

    def swap_before_pinned_reuse(candidate):
        nonlocal nested_lstats
        if Path(candidate) == nested:
            nested_lstats += 1
            if nested_lstats == 2:
                root.rename(displaced)
                (root / "src").mkdir(parents=True)
        return real_lstat(candidate)

    monkeypatch.setattr(resolver_module.os, "lstat", swap_before_pinned_reuse)

    batch = ProjectInstructionResolver().resolve_targets(
        root,
        [nested],
        max_bytes=100,
        dispatch_started_wall_ns=time.time_ns() + 1_000_000_000,
        pinned_by_canonical_path={source.canonical_path: source},
        expected_binding_identity=identity,
    )

    assert nested_lstats >= 2
    assert batch.sources == ()
    assert [outcome.code for outcome in batch.outcomes] == ["resolution_failed"]


def test_nested_pinned_reuse_rechecks_root_after_metadata_validation(
    tmp_path, monkeypatch
):
    root = tmp_path / "selected"
    nested = root / "src"
    nested.mkdir(parents=True)
    source = _source(root, "src/AGENTS.md", "pinned")
    identity = capture_binding_root_identity(root)
    displaced = tmp_path / "displaced"
    real_validator = resolver_module._valid_pinned_source

    def validate_then_swap(**kwargs):
        valid = real_validator(**kwargs)
        root.rename(displaced)
        (root / "src").mkdir(parents=True)
        return valid

    monkeypatch.setattr(resolver_module, "_valid_pinned_source", validate_then_swap)

    batch = ProjectInstructionResolver().resolve_targets(
        root,
        [nested],
        max_bytes=100,
        dispatch_started_wall_ns=time.time_ns() + 1_000_000_000,
        pinned_by_canonical_path={source.canonical_path: source},
        expected_binding_identity=identity,
    )

    assert batch.sources == ()
    assert [outcome.code for outcome in batch.outcomes] == ["resolution_failed"]


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
