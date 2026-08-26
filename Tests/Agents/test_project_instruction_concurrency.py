from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path

from tldw_chatbook.Agents.agent_models import (
    ToolCall,
    ToolCatalogEntry,
    ToolResult,
    ToolSchema,
)
from tldw_chatbook.Agents.project_instruction_resolver import (
    InstructionChainDelivery,
    InstructionSnapshot,
    InstructionSource,
    ProjectInstructionResolver,
)
from tldw_chatbook.Agents.project_instruction_runtime import (
    InstructionActivationLedger,
    InstructionChainPayloadState,
)
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry, ToolPathTarget


def _source(root: Path) -> InstructionSource:
    body = "root"
    return InstructionSource(
        canonical_path=root / "AGENTS.md",
        relative_path="AGENTS.md",
        scope=".",
        kind="standard",
        body=body,
        byte_count=len(body),
        digest=hashlib.sha256(body.encode()).hexdigest(),
    )


def _ledger(root: Path, resolver=None) -> InstructionActivationLedger:
    source = _source(root)
    return InstructionActivationLedger(
        InstructionSnapshot(
            binding_id="binding",
            binding_root=root,
            locator_fingerprint="fingerprint",
            dispatch_started_wall_ns=time.time_ns() + 1_000_000_000,
            startup_source=source,
            global_outcomes=(),
            primary_delivery=InstructionChainDelivery((source.digest,), ()),
            warning_codes=(),
        ),
        nested_max_bytes=6,
        resolver=resolver,
    )


def _payload() -> InstructionChainPayloadState:
    state = InstructionChainPayloadState(
        request_builder=lambda messages, schemas: (messages, schemas),
        safe_token_allowance=lambda request, rows: 1_000,
        count_tokens=lambda rows: len(rows),
    )
    state.capture(messages=[], active_schemas=(), calls=[])
    return state


class _Provider:
    def __init__(self, target: Path) -> None:
        self.target = target

    def list_catalog(self):
        name = self.target.name
        return [ToolCatalogEntry(f"fake:{name}", name, name, "local")]

    def load_schema(self, tool_id):
        return ToolSchema(tool_id, self.target.name, "", {})

    def invoke(self, tool_id, args):
        return ToolResult(ok=True)

    def path_targets(self, tool_id, args):
        return (ToolPathTarget(self.target / "file.py", "exact"),)


def _registry(*targets: Path) -> ToolCatalogRegistry:
    registry = ToolCatalogRegistry()
    for target in targets:
        registry.register_provider(_Provider(target))
    return registry


class _BlockingResolver(ProjectInstructionResolver):
    def __init__(self, entered: threading.Barrier, release: threading.Barrier) -> None:
        self.entered = entered
        self.release = release
        self.first = True

    def resolve_targets(self, *args, **kwargs):
        if self.first:
            self.first = False
            self.entered.wait()
            self.release.wait()
        return super().resolve_targets(*args, **kwargs)


def test_first_lock_wins_shared_nested_byte_budget_without_sleeps(tmp_path: Path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    (first_dir / "AGENTS.md").write_text("111111")
    (second_dir / "AGENTS.md").write_text("222222")
    entered = threading.Barrier(2)
    release = threading.Barrier(2)
    ledger = _ledger(tmp_path, _BlockingResolver(entered, release))
    registry = _registry(first_dir, second_dir)
    payload = _payload()
    results = {}

    def prepare(name: str):
        results[name] = ledger.prepare(
            [ToolCall(name, {}, name)], name, registry, payload
        )

    winner = threading.Thread(target=prepare, args=("first",))
    loser = threading.Thread(target=prepare, args=("second",))
    winner.start()
    entered.wait()
    loser.start()
    release.wait()
    winner.join()
    loser.join()

    assert any("111111" in row["content"] for row in results["first"].rows)
    assert all("222222" not in row["content"] for row in results["second"].rows)
    assert any(
        "omitted_byte_budget" in row["content"] for row in results["second"].rows
    )
    assert ledger.remaining_nested_bytes == 0


def test_concurrent_later_chain_receives_first_chains_activated_source(tmp_path: Path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "AGENTS.md").write_text("shared")
    ledger = _ledger(tmp_path)
    registry = _registry(nested)
    payload = _payload()
    start = threading.Barrier(3)
    results = {}

    def prepare(chain: str):
        start.wait()
        results[chain] = ledger.prepare(
            [ToolCall("nested", {}, chain)], chain, registry, payload
        )

    threads = [threading.Thread(target=prepare, args=(chain,)) for chain in ("a", "b")]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join()

    assert all(
        any("shared" in row["content"] for row in result.rows)
        for result in results.values()
    )
    assert ledger.remaining_nested_bytes == 0


def test_one_batch_admits_deepest_source_deterministically(tmp_path: Path):
    target = tmp_path / "src" / "pkg"
    target.mkdir(parents=True)
    (tmp_path / "src" / "AGENTS.md").write_text("broad!")
    (target / "AGENTS.md").write_text("deep!!")
    ledger = _ledger(tmp_path)
    payload = _payload()

    result = ledger.prepare(
        [ToolCall("pkg", {}, "call")], "primary", _registry(target), payload
    )

    rendered = "\n".join(row["content"] for row in result.rows)
    assert "deep!!" in rendered
    assert "broad!" not in rendered
    assert "omitted_byte_budget" in rendered
