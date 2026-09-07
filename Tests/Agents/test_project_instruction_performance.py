from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from tldw_chatbook.Agents import project_instruction_resolver as resolver_module
from tldw_chatbook.Agents.project_instruction_resolver import (
    ProjectInstructionResolver,
)


TREE_DEPTH = 32
DECOY_BRANCHES = 32


def _deep_tree(root: Path) -> tuple[Path, tuple[Path, ...]]:
    target = root / "target"
    target.mkdir()
    chain = [target]
    (target / "AGENTS.md").write_text("level 0")
    for depth in range(1, TREE_DEPTH):
        target /= f"level-{depth:02d}"
        target.mkdir()
        chain.append(target)
        (target / "AGENTS.md").write_text(f"level {depth}")
    for index in range(DECOY_BRANCHES):
        decoy = root / f"decoy-{index:02d}" / "nested"
        decoy.mkdir(parents=True)
        (decoy / "AGENTS.md").write_text("must not be inspected")
    return target, tuple(chain)


def test_startup_inspects_only_the_binding_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "AGENTS.md").write_text("root guidance")
    _deep_tree(root)
    inspected: list[Path] = []
    real_read_candidate = resolver_module._read_candidate

    def counted_read_candidate(**kwargs: Any):
        inspected.append(kwargs["root"])
        return real_read_candidate(**kwargs)

    monkeypatch.setattr(resolver_module, "_read_candidate", counted_read_candidate)
    started = time.perf_counter_ns()

    result = ProjectInstructionResolver().resolve_startup(
        binding_id="binding-1",
        binding_root=root,
        locator_fingerprint="fingerprint",
        max_bytes=32_768,
        dispatch_started_wall_ns=time.time_ns() + 1_000_000_000,
    )

    elapsed_ns = time.perf_counter_ns() - started
    print(f"startup_elapsed_ns={elapsed_ns}")
    assert result.source is not None
    assert tuple(dict.fromkeys(inspected)) == (root,)


def test_nested_activation_inspects_exactly_the_root_to_target_depth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    target, expected_chain = _deep_tree(root)
    inspected: list[Path] = []
    real_resolve_directory = resolver_module._resolve_nested_directory

    def counted_resolve_directory(**kwargs: Any):
        inspected.append(kwargs["directory"])
        return real_resolve_directory(**kwargs)

    monkeypatch.setattr(
        resolver_module,
        "_resolve_nested_directory",
        counted_resolve_directory,
    )
    started = time.perf_counter_ns()

    result = ProjectInstructionResolver().resolve_targets(
        root,
        [target],
        max_bytes=32_768,
        dispatch_started_wall_ns=time.time_ns() + 1_000_000_000,
        pinned_by_canonical_path={},
    )

    elapsed_ns = time.perf_counter_ns() - started
    print(f"nested_elapsed_ns={elapsed_ns}")
    assert tuple(inspected) == expected_chain
    assert len(result.sources) == TREE_DEPTH
    assert all("decoy" not in source.relative_path for source in result.sources)
