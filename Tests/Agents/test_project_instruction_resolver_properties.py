from __future__ import annotations

import hashlib
import tempfile
import time
from pathlib import Path

from hypothesis import given, strategies as st

from tldw_chatbook.Agents.project_instruction_resolver import (
    InstructionSource,
    ProjectInstructionResolver,
    admit_sources,
)


def _source(index: int, byte_count: int) -> InstructionSource:
    body = "x" * byte_count
    return InstructionSource(
        canonical_path=Path(f"/memory-only/{index}/AGENTS.md"),
        relative_path=f"scope-{index}/AGENTS.md",
        scope=f"scope-{index}",
        kind="standard",
        body=body,
        byte_count=byte_count,
        digest=hashlib.sha256(f"{index}:{body}".encode()).hexdigest(),
    )


@given(
    sizes=st.lists(st.integers(min_value=0, max_value=128), max_size=12),
    budget=st.integers(min_value=0, max_value=512),
)
def test_token_admission_never_exceeds_budget_and_never_truncates(
    sizes: list[int], budget: int
) -> None:
    sources = tuple(_source(index, size) for index, size in enumerate(sizes))

    delivery = admit_sources(
        sources,
        safe_input_tokens=budget,
        count_tokens=lambda source: source.byte_count,
    )

    by_digest = {source.digest: source for source in sources}
    admitted = [by_digest[digest] for digest in delivery.source_digests]
    assert sum(source.byte_count for source in admitted) <= budget
    assert set(delivery.source_digests).isdisjoint(
        {
            source.digest
            for source in sources
            if source.relative_path
            in {outcome.relative_path for outcome in delivery.outcomes}
        }
    )


@given(
    size=st.integers(min_value=1, max_value=256),
    budget=st.integers(min_value=0, max_value=256),
)
def test_startup_byte_budget_admits_or_omits_the_whole_source(
    size: int, budget: int
) -> None:
    with tempfile.TemporaryDirectory(
        dir=Path(tempfile.gettempdir()).resolve()
    ) as directory:
        root = Path(directory)
        (root / "AGENTS.md").write_bytes(b"x" * size)

        candidate = ProjectInstructionResolver().resolve_startup(
            binding_id="binding",
            binding_root=root,
            locator_fingerprint="fingerprint",
            max_bytes=budget,
            dispatch_started_wall_ns=time.time_ns(),
        )

        if size <= budget:
            assert candidate.source is not None
            assert candidate.source.byte_count == size
            assert candidate.outcomes == ()
        else:
            assert candidate.source is None
            assert [outcome.code for outcome in candidate.outcomes] == [
                "omitted_byte_budget"
            ]


@given(
    component=st.text(
        alphabet=st.characters(
            whitelist_categories=("Ll", "Lu", "Nd"),
            whitelist_characters="-_",
        ),
        min_size=1,
        max_size=20,
    ),
    body=st.text(min_size=1, max_size=100).filter(lambda value: bool(value.strip())),
)
def test_resolved_source_is_canonically_confined_to_selected_root(
    component: str, body: str
) -> None:
    with tempfile.TemporaryDirectory(
        dir=Path(tempfile.gettempdir()).resolve()
    ) as directory:
        root = Path(directory) / component
        root.mkdir()
        raw = body.encode("utf-8")
        (root / "AGENTS.md").write_bytes(raw)

        candidate = ProjectInstructionResolver().resolve_startup(
            binding_id="binding",
            binding_root=root,
            locator_fingerprint="fingerprint",
            max_bytes=len(raw),
            dispatch_started_wall_ns=time.time_ns(),
        )

        assert candidate.source is not None
        assert candidate.source.canonical_path.parent == root.resolve()
        assert candidate.source.canonical_path.is_relative_to(root.resolve())
        assert candidate.source.byte_count == len(raw)
