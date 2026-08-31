from __future__ import annotations

import hashlib
import os
import time
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents import project_instruction_resolver as resolver_module
from tldw_chatbook.Agents.project_instruction_resolver import (
    InstructionChainDelivery,
    InstructionOutcome,
    InstructionSnapshot,
    InstructionSource,
    ProjectInstructionResolver,
    InstructionPromotionSnapshotError,
    StartupInstructionCandidate,
    admit_sources,
)


def test_promotion_snapshot_captures_target_state_and_effective_chain(
    tmp_path: Path,
) -> None:
    (tmp_path / "AGENTS.md").write_text("root")
    nested = tmp_path / "nested"
    nested.mkdir()
    target = nested / "AGENTS.override.md"
    target.write_text("nested")

    snapshot = ProjectInstructionResolver().snapshot_promotion_target(
        binding_id="binding-1",
        binding_root=tmp_path,
        locator_fingerprint="fingerprint",
        target_path=target,
        activation_revision=7,
    )

    assert snapshot.target_relative_path == "nested/AGENTS.override.md"
    assert snapshot.expected_absent is False
    assert snapshot.expected_sha256 == hashlib.sha256(b"nested").hexdigest()
    assert snapshot.effective_chain == (
        ("AGENTS.md", "standard", hashlib.sha256(b"root").hexdigest()),
        (
            "nested/AGENTS.override.md",
            "override",
            hashlib.sha256(b"nested").hexdigest(),
        ),
    )
    assert snapshot.activation_revision == 7
    assert len(snapshot.effective_chain_digest) == 64
    assert len(snapshot.root_identity_digest) == 64
    assert str(tmp_path) not in repr(snapshot)


def test_promotion_snapshot_represents_missing_target_as_expected_absent(
    tmp_path: Path,
) -> None:
    snapshot = ProjectInstructionResolver().snapshot_promotion_target(
        binding_id="binding-1",
        binding_root=tmp_path,
        locator_fingerprint="fingerprint",
        target_path=tmp_path / "AGENTS.md",
        activation_revision=0,
    )

    assert snapshot.expected_absent is True
    assert snapshot.expected_sha256 is None
    assert snapshot.effective_chain == ()


def test_promotion_snapshot_rejects_symlink_target(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.write_text("outside")
    selected = tmp_path / "selected"
    selected.mkdir()
    (selected / "AGENTS.md").symlink_to(outside)

    with pytest.raises(InstructionPromotionSnapshotError) as failure:
        ProjectInstructionResolver().snapshot_promotion_target(
            binding_id="binding-1",
            binding_root=selected,
            locator_fingerprint="fingerprint",
            target_path=selected / "AGENTS.md",
            activation_revision=0,
        )

    assert failure.value.code == "invalid_target"


def _resolve(
    root: Path,
    *,
    max_bytes: int = 32_768,
    dispatch_started_wall_ns: int | None = None,
):
    return ProjectInstructionResolver().resolve_startup(
        binding_id="binding-1",
        binding_root=root,
        locator_fingerprint="locator-sha256",
        max_bytes=max_bytes,
        dispatch_started_wall_ns=(
            time.time_ns()
            if dispatch_started_wall_ns is None
            else dispatch_started_wall_ns
        ),
    )


def _source(*, body: str, relative_path: str = "AGENTS.md") -> InstructionSource:
    raw = body.encode()
    return InstructionSource(
        canonical_path=Path("/not-persisted") / relative_path,
        relative_path=relative_path,
        scope=".",
        kind="standard",
        body=body,
        byte_count=len(raw),
        digest=hashlib.sha256(raw).hexdigest(),
    )


def test_startup_selects_only_binding_root_override(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("standard")
    (tmp_path / "AGENTS.override.md").write_text("override")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "AGENTS.override.md").write_text("nested")

    candidate = _resolve(tmp_path)

    assert candidate.source is not None
    assert candidate.source.kind == "override"
    assert candidate.source.body == "override"
    assert candidate.source.relative_path == "AGENTS.override.md"
    assert candidate.source.scope == "."
    assert candidate.outcomes == ()


@pytest.mark.parametrize("override", [b"", b" \n\t\r"])
def test_empty_or_whitespace_override_falls_back_to_standard(
    tmp_path: Path, override: bytes
) -> None:
    (tmp_path / "AGENTS.override.md").write_bytes(override)
    (tmp_path / "AGENTS.md").write_text("standard")

    candidate = _resolve(tmp_path)

    assert candidate.source is not None
    assert candidate.source.kind == "standard"
    assert candidate.source.body == "standard"


@pytest.mark.parametrize("bad_override", [b"\xff", b"\xef\xbb\xbf\xff"])
def test_invalid_override_suppresses_standard_fallback(
    tmp_path: Path, bad_override: bytes
) -> None:
    (tmp_path / "AGENTS.override.md").write_bytes(bad_override)
    (tmp_path / "AGENTS.md").write_text("must not load")

    candidate = _resolve(tmp_path)

    assert candidate.source is None
    assert [(item.relative_path, item.code) for item in candidate.outcomes] == [
        ("AGENTS.override.md", "invalid")
    ]


def test_unreadable_override_suppresses_standard_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = tmp_path / "AGENTS.override.md"
    override.write_text("override")
    (tmp_path / "AGENTS.md").write_text("must not load")
    real_open = resolver_module.os.open

    def refuse_override(path: os.PathLike[str] | str, flags: int) -> int:
        if Path(path) == override:
            raise PermissionError("private detail must not escape")
        return real_open(path, flags)

    monkeypatch.setattr(resolver_module.os, "open", refuse_override)

    candidate = _resolve(tmp_path)

    assert candidate.source is None
    assert [(item.relative_path, item.code) for item in candidate.outcomes] == [
        ("AGENTS.override.md", "resolution_failed")
    ]


def test_oversized_override_is_not_read_and_suppresses_standard_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = tmp_path / "AGENTS.override.md"
    override.write_bytes(b"12345")
    (tmp_path / "AGENTS.md").write_text("must not load")
    real_open = resolver_module.os.open
    opened: list[Path] = []

    def recording_open(path: os.PathLike[str] | str, flags: int) -> int:
        opened.append(Path(path))
        return real_open(path, flags)

    monkeypatch.setattr(resolver_module.os, "open", recording_open)

    candidate = _resolve(tmp_path, max_bytes=4)

    assert candidate.source is None
    assert override not in opened
    assert [(item.relative_path, item.code) for item in candidate.outcomes] == [
        ("AGENTS.override.md", "omitted_byte_budget")
    ]


def test_utf8_bom_is_removed_from_body_but_counted_and_hashed(tmp_path: Path) -> None:
    raw = b"\xef\xbb\xbfroot guidance"
    (tmp_path / "AGENTS.md").write_bytes(raw)

    candidate = _resolve(tmp_path)

    assert candidate.source is not None
    assert candidate.source.body == "root guidance"
    assert candidate.source.byte_count == len(raw)
    assert candidate.source.digest == hashlib.sha256(raw).hexdigest()
    assert candidate.source.canonical_path == (tmp_path / "AGENTS.md").resolve()


def test_sensitive_source_fields_are_not_exposed_by_default_repr(
    tmp_path: Path,
) -> None:
    secret_body = "private project guidance"
    (tmp_path / "AGENTS.md").write_text(secret_body)

    candidate = _resolve(tmp_path)

    assert candidate.source is not None
    rendered = repr(candidate)
    assert secret_body not in rendered
    assert str(tmp_path) not in rendered
    assert candidate.source.digest not in rendered


def test_standard_file_requires_strict_utf8(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_bytes(b"valid\xffinvalid")

    candidate = _resolve(tmp_path)

    assert candidate.source is None
    assert [(item.relative_path, item.code) for item in candidate.outcomes] == [
        ("AGENTS.md", "invalid")
    ]


def test_startup_does_not_discover_parent_sibling_or_global_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "selected"
    sibling = tmp_path / "sibling"
    fake_home = tmp_path / "home"
    root.mkdir()
    sibling.mkdir()
    fake_home.mkdir()
    (tmp_path / "AGENTS.md").write_text("parent")
    (sibling / "AGENTS.md").write_text("sibling")
    (fake_home / "AGENTS.md").write_text("global")
    monkeypatch.setenv("HOME", str(fake_home))

    candidate = _resolve(root)

    assert candidate.source is None
    assert candidate.outcomes == ()


def test_startup_never_recursively_walks_the_binding(
    tmp_path: Path, monkeypatch
) -> None:
    nested = tmp_path / "one" / "two"
    nested.mkdir(parents=True)
    (nested / "AGENTS.md").write_text("nested")

    def recursive_discovery_is_forbidden(*_args, **_kwargs):
        raise AssertionError("startup discovery must remain O(1)")

    monkeypatch.setattr(resolver_module.os, "walk", recursive_discovery_is_forbidden)
    monkeypatch.setattr(resolver_module.os, "scandir", recursive_discovery_is_forbidden)

    candidate = _resolve(tmp_path)

    assert candidate.source is None


def test_startup_examines_only_the_binding_root_and_two_candidate_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "AGENTS.md").write_text("root")
    real_lstat = resolver_module.os.lstat
    examined: list[Path] = []

    def recording_lstat(path: os.PathLike[str] | str):
        examined.append(Path(path))
        return real_lstat(path)

    monkeypatch.setattr(resolver_module.os, "lstat", recording_lstat)

    candidate = _resolve(tmp_path)

    assert candidate.source is not None
    inside_binding = {path for path in examined if path.is_relative_to(tmp_path)}
    assert inside_binding == {
        tmp_path,
        tmp_path / "AGENTS.override.md",
        tmp_path / "AGENTS.md",
    }


def test_root_replacement_between_override_and_standard_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "selected"
    root.mkdir()
    (root / "AGENTS.md").write_text("original standard")
    replaced = tmp_path / "replaced"
    override = root / "AGENTS.override.md"
    real_lstat = resolver_module.os.lstat
    raced = False

    def replace_root_on_missing_override(path: os.PathLike[str] | str):
        nonlocal raced
        try:
            return real_lstat(path)
        except FileNotFoundError:
            if Path(path) == override and not raced:
                raced = True
                root.rename(replaced)
                root.mkdir()
                (root / "AGENTS.md").write_text("replacement standard")
            raise

    monkeypatch.setattr(resolver_module.os, "lstat", replace_root_on_missing_override)

    candidate = _resolve(root, dispatch_started_wall_ns=2**63 - 1)

    assert raced is True
    assert candidate.source is None
    assert [(item.relative_path, item.code) for item in candidate.outcomes] == [
        ("AGENTS.override.md", "resolution_failed")
    ]


def test_override_created_between_candidates_suppresses_standard_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    standard = tmp_path / "AGENTS.md"
    override = tmp_path / "AGENTS.override.md"
    standard.write_text("standard")
    dispatch_started = time.time_ns() + 1_000_000_000
    real_lstat = resolver_module.os.lstat
    created = False

    def create_override_before_standard(path: os.PathLike[str] | str):
        nonlocal created
        if Path(path) == standard and not created:
            created = True
            override.write_text("new override")
            os.utime(
                override,
                ns=(dispatch_started + 1, dispatch_started + 1),
            )
        return real_lstat(path)

    monkeypatch.setattr(resolver_module.os, "lstat", create_override_before_standard)

    candidate = _resolve(tmp_path, dispatch_started_wall_ns=dispatch_started)

    assert created is True
    assert candidate.source is None
    assert [(item.relative_path, item.code) for item in candidate.outcomes] == [
        ("AGENTS.override.md", "stale")
    ]


def test_empty_override_mutating_between_candidates_suppresses_standard_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = tmp_path / "AGENTS.override.md"
    standard = tmp_path / "AGENTS.md"
    override.write_text(" \n")
    standard.write_text("standard")
    dispatch_started = time.time_ns() + 1_000_000_000
    real_lstat = resolver_module.os.lstat
    mutated = False

    def mutate_override_before_standard(path: os.PathLike[str] | str):
        nonlocal mutated
        if Path(path) == standard and not mutated:
            mutated = True
            override.write_text("now authoritative")
            os.utime(override, ns=(dispatch_started - 1, dispatch_started - 1))
        return real_lstat(path)

    monkeypatch.setattr(resolver_module.os, "lstat", mutate_override_before_standard)

    candidate = _resolve(tmp_path, dispatch_started_wall_ns=dispatch_started)

    assert mutated is True
    assert candidate.source is None
    assert [(item.relative_path, item.code) for item in candidate.outcomes] == [
        ("AGENTS.override.md", "resolution_failed")
    ]


def test_override_created_while_standard_is_absent_reports_stale_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    standard = tmp_path / "AGENTS.md"
    override = tmp_path / "AGENTS.override.md"
    dispatch_started = time.time_ns() + 1_000_000_000
    real_lstat = resolver_module.os.lstat
    created = False

    def create_override_on_missing_standard(path: os.PathLike[str] | str):
        nonlocal created
        try:
            return real_lstat(path)
        except FileNotFoundError:
            if Path(path) == standard and not created:
                created = True
                override.write_text("late override")
                os.utime(
                    override,
                    ns=(dispatch_started + 1, dispatch_started + 1),
                )
            raise

    monkeypatch.setattr(
        resolver_module.os, "lstat", create_override_on_missing_standard
    )

    candidate = _resolve(tmp_path, dispatch_started_wall_ns=dispatch_started)

    assert created is True
    assert candidate.source is None
    assert [(item.relative_path, item.code) for item in candidate.outcomes] == [
        ("AGENTS.override.md", "stale")
    ]


@pytest.mark.parametrize(
    ("standard_bytes", "max_bytes"),
    [(b"\xff", 16), (b"12345", 4)],
    ids=["invalid-standard", "omitted-standard"],
)
def test_empty_override_mutation_supersedes_invalid_or_omitted_standard_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    standard_bytes: bytes,
    max_bytes: int,
) -> None:
    override = tmp_path / "AGENTS.override.md"
    standard = tmp_path / "AGENTS.md"
    override.write_text(" \n")
    standard.write_bytes(standard_bytes)
    dispatch_started = time.time_ns() + 1_000_000_000
    real_lstat = resolver_module.os.lstat
    mutated = False

    def mutate_override_before_standard(path: os.PathLike[str] | str):
        nonlocal mutated
        if Path(path) == standard and not mutated:
            mutated = True
            override.write_text("x")
            os.utime(override, ns=(dispatch_started - 1, dispatch_started - 1))
        return real_lstat(path)

    monkeypatch.setattr(resolver_module.os, "lstat", mutate_override_before_standard)

    candidate = _resolve(
        tmp_path,
        max_bytes=max_bytes,
        dispatch_started_wall_ns=dispatch_started,
    )

    assert mutated is True
    assert candidate.source is None
    assert [(item.relative_path, item.code) for item in candidate.outcomes] == [
        ("AGENTS.override.md", "resolution_failed")
    ]


def test_ancestor_disappearing_after_root_validation_is_not_treated_as_absent_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "AGENTS.md").write_text("must not load")
    real_lstat = resolver_module.os.lstat
    root_calls = 0

    def disappearing_root(path: os.PathLike[str] | str):
        nonlocal root_calls
        if Path(path) == tmp_path:
            root_calls += 1
            if root_calls > 1:
                raise FileNotFoundError
        return real_lstat(path)

    monkeypatch.setattr(resolver_module.os, "lstat", disappearing_root)

    candidate = _resolve(tmp_path)

    assert candidate.source is None
    assert [(item.relative_path, item.code) for item in candidate.outcomes] == [
        ("AGENTS.override.md", "resolution_failed")
    ]


def test_ancestor_metadata_becoming_unverifiable_omits_only_selected_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "AGENTS.md").write_text("must not load")
    real_lstat = resolver_module.os.lstat
    root_calls = 0

    def losing_identity(path: os.PathLike[str] | str):
        nonlocal root_calls
        value = real_lstat(path)
        if Path(path) != tmp_path:
            return value
        root_calls += 1
        if root_calls == 1:
            return value
        return SimpleNamespace(st_mode=value.st_mode)

    monkeypatch.setattr(resolver_module.os, "lstat", losing_identity)

    candidate = _resolve(tmp_path)

    assert candidate.source is None
    assert [(item.relative_path, item.code) for item in candidate.outcomes] == [
        ("AGENTS.override.md", "resolution_failed")
    ]


def test_symlinked_instruction_file_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "real.md"
    target.write_text("do not follow")
    (tmp_path / "AGENTS.md").symlink_to(target)

    candidate = _resolve(tmp_path)

    assert candidate.source is None
    assert [item.code for item in candidate.outcomes] == ["invalid"]


def test_symlinked_binding_root_is_refused(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    (real_root / "AGENTS.md").write_text("do not infer through linked root")
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)

    candidate = _resolve(linked_root)

    assert candidate.source is None
    assert [item.code for item in candidate.outcomes] == ["resolution_failed"]


def test_symlinked_parent_component_is_refused(tmp_path: Path) -> None:
    real_parent = tmp_path / "real-parent"
    root = real_parent / "selected"
    root.mkdir(parents=True)
    (root / "AGENTS.md").write_text("do not infer through linked ancestor")
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    candidate = _resolve(linked_parent / "selected")

    assert candidate.source is None
    assert [item.code for item in candidate.outcomes] == ["resolution_failed"]


@pytest.mark.skipif(
    not hasattr(os, "O_NOFOLLOW"), reason="platform does not expose O_NOFOLLOW"
)
def test_posix_open_uses_no_follow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instruction = tmp_path / "AGENTS.md"
    instruction.write_text("safe")
    real_open = resolver_module.os.open
    observed_flags: list[int] = []

    def recording_open(path: os.PathLike[str] | str, flags: int) -> int:
        observed_flags.append(flags)
        return real_open(path, flags)

    monkeypatch.setattr(resolver_module.os, "open", recording_open)

    candidate = _resolve(tmp_path)

    assert candidate.source is not None
    assert observed_flags
    assert observed_flags[0] & os.O_NOFOLLOW


@pytest.mark.parametrize(
    ("constant_name", "sentinel"),
    [("_BINARY", 1 << 28), ("_NONBLOCK", 1 << 29)],
)
def test_descriptor_open_includes_required_platform_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    constant_name: str,
    sentinel: int,
) -> None:
    instruction = tmp_path / "AGENTS.md"
    instruction.write_text("safe descriptor")
    real_open = resolver_module.os.open
    observed_flags: list[int] = []
    monkeypatch.setattr(resolver_module, constant_name, sentinel, raising=False)

    def recording_open(path: os.PathLike[str] | str, flags: int) -> int:
        observed_flags.append(flags)
        return real_open(path, flags & ~sentinel)

    monkeypatch.setattr(resolver_module.os, "open", recording_open)

    candidate = _resolve(tmp_path)

    assert candidate.source is not None
    assert observed_flags[0] & sentinel


@pytest.mark.skipif(
    not hasattr(os, "mkfifo") or not hasattr(os, "O_NONBLOCK"),
    reason="platform does not expose FIFO/nonblocking descriptor primitives",
)
def test_raced_regular_file_to_fifo_fails_closed_without_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instruction = tmp_path / "AGENTS.md"
    instruction.write_text("race target")
    real_open = resolver_module.os.open
    raced = False

    def race_to_fifo(path: os.PathLike[str] | str, flags: int) -> int:
        nonlocal raced
        if Path(path) == instruction and not raced:
            raced = True
            instruction.unlink()
            os.mkfifo(instruction)
            assert flags & os.O_NONBLOCK
        return real_open(path, flags)

    monkeypatch.setattr(resolver_module.os, "open", race_to_fifo)

    candidate = _resolve(tmp_path)

    assert raced is True
    assert candidate.source is None
    assert [item.code for item in candidate.outcomes] == ["resolution_failed"]


def test_missing_posix_no_follow_still_uses_identity_checked_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "AGENTS.md").write_text("portable")
    monkeypatch.setattr(resolver_module, "_NOFOLLOW", None)

    candidate = _resolve(tmp_path)

    assert candidate.source is not None
    assert candidate.source.body == "portable"


def test_file_identity_change_during_read_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "AGENTS.md").write_text("unstable")
    real_fstat = resolver_module.os.fstat
    calls = 0

    def changing_fstat(descriptor: int):
        nonlocal calls
        calls += 1
        value = real_fstat(descriptor)
        if calls < 2:
            return value
        return SimpleNamespace(
            st_dev=value.st_dev,
            st_ino=value.st_ino + 1,
            st_mode=value.st_mode,
            st_size=value.st_size,
            st_mtime_ns=value.st_mtime_ns,
        )

    monkeypatch.setattr(resolver_module.os, "fstat", changing_fstat)

    candidate = _resolve(tmp_path)

    assert candidate.source is None
    assert [item.code for item in candidate.outcomes] == ["resolution_failed"]


def test_unrelated_ancestor_mtime_drift_does_not_invalidate_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instruction = tmp_path / "AGENTS.md"
    instruction.write_text("stable file")
    real_lstat = resolver_module.os.lstat
    real_open = resolver_module.os.open
    file_opened = False

    def changing_directory_mtime(path: os.PathLike[str] | str):
        value = real_lstat(path)
        if Path(path) != tmp_path or not file_opened:
            return value
        return SimpleNamespace(
            st_dev=value.st_dev,
            st_ino=value.st_ino,
            st_mode=value.st_mode,
            st_size=value.st_size + 1,
            st_mtime_ns=value.st_mtime_ns + 1,
        )

    def mark_open(path: os.PathLike[str] | str, flags: int) -> int:
        nonlocal file_opened
        descriptor = real_open(path, flags)
        file_opened = True
        return descriptor

    monkeypatch.setattr(resolver_module.os, "lstat", changing_directory_mtime)
    monkeypatch.setattr(resolver_module.os, "open", mark_open)

    candidate = _resolve(tmp_path)

    assert candidate.source is not None
    assert candidate.source.body == "stable file"


def test_pre_open_name_and_descriptor_identity_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "AGENTS.md").write_text("raced")
    real_fstat = resolver_module.os.fstat
    first = True

    def mismatched_first_fstat(descriptor: int):
        nonlocal first
        value = real_fstat(descriptor)
        if not first:
            return value
        first = False
        return SimpleNamespace(
            st_dev=value.st_dev,
            st_ino=value.st_ino + 1,
            st_mode=value.st_mode,
            st_size=value.st_size,
            st_mtime_ns=value.st_mtime_ns,
        )

    monkeypatch.setattr(resolver_module.os, "fstat", mismatched_first_fstat)

    candidate = _resolve(tmp_path)

    assert candidate.source is None
    assert [item.code for item in candidate.outcomes] == ["resolution_failed"]


def test_ancestor_identity_change_during_read_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instruction = tmp_path / "AGENTS.md"
    instruction.write_text("unstable root")
    real_lstat = resolver_module.os.lstat
    real_open = resolver_module.os.open
    file_opened = False

    def changing_lstat(path: os.PathLike[str] | str):
        value = real_lstat(path)
        if Path(path) != tmp_path or not file_opened:
            return value
        return SimpleNamespace(
            st_dev=value.st_dev,
            st_ino=value.st_ino + 1,
            st_mode=value.st_mode,
            st_size=value.st_size,
            st_mtime_ns=value.st_mtime_ns,
        )

    def mark_open(path: os.PathLike[str] | str, flags: int) -> int:
        nonlocal file_opened
        descriptor = real_open(path, flags)
        file_opened = True
        return descriptor

    monkeypatch.setattr(resolver_module.os, "lstat", changing_lstat)
    monkeypatch.setattr(resolver_module.os, "open", mark_open)

    candidate = _resolve(tmp_path)

    assert candidate.source is None
    assert [item.code for item in candidate.outcomes] == ["resolution_failed"]


def test_source_newer_than_dispatch_cutoff_is_stale(tmp_path: Path) -> None:
    instruction = tmp_path / "AGENTS.md"
    instruction.write_text("next dispatch only")
    file_mtime = instruction.stat().st_mtime_ns

    candidate = _resolve(tmp_path, dispatch_started_wall_ns=file_mtime - 1)

    assert candidate.source is None
    assert [(item.relative_path, item.code) for item in candidate.outcomes] == [
        ("AGENTS.md", "stale")
    ]


def test_growing_file_read_is_capped_at_limit_plus_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instruction = tmp_path / "AGENTS.md"
    instruction.write_bytes(b"1234")
    real_read = resolver_module.os.read
    requests: list[int] = []
    grew = False

    def growing_read(descriptor: int, count: int) -> bytes:
        nonlocal grew
        requests.append(count)
        if not grew:
            grew = True
            with instruction.open("ab") as stream:
                stream.write(b"56789")
        return real_read(descriptor, count)

    monkeypatch.setattr(resolver_module.os, "read", growing_read)

    candidate = _resolve(tmp_path, max_bytes=4)

    assert candidate.source is None
    assert requests
    assert max(requests) <= 5
    assert sum(requests) <= 5
    assert [item.code for item in candidate.outcomes] == ["resolution_failed"]


def test_windows_reparse_attribute_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "AGENTS.md").write_text("linked")
    monkeypatch.setattr(resolver_module, "_WINDOWS", True)
    monkeypatch.setattr(resolver_module, "_REPARSE_POINT", 0x400)
    real_lstat = resolver_module.os.lstat

    def reparse_lstat(path: os.PathLike[str] | str):
        value = real_lstat(path)
        return SimpleNamespace(
            st_dev=value.st_dev,
            st_ino=value.st_ino,
            st_mode=value.st_mode,
            st_size=value.st_size,
            st_mtime_ns=value.st_mtime_ns,
            st_file_attributes=(0x400 if Path(path) == tmp_path / "AGENTS.md" else 0),
        )

    monkeypatch.setattr(resolver_module.os, "lstat", reparse_lstat)

    candidate = _resolve(tmp_path)

    assert candidate.source is None
    assert [item.code for item in candidate.outcomes] == ["invalid"]


def test_windows_raced_in_reparse_attribute_is_refused_after_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instruction = tmp_path / "AGENTS.md"
    instruction.write_text("raced link")
    monkeypatch.setattr(resolver_module, "_WINDOWS", True)
    monkeypatch.setattr(resolver_module, "_REPARSE_POINT", 0x400)
    real_lstat = resolver_module.os.lstat
    real_fstat = resolver_module.os.fstat
    real_open = resolver_module.os.open
    file_opened = False

    def with_reparse_state(path: os.PathLike[str] | str):
        value = real_lstat(path)
        return SimpleNamespace(
            st_dev=value.st_dev,
            st_ino=value.st_ino,
            st_mode=value.st_mode,
            st_size=value.st_size,
            st_mtime_ns=value.st_mtime_ns,
            st_file_attributes=(
                0x400 if Path(path) == instruction and file_opened else 0
            ),
        )

    def with_descriptor_attributes(descriptor: int):
        value = real_fstat(descriptor)
        return SimpleNamespace(
            st_dev=value.st_dev,
            st_ino=value.st_ino,
            st_mode=value.st_mode,
            st_size=value.st_size,
            st_mtime_ns=value.st_mtime_ns,
            st_file_attributes=0,
        )

    def mark_open(path: os.PathLike[str] | str, flags: int) -> int:
        nonlocal file_opened
        descriptor = real_open(path, flags)
        file_opened = True
        return descriptor

    monkeypatch.setattr(resolver_module.os, "lstat", with_reparse_state)
    monkeypatch.setattr(resolver_module.os, "fstat", with_descriptor_attributes)
    monkeypatch.setattr(resolver_module.os, "open", mark_open)

    candidate = _resolve(tmp_path)

    assert candidate.source is None
    assert [item.code for item in candidate.outcomes] == ["resolution_failed"]


def test_missing_required_platform_metadata_fails_only_source_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "AGENTS.md").write_text("cannot verify")
    monkeypatch.setattr(resolver_module, "_WINDOWS", True)
    monkeypatch.setattr(resolver_module, "_REPARSE_POINT", 0x400)
    real_lstat = resolver_module.os.lstat

    def metadata_without_windows_attributes(path: os.PathLike[str] | str):
        value = real_lstat(path)
        fields = dict(
            st_dev=value.st_dev,
            st_ino=value.st_ino,
            st_mode=value.st_mode,
            st_size=value.st_size,
            st_mtime_ns=value.st_mtime_ns,
        )
        if Path(path) != tmp_path / "AGENTS.md":
            fields["st_file_attributes"] = 0
        return SimpleNamespace(**fields)

    monkeypatch.setattr(
        resolver_module.os, "lstat", metadata_without_windows_attributes
    )

    candidate = _resolve(tmp_path)

    assert candidate.source is None
    assert [(item.relative_path, item.code) for item in candidate.outcomes] == [
        ("AGENTS.md", "resolution_failed")
    ]


@pytest.mark.parametrize("attributes", [None, "not-a-number", object()])
def test_unusable_windows_reparse_metadata_fails_source_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attributes: object
) -> None:
    instruction = tmp_path / "AGENTS.md"
    instruction.write_text("cannot verify")
    monkeypatch.setattr(resolver_module, "_WINDOWS", True)
    monkeypatch.setattr(resolver_module, "_REPARSE_POINT", 0x400)
    real_lstat = resolver_module.os.lstat

    def unusable_attributes(path: os.PathLike[str] | str):
        value = real_lstat(path)
        return SimpleNamespace(
            st_dev=value.st_dev,
            st_ino=value.st_ino,
            st_mode=value.st_mode,
            st_size=value.st_size,
            st_mtime_ns=value.st_mtime_ns,
            st_file_attributes=(attributes if Path(path) == instruction else 0),
        )

    monkeypatch.setattr(resolver_module.os, "lstat", unusable_attributes)

    candidate = _resolve(tmp_path)

    assert candidate.source is None
    assert [(item.relative_path, item.code) for item in candidate.outcomes] == [
        ("AGENTS.md", "resolution_failed")
    ]


def test_token_admission_is_whole_source_and_prioritizes_specific_sources() -> None:
    broad = _source(body="123", relative_path="AGENTS.md")
    specific = _source(body="4567", relative_path="src/AGENTS.md")

    delivery = admit_sources(
        (broad, specific),
        safe_input_tokens=4,
        count_tokens=lambda source: source.byte_count,
    )

    assert delivery.source_digests == (specific.digest,)
    assert [(item.relative_path, item.code) for item in delivery.outcomes] == [
        ("AGENTS.md", "omitted_token_budget")
    ]


def test_token_admission_rejects_non_positive_allowance() -> None:
    source = _source(body="x")

    delivery = admit_sources(
        (source,), safe_input_tokens=0, count_tokens=lambda item: item.byte_count
    )

    assert delivery.source_digests == ()
    assert [item.code for item in delivery.outcomes] == ["omitted_token_budget"]


@pytest.mark.parametrize("estimate", [0, -1, 1.5, True, None])
def test_token_admission_omits_invalid_estimates(estimate: object) -> None:
    source = _source(body="x")

    delivery = admit_sources(
        (source,), safe_input_tokens=100, count_tokens=lambda _source: estimate
    )

    assert delivery.source_digests == ()
    assert [item.code for item in delivery.outcomes] == ["omitted_token_budget"]


def test_zero_estimate_is_omitted_even_with_zero_safe_allowance() -> None:
    source = _source(body="x")

    delivery = admit_sources(
        (source,), safe_input_tokens=0, count_tokens=lambda _source: 0
    )

    assert delivery.source_digests == ()
    assert [item.code for item in delivery.outcomes] == ["omitted_token_budget"]


def test_token_estimator_exception_is_sanitized_to_omission() -> None:
    source = _source(body="x")

    def failed_estimator(_source: InstructionSource) -> int:
        raise RuntimeError("sensitive estimator detail")

    delivery = admit_sources(
        (source,), safe_input_tokens=100, count_tokens=failed_estimator
    )

    assert delivery.source_digests == ()
    assert [item.code for item in delivery.outcomes] == ["omitted_token_budget"]


def test_absolute_path_failure_returns_content_free_resolution_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path("relative-root")

    def missing_cwd(_path: Path) -> Path:
        raise FileNotFoundError("sensitive cwd detail")

    monkeypatch.setattr(Path, "absolute", missing_cwd)

    candidate = ProjectInstructionResolver().resolve_startup(
        binding_id="binding",
        binding_root=root,
        locator_fingerprint="fingerprint",
        max_bytes=1024,
        dispatch_started_wall_ns=time.time_ns(),
    )

    assert candidate.binding_root == root
    assert candidate.source is None
    assert [(item.relative_path, item.code) for item in candidate.outcomes] == [
        (".", "resolution_failed")
    ]


def test_public_value_objects_are_frozen_and_candidate_keeps_handoff_fields(
    tmp_path: Path,
) -> None:
    (tmp_path / "AGENTS.md").write_text("root")

    candidate = _resolve(tmp_path)

    assert isinstance(candidate, StartupInstructionCandidate)
    assert candidate.binding_id == "binding-1"
    assert candidate.binding_root == tmp_path.resolve()
    assert candidate.locator_fingerprint == "locator-sha256"
    assert candidate.dispatch_started_wall_ns > 0
    assert candidate.source is not None
    delivery = InstructionChainDelivery((candidate.source.digest,), ())
    snapshot = InstructionSnapshot(
        binding_id=candidate.binding_id,
        binding_root=candidate.binding_root,
        locator_fingerprint=candidate.locator_fingerprint,
        dispatch_started_wall_ns=candidate.dispatch_started_wall_ns,
        startup_source=candidate.source,
        global_outcomes=(InstructionOutcome("AGENTS.md", ".", "invalid"),),
        primary_delivery=delivery,
        warning_codes=("invalid",),
    )
    assert snapshot.primary_delivery is delivery
    assert hasattr(InstructionOutcome, "__slots__")
    assert hasattr(StartupInstructionCandidate, "__slots__")
    assert hasattr(InstructionChainDelivery, "__slots__")
    assert hasattr(InstructionSnapshot, "__slots__")
    with pytest.raises(FrozenInstanceError):
        candidate.binding_id = "changed"  # type: ignore[misc]
