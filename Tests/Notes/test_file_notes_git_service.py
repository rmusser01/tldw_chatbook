from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import Awaitable, Mapping
from pathlib import Path

import pytest

from tldw_chatbook.Notes.file_notes_git_service import (
    AsyncGitProcessRunner,
    DiscoveryResult,
    FileNotesGitService,
    GitCommandResult,
    GitIndexParseError,
    GitMutationAdmissionError,
    GitShutdownAffinityError,
    GitStatusAdmissionError,
    PorcelainV2ParseError,
    PorcelainPathOutsideSessionError,
    PorcelainRecord,
    build_git_environment,
    build_index_argv,
    build_file_notes_session_owner,
    build_stage_argv,
    build_status_argv,
    build_unstage_argv,
    build_update_index_payload,
    classify_session_rows,
    coalesce_session_changes,
    compute_stage_closure,
    compute_unstage_closure,
    ownership_signature_matches,
    parse_index_entries_z,
    parse_porcelain_v2_z,
    sanitize_git_stderr,
    stage_group_is_closed,
    stage_pathspecs,
    unstage_group_is_closed,
)
from tldw_chatbook.Notes.file_notes_session_owner import (
    FileNotesSessionOwner,
    FileSystemIdentity,
    HeadIdentity,
    IndexBaseline,
    IndexEntry,
    RepositoryIdentity,
    SequencedSessionChange,
    SessionBinding,
    SessionChange,
    SessionChangeGroup,
    SessionGitRow,
    SessionGitStatus,
    StagingOwnership,
)

OID_A = "a" * 40
OID_B = "b" * 40
OID_C = "c" * 40
ZERO_OID = "0" * 40


def _change(
    sequence: int,
    action: str,
    path: str,
    destination: str | None = None,
) -> SequencedSessionChange:
    return SequencedSessionChange(
        sequence,
        SessionChange(action, path, destination),  # type: ignore[arg-type]
    )


def _entry(
    path: str,
    *,
    mode: str = "100644",
    object_id: str = OID_A,
    stage: int = 0,
    flags: tuple[str, ...] = (),
) -> IndexEntry:
    return IndexEntry(
        path=path,
        mode=mode,
        object_id=object_id,
        stage=stage,
        semantic_flags=flags,
    )


def _repository() -> RepositoryIdentity:
    worktree = FileSystemIdentity(device=1, inode=10)
    git_dir = FileSystemIdentity(device=1, inode=11)
    common_dir = FileSystemIdentity(device=1, inode=12)
    return RepositoryIdentity(
        worktree_root="/repo",
        git_dir="/repo/.git",
        git_common_dir="/repo/.git",
        worktree_identity=worktree,
        git_dir_identity=git_dir,
        git_common_dir_identity=common_dir,
    )


def _ownership(
    group: SessionChangeGroup,
    entries: Mapping[str, IndexEntry | None],
    *,
    topology: tuple[str, ...] | None = None,
    topology_group: SessionChangeGroup | None = None,
) -> StagingOwnership:
    approved_group = topology_group or group
    approved_endpoints = topology or approved_group.endpoints
    return StagingOwnership(
        repository=_repository(),
        head=HeadIdentity.attached("refs/heads/main", OID_B),
        approved_endpoint_topology=approved_endpoints,
        approved_move_edges=(
            ()
            if topology is not None
            else approved_group.move_edges
        ),
        approved_current_path=(
            approved_endpoints[-1]
            if topology is not None
            else approved_group.current_path
        ),
        original_baselines={
            path: IndexBaseline(entry=None) for path in entries
        },
        post_stage_entries=entries,
    )


def _single_group(path: str = "note.md") -> SessionChangeGroup:
    return SessionChangeGroup(
        group_id=1,
        endpoints=(path,),
        source_path=path,
        destination_path=None,
        current_path=path,
        latest_action="modified",
        latest_sequence=1,
    )


def test_repeated_edits_retain_the_earliest_sequence_as_group_id() -> None:
    groups = coalesce_session_changes(
        (
            _change(7, "modified", "note.md"),
            _change(11, "modified", "note.md"),
            _change(15, "modified", "note.md"),
        )
    )

    assert len(groups) == 1
    assert groups[0].group_id == 7
    assert groups[0].latest_sequence == 15
    assert groups[0].endpoints == ("note.md",)


@pytest.mark.parametrize(
    ("changes", "expected_latest_action"),
    [
        (
            (
                _change(1, "created", "draft.md"),
                _change(2, "deleted", "draft.md"),
            ),
            "deleted",
        ),
        (
            (
                _change(1, "deleted", "tracked.md"),
                _change(2, "restored", "tracked.md"),
            ),
            "restored",
        ),
    ],
)
def test_create_delete_and_delete_restore_remain_one_visible_group(
    changes: tuple[SequencedSessionChange, ...],
    expected_latest_action: str,
) -> None:
    groups = coalesce_session_changes(changes)

    assert len(groups) == 1
    assert groups[0].group_id == 1
    assert groups[0].latest_action == expected_latest_action
    assert groups[0].endpoints == (changes[0].change.relative_path,)


def test_move_lineage_is_inseparable_and_retains_original_and_final_paths() -> None:
    groups = coalesce_session_changes(
        (
            _change(2, "moved", "one.md", "two.md"),
            _change(4, "moved", "two.md", "three.md"),
            _change(6, "modified", "three.md"),
        )
    )

    assert len(groups) == 1
    assert groups[0].group_id == 2
    assert groups[0].endpoints == ("one.md", "two.md", "three.md")
    assert groups[0].source_path == "one.md"
    assert groups[0].destination_path == "three.md"
    assert groups[0].current_path == "three.md"
    assert groups[0].display_text == "one.md → three.md"


def test_reusing_an_old_move_source_starts_a_new_group() -> None:
    groups = coalesce_session_changes(
        (
            _change(1, "moved", "one.md", "two.md"),
            _change(2, "modified", "one.md"),
            _change(3, "moved", "two.md", "three.md"),
        )
    )

    assert [(group.group_id, group.endpoints) for group in groups] == [
        (1, ("one.md", "two.md", "three.md")),
        (2, ("one.md",)),
    ]


def test_only_the_active_path_mapping_extends_a_move_group() -> None:
    groups = coalesce_session_changes(
        (
            _change(1, "moved", "a.md", "b.md"),
            _change(2, "modified", "a.md"),
            _change(3, "modified", "b.md"),
        )
    )

    assert groups[0].latest_sequence == 3
    assert groups[0].current_path == "b.md"
    assert groups[1].group_id == 2
    assert groups[1].latest_sequence == 2


def test_display_text_sanitizes_controls_without_changing_raw_paths() -> None:
    groups = coalesce_session_changes(
        (_change(1, "moved", "line\none.md", "tab\tfinal.md"),)
    )

    assert groups[0].endpoints == ("line\none.md", "tab\tfinal.md")
    assert groups[0].display_text == r"line\none.md → tab\tfinal.md"


@pytest.mark.parametrize("control", ["\x80", "\x9b", "\x9f"])
def test_display_text_sanitizes_c1_controls_without_changing_paths(
    control: str,
) -> None:
    path = f"c1-{control}.md"
    (group,) = coalesce_session_changes((_change(1, "modified", path),))

    assert group.endpoints == (path,)
    assert group.display_text == f"c1-\\x{ord(control):02x}.md"


def test_display_sanitizes_surrogateescaped_bytes_but_raw_path_round_trips() -> None:
    raw_path = b"byte-\x9b.md"
    path = os.fsdecode(raw_path)
    (group,) = coalesce_session_changes((_change(1, "modified", path),))

    assert group.endpoints == (path,)
    assert os.fsencode(group.endpoints[0]) == raw_path
    assert group.display_text == r"byte-\x9b.md"


def test_session_group_copies_endpoint_sequences_to_an_immutable_tuple() -> None:
    endpoints = ["one.md"]
    group = SessionChangeGroup(
        group_id=1,
        endpoints=endpoints,  # type: ignore[arg-type]
        source_path="one.md",
        destination_path=None,
        current_path="one.md",
        latest_action="modified",
        latest_sequence=1,
    )
    endpoints.append("late.md")

    assert group.endpoints == ("one.md",)


def test_parse_porcelain_v2_z_preserves_all_supported_record_paths_as_bytes() -> None:
    non_utf8_path = os.fsdecode(b"bad-\xff.md")
    payload = (
        f"1 .M N... 100644 100644 100644 {OID_A} {OID_A} "
        "ordinary name.md\0"
    ).encode()
    payload += (
        f"2 R. N... 100644 100644 100644 {OID_A} {OID_B} R100 "
        "renamed final.md\0-leading-source.md\0"
    ).encode()
    payload += (
        f"u UU N... 100644 100644 100644 100644 {OID_A} {OID_B} "
        f"{OID_C} conflict\tline\n.md\0"
    ).encode()
    payload += b"? bad-\xff.md\0"
    payload += b"! :(glob)**/[draft]?.md\0"
    allowed = frozenset(
        {
            "ordinary name.md",
            "renamed final.md",
            "-leading-source.md",
            "conflict\tline\n.md",
            non_utf8_path,
            ":(glob)**/[draft]?.md",
        }
    )

    records = parse_porcelain_v2_z(payload, allowed_paths=allowed)

    assert [record.kind for record in records] == [
        "ordinary",
        "rename",
        "unmerged",
        "untracked",
        "ignored",
    ]
    assert records[0].path == "ordinary name.md"
    assert records[0].index_status == "."
    assert records[0].worktree_status == "M"
    assert records[1].path == "renamed final.md"
    assert records[1].original_path == "-leading-source.md"
    assert records[1].score == "R100"
    assert records[2].path == "conflict\tline\n.md"
    assert records[3].path == non_utf8_path
    assert os.fsencode(records[3].path) == b"bad-\xff.md"
    assert records[4].path == ":(glob)**/[draft]?.md"


def test_parse_porcelain_v2_z_fails_the_complete_result_outside_whitelist() -> None:
    payload = (
        f"1 .M N... 100644 100644 100644 {OID_A} {OID_A} "
        "allowed.md\0"
        f"1 .M N... 100644 100644 100644 {OID_A} {OID_A} "
        "outside.md\0"
    ).encode()

    with pytest.raises(
        PorcelainPathOutsideSessionError,
        match="outside.md",
    ):
        parse_porcelain_v2_z(
            payload,
            allowed_paths=frozenset({"allowed.md"}),
        )


@pytest.mark.parametrize(
    ("allowed_paths", "outside_path"),
    [
        (frozenset({"destination.md"}), "source.md"),
        (frozenset({"source.md"}), "destination.md"),
    ],
)
def test_rename_current_and_original_paths_are_both_whitelist_checked(
    allowed_paths: frozenset[str],
    outside_path: str,
) -> None:
    payload = (
        f"2 R. N... 100644 100644 100644 {OID_A} {OID_B} R100 "
        "destination.md\0source.md\0"
    ).encode()

    with pytest.raises(
        PorcelainPathOutsideSessionError,
        match=outside_path,
    ):
        parse_porcelain_v2_z(payload, allowed_paths=allowed_paths)


@pytest.mark.parametrize(
    "payload",
    [
        b"1 .M N... 100644 100644 100644 " + OID_A.encode() + b" ",
        (
            b"2 R. N... 100644 100644 100644 "
            + OID_A.encode()
            + b" "
            + OID_B.encode()
            + b" R100 destination.md\0"
        ),
        b"x unsupported.md\0",
    ],
)
def test_parse_porcelain_v2_z_rejects_malformed_or_truncated_framing(
    payload: bytes,
) -> None:
    with pytest.raises(PorcelainV2ParseError):
        parse_porcelain_v2_z(
            payload,
            allowed_paths=frozenset({"destination.md"}),
        )


def test_git_rename_pairing_does_not_merge_chatbook_groups() -> None:
    groups = coalesce_session_changes(
        (
            _change(1, "modified", "source.md"),
            _change(2, "modified", "destination.md"),
        )
    )
    payload = (
        f"2 R. N... 100644 100644 100644 {OID_A} {OID_B} R100 "
        "destination.md\0source.md\0"
    ).encode()
    records = parse_porcelain_v2_z(
        payload,
        allowed_paths=frozenset({"source.md", "destination.md"}),
    )

    rows = classify_session_rows(groups, records, {}, {})

    assert [row.group_id for row in rows] == [1, 2]


@pytest.mark.parametrize(
    (
        "record",
        "entry",
        "owned",
        "topology",
        "expected_state",
        "expected_stage_action",
        "expected_unstage",
    ),
    [
        (
            PorcelainRecord("ordinary", "note.md", ".", "M"),
            _entry("note.md"),
            False,
            None,
            "unstaged",
            "stage",
            False,
        ),
        (
            PorcelainRecord("ordinary", "note.md", "M", "."),
            _entry("note.md", object_id=OID_B),
            True,
            None,
            "owned",
            None,
            True,
        ),
        (
            PorcelainRecord("ordinary", "note.md", "M", "M"),
            _entry("note.md", object_id=OID_B),
            True,
            None,
            "owned_newer_edits",
            "stage_update",
            True,
        ),
        (
            PorcelainRecord("ordinary", "note.md", "M", "."),
            _entry("note.md", object_id=OID_B),
            True,
            ("old-name.md",),
            "owned_topology_changed",
            "stage_update",
            False,
        ),
        (
            PorcelainRecord("ordinary", "note.md", "M", "."),
            _entry("note.md", object_id=OID_B),
            False,
            None,
            "external_staged",
            None,
            False,
        ),
        (
            PorcelainRecord("ordinary", "note.md", "M", "M"),
            _entry("note.md", object_id=OID_B),
            False,
            None,
            "external_partial",
            None,
            False,
        ),
        (
            None,
            _entry("note.md"),
            False,
            None,
            "clean",
            None,
            False,
        ),
        (
            PorcelainRecord("ignored", "note.md"),
            None,
            False,
            None,
            "ignored",
            None,
            False,
        ),
        (
            PorcelainRecord("unmerged", "note.md", "U", "U"),
            _entry("note.md", stage=2),
            False,
            None,
            "conflict",
            None,
            False,
        ),
        (
            None,
            _entry("note.md", flags=("skip-worktree",)),
            False,
            None,
            "unsupported",
            None,
            False,
        ),
        (
            PorcelainRecord("nested_repository", "note.md"),
            None,
            False,
            None,
            "nested_repository",
            None,
            False,
        ),
        (
            PorcelainRecord("unavailable", None, message="Git is unavailable"),
            None,
            False,
            None,
            "unavailable",
            None,
            False,
        ),
        (
            PorcelainRecord("error", None, message="status failed"),
            None,
            False,
            None,
            "error",
            None,
            False,
        ),
    ],
)
def test_frozen_row_action_policy(
    record: PorcelainRecord | None,
    entry: IndexEntry | None,
    owned: bool,
    topology: tuple[str, ...] | None,
    expected_state: str,
    expected_stage_action: str | None,
    expected_unstage: bool,
) -> None:
    group = _single_group()
    entries = {} if entry is None else {entry.path: entry}
    ownership = (
        {
            group.group_id: _ownership(
                group,
                entries,
                topology=topology,
            )
        }
        if owned
        else {}
    )

    (row,) = classify_session_rows(
        (group,),
        () if record is None else (record,),
        entries,
        ownership,
    )

    assert row.state == expected_state
    assert row.stage_action == expected_stage_action
    assert row.stage_eligible is (expected_stage_action is not None)
    assert row.unstage_eligible is expected_unstage


def test_changed_topology_needs_matching_owned_post_stage_entries_for_update() -> None:
    group = SessionChangeGroup(
        group_id=1,
        endpoints=("note.md", "renamed.md"),
        source_path="note.md",
        destination_path="renamed.md",
        current_path="renamed.md",
        latest_action="moved",
        latest_sequence=2,
    )
    originally_owned = _entry("note.md", object_id=OID_A)
    externally_replaced = _entry("note.md", object_id=OID_B)
    ownership = _ownership(
        group,
        {"note.md": originally_owned},
        topology=("note.md",),
    )

    (row,) = classify_session_rows(
        (group,),
        (PorcelainRecord("ordinary", "note.md", "M", "."),),
        {"note.md": externally_replaced},
        {group.group_id: ownership},
    )

    assert row.state == "external_staged"
    assert row.stage_action is None
    assert not row.unstage_eligible


def test_owned_group_with_externally_staged_second_endpoint_is_not_owned() -> None:
    group = SessionChangeGroup(
        group_id=1,
        endpoints=("old.md", "new.md"),
        source_path="old.md",
        destination_path="new.md",
        current_path="new.md",
        latest_action="moved",
        latest_sequence=2,
        move_edges=(("old.md", "new.md"),),
    )
    owned_entry = _entry("old.md", object_id=OID_A)
    external_entry = _entry("new.md", object_id=OID_B)
    ownership = _ownership(
        group,
        {"old.md": owned_entry},
    )

    (row,) = classify_session_rows(
        (group,),
        (
            PorcelainRecord("ordinary", "old.md", "M", "."),
            PorcelainRecord("ordinary", "new.md", "A", "."),
        ),
        {"old.md": owned_entry, "new.md": external_entry},
        {group.group_id: ownership},
    )

    assert row.state == "external_staged"
    assert row.stage_action is None
    assert not row.unstage_eligible


def test_topology_changed_group_with_new_external_stage_cannot_stage_update() -> None:
    before = coalesce_session_changes((_change(1, "modified", "old.md"),))[0]
    after = coalesce_session_changes(
        (
            _change(1, "modified", "old.md"),
            _change(2, "moved", "old.md", "new.md"),
        )
    )[0]
    owned_entry = _entry("old.md", object_id=OID_A)
    external_entry = _entry("new.md", object_id=OID_B)
    ownership = _ownership(
        after,
        {"old.md": owned_entry},
        topology_group=before,
    )

    (row,) = classify_session_rows(
        (after,),
        (
            PorcelainRecord("ordinary", "old.md", "M", "."),
            PorcelainRecord("ordinary", "new.md", "A", "M"),
        ),
        {"old.md": owned_entry, "new.md": external_entry},
        {after.group_id: ownership},
    )

    assert row.state == "external_partial"
    assert row.stage_action is None
    assert not row.unstage_eligible


def test_move_reversal_changes_exact_topology_despite_same_endpoint_set() -> None:
    staged_group = coalesce_session_changes(
        (_change(1, "moved", "a.md", "b.md"),)
    )[0]
    reversed_group = coalesce_session_changes(
        (
            _change(1, "moved", "a.md", "b.md"),
            _change(2, "moved", "b.md", "a.md"),
        )
    )[0]
    post_entry = _entry("b.md", object_id=OID_B)
    ownership = _ownership(
        reversed_group,
        {"b.md": post_entry},
        topology_group=staged_group,
    )

    assert staged_group.endpoints == reversed_group.endpoints
    assert staged_group.topology_signature != reversed_group.topology_signature
    (row,) = classify_session_rows(
        (reversed_group,),
        (),
        {"b.md": post_entry},
        {reversed_group.group_id: ownership},
    )
    assert row.state == "owned_topology_changed"
    assert row.stage_action == "stage_update"
    assert not row.unstage_eligible


def test_newer_body_edit_preserves_move_topology_and_unstage_eligibility() -> None:
    staged_group = coalesce_session_changes(
        (_change(1, "moved", "a.md", "b.md"),)
    )[0]
    edited_group = coalesce_session_changes(
        (
            _change(1, "moved", "a.md", "b.md"),
            _change(2, "modified", "b.md"),
        )
    )[0]
    post_entry = _entry("b.md", object_id=OID_B)
    ownership = _ownership(
        edited_group,
        {"b.md": post_entry},
        topology_group=staged_group,
    )

    assert staged_group.topology_signature == edited_group.topology_signature
    (row,) = classify_session_rows(
        (edited_group,),
        (PorcelainRecord("ordinary", "b.md", "M", "M"),),
        {"b.md": post_entry},
        {edited_group.group_id: ownership},
    )
    assert row.state == "owned_newer_edits"
    assert row.stage_action == "stage_update"
    assert row.unstage_eligible


def test_shared_effective_path_disables_every_sibling_group_and_action() -> None:
    groups = coalesce_session_changes(
        (
            _change(1, "moved", "a.md", "b.md"),
            _change(2, "modified", "a.md"),
        )
    )
    records = (
        PorcelainRecord("untracked", "a.md"),
        PorcelainRecord("ordinary", "b.md", "M", "."),
    )
    post_entry = _entry("b.md", object_id=OID_B)
    ownership = {
        groups[0].group_id: _ownership(
            groups[0],
            {"b.md": post_entry},
        )
    }

    rows = classify_session_rows(
        groups,
        records,
        {"b.md": post_entry},
        ownership,
    )

    assert [group.endpoints for group in groups] == [
        ("a.md", "b.md"),
        ("a.md",),
    ]
    assert [row.state for row in rows] == [
        "ambiguous_lineage",
        "ambiguous_lineage",
    ]
    assert all(
        row.disabled_reason
        == "Ambiguous session lineage: effective path belongs to multiple groups"
        for row in rows
    )
    assert all(not row.stage_eligible for row in rows)
    assert all(not row.unstage_eligible for row in rows)
    assert stage_pathspecs(groups[0], records, groups=groups) == ()
    assert stage_pathspecs(groups[1], records, groups=groups) == ()
    selected_or_bulk_pathspecs = tuple(
        pathspec
        for row in rows
        if row.stage_eligible
        for pathspec in stage_pathspecs(
            row.group,
            records,
            groups=groups,
        )
    )
    assert selected_or_bulk_pathspecs == ()


def test_inactive_historical_endpoint_without_effective_path_is_not_ambiguous() -> None:
    groups = coalesce_session_changes(
        (
            _change(1, "moved", "a.md", "b.md"),
            _change(2, "modified", "a.md"),
        )
    )
    records = (PorcelainRecord("untracked", "b.md"),)

    rows = classify_session_rows(groups, records, {}, {})

    assert [row.state for row in rows] == ["unstaged", "clean"]
    assert rows[0].stage_eligible
    assert not rows[1].stage_eligible
    assert stage_pathspecs(groups[0], records, groups=groups) == (
        os.fsencode("b.md"),
    )
    assert stage_pathspecs(groups[1], records, groups=groups) == ()


def test_tracked_ancestor_and_descendant_paths_expand_stage_closure() -> None:
    entries = {
        "ancestor": _entry("ancestor"),
        "docs/one.md": _entry("docs/one.md"),
        "docs/deep/two.md": _entry("docs/deep/two.md"),
        "docs-old.md": _entry("docs-old.md"),
    }

    assert compute_stage_closure({"ancestor/child.md", "docs"}, entries) == (
        frozenset(
            {
                "ancestor",
                "ancestor/child.md",
                "docs",
                "docs/one.md",
                "docs/deep/two.md",
            }
        )
    )


def test_out_of_lineage_stage_closure_blocks_the_whole_group() -> None:
    group = _single_group("docs")
    entries = {"docs/outside.md": _entry("docs/outside.md")}

    assert not stage_group_is_closed(group, entries)
    (row,) = classify_session_rows(
        (group,),
        (PorcelainRecord("ordinary", "docs", ".", "M"),),
        entries,
        {},
    )
    assert row.state == "unsafe_closure"
    assert not row.stage_eligible


def test_unstage_closure_includes_current_index_replacement_conflicts() -> None:
    baselines = {
        "file": IndexBaseline(_entry("file", object_id=OID_B)),
        "tree/leaf.md": IndexBaseline(_entry("tree/leaf.md", object_id=OID_C)),
        "absent": IndexBaseline(None),
    }
    current = {
        "file/child.md": _entry("file/child.md"),
        "tree": _entry("tree"),
        "absent/untouched.md": _entry("absent/untouched.md"),
        "unrelated.md": _entry("unrelated.md"),
    }

    assert compute_unstage_closure(baselines, current) == frozenset(
        {
            "file",
            "file/child.md",
            "tree",
            "tree/leaf.md",
            "absent",
        }
    )


def test_out_of_lineage_unstage_replacement_closure_blocks_the_group() -> None:
    group = _single_group("tree")
    baselines = {"tree": IndexBaseline(_entry("tree", object_id=OID_B))}
    current = {"tree/external.md": _entry("tree/external.md")}
    ownership = _ownership(group, {})

    assert not unstage_group_is_closed(
        group,
        baselines,
        current,
        ownership,
    )


def test_unstage_closure_requires_exact_same_group_owned_conflicts() -> None:
    group = SessionChangeGroup(
        group_id=1,
        endpoints=("tree", "tree/child.md"),
        source_path="tree",
        destination_path="tree/child.md",
        current_path="tree/child.md",
        latest_action="moved",
        latest_sequence=2,
        move_edges=(("tree", "tree/child.md"),),
    )
    baselines = {"tree": IndexBaseline(_entry("tree", object_id=OID_A))}
    current_child = _entry("tree/child.md", object_id=OID_B)
    unowned = _ownership(group, {"tree": None})
    mismatched = _ownership(
        group,
        {
            "tree": None,
            "tree/child.md": _entry("tree/child.md", object_id=OID_C),
        },
    )
    owned = _ownership(
        group,
        {"tree": None, "tree/child.md": current_child},
    )

    assert not unstage_group_is_closed(
        group,
        baselines,
        {"tree/child.md": current_child},
        unowned,
    )
    assert not unstage_group_is_closed(
        group,
        baselines,
        {"tree/child.md": current_child},
        mismatched,
    )
    assert unstage_group_is_closed(
        group,
        baselines,
        {"tree/child.md": current_child},
        owned,
    )


def test_transient_move_endpoints_are_omitted_from_stage_pathspecs() -> None:
    group = SessionChangeGroup(
        group_id=1,
        endpoints=("old.md", "temporary.md", "final [1].md"),
        source_path="old.md",
        destination_path="final [1].md",
        current_path="final [1].md",
        latest_action="moved",
        latest_sequence=2,
    )
    records = (
        PorcelainRecord("ordinary", "old.md", ".", "D"),
        PorcelainRecord("untracked", "final [1].md"),
    )

    assert stage_pathspecs(
        group,
        records,
        groups=(group,),
    ) == (
        os.fsencode("old.md"),
        os.fsencode("final [1].md"),
    )


def test_clean_tracked_endpoints_are_omitted_from_stage_pathspecs() -> None:
    group = SessionChangeGroup(
        group_id=1,
        endpoints=("changed.md", "clean.md"),
        source_path="changed.md",
        destination_path="clean.md",
        current_path="clean.md",
        latest_action="moved",
        latest_sequence=2,
    )

    assert stage_pathspecs(
        group,
        (PorcelainRecord("untracked", "changed.md"),),
        groups=(group,),
    ) == (os.fsencode("changed.md"),)


@pytest.mark.parametrize(
    ("entry", "expected_reason"),
    [
        (
            _entry("note.md", flags=("skip-worktree",)),
            "skip-worktree",
        ),
        (
            _entry("note.md", flags=("assume-unchanged",)),
            "assume-unchanged",
        ),
        (
            _entry("note.md", object_id=ZERO_OID),
            "intent-to-add",
        ),
    ],
)
def test_nondefault_semantics_and_intent_to_add_are_blocked(
    entry: IndexEntry,
    expected_reason: str,
) -> None:
    (row,) = classify_session_rows(
        (_single_group(),),
        (),
        {"note.md": entry},
        {},
    )
    assert row.state == "unsupported"
    assert row.disabled_reason is not None
    assert expected_reason in row.disabled_reason
    assert not row.stage_eligible
    assert not row.unstage_eligible


def test_ownership_signature_matches_repository_head_topology_and_entries() -> None:
    group = _single_group()
    post_entry = _entry("note.md", object_id=OID_B)
    ownership = _ownership(group, {"note.md": post_entry})
    repository = ownership.repository
    head = ownership.head

    assert ownership_signature_matches(
        ownership,
        repository=repository,
        head=head,
        topology=group.topology_signature,
        current_index_entries={"note.md": post_entry},
    )
    assert not ownership_signature_matches(
        ownership,
        repository=repository,
        head=HeadIdentity.detached(OID_B),
        topology=group.topology_signature,
        current_index_entries={"note.md": post_entry},
    )
    assert not ownership_signature_matches(
        ownership,
        repository=repository,
        head=head,
        topology=(
            ("note.md", "new.md"),
            (),
            "new.md",
        ),
        current_index_entries={"note.md": post_entry},
    )
    assert not ownership_signature_matches(
        ownership,
        repository=repository,
        head=head,
        topology=group.topology_signature,
        current_index_entries={
            "note.md": _entry(
                "note.md",
                object_id=OID_B,
                flags=("assume-unchanged",),
            )
        },
    )


def test_ownership_signature_preserves_an_exact_post_stage_absence() -> None:
    group = _single_group("deleted.md")
    ownership = StagingOwnership(
        repository=_repository(),
        head=HeadIdentity.detached(OID_A),
        approved_endpoint_topology=group.endpoints,
        approved_move_edges=group.move_edges,
        approved_current_path=group.current_path,
        original_baselines={
            "deleted.md": IndexBaseline(_entry("deleted.md"))
        },
        post_stage_entries={"deleted.md": None},
    )

    assert ownership_signature_matches(
        ownership,
        repository=ownership.repository,
        head=ownership.head,
        topology=group.topology_signature,
        current_index_entries={},
    )
    assert not ownership_signature_matches(
        ownership,
        repository=ownership.repository,
        head=ownership.head,
        topology=group.topology_signature,
        current_index_entries={"deleted.md": _entry("deleted.md")},
    )


def test_head_identity_distinguishes_attached_detached_and_explicit_unborn() -> None:
    attached = HeadIdentity.attached("refs/heads/main", OID_A)
    detached = HeadIdentity.detached(OID_A)
    unborn = HeadIdentity.unborn("refs/heads/main")

    assert (attached.kind, attached.branch, attached.object_id) == (
        "attached",
        "refs/heads/main",
        OID_A,
    )
    assert (detached.kind, detached.branch, detached.object_id) == (
        "detached",
        None,
        OID_A,
    )
    assert (unborn.kind, unborn.branch, unborn.object_id) == (
        "unborn",
        "refs/heads/main",
        None,
    )
    assert len({attached, detached, unborn}) == 3


_REDIRECTING_GIT_ENVIRONMENT = {
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_COMMON_DIR",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_NAMESPACE",
    "GIT_CEILING_DIRECTORIES",
    "GIT_DISCOVERY_ACROSS_FILESYSTEM",
    "GIT_SHALLOW_FILE",
    "GIT_GRAFT_FILE",
    "GIT_REPLACE_REF_BASE",
    "GIT_NO_REPLACE_OBJECTS",
    "GIT_EXEC_PATH",
    "GIT_PREFIX",
    "GIT_CONFIG",
    "GIT_CONFIG_SYSTEM",
    "GIT_CONFIG_GLOBAL",
    "GIT_CONFIG_NOSYSTEM",
    "GIT_CONFIG_PARAMETERS",
    "GIT_GLOB_PATHSPECS",
    "GIT_NOGLOB_PATHSPECS",
    "GIT_LITERAL_PATHSPECS",
    "GIT_ICASE_PATHSPECS",
}


def test_git_environment_removes_repository_and_config_injection() -> None:
    ambient = {
        **{key: f"hostile-{key}" for key in _REDIRECTING_GIT_ENVIRONMENT},
        "GIT_CONFIG_COUNT": "2",
        "GIT_CONFIG_KEY_0": "core.fsmonitor",
        "GIT_CONFIG_VALUE_0": "hostile-hook",
        "GIT_CONFIG_KEY_37": "diff.external",
        "GIT_CONFIG_VALUE_37": "hostile-diff",
        "HOME": "/private/home",
        "PATH": "/private/bin",
        "LANG": "C.UTF-8",
        "FILTER_HELPER_CONTEXT": "preserved",
        "GIT_AUTHOR_NAME": "ordinary-config-remains",
    }

    sanitized = build_git_environment(ambient)

    assert not (_REDIRECTING_GIT_ENVIRONMENT & sanitized.keys())
    assert all(
        not key.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_"))
        for key in sanitized
    )
    assert "GIT_CONFIG_COUNT" not in sanitized
    assert sanitized["HOME"] == "/private/home"
    assert sanitized["FILTER_HELPER_CONTEXT"] == "preserved"
    assert sanitized["GIT_AUTHOR_NAME"] == "ordinary-config-remains"
    assert sanitized["GIT_TERMINAL_PROMPT"] == "0"
    assert "LC_ALL" not in sanitized


def test_status_environment_and_argv_disable_side_channel_writes() -> None:
    environment = build_git_environment(
        {
            "PATH": "/private/bin",
            "LANG": "de_DE.UTF-8",
            "LC_ALL": "de_DE.UTF-8",
            "FILTER_HELPER_CONTEXT": "preserved",
        },
        for_status=True,
    )
    argv = build_status_argv(
        "/private/bin/git",
        (os.fsencode("-leading.md"), os.fsencode("line\nbreak.md")),
    )

    assert environment["GIT_OPTIONAL_LOCKS"] == "0"
    assert environment["LANG"] == "de_DE.UTF-8"
    assert environment["LC_ALL"] == "de_DE.UTF-8"
    assert environment["FILTER_HELPER_CONTEXT"] == "preserved"
    assert argv[:2] == ("/private/bin/git", "--literal-pathspecs")
    assert ("-c", "core.fsmonitor=false") == argv[2:4]
    pairs = tuple(
        argv[index : index + 2]
        for index in range(len(argv) - 1)
    )
    assert ("-c", "status.renames=false") in pairs
    assert "--porcelain=v2" in argv
    assert "-z" in argv
    assert "--untracked-files=all" in argv
    assert "--ignored=matching" in argv
    assert "--no-renames" in argv
    boundary = argv.index("--")
    assert argv[boundary + 1 :] == (
        os.fsencode("-leading.md"),
        os.fsencode("line\nbreak.md"),
    )


def test_runner_api_is_direct_argv_and_command_results_stay_bytes() -> None:
    signature = inspect.signature(AsyncGitProcessRunner.run)

    assert "shell" not in signature.parameters
    assert "argv" in signature.parameters
    result = GitCommandResult(
        returncode=0,
        stdout=b"\xffstdout\0",
        stderr=b"\xfestderr",
    )
    assert isinstance(result.stdout, bytes)
    assert isinstance(result.stderr, bytes)


def test_git_stderr_is_bounded_and_control_sanitized() -> None:
    diagnostic = sanitize_git_stderr(
        b"bad\npath\rwith\tcontrols\x00\x1b" + (b"x" * 10_000),
        limit=80,
    )

    assert len(diagnostic.encode("utf-8", "surrogateescape")) <= 80
    assert "\n" not in diagnostic
    assert "\r" not in diagnostic
    assert "\t" not in diagnostic
    assert "\x00" not in diagnostic
    assert "\x1b" not in diagnostic


class _DiscoveryFailureRunner:
    def __init__(self, result: GitCommandResult) -> None:
        self.result = result
        self.calls: list[tuple[str | bytes, ...]] = []

    async def run(
        self,
        argv: tuple[str | bytes, ...],
        *,
        cwd: str,
        environment: Mapping[str, str],
        stdin: bytes | None = None,
        timeout: float | None = None,
    ) -> GitCommandResult:
        del cwd, environment, stdin, timeout
        self.calls.append(tuple(argv))
        return self.result

    def shutdown(self) -> None:
        return None


class _LocalizedDiscoveryRunner:
    def __init__(self) -> None:
        self.locales: list[str | None] = []

    async def run(
        self,
        argv: tuple[str | bytes, ...],
        *,
        cwd: str,
        environment: Mapping[str, str],
        stdin: bytes | None = None,
        timeout: float | None = None,
    ) -> GitCommandResult:
        del argv, cwd, stdin, timeout
        locale = environment.get("LC_ALL")
        self.locales.append(locale)
        if locale == "C":
            return GitCommandResult(
                128,
                b"",
                b"fatal: not a git repository\n",
            )
        return GitCommandResult(
            128,
            b"",
            "schwerwiegend: kein Git-Repository\n".encode(),
        )

    def shutdown(self) -> None:
        return None


class _HeadDiscoveryRunner:
    def __init__(
        self,
        root: Path,
        *,
        symbolic: GitCommandResult,
        revision: GitCommandResult,
        reference: GitCommandResult | None = None,
    ) -> None:
        self.root = root
        self.symbolic = symbolic
        self.revision = revision
        self.reference = reference
        self.calls: list[tuple[str | bytes, ...]] = []

    async def run(
        self,
        argv: tuple[str | bytes, ...],
        *,
        cwd: str,
        environment: Mapping[str, str],
        stdin: bytes | None = None,
        timeout: float | None = None,
    ) -> GitCommandResult:
        del cwd, environment, stdin, timeout
        command = tuple(argv)
        self.calls.append(command)
        text = tuple(os.fsdecode(argument) for argument in command)
        if "--is-inside-work-tree" in text:
            return GitCommandResult(0, b"true\n", b"")
        if "--show-toplevel" in text:
            return GitCommandResult(0, os.fsencode(self.root) + b"\n", b"")
        if "--absolute-git-dir" in text or "--git-common-dir" in text:
            return GitCommandResult(
                0,
                os.fsencode(self.root / ".git") + b"\n",
                b"",
            )
        if "symbolic-ref" in text:
            return self.symbolic
        if "HEAD^{commit}" in text:
            return self.revision
        if "show-ref" in text:
            assert self.reference is not None
            return self.reference
        raise AssertionError(f"Unexpected Git command: {text!r}")

    def shutdown(self) -> None:
        return None


async def _discover_with_head_results(
    tmp_path: Path,
    *,
    symbolic: GitCommandResult,
    revision: GitCommandResult,
    reference: GitCommandResult | None = None,
) -> tuple[DiscoveryResult, _HeadDiscoveryRunner]:
    root = tmp_path / "notes"
    root.mkdir()
    (root / ".git").mkdir()
    owner = FileNotesSessionOwner()
    binding = owner.select_root(root)
    runner = _HeadDiscoveryRunner(
        root,
        symbolic=symbolic,
        revision=revision,
        reference=reference,
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="/private/bin/git",
        environment={"PATH": "/private/bin"},
    )

    discovery = await service.discover(binding)
    await service.shutdown()
    return discovery, runner


async def _discover_with_failure(
    tmp_path: Path,
    result: GitCommandResult,
) -> tuple[DiscoveryResult, _DiscoveryFailureRunner]:
    root = tmp_path / "notes"
    root.mkdir()
    owner = FileNotesSessionOwner()
    binding = owner.select_root(root)
    runner = _DiscoveryFailureRunner(result)
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="/private/bin/git",
        environment={"PATH": "/private/bin"},
    )

    discovery = await service.discover(binding)
    await service.shutdown()
    return discovery, runner


@pytest.mark.asyncio
async def test_discover_preserves_genuine_not_repository_result(
    tmp_path: Path,
) -> None:
    discovery, runner = await _discover_with_failure(
        tmp_path,
        GitCommandResult(
            128,
            b"",
            b"fatal: not a git repository (or any parent): .git\n",
        ),
    )

    assert discovery.state == "not_repository"
    assert len(runner.calls) == 1


@pytest.mark.asyncio
async def test_discovery_forces_stable_locale_before_classifying_diagnostics(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    owner = FileNotesSessionOwner()
    binding = owner.select_root(root)
    runner = _LocalizedDiscoveryRunner()
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={
            "PATH": "/bin",
            "LANG": "de_DE.UTF-8",
            "LC_ALL": "de_DE.UTF-8",
        },
    )

    discovery = await service.discover(binding)
    await service.shutdown()

    assert discovery.state == "not_repository"
    assert runner.locales == ["C"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stderr", "expected_fragments"),
    [
        (
            (
                b"fatal: detected dubious ownership in repository at '/notes'\n"
                b"git config --global --add safe.directory /notes\n"
            ),
            ("dubious ownership", "safe.directory"),
        ),
        (
            b"fatal: could not open '.git/HEAD': Permission denied\n",
            ("permission denied",),
        ),
    ],
)
async def test_discover_surfaces_repository_safety_refusal(
    tmp_path: Path,
    stderr: bytes,
    expected_fragments: tuple[str, ...],
) -> None:
    discovery, runner = await _discover_with_failure(
        tmp_path,
        GitCommandResult(128, b"", stderr),
    )

    assert discovery.state == "unsafe_root"
    assert discovery.message is not None
    message = discovery.message.lower()
    assert all(fragment in message for fragment in expected_fragments)
    assert "\n" not in message
    assert len(runner.calls) == 1
    assert all(
        "config" not in tuple(os.fsdecode(argument) for argument in call)
        for call in runner.calls
    )


@pytest.mark.asyncio
async def test_discover_bounds_and_sanitizes_hostile_failure_stderr(
    tmp_path: Path,
) -> None:
    discovery, _runner = await _discover_with_failure(
        tmp_path,
        GitCommandResult(
            128,
            b"",
            (
                b"fatal: detected dubious ownership\x00\x1b\n"
                + (b"x" * 10_000)
            ),
        ),
    )

    assert discovery.state == "unsafe_root"
    assert discovery.message is not None
    message = discovery.message
    assert len(message.encode("utf-8", "surrogateescape")) <= 4200
    assert "\n" not in message
    assert "\r" not in message
    assert "\t" not in message
    assert "\x00" not in message
    assert "\x1b" not in message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("symbolic", "revision", "reference", "expected"),
    [
        (
            GitCommandResult(0, b"refs/heads/main\n", b""),
            GitCommandResult(0, OID_A.encode("ascii") + b"\n", b""),
            None,
            HeadIdentity.attached("refs/heads/main", OID_A),
        ),
        (
            GitCommandResult(1, b"", b""),
            GitCommandResult(0, OID_B.encode("ascii") + b"\n", b""),
            None,
            HeadIdentity.detached(OID_B),
        ),
        (
            GitCommandResult(0, b"refs/heads/new\n", b""),
            GitCommandResult(1, b"", b""),
            GitCommandResult(2, b"", b""),
            HeadIdentity.unborn("refs/heads/new"),
        ),
    ],
)
async def test_head_semantics_require_expected_exit_combinations(
    tmp_path: Path,
    symbolic: GitCommandResult,
    revision: GitCommandResult,
    reference: GitCommandResult | None,
    expected: HeadIdentity,
) -> None:
    discovery, _runner = await _discover_with_head_results(
        tmp_path,
        symbolic=symbolic,
        revision=revision,
        reference=reference,
    )

    assert discovery.state == "ready"
    assert discovery.head == expected


@pytest.mark.asyncio
async def test_head_operational_failure_is_sanitized_and_never_detached(
    tmp_path: Path,
) -> None:
    discovery, runner = await _discover_with_head_results(
        tmp_path,
        symbolic=GitCommandResult(
            128,
            b"",
            b"fatal: permission denied\x00\x1b\n",
        ),
        revision=GitCommandResult(0, OID_A.encode("ascii") + b"\n", b""),
    )

    assert discovery.state == "unsupported"
    assert discovery.head is None
    assert discovery.message is not None
    assert "permission denied" in discovery.message
    assert "\n" not in discovery.message
    assert "\x00" not in discovery.message
    assert "\x1b" not in discovery.message
    assert not any(
        "HEAD^{commit}" in tuple(os.fsdecode(item) for item in call)
        for call in runner.calls
    )


@pytest.mark.asyncio
async def test_existing_unresolvable_branch_is_not_misclassified_as_unborn(
    tmp_path: Path,
) -> None:
    discovery, runner = await _discover_with_head_results(
        tmp_path,
        symbolic=GitCommandResult(0, b"refs/heads/main\n", b""),
        revision=GitCommandResult(1, b"", b""),
        reference=GitCommandResult(0, b"", b""),
    )

    assert discovery.state == "unsupported"
    assert discovery.head is None
    assert discovery.message is not None
    assert "does not resolve to a commit" in discovery.message
    assert any(
        "show-ref" in tuple(os.fsdecode(item) for item in call)
        for call in runner.calls
    )


class _CompletedProcess:
    returncode = 0

    def __init__(self) -> None:
        self.stdin: bytes | None = None

    async def communicate(
        self,
        stdin: bytes | None,
    ) -> tuple[bytes, bytes]:
        self.stdin = stdin
        return b"stdout\0", b"stderr"

    async def wait(self) -> int:
        return self.returncode

    def terminate(self) -> None:
        raise AssertionError("completed process must not be terminated")

    def kill(self) -> None:
        raise AssertionError("completed process must not be killed")


@pytest.mark.asyncio
async def test_runner_passes_direct_argv_and_byte_stdin_to_exec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = _CompletedProcess()
    captured: dict[str, object] = {}

    async def fake_create_subprocess_exec(
        *argv: str | bytes,
        **kwargs: object,
    ) -> _CompletedProcess:
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return child

    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    runner = AsyncGitProcessRunner()

    result = await runner.run(
        ("git", b"status"),
        cwd="/repo",
        environment={"PATH": "/bin"},
        stdin=b"input\0bytes",
        timeout=1,
    )

    assert captured["argv"] == ("git", b"status")
    assert "shell" not in captured["kwargs"]  # type: ignore[operator]
    assert child.stdin == b"input\0bytes"
    assert result == GitCommandResult(0, b"stdout\0", b"stderr")


class _StubbornProcess:
    returncode: int | None = None

    def __init__(self) -> None:
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls = 0
        self.communicate_started = asyncio.Event()
        self._never = asyncio.Event()

    async def communicate(
        self,
        stdin: bytes | None,
    ) -> tuple[bytes, bytes]:
        del stdin
        self.communicate_started.set()
        await self._never.wait()
        return b"", b""

    async def wait(self) -> int:
        self.wait_calls += 1
        await self._never.wait()
        return 0

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1


class _SignalFailureProcess(_StubbornProcess):
    def __init__(
        self,
        *,
        terminate_error: OSError | None = None,
        kill_error: OSError | None = None,
    ) -> None:
        super().__init__()
        self.terminate_error = terminate_error
        self.kill_error = kill_error

    def terminate(self) -> None:
        super().terminate()
        if self.terminate_error is not None:
            raise self.terminate_error

    def kill(self) -> None:
        super().kill()
        if self.kill_error is not None:
            raise self.kill_error


@pytest.mark.asyncio
async def test_runner_timeout_terminates_then_kills_with_two_bounded_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = _StubbornProcess()

    async def fake_create_subprocess_exec(
        *argv: str | bytes,
        **kwargs: object,
    ) -> _StubbornProcess:
        del argv, kwargs
        return child

    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    runner = AsyncGitProcessRunner(
        terminate_timeout=0.001,
        kill_timeout=0.001,
    )

    result = await runner.run(
        ("git", "status"),
        cwd="/repo",
        environment={},
        timeout=0.001,
    )

    assert child.terminate_calls == 1
    assert child.kill_calls == 1
    assert child.wait_calls == 2
    assert result.timed_out
    assert result.termination_uncertain


@pytest.mark.asyncio
async def test_runner_shutdown_returns_retained_finite_settlement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = _StubbornProcess()

    async def fake_create_subprocess_exec(
        *argv: str | bytes,
        **kwargs: object,
    ) -> _StubbornProcess:
        del argv, kwargs
        return child

    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    runner = AsyncGitProcessRunner(
        terminate_timeout=0.001,
        kill_timeout=0.001,
    )
    command = asyncio.create_task(
        runner.run(
            ("git", "status"),
            cwd="/repo",
            environment={},
            timeout=None,
        )
    )
    await child.communicate_started.wait()

    settlement = runner.shutdown()

    try:
        assert inspect.isawaitable(settlement)
        assert runner.shutdown() is settlement
        cancelled_waiter = asyncio.ensure_future(settlement)
        cancelled_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_waiter
        await asyncio.wait_for(settlement, timeout=1)
        assert command.done()
        result = await command
        assert result.termination_uncertain
        assert child.terminate_calls == 1
        assert child.kill_calls == 1
        assert child.wait_calls == 2
        assert child in runner._processes
    finally:
        child._never.set()
        command.cancel()
        await asyncio.gather(command, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("terminate_error", "kill_error"),
    [
        (PermissionError("terminate denied"), None),
        (None, OSError("kill failed")),
        (
            PermissionError("terminate denied"),
            OSError("kill failed"),
        ),
    ],
)
async def test_runner_signal_failures_are_uncertain_and_remain_tracked(
    monkeypatch: pytest.MonkeyPatch,
    terminate_error: OSError | None,
    kill_error: OSError | None,
) -> None:
    child = _SignalFailureProcess(
        terminate_error=terminate_error,
        kill_error=kill_error,
    )

    async def fake_create_subprocess_exec(
        *argv: str | bytes,
        **kwargs: object,
    ) -> _SignalFailureProcess:
        del argv, kwargs
        return child

    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    runner = AsyncGitProcessRunner(
        terminate_timeout=0.001,
        kill_timeout=0.001,
    )
    command = asyncio.create_task(
        runner.run(
            ("git", "status"),
            cwd="/repo",
            environment={},
            timeout=None,
        )
    )
    await child.communicate_started.wait()

    settlement = runner.shutdown()
    confirmed = await asyncio.wait_for(settlement, timeout=1)
    result = await asyncio.wait_for(command, timeout=1)

    assert not confirmed
    assert result.termination_uncertain
    assert child.terminate_calls == 1
    assert child.kill_calls == 1
    assert child.wait_calls == 2
    assert child in runner._processes


@pytest.mark.asyncio
async def test_runner_shutdown_fails_when_owned_run_task_raises() -> None:
    runner = AsyncGitProcessRunner()
    runner._loop = asyncio.get_running_loop()
    runner._shutdown_event = asyncio.Event()

    async def fail() -> None:
        await asyncio.sleep(0)
        raise RuntimeError("owned task failed")

    task = asyncio.create_task(fail())
    runner._run_tasks.add(task)  # type: ignore[arg-type]
    task.add_done_callback(runner._run_tasks.discard)  # type: ignore[arg-type]

    settlement = runner.shutdown()

    assert not await asyncio.wait_for(settlement, timeout=1)
    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_runner_wrong_loop_shutdown_is_retryable_before_sealing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = _StubbornProcess()

    async def fake_create_subprocess_exec(
        *argv: str | bytes,
        **kwargs: object,
    ) -> _StubbornProcess:
        del argv, kwargs
        return child

    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    runner = AsyncGitProcessRunner(
        terminate_timeout=0.001,
        kill_timeout=0.001,
    )
    command = asyncio.create_task(
        runner.run(
            ("git", "status"),
            cwd="/repo",
            environment={},
            timeout=None,
        )
    )
    await asyncio.wait_for(child.communicate_started.wait(), timeout=1)

    try:
        with pytest.raises(GitShutdownAffinityError):
            await asyncio.wait_for(
                asyncio.to_thread(runner.shutdown),
                timeout=1,
            )

        assert not runner._sealed
        assert runner._shutdown_event is not None
        assert not runner._shutdown_event.is_set()
        assert child.terminate_calls == 0
        assert child.kill_calls == 0

        settlement = runner.shutdown()
        assert not await asyncio.wait_for(settlement, timeout=1)
        assert (await asyncio.wait_for(command, timeout=1)).termination_uncertain
    finally:
        child._never.set()
        command.cancel()
        await asyncio.gather(command, return_exceptions=True)


def test_index_command_is_complete_nul_safe_and_has_explicit_boundary() -> None:
    argv = build_index_argv("/private/bin/git")

    assert argv[:2] == ("/private/bin/git", "--literal-pathspecs")
    assert argv[-1] == "--"
    assert "ls-files" in argv
    assert "-z" in argv
    assert "--stage" in argv
    assert "-v" in argv


def test_stage_command_is_one_exact_fail_fast_literal_path_operation() -> None:
    argv = build_stage_argv(
        "git",
        (b"notes/-leading.md", b"notes/[literal]*.md"),
    )

    assert argv == (
        "git",
        "--literal-pathspecs",
        "-c",
        "add.ignoreErrors=false",
        "add",
        "--all",
        "--",
        b"notes/-leading.md",
        b"notes/[literal]*.md",
    )


def test_unstage_api_has_exact_synchronous_admission_signature() -> None:
    signature = inspect.signature(FileNotesGitService.start_unstage)

    assert tuple(signature.parameters) == ("self", "binding", "group_ids")
    assert signature.return_annotation == "asyncio.Task[GitActionResult]"


def test_unstage_command_is_one_exact_no_worktree_index_operation() -> None:
    argv = build_unstage_argv("/private/bin/git")

    assert argv == (
        "/private/bin/git",
        "update-index",
        "-z",
        "--index-info",
    )
    assert not {"checkout", "restore", "reset", "read-tree"}.intersection(argv)


def test_unstage_payload_removes_owned_conflicts_before_exact_baselines() -> None:
    group = SessionChangeGroup(
        group_id=1,
        endpoints=("created.md", "tree", "tree/owned.md", "tracked.md"),
        source_path="tree",
        destination_path="tree/owned.md",
        current_path="tree/owned.md",
        latest_action="moved",
        latest_sequence=1,
        move_edges=(("tree", "tree/owned.md"),),
    )
    tracked_baseline = _entry("tracked.md", object_id=OID_A)
    tree_baseline = _entry("tree", object_id=OID_B)
    owned_child = _entry("tree/owned.md", object_id=OID_C)
    ownership = StagingOwnership(
        repository=_repository(),
        head=HeadIdentity.attached("refs/heads/main", OID_A),
        approved_endpoint_topology=group.endpoints,
        approved_move_edges=group.move_edges,
        approved_current_path=group.current_path,
        original_baselines={
            "tracked.md": IndexBaseline(tracked_baseline),
            "created.md": IndexBaseline(None),
            "tree": IndexBaseline(tree_baseline),
        },
        post_stage_entries={
            "tracked.md": _entry("tracked.md", object_id=OID_C),
            "created.md": _entry("created.md", object_id=OID_C),
            "tree": None,
            "tree/owned.md": owned_child,
        },
    )

    payload = build_update_index_payload(
        ownership,
        {
            "tracked.md": ownership.post_stage_entries["tracked.md"],
            "created.md": ownership.post_stage_entries["created.md"],
            "tree/owned.md": owned_child,
        },
    )

    assert payload == (
        b"0 " + ZERO_OID.encode() + b"\ttree/owned.md\0"
        b"0 " + ZERO_OID.encode() + b"\tcreated.md\0"
        b"100644 " + OID_A.encode() + b" 0\ttracked.md\0"
        b"100644 " + OID_B.encode() + b" 0\ttree\0"
    )


def test_unstage_payload_uses_one_exact_repository_object_id_width() -> None:
    group = _single_group("wide.md")
    wide_oid = "d" * 64
    wide_baseline = _entry("wide.md", object_id=wide_oid)
    ownership = StagingOwnership(
        repository=_repository(),
        head=HeadIdentity.attached("refs/heads/main", wide_oid),
        approved_endpoint_topology=group.endpoints,
        approved_move_edges=group.move_edges,
        approved_current_path=group.current_path,
        original_baselines={"wide.md": IndexBaseline(wide_baseline)},
        post_stage_entries={
            "wide.md": _entry("wide.md", object_id="e" * 64),
        },
    )

    payload = build_update_index_payload(
        ownership,
        {"wide.md": ownership.post_stage_entries["wide.md"]},
    )

    assert payload == b"100644 " + wide_oid.encode() + b" 0\twide.md\0"

    mixed_width = StagingOwnership(
        repository=ownership.repository,
        head=ownership.head,
        approved_endpoint_topology=ownership.approved_endpoint_topology,
        approved_move_edges=ownership.approved_move_edges,
        approved_current_path=ownership.approved_current_path,
        original_baselines={
            "wide.md": IndexBaseline(_entry("wide.md", object_id=OID_A)),
        },
        post_stage_entries=ownership.post_stage_entries,
    )
    with pytest.raises(ValueError, match="object ID width"):
        build_update_index_payload(
            mixed_width,
            {"wide.md": ownership.post_stage_entries["wide.md"]},
        )


@pytest.mark.parametrize(
    "relative_path",
    ("../escape.md", "/absolute.md", "empty//component.md", ".git/config"),
)
def test_unstage_index_path_mapper_rejects_unsafe_components(
    tmp_path: Path,
    relative_path: str,
) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    notes_root = repository_root / "notes"
    notes_root.mkdir()
    repository = _repository_at(repository_root)
    service = FileNotesGitService(FileNotesSessionOwner(), git_executable="git")

    mapped, row = service._map_group_for_unstage(
        notes_root,
        repository,
        _single_group(relative_path),
    )

    assert mapped is None
    assert row is not None
    assert row.state == "unsupported"


def test_unstage_index_path_mapper_rejects_nested_git_and_symlink_boundaries(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    notes_root = repository_root / "notes"
    notes_root.mkdir()
    nested = notes_root / "nested"
    nested.mkdir()
    (nested / ".git").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (notes_root / "linked").symlink_to(outside, target_is_directory=True)
    repository = _repository_at(repository_root)
    service = FileNotesGitService(FileNotesSessionOwner(), git_executable="git")

    nested_mapping, nested_row = service._map_group_for_unstage(
        notes_root,
        repository,
        _single_group("nested/note.md"),
    )
    linked_mapping, linked_row = service._map_group_for_unstage(
        notes_root,
        repository,
        _single_group("linked/note.md"),
    )

    assert nested_mapping is None
    assert nested_row is not None
    assert nested_row.state == "nested_repository"
    assert linked_mapping is None
    assert linked_row is not None
    assert linked_row.state == "unsupported"


def test_unstage_index_path_mapper_preserves_bytes_and_root_containment(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    notes_root = repository_root / "notes"
    notes_root.mkdir()
    outside_root = tmp_path / "outside"
    outside_root.mkdir()
    repository = _repository_at(repository_root)
    service = FileNotesGitService(FileNotesSessionOwner(), git_executable="git")
    byte_path = os.fsdecode(b"raw-\xff.md")

    mapped, row = service._map_group_for_unstage(
        notes_root,
        repository,
        _single_group(byte_path),
    )
    outside_mapping, outside_row = service._map_group_for_unstage(
        outside_root,
        repository,
        _single_group("note.md"),
    )

    assert row is None
    assert mapped is not None
    assert os.fsencode(mapped.endpoints[0]) == b"notes/raw-\xff.md"
    assert outside_mapping is None
    assert outside_row is not None
    assert outside_row.state == "unsupported"


def test_parse_index_entries_preserves_stage_and_semantic_flags() -> None:
    payload = (
        b"H 100644 " + OID_A.encode("ascii") + b" 0\tnormal.md\0"
        b"S 100755 " + OID_B.encode("ascii") + b" 0\tsparse.md\0"
        b"h 100644 " + OID_C.encode("ascii") + b" 2\tassumed.md\0"
    )

    entries = parse_index_entries_z(payload)

    assert entries == (
        _entry("normal.md"),
        _entry(
            "sparse.md",
            mode="100755",
            object_id=OID_B,
            flags=("skip-worktree",),
        ),
        _entry(
            "assumed.md",
            object_id=OID_C,
            stage=2,
            flags=("assume-unchanged",),
        ),
    )


def test_unmerged_index_stages_are_preserved_and_classified_as_conflict() -> None:
    payload = b"".join(
        b"H 100644 "
        + object_id.encode("ascii")
        + f" {stage}\tconflict.md\0".encode()
        for stage, object_id in (
            (1, OID_A),
            (2, OID_B),
            (3, OID_C),
        )
    )
    group = coalesce_session_changes(
        (_change(1, "modified", "conflict.md"),)
    )[0]

    entries = parse_index_entries_z(payload)
    (row,) = classify_session_rows((group,), (), entries, {})

    assert tuple(entry.stage for entry in entries) == (1, 2, 3)
    assert row.state == "conflict"
    assert row.disabled_reason == "Git conflict"


@pytest.mark.parametrize(
    "payload",
    [
        b"H 100644 " + OID_A.encode("ascii") + b" 0\tmissing-nul.md",
        b"H malformed\0",
        (
            b"H 100644 " + OID_A.encode("ascii") + b" 0\tduplicate.md\0"
            b"H 100644 " + OID_A.encode("ascii") + b" 0\tduplicate.md\0"
        ),
        b"H 100644 not-an-object 0\tbad.md\0",
        b"H 100644 " + OID_A.encode("ascii") + b" x\tbad-stage.md\0",
    ],
)
def test_parse_index_entries_fails_closed_on_malformed_or_duplicate_data(
    payload: bytes,
) -> None:
    with pytest.raises(GitIndexParseError):
        parse_index_entries_z(payload)


def _repository_at(root: Path) -> RepositoryIdentity:
    git_dir = root / ".git"
    git_dir.mkdir(exist_ok=True)
    worktree_stat = root.stat(follow_symlinks=False)
    git_stat = git_dir.stat(follow_symlinks=False)
    return RepositoryIdentity(
        worktree_root=str(root),
        git_dir=str(git_dir),
        git_common_dir=str(git_dir),
        worktree_identity=FileSystemIdentity(
            worktree_stat.st_dev,
            worktree_stat.st_ino,
        ),
        git_dir_identity=FileSystemIdentity(
            git_stat.st_dev,
            git_stat.st_ino,
        ),
        git_common_dir_identity=FileSystemIdentity(
            git_stat.st_dev,
            git_stat.st_ino,
        ),
    )


class _DelayedStatusRunner:
    def __init__(self, *, index_path: str = "note.md") -> None:
        self.first_index_started = asyncio.Event()
        self.release_first_index = asyncio.Event()
        self.status_completed = asyncio.Event()
        self.calls: list[tuple[str | bytes, ...]] = []
        self.query_count = 0
        self.shutdown_calls = 0
        self.add_seen = False
        self.index_path = index_path

    async def run(
        self,
        argv: tuple[str | bytes, ...],
        *,
        cwd: str,
        environment: Mapping[str, str],
        stdin: bytes | None = None,
        timeout: float | None = None,
    ) -> GitCommandResult:
        del environment, stdin, timeout
        command = tuple(argv)
        self.calls.append(command)
        text = tuple(os.fsdecode(argument) for argument in command)
        if "config" in text:
            return GitCommandResult(1, b"", b"")
        if "symbolic-ref" in text:
            return GitCommandResult(0, b"refs/heads/main\n", b"")
        if "--show-toplevel" in text:
            return GitCommandResult(
                0,
                os.fsencode(cwd) + b"\n",
                b"",
            )
        if "--absolute-git-dir" in text or "--git-common-dir" in text:
            return GitCommandResult(
                0,
                os.fsencode(Path(cwd) / ".git") + b"\n",
                b"",
            )
        if "rev-parse" in text:
            return GitCommandResult(0, OID_A.encode("ascii") + b"\n", b"")
        if "ls-files" in text:
            self.query_count += 1
            if self.query_count == 1:
                self.first_index_started.set()
                await self.release_first_index.wait()
            if self.add_seen:
                return GitCommandResult(
                    0,
                    b"H 100644 "
                    + OID_B.encode()
                    + b" 0\t"
                    + os.fsencode(self.index_path)
                    + b"\0",
                    b"",
                )
            return GitCommandResult(0, b"", b"")
        if "status" in text:
            boundary = command.index("--")
            payload = b"".join(
                b"? " + os.fsencode(argument) + b"\0"
                for argument in command[boundary + 1 :]
            )
            self.status_completed.set()
            return GitCommandResult(0, payload, b"")
        if "add" in text:
            self.add_seen = True
            return GitCommandResult(0, b"", b"")
        raise AssertionError(f"Unexpected Git command: {text!r}")

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        self.release_first_index.set()


class _DelayedUnstageRunner(_DelayedStatusRunner):
    def __init__(self, *, index_path: str = "note.md") -> None:
        super().__init__(index_path=index_path)
        self.add_seen = True
        self.update_stdins: list[bytes] = []

    async def run(
        self,
        argv: tuple[str | bytes, ...],
        *,
        cwd: str,
        environment: Mapping[str, str],
        stdin: bytes | None = None,
        timeout: float | None = None,
    ) -> GitCommandResult:
        text = tuple(os.fsdecode(argument) for argument in argv)
        if "update-index" in text:
            self.calls.append(tuple(argv))
            assert stdin is not None
            self.update_stdins.append(stdin)
            self.add_seen = False
            return GitCommandResult(0, b"", b"")
        return await super().run(
            argv,
            cwd=cwd,
            environment=environment,
            stdin=stdin,
            timeout=timeout,
        )


class _PausedUnstageRunner(_DelayedUnstageRunner):
    def __init__(self) -> None:
        super().__init__()
        self.release_first_index.set()
        self.update_started = asyncio.Event()
        self.release_update = asyncio.Event()

    async def run(
        self,
        argv: tuple[str | bytes, ...],
        *,
        cwd: str,
        environment: Mapping[str, str],
        stdin: bytes | None = None,
        timeout: float | None = None,
    ) -> GitCommandResult:
        text = tuple(os.fsdecode(argument) for argument in argv)
        if "update-index" in text:
            self.calls.append(tuple(argv))
            assert stdin is not None
            self.update_stdins.append(stdin)
            self.update_started.set()
            await self.release_update.wait()
            self.add_seen = False
            return GitCommandResult(0, b"", b"")
        return await super().run(
            argv,
            cwd=cwd,
            environment=environment,
            stdin=stdin,
            timeout=timeout,
        )

    def shutdown(self) -> Awaitable[bool]:
        self.shutdown_calls += 1
        self.release_update.set()

        async def settle() -> bool:
            await asyncio.sleep(0)
            return True

        return asyncio.create_task(settle())


class _HeadFailureStatusRunner(_DelayedStatusRunner):
    def __init__(self) -> None:
        super().__init__()
        self.release_first_index.set()

    async def run(
        self,
        argv: tuple[str | bytes, ...],
        *,
        cwd: str,
        environment: Mapping[str, str],
        stdin: bytes | None = None,
        timeout: float | None = None,
    ) -> GitCommandResult:
        text = tuple(os.fsdecode(argument) for argument in argv)
        if "symbolic-ref" in text:
            self.calls.append(tuple(argv))
            return GitCommandResult(
                128,
                b"",
                b"fatal: HEAD permission denied\x00\x1b\n",
            )
        return await super().run(
            argv,
            cwd=cwd,
            environment=environment,
            stdin=stdin,
            timeout=timeout,
        )


class _TwoPhaseStatusRunner(_DelayedStatusRunner):
    def __init__(self) -> None:
        super().__init__()
        self.second_index_started = asyncio.Event()
        self.release_second_index = asyncio.Event()

    async def run(
        self,
        argv: tuple[str | bytes, ...],
        *,
        cwd: str,
        environment: Mapping[str, str],
        stdin: bytes | None = None,
        timeout: float | None = None,
    ) -> GitCommandResult:
        text = tuple(os.fsdecode(argument) for argument in argv)
        if "ls-files" not in text:
            return await super().run(
                argv,
                cwd=cwd,
                environment=environment,
                stdin=stdin,
                timeout=timeout,
            )
        self.calls.append(tuple(argv))
        self.query_count += 1
        if self.query_count == 1:
            self.first_index_started.set()
            await self.release_first_index.wait()
        elif self.query_count == 2:
            self.second_index_started.set()
            await self.release_second_index.wait()
        return GitCommandResult(0, b"", b"")


def _status_service(
    tmp_path: Path,
) -> tuple[
    FileNotesSessionOwner,
    object,
    FileNotesGitService,
    _DelayedStatusRunner,
]:
    root = tmp_path / "notes"
    root.mkdir()
    owner = FileNotesSessionOwner()
    binding = owner.select_root(root)
    repository = _repository_at(root)
    assert owner.publish_trust(binding, repository)
    runner = _DelayedStatusRunner()
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={"PATH": "/bin"},
    )
    return owner, binding, service, runner


def _unstage_service(
    tmp_path: Path,
    *,
    runner: _DelayedUnstageRunner | None = None,
    relative_path: str = "note.md",
    create_file: bool = True,
) -> tuple[
    FileNotesSessionOwner,
    SessionBinding,
    FileNotesGitService,
    _DelayedUnstageRunner,
]:
    root = tmp_path / "notes"
    root.mkdir()
    if create_file:
        (root / relative_path).write_text("owned\n", encoding="utf-8")
    owner = FileNotesSessionOwner()
    binding = owner.select_root(root)
    assert owner.record_change(
        binding,
        SessionChange("created", relative_path),
    )
    repository = _repository_at(root)
    assert owner.publish_trust(binding, repository)
    group = _single_group(relative_path)
    assert owner.publish_ownership(
        binding,
        {
            1: StagingOwnership(
                repository=repository,
                head=HeadIdentity.attached("refs/heads/main", OID_A),
                approved_endpoint_topology=group.endpoints,
                approved_move_edges=group.move_edges,
                approved_current_path=group.current_path,
                original_baselines={relative_path: IndexBaseline(None)},
                post_stage_entries={
                    relative_path: _entry(relative_path, object_id=OID_B),
                },
            )
        },
    )
    selected_runner = runner or _DelayedUnstageRunner(
        index_path=relative_path,
    )
    service = FileNotesGitService(
        owner,
        runner=selected_runner,
        git_executable="git",
        environment={"PATH": "/bin"},
    )
    return owner, binding, service, selected_runner


def _publish_actionable_status(
    owner: FileNotesSessionOwner,
    binding: SessionBinding,
    repository: RepositoryIdentity,
    group: SessionChangeGroup,
) -> SessionGitStatus:
    generation = owner.next_status_generation(binding)
    assert generation is not None
    status = SessionGitStatus(
        binding_generation=binding.generation,
        status_generation=generation,
        state="ready",
        rows=(
            SessionGitRow(
                group,
                "unstaged",
                stage_action="stage",
            ),
        ),
        repository=repository,
        head=HeadIdentity.attached("refs/heads/main", OID_A),
    )
    assert owner.publish_status(binding, status)
    return status


class _ObservedMismatchThenFailureRunner(_DelayedStatusRunner):
    def __init__(self, scenario: str) -> None:
        super().__init__()
        self.release_first_index.set()
        self.scenario = scenario
        self.mismatch_observed = False
        self.later_failure_observed = False

    async def run(
        self,
        argv: tuple[str | bytes, ...],
        *,
        cwd: str,
        environment: Mapping[str, str],
        stdin: bytes | None = None,
        timeout: float | None = None,
    ) -> GitCommandResult:
        text = tuple(os.fsdecode(argument) for argument in argv)
        if (
            self.scenario == "head_then_index_failure"
            and "HEAD^{commit}" in text
        ):
            self.calls.append(tuple(argv))
            self.mismatch_observed = True
            return GitCommandResult(0, OID_C.encode("ascii") + b"\n", b"")
        if "ls-files" in text:
            self.calls.append(tuple(argv))
            if self.scenario == "head_then_index_failure":
                self.later_failure_observed = True
                return GitCommandResult(1, b"", b"index unavailable")
            self.mismatch_observed = True
            return GitCommandResult(
                0,
                b"H 100644 " + OID_C.encode("ascii") + b" 0\tnote.md\0",
                b"",
            )
        if (
            self.scenario == "index_then_status_parse_failure"
            and "status" in text
        ):
            self.calls.append(tuple(argv))
            self.later_failure_observed = True
            return GitCommandResult(0, b"? note.md", b"")
        return await super().run(
            argv,
            cwd=cwd,
            environment=environment,
            stdin=stdin,
            timeout=timeout,
        )


def _status_service_with_owned_note(
    tmp_path: Path,
    runner: _ObservedMismatchThenFailureRunner,
) -> tuple[
    FileNotesSessionOwner,
    SessionBinding,
    FileNotesGitService,
]:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "note.md").write_text("note\n", encoding="utf-8")
    owner = FileNotesSessionOwner()
    binding = owner.select_root(root)
    assert owner.record_change(binding, SessionChange("modified", "note.md"))
    repository = _repository_at(root)
    assert owner.publish_trust(binding, repository)
    group = _single_group("note.md")
    assert owner.publish_ownership(
        binding,
        {
            group.group_id: StagingOwnership(
                repository=repository,
                head=HeadIdentity.attached("refs/heads/main", OID_A),
                approved_endpoint_topology=group.endpoints,
                approved_move_edges=group.move_edges,
                approved_current_path=group.current_path,
                original_baselines={"note.md": IndexBaseline(None)},
                post_stage_entries={
                    "note.md": _entry("note.md", object_id=OID_B)
                },
            )
        },
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
    )
    return owner, binding, service


@pytest.mark.asyncio
async def test_observed_head_mismatch_revokes_before_later_index_failure(
    tmp_path: Path,
) -> None:
    runner = _ObservedMismatchThenFailureRunner("head_then_index_failure")
    owner, binding, service = _status_service_with_owned_note(tmp_path, runner)

    result = await service.start_status(
        binding,
        owner.snapshot(binding).changes,
    )

    assert runner.mismatch_observed
    assert runner.later_failure_observed
    assert result.state == "stale"
    assert not owner.snapshot(binding).staging_ownership


@pytest.mark.asyncio
async def test_observed_index_mismatch_revokes_before_later_status_parse_failure(
    tmp_path: Path,
) -> None:
    runner = _ObservedMismatchThenFailureRunner(
        "index_then_status_parse_failure"
    )
    owner, binding, service = _status_service_with_owned_note(tmp_path, runner)

    result = await service.start_status(
        binding,
        owner.snapshot(binding).changes,
    )

    assert runner.mismatch_observed
    assert runner.later_failure_observed
    assert result.state == "error"
    assert not owner.snapshot(binding).staging_ownership


@pytest.mark.asyncio
async def test_head_uncertainty_retains_disabled_rows_and_clears_ownership(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "note.md").write_text("note\n", encoding="utf-8")
    owner = FileNotesSessionOwner()
    binding = owner.select_root(root)
    repository = _repository_at(root)
    assert owner.publish_trust(binding, repository)
    group = _single_group("note.md")
    _publish_actionable_status(owner, binding, repository, group)
    assert owner.publish_ownership(
        binding,
        {group.group_id: _ownership(group, {"note.md": _entry("note.md")})},
    )
    runner = _HeadFailureStatusRunner()
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
    )

    result = await service.start_status(
        binding,
        (_change(1, "modified", "note.md"),),
    )

    assert result.state == "error"
    assert result.message is not None
    assert "permission denied" in result.message
    assert "\n" not in result.message
    assert "\x00" not in result.message
    assert "\x1b" not in result.message
    assert tuple(row.group_id for row in result.rows) == (group.group_id,)
    assert all(not row.stage_eligible for row in result.rows)
    assert all(not row.unstage_eligible for row in result.rows)
    assert all(row.disabled_reason for row in result.rows)
    snapshot = owner.snapshot(binding)
    assert snapshot.git_status == result
    assert not snapshot.staging_ownership


@pytest.mark.asyncio
async def test_repository_identity_loss_never_retains_previous_rows(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    owner = FileNotesSessionOwner()
    binding = owner.select_root(root)
    repository = _repository_at(root)
    assert owner.publish_trust(binding, repository)
    group = _single_group("note.md")
    _publish_actionable_status(owner, binding, repository, group)
    assert owner.publish_ownership(
        binding,
        {group.group_id: _ownership(group, {"note.md": _entry("note.md")})},
    )
    (root / ".git").rename(root / ".git-replaced")
    (root / ".git").mkdir()
    runner = _DelayedStatusRunner()
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
    )

    result = await service.start_status(
        binding,
        (_change(1, "modified", "note.md"),),
    )

    assert result.state == "stale"
    assert not result.rows
    snapshot = owner.snapshot(binding)
    assert snapshot.trusted_repository is None
    assert snapshot.git_status is None
    assert not snapshot.staging_ownership


@pytest.mark.asyncio
async def test_retained_status_is_read_only_and_exact_binding(
    tmp_path: Path,
) -> None:
    _owner, binding, service, runner = _status_service(tmp_path)
    assert isinstance(binding, SessionBinding)
    task = service.start_status(
        binding,
        (_change(1, "modified", "note.md"),),
    )
    await runner.first_index_started.wait()

    assert service.retained_status(binding) is task
    assert (
        service.retained_status(
            SessionBinding(binding.root_key, binding.generation + 1)
        )
        is None
    )

    runner.release_first_index.set()
    await task
    await asyncio.sleep(0)
    assert service.retained_status(binding) is None


@pytest.mark.asyncio
async def test_hidden_change_invalidates_admitted_status_until_reopen(
    tmp_path: Path,
) -> None:
    owner, binding, service, runner = _status_service(tmp_path)
    assert isinstance(binding, SessionBinding)
    root = tmp_path / "notes"
    first_path = "one.md"
    second_path = "two.md"
    (root / first_path).write_text("one\n", encoding="utf-8")
    assert owner.record_change(binding, SessionChange("created", first_path))

    first = service.start_status(binding, owner.snapshot(binding).changes)
    await asyncio.wait_for(runner.first_index_started.wait(), timeout=1)

    (root / second_path).write_text("two\n", encoding="utf-8")
    assert owner.record_change(binding, SessionChange("created", second_path))
    assert owner.clear_status(binding)
    assert runner.query_count == 1

    runner.release_first_index.set()
    result = await asyncio.wait_for(first, timeout=1)

    assert result.state == "ready"
    assert tuple(row.group.current_path for row in result.rows) == (first_path,)
    assert owner.snapshot(binding).git_status is None
    assert runner.query_count == 1

    refreshed = service.start_status(
        binding,
        owner.snapshot(binding).changes,
    )
    refreshed_result = await asyncio.wait_for(refreshed, timeout=1)

    assert runner.query_count == 2
    assert tuple(
        row.group.current_path for row in refreshed_result.rows
    ) == (first_path, second_path)
    assert owner.snapshot(binding).git_status == refreshed_result


@pytest.mark.asyncio
async def test_hidden_change_invalidates_earlier_coalesced_status_request(
    tmp_path: Path,
) -> None:
    owner, binding, service, runner = _status_service(tmp_path)
    assert isinstance(binding, SessionBinding)
    root = tmp_path / "notes"
    paths = ("one.md", "two.md", "three.md")
    for path in paths:
        (root / path).write_text(path, encoding="utf-8")

    assert owner.record_change(binding, SessionChange("created", paths[0]))
    first = service.start_status(binding, owner.snapshot(binding).changes)
    await asyncio.wait_for(runner.first_index_started.wait(), timeout=1)

    assert owner.record_change(binding, SessionChange("created", paths[1]))
    assert owner.clear_status(binding)
    assert (
        service.start_status(binding, owner.snapshot(binding).changes) is first
    )
    assert owner.record_change(binding, SessionChange("created", paths[2]))
    assert owner.clear_status(binding)

    runner.release_first_index.set()
    result = await asyncio.wait_for(first, timeout=1)

    assert runner.query_count == 2
    assert tuple(row.group.current_path for row in result.rows) == paths[:2]
    assert owner.snapshot(binding).git_status is None

    refreshed = service.start_status(
        binding,
        owner.snapshot(binding).changes,
    )
    refreshed_result = await asyncio.wait_for(refreshed, timeout=1)

    assert runner.query_count == 3
    assert tuple(
        row.group.current_path for row in refreshed_result.rows
    ) == paths
    assert owner.snapshot(binding).git_status == refreshed_result


@pytest.mark.asyncio
async def test_ten_status_triggers_coalesce_to_active_plus_latest_rerun(
    tmp_path: Path,
) -> None:
    owner, binding, service, runner = _status_service(tmp_path)
    del owner
    paths = tuple(f"note-{index}.md" for index in range(11))
    for path in paths:
        (tmp_path / "notes" / path).write_text("note\n", encoding="utf-8")
    first = service.start_status(
        binding,  # type: ignore[arg-type]
        (_change(1, "modified", paths[0]),),
    )
    await runner.first_index_started.wait()

    coalesced = [
        service.start_status(
            binding,  # type: ignore[arg-type]
            (_change(index + 2, "modified", path),),
        )
        for index, path in enumerate(paths[1:])
    ]

    assert all(task is first for task in coalesced)
    runner.release_first_index.set()
    result = await asyncio.wait_for(first, timeout=1)
    assert runner.query_count == 2
    assert tuple(row.group.current_path for row in result.rows) == (paths[-1],)


@pytest.mark.asyncio
async def test_trigger_during_final_rerun_marks_result_stale_without_third_child(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    paths = ("one.md", "two.md", "three.md")
    for path in paths:
        (root / path).write_text(path, encoding="utf-8")
    owner = FileNotesSessionOwner()
    binding = owner.select_root(root)
    repository = _repository_at(root)
    assert owner.publish_trust(binding, repository)
    runner = _TwoPhaseStatusRunner()
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={"PATH": "/bin"},
    )
    first = service.start_status(
        binding,
        (_change(1, "modified", paths[0]),),
    )
    await asyncio.wait_for(runner.first_index_started.wait(), timeout=1)
    assert (
        service.start_status(
            binding,
            (_change(2, "modified", paths[1]),),
        )
        is first
    )
    runner.release_first_index.set()
    await asyncio.wait_for(runner.second_index_started.wait(), timeout=1)

    assert (
        service.start_status(
            binding,
            (_change(3, "modified", paths[2]),),
        )
        is first
    )
    runner.release_second_index.set()
    result = await asyncio.wait_for(first, timeout=1)

    assert runner.query_count == 2
    assert result.state == "stale"
    assert result.message is not None
    assert "newer" in result.message.lower()
    assert tuple(row.group.current_path for row in result.rows) == (paths[1],)
    assert all(not row.stage_eligible for row in result.rows)
    assert all(not row.unstage_eligible for row in result.rows)

    refreshed = service.start_status(
        binding,
        (_change(3, "modified", paths[2]),),
    )
    assert refreshed is not first
    refreshed_result = await asyncio.wait_for(refreshed, timeout=1)
    assert runner.query_count == 3
    assert refreshed_result.state == "ready"
    assert tuple(
        row.group.current_path for row in refreshed_result.rows
    ) == (paths[2],)


@pytest.mark.asyncio
async def test_pending_status_rerun_is_suppressed_by_admitted_mutation(
    tmp_path: Path,
) -> None:
    owner, binding, service, runner = _status_service(tmp_path)
    root = tmp_path / "notes"
    (root / "one.md").write_text("one\n", encoding="utf-8")
    (root / "two.md").write_text("two\n", encoding="utf-8")
    first = service.start_status(
        binding,  # type: ignore[arg-type]
        (_change(1, "modified", "one.md"),),
    )
    await runner.first_index_started.wait()
    assert (
        service.start_status(
            binding,  # type: ignore[arg-type]
            (_change(2, "modified", "two.md"),),
        )
        is first
    )
    mutation = owner.try_acquire_mutation(binding)  # type: ignore[arg-type]
    assert mutation is not None

    runner.release_first_index.set()
    result = await asyncio.wait_for(first, timeout=1)

    assert runner.query_count == 1
    assert result.state == "stale"
    assert owner.try_acquire_status(binding) is None  # type: ignore[arg-type]
    mutation.release()
    status_lease = owner.try_acquire_status(binding)  # type: ignore[arg-type]
    assert status_lease is not None
    status_lease.release()


@pytest.mark.asyncio
async def test_status_after_mutation_admission_starts_no_child_or_rerun(
    tmp_path: Path,
) -> None:
    owner, binding, service, runner = _status_service(tmp_path)
    mutation = owner.try_acquire_mutation(binding)  # type: ignore[arg-type]
    assert mutation is not None

    with pytest.raises(GitStatusAdmissionError) as error:
        service.start_status(
            binding,  # type: ignore[arg-type]
            (_change(1, "modified", "note.md"),),
        )

    assert error.value.reason == "mutation_active"
    assert not runner.calls
    mutation.release()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["stage", "unstage"])
async def test_stage_and_unstage_transition_refusal_is_synchronous(
    tmp_path: Path,
    action: str,
) -> None:
    owner, binding, service, runner = _status_service(tmp_path)
    transition = owner.try_acquire_transition(binding, "path")  # type: ignore[arg-type]
    assert transition is not None

    with pytest.raises(GitMutationAdmissionError) as error:
        getattr(service, f"start_{action}")(binding, (1,))

    assert error.value.reason == "transition_active"
    assert service._action_cycle is None
    assert service._action_waiter is None
    assert not runner.calls
    transition.release()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["stage", "unstage"])
async def test_stage_and_unstage_hold_mutation_lease_while_status_settles(
    tmp_path: Path,
    action: str,
) -> None:
    owner, binding, service, runner = _status_service(tmp_path)
    root = tmp_path / "notes"
    (root / "note.md").write_text("note\n", encoding="utf-8")
    assert owner.record_change(
        binding,  # type: ignore[arg-type]
        SessionChange("created", "note.md"),
    )
    status = service.start_status(
        binding,  # type: ignore[arg-type]
        (_change(1, "created", "note.md"),),
    )
    await runner.first_index_started.wait()

    action_waiter = getattr(service, f"start_{action}")(binding, (1,))

    assert owner.try_acquire_transition(binding, "root") is None  # type: ignore[arg-type]
    with pytest.raises(GitStatusAdmissionError) as error:
        service.start_status(
            binding,  # type: ignore[arg-type]
            (_change(2, "modified", "note.md"),),
        )
    assert error.value.reason == "mutation_active"
    assert not action_waiter.done()

    runner.release_first_index.set()
    await status
    result = await action_waiter
    assert result.state in {"success", "blocked"}
    transition = owner.try_acquire_transition(binding, "root")  # type: ignore[arg-type]
    assert transition is not None
    transition.release()


@pytest.mark.asyncio
async def test_cancelled_unstage_waiter_does_not_cancel_retained_action(
    tmp_path: Path,
) -> None:
    paused_runner = _PausedUnstageRunner()
    owner, binding, service, runner = _unstage_service(
        tmp_path,
        runner=paused_runner,
    )

    waiter = service.start_unstage(binding, (1,))
    await asyncio.wait_for(runner.update_started.wait(), timeout=1)
    retained = service._action_cycle
    assert retained is not None

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert not retained.cancelled()
    assert not retained.done()

    runner.release_update.set()
    result = await asyncio.wait_for(retained, timeout=1)

    assert result.state == "success"
    assert result.unstaged_group_ids == (1,)
    assert not owner.snapshot(binding).staging_ownership
    transition = owner.try_acquire_transition(binding, "screen")
    assert transition is not None
    transition.release()


@pytest.mark.asyncio
async def test_shutdown_settles_active_unstage_and_releases_mutation(
    tmp_path: Path,
) -> None:
    paused_runner = _PausedUnstageRunner()
    owner, binding, service, runner = _unstage_service(
        tmp_path,
        runner=paused_runner,
    )
    unstage = service.start_unstage(binding, (1,))
    await asyncio.wait_for(runner.update_started.wait(), timeout=1)

    settlement = service.shutdown()
    await asyncio.wait_for(settlement, timeout=1)

    result = await unstage
    assert result.state == "uncertain"
    assert runner.shutdown_calls == 1
    assert not owner.snapshot(binding).staging_ownership
    transition = owner.try_acquire_transition(binding, "screen")
    assert transition is not None
    transition.release()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw_path",
    (b"raw-\xff.md", b"tab\tand-newline\n.md"),
    ids=("surrogateescaped", "tab-newline"),
)
async def test_unstage_execution_preserves_exact_filename_bytes_in_stdin(
    tmp_path: Path,
    raw_path: bytes,
) -> None:
    # macOS rejects creating raw non-UTF-8 names, so exercise that byte
    # boundary through the retained runner while still using start_unstage.
    relative_path = os.fsdecode(raw_path)
    assert os.fsencode(relative_path) == raw_path
    byte_runner = _DelayedUnstageRunner(index_path=relative_path)
    byte_runner.release_first_index.set()
    _, binding, service, runner = _unstage_service(
        tmp_path,
        runner=byte_runner,
        relative_path=relative_path,
        create_file=False,
    )

    result = await service.start_unstage(binding, (1,))

    assert result.state == "success"
    assert result.unstaged_group_ids == (1,)
    assert runner.update_stdins == [
        b"0 " + b"0" * 40 + b"\t" + raw_path + b"\0",
    ]
    assert runner.update_stdins[0].count(b"\0") == 1


@pytest.mark.asyncio
async def test_trigger_after_mutation_admission_cannot_piggyback_active_cycle(
    tmp_path: Path,
) -> None:
    owner, binding, service, runner = _status_service(tmp_path)
    root = tmp_path / "notes"
    (root / "one.md").write_text("one\n", encoding="utf-8")
    (root / "two.md").write_text("two\n", encoding="utf-8")
    first = service.start_status(
        binding,  # type: ignore[arg-type]
        (_change(1, "modified", "one.md"),),
    )
    await runner.first_index_started.wait()
    mutation = owner.try_acquire_mutation(binding)  # type: ignore[arg-type]
    assert mutation is not None

    with pytest.raises(GitStatusAdmissionError) as error:
        service.start_status(
            binding,  # type: ignore[arg-type]
            (_change(2, "modified", "two.md"),),
        )

    assert error.value.reason == "mutation_active"
    runner.release_first_index.set()
    await asyncio.wait_for(first, timeout=1)
    assert runner.query_count == 1
    mutation.release()


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_cancel_service_status_publication(
    tmp_path: Path,
) -> None:
    owner, binding, service, runner = _status_service(tmp_path)
    (tmp_path / "notes" / "note.md").write_text("note\n", encoding="utf-8")
    waiter = service.start_status(
        binding,  # type: ignore[arg-type]
        (_change(1, "modified", "note.md"),),
    )
    await runner.first_index_started.wait()

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    runner.release_first_index.set()
    await asyncio.wait_for(runner.status_completed.wait(), timeout=1)
    await asyncio.sleep(0)

    status = owner.snapshot(binding).git_status  # type: ignore[arg-type]
    assert status is not None
    assert status.state == "ready"
    lease = owner.try_acquire_status(binding)  # type: ignore[arg-type]
    assert lease is not None
    lease.release()


def test_production_builder_attaches_exactly_one_git_service() -> None:
    owner = build_file_notes_session_owner()

    attached = owner.attached_git_service()

    assert isinstance(attached, FileNotesGitService)
    with pytest.raises(RuntimeError, match="already attached"):
        owner.attach_git_service(FileNotesGitService(owner))
    owner.shutdown()


@pytest.mark.asyncio
async def test_owner_exposes_retained_git_shutdown_settlement() -> None:
    release = asyncio.Event()

    class AttachedService:
        def __init__(self) -> None:
            self.shutdown_calls = 0
            self.settlement: asyncio.Task[bool] | None = None

        def shutdown(self) -> asyncio.Task[bool]:
            self.shutdown_calls += 1
            if self.settlement is None:
                self.settlement = asyncio.create_task(release.wait())
            return self.settlement

    service = AttachedService()
    owner = FileNotesSessionOwner()
    owner.attach_git_service(service)

    owner.shutdown()
    settlement = asyncio.create_task(owner.settle_git_shutdown())
    await asyncio.sleep(0)

    assert service.shutdown_calls == 1
    assert not settlement.done()
    release.set()
    await asyncio.wait_for(settlement, timeout=1)
    await owner.settle_git_shutdown()


@pytest.mark.asyncio
async def test_owner_wrong_loop_shutdown_can_retry_on_owning_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    owner = FileNotesSessionOwner()
    binding = owner.select_root(root)
    repository = _repository_at(root)
    assert owner.publish_trust(binding, repository)
    child = _StubbornProcess()

    async def fake_create_subprocess_exec(
        *argv: str | bytes,
        **kwargs: object,
    ) -> _StubbornProcess:
        del argv, kwargs
        return child

    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    runner = AsyncGitProcessRunner(
        terminate_timeout=0.001,
        kill_timeout=0.001,
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
    )
    owner.attach_git_service(service)
    status = service.start_status(
        binding,
        (_change(1, "modified", "note.md"),),
    )
    await asyncio.wait_for(child.communicate_started.wait(), timeout=1)

    try:
        with pytest.raises(GitShutdownAffinityError):
            await asyncio.wait_for(
                asyncio.to_thread(owner.shutdown),
                timeout=1,
            )

        assert owner.attached_git_service() is service
        assert owner.record_change(
            binding,
            SessionChange("modified", "retry.md"),
        )
        assert not service._sealed
        assert not runner._sealed
        assert child.terminate_calls == 0
        assert child.kill_calls == 0

        await asyncio.wait_for(owner.shutdown_async(), timeout=1)
        assert owner.attached_git_service() is None
        assert child.terminate_calls == 1
        assert child.kill_calls == 1
        assert (await status).state == "stale"
    finally:
        child._never.set()
        status.cancel()
        await asyncio.gather(status, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_creation", [1, 2])
async def test_status_task_creation_failure_releases_admitted_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_creation: int,
) -> None:
    owner, binding, service, _runner = _status_service(tmp_path)
    loop = asyncio.get_running_loop()
    original_create_task = loop.create_task
    failed_coroutines: list[object] = []
    created_tasks: list[asyncio.Task[object]] = []
    creation_count = 0

    def fail_task_creation(
        coroutine: object,
        *args: object,
        **kwargs: object,
    ) -> asyncio.Task[object]:
        nonlocal creation_count
        creation_count += 1
        if creation_count == failed_creation:
            failed_coroutines.append(coroutine)
            raise RuntimeError("task creation failed")
        task = original_create_task(  # type: ignore[arg-type]
            coroutine,
            *args,
            **kwargs,
        )
        created_tasks.append(task)  # type: ignore[arg-type]
        return task  # type: ignore[return-value]

    with monkeypatch.context() as context:
        context.setattr(loop, "create_task", fail_task_creation)
        with pytest.raises(RuntimeError, match="task creation failed"):
            service.start_status(
                binding,  # type: ignore[arg-type]
                (_change(1, "modified", "note.md"),),
            )

    assert len(failed_coroutines) == 1
    assert getattr(failed_coroutines[0], "cr_frame", None) is None
    await asyncio.sleep(0)
    assert all(task.done() for task in created_tasks)

    lease = owner.try_acquire_status(binding)  # type: ignore[arg-type]
    assert lease is not None
    lease.release()


def test_create_task_closes_coroutine_without_running_loop() -> None:
    service = FileNotesGitService(FileNotesSessionOwner())

    async def operation() -> SessionGitStatus:
        raise AssertionError("closed coroutine must not run")

    coroutine = operation()

    with pytest.raises(RuntimeError, match="no running event loop"):
        service._create_task(coroutine)

    assert coroutine.cr_frame is None


@pytest.mark.asyncio
async def test_shutdown_seals_status_and_prevents_late_publication(
    tmp_path: Path,
) -> None:
    owner, binding, service, runner = _status_service(tmp_path)
    (tmp_path / "notes" / "note.md").write_text("note\n", encoding="utf-8")
    group = _single_group("note.md")
    assert owner.publish_ownership(
        binding,  # type: ignore[arg-type]
        {1: _ownership(group, {"note.md": _entry("note.md")})},
    )
    waiter = service.start_status(
        binding,  # type: ignore[arg-type]
        (_change(1, "modified", "note.md"),),
    )
    await runner.first_index_started.wait()

    settlement = service.shutdown()
    assert service.shutdown() is settlement

    with pytest.raises(GitStatusAdmissionError) as error:
        service.start_status(
            binding,  # type: ignore[arg-type]
            (_change(2, "modified", "note.md"),),
    )
    assert error.value.reason == "shutdown"
    assert inspect.isawaitable(settlement)
    cancelled_waiter = asyncio.ensure_future(settlement)
    cancelled_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_waiter
    await asyncio.wait_for(settlement, timeout=1)
    assert waiter.done()
    await waiter
    snapshot = owner.snapshot(binding)  # type: ignore[arg-type]
    assert snapshot.git_status is None
    assert not snapshot.staging_ownership
    assert runner.shutdown_calls == 1


class _UncertainStatusRunner(_DelayedStatusRunner):
    async def run(
        self,
        argv: tuple[str | bytes, ...],
        *,
        cwd: str,
        environment: Mapping[str, str],
        stdin: bytes | None = None,
        timeout: float | None = None,
    ) -> GitCommandResult:
        text = tuple(os.fsdecode(argument) for argument in argv)
        if "ls-files" in text:
            self.calls.append(tuple(argv))
            self.query_count += 1
            return GitCommandResult(
                None,
                b"",
                b"uncertain\x00child\n",
                timed_out=True,
                termination_uncertain=True,
            )
        return await super().run(
            argv,
            cwd=cwd,
            environment=environment,
            stdin=stdin,
            timeout=timeout,
        )


@pytest.mark.asyncio
async def test_uncertain_child_termination_publishes_stale_and_clears_ownership(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "note.md").write_text("note\n", encoding="utf-8")
    owner = FileNotesSessionOwner()
    binding = owner.select_root(root)
    repository = _repository_at(root)
    assert owner.publish_trust(binding, repository)
    group = _single_group("note.md")
    assert owner.publish_ownership(
        binding,
        {1: _ownership(group, {"note.md": _entry("note.md")})},
    )
    runner = _UncertainStatusRunner()
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
    )

    status = await service.start_status(
        binding,
        (_change(1, "modified", "note.md"),),
    )

    assert status.state == "stale"
    assert status.message is not None
    assert "uncertain" in status.message
    snapshot = owner.snapshot(binding)
    assert snapshot.git_status == status
    assert not snapshot.staging_ownership


class _OutsideWhitelistRunner(_DelayedStatusRunner):
    async def run(
        self,
        argv: tuple[str | bytes, ...],
        *,
        cwd: str,
        environment: Mapping[str, str],
        stdin: bytes | None = None,
        timeout: float | None = None,
    ) -> GitCommandResult:
        text = tuple(os.fsdecode(argument) for argument in argv)
        if "status" in text:
            self.calls.append(tuple(argv))
            return GitCommandResult(0, b"? outside.md\0", b"")
        return await super().run(
            argv,
            cwd=cwd,
            environment=environment,
            stdin=stdin,
            timeout=timeout,
        )


@pytest.mark.asyncio
async def test_status_rejects_git_output_outside_repo_coordinate_whitelist(
    tmp_path: Path,
) -> None:
    owner, binding, service, original_runner = _status_service(tmp_path)
    (tmp_path / "notes" / "note.md").write_text("note\n", encoding="utf-8")
    runner = _OutsideWhitelistRunner()
    runner.release_first_index.set()
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
    )
    del original_runner

    status = await service.start_status(
        binding,  # type: ignore[arg-type]
        (_change(1, "modified", "note.md"),),
    )

    assert status.state == "error"
    assert status.message is not None
    assert "outside the session whitelist" in status.message


class _PreAddEndpointRaceRunner(_DelayedStatusRunner):
    def __init__(
        self,
        *,
        root: Path,
        relative_path: str,
        replacement: str,
    ) -> None:
        super().__init__(index_path=relative_path)
        self.release_first_index.set()
        self.root = root
        self.relative_path = relative_path
        self.replacement = replacement
        self.identity_checks = 0

    async def run(
        self,
        argv: tuple[str | bytes, ...],
        *,
        cwd: str,
        environment: Mapping[str, str],
        stdin: bytes | None = None,
        timeout: float | None = None,
    ) -> GitCommandResult:
        text = tuple(os.fsdecode(argument) for argument in argv)
        if "--show-toplevel" in text:
            self.identity_checks += 1
            if self.identity_checks == 2:
                if self.replacement == "directory":
                    endpoint = self.root / self.relative_path
                    endpoint.unlink()
                    endpoint.mkdir()
                    (endpoint / "external.md").write_text(
                        "outside session lineage\n",
                        encoding="utf-8",
                    )
                else:
                    (self.root / "nested" / ".git").mkdir()
        return await super().run(
            argv,
            cwd=cwd,
            environment=environment,
            stdin=stdin,
            timeout=timeout,
        )


class _PreAddOwnedEndpointRaceRunner(_PreAddEndpointRaceRunner):
    def __init__(self, *, root: Path) -> None:
        super().__init__(
            root=root,
            relative_path="note.md",
            replacement="directory",
        )
        self.add_seen = True
        self.add_calls = 0

    async def run(
        self,
        argv: tuple[str | bytes, ...],
        *,
        cwd: str,
        environment: Mapping[str, str],
        stdin: bytes | None = None,
        timeout: float | None = None,
    ) -> GitCommandResult:
        text = tuple(os.fsdecode(argument) for argument in argv)
        if "add" in text:
            self.add_calls += 1
        if "status" in text:
            self.calls.append(tuple(argv))
            payload = (
                f"1 .M N... 100644 100644 100644 {OID_B} {OID_B} "
                "note.md\0"
            ).encode()
            return GitCommandResult(0, payload, b"")
        return await super().run(
            argv,
            cwd=cwd,
            environment=environment,
            stdin=stdin,
            timeout=timeout,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("replacement", "relative_path"),
    [
        ("directory", "note.md"),
        ("nested_repository", "nested/note.md"),
    ],
)
async def test_stage_rechecks_endpoint_safety_after_final_repository_revalidation(
    tmp_path: Path,
    replacement: str,
    relative_path: str,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    endpoint = root / relative_path
    endpoint.parent.mkdir(parents=True, exist_ok=True)
    endpoint.write_text("session note\n", encoding="utf-8")
    owner = FileNotesSessionOwner()
    binding = owner.select_root(root)
    assert owner.record_change(binding, SessionChange("created", relative_path))
    repository = _repository_at(root)
    assert owner.publish_trust(binding, repository)
    runner = _PreAddEndpointRaceRunner(
        root=root,
        relative_path=relative_path,
        replacement=replacement,
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
    )

    result = await service.start_stage(binding, (1,))

    assert runner.identity_checks >= 2
    assert result.state == "blocked"
    assert result.blocked_group_ids == (1,)
    assert not runner.add_seen
    assert not owner.snapshot(binding).staging_ownership


@pytest.mark.asyncio
async def test_blocked_pre_add_stage_update_retains_existing_ownership(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "note.md").write_text("newer session edit\n", encoding="utf-8")
    owner = FileNotesSessionOwner()
    binding = owner.select_root(root)
    assert owner.record_change(binding, SessionChange("modified", "note.md"))
    repository = _repository_at(root)
    assert owner.publish_trust(binding, repository)
    group = _single_group("note.md")
    saved_ownership = StagingOwnership(
        repository=repository,
        head=HeadIdentity.attached("refs/heads/main", OID_A),
        approved_endpoint_topology=group.endpoints,
        approved_move_edges=group.move_edges,
        approved_current_path=group.current_path,
        original_baselines={
            "note.md": IndexBaseline(_entry("note.md", object_id=OID_A))
        },
        post_stage_entries={
            "note.md": _entry("note.md", object_id=OID_B)
        },
    )
    assert owner.publish_ownership(binding, {1: saved_ownership})
    runner = _PreAddOwnedEndpointRaceRunner(root=root)
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
    )

    result = await service.start_stage(binding, (1,))

    assert result.state == "blocked"
    assert result.blocked_group_ids == (1,)
    assert runner.add_calls == 0
    assert owner.snapshot(binding).staging_ownership == {1: saved_ownership}


class _PostflightStageRaceRunner(_DelayedStatusRunner):
    def __init__(
        self,
        *,
        race: str,
        owner: FileNotesSessionOwner,
        binding: SessionBinding,
        root: Path,
    ) -> None:
        super().__init__()
        self.release_first_index.set()
        self.race = race
        self.owner = owner
        self.binding = binding
        self.root = root
        self.add_seen = False

    async def run(
        self,
        argv: tuple[str | bytes, ...],
        *,
        cwd: str,
        environment: Mapping[str, str],
        stdin: bytes | None = None,
        timeout: float | None = None,
    ) -> GitCommandResult:
        text = tuple(os.fsdecode(argument) for argument in argv)
        if "add" in text:
            self.calls.append(tuple(argv))
            self.add_seen = True
            if self.race == "topology":
                assert self.owner.record_change(
                    self.binding,
                    SessionChange("moved", "note.md", "next.md"),
                )
            return GitCommandResult(0, b"", b"")
        if self.add_seen and self.race == "identity" and "--show-toplevel" in text:
            replacement = self.root / "replacement"
            replacement.mkdir(exist_ok=True)
            self.calls.append(tuple(argv))
            return GitCommandResult(0, os.fsencode(replacement) + b"\n", b"")
        if (
            self.add_seen
            and self.race == "head"
            and "rev-parse" in text
            and "HEAD^{commit}" in text
        ):
            self.calls.append(tuple(argv))
            return GitCommandResult(0, OID_B.encode() + b"\n", b"")
        if self.add_seen and self.race == "semantic" and "ls-files" in text:
            self.calls.append(tuple(argv))
            return GitCommandResult(
                0,
                b"S 100644 " + OID_B.encode() + b" 0\tnote.md\0",
                b"",
            )
        if self.add_seen and self.race == "index_topology" and "ls-files" in text:
            self.calls.append(tuple(argv))
            return GitCommandResult(
                0,
                b"H 100644 " + OID_B.encode() + b" 0\tnote.md/child\0",
                b"",
            )
        return await super().run(
            argv,
            cwd=cwd,
            environment=environment,
            stdin=stdin,
            timeout=timeout,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "race",
    ["identity", "head", "index_topology", "semantic", "topology"],
)
async def test_stage_postflight_races_publish_no_ownership(
    tmp_path: Path,
    race: str,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "note.md").write_text("note\n", encoding="utf-8")
    owner = FileNotesSessionOwner()
    binding = owner.select_root(root)
    assert owner.record_change(binding, SessionChange("created", "note.md"))
    repository = _repository_at(root)
    assert owner.publish_trust(binding, repository)
    runner = _PostflightStageRaceRunner(
        race=race,
        owner=owner,
        binding=binding,
        root=root,
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
    )

    result = await service.start_stage(binding, (1,))

    assert result.state == "uncertain"
    assert not owner.snapshot(binding).staging_ownership
    transition = owner.try_acquire_transition(binding, "path")
    assert transition is not None
    transition.release()


class _ShutdownDuringStageRunner(_DelayedStatusRunner):
    def __init__(self) -> None:
        super().__init__()
        self.release_first_index.set()
        self.add_started = asyncio.Event()
        self.release_add = asyncio.Event()

    async def run(
        self,
        argv: tuple[str | bytes, ...],
        *,
        cwd: str,
        environment: Mapping[str, str],
        stdin: bytes | None = None,
        timeout: float | None = None,
    ) -> GitCommandResult:
        text = tuple(os.fsdecode(argument) for argument in argv)
        if "add" in text:
            self.calls.append(tuple(argv))
            self.add_started.set()
            await self.release_add.wait()
            return GitCommandResult(
                None,
                b"",
                b"shutdown",
                termination_uncertain=True,
            )
        return await super().run(
            argv,
            cwd=cwd,
            environment=environment,
            stdin=stdin,
            timeout=timeout,
        )

    def shutdown(self) -> Awaitable[bool]:
        self.shutdown_calls += 1
        self.release_add.set()

        async def settle() -> bool:
            await asyncio.sleep(0)
            return True

        return asyncio.create_task(settle())


@pytest.mark.asyncio
async def test_shutdown_settles_active_stage_and_releases_mutation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "note.md").write_text("note\n", encoding="utf-8")
    owner = FileNotesSessionOwner()
    binding = owner.select_root(root)
    assert owner.record_change(binding, SessionChange("created", "note.md"))
    repository = _repository_at(root)
    assert owner.publish_trust(binding, repository)
    runner = _ShutdownDuringStageRunner()
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
    )
    stage = service.start_stage(binding, (1,))
    await runner.add_started.wait()

    settlement = service.shutdown()
    await asyncio.wait_for(settlement, timeout=1)

    result = await stage
    assert result.state == "uncertain"
    assert runner.shutdown_calls == 1
    assert not owner.snapshot(binding).staging_ownership
    transition = owner.try_acquire_transition(binding, "screen")
    assert transition is not None
    transition.release()
