from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import Mapping
from pathlib import Path

import pytest

from tldw_chatbook.Notes.file_notes_git_service import (
    AsyncGitProcessRunner,
    DiscoveryResult,
    FileNotesGitService,
    GitCommandResult,
    GitIndexParseError,
    GitStatusAdmissionError,
    PorcelainV2ParseError,
    PorcelainPathOutsideSessionError,
    PorcelainRecord,
    build_git_environment,
    build_index_argv,
    build_file_notes_session_owner,
    build_status_argv,
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
    SessionChange,
    SessionChangeGroup,
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


def test_status_environment_and_argv_disable_side_channel_writes() -> None:
    environment = build_git_environment(
        {"PATH": "/private/bin"},
        for_status=True,
    )
    argv = build_status_argv(
        "/private/bin/git",
        (os.fsencode("-leading.md"), os.fsencode("line\nbreak.md")),
    )

    assert environment["GIT_OPTIONAL_LOCKS"] == "0"
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
        assert not runner._processes
        assert not runner._cleanup_tasks
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
    def __init__(self) -> None:
        self.first_index_started = asyncio.Event()
        self.release_first_index = asyncio.Event()
        self.status_completed = asyncio.Event()
        self.calls: list[tuple[str | bytes, ...]] = []
        self.query_count = 0
        self.shutdown_calls = 0

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
            return GitCommandResult(0, b"", b"")
        if "status" in text:
            boundary = command.index("--")
            payload = b"".join(
                b"? " + os.fsencode(argument) + b"\0"
                for argument in command[boundary + 1 :]
            )
            self.status_completed.set()
            return GitCommandResult(0, payload, b"")
        raise AssertionError(f"Unexpected Git command: {text!r}")

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        self.release_first_index.set()


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
async def test_status_task_creation_failure_releases_admitted_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, binding, service, _runner = _status_service(tmp_path)

    def fail_task_creation(coroutine: object) -> asyncio.Task[object]:
        close = getattr(coroutine, "close", None)
        if close is not None:
            close()
        raise RuntimeError("task creation failed")

    monkeypatch.setattr(service, "_create_task", fail_task_creation)

    with pytest.raises(RuntimeError, match="task creation failed"):
        service.start_status(
            binding,  # type: ignore[arg-type]
            (_change(1, "modified", "note.md"),),
        )

    lease = owner.try_acquire_status(binding)  # type: ignore[arg-type]
    assert lease is not None
    lease.release()


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
