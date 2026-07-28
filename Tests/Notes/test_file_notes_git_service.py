from __future__ import annotations

import os
from collections.abc import Mapping

import pytest

from tldw_chatbook.Notes.file_notes_git_service import (
    PorcelainV2ParseError,
    PorcelainPathOutsideSessionError,
    PorcelainRecord,
    classify_session_rows,
    coalesce_session_changes,
    compute_stage_closure,
    compute_unstage_closure,
    index_entry_has_unsupported_semantics,
    index_entry_signature,
    ownership_signature_matches,
    parse_porcelain_v2_z,
    stage_group_is_closed,
    stage_pathspecs,
    unstage_group_is_closed,
)
from tldw_chatbook.Notes.file_notes_session_owner import (
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
        {"old.md": _entry("old.md")},
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
        {"clean.md": _entry("clean.md")},
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
    assert index_entry_has_unsupported_semantics(entry)

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


def test_index_signature_is_exact_for_mode_object_stage_and_flags() -> None:
    base = _entry(
        "note.md",
        mode="100755",
        object_id=OID_C,
        stage=2,
        flags=("skip-worktree", "assume-unchanged"),
    )

    assert index_entry_signature(base) == (
        "100755",
        OID_C,
        2,
        ("assume-unchanged", "skip-worktree"),
    )
    assert index_entry_signature(
        _entry(
            "note.md",
            mode="100644",
            object_id=OID_C,
            stage=2,
            flags=("skip-worktree", "assume-unchanged"),
        )
    ) != index_entry_signature(base)
    assert index_entry_signature(
        _entry(
            "note.md",
            mode="100755",
            object_id=OID_A,
            stage=2,
            flags=("skip-worktree", "assume-unchanged"),
        )
    ) != index_entry_signature(base)
    assert index_entry_signature(
        _entry(
            "note.md",
            mode="100755",
            object_id=OID_C,
            stage=1,
            flags=("skip-worktree", "assume-unchanged"),
        )
    ) != index_entry_signature(base)
    assert index_entry_signature(
        _entry(
            "note.md",
            mode="100755",
            object_id=OID_C,
            stage=2,
            flags=("skip-worktree",),
        )
    ) != index_entry_signature(base)


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
