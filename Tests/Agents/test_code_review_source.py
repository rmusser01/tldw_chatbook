"""Real Git/source authority controls for immutable review evidence."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import unicodedata
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents.code_review_models import (
    CommitRangeTarget,
    ListSourceQuery,
    PatchTarget,
    PullRequestTarget,
    ResolveSourceQuery,
    ReviewRefused,
    ReviewRunBinding,
    SearchSourceQuery,
    WorkingTreeTarget,
)
from tldw_chatbook.Agents.code_review_source import (
    ReviewSourceAuthority,
    capture_review_source,
    query_review_source,
    read_review_source,
    use_review_source_authority,
)
from tldw_chatbook.Agents.code_review_store import ReviewStore
from tldw_chatbook.Agents.local_tool_provider import RunAdmittedWorkspaceRoot
from tldw_chatbook.Agents.run_context import use_run_id
from tldw_chatbook.Chat.console_scratch_space import ConsoleScratchSpaceManager


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root).decode().strip()


@pytest.fixture
def source_case(tmp_path, request):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "Test")
    git(root, "config", "user.email", "test@example.invalid")
    for name, text in {
        "main.py": "original\n",
        "helpers.py": "old helper\n",
        "test_main.py": "unchanged test\n",
        "secret.txt": "SECRET\n",
    }.items():
        (root / name).write_text(text)
    git(root, "add", ".")
    git(root, "commit", "-qm", "base")
    base = git(root, "rev-parse", "HEAD")
    (root / "main.py").write_text("from helpers import helper\ncommitted\n")
    git(root, "commit", "-qam", "head")
    head = git(root, "rev-parse", "HEAD")
    (root / "main.py").write_text("staged bytes\n")
    git(root, "add", "main.py")
    (root / "main.py").write_text("captured bytes\n")
    (root / "extra.py").write_text("selected untracked\n")
    (root / "binary").write_bytes(b"\x00\xff")
    # Catalog-only tests never capture sources; retain escape-link coverage
    # for every source/capture fixture consumer.
    if getattr(request, "param", None) != "catalog_only":
        (root / "escape").symlink_to(tmp_path / "outside")
    (tmp_path / "outside").write_text("outside secret")
    admitted = RunAdmittedWorkspaceRoot(
        workspace_id="workspace",
        binding_id="binding",
        alias="project",
        root=root,
        locator_fingerprint="fingerprint",
        root_identity=(("root", 1, 2, 3),),
        allow_write=False,
        guard=lambda write: not write,
        exclusions_provider=lambda: (root / "secret.txt",),
    )
    authority = ReviewSourceAuthority(admitted=admitted, source_epoch=0)
    manager = ConsoleScratchSpaceManager(temp_parent=tmp_path)
    store = ReviewStore(manager, "chat")
    case = SimpleNamespace(
        root=root,
        file=root / "main.py",
        helper=root / "helpers.py",
        base=base,
        head=head,
        store=store,
        authority=authority,
    )
    case.working_tree_target = WorkingTreeTarget(
        kind="working_tree",
        binding_ref="binding",
        staged=True,
        unstaged=True,
        untracked_paths=("extra.py",),
        context_paths=("helpers.py",),
    )
    case.committed_target = CommitRangeTarget(
        kind="commit_range",
        binding_ref="binding",
        base_commit=base,
        head_commit=head,
        comparison="direct",
    )

    def capture(target=None):
        context = store.create(target_digest="0" * 64)
        manifest = capture_review_source(
            target or case.working_tree_target, authority, store, context.review_id
        )
        store.bind_attempt(
            context.review_id,
            ReviewRunBinding(
                review_id=context.review_id,
                slot_id="R1",
                attempt_id="a1",
                role="reviewer",
                target_digest=manifest.target_digest,
                source_epoch=0,
            ),
        )
        return manifest

    case.capture = capture

    @contextmanager
    def bound(manifest):
        with (
            use_run_id(manifest.review_id),
            use_review_source_authority(store, authority),
        ):
            yield

    case.bound = bound

    def resolve(manifest, path="main.py", side="worktree"):
        return query_review_source(
            manifest.review_id,
            ResolveSourceQuery(op="resolve", relative_path=path, side=side),
        ).records[0]["source_ref"]

    case.resolve = resolve
    yield case
    store.close()
    assert manager.dispose()


def test_worktree_read_is_frozen_after_capture(source_case):
    c = source_case
    manifest = c.capture()
    c.file.write_text("newer bytes\n")
    with c.bound(manifest):
        page = read_review_source(
            manifest, c.resolve(manifest), start_line=1, max_tokens=1024
        )
    assert page.text == "captured bytes\n"
    assert page.source_digest == hashlib.sha256(b"captured bytes\n").hexdigest()


def test_resolve_unchanged_helper_uses_pinned_commit(source_case):
    c = source_case
    manifest = c.capture(c.committed_target)
    c.helper.write_text("new helper\n")
    with c.bound(manifest):
        for path, expected in [
            ("helpers.py", "old helper\n"),
            ("test_main.py", "unchanged test\n"),
        ]:
            ref = c.resolve(manifest, path, "head")
            assert (
                read_review_source(manifest, ref, start_line=1, max_tokens=1024).text
                == expected
            )
        names = query_review_source(
            manifest.review_id,
            ListSourceQuery(op="list", relative_dir=".", side="head"),
        )
        assert "secret.txt" not in names.model_dump_json()
        hits = query_review_source(
            manifest.review_id,
            SearchSourceQuery(
                op="search", text="old helper", relative_dir=".", side="head"
            ),
        )
        assert hits.records[0]["start_line"] == 1


def test_mutable_uncaptured_context_refuses(source_case):
    c = source_case
    manifest = c.capture(c.working_tree_target.model_copy(update={"context_paths": ()}))
    with c.bound(manifest), pytest.raises(ReviewRefused, match="context_not_captured"):
        c.resolve(manifest, "helpers.py")


def test_deep_range_and_complete_small_pages(source_case):
    c = source_case
    c.file.write_text("".join(f'{n}: café "quoted"\\\n' for n in range(1, 50012)))
    manifest = c.capture()
    with c.bound(manifest):
        ref = c.resolve(manifest)
        page = read_review_source(
            manifest,
            ref,
            start_line=50000,
            end_line=50010,
            max_tokens=1024,
            max_result_chars=1024,
        )
        received = page.text
        assert page.start_line == 50000
        while page.cursor:
            assert len(page.model_dump_json()) <= 1024
            previous_end = page.end_line
            page = read_review_source(
                manifest,
                ref,
                start_line=50000,
                end_line=50010,
                max_tokens=1024,
                max_result_chars=1024,
                cursor=page.cursor,
            )
            assert page.start_line == previous_end + 1
            received += page.text
        assert received == "".join(
            f'{n}: café "quoted"\\\n' for n in range(50000, 50011)
        )


@pytest.mark.parametrize("start,end", [(0, 1), (2, 1), (True, 1), (1.0, 1), (1, 999)])
def test_invalid_ranges(source_case, start, end):
    c = source_case
    manifest = c.capture()
    with c.bound(manifest), pytest.raises(ReviewRefused, match="invalid_source_range"):
        read_review_source(
            manifest,
            c.resolve(manifest),
            start_line=start,
            end_line=end,
            max_tokens=1024,
        )


def test_worktree_layers_selected_untracked_and_no_writes(source_case):
    c = source_case
    before = git(c.root, "status", "--porcelain=v1", "-z")
    manifest = c.capture()
    with c.bound(manifest):
        for side, expected in [
            ("base", "from helpers"),
            ("staged", "staged bytes"),
            ("worktree", "captured bytes"),
        ]:
            ref = c.resolve(manifest, side=side)
            assert (
                expected
                in read_review_source(manifest, ref, start_line=1, max_tokens=1024).text
            )
        assert c.resolve(manifest, "extra.py")
        with pytest.raises(ReviewRefused, match="context_not_captured"):
            c.resolve(manifest, "binary")
    assert git(c.root, "status", "--porcelain=v1", "-z") == before


@pytest.mark.parametrize("kind", ["pr", "commit_range"])
def test_committed_targets_and_lazy_inventory_preserve_identity(source_case, kind):
    c = source_case
    args = c.committed_target.model_dump()
    args["kind"] = kind
    if kind == "pr":
        args["pr_ref"] = "https://example.invalid/pull/1"
    target = PullRequestTarget(**args) if kind == "pr" else CommitRangeTarget(**args)
    manifest = c.capture(target)
    digest = manifest.target_digest
    with c.store.lease(manifest.review_id) as context:
        revision = context.revision
    git(c.root, "add", "main.py", "helpers.py")
    git(c.root, "commit", "-qm", "move head")
    with c.bound(manifest):
        ref = c.resolve(manifest, "helpers.py", "head")
        assert (
            read_review_source(manifest, ref, start_line=1, max_tokens=1024).text
            == "old helper\n"
        )
    with c.store.lease(manifest.review_id) as context:
        assert context.revision == revision
        assert context.target_digest == digest
        assert context.source_state.inventory_revision > 1


def test_patch_preimages_are_frozen_and_checkout_untouched(source_case):
    c = source_case
    patch = (
        b"--- a/main.py\n+++ b/main.py\n@@ -1 +1 @@\n-captured bytes\n+patched bytes\n"
    )
    c.authority = replace(
        c.authority, resolve_patch=lambda ref: patch if ref == "accepted" else None
    )
    target = PatchTarget(
        kind="patch",
        binding_ref="binding",
        patch_ref="accepted",
        context_paths=("helpers.py",),
    )
    context = c.store.create(target_digest="0" * 64)
    manifest = capture_review_source(target, c.authority, c.store, context.review_id)
    c.store.bind_attempt(
        context.review_id,
        ReviewRunBinding(
            review_id=context.review_id,
            slot_id="R1",
            attempt_id="a1",
            role="reviewer",
            target_digest=manifest.target_digest,
            source_epoch=0,
        ),
    )
    c.helper.write_text("new helper\n")
    with (
        use_run_id(context.review_id),
        use_review_source_authority(c.store, c.authority),
    ):
        for path, side, expected in [
            ("main.py", "base", "captured bytes\n"),
            ("main.py", "head", "patched bytes\n"),
            ("helpers.py", "head", "old helper\n"),
        ]:
            ref = c.resolve(manifest, path, side)
            assert (
                read_review_source(manifest, ref, start_line=1, max_tokens=1024).text
                == expected
            )
    assert c.file.read_text() == "captured bytes\n"


def test_patch_mismatch_never_publishes(source_case):
    c = source_case
    authority = replace(
        c.authority,
        resolve_patch=lambda ref: (
            b"--- a/main.py\n+++ b/main.py\n@@ -1 +1 @@\n-wrong\n+patched\n"
        ),
    )
    context = c.store.create(target_digest="0" * 64)
    with pytest.raises(ReviewRefused, match="patch_preimage_mismatch"):
        capture_review_source(
            PatchTarget(kind="patch", binding_ref="binding", patch_ref="accepted"),
            authority,
            c.store,
            context.review_id,
        )
    assert context.source_state is None
    assert not context.sources
    assert c.store.usage().retained_bytes == 0


def test_mutating_during_capture_refuses_without_publication(source_case, monkeypatch):
    from tldw_chatbook.Agents import code_review_source as module

    c = source_case
    original = module._read_mutable
    calls = 0

    def drift(root, exclusions, path):
        nonlocal calls
        data = original(root, exclusions, path)
        if path == "main.py":
            calls += 1
            if calls == 1:
                c.file.write_text("changed in capture\n")
        return data

    monkeypatch.setattr(module, "_read_mutable", drift)
    context = c.store.create(target_digest="0" * 64)
    with pytest.raises(ReviewRefused, match="source_drift"):
        capture_review_source(
            c.working_tree_target, c.authority, c.store, context.review_id
        )
    assert context.source_state is None
    assert c.store.usage().retained_bytes == 0


@pytest.mark.parametrize(
    "path,reason",
    [
        ("binary", "binary_source"),
        ("escape", "source_path_unavailable"),
        ("missing", "source_unavailable"),
    ],
)
def test_required_unsupported_selection_blocks_capture(source_case, path, reason):
    c = source_case
    target = c.working_tree_target.model_copy(update={"untracked_paths": (path,)})
    with pytest.raises(ReviewRefused, match=reason):
        c.capture(target)


def test_optional_context_omissions_are_body_free_gaps(source_case):
    c = source_case
    manifest = c.capture(
        c.working_tree_target.model_copy(
            update={"context_paths": ("binary", "missing", "secret.txt")}
        )
    )
    assert manifest.gaps
    assert all(gap.source_ref is None for gap in manifest.gaps)
    assert "secret.txt" not in repr(manifest)


def test_source_range_cursor_is_pinned_and_scope_bound(source_case):
    c = source_case
    c.file.write_text("".join(f"{n}:" + "x" * 150 + "\n" for n in range(40)))
    manifest = c.capture()
    with c.bound(manifest):
        ref = c.resolve(manifest)
        first = read_review_source(
            manifest,
            ref,
            start_line=5,
            end_line=30,
            max_tokens=1024,
            max_result_chars=1024,
        )
        assert first.cursor
        second = read_review_source(
            manifest,
            ref,
            start_line=5,
            end_line=30,
            max_tokens=1024,
            max_result_chars=1024,
            cursor=first.cursor,
        )
        replay = read_review_source(
            manifest,
            ref,
            start_line=5,
            end_line=30,
            max_tokens=1024,
            max_result_chars=1024,
            cursor=first.cursor,
        )
        assert replay == second
        with pytest.raises(ReviewRefused, match="source_cursor_conflict"):
            read_review_source(
                manifest,
                ref,
                start_line=1,
                end_line=30,
                max_tokens=1024,
                cursor=first.cursor,
            )
        with pytest.raises(ReviewRefused, match="source_page_capacity_changed"):
            read_review_source(
                manifest,
                ref,
                start_line=5,
                end_line=30,
                max_tokens=1024,
                max_result_chars=500,
                cursor=first.cursor,
            )
        other = c.resolve(manifest, "helpers.py")
        with pytest.raises(ReviewRefused):
            read_review_source(
                manifest, other, start_line=1, max_tokens=1024, cursor=first.cursor
            )
        with pytest.raises(ReviewRefused, match="invalid_source_cursor"):
            read_review_source(
                manifest,
                ref,
                start_line=5,
                end_line=30,
                max_tokens=1024,
                cursor="forged",
            )
    c.store.bind_attempt(
        "foreign",
        ReviewRunBinding(
            review_id=manifest.review_id,
            slot_id="R2",
            attempt_id="a1",
            role="reviewer",
            target_digest=manifest.target_digest,
            source_epoch=0,
        ),
    )
    with (
        use_run_id("foreign"),
        use_review_source_authority(c.store, c.authority),
        pytest.raises(ReviewRefused, match="source_cursor_conflict"),
    ):
        read_review_source(
            manifest,
            ref,
            start_line=5,
            end_line=30,
            max_tokens=1024,
            cursor=first.cursor,
        )


def test_oversized_line_and_control_refuse_explicitly(source_case):
    c = source_case
    c.file.write_text("x" * 6000 + "\n")
    manifest = c.capture()
    with c.bound(manifest):
        ref = c.resolve(manifest)
        with pytest.raises(
            ReviewRefused, match="source_span_too_large_for_tool_result"
        ) as caught:
            read_review_source(
                manifest, ref, start_line=1, max_tokens=1024, max_result_chars=1024
            )
        assert caught.value.gap.source_ref == ref
        with pytest.raises(
            ReviewRefused, match="review_control_too_large_for_tool_result"
        ):
            read_review_source(
                manifest, ref, start_line=1, max_tokens=1024, max_result_chars=10
            )


def test_search_and_list_small_pages_never_skip_entries(source_case):
    c = source_case
    for n in range(5):
        (c.root / f"context-{n:02}.py").write_text("needle\n" * 8)
    git(c.root, "add", ".")
    git(c.root, "commit", "-qm", "context")
    target = c.committed_target.model_copy(
        update={"head_commit": git(c.root, "rev-parse", "HEAD")}
    )
    # Binary and escaping symlink are unchanged context, not required scope.
    target = target.model_copy(update={"base_commit": target.head_commit})
    manifest = c.capture(target)
    with c.bound(manifest):
        for query in [
            ListSourceQuery(op="list", relative_dir=".", side="head"),
            SearchSourceQuery(
                op="search", text="needle", relative_dir=".", side="head"
            ),
        ]:
            page = query_review_source(manifest.review_id, query, max_result_chars=1024)
            found = list(page.records)
            cursors = set()
            while page.cursor:
                assert len(page.model_dump_json()) <= 1024
                assert page.cursor not in cursors
                cursors.add(page.cursor)
                assert len(cursors) < 500
                page = query_review_source(
                    manifest.review_id, query, cursor=page.cursor, max_result_chars=1024
                )
                found.extend(page.records)
            selected = [
                record
                for record in found
                if record["relative_path"].startswith("context-")
            ]
            assert len(selected) == (5 if query.op == "list" else 40)
            assert len(
                {json.dumps(record, sort_keys=True) for record in selected}
            ) == len(selected)


def test_release_discards_source_state_and_all_refs(source_case):
    c = source_case
    manifest = c.capture()
    with c.store.lease(manifest.review_id) as context:
        assert context.source_state is not None
    c.store.release(manifest.review_id, expected_revision=context.revision)
    assert context.source_state is None
    with c.bound(manifest), pytest.raises(ReviewRefused):
        c.resolve(manifest)


def test_symlink_swap_between_validation_and_open_never_reads_escape(
    source_case, monkeypatch
):
    from tldw_chatbook.Agents import code_review_source as module

    c = source_case
    original = module._safe

    def swap(root, exclusions, path, **kwargs):
        result = original(root, exclusions, path, **kwargs)
        c.file.unlink()
        c.file.symlink_to(c.root.parent / "outside")
        return result

    monkeypatch.setattr(module, "_safe", swap)
    with pytest.raises(ReviewRefused):
        module._read_mutable(c.root, (), "main.py")


def test_required_deleted_and_renamed_files_keep_exact_sides(source_case):
    c = source_case
    git(c.root, "reset", "--hard", c.head)
    git(c.root, "mv", "main.py", "renamed.py")
    git(c.root, "rm", "helpers.py")
    git(c.root, "commit", "-qm", "rename and delete")
    target = c.committed_target.model_copy(
        update={"base_commit": c.head, "head_commit": git(c.root, "rev-parse", "HEAD")}
    )
    manifest = c.capture(target)
    with c.bound(manifest):
        assert c.resolve(manifest, "main.py", "base")
        assert c.resolve(manifest, "renamed.py", "head")
        assert c.resolve(manifest, "helpers.py", "base")
        with pytest.raises(ReviewRefused, match="source_absent"):
            c.resolve(manifest, "helpers.py", "head")


def test_deleted_nested_directory_is_valid_commit_comparison(source_case):
    c = source_case
    git(c.root, "reset", "--hard", c.head)
    (c.root / "nested").mkdir()
    (c.root / "nested" / "old.py").write_text("old\n")
    git(c.root, "add", "nested")
    git(c.root, "commit", "-qm", "nested base")
    base = git(c.root, "rev-parse", "HEAD")
    git(c.root, "rm", "-r", "nested")
    git(c.root, "commit", "-qm", "nested removal")
    manifest = c.capture(
        c.committed_target.model_copy(
            update={
                "base_commit": base,
                "head_commit": git(c.root, "rev-parse", "HEAD"),
            }
        )
    )
    assert any(
        doc.relative_path == "nested/old.py" and doc.side == "base"
        for doc in manifest.initial_sources
    )


def test_empty_document_has_complete_zero_line_read(source_case):
    c = source_case
    c.file.write_text("")
    manifest = c.capture()
    with c.bound(manifest):
        page = read_review_source(
            manifest, c.resolve(manifest), start_line=1, max_tokens=1024
        )
    assert page.text == "" and page.cursor is None
    assert page.start_line is None and page.end_line is None


def test_replayed_query_rechecks_new_exclusions(source_case):
    c = source_case
    excluded = [c.root / "secret.txt"]
    authority = replace(
        c.authority,
        admitted=replace(
            c.authority.admitted, exclusions_provider=lambda: tuple(excluded)
        ),
    )
    context = c.store.create(target_digest="0" * 64)
    manifest = capture_review_source(
        c.committed_target, authority, c.store, context.review_id
    )
    c.store.bind_attempt(
        "query-run",
        ReviewRunBinding(
            review_id=manifest.review_id,
            slot_id="R1",
            attempt_id="a1",
            role="reviewer",
            target_digest=manifest.target_digest,
            source_epoch=0,
        ),
    )
    query = ListSourceQuery(op="list", relative_dir=".", side="head")
    with use_run_id("query-run"), use_review_source_authority(c.store, authority):
        first = query_review_source(manifest.review_id, query, max_result_chars=550)
        assert first.cursor
        second = query_review_source(
            manifest.review_id, query, cursor=first.cursor, max_result_chars=550
        )
        excluded.append(c.root / second.records[0]["relative_path"])
        with pytest.raises(ReviewRefused, match="source_path_unavailable"):
            query_review_source(
                manifest.review_id, query, cursor=first.cursor, max_result_chars=550
            )


def test_patch_add_refuses_existing_unmatched_preimage(source_case):
    c = source_case
    patch = b"--- /dev/null\n+++ b/main.py\n@@ -0,0 +1 @@\n+replacement\n"
    authority = replace(c.authority, resolve_patch=lambda ref: patch)
    context = c.store.create(target_digest="0" * 64)
    with pytest.raises(ReviewRefused, match="patch_preimage_mismatch"):
        capture_review_source(
            PatchTarget(kind="patch", binding_ref="binding", patch_ref="accepted"),
            authority,
            c.store,
            context.review_id,
        )


def test_patch_bytes_above_capture_metadata_allowance_are_accepted(source_case):
    c = source_case
    patch = b"--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+" + b"a" * 8000 + b"\n"
    authority = replace(c.authority, resolve_patch=lambda ref: patch)
    context = c.store.create(target_digest="0" * 64)
    manifest = capture_review_source(
        PatchTarget(kind="patch", binding_ref="binding", patch_ref="accepted"),
        authority,
        c.store,
        context.review_id,
    )
    assert manifest.initial_sources[0].byte_count == 8001


def test_missing_pr_history_refuses_no_acquisition(source_case):
    c = source_case
    target = PullRequestTarget(
        kind="pr",
        binding_ref="binding",
        pr_ref="https://example.invalid/1",
        base_commit=c.base,
        head_commit="f" * 40,
        comparison="direct",
    )
    with pytest.raises(ReviewRefused, match="history_unavailable"):
        c.capture(target)


def test_source_queries_reject_outside_scope_and_unknown_revision_paths(source_case):
    c = source_case
    manifest = c.capture(c.committed_target)
    with c.bound(manifest):
        with pytest.raises(ReviewRefused, match="source_path_unavailable"):
            c.resolve(manifest, "secret.txt", "head")
        with pytest.raises(ReviewRefused, match="source_unavailable"):
            c.resolve(manifest, c.head, "head")
        with pytest.raises(ReviewRefused, match="invalid_source_side"):
            c.resolve(manifest, "main.py", "worktree")
        with pytest.raises(ReviewRefused, match="invalid_source_query"):
            query_review_source(
                manifest.review_id,
                {"op": "search", "text": "foo", "relative_dir": "../", "side": "head"},
            )


def test_capture_refuses_after_run_binding_and_after_prior_capture(source_case):
    c = source_case
    manifest = c.capture()
    with pytest.raises(ReviewRefused, match="source_already_published"):
        capture_review_source(
            c.working_tree_target, c.authority, c.store, manifest.review_id
        )


def test_merge_base_comparison_records_actual_common_ancestor(source_case):
    c = source_case
    git(c.root, "reset", "--hard", c.head)
    git(c.root, "checkout", "-qb", "other", c.base)
    (c.root / "other.py").write_text("other\n")
    git(c.root, "add", "other.py")
    git(c.root, "commit", "-qm", "other branch")
    other = git(c.root, "rev-parse", "HEAD")
    manifest = c.capture(
        c.committed_target.model_copy(
            update={"base_commit": other, "comparison": "merge_base"}
        )
    )
    assert dict(manifest.revisions) == {"base": c.base, "head": c.head}
    assert manifest.comparison == "merge_base"


def test_committed_patch_preimage_ignores_dirty_checkout(source_case):
    c = source_case
    patch = b"--- a/main.py\n+++ b/main.py\n@@ -1 +1 @@\n-original\n+patched\n"
    authority = replace(c.authority, resolve_patch=lambda ref: patch)
    context = c.store.create(target_digest="0" * 64)
    manifest = capture_review_source(
        PatchTarget(
            kind="patch",
            binding_ref="binding",
            patch_ref="accepted",
            preimage_commit=c.base,
        ),
        authority,
        c.store,
        context.review_id,
    )
    assert (
        next(doc for doc in manifest.initial_sources if doc.side == "base").digest
        == hashlib.sha256(b"original\n").hexdigest()
    )
    assert c.file.read_text() == "captured bytes\n"


def test_cursor_mapping_is_body_free_and_does_not_advance(source_case):
    from tldw_chatbook.Agents.code_review_source import (
        original_source_path,
        source_cursor_selection,
    )

    c = source_case
    c.file.write_text("line\n" * 1000)
    manifest = c.capture()
    with c.bound(manifest):
        ref = c.resolve(manifest)
        page = read_review_source(
            manifest,
            ref,
            start_line=10,
            end_line=1000,
            max_tokens=1024,
            max_result_chars=1024,
        )
        usage = c.store.usage()
        selection = source_cursor_selection(manifest.review_id, page.cursor)
        assert selection == {
            "op": "content",
            "source_ref": ref,
            "start_line": 10,
            "end_line": 1000,
        }
        assert original_source_path(manifest.review_id, ref) == c.file
        assert c.store.usage() == usage


def test_discovery_page_budget_resumes_nested_inventory(source_case, monkeypatch):
    from tldw_chatbook.Agents import code_review_source as module

    c = source_case
    git(c.root, "reset", "--hard", c.head)
    for name in ("a", "b", "c"):
        (c.root / name).mkdir()
        (c.root / name / "file.py").write_text("needle\n")
    git(c.root, "add", "a", "b", "c")
    git(c.root, "commit", "-qm", "nested context")
    head = git(c.root, "rev-parse", "HEAD")
    manifest = c.capture(
        c.committed_target.model_copy(update={"base_commit": head, "head_commit": head})
    )
    monkeypatch.setattr(module, "MAX_LIST_ENTRIES", 1)
    with c.bound(manifest):
        query = SearchSourceQuery(
            op="search", text="needle", relative_dir=".", side="head"
        )
        page = query_review_source(manifest.review_id, query, max_result_chars=1024)
        found = list(page.records)
        turns = 0
        while page.cursor:
            turns += 1
            assert turns < 15
            page = query_review_source(
                manifest.review_id, query, cursor=page.cursor, max_result_chars=1024
            )
            found.extend(page.records)
    assert {r["relative_path"] for r in found} == {
        "a/file.py",
        "b/file.py",
        "c/file.py",
    }


def test_source_authority_rejects_remote_without_local_git(source_case, monkeypatch):
    from tldw_chatbook.Agents import code_review_source as module
    from tldw_chatbook.Tools.remote_root_types import RemoteRoot

    c = source_case
    from pathlib import PurePosixPath

    remote = RemoteRoot(
        alias="project",
        canonical_locator="ssh://example.invalid/project",
        root=PurePosixPath("/project"),
        binding_id="binding",
    )

    def deny(*args, **kwargs):
        raise AssertionError("local Git must not be called")

    authority = replace(
        c.authority, admitted=replace(c.authority.admitted, root=remote)
    )
    monkeypatch.setattr(module, "ReviewGitReader", deny)
    context = c.store.create(target_digest="0" * 64)
    with pytest.raises(ReviewRefused, match="target_acquisition_unavailable"):
        capture_review_source(c.committed_target, authority, c.store, context.review_id)


def test_indivisible_query_entry_refuses_without_clipping(source_case):
    c = source_case
    relative = "/".join(["a" * 100, "b" * 100, "c" * 100 + ".py"])
    path = c.root / relative
    path.parent.mkdir(parents=True)
    path.write_text("small\n")
    git(c.root, "add", relative)
    git(c.root, "commit", "-qm", "long name")
    head = git(c.root, "rev-parse", "HEAD")
    manifest = c.capture(
        c.committed_target.model_copy(update={"base_commit": head, "head_commit": head})
    )
    with (
        c.bound(manifest),
        pytest.raises(
            ReviewRefused, match="review_record_too_large_for_tool_result"
        ) as caught,
    ):
        query_review_source(
            manifest.review_id,
            ResolveSourceQuery(op="resolve", relative_path=relative, side="head"),
            max_result_chars=512,
        )
    assert caught.value.gap.reason == "review_record_too_large_for_tool_result"


def test_source_query_is_literal_and_conflicting_cursor_refuses(source_case):
    c = source_case
    c.file.write_text("literal .*\n" + "plain\n" * 200)
    manifest = c.capture()
    with c.bound(manifest):
        hits = query_review_source(
            manifest.review_id,
            SearchSourceQuery(
                op="search", text=".*", relative_dir=".", side="worktree"
            ),
        )
        assert len(hits.records) == 1
        page = query_review_source(
            manifest.review_id,
            SearchSourceQuery(
                op="search", text="plain", relative_dir=".", side="worktree"
            ),
            max_result_chars=1024,
        )
        assert page.cursor
        with pytest.raises(ReviewRefused, match="source_cursor_conflict"):
            query_review_source(
                manifest.review_id,
                SearchSourceQuery(
                    op="search", text="other", relative_dir=".", side="worktree"
                ),
                cursor=page.cursor,
            )


def test_capture_storage_failure_leaves_no_manifest(source_case):
    from tldw_chatbook.Agents.code_review_models import ReviewLimits

    c = source_case
    # Use the same real scratch owner with an independent tab and tiny source cap.
    small = ReviewStore(
        c.store._manager, "small-chat", limits=ReviewLimits(source_max_bytes=5)
    )
    context = small.create(target_digest="0" * 64)
    with pytest.raises(ReviewRefused, match="storage_quota_exceeded"):
        capture_review_source(
            c.working_tree_target, c.authority, small, context.review_id
        )
    assert context.source_state is None and not context.sources
    assert small.usage().retained_bytes == 0
    small.close()


def test_source_paths_normalize_without_expanding_authority(source_case):
    c = source_case
    manifest = c.capture(
        c.working_tree_target.model_copy(
            update={"scope": "./", "context_paths": ("./helpers.py",)}
        )
    )
    with c.bound(manifest):
        assert c.resolve(manifest, "./helpers.py") == c.resolve(manifest, "helpers.py")


def test_captured_worktree_deletion_is_explicit_absence(source_case):
    c = source_case
    c.file.unlink()
    manifest = c.capture()
    assert ("worktree", "main.py") in manifest.absent_sources
    with c.bound(manifest), pytest.raises(ReviewRefused, match="source_absent"):
        c.resolve(manifest)


def test_mutable_query_rejects_side_outside_target(source_case):
    c = source_case
    manifest = c.capture()
    with c.bound(manifest), pytest.raises(ReviewRefused, match="invalid_source_side"):
        query_review_source(
            manifest.review_id,
            ListSourceQuery(op="list", relative_dir=".", side="head"),
        )


def test_query_coverage_gap_survives_record_page_boundary(source_case):
    c = source_case
    for name, data in [
        ("a.py", b"safe\n"),
        ("b.bin", b"\x00binary"),
        ("c" + ("x" * 120) + ".py", b"safe\n"),
    ]:
        (c.root / name).write_bytes(data)
    git(c.root, "add", ".")
    git(c.root, "commit", "-qm", "query omissions")
    head = git(c.root, "rev-parse", "HEAD")
    manifest = c.capture(
        c.committed_target.model_copy(update={"base_commit": head, "head_commit": head})
    )
    with c.bound(manifest):
        query = ListSourceQuery(op="list", relative_dir=".", side="head")
        page = query_review_source(manifest.review_id, query, max_result_chars=600)
        gaps = list(page.gaps)
        while page.cursor:
            page = query_review_source(
                manifest.review_id, query, cursor=page.cursor, max_result_chars=600
            )
            gaps.extend(page.gaps)
    assert any(gap.reason == "binary_source" for gap in gaps)


def test_foreign_chat_and_guessed_manifest_cannot_read_source(source_case):
    c = source_case
    manifest = c.capture()
    foreign = ReviewStore(c.store._manager, "foreign-chat")
    with (
        use_run_id(manifest.review_id),
        use_review_source_authority(foreign, c.authority),
        pytest.raises(ReviewRefused, match="unknown_review"),
    ):
        query_review_source(
            manifest.review_id,
            ListSourceQuery(op="list", relative_dir=".", side="worktree"),
        )
    with (
        c.bound(manifest),
        pytest.raises(ReviewRefused, match="invalid_source_manifest"),
    ):
        read_review_source(
            replace(manifest), c.resolve(manifest), start_line=1, max_tokens=1024
        )
    foreign.close()


def test_lazy_history_failure_uses_review_refusal(source_case):
    c = source_case
    manifest = c.capture(c.committed_target)
    git(c.root, "config", "remote.origin.promisor", "true")
    with c.bound(manifest), pytest.raises(ReviewRefused, match="history_unavailable"):
        c.resolve(manifest, "helpers.py", "head")


@pytest.mark.parametrize("create", [False, True])
def test_repeated_patch_target_refuses_before_capture(source_case, create):
    from tldw_chatbook.Tools.patch_tool_impls import parse_patch_targets

    c = source_case
    if create:
        patch = (
            b"--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+first\n"
            b"--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+second\n"
        )
    else:
        c.file.write_text("a\nb\n")
        patch = (
            b"--- a/main.py\n+++ b/main.py\n@@ -1 +1 @@\n-a\n+A\n"
            b"--- a/main.py\n+++ b/main.py\n@@ -2 +2 @@\n-b\n+B\n"
        )
    assert len(parse_patch_targets(patch.decode())) == 2
    authority = replace(c.authority, resolve_patch=lambda ref: patch)
    context = c.store.create(target_digest="0" * 64)
    before = c.file.read_bytes()
    with pytest.raises(ReviewRefused, match="duplicate_patch_target"):
        capture_review_source(
            PatchTarget(kind="patch", binding_ref="binding", patch_ref="accepted"),
            authority,
            c.store,
            context.review_id,
        )
    assert context.source_state is None and not context.sources
    assert not context.source_finalized
    assert c.store.usage().retained_bytes == 0
    assert c.file.read_bytes() == before
    assert not (c.root / "new.py").exists()


@pytest.mark.parametrize("unstaged", [False, True])
def test_selected_untracked_file_staged_during_capture_refuses(
    source_case, monkeypatch, unstaged
):
    from tldw_chatbook.Agents import code_review_source as module

    c = source_case
    original = module._read_mutable
    staged = False

    def stage_after_read(root, exclusions, path):
        nonlocal staged
        data = original(root, exclusions, path)
        if path == "extra.py" and not staged:
            git(c.root, "add", "extra.py")
            staged = True
        return data

    monkeypatch.setattr(module, "_read_mutable", stage_after_read)
    target = c.working_tree_target.model_copy(
        update={"staged": False, "unstaged": unstaged}
    )
    context = c.store.create(target_digest="0" * 64)
    before = (c.root / "extra.py").read_bytes()
    with pytest.raises(ReviewRefused, match="source_drift"):
        capture_review_source(target, c.authority, c.store, context.review_id)
    assert staged
    assert (c.root / "extra.py").read_bytes() == before
    assert git(c.root, "ls-files", "--", "extra.py") == "extra.py"
    assert context.source_state is None and not context.sources
    assert not context.source_finalized
    assert c.store.usage().retained_bytes == 0


def _git_calls(monkeypatch):
    """Record every Git argv the review reader runs, still executing it."""
    from tldw_chatbook.Tools import git_tool_impls

    calls = []
    original = git_tool_impls._run_git_bytes

    def counted(args, **kwargs):
        calls.append(tuple(args[1:]))
        return original(args, **kwargs)

    monkeypatch.setattr(git_tool_impls, "_run_git_bytes", counted)
    return calls


def _commit_paths(root, message, *paths):
    git(root, "add", "--", *paths)
    git(root, "commit", "-qm", message, "--", *paths)
    return git(root, "rev-parse", "HEAD")


def test_committed_capture_lists_each_pinned_directory_once(source_case, monkeypatch):
    c = source_case
    for index in range(8):
        path = c.root / "pkg" / "deep" / f"m{index}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"value = {index}\n")
    head = _commit_paths(c.root, "package", "pkg")
    calls = _git_calls(monkeypatch)
    manifest = c.capture(
        c.committed_target.model_copy(
            update={"base_commit": c.head, "head_commit": head}
        )
    )
    assert len([doc for doc in manifest.initial_sources if doc.side == "head"]) == 8
    # Base lists "." once; head lists ".", "pkg" and "pkg/deep" once each.
    assert len([call for call in calls if call[0] == "ls-tree"]) == 4
    assert len([call for call in calls if call[:2] == ("rev-parse", "--verify")]) == 2


def test_oversized_change_set_refuses_before_reading_entries(
    source_case, monkeypatch, tmp_path
):
    from tldw_chatbook.Agents.code_review_models import ReviewLimits

    c = source_case
    for index in range(4):
        (c.root / f"wide{index}.py").write_text("wide\n")
    head = _commit_paths(c.root, "wide", *(f"wide{index}.py" for index in range(4)))
    (tmp_path / "small").mkdir()
    manager = ConsoleScratchSpaceManager(temp_parent=tmp_path / "small")
    store = ReviewStore(manager, "small", limits=ReviewLimits(review_max_files=5))
    calls = _git_calls(monkeypatch)
    context = store.create(target_digest="0" * 64)
    with pytest.raises(ReviewRefused, match="storage_quota_exceeded"):
        capture_review_source(
            c.committed_target.model_copy(
                update={"base_commit": c.head, "head_commit": head}
            ),
            c.authority,
            store,
            context.review_id,
        )
    assert not [call for call in calls if call[0] == "ls-tree"]
    store.close()
    assert manager.dispose()


def test_capture_never_runs_the_source_guard_under_the_store_lock(source_case):
    """The Console guard may hop to the UI thread, which takes this lock on Stop."""
    import threading

    c = source_case
    blocked = []

    def ui_thread_takes_store_lock():
        if c.store._tab.lock.acquire(timeout=1):
            c.store._tab.lock.release()
        else:
            blocked.append(True)

    def guard(write):
        thread = threading.Thread(target=ui_thread_takes_store_lock)
        thread.start()
        thread.join()
        return not write

    authority = replace(
        c.authority, admitted=replace(c.authority.admitted, guard=guard)
    )
    context = c.store.create(target_digest="0" * 64)
    capture_review_source(c.working_tree_target, authority, c.store, context.review_id)
    assert not blocked


def test_exclusion_resolving_outside_the_root_is_ignored(source_case, tmp_path):
    c = source_case
    admitted = replace(
        c.authority.admitted,
        exclusions_provider=lambda: (tmp_path / "outside", c.root / "secret.txt"),
    )
    authority = replace(c.authority, admitted=admitted)
    context = c.store.create(target_digest="0" * 64)
    manifest = capture_review_source(
        c.working_tree_target.model_copy(update={"context_paths": ("secret.txt",)}),
        authority,
        c.store,
        context.review_id,
    )
    assert "secret.txt" not in {doc.relative_path for doc in manifest.initial_sources}
    assert "secret.txt" not in repr(manifest)


def test_unreadable_changed_files_become_coverage_gaps(source_case):
    c = source_case
    (c.root / "image.bin").write_bytes(b"\x89PNG\x00\x01")
    (c.root / "alias").symlink_to("helpers.py")
    (c.root / "secret.txt").write_text("ROTATED\n")
    (c.root / "helpers.py").write_text("new helper\n")
    head = _commit_paths(
        c.root, "mixed", "image.bin", "alias", "secret.txt", "helpers.py"
    )
    manifest = c.capture(
        c.committed_target.model_copy(
            update={"base_commit": c.head, "head_commit": head}
        )
    )
    captured = {(doc.side, doc.relative_path) for doc in manifest.initial_sources}
    assert {("base", "helpers.py"), ("head", "helpers.py")} <= captured
    assert not {path for _, path in captured} & {"image.bin", "alias"}
    reasons = [gap.reason for gap in manifest.gaps]
    assert "binary_source: image.bin" in reasons
    assert "unsupported_source_type: alias" in reasons
    assert "excluded_source: 1 changed file(s) withheld" in reasons
    assert "secret.txt" not in repr(manifest)


def test_unstaged_binary_change_is_a_gap_not_a_refusal(source_case):
    c = source_case
    (c.root / "data.bin").write_bytes(b"\x00first")
    _commit_paths(c.root, "data", "data.bin")
    (c.root / "data.bin").write_bytes(b"\x00second")
    manifest = c.capture()
    assert "main.py" in {doc.relative_path for doc in manifest.initial_sources}
    assert [gap.reason for gap in manifest.gaps].count("binary_source: data.bin") == 1


def test_changed_submodule_is_an_unsupported_type_gap(source_case):
    c = source_case
    git(c.root, "update-index", "--add", "--cacheinfo", f"160000,{c.base},vendor/lib")
    git(c.root, "commit", "-qm", "submodule")
    head = git(c.root, "rev-parse", "HEAD")
    manifest = c.capture(
        c.committed_target.model_copy(
            update={"base_commit": c.head, "head_commit": head}
        )
    )
    assert ("head", "vendor/lib") not in manifest.absent_sources
    assert "unsupported_source_type: vendor/lib" in [g.reason for g in manifest.gaps]
    assert "main.py" in {doc.relative_path for doc in manifest.initial_sources}


def test_oversized_changed_file_is_a_gap_but_a_selected_file_refuses(
    source_case, monkeypatch
):
    from tldw_chatbook.Agents import code_review_source as module

    c = source_case
    (c.root / "big.py").write_text("x" * 200 + "\n")
    (c.root / "small.py").write_text("small\n")
    head = _commit_paths(c.root, "sizes", "big.py", "small.py")
    monkeypatch.setattr(module, "MAX_READ_FILE_BYTES", 100)
    manifest = c.capture(
        c.committed_target.model_copy(
            update={"base_commit": c.head, "head_commit": head}
        )
    )
    assert "source_truncated: big.py" in [gap.reason for gap in manifest.gaps]
    assert ("head", "small.py") in {
        (doc.side, doc.relative_path) for doc in manifest.initial_sources
    }
    (c.root / "large-untracked.py").write_text("y" * 200 + "\n")
    with pytest.raises(ReviewRefused, match="source_truncated"):
        c.capture(
            c.working_tree_target.model_copy(
                update={"untracked_paths": ("large-untracked.py",)}
            )
        )


def _binary_changes(c, count):
    names = [f"blob{index:02}.bin" for index in range(count)]
    for name in names:
        (c.root / name).write_bytes(b"\x00first")
    _commit_paths(c.root, "binaries", *names)
    for name in names:
        (c.root / name).write_bytes(b"\x00second")
    return names


def test_first_scope_listing_reports_bounded_capture_gaps(source_case):
    c = source_case
    _binary_changes(c, 10)
    (c.root / "secret.txt").write_text("ROTATED\n")
    manifest = c.capture()
    query = ListSourceQuery(op="list", relative_dir=".", side="worktree")
    with c.bound(manifest):
        first = query_review_source(manifest.review_id, query)
        again = query_review_source(manifest.review_id, query)
    reasons = [gap.reason for gap in first.gaps]
    assert reasons[0] == "excluded_source: 1 changed file(s) withheld"
    assert reasons[1:8] == [f"binary_source: blob{index:02}.bin" for index in range(7)]
    assert reasons[8] == "capture_gaps_omitted: 3 more"
    assert [gap.reason for gap in again.gaps] == reasons
    assert "secret.txt" not in first.model_dump_json()


def test_non_utf8_changed_file_is_a_gap_but_a_selected_file_refuses(source_case):
    c = source_case
    (c.root / "latin1.txt").write_bytes(b"caf\xe9\n")
    (c.root / "plain.py").write_text("plain\n")
    head = _commit_paths(c.root, "encodings", "latin1.txt", "plain.py")
    manifest = c.capture(
        c.committed_target.model_copy(
            update={"base_commit": c.head, "head_commit": head}
        )
    )
    assert "unsupported_source_encoding: latin1.txt" in [
        g.reason for g in manifest.gaps
    ]
    assert ("head", "plain.py") in {
        (doc.side, doc.relative_path) for doc in manifest.initial_sources
    }
    (c.root / "selected.txt").write_bytes(b"caf\xe9\n")
    with pytest.raises(ReviewRefused, match="unsupported_source_encoding"):
        c.capture(
            c.working_tree_target.model_copy(
                update={"untracked_paths": ("selected.txt",)}
            )
        )


def test_working_tree_capture_reads_the_index_once_per_pass(source_case, monkeypatch):
    c = source_case
    names = [f"staged{index}.py" for index in range(6)]
    for name in names:
        (c.root / name).write_text("first\n")
    _commit_paths(c.root, "staged files", *names)
    for name in names:
        (c.root / name).write_text("staged change\n")
    git(c.root, "add", "--", *names)
    calls = _git_calls(monkeypatch)
    manifest = c.capture()
    staged = {
        doc.relative_path for doc in manifest.initial_sources if doc.side == "staged"
    }
    assert set(names) <= staged
    # One index read for capture and one fresh read for the drift re-check,
    # covering the changed and the untracked paths alike.
    listings = [call for call in calls if call[0] == "ls-files"]
    assert len(listings) == 2
    assert all(call[1] == "--stage" for call in listings)


def test_index_change_between_capture_passes_is_still_drift(source_case, monkeypatch):
    from tldw_chatbook.Agents import code_review_source as module

    c = source_case
    original = module._read_mutable
    restaged = False

    def restage_after_read(root, exclusions, path):
        nonlocal restaged
        data = original(root, exclusions, path)
        if path == "main.py" and not restaged:
            (c.root / "main.py").write_text("restaged during capture\n")
            git(c.root, "add", "main.py")
            (c.root / "main.py").write_text(data.decode())
            restaged = True
        return data

    monkeypatch.setattr(module, "_read_mutable", restage_after_read)
    context = c.store.create(target_digest="0" * 64)
    with pytest.raises(ReviewRefused, match="source_drift"):
        capture_review_source(
            c.working_tree_target, c.authority, c.store, context.review_id
        )
    assert restaged


_NFD_NAIVE = unicodedata.normalize("NFD", "na\u00efve.py")


@pytest.mark.parametrize("variant", ["MAIN.PY", "Main.py", "NA\u00cfVE.py", _NFD_NAIVE])
def test_variant_spelling_of_a_tracked_file_is_not_an_untracked_selection(
    source_case, variant
):
    """A case/normalization-insensitive filesystem opens the tracked file.

    Only the exact stored spelling is listed, so the variant refuses; where the
    filesystem does not alias it, the variant is simply a missing file.
    """
    c = source_case
    (c.root / "na\u00efve.py").write_text("tracked\n")
    _commit_paths(c.root, "non-ascii", "na\u00efve.py")
    aliased = (c.root / variant).exists()
    with pytest.raises(ReviewRefused) as refused:
        c.capture(
            c.working_tree_target.model_copy(update={"untracked_paths": (variant,)})
        )
    assert refused.value.reason == (
        "invalid_untracked_selection" if aliased else "source_unavailable"
    )


def test_distinct_case_spelling_is_accepted_on_a_case_sensitive_filesystem(
    source_case, monkeypatch
):
    """A real MAIN.PY beside tracked main.py is an untracked selection."""
    from tldw_chatbook.Agents import code_review_source as module

    c = source_case
    if (c.root / "MAIN.PY").exists():
        # Case-insensitive volume: simulate the listing a case-sensitive one
        # gives when both spellings are stored.
        listed = module._listed_names
        monkeypatch.setattr(
            module,
            "_listed_names",
            lambda directory: (
                listed(directory) | {"MAIN.PY"}
                if directory == c.root
                else listed(directory)
            ),
        )
    else:
        (c.root / "MAIN.PY").write_text("distinct untracked file\n")
    manifest = c.capture(
        c.working_tree_target.model_copy(update={"untracked_paths": ("MAIN.PY",)})
    )
    assert ("worktree", "MAIN.PY") in {
        (doc.side, doc.relative_path) for doc in manifest.initial_sources
    }


def test_nfd_spelling_of_an_excluded_non_ascii_file_is_refused(source_case, tmp_path):
    """An NFD spelling escapes the (unnormalized) exclusion match; the listing
    check keeps it from reading the excluded NFC file on APFS."""
    c = source_case
    excluded = c.root / "na\u00efve.txt"
    excluded.write_text("private\n")
    authority = replace(
        c.authority,
        admitted=replace(c.authority.admitted, exclusions_provider=lambda: (excluded,)),
    )
    variant = unicodedata.normalize("NFD", "na\u00efve.txt")
    context = c.store.create(target_digest="0" * 64)
    with pytest.raises(ReviewRefused):
        capture_review_source(
            c.working_tree_target.model_copy(update={"untracked_paths": (variant,)}),
            authority,
            c.store,
            context.review_id,
        )
    context = c.store.create(target_digest="0" * 64)
    manifest = capture_review_source(
        c.working_tree_target.model_copy(update={"context_paths": (variant,)}),
        authority,
        c.store,
        context.review_id,
    )
    assert variant not in {doc.relative_path for doc in manifest.initial_sources}
    assert "optional_context_unavailable" in [gap.reason for gap in manifest.gaps]


@pytest.mark.parametrize("variant", ["SECRET.TXT", "Secret.Txt"])
def test_case_variant_of_an_excluded_file_never_reaches_a_reviewer(
    source_case, variant
):
    """Exclusions casefold (local_tool_impls._is_relative_sensitive_path)."""
    c = source_case
    with pytest.raises(ReviewRefused, match="source_path_unavailable"):
        c.capture(
            c.working_tree_target.model_copy(update={"untracked_paths": (variant,)})
        )
    manifest = c.capture(
        c.working_tree_target.model_copy(update={"context_paths": (variant,)})
    )
    assert "optional_context_unavailable" in [gap.reason for gap in manifest.gaps]
    assert variant.casefold() not in repr(manifest).casefold()


@pytest.mark.parametrize("variant", [".NETRC", ".Git-Credentials"])
def test_case_variant_of_a_sensitive_name_is_refused(source_case, variant):
    """The name rule is casefolded (Utils/sensitive_paths._SENSITIVE_FILE_NAMES)."""
    c = source_case
    (c.root / variant.lower()).write_text("machine example login me password pw\n")
    with pytest.raises(ReviewRefused, match="source_path_unavailable"):
        c.capture(
            c.working_tree_target.model_copy(update={"untracked_paths": (variant,)})
        )


def test_context_path_respelled_before_the_drift_check_refuses(
    source_case, monkeypatch
):
    """A case-only rename keeps the bytes; only a fresh listing sees it."""
    from tldw_chatbook.Agents import code_review_source as module

    c = source_case
    original = module._read_mutable
    renamed = False

    def respell_after_read(root, exclusions, path):
        nonlocal renamed
        data = original(root, exclusions, path)
        if path == "helpers.py" and not renamed:
            (c.root / "helpers.py").rename(c.root / "helpers.tmp")
            (c.root / "helpers.tmp").rename(c.root / "Helpers.py")
            renamed = True
        return data

    monkeypatch.setattr(module, "_read_mutable", respell_after_read)
    context = c.store.create(target_digest="0" * 64)
    with pytest.raises(ReviewRefused, match="source_drift"):
        capture_review_source(
            c.working_tree_target, c.authority, c.store, context.review_id
        )
    assert renamed


def test_context_path_dropped_to_a_gap_is_not_rechecked_for_spelling(
    source_case, monkeypatch
):
    """A gap is never read, so its respelling is not drift (content is ignored too)."""
    from tldw_chatbook.Agents import code_review_source as module

    c = source_case
    (c.root / "notes.py").write_text("notes\n")
    original = module._read_mutable

    def respell_then_fail(root, exclusions, path):
        if path == "notes.py":
            (c.root / "notes.py").rename(c.root / "notes.tmp")
            (c.root / "notes.tmp").rename(c.root / "Notes.py")
            raise module.LocalToolError("unreadable")
        return original(root, exclusions, path)

    monkeypatch.setattr(module, "_read_mutable", respell_then_fail)
    manifest = c.capture(
        c.working_tree_target.model_copy(update={"context_paths": ("notes.py",)})
    )
    assert "optional_context_unavailable" in [gap.reason for gap in manifest.gaps]


@pytest.mark.skipif(
    (hasattr(os, "geteuid") and os.geteuid() == 0) or os.name == "nt",
    reason="root can list any directory; Windows ignores the POSIX mode bits",
)
def test_unlistable_directory_is_unavailable_not_a_variant_spelling(source_case):
    c = source_case
    locked = c.root / "locked"
    locked.mkdir()
    (locked / "file.py").write_text("readable through a search-only directory\n")
    locked.chmod(0o100)
    try:
        with pytest.raises(ReviewRefused) as refused:
            c.capture(
                c.working_tree_target.model_copy(
                    update={"untracked_paths": ("locked/file.py",)}
                )
            )
    finally:
        locked.chmod(0o700)
    assert refused.value.reason == "source_unavailable"


def test_patch_preimage_variant_spelling_is_refused(source_case):
    c = source_case
    current = c.file.read_text().strip()
    patch = (
        f"--- a/MAIN.PY\n+++ b/MAIN.PY\n@@ -1 +1 @@\n-{current}\n+patched\n"
    ).encode()
    authority = replace(c.authority, resolve_patch=lambda ref: patch)
    context = c.store.create(target_digest="0" * 64)
    with pytest.raises(ReviewRefused, match="patch_preimage_mismatch"):
        capture_review_source(
            PatchTarget(kind="patch", binding_ref="binding", patch_ref="accepted"),
            authority,
            c.store,
            context.review_id,
        )


def test_listings_are_memoized_per_pass_and_fresh_for_drift(source_case, monkeypatch):
    from collections import Counter
    from pathlib import Path

    from tldw_chatbook.Agents import code_review_source as module

    c = source_case
    (c.root / "pkg").mkdir()
    for name in ("a.py", "b.py"):
        (c.root / "pkg" / name).write_text(f"{name}\n")
    calls = Counter()
    original = module._listed_names

    def counted(directory):
        calls[Path(directory)] += 1
        return original(directory)

    monkeypatch.setattr(module, "_listed_names", counted)
    c.capture(
        c.working_tree_target.model_copy(
            update={"untracked_paths": ("pkg/a.py", "pkg/b.py")}
        )
    )
    root = c.root.resolve()
    # One listing per directory for capture, and a fresh one for drift.
    assert calls[root] == 2
    assert calls[root / "pkg"] == 2
