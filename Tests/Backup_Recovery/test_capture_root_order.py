"""Candidate ordering saves native walks without granting lexical authority."""

import os

import pytest

from tldw_chatbook.Backup_Recovery import storage_admission as storage


@pytest.mark.parametrize("directory", [False, True])
def test_capture_tries_matching_root_before_unrelated_roots(
    tmp_path, monkeypatch, directory
):
    unrelated = []
    for index in range(20):
        root = tmp_path / f"unrelated-{index}"
        root.mkdir()
        unrelated.append(root)
    owner = tmp_path / "owned"
    if directory:
        owner.mkdir()
        selected = owner / "selected"
    else:
        selected = owner
    selected.write_bytes(b"native source")
    observed = []
    original = storage._contains_owned_path

    def contains(root, path):
        observed.append(root)
        return original(root, path)

    monkeypatch.setattr(storage, "_contains_owned_path", contains)
    assert storage._contains_capture_path((*unrelated, owner), selected)
    assert observed == [owner]


def test_capture_retains_nonlexical_hardlink_alias_fallback(tmp_path):
    owner, alias = tmp_path / "owned", tmp_path / "alias"
    owner.write_bytes(b"same native file")
    os.link(owner, alias)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    assert storage._contains_capture_path((unrelated, owner), alias)


@pytest.mark.parametrize("relationship", ["parent", "file_child", "prefix_sibling"])
def test_capture_candidate_order_does_not_expand_owned_scope(tmp_path, relationship):
    owner = tmp_path / "owned"
    if relationship == "prefix_sibling":
        owner.mkdir()
        selected = tmp_path / "owned-other"
        selected.write_bytes(b"outside")
    else:
        owner.write_bytes(b"declared regular file")
        selected = owner.parent if relationship == "parent" else owner / "child"
    if relationship == "file_child":
        with pytest.raises(OSError):
            storage._contains_capture_path((owner,), selected)
    else:
        assert not storage._contains_capture_path((owner,), selected)


def test_capture_lexical_match_still_runs_native_refusal(tmp_path, monkeypatch):
    selected = tmp_path / "owned"
    selected.write_bytes(b"native source")
    failure = OSError("unsafe_directory")

    def refuse(root, path):
        assert root == path == selected
        raise failure

    monkeypatch.setattr(storage, "_contains_owned_path", refuse)
    with pytest.raises(OSError) as caught:
        storage._contains_capture_path((selected,), selected)
    assert caught.value is failure


def test_capture_lexical_match_does_not_bypass_negative_native_result(
    tmp_path, monkeypatch
):
    selected = tmp_path / "owned"
    fallback = tmp_path / "alias-owner"
    observed = []

    def contains(root, path):
        observed.append(root)
        return root == fallback

    monkeypatch.setattr(storage, "_contains_owned_path", contains)
    assert storage._contains_capture_path((fallback, selected), selected)
    assert observed == [selected, fallback]
