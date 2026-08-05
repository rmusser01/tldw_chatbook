# Property-based tests for the workspace-local tool cores (fs_edit,
# path confinement), following the repo's Hypothesis style
# (Tests/Media_DB/test_media_db_properties.py).

from hypothesis import HealthCheck, given, settings, strategies as st

from tldw_chatbook.Tools.local_tool_impls import (
    LocalToolError,
    edit_file,
    resolve_workspace_path,
)

# tmp_path is function-scoped, so Hypothesis would flag it per example;
# file I/O per example also makes the default deadline flaky. Both are
# expected here — suppress rather than fight them.
_PROPERTY_SETTINGS = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


@given(
    prefix=st.text(max_size=50), needle=st.text(min_size=1, max_size=10),
    suffix=st.text(max_size=50), replacement=st.text(max_size=20),
)
@_PROPERTY_SETTINGS
def test_edit_replaces_exactly_one_occurrence(tmp_path, prefix, needle, suffix, replacement):
    ws = tmp_path / "ws"; ws.mkdir(exist_ok=True)
    content = prefix + needle + suffix
    if content.count(needle) != 1:
        return  # only unique-match inputs are in scope for this property
    try:
        original = content.encode("utf-8")
    except UnicodeEncodeError:
        return  # file content must be UTF-8 for fs_edit to apply at all
    (ws / "f.txt").write_bytes(original)
    try:
        edit_file("f.txt", needle, replacement, workspace_root=ws)
    except LocalToolError:
        # refused edit (unencodable replacement, identical strings, …) —
        # a failed edit must never mutate the file
        assert (ws / "f.txt").read_bytes() == original
        return
    # Oracle: Python's own leftmost single replace. Positional
    # reconstruction (prefix+replacement+suffix) is UNSOUND here:
    # str.count is non-overlapping, so a self-overlapping needle
    # (prefix="a", needle="aa") can count==1 while str.replace picks an
    # occurrence earlier than len(prefix).
    expected = content.replace(needle, replacement, 1)
    assert (ws / "f.txt").read_bytes() == expected.encode("utf-8")
    if content.find(needle) == len(prefix):
        # the replaced occurrence IS the constructed one — the positional
        # invariant holds as an additional check in that case
        assert expected == prefix + replacement + suffix


@given(
    prefix=st.text(max_size=50), needle=st.text(min_size=1, max_size=10),
    suffix=st.text(max_size=50), replacement=st.text(max_size=20),
)
@_PROPERTY_SETTINGS
def test_edit_replace_all_replaces_every_occurrence(tmp_path, prefix, needle, suffix, replacement):
    ws = tmp_path / "ws"; ws.mkdir(exist_ok=True)
    content = prefix + needle + suffix
    if content.count(needle) < 1:
        return  # only matching inputs are in scope for this property
    try:
        original = content.encode("utf-8")
    except UnicodeEncodeError:
        return  # file content must be UTF-8 for fs_edit to apply at all
    (ws / "f.txt").write_bytes(original)
    try:
        edit_file("f.txt", needle, replacement, workspace_root=ws, replace_all=True)
    except LocalToolError:
        assert (ws / "f.txt").read_bytes() == original  # failed edit: no mutation
        return
    expected = content.replace(needle, replacement)
    assert (ws / "f.txt").read_bytes() == expected.encode("utf-8")


@given(path=st.text(min_size=1))
@_PROPERTY_SETTINGS
def test_workspace_confinement_never_escapes(tmp_path, path):
    ws = tmp_path / "ws"; ws.mkdir(exist_ok=True)
    try:
        resolved = resolve_workspace_path(path, ws)
    except (LocalToolError, ValueError):
        return  # refusal is fine
    assert resolved.is_relative_to(ws.resolve())
