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
    # newline="" both sides: universal-newline mode would silently turn \r
    # into \n and break the exact-replacement round trip.
    (ws / "f.txt").write_text(content, newline="")
    edit_file("f.txt", needle, replacement, workspace_root=ws)
    with open(ws / "f.txt", encoding="utf-8", newline="") as fh:
        assert fh.read() == prefix + replacement + suffix


@given(path=st.text(min_size=1))
@_PROPERTY_SETTINGS
def test_workspace_confinement_never_escapes(tmp_path, path):
    ws = tmp_path / "ws"; ws.mkdir(exist_ok=True)
    try:
        resolved = resolve_workspace_path(path, ws)
    except (LocalToolError, ValueError):
        return  # refusal is fine
    assert str(resolved).startswith(str(ws.resolve()))
