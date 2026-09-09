"""Selected sources extend existing read authority without granting writes/instructions."""

import pytest

from Tests.Agents.test_local_tool_provider import ALLOW, ASK
from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider


@pytest.mark.parametrize("name", ["fs_read", "fs_list"])
def test_selected_source_reads_refuse_siblings_and_escapes(tmp_path, name):
    primary, source, sibling = (tmp_path / n for n in ("primary", "source", "sibling"))
    for root in (primary, source, sibling):
        root.mkdir()
        (root / "data.txt").write_text("selected source data")
    (source / "escape").symlink_to(sibling, target_is_directory=True)
    provider = LocalToolProvider(
        workspace_root=primary, read_only_roots=(source,), resolve_state=lambda _: ALLOW
    )
    target = source / "data.txt" if name == "fs_read" else source
    result = provider.invoke("local:" + name, {"path": str(target)})
    assert (
        result.ok
        and ("selected source data" if name == "fs_read" else "data.txt")
        in result.content
    )
    assert (
        provider.path_targets("local:" + name, {"path": str(target)})[0].kind
        == "outside"
    )
    for other in (sibling, source / ".." / "sibling", source / "escape"):
        path = other / "data.txt" if name == "fs_read" else other
        assert not provider.invoke("local:" + name, {"path": str(path)}).ok
    assert not provider.invoke(
        "local:fs_write", {"path": str(source / "data.txt"), "content": "bad"}
    ).ok
    assert not provider.invoke(
        "local:fs_edit",
        {
            "path": str(source / "data.txt"),
            "old_string": "selected",
            "new_string": "bad",
        },
    ).ok
    (primary / "data.txt").write_text("invalid")
    assert provider.invoke(
        "local:fs_edit",
        {"path": "data.txt", "old_string": "invalid", "new_string": "valid"},
    ).ok
    assert (source / "data.txt").read_text() == "selected source data"
    assert (primary / "data.txt").read_text() == "valid"


def test_source_guard_is_checked_after_permission_wait(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "data").write_text("never read stale source")
    valid = True

    def approve(calls):
        nonlocal valid
        valid = False
        return {"fs_read": "approve_once"}

    provider = LocalToolProvider(
        workspace_root=tmp_path / "primary",
        read_only_roots=(source,),
        root_guard=lambda: valid,
        resolve_state=lambda _: ASK,
        approval_callback=approve,
    )
    from tldw_chatbook.Agents.local_tool_provider import LOCAL_ROOT_CHANGED_REFUSAL

    result = provider.invoke("local:fs_read", {"path": str(source / "data")})
    assert not valid
    assert not result.ok and result.error == LOCAL_ROOT_CHANGED_REFUSAL


def test_ordinary_provider_does_not_gain_source_access(tmp_path):
    primary = tmp_path / "primary"
    primary.mkdir()
    other = tmp_path / "secret"
    other.write_text("unselected")
    provider = LocalToolProvider(workspace_root=primary, resolve_state=lambda _: ALLOW)
    assert not provider.invoke("local:fs_read", {"path": str(other)}).ok
