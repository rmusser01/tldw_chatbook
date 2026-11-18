"""Source authority is independent of publisher/display metadata."""

import pytest


@pytest.mark.parametrize(
    "value",
    [
        "http://example.test/pkg",
        "git://example.test/pkg",
        "file:///etc",
        "/etc",
        "../pkg",
        "https://user:secret@example.test/pkg",
        "https://example.test/../pkg",
        "https://example.test/pkg?token=secret",
        "https://example.test/pkg#ref=-upload-pack=evil",
        "https://example.test/pkg#subdir=../outside",
        "https://example.test/pkg#ref=main&ref=other",
    ],
)
def test_untrusted_locator_rejected_without_exposing_input(value):
    from tldw_chatbook.Plugins.git_source import parse_source_locator

    with pytest.raises(ValueError) as caught:
        parse_source_locator(value)
    assert value not in str(caught.value)


def test_source_selectors_and_origin_are_separate():
    from tldw_chatbook.Plugins.git_source import parse_source_locator

    source = parse_source_locator(
        "github:team/plugins#ref=release/v1&subdir=tools/review"
    )
    assert source.transport == "https"
    assert source.origin == "https://github.com"
    assert source.repository == "/team/plugins.git"
    assert source.ref == "release/v1"
    assert source.subdirectory == "tools/review"
    assert source.url == "https://github.com/team/plugins.git"


def test_ssh_username_and_explicit_port_are_part_of_source():
    from tldw_chatbook.Plugins.git_source import parse_source_locator

    source = parse_source_locator("ssh://git@example.test:2222/team/plugins.git")
    assert source.transport == "ssh"
    assert source.origin == "ssh://git@example.test:2222"
    assert source.ref == "HEAD"


def test_only_explicit_local_selection_grants_host_path(tmp_path):
    from tldw_chatbook.Plugins.git_source import (
        parse_source_locator,
        select_local_source,
    )

    with pytest.raises(ValueError):
        parse_source_locator(str(tmp_path))
    source = select_local_source(tmp_path)
    assert source.transport == "local"
    assert source.repository == str(tmp_path.resolve())
    with pytest.raises(ValueError):
        source.repository = "/elsewhere"
