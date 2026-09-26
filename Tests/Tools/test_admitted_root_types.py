"""Phase 3a (task 15): the ``LocalRoot | RemoteRoot`` admitted-root union.

The descriptors that give remote workspace roots their own TYPE so every
laptop-disk call site fails loudly at the boundary instead of silently
reading the laptop's filesystem (spec: "Remote roots never touch the
laptop's disk"). ``LocalRoot`` wraps a plain ``Path``; ``RemoteRoot`` is a
pure-Python descriptor (alias, canonical locator, POSIX root, binding id)
that can never be mistaken for a laptop path.
"""

import dataclasses
from pathlib import Path, PurePosixPath

import pytest

from tldw_chatbook.Tools.remote_root_types import (
    AdmittedRoot,
    LocalRoot,
    RemoteRoot,
    display_uri,
    is_remote,
    local_root_path,
)


def test_local_root_is_frozen_and_normalizes_to_path() -> None:
    root = LocalRoot(Path("/tmp/workspace"))

    assert isinstance(root.path, Path)
    assert root.path == Path("/tmp/workspace")
    with pytest.raises(dataclasses.FrozenInstanceError):
        root.path = Path("/elsewhere")  # type: ignore[misc]


def test_local_root_coerces_string_paths() -> None:
    root = LocalRoot("/tmp/workspace")

    assert root.path == Path("/tmp/workspace")
    assert root == LocalRoot(Path("/tmp/workspace"))


def test_remote_root_is_frozen_and_keeps_pure_posix_root() -> None:
    root = RemoteRoot(
        alias="proj",
        canonical_locator="devbox:/srv/www",
        root=PurePosixPath("/srv/www"),
        binding_id="binding-17",
    )

    assert root.root == PurePosixPath("/srv/www")
    assert root.alias == "proj"
    assert root.canonical_locator == "devbox:/srv/www"
    assert root.binding_id == "binding-17"
    with pytest.raises(dataclasses.FrozenInstanceError):
        root.alias = "other"  # type: ignore[misc]


def test_remote_root_rejects_blank_identifiers_and_relative_roots() -> None:
    with pytest.raises(ValueError):
        RemoteRoot(alias="", canonical_locator="devbox:/srv", root="/srv", binding_id="b")
    with pytest.raises(ValueError):
        RemoteRoot(alias="proj", canonical_locator="", root="/srv", binding_id="b")
    with pytest.raises(ValueError):
        RemoteRoot(alias="proj", canonical_locator="devbox:/srv", root="srv/www", binding_id="b")
    with pytest.raises(ValueError):
        RemoteRoot(alias="proj", canonical_locator="devbox:/srv", root="/srv", binding_id="  ")


def test_remote_root_coerces_string_root_to_pure_posix() -> None:
    root = RemoteRoot(
        alias="proj", canonical_locator="devbox:/srv", root="/srv/www", binding_id="b"
    )

    assert root.root == PurePosixPath("/srv/www")


def test_is_remote_discriminates_every_root_form() -> None:
    assert is_remote(
        RemoteRoot(alias="a", canonical_locator="h:/r", root="/r", binding_id="b")
    )
    assert not is_remote(LocalRoot(Path("/tmp/w")))
    assert not is_remote(Path("/tmp/w"))


def test_display_uri_renders_local_file_scheme() -> None:
    assert display_uri(LocalRoot(Path("/tmp/workspace"))) == "file:///tmp/workspace"


def test_display_uri_renders_remote_ssh_scheme() -> None:
    root = RemoteRoot(
        alias="proj", canonical_locator="devbox/srv/www", root="/srv/www", binding_id="b"
    )

    # The context-note form Task 18 consumes: ssh:// + canonical locator.
    assert display_uri(root) == "ssh://devbox/srv/www"


@pytest.mark.parametrize(
    ("locator", "expected"),
    [
        # Real canonical locators (Task 5's locator_string) already carry
        # the scheme -- they render verbatim, never double-prefixed.
        ("ssh://devbox/srv/www", "ssh://devbox/srv/www"),
        ("ssh://me@devbox/srv/www", "ssh://me@devbox/srv/www"),
        # An explicit NON-default port survives verbatim.
        ("ssh://devbox:2222/srv/www", "ssh://devbox:2222/srv/www"),
        ("ssh://me@[::1]:2222/srv/www", "ssh://me@[::1]:2222/srv/www"),
        # The redundant default port is stripped (locator_string rule).
        ("ssh://devbox:22/srv/www", "ssh://devbox/srv/www"),
        ("ssh://[::1]:22/srv/www", "ssh://[::1]/srv/www"),
        # Hand-built colon forms: the empty-port authority normalizes
        # away instead of rendering "ssh://devbox:/srv/www".
        ("devbox:/srv/www", "ssh://devbox/srv/www"),
        ("me@devbox:/srv/www", "ssh://me@devbox/srv/www"),
        ("devbox:22/srv/www", "ssh://devbox/srv/www"),
        # Path-less locator stays path-less.
        ("ssh://devbox", "ssh://devbox"),
    ],
)
def test_display_uri_normalizes_ssh_locator_forms(locator: str, expected: str) -> None:
    root = RemoteRoot(
        alias="proj", canonical_locator=locator, root="/srv/www", binding_id="b"
    )

    assert display_uri(root) == expected


def test_local_root_path_passes_plain_paths_through_unchanged() -> None:
    plain = Path("/tmp/workspace").resolve()

    assert local_root_path(plain, site="unit-test") is plain


def test_local_root_path_unwraps_local_root_to_its_path() -> None:
    path = Path("/tmp/workspace")

    assert local_root_path(LocalRoot(path), site="unit-test") == path


def test_local_root_path_fails_loud_for_remote_roots() -> None:
    root = RemoteRoot(
        alias="proj", canonical_locator="devbox:/srv/www", root="/srv/www", binding_id="b"
    )

    with pytest.raises(TypeError) as caught:
        local_root_path(root, site="ledger hashing")

    assert "remote root reached laptop-disk path" in str(caught.value)
    assert "ledger hashing" in str(caught.value)


def test_union_alias_covers_both_descriptors() -> None:
    roots: tuple[AdmittedRoot, ...] = (
        LocalRoot(Path("/tmp/w")),
        RemoteRoot(alias="a", canonical_locator="h:/r", root="/r", binding_id="b"),
    )

    assert len(roots) == 2
