from pathlib import Path

import pytest

from tldw_chatbook.Utils.private_paths import (
    PrivatePathError,
    PrivatePathResult,
    PrivatePathStatus,
    lexical_path,
)


@pytest.mark.parametrize(
    ("status", "verified_private", "usable"),
    [
        (PrivatePathStatus.CREATED_PRIVATE, True, True),
        (PrivatePathStatus.HARDENED_PRIVATE, True, True),
        (PrivatePathStatus.ALREADY_PRIVATE, True, True),
        (PrivatePathStatus.UNVERIFIED_PLATFORM, False, True),
        (PrivatePathStatus.UNSAFE_PARENT, False, False),
        (PrivatePathStatus.WRONG_OWNER, False, False),
        (PrivatePathStatus.LINK_OR_NON_REGULAR, False, False),
        (PrivatePathStatus.OPERATION_FAILED, False, False),
    ],
)
def test_private_path_result_classifies_posture(status, verified_private, usable):
    result = PrivatePathResult(Path("/tmp/config.toml"), status)

    assert result.verified_private is verified_private
    assert result.usable is usable


def test_private_path_error_exposes_bounded_result_without_original_exception():
    result = PrivatePathResult(
        Path("/tmp/config.toml"),
        PrivatePathStatus.UNSAFE_PARENT,
        reason="shared_writable_parent",
    )

    error = PrivatePathError(result)

    assert error.result is result
    assert "shared_writable_parent" in str(error)


def test_lexical_path_normalizes_without_resolving_symlink(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    monkeypatch.chdir(tmp_path)

    selected = lexical_path(Path("alias") / ".." / "alias" / "config.toml")

    assert selected == alias / "config.toml"
    assert selected != real / "config.toml"


def test_lexical_path_rejects_nul():
    with pytest.raises(ValueError, match="NUL"):
        lexical_path("bad\x00path")
