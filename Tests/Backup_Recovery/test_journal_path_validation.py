"""Preserve native journal path validation without duplicate parsing."""

from pathlib import PurePosixPath, PureWindowsPath

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from tldw_chatbook.Backup_Recovery import journal


@pytest.mark.parametrize(
    ("flavor", "value", "valid"),
    [
        (PurePosixPath, "/", True),
        (PurePosixPath, "/profile/config.toml", True),
        (PurePosixPath, "//server/share/file", True),
        (PurePosixPath, "/profile/./file", True),
        (PurePosixPath, "/profile//file/", True),
        (PurePosixPath, "/profile/.hidden", True),
        (PurePosixPath, "/profile/name..txt", True),
        (PurePosixPath, "/profile/.../file", True),
        (PurePosixPath, "/profile/%2e%2e/file", True),
        (PurePosixPath, "/profile/\uff0e\uff0e/file", True),
        (PurePosixPath, "", False),
        (PurePosixPath, "relative/file", False),
        (PurePosixPath, "/profile/../file", False),
        (PurePosixPath, "/profile/..", False),
        (PurePosixPath, "/profile/null\0file", False),
        (PurePosixPath, "C:/profile/file", False),
        (PureWindowsPath, "C:\\profile\\file", True),
        (PureWindowsPath, "C:/profile/file", True),
        (PureWindowsPath, "C:\\profile\\.\\file", True),
        (PureWindowsPath, "\\\\server\\share\\file", True),
        (PureWindowsPath, "//server/share/file", True),
        (PureWindowsPath, "\\\\?\\C:\\profile\\file", True),
        (PureWindowsPath, "C:\\profile\\name..txt", True),
        (PureWindowsPath, "C:\\profile\\...\\file", True),
        (PureWindowsPath, "C:\\profile\\%2e%2e\\file", True),
        (PureWindowsPath, "", False),
        (PureWindowsPath, "relative\\file", False),
        (PureWindowsPath, "C:profile\\file", False),
        (PureWindowsPath, "\\profile\\file", False),
        (PureWindowsPath, "/profile/file", False),
        (PureWindowsPath, "C:\\profile\\..\\file", False),
        (PureWindowsPath, "\\\\server\\share\\..\\file", False),
        (PureWindowsPath, "\\\\?\\C:\\profile\\..\\file", False),
        (PureWindowsPath, "C:\\profile\\null\0file", False),
    ],
)
def test_journal_path_validator_preserves_platform_boundaries(
    monkeypatch, flavor, value, valid
):
    monkeypatch.setattr(journal, "Path", flavor)
    if valid:
        assert journal._Object.absolute_path(value) is value
    else:
        with pytest.raises(ValueError, match="^artifact_path_invalid$"):
            journal._Object.absolute_path(value)


@pytest.mark.parametrize("flavor", [PurePosixPath, PureWindowsPath])
@pytest.mark.parametrize("model_name", ["_Object", "_Directory", "_MobileWitness"])
def test_journal_models_construct_one_fresh_path_without_ordinary_parts(
    monkeypatch, flavor, model_name
):
    paths, parts = [], []

    class ObservedPath(flavor):
        @property
        def parts(self):
            parts.append(self)
            return super().parts

    def construct(value):
        path = ObservedPath(value)
        paths.append(path)
        return path

    monkeypatch.setattr(journal, "Path", construct)
    value = (
        "/profile/config.toml"
        if flavor is PurePosixPath
        else "C:\\profile\\config.toml"
    )
    payload = {"path": value, "device": 1, "inode": 2}
    if model_name == "_Object":
        payload.update(kind="file", sha256="a" * 64, size=0)
    elif model_name == "_MobileWitness":
        payload["mode"] = 0o100600
    model = getattr(journal, model_name)
    for _ in range(2):
        assert model.model_validate(payload).path == value
    assert len(paths) == 2
    assert paths[0] is not paths[1]
    assert not parts


@pytest.mark.parametrize(
    "value", [None, False, 1, b"/profile/file", PurePosixPath("/profile/file")]
)
def test_journal_path_field_keeps_native_strict_string_requirement(value):
    with pytest.raises(ValidationError):
        journal._MobileWitness.model_validate(
            {"path": value, "device": 1, "inode": 2, "mode": 0o100600}
        )


_PATHS = st.one_of(
    st.text(max_size=256),
    st.builds(
        lambda prefix, value: prefix + value,
        st.sampled_from(
            ["/", "//", "C:\\", "C:", "\\", "\\\\server\\share\\", "\\\\?\\C:\\"]
        ),
        st.text(max_size=256),
    ),
)


@pytest.mark.parametrize("flavor", [PurePosixPath, PureWindowsPath])
@given(value=_PATHS)
def test_journal_path_prefilter_matches_original_native_predicate(flavor, value):
    # Compatibility oracle: exact original predicate, without the prefilter.
    invalid = (
        not flavor(value).is_absolute() or ".." in flavor(value).parts or "\0" in value
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(journal, "Path", flavor)
        if invalid:
            with pytest.raises(ValueError, match="^artifact_path_invalid$"):
                journal._Object.absolute_path(value)
        else:
            assert journal._Object.absolute_path(value) is value
