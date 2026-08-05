import os
from pathlib import Path

import pytest

from tldw_chatbook.Chatbooks import database_paths


def test_chatbook_database_paths_use_canonical_runtime_getters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
        "ChaChaNotes": tmp_path / "chachanotes.db",
        "Prompts": tmp_path / "prompts.db",
        "Media": tmp_path / "media.db",
    }
    monkeypatch.setattr(
        database_paths.config,
        "get_chachanotes_db_path",
        lambda: expected["ChaChaNotes"],
    )
    monkeypatch.setattr(
        database_paths.config,
        "get_prompts_db_path",
        lambda: expected["Prompts"],
    )
    monkeypatch.setattr(
        database_paths.config,
        "get_media_db_path",
        lambda: expected["Media"],
    )

    assert database_paths.get_chatbook_database_paths() == {
        name: str(path) for name, path in expected.items()
    }


def test_private_chatbooks_directory_uses_canonical_runtime_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_data_dir = tmp_path / "runtime-data"
    user_data_dir.mkdir(mode=0o700)
    chatbooks_dir = user_data_dir / "chatbooks"
    chatbooks_dir.mkdir(mode=0o755)
    monkeypatch.setattr(
        database_paths.config,
        "get_user_data_dir",
        lambda: user_data_dir,
    )

    selected = database_paths.get_private_chatbooks_dir()

    assert selected == chatbooks_dir
    assert selected.stat().st_mode & 0o777 == 0o700


@pytest.mark.skipif(os.name != "posix", reason="POSIX export directory contract")
def test_chatbook_directory_helper_hardens_export_directory(tmp_path: Path) -> None:
    export_dir = tmp_path / "Documents" / "Chatbooks"
    export_dir.mkdir(parents=True, mode=0o755)

    previous = os.umask(0)
    try:
        selected = database_paths.secure_chatbook_directory(export_dir)
    finally:
        os.umask(previous)

    assert selected == export_dir
    assert export_dir.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize(
    "relative_path",
    [
        "tldw_chatbook/UI/Wizards/ChatbookCreationWizard.py",
        "tldw_chatbook/UI/Wizards/ChatbookImportWizard.py",
        "tldw_chatbook/UI/ChatbookExportManagementWindow.py",
        "tldw_chatbook/UI/ChatbookCreationWindow.py",
    ],
)
def test_chatbook_surfaces_do_not_embed_database_defaults(
    relative_path: str,
) -> None:
    source = (Path(__file__).parents[2] / relative_path).read_text(encoding="utf-8")

    assert "get_chatbook_database_paths" in source
    assert "chachanotes_db_path" not in source
    assert "prompts_db_path" not in source
    assert "media_db_path" not in source


@pytest.mark.parametrize(
    "relative_path",
    [
        "tldw_chatbook/UI/Wizards/ChatbookCreationWizard.py",
        "tldw_chatbook/UI/ChatbookExportManagementWindow.py",
        "tldw_chatbook/UI/Chatbooks_Window_Improved.py",
        "tldw_chatbook/UI/ChatbookCreationWindow.py",
    ],
)
def test_chatbook_export_directory_surfaces_use_private_chatbooks_dir_accessor(
    relative_path: str,
) -> None:
    """The four Chatbook windows default the export dir via the accessor.

    Regression guard for task-984: reconciles the export directory default
    across all four windows onto ``get_private_chatbooks_dir()`` instead of
    the hardcoded ``~/Documents/Chatbooks`` literal.
    """
    source = (Path(__file__).parents[2] / relative_path).read_text(encoding="utf-8")

    assert "get_private_chatbooks_dir" in source
    assert '"Documents" / "Chatbooks"' not in source
