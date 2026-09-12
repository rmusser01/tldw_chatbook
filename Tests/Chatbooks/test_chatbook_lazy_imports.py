"""Import-closure contracts for first-use Chatbook archive engines."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CREATOR_MODULE = "tldw_chatbook.Chatbooks.chatbook_creator"
IMPORTER_MODULE = "tldw_chatbook.Chatbooks.chatbook_importer"
ERROR_MODULE = "tldw_chatbook.Chatbooks.error_handler"


def _run_isolated(tmp_path: Path, source: str) -> subprocess.CompletedProcess[str]:
    data_home = tmp_path / "data"
    config_home = tmp_path / "config"
    home = tmp_path / "home"
    for path in (data_home, config_home, home):
        path.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "HOME": str(home),
        "PYTHONPATH": str(REPO_ROOT),
        "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring",
        "TLDW_TEST_MODE": "1",
        "TLDW_CONFIG_PATH": str(config_home / "config.toml"),
        "USERPROFILE": str(home),
        "XDG_CONFIG_HOME": str(config_home),
        "XDG_DATA_HOME": str(data_home),
    }
    env.pop("PYTEST_CURRENT_TEST", None)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )


def _assert_succeeded(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, (
        f"isolated Chatbook probe failed (rc={result.returncode})\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )


def test_service_exports_do_not_import_archive_engines(tmp_path: Path) -> None:
    result = _run_isolated(
        tmp_path,
        f"""
        import sys

        from tldw_chatbook.Chatbooks import LocalChatbookService, ServerChatbookService

        assert LocalChatbookService.__name__ == "LocalChatbookService"
        assert ServerChatbookService.__name__ == "ServerChatbookService"
        assert {CREATOR_MODULE!r} not in sys.modules
        assert {IMPORTER_MODULE!r} not in sys.modules
        assert {ERROR_MODULE!r} not in sys.modules
        """,
    )

    _assert_succeeded(result)


def test_lazy_public_exports_keep_class_identity(tmp_path: Path) -> None:
    result = _run_isolated(
        tmp_path,
        """
        import tldw_chatbook.Chatbooks as chatbooks
        from tldw_chatbook.Chatbooks.chatbook_creator import ChatbookCreator as DirectCreator
        from tldw_chatbook.Chatbooks.chatbook_importer import ChatbookImporter as DirectImporter
        from tldw_chatbook.Chatbooks.chatbook_models import (
            Chatbook as DirectChatbook,
            ChatbookContent as DirectChatbookContent,
            ChatbookManifest as DirectChatbookManifest,
        )
        from tldw_chatbook.Chatbooks.error_handler import (
            ChatbookError as DirectChatbookError,
            ChatbookErrorHandler as DirectChatbookErrorHandler,
            ChatbookErrorType as DirectChatbookErrorType,
        )
        from tldw_chatbook.Chatbooks.local_chatbook_service import (
            LocalChatbookService as DirectLocalChatbookService,
        )
        from tldw_chatbook.Chatbooks.server_chatbook_service import (
            ServerChatbookService as DirectServerChatbookService,
        )

        assert chatbooks.ChatbookCreator is DirectCreator
        assert chatbooks.ChatbookImporter is DirectImporter
        assert chatbooks.Chatbook is DirectChatbook
        assert chatbooks.ChatbookContent is DirectChatbookContent
        assert chatbooks.ChatbookManifest is DirectChatbookManifest
        assert chatbooks.ChatbookError is DirectChatbookError
        assert chatbooks.ChatbookErrorHandler is DirectChatbookErrorHandler
        assert chatbooks.ChatbookErrorType is DirectChatbookErrorType
        assert chatbooks.LocalChatbookService is DirectLocalChatbookService
        assert chatbooks.ServerChatbookService is DirectServerChatbookService
        assert set(chatbooks.__all__) == {
            "ChatbookCreator",
            "ChatbookImporter",
            "Chatbook",
            "ChatbookManifest",
            "ChatbookContent",
            "LocalChatbookService",
            "ServerChatbookService",
            "ChatbookError",
            "ChatbookErrorHandler",
            "ChatbookErrorType",
        }
        """,
    )

    _assert_succeeded(result)


def test_local_service_loads_archive_engines_on_first_operation(tmp_path: Path) -> None:
    result = _run_isolated(
        tmp_path,
        f"""
        import asyncio
        import os
        from pathlib import Path
        import sys

        from tldw_chatbook.Chatbooks import LocalChatbookService

        assert {CREATOR_MODULE!r} not in sys.modules
        assert {IMPORTER_MODULE!r} not in sys.modules

        root = Path(os.environ["XDG_DATA_HOME"]) / "chatbook-first-use"
        root.mkdir(parents=True, exist_ok=True)
        archive = root / "empty.chatbook.zip"
        service = LocalChatbookService({{}}, registry_path=root / "registry.json")

        exported = asyncio.run(
            service.export_chatbook(
                {{"name": "Empty", "content_selections": {{}}, "output_path": archive}}
            )
        )
        assert exported["success"] is True
        assert archive.is_file()
        assert {CREATOR_MODULE!r} in sys.modules
        assert {IMPORTER_MODULE!r} not in sys.modules

        previewed = asyncio.run(service.preview_chatbook(archive))
        assert previewed["success"] is True
        assert {IMPORTER_MODULE!r} in sys.modules

        rejected = asyncio.run(service.import_chatbook(root / "missing.txt", {{}}))
        assert rejected["success"] is False
        """,
    )

    _assert_succeeded(result)
