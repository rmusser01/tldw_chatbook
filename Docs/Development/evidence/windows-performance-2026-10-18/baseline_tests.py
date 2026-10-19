"""Run focused compatibility failures with original HEAD UI modules in memory."""

import importlib.util
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Tests.windows_private_fixture_runner import user_fixture_default_owner  # noqa: E402


class OriginalSources:
    def pytest_sessionstart(self):
        for relative in (
            "tldw_chatbook/UI/Library_Modules/library_notes_sync_controller.py",
            "tldw_chatbook/UI/Screens/library_screen.py",
            "tldw_chatbook/UI/Screens/settings_screen.py",
        ):
            source = subprocess.check_output(
                [
                    "git",
                    "-c",
                    f"safe.directory={ROOT.as_posix()}",
                    "show",
                    f"HEAD:{relative}",
                ],
                cwd=ROOT,
            )
            name = relative[:-3].replace("/", ".")
            spec = importlib.util.spec_from_file_location(name, ROOT / relative)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            exec(compile(source, str(ROOT / relative), "exec"), module.__dict__)


if __name__ == "__main__":
    with user_fixture_default_owner():
        import pytest

        raise SystemExit(pytest.main(sys.argv[1:], plugins=[OriginalSources()]))
