"""Replay selected failures with pre-change Console modules, without file swaps."""

import importlib.abc
import importlib.machinery
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
BASE_REVISION = "efb71994b80efc0ba578979c3f4d7b91f456eeaa"
sys.path.insert(0, str(ROOT))
MODULES = {
    "tldw_chatbook.Chat.console_runtime",
    "tldw_chatbook.UI.Console_Modules.image",
    "tldw_chatbook.UI.Console_Modules.video",
    "tldw_chatbook.UI.Console_Modules.workspace",
    "tldw_chatbook.UI.Navigation.main_navigation",
    "tldw_chatbook.UI.Navigation.nav_overflow_menu",
    "tldw_chatbook.UI.Screens.chat_screen",
    "tldw_chatbook.app_service_wiring",
}


class BaselineLoader(importlib.abc.Loader):
    def __init__(self, filename, source):
        self.filename, self.source = filename, source

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        exec(compile(self.source, self.filename, "exec"), module.__dict__)


class BaselineFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in MODULES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        relative = Path(spec.origin).relative_to(ROOT).as_posix()
        source = subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={ROOT.as_posix()}",
                "show",
                f"{BASE_REVISION}:{relative}",
            ],
            cwd=ROOT,
        )
        spec.loader = BaselineLoader(spec.origin, source)
        return spec


sys.meta_path.insert(0, BaselineFinder())
if __name__ == "__main__":
    from Tests.windows_private_fixture_runner import main

    raise SystemExit(main())
