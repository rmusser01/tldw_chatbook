"""Load the exact pre-presentation modules without swapping the working tree."""

import importlib.abc
import importlib.machinery
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
BASE_REVISION = "ac0c13581a54327b1fd38a3b6ddcce45cdb416e0"
MODULES = {
    "tldw_chatbook.Utils.textual_css_fastpath",
    "tldw_chatbook.UI.Screens.chat_screen",
    "tldw_chatbook.UI.Console_Modules.workspace",
}


class OriginalLoader(importlib.abc.Loader):
    def __init__(self, filename, source):
        self.filename, self.source = filename, source

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        exec(compile(self.source, self.filename, "exec"), module.__dict__)


class OriginalFinder(importlib.abc.MetaPathFinder):
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
        spec.loader = OriginalLoader(spec.origin, source)
        return spec


sys.meta_path.insert(0, OriginalFinder())
