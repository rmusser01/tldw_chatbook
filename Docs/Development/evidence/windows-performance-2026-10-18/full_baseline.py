"""Pin the pre-FULL-preparation modules without changing the working tree."""

import importlib.abc
import importlib.machinery
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
BASE_REVISION = "2a9cf92464982cb0d53e6452ade220e4e4884976"
MODULES = {
    "tldw_chatbook.UI.Console_Modules.config_sync",
    "tldw_chatbook.UI.Screens.chat_screen",
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
