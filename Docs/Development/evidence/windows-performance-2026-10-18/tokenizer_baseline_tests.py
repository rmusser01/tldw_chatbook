"""Replay selected failures with pre-change Console modules, without file swaps."""

import importlib.abc
import importlib.machinery
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
BASE_REVISION = "69771c1811e734421f0bdf31ed90969f8c1f0c63"
sys.path.insert(0, str(ROOT))
MODULES = {
    "tldw_chatbook.UI.Console_Modules.console_spend_projection",
    "tldw_chatbook.UI.Console_Modules.context_spend",
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
from Tests.windows_private_fixture_runner import main  # noqa: E402 - install the source loader first.

raise SystemExit(main())
