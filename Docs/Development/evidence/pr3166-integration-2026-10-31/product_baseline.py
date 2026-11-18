"""Load exact base versions of PR3166 product modules without editing the tree."""

import importlib.abc
import importlib.util
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
BASE = "b1ca1b11fa44a8e0ff0a09dfbffe940072861ba4"
paths = subprocess.check_output(
    ["git", "diff", "--name-only", BASE, "--", "tldw_chatbook"], cwd=ROOT, text=True
).splitlines()
SOURCES = {}
for path in paths:
    if not path.endswith(".py"):
        continue
    result = subprocess.run(
        ["git", "show", f"{BASE}:{path}"], cwd=ROOT, capture_output=True
    )
    if result.returncode == 0:
        SOURCES[path.removesuffix(".py").replace("/", ".")] = (
            path,
            result.stdout.decode("utf-8"),
        )


class BaseSource(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in SOURCES:
            return importlib.util.spec_from_file_location(
                fullname, ROOT / SOURCES[fullname][0], loader=self
            )
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        path, source = SOURCES[module.__name__]
        exec(compile(source, str(ROOT / path), "exec"), module.__dict__)


sys.meta_path.insert(0, BaseSource())
