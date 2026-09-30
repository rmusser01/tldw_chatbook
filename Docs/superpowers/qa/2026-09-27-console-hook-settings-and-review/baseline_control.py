import importlib.abc
import importlib.util
import pathlib
import subprocess
import sys

root = pathlib.Path.cwd()
ref = "857b3dd7d0"
paths = subprocess.check_output(
    ["git", "diff", "--name-only", ref, "--", "tldw_chatbook"], text=True
).splitlines()
sources = {}
styles = {}
for relative in paths:
    result = subprocess.run(
        ["git", "show", ref + ":" + relative],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        continue
    if relative.endswith(".py"):
        sources[relative[:-3].replace("/", ".")] = (str(root / relative), result.stdout)
    if relative.endswith(".tcss"):
        styles[str((root / relative).resolve())] = result.stdout


class BaselineLoader(importlib.abc.Loader):
    def __init__(self, path, source):
        self.path, self.source = path, source

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__file__ = self.path
        exec(compile(self.source, self.path, "exec"), module.__dict__)  # noqa: S102 -- run only the pinned committed baseline


class BaselineFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in sources:
            source_path, source = sources[fullname]
            return importlib.util.spec_from_file_location(
                fullname, source_path, loader=BaselineLoader(source_path, source)
            )


original_read = pathlib.Path.read_text


def baseline_read(path, *args, **kwargs):
    return styles.get(str(path.resolve())) or original_read(path, *args, **kwargs)


pathlib.Path.read_text = baseline_read
sys.meta_path.insert(0, BaselineFinder())
sys.path.insert(0, str(root))
import pytest


class BaselineCategoryContract:
    def pytest_collection_modifyitems(self, items):
        for item in items:
            floor = getattr(item.module, "PER_CATEGORY_MIN_SETTINGS", None)
            if floor is not None:
                floor.pop(
                    "hooks", None
                )  # This category does not exist on the baseline.


raise SystemExit(pytest.main(sys.argv[1:], plugins=[BaselineCategoryContract()]))
