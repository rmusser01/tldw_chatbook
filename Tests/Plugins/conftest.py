"""Real local packages, authored independently from the parser."""

import json
import shutil
from pathlib import Path

import pytest


@pytest.fixture
def native_package(tmp_path):
    count = 0

    def create(*, requires=None, extension=None):
        nonlocal count
        count += 1
        root = tmp_path / f"package-{count}"
        shutil.copytree(Path(__file__).parent / "fixtures" / "native", root)
        manifest = json.loads((root / "plugin.json").read_text())
        ext = {"version": 1} if extension is None else extension
        if requires is not None:
            ext["requires"] = requires
        manifest["extensions"]["io.github.rmusser01.chatbook"] = ext
        (root / "plugin.json").write_text(json.dumps(manifest))
        return root

    return create
