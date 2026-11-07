"""Paired browse controls with exact source and explicit native-profile custody."""

import pathlib
import subprocess
import sys
import types

import pytest

sys.path.insert(0, str(pathlib.Path.cwd()))
from Tests import conftest as private_bootstrap

baseline = "--baseline" in sys.argv
bootstrap = "--bootstrap" in sys.argv
if baseline:
    path = "tldw_chatbook/UI/Library_Modules/library_prompt_browse_controller.py"
    source = subprocess.check_output(["git", "show", f"b1108874cab28d23f660aa22c11c863466a84a8e:{path}"], text=True)
    name = "tldw_chatbook.UI.Library_Modules.library_prompt_browse_controller"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    sys.modules[name] = module


class BootstrapProfile:
    def pytest_collection_modifyitems(self, items):
        for item in items:
            item.add_marker(pytest.mark.bootstrap_profile)


label = ("baseline" if baseline else "current") + ("-profile" if bootstrap else "")
raise SystemExit(
    pytest.main(
        [
            "Tests/UI/test_library_prompt_browse_controller.py",
            "-q",
            "-p",
            "no:cacheprovider",
            f"--basetemp=/private/tmp/mixins-pr3346-size-repair/{label}-browse-tmp",
        ],
        plugins=[BootstrapProfile()] if bootstrap else [],
    )
)
