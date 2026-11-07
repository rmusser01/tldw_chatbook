"""Run final regressions with only the exact reviewed baseline export helper."""

import subprocess
import sys
import types

import pytest

revision = "b1108874cab28d23f660aa22c11c863466a84a8e"
path = "tldw_chatbook/DB/prompt_export.py"
source = subprocess.check_output(["git", "show", f"{revision}:{path}"], text=True)
name = "tldw_chatbook.DB.prompt_export"
baseline = types.ModuleType(name)
baseline.__file__ = f"<{revision}:{path}>"
exec(compile(source, baseline.__file__, "exec"), baseline.__dict__)
sys.modules[name] = baseline
raise SystemExit(
    pytest.main(
        [
            "Tests/Prompts_DB/test_prompt_export_mixins.py",
            "-q",
            "-p",
            "no:cacheprovider",
            "--basetemp=/private/tmp/mixins-pr3346-export-fix/baseline-final-tmp",
        ]
    )
)
