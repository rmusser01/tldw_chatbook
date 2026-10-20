"""Pin original presentation code in pytest, including private-profile children."""

from pathlib import Path
import subprocess
import sys
import runpy

ROOT = Path(__file__).resolve().parents[4]
LOADER = Path(__file__).with_name("presentation_baseline.py").resolve()
RECEIPTS = Path(__file__).with_name("presentation-baseline-receipts.jsonl").resolve()
if "--return-matcher" in sys.argv:
    sys.argv.remove("--return-matcher")
    LOADER = Path(__file__).with_name("return_matcher_baseline.py").resolve()
    RECEIPTS = (
        Path(__file__).with_name("return-matcher-baseline-receipts.jsonl").resolve()
    )
    sys.path.insert(0, str(LOADER.parent))
if "--control-bar" in sys.argv:
    sys.argv.remove("--control-bar")
    LOADER = Path(__file__).with_name("control_bar_baseline.py").resolve()
    RECEIPTS = Path(__file__).with_name("control-bar-baseline-receipts.jsonl").resolve()
    sys.path.insert(0, str(LOADER.parent))
sys.path.insert(0, str(ROOT))
PRELUDE = f"""
import runpy,sys,hashlib,json,os
sys.path.insert(0, {str(ROOT)!r})
sys.path.insert(0, {str(LOADER.parent)!r})
definition = runpy.run_path({str(LOADER)!r})
original_load = definition['OriginalLoader'].exec_module
def audited_load(self, module):
 original_load(self, module)
 with open({str(RECEIPTS)!r}, 'a', encoding='utf-8') as receipt:
  receipt.write(json.dumps({{'pid': os.getpid(), 'module': module.__name__, 'revision': definition['BASE_REVISION'], 'sha256': hashlib.sha256(self.source).hexdigest()}})+'\\n')
definition['OriginalLoader'].exec_module = audited_load
"""
exec(PRELUDE)
OriginalPopen = subprocess.Popen


class BaselineChildren(OriginalPopen):
    def __init__(self, args, *positional, **kwargs):
        if isinstance(args, (tuple, list)) and list(args[1:3]) == ["-m", "pytest"]:
            args = [
                args[0],
                "-c",
                PRELUDE + "\nrunpy.run_module('pytest',run_name='__main__')",
                *args[3:],
            ]
        super().__init__(args, *positional, **kwargs)


subprocess.Popen = BaselineChildren
if LOADER.name in {"return_matcher_baseline.py", "control_bar_baseline.py"}:
    runpy.run_path(str(LOADER.with_name("matching_tests.py")), run_name="__main__")
from Tests.windows_private_fixture_runner import main  # noqa: E402

raise SystemExit(main())
