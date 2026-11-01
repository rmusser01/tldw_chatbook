"""Pin baseline storage in targeted pytest runs and their Python -c children."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
LOADER = Path(__file__).resolve().with_name("storage_baseline.py")
RECEIPTS = (
    Path(__file__).resolve().with_name("storage-lifetimes-baseline-receipts.jsonl")
)
sys.path.insert(0, str(ROOT))

PRELUDE = f"""
import runpy as _baseline_runpy,sys as _baseline_sys
assert 'tldw_chatbook.Backup_Recovery.storage_admission' not in _baseline_sys.modules
_baseline_definition = _baseline_runpy.run_path({str(LOADER)!r})
_baseline_load = _baseline_definition['OriginalLoader'].exec_module
def _baseline_audited_load(self,module):
 _baseline_load(self,module)
 if module.__name__.endswith('.storage_admission'):
  assert 0.01 in module._Acquisition.initializing.__wrapped__.__code__.co_consts
 import hashlib as _baseline_hashlib,json as _baseline_json,os as _baseline_os
 with open({str(RECEIPTS)!r},'a',encoding='utf-8') as receipt:
  receipt.write(_baseline_json.dumps({{'pid':_baseline_os.getpid(),'module':module.__name__,'revision':'579de70851ca4c655947f9962671be774296c80f','sha256':_baseline_hashlib.sha256(self.source).hexdigest()}})+'\\n')
_baseline_definition['OriginalLoader'].exec_module = _baseline_audited_load
"""
exec(PRELUDE)

OriginalPopen = subprocess.Popen


class BaselineChildren(OriginalPopen):
    def __init__(self, args, *positional, **kwargs):
        if isinstance(args, (list, tuple)) and "-c" in args:
            index = args.index("-c") + 1
            if index < len(args) and "tldw_chatbook" in args[index]:
                args = list(args)
                args[index] = PRELUDE + "\n" + args[index]
        super().__init__(args, *positional, **kwargs)


subprocess.Popen = BaselineChildren
from Tests.windows_private_fixture_runner import main  # noqa: E402

raise SystemExit(main())
