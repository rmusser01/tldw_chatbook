"""Execute pinned original UI bytes with matching original on-disk source."""

import hashlib
from pathlib import Path

import memory_baseline as baseline
import presentation_baseline as original

BASE_REVISION = baseline.BASE_REVISION
LOADED_SOURCES = {}
ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = ROOT / ".tmp-memory-baseline-sources"


class OriginalLoader(baseline.OriginalLoader):
    def exec_module(self, module):
        relative = Path(self.filename).relative_to(ROOT)
        target = SOURCE_ROOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != self.source:
                raise RuntimeError("Pinned baseline source differs from existing file")
        else:
            with target.open("xb") as output:
                output.write(self.source)
        self.filename = str(target)
        module.__file__ = str(target)
        module.__spec__.origin = str(target)
        super().exec_module(module)
        LOADED_SOURCES[module.__name__] = {
            "revision": BASE_REVISION,
            "sha256": hashlib.sha256(self.source).hexdigest(),
            "path": str(target),
        }


original.OriginalLoader = OriginalLoader
