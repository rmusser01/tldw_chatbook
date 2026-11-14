"""Compatibility alias: splash effects now live in ``tldw_chatbook.UI.Splash_Screens``.

Every ``tldw_chatbook.Utils.Splash_Screens[.x.y]`` import resolves to the very
same module object as its ``tldw_chatbook.UI.Splash_Screens`` twin (no second
copy, so the effect registry stays single). Importing this package costs
nothing at startup; the alias only acts when an old path is imported.
"""

import importlib
import importlib.abc
import importlib.util
import sys

_OLD = __name__
_NEW = "tldw_chatbook.UI.Splash_Screens"


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target):
        self._target = target
        self._real_spec = None

    def create_module(self, spec):
        module = importlib.import_module(self._target)
        self._real_spec = module.__spec__
        return module

    def exec_module(self, module):
        # Already executed under its real name. The import system just stamped the
        # alias spec onto the shared module; put the real one back so the UI
        # module keeps its identity and importlib.reload still works.
        module.__spec__ = self._real_spec


# Dated exception (TASK-36953.2, see Docs/Architecture/layer-map.md): this finder
# reaches tldw_chatbook.UI.Splash_Screens.* by a computed name, which the AST
# layer guard cannot see. It goes when this shim is retired.
class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_OLD + "."):
            return None
        real = _NEW + fullname[len(_OLD):]
        if importlib.util.find_spec(real) is None:
            return None
        return importlib.util.spec_from_loader(fullname, _AliasLoader(real))


if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

# The package itself becomes the real one. This is the one Utils -> UI edge, kept
# only for old import paths (user custom cards may use them); it has an explicit
# row in Tests/Architecture/layer_direction_allowlist.tsv and goes with the shim.
sys.modules[__name__] = importlib.import_module("tldw_chatbook.UI.Splash_Screens")
