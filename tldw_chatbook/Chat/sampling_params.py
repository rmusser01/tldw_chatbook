"""Moved to ``tldw_chatbook.Provider_Contract.sampling_params`` (TASK-36944.1).

This name is an alias of the moved module, not a copy: imports, ``isinstance``
checks and test patches through either path reach the same objects.
"""

import sys

import tldw_chatbook.Provider_Contract.sampling_params as _moved

sys.modules[__name__] = _moved
