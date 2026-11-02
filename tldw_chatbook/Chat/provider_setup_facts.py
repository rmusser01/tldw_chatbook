"""Moved to ``tldw_chatbook.Provider_Contract.provider_setup_facts`` (TASK-36944.1).

This name is an alias of the moved module, not a copy: imports, ``isinstance``
checks and test patches through either path reach the same objects.
"""

import sys

import tldw_chatbook.Provider_Contract.provider_setup_facts as _moved

sys.modules[__name__] = _moved
