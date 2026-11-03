"""Compatibility alias for the installed core recovery adapter owner."""

import sys

from tldw_chatbook.Backup_Recovery import core_adapters as _implementation

sys.modules[__name__] = _implementation
