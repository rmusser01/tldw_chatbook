"""Import-time optional dependency stubs for focused UI tests.

Import this module before application modules that probe the unrelated MLX stack.
Keeping the registration behind an import preserves contiguous import groups in
the focused test modules that need it.
"""

from __future__ import annotations

import sys
import types

sys.modules.setdefault("parakeet_mlx", types.ModuleType("parakeet_mlx"))
