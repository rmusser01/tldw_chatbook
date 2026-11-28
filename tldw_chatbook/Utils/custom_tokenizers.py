"""Compatibility module alias for the canonical custom-tokenizer service."""

import sys

from tldw_chatbook.Chat import custom_tokenizers as _implementation

# One module dictionary preserves the manager, native functions and live patches.
sys.modules[__name__] = _implementation
