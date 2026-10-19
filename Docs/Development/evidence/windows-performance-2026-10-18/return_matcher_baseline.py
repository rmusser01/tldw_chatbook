"""Restore only the exact pre-subject-compound CSS matcher for timing controls."""

import presentation_baseline as original

BASE_REVISION = "48339e4d6d2fa82b561e6fa1651f8e82928177b2"
original.BASE_REVISION = BASE_REVISION
original.MODULES = {"tldw_chatbook.Utils.textual_css_fastpath"}
OriginalLoader = original.OriginalLoader
