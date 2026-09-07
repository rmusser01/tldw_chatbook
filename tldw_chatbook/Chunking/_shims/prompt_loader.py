# tldw_chatbook/Chunking/_shims/prompt_loader.py
"""Flat alias for the phase-1 prompt_loader shim (plan-documented layout).

The load-bearing module is ``_shims.Utils.prompt_loader`` -- the dotted path
the vendored engine imports (engine/strategies/rolling_summarize.py:13).
This flat re-export keeps the layout documented in the phase-A plan
importable under ``_shims.prompt_loader`` too; both names share one
implementation.
"""
from .Utils.prompt_loader import _KNOWN, load_prompt

__all__ = ["load_prompt", "_KNOWN"]
