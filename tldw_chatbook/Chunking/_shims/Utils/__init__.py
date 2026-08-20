# tldw_chatbook/Chunking/_shims/Utils/__init__.py
"""Upstream ``app.core.Utils`` package-level shim (spec §5.3).

The vendored engine imports
``tldw_chatbook.Chunking._shims.Utils.prompt_loader`` unguarded at module
level (engine/strategies/rolling_summarize.py:13), so the capital-``U``
package must exist for the engine's module graph to resolve. Phase 1 ships
only prompt_loader here.
"""
