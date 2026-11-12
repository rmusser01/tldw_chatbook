"""
tldw_chatbook - A Textual TUI for chatting with LLMs

A sophisticated Terminal User Interface (TUI) application built with the Textual
framework for interacting with various Large Language Model APIs. Provides a
complete ecosystem for AI-powered interactions including conversation management,
character/persona chat, notes with bidirectional file sync, media ingestion,
and advanced RAG (Retrieval-Augmented Generation) capabilities.
"""

# Disable progress bars early to prevent interference with TUI
# This must be done before any libraries that use progress bars are imported
import os
import sys

from .Utils.tiktoken_runtime import (
    arm_bundled_tiktoken_runtime as _arm_bundled_tiktoken_runtime,
)

# Phase 0c: register the armer instead of importing tiktoken eagerly. The
# bundled offline assets are still selected before tiktoken's first use —
# see ``arm_bundled_tiktoken_runtime`` — but importing this package (which
# every ``tldw_chatbook`` import runs, the stdlib-only pinned workspace
# worker's included) no longer pulls the third-party tiktoken modules in.
_arm_bundled_tiktoken_runtime()

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# task-2119 (security): Loguru's `diagnose` option defaults to True and, on
# any record logged via `logger.opt(exception=True)`, dumps every stack
# frame's LOCAL VARIABLES alongside the traceback -- not just the source
# line. Provider request handlers in LLM_Calls/LLM_API_Calls.py hold the raw
# `Authorization`/`x-api-key` header dict and the resolved API key in scope
# across ~30 exception handlers; with diagnose left at its default, an
# ordinary transient error (timeout, 429, connection reset) prints the live
# credential in cleartext to whatever sink is active. Live-confirmed via a
# genuine Moonshot 429 on 2026-08-03: a verification script that imported
# this package's adapters directly (never running the real app's own
# `Logging_Config.configure_application_logging`) hit loguru's auto-init
# default sink, which was still diagnose=True.
#
# The real app's own sink, and `Metrics/logger_config.py`'s legacy sink,
# both now also pass `diagnose=False` explicitly where they call
# `logger.add(...)` -- that per-sink kwarg is the actual, order-independent
# fix. This block additionally neutralizes loguru's own auto-init default
# sink (created the instant anything, anywhere, does
# `from loguru import logger`, before any application code has run) by
# replacing it outright, and sets the library-wide default for any *future*
# `logger.add(...)` call that forgets to pass `diagnose=` explicitly. Doing
# both (not just the env var) matters: `LOGURU_DIAGNOSE` only takes effect
# for a given `add()` call if it was set before loguru's `_logger.py` module
# was first imported anywhere in the process -- it becomes a bound default
# on `Logger.add`'s signature at that point, not a value read fresh on every
# call -- and `Tests/conftest.py` imports loguru ahead of this package for
# fixture-ordering reasons, which would otherwise defeat the env var alone
# for the whole test session. The explicit `remove(0)` + `add(diagnose=False)`
# below is not subject to that ordering: it runs the moment this package is
# imported, which is unconditionally required before any of its own
# exception handlers can execute.
#
# Only loguru's auto-init sink (always handler id 0) is replaced. A host
# application that configured loguru before importing this package keeps its
# own sinks untouched -- if it removed the default sink, `remove(0)` raises
# ValueError and this block installs nothing, because that host owns the
# logging configuration (and any diagnose=True sink it installed on purpose).
#
# Phase 0c (pinned worker import closure): the sink replacement is gated on
# loguru ALREADY being imported, so importing this package (which every
# ``tldw_chatbook`` import runs, the stdlib-only workspace worker's
# included) no longer pulls loguru in itself. The two arms together keep
# the security invariant above intact:
#
# * loguru already in ``sys.modules`` (the ``Tests/conftest.py`` case, and
#   any host that imported it first): the explicit ``remove(0)`` +
#   ``add(diagnose=False, backtrace=True)`` below runs exactly as before,
#   because in this arm the env var alone provably arrived too late.
# * loguru not yet imported: the eager ``LOGURU_DIAGNOSE=0`` setdefault
#   above lands before loguru's first import anywhere (nothing in this
#   package can import loguru before this init ran), so loguru's own
#   auto-init sink is constructed with ``diagnose=False`` and its default
#   ``backtrace=True`` -- the same sink configuration the explicit
#   replacement below installs.
os.environ.setdefault("LOGURU_DIAGNOSE", "0")
if "loguru" in sys.modules:
    try:
        from loguru import logger as _pkg_init_loguru_logger

        try:
            _pkg_init_loguru_logger.remove(0)
        except ValueError:
            pass
        else:
            _pkg_init_loguru_logger.add(sys.stderr, diagnose=False, backtrace=True)
    except Exception:
        # Logging safety must never be the reason package import fails.
        pass


def _install_textual_compatibility_shims() -> None:
    """Keep older Textual widget access patterns working across dependency bumps."""
    try:
        from textual.widgets import Static
    except Exception:
        return

    if not hasattr(Static, "renderable"):
        Static.renderable = property(  # type: ignore[attr-defined]
            lambda self: self.content,
            lambda self, value: self.update(value),
        )
    _install_zero_width_fold_guard()


#: Marks the guarded ``Content._wrap_and_format`` (idempotent, detectable).
ZERO_WIDTH_FOLD_GUARD = "_tldw_zero_width_fold_guard"


def _install_zero_width_fold_guard() -> None:
    """Crop instead of folding text at a zero fold width (TASK-38740).

    Textual 8.2.8 folds a line with ``range(0, length, width)`` (no wrap) or
    Rich's ``chop_cells`` (wrap), and both raise ``ValueError`` when the fold
    width is 0. A Button's label hits it when its content box is exactly its
    two ``line-pad`` cells, an OptionList option when its box is 0. In layout
    that exits the app, and no stylesheet rule can rule the width out: a
    grid's auto row ignores ``min-width`` and ``max-width: 100%`` follows a
    2-cell parent. Every height and paint of a ``Content`` goes through this
    one method, so the guard lives here: at that width the text renders as
    ``text-wrap: nowrap; text-overflow: clip``, one line cropped to the box.
    ``text-overflow: ellipsis`` everywhere would also stop the raise, but in
    this Textual a wrapped line then paints two cells wider than its widget
    and shifts the rest of the row.

    TASK-38952: that over-paint is cut here too. A wrapped box at or below
    its line-pad renders the same cropped line, and a wrapped line wider
    than the box minus its line-pad is cut to fit before the pad.
    """
    from textual.content import Content

    upstream = Content._wrap_and_format
    if getattr(upstream, ZERO_WIDTH_FOLD_GUARD, False):
        return

    def _wrap_and_format(
        self,
        width,
        align="left",
        overflow="fold",
        no_wrap=False,
        line_pad=0,
        *args,
        **kwargs,
    ):
        zero = 0 if no_wrap else 2 * line_pad
        if (overflow == "fold" and width == zero) or (
            # TASK-38952: below the line-pad too (a 1-cell Button painted 3).
            not no_wrap and line_pad and 0 <= width <= zero
        ):
            # One cropped line: a wrapped crop would add the line-pad on top.
            overflow, no_wrap = "clip", True
        lines = upstream(
            self, width, align, overflow, no_wrap, line_pad, *args, **kwargs
        )
        if not no_wrap and line_pad and width > zero:
            # TASK-38952: upstream cuts a wrapped line to the full width and
            # then pads it, so a line wider than the box minus its line-pad
            # (an unbreakable word, trailing spaces) painted 2 cells past it.
            for line in lines:
                content = line.content
                if content.cell_length > width:
                    core = content[line_pad : len(content) - line_pad]
                    line.content = core.truncate(
                        width - 2 * line_pad, ellipsis=overflow == "ellipsis"
                    ).pad(line_pad, line_pad)
        return lines

    setattr(_wrap_and_format, ZERO_WIDTH_FOLD_GUARD, True)
    _wrap_and_format.__wrapped__ = upstream  # type: ignore[attr-defined]
    Content._wrap_and_format = _wrap_and_format  # type: ignore[method-assign]


__version__ = "0.2.4"
__author__ = "Robert Musser"
__email__ = "contact@rmusser.net"
__license__ = "AGPLv3+"

# Version tuple for programmatic comparison
VERSION_TUPLE = (0, 2, 4)

# Export key components when package is imported
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "VERSION_TUPLE",
]
