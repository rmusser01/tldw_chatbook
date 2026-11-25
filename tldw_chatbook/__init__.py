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
from functools import cache, partial, wraps
from typing import TYPE_CHECKING, TypeVar

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

# TASK-33621.113: the settings above disable tqdm's bars, but a disabled bar
# still builds tqdm's process-wide write lock, and on macOS its
# multiprocessing.RLock starts CPython's resource tracker with
# ``sys.stderr.fileno()``. Inside the running app that is Textual's capture
# (``-1``), so the first Hugging Face download from ANY feature raised
# ``ValueError: bad value(s) in fds_to_keep`` and left tqdm's class lock held,
# wedging every later download until restart. Give tqdm the thread-only lock
# it already falls back to when a multiprocessing lock cannot be built
# (``mp_lock = None``; nothing here shares bars across processes) as
# ``tqdm.std`` is first imported, the same arming pattern as tiktoken above.
# Nothing is imported here, so the boot import census is unchanged. Console
# dictation keeps its own guard (``Utils/model_download_guard.py``), which
# also repairs a lock that was wedged before this ran.
# Incompatible: ``tqdm.contrib.concurrent.process_map``. It shares the lock's
# ``mp_lock`` with its workers (``ensure_lock(lock_name="mp_lock")``), which is
# None here, so tqdm 4.67.3 calls ``set_lock(None)`` and the next bar raises
# TypeError. Nothing in this repo or its dependencies calls it today;
# ``thread_map`` is unaffected.
_TQDM_LOCK_MODULE = "tqdm.std"
#: Marks an armer from ANY execution of this init (a reload defines a new
#: class), so two never coexist or ask each other for ``tqdm.std``.
_TQDM_ARMER_MARKER = "_tldw_tqdm_lock_armer"


def _decide_tqdm_write_lock(module: object) -> None:
    """Pre-decide tqdm's lock as thread-only, unless tqdm already decided it."""
    lock_type = getattr(module, "TqdmDefaultWriteLock", None)
    if lock_type is not None and not hasattr(lock_type, "mp_lock"):
        lock_type.mp_lock = None


class _TqdmLockArmingLoader:
    """Loader wrapper that decides tqdm's lock once ``tqdm.std`` has executed.

    No bar can exist before ``exec_module`` returns, so deciding there comes
    strictly before tqdm's first lock build.
    """

    def __init__(self, loader: object, finder: "_TqdmLockArmingFinder") -> None:
        self._loader = loader
        self._finder = finder

    def create_module(self, spec: object) -> object:
        return self._loader.create_module(spec)

    def exec_module(self, module: object) -> None:
        # A failed import leaves the finder armed for the next attempt.
        self._loader.exec_module(module)
        _decide_tqdm_write_lock(module)
        self._finder.discard()

    def __getattr__(self, name: str) -> object:
        return getattr(self._loader, name)


class _TqdmLockArmingFinder:
    """Meta-path finder that arms ``tqdm.std``'s first import, then retires."""

    _tldw_tqdm_lock_armer = True

    def find_spec(
        self, fullname: str, path: object = None, target: object = None
    ) -> object:
        if fullname != _TQDM_LOCK_MODULE:
            return None
        # Ask the remaining finders, as the import system would after us; never
        # another armer, which would ask us back (unbounded recursion).
        for finder in tuple(sys.meta_path):
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None or getattr(finder, _TQDM_ARMER_MARKER, False):
                continue
            spec = find_spec(fullname, path, target)
            if spec is not None:
                break
        else:
            return None
        if spec.loader is not None and hasattr(spec.loader, "exec_module"):
            spec.loader = _TqdmLockArmingLoader(spec.loader, self)
        return spec

    def discard(self) -> None:
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass


def _arm_tqdm_write_lock() -> None:
    """Decide tqdm's lock now if tqdm is loaded, else at its first import."""
    module = sys.modules.get(_TQDM_LOCK_MODULE)
    if module is not None:
        _decide_tqdm_write_lock(module)
        return
    if not any(getattr(f, _TQDM_ARMER_MARKER, False) for f in sys.meta_path):
        sys.meta_path.insert(0, _TqdmLockArmingFinder())


_arm_tqdm_write_lock()

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


_strip_caches_installed = False


def install_lazy_strip_caches() -> bool:
    """Install the verified allocation adapter, leaving unknown native shapes alone."""
    global _strip_caches_installed
    if _strip_caches_installed:
        return True

    import logging
    from functools import wraps
    from threading import RLock
    from types import MemberDescriptorType
    from typing import Any

    import textual
    from textual import strip as strip_module
    from textual.cache import FIFOCache
    from textual.strip import Strip

    _CACHE_SIZES = {
        "_divide_cache": 4,
        "_crop_cache": 16,
        "_style_cache": 16,
        "_filter_cache": 4,
        "_line_length_cache": 4,
        "_crop_extend_cache": 4,
        "_offsets_cache": 4,
    }

    _allocation_lock = RLock()

    def _lazy_cache(slot: MemberDescriptorType, capacity: int) -> property:
        def get(instance: Strip) -> FIFOCache[Any, Any]:
            cache = slot.__get__(instance, Strip)
            if cache is None:
                with _allocation_lock:
                    cache = slot.__get__(instance, Strip)
                    if cache is None:
                        cache = FIFOCache(capacity)
                        slot.__set__(instance, cache)
            return cache

        return property(get, slot.__set__, slot.__delete__)

    slots = {name: Strip.__dict__.get(name) for name in _CACHE_SIZES}
    constructor = Strip.__init__
    expected_slots = {
        *_CACHE_SIZES,
        "_segments",
        "_cell_length",
        "_render_cache",
        "_link_ids",
        "_cell_count",
    }
    if (
        textual.__version__ != "8.2.8"
        or set(getattr(Strip, "__slots__", ())) != expected_slots
        or any(
            type(Strip.__dict__.get(name)) is not MemberDescriptorType
            for name in expected_slots
        )
        or getattr(getattr(constructor, "__code__", None), "co_filename", None)
        != strip_module.__file__
    ):
        logging.getLogger(__name__).warning(
            "Lazy Strip caches not installed: unsupported Textual version or native shape"
        )
        return False

    @wraps(constructor)
    def initialize(instance, segments, cell_length=None):
        if type(instance) is not Strip:
            constructor(instance, segments, cell_length)
            return
        # Native 8.2.8 data initialization; only its seven FIFO allocations are deferred.
        instance._segments = list(segments)
        instance._cell_length = cell_length
        for slot in slots.values():
            slot.__set__(instance, None)
        instance._render_cache = None
        instance._link_ids = None
        instance._cell_count = None

    properties = {
        name: _lazy_cache(slots[name], capacity)
        for name, capacity in _CACHE_SIZES.items()
    }
    for name, lazy in properties.items():
        setattr(Strip, name, lazy)
    Strip.__init__ = initialize
    _strip_caches_installed = True
    return True


_style_colors_installed = False


def install_owner_free_style_colors() -> bool:
    """Adapt only the verified native color calculation, otherwise leave it alone."""
    global _style_colors_installed
    if _style_colors_installed:
        return True

    import logging
    from dataclasses import replace
    from dis import get_instructions
    from functools import lru_cache

    import textual
    from textual import _styles_cache as styles_cache_module
    from textual._styles_cache import StylesCache
    from textual.color import Color
    from textual.style import Style

    native = StylesCache.__dict__.get("get_inner_outer")
    compute = getattr(native, "__wrapped__", None)
    code = getattr(compute, "__code__", None)
    parameters = getattr(native, "cache_parameters", None)
    if (
        textual.__version__ != "8.2.8"
        or code is None
        or code.co_filename != styles_cache_module.__file__
        or code.co_argcount != 3
        or code.co_varnames[:3] != ("cls", "base_background", "background")
        or any(
            instruction.opname.startswith("LOAD_")
            # 3.13 superinstructions (LOAD_FAST_LOAD_FAST) carry a name tuple.
            and "cls"
            in (
                instruction.argval
                if isinstance(instruction.argval, tuple)
                else (instruction.argval,)
            )
            for instruction in get_instructions(compute)
        )
        or not callable(parameters)
        or parameters() != {"maxsize": 1024, "typed": False}
    ):
        logging.getLogger(__name__).warning(
            "Owner-free style colors not installed: unsupported Textual version or native shape"
        )
        return False

    @lru_cache(maxsize=1024)
    def colors(base_background: Color, background: Color) -> tuple[Style, Style]:
        return compute(None, base_background, background)

    def detached_colors(
        owner: StylesCache, base_background: Color, background: Color
    ) -> tuple[Style, Style]:
        # Frozen Textual styles still memoize mutable, color-system-specific ANSI.
        templates = colors(base_background, background)
        if type(owner) is StylesCache:
            previous = getattr(owner, "_tldw_current_color_result", None)
            if previous is not None and previous[0] is templates:
                return previous[1]
        inner, outer = templates
        result = replace(inner), replace(outer)
        if type(owner) is StylesCache:
            owner._tldw_current_color_result = templates, result
        return result

    detached_colors.cache_parameters = colors.cache_parameters
    detached_colors.cache_info = colors.cache_info
    detached_colors.cache_clear = colors.cache_clear
    StylesCache.get_inner_outer = detached_colors
    _style_colors_installed = True
    return True


_box_models_installed = False


def install_owner_local_box_models() -> bool:
    """Adapt verified native geometry without changing its calculation or LRU."""
    global _box_models_installed
    if _box_models_installed:
        return True

    import hashlib
    import inspect
    import logging
    from fractions import Fraction
    from functools import wraps
    from itertools import islice
    from tokenize import TokenError

    import textual
    from textual import widget as widget_module
    from textual.box_model import BoxModel
    from textual.cache import LRUCache
    from textual.geometry import Size, Spacing
    from textual.widget import Widget

    _NATIVE_DIGEST = "1ccca76e5161f18cde4146808f94c9e0d840a9b41c82958f313f85e1a8580f02"

    def _source(method) -> str:
        # inspect.getsource would retain entire native files in global linecache.
        code = method.__code__
        with open(code.co_filename, encoding="utf-8") as source:
            return "".join(
                inspect.getblock(list(islice(source, code.co_firstlineno - 1, None)))
            )

    def _immutable(model: object) -> bool:
        return (
            type(model) is BoxModel
            and type(model.width) is Fraction
            and type(model.height) is Fraction
            and type(model.margin) is Spacing
        )

    native = Widget._get_box_model
    try:
        digest = hashlib.sha256(
            "".join(
                _source(method) for method in (native, LRUCache.get, LRUCache.set)
            ).encode()
        ).hexdigest()
    except (AttributeError, OSError, SyntaxError, TokenError, TypeError):
        digest = None
    if (
        textual.__version__ != "8.2.8"
        or getattr(getattr(native, "__code__", None), "co_filename", None)
        != widget_module.__file__
        or digest != _NATIVE_DIGEST
        or LRUCache.__setitem__ is not LRUCache.set
        or set(LRUCache.__slots__)
        != {"_maxsize", "_cache", "_full", "_head", "hits", "misses"}
    ):
        logging.getLogger(__name__).warning(
            "Owner-local BoxModel reuse not installed: unsupported Textual version or native shape"
        )
        return False

    @wraps(native)
    def get_box_model(
        self: Widget,
        container: Size,
        viewport: Size,
        width_fraction: Fraction,
        height_fraction: Fraction,
        constrain_width: bool = False,
        greedy: bool = True,
    ) -> BoxModel:
        cache = self._box_model_cache
        eligible = (
            type(cache) is LRUCache and type(self)._get_box_model is get_box_model
        )
        previous = cache._head if eligible else None
        misses = cache.misses if eligible else 0
        result = native(
            self,
            container,
            viewport,
            width_fraction,
            height_fraction,
            constrain_width,
            greedy,
        )
        if eligible and self._box_model_cache is cache:
            head = cache._head
            if (
                previous
                and len(previous) == 4
                and len(head) == 4
                and head is not previous
                and head[1] is previous
                and cache.misses == misses + 1
                and head[3] is result
                and _immutable(result)
                and _immutable(previous[3])
                and result == previous[3]
            ):
                # A clear/replacement/reentrant cache change cannot reuse this link.
                head[3] = previous[3]
                return previous[3]
        return result

    Widget._get_box_model = get_box_model
    _box_models_installed = True
    return True


_adapter_installs_attempted = False


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
    _install_queued_message_keeper()
    # Optional allocation adapters: any failure (e.g. missing Textual
    # dist-info metadata in a packaged build) leaves native Textual in place.
    # Both UI and Widgets imports land here; try (and warn) once per process.
    global _adapter_installs_attempted
    if _adapter_installs_attempted:
        return
    _adapter_installs_attempted = True
    for install in (
        install_lazy_strip_caches,
        install_owner_free_style_colors,
        install_owner_local_box_models,
    ):
        try:
            install()
        except Exception as error:  # noqa: BLE001 - an optimisation must never stop the UI importing.
            import logging

            logging.getLogger(__name__).warning(
                "%s not installed: %s", install.__name__, type(error).__name__
            )


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


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from textual.widget import Widget

_T = TypeVar("_T")

#: Marks the message-keeping ``App._prune`` (idempotent, detectable).
QUEUED_MESSAGE_KEEPER = "_tldw_queued_message_keeper"


@cache
def _kept_messages() -> tuple[tuple[type, ...], tuple[type, ...]]:
    """The widgets whose queued user action a removal keeps, and the actions.

    TASK-42150: only messages that a user's own input posts. ``Select``,
    ``Checkbox`` and ``Switch`` post ``Changed`` on a mount echo or a
    programmatic ``value =`` as well, so one from a removed generation is not
    handed on as if the user had chosen it.
    """
    from textual.widgets import Button, Input, ListView

    return (Button, Input, ListView), (
        Button.Pressed,
        Input.Submitted,
        ListView.Selected,
    )


async def _forward_a_dropped_message(node, on_message, message) -> None:
    """Run ``node``'s own dispatch, then hand on a kept message its closing parent drops."""
    await on_message(message)
    if (
        not isinstance(message, _kept_messages()[1])
        or message._stop_propagation
        or node.is_parent_active
    ):
        return
    target = node.parent
    while target is not None and target._pruning:
        target = target.parent
    # Never past a removed screen to the app, nor past the exit fence.
    if target is not None and not target.is_dom_root and not target._closing:
        if target.is_attached:
            message._bubble_to(target)


def keep_queued_messages(nodes: "Iterable[Widget]") -> None:
    """Deliver the user actions that removing ``nodes`` would drop.

    TASK-41703, generalised by TASK-42014 and TASK-42150. Textual removes a
    subtree top down: a removed node stops taking messages first, and a node
    below it that then handles a queued user action (``_kept_messages``)
    finds its parent closing, so ``MessagePump._on_message`` does not bubble
    it. Each node between such a sender and the removed node hands the
    message to its nearest ancestor that is not being removed instead (the
    container whose children go), which bubbles it on. Handlers on the
    removed nodes it skips do not run. ``App._prune``, which every removal
    goes through, calls this once the nodes are marked as pruning.
    """
    senders = _kept_messages()[0]
    for top in nodes:
        stays = top.parent
        # A removed screen's nodes could only reach the app, and a node whose
        # parent is going was walked with that parent.
        if stays is None or stays.is_dom_root or stays._pruning:
            continue
        for sender in top.walk_children():
            if not isinstance(sender, senders):
                continue
            node = sender
            while (parent := node.parent) is not None and parent._pruning:
                if "_on_message" not in vars(node):
                    node._on_message = partial(
                        _forward_a_dropped_message, node, node._on_message
                    )
                node = parent


def rendered_target(targets: "Mapping[Widget, _T]", sender: "Widget") -> "_T | None":
    """What ``sender`` was rendered for, or None when it is from an earlier
    rebuild or is being removed (TASK-42150 review r19rc-rev1).

    A press can reach its handler after the rows it came from were rebuilt:
    the keeper hands one below a removed row to a live ancestor, and one that
    had already left the row is handled after the rebuild too. A position read
    from the sender's id then names whatever row holds that position now, so
    resolve the target from the sender instance: the builder maps each
    control it renders to its target, a new map per rebuild.
    """
    if not sender.is_attached or sender._pruning:
        return None
    return targets.get(sender)


def _install_queued_message_keeper() -> None:
    """Make every removal keep a user action queued below what it removes."""
    from textual.app import App

    native = App._prune
    if getattr(native, QUEUED_MESSAGE_KEEPER, False):
        return

    @wraps(native)
    def _prune(self, *nodes, parent=None):
        removal = native(self, *nodes, parent=parent)
        # Synchronous: the nodes are marked pruning and none has run since.
        keep_queued_messages(nodes)
        return removal

    setattr(_prune, QUEUED_MESSAGE_KEEPER, True)
    App._prune = _prune  # type: ignore[method-assign]


__version__ = "0.2.3"
__author__ = "Robert Musser"
__email__ = "contact@rmusser.net"
__license__ = "AGPLv3+"

# Version tuple for programmatic comparison
VERSION_TUPLE = (0, 2, 3)

# Export key components when package is imported
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "VERSION_TUPLE",
]
