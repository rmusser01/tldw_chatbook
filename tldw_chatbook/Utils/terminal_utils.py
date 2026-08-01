# tldw_chatbook/Utils/terminal_utils.py
# Description: Terminal capability detection utilities
#
# Imports
#
# Standard Library
import importlib
import os
import time
from typing import Dict, Literal

#
# Local Imports
from loguru import logger

from ..Metrics.metrics_logger import log_counter, log_histogram
#
#######################################################################################################################
#
# Functions:


def warm_up_image_protocol() -> bool:
    """Resolve textual_image's rendering protocol while the terminal is free.

    task-1650: ``textual_image`` chooses its renderer exactly once, at
    import time, by writing an escape query and reading the terminal's
    reply (see ``textual_image/renderable/__init__.py``). Every app-side
    import is lazy -- nested inside functions that run in the LIVE app --
    and by then Textual holds the terminal in raw mode and owns stdin, so
    the query never gets an answer and selection silently degrades to
    half-cell rendering. The result is pixelated avatars and inline images
    in Kitty/iTerm2 with no exception and no log line. textual_image warns
    about exactly this in its own source ("querying the terminal isn't
    possible anymore once Textual is started").

    Call this ONCE from each entry point before ``App.run()``. Importing
    the top-level ``textual_image`` package is not sufficient, which is
    also why this cannot route through ``optional_deps.check_dependency``:
    that helper imports the TOP-LEVEL package, and the protocol choice
    lives in the ``renderable`` submodule that only ``textual_image.widget``
    pulls in.

    Outcomes are logged and counted (Qodo PR #1150): a silent degrade is
    the very symptom this function exists to remove.

    Returns:
        True if the protocol-selecting import completed; False when the
        optional dependency is absent or its terminal query failed (both
        leave the app on its mosaic/pixels fallbacks, which always work).
    """
    # Qodo PR #1150: availability goes through the central helper so
    # DEPENDENCIES_AVAILABLE stays consistent with the rest of the
    # codebase. The submodule import below is still done directly and
    # deliberately -- check_dependency() imports only the TOP-LEVEL
    # package, which never loads textual_image.renderable where the
    # protocol is chosen, so routing the whole thing through it would
    # silently reinstate the bug this function exists to fix.
    from .optional_deps import check_dependency

    try:
        if not check_dependency("textual_image"):
            raise ImportError("textual_image unavailable")
        importlib.import_module("textual_image.widget")
    except ImportError:
        # Optional dependency absent -- expected, not a defect.
        logger.debug(
            "textual_image not installed; images use mosaic/pixels rendering."
        )
        log_counter(
            "terminal_utils_image_protocol_warmup",
            labels={"result": "missing_dependency"},
        )
        return False
    except Exception as exc:
        # A terminal that does not answer the capability query raises here
        # (observed: TerminalError under a bare pty). Rendering still
        # degrades gracefully, but say so rather than failing silently.
        logger.warning(
            "Image protocol warm-up failed ({}); falling back to mosaic/pixels "
            "rendering.",
            type(exc).__name__,
        )
        log_counter(
            "terminal_utils_image_protocol_warmup", labels={"result": "query_failed"}
        )
        return False

    selected = "unknown"
    try:
        import textual_image.renderable as _renderable

        selected = _renderable.Image.__module__.rsplit(".", 1)[-1]
    except Exception:  # pragma: no cover - introspection only
        pass
    logger.info("Image render protocol selected: {}", selected)
    log_counter(
        "terminal_utils_image_protocol_warmup",
        labels={"result": "ok", "protocol": selected},
    )
    return True


def detect_terminal_capabilities() -> Dict[str, any]:
    """
    Detect terminal image support capabilities.

    Returns:
        Dictionary with:
        - sixel: Whether sixel graphics are supported
        - tgp: Whether terminal graphics protocol is supported
        - unicode: Whether unicode is supported (assumed True)
        - recommended_mode: Recommended rendering mode ('pixels', 'regular', or 'auto')
        - terminal_type: Detected terminal name ('kitty', 'wezterm', 'iterm2', ... or 'unknown')
    """
    start_time = time.time()
    log_counter("terminal_utils_detect_capabilities_attempt")

    term = os.environ.get("TERM", "").lower()
    term_program = os.environ.get("TERM_PROGRAM", "").lower()

    capabilities = {
        "sixel": False,
        "tgp": False,
        "unicode": True,  # Assume unicode support by default
        "recommended_mode": "pixels",  # Default to rich-pixels
    }

    # tmux/screen panes inherit the HOST terminal's env (TERM_PROGRAM,
    # ITERM_SESSION_ID, ...) but tmux does not pass graphics escape
    # sequences (TGP/iTerm2/sixel) through to that host terminal, so
    # trusting the leaked identity paints nothing at all. Half-block
    # pixels are the only rendering that survives a multiplexer.
    if os.environ.get("TMUX") or term.startswith(("screen", "tmux")):
        capabilities["terminal_type"] = "tmux"
        duration = time.time() - start_time
        log_histogram("terminal_utils_detect_capabilities_duration", duration)
        log_counter(
            "terminal_utils_detect_capabilities_result",
            labels={
                "terminal_type": "tmux",
                "has_sixel": str(capabilities["sixel"]),
                "has_tgp": str(capabilities["tgp"]),
                "recommended_mode": capabilities["recommended_mode"],
            },
        )
        return capabilities

    # Check for specific terminals that support advanced graphics

    # Kitty terminal
    if "kitty" in term or "kitty" in term_program:
        capabilities["tgp"] = True
        capabilities["recommended_mode"] = "regular"

    # WezTerm
    elif "wezterm" in term:
        capabilities["tgp"] = True
        capabilities["sixel"] = True
        capabilities["recommended_mode"] = "regular"

    # iTerm2
    elif "iterm" in term_program or os.environ.get("ITERM_SESSION_ID"):
        capabilities["tgp"] = True
        capabilities["recommended_mode"] = "regular"

    # Alacritty (limited support)
    elif "alacritty" in term or "alacritty" in term_program:
        # Alacritty has limited image support
        capabilities["recommended_mode"] = "pixels"

    # XTerm with 256 colors (may support sixel)
    elif "xterm" in term and "256color" in term:
        # Some xterm builds support sixel
        capabilities["sixel"] = True
        capabilities["recommended_mode"] = "regular"

    # Konsole
    elif "konsole" in term_program:
        capabilities["sixel"] = True
        capabilities["recommended_mode"] = "regular"

    # GNOME Terminal / VTE-based terminals
    elif os.environ.get("VTE_VERSION"):
        # Most VTE-based terminals don't support advanced graphics
        capabilities["recommended_mode"] = "pixels"

    # Windows Terminal
    elif "windows-terminal" in term_program or os.environ.get("WT_SESSION"):
        # Windows Terminal has limited image support
        capabilities["recommended_mode"] = "pixels"

    # Log detected capabilities
    duration = time.time() - start_time
    log_histogram("terminal_utils_detect_capabilities_duration", duration)

    # Determine terminal type for metrics
    terminal_type = "unknown"
    if "kitty" in term or "kitty" in term_program:
        terminal_type = "kitty"
    elif "wezterm" in term:
        terminal_type = "wezterm"
    elif "iterm" in term_program or os.environ.get("ITERM_SESSION_ID"):
        terminal_type = "iterm2"
    elif "alacritty" in term or "alacritty" in term_program:
        terminal_type = "alacritty"
    elif "xterm" in term:
        terminal_type = "xterm"
    elif "konsole" in term_program:
        terminal_type = "konsole"
    elif os.environ.get("VTE_VERSION"):
        terminal_type = "vte_based"
    elif "windows-terminal" in term_program or os.environ.get("WT_SESSION"):
        terminal_type = "windows_terminal"

    log_counter(
        "terminal_utils_detect_capabilities_result",
        labels={
            "terminal_type": terminal_type,
            "has_sixel": str(capabilities["sixel"]),
            "has_tgp": str(capabilities["tgp"]),
            "recommended_mode": capabilities["recommended_mode"],
        },
    )

    capabilities["terminal_type"] = terminal_type
    return capabilities


def get_image_render_mode(config_mode: str = "auto") -> Literal["pixels", "regular"]:
    """
    Determine the best image rendering mode based on terminal capabilities and config.

    Args:
        config_mode: Configuration mode ('auto', 'pixels', 'regular')

    Returns:
        Either 'pixels' or 'regular' rendering mode
    """
    log_counter(
        "terminal_utils_get_render_mode_attempt", labels={"config_mode": config_mode}
    )

    if config_mode == "pixels":
        log_counter(
            "terminal_utils_render_mode_result",
            labels={"mode": "pixels", "reason": "config"},
        )
        return "pixels"
    elif config_mode == "regular":
        log_counter(
            "terminal_utils_render_mode_result",
            labels={"mode": "regular", "reason": "config"},
        )
        return "regular"
    else:  # auto mode
        capabilities = detect_terminal_capabilities()
        # If terminal supports advanced graphics and textual-image is available
        if capabilities["recommended_mode"] == "regular":
            # Check if textual-image is available
            try:
                import textual_image  # noqa: F401

                log_counter(
                    "terminal_utils_render_mode_result",
                    labels={"mode": "regular", "reason": "auto_capable"},
                )
                return "regular"
            except ImportError:
                # Fall back to pixels if textual-image not available
                log_counter(
                    "terminal_utils_render_mode_result",
                    labels={
                        "mode": "pixels",
                        "reason": "auto_fallback_no_textual_image",
                    },
                )
                return "pixels"

        log_counter(
            "terminal_utils_render_mode_result",
            labels={"mode": "pixels", "reason": "auto_default"},
        )
        return "pixels"


def is_image_support_available() -> bool:
    """
    Check if any form of image support is available.

    Returns:
        True if images can be displayed (either mode)
    """
    log_counter("terminal_utils_check_image_support_attempt")

    # Check for rich-pixels (should always be available if PIL is installed)
    try:
        import rich_pixels  # noqa: F401
        import PIL  # noqa: F401

        log_counter(
            "terminal_utils_image_support_result",
            labels={"available": "true", "library": "rich_pixels"},
        )
        return True
    except ImportError:
        pass

    # Check for textual-image
    try:
        import textual_image  # noqa: F401

        log_counter(
            "terminal_utils_image_support_result",
            labels={"available": "true", "library": "textual_image"},
        )
        return True
    except ImportError:
        pass

    log_counter(
        "terminal_utils_image_support_result",
        labels={"available": "false", "library": "none"},
    )
    return False


#
#
#######################################################################################################################
