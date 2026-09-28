"""Process entry points for ``tldw_chatbook`` (moved from ``app.py``, TASK-33011).

Early logging, the generated-CSS staleness check and its build manifest, the
argument parser, the ``textual serve`` factory (``get_app``), the console-script
runner (``main_cli_runner``) and the body of ``python -m tldw_chatbook.app``
(``_run_module_main``). The bodies are unchanged from ``app.py``; the only
edits are three ``noqa`` markers for lint the function scope newly exposes.

This module is reached only through ``cli.py``'s lazy import, ``python -m`` and
the lazy ``tldw_chatbook.app.__getattr__`` re-export -- never on the in-process
boot path -- so nothing here counts against the UI-ready module census.
"""

# ADR-126: importing ``tldw_chatbook.app`` first runs its recovery fence
# (``admit_startup``) before any config or runtime import below.
from tldw_chatbook.app import TldwCli  # noqa: I001 -- the fence must import first

import argparse
import hashlib
import logging
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

from loguru import logger as loguru_logger
from textual.css.query import QueryError

from tldw_chatbook.config import (
    load_cli_config_and_ensure_existence,
    load_settings,
    set_encryption_password,
)
from tldw_chatbook.css import build_css, widget_css
from tldw_chatbook.Logging_Config import configure_application_logging
from tldw_chatbook.Metrics.metrics import init_metrics_server
from tldw_chatbook.Metrics.Otel_Metrics import init_metrics as init_otel_metrics
from tldw_chatbook.Utils.app_shutdown import (
    arm_exit_watchdog,
    install_termination_handlers,
)
from tldw_chatbook.Utils.Emoji_Handling import (
    EMOJI_TITLE_BRAIN,
    FALLBACK_TITLE_BRAIN,
    get_char,
    supports_emoji,
)


# Initialize logging at the earliest possible point
def initialize_early_logging() -> object:
    """Initialize logging as early as possible to capture all logs from startup.

    Returns:
        The minimal app-like object logging was configured against.
    """

    # Create a temporary app-like object with just enough attributes for configure_application_logging
    class EarlyLoggingApp:
        def __init__(self):
            self.app_config = load_settings()
            self._rich_log_handler = None

        def query_one(self, *args, **kwargs):
            # This will fail in configure_application_logging, but that's expected
            # for early logging - we just want to set up file and console logging
            raise QueryError("Early logging setup - UI not available yet")

    # Configure logging with our minimal app-like object
    early_app = EarlyLoggingApp()
    configure_application_logging(early_app)
    logging.info("Early logging initialization complete")
    loguru_logger.info("Early logging initialization complete (loguru)")
    return early_app


def _is_source_tree(package_root: Path) -> bool:
    """Return whether package files are inside a build-capable source tree."""

    return (package_root.parent / "pyproject.toml").is_file()


#: A class-level ``BUNDLED_CSS`` / ``BUNDLED_SCREEN_CSS`` *assignment*, which is
#: what makes a module an input to the generated stylesheets. Anchored on the
#: assignment rather than matching the bare name anywhere in the file: four
#: package modules -- including this one, via ``_generated_css_is_stale``'s own
#: docstring -- discuss the marker while declaring nothing, and a plain substring
#: test made every edit to any of them rebuild the CSS on the next boot, quietly
#: rewriting the committed bundle's ``Generated:`` timestamp. A module that has
#: just *gained* a declaration is still caught: a declaration is an assignment.
_BUNDLED_CSS_DECLARATION_RE = re.compile(r"^\s*BUNDLED_(?:SCREEN_)?CSS\s*[:=]", re.M)


def _load_css_build_manifest(css_dir: Path) -> dict[str, list] | None:
    """Load the builder's content manifest, or ``None`` when absent/invalid.

    The manifest is written by ``build_css.write_build_manifest`` beside the
    generated sheets; see TASK-18910. Each entry is ``[sha256, mtime_at_build]``.
    Any read/parse/shape problem returns ``None`` so the caller falls back to
    the legacy mtime rule -- a broken manifest costs one spurious rebuild,
    never a missed one.
    """
    try:
        import json

        with open(css_dir / build_css.BUILD_MANIFEST_FILENAME, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not data:
        # An empty manifest is treated as absent: the max() over its entries
        # would raise, and an empty build is not a state the builder can
        # produce (it always records at least the CSS_MODULES that exist).
        return None
    manifest: dict[str, list] = {}
    for key, value in data.items():
        if (
            not isinstance(key, str)
            or not isinstance(value, list)
            or len(value) != 2
            or not isinstance(value[0], str)
            or not isinstance(value[1], (int, float))
        ):
            return None  # unknown shape: treat as absent
        manifest[key] = value
    return manifest


def _save_css_build_manifest(css_dir: Path, manifest: dict[str, list]) -> None:
    """Persist an updated manifest (mtime refreshes after hash confirmation).

    Best-effort: a write failure costs one re-hash on the next boot, never a
    missed or spurious rebuild -- the in-memory decision has already been
    made with the correct data.
    """
    try:
        import json

        (css_dir / build_css.BUILD_MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def _generated_css_is_stale(package_root: Path) -> tuple[bool, str]:
    """Return whether the generated stylesheets need rebuilding, and why.

    Source-tree boots rebuild the CSS when its inputs have moved on. Before
    TASK-15450 every input was a ``.tcss`` module, so checking those mtimes was
    exhaustive. Four of the five generated sheets are now built from class-level
    ``BUNDLED_CSS`` / ``BUNDLED_SCREEN_CSS`` literals in Python modules, so a
    widget-CSS edit would otherwise have *no effect* until someone remembered to
    run ``build_css.py`` by hand -- where editing ``DEFAULT_CSS`` used to take
    effect on the very next run, because Textual read it straight off the class.

    A Python module counts as an input only if it is *newer than the build* and
    actually mentions the marker. Both halves matter. Treating every ``.py`` as
    an input was tried first and is wrong: editing ``app.py`` -- or any of the
    ~1,640 files in this package -- would then re-run the build subprocess on
    every single developer boot. Reading files to find the marker is likewise
    only affordable because the mtime test has already narrowed the set, which is
    normally empty. Checking the marker rather than the list of modules the
    sheets currently name is what catches a module that has just *gained* a
    ``BUNDLED_CSS`` declaration -- exactly the file a "nothing happened" bug
    report starts from.

    Cost: one ``os.walk`` of the package, ~0.3 ms warm for ~1,640 files, plus a
    read of each file changed since the last build (normally none). It runs only
    under ``_is_source_tree`` -- for developers, never for a wheel install -- and
    never on the per-frame or per-keystroke paths.

    Known gap: *deleting* a module that carried ``BUNDLED_CSS`` leaves no newer
    file behind, so it is not detected here. The CSS bundle guard in CI covers
    that; this check is a dev-loop convenience, not the authority.

    Args:
        package_root: The installed ``tldw_chatbook`` package directory.

    Returns:
        ``(stale, reason)``; ``reason`` is a log-ready phrase, empty when fresh.
    """
    css_dir = package_root / "css"
    generated = [
        css_dir / "tldw_cli_modular.tcss",
        css_dir / build_css.WIDGET_DEFAULTS_SELF_FILENAME,
        css_dir / build_css.WIDGET_DEFAULTS_SCOPED_FILENAME,
        css_dir / build_css.SCREEN_CSS_SELF_FILENAME,
        css_dir / build_css.SCREEN_CSS_SCOPED_FILENAME,
        # TASK-25812 (Qodo #2281) / TASK-24459: the per-screen sheets split
        # from the screen-owned modules are generated outputs too -- a
        # missing or stale one must trigger the same rebuild, or visiting
        # that screen loads nothing (the bundle no longer carries its
        # rules). Required only when ALL source modules are part of this
        # tree, mirroring the builders' own skip for partial/scratch
        # checkouts.
        *(
            css_dir / name
            for split in build_css.SCREEN_OWNED_SPLITS
            if all((css_dir / module).is_file() for module in split.modules)
            for name in split.sheets.values()
        ),
    ]
    missing = [path.name for path in generated if not path.is_file()]
    if missing:
        return True, f"generated stylesheet(s) not found: {', '.join(missing)}"

    # Compare against the OLDEST generated sheet: any one of them being behind
    # its sources is enough to require a rebuild.
    oldest = min(path.stat().st_mtime for path in generated)

    # TASK-18910: when the builder's content manifest is present it is
    # AUTHORITATIVE. Each recorded input is mtime-compared first and hashed
    # when its mtime differs from the recorded build time IN EITHER
    # DIRECTION -- which removes the false positives (branch switch /
    # ``git checkout`` / stash pop rewrite mtimes without changing content;
    # each cost a ~0.7 s synchronous rebuild) while still catching content
    # restored with a preserved or backdated timestamp (``cp -p``,
    # rsync -a), which a "newer than the build" test alone would treat as
    # unchanged. It also closes a masking gap the pure-mtime rule had: a
    # pull that brings regenerated sheets (new sheet mtimes) together with
    # a source edit made without a local rebuild never fired, because the
    # edited source was no longer "newer than the build". Inputs whose
    # hash confirms unchanged content have their recorded mtime refreshed
    # so a one-time mtime move does not re-hash on every later boot. No
    # manifest (first boot after the change, or a wheel install) keeps the
    # legacy mtime rule; the manifest self-heals on the next rebuild.
    manifest = _load_css_build_manifest(css_dir)
    if manifest is not None:

        from .Utils.path_validation import validate_path

        def _sha256(path: Path) -> str | None:
            digest = hashlib.sha256()
            try:
                with open(path, "rb") as handle:
                    for chunk in iter(
                        lambda: handle.read(build_css.HASH_CHUNK_SIZE_BYTES), b""
                    ):
                        digest.update(chunk)
            except OSError:
                return None
            return digest.hexdigest()

        # A "newer than the build" reference for the declaration scan below:
        # the newest mtime recorded in the manifest (any input mtime past it
        # is one the build never saw, whether or not it is in the manifest).
        newest_recorded = max(entry[1] for entry in manifest.values())

        manifest_dirty = False
        seen = set()
        for key, entry in sorted(manifest.items()):
            recorded_hash, recorded_mtime = entry[0], entry[1]
            # Manifest keys are joined into filesystem paths; a hand-edited
            # manifest must not be able to point the stat/hash reads outside
            # the package (Qodo security finding on PR #1831).
            try:
                source = validate_path(key, package_root, allow_hidden=True)
            except ValueError:
                return True, f"{key} in the build manifest escapes the package"
            try:
                source_mtime = source.stat().st_mtime
            except OSError:
                # Deleted input: the sheets still carry its rules, so a
                # rebuild is required (the pre-manifest code could not see
                # deletions at all -- see its "Known gap" note).
                return True, f"{key} (recorded in the build manifest) was deleted"
            seen.add(key)
            if source_mtime == recorded_mtime:
                continue  # unchanged since the build; skip hashing
            if _sha256(source) != recorded_hash:
                return True, f"{key} changed since the build"
            # Hash-confirmed unchanged: refresh the recorded mtime so this
            # mtime move is not re-hashed on every subsequent boot.
            manifest[key] = [recorded_hash, source_mtime]
            manifest_dirty = True

        if manifest_dirty:
            _save_css_build_manifest(css_dir, manifest)

        # A module that has GAINED a BUNDLED_CSS declaration since the build
        # is not in the manifest; catch it by scanning declarations in any
        # .py newer than the newest recorded build input. A backdated NEW
        # carrier cannot be distinguished from pre-build files by mtime, so
        # the scan also admits files older than the build when they were
        # not part of the recorded set and sit in a CSS-declaring
        # neighbourhood -- bounded by the manifest's own key set: any .py
        # NOT in the manifest is either new or predates the manifest, and
        # reading it once is cheap relative to a rebuild.
        skip = {"__pycache__", *widget_css.EXCLUDED_DIRS}
        for dirpath, dirnames, filenames in os.walk(package_root):
            dirnames[:] = [name for name in dirnames if name not in skip]
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                source = os.path.join(dirpath, filename)
                key = Path(source).relative_to(package_root).as_posix()
                if key in seen:
                    continue  # already verified above
                try:
                    if os.stat(source).st_mtime <= newest_recorded:
                        continue
                    with open(source, "r", encoding="utf-8", errors="ignore") as handle:
                        text = handle.read()
                except OSError:
                    continue
                if _BUNDLED_CSS_DECLARATION_RE.search(text):
                    return (
                        True,
                        f"{filename} gained a BUNDLED_CSS declaration since the build",
                    )
        return False, ""

    # Legacy path (no manifest): the pre-TASK-18910 mtime rule, unchanged.
    for subdir in ("core", "layout", "components", "features", "utilities"):
        subdir_path = css_dir / subdir
        if not subdir_path.is_dir():
            continue
        for module in subdir_path.glob("*.tcss"):
            if module.stat().st_mtime > oldest:
                return True, f"CSS module {module.name} is newer than the build"

    skip = {"__pycache__", *widget_css.EXCLUDED_DIRS}
    for dirpath, dirnames, filenames in os.walk(package_root):
        # Match the builder's own view of what an input is: `iter_blocks` skips
        # these directories, so a vendored file mentioning the marker must not
        # trigger a rebuild that would then ignore it.
        dirnames[:] = [name for name in dirnames if name not in skip]
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            source = os.path.join(dirpath, filename)
            try:
                if os.stat(source).st_mtime <= oldest:
                    continue
                with open(source, "r", encoding="utf-8", errors="ignore") as handle:
                    text = handle.read()
            except OSError:
                continue  # vanished mid-walk; not our problem to report
            if _BUNDLED_CSS_DECLARATION_RE.search(text):
                return True, f"{filename} carries widget CSS newer than the build"

    return False, ""


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the tldw-cli argument parser (extracted from main_cli_runner() for testability)."""
    parser = argparse.ArgumentParser(
        description="tldw chatbook - A Textual TUI for chatting with LLMs",
        prog="tldw-cli",
    )
    parser.add_argument(
        "--serve", action="store_true", help="Run the application as a web server"
    )
    parser.add_argument(
        "--host", type=str, help="Host address for web server (default: localhost)"
    )
    parser.add_argument("--port", type=int, help="Port for web server (default: 8000)")
    parser.add_argument("--web-title", type=str, help="Title for the web page")
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode for web server"
    )
    parser.add_argument(
        "--focus",
        action="store_true",
        help="Start chrome-free in the Console (hides nav bar and workbench header)",
    )
    return parser


# --- Main execution block ---
def _run_module_main() -> None:
    # Record the launch directory first, before anything can chdir -- the
    # `python -m tldw_chatbook.app` path does not route through
    # main_cli_runner, so it needs its own capture (set-once; harmless if
    # already recorded). See workspace_context_note for why this matters.
    from tldw_chatbook.Tools.workspace_file_roots import (
        set_launch_cwd as _set_launch_cwd,
    )

    _set_launch_cwd()

    # Initialize logging first
    early_logging_app = initialize_early_logging()  # noqa: F841 -- verbatim

    try:
        load_cli_config_and_ensure_existence()
    except Exception as e_cfg_main:
        logging.error(
            f"Could not ensure creation of effective config file: {e_cfg_main}",
            exc_info=True,
        )

    # TASK-26040: persist any pending forward config migration once at boot.
    # A no-op (no lock, no file read) until a real migration is registered.
    try:
        from tldw_chatbook.config import migrate_config_file_if_needed
        migrate_config_file_if_needed()
    except Exception as e_cfg_migrate:
        logging.error(
            f"Config schema migration failed; the original file was left "
            f"untouched: {e_cfg_migrate}",
            exc_info=True,
        )

    # --- Initialize Metrics Systems ---
    # Initialize Prometheus metrics server
    try:
        # Opt-in only: init_metrics_server checks [metrics] enabled before it
        # binds anything, and resolves port/bind address itself (TASK-25914).
        # It previously read METRICS_PORT here with a "8000" fallback, which
        # meant the env default silently overrode a configured port.
        init_metrics_server()
    except Exception as exc:
        loguru_logger.warning(
            "Prometheus metrics initialization failed (exception_type={}).",
            type(exc).__name__,
        )
        # Continue without metrics server - metrics are still collected

    # Initialize OpenTelemetry metrics
    try:
        # Initialize OpenTelemetry for advanced metrics collection
        # This complements the existing Prometheus metrics
        init_otel_metrics()
    except Exception as exc:
        loguru_logger.warning(
            "OpenTelemetry metrics initialization failed (exception_type={}).",
            type(exc).__name__,
        )
        # Continue without OpenTelemetry - the app still has Prometheus metrics

    # --- Emoji Check ---
    emoji_is_supported = supports_emoji()  # Call it once
    loguru_logger.info(f"Terminal emoji support detected: {emoji_is_supported}")
    loguru_logger.info(
        f"Using brain: {get_char(EMOJI_TITLE_BRAIN, FALLBACK_TITLE_BRAIN)}"
    )
    loguru_logger.info("-" * 30)

    # --- CSS File Handling ---
    package_root = Path(__file__).parent
    if _is_source_tree(package_root):
        try:
            css_dir = package_root / "css"
            css_dir.mkdir(exist_ok=True)

            # Check if modular CSS needs to be built
            build_script_path = css_dir / "build_css.py"

            # Check whether any input -- a .tcss module or a Python module
            # carrying BUNDLED_CSS -- has moved on since the last build.
            should_rebuild, reason = _generated_css_is_stale(package_root)  # noqa: RUF059 -- verbatim
            if should_rebuild:
                logging.info("Generated CSS is stale during module entry; rebuilding")

            if should_rebuild and build_script_path.exists():
                logging.info("Building modular CSS...")
                import subprocess

                # Build CSS synchronously before starting the app
                result = subprocess.run(
                    [sys.executable, str(build_script_path)],
                    cwd=str(css_dir),
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    logging.info("Successfully built modular CSS")
                else:
                    logging.error(f"Failed to build modular CSS: {result.stderr}")

        except Exception as e_css_main:
            logging.error(f"Error handling CSS file: {e_css_main}", exc_info=True)

    # --- Check for encrypted config (config will be created if it doesn't exist) ---
    try:
        config_data = load_cli_config_and_ensure_existence()
        encryption_config = config_data.get("encryption", {})

        if encryption_config.get("enabled", False):
            loguru_logger.info("Config file encryption is enabled. Password required.")

            # Import password dialog dependencies here to avoid circular imports
            import asyncio  # noqa: F401 -- verbatim; was a module-scope rebind
            from textual.app import App
            from tldw_chatbook.Widgets.password_dialog import PasswordDialog

            class PasswordPromptApp(App):
                """Minimal app to prompt for password."""

                def __init__(self):
                    super().__init__()
                    self.password = None

                async def on_mount(self) -> None:
                    """Show password dialog immediately on mount."""
                    password = await self.push_screen(
                        PasswordDialog(
                            mode="unlock",
                            title="Unlock Configuration",
                            message="Enter your master password to decrypt the configuration file.",
                            on_submit=lambda p: None,
                            on_cancel=lambda: None,
                        ),
                        wait_for_dismiss=True,
                    )

                    if password:
                        # Verify password
                        from tldw_chatbook.Utils.config_encryption import (
                            config_encryption,
                        )

                        password_verifier = encryption_config.get(
                            "password_verifier", ""
                        )
                        if password_verifier and config_encryption.verify_password(
                            password, password_verifier
                        ):
                            self.password = password
                            self.exit()
                        else:
                            self.notify(
                                "Invalid password. Please try again.", severity="error"
                            )
                            # Re-show the dialog
                            await self.on_mount()
                    else:
                        # User cancelled
                        loguru_logger.error(
                            "Password required but not provided. Exiting."
                        )
                        self.exit()

            # Run the password prompt app
            password_app = PasswordPromptApp()
            password_app.run()

            if password_app.password:
                # Set the password for the session
                set_encryption_password(password_app.password)
                loguru_logger.info("Configuration decrypted successfully.")
            else:
                # Exit if no password provided
                loguru_logger.error("Cannot proceed without decryption password.")
                sys.exit(1)

    except Exception as e:
        loguru_logger.error(f"Error checking config encryption: {e}")
        # Continue without encryption if there's an error

    # task-1650: resolve textual_image's rendering protocol NOW, while the
    # terminal still answers escape queries. Textual takes raw mode in
    # run() below, after which the query silently fails and every image
    # surface degrades to half-cell rendering.
    from .Utils.terminal_utils import warm_up_image_protocol

    warm_up_image_protocol()

    # argparse terminates here on --help (exit 0) and invalid arguments
    # (exit 2), same as the console-script path -- no guard: swallowing
    # SystemExit would print usage and then launch the TUI anyway.
    _main_args = _build_arg_parser().parse_args()

    # task-18908: --serve historically only worked via the console-script
    # entry; this __main__ path parsed the flags and then ignored them,
    # silently binding the config default port. Route them exactly like
    # main_cli_runner does.
    if _main_args.serve:
        from .Web_Server.serve import check_web_server_available, run_web_server

        if not check_web_server_available():
            loguru_logger.error("Web server feature is not available!")
            loguru_logger.error("Install with: pip install tldw_chatbook[web]")
            raise SystemExit(1)

        loguru_logger.info("Starting tldw_chatbook in web server mode")
        run_web_server(
            host=_main_args.host,
            port=_main_args.port,
            title=_main_args.web_title,
            debug=_main_args.debug,
        )
        raise SystemExit(0)

    # task-19561: `python -m tldw_chatbook.app` installed no signal handlers
    # at all, so SIGTERM took the process out with the kernel default -- even
    # more abrupt than the console script's `os._exit(0)`. Both entry points
    # now share one bounded, graceful mechanism.
    install_termination_handlers()

    # task-21100: pending ChaChaNotes migrations replay inside TldwCli's
    # constructor, before anything can paint -- the terminal is the only
    # surface that exists at this phase, so say what the pause is there.
    from tldw_chatbook.Utils.db_upgrade_notice import (
        print_db_upgrade_notice_if_pending,
    )

    print_db_upgrade_notice_if_pending()

    # Create instance with early logging flag
    app_instance = TldwCli()
    app_instance._cli_focus_override = bool(_main_args.focus)
    # Set the early logging flag so _setup_logging knows logging was already initialized
    app_instance._early_logging_initialized = True
    try:
        app_instance.run()
    except KeyboardInterrupt:
        loguru_logger.info("--- KeyboardInterrupt received ---")
    except Exception:
        loguru_logger.exception("--- CRITICAL ERROR DURING app.run() ---")
        traceback.print_exc()  # Make sure traceback prints
    finally:
        # This might run even if app exits early internally in run()
        loguru_logger.info("--- FINALLY block after app.run() ---")
        # Everything from here is interpreter teardown -- `asyncio.run`'s
        # executor join, `threading._shutdown()`, `atexit`. None of it is
        # interruptible from Python, so this is the last point a bound can
        # be placed on it. Idempotent: a SIGTERM-armed watchdog already
        # holds a tighter deadline and this call leaves it alone.
        arm_exit_watchdog(reason="interpreter exit")

    loguru_logger.info("--- AFTER app.run() call (if not crashed hard) ---")


# Entry point for the tldw-chatbook command
def get_app() -> TldwCli:
    """Entry point for textual serve.

    Returns:
        The TldwCli app instance, not yet running.
    """
    # Configure logging to suppress verbose debug messages early

    # Suppress various verbose loggers
    logging.getLogger("torio._extension.utils").setLevel(logging.WARNING)
    logging.getLogger("torio").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("fsevents").setLevel(logging.WARNING)

    # Ensure CSS is built

    package_root = Path(__file__).parent
    if _is_source_tree(package_root):
        css_dir = package_root / "css"
        build_script_path = css_dir / "build_css.py"

        # Same staleness rule as the main entry points: a missing generated
        # sheet, or any input newer than the build (TASK-15450).
        stale, reason = _generated_css_is_stale(package_root)
        if stale and build_script_path.exists():
            print(f"Building modular CSS: {reason}")

            subprocess.run([sys.executable, str(build_script_path)], check=True)

    return TldwCli()


def main_cli_runner() -> object:
    """Entry point for the tldw-chatbook command.

    This function is referenced in pyproject.toml as the entry point for the tldw-chatbook command.
    It initializes logging early and then runs the TldwCli app.

    Returns:
        The app's pending recovery-restart request, or None when there is none
        (``cli.main_cli_runner`` passes it on).
    """
    # Record the launch directory at the earliest point in the process, before
    # anything can chdir. The workspace-context note appended to agent prompts
    # expresses workspace roots relative to this (never as absolute host
    # paths). Set-once: harmless if another entry path already recorded it.
    from tldw_chatbook.Tools.workspace_file_roots import set_launch_cwd

    set_launch_cwd()

    # Configure logging to suppress verbose debug messages early
    import warnings

    # Suppress various verbose loggers
    logging.getLogger("torio._extension.utils").setLevel(logging.WARNING)
    logging.getLogger("torio").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)

    # Suppress torchaudio and FFmpeg warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
    warnings.filterwarnings("ignore", message=".*FFmpeg.*")

    # Set environment variable to suppress FFmpeg output
    os.environ["TORCHAUDIO_LOG_LEVEL"] = "ERROR"

    # task-19561: SIGTERM used to be answered here by `os._exit(0)` from
    # inside the handler, after an `atexit`-registered `force_cleanup` that
    # tried to daemonize already-started threads (a `RuntimeError` every
    # time) and cleared `concurrent.futures.thread._threads_queues` (which
    # only ever robs `_python_exit` of the sentinels that let idle executor
    # threads finish). `os._exit` skipped Textual's `on_unmount` entirely --
    # no database closed, no transaction rolled back, and any row already
    # flipped to `running` stranded there permanently. The handlers below
    # run the ordinary shutdown path and keep a hard exit only as the
    # bounded, last-resort escape. See `Utils/app_shutdown.py`.
    install_termination_handlers()

    # Initialize logging first
    initialize_early_logging()

    try:
        load_cli_config_and_ensure_existence()
    except Exception as e_cfg_main:
        logging.error(
            f"Could not ensure creation of effective config file: {e_cfg_main}",
            exc_info=True,
        )

    # --- Emoji Check ---
    emoji_is_supported = supports_emoji()  # Call it once
    loguru_logger.info(f"Terminal emoji support detected: {emoji_is_supported}")
    loguru_logger.info(
        f"Using brain: {get_char(EMOJI_TITLE_BRAIN, FALLBACK_TITLE_BRAIN)}"
    )
    loguru_logger.info("-" * 30)

    # --- CSS File Handling ---
    package_root = Path(__file__).parent
    if _is_source_tree(package_root):
        try:
            css_dir = package_root / "css"
            css_dir.mkdir(exist_ok=True)

            # Check if modular CSS needs to be built
            build_script_path = css_dir / "build_css.py"

            # Check whether any input -- a .tcss module or a Python module
            # carrying BUNDLED_CSS -- has moved on since the last build.
            should_rebuild, reason = _generated_css_is_stale(package_root)
            if should_rebuild:
                logging.info("Generated CSS is stale during CLI entry; rebuilding")

            if should_rebuild and build_script_path.exists():
                logging.info("Building modular CSS...")

                # Build CSS synchronously before starting the app
                result = subprocess.run(
                    [sys.executable, str(build_script_path)],
                    cwd=str(css_dir),
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    logging.info("Successfully built modular CSS")
                else:
                    logging.error(f"Failed to build modular CSS: {result.stderr}")

        except Exception as e_css_main:
            logging.error(f"Error handling CSS file: {e_css_main}", exc_info=True)

    # Parse command line arguments
    args = _build_arg_parser().parse_args()

    # If --serve flag is provided, run as web server
    if args.serve:
        # Check if web server dependencies are available
        from .Web_Server.serve import check_web_server_available, run_web_server

        if not check_web_server_available():
            loguru_logger.error("\n" + "=" * 60)
            loguru_logger.error("Web server feature is not available!")
            loguru_logger.error("=" * 60)
            loguru_logger.error(
                "\nThe required dependency 'textual-serve' is not installed."
            )
            loguru_logger.error("\nTo install it, run:")
            loguru_logger.error("  pip install tldw_chatbook[web]")
            loguru_logger.error("\nFor development installations:")
            loguru_logger.error('  pip install -e ".[web]"')
            loguru_logger.error("\n" + "=" * 60 + "\n")
            return

        loguru_logger.info("Starting tldw_chatbook in web server mode")
        run_web_server(
            host=args.host, port=args.port, title=args.web_title, debug=args.debug
        )
        return  # Exit after web server stops

    # Otherwise, run as normal TUI app
    # task-1650: resolve textual_image's rendering protocol NOW, while the
    # terminal still answers escape queries. Textual takes raw mode in
    # run() below, after which the query silently fails and every image
    # surface degrades to half-cell rendering.
    from .Utils.terminal_utils import warm_up_image_protocol

    warm_up_image_protocol()

    # task-21100: pending ChaChaNotes migrations replay inside TldwCli's
    # constructor, before anything can paint -- the terminal is the only
    # surface that exists at this phase, so say what the pause is there.
    from .Utils.db_upgrade_notice import print_db_upgrade_notice_if_pending

    print_db_upgrade_notice_if_pending()

    # Create instance with early logging flag
    app_instance = TldwCli()
    app_instance._cli_focus_override = bool(args.focus)
    app_instance._recovery_restart_available = True
    # Set the early logging flag so _setup_logging knows logging was already initialized
    app_instance._early_logging_initialized = True
    recovery_restart_request = None
    try:
        app_instance.run()
        recovery_restart_request = getattr(app_instance, "_recovery_restart_request", None)
    except KeyboardInterrupt:
        loguru_logger.info("--- KeyboardInterrupt received ---")
    except Exception:
        loguru_logger.exception("--- CRITICAL ERROR DURING app.run() ---")
        traceback.print_exc()  # Make sure traceback prints
    finally:
        # This might run even if app exits early internally in run()
        loguru_logger.info("--- FINALLY block after app.run() ---")
        # Bound interpreter teardown (see the identical call in the
        # `__main__` block for why this is the last placeable bound).
        arm_exit_watchdog(reason="interpreter exit")

    loguru_logger.info("--- AFTER app.run() call (if not crashed hard) ---")
    return recovery_restart_request
