#!/usr/bin/env python3
"""Run speculative Console voice from one exact local source checkout.

This file is deliberately outside the application package and excluded from release
artifacts. It changes only process-local UI selection seams after validating an
explicitly selected commit-scoped data profile; packaged installs keep using their configured engine,
qualification manifest, and normal profile without an environment, config, or CLI
override.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.machinery import PathFinder
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
_COMMIT = re.compile(r"[0-9a-f]{40}")
_MAX_SAY_WAV_BYTES = 32 * 1024 * 1024


class DevelopmentVoiceEntryError(RuntimeError):
    """The development launcher is not bound to the requested source checkout."""


@dataclass(frozen=True)
class SourceCheckoutIdentity:
    """Content-free identity printed before the development app starts."""

    root: Path
    branch: str
    head: str
    profile: str


def _git(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ("git", "-C", str(repo_root), *args),
            check=True,
            capture_output=True,
            text=True,
            env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"),
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DevelopmentVoiceEntryError(
            "Could not identify the Git worktree."
        ) from exc
    return result.stdout.strip()


def validate_source_checkout(
    *,
    repo_root: Path,
    package_root: Path,
    expected_head: str,
) -> SourceCheckoutIdentity:
    """Require one non-detached checkout and its matching imported package."""

    root = Path(repo_root).resolve()
    package = Path(package_root).resolve()
    git_marker = root / ".git"
    if not git_marker.exists() or git_marker.is_symlink():
        raise DevelopmentVoiceEntryError("The launcher requires a Git worktree.")
    if _COMMIT.fullmatch(expected_head) is None:
        raise DevelopmentVoiceEntryError("--expect-head requires a full commit hash.")
    if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise DevelopmentVoiceEntryError("The launcher is outside the Git worktree.")
    if package != (root / "tldw_chatbook").resolve():
        raise DevelopmentVoiceEntryError(
            "The imported tldw_chatbook package is not from this worktree."
        )
    head = _git(root, "rev-parse", "HEAD")
    if head != expected_head:
        raise DevelopmentVoiceEntryError(
            f"Checkout HEAD is {head}; expected {expected_head}."
        )
    branch = _git(root, "branch", "--show-current")
    if not branch:
        raise DevelopmentVoiceEntryError("The launcher refuses a detached checkout.")
    return SourceCheckoutIdentity(
        root=root,
        branch=branch,
        head=head,
        profile=f"speculative_voice_dev_{head[:12]}",
    )


async def _render_macos_say_wav(text: str) -> bytes:
    """Render one phrase to bounded PCM WAV with macOS's local synthesizer."""

    if sys.platform != "darwin" or not Path("/usr/bin/say").is_file():
        raise DevelopmentVoiceEntryError("macOS local speech is unavailable.")
    if type(text) is not str or not text.strip():
        raise ValueError("Speech text must be non-empty.")

    descriptor, output_name = tempfile.mkstemp(
        prefix="tldw-speculative-voice-",
        suffix=".wav",
    )
    os.close(descriptor)
    output_path = Path(output_name)
    process: asyncio.subprocess.Process | None = None
    try:
        process = await asyncio.create_subprocess_exec(
            "/usr/bin/say",
            "-o",
            str(output_path),
            "--data-format=LEI16@24000",
            "--file-format=WAVE",
            "--",
            text,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            return_code = await process.wait()
        except asyncio.CancelledError:
            process.terminate()
            await process.wait()
            raise
        if return_code != 0:
            raise RuntimeError("macOS local speech synthesis failed")
        size = output_path.stat().st_size
        if not 44 <= size <= _MAX_SAY_WAV_BYTES:
            raise RuntimeError("macOS local speech returned invalid audio")
        return await asyncio.to_thread(output_path.read_bytes)
    finally:
        if process is not None and process.returncode is None:
            process.terminate()
            await process.wait()
        output_path.unlink(missing_ok=True)


async def _synthesize_macos_say(_self: object, *, text: str):
    """Adapt the local development exporter to the production TTS response seam."""

    from tldw_chatbook.TTS.adapter_types import TTSAudioResponse

    body = await _render_macos_say_wav(text)

    async def stream():
        yield body

    return TTSAudioResponse(
        provider_id="macos_say_development",
        model_id="system_voice",
        audio_format="wav",
        content_type="audio/wav",
        byte_stream=stream(),
        sample_rate=24_000,
        metadata={"channels": 1},
    )


async def _check_macos_say_tts() -> int:
    """Validate the local exporter through the production PCM normalizer."""

    from tldw_chatbook.TTS.pcm_stream import iter_normalized_pcm_frames

    response = await _synthesize_macos_say(None, text="Voice system check.")
    frame_count = 0
    async with response:
        async for _frame in iter_normalized_pcm_frames(
            audio_format=response.audio_format,
            sample_rate=response.sample_rate,
            channels=1,
            byte_stream=response.byte_stream,
        ):
            frame_count += 1
    if frame_count < 10:
        raise DevelopmentVoiceEntryError(
            "macOS local speech produced insufficient normalized audio."
        )
    return frame_count


def _checkout_identity(expected_head: str) -> SourceCheckoutIdentity:
    """Validate source identity without importing the application package."""
    package = sys.modules.get("tldw_chatbook")
    if package is None:
        spec = PathFinder.find_spec("tldw_chatbook", [str(REPO_ROOT)])
        package_file = None if spec is None else spec.origin
    else:
        package_file = getattr(package, "__file__", None)
    expected_package = REPO_ROOT / "tldw_chatbook" / "__init__.py"
    if package_file is None or Path(package_file).resolve() != expected_package:
        raise DevelopmentVoiceEntryError("The package is not from this worktree.")
    identity = validate_source_checkout(
        repo_root=REPO_ROOT,
        package_root=expected_package.parent,
        expected_head=expected_head,
    )
    config = sys.modules.get("tldw_chatbook.config")
    if config is not None:
        config_file = getattr(config, "__file__", None)
        if (
            config_file is None
            or Path(config_file).resolve()
            != (identity.root / "tldw_chatbook" / "config.py").resolve()
        ):
            raise DevelopmentVoiceEntryError(
                "The imported config is not from this worktree."
            )
    return identity


def _check_selected_config(identity: SourceCheckoutIdentity) -> Path:
    """Read the explicit input without creating it or admitting startup."""
    if not os.environ.get("TLDW_CONFIG_PATH", "").strip():
        raise DevelopmentVoiceEntryError(
            "Select the prepared commit profile with TLDW_CONFIG_PATH before launch."
        )
    from tldw_chatbook.Backup_Recovery import bootstrap, profile_paths
    from tldw_chatbook.Backup_Recovery.storage_admission import _read_recovery_file

    selected = profile_paths.effective_config_path()
    allowed, _reason = bootstrap.startup_permission(
        selected, bootstrap.default_bootstrap_root()
    )
    if not allowed:
        return selected
    config = sys.modules.get("tldw_chatbook.config")
    if config is not None:
        # The guarded read must refuse source drift before reading a new selector.
        try:
            snapshot = config.read_cli_config_snapshot()
        except (OSError, ValueError):
            return selected
        if snapshot.path != selected:
            raise DevelopmentVoiceEntryError(
                "The loaded config source is not selected."
            )
    try:
        data = _read_recovery_file("config", selected, max_bytes=16 * 1024**2)
    except FileNotFoundError:
        raise DevelopmentVoiceEntryError(
            "The selected commit config must already exist."
        ) from None
    except (OSError, ValueError, RuntimeError):
        # Ordinary unlock owns unsafe/unreadable input and its recovery outcome.
        return selected
    try:
        document = tomllib.loads(data.decode("utf-8"))
    except (UnicodeError, ValueError):
        return selected
    encryption = document.get("encryption", {})
    if (
        not isinstance(encryption, dict)
        or encryption.get("enabled", False) is not False
    ):
        return selected
    general = document.get("general", {})
    if not isinstance(general, dict):
        return selected
    name = general.get("users_name")
    if isinstance(name, str) and name.startswith("enc:"):
        return selected
    if name != identity.profile:
        raise DevelopmentVoiceEntryError(
            f"The selected config must have general.users_name={identity.profile}."
        )
    return selected


@contextmanager
def development_voice_enabled(
    *, expected_head: str, macos_say_tts: bool = False
) -> Iterator[SourceCheckoutIdentity]:
    """Enable speculative selection only after ordinary startup preparation."""

    original_path = sys.path[:]
    sys.path.insert(0, str(REPO_ROOT))
    try:
        identity = _checkout_identity(expected_head)
        config = sys.modules.get("tldw_chatbook.config")
        if config is None:
            raise DevelopmentVoiceEntryError(
                "Use the source main entry to prepare the selected config first."
            )
        if not os.environ.get("TLDW_CONFIG_PATH", "").strip():
            raise DevelopmentVoiceEntryError("The prepared config must be explicit.")
        from tldw_chatbook.Backup_Recovery import profile_paths

        selected = profile_paths.effective_config_path()
        snapshot = config.read_cli_config_snapshot()
        if (
            config.get_cli_config_path() != selected
            or snapshot.path != selected
            or snapshot.serialized is None
        ):
            raise DevelopmentVoiceEntryError(
                "The prepared config source is not selected."
            )
        installed = config.load_cli_config_and_ensure_existence()
        atomic = config.get_atomic_config_snapshot().values
        for values in (installed, atomic):
            if (
                profile_paths.setting(values, "general", "users_name")
                != identity.profile
            ):
                raise DevelopmentVoiceEntryError(
                    "The loaded commit profile does not match."
                )
        if config.get_user_folder_name() != identity.profile:
            raise DevelopmentVoiceEntryError("The loaded commit profile does not match.")
        # Let the normal owner resolve any durable fallback before comparing paths.
        data_dir = config.get_user_data_dir()
        if any(profile_paths.user_data_dir(values) != data_dir for values in (installed, atomic)):
            raise DevelopmentVoiceEntryError("The loaded profile data paths disagree.")
        if (
            config.get_chachanotes_db_path(ignore_override=True)
            != data_dir / profile_paths.database_leaf("chachanotes_db_path")
        ):
            raise DevelopmentVoiceEntryError(
                "The loaded profile default paths disagree."
            )
        from tldw_chatbook.Chat.console_speculative_voice_session import (
            _LazyHandsFreeTts,
        )
        from tldw_chatbook.UI.Console_Modules import hands_free

        original_gate = hands_free.speculative_voice_qualified
        original_resolver = hands_free.resolve_handsfree_engine
        original_tts = _LazyHandsFreeTts.synthesize_hands_free
        hands_free.speculative_voice_qualified = lambda: True
        hands_free.resolve_handsfree_engine = lambda: "pipeline"
        if macos_say_tts:
            _LazyHandsFreeTts.synthesize_hands_free = _synthesize_macos_say
        try:
            yield identity
        finally:
            _LazyHandsFreeTts.synthesize_hands_free = original_tts
            hands_free.speculative_voice_qualified = original_gate
            hands_free.resolve_handsfree_engine = original_resolver
    finally:
        sys.path[:] = original_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the normal Chatbook UI with speculative voice selected only in "
            "this exact source checkout."
        )
    )
    parser.add_argument(
        "--expect-head",
        required=True,
        help="Full commit hash that the checkout must match before launch.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate and print checkout identity without starting the app.",
    )
    parser.add_argument(
        "--macos-say-tts",
        action="store_true",
        help=(
            "Use the local macOS speech exporter for this guarded development "
            "process without changing saved TTS settings."
        ),
    )
    parser.add_argument(
        "--check-macos-say-tts",
        action="store_true",
        help=(
            "Render a fixed local phrase through the production PCM normalizer "
            "and exit without opening the app."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate, enable the process-local gate, and run the ordinary app entry."""

    args = _parser().parse_args(argv)
    identity = _checkout_identity(args.expect_head)
    if args.check:
        print("mode=source-checkout-only")
        print(f"root={identity.root}")
        print(f"branch={identity.branch}")
        print(f"head={identity.head}")
        print(f"expected_profile={identity.profile}")
        print("profile_admission=not-performed")
        return 0
    original_argv = sys.argv
    original_path = sys.path[:]
    try:
        sys.path.insert(0, str(REPO_ROOT))
        sys.argv = [original_argv[0]]
        from tldw_chatbook.Utils.private_paths import PrivatePathError
        from tldw_chatbook.Utils.startup_errors import emit_private_path_startup_error
        from tldw_chatbook.Utils.startup_logging import quiet_startup_stderr

        quiet_startup_stderr()
        selected = _check_selected_config(identity)
        from tldw_chatbook.Backup_Recovery.launcher import startup_unlock
        from tldw_chatbook.Backup_Recovery.profile_paths import effective_config_path

        stopped = startup_unlock()
        if stopped is not None:
            return stopped
        if effective_config_path() != selected:
            raise DevelopmentVoiceEntryError("The selected config changed during startup.")
        __import__("tldw_chatbook.config")
        from tldw_chatbook.cli import _run_prepared_app

        with development_voice_enabled(
            expected_head=args.expect_head,
            macos_say_tts=args.macos_say_tts or args.check_macos_say_tts,
        ) as identity:
            print("mode=source-checkout-only")
            print("engine=speculative-pipeline")
            print(f"root={identity.root}")
            print(f"branch={identity.branch}")
            print(f"head={identity.head}")
            print(f"profile={identity.profile}")
            print(
                "tts="
                f"{'macos-say-local' if args.macos_say_tts or args.check_macos_say_tts else 'configured'}"
            )
            if args.check_macos_say_tts:
                print(f"tts_normalized_frames={asyncio.run(_check_macos_say_tts())}")
                return 0
            result = _run_prepared_app()
            return result if isinstance(result, int) else 0
    except PrivatePathError as exc:
        emit_private_path_startup_error(exc)
        raise SystemExit(1) from None
    finally:
        sys.argv = original_argv
        sys.path[:] = original_path


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DevelopmentVoiceEntryError as exc:
        print(f"speculative voice development entry refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
