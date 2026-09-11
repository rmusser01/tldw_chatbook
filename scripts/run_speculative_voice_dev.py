#!/usr/bin/env python3
"""Run speculative Console voice from one exact local source checkout.

This file is deliberately outside the application package and excluded from release
artifacts. It changes only process-local UI selection seams and selects a
commit-scoped data profile; packaged installs keep using their configured engine,
qualification manifest, and normal profile without an environment, config, or CLI
override.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


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


@contextmanager
def development_voice_enabled(
    *, expected_head: str, macos_say_tts: bool = False
) -> Iterator[SourceCheckoutIdentity]:
    """Enable speculative selection only for this guarded process lifetime."""

    original_path = sys.path[:]
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import tldw_chatbook

        package_file = getattr(tldw_chatbook, "__file__", None)
        if package_file is None:
            raise DevelopmentVoiceEntryError("The imported package has no source path.")
        identity = validate_source_checkout(
            repo_root=REPO_ROOT,
            package_root=Path(package_file).resolve().parent,
            expected_head=expected_head,
        )
        from tldw_chatbook import config
        from tldw_chatbook.Chat.console_speculative_voice_session import (
            _LazyHandsFreeTts,
        )
        from tldw_chatbook.UI.Console_Modules import hands_free

        original_gate = hands_free.speculative_voice_qualified
        original_resolver = hands_free.resolve_handsfree_engine
        original_profile = config.get_user_folder_name
        original_tts = _LazyHandsFreeTts.synthesize_hands_free
        hands_free.speculative_voice_qualified = lambda: True
        hands_free.resolve_handsfree_engine = lambda: "pipeline"
        config.get_user_folder_name = lambda: identity.profile
        if macos_say_tts:
            _LazyHandsFreeTts.synthesize_hands_free = _synthesize_macos_say
        try:
            yield identity
        finally:
            _LazyHandsFreeTts.synthesize_hands_free = original_tts
            config.get_user_folder_name = original_profile
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
        if args.check:
            return 0
        from tldw_chatbook.cli import main_cli_runner

        original_argv = sys.argv
        try:
            sys.argv = [original_argv[0]]
            result = main_cli_runner()
        finally:
            sys.argv = original_argv
        return result if isinstance(result, int) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DevelopmentVoiceEntryError as exc:
        print(f"speculative voice development entry refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
