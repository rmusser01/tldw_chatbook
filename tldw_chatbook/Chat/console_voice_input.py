"""Headless Console voice-dictation controller.

Deliberately free of Textual imports: the widget layer owns rendering and
threading policy, this module owns availability, provider resolution, and the
dictation state machine. That split is what makes the state machine unit
testable without a running app.
"""

from __future__ import annotations

# Detection itself lives in `Utils/local_stt_providers`, but these two stay
# imported here because they are the seams the provider-detection tests patch
# (`monkeypatch.setattr(cvi.importlib.util, "find_spec", ...)` and
# `monkeypatch.setattr(cvi.sys, "platform", ...)` both mutate the real module
# objects, which the detection helpers then read).
import importlib.util  # noqa: F401 - patched seam; see comment above
import string
import sys  # noqa: F401 - patched seam; see comment above
import threading
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

from ..config import get_cli_setting

# The provider catalogue lives in `Utils/local_stt_providers.py`, not here: the
# dictation service's privacy allowlist consumes the identical tuple, and when
# these were two separate lists they drifted twice -- once on a misspelled id,
# once on this module growing from three providers to seven while the service's
# allowlist stayed at three. Re-exported under the old names because they are
# this module's published surface (and the seams tests monkeypatch).
from ..Utils.local_stt_providers import (  # noqa: F401 - re-exported API
    DARWIN_ONLY_PROVIDERS,
    LOCAL_PROVIDER_MODULES,
    LOCAL_STT_PROVIDERS,
    installed_local_providers,
    module_installed as _module_installed,
    provider_installed as _provider_installed,
)

# Capture backends, in preference order. AudioRecordingService picks between
# them itself; we only need to know whether at least one exists.
CAPTURE_MODULES: tuple[str, ...] = ("pyaudio", "sounddevice")

# `AudioRecordingService.__init__` raises `AudioRecordingError` unconditionally
# when NumPy is missing (`Audio/recording_service.py:127`), regardless of which
# backend was chosen -- so a backend resolving is not sufficient for capture to
# actually work. Without this, `capture_available()` could report a live
# microphone that deterministically fails the instant `start()` builds the
# service.
CAPTURE_REQUIRED_MODULES: tuple[str, ...] = ("numpy",)

CAPTURE_REASON = "No microphone backend installed."
CAPTURE_REMEDY = (
    "Microphone support isn't installed. "
    "Install with: pip install 'tldw_chatbook[speech_recording]'"
)
PROVIDER_REASON = "No speech-to-text provider installed."
PROVIDER_REMEDY = (
    "No speech-to-text provider installed. "
    "Install with: pip install 'tldw_chatbook[transcription_faster_whisper]'"
)

# --- Model preparation (warm-up) ------------------------------------------
#
# `TranscriptionService` has no preload API: the *only* way to make it load a
# model is to transcribe something. It also keeps `_model_cache` per instance
# (`transcription_service.py`), and the Console builds a fresh service on every
# press, so every capture pays a model load -- a 1.4 GB download on a fresh
# machine, ~1 s from disk afterwards. Doing that lazily on the first audio
# chunk means the load happens *while the user is already speaking* and behind
# the stop-side thread join, which is how a perfect capture came back as "No
# audio was captured from the microphone." Warming here moves the whole cost
# into `preparing`, before the microphone is ever opened.
WARMUP_SAMPLE_RATE = 16_000
WARMUP_CHANNELS = 1
WARMUP_SAMPLE_WIDTH = 2
WARMUP_DURATION_MS = 500
#: Half a second of digital silence: long enough that every provider's own
#: framing accepts it, short enough that a warm load stays imperceptible.
WARMUP_PCM = b"\x00" * (
    WARMUP_SAMPLE_RATE * WARMUP_CHANNELS * WARMUP_SAMPLE_WIDTH * WARMUP_DURATION_MS
    // 1000
)
# The chip these render into is `VOICE_CHIP_MAX_WIDTH = 42` cells and exactly
# one row high, so anything longer is silently cut mid-sentence -- an earlier
# draft of this ended on "...(first run may" and the duration warning, the only
# reason a separate first-run string exists, never reached anyone. Keep both
# under WARMUP_MESSAGE_MAX_CELLS *including* the chip's "◌ " prefix; the long
# explanation goes in WARMUP_DETAIL_FIRST_RUN, which the screen shows as a
# toast where it has room to be read.
WARMUP_MESSAGE_MAX_CELLS = 40
#: Shown in the chip while the model loads for the first time in this process.
WARMUP_MESSAGE_FIRST_RUN = "Preparing speech model…"
#: Shown on later presses, where the model is already on disk and only the
#: per-instance load remains (~1 s measured).
WARMUP_MESSAGE = "Loading speech model…"
#: The part that does not fit in a 42-cell chip. Deliberately open-ended about
#: duration: the real cost depends on the model and the network, and a
#: hard-coded number would be a promise this cannot keep.
WARMUP_DETAIL_FIRST_RUN = (
    "Preparing the speech model for the first time. The first run downloads it "
    "and can take several minutes. Nothing is being recorded yet — the "
    "microphone opens once the model is ready."
)
WARMUP_REASON_TEMPLATE = "The '{provider}' speech model could not be prepared."
WARMUP_REMEDY = (
    "Check the transcription provider and model in Settings, and that the "
    "model can be downloaded or is already on disk."
)
#: A warm-up that fails *after* the model loaded (the silence transcription
#: itself erroring) is weak evidence: it costs the user nothing to try the real
#: capture, and Fix 2/3 mean a degraded capture can no longer be misreported as
#: a dead microphone. Fatal warm-up is reserved for a transcription service that
#: cannot be built at all.
WARMUP_DEGRADED_REASON_TEMPLATE = (
    "Could not pre-load the '{provider}' speech model; the first part of this "
    "capture may be slow."
)

# --- Stop-side outcomes ----------------------------------------------------
#
# The transcription worker was still running when the service's join expired,
# so the transcript is not empty because the microphone was silent -- it is
# empty because the work never finished. Saying "no audio was captured" here
# blames the one component that was demonstrably working.
TRANSCRIPTION_INCOMPLETE_REASON = "Transcription did not finish before dictation stopped."
TRANSCRIPTION_INCOMPLETE_REMEDY = (
    "The speech model was still running. Try a shorter capture or a faster "
    "model, or raise dictation.stop_join_timeout_seconds."
)
NO_CAPTURE_MESSAGE = "No audio was captured from the microphone."
NO_SPEECH_MESSAGE = "Transcription returned no speech."


@dataclass(frozen=True)
class Availability:
    """Whether dictation can run, and what to do about it if not."""

    ok: bool
    kind: str = "ok"  # "ok" | "missing-capture" | "missing-provider"
    reason: str = ""
    remedy: str = ""


def capture_available() -> bool:
    """Return True when at least one audio capture backend is installed.

    Returns:
        True when a `CAPTURE_MODULES` backend resolves AND every module in
        `CAPTURE_REQUIRED_MODULES` (NumPy) also resolves -- a backend alone
        is not enough, since `AudioRecordingService` refuses to construct
        without NumPy no matter which backend it picked.
    """
    if not all(_module_installed(name) for name in CAPTURE_REQUIRED_MODULES):
        return False
    return any(_module_installed(name) for name in CAPTURE_MODULES)


def probe() -> Availability:
    """Report whether dictation is usable, distinguishing the two failures.

    Returns:
        `Availability(ok=True)` when a capture backend and a transcription
        provider are both present; otherwise `ok=False` with `kind` set to
        `"missing-capture"` or `"missing-provider"` and a UI-ready
        `reason`/`remedy` pair for whichever is absent.
    """
    if not capture_available():
        logger.debug("Console dictation unavailable: no capture backend")
        return Availability(
            ok=False,
            kind="missing-capture",
            reason=CAPTURE_REASON,
            remedy=CAPTURE_REMEDY,
        )
    if not installed_local_providers():
        logger.debug("Console dictation unavailable: no transcription provider")
        return Availability(
            ok=False,
            kind="missing-provider",
            reason=PROVIDER_REASON,
            remedy=PROVIDER_REMEDY,
        )
    return Availability(ok=True)


DEFAULT_LANGUAGE = "en"

#: The provider `resolve()` picks a dictation-specific fast model for when
#: `dictation.model` is unset -- see `_dictation_model_override` and the
#: `model` block in `resolve()` below.
DICTATION_FAST_MODEL_PROVIDER = "faster-whisper"
#: Measured on real hardware (a loaded machine, one short "console stop" WAV):
#: the transcription stack's own default faster-whisper model,
#: `distil-large-v3`, took 11.47s to transcribe it -- dead on arrival for a
#: spoken command, and with zero feedback while it ran (see
#: `VoiceSegmentTranscribing`). `base` measured 1.43s on the identical WAV,
#: transcribing it correctly. `dictation.model`, when set, always wins over
#: this; this is only the *unset* default for a dictation capture, and it
#: never changes what the transcription stack itself uses elsewhere.
DICTATION_FAST_MODEL_DEFAULT = "base"


@dataclass(frozen=True)
class EffectiveConfig:
    """The transcription settings dictation will actually run with."""

    provider: str
    model: str | None
    language: str
    configured_provider: str
    was_overridden: bool
    #: True when `model` is NOT what `transcription.default_model` would have
    #: produced -- an explicit `dictation.model` override; (when that key is
    #: unset and `provider` resolved to `DICTATION_FAST_MODEL_PROVIDER`) the
    #: dictation-specific fast default; or (when that key is unset and
    #: `provider` resolved to anything else) `configured_model` naming
    #: something that is being deliberately discarded in favor of `None` --
    #: see the `model` block in `resolve()`: a non-faster-whisper provider
    #: never inherits `transcription.default_model`, since that value may
    #: name a model belonging to an entirely different provider (a Whisper
    #: model handed to parakeet-mlx 404s trying to load it as a HuggingFace
    #: repo). Mirrors `was_overridden`'s provenance role, scoped to the model
    #: rather than the provider: `was_overridden` means "the configured
    #: provider wasn't available," this means "dictation chose a different
    #: model than the transcription stack on purpose," which is a distinct
    #: reason and never itself a failure.
    model_overridden_for_dictation: bool = False
    #: `transcription.default_model`'s raw configured value, independent of
    #: what `model` ended up being -- the transcription stack's own answer to
    #: "what model would this be without dictation's fast-default policy."
    #: `None` when nothing is configured there.
    configured_model: str | None = None
    #: True only for the fast-default branch specifically (unset
    #: `dictation.model`, `provider` resolved to `DICTATION_FAST_MODEL_PROVIDER`)
    #: AND `configured_model` names something other than
    #: `DICTATION_FAST_MODEL_DEFAULT` -- i.e. the fast default actually
    #: displaced a value the user configured elsewhere, not merely a bare
    #: default winning over an equally-unconfigured slot. False for an
    #: explicit `dictation.model` override (the user's own deliberate choice
    #: needs no advisory) and false when there was nothing configured to
    #: displace. Drives `VoiceDictationModelDefaulted` (review finding L1).
    fast_default_displaced_configured_model: bool = False


def _dictation_model_override() -> str | None:
    """Read `dictation.model`, the dictation-specific model override.

    Same warn+fallback shape as this module's other config readers (see
    `command_prefix()`/`warm_before_capture_enabled()`): a blank or
    whitespace-only value is treated as unset, not an error, and a non-string
    value is logged and ignored rather than propagated as a confusing type
    into `EffectiveConfig.model`.

    Returns:
        The configured model name, stripped, or `None` when unset, blank, or
        not a string.
    """
    raw = get_cli_setting("dictation", "model", None)
    if raw is None:
        return None
    if not isinstance(raw, str):
        logger.warning(
            "dictation.model must be a string (got {!r}); ignoring",
            raw,
        )
        return None
    value = raw.strip()
    return value or None


def resolve() -> EffectiveConfig | None:
    """Choose the provider before the dictation service gets the chance.

    `LazyLiveDictationService._initialize_streaming_transcriber` rewrites the
    provider to `parakeet-mlx` whenever privacy mode is on and the chosen
    provider is not local -- silently, and to an Apple-Silicon-only provider.
    This function only ever returns something from `LOCAL_STT_PROVIDERS`, which
    is the identical tuple that privacy check consumes, so that branch cannot
    fire on anything this returns.

    That guarantee is structural, not incidental: it held by luck when both
    lists happened to name the same three providers, and broke the moment this
    module's catalogue grew to seven while the service's allowlist stayed at
    three -- the Console warmed and announced one model, then transcribed with
    another. Both now read `Utils/local_stt_providers.LOCAL_STT_PROVIDERS`.

    Model resolution is provider-scoped and never inherits
    `transcription.default_model` across providers: `dictation.model` wins
    when set (for any provider); failing that, a `DICTATION_FAST_MODEL_PROVIDER`
    (`faster-whisper`) resolution defaults to `DICTATION_FAST_MODEL_DEFAULT`
    rather than inheriting whatever (potentially much slower) model the
    transcription stack is configured with (see `_dictation_model_override`'s
    docstring for the measured numbers behind that default); and every other
    provider gets `None`, letting that provider's own transcription path load
    its own default. `transcription.default_model` is a single, provider-
    agnostic config key, so its value belongs to whichever provider the
    transcription stack itself is configured for -- handing that name to a
    *different* resolved provider is not a fallback, it is a wrong argument:
    parakeet-mlx asked to load a Whisper model name (e.g. `distil-large-v3`)
    tries to fetch it as a HuggingFace repo and 404s, which previously killed
    the capture outright (live-reproduced). `None` flows through
    `warm_transcription_model()`/`LazyLiveDictationService` to
    `TranscriptionService.transcribe_buffer()`/`create_streaming_transcriber()`
    unchanged -- both already do `model or <their own default>`, which is
    exactly the "no model given" case they use for direct calls, so this
    needs no matching change on that side.

    Returns:
        The settings to run with, or None when no local provider is installed.
    """
    installed = installed_local_providers()
    if not installed:
        return None

    # Key names matter and are easy to get wrong: the [transcription] section
    # uses `default_provider`/`default_model`/`default_language` (config.py:3333),
    # and the raw TOML section `STTSettings` is stored in the loaded config under
    # `STT_settings` (config.py:1548). Reading `provider`/`model`/`language` or
    # `STTSettings` silently returns the default and defeats this whole function.
    configured = get_cli_setting(
        "transcription", "default_provider", None
    ) or get_cli_setting("STT_settings", "default_stt_provider", "")
    configured = str(configured or "")

    if configured in installed:
        provider = configured
    else:
        # Preference order is LOCAL_PROVIDER_MODULES' declaration order.
        provider = installed[0]
        if configured:
            logger.info(
                "Console dictation provider '{}' unavailable; using '{}'",
                configured,
                provider,
            )

    # `dictation.model` (when set) always wins; failing that, a
    # `faster-whisper` resolution gets a dictation-specific fast default
    # rather than inheriting `transcription.default_model` (typically
    # distil-large-v3, measured ~11.5s per short segment on a loaded machine
    # -- see `DICTATION_FAST_MODEL_DEFAULT`). Every OTHER provider gets
    # `None`: `transcription.default_model` is not scoped to any particular
    # provider, so it may well name a model that belongs to a provider other
    # than the one that just resolved (see the `model` docstring block above
    # `resolve()` and `EffectiveConfig.model_overridden_for_dictation`).
    #
    # Read once, up front: needed both to decide whether the `else` branch's
    # `None` is itself a displacement (`model_overridden_for_dictation`) and,
    # in the fast-default branch, to tell "displaced a value the user
    # actually configured" apart from "there was nothing there to displace"
    # (`fast_default_displaced_configured_model`, review finding L1).
    configured_model_raw = get_cli_setting("transcription", "default_model", None)
    configured_model = str(configured_model_raw) if configured_model_raw else None

    dictation_model = _dictation_model_override()
    fast_default_displaced_configured_model = False
    if dictation_model is not None:
        model = dictation_model
        model_overridden_for_dictation = True
    elif provider == DICTATION_FAST_MODEL_PROVIDER:
        model = DICTATION_FAST_MODEL_DEFAULT
        model_overridden_for_dictation = True
        fast_default_displaced_configured_model = bool(
            configured_model and configured_model != DICTATION_FAST_MODEL_DEFAULT
        )
    else:
        # Not `configured_model`: see the provider-scoping note above. `None`
        # lets the provider's own transcription path pick its own default
        # (`TranscriptionService.transcribe_buffer()`/
        # `create_streaming_transcriber()` both already do
        # `model or <their own default>`).
        model = None
        model_overridden_for_dictation = configured_model is not None

    language = get_cli_setting("transcription", "default_language", DEFAULT_LANGUAGE)

    return EffectiveConfig(
        provider=provider,
        model=model,
        language=str(language or DEFAULT_LANGUAGE),
        configured_provider=configured,
        was_overridden=bool(configured) and provider != configured,
        model_overridden_for_dictation=model_overridden_for_dictation,
        configured_model=configured_model,
        fast_default_displaced_configured_model=fast_default_displaced_configured_model,
    )


STATE_UNAVAILABLE = "unavailable"
STATE_IDLE = "idle"
STATE_PREPARING = "preparing"
STATE_LISTENING = "listening"
STATE_FINISHING = "finishing"
STATE_ERROR = "error"


@dataclass(frozen=True)
class VoicePartial:
    """In-flight recognizer text; superseded by the next partial or final."""

    text: str


@dataclass(frozen=True)
class VoiceFinal:
    """A segment the recognizer finalized on the silence threshold."""

    text: str


@dataclass(frozen=True)
class VoiceSegmentTranscribing:
    """The silence gate closed a segment; its transcription is starting or done.

    Recognizer-driven, exactly like `VoicePartial`: fired from
    `LazyLiveDictationService._transcribe_segment_audio`, on the processing
    thread, TWICE per segment -- `done=False` right before the call that can
    take seconds, `done=True` right after it returns, unconditionally
    (`dictation_service_lazy.py`'s module docstring has the measured
    latencies, and `_transcribe_segment_audio`'s docstring has the
    unconditional-completion rationale). Under the segment-at-silence
    architecture there is otherwise no signal at all in that gap -- no live
    partial text, nothing -- so without the `done=False` half a multi-second
    pause between the silence pause and the next `VoiceFinal`/`VoiceCommand`
    looks identical to a dead capture.

    The `done=True` half exists for a narrower but load-bearing reason
    (review finding M1): a segment that transcribes to blank/whitespace --
    routine for room noise or a too-short VAD sliver -- fires neither
    `VoicePartial` nor `VoiceFinal`, so without an unconditional completion
    signal a consumer that shows a transcribing indication on `done=False`
    and clears it on the next final/command/state-change would have nothing
    to clear it on; the indication would stick for the rest of the capture,
    claiming work is in flight when it is not. Consumers therefore clear the
    indication on `done=True` OR on the next final/command/state-change,
    whichever comes first; see `ConsoleComposerBar.set_voice_segment_transcribing`.

    Carries no payload beyond `done` -- there is nothing else to say.

    Not proof the recognizer produced anything: it only proves the silence
    gate fired (and, for `done=True`, that the transcription call returned).
    `ConsoleStreamingDictationSession._handle_event` deliberately does NOT
    set `_heard_recognizer_output` for either half of this event -- see that
    method's docstring for why the distinction matters for the silent-capture
    messaging in `stop_and_transcribe()`.

    Attributes:
        done: False for the "started" signal, True for the "completed"
            signal (fired unconditionally, blank result or not).
    """

    done: bool = False


@dataclass(frozen=True)
class VoiceCommand:
    """A finalized segment that matched the spoken-command grammar.

    Kept as dumb as its `VoiceFinal`/`VoicePartial` siblings: it will later
    cross a thread boundary through the same `post_message` wrappers, so it
    carries nothing beyond the resolved command name.

    Attributes:
        name: One of `COMMAND_PHRASES`' values (e.g. `"send"`,
            `"new-paragraph"`).
    """

    name: str


@dataclass(frozen=True)
class VoiceStateChanged:
    """The controller's state machine transitioned to `state`."""

    state: str


@dataclass(frozen=True)
class VoiceFailed:
    """Dictation could not proceed; `reason`/`remedy` are UI-ready text."""

    reason: str
    remedy: str = ""


@dataclass(frozen=True)
class VoiceProviderOverridden:
    """The `configured` provider was unavailable; `effective` is what ran instead."""

    configured: str
    effective: str


@dataclass(frozen=True)
class VoiceDictationModelDefaulted:
    """Dictation's fast-model default (`effective`) displaced `configured`.

    Emitted only when `EffectiveConfig.fast_default_displaced_configured_model`
    is True: `dictation.model` was unset, the resolved provider is
    `DICTATION_FAST_MODEL_PROVIDER`, and the transcription stack has its own
    `transcription.default_model` configured to something else. A user who
    deliberately set that value otherwise gets `DICTATION_FAST_MODEL_DEFAULT`
    with no runtime signal at all (review finding L1) -- the only prior
    disclosure was a comment in the config guide. Mirrors
    `VoiceProviderOverridden`'s shape and its two-tier once-per-run latch
    (per-controller `_model_default_announced`, app-instance
    `_console_dictation_model_default_notified`).

    Attributes:
        configured: `transcription.default_model`'s raw configured value.
        effective: The model dictation is actually using instead
            (`DICTATION_FAST_MODEL_DEFAULT`).
    """

    configured: str
    effective: str


@dataclass(frozen=True)
class VoiceModelPreparing:
    """The speech model is loading, before the microphone opens.

    Emitted from `preparing`, so the UI can say *why* it is sitting there --
    a first-run model download is minutes long and looks exactly like a hang
    otherwise.

    Attributes:
        message: Chip-sized status text, short enough to paint whole in the
            composer's one-row voice chip.
        detail: The longer explanation, for a surface with room to show it.
            Empty when there is nothing more to say than `message`.
        first_run: True when this process has not loaded this model before.
    """

    message: str
    detail: str = ""
    first_run: bool = False


@dataclass(frozen=True)
class VoiceModelWarmupFailed:
    """Pre-loading the model failed, but the capture is going ahead anyway.

    Not a `VoiceFailed`: nothing has gone wrong with the microphone or with the
    session, and treating it as fatal would make one transient error render
    dictation permanently unusable -- the Console warms on *every* press.
    """

    reason: str
    remedy: str = ""


@dataclass(frozen=True)
class VoiceVadUnavailable:
    """The capture started, but the recorder's voice-activity detection did not.

    Dictation itself still works -- the recorder falls back to forwarding
    every frame -- but a segment then only finalizes when the capture stops,
    so a command spoken mid-capture (see `COMMAND_PHRASES`) can never fire.
    """


#: Shown once per app run when `VoiceVadUnavailable` fires (see
#: `ConsoleVoiceInputController._maybe_report_vad_unavailable`). Not spoken:
#: the microphone is open when this would need to be said, and speaking over
#: an open mic is exactly what `spoken_feedback` avoids everywhere else.
VAD_UNAVAILABLE_MESSAGE = (
    "Voice input is degraded: voice-activity detection (webrtcvad) is not "
    "installed in this Python environment. NOTHING will appear until you "
    "press stop, and voice commands will not fire. Run the app from a "
    "Python with the speech_recording extras installed."
)


@dataclass(frozen=True)
class CaptureOutcome:
    """What the dictation service reported about a finished capture.

    Exists so an empty transcript can be attributed correctly instead of
    always being blamed on the microphone.

    Attributes:
        captured_bytes: PCM bytes the recorder actually delivered, or None
            when the service did not say (older/fake services).
        transcription_complete: False when the service's processing thread was
            still working at the moment its join expired, so anything still in
            flight was dropped.
    """

    captured_bytes: int | None = None
    transcription_complete: bool = True


# --- Spoken-command grammar -------------------------------------------------
#
# A finalized segment is either a command ("Console, send.") or dictated text
# -- there is no third option, and an ambiguous segment always resolves to
# text. Whole-segment match only, against the *normalized* segment: matching
# on a substring or a prefix would make "Console send button is broken" (a
# sentence a user might actually dictate) fire the send command.
DEFAULT_COMMAND_PREFIX = "console"

#: Normalized phrase (after `normalize_spoken`, prefix stripped) -> command
#: name. Command names are kebab-case, independent of the spoken phrasing.
COMMAND_PHRASES: dict[str, str] = {
    "new paragraph": "new-paragraph",
    "new line": "new-line",
    "stop": "stop",
    "send": "send",
    "discard": "discard",
    "read that back": "read-that-back",
    "new session": "new-session",
}

#: Recognizer mis-hearings observed on real hardware, mapped to the phrase
#: the speaker actually said. Consulted ONLY after an exact `COMMAND_PHRASES`
#: match fails, and only for the whole prefix-stripped remainder -- so the
#: fail-open rule ("ambiguous resolves to text") still governs everything not
#: literally in this table. Every entry must name the incident that earned
#: it; do not add speculative homophones, each one widens what can no longer
#: be dictated as text.
MISHEARD_PHRASES: dict[str, str] = {
    # parakeet-mlx, live 2026-07-31: the user's spoken "stop" repeatedly
    # transcribed as "dot" ("Console dot.").
    "dot": "stop",
}

#: Mis-heard prefix variants, same evidence rule as `MISHEARD_PHRASES`.
#: parakeet-mlx, live 2026-07-31: "console" transcribed as "consoles"
#: ("Consoles. Stop.").
MISHEARD_PREFIXES: tuple[str, ...] = ("consoles",)


def normalize_spoken(text: str) -> str:
    """Fold recognizer output down to the shape the grammar matches against.

    Lowercases, removes ALL punctuation, and collapses whitespace.
    Punctuation is stripped entirely rather than just trimmed from the ends:
    recognizers commonly emit an internal comma after the command prefix
    ("Console, send."), and preserving it would mean that -- the single most
    natural phrasing -- could never match.

    "All punctuation" means every character whose `unicodedata.category`
    starts with `"P"` (the Unicode punctuation categories: Po, Pc, Pd, Ps,
    Pe, Pi, Pf), plus everything in ASCII `string.punctuation` as a belt.
    The two are not the same set: several ASCII punctuation characters
    (`$+<=>^`|~`) are Unicode Symbol characters (category `S`), not
    Punctuation, so `unicodedata` alone would miss them. Conversely, plain
    `string.punctuation` alone would miss the Unicode marks Whisper-family
    recognizers actually emit -- right single quote U+2019, em dash U+2014,
    ellipsis U+2026 -- so a hesitant "Console… send" or a curly-quoted
    "Console, 'send'" would fail open to text and the command would silently
    never fire.

    Args:
        text: Raw recognizer output for one finalized segment.

    Returns:
        The normalized text, e.g. `"Console, send."` -> `"console send"`,
        `"Console… send"` -> `"console send"`.
    """
    lowered = text.lower()
    kept = [
        ch
        for ch in lowered
        if not (unicodedata.category(ch).startswith("P") or ch in string.punctuation)
    ]
    return " ".join("".join(kept).split())


def command_prefix() -> str:
    """Return the configured wake phrase that precedes every voice command.

    Reads `dictation.command_prefix`. A blank (empty or whitespace-only)
    value is treated the same as unset, since it would otherwise make every
    normalized segment match every command phrase.

    Returns:
        The configured prefix, lowercased and stripped, or
        `DEFAULT_COMMAND_PREFIX` when unset or blank.
    """
    configured = get_cli_setting("dictation", "command_prefix", None)
    prefix = str(configured or "").strip().lower()
    return prefix or DEFAULT_COMMAND_PREFIX


def classify_segment(text: str) -> "VoiceCommand | VoiceFinal":
    """Classify one finalized segment as a spoken command or dictated text.

    Matches only the whole normalized segment against
    `f"{command_prefix()} {phrase}"` for each phrase in `COMMAND_PHRASES`.
    Anything else -- including a correctly prefixed typo, or the prefix
    followed by unrelated words -- fails open to `VoiceFinal` with the
    original, unmodified text, since misrecognizing dictated text as a
    command silently discards what the user said.

    Exact matches are tried first. When none hits, the whole segment is
    retried against `MISHEARD_PHRASES`/`MISHEARD_PREFIXES` -- curated,
    incident-backed recognizer mis-hearings ("Console dot." for "console
    stop") -- still whole-segment-only, so ordinary prose can no more match
    an alias than it could match the real phrase.

    Args:
        text: Raw recognizer output for one finalized segment.

    Returns:
        A `VoiceCommand` when the segment matches the grammar exactly,
        otherwise a `VoiceFinal` carrying `text` unchanged.
    """
    normalized = normalize_spoken(text)
    configured_prefix = command_prefix()
    prefixes = (configured_prefix, *(
        alias for alias in MISHEARD_PREFIXES if configured_prefix == DEFAULT_COMMAND_PREFIX
    ))
    for prefix in prefixes:
        marker = f"{prefix} "
        if not normalized.startswith(marker):
            continue
        remainder = normalized[len(marker):]
        name = COMMAND_PHRASES.get(remainder)
        if name is None:
            corrected = MISHEARD_PHRASES.get(remainder)
            if corrected is not None:
                name = COMMAND_PHRASES.get(corrected)
        if name is not None:
            return VoiceCommand(name)
    return VoiceFinal(text)


#: (provider, model) pairs already warmed *in this process*. Only drives which
#: preparing message is shown -- the load itself is repeated every press,
#: because `TranscriptionService._model_cache` is per instance and the Console
#: builds a fresh service each time.
_WARMED_MODELS: set[tuple[str, str]] = set()
_WARMED_MODELS_LOCK = threading.Lock()


def _warmup_key(effective: EffectiveConfig) -> tuple[str, str]:
    return (effective.provider, effective.model or "")


def _is_first_warmup(effective: EffectiveConfig) -> bool:
    """Return True when this provider/model has not been warmed in this run."""
    with _WARMED_MODELS_LOCK:
        return _warmup_key(effective) not in _WARMED_MODELS


def _mark_warmed(effective: EffectiveConfig) -> None:
    """Record that this provider/model has now been loaded at least once."""
    with _WARMED_MODELS_LOCK:
        _WARMED_MODELS.add(_warmup_key(effective))


def reset_model_warmup_state() -> None:
    """Forget which models have been warmed. For tests only."""
    with _WARMED_MODELS_LOCK:
        _WARMED_MODELS.clear()


def warmup_target(service: Any) -> Any | None:
    """Return the transcription service a dictation service will really use.

    Warming a throwaway `TranscriptionService()` would only prime the on-disk
    HuggingFace cache; the ~1 s per-instance load would still land on the first
    audio chunk. This reaches the dictation service's *own* lazily built
    transcriber, which is the identical object `_process_audio_buffer()` uses,
    so the warmed `_model_cache` entry is the one the capture hits.

    Args:
        service: The dictation service that is about to start capturing.

    Returns:
        The transcription service to warm, or None when this service does not
        expose one (test fakes).
    """
    # Checked on the *class* so a fake without the property is skipped without
    # invoking anything, and so a genuine AttributeError raised inside the real
    # property is never silently swallowed by a `getattr` default.
    if not hasattr(type(service), "transcription_service"):
        return None
    return service.transcription_service


def warm_before_capture_enabled() -> bool:
    """Return whether the model should be pre-loaded before capture opens.

    `dictation.warm_model_before_capture`, default True. The escape hatch
    exists because warming runs on *every* press (the Console builds a fresh
    service each time), so a provider that chokes on digital silence would
    otherwise degrade every capture forever.

    Returns:
        True when the warm-up should run.
    """
    raw = get_cli_setting("dictation.warm_model_before_capture", True)
    if isinstance(raw, str):
        return raw.strip().lower() not in {"false", "no", "0", "off"}
    return bool(raw)


def warm_transcription_model(transcriber: Any, effective: EffectiveConfig) -> None:
    """Load the speech model by transcribing silence through it.

    There is no preload/warm entry point on `TranscriptionService`; performing
    a transcription is the only way to populate its model cache. The arguments
    mirror `LazyLiveDictationService._process_audio_buffer()` exactly so the
    cache entry this creates is the one the capture will hit.

    Args:
        transcriber: The transcription service to warm.
        effective: The provider/model/language the capture will run with.

    Raises:
        Exception: Whatever the provider raises; the caller reports it as a
            model/provider failure.
    """
    transcriber.transcribe_buffer(
        audio_data=WARMUP_PCM,
        sample_rate=WARMUP_SAMPLE_RATE,
        channels=WARMUP_CHANNELS,
        sample_width=WARMUP_SAMPLE_WIDTH,
        provider=effective.provider,
        model=effective.model,
        language=effective.language,
    )


def capture_outcome_from(result: Any) -> CaptureOutcome:
    """Read a `DictationResult` without depending on its exact type.

    Args:
        result: Whatever `stop_dictation()` returned (services under test
            return None).

    Returns:
        The outcome, with unknown fields left as their "not reported" defaults.
    """
    if result is None:
        return CaptureOutcome()
    captured = getattr(result, "captured_bytes", None)
    complete = getattr(result, "transcription_complete", True)
    return CaptureOutcome(
        captured_bytes=int(captured) if isinstance(captured, int) else None,
        transcription_complete=bool(complete),
    )


def default_service_factory(**kwargs: Any) -> Any:
    """Build a LazyLiveDictationService, importing it as late as possible.

    The import lives in the function body on purpose: `tldw_chatbook.Audio`
    (the package) chains to `transcription_service`, which imports
    faster-whisper and NeMo at module scope. Importing the submodule directly,
    at call time, keeps that cost off app start entirely.

    Args:
        **kwargs: Forwarded verbatim to `LazyLiveDictationService.__init__`
            (e.g. `transcription_provider`, `transcription_model`, `language`,
            `enable_commands`).

    Returns:
        The constructed `LazyLiveDictationService`.
    """
    from ..Audio.dictation_service_lazy import LazyLiveDictationService

    return LazyLiveDictationService(**kwargs)


class ConsoleVoiceInputController:
    """Own the dictation lifecycle without touching the UI.

    Threading policy lives in the caller: `spawn` runs a thunk off the UI
    thread (a Textual worker in the app, a direct call in tests), because both
    `start_dictation()` and `stop_dictation()` (a thread join) block.
    """

    #: How often the abandon-aware warm-up wait re-checks. Small enough that
    #: quitting mid-download feels immediate, large enough not to spin.
    WARMUP_POLL_SECONDS = 0.05

    def __init__(
        self,
        *,
        emit: Callable[..., None],
        spawn: Callable[[Callable[[], None]], None],
        service_factory: Callable[..., Any] = default_service_factory,
    ) -> None:
        """Build the controller.

        Args:
            emit: Called for every controller event. Most calls pass just the
                event; the three recognizer-driven closures `_run_begin()`
                wires (`on_partial_transcript`, `on_final_transcript`, and,
                via `_report_service_error`, `on_error`) also pass this
                attempt's capture-generation token as a second, optional
                argument -- see `start()`'s `capture_generation` parameter.
            spawn: Runs a thunk off the UI thread.
            service_factory: Builds the dictation service.
        """
        self._emit = emit
        self._spawn = spawn
        self._service_factory = service_factory
        self._service: Any | None = None
        self._state = STATE_IDLE
        self._state_lock = threading.Lock()
        self._override_announced = False
        # Same shape as `_override_announced`, for `VoiceDictationModelDefaulted`
        # (review finding L1) -- see that event's docstring for the two-tier
        # scheme.
        self._model_default_announced = False
        # Same shape as `_override_announced`: latches `VoiceVadUnavailable`
        # to once per controller instance. `_handle_console_dictation_event`
        # (chat_screen.py) latches it a second time on `self.app_instance`,
        # the same two-tier scheme `VoiceProviderOverridden` uses, since a
        # fresh controller is built on every new dictation session.
        self._vad_unavailable_announced = False
        self.save_audio_requested = False
        # One-way latch: once `abandon()` has run, an in-flight `_begin()`
        # (still building/starting a service on another thread, a cold model
        # load can take tens of seconds) must release what it built instead
        # of transitioning to `listening`. Never reset -- `abandon()` is a
        # teardown path (unmount, app quit); the controller is not expected
        # to `start()` again afterward.
        self._abandoned = False
        # Per-attempt (not per-instance) latch: set when the service reports a
        # real cause through `on_error`, cleared at the top of every
        # `_run_begin()` so a failed attempt can never silence a later one.
        self._error_reported = False
        # What the service said about the last finished capture. Reset per
        # attempt in `start()`; read by the caller to tell a silent microphone
        # apart from a transcription that never finished.
        self._last_capture_outcome = CaptureOutcome()
        # The caller's opaque capture-generation token for the attempt
        # `start()` is about to begin, set alongside the other per-attempt
        # state under `_state_lock`. `_run_begin()` reads it once into a
        # local before wiring `on_partial_transcript`/`on_final_transcript`/
        # `on_error`, so those closures bind *this* attempt's token even if a
        # later `start()` overwrites this attribute before an orphaned
        # processing thread from THIS attempt finally calls one of them (see
        # `start()`'s `capture_generation` parameter).
        self._pending_capture_generation: int | None = None

    @property
    def state(self) -> str:
        """The controller's current state.

        Returns:
            One of the `STATE_*` constants (`STATE_UNAVAILABLE`, `STATE_IDLE`,
            `STATE_PREPARING`, `STATE_LISTENING`, `STATE_FINISHING`,
            `STATE_ERROR`).
        """
        return self._state

    @property
    def last_capture_outcome(self) -> CaptureOutcome:
        """What the service reported about the most recently stopped capture."""
        return self._last_capture_outcome

    @property
    def is_active(self) -> bool:
        """True while a microphone is or is about to be live."""
        return self._state in (STATE_PREPARING, STATE_LISTENING, STATE_FINISHING)

    def _set_state(self, state: str) -> None:
        self._state = state
        self._emit(VoiceStateChanged(state))

    def _emit_capture_event(self, event: Any, generation: int | None = None) -> None:
        """Forward a recognizer-driven event, adding the generation only when known.

        `emit` (the caller-supplied callback, e.g.
        `ConsoleStreamingDictationSession._handle_event`) is called with just
        the event whenever `generation` is `None` -- preserving the original
        single-argument contract every non-Console caller and every existing
        test double still relies on -- and with `(event, generation)` only
        when a real token is in play (a capture wired through
        `ConsoleStreamingDictationSession`, whose `start()` always supplies
        one). This is what lets `on_partial_transcript`/`on_final_transcript`
        (in `_run_begin()`) and `_fail()`'s `VoiceFailed` emit unconditionally
        pass whatever `capture_generation` they were bound with, including
        `None`, without changing `emit`'s call arity for every caller that
        never opted into generation tracking.

        Args:
            event: The event to forward.
            generation: This attempt's capture-generation token, or `None`.
        """
        if generation is None:
            self._emit(event)
        else:
            self._emit(event, generation)

    def _fail(
        self,
        reason: str,
        remedy: str = "",
        *,
        generation: int | None = None,
    ) -> None:
        # Mutate first so a throwing `emit` cannot leave the machine wedged,
        # but keep VoiceFailed ahead of VoiceStateChanged(idle): the UI clears
        # its pending-send on the failure and fires it on the idle transition,
        # so reversing these would send the message on a failed dictation.
        #
        # `generation` is only ever non-`None` when this came from
        # `_report_service_error()`'s delayed, capture-bound path (see its
        # docstring): every other caller is synchronous within the current
        # attempt's own call chain and passes nothing, which
        # `ConsoleStreamingDictationSession._handle_event` treats as "always
        # current" -- there is no orphaned thread to be stale relative to.
        # The paired `VoiceStateChanged(idle)` never carries one: Task 3's
        # screen-side session-identity/state guards already cover a stale
        # idle transition, and `_handle_event` does not gate that event type.
        self._state = STATE_IDLE
        self._emit_capture_event(VoiceFailed(reason=reason, remedy=remedy), generation)
        self._emit(VoiceStateChanged(STATE_IDLE))

    def start(self, *, capture_generation: int | None = None) -> None:
        """Begin capture. Rejected unless currently idle and never abandoned.

        Args:
            capture_generation: The caller's opaque token for this attempt.
                Stored so `_run_begin()` can read it once, before wiring
                `on_partial_transcript`/`on_final_transcript`/`on_error`, and
                bind it into those specific closures -- the ones a real
                orphaned processing thread can still call after a *later*
                capture has already reassigned `self._pending_capture_generation`.
                Reading `self._pending_capture_generation` dynamically inside
                those closures instead would report whichever capture is
                current at call time, not the one that produced the event,
                defeating the whole point. `None` (the default, and every
                caller except `ConsoleStreamingDictationSession`) means the
                caller does not distinguish captures, in which case nothing
                downstream is gated.
        """
        with self._state_lock:
            if self._abandoned or self._state != STATE_IDLE:
                logger.debug(
                    "Console dictation start ignored (abandoned={}, state={})",
                    self._abandoned,
                    self._state,
                )
                return
            self._last_capture_outcome = CaptureOutcome()
            self._state = STATE_PREPARING
            self._pending_capture_generation = capture_generation
        self._emit(VoiceStateChanged(STATE_PREPARING))

        # Each `try` below covers only the call that can crash unexpectedly,
        # never the `_fail()` that handles its result: `_fail()`'s own emit
        # can itself raise (that's the whole point of Finding 2), and if that
        # raise were caught by one of these `except` blocks it would trigger
        # a second, mislabeled `_fail()` describing the plumbing exception
        # instead of the real cause.
        try:
            availability = probe()
        except Exception as exc:  # noqa: BLE001 - a probe crash must not wedge preparing
            logger.opt(exception=True).warning("Console dictation availability probe crashed")
            self._fail(str(exc))
            return

        if not availability.ok:
            self._fail(availability.reason, availability.remedy)
            return

        try:
            effective = resolve()
        except Exception as exc:  # noqa: BLE001 - a resolve crash must not wedge preparing
            logger.opt(exception=True).warning("Console dictation provider resolution crashed")
            self._fail(str(exc))
            return

        if effective is None:
            self._fail(PROVIDER_REASON, PROVIDER_REMEDY)
            return

        try:
            if effective.was_overridden and not self._override_announced:
                self._override_announced = True
                self._emit(
                    VoiceProviderOverridden(
                        configured=effective.configured_provider,
                        effective=effective.provider,
                    )
                )

            if (
                effective.fast_default_displaced_configured_model
                and not self._model_default_announced
            ):
                self._model_default_announced = True
                self._emit(
                    VoiceDictationModelDefaulted(
                        configured=effective.configured_model or "",
                        effective=effective.model or "",
                    )
                )

            self._spawn(lambda: self._begin(effective))
        except Exception as exc:  # noqa: BLE001 - override-announce/spawn must not wedge preparing
            logger.opt(exception=True).warning("Console dictation could not be spawned")
            self._fail(str(exc))
            return

    def _begin(self, effective: EffectiveConfig) -> None:
        """Blocking half of start(); always runs via `spawn`.

        A thread boundary: when `spawn` is inline -- the default in nearly
        every test, and any future ad-hoc caller -- this method runs
        synchronously inside `start()`'s own try/except around the `spawn()`
        call. That guard exists to catch a real `spawn()` failing to
        *schedule* work and must stay in place, so nothing raised in here may
        propagate back through `spawn()` into it: `_run_begin()`'s own
        `_fail()` calls have a raising emit as their whole reason for
        existing (Finding 2), and letting that reach `start()`'s guard would
        fire a second, mislabeled `VoiceFailed` describing this method's
        plumbing instead of the real cause -- the exact cascade N1 fixed in
        `start()`, recurring one call frame deeper.
        """
        try:
            self._run_begin(effective)
        except Exception:  # noqa: BLE001 - nothing may escape _begin(); see docstring
            logger.opt(exception=True).warning("Console dictation _begin() raised unexpectedly")

    def _run_begin(self, effective: EffectiveConfig) -> None:
        """The actual work of `_begin()`, shielded from its caller by `_begin()`."""
        # Cleared per attempt, before anything can set it: `on_error` fires
        # synchronously from inside `start_dictation()` (see
        # `_report_service_error`), and a latch left over from an earlier
        # failed attempt would silence this attempt's fallback report.
        self._error_reported = False
        # Read once, into a local, before `on_partial_transcript`/
        # `on_final_transcript`/`on_error` are wired below: those three
        # closures bind `capture_generation` as an early-evaluated default
        # argument, so a later `start()` overwriting
        # `self._pending_capture_generation` cannot change what an already
        # orphaned processing thread from THIS attempt reports when it
        # finally calls one of them.
        capture_generation = self._pending_capture_generation
        try:
            service = self._service_factory(
                transcription_provider=effective.provider,
                transcription_model=effective.model,
                language=effective.language,
                enable_commands=False,  # V2 owns voice commands, not V1
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
            logger.opt(exception=True).warning(
                "Console dictation service could not be built"
            )
            self._fail(str(exc))
            return

        # Before the microphone opens, never after: a model load that happens
        # on the first audio chunk runs while the user is already speaking and
        # behind the stop-side join, which is exactly how a good capture came
        # back as "No audio was captured from the microphone."
        if not self._prepare_speech_model(service, effective):
            return

        try:
            started = service.start_dictation(
                on_partial_transcript=lambda text, _gen=capture_generation: (
                    self._emit_capture_event(VoicePartial(text), _gen)
                ),
                on_final_transcript=lambda text, _gen=capture_generation: (
                    self._emit_capture_event(classify_segment(text), _gen)
                ),
                on_segment_transcribing=lambda done, _gen=capture_generation: (
                    self._emit_capture_event(VoiceSegmentTranscribing(done=done), _gen)
                ),
                on_state_change=lambda _state: None,  # our state machine is authoritative
                on_error=lambda error, _gen=capture_generation: self._report_service_error(
                    error, generation=_gen
                ),
                save_audio=self.save_audio_requested,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
            logger.opt(exception=True).warning("Console dictation failed to start")
            # `on_error` is invoked from *inside* `start_dictation()`, i.e.
            # from inside the `try` above, so this `exc` can be the real
            # cause's own `_fail()` emit raising rather than a start failure.
            # Reporting again would bury the real cause under plumbing --
            # the same latch `_fail_not_started()` consults, for the same
            # reason.
            if self._error_reported:
                logger.debug(
                    "Console dictation start crashed after the service reported the cause"
                )
                return
            self._fail(str(exc))
            return

        # Claim the freshly built service unless `abandon()` won the race
        # while the factory/`start_dictation()` call (a cold model load can
        # take tens of seconds) was still in flight. That check happened
        # against no service to release, so it's on us to release this one.
        with self._state_lock:
            if self._abandoned:
                claimed = False
            else:
                self._service = service
                claimed = True

        if not claimed:
            self._release(service)
            return

        if not started:
            self._claim_service()  # drop it; the service already cleaned up
            self._fail_not_started()
            return

        if self._enter_listening():
            self._maybe_report_vad_unavailable(service)

    def _prepare_speech_model(self, service: Any, effective: EffectiveConfig) -> bool:
        """Load the speech model while still in `preparing`, before capture.

        The whole point of the `preparing` phase: on a fresh machine this is a
        1.4 GB download, and it used to happen lazily on the first audio chunk
        -- recording into a void, then losing the download to the stop-side
        thread join. Doing it here costs a warm press nothing extra (the load
        it performs is the one the first chunk would have performed anyway).

        Two failure modes, deliberately weighted differently:

        * The transcription service cannot be *built* -- models genuinely
          absent. Fatal, reported as a model/provider problem.
        * The silence transcription itself errors after the service was built.
          Not fatal: the Console warms on every press, so making this fatal
          would turn one transient error into permanently unusable dictation.
          It is announced and the capture proceeds, which is only safe because
          an empty result can no longer be misreported as a dead microphone.

        Opt out entirely with `dictation.warm_model_before_capture = false`.

        Args:
            service: The dictation service that is about to start capturing.
            effective: The provider/model/language the capture will run with.

        Returns:
            True when capture may proceed. False when this attempt is over --
            either the model could not be prepared at all (already reported
            through `_fail`) or `abandon()` landed while the model was loading.
        """
        if self._abandoned:
            self._release(service)
            return False
        if not warm_before_capture_enabled():
            logger.debug(
                "Console dictation model warm-up disabled by "
                "dictation.warm_model_before_capture"
            )
            return True
        try:
            transcriber = warmup_target(service)
        except Exception as exc:  # noqa: BLE001 - reported as a model failure
            logger.opt(exception=True).warning(
                "Console dictation transcription service could not be built"
            )
            self._release(service)
            self._fail(self._warmup_failure(effective, exc), WARMUP_REMEDY)
            return False
        if transcriber is None:
            return True

        first_run = _is_first_warmup(effective)
        self._emit_quietly(
            VoiceModelPreparing(
                message=WARMUP_MESSAGE_FIRST_RUN if first_run else WARMUP_MESSAGE,
                detail=WARMUP_DETAIL_FIRST_RUN if first_run else "",
                first_run=first_run,
            ),
            "model-preparing notice",
        )

        error = self._run_warmup_off_the_executor(transcriber, effective)
        if self._abandoned:
            # A first-run warm-up runs for minutes; the screen can unmount
            # inside it. Never hand a live microphone to a torn-down machine.
            self._release(service)
            return False
        if error is not None:
            logger.opt(exception=error).warning("Console dictation model warm-up failed")
            self._emit_quietly(
                VoiceModelWarmupFailed(
                    reason=(
                        f"{WARMUP_DEGRADED_REASON_TEMPLATE.format(provider=effective.provider)}"
                        f" {error}"
                    ).strip(),
                    remedy=WARMUP_REMEDY,
                ),
                "model warm-up warning",
            )
            return True
        _mark_warmed(effective)
        return True

    def _run_warmup_off_the_executor(
        self, transcriber: Any, effective: EffectiveConfig
    ) -> BaseException | None:
        """Perform the warm-up on a daemon thread, waiting abandon-aware.

        `_run_begin()` runs on the default asyncio executor (via the screen's
        `asyncio.to_thread`), and `asyncio.run()` **joins that executor at
        shutdown**. Doing a multi-minute model download directly on it means a
        user who gets bored and quits sits in front of a dead terminal until
        the download finishes -- `abandon()` cannot interrupt a blocking C
        call. Running it on a daemon thread and waiting here means `abandon()`
        releases this frame promptly, the executor thread returns, and the
        process can exit; the orphaned download dies with it.

        Args:
            transcriber: The transcription service to warm.
            effective: The provider/model/language the capture will run with.

        Returns:
            None when the model loaded, the exception when it did not, or a
            `RuntimeError` placeholder when `abandon()` ended the wait early
            (the caller checks `_abandoned` first, so that value is unused).
        """
        done = threading.Event()
        box: dict[str, BaseException] = {}

        def _work() -> None:
            try:
                warm_transcription_model(transcriber, effective)
            except BaseException as exc:  # noqa: BLE001 - handed back to the caller
                box["error"] = exc
            finally:
                done.set()

        threading.Thread(
            target=_work, daemon=True, name="ConsoleDictationModelWarmup"
        ).start()
        while not done.wait(self.WARMUP_POLL_SECONDS):
            if self._abandoned:
                logger.debug(
                    "Console dictation abandoned while the speech model was loading; "
                    "leaving the load to finish on its daemon thread"
                )
                return RuntimeError("abandoned during model warm-up")
        return box.get("error")

    def _emit_quietly(self, event: Any, description: str) -> None:
        """Emit an advisory event, swallowing a plumbing failure.

        Progress and degraded-mode notices are cosmetic: a raising `emit` must
        never cost the user their capture. Real failures still go through
        `_fail()`, whose raising-emit contract is deliberately different.

        Args:
            event: The event to emit.
            description: What it was, for the log line if emitting fails.
        """
        try:
            self._emit(event)
        except Exception:  # noqa: BLE001 - advisory copy must never abort a start
            logger.opt(exception=True).debug(
                "Console dictation {} could not be emitted", description
            )

    @staticmethod
    def _warmup_failure(effective: EffectiveConfig, exc: Exception) -> str:
        """Phrase a warm-up failure as a model problem, never a microphone one."""
        return f"{WARMUP_REASON_TEMPLATE.format(provider=effective.provider)} {exc}".strip()

    def _report_service_error(self, error: Any, *, generation: int | None = None) -> None:
        """Turn a service-reported error into a failure, recording that we did.

        `LazyLiveDictationService` reports through this callback
        *synchronously, from inside `start_dictation()`*, and then returns
        `False` rather than raising -- all three of its failure branches do
        (`dictation_service_lazy.py` lines 285-290, 323-329 and 332-335, each
        `self._notify_error(...)` followed by `return False`). Without the
        latch, that one failure produces two `VoiceFailed` events: the real
        cause from here, then `_fail_not_started()`'s generic one, which
        arrives *last* and buries the actionable diagnostic in the UI.

        This is also the path an orphaned processing thread uses to report a
        recognizer error long after its own capture's `stop_dictation()` join
        gave up. Fix round 1 (Finding, HIGH): tagging the eventual `VoiceFailed`
        emit with `generation` is not enough on its own -- `_claim_service()`/
        `_release()` below reach for `self._service`, i.e. whatever capture is
        CURRENTLY live, not the one that produced this report, and `_fail()`
        unconditionally flips the FSM to `idle`. Left unchecked, a stale call
        would rip the microphone out from under a live capture 2 and silently
        idle the controller, while the session's own generation gate in
        `ConsoleStreamingDictationSession._handle_event` only swallows the
        *notification* -- the screen would show "Rec ●" over a dead
        microphone with no toast at all, worse than doing nothing. So the
        check happens here, first, before the latch, before any claim/release,
        before `_fail()` ever runs: a stale call is a no-op start to finish.

        Args:
            error: The exception the service reported.
            generation: This attempt's capture-generation token, bound at
                `on_error`'s creation time in `_run_begin()`. `None` when the
                caller does not distinguish captures. Compared against
                `self._pending_capture_generation` -- the token the most
                recent `start()` recorded -- which is exactly what changed if
                a newer capture has begun since this callback was wired.
        """
        if generation is not None and generation != self._pending_capture_generation:
            logger.debug(
                "Console dictation ignoring a stale service error (generation "
                "{}, current generation {})",
                generation,
                self._pending_capture_generation,
            )
            return
        # Set before `_fail()`, which emits and can therefore raise: the
        # report has happened either way, and the service's own
        # `_notify_error()` only logs whatever escapes this callback.
        self._error_reported = True
        # Claim and release the service *before* reporting: a mid-session
        # error (state LISTENING, `self._service` already claimed by
        # `_run_begin()`) must not leave a live recorder behind an idle
        # machine, and must not be silently orphaned when a retry claims a
        # second service. During the startup path there is nothing to claim
        # yet (`_claim_service()` returns None), so this is a no-op there.
        # `_release()` never raises, so it cannot disturb the `_fail()`
        # raising-emit contract this callback depends on (see
        # `_run_begin`'s `except` and the `_error_reported` latch above).
        service = self._claim_service()
        if service is not None:
            self._release(service)
        self._fail(str(error), generation=generation)

    def _fail_not_started(self) -> None:
        """Report that `start_dictation()` returned `False`.

        Stays quiet in two cases. If `abandon()` landed in the narrow window
        between the claim above and this check, the controller is already
        idle and torn down, so a `VoiceFailed`/`VoiceStateChanged(idle)` pair
        here would be noise on top of teardown. And if the service already
        told us *why* it could not start (see `_report_service_error`), this
        generic message would land second and bury that real cause.
        """
        if self._abandoned:
            return
        if self._error_reported:
            logger.debug(
                "Console dictation start failed; real cause already reported by the service"
            )
            return
        self._fail("Could not start the microphone.")

    def _enter_listening(self) -> bool:
        """Atomically transition to `listening`, re-checking abandonment.

        Between claiming the service above (under `_state_lock`) and this
        call, `abandon()` may have run on another thread -- a real one in
        production, since `_begin()` runs on a worker thread while `abandon()`
        fires from the UI thread -- and already released the microphone and
        returned the machine to idle. Re-checking `_abandoned` here, under
        the same lock, closes that window instead of stomping the state back
        to `listening` with no service behind it.

        Returns:
            True once the transition to `listening` actually happened; False
            when `abandon()` won the race and the machine is idle instead.
            The caller uses this to skip work (e.g. the VAD-unavailable
            check) that only makes sense for a capture that is actually live.
        """
        with self._state_lock:
            if self._abandoned:
                return False
            self._state = STATE_LISTENING
        self._emit(VoiceStateChanged(STATE_LISTENING))
        return True

    def _maybe_report_vad_unavailable(self, service: Any) -> None:
        """Emit `VoiceVadUnavailable` once per controller instance.

        Voice commands rely on the recorder's own VAD to gate mid-capture
        finalization (`LazyLiveDictationService.SILENCE_THRESHOLD_SECONDS`'s
        docstring has the mechanism); without it the recorder forwards every
        frame and a segment only finalizes when the capture stops, so a
        command spoken mid-capture never fires. That degrade path is
        otherwise silent, so this tells the user once.

        `service._audio_service` is read with the same defensive
        `getattr`-only pattern `_release()` uses: a fake or a service that
        has not populated the attribute yet reports neither `True` nor
        `False` from the inner `getattr`, and only an explicit `False` (VAD
        was requested and did not come up) counts as degraded.

        Args:
            service: The dictation service this capture just started on.
        """
        if self._vad_unavailable_announced:
            return
        audio = getattr(service, "_audio_service", None)
        if getattr(audio, "use_vad", None) is False:
            self._vad_unavailable_announced = True
            logger.warning(
                "Console dictation started without VAD; voice commands "
                "cannot finalize mid-capture this session"
            )
            self._emit(VoiceVadUnavailable())

    def stop(self) -> None:
        """End capture and commit. No-op unless currently listening."""
        with self._state_lock:
            if self._state != STATE_LISTENING:
                logger.debug("Console dictation stop ignored in state {}", self._state)
                return
            self._state = STATE_FINISHING
        # Same guard `start()` carries, for the same reason: nothing else
        # unwinds `finishing`, so a raising emit or a `spawn()` that fails to
        # schedule would wedge the machine there forever with `is_active`
        # true. `_finish()` is an exception boundary (see its docstring), so
        # with an inline `spawn` this `try` cannot transitively swallow
        # `_finish()`'s own `_fail()` and cascade a mislabeled failure.
        try:
            self._emit(VoiceStateChanged(STATE_FINISHING))
            self._spawn(self._finish)
        except Exception as exc:  # noqa: BLE001 - finishing must never wedge
            logger.opt(exception=True).warning("Console dictation could not be finished")
            # The microphone is live and no worker will ever run `_finish()`
            # now, so drop it here rather than leave it recording behind an
            # idle state machine.
            service = self._claim_service()
            if service is not None:
                self._release(service)
            self._fail(str(exc))
            return

    def _finish(self) -> None:
        """Blocking half of stop(); always runs via `spawn`.

        An exception boundary, exactly like `_begin()`: with an inline
        `spawn` this runs synchronously inside `stop()`'s try around the
        `spawn()` call, and `_run_finish()`'s `_fail()` has a raising emit as
        its whole reason for existing -- letting that reach `stop()`'s guard
        would re-fire a second, mislabeled `VoiceFailed` describing this
        method's plumbing instead of the real cause.
        """
        try:
            self._run_finish()
        except Exception:  # noqa: BLE001 - nothing may escape _finish(); see docstring
            logger.opt(exception=True).warning("Console dictation _finish() raised unexpectedly")

    def _run_finish(self) -> None:
        """The actual work of `_finish()`, shielded from its caller by `_finish()`."""
        service = self._claim_service()
        try:
            result = service.stop_dictation() if service is not None else None
        except Exception as exc:  # noqa: BLE001
            logger.opt(exception=True).warning("Console dictation failed to stop")
            # The service was already claimed, so nothing else will ever
            # release it: without this, a `stop_dictation()` that raises
            # (`Thread.join(timeout=nan)` -> ValueError is one real way to get
            # here) leaves a live microphone behind an idle state machine.
            # `_release` never raises, so it cannot disturb the raising-emit
            # contract `_fail()` depends on.
            if service is not None:
                self._release(service)
            self._fail(str(exc))
            return
        # How many bytes the recorder delivered, and whether the transcription
        # thread actually finished. Without this the caller can only guess from
        # an empty transcript -- and it guessed "microphone", every time.
        self._last_capture_outcome = capture_outcome_from(result)
        # Belt-and-braces: `LazyLiveDictationService.stop_dictation()` has
        # historically returned successfully without releasing capture, so
        # the Console does not trust the dependency alone. `stop_recording()`
        # early-returns when not already recording, so releasing again here
        # cannot double-stop a service that already released itself
        # correctly -- it logs a warning at worst.
        if service is not None:
            self._release(service)
        self._enter_idle()

    def _enter_idle(self) -> None:
        """Atomically return to `idle`, re-checking abandonment.

        The mirror of `_enter_listening()`, and needed for the same reason:
        `_run_finish()` runs on a worker thread while `abandon()` fires from
        the UI thread, so teardown can complete while this is still in
        flight. Announcing `idle` again afterwards would emit a state change
        for a controller that has already been torn down -- and a later task
        treats `VoiceStateChanged(idle)` as the trigger to send a deferred
        message.
        """
        with self._state_lock:
            if self._abandoned:
                return
            self._state = STATE_IDLE
        self._emit(VoiceStateChanged(STATE_IDLE))

    def _claim_service(self) -> Any | None:
        """Take sole ownership of the current service, under `_state_lock`.

        Every other read-and-clear of `self._service` is serialized against
        `abandon()` this way. Without the lock, `abandon()` on the UI thread
        and `_run_finish()` on a worker can both come away with the same
        service (double release), or `_run_finish()` can call
        `stop_dictation()` on one `abandon()` has already released -- which
        lands in its `except` and reports a spurious failure after teardown.

        Returns:
            The service that was held, or None if there was none to take.
        """
        with self._state_lock:
            service, self._service = self._service, None
        return service

    def abandon(self) -> None:
        """Release the microphone without waiting on the 2s join.

        For teardown paths (unmount, app quit) where blocking would show up
        as a hang. Best effort by design. Safe to call from any state,
        including mid-`preparing`: sets a one-way latch that `_begin()`
        checks after it finishes building/starting a service, so a service
        that only comes into existence after this call still gets released
        instead of handed off to `listening`.
        """
        with self._state_lock:
            self._abandoned = True
            service, self._service = self._service, None
            self._state = STATE_IDLE
        if service is not None:
            self._release(service)

    def _release(self, service: Any) -> None:
        """Best-effort microphone release, skipping the 2s join. Never raises.

        Used by `abandon()` at teardown, by `_run_begin()` when `abandon()`
        won the race, and by `stop()` when no worker will ever run
        `_finish()`.

        Beyond the audio stream, this must also tell the service's
        `DictationProcessor` daemon thread (`LazyLiveDictationService.
        _processing_loop`) to exit. That thread only ever gets stopped from
        inside `stop_dictation()`'s `stop_processing.set()` -- exactly the
        blocking join path `abandon()` exists to skip -- so without this the
        thread (and the service instance it holds a reference to) survives
        forever after every abandoned or mid-session-failed capture. No join
        here: `abandon()`'s entire point is that it never blocks.

        Args:
            service: The dictation service instance to release.
        """
        try:
            audio = getattr(service, "_audio_service", None)
            if audio is not None and hasattr(audio, "stop_recording"):
                audio.stop_recording()
        except Exception:  # noqa: BLE001 - teardown must never raise
            logger.opt(exception=True).debug("Console dictation abandon failed")
        try:
            stop_processing = getattr(service, "stop_processing", None)
            if stop_processing is not None and hasattr(stop_processing, "set"):
                stop_processing.set()
        except Exception:  # noqa: BLE001 - teardown must never raise
            logger.opt(exception=True).debug(
                "Console dictation abandon failed to stop the processing thread"
            )
