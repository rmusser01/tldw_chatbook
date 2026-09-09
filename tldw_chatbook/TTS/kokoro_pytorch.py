"""Local voice handling around the official Kokoro PyTorch runtime."""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tldw_chatbook.TTS.adapter_types import TTSOperationError
from tldw_chatbook.Utils.optional_deps import check_dependency

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path.home() / ".config" / "tldw_cli" / "models" / "kokoro"
SAMPLE_RATE = 24000
MODEL_REPO = "hexgrad/Kokoro-82M"
MODEL_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
VOICE_PATTERN = re.compile(r"([a-z]+_[a-z]+)(?:\((\d+)\))?")
LANGUAGE_CODES = {
    "en": "a",
    "en-us": "a",
    "en-gb": "b",
    "es": "e",
    "fr": "f",
    "fr-fr": "f",
    "hi": "h",
    "it": "i",
    "pt": "p",
    "pt-br": "p",
    "ja": "j",
    "zh": "z",
    "zh-cn": "z",
}


def require_runtime() -> tuple[Any, Any]:
    """Load the optional runtime only for a deliberate PyTorch operation."""
    if not check_dependency("kokoro", "kokoro_pytorch"):
        raise TTSOperationError(
            code="dependency_missing",
            message=(
                "Kokoro PyTorch requires the optional kokoro runtime. "
                "On Python 3.11 or 3.12, install 'tldw_chatbook[local_tts]'. "
                "Python 3.13+ is not supported by kokoro 0.9.4; select ONNX instead."
            ),
            retryable=False,
            operation_id="kokoro_pytorch",
            recovery_action="install_kokoro_pytorch",
        )
    from kokoro import KModel, KPipeline

    return KModel, KPipeline


class _CpuSTFT(torch.nn.Module):
    """Keep upstream Fourier math on CPU where complex operations are supported."""

    def __init__(self, stft: torch.nn.Module) -> None:
        super().__init__()
        self.stft = stft

    def transform(self, input_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        magnitude, phase = self.stft.transform(input_data.cpu())
        return magnitude.to(input_data.device), phase.to(input_data.device)

    def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        return self.stft.inverse(magnitude.cpu(), phase.cpu()).to(magnitude.device)


class KokoroModel:
    """Share one official model between lazily initialized language pipelines."""

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self.model_path = model_path
        self.device = device
        self.model = None
        self.config = None
        self.pipelines: dict[str, Any] = {}

    def load(self) -> None:
        """Load the configured checkpoint using Kokoro's actual architecture."""
        if self.model is not None:
            return
        model_type, _ = require_runtime()
        config_path = Path(self.model_path).with_name("config.json")
        if not config_path.is_file():
            from huggingface_hub import hf_hub_download

            config_path = Path(
                hf_hub_download(
                    repo_id=MODEL_REPO,
                    filename="config.json",
                    revision=MODEL_REVISION,
                )
            )
        # KModel loads the component state dictionaries with weights_only=True.
        model = model_type(
            repo_id=MODEL_REPO,
            config=str(config_path),
            model=self.model_path,
        )
        if torch.device(self.device).type == "mps":
            # Install after loading the checkpoint so decoder state keys are
            # unchanged. TorchSTFT has no learned weights. The neural layers
            # remain on MPS; only unsupported complex Fourier work uses CPU.
            model.decoder.generator.stft = _CpuSTFT(model.decoder.generator.stft)
        model = model.to(self.device).eval()
        self.config = str(config_path)
        self.model = model
        logger.info("Loaded official Kokoro PyTorch model on %s", self.device)

    def pipeline(self, language: str) -> Any:
        """Return the pipeline for a supported language, sharing model weights."""
        language = language.lower().replace("_", "-")
        language = LANGUAGE_CODES.get(language, language)
        if language not in set("abefhipjz"):
            raise ValueError(f"Unsupported Kokoro language: {language}")
        self.load()
        if language not in self.pipelines:
            _, pipeline_type = require_runtime()
            try:
                pipeline = pipeline_type(
                    lang_code=language,
                    model=self.model,
                    repo_id=MODEL_REPO,
                )
            except ImportError as error:
                if language not in {"j", "z"}:
                    raise
                raise TTSOperationError(
                    code="dependency_missing",
                    message=(
                        "Kokoro language dependencies are missing. Install "
                        "'misaki[ja]' for Japanese or 'misaki[zh]' for Chinese "
                        "in the TTS environment."
                    ),
                    retryable=False,
                    operation_id="kokoro_pytorch",
                    recovery_action="install_kokoro_language_extras",
                ) from error
            except SystemExit as error:
                # Misaki invokes spaCy's CLI installer for missing English
                # assets. A failed installation must not exit the application.
                raise TTSOperationError(
                    code="configuration_invalid",
                    message=(
                        "Kokoro language setup failed. For English, run "
                        "'python -m spacy download en_core_web_sm' in the TTS "
                        "environment, then retry."
                    ),
                    retryable=False,
                    operation_id="kokoro_pytorch",
                    recovery_action="install_kokoro_language",
                ) from error
            self.pipelines[language] = pipeline
        return self.pipelines[language]


def build_model(model_path: str, device: str = "cpu") -> KokoroModel:
    """Build an official Kokoro model from a local v1 checkpoint."""
    model = KokoroModel(model_path, device)
    model.load()
    return model


def load_voice(voice_path: str, device: str = "cpu") -> torch.Tensor:
    """Load a local tensor voice pack without unrestricted pickle execution."""
    voice_tensor = torch.load(voice_path, map_location="cpu", weights_only=True)
    if isinstance(voice_tensor, dict):
        voice_tensor = voice_tensor.get("voice", voice_tensor.get("embedding"))
    if not isinstance(voice_tensor, torch.Tensor):
        raise ValueError(f"Invalid voice file format: {voice_path}")  # noqa: TRY004 - malformed file content
    return voice_tensor.to(device=device, dtype=torch.float32)


def parse_voice_mix(voice_string: str) -> list[tuple[str, float]]:
    """Parse weighted local voices such as ``af_bella(2)+af_sky(1)``."""
    voices = []
    for part in voice_string.split("+"):
        part = part.strip()
        match = VOICE_PATTERN.fullmatch(part)
        if match:
            voices.append((match[1], float(match[2]) if match[2] else 1.0))
        elif part:
            voices.append((part, 1.0))
    return voices


def mix_voices(
    voice_tensors: list[tuple[torch.Tensor, float]],
    normalize: bool = True,
) -> torch.Tensor:
    """Blend full voice packs without broadcasting into extra dimensions."""
    if not voice_tensors:
        raise ValueError("No voice tensors provided")
    tensors, weights = zip(*voice_tensors)
    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("Voice weights must be finite and nonnegative")
    total = sum(weights)
    if total <= 0:
        raise ValueError("Voice weights must have a positive total")
    if normalize:
        weights = [weight / total for weight in weights]
    stacked = torch.stack(tensors)
    factors = torch.tensor(weights, device=stacked.device, dtype=stacked.dtype)
    factors = factors.reshape((-1,) + (1,) * tensors[0].ndim)
    return (stacked * factors).sum(dim=0)


def generate(
    model: KokoroModel,
    text: str,
    voice: torch.Tensor | str | dict[str, float],
    lang: str = "a",
    speed: float = 1.0,
    voice_dir: str | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, str]:
    """Synthesize every upstream segment, preserving its waveform and speed.

    Args:
        model: Shared Kokoro model wrapper.
        text: Text to phonemize and synthesize.
        voice: Complete voice pack, local voice name, or weighted local voices.
        lang: Kokoro language code or supported locale alias.
        speed: Duration multiplier consumed once by the neural model.
        voice_dir: Directory for named local voice packs.
        **kwargs: Reserved compatibility options from existing callers.

    Returns:
        Complete mono float32 waveform at 24 kHz and generated phonemes.

    Raises:
        ValueError: Language, speed or generated audio is invalid.
        TTSOperationError: Runtime setup or the text's phoneme length is invalid.
    """
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("Kokoro speed must be finite and positive")
    pipeline = model.pipeline(lang)
    if pipeline.lang_code not in "ab":
        # Upstream 0.9.4 silently truncates oversized non-English sentences to
        # 510 phonemes. Reject them before inference so missing speech cannot
        # be reported as successful. Newlines provide explicit safe boundaries.
        for paragraph in text.splitlines():
            if paragraph.strip():
                phonemes, _ = pipeline.g2p(paragraph)
                if len(phonemes or "") > 510:
                    raise TTSOperationError(
                        code="request_invalid",
                        message=(
                            "Kokoro input exceeds 510 phonemes per non-English "
                            "segment. Split long text with newlines and retry."
                        ),
                        retryable=False,
                        operation_id="kokoro_pytorch",
                        recovery_action="split_kokoro_text",
                    )
    if isinstance(voice, (str, dict)):
        specs = parse_voice_mix(voice) if isinstance(voice, str) else voice.items()
        voice = mix_voices(
            [
                (load_voice(_find_voice_file(name, voice_dir)), weight)
                for name, weight in specs
            ]
        )
    if not isinstance(voice, torch.Tensor):
        raise TypeError("Kokoro requires a tensor voice pack")
    # KPipeline's tensor detection requires a CPU FloatTensor; it moves the
    # selected style to the actual model device itself, including MPS.
    voice = voice.detach().to(device="cpu", dtype=torch.float32)
    chunks = []
    phonemes = []
    with torch.inference_mode():
        for result in pipeline(text, voice=voice, speed=speed):
            if result.audio is None:
                raise ValueError("Kokoro returned no audio")
            samples = result.audio.detach().cpu().numpy()
            if samples.ndim != 1 or not np.isfinite(samples).all():
                raise ValueError("Kokoro returned invalid audio")
            if samples.size:
                chunks.append(np.asarray(samples, dtype=np.float32))
                phonemes.append(result.phonemes)
    if not chunks:
        raise ValueError("Kokoro returned no audio for this text")
    return np.concatenate(chunks), " ".join(phonemes)


def _find_voice_file(voice_name: str, voice_dir: str | None = None) -> str:
    directory = (
        Path(voice_dir) if voice_dir is not None else DEFAULT_MODEL_PATH / "voices"
    )
    for extension in (".pt", ".pth", ".ckpt"):
        path = directory / f"{voice_name}{extension}"
        if path.is_file():
            return str(path)
    if Path(voice_name).is_file():
        return voice_name
    raise FileNotFoundError(f"Voice file not found: {voice_name}")


def get_available_voices(voice_dir: str | None = None) -> list[str]:
    """List the installed local PyTorch voice packs."""
    directory = (
        Path(voice_dir) if voice_dir is not None else DEFAULT_MODEL_PATH / "voices"
    )
    return sorted(path.stem for path in directory.glob("*.pt"))
