"""Optional model-aligned Kokoro language frontends (ADR-142)."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from tldw_chatbook.TTS.adapter_types import TTSOperationError
from tldw_chatbook.Utils.optional_deps import check_dependency


def japanese_dictionary_error(
    error: RuntimeError, *, operation_id: str
) -> TTSOperationError | None:
    """Classify a MeCab setup failure without exposing its private file paths."""
    if "mecab" not in str(error).lower():
        return None
    return TTSOperationError(
        code="configuration_invalid",
        message=(
            "Kokoro Japanese dictionary setup failed. Install 'misaki[ja]' "
            "and run 'python -m unidic download' in the TTS environment, "
            "then retry."
        ),
        retryable=False,
        operation_id=operation_id,
        recovery_action="install_kokoro_japanese_dictionary",
    )


def prepare_onnx_text(
    text: str, language: str, cache: dict[str, Any]
) -> tuple[str, str, bool]:
    """Prepare text or v1 phonemes without loading a PyTorch model.

    Args:
        text: Complete requested text.
        language: Voice-derived or explicitly selected language.
        cache: Frontends owned by the calling backend instance.

    Returns:
        Input text/phonemes, upstream locale, and the is_phonemes flag.

    Raises:
        TTSOperationError: Language dependencies or dictionary setup is missing.
        RuntimeError: The upstream frontend encounters another failure.
    """
    language = language.lower().replace("_", "-")
    if language in {"fr", "fr-fr"}:
        return text, "fr-fr", False
    if language in {"ja", "j"}:
        locale, module_name, class_name, options = "ja", "misaki.ja", "JAG2P", {}
    elif language in {"zh", "zh-cn", "cmn", "z"}:
        locale, module_name, class_name, options = (
            "cmn",
            "misaki.zh",
            "ZHG2P",
            {"version": None},
        )
    else:
        return text, language, False
    extra = "ja" if locale == "ja" else "zh"
    try:
        if locale not in cache:
            if not check_dependency(module_name, f"kokoro_{extra}"):
                raise ImportError("Optional Kokoro language frontend unavailable")
            cache[locale] = getattr(import_module(module_name), class_name)(**options)
        phonemes, _ = cache[locale](text)
    except ImportError as error:
        # Some extras are imported lazily by frontend construction or use.
        raise TTSOperationError(
            code="dependency_missing",
            message=(
                f"Kokoro {extra} requires 'misaki[{extra}]'. "
                "Install it in the TTS environment, then retry."
            ),
            retryable=False,
            operation_id="kokoro_onnx",
            recovery_action="install_kokoro_language_extras",
        ) from error
    except RuntimeError as error:
        setup_error = (
            japanese_dictionary_error(error, operation_id="kokoro_onnx")
            if locale == "ja"
            else None
        )
        if setup_error is not None:
            raise setup_error from error
        raise
    return phonemes, locale, True
