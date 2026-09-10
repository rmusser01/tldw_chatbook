"""Language frontend contracts without models, dictionaries or audio hardware."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.TTS import kokoro_languages as languages
from tldw_chatbook.TTS.adapter_types import TTSOperationError


@pytest.mark.parametrize(
    "language,expected",
    [("fr", "fr-fr"), ("fr_FR", "fr-fr"), ("en-us", "en-us"), ("hi", "hi")],
)
def test_espeak_locales_do_not_import_optional_east_asian_frontends(
    monkeypatch, language, expected
):
    def unexpected(*args):
        pytest.fail("This language must not import an optional frontend")

    monkeypatch.setattr(languages, "check_dependency", unexpected)
    assert languages.prepare_onnx_text("text", language, {}) == (
        "text",
        expected,
        False,
    )


@pytest.mark.parametrize(
    "language,module_name,class_name,expected_language,options",
    [
        ("ja", "misaki.ja", "JAG2P", "ja", {}),
        ("zh", "misaki.zh", "ZHG2P", "cmn", {"version": None}),
        ("zh-cn", "misaki.zh", "ZHG2P", "cmn", {"version": None}),
    ],
)
def test_onnx_uses_model_aligned_phonemes_and_reuses_its_frontend(
    monkeypatch, language, module_name, class_name, expected_language, options
):
    imports, constructions, texts = [], [], []

    def frontend(text):
        texts.append(text)
        return "ɕi phonemes", None

    def construct(**kwargs):
        constructions.append(kwargs)
        return frontend

    def import_module(name):
        imports.append(name)
        return SimpleNamespace(**{class_name: construct})

    monkeypatch.setattr(languages, "check_dependency", lambda *args: True)
    monkeypatch.setattr(languages, "import_module", import_module)
    cache = {}
    for text in ("最初", "次の文章"):
        assert languages.prepare_onnx_text(text, language, cache) == (
            "ɕi phonemes",
            expected_language,
            True,
        )
    assert imports == [module_name]
    assert constructions == [options]
    assert texts == ["最初", "次の文章"]


@pytest.mark.parametrize("language", ["ja", "zh"])
def test_missing_optional_language_extras_are_actionable(monkeypatch, language):
    monkeypatch.setattr(languages, "check_dependency", lambda *args: False)
    with pytest.raises(TTSOperationError) as caught:
        languages.prepare_onnx_text("text", language, {})
    error = caught.value
    assert error.code == "dependency_missing"
    assert not error.retryable
    assert f"misaki[{language}]" in str(error)
    assert error.recovery_action == "install_kokoro_language_extras"


@pytest.mark.parametrize("language,class_name", [("ja", "JAG2P"), ("zh", "ZHG2P")])
@pytest.mark.parametrize("stage", ["construction", "phonemization"])
def test_lazy_frontend_import_failure_remains_actionable(
    monkeypatch, language, class_name, stage
):
    def missing(*args, **kwargs):
        raise ModuleNotFoundError("Missing language dependency at /PRIVATE/path")

    constructor = missing if stage == "construction" else lambda **kwargs: missing
    monkeypatch.setattr(languages, "check_dependency", lambda *args: True)
    monkeypatch.setattr(
        languages,
        "import_module",
        lambda _: SimpleNamespace(**{class_name: constructor}),
    )
    with pytest.raises(TTSOperationError) as caught:
        languages.prepare_onnx_text("text", language, {})
    assert caught.value.code == "dependency_missing"
    assert not caught.value.retryable
    assert f"misaki[{language}]" in str(caught.value)
    assert "PRIVATE" not in str(caught.value)


def test_missing_mecab_dictionary_is_setup_guidance_without_private_paths():
    error = languages.japanese_dictionary_error(
        RuntimeError("Failed initializing MeCab: /PRIVATE/user/dicdir/mecabrc"),
        operation_id="kokoro_pytorch",
    )
    assert error.code == "configuration_invalid"
    assert not error.retryable
    assert "python -m unidic download" in str(error)
    assert "PRIVATE" not in str(error)
    assert error.operation_id == "kokoro_pytorch"


def test_unrelated_runtime_failure_is_not_mislabeled_dictionary_setup():
    assert (
        languages.japanese_dictionary_error(
            RuntimeError("out of memory"), operation_id="kokoro_pytorch"
        )
        is None
    )
