"""omnivoice_tts optional-feature registration."""

from tldw_chatbook.Utils import optional_deps


def test_omnivoice_tts_feature_registered() -> None:
    info = optional_deps.OPTIONAL_FEATURES["omnivoice_tts"]
    assert info.extra == "omnivoice_tts"
    assert set(info.package_dependencies) >= {"onnxruntime", "tokenizers"}


def test_omnivoice_tts_defaults_false() -> None:
    assert optional_deps.DEPENDENCIES_AVAILABLE["omnivoice_tts"] is False
