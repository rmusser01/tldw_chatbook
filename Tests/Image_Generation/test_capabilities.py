import pytest


def test_reference_image_capable_backends_frozenset():
    from tldw_chatbook.Image_Generation.capabilities import REFERENCE_IMAGE_CAPABLE_BACKENDS
    assert REFERENCE_IMAGE_CAPABLE_BACKENDS == frozenset({"fal", "gemini", "fireworks"})
    assert isinstance(REFERENCE_IMAGE_CAPABLE_BACKENDS, frozenset)


@pytest.mark.parametrize("backend", ["fal", "gemini", "fireworks"])
def test_resolve_backend_reference_image_capability_new_backends_supported(backend):
    from tldw_chatbook.Image_Generation.capabilities import resolve_backend_reference_image_capability
    capability = resolve_backend_reference_image_capability(backend)
    assert capability.supported is True
    assert capability.reason is None


@pytest.mark.parametrize(
    "backend",
    ["stable_diffusion_cpp", "swarmui", "openrouter", "novita", "together", "modelstudio"],
)
def test_resolve_backend_reference_image_capability_legacy_backends_unsupported(backend):
    # The new REFERENCE_IMAGE_CAPABLE_BACKENDS frozenset is the primary gate
    # here -- even modelstudio, which has a (dormant) per-model reference
    # image map, is unsupported at this backend-level query.
    from tldw_chatbook.Image_Generation.capabilities import resolve_backend_reference_image_capability
    capability = resolve_backend_reference_image_capability(backend)
    assert capability.supported is False


def test_resolve_backend_reference_image_capability_unknown_backend_unsupported():
    from tldw_chatbook.Image_Generation.capabilities import resolve_backend_reference_image_capability
    assert resolve_backend_reference_image_capability("not-a-real-backend").supported is False
    assert resolve_backend_reference_image_capability(None).supported is False


def test_resolve_backend_reference_image_capability_case_insensitive():
    from tldw_chatbook.Image_Generation.capabilities import resolve_backend_reference_image_capability
    assert resolve_backend_reference_image_capability("FAL").supported is True
    assert resolve_backend_reference_image_capability(" Gemini ").supported is True


def test_resolve_reference_image_capability_new_backends_ignore_model():
    # fal/gemini/fireworks have no per-model gating -- any model (or None)
    # is accepted once the backend itself is in the capable set.
    from tldw_chatbook.Image_Generation.capabilities import resolve_reference_image_capability
    assert resolve_reference_image_capability("fal", "any-model").supported is True
    assert resolve_reference_image_capability("gemini", None).supported is True
    assert resolve_reference_image_capability("fireworks", "").supported is True


def test_resolve_reference_image_capability_modelstudio_dormant_map_unchanged():
    # The dormant per-model map is still fully intact and reachable through
    # this model-aware function (used directly by the Model Studio adapter)
    # -- task-3 must not remove or further restrict it there.
    from tldw_chatbook.Image_Generation.capabilities import resolve_reference_image_capability
    assert resolve_reference_image_capability("modelstudio", "qwen-image-2.0").supported is True
    assert resolve_reference_image_capability("modelstudio", "qwen-image-edit").supported is True
    assert resolve_reference_image_capability("modelstudio", "some-unrelated-model").supported is False


def test_resolve_reference_image_capability_legacy_non_modelstudio_backend_unsupported():
    from tldw_chatbook.Image_Generation.capabilities import resolve_reference_image_capability
    assert resolve_reference_image_capability("swarmui", "anything").supported is False
