import pytest

from tldw_chatbook.Image_Generation.capabilities import ResolvedReferenceImage

@pytest.fixture
def rv():
    from tldw_chatbook.Image_Generation import request_validation as m
    return m

def _codes(issues):
    return {i.path for i in issues}

def _messages(issues):
    return {i.message for i in issues}

def _ref(**overrides):
    defaults = dict(
        file_id=1,
        filename="ref.png",
        mime_type="image/png",
        width=64,
        height=64,
        bytes_len=4,
        content=b"1234",
        temp_path=None,
    )
    defaults.update(overrides)
    return ResolvedReferenceImage(**defaults)

def test_valid_request_has_no_issues(rv):
    ok = {"backend": "swarmui", "prompt": "cat", "width": 512, "height": 512, "steps": 20, "cfg_scale": 7.0, "extra_params": {}}
    assert rv.validate_image_generation_request(ok) == []

def test_oversize_dimensions_flagged(rv):
    bad = {"backend": "swarmui", "prompt": "cat", "width": 9000, "height": 9000, "extra_params": {}}
    issues = rv.validate_image_generation_request(bad)
    assert any("width" in p for p in _codes(issues))

def test_negative_cfg_scale_flagged(rv):
    bad = {"backend": "swarmui", "prompt": "cat", "cfg_scale": -1.0, "extra_params": {}}
    assert any("cfg_scale" in p for p in _codes(rv.validate_image_generation_request(bad)))

def test_extra_params_not_in_allowlist_flagged(rv):
    bad = {"backend": "swarmui", "prompt": "cat", "extra_params": {"totally_unknown": 1}}
    issues = rv.validate_image_generation_request(bad)
    assert any("extra_params" in p for p in _codes(issues))


def test_reference_image_absent_key_no_issues(rv):
    # Existing callers that never populate "reference_image" at all must be
    # completely unaffected by the new choke-point checks.
    ok = {"backend": "swarmui", "prompt": "cat", "extra_params": {}}
    assert rv.validate_image_generation_request(ok) == []


def test_reference_image_none_no_issues(rv):
    ok = {"backend": "swarmui", "prompt": "cat", "extra_params": {}, "reference_image": None}
    assert rv.validate_image_generation_request(ok) == []


@pytest.mark.parametrize(
    "backend",
    ["stable_diffusion_cpp", "swarmui", "openrouter", "novita", "together", "modelstudio"],
)
def test_reference_image_refused_for_legacy_backends(rv, backend):
    bad = {"backend": backend, "prompt": "cat", "extra_params": {}, "reference_image": _ref()}
    issues = rv.validate_image_generation_request(bad)
    assert issues == [
        rv.ImageGenerationValidationIssue(
            code="image_params_invalid",
            message=f"backend {backend!r} does not support reference images",
            path="reference_image",
        )
    ]


@pytest.mark.parametrize("backend", ["fal", "gemini"])
def test_reference_image_accepted_for_new_backends(rv, backend):
    ok = {"backend": backend, "prompt": "cat", "extra_params": {}, "reference_image": _ref()}
    assert rv.validate_image_generation_request(ok) == []


def test_reference_image_webp_accepted(rv):
    ok = {"backend": "fal", "prompt": "cat", "extra_params": {}, "reference_image": _ref(mime_type="image/webp")}
    assert rv.validate_image_generation_request(ok) == []


def test_reference_image_gif_refused(rv):
    bad = {"backend": "fal", "prompt": "cat", "extra_params": {}, "reference_image": _ref(mime_type="image/gif")}
    issues = rv.validate_image_generation_request(bad)
    assert "reference image mime 'image/gif' is not supported (png/jpeg/webp)" in _messages(issues)


def test_reference_image_oversize_refused(rv):
    bad = {
        "backend": "fal",
        "prompt": "cat",
        "extra_params": {},
        "reference_image": _ref(bytes_len=rv.IMAGE_GEN_REFERENCE_MAX_BYTES + 1),
    }
    issues = rv.validate_image_generation_request(bad)
    assert "reference image exceeds the 10MB limit" in _messages(issues)


def test_reference_image_at_exact_cap_not_refused(rv):
    ok = {
        "backend": "fal",
        "prompt": "cat",
        "extra_params": {},
        "reference_image": _ref(bytes_len=rv.IMAGE_GEN_REFERENCE_MAX_BYTES),
    }
    assert rv.validate_image_generation_request(ok) == []


def test_reference_image_no_content_refused(rv):
    bad = {
        "backend": "fal",
        "prompt": "cat",
        "extra_params": {},
        "reference_image": _ref(content=None, temp_path="/tmp/ref.png"),
    }
    issues = rv.validate_image_generation_request(bad)
    assert "reference image has no content bytes" in _messages(issues)


def test_reference_image_multiple_problems_all_reported(rv):
    # Unsupported backend + bad mime + oversize + no content, all at once --
    # the checks must not short-circuit each other.
    bad = {
        "backend": "swarmui",
        "prompt": "cat",
        "extra_params": {},
        "reference_image": _ref(
            mime_type="image/gif",
            bytes_len=rv.IMAGE_GEN_REFERENCE_MAX_BYTES + 1,
            content=None,
            temp_path="/tmp/ref.gif",
        ),
    }
    issues = rv.validate_image_generation_request(bad)
    messages = _messages(issues)
    assert "backend 'swarmui' does not support reference images" in messages
    assert "reference image mime 'image/gif' is not supported (png/jpeg/webp)" in messages
    assert "reference image exceeds the 10MB limit" in messages
    assert "reference image has no content bytes" in messages
    assert len(issues) == 4
