import pytest


@pytest.fixture(autouse=True)
def _reset():
    from tldw_chatbook.Video_Generation import adapter_registry as r
    r.reset_registry()
    yield
    r.reset_registry()


def _request(**overrides):
    from tldw_chatbook.Video_Generation.worker import build_request
    kwargs = {"backend": "fake", "prompt": "a kite over the harbor"}
    kwargs.update(overrides)
    return build_request(**kwargs)


def _register_fake(result=None):
    from tldw_chatbook.Video_Generation.adapter_registry import get_registry
    from tldw_chatbook.Video_Generation.adapters.base import VideoGenResult

    class FakeAdapter:
        name = "fake"
        supported_formats = {"mp4"}

        def generate(self, request):
            return result or VideoGenResult(content=b"vid", content_type="video/mp4", bytes_len=3)

    registry = get_registry()
    registry._enabled_backends = ["fake"]
    registry._default_backend = "fake"
    registry.register_adapter("fake", FakeAdapter)
    return registry


def test_run_generation_refuses_disabled_backend():
    from tldw_chatbook.Video_Generation.worker import run_generation
    from tldw_chatbook.Video_Generation.exceptions import VideoGenerationError

    with pytest.raises(VideoGenerationError, match="not enabled/available"):
        run_generation(_request(backend="fake"))


def test_run_generation_refuses_invalid_request_before_dispatch():
    from tldw_chatbook.Video_Generation.worker import run_generation
    from tldw_chatbook.Video_Generation.exceptions import VideoGenerationError

    _register_fake()
    # duration exceeds the configured cap (default 15): the choke point must
    # reject before the adapter's generate() is ever called.
    with pytest.raises(VideoGenerationError, match="Invalid video generation request"):
        run_generation(_request(duration_seconds=999))


def test_run_generation_happy_path_calls_adapter():
    from tldw_chatbook.Video_Generation.worker import run_generation

    _register_fake()
    result = run_generation(_request(duration_seconds=5, fps=24))
    assert result.content == b"vid"
    assert result.content_type == "video/mp4"


def test_build_request_defaults():
    req = _request()
    assert req.format == "mp4"
    assert req.extra_params == {}
    assert req.reference_assets == ()
    assert req.duration_seconds is None
