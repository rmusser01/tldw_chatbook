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


def _register_fake(result=None, dispatches=None):
    from tldw_chatbook.Video_Generation.adapter_registry import get_registry
    from tldw_chatbook.Video_Generation.adapters.base import VideoGenResult

    class FakeAdapter:
        name = "fake"
        supported_formats = {"mp4", "webm"}

        def generate(self, request):
            if dispatches is not None:
                dispatches.append(request)
            return result or VideoGenResult(
                content=b"vid", content_type="video/mp4", container="mp4", bytes_len=3
            )

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
    assert result.container == "mp4"


def test_run_generation_refuses_unknown_format_before_adapter_dispatch():
    from tldw_chatbook.Video_Generation.exceptions import VideoGenerationError
    from tldw_chatbook.Video_Generation.worker import run_generation

    dispatches = []
    _register_fake(dispatches=dispatches)

    with pytest.raises(VideoGenerationError, match="request format"):
        run_generation(_request(video_format="mov"))

    assert dispatches == []


@pytest.mark.parametrize(
    ("requested", "container", "content_type"),
    [
        ("mp4", "webm", "video/webm"),
        ("mp4", "mp4", "video/webm"),
        ("webm", "webm", "video/mp4"),
        ("mp4", "mov", "video/mp4"),
        ("mp4", "mp4", "application/octet-stream"),
    ],
)
def test_run_generation_rejects_unknown_or_contradictory_result_facts(
    requested, container, content_type
):
    from tldw_chatbook.Video_Generation.adapters.base import VideoGenResult
    from tldw_chatbook.Video_Generation.exceptions import VideoGenerationError
    from tldw_chatbook.Video_Generation.worker import run_generation

    dispatches = []
    result = VideoGenResult(
        content=b"PRIVATE-BYTES",
        content_type=content_type,
        container=container,
        bytes_len=13,
    )
    _register_fake(result=result, dispatches=dispatches)

    with pytest.raises(VideoGenerationError, match="result format") as exc_info:
        run_generation(_request(video_format=requested))

    assert len(dispatches) == 1
    assert "PRIVATE-BYTES" not in str(exc_info.value)
    assert container not in str(exc_info.value)
    assert content_type not in str(exc_info.value)


@pytest.mark.parametrize("case", ["missing", "malformed", "spoof", "raising"])
def test_run_generation_contains_malformed_or_hostile_result_containers(case):
    from types import SimpleNamespace

    from tldw_chatbook.Video_Generation.exceptions import VideoGenerationError
    from tldw_chatbook.Video_Generation.worker import run_generation

    class EqualitySpoof:
        def __eq__(self, _other):
            return True

    class EqualityTrap:
        def __eq__(self, _other):
            raise RuntimeError("PRIVATE-EQUALITY-ERROR")

    containers = {
        "malformed": None,
        "spoof": EqualitySpoof(),
        "raising": EqualityTrap(),
    }
    result = SimpleNamespace(content_type="video/mp4")
    if case != "missing":
        result.container = containers[case]
    _register_fake(result=result)

    with pytest.raises(VideoGenerationError) as exc_info:
        run_generation(_request())

    assert str(exc_info.value) == "Invalid video generation result format"
    assert "PRIVATE-EQUALITY-ERROR" not in str(exc_info.value)


def test_run_generation_suppresses_hostile_result_property_traceback_details():
    import traceback

    from tldw_chatbook.Video_Generation.exceptions import VideoGenerationError
    from tldw_chatbook.Video_Generation.worker import run_generation

    class HostileResult:
        @property
        def container(self):
            raise RuntimeError("PRIVATE-RESULT-PROPERTY")

    _register_fake(result=HostileResult())

    with pytest.raises(VideoGenerationError) as exc_info:
        run_generation(_request())

    error = exc_info.value
    formatted = "".join(traceback.format_exception(error))
    assert str(error) == "Invalid video generation result format"
    assert error.__cause__ is None
    assert error.__context__ is not None
    assert error.__suppress_context__ is True
    assert "PRIVATE-RESULT-PROPERTY" not in formatted


def test_build_request_defaults():
    req = _request()
    assert req.format == "mp4"
    assert req.extra_params == {}
    assert req.reference_assets == ()
    assert req.duration_seconds is None
