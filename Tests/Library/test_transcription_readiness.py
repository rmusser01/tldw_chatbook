"""Download-free selected-backend checks before Library queue creation."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.Library.transcription_readiness import (
    TranscriptionReadinessError,
    check_transcription_readiness,
)


def _runtime(**attributes):
    return SimpleNamespace(**attributes)


@pytest.mark.parametrize(
    "provider,module,extra",
    [
        ("default", "faster_whisper", "transcription_faster_whisper"),
        ("faster-whisper", "faster_whisper", "transcription_faster_whisper"),
        ("parakeet-onnx", "onnx_asr", "transcription_parakeet_onnx"),
        ("transcribe-cpp", "transcribe_cpp", "transcription_transcribe_cpp"),
    ],
)
def test_missing_selected_package_names_its_install_remedy(provider, module, extra):
    def missing(name):
        assert name == module
        raise ModuleNotFoundError(name)

    with pytest.raises(TranscriptionReadinessError) as caught:
        check_transcription_readiness(
            {"transcription_provider": provider}, import_module=missing
        )
    assert caught.value.reason == "package_missing"
    assert extra in str(caught.value)
    assert "Python environment" in str(caught.value)


@pytest.mark.parametrize("vocabulary", ["vocabulary.json", "vocabulary.txt"])
def test_whisper_checks_exact_selected_cache_without_downloading(tmp_path, vocabulary):
    calls = []
    model = tmp_path / "cached"
    model.mkdir()
    (model / "model.bin").write_bytes(b"weights")
    (model / "config.json").write_text("{}")
    (model / "tokenizer.json").write_text("{}")
    (model / vocabulary).write_text("[]")

    def cached(name, *, local_files_only):
        calls.append((name, local_files_only))
        return str(model)

    def importer(name):
        assert name in {"faster_whisper", "faster_whisper.utils"}
        return _runtime(download_model=cached)

    check_transcription_readiness(
        {"transcription_provider": "faster-whisper", "transcription_model": "small.en"},
        import_module=importer,
    )
    assert calls == [("small.en", True)]


@pytest.mark.parametrize(
    "absent", ["cache", "model.bin", "config.json", "tokenizer.json", "vocabulary.json"]
)
def test_whisper_missing_or_incomplete_cache_has_model_remedy(tmp_path, absent):
    for filename in ("model.bin", "config.json", "tokenizer.json", "vocabulary.json"):
        if filename != absent:
            (tmp_path / filename).write_text("test")

    def cached(*args, **kwargs):
        if absent == "cache":
            raise FileNotFoundError("/private/cache/token")
        return str(tmp_path)

    with pytest.raises(TranscriptionReadinessError) as caught:
        check_transcription_readiness(
            {"transcription_provider": "faster-whisper", "transcription_model": "base"},
            import_module=lambda name: _runtime(download_model=cached),
        )
    assert caught.value.reason == "model_missing"
    assert "base" in str(caught.value)
    assert "download" in str(caught.value).lower()
    assert str(tmp_path) not in str(caught.value)
    assert "/private" not in str(caught.value)


def test_parakeet_uses_exact_language_precision_and_source_service():
    calls = []
    service = _runtime(resolve=lambda key, **kwargs: calls.append((key, kwargs)))
    check_transcription_readiness(
        {
            "transcription_provider": "parakeet-onnx",
            "language": "de",
            "transcription_precision": "f32",
        },
        import_module=lambda name: _runtime(),
        parakeet_source_service=service,
    )
    assert calls[0][0].value == "v3_f32"


def test_external_parakeet_preserves_existing_folder_and_vad_consent_flow():
    def unexpected(*args, **kwargs):
        raise AssertionError(
            "external verification belongs to its existing consent flow"
        )

    check_transcription_readiness(
        {
            "transcription_provider": "parakeet-onnx",
            "transcription_model_dir": "/private/model",
        },
        import_module=lambda name: _runtime(),
        parakeet_source_service=_runtime(resolve=unexpected),
    )


def test_missing_parakeet_bundle_is_not_called_incompatible():
    def missing(*args, **kwargs):
        raise FileNotFoundError("/private/not-installed")

    with pytest.raises(TranscriptionReadinessError) as caught:
        check_transcription_readiness(
            {"transcription_provider": "parakeet-onnx"},
            import_module=lambda name: _runtime(),
            parakeet_source_service=_runtime(resolve=missing),
        )
    assert caught.value.reason == "model_missing"
    assert "Install" in str(caught.value)
    assert "Parakeet" in str(caught.value)
    assert "/private" not in str(caught.value)


def test_missing_managed_vad_names_vad_instead_of_the_asr_model():
    from tldw_chatbook.STT.parakeet_sources import (
        ParakeetSourceError,
        ParakeetSourceErrorCode,
    )

    def missing(*args, **kwargs):
        raise ParakeetSourceError(ParakeetSourceErrorCode.VAD_UNAVAILABLE)

    with pytest.raises(TranscriptionReadinessError) as caught:
        check_transcription_readiness(
            {"transcription_provider": "parakeet-onnx"},
            import_module=lambda name: _runtime(),
            parakeet_source_service=_runtime(resolve=missing),
        )
    assert caught.value.reason == "vad_missing"
    assert "Silero VAD" in str(caught.value)
    assert "Install" in str(caught.value)


def test_transcribe_cpp_requires_a_selected_gguf_before_queueing():
    with pytest.raises(TranscriptionReadinessError) as caught:
        check_transcription_readiness(
            {"transcription_provider": "transcribe-cpp"},
            import_module=lambda name: _runtime(),
        )
    assert caught.value.reason == "model_missing"
    assert "GGUF" in str(caught.value)


def test_dispatch_classifies_absent_parakeet_files_as_missing_model():
    from tldw_chatbook.app_ingest_queue import LibraryIngestQueueMixin
    from tldw_chatbook.STT.contracts import TranscriptionFailureCode

    code, _ = LibraryIngestQueueMixin._classify_local_stt_dispatch_error(
        "parakeet-onnx", FileNotFoundError("/private/missing")
    )
    assert code is TranscriptionFailureCode.MODEL_NOT_INSTALLED


@pytest.mark.parametrize("provider", ["parakeet-onnx", "transcribe-cpp"])
def test_closed_translate_checkbox_does_not_change_selected_route(provider):
    # The form retains checkbox values when their provider gate closes.
    with pytest.raises(TranscriptionReadinessError) as caught:
        check_transcription_readiness(
            {"transcription_provider": provider, "translate_to_english": True},
            import_module=lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name)),
        )
    assert caught.value.reason == "package_missing"


def test_explicit_translation_target_wins_over_checkbox():
    from tldw_chatbook.Local_Ingestion.stt_batch_routing import BatchSTTRoutingError

    with pytest.raises(BatchSTTRoutingError, match="only supports target en"):
        check_transcription_readiness(
            {
                "transcription_provider": "faster-whisper",
                "translate_to_english": True,
                "translation_target_language": "de",
            },
            import_module=lambda name: _runtime(),
        )


def test_whisper_accepts_model_directory_with_shell_punctuation(tmp_path):
    root = tmp_path / "Whisper;custom"
    root.mkdir()
    for name in ("model.bin", "config.json", "tokenizer.json", "vocabulary.json"):
        (root / name).write_text("test")
    check_transcription_readiness(
        {"transcription_provider": "default"},
        default_model=str(root),
        import_module=lambda name: _runtime(),
    )


@pytest.mark.parametrize(
    "model", ["distil-large-v3", "deepdml/faster-whisper-large-v3-turbo-ct2"]
)
def test_missing_configured_whisper_catalog_model_names_exact_download(model):
    def missing(*args, **kwargs):
        raise FileNotFoundError("/private/cache")

    with pytest.raises(TranscriptionReadinessError) as caught:
        check_transcription_readiness(
            {"transcription_provider": "default"},
            default_model=model,
            import_module=lambda name: _runtime(download_model=missing),
        )
    assert caught.value.reason == "model_missing"
    assert f"download_model('{model}')" in str(caught.value)
    assert "/private" not in str(caught.value)


def test_missing_custom_whisper_path_is_not_echoed_in_remedy(tmp_path):
    model = str(tmp_path / "private-model")

    def missing(*args, **kwargs):
        raise FileNotFoundError(model)

    with pytest.raises(TranscriptionReadinessError) as caught:
        check_transcription_readiness(
            {"transcription_provider": "default"},
            default_model=model,
            import_module=lambda name: _runtime(download_model=missing),
        )
    assert caught.value.reason == "model_missing"
    assert model not in str(caught.value)


@pytest.mark.parametrize("saved_model", [None, "distil-large-v3", "org/custom-whisper"])
def test_auto_uses_fresh_base_without_rewriting_saved_model(
    tmp_path, monkeypatch, saved_model
):
    from Tests.Backup_Recovery.config_test_support import select_config_source
    from tldw_chatbook.Local_Ingestion import transcription_service
    from tldw_chatbook.UI.Screens import library_screen

    path = tmp_path / "config.toml"
    original = None
    if saved_model is not None:
        original = f'[transcription]\ndefault_model = "{saved_model}"\ndevice = "cpu"\n'.encode()
        path.write_bytes(original)
    source = select_config_source(
        monkeypatch, path, transcription_service.__dict__, library_screen.__dict__
    )
    backend = transcription_service._LegacyTranscriptionBackend()
    expected = saved_model or "base"
    assert backend.config["default_model"] == expected
    assert source.get_cli_setting("transcription.default_model", "base") == expected
    payload = library_screen._read_library_ingest_options_from_config()
    assert payload["transcription_default_model"] == expected
    from types import SimpleNamespace
    from tldw_chatbook.UI.Library_Modules.library_ingest_state import LibraryIngestState
    from tldw_chatbook.Library.library_ingest_jobs import LibraryIngestJobRegistry
    from tldw_chatbook.app_ingest_queue import LibraryIngestQueueMixin

    form = _transcription_preview({"transcription_provider": "default"}, expected).form
    form.analyze = False
    form.chunk = False
    registry = LibraryIngestJobRegistry()
    owner = SimpleNamespace(
        _ingest_state=LibraryIngestState(form=form),
        _media_state=SimpleNamespace(analyze_running=False),
        _library_ingest_registry=lambda: registry,
        app_instance=SimpleNamespace(media_db=object()),
    )
    library_screen.LibraryScreen._load_library_ingest_options_from_config(owner)
    assert owner._transcription_default_model == expected
    state = library_screen.LibraryScreen._build_library_ingest_state(owner)
    label = expected if saved_model != "org/custom-whisper" else "custom model"
    assert f"Faster Whisper · {label} (Auto)" in state.commit_summary_line

    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"")
    job = registry.submit(
        source_path=str(audio),
        detected_type="audio",
        chunk_enabled=False,
        perform_analysis=False,
        ingest_options={
            "audio_video": {
                "transcription_provider": "default",
                "transcription_model": "large-v3",
            }
        },
    )
    options = LibraryIngestQueueMixin._ingest_job_options(
        SimpleNamespace(media_db=None), job
    )
    observed = []

    def transcribe_native(path, model, *args, **kwargs):
        observed.append(model)
        return {"text": "probe", "segments": []}

    monkeypatch.setattr(transcription_service, "FASTER_WHISPER_AVAILABLE", True)
    monkeypatch.setattr(backend, "_transcribe_with_faster_whisper", transcribe_native)
    backend.transcribe(
        str(audio),
        provider=options["transcription_provider"],
        model=options["transcription_model"],
        language="en",
        batch_route_resolved=True,
    )
    assert observed == [expected]
    if original is not None:
        assert path.read_bytes() == original
    else:
        import tomllib

        assert (
            tomllib.loads(path.read_text(encoding="utf-8"))["transcription"][
                "default_model"
            ]
            == "base"
        )


def _transcription_preview(
    options, default_model="base", *, group="audio_video", server=False
):
    from tldw_chatbook.Library.ingest_types import PreflightResult
    from tldw_chatbook.Library.library_ingest_state import (
        LibraryIngestFormState,
        build_library_ingest_state,
    )

    path = "clip.mp4" if group == "audio_video" else "notes.txt"
    form = LibraryIngestFormState(path=path, type_options={"audio_video": options})
    form.preflight = PreflightResult(
        type_groups={group: [path]},
        warnings=[],
        errors=[],
        total_files=1,
        total_size=1,
        truncated=False,
    )
    return build_library_ingest_state(
        (),
        form=form,
        transcription_default_model=default_model,
        ingest_backend="server" if server else "local",
        runtime_source="server" if server else "local",
        server_ingest_available=server,
        transcribe_cpp_configured=True,
    )


@pytest.mark.parametrize(
    "options,default_model,expected",
    [
        (
            {"transcription_provider": "default", "transcription_model": "large-v3"},
            "base",
            "Faster Whisper · base (Auto)",
        ),
        (
            {"transcription_provider": "default", "transcription_model": "base"},
            "distil-large-v3",
            "Faster Whisper · distil-large-v3 (Auto)",
        ),
        (
            {
                "transcription_provider": "faster-whisper",
                "transcription_model": "small.en",
            },
            "distil-large-v3",
            "Faster Whisper · small.en",
        ),
        (
            {
                "transcription_provider": "faster-whisper",
                "model": "large-v3",
                "transcription_model": "base",
            },
            "base",
            "Faster Whisper · large-v3",
        ),
        (
            {"transcription_provider": "parakeet-onnx", "language": "en"},
            "base",
            "Parakeet ONNX · nemo-parakeet-tdt-0.6b-v2 · INT8",
        ),
        (
            {
                "transcription_provider": "parakeet-onnx",
                "language": "de",
                "transcription_precision": "f32",
            },
            "base",
            "Parakeet ONNX · nemo-parakeet-tdt-0.6b-v3 · F32",
        ),
        (
            {"transcription_provider": "transcribe-cpp"},
            "base",
            "transcribe.cpp · configured GGUF",
        ),
    ],
)
def test_import_summary_names_the_effective_transcription_model(
    options, default_model, expected
):
    state = _transcription_preview(options, default_model)
    assert f"Transcription: {expected}" in state.commit_summary_line


@pytest.mark.parametrize("group,server", [("generic", False), ("audio_video", True)])
def test_local_transcription_preview_is_absent_for_other_import_targets(group, server):
    state = _transcription_preview(
        {"transcription_provider": "default"}, group=group, server=server
    )
    assert "Transcription:" not in state.commit_summary_line


def test_custom_transcription_model_preview_does_not_expose_path_or_markup():
    state = _transcription_preview(
        {"transcription_provider": "default"}, "/private/[red]weights/model"
    )
    assert "Faster Whisper · custom model (Auto)" in state.commit_summary_line
    assert "/private" not in state.commit_summary_line
    assert "[red]" not in state.commit_summary_line


def test_selected_unavailable_server_backend_does_not_advertise_local_model():
    from tldw_chatbook.Library.library_ingest_state import build_library_ingest_state

    form = _transcription_preview({"transcription_provider": "default"}).form
    state = build_library_ingest_state(
        (), form=form, ingest_backend="server", server_ingest_available=False
    )
    assert "Transcription:" not in state.commit_summary_line


def test_direct_local_preview_does_not_claim_a_missing_gguf_is_configured():
    from tldw_chatbook.Library.library_ingest_state import build_library_ingest_state

    form = _transcription_preview({"transcription_provider": "transcribe-cpp"}).form
    state = build_library_ingest_state((), form=form, transcribe_cpp_configured=False)
    assert "transcribe.cpp · choose a GGUF model" in state.commit_summary_line
    assert "configured GGUF" not in state.commit_summary_line
