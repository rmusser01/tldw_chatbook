"""TASK-2062.3 browser-retirement and preservation ratchets."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
from tldw_chatbook.UI.Screens.llm_screen import MODELS_RAIL_SECTIONS


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_ROOT = ROOT / "tldw_chatbook"
RETIRED_FILES = (
    PRODUCTION_ROOT / "LLM_Calls/huggingface_api.py",
    *(
        PRODUCTION_ROOT / "Widgets/HuggingFace" / name
        for name in (
            "__init__.py",
            "model_browser_widget.py",
            "model_search_widget.py",
            "model_card_viewer.py",
            "download_manager.py",
            "local_models_widget.py",
        )
    ),
)
RETIRED_MODULES = (
    "tldw_chatbook.LLM_Calls.huggingface_api",
    "tldw_chatbook.Widgets.HuggingFace.model_browser_widget",
    "tldw_chatbook.Widgets.HuggingFace.model_search_widget",
    "tldw_chatbook.Widgets.HuggingFace.model_card_viewer",
    "tldw_chatbook.Widgets.HuggingFace.download_manager",
    "tldw_chatbook.Widgets.HuggingFace.local_models_widget",
)


def _production_python_with(needle: str) -> tuple[str, ...]:
    return tuple(
        path.relative_to(ROOT).as_posix()
        for path in PRODUCTION_ROOT.rglob("*.py")
        if needle in path.read_text(encoding="utf-8")
    )


def _rail_keys() -> tuple[str, ...]:
    return tuple(key for _section, rows in MODELS_RAIL_SECTIONS for key, _label in rows)


def test_legacy_models_downloader_files_are_retired() -> None:
    assert not [path.relative_to(ROOT) for path in RETIRED_FILES if path.exists()]


@pytest.mark.parametrize("module_name", RETIRED_MODULES)
def test_legacy_models_downloader_modules_are_not_importable(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize(
    "retired_reference",
    (
        "Widgets.HuggingFace",
        "HuggingFaceModelBrowser",
        "ModelSearchWidget",
        "ModelCardViewer",
        "DownloadManager",
        "LocalModelsWidget",
        "LLM_Calls.huggingface_api",
    ),
)
def test_production_has_no_legacy_browser_reference(retired_reference: str) -> None:
    assert _production_python_with(retired_reference) == ()


def test_models_rail_and_view_mapping_have_no_legacy_browser_destination() -> None:
    view_mapping = LLMManagementWindow(None).view_mapping
    assert "download-models" not in _rail_keys()
    assert "download-models" not in view_mapping
    assert "llm-view-download-models" not in view_mapping.values()


def test_production_has_no_retired_download_models_empty_state_action() -> None:
    assert _production_python_with("empty-state-download-models") == ()


def test_installed_external_and_remote_rail_destinations_remain() -> None:
    keys = _rail_keys()
    assert "installed" in keys
    assert "external" in keys
    assert "remote" in keys


def test_configured_legacy_download_root_still_reaches_installed_view() -> None:
    source = inspect.getsource(LLMManagementWindow._mount_deferred_views)
    assert '"model_download_dir"' in source
    assert "legacy_dir = Path(str(configured)).expanduser()" in source
    assert "legacy_dir=legacy_dir" in source


def test_transformers_local_directory_controls_remain() -> None:
    source = (PRODUCTION_ROOT / "UI/LLM_Management_Window.py").read_text(
        encoding="utf-8"
    )
    for fragment in (
        'id="transformers-models-dir-path"',
        '"Browse Dir",',
        'id="transformers-browse-models-dir-button"',
        '"List Local Models",',
        'id="transformers-list-local-models-button"',
    ):
        assert fragment in source


def test_llamacpp_and_llamafile_external_gguf_controls_remain() -> None:
    source = (PRODUCTION_ROOT / "UI/LLM_Management_Window.py").read_text(
        encoding="utf-8"
    )
    assert LLMManagementWindow.GGUF_PROVIDERS == ("llamacpp", "llamafile")
    assert '"External GGUF", GGUFSourceMode.EXTERNAL.value' in source
    for fragment in (
        'id=f"{provider}-gguf-source-mode"',
        'id=f"{provider}-model-path"',
        'id=f"{provider}-browse-model-button"',
    ):
        assert fragment in source


def test_remote_acquisition_and_hugging_face_inference_owners_remain() -> None:
    remote_view = importlib.import_module("tldw_chatbook.UI.Screens.model_remote_view")
    remote_adapter = importlib.import_module(
        "tldw_chatbook.Model_Artifacts.remote_huggingface"
    )
    llm_api = importlib.import_module("tldw_chatbook.LLM_Calls.LLM_API_Calls")
    summarization = importlib.import_module(
        "tldw_chatbook.LLM_Calls.Summarization_General_Lib"
    )

    assert hasattr(remote_view, "RemoteView")
    assert hasattr(remote_adapter, "HuggingFaceRemoteAdapter")
    assert hasattr(llm_api, "chat_with_huggingface")
    assert hasattr(summarization, "summarize_with_huggingface")


def test_model_download_dir_is_only_config_and_installed_legacy_scan_wiring() -> None:
    assert set(_production_python_with("model_download_dir")) == {
        "tldw_chatbook/config.py",
        "tldw_chatbook/UI/LLM_Management_Window.py",
    }
