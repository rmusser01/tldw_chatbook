"""TASK-2062.3 browser-retirement and preservation ratchets."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML
from tldw_chatbook.Event_Handlers.LLM_Management_Events.llm_management_events_transformers import (
    TRANSFORMERS_BUTTON_HANDLERS,
)
from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
from tldw_chatbook.UI.Screens.llm_screen import MODELS_RAIL_SECTIONS


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_ROOT = ROOT / "tldw_chatbook"
TRANSFORMERS_EVENTS_PATH = (
    PRODUCTION_ROOT
    / "Event_Handlers/LLM_Management_Events/llm_management_events_transformers.py"
)
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
    """Reject any surviving legacy downloader file."""
    assert not [path.relative_to(ROOT) for path in RETIRED_FILES if path.exists()]


@pytest.mark.parametrize("module_name", RETIRED_MODULES)
def test_legacy_models_downloader_modules_are_not_importable(module_name: str) -> None:
    """Keep every retired downloader module non-importable.

    Args:
        module_name: Retired import path under test.
    """
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
    """Reject production references to retired browser owners.

    Args:
        retired_reference: Retired symbol or module fragment under test.
    """
    assert _production_python_with(retired_reference) == ()


def test_models_rail_and_view_mapping_have_no_legacy_browser_destination() -> None:
    """Keep the retired browser absent from Models navigation and view routing."""
    view_mapping = LLMManagementWindow(None).view_mapping
    assert "download-models" not in _rail_keys()
    assert "download-models" not in view_mapping
    assert "llm-view-download-models" not in view_mapping.values()


def test_production_has_no_retired_download_models_empty_state_action() -> None:
    """Keep production empty states free of the retired download action."""
    assert _production_python_with("empty-state-download-models") == ()


def test_installed_external_and_remote_rail_destinations_remain() -> None:
    """Preserve the Installed, External, and Remote Models destinations."""
    keys = _rail_keys()
    assert "installed" in keys
    assert "external" in keys
    assert "remote" in keys


def test_configured_legacy_download_root_still_reaches_installed_view() -> None:
    """Preserve the configured legacy root as an Installed read-only scan path."""
    source = inspect.getsource(LLMManagementWindow._mount_deferred_views)
    assert '"model_download_dir"' in source
    assert "legacy_dir = Path(str(configured)).expanduser()" in source
    assert "legacy_dir=legacy_dir" in source


def test_transformers_local_directory_controls_remain() -> None:
    """Preserve Transformers local directory browse and list controls."""
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


def test_transformers_cache_hint_uses_optional_dependency_guard() -> None:
    """Keep the surviving cache hint behind the shared optional-deps guard."""
    source = TRANSFORMERS_EVENTS_PATH.read_text(encoding="utf-8")

    assert 'get_safe_import("huggingface_hub")' in source
    assert 'importlib.import_module("huggingface_hub.constants")' in source
    assert "from huggingface_hub import" not in source


def test_transformers_direct_downloader_is_retired() -> None:
    """Keep every Transformers direct-download control and handler retired."""
    window_source = (PRODUCTION_ROOT / "UI/LLM_Management_Window.py").read_text(
        encoding="utf-8"
    )
    events_source = TRANSFORMERS_EVENTS_PATH.read_text(encoding="utf-8")

    for fragment in (
        '"Download New Model:"',
        'id="transformers-download-repo-id"',
        'id="transformers-download-revision"',
        'id="transformers-download-model-button"',
    ):
        assert fragment not in window_source

    for fragment in (
        "_valid_huggingface_repo_id",
        "_valid_huggingface_revision",
        "run_transformers_model_download_worker",
        "handle_transformers_download_model_button_pressed",
        "subprocess.Popen",
        "functools.partial",
        "target_model_specific_dir.mkdir",
        '"huggingface-cli"',
    ):
        assert fragment not in events_source

    assert set(TRANSFORMERS_BUTTON_HANDLERS) == {
        "transformers-list-local-models-button",
        "transformers-browse-models-dir-button",
    }


def test_llamacpp_and_llamafile_external_gguf_controls_remain() -> None:
    """Preserve External GGUF controls for llama.cpp and llamafile."""
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
    """Preserve managed remote acquisition and Hugging Face inference owners."""
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
    """Constrain model_download_dir to config and the Installed scan seam."""
    assert set(_production_python_with("model_download_dir")) == {
        "tldw_chatbook/config.py",
        "tldw_chatbook/UI/LLM_Management_Window.py",
    }


def test_default_llm_management_config_only_keeps_installed_legacy_scan_root() -> None:
    """Keep only the Installed legacy scan root in default Models config."""
    assert DEFAULT_CONFIG_FROM_TOML["llm_management"] == {
        "model_download_dir": "~/Downloads/tldw_models"
    }
    config_source = (PRODUCTION_ROOT / "config.py").read_text(encoding="utf-8")
    assert (
        'model_download_dir = "~/Downloads/tldw_models"'
        "  # Legacy read-only scan root for Installed models"
    ) in config_source
