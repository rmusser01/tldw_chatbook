from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button

from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow


class _LLMWindowHost(App):
    def compose(self) -> ComposeResult:
        app_instance = SimpleNamespace(
            app_config={"llm_management": {"model_download_dir": "/private/tmp/tldw-models"}}
        )
        yield LLMManagementWindow(app_instance)


@pytest.mark.asyncio
async def test_llm_runtime_browse_controls_explain_expected_paths(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Widgets.HuggingFace.model_search_widget.ModelSearchWidget._initial_browse",
        lambda self: None,
    )
    monkeypatch.setattr(
        "tldw_chatbook.Widgets.HuggingFace.download_manager.DownloadManager.on_mount",
        lambda self: None,
    )
    monkeypatch.setattr(
        "tldw_chatbook.Widgets.HuggingFace.local_models_widget.LocalModelsWidget.on_mount",
        lambda self: None,
    )

    expected_tooltips = {
        "llamacpp-browse-exec-button": "Choose the llama.cpp server executable.",
        "llamacpp-browse-model-button": "Choose a GGUF model file for llama.cpp.",
        "llamafile-browse-exec-button": "Choose the llamafile executable.",
        "llamafile-browse-model-button": "Choose an optional external GGUF model for llamafile.",
        "vllm-browse-python-button": "Choose the Python interpreter used to launch vLLM.",
        "vllm-browse-model-button": "Choose a local model directory for vLLM, or type a Hugging Face repo ID.",
        "onnx-browse-python-button": "Choose the Python interpreter used to launch the ONNX server.",
        "onnx-browse-script-button": "Choose the ONNX server script to run.",
        "onnx-browse-model-button": "Choose the ONNX model file or directory to load.",
        "transformers-browse-models-dir-button": "Choose the local Transformers models root directory.",
        "transformers-browse-script-button": "Choose the custom Transformers server script to run.",
        "mlx-browse-model-button": "Choose a local MLX model path, or type a Hugging Face repo ID.",
        "ollama-browse-exec-button": "Choose the Ollama executable.",
        "ollama-browse-modelfile-button": "Choose the Modelfile used to create an Ollama model.",
    }

    app = _LLMWindowHost()

    async with app.run_test() as pilot:
        await pilot.pause()
        window = app.query_one(LLMManagementWindow)

        for button_id, expected_tooltip in expected_tooltips.items():
            button = window.query_one(f"#{button_id}", Button)
            assert str(button.tooltip) == expected_tooltip
