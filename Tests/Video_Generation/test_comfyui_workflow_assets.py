import json
from pathlib import Path

import pytest


WORKFLOW_DIR = Path(__file__).parents[2] / "tldw_chatbook" / "Video_Generation" / "workflows"
WORKFLOW_NAMES = ("minimax_h3_t2v.json", "minimax_h3_t2v_spectrum.json")
FRAME_GRID_EXPRESSION = (
    "max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17"
)
COMMON_CLASS_TYPES = {
    "92": "SaveVideo",
    "105:6": "UNETLoader",
    "105:9": "BasicScheduler",
    "105:10": "VAEDecode",
    "105:11": "VAELoader",
    "105:13": "CLIPLoader",
    "105:14": "SamplerCustomAdvanced",
    "105:15": "RandomNoise",
    "105:16": "BasicGuider",
    "105:17": "KSamplerSelect",
    "105:23": "VAEDecodeAudio",
    "105:24": "VAELoader",
    "105:91": "CreateVideo",
    "105:104": "MiniMaxH3ImageToVideo",
    "105:107": "ComfyMathExpression",
    "105:111": "PrimitiveFloat",
}
SAFE_FILLER = (
    "An atmospheric cinematic shot of a red sailboat crossing a calm lake at sunrise. "
    "Gentle wind ripples the water and nearby reeds while the camera slowly tracks from "
    "left to right. Natural ambient sound with distant birds and soft water. No text, "
    "logos, or watermarks."
)


def _load(name: str) -> dict:
    return json.loads((WORKFLOW_DIR / name).read_text(encoding="utf-8"))


def _prompts(graph: dict) -> list[str]:
    return [
        node["inputs"]["prompt"]
        for node in graph.values()
        if node.get("class_type") == "MiniMaxH3ImageToVideo"
    ]


def test_h3_assets_are_sanitized_api_graphs():
    for name in WORKFLOW_NAMES:
        graph = _load(name)
        assert graph and all(node.get("class_type") for node in graph.values())
        assert _prompts(graph) == [SAFE_FILLER]
        assert graph["105:104"]["inputs"]["width"] == 864
        assert graph["105:104"]["inputs"]["height"] == 480
        assert graph["105:104"]["inputs"]["length"] == ["105:107", 1]
        assert graph["105:104"]["_meta"]["title"] == "Prompt Width Height"
        assert graph["105:15"]["inputs"]["noise_seed"] == 0
        assert graph["105:15"]["_meta"]["title"] == "Seed"
        assert graph["105:111"]["inputs"]["value"] == 5
        assert graph["105:111"]["_meta"]["title"] == "Duration"
        assert graph["105:91"]["inputs"]["fps"] == 24
        assert graph["105:91"]["_meta"]["title"] == "Native FPS"
        assert graph["92"]["inputs"]["format"] == "mp4"
        assert "115" not in graph


def test_spectrum_is_opt_in_and_preserves_model_routes():
    base = _load("minimax_h3_t2v.json")
    spectrum = _load("minimax_h3_t2v_spectrum.json")
    assert "SpectrumApplyMiniMaxH3" not in {
        node["class_type"] for node in base.values()
    }
    assert spectrum["105:120"]["class_type"] == "SpectrumApplyMiniMaxH3"
    assert spectrum["105:9"]["inputs"]["model"] == ["105:120", 0]
    assert spectrum["105:16"]["inputs"]["model"] == ["105:120", 0]


@pytest.mark.parametrize(
    ("name", "effective_model"),
    [
        ("minimax_h3_t2v.json", ["105:6", 0]),
        ("minimax_h3_t2v_spectrum.json", ["105:120", 0]),
    ],
)
def test_h3_assets_preserve_model_sampler_and_audio_topology(name, effective_model):
    graph = _load(name)
    expected_class_types = dict(COMMON_CLASS_TYPES)
    if name.endswith("_spectrum.json"):
        expected_class_types["105:120"] = "SpectrumApplyMiniMaxH3"
    assert {
        node_id: node["class_type"] for node_id, node in graph.items()
    } == expected_class_types

    assert graph["105:6"]["inputs"]["unet_name"] == (
        "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    )
    assert graph["105:13"]["inputs"]["clip_name"] == (
        "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    )
    assert graph["105:13"]["inputs"]["type"] == "minimax"
    assert graph["105:11"]["inputs"]["vae_name"] == (
        "minimax_h3_video_vae_fp16.safetensors"
    )
    assert graph["105:24"]["inputs"]["vae_name"] == (
        "minimax_h3_audio_vae_fp32.safetensors"
    )

    assert graph["105:17"]["inputs"]["sampler_name"] == "res_multistep"
    assert graph["105:9"]["inputs"] == {
        "scheduler": "simple",
        "steps": 20,
        "denoise": 1,
        "model": effective_model,
    }
    assert graph["105:14"]["inputs"] == {
        "noise": ["105:15", 0],
        "guider": ["105:16", 0],
        "sampler": ["105:17", 0],
        "sigmas": ["105:9", 0],
        "latent_image": ["105:104", 1],
    }
    assert graph["105:16"]["inputs"] == {
        "model": effective_model,
        "conditioning": ["105:104", 0],
    }
    assert graph["105:104"]["inputs"]["clip"] == ["105:13", 0]
    assert graph["105:104"]["inputs"]["vae"] == ["105:11", 0]
    if name.endswith("_spectrum.json"):
        assert graph["105:120"]["inputs"]["model"] == ["105:6", 0]

    assert graph["105:10"]["inputs"] == {
        "samples": ["105:14", 0],
        "vae": ["105:11", 0],
    }
    assert graph["105:23"]["inputs"] == {
        "samples": ["105:14", 0],
        "vae": ["105:24", 0],
    }
    assert graph["105:91"]["inputs"]["images"] == ["105:10", 0]
    assert graph["105:91"]["inputs"]["audio"] == ["105:23", 0]
    assert graph["105:91"]["inputs"]["bit_depth"] == 8
    assert graph["92"]["inputs"]["video"] == ["105:91", 0]
    assert graph["92"]["inputs"]["codec"] == "auto"


@pytest.mark.parametrize("name", WORKFLOW_NAMES)
def test_h3_duration_drives_preserved_frame_grid_expression(name):
    graph = _load(name)

    assert graph["105:107"]["inputs"] == {
        "expression": FRAME_GRID_EXPRESSION,
        "values.a": ["105:111", 0],
    }
    assert graph["105:104"]["inputs"]["length"] == ["105:107", 1]


def test_obsolete_assets_are_not_shipped():
    assert not (WORKFLOW_DIR / "wan22_t2v.json").exists()
    assert not (WORKFLOW_DIR / "svd_xt_i2v.json").exists()
