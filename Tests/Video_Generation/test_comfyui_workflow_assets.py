import json
from pathlib import Path


WORKFLOW_DIR = Path(__file__).parents[2] / "tldw_chatbook" / "Video_Generation" / "workflows"
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
    for name in ("minimax_h3_t2v.json", "minimax_h3_t2v_spectrum.json"):
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


def test_obsolete_assets_are_not_shipped():
    assert not (WORKFLOW_DIR / "wan22_t2v.json").exists()
    assert not (WORKFLOW_DIR / "svd_xt_i2v.json").exists()
