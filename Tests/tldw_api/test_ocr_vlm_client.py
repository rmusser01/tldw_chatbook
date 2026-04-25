from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    OCRBackendsResponse,
    OCRPointsPreloadResponse,
    TLDWAPIClient,
    VLMBackendsResponse,
)


@pytest.mark.asyncio
async def test_ocr_and_vlm_backend_discovery_routes_wire(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "points": {
                    "available": True,
                    "mode": "transformers",
                    "sglang_reachable": False,
                },
                "mineru": {
                    "available": False,
                    "pdf_only": True,
                    "document_level": True,
                    "error": "missing dependency",
                },
            },
            {"status": "ok", "mode": "transformers"},
            {
                "qwen2_vl": {
                    "available": True,
                    "mode": "remote",
                }
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    ocr = await client.list_ocr_backends()
    preload = await client.preload_ocr_points_transformers()
    vlm = await client.list_vlm_backends()

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/ocr/backends")
    assert mocked.await_args_list[1].args[:2] == ("POST", "/api/v1/ocr/points/preload")
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/vlm/backends")
    assert isinstance(ocr, OCRBackendsResponse)
    assert ocr.root["points"].mode == "transformers"
    assert ocr.root["mineru"].error == "missing dependency"
    assert isinstance(preload, OCRPointsPreloadResponse)
    assert preload.status == "ok"
    assert isinstance(vlm, VLMBackendsResponse)
    assert vlm.root["qwen2_vl"]["available"] is True
