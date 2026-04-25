import pytest

from tldw_chatbook.Media import ServerMediaReadingService


class FakeDocumentIntelligenceClient:
    def __init__(self):
        self.calls = []

    async def get_document_outline(self, media_id):
        self.calls.append(("get_document_outline", media_id))
        return {"media_id": media_id, "has_outline": True, "entries": [], "total_pages": 1}

    async def get_document_figures(self, media_id, **params):
        self.calls.append(("get_document_figures", media_id, params))
        return {"media_id": media_id, "has_figures": False, "figures": [], "total_count": 0}

    async def get_document_references(self, media_id, **params):
        self.calls.append(("get_document_references", media_id, params))
        return {"media_id": media_id, "has_references": False, "references": []}

    async def generate_document_insights(self, media_id, request_data):
        self.calls.append(("generate_document_insights", media_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {"media_id": media_id, "insights": [], "model_used": "test-model", "cached": False}


@pytest.mark.asyncio
async def test_server_media_service_routes_document_intelligence_operations():
    client = FakeDocumentIntelligenceClient()
    service = ServerMediaReadingService(client=client)

    outline = await service.get_document_outline(7)
    figures = await service.get_document_figures(7, min_size=80)
    references = await service.get_document_references(7, enrich=True, limit=10)
    insights = await service.generate_document_insights(7, categories=["summary"], model="test-model", force=True)

    assert outline["has_outline"] is True
    assert figures["has_figures"] is False
    assert references["has_references"] is False
    assert insights["model_used"] == "test-model"
    assert client.calls == [
        ("get_document_outline", 7),
        ("get_document_figures", 7, {"min_size": 80}),
        (
            "get_document_references",
            7,
            {
                "enrich": True,
                "reference_index": None,
                "offset": 0,
                "limit": 10,
                "parse_cap": None,
                "search": None,
            },
        ),
        (
            "generate_document_insights",
            7,
            {
                "categories": ["summary"],
                "model": "test-model",
                "max_content_length": 5000,
                "force": True,
            },
        ),
    ]
