from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import TLDWAPIClient


@pytest.mark.asyncio
async def test_web_scraping_client_routes_management_subset(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {"initialized": True, "queue": {"active": 1}},
            {"job_id": "job-1", "status": "running"},
            {"status": "cancelled", "job_id": "job-1"},
            {"task_id": "task-1", "progress": {"pages_scraped": 2}, "status": "in_progress"},
            {"domain": "example.com", "cookies": [{"name": "session"}], "cookie_count": 1},
            {"status": "success", "domain": "example.com"},
            {"url": "https://example.com", "is_duplicate": False},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    status = await client.get_web_scraping_status()
    job = await client.get_web_scraping_job_status("job-1")
    cancelled = await client.cancel_web_scraping_job("job-1")
    progress = await client.get_web_scraping_progress("task-1")
    cookies = await client.get_web_scraping_cookies("example.com")
    set_cookies = await client.set_web_scraping_cookies("example.com", [{"name": "session", "value": "abc"}])
    duplicate = await client.check_web_scraping_duplicate("https://example.com")

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/web-scraping/status")
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/web-scraping/job/job-1")
    assert mocked.await_args_list[2].args[:2] == ("DELETE", "/api/v1/web-scraping/job/job-1")
    assert mocked.await_args_list[3].args[:2] == ("GET", "/api/v1/web-scraping/progress/task-1")
    assert mocked.await_args_list[4].args[:2] == ("GET", "/api/v1/web-scraping/cookies/example.com")
    assert mocked.await_args_list[5].args[:2] == ("POST", "/api/v1/web-scraping/cookies/example.com")
    assert mocked.await_args_list[5].kwargs["json_data"] == [{"name": "session", "value": "abc"}]
    assert mocked.await_args_list[6].args[:2] == ("GET", "/api/v1/web-scraping/duplicates/check")
    assert mocked.await_args_list[6].kwargs["params"] == {"url": "https://example.com"}

    assert status["queue"]["active"] == 1
    assert job["status"] == "running"
    assert cancelled["status"] == "cancelled"
    assert progress["progress"]["pages_scraped"] == 2
    assert cookies["cookie_count"] == 1
    assert set_cookies["status"] == "success"
    assert duplicate["is_duplicate"] is False
