import pytest

from tldw_chatbook.Web_Scraping_Interop import WebScrapingScopeService


class FakeServerWebScrapingService:
    def __init__(self):
        self.calls = []

    async def get_status(self):
        self.calls.append(("get_status",))
        return {"record_id": "server:web_scraping:status"}

    async def get_job_status(self, job_id):
        self.calls.append(("get_job_status", job_id))
        return {"record_id": f"server:web_scraping_job:{job_id}"}

    async def cancel_job(self, job_id):
        self.calls.append(("cancel_job", job_id))
        return {"record_id": f"server:web_scraping_job:{job_id}"}

    async def get_progress(self, task_id):
        self.calls.append(("get_progress", task_id))
        return {"record_id": f"server:web_scraping_progress:{task_id}"}

    async def get_cookies(self, domain):
        self.calls.append(("get_cookies", domain))
        return {"record_id": f"server:web_scraping_cookies:{domain}"}

    async def set_cookies(self, domain, cookies):
        self.calls.append(("set_cookies", domain, cookies))
        return {"record_id": f"server:web_scraping_cookies:{domain}"}

    async def check_duplicate(self, url):
        self.calls.append(("check_duplicate", url))
        return {"record_id": f"server:web_scraping_duplicate:{url}"}


@pytest.mark.asyncio
async def test_web_scraping_scope_service_routes_server_owned_operations():
    server = FakeServerWebScrapingService()
    scope = WebScrapingScopeService(server_service=server)

    status = await scope.get_status(mode="server")
    job = await scope.get_job_status(mode="server", job_id="job-1")
    cancelled = await scope.cancel_job(mode="server", job_id="job-1")
    progress = await scope.get_progress(mode="server", task_id="task-1")
    cookies = await scope.get_cookies(mode="server", domain="example.com")
    set_cookies = await scope.set_cookies(mode="server", domain="example.com", cookies=[{"name": "session"}])
    duplicate = await scope.check_duplicate(mode="server", url="https://example.com")

    assert status["record_id"] == "server:web_scraping:status"
    assert job["record_id"] == "server:web_scraping_job:job-1"
    assert cancelled["record_id"] == "server:web_scraping_job:job-1"
    assert progress["record_id"] == "server:web_scraping_progress:task-1"
    assert cookies["record_id"] == "server:web_scraping_cookies:example.com"
    assert set_cookies["record_id"] == "server:web_scraping_cookies:example.com"
    assert duplicate["record_id"] == "server:web_scraping_duplicate:https://example.com"
    assert server.calls == [
        ("get_status",),
        ("get_job_status", "job-1"),
        ("cancel_job", "job-1"),
        ("get_progress", "task-1"),
        ("get_cookies", "example.com"),
        ("set_cookies", "example.com", [{"name": "session"}]),
        ("check_duplicate", "https://example.com"),
    ]


@pytest.mark.asyncio
async def test_web_scraping_scope_service_rejects_local_mode_before_dispatch():
    server = FakeServerWebScrapingService()
    scope = WebScrapingScopeService(server_service=server)

    with pytest.raises(ValueError, match="server-only"):
        await scope.get_status(mode="local")

    assert server.calls == []


def test_web_scraping_scope_service_reports_local_unavailability_and_deferred_controls():
    scope = WebScrapingScopeService(server_service=FakeServerWebScrapingService())

    local_report = scope.list_unsupported_capabilities(mode="local")
    server_report = scope.list_unsupported_capabilities(mode="server")

    assert local_report[0]["operation_id"] == "media.web_scraping.remote_only.local"
    assert "media.web_scraping.status.server" in local_report[0]["affected_action_ids"]
    assert server_report == [
        {
            "operation_id": "media.web_scraping.service_controls.server",
            "source": "server",
            "supported": False,
            "reason_code": "deferred_server_process_control",
            "user_message": "Server web-scraping initialize/shutdown controls are intentionally not exposed through Chatbook.",
            "affected_action_ids": [
                "media.web_scraping.service.initialize.server",
                "media.web_scraping.service.shutdown.server",
            ],
        }
    ]
