from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.CCP_Modules.ccp_prompt_handler import CCPPromptHandler


class FakePromptScopeService:
    def __init__(self):
        self.calls = []

    async def list_prompts(self, **kwargs):
        self.calls.append(("list_prompts", kwargs))
        return {
            "items": [
                {
                    "id": "server:prompt:alpha",
                    "backend": "server",
                    "source_id": "alpha",
                    "name": "Alpha Prompt",
                    "author": "Server Writer",
                    "details": "Remote prompt details",
                    "system_prompt": "Remote system",
                    "user_prompt": "Remote user",
                    "keywords": ["remote", "alpha"],
                },
                {
                    "id": "server:prompt:beta",
                    "backend": "server",
                    "source_id": "beta",
                    "name": "Beta Prompt",
                    "author": "Server Writer",
                    "details": "Other details",
                    "system_prompt": "Other system",
                    "user_prompt": "Other user",
                    "keywords": ["remote"],
                },
            ],
            "total_pages": 1,
            "current_page": 1,
            "total_items": 2,
        }

    async def record_prompt_usage(self, **kwargs):
        self.calls.append(("record_prompt_usage", kwargs))
        return {
            "id": "server:prompt:alpha",
            "backend": "server",
            "source_id": "alpha",
            "name": "Alpha Prompt",
            "usage_count": 4,
        }

    async def list_prompt_versions(self, **kwargs):
        self.calls.append(("list_prompt_versions", kwargs))
        if kwargs.get("mode") == "local":
            raise ValueError("Local prompt version history is unavailable.")
        return [
            {"id": "server:prompt:alpha:v2", "backend": "server", "version": 2, "name": "Alpha Prompt"},
            {"id": "server:prompt:alpha:v1", "backend": "server", "version": 1, "name": "Alpha Prompt"},
        ]

    async def restore_prompt_version(self, **kwargs):
        self.calls.append(("restore_prompt_version", kwargs))
        if kwargs.get("mode") == "local":
            raise ValueError("Local prompt version restore is unavailable.")
        return {
            "id": "server:prompt:alpha",
            "backend": "server",
            "source_id": "alpha",
            "name": "Alpha Prompt Restored",
            "usage_count": 4,
            "version": kwargs["version"],
        }


class FakeWindow:
    def __init__(self, *, backend="server"):
        self.app_instance = SimpleNamespace(
            current_runtime_backend=backend,
            prompt_scope_service=FakePromptScopeService(),
        )
        self.messages = []
        self.notifications = []

    def post_message(self, message):
        self.messages.append(message)

    def notify(self, message, **kwargs):
        self.notifications.append((message, kwargs))


class FakeStatic:
    def __init__(self):
        self.value = ""

    def update(self, value):
        self.value = value


class FakeInput:
    def __init__(self, value=""):
        self.value = value


@pytest.mark.asyncio
async def test_prompt_search_uses_active_scope_service_backend(monkeypatch):
    window = FakeWindow(backend="server")
    handler = CCPPromptHandler(window)

    async def noop_update():
        return None

    monkeypatch.setattr(handler, "_update_search_results_ui", noop_update)

    await handler.handle_search("alpha")

    assert window.app_instance.prompt_scope_service.calls == [
        (
            "list_prompts",
            {
                "mode": "server",
                "page": 1,
                "per_page": 100,
                "include_deleted": False,
            },
        )
    ]
    assert [prompt["id"] for prompt in handler.search_results] == ["server:prompt:alpha"]


@pytest.mark.asyncio
async def test_prompt_load_selected_uses_stable_prompt_result_mapping(monkeypatch):
    window = FakeWindow(backend="server")
    handler = CCPPromptHandler(window)
    handler._prompt_result_ids = {"prompt-result-0": "server:prompt:alpha"}

    class FakePromptList:
        highlighted_child = SimpleNamespace(id="prompt-result-0")

    def query_one(selector, widget_type):
        assert selector == "#ccp-prompts-listview"
        return FakePromptList()

    window.query_one = query_one
    loaded = []

    async def capture_load(prompt_identifier):
        loaded.append(prompt_identifier)

    monkeypatch.setattr(handler, "load_prompt", capture_load)

    await handler.handle_load_selected()

    assert loaded == ["server:prompt:alpha"]


@pytest.mark.asyncio
async def test_prompt_usage_control_routes_through_active_scope_service():
    window = FakeWindow(backend="server")
    usage_display = FakeStatic()

    def query_one(selector, widget_type=None):
        assert selector == "#ccp-editor-prompt-usage-display"
        return usage_display

    window.query_one = query_one
    handler = CCPPromptHandler(window)
    handler.current_prompt_id = "server:prompt:alpha"
    handler.current_prompt_data = {"id": "server:prompt:alpha", "usage_count": 3}

    await handler.handle_record_prompt_usage()

    assert window.app_instance.prompt_scope_service.calls == [
        (
            "record_prompt_usage",
            {
                "mode": "server",
                "prompt_identifier": "alpha",
            },
        )
    ]
    assert handler.current_prompt_data["usage_count"] == 4
    assert usage_display.value == "Usage: 4"


@pytest.mark.asyncio
async def test_prompt_versions_and_restore_use_active_scope_service():
    window = FakeWindow(backend="server")
    version_status = FakeStatic()
    version_input = FakeInput("2")

    def query_one(selector, widget_type=None):
        if selector == "#ccp-editor-prompt-version-status":
            return version_status
        if selector == "#ccp-editor-prompt-version-input":
            return version_input
        if selector == "#ccp-editor-prompt-usage-display":
            return FakeStatic()
        raise AssertionError(f"Unexpected selector: {selector}")

    window.query_one = query_one
    handler = CCPPromptHandler(window)
    handler.current_prompt_id = "server:prompt:alpha"
    handler.current_prompt_data = {"id": "server:prompt:alpha", "name": "Alpha Prompt"}

    await handler.handle_list_prompt_versions()
    await handler.handle_restore_prompt_version()

    assert window.app_instance.prompt_scope_service.calls == [
        (
            "list_prompt_versions",
            {
                "mode": "server",
                "prompt_identifier": "alpha",
            },
        ),
        (
            "restore_prompt_version",
            {
                "mode": "server",
                "prompt_identifier": "alpha",
                "version": 2,
            },
        ),
    ]
    assert "v2" in version_status.value
    assert "restored v2" in version_status.value
    assert handler.current_prompt_data["name"] == "Alpha Prompt Restored"


@pytest.mark.asyncio
async def test_prompt_versions_surface_local_unavailable_status():
    window = FakeWindow(backend="local")
    version_status = FakeStatic()

    def query_one(selector, widget_type=None):
        assert selector == "#ccp-editor-prompt-version-status"
        return version_status

    window.query_one = query_one
    handler = CCPPromptHandler(window)
    handler.current_prompt_id = "local:prompt:1"

    await handler.handle_list_prompt_versions()

    assert window.app_instance.prompt_scope_service.calls == [
        (
            "list_prompt_versions",
            {
                "mode": "local",
                "prompt_identifier": "1",
            },
        )
    ]
    assert "Local prompt version history is unavailable" in version_status.value
