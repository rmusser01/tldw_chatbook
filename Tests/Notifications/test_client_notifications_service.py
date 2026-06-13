from tldw_chatbook.Notifications import ClientNotificationsDB, ClientNotificationsService


class FakePolicyEnforcer:
    def __init__(self):
        self.calls = []

    def require_allowed(self, *, action_id):
        self.calls.append(action_id)


def test_client_notifications_service_routes_queue_and_settings_with_policy(tmp_path):
    store = ClientNotificationsDB(tmp_path / "notifications.db")
    first = store.insert_notification(category="watchlists", title="A", message="A")
    second = store.insert_notification(category="research", title="B", message="B")
    policy = FakePolicyEnforcer()
    service = ClientNotificationsService(store=store, policy_enforcer=policy)

    queue = service.list_queue(limit=10)
    observed = service.observe_queue(after_id=first["id"], limit=10)
    updated = service.update_notification(first["id"], is_read=True, is_dismissed=True)
    settings = service.update_settings(enabled=False, toast_enabled=False)
    fetched_settings = service.get_settings()

    assert [row["id"] for row in queue] == [second["id"], first["id"]]
    assert [row["id"] for row in observed] == [second["id"]]
    assert updated["is_read"] is True
    assert updated["is_dismissed"] is True
    assert settings["enabled"] is False
    assert settings["toast_enabled"] is False
    assert fetched_settings["enabled"] is False
    assert policy.calls == [
        "notifications.queue.list.local",
        "notifications.queue.observe.local",
        "notifications.queue.update.local",
        "notifications.settings.update.local",
        "notifications.settings.list.local",
    ]
