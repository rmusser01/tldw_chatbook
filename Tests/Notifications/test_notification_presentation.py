from tldw_chatbook.Notifications.notification_presentation import NotificationPresentationStore


def test_presentation_updates_local_delivery_without_changing_server_owned_state():
    store = NotificationPresentationStore()
    store.upsert_server_state(
        "event-1",
        read_state="unread",
        dismiss_state="active",
        reminder_state="scheduled",
    )

    record = store.mark_delivered("event-1", presented_at="2026-04-28T12:00:00Z")

    assert record.local_delivery_state == "delivered"
    assert record.presented_at == "2026-04-28T12:00:00Z"
    assert record.server_read_state == "unread"
    assert record.server_dismiss_state == "active"
    assert record.server_reminder_state == "scheduled"


def test_presentation_updates_server_owned_state_without_changing_local_delivery():
    store = NotificationPresentationStore()
    store.mark_failed("event-1", delivery_error="toast unavailable")

    record = store.upsert_server_state(
        "event-1",
        read_state="read",
        dismiss_state="dismissed",
        reminder_state="completed",
    )

    assert record.local_delivery_state == "failed"
    assert record.delivery_error == "toast unavailable"
    assert record.server_read_state == "read"
    assert record.server_dismiss_state == "dismissed"
    assert record.server_reminder_state == "completed"


def test_presentation_updates_server_reminder_state_without_changing_delivery_read_or_dismiss():
    store = NotificationPresentationStore()
    store.upsert_server_state("event-1", read_state="unread", dismiss_state="active")
    store.mark_delivered("event-1", presented_at="2026-04-28T12:00:00Z")

    record = store.upsert_server_state("event-1", reminder_state="snoozed")

    assert record.local_delivery_state == "delivered"
    assert record.presented_at == "2026-04-28T12:00:00Z"
    assert record.server_read_state == "unread"
    assert record.server_dismiss_state == "active"
    assert record.server_reminder_state == "snoozed"
