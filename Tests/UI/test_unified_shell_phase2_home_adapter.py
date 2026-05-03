from pathlib import Path


PHASE_2_ROOT = Path("Docs/superpowers/qa/unified-shell/phase-2")
EVIDENCE = PHASE_2_ROOT / "2026-05-03-home-active-work-adapter-contract.md"
DETAIL_CONSOLE_EVIDENCE = PHASE_2_ROOT / "2026-05-03-home-detail-console-adapter-actions.md"
ITEM_CONTEXT_EVIDENCE = PHASE_2_ROOT / "2026-05-03-home-active-work-item-context.md"
LOCAL_NOTIFICATION_EVIDENCE = PHASE_2_ROOT / "2026-05-03-home-local-notification-snapshot.md"
NOTIFICATION_REVIEW_EVIDENCE = PHASE_2_ROOT / "2026-05-03-home-notification-review-routing.md"
README = PHASE_2_ROOT / "README.md"
ROADMAP = Path("Docs/superpowers/trackers/unified-shell-maturity-roadmap.md")
TASK_4_1 = Path("backlog/tasks/task-4.1 - Phase-2.1-Add-Home-active-work-adapter-contract.md")
TASK_4_2 = Path(
    "backlog/tasks/task-4.2 - Phase-2.2-Route-Home-detail-and-Console-actions-through-active-work-adapter.md"
)
TASK_4_3 = Path("backlog/tasks/task-4.3 - Phase-2.3-Bind-Home-controls-to-active-work-item-context.md")
TASK_4_4 = Path(
    "backlog/tasks/task-4.4 - Phase-2.4-Wire-Home-to-local-notification-snapshot-adapter.md"
)
TASK_4_5 = Path(
    "backlog/tasks/task-4.5 - Phase-2.5-Route-Home-notification-review-to-the-notifications-inbox.md"
)


def test_phase_two_home_adapter_evidence_exists_and_records_verification():
    assert EVIDENCE.exists()
    text = EVIDENCE.read_text(encoding="utf-8")

    assert "TASK-4.1" in text
    assert "Home active-work adapter contract" in text
    assert "Tests/Home/test_active_work_adapter.py" in text
    assert "Tests/UI/test_home_screen.py" in text
    assert "passed" in text
    assert "UnavailableHomeActiveWorkAdapter" in text


def test_phase_two_home_adapter_is_linked_from_index_roadmap_and_task():
    evidence_name = EVIDENCE.name

    assert evidence_name in README.read_text(encoding="utf-8")

    roadmap_text = ROADMAP.read_text(encoding="utf-8")
    assert "TASK-4.1" in roadmap_text
    assert evidence_name in roadmap_text
    assert "| Phase 2 | `Docs/superpowers/qa/unified-shell/phase-2/` | in-progress |" in roadmap_text

    task_text = TASK_4_1.read_text(encoding="utf-8")
    assert "status: Done" in task_text
    assert "- [x] #1" in task_text
    assert "- [x] #4" in task_text


def test_phase_two_detail_console_evidence_exists_and_records_verification():
    assert DETAIL_CONSOLE_EVIDENCE.exists()
    text = DETAIL_CONSOLE_EVIDENCE.read_text(encoding="utf-8")

    assert "TASK-4.2" in text
    assert "Open details" in text
    assert "Open in Console" in text
    assert "Tests/Home/test_active_work_adapter.py" in text
    assert "Tests/UI/test_home_screen.py" in text
    assert "18 passed" in text
    assert "HomeConsoleLaunch" in text


def test_phase_two_detail_console_evidence_is_linked_from_index_roadmap_and_task():
    evidence_name = DETAIL_CONSOLE_EVIDENCE.name

    assert evidence_name in README.read_text(encoding="utf-8")

    roadmap_text = ROADMAP.read_text(encoding="utf-8")
    assert "TASK-4.2" in roadmap_text
    assert evidence_name in roadmap_text

    task_text = TASK_4_2.read_text(encoding="utf-8")
    assert "status: Done" in task_text
    assert "- [x] #1" in task_text
    assert "- [x] #4" in task_text


def test_phase_two_item_context_evidence_exists_and_records_verification():
    assert ITEM_CONTEXT_EVIDENCE.exists()
    text = ITEM_CONTEXT_EVIDENCE.read_text(encoding="utf-8")

    assert "TASK-4.3" in text
    assert "HomeActiveWorkItem" in text
    assert "target_id" in text
    assert "Tests/Home/test_dashboard_state.py" in text
    assert "Tests/UI/test_home_screen.py" in text
    assert "31 passed" in text


def test_phase_two_item_context_evidence_is_linked_from_index_roadmap_and_task():
    evidence_name = ITEM_CONTEXT_EVIDENCE.name

    assert evidence_name in README.read_text(encoding="utf-8")

    roadmap_text = ROADMAP.read_text(encoding="utf-8")
    assert "TASK-4.3" in roadmap_text
    assert evidence_name in roadmap_text

    task_text = TASK_4_3.read_text(encoding="utf-8")
    assert "status: Done" in task_text
    assert "- [x] #1" in task_text
    assert "- [x] #5" in task_text


def test_phase_two_local_notification_evidence_exists_and_records_verification():
    assert LOCAL_NOTIFICATION_EVIDENCE.exists()
    text = LOCAL_NOTIFICATION_EVIDENCE.read_text(encoding="utf-8")

    assert "TASK-4.4" in text
    assert "LocalNotificationHomeActiveWorkAdapter" in text
    assert "notification_count" in text
    assert "ClientNotificationsService.list_queue" in text
    assert "Tests/UI/test_screen_navigation.py" in text
    assert "58 passed" in text
    assert "66 passed" in text


def test_phase_two_local_notification_evidence_is_linked_from_index_roadmap_and_task():
    evidence_name = LOCAL_NOTIFICATION_EVIDENCE.name

    assert evidence_name in README.read_text(encoding="utf-8")

    roadmap_text = ROADMAP.read_text(encoding="utf-8")
    assert "TASK-4.4" in roadmap_text
    assert evidence_name in roadmap_text

    task_text = TASK_4_4.read_text(encoding="utf-8")
    assert "status: Done" in task_text
    assert "- [x] #1" in task_text
    assert "- [x] #5" in task_text


def test_phase_two_notification_review_routing_evidence_exists_and_records_verification():
    assert NOTIFICATION_REVIEW_EVIDENCE.exists()
    text = NOTIFICATION_REVIEW_EVIDENCE.read_text(encoding="utf-8")

    assert "TASK-4.5" in text
    assert "review_notifications" in text
    assert "pending_subscription_initial_tab" in text
    assert "SubscriptionWindow.initial_tab" in text
    assert "Tests/UI/test_home_screen.py" in text
    assert "3 passed" in text
    assert "91 passed" in text


def test_phase_two_notification_review_routing_evidence_is_linked_from_index_roadmap_and_task():
    evidence_name = NOTIFICATION_REVIEW_EVIDENCE.name

    assert evidence_name in README.read_text(encoding="utf-8")

    roadmap_text = ROADMAP.read_text(encoding="utf-8")
    assert "TASK-4.5" in roadmap_text
    assert evidence_name in roadmap_text

    task_text = TASK_4_5.read_text(encoding="utf-8")
    assert "status: Done" in task_text
    assert "- [x] #1" in task_text
    assert "- [x] #5" in task_text
