"""task-31384: the controller's legacy round attributes are aliases of the host.

Eleven test files and ``ChatScreen._current_park_round_ids`` read the
historical per-kind names; the host must be the single owner behind them.
"""

from __future__ import annotations

from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_interrupt_rounds import KIND_SETTER_ATTRS


def test_every_kind_has_a_setter_attribute_on_the_controller():
    controller = ConsoleChatController(store=ConsoleChatStore(), provider_gateway=None)
    for attr in KIND_SETTER_ATTRS.values():
        assert hasattr(controller, attr), attr


def test_legacy_registry_payload_and_lock_names_alias_the_host():
    controller = ConsoleChatController(store=ConsoleChatStore(), provider_gateway=None)
    host = controller._interrupt_host
    for kind, registry, payloads in [
        ("approval", "_pending_approval_rounds", "_parked_approval_payloads"),
        ("skill_install", "_pending_skill_install_rounds", "_parked_skill_install_payloads"),
        ("skill_script", "_pending_skill_script_rounds", "_parked_skill_script_payloads"),
        ("worktree_merge", "_pending_worktree_merge_rounds", "_parked_worktree_merge_payloads"),
        ("question", "_pending_question_rounds", "_parked_question_payloads"),
    ]:
        assert getattr(controller, registry) is host.registries[kind], registry
        assert getattr(controller, payloads) is host.payloads[kind], payloads
    for lock in (
        "_approval_state_lock",
        "_pending_skill_install_lock",
        "_pending_skill_script_lock",
        "_pending_worktree_merge_lock",
        "_pending_question_lock",
    ):
        assert getattr(controller, lock) is host.lock, lock


def test_store_based_helpers_work_over_any_dict_under_the_one_lock():
    controller = ConsoleChatController(store=ConsoleChatStore(), provider_gateway=None)
    payload = {"round_id": "r1", "session_id": "s1"}
    assert controller._park_round_payload(controller._parked_question_payloads, "r1", payload) is True
    assert controller._head_round_payload(controller._parked_question_payloads, "s1") is payload
    assert controller._head_round_payload(controller._parked_approval_payloads, "s1") is None
    controller._unpark_round_payload(controller._parked_question_payloads, "r1")
    assert controller._head_round_payload(controller._parked_question_payloads, "s1") is None
    # A dict the host does not own (test doubles and the skill bridges'
    # own remount calls pass one) is handled the same way.
    foreign: dict = {}
    assert controller._park_round_payload(foreign, "x", {"round_id": "x", "session_id": "s9"}) is True
    assert controller._session_round_payloads(foreign, "s9")[0]["round_id"] == "x"


def test_approvals_register_the_permission_summary_as_the_after_remount_hook():
    controller = ConsoleChatController(store=ConsoleChatStore(), provider_gateway=None)
    hook = controller._interrupt_host.after_remount["approval"]
    assert hook.__func__ is ConsoleChatController._maybe_fire_permission_summary
    assert set(controller._interrupt_host.after_remount) == {"approval"}
