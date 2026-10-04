from __future__ import annotations


def legacy_host_test_bindings(seams):
    """Illustrate individual fake bindings, not a production receiver adapter.

    Expand these keyword arguments in the updated host fixture. New owner behavior
    requires its own exact callbacks; this does not silently fabricate them.
    No application code may call this illustrative fixture fragment.
    """
    from tldw_chatbook.Chat.console_interrupt_rounds import InterruptRoundHost

    return InterruptRoundHost(
        read_controller__announce_hidden_decision=lambda: getattr(
            seams, "_announce_hidden_decision", None
        ),
        read_controller__approval_view_is_detached=lambda: getattr(
            seams, "_approval_view_is_detached", None
        ),
        read_controller__forget_hidden_decision=lambda: getattr(
            seams, "_forget_hidden_decision", None
        ),
        read_controller__is_session_cancelled=lambda: getattr(
            seams, "_is_session_cancelled", None
        ),
        read_controller__marshal_pending_decision_projection=lambda: getattr(
            seams, "_marshal_pending_decision_projection", None
        ),
        read_controller__notify_run_hook_approval=lambda: getattr(
            seams, "_notify_run_hook_approval", None
        ),
        read_controller__publish_pending_decision=lambda: getattr(
            seams, "_publish_pending_decision", None
        ),
        read_controller__refresh_answerable_decision=lambda: getattr(
            seams, "_refresh_answerable_decision", None
        ),
        read_controller_add_pending_round=lambda: getattr(
            seams, "add_pending_round", None
        ),
        read_controller_announce_hidden_decision=lambda: getattr(
            seams, "announce_hidden_decision", None
        ),
        read_controller_app=lambda: getattr(seams, "app", None),
        read_controller_discard_pending_round=lambda: getattr(
            seams, "discard_pending_round", None
        ),
        read_controller_expire_pending_decisions=lambda: getattr(
            seams, "expire_pending_decisions", None
        ),
        read_controller_on_pending_rounds_changed=lambda: getattr(
            seams, "on_pending_rounds_changed", None
        ),
        read_controller_park_pending_approval=lambda: getattr(
            seams, "park_pending_approval", None
        ),
        read_controller_set_pending_approval=lambda: getattr(
            seams, "set_pending_approval", None
        ),
        read_controller_set_pending_decision=lambda: getattr(
            seams, "set_pending_decision", None
        ),
        read_controller_set_pending_question=lambda: getattr(
            seams, "set_pending_question", None
        ),
        read_controller_set_pending_skill_install=lambda: getattr(
            seams, "set_pending_skill_install", None
        ),
        read_controller_set_pending_skill_script=lambda: getattr(
            seams, "set_pending_skill_script", None
        ),
        read_controller_set_pending_worktree_merge=lambda: getattr(
            seams, "set_pending_worktree_merge", None
        ),
        read_controller_store=lambda: getattr(seams, "store", None),
    )
