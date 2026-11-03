"""Remote round projection and native decision settlement on the existing host."""

from __future__ import annotations

from . import console_interrupt_rounds as host


def remote_rounds(self, session_id: str) -> tuple[dict, ...]:
    """Allowlisted retained decision metadata, independent of mounted cards."""
    import json

    rows = []
    sessions = getattr(self.read_controller_store(), "sessions", lambda: ())()
    session = next((row for row in sessions if row.id == session_id), None)
    incarnation = getattr(session, "incarnation_id", None)
    # Payloads are retained native values; snapshot their ownership under the
    # registry lock, then redact/format outside it so answers cannot block on JSON.
    with self.lock:
        captured = [
            (
                kind,
                round_id,
                dict(state),
                dict(self.payloads[kind].get(round_id, {})),
            )
            for kind, states in self.registries.items()
            for round_id, state in states.items()
            if state.get("session_id") == session_id
            and not state.get("settled")
            and state.get("remote_incarnation_id") == incarnation
            and not state["event"].is_set()
        ]
        rendered_rounds = {
            rendered[3]
            for target, _kinds, rendered in self._decision_views.values()
            if target == session_id and rendered is not None
        }
    for kind, round_id, state, payload in captured:
        supported = False
        review = {}
        if kind == "approval" and payload.get("calls"):
            from .approval_review import approval_options, remote_argument_summary

            try:
                calls = payload["calls"]
                supported = all(
                    not call.get("full_command")
                    and not (
                        call.get("server_key") == "local:__local__"
                        and call.get("tool_name") == "shell_exec"
                    )
                    for call in calls
                )
                if supported:
                    review["calls"] = [
                        {
                            **{
                                key: call.get(key)
                                for key in (
                                    "call_id",
                                    "llm_name",
                                    "tool_name",
                                    "options",
                                    "effects",
                                    "reason",
                                    "path_precheck_failed",
                                )
                            },
                            "options": [
                                value
                                for _label, value in approval_options(
                                    call.get("options")
                                )
                            ],
                            "arguments_summary": remote_argument_summary(
                                call.get("arguments")
                            ),
                        }
                        for call in calls
                    ]
            except Exception:  # noqa: BLE001 - untrusted review failures must hide raw data
                supported, review = False, {}
        elif kind == "question" and payload.get("questions"):
            from tldw_chatbook.Agents.ask_user_questions import validate_questions

            try:
                review["questions"] = validate_questions(payload["questions"])
                supported = True
            except ValueError:
                supported, review = False, {}
        try:
            if len(json.dumps(review).encode()) > 16000:
                supported, review = False, {}
        except (TypeError, ValueError):
            supported, review = False, {}
        answerable = round_id in rendered_rounds
        row = {
            "kind": kind,
            "round_id": round_id,
            "run_id": state.get("run_id") or None,
            "supported": supported,
            "answerable": supported and answerable,
            **review,
        }
        if not supported:
            row.update(
                reason="This review requires the terminal.", action="open_terminal"
            )
        rows.append(row)
    lock = self.read_controller__pending_chat_create_lock()
    if lock is not None:
        with lock:
            rows.extend(
                {
                    "kind": "chat_create",
                    "round_id": round_id,
                    "run_id": state.get("run_id") or None,
                    "supported": False,
                    "answerable": False,
                    "reason": "This review requires the terminal.",
                    "action": "open_terminal",
                }
                for round_id, state in self.read_controller__pending_chat_create_rounds().items()
                if state.get("session_id") == session_id and not state["event"].is_set()
            )
    return tuple(rows)


def set_remote_answerable(
    self, target: host.ConsoleTarget, *, answerable: bool
) -> None:
    """Reuse native rendered-card accounting for one exact live round."""
    owner = ("remote", target)
    if not answerable:
        self.set_decision_view(owner, None)
        return
    session = next(
        (
            row
            for row in self.read_controller_store().sessions()
            if row.id == target.session_id
        ),
        None,
    )
    if session is None or session.incarnation_id != target.incarnation_id:
        raise ValueError("Console target incarnation changed.")
    row = next(
        (
            row
            for row in self.remote_rounds(target.session_id)
            if row["round_id"] == target.round_id and row["run_id"] == target.run_id
        ),
        None,
    )
    if row is None or not row["supported"]:
        raise ValueError("Decision is no longer remotely answerable.")
    self.set_decision_view(
        owner, target.session_id, kinds=(row["kind"],), decision_id=target.round_id
    )


def resolve_remote_round(
    self,
    target: host.ConsoleTarget,
    *,
    kind: str,
    answer: dict,
    _on_outcome: host.Callable[[str], None] | None = None,
) -> dict:
    """Validate exact native scope before calling the original kind resolver."""
    session = next(
        (
            row
            for row in self.read_controller_store().sessions()
            if row.id == target.session_id
        ),
        None,
    )
    if session is None or session.incarnation_id != target.incarnation_id:
        raise ValueError("Console target incarnation changed.")
    if target.round_id is None or target.handle_id is not None:
        raise ValueError("An exact decision round is required.")
    with self.lock:
        key = (
            target.session_id,
            target.incarnation_id,
            target.run_id,
            kind,
            target.round_id,
        )
        outcome = self._remote_outcomes.get(key)
        if outcome is not None:
            if _on_outcome is not None:
                return {
                    "state": "outcome_unknown"
                    if outcome["state"] == "unknown"
                    else "refused",
                    "native_id": target.round_id,
                    "reason": "already_resolved",
                }
            return dict(outcome)
        state = self.registries.get(kind, {}).get(target.round_id)
        if (
            state is None
            or state.get("remote_incarnation_id") != target.incarnation_id
            or state.get("session_id") != target.session_id
            or (state.get("run_id") or None) != target.run_id
        ):
            raise ValueError("Decision target is no longer current.")
        if state.get("settled") or state["event"].is_set():
            return {
                "state": "refused" if _on_outcome is not None else "resolved",
                "native_id": target.round_id,
                "reason": "already_resolved",
            }
    row = next(
        (
            row
            for row in self.remote_rounds(target.session_id)
            if row["kind"] == kind and row["round_id"] == target.round_id
        ),
        None,
    )
    if row is None or not row["supported"]:
        raise ValueError("Open terminal to answer this decision.")
    if type(answer) is not dict:
        raise ValueError("Decision answer must be an object.")
    recorded = False

    def on_recorded(native_state):
        nonlocal recorded
        recorded = True
        if _on_outcome is not None:
            native_state["remote_receipt_outcome"] = _on_outcome

    if kind == "approval":
        if set(answer) != {"decisions"} or type(answer["decisions"]) is not dict:
            raise ValueError("Expected exact approval decisions.")
        allowed = {
            call.get("call_id") or call["llm_name"]: set(call["options"] or [])
            | {"deny"}
            for call in row["calls"]
        }
        if set(answer["decisions"]) != set(allowed) or any(
            type(value) is not str or value not in allowed[key]
            for key, value in answer["decisions"].items()
        ):
            raise ValueError("Approval choices do not match the native review.")
        self.resolve_pending_approval(
            answer["decisions"], round_id=target.round_id, _on_recorded=on_recorded
        )
    elif kind == "question":
        from tldw_chatbook.Agents.ask_user_questions import validate_answers

        if set(answer) != {"answers"}:
            raise ValueError("Expected question answers.")
        answers = validate_answers(answer["answers"])
        if len(answers) != len(row["questions"]) or any(
            response["question"] != question["question"]
            or not set(response["selected"])
            <= {option["label"] for option in question["options"]}
            or (not question.get("multiSelect") and len(response["selected"]) > 1)
            for response, question in zip(answers, row["questions"], strict=True)
        ):
            raise ValueError("Answers do not match the native question round.")
        self.resolve_pending_question(
            answers, request_id=target.round_id, _on_recorded=on_recorded
        )
    else:
        raise ValueError("Open terminal to answer this decision.")
    self.set_remote_answerable(target, answerable=False)
    if _on_outcome is not None:
        return {
            "state": "accepted" if recorded else "refused",
            "native_id": target.round_id,
            "reason": "answer_recorded" if recorded else "already_resolved",
        }
    return {
        "state": "resolved",
        "native_id": target.round_id,
        "reason": state.get("terminal_reason") or "answered",
    }


def resolve_pending_approval(
    self,
    decisions: dict[str, str],
    *,
    round_id: str | None,
    _on_recorded: host.Callable[[dict], None] | None = None,
) -> None:
    """UI THREAD: apply the user's batch decision, releasing the waiting worker thread.

    Called by ``ChatScreen``'s ``ChatApprovalCard.ApprovalDecided``
    handler, which forwards ``event.round_id`` -- the SAME id
    ``request_mcp_approvals`` stamped into the payload the card was
    built from (``ChatApprovalCard.set_batch`` stashes it;
    ``_submit_batch_decisions`` echoes it back on submit, mirroring
    ``resolve_pending_skill_script``'s identical ``request_id``
    round-trip).

    Fix round 1 (review CRITICAL finding): resolves ONLY the round
    whose id matches ``round_id`` -- never "whichever round belongs to
    the currently active session". ``ApprovalDecided`` travels as an
    async Textual message: a ``switch_session`` landing in the gap
    between the user's click and this handler running would otherwise
    let session A's decision resolve session B's completely different,
    unreviewed batch (or, for the same session, let a STALE decision
    from an already-ended round 1 resolve a newer round 2 that
    happened to arm before the stale message was delivered). A
    mismatched or stale ``round_id`` -- including one belonging to a
    round that already resolved and was popped -- is a safe no-op: the
    real round (if any) stays pending and its card re-derives
    unchanged on the next visit; nothing is ever auto-approved or
    denied-by-accident here.

    TASK-913 (AC#2): ``round_id=None`` no longer falls back to
    "whichever round belongs to the currently active session" -- it
    fails closed immediately, mirroring
    ``resolve_pending_skill_script``'s/``resolve_pending_skill_install``'s
    identical ``if request_id is None: return`` contract. Production
    (``ChatApprovalCard``/``ChatScreen``) has only ever had a single
    emitter (``ChatApprovalCard._submit_batch_decisions``) and it
    always threads the real ``round_id`` through; the active-session
    fallback existed only for legacy direct-call tests, which have
    been migrated to pass the real round id captured from the
    mounted/parked payload instead.

    A no-op both when ``round_id`` is ``None`` and when it doesn't
    match any currently-armed round (e.g. a stale message arriving
    after a timeout/cancellation already resolved and cleared it) --
    the real round (if any) stays pending and undecided; nothing is
    ever auto-approved or denied-by-accident here.

    NOTE: Snapshots the round's ``decisions``/``event`` into locals to
    avoid TOCTOU race: the worker thread's ``finally`` block pops the
    round entry out of ``_pending_approval_rounds`` concurrently. Guard
    and act only on the snapshots.

    Args:
        decisions: The user's per-``llm_name`` decision strings
            (``approve_once``/``approve_session``/``always_allow``/
            ``deny``) to merge into the round's shared decisions dict.
        round_id: The specific round to resolve (the id stamped onto
            the card the user actually decided). ``None`` (the
            default) never matches an armed round, so an un-migrated
            or malformed caller fails closed by omission.
    """
    # TASK-913 (AC#2): fail closed on a missing round_id rather than
    # scanning `_pending_approval_rounds.values()` for "whichever round
    # belongs to the active session" -- that active-session fallback
    # was production-unreachable (see docstring) and is now removed
    # entirely, taking its AC#1 lock-guarded-snapshot protection with
    # it (moot once the scan itself is gone). The remaining branch's
    # `.get()` read stays guarded: the worker thread's own registration
    # (`request_mcp_approvals`) and teardown (its `finally`) can mutate
    # this dict concurrently.
    if round_id is None:
        return
    with self.lock:
        round_state = self.registries["approval"].get(round_id)
        if (
            round_state is None
            or round_state.get("settled")
            or round_state["event"].is_set()
        ):
            return
        round_state["settled"] = True
        round_state["terminal_reason"] = "user"
        decisions_dict = round_state["decisions"]
        decisions_dict.update(decisions or {})
        if isinstance(decisions_dict, self.read_global_ApprovalDecisions()):
            answers = self.read_global_ApprovalDecisions()(
                decisions or {},
                denial_reasons=getattr(decisions, "denial_reasons", {}),
            )
            decisions_dict.denial_reasons = {
                key: reason
                for key, reason in answers.denial_reasons.items()
                if key in round_state["names"]
            }
        if _on_recorded is not None:
            _on_recorded(round_state)
        approval_event = round_state["event"]
    approval_event.set()


def resolve_pending_question(
    self,
    answers: list[dict[str, host.Any]],
    request_id: str | None,
    *,
    _on_recorded: host.Callable[[dict], None] | None = None,
) -> None:
    """UI THREAD: hand the card's answers to the waiting worker thread.

    Strict ``request_id`` match, exactly like
    ``resolve_pending_skill_script``: a resolve with no id, or an id
    from any round but the armed one, is silently dropped. The answers
    are validated (``ask_user_questions.validate_answers``) before the
    worker sees them; a malformed list is dropped the same way.

    Args:
        answers: One PRD A6 answer dict per question, in order.
        request_id: The armed round's id as echoed back by the card.
    """
    from tldw_chatbook.Agents.ask_user_questions import (
        AskUserValidationError,
        validate_answers,
    )

    if request_id is None:
        return
    try:
        clean_answers = validate_answers(answers)
    except AskUserValidationError:
        return  # fail closed: a malformed resolve is dropped, never partially applied
    with self.lock:
        round_state = self.registries["question"].get(request_id)
        if (
            round_state is None
            or round_state.get("settled")
            or round_state["event"].is_set()
        ):
            return
        round_state["settled"] = True
        round_state["terminal_reason"] = "user"
        round_state["decision"]["answers"] = clean_answers
        if _on_recorded is not None:
            _on_recorded(round_state)
        event = round_state["event"]
    event.set()


def resolve_pending_skill_install(self, allow: bool, *, request_id: str | None) -> None:
    """UI THREAD: apply the user's Allow/Deny, releasing the worker thread.

    TASK-910: strict match against ``request_id``, mirroring
    ``resolve_pending_skill_script``'s identical contract -- a resolve
    carrying no id, or an id belonging to any round other than the one
    it names, is silently dropped rather than resolved. This closes the
    same stale-late-click hazard ``resolve_pending_skill_script``'s own
    docstring documents: once two sessions can each have their own
    concurrent install-confirm round (TASK-910 parking), "whichever
    round happens to be active" is no longer a safe fallback the way it
    was pre-TASK-910 (a single global slot could only ever have one
    candidate).

    Args:
        allow: True to allow the pending install, False to deny it.
        request_id: The armed round's id, as echoed back by the UI
            (``SkillInstallConfirmCard.InstallDecided.request_id``).
            ``None`` (the default) never matches an armed round, so an
            un-migrated or malformed caller fails closed by omission.
    """
    if request_id is None:
        return
    with self.lock:
        round_state = self.registries["skill_install"].get(request_id)
        if round_state is None or round_state.get("settled"):
            return
        round_state["settled"] = True
        round_state["terminal_reason"] = "user"
        round_state["decision"]["allow"] = bool(allow)
        event = round_state["event"]
    event.set()
