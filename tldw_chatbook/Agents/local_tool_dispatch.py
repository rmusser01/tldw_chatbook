"""Native execution methods behind the existing service; ownership stays on the caller."""

from __future__ import annotations

from . import local_tool_provider as native_host


def _invoke_detailed(
    self, tool_id: str, args: dict
) -> native_host.LocalToolInvocationResult:
    """Execute one tool call. Never raises across the boundary.

    Fail-closed: only an explicit "allow" verdict executes; "deny" and
    any unrecognized verdict refuse with LOCAL_DENY_REFUSAL when the
    resolver genuinely said Off (``PERMISSION_OFF``) and with
    LOCAL_USER_DENY_REFUSAL otherwise -- the user's own card Deny, which
    used to share the Off string and left the transcript unable to name
    either authority (Qodo #7). Both mirror
    MCPToolProvider._apply_verdict's fallthrough, "gate_error" (Fix
    Round H, Item 1: the permission resolver raised rather than
    genuinely resolving) with LOCAL_GATE_ERROR_REFUSAL -- a DIFFERENT
    string from LOCAL_DENY_REFUSAL, since the tool's actual configured
    state was never determined and LOCAL_DENY_REFUSAL's "set to Off"
    claim would be false -- "timeout" with LOCAL_TIMEOUT_REFUSAL, and
    "no_callback" with the constructor's ``no_callback_refusal``
    override when set (LOCAL_TIMEOUT_REFUSAL otherwise).

    Audit (MCP parity): refusals are recorded via the optional
    ``record_decision`` seam, and task-32280's fix round gave each
    REFUSER its own token rather than one flat "denied": ``"denied"``
    only for a person's card Deny, ``"denied-policy"`` for a configured
    Off, ``"denied-killswitch"`` for the kill switch,
    ``"denied-unresolved"`` for a gate that raised instead of
    resolving, and ``"denied-timeout"`` for timeout/no_callback
    (matching the refusal copy the model actually saw). Successful
    executions record nothing: MCPToolProvider records those
    service-side via execute_hub_tool, which has no local analogue.
    """
    from .automatic_work_runtime import current_automatic_work
    from .automatic_work_budget import AutomaticWorkRefused

    name = tool_id.split(":", 1)[1] if ":" in tool_id else tool_id
    spec = self._specs.get(name)
    if spec is None:
        return native_host.LocalToolInvocationResult(
            result=native_host.ToolResult(
                ok=False, error=f"Unknown local tool: {name}"
            ),
            final_gate="not_checked",
            approval_consumed=False,
            reason_code=native_host.LocalToolInvocationReason.UNKNOWN_TOOL,
            dispatch_started=False,
            provider_terminal=native_host.LocalProviderTerminal.NOT_STARTED,
        )
    automatic_work = current_automatic_work()
    automatic_refusal = None
    from .workflow_task_scope import check_task_provider

    try:
        check_task_provider(self, tool_id, args)
    except Exception:  # noqa: BLE001 - authority failures must refuse
        automatic_refusal = "workflow task scope refused"
    if automatic_work is not None:
        try:
            automatic_work.check()
        except AutomaticWorkRefused as exc:
            automatic_refusal = f"automatic tool call refused: {exc}"
        except Exception:
            automatic_refusal = "automatic tool call refused: chain unavailable"
        if name == "web_deep_search":
            automatic_refusal = "web_deep_search is unavailable during automatic follow-up; request it in a manual message"
    if automatic_refusal is not None:
        return native_host.LocalToolInvocationResult(
            result=native_host.ToolResult.blocked(automatic_refusal),
            final_gate="not_checked",
            approval_consumed=False,
            reason_code=native_host.LocalToolInvocationReason.AUTHORITY_UNAVAILABLE,
            dispatch_started=False,
            provider_terminal=native_host.LocalProviderTerminal.NOT_STARTED,
        )
    try:
        authority, clean_args = self._select_admitted_root(name, args)
    except ValueError as exc:
        return native_host.LocalToolInvocationResult(
            result=native_host.ToolResult(ok=False, error=str(exc)),
            final_gate="not_checked",
            approval_consumed=False,
            reason_code=native_host.LocalToolInvocationReason.INVALID_ARGUMENTS,
            dispatch_started=False,
            provider_terminal=native_host.LocalProviderTerminal.NOT_STARTED,
        )
    write = self._is_mutating_path_tool(spec)
    if not self._authority_is_valid(authority, write=write):
        return native_host.LocalToolInvocationResult(
            result=native_host.ToolResult.blocked(
                native_host.LOCAL_ROOT_CHANGED_REFUSAL
            ),
            final_gate="not_checked",
            approval_consumed=False,
            reason_code=native_host.LocalToolInvocationReason.ROOT_CHANGED,
            dispatch_started=False,
            provider_terminal=native_host.LocalProviderTerminal.NOT_STARTED,
        )
    if self._kill_switch_engaged():
        # task-32280 fix round: the switch refused, not the user.
        self._record_decision_safe(
            self.hub_tool_for(name), native_host.KILL_SWITCH_DENIED_DECISION
        )
        return native_host.LocalToolInvocationResult(
            result=native_host.ToolResult.blocked(
                native_host.LOCAL_KILL_SWITCH_REFUSAL
            ),
            final_gate="kill_switch",
            approval_consumed=False,
            reason_code=native_host.LocalToolInvocationReason.PERMISSION_OFF,
            dispatch_started=False,
            provider_terminal=native_host.LocalProviderTerminal.NOT_STARTED,
        )
    if authority is not None and name in native_host._PATH_AUTHORITY_LOCAL_NAMES:
        # TASK-28238 P2 T3 review fix: this cache was construction-time
        # -immutable before agent admit/retire made it mutable
        # mid-flight, so a retire racing this lookup (the run's mapping
        # in `_agent_roots` still resolved a moment ago, but its alias
        # entry here is now gone) is now reachable. Falling back to the
        # generic `spec` would be dishonest -- it closes over the
        # PROVIDER's own base root, not `authority.root`, so it would
        # dispatch a path tool against the wrong root instead of
        # refusing. An explicit blocked result is the honest one here.
        # TASK-34351: only a retired agent worktree reaches this branch
        # (static Console roots are never retired), so the refusal names
        # the released worktree, not private scratch.
        authority_specs = self._path_specs_by_alias.get(authority.alias)
        if authority_specs is None or name not in authority_specs:
            return native_host.LocalToolInvocationResult(
                result=native_host.ToolResult.blocked(
                    native_host.LOCAL_RUN_WORKTREE_RELEASED_REFUSAL
                ),
                final_gate="not_checked",
                approval_consumed=False,
                reason_code=native_host.LocalToolInvocationReason.AUTHORITY_UNAVAILABLE,
                dispatch_started=False,
                provider_terminal=native_host.LocalProviderTerminal.NOT_STARTED,
            )
        selected_spec = authority_specs[name]
    else:
        selected_spec = spec
    from tldw_chatbook.Chat.console_ssh_custody import bind_model_ssh

    bind_model_ssh(self, authority, selected_spec)
    promotion_invocation = self._invoke_promotion(
        name,
        args,
        clean_args=dict(clean_args),
        spec=selected_spec,
        authority=authority,
    )
    if promotion_invocation is not None:
        promotion_result, approval_consumed = promotion_invocation
        if promotion_result.ok:
            return native_host.LocalToolInvocationResult(
                result=promotion_result,
                final_gate="not_checked",
                approval_consumed=approval_consumed,
                reason_code=native_host.LocalToolInvocationReason.HANDLER_RETURNED,
                dispatch_started=True,
                provider_terminal=native_host.LocalProviderTerminal.RETURNED,
            )
        return native_host.LocalToolInvocationResult(
            result=promotion_result,
            final_gate="not_checked",
            approval_consumed=approval_consumed,
            reason_code=native_host.LocalToolInvocationReason.APPROVAL_REFUSED,
            dispatch_started=False,
            provider_terminal=native_host.LocalProviderTerminal.NOT_STARTED,
        )
    # PR2a Task 5: only the DISPATCHING run's own stamp may resolve
    # this call. `ToolProvider.invoke` has no run parameter, so the run
    # id rides `run_context` (bound by `AgentService` around each
    # invocation); `""` outside any run matches no stamp a review hook
    # writes, so such a call resolves through the fresh gate below.
    gate = self._verdict_for(name, args, native_host.current_run_id())
    if gate.verdict == "allow":
        dispatch_started = False
        provider_terminal = native_host.LocalProviderTerminal.NOT_STARTED

        def _invoke_allowed() -> native_host.LocalToolInvocationResult:
            nonlocal dispatch_started, provider_terminal
            try:
                check_task_provider(self, tool_id, args)
            except Exception:  # noqa: BLE001 - authority failures must refuse
                return native_host.LocalToolInvocationResult(
                    result=native_host.ToolResult.blocked(
                        "workflow task scope refused"
                    ),
                    final_gate=gate.verdict,
                    approval_consumed=gate.approval_consumed,
                    reason_code=native_host.LocalToolInvocationReason.AUTHORITY_UNAVAILABLE,
                    dispatch_started=False,
                    provider_terminal=native_host.LocalProviderTerminal.NOT_STARTED,
                )
            if not self._authority_is_valid(authority, write=write):
                return native_host.LocalToolInvocationResult(
                    result=native_host.ToolResult.blocked(
                        native_host.LOCAL_ROOT_CHANGED_REFUSAL
                    ),
                    final_gate=gate.verdict,
                    approval_consumed=gate.approval_consumed,
                    reason_code=native_host.LocalToolInvocationReason.ROOT_CHANGED,
                    dispatch_started=False,
                    provider_terminal=native_host.LocalProviderTerminal.NOT_STARTED,
                )
            # TASK-34351: resolve the handler once, before any side
            # effect. The run's worktree may have been retired while its
            # approval card waited; re-indexing the alias cache at the
            # handler site raised a bare KeyError whose text (the alias)
            # reached the model as the error.
            if (
                authority is not None
                and name in native_host._PATH_AUTHORITY_LOCAL_NAMES
            ):
                handler_spec = self._path_specs_by_alias.get(authority.alias, {}).get(
                    name
                )
                if handler_spec is None:
                    return native_host.LocalToolInvocationResult(
                        result=native_host.ToolResult.blocked(
                            native_host.LOCAL_RUN_WORKTREE_RELEASED_REFUSAL
                        ),
                        final_gate=gate.verdict,
                        approval_consumed=gate.approval_consumed,
                        reason_code=native_host.LocalToolInvocationReason.AUTHORITY_UNAVAILABLE,
                        dispatch_started=False,
                        provider_terminal=native_host.LocalProviderTerminal.NOT_STARTED,
                    )
            else:
                handler_spec = spec
            # Phase 3c (Task 17): redaction strips the authority's root
            # locator from results/paths. A local root unwraps to its
            # Path (byte-identical); a RemoteRoot redacts LEXICALLY
            # against the descriptor's locator spellings (canonical
            # locator / remote path / display URI) — no laptop disk,
            # no wall.
            if authority is None:
                redaction_root: "native_host.Path | native_host.RemoteRoot | None" = (
                    self._result_redaction_root
                )
            elif isinstance(authority.root, native_host.RemoteRoot):
                redaction_root = authority.root
            else:
                redaction_root = native_host.local_root_path(
                    authority.root, site="result redaction"
                )
            if automatic_work is not None:
                try:
                    automatic_work.check()
                except Exception as exc:
                    reason = (
                        str(exc)
                        if isinstance(exc, AutomaticWorkRefused)
                        else "chain unavailable"
                    )
                    return native_host.LocalToolInvocationResult(
                        result=native_host.ToolResult.blocked(
                            f"automatic tool call refused: {reason}"
                        ),
                        final_gate=gate.verdict,
                        approval_consumed=gate.approval_consumed,
                        reason_code=native_host.LocalToolInvocationReason.AUTHORITY_UNAVAILABLE,
                        dispatch_started=False,
                        provider_terminal=native_host.LocalProviderTerminal.NOT_STARTED,
                    )
            from tldw_chatbook.Chat.console_ssh_custody import bind_model_ssh

            bind_model_ssh(self, authority, handler_spec)
            dispatch_started = True
            # Phase 3b (Task 16): the CAS/ledger sites below dispatch
            # on the authority's root TYPE. A local root keeps the
            # exact pre-Phase-3b behaviour (provider-side resolve +
            # laptop hash); a RemoteRoot NEVER reaches a laptop-disk
            # call -- its stamps come from worker-reported sha256/size
            # values in the fs_read/fs_write/fs_edit/fs_patch
            # responses, so the laptop's same-named copy of a remote
            # path can never be read, hashed, or stamped.
            ledger_root_source = authority.root if authority is not None else self._root
            remote_ledger = isinstance(ledger_root_source, native_host.RemoteRoot)
            if remote_ledger:
                ledger_root: "native_host.Path | native_host.RemoteRoot" = (
                    ledger_root_source
                )
                # Remote roots are constructed with an executor in the
                # slot (Phase 3a guards), so this is never None here.
                ledger_executor = (
                    authority.workspace_executor if authority is not None else None
                )
            else:
                ledger_root = native_host.local_root_path(
                    ledger_root_source,
                    site="read/write CAS ledger hashing",
                )
                ledger_executor = None
            # ruling 1 (TASK-28238 P1 T3): `clean_args` is read by the
            # fs_read branch below BEFORE the fs_write branch's
            # would-be reassignment; assigning to `clean_args` anywhere
            # in this closure makes Python treat the name as local to
            # the whole function, and that earlier read would then hit
            # an unbound local. `dispatch_args` is a separate alias so
            # the fs_read branch's `clean_args` reads stay closure
            # reads of the enclosing scope's variable.
            dispatch_args = clean_args
            stale_guard = None
            if name == "fs_read" and not remote_ledger:
                try:
                    self._record_fs_read_observation(clean_args, ledger_root)
                except Exception:  # noqa: BLE001 - observation must never affect dispatch
                    pass
            elif name == "fs_read":
                # Remote: the stamps live in the response the worker
                # is ABOUT to produce -- recorded post-dispatch below.
                pass
            elif name == "fs_write":
                workflow_write = (
                    automatic_work is not None
                    and automatic_work.attempt_kind == "workflow_phase"
                    and clean_args.get("dry_run") is not True
                )
                guard_args = clean_args
                if workflow_write:
                    # Model CAS cannot replace this run's actual observation.
                    guard_args = {
                        key: value
                        for key, value in clean_args.items()
                        if key not in {"expected_sha256", "expected_absent"}
                    }
                stale_guard = self._fs_write_guard_injection(
                    guard_args,
                    ledger_root,
                )
                if workflow_write:
                    observed = stale_guard[0] if stale_guard is not None else {}
                    explicit_absence = (
                        clean_args.get("expected_absent") is True
                        and "expected_sha256" not in clean_args
                    )
                    conflict = (
                        any(
                            key in clean_args and clean_args[key] != observed.get(key)
                            for key in ("expected_sha256", "expected_absent")
                        )
                        if stale_guard is not None
                        else not explicit_absence
                    )
                    if conflict:
                        dispatch_started = False
                        return native_host.LocalToolInvocationResult(
                            result=native_host.ToolResult.blocked(
                                "workflow fs_write requires this run's observed content "
                                "or an atomic absent-file precondition; read the file again"
                            ),
                            final_gate=gate.verdict,
                            approval_consumed=gate.approval_consumed,
                            reason_code=native_host.LocalToolInvocationReason.INVALID_ARGUMENTS,
                            dispatch_started=False,
                            provider_terminal=native_host.LocalProviderTerminal.NOT_STARTED,
                        )
                if stale_guard is not None:
                    dispatch_args = stale_guard[0]
            elif (
                name in {"fs_edit", "fs_patch"}
                and clean_args.get("dry_run") is not True
            ):
                # A preview never writes, so there is no clobber risk
                # for the pre-check to guard against (fs_write's own
                # injection already skips on dry_run the same way).
                _stale = self._stale_targets_for(
                    name,
                    clean_args,
                    ledger_root,
                    executor=ledger_executor,
                )
                if _stale:
                    shown, stamp, now = _stale[0]
                    message = self._stale_write_refusal(shown, stamp, now)
                    if len(_stale) > 1:
                        message += f" (+{len(_stale) - 1} more stale targets)"
                    provider_terminal = native_host.LocalProviderTerminal.RETURNED
                    return native_host.LocalToolInvocationResult(
                        result=native_host.ToolResult.blocked(message),
                        final_gate=gate.verdict,
                        approval_consumed=gate.approval_consumed,
                        reason_code=native_host.LocalToolInvocationReason.HANDLER_RETURNED,
                        dispatch_started=True,
                        provider_terminal=provider_terminal,
                    )
            try:
                check_task_provider(self, tool_id, args)
            except Exception:  # noqa: BLE001 - authority failures must refuse
                return native_host.LocalToolInvocationResult(
                    result=native_host.ToolResult.blocked(
                        "workflow task scope refused"
                    ),
                    final_gate=gate.verdict,
                    approval_consumed=gate.approval_consumed,
                    reason_code=native_host.LocalToolInvocationReason.AUTHORITY_UNAVAILABLE,
                    dispatch_started=False,
                    provider_terminal=native_host.LocalProviderTerminal.NOT_STARTED,
                )
            try:
                raw_output = handler_spec.handler(dispatch_args)
                result = native_host.ToolResult(
                    ok=True,
                    content=self._bounded_result(
                        native_host.redact_root_locator(
                            raw_output,
                            redaction_root,
                        ),
                        invocation_id=name,
                    ),
                )
                if name == "fs_read" and remote_ledger:
                    # Phase 3b: record the remote observation from the
                    # worker's OWN response -- the raw pre-redaction,
                    # pre-bounding text (a bounded tail cannot be
                    # parsed). Never affects dispatch.
                    try:
                        self._record_fs_read_observation(
                            clean_args,
                            ledger_root,
                            worker_result=native_host._response_result_text(raw_output),
                        )
                    except Exception:  # noqa: BLE001 - observation must never affect dispatch
                        pass
                if name in {"fs_write", "fs_edit", "fs_patch"}:
                    self._update_ledger_after_write(
                        name,
                        clean_args,
                        ledger_root,
                        result_text=native_host._response_result_text(raw_output),
                    )
                provider_terminal = native_host.LocalProviderTerminal.RETURNED
                return native_host.LocalToolInvocationResult(
                    result=native_host.replace(
                        result, approval_decision=gate.approval_decision
                    )
                    if result.outcome != "blocked"
                    else result,
                    final_gate=gate.verdict,
                    approval_consumed=gate.approval_consumed,
                    reason_code=native_host.LocalToolInvocationReason.HANDLER_RETURNED,
                    dispatch_started=True,
                    provider_terminal=provider_terminal,
                )
            except native_host.WorkspaceToolExecutionError as exc:
                if (
                    exc.code == "cleanup_unproven"
                    and authority is not None
                    and authority.on_cleanup_unproven is not None
                ):
                    try:
                        authority.on_cleanup_unproven()
                    except Exception as observer_exc:  # noqa: BLE001 - preserve refusal
                        native_host.logger.warning(
                            "Cleanup observer failed (exception_type={})",
                            type(observer_exc).__name__,
                        )
                if stale_guard is not None and native_host._is_cas_precondition_failure(
                    str(exc)
                ):
                    provider_terminal = native_host.LocalProviderTerminal.RETURNED
                    return native_host.LocalToolInvocationResult(
                        result=native_host.ToolResult.blocked(
                            self._stale_write_refusal(
                                str(clean_args.get("path", "")),
                                stale_guard[1],
                                native_host._cas_failure_now(stale_guard[2]),
                            )
                        ),
                        final_gate=gate.verdict,
                        approval_consumed=gate.approval_consumed,
                        reason_code=native_host.LocalToolInvocationReason.HANDLER_RETURNED,
                        dispatch_started=True,
                        provider_terminal=provider_terminal,
                    )
                if remote_ledger and name == "fs_read":
                    self._record_remote_read_failure_safe(
                        clean_args, ledger_root, str(exc)
                    )
                provider_terminal = native_host.LocalProviderTerminal.RAISED
                result = native_host._workspace_execution_error_result(
                    exc,
                    redaction_root=redaction_root,
                )
                return native_host.LocalToolInvocationResult(
                    result=native_host.replace(
                        result, approval_decision=gate.approval_decision
                    )
                    if result.outcome != "blocked"
                    else result,
                    final_gate=gate.verdict,
                    approval_consumed=gate.approval_consumed,
                    reason_code=native_host._workspace_execution_error_reason(exc),
                    dispatch_started=True,
                    provider_terminal=provider_terminal,
                )
            except Exception as exc:  # noqa: BLE001 — protocol boundary
                if (
                    remote_ledger
                    and stale_guard is not None
                    and native_host._is_cas_precondition_failure(str(exc))
                ):
                    # Phase 3b: the remote transport executor raises its
                    # own typed error (not WorkspaceToolExecutionError)
                    # for a worker-side CAS refusal; relabel it with
                    # the same stale-write refusal the local path gets.
                    # No fresh worker value exists (failures carry no
                    # stamps) -- see `_REMOTE_NOW_DISPLAY`.
                    provider_terminal = native_host.LocalProviderTerminal.RETURNED
                    return native_host.LocalToolInvocationResult(
                        result=native_host.ToolResult.blocked(
                            self._stale_write_refusal(
                                str(clean_args.get("path", "")),
                                stale_guard[1],
                                native_host._REMOTE_NOW_DISPLAY,
                            )
                        ),
                        final_gate=gate.verdict,
                        approval_consumed=gate.approval_consumed,
                        reason_code=native_host.LocalToolInvocationReason.HANDLER_RETURNED,
                        dispatch_started=True,
                        provider_terminal=provider_terminal,
                    )
                if remote_ledger and name == "fs_read":
                    self._record_remote_read_failure_safe(
                        clean_args, ledger_root, str(exc)
                    )
                provider_terminal = native_host.LocalProviderTerminal.RAISED
                error = native_host.redact_root_locator(
                    str(exc) or repr(exc),
                    redaction_root,
                )
                return native_host.LocalToolInvocationResult(
                    result=native_host.ToolResult(
                        ok=False,
                        error=error[: native_host._MAX_ERROR_CHARS],
                        approval_decision=gate.approval_decision,
                    ),
                    final_gate=gate.verdict,
                    approval_consumed=gate.approval_consumed,
                    reason_code=native_host.LocalToolInvocationReason.HANDLER_RAISED,
                    dispatch_started=True,
                    provider_terminal=provider_terminal,
                )

        authority_scope = (
            authority.authority_scope
            if authority is not None
            else self._authority_scope
        )
        if (
            authority_scope is not None
            and name in native_host._PATH_AUTHORITY_LOCAL_NAMES
        ):
            try:
                with authority_scope():
                    return _invoke_allowed()
            except Exception:  # noqa: BLE001 - lease failure is fail-closed
                return native_host.LocalToolInvocationResult(
                    result=native_host.ToolResult.blocked(
                        native_host.LOCAL_AUTHORITY_UNAVAILABLE_REFUSAL
                    ),
                    final_gate=gate.verdict,
                    approval_consumed=gate.approval_consumed,
                    reason_code=native_host.LocalToolInvocationReason.AUTHORITY_UNAVAILABLE,
                    dispatch_started=dispatch_started,
                    provider_terminal=provider_terminal,
                )
        return _invoke_allowed()
    if gate.verdict == "timeout":
        self._record_decision_safe(self.hub_tool_for(name), "denied-timeout")
        result = native_host.ToolResult.blocked(native_host.LOCAL_TIMEOUT_REFUSAL)
    elif gate.verdict == "no_callback":
        self._record_decision_safe(self.hub_tool_for(name), "denied-timeout")
        refusal = (
            self._no_callback_refusal
            if self._no_callback_refusal is not None
            else native_host.LOCAL_TIMEOUT_REFUSAL
        )
        result = native_host.ToolResult.blocked(refusal)
    elif gate.verdict == "gate_error":
        # Fix Round H, Item 1: the resolver raised rather than
        # genuinely resolving to "deny" -- still fails closed (the tool
        # does not run), but the reason told to the model is honest:
        # the permission check itself failed, not a configured Off.
        #
        # task-32280 fix round: the AUDIT row now says the same thing
        # the model was told. It used to record plain "denied", which
        # Audit renders as "Denied by you" -- a claim about a person
        # for a gate that never resolved at all.
        self._record_decision_safe(
            self.hub_tool_for(name), native_host.UNRESOLVED_DENIED_DECISION
        )
        result = native_host.ToolResult.blocked(native_host.LOCAL_GATE_ERROR_REFUSAL)
    else:
        # "deny" and any unrecognized verdict fail closed the same way
        # to the MODEL -- but not to the audit log. task-32280 fix
        # round: `_verdict_for()` already knows which of the two this
        # is (`refusal_reason`), and throwing that away here is what
        # made a configured Off indistinguishable from a person
        # pressing Deny. PERMISSION_OFF is the configured state; an
        # unrecognized verdict resolves nothing, so it is neither.
        # Qodo #7 (lane B): the TEXT names who refused too --
        # `LOCAL_DENY_REFUSAL` only where the resolver actually said
        # "deny" (a configured Off), the user's own string everywhere
        # else, so the bridge renders "blocked (Off)" vs "denied by you".
        reason = gate.refusal_reason
        if reason == native_host.LocalToolInvocationReason.PERMISSION_OFF:
            decision = native_host.POLICY_DENIED_DECISION
        elif reason == native_host.LocalToolInvocationReason.APPROVAL_REFUSED:
            decision = "denied"
        else:
            decision = native_host.UNRESOLVED_DENIED_DECISION
        self._record_decision_safe(self.hub_tool_for(name), decision)
        result = native_host.ToolResult.blocked(
            native_host.LOCAL_DENY_REFUSAL
            if reason is native_host.LocalToolInvocationReason.PERMISSION_OFF
            else (gate.denial_refusal or native_host.LOCAL_USER_DENY_REFUSAL)
        )
    return native_host.LocalToolInvocationResult(
        result=native_host.replace(result, approval_decision=gate.approval_decision),
        final_gate=gate.verdict,
        approval_consumed=gate.approval_consumed,
        reason_code=(
            gate.refusal_reason
            or native_host.LocalToolInvocationReason.APPROVAL_REFUSED
        ),
        dispatch_started=False,
        provider_terminal=native_host.LocalProviderTerminal.NOT_STARTED,
    )
