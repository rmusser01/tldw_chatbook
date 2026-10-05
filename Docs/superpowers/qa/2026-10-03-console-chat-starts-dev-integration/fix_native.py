from pathlib import Path
p=Path('tldw_chatbook/Chat/console_chat_start.py');s=p.read_text().replace('from .console_chat_models import','from tldw_chatbook.DB.base_db import operation_owned_connection, run_owned_db_call\n\nfrom .console_chat_models import',1)
s=s.replace('        self.bridge = owner._controller._agent_bridge','        self.bridge = owner._controller._agent_bridge\n        self.store = owner._controller.store\n        self.database = self.bridge.runs_db\n        self.ledger = self.database.automatic_work\n        self.capacity_owner = owner._controller._fleet_wake\n        self.runtime_owner_id = self.capacity_owner.runtime_owner_id')
s=s.replace('            and authorization.owner is self','            and authorization.owner is self\n            and self._controller.store is authorization.store')
s=s.replace('self, request: AgentChatStartRequest, outcome: AgentChatStartOutcome\n','self, request: AgentChatStartRequest, outcome: AgentChatStartOutcome, *, store: Any = None\n')
s=s.replace('        store = self._controller.store\n        publication', '        store = store if store is not None else self._controller.store\n        publication')
s=s.replace('            asyncio.to_thread(\n                store.persistence.update_agent_handoff_launch,','            run_owned_db_call(\n                store.persistence.db,\n                store.persistence.update_agent_handoff_launch,')
s=s.replace('if target is not None and target.incarnation_id == request.session_incarnation:', 'if self._controller.store is store and target is not None and target.incarnation_id == request.session_incarnation:')
s=s.replace('item.request, AgentChatStartOutcome(status, reason)','item.request, AgentChatStartOutcome(status, reason), store=item.store')
s=s.replace('        controller = self._controller\n        if (\n            self._disposed', '        controller = self._controller\n        if getattr(controller, "_maintenance_paused", False):\n            return "maintenance"\n        if (\n            self._disposed')
s=s.replace('        owner = self._controller._fleet_wake\n        ledger = self._controller._agent_bridge.runs_db.automatic_work','        owner = item.capacity_owner\n        ledger = item.ledger')
s=s.replace('            asyncio.to_thread(\n                ledger.prepare_chat_start,','            run_owned_db_call(\n                item.database,\n                ledger.prepare_chat_start,')
s=s.replace('owner_id=owner.runtime_owner_id,','owner_id=item.runtime_owner_id,').replace('            owner.runtime_owner_id,','            item.runtime_owner_id,')
s=s.replace('                            asyncio.to_thread(\n                                item.context.ledger.complete_chat_start,','                            run_owned_db_call(\n                                item.database,\n                                item.context.ledger.complete_chat_start,')
s=s.replace('                            asyncio.to_thread(\n                                item.context.ledger.abort_chat_start,','                            run_owned_db_call(\n                                item.database,\n                                item.context.ledger.abort_chat_start,')
s=s.replace('            controller._fleet_wake.release_automatic_primary(','            item.capacity_owner.release_automatic_primary(')
s=s.replace('        item.accepted = item.context.ledger.accept_chat_start(\n            request.attempt_id,\n            owner_id=item.context.owner_id,\n            limits=AutomaticWorkLimits.from_settings(),\n        )','        with operation_owned_connection(item.database):\n            item.accepted = item.context.ledger.accept_chat_start(\n                request.attempt_id,\n                owner_id=item.context.owner_id,\n                limits=AutomaticWorkLimits.from_settings(),\n            )')
p.write_text(s)
p=Path('tldw_chatbook/Chat/console_fleet_wake.py');s=p.read_text().replace('        if self._disposed or getattr(self._controller, "_disposed", False):','        if self._disposed or getattr(self._controller, "_disposed", False) or getattr(self._controller, "_maintenance_paused", False):',1);p.write_text(s)
p=Path('tldw_chatbook/Chat/console_chat_controller.py');s=p.read_text().replace('            or self._fleet_wake._delivery_tasks\n','            or self._fleet_wake._delivery_tasks\n            or self._chat_start.tasks()\n',1)
s=s.replace('        if origin in {\n            ConsoleSubmissionOrigin.MANUAL,\n            ConsoleSubmissionOrigin.AGENT_CHAT_START,\n        }:\n            try:\n                from tldw_chatbook.Agents.run_hooks', '        if origin is ConsoleSubmissionOrigin.MANUAL:\n            try:\n                from tldw_chatbook.Agents.run_hooks',1)
s=s.replace('            if session.id in busy or len(busy) >= self.max_parallel_runs:\n                return ConsoleSubmitResult(False, False, "A run is already preparing.")','            if origin is ConsoleSubmissionOrigin.AGENT_CHAT_START and self._chat_start.authorizes(chat_start_authorization, session.id):\n                busy = [identity for identity in busy if identity != session.id]\n            if session.id in busy or len(busy) >= self.max_parallel_runs:\n                return ConsoleSubmitResult(False, False, "A run is already preparing.")',1)
s=s.replace('        agent_db = getattr(self._agent_bridge, "agent_runs_db", None)','        agent_db = getattr(self._agent_bridge, "runs_db", None) or getattr(self._agent_bridge, "agent_runs_db", None)',1)
# Refuse target consent before opening any provisional MCP approval. Accepted tools remain normal.
needle='        if not pending:\n'
pos=s.index(needle,s.index('    def request_mcp_approvals('))
s=s[:pos]+'''        start = self._chat_start._active.get(session_id)
        if start is not None and not start.accepted:
            start.withdrawal_reason = "target_consent_required"
            return {call.id: "deny" for call in pending}
'''+s[pos:]
p.write_text(s)
p=Path('tldw_chatbook/Chat/console_agent_bridge.py');s=p.read_text().replace('    denials = {"fork_chat": 0, "new_chat": 0}\n','    denials = {"fork_chat": 0, "new_chat": 0}\n    remembered: set[object] = set()\n',1)
start=s.index('        # Qodo 2761 round, finding 1: NO closure-local remember memo.',s.index('def build_chat_create_tool_closures'))
end=s.index('        # Broad-catch the EXECUTE phase',start)
s=s[:start]+'''        # Only new_chat's trusted preparation verifies a live primary on every
        # invocation. Fork/sub-agent requests continue through native confirmation.
        if tool != "new_chat" or grant_scope not in remembered:
            try:
                decision = confirm(dict(payload))
            except Exception:
                decision = {"allow": False, "remember": False}
            if not isinstance(decision, Mapping) or not decision.get("allow", False):
                denials[tool] += 1
                _release_chat_creation_token(payload)
                return ToolResult(ok=False, error="The user declined. Do not retry this turn.")
            if tool == "new_chat" and decision.get("remember", False):
                remembered.add(grant_scope)
'''+s[end:];p.write_text(s)
print('Native finite ownership, maintenance, input and grant contracts reconciled')
