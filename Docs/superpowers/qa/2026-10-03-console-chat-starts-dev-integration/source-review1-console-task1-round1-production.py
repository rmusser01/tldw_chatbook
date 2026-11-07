from pathlib import Path
root=Path('/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook')
def replace(path, old, new):
 p=root/path;s=p.read_text();assert s.count(old)==1,(path,s.count(old));p.write_text(s.replace(old,new))
replace('tldw_chatbook/Chat/console_dispatch_repository.py',
'''            or type(acceptance.user_root_fork) is not bool
''', '''            or (
                acceptance.origin == "agent_chat_start"
                and acceptance.continuation_receipt is not None
            )
            or type(acceptance.user_root_fork) is not bool
''')
replace('tldw_chatbook/UI/Console_Modules/workspace.py','import inspect\nimport re','import inspect\nimport json\nimport re')
replace('tldw_chatbook/UI/Console_Modules/workspace.py','''from typing import TYPE_CHECKING, Any, Optional
import asyncio
from datetime import datetime, timezone
import inspect
import json
import re
import time
''','''from typing import TYPE_CHECKING, Any, Optional
''')
replace('tldw_chatbook/Chat/console_chat_controller.py','''        """Require live primary execution and this exact source object lifetime."""''','''        """Require the captured primary or child execution and source lifetime."""''')
replace('tldw_chatbook/Chat/console_chat_controller.py','''        cancel = self._active_cancel_events.get(session.id)
        if cancel is None or cancel.is_set():
            return False
        if self._active_assistant_message_ids.get(session.id) != payload.get(
            "source_message_id"
        ):
            return False
        try:
            if bridge.live_primary_run_id(session.persisted_conversation_id) != run_id:
                return False
            row = bridge.runs_db.get_run(run_id)
            return bool(
                row
                and row["conversation_id"] == session.persisted_conversation_id
                and row["agent_kind"] == "primary"
                and row["status"] not in {"done", "error", "cancelled", "abandoned"}
            )
        except Exception:
            return False
''','''        try:
            row = bridge.runs_db.get_run(run_id)
            if not row or row["conversation_id"] != session.persisted_conversation_id:
                return False
            if payload.get("source_agent_kind") == "subagent":
                # TASK32531 children may survive their parent's turn. Bind to
                # the trusted child actor, never the session's next primary.
                actor = current_run_actor()
                parent_id = payload.get("source_parent_run_id")
                parent = bridge.runs_db.get_run(parent_id) if parent_id else None
                return bool(
                    actor is not None
                    and actor.kind == row["agent_kind"] == "subagent"
                    and actor.run_id == run_id
                    and actor.parent_run_id == row.get("parent_run_id") == parent_id
                    and row["status"] == "running"
                    and parent
                    and parent["conversation_id"] == session.persisted_conversation_id
                    and payload.get("destination") == "same_workspace"
                    and payload.get("mode") == "draft"
                )
            cancel = self._active_cancel_events.get(session.id)
            return bool(
                cancel is not None
                and not cancel.is_set()
                and self._active_assistant_message_ids.get(session.id)
                == payload.get("source_message_id")
                and bridge.live_primary_run_id(session.persisted_conversation_id) == run_id
                and row["agent_kind"] == "primary"
                and row["status"] not in {"done", "error", "cancelled", "abandoned"}
            )
        except Exception:
            return False
''')
replace('tldw_chatbook/Chat/console_chat_controller.py','''        public = validate_new_chat_arguments(payload)
        if not self._chat_creation_source_live(
            payload
''','''        public = validate_new_chat_arguments(payload)
        actor = current_run_actor()
        child = actor is not None and actor.kind == "subagent"
        # Children retain draft creation in the parent's workspace only;
        # destination selection and bounded starts remain primary authority.
        if child and (public["destination"] != "same_workspace" or public["mode"] != "draft"):
            raise PermissionError("primary_creation_authority_required")
        payload = {
            **payload,
            **public,
            "source_agent_kind": "subagent" if child else "primary",
            "source_parent_run_id": actor.parent_run_id if child else None,
        }
        if not self._chat_creation_source_live(
            payload
''')
replace('tldw_chatbook/Chat/console_chat_controller.py','''            "source_run_id": payload["source_run_id"],
            "source_message_id": payload["source_message_id"],
''','''            "source_run_id": payload["source_run_id"],
            "source_agent_kind": payload["source_agent_kind"],
            "source_parent_run_id": payload["source_parent_run_id"],
            "source_message_id": payload["source_message_id"],
''')
replace('tldw_chatbook/Chat/console_chat_controller.py','''                "approved": grant
                in self._chat_create_session_grants.get(source.id, set()),
''','''                "approved": not child and grant
                in self._chat_create_session_grants.get(source.id, set()),
''')
replace('tldw_chatbook/Chat/console_chat_controller.py','''            remember = bool(decision.get("remember", False))
            # Record a standing grant only on an allow that ALSO asked to
''','''            remember = (
                requesting_kind == AGENT_KIND_PRIMARY
                and bool(decision.get("remember", False))
            )
            # Record a standing grant only on an allow that ALSO asked to
''')
replace('tldw_chatbook/Chat/console_agent_bridge.py','''        # Only new_chat's trusted preparation verifies a live primary on every
        # invocation. Fork/sub-agent requests continue through native confirmation.
''','''        # Trusted preparation rechecks the captured primary or child actor.
        # Child decisions never remember, so every child request confirms anew.
''')
