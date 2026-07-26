# Workspace UAT script (as executed 2026-07-26)

Reproducible recipe for the baseline in `report.md`. Driver: `.claude/skills/verify/SKILL.md` (tmux + SGR clicks), isolated socket to avoid colliding with other sessions' `verify` socket.

## Setup

```bash
SCRATCH=<scratch-dir>; mkdir -p $SCRATCH/ws-uat/captures
printf '[general]\nusers_name = "uat_ws"\n' > $SCRATCH/ws-uat/config.toml
rm -rf ~/.local/share/tldw_cli/uat_ws
tmux -L wsuat new-session -d -x 235 -y 52 \
  "cd <worktree> && TLDW_CONFIG_PATH=$SCRATCH/ws-uat/config.toml <repo>/.venv/bin/python -m tldw_chatbook.app"
sleep 14
```

Requires a llama.cpp-compatible server at `http://127.0.0.1:9099` for scenarios 2, 8, 20 (all other scenarios run without it).

## Act 1 — fresh first-run (captures 01–11)

1. Capture initial Console (setup card; rail hidden until provider setup).
2. Setup card → "Set up provider" → Settings/Providers: select llama.cpp (dropdown needs wheel-scroll to reach local providers), model = server model name, endpoint = `http://127.0.0.1:9099`, Test Provider, save via `Esc` then `s`.
3. Back to Console: capture Session section (Workspace row, Switch, RAG Scope, recovery copy, blank Scope row, empty browser sections).
4. Attempt workspace creation: the New button is invisible (clipped) — sweep-click the blank strip right of Switch (~cols 31–41 of the pane, same row as Switch) until `Workspace` value changes. Record which column hits.
5. Switcher round-trip: Switch → capture modal → Escape → Switch → click "Default".
6. First chat: send "Reply with only the word ok."; capture Scope row ("This conversation") and browser placement (under Chats).
7. Details tray: scroll rail, click the `Details ▸` caret (clicking the word may miss), capture rows incl. the wrapped "handoff" orphan and truncations.
8. Settings → Overview: capture "Server, sync, workspace, and handoff" rows + boundary/recovery copy.

## Act 2 seeding (app stopped: `tmux -L wsuat send-keys C-q`)

```python
# TLDW_CONFIG_PATH=<scratch>/ws-uat/config.toml, cwd = worktree
from tldw_chatbook.config import get_workspaces_db_path, get_chachanotes_db_path
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService

registry = LocalWorkspaceRegistryService(WorkspaceDB(get_workspaces_db_path(), "uat-seed"))
# NOTE: Act 1 already created workspace-local-1 via the invisible New button.
registry.create_workspace(workspace_id="workspace-local-2", name="Workspace 2")
registry.set_active_workspace("workspace-local-1")
svc = ChatPersistenceService(CharactersRAGDB(get_chachanotes_db_path(), "uat-seed"), workspace_registry=registry)
for ws, titles in [("workspace-local-1", ["Client A kickoff", "Client A budget", "Client A retro"]),
                   ("workspace-local-2", ["Client B kickoff", "Client B scope"])]:
    for t in titles:
        cid = svc.create_conversation(assistant_kind="generic", assistant_id="console",
                                      conversation_title=t, scope_type="workspace", workspace_id=ws)
        svc.create_message(conversation_id=cid, sender="user", content=f"Notes for {t}")
        svc.create_message(conversation_id=cid, sender="assistant", content="Acknowledged.")
registry.link_membership("workspace-local-1", item_type="conversation",
                         item_id="conv-ghost-uat", title="Ghost chat")  # ghost: no ChaChaNotes record
```

Relaunch as in Setup.

## Act 2 — seeded (captures 12–29)

9. Grouped browser: active group expanded, others collapsed; scroll for full listing.
10. Cross-workspace click: expand the inactive group, click a row → verify `Workspace` value changed with no toast (scroll rail top to see it).
11. Search "budget" → "1 match" + force-expand; Clear.
12. Star a row → verify Starred section entry.
13. Details tray in the new workspace: handoff rows ("<title> - reference"), duplicate Handoff labels, ACP rows.
14. Library: rail → expand the collapsed **Details** disclosure → scroll to bottom for Workspace/Actions groups. Click "Use in Console" (expect nothing visible), then "Create local workspace" — verify via DB (`workspace_records.active`), observe rail recompose-to-top + re-collapse, then Console rail shows the new workspace.
15. Ghost row: click "Ghost chat" → expect no reaction (capture ≤0.5s after click).
16. Rail-state: toggle Details open, switch workspace via modal, switch back, re-check caret; dump `[console.rail_state]` keys from the scratch config.
17. Settings → Storage: capture `Workspaces` path input + resolved-path caption + restart guidance.
18. Live separation: new tab → "Say only: alpha"; switch to Default → new tab → "Say only: beta"; verify placement in UI and DB:
    `sqlite3 ~/.local/share/tldw_cli/uat_ws/tldw_chatbook_ChaChaNotes.db "SELECT title, workspace_id FROM conversations WHERE deleted=0;"`

## Teardown

```bash
tmux -L wsuat kill-server
rm -rf ~/.local/share/tldw_cli/uat_ws
```
