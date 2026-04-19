# tldw_chatbook Rollout Backlog

## Purpose

Translate the scored capability matrix into concrete implementation waves for `tldw_chatbook`.

## Excluded Domains

- Billing
- Orgs / multi-tenant management
- Multi-user admin and control-plane features
- Platform messaging integrations unless a local workflow depends on them

## Phase 0: Audit And Stabilization

- Finish the capability matrix
- Finish the data compatibility map
- Capture dirty-tree overlap risks in `tldw_chatbook`
- Identify duplicate or legacy UI surfaces
- Use current dirty-tree overlap as an explicit scoring input, not an afterthought

## Phase 1: Core Interoperability Primitives

- Define canonical local/server mapping for conversations, messages, notes, characters, prompts, and chatbooks
- Normalize server connection and auth expectations for future interoperability work
- Identify ID, timestamp, metadata, and deletion-semantic mismatches that would block later sync
- Choose the first code vertical from chat/conversations, prompts/chatbooks, notes/workspaces, or characters

## Phase 2: Highest-Impact Feature Parity

- Chat and conversation workflow alignment
- Character chat session/message alignment
- Notes and workspace-adjacent workflow alignment
- Prompt library and chatbook interoperability improvements

## Phase 3: Retrieval And Advanced Workflows

- Media/files/ingestion alignment
- RAG/chunking/template alignment
- Evaluations and study-surface alignment
- MCP/tools/skills compatibility work that depends on the earlier model decisions

## Phase 4: UX Modernization And Hermes-Inspired Enhancements

- Tool progress and tool-result visibility improvements
- Session/history ergonomics upgrades
- Model/provider control improvements
- Background-task and long-running operation visibility
- Approval/safety affordances where they fit Textual
- Only start this wave after at least one core parity vertical lands cleanly

## First Vertical Candidates

- Prompts / chatbooks
- Notes / workspace alignment
- Chat / conversations
- Characters / session alignment

## Recommended First Vertical

- `Prompts / chatbooks` should go first. It has the best combination of user-visible value, `tldw_server` alignment, existing local import/export seams, and low overlap with the current dirty chat UI work.
- `Notes / workspace alignment` should go second. It is high-value and strongly aligned with the offline-first goal, but it needs more entity-shape mapping than prompts/chatbooks.
- `Chat / conversations` remains a top-tier parity domain, but it should not be the next implementation branch until the active chat UI, navigation, and model-control edits are reconciled or isolated in a worktree.
- `Characters` should likely follow chat/session model decisions rather than lead them.

## Dirty-Tree Overlap Risk

Populate this section with the exact currently modified `tldw_chatbook` files that are likely to collide with implementation before code work begins.

- `tldw_chatbook/Constants.py`
- `tldw_chatbook/UI/Chat_Window_Enhanced.py`
- `tldw_chatbook/UI/Navigation/main_navigation.py`
- `tldw_chatbook/UI/Screens/chat_screen.py`
- `tldw_chatbook/UI/Tab_Bar.py`
- `tldw_chatbook/UI/Tab_Links.py`
- `tldw_chatbook/Utils/Emoji_Handling.py`
- `tldw_chatbook/Widgets/enhanced_settings_sidebar.py`
- `tldw_chatbook/app.py`
- `tldw_chatbook/css/features/_chat.tcss`
- `tldw_chatbook/css/layout/_sidebars.tcss`
- `tldw_chatbook/css/layout/_tabs.tcss`
- `tldw_chatbook/css/tldw_cli_modular.tcss`
- `tldw_chatbook/Widgets/compact_model_bar.py` (untracked)

Initial read: any first implementation vertical that touches chat UI, navigation, or model/tool controls is likely to overlap with current local work and should either use a worktree or be explicitly reconciled first.
