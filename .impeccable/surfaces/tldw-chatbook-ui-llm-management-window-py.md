---
version: 1
slug: "tldw-chatbook-ui-llm-management-window-py"
primary_target: "tldw_chatbook/UI/LLM_Management_Window.py"
related_targets: ["tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_vllm.py","tldw_chatbook/UI/Screens/llm_screen.py"]
---

# vLLM Lab setup

- **Scope and mode:** Extend the established Lab Models destination; Operate.
- **Audience:** First-time local-model users, repeat vLLM operators, and keyboard/security-conscious terminal users.
- **Job:** Resolve prerequisites, launch or connect, prove the served chat model, and adopt it into Console with explicit scope.
- **Primary action:** The action advances by state: Check setup, Start vLLM, Stop, Restart with draft, then Use in Console.
- **Proof:** Ready requires current-generation `/health` and `/v1/models` evidence; process liveness alone is insufficient.
- **Constraints:** Preserve Lab's terminal-native visual system; show state in text; keep secrets, local paths, raw commands, and unrestricted output out of cross-surface state and logs; contain focus and geometry at 80x24, 100x30, and 120x40.
- **Direction:** Readiness-first workbench with immutable Current server and editable Next restart configuration. Advanced options and bounded diagnostics are secondary.
- **Memorable moment:** The loading state resolves into a verified endpoint/model and activates Use in Console without retyping either value.
- **Unresolved:** None for the approved complete-redesign scope; implementation details are sequenced by TASK-31214 through TASK-31221.
