# Character chat selection repair

Starting a character chat must retain the visible worked chat's provider/model
and keep the new character conversation selected after settings dialogs close.
The investigation reproduced two independent stale-state paths: screen control
mirrors can override the current conversation pair, and a global character chat
can leave the registry on the previous named workspace.

Use the active non-pristine session's provider/model to resolve the established
new-session generation defaults. Keep current canonical-default convergence for
untouched sessions and the existing control-derived fallback when no source chat
exists. The character card still owns the new system prompt and greeting; do not
copy another conversation's prompt, prefill, or settings persistence identity.

Both Personas Start Chat and the Console character picker's New Chat use this
shared resolution. Align the workspace registry to the actual target session
through the existing tab-activation seam before activating it. This also applies
to an existing duplicate handoff and a repurposed pristine session.

Alternatives considered: updating global control mirrors on every chat-only
Apply would spread a conversation-local edit into later creation policy; removing
resume reconciliation would break deliberate workspace changes from Settings and
Library. Repair the character-launch inputs and activation instead.

ADR required: no

Existing governance: ADR-006 and ADR-095 own model-default resolution and
source-owned handoffs; TASK-15120 defines registry alignment when opening global
conversations. This repair changes no storage, permissions, or ownership boundary.

Verify real launch methods, untouched-session convergence, both launch routes,
registry state, and source-chat preservation. UAT must exercise a named workspace,
stale vLLM controls, llama.cpp selection, Apply, Cancel, and a local HTTP reply.
