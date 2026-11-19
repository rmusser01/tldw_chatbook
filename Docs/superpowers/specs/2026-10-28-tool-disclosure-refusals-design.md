# Tool disclosure refusal recovery

The user approved distinguishing unavailable tools from unloaded schemas after a Console project-inspection report. The reporting environment was Fedora Linux; the recovery logic is platform independent. The report does not establish which condition occurred; both currently produce the same refusal.

Allowed catalog tools whose schemas are absent must stay blocked before dispatch, but receive a concrete `load_tools` recovery instruction. Tools excluded from the run retain the existing generic refusal, including when their schema happens to be present. Ordinary and skill dispatch share the same check. Refusals remain blocked, not started, and carry no invented approval decision.

ADR-104 intentionally replaces the catalog working set. Preserve replacement, model-context budgeting, authorization, approval, hooks, and catalog confidentiality. Do not auto-load or execute guessed names, add retries, or infer permission from a refusal string.

Approach: share a small refusal helper at the service dispatch boundary and test recovery through the real service and SQLite run store. Automatically loading schemas would change the disclosure contract; changing replacement into accumulation would contradict ADR-104.

ADR required: yes
ADR path: backlog/decisions/252-tool-availability-refusals-before-hook-preparation.md; existing: backlog/decisions/104-token-budgeted-agent-tool-disclosure.md; backlog/decisions/078-structured-agent-tool-outcome-provenance.md; backlog/decisions/154-agent-denial-streak-boundary.md
Reason: ADR-252 records the additive typed refusal contract across the service/runtime preparation seam; ADR-104/078/154 continue to govern disclosure and authority.

Related repair: ADR-252 records the typed preparation refusal contract. The shared availability check also verifies catalog membership before promising reload recovery.
