# MCP master-switch ordering and receipt review

TASK-32793 continues the component workflow review under ADR-150/161.
ADR required: yes.
ADR path: backlog/decisions/169-mcp-local-config-save-lifetime.md.
Reason: extend ADR-168 app ownership to both existing controls for the same
local-tools master setting, preserving all runtime permission boundaries.

1. Hold an older off-write while reversing to on; reproduce out-of-order
   persistence. Hold a write across refresh and reproduce its lost pending label.
   Return committed-file/cache-failed and reproduce the false failure receipt.
2. Extend the existing root owner into a narrow local configuration owner with
   one FIFO and separate per-key latest/confirmed receipts. Preserve existing
   root draft, generation/file identity and shutdown invariants.
3. Admit Tools and Servers master requests synchronously. Project latest pending
   choices and terminal outcomes into both controls, preserving focus and root
   drafts. Refuse stale configuration requests; keep catalog failure separate.
4. Test overlap, cross-control reversal, observer destruction, shutdown, retries,
   partial caches, no-op warnings, external configuration changes and old root
   behavior. Use independent review of ownership and publication boundaries.
5. Qualify real private persistence and pending navigation in dark/light at
   80x24 and 170x48. Inspect actual screenshots, unchanged permission profiles,
   normal shutdown and private data/default-profile receipts.
6. Update review ledgers, task notes and draft PR2707. No full suite or merge.
