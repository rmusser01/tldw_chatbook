# Focused review — TASK-32665

Independent read-only reviewer: `/root/compact_picker_review`.

The first review reproduced three validation failures within AC2: a long missing
path consumed the compact listing; correcting to the current folder left a stale
error because unchanged reactive values emit no location event; a 260-character
component raised ENAMETOOLONG outside the validator's exception boundary.

The final review repeated bounded probes after the repairs. Overlong names now
return validation errors, a 527-character diagnostic preserves listing rows
across resize, and correcting to the current directory clears the error. Path,
selection, focus and highlighted folder remain intact. No actionable findings
remain in the reviewed production delta.

Root review also corrected a caller-test readiness race: wait for the provider
Select itself after expanding its parent before assigning its value. All 16
cases in that module pass, with no wait-budget increase.
