# Windows startup 238564 independent review

**The 60-second startup acceptance still fails.** Completed run34912728812/job104203594626 checked out exact `238564276d7efc84e0954ace75f4c6e62493db3c` (local log lines36,199,202). The full local log includes job cleanup. This review used only that existing log and exact Git source; no new remote run, app boot or source edit.

## Exact outcomes

- Native/non-UI GGUF selection: **31 passed in8.17s** (line533).
- Four separately launched full-app GGUF cases: **all four timed out** under the existing60s limit; none completed its final assertions. They are claim-authority/recompose, external-copy keyboard/geometry, and provider-width cases for llamacpp and llamafile. This is four observed terminal timeouts, not a missing-outcome inference or pytest skip.
- Same-node diagnostic with pinned dev `4631b60f8dd9623fc55bf16f4a37e29fcb1240c7`: **1 passed,1 warning in43.19s**, test-call33.55s and teardown1.16s (lines1694–1706).
- Current candidate diagnostic: **timeout**, original60s pytest limit retained; step exits1 at2125. Its final profile elapsed68.734s is observer-relative time beginning before pytest setup, not an increased test timeout.

All **nine bounded profiles** have `source_matches=true`, with empty observer-error lists: four dev samples at12.969/28.360/43.266/45.750s; five candidate samples at13.078/26.156/40.156/54.375/68.734s. This proves the observer found the imported package beneath its requested source directory. It is not a complete installed-wheel or per-file Git integrity receipt. The diagnostic source is unchanged from043d.

## Actual terminal paths

The primary claim-authority case reaches post-attach/screen-resume reconciliation: workspace registry reconciliation → ensure Console store → Canvas bind/live policy getter → config/native acquisition and Windows security checks (841–891). A concurrent FTS startup worker is acquiring native authority while seeding builtin Persona content (707–729). This does not prove which worker consumed most elapsed time.

The other three primary main-thread dumps retain only lower native frame fragments: external-copy is in bootstrap parent/open-handle work (1092–1104); both width cases are in native pinned-directory/fstat/security/SID work (1307–1315,1482–1490). Their higher caller chains are absent, so the earlier043d summary/readiness caller must **not** be copied onto these cases. Worker samples independently show builtin asset processing or Notes FTS acquisition. Some timeout dumps display source statements inconsistent with their sampled frame line; function identities and retained chain are stronger evidence than those printed statements.

The profiled candidate's terminal path is newly specific: **CompactModelBar.compose:46 → get_cli_providers_and_models:8719 → decorated load_settings → config_participants.operation:343 timed config lock acquisition** (2114–2123). A concurrent worker is **ChatScreen._save_console_rail_preferences:12995 → actual settings mutation → config write/interprocess lock → private text creation/descriptor close → raw._runtime_operation/_check:322 waiting storage._lock** (1940–1971).

Thus there is observed config-lock contention during compact-bar composition alongside a real rail-preference write. There is no complete observed reverse dependency identifying the storage-lock owner, no proof of a permanent lock cycle, and no evidence that this individual getter spent60s. The CompactModelBar body contains one native provider/config getter followed by pure projection and widget yields; it offers no demonstrated adjacent same-source read group to merge. The rail save is a real write under its existing preference/config locks, not a redundant read that can be cached or omitted.

## Comparable measurements and limits

The primary constructors completed in6.526/7.599/6.406/6.820s. Under the same diagnostic observer the completed constructor interval was **candidate23.712153s versus dev4.620144s**. Both are the same lifecycle boundary, but the implementations and concurrent work differ. This supports extra candidate startup cost; it does not isolate a fix or predict unprofiled savings. Pinned dev also reports its existing SQLite privacy-unverified warning; the candidate's native authority work is not equivalent to that baseline's assurance contract.

At the candidate's last incomplete snapshot:477 acquire_storage calls/34.235184s inclusive;607 raw-scope profiler entries/37.579476s;172 CLI-load calls/24.359715s;84 Canvas policy reads/21.878068s;2417 registry calls/18.131535s. Windows open_handle has241907 calls/5.126758s self; security137977 calls/4.248539s self. These totals overlap and must not be added. Generator/coroutine profiler entry counts (including22 compose_content entries) do not prove22 fresh compositions. Omitted top20 entries are unknown, not zero; cProfile is main-thread observation and includes overhead.

Compared with043d, acceptance is unchanged (31 non-UI pass plus four primary timeouts; dev diagnostic passes and candidate times out). The final callback now includes compact model composition and a rail preference writer, whereas043d retained initial summary/Canvas composition and other primary post-attach paths. Candidate477 acquisitions at68.734s versus earlier323 at60.203s describe different unfinished workloads, not measured regression or improvement. The historical86ad/043d aggregate counts cannot establish a new loop or justify grouping unrelated effectful calls.

No guard skip, cache, timeout increase, summary-wide operation or speculative optimization is supported by this artifact. If lock contention is investigated next, record the exact current config writer plus storage-lock holder/phase using bounded native identities; a terminal wait alone does not identify an ordering defect. Keep the60s startup regression criterion separate from backup workflow observation budgets.

## Evidence hashes

- Log `/private/tmp/uat-238564-windows-startup-job.log`: `6a1b9c75450bfb15c1b4f4a2000cab3a48bb79f4f3ea3c45abed5f1db12e1f85`.
- Parsed bounded profiles `/private/tmp/uat-238564-windows-startup-profiles.json`: `89894beb1d3d79f12fa7756b23dc3e9125cb8688b0e9ff2e3176dc83a36734ff`.
- Exact-source diagnostic Python: `3b1a8ff0d4b8e55dad9343e8e0d7e43967d738bfc26f0162788f7eb4ee6c4ebc`.
- Compact accounting/profile receipt: `/private/tmp/uat-238564-windows-startup-summary.json`.
