# Windows 05e startup evidence review

Run **34914010324**, job **104207556802**, tested source **05e07702abedcd3b1dac1058385dddb6f880c343**. Independently parsed the saved full log and checked its SHA256 against the supplied summary. No reruns or source changes.

- **31 non-UI cases passed in 9.55s** (log line 534).
- **All four primary full-app cases hit their unchanged 60s timeout**: claim authority/recompose; external-copy keyboard geometry; supported-width llama.cpp; supported-width llamafile. Each ran separately; paired opening/closing timeout banners are not eight failures. Primary step exited 1 (line 1974).
- The separate profiled comparison used the same claim-authority case: pinned dev **4631b60f** passed in **51.11s**, including a 37.30s test call; candidate **05e07702** timed out at the unchanged 60s criterion and exited 1. This is a completed baseline versus incomplete candidate, not two completed timing measurements.
- **Eight profile records: four dev and four candidate; all eight `source_matches=true`; all eight `observer_errors=[]`.** Dev observation elapsed times: 13.391, 29.797, 47.594, 54.625s; candidate: 14.656, 29.875, 45.218, 60.640s. These observer elapsed times have a different origin from the pytest deadline. Source-match receipts validate the diagnostic source selection; they are not installed-wheel Git-blob receipts. Absence of observer errors does not mean absence of application failure.

The candidate terminal main-thread stack (lines 2422–2474) is still in Console composition: settings summary → active settings readiness → store/Canvas binding → live Canvas configuration getter → checked config/raw operation → storage admission authority → registry → native Windows parent/handle work. This is evidence of unfinished checked startup work, not a demonstrated deadlock or an identified faulty guard. The last bounded profile attributes 31.947s inclusive to 337 `acquire_storage` calls and 42.848s inclusive to config wrappers; these overlap and must not be summed or treated as independent costs. The stack does not prove which particular repeated read is avoidable, which native operation dominates wall time, or a safe correction.

**Conclusion:** current candidate fails the existing Windows startup criterion while the pinned dev comparator completes it; the regression signal remains real. The log does not isolate its root cause, establish that the latest telemetry commit introduced it, or justify a cache, guard change, wider config lifetime, or timeout increase. Historical aggregate timing and incomplete workloads cannot establish a Windows speedup or slowdown ratio. Preserve this failure separately from passing non-UI and backup workflow evidence.

Evidence: `/private/tmp/uat-05e077-windows-startup-job.log`; `/private/tmp/uat-05e077-windows-startup-summary.json`.

Log SHA256: `32efe7ebc8e3cbc2eba12d36b4e059d2595e3f14ff1776f5104e285d2e8a6be3`.
Summary SHA256: `5c458512bdcdfe37bfa2965eb707f42a271fd186411204b4354791561fb22298`.
