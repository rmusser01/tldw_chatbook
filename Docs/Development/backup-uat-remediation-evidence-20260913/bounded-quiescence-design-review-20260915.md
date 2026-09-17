# Bounded final-quiescence observation review

APPROVED as a test-only observation correction; no source edits or tests executed by reviewer.

The proposed asyncio.timeout(2) loop awaits pilot.pause only while section._reconcile_scheduled is true, immediately before the unchanged false assertion. Existing _settle in Tests/UI/test_console_bounded_section.py:27 calls pause exactly three times. Installed Textual pilot.py:535–547 waits for screen/idle and then calls screen._on_timer_update synchronously as its last action. That update may enqueue a valid later layout callback. ConsoleBoundedSection._reconcile_native also explicitly schedules fresh layout after viewport or fixed chrome/hint height changes. Neither API guarantees that exactly three pauses consumes every final callback.

The actual Windows case already passed geometry,scroll,hint,focus and allocation assertions before failing only the pending flag. This supports checking eventual finite quiescence rather than a fixed scheduler-turn count. It does not prove the exact timer event that set the flag in that run.

The two-second timeout must propagate normally, with no exception suppression or flag mutation. It will refuse perpetual rescheduling/no-clearing and is inside the unchanged45s child ceiling. Earlier covered callback stability and scoped-demotion assertions remain exact, as do unmount refusal and all restoration geometry assertions. The final .04s callback-count check remains after the no-pending assertion and catches renewed reconciliation following apparent quiescence. The test accepts finite post-resume convergence; it does not impose a callback-count budget during that convergence, which was not an explicit prior behavior contract.

No product change, native guard change or deadline extension is needed. Root should verify the final patch preserves those assertions and fresh native Windows passes; this design review is not Windows acceptance.
