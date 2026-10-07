# Details page test synchronization

The current-dev Details file initially failed after exhausting a fixed 30-pause loop, then reading a page that its worker had not started yet. A test-only trace captured requested page 8, painted page 7, cached pages 6 and 7, no cancellation and no worker exception. Under this Windows Proactor loop, the nominal thirty 10 ms pauses consumed about 61 ms; monotonic resolution was 15.625 ms. Awaiting the already-requested worker delivered and painted the page. These observations diagnose test synchronization and do not measure product latency.

The final repair adds a stdlib asyncio import and replaces only that polling loop with `asyncio.wait_for(app.workers.wait_for_complete(), timeout=5)`. It then asserts the expected painted page index and equality between the displayed TextArea and cached page. The 300 target reconstruction, redaction, 4,096-character page limit, 30-page ceiling and no-decision assertions remain. Production helpers, card behavior, page requests and authority checks are unchanged. Temporary traces and wrappers were restored before applying the repair.

## Functional verification

Interpreter: `C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe`, Python 3.12.10. Both commands used the repository private control launcher, verified checkout module origins and original-home protection before redirection, and reported `[true, "startup_allowed"]`.

| Command target | Result | Preserved receipt |
| --- | --- | --- |
| `Tests/UI/test_approval_details.py -k test_mounted_grouped_targets_have_bounded_preview_and_complete_details` | 1 passed, 18 deselected, 1 warning in 4.93 s | `details-focused-green.log` |
| `Tests/UI/test_approval_details.py` | 19 passed, 1 warning in 13.53 s | `details-suite-green.log` |

`python -m ruff check --no-cache Tests/UI/test_approval_details.py` passed. `python -m ruff format --check --no-cache Tests/UI/test_approval_details.py` reported one file already formatted. Default-cache attempts were refused by the restricted executor without source changes; the no-cache checks succeeded.

Final repaired test SHA256: `a40a19fdfbc8f46a5571defe95e562e252530a502116dd51b86119185bf4cbb5`. The repaired file has no temporary diagnostic probes.

The test/helper/card matched the original reviewed implementation before this repair. The original checkout was not rerun during diagnosis, so current suite-order failures are not attributed to it beyond the shared fixed-polling implementation. Earlier failed receipts remain recorded. These focused results do not qualify native/browser presentation, responsiveness targets, geometry/theme/Inspect coverage or actual Windows dispatch.

ADR required: no new ADR. Existing ADR-221 applies; this changes test synchronization without changing product behavior or boundaries.
