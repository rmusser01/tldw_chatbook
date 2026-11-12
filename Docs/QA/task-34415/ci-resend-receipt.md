# Final-head UI3 refused-echo selection receipt

Current published head `15912073f0f4f39fbf9c29984dc0ec133bda2e07` failed
[Derived Artifacts 37583445896](https://github.com/rmusser01/tldw_chatbook/actions/runs/37583445896).
UI3's original `test_console_r_resends_a_refused_echo_as_one_message` timed out
waiting for the Resend button: **1 failed / 576 passed / four warnings in
768.99s**. The warnings are inherited asyncio marks on four synchronous model
switcher tests. PR, UI1/2/4 and Perf passed, but required derived reproduction
failed. This head is not merge evidence; TASK34415 remains In Progress.

## Actual boundary and correction

The original screen-wide text wait can observe the provider refusal in controls
before the refused USER echo reaches the transcript model. `select_message`
intentionally ignores an absent message; later publication does not repeat that
selection. Resend therefore remains unselected even when the echo becomes visible.

A call-through probe holds the original transcript publication for **250ms**
once an actual failed USER exists. It neither invents model data nor replaces
selection, getters, profiles, admission or rendered results. The original node
then observes refusal with `model_present=False`; real selection leaves
`selected=False`; publication arrives later and the exact original Resend
selector fails. This proves a supported publication race, not the hosted
runner's unrecorded internal ordering or a unique cause.

Only the existing refusal-text wait is scoped to the actual transcript before
selection. Its original 80 attempts, pause, two-second action-selector deadline,
keyboard dispatch and every message/recovery/draft assertion remain. No new
wait/helper, deadline increase, production/profile/workflow change or cancellation.
Import sorting and one whitespace-only formatter change accompany the repair.
All other module AST nodes, after normalizing from-import name order, and all
original assertions/call arguments except that declared text scope are identical.

## Retained outcomes

- Original isolated case: **1 passed in 8.70s**; late unrelated pytest garbage
  cleanup warnings remain in the original log, not claimed warning-free.
- First observer: **setup-invalid AttributeError**, with two sandbox pytest-cache
  warnings. Retained, not credited as RED.
- Valid observation only: **1 passed in 6.81s**. Not a hosted reproduction.
- Final-probe original RED: **1 failed in 8.41s**, exact Resend selector failure.
- Identical delayed-publication GREEN: **1 passed in 7.38s**; final formatted
  source GREEN: **1 passed in 7.16s**. No pytest warnings in these valid probes.
- Ordinary affected module: **13 passed in 37.81s**, no pytest warnings. Both
  unchanged private-profile children pass separately in **8.16s** and **7.46s**,
  without pytest warnings. This run precedes only import sorting and whitespace
  formatting; the AST identity check and final original-node GREEN cover that
  mechanical cleanup without rerunning unrelated suites.
- Changed file: Ruff and format clean. An initial non-escalated Ruff invocation
  could not create its cache; it is not credited as a static pass. Native scoped
  checks succeeded afterward; the inherited import-sort and format debt were
  removed mechanically, without changing test logic.
- All eleven current artifact guards pass in the retained `preflight.log`.

The controlled runs use the original node and unchanged fixture mode:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=/private/tmp/pr3034-resend-probe-ahTaHd \
python -m pytest \
  Tests/UI/test_console_turn_resend_ui.py::test_console_r_resends_a_refused_echo_as_one_message \
  -p pytest_asyncio.plugin -p pytest_timeout -p resend_probe \
  --resend-probe-mode=hold -q -s
```

The runnable `resend_probe.py` is byte-identical to the published `.txt` source.
Controlled RED/GREEN and module runs have separate fresh basetemp, log and XML
paths in the manifest. The original isolated run used pytest's default temporary
root and retained its housekeeping warnings. No existing test directory was
deleted or reused by this work.

The [exact diagnostic source](ci-resend-probe.txt) and
[SHA-256 manifest](ci-resend-sha256.txt) are the published evidence closure.
Complete hosted failure, original/invalid/valid probe, parent/private-child logs
and XML remain **local**, under `/private/tmp/pr3034-resend-probe-ahTaHd`,
`/private/tmp/pr3034-ui3-resend-U2DVQP`, and
`/private/tmp/pr3034-ui3-failed-37583445896.log`. Only hand-audited outcomes,
hashes and non-secret probe source are published; no diagnostic dump publication
or full-copy claim. Earlier failures, warnings and qualification gaps are retained.

ADR required: no. Routine test readiness under ADR126/ADR120; production
source/admission, ownership and qualification boundaries are unchanged.
Fresh independent read-only review of all seven staged paths against `15912073`
found no Critical, Important or Minor issues. It confirmed the actual flow,
all five journey assertions, preserved bounds, neighboring/private-profile
behavior, exact probe and all thirty raw/source hashes. Unchanged production
and other tests preserve earlier review applicability. This confirms scoped
publication readiness, not merge clearance; new exact-head CI and bounded
closeout remain required. No task or parent qualification closeout is claimed.
