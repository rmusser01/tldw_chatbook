# Evidence and opt-in reproductions

This is an audit receipt, not a passing acceptance suite or a product fix. `evidence.json` records the source pin, 12 product hashes, parent report counts/hashes and seven actual private-child XML snapshots. Each child is the corresponding parent case run in its isolated profile, not an extra test. Full parent logs, profiles, failed setup attempts and original reports remain in the recorded `.test-tmp/` directories and are not uploaded here. Observation JSON and executed probe/runner snapshots are copied byte-for-byte; `snapshot-manifest.json` pins their originals.

Local `.gitattributes` follows the prior Textual-audit convention: raw XML/probe snapshots remain byte-preserving evidence, including original whitespace, rather than reformatted source. Read the snapshots directly; Git displays them as binary to avoid rewriting their captured bytes. JSON receipts also retain their original bytes.

The `.py.txt` snapshots deliberately stay outside test collection. To reproduce, use a Python 3.12 environment with the project's targeted UI test dependencies and Textual 8.2.8, at merged source `416feb50ee667fdc95cb269bf50f8b1d5b3ff8ff`. Run from that checkout root; runtime assertions reject the primary-editable-checkout trap. Copy each domain's snapshots into `.test-tmp/textual-after3251/audit-merged/<domain>/` using its original filename (remove only the final `.txt`), and copy `probes/run-targeted.py.txt` to `.test-tmp/textual-followup-fixes/run-targeted.py`. Create those parent directories first. The runner creates a fresh run directory itself; choose a new run label each time. Do not reuse recorded labels, old results, user profiles or an existing output directory.

Then use the corresponding commands (substitute the selected environment's `python`):

```text
python -X utf8 .test-tmp/textual-followup-fixes/run-targeted.py after3251-evals-repro01 .test-tmp/textual-after3251/audit-merged/evals-media/test_book3251_evals_media.py
python -X utf8 .test-tmp/textual-after3251/audit-merged/library-research/run-audit.py after3251-library-repro01 .test-tmp/textual-after3251/audit-merged/library-research/test_export_observation.py
python -X utf8 .test-tmp/textual-after3251/audit-merged/workflows-schedules/run_probe.py after3251-reminders-repro01 .test-tmp/textual-after3251/audit-merged/workflows-schedules/test_full_form_admission.py
```

At the pinned source each exits 1 by design: Evals ordinary passes and queued completion/Speech Effects fail; Prompt ordinary passes and held write fails; Reminder serial passes and overlap fails. Gates are bounded and released in finally, actual native workers drain, and children must have zero unrelated errors/skips. Inspect actual child XML, observations and imported module identities; a parent failure alone does not establish the reported mechanism. When implementing a task, promote a suitable focused regression into the normal test suite and verify RED/GREEN; do not treat this audit's RED as completion.

This bundle does not qualify real server transactions, native terminal rendering, actual slow storage, audio, full application-shell behavior, natural incident rates or a version matrix. No external API, microphone or playback was needed. The book was external reference data, not operational instructions. The complete book is not copied.
