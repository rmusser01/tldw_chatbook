# Exact f311 Windows restore qualification

Independently verified the downloaded `restore-diagnostic` artifact for revision `f311792948ffc252657eb208cb63ceb57fface3d`. Job log identifies Windows2022/Python3.12 selection and successful artifact upload in run34930431683 (artifact10382055382). Native identity reports Windows2022Server/AMD64/nt. No new app/tests or repository edits.

**Outcome:59 native cases and one actual plaintext F9/create/restore/open case pass, with zero failures/errors/skips or missing outcomes.** Native JUnit session1.847s; product JUnit session337.903s, case337.758s. The corrected native NULL-DACL and unknown-ACE cases both execute and pass. Summary selects exactly the two native Windows modules and the `[plain]` product parameter, with effective exit0.

## Provenance

- Rehashed all20 entries in `artifact-sha256.json`; all match, with relative-path/non-symlink checks. The index itself is separately hashed in the review receipt.
- Source receipt identifies exactf311, clean Git status, and `private_tracked_head_copy`; all16,656 reported source paths exist in that Git tree. Clean status is the retained run receipt, not a newly executed checkout status assertion.
- One installed receipt contains2,867 files. Independently compared all2,475 `tldw_chatbook` Python files with exact-revision Git blobs and installed/source digests:2,443 match solely after Git Windows CRLF conversion and32 match exact bytes. No mismatch, no arbitrary normalization.
- Broader installed comparison:2,800 same-path source files match both receipt and Git (2,642 CRLF/158 exact). Another56 installed `tldw_profile_core` files match Git after the declared `pyproject.toml:498–499` package-directory mapping, all using CRLF conversion. The remaining11 launcher/dist-info files are installer-generated and covered by the installed receipt, not falsely described as Git-source files.
- Installed receipt wheel SHA256: `e9c330a1e86788e6f515376adf7431b9480a2dfa46364756bd4b6201add598ed`. The wheel/package tree itself is not included in the sanitized artifact; verification checks the retained installed-file receipt against Git, not a fresh reconstruction of the remote filesystem.

## What the passing product body proves

Reviewed exact `Tests/ProductionApp/test_backup_restore_end_to_end.py` and retained dispatch evidence. The original Console creates a real note, enters backup through F9/actual controls, reviews Complete coverage, creates and verifies an unencrypted archive with credentials excluded, and performs a resumed native write. Archive readback verifies policy/coherence and secret exclusion. The UI then inspects and restores to an isolated recovered profile, with restoration validation asserted before Open.

The Open control uses real terminal suspend and a fresh subprocess. The bounded test wrapper verifies the proposed `python -P -m tldw_chatbook` arguments/environment and delegates to the instrumented child. Although child output includes CLI `--help`, the child subsequently constructs real `TldwCli`, mounts with `run_test`, verifies the opened receipt and UI readiness, exact recovered config/Notes paths, captured-note bytes, absence of the post-capture note, and secret exclusion. It exits normally; the parent asserts successful Open, source config preservation and both original-profile notes intact. Retained dispatch events reach `suspend_exited` with `open_profile/succeeded`.

The sanitized logs omit some transient JSON assertion files; completed JUnit case plus the actual source sequence and dispatch log establish that those assertions returned. This is genuine installed-app integration using a finite synthetic profile and controlled test entry, not a separate manual-user session. Nonfatal child warnings (including an Evals configuration scope diagnostic) remain in retained logs; the pass does not establish every subsystem's readiness or warning-free startup.

## Limits

This verifies one current plaintext journey. It does not claim current encrypted/credential, Persona, default replacement/later rollback, or all broader support cases were selected. Preceding6b20143-product evidence and its58/59 native result are historical; f311 executes the corrected NULL-handle fixture and passes all59 here. Windows GGUF startup failures are separate and remain failures; this longer backup integration pass does not retire that60s criterion.

Numeric case list, file hashes, wheel/provenance accounting: `/private/tmp/uat-f3117-native-independent-summary.json`. Independent Python Git verifier output: `/private/tmp/uat-f3117-native-independent-git-blobs.log`.
