# Power persona keyboard UAT — c5c05181d834d4b7596107a154216a330af8261d

Artifact root: `/private/tmp/chatbook-backup-uat-repeat-20260913-07b3zfn3`. Ordinary installed `python -P -m tldw_chatbook`, dedicated tmux `power`, synthetic HOME/config, null keyring, no product edits or widget/Pilot assignment. Actions and terminal text/ANSI frames are in this evidence directory.

## Results

- Missing encryption password rejected (017); mismatched confirmation rejected (022); Partial archive acknowledgement required and accepted (027–029).
- Initial capture refused `review_required` with missing identity/required sources (031), no archive. The second-profile fixture had not yet initialized its databases. After normal installed startup initialized that fixture, capture was repeated using a different output filename. This corrected-fixture pass does not erase the first refusal.
- Two-profile encrypted credential-inclusive backup, each selecting the same 1,800-file external research folder, verified in UI (073). Null keyring required explicit acknowledgement of 42 unavailable credential scopes; acknowledged through keyboard controls (059–065).
- Archive: `../power/power-multi-initialized.tldw-backup.zip.age`; SHA256 `86324ee519464336a064867bd37d6bc499fcb88d20e8290df31b5a947cfeee9e`. Inspection reports 3,868 files, 54,226,805 payload bytes, Partial, credentials include. Wrong password rejected (079); correct password verified (082).
- Isolated restore validated (103) to `../power/restore-locations/{one,two,research}`. Exactly 1,800 restored research files match all original SHA256 values; source files unchanged (`power-restored-research-verification.json`). Profiles: `7a3079e053a64b3bbd31f166ea0f2018` and `a88f5897259c45e9bd10448de500fd6b`.
- Selected nonempty Notes/chats group extracted through review and confirmation (110–111): five files, 21,667,840 bytes; every byte hash matches the verified manifest (`power-extraction-verification.json`). Output `../power/manual-extraction` includes inert mapping.
- Read-only acquisition of the encrypted archive confirms the synthetic provider key is retained in its config member. Isolated configs are sanitized and the encrypted recovery archive is retained (`power-credential-verification.json`). External keyring scopes remain explicitly omitted as reviewed.
- Native Open in new process launched first restored profile and mounted Settings (116–117); mounted receipt exists. **Failure:** normal Ctrl+Q returns to original UI with `Failed: opening profile` / `backup_operation_failed` (118). Read-only receipt validation reproduces `isolated_activation_required`; `power-native-open-readback.txt` records traceback. This flow remains a failure pending correction/retest.
- Actual keyboard cancellation passed: Running capturing (129), Cancellation requested (130), Cancelled capturing (131). `../power/power-cancel.tldw-backup.zip.age` absent; 1,800 source files and both source configs unchanged across 1,802 SHA256 checks (`power-cancel-verification.json`).

No novice/newcomer sessions or evidence were changed. The sole unresolved product finding in this persona is the parent failure after native restored-profile exit. Terminal evidence, rather than desktop pixel screenshots, records these real keyboard actions.
