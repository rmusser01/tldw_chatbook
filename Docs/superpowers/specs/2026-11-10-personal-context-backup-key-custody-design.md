# Personal Context backup and key custody

**Status:** Approved by the owner, 2026-11-10 (all 11 decisions as written; see "Decisions"). Implementation is filed as TASK-40133 to TASK-40136, TASK-40138 and TASK-40139, with follow-ups TASK-40137 (an investigation TASK-40139 depends on) and TASK-40140.
**Task:** TASK-39148. **Base:** dev `ae9f9e8d21`, 2026-11-10. Line numbers are that commit's.

## Owner rulings

- **2026-11-06** (TASK-39110 AC#2, quoted in TASK-39148): "Capture it: back up the database and design key custody so a restore can read it — a larger feature that needs its own design task."
- **2026-11-10:** "Draft it now."
- **2026-11-10** (AskUserQuestion): "Approve all 11". This approved every decision in "Decisions" as written.

## Constraints this design holds to

1. **No impersonation.** A restored device must never act as the old device towards the server.
2. **No key in a readable archive.** The profile key is only ever inside an age-encrypted archive. An unencrypted archive never holds it.
3. **No silent data loss.** A restore never deletes or rewrites the Personal Context database because its key is missing. It never creates a replacement key, and it never overwrites or deletes a keyring entry. ADR-102:210-211 says: "Missing or changed key protectors lock the profile and never trigger silent replacement." ADR-126 decision 8 (:181) says: "Never overwrite shared keyring entries."
4. **Told before, not after.** The backup preview and the restore plan say what will happen to Personal Context before anything is written.

## Facts from code

**Records and keys**

- **F1.** Each record version is AES-256-GCM encrypted under a random DEK. That DEK is wrapped by the profile `encryption_key` (`Personal_Context/crypto.py:39-71`). Each row also has an HMAC tag made with the profile `integrity_key`. A read checks the row's `key_version` against the active key (`repository.py:1323-1330`), then checks both the AEAD and the HMAC (`:1343-1354`). Reading therefore needs **both** keys and the matching version.
  - The file is ordinary SQLite with ciphertext columns, not SQLCipher. It can be copied and schema-validated without the key.
- **F2.** The key material is 65 bytes: a version byte, `encryption_key` and `integrity_key` (`key_protector.py:75-84`). It is stored in the OS keyring under service `tldw_chatbook.personal_context` (`:28`). The account is `profile_ref = "personal-context:" + sha256(absolute db path)` (`repository.py:364-367`).
  - **The key is bound to the database's absolute path, not to the profile.** Another path, machine or OS account means no key.
  - Schema migrations run in `_inspect_schema` **before** the key is loaded (`repository.py:373-378`; migrations at `:630-733`).
- **F3.** Production always uses `KeyringProfileKeyProtector`. It is the repository default (`repository.py:368`), `bootstrap.py:44-49` passes none, and drafts use it too (`interview_launch.py:97-113`).
  - `PassphraseProfileKeyProtector` (`key_protector.py:209-380`) and `InterviewPassphraseKeyProtector` (`:383-426`), which builds one per interview session, have no production caller. Only the package export and `Tests/Personal_Context/test_key_protector.py` use them. **No real profile has a passphrase bundle today.**
  - `PassphraseProfileKeyProtector`'s bundle also binds the path-derived `profile_ref` into its AAD and KDF salt (`:238-239`, `:333-336`). A bundle restored to another path would not open even with the right passphrase.
- **F4.** An existing database whose key is missing raises `ProfileLockedError` from `load()` (`repository.py:380-387`, `key_protector.py:161-165`). Bootstrap then returns a locked service with `profile_present=True` (`bootstrap.py:59-73`).
  - Only a **new** database calls `load_or_create` (`repository.py:374-378`). Today, a restored database whose key is missing is therefore already locked, kept, and never re-keyed.
- **F5.** `encryption_key` never rotates.
  - The first link keeps it, replaces `integrity_key` with the server's, and bumps `key_version` (`repository.py:2051-2055`). It then writes the keyring entry (`:2612-2618`).
  - Remove local profile deletes whichever account the repository loaded (`:4972-4997`). Reinitialising creates a new one (`:1703`).
  - So the same path can hold a different key than an older backup of that path did.
  - Creating a database (`repository.py:374-378`) and Start fresh (`reinitialize_destroyed_profile`, `:1703`, from `service.py:606-635`) both call `load_or_create` on the path account. That adopts any existing entry (`key_protector.py:144-148`).
  - A failed re-initialisation deletes the path account even when it only loaded it (`repository.py:1752-1754`). The secure-removal retry deletes it as well (`service.py:1952-1963` → `repository.py:4995`).
  - `key_version` starts at 1 (`key_protector.py:48`) and only a link bumps it. Two unrelated keys therefore usually share version 1, so the version cannot tell keys apart. A wrong key fails authentication instead: `ProfileIntegrityError`, reported as `profile_integrity_invalid` (`bootstrap.py:69-73`), which reads like tampering.

**Link keys and sync state**

- **F6.** After linking, `integrity_key` is a server-issued key. It is unwrapped by the device's RSA key (`link_service.py:359`, `link_key_custody.py:147-162`). It authenticates sync envelopes in both directions (`Sync_Interop/personal_context_adapter.py:229-271`). Envelopes carry `device_id` and `integrity_key_id` (`Sync_Interop/envelope_builder.py:50-104`).
- **F7.** Three kinds of key live only in the OS keyring:
  - the device RSA wrapping key: one per OS account, `tldw_chatbook.personal_context.link` / `device-wrapping-rsa-v1` (`link_key_custody.py:19-21`, `:205-248`);
  - staged first-link integrity keys (`:401-479`);
  - dataset staging keys (`:481-547`).

  First link sends the public wrapping key and receives a `device_id` (`link_service.py:166-185`).
- **F8.** Link state is not in the Personal Context database. It is in `tldw_chatbook_sync_state.db`:
  - the `personal_context_link_state` table (device, dataset, key ids, cursors), at `Sync_Interop/recovery.py:33`;
  - the shared `sync_profile_state.device_id`, at `:39`.

  Backup **already captures** that database as `runtime.sync_state` (`Sync_Interop/recovery.py:79-90`).
  - On restart, sync resumes only if the dataset staging key loads from the keyring (`app.py:2625-2662`). Otherwise it fails closed with `personal_context_staging_key_unavailable`.
  - A new link refuses when the stored `device_id` differs (`link_service.py:1014-1028`, `personal_context_sync_profile_binding_conflict`).

**Backup and restore today**

- **F9.** Archive encryption (age v1, scrypt passphrase) is optional. It becomes mandatory only when credentials are included.
  - `_capture_options` refuses credentials without encryption (`Backup_Recovery/capture.py:301-302`). `start_backup` calls it at `recovery_service.py:1247`.
  - The reader refuses the same combination (`archive_reader.py:172-173`).
  - Separately, `recovery_service.py:1250-1251` requires a password exactly when `encrypted` is set.
  - The passphrase never reaches capture code (`capture.py:236-250`; `recovery_service.py:240-242`: "Passwords belong only to queued/running call frames").
- **F10.** Keyring secrets already travel in archives. They go into `payload/credential-recovery.json` inside the age container, in `include` mode (user backups) and `rollback` mode (safety copies) (`capture.py:519-530`, `credentials.py:1185-1191`).
  - The closest precedent is the **citation key**. It is a 32-byte keyring key referenced from a SQLite column through `SQLITE_CREDENTIAL_COLUMNS` (`credential_policies.py:16-31`) and captured at `credentials.py:758-772`. It is never written back to the keyring on restore.
  - Credential material has no extraction group (`group_selection.py:96`).
  - In `exclude` mode, which is the default, every `SQLITE_CREDENTIAL_COLUMNS` owner's database is rewritten in staging (`credentials.py:1100-1101`). A citation column becomes `recovery:setup_required` (`credentials.py:704-706`).
- **F11.** Restore never overwrites a keyring entry.
  - A remappable server credential goes to a fresh `recovery_<hex>_<purpose>` scope with set-if-absent (`server_credentials.py:688-695`). The staged reference is then rewritten to point at it (`credentials.py:1231-1302`).
  - Everything else is retained encrypted in `<control>/replacement-<op>/credentials.age` or `isolated-<op>/credentials.age`, after an acknowledgement (`replacement.py:1604-1620`, `isolated_restore.py:120-137`).
  - This is ADR-126 decision 8 (:197-199): "restore into a new scope through the owning adapter rather than overwrite shared credentials".
- **F12.** Isolated restored profiles have no keyring. They run with `keyring.backends.null.Keyring` (`isolated_restore.py:393`, `:423-426`), as does the portable relocation host (`portable_relocation.py:250`). `KeyringProfileKeyProtector` refuses a null backend (`key_protector.py:138-141`).
- **F13.** Replace mode does three things in this order:
  1. It captures every affected original through its owner's own `capture`.
  2. It writes an age-encrypted rollback copy under a fresh rollback password, with `credential_mode="rollback"`, and verifies it by reading it back.
  3. Only then does it publish (`replacement.py:314-659`, `:445-469`, `:536-547`).

  Rollback copies are kept until the user deletes them (ADR-126:182; `recovery_copies.py:187-215`).
- **F14.** Every restore mints a new `installation_id` (`isolated_restore.py:96-98`, `replacement.py:1629-1630`).
  - Every owner with `activation_required=True` is bound into the restored generation's review. On the forward path this happens in `finalize_candidate` (`publication.py:592-604`), which is called from `replacement.py:1682` and `isolated_restore.py:260`. `_rollback_replacement` repeats it (`replacement.py:2584-2596`). Its consumers stay off until `execution_allowed((owner,), path)` passes (`activation.py:224-227`). Precedents are `app.py:4717` and `app_service_wiring.py:3348`.
  - No Personal Context code calls it today.
- **F15.** The current row is `_LocalSQLite(..., "personal_context_repository_capture_unsupported", excluded=False)` (`config_adapter.py:1163-1176`). It blocks every preview that finds the file (`config_adapter.py:705-712`; `inventory.py:57`).
  - Its SQLite policy lacks `centralized_backup_allowed` / `recovery_capture_allowed` (`DB/private_sqlite.py:470-475`).
  - The smallest captured-owner precedent is `Sync_Interop/recovery.py` (90 lines).
- **F16.** Detached import is credential-excluded by design (`detached_layout.py:1`, `:492-493`). It has a fixed owner list at `:929-956`, beside `notes.sync_state`.
  - A detached source holding either Personal Context database is refused as `detached_unknown_content` (`:1146-1153`). Neither file is in `known` (`:1037-1058`).
- **F17.** Personal Context already writes a passphrase-encrypted whole-profile snapshot, including device-only records (`export_service.py:223-271`). It has no import: `load_recovery_export` is called only by tests, and the User Guide says so (`personal-context-profile.md:66-67`).
- **F18.** On device-only records, the ADRs say:
  - ADR-102:82: "`device_only` records never leave Chatbook."
  - ADR-203:35-37 (Accepted) narrows this to "a hard ceiling against Sync and automatic remote model egress". It distinguishes qualified Chatbook-owned local model execution, and separately authorized owner exports, from automatic external disclosure.

## Q1. Does the archive carry the key, and under what credential?

The code facts are F2, F3, F9, F10 and F17. The backup's own credential is the archive password, and capture never sees it. Wrapping the key a second time under that same password would add no protection: the password that opens the archive would also open the wrap.

**Option A: a credential record inside the encrypted archive.** (Recommended.)
- Add a `personal_context` credential kind, referenced from the repository like the citation key (F10).
- Unlike the citation column, the reference is a non-secret account name. `exclude` mode keeps it unchanged and never passes it through the citation sanitizer (F10), so a key-less archive still names the account that a same-device restore needs.
- It is captured in `include` and `rollback` modes, both of which force encryption (F9).
- Pros:
  - There is one secret, the backup password.
  - It reuses the existing capture, acknowledgement and retention machinery.
  - The key never sits in an unencrypted archive.
  - It is not extractable, because credentials have no group (F10).
- Cost: inside a decrypted archive (the private inspection workspace), the key sits beside the database. The backup password then protects Personal Context exactly as it protects the chats, notes and server credentials in the same archive. Personal Context's own at-rest encryption adds no second factor there.

**Option B: a separate Personal Context passphrase.**
- Wrap the key material under a passphrase typed for Personal Context at backup time (scrypt + AES-GCM, AAD bound to `profile_id`, not the path).
- Pros: it works in unencrypted archives, and Personal Context keeps an independent secret inside the archive.
- Cons:
  - It adds a second secret and a new prompt to Backup & Restore and to restore.
  - A forgotten Personal Context passphrase leaves restored data permanently unreadable.

**Option C: no key in the archive.**
- Restore can read records only where the original keyring entry still exists for that exact path (F2): the same machine and OS account, in replace mode.
- Moving to another machine would depend on F17's export, whose import does not exist.
- This does not meet the ruling's "design key custody so a restore can read it" when the device is lost.

**Recommendation: A.**
- On the passphrase bundle: no bundle exists anywhere (F3), so no archive needs to hold one. Leave passphrase custody unshipped.
- If it ever ships, its bundle must bind to `profile_id`, not the path. Otherwise a restore cannot open it (F3).

### Q1a. A backup without credentials

**Option A: capture the database without its key.** (Recommended.)
- The preview says: "Database only: readable only on this device, and only until Personal Context is linked or removed."
  - The first link overwrites the named account in place (`repository.py:2051-2055`, `:2613`), and Remove deletes it. After either, every earlier key-less backup fails to authenticate, even on this device.
  - `exclude` is the default capture mode (`capture.py:296`).
- The restore plan raises an acknowledged issue.
- Such an archive still serves the common case of repairing a profile on the same device. The key is still in this device's keyring, and the archive keeps the database's key reference unchanged (Q1, option A).

**Option B: refuse Personal Context capture unless credentials are included.**
- The row stays `unsupported` and the preview stays incomplete.
- This forces the user to include every credential just to back up Personal Context.

**Option C: capture silently.** This is ruled out by constraint 4.

**Recommendation: A.** A key-less archive holds only ciphertext (F1), so it stays within constraint 2.

## Q2. Restore onto another machine or OS account

The code facts are F2, F4, F11, F12 and F14.

### What re-wraps the key

Today the keyring account is derived from the path (F2), so a different path never finds the key. The design gives the repository an explicit, non-secret **key reference**.

**How the reference is stored and read.**
- Schema v9 adds `profile_meta.key_account`. Empty means the legacy path-derived account.
- Migrations run before the key loads (F2), so the migration only adds the empty column. The repository writes the reference the first time the path-derived account's key authenticates this database. So:
  - a v8 database found at a moved path stays unstamped and locked;
  - it opens again once it is moved back;
  - nothing changes for today's users.
- The repository looks the key up by the stored account, or by the path-derived account while the reference is empty. **It uses a key only after that key authenticates this database's manifest head.** Detection is never by `key_version` (F5).
- A stamped database that is moved to another path keeps finding its key on the same keyring.
- **Creating or re-creating a profile** (the first profile, or Start fresh) writes its key to a **new, unused account** (set-if-absent) and records that account in the same `profile_meta` row.
  - It never adopts an existing keyring entry: no `load_or_create` on the path account (F5).
  - Failure cleanup deletes only an account it created.
  - A database with no profile yet holds no key, and has nothing to authenticate.

**How backup treats the reference.**
- Backup registers `(personal_context.repository, profile_meta, key_account)` as a credential column of a new kind.
- `include` and `rollback` modes record the key material.
- `exclude` mode leaves the reference exactly as it is, and never applies the citation sanitizer (F10).
- A database whose reference is still empty carries no key record.

**How a replace restore installs the key.** Restore **always** installs into a fresh account (`personal-context:recovery_<hex>`, set-if-absent) and rewrites the staged reference, following the server-credential remap (F11). A restore therefore never leaves two databases, live or retained, naming one account.
- The material comes from the archive's credential record. For a key-less archive, it is copied from the account the archive names on this device. If that reference is empty, it is copied from the target path's legacy account.
- Either way, the material is used only after it authenticates the restored database's manifest head.
- If nothing authenticates, the staged reference becomes the marker `recovery:setup_required:<reason>`, which records the restore case. The database then names no account and stays locked.
- An isolated restore cannot write the keyring (F12), so it stamps the marker too. The isolated copy never names the live profile's account.
- The staged reference is always rewritten, either to a fresh account or to the marker; capture's `exclude` treatment does not apply here.
  - **Landing order.** Before readable restores ship, no restore publishes the repository.
    - The live database is left byte-for-byte untouched, and the archive keeps the captured copy.
    - The restore plan says Personal Context will not be restored.
    - Replace restores start publishing it, with a fresh account or the marker, when readable restores ship. Isolated restores start when the restore-without-a-key handling ships.
    - A restore that will leave Personal Context locked says so in its plan before it runs.
- After publication, restore opens the database read-only and authenticates its manifest head again. If that fails, it deletes only the account it just created, and Personal Context stays locked (Q2 copy below).

**Operations that change or delete an account.** The first link's key replace (`repository.py:2612-2618`) and Remove local profile (`:4994-4997`) change or delete only the account this database names, and only after that account's key has authenticated this database.
- Remove records the account it removes in `profile_meta`, so the secure-removal retry deletes exactly that account and nothing else.
- Every restore stamps a fresh account or the marker, and every new profile gets a new account. A retained restore copy therefore never names the live database's account, and neither operation can break a rollback.
- Known ceiling: a user's own manual copy of the database file shares its key account, so linking or removing one copy affects the other.

This "re-wraps" the key in one sense only: the key moves from the encrypted archive into the target keyring. **Nothing re-encrypts records.**

### Re-keying

**Option A: install the key as it is.** (Recommended for v1.)
- Anyone who holds the archive and its password already has this data, and in include mode also the server credential.

**Option B: re-encrypt every row under a fresh `encryption_key` during restore.**
- This protects future copies of the live database against someone who holds an old archive plus a later disk image.
- It needs a new full-rewrite path in a 5,000-line repository. The only rewrite today is the first-link rebaseline, `repository.py:1824-2621`.

**Option C: B, and also replace `integrity_key` with a fresh local key.**
- This also drops the server-shared key, so the next link has to rebaseline from scratch.

**Recommendation: A.** File B as a separate follow-up if the owner wants forward secrecy after a restore.

### What the user sees when the key is missing

Personal Context is locked, as it already is (F4). No keyring entry is created. Every table except `profile_meta.key_account` is unchanged from the archived copy. `key_account` carries the marker, and the marker records the reason. The reason is therefore still named correctly after a restart.

The lock gets reasons and copy that name the cause:

| Situation | Reason | Copy (draft) |
|---|---|---|
| The archive had no key (Q1a), and no account on this device authenticates the restored database | `personal_context_key_not_in_backup` | "Personal Context is locked: this backup did not include its key, and this device does not have it. Your profile data is kept. To read it, restore again with Replace from an encrypted backup that includes credentials." |
| Isolated restored profile (F12), when the archive included the key or an account on this device authenticates the restored database | `personal_context_keyring_unavailable` | "Personal Context can't open in a separately restored copy, because that copy has no access to the system keyring. Its data and key are kept. Restore with Replace to use it." |
| Replace restore when this device has no secure keyring | `personal_context_key_install_unavailable` | "Personal Context was restored but stays locked: this device has no secure keyring to hold its key. The key is kept in the encrypted restore copy. Set up a secure keyring, then restore again." |
| The stored account's key does not authenticate the database's manifest, including a different key with the same `key_version` (F5) | `personal_context_key_mismatch` | "Personal Context is locked: the key on this device belongs to a different copy of the profile. Your data is kept. Use Backup & Restore to roll back or restore the matching copy, then choose Try again." |

The restore plan shows the first three as acknowledged issues **before** the restore runs (constraint 4).

An isolated restore uses the key-not-in-backup reason and copy only when row 1's condition holds: the archive had no key, and no account on this device authenticates the restored database. Otherwise it uses `personal_context_keyring_unavailable`, whose "Restore with Replace" is then true.
- `restore_isolated` (`isolated_restore.py:49`) runs in the live process with the device keyring, so the check is possible there.
- Only the launched isolated copy uses the null backend (`:393`, `:423-426`).

A locked service refuses Remove local profile and Start fresh (`service.py:184-186`). The copy therefore names only what the user can do while locked: restore or roll back through Backup & Restore, then choose **Try again**.

The fourth is detected by authentication when the repository opens.
- A manifest-authentication failure at open is reported as this key mismatch. It replaces the tamper-like `profile_integrity_invalid`, even though a tampered manifest fails the same way (`repository.py:1341-1352`).
- Integrity failures beyond the manifest stay `profile_integrity_invalid`.

After a restore a key mismatch cannot arise, because restore stamps only an account that authenticated. It covers other causes: a manual copy, an unlinked profile from another machine, or a legacy path account that another database's key now occupies.

## Q3. Link and sync key custody

The code facts are F6, F7, F8 and F14.

**Option A: never capture link keys, and never resume a restored link.** (Recommended.)
- The device wrapping key, staged integrity keys and dataset staging keys stay keyring-only and never enter any archive.
- **Generic gate.** After every restore, Personal Context sync stays off until the restored generation is reviewed.
  - That review approves `personal_context.repository` only after a restore that published neither `personal_context.repository` nor `runtime.sync_state`.
  - Personal Context sync also needs the shared owners `config` and `runtime.sync_state`. This review never approves them; see the investigation task (TASK-40137). So the review's copy does not claim sync resumes.
  - After a restore that published either owner, sync stays off with a named reason shown to the user, until the review that clears the link state approves `personal_context.repository`.
  - **The Personal Context review approves only Personal Context's own owner, `personal_context.repository`. It never approves `config` or `runtime.sync_state`.**
    - Approval is per owner and generation (`activation.py:351-384`).
    - Every Sync V2 call also needs those two shared owners approved (`_sync_call`, `server_sync_service.py:85-88`). That includes `LocalFirstSyncService.sync_once`, which syncs Notes, Chat and Personal Context in one call (`local_first_sync_service.py:70-71`), and the link calls (`link_service.py:170`, `:730` → `server_sync_service.py:710-712`, `:811-813`).
    - Approving them would turn Notes and Chat sync back on with the restored device id.
- **Who approves the shared owners.** Who may approve `config` and `runtime.sync_state` after a restore, and through what review, without letting Notes and Chat resume as the old device, is TASK-40137's ruling.
  - The fresh Personal Context link gets past `_sync_call` under that ruling.
  - If the ruling needs an owner decision, the link task stops and brings it to the owner.
- **Fresh link.** A restore that publishes `personal_context.repository` or `runtime.sync_state` also forces a fresh reviewed first link. On a new machine, that link registers a new device with a new wrapping key.
  - This includes an automation-only restore, which publishes only `runtime.sync_state` and can rewind `personal_context_link_state` under a live repository.
- Mechanism:
  - The new capture adapter is a `_SQLiteDeclaration` subclass, so it is `activation_required` by default (F14).
  - Every restore binds all activation-required owners (`publication.py:592-604`), so the gate fires whichever groups were restored.
  - `_load_personal_context_sync_runtime` (`app.py:2625`) checks `execution_allowed` for `personal_context.repository` and `runtime.sync_state`.
  - For a restore that published either owner, the Personal Context review clears that server profile's `personal_context_link_state` before approving, so the next **Link to home server** starts fresh.
  - The fresh link rebinds Personal Context's device without changing Notes and Chat sync state (device id, dataset cursors, outbox) for that profile. Note that `_seed_sync_profile` carries `dataset_cursors` forward (`link_service.py:1029`).
  - What Notes and Chat sync do after a cross-machine restore is investigated in TASK-40137.
- The restored records stay readable locally. The integrity key restored with them keeps them authenticated, and the next link rebaselines it (F5).

**Option B: capture and restore the link keys.**
- The restored device would hold the old `device_id` (already restored by `runtime.sync_state`, F8), the old wrapping key and the old cursors. It *is* the old device, so this violates constraint 1. It also splits the device in two when the old device is still running.

**Option C: re-wrap link keys to a new device during restore.**
- This needs a server-side device-migration protocol, which does not exist.

**Recommendation: A.** Two further points:
- **Same machine.** The wrapping key is still in the keyring, but the restored cursors and outbox are stale, so a fresh link still applies. First-link reconciliation is the reviewed tool for converging two lineages.
- **Old device registration.** It stays on the server, and Chatbook cannot unregister it (User Guide :335). The copy says so: "Personal Context sync is off because this profile was restored. Link to home server again. If the old device is lost, remove it on your server."
- **Server-trusted sync state.** It is in `runtime.sync_state` and is captured as it is today. The link reset above makes it inert for Personal Context.

## Q4. Rollback and replacement: a restored database whose key no longer matches

The code facts are F5, F11 and F13. The same path can hold a different key than the archive's: after a link, after Remove local profile and a new profile, or on another machine. The database being replaced can be read only with the key now in the keyring.

**Option A: a fresh account and an explicit key reference.** (Recommended.)
- This is Q2's design. Restore never touches the target's existing entry, because it always installs into a fresh account:
  - The original database stays readable with its own key.
  - The held original and the rollback copy keep working.
  - A later rollback puts back the original database, whose reference still names its untouched account.
  - Linking or removing the restored profile changes only that profile's own fresh account. A rollback after either still opens the original records.
- The rollback copy also escrows the live key, because the key is a credential record in `rollback` mode (F13). That covers a keyring that is lost between restore and rollback.
- Journaled steps keep `recover --finish|--rollback` idempotent:
  - "key installed into account X" is journaled before publication;
  - the restore deletes only an account that it created itself.
- This follows ADR-126 decision 8 as written.

**Option B: overwrite the path-keyed entry after escrowing it in the verified rollback copy.**
- This needs no schema change.
- It needs an ADR-126 amendment, because it overwrites a keyring entry.
- A later rollback must re-install the old key, and a crash between the two steps leaves the original unreadable until `recover` runs.

**Option C: refuse replace-restore while a live key exists.**
- The user's only way forward would be Remove local profile. That deletes the key the held original needs, which is a data-loss trap.

**Recommendation: A.** A key that does not match is detected by authenticating the manifest head with the loaded key, never by comparing `key_version`, because unrelated keys usually share version 1 (F5). It gets the `personal_context_key_mismatch` copy in Q2.

## Q5. Should the drafts exclusion join the detached-layout fixed owner list?

The code facts are F16 and `config_adapter.py:693-706`. An excluded `_LocalSQLite` yields `intentionally_excluded` rows. The drafts key is keyring-only (`interview_launch.py:101`), and the drafts are local-only by policy (User Guide :350-351).

**Option A: join, as excluded.** (Recommended.)
- Add `personal_context.interview_drafts` to the fixed list.
- Add its four paths to `known`, as `notes.sync_state` does at `:1046-1049`.
- Add one review line: "Personal Context interview drafts stay on the original device."
- Without this, every portable import from a profile that ever started an interview is refused.

**Option B: keep refusing.**
- Every such portable import stays blocked by a cause the user cannot see.

**Option C: import the draft bytes.**
- This contradicts the local-only policy, and the bytes are unreadable without their keyring key anyway.

**Recommendation: A.** For the repository in a detached source, refuse by name rather than as unknown content:
- Add it to the same list. Until it has a key-carrying path, its row is `unsupported`, so the refusal reads `detached_owner_unavailable:personal_context.repository`.
- Detached import is credential-excluded, and the relocation host has a null keyring (F12, F16), so the key cannot come along.
- Copy: "Personal Context can't be moved this way because its key stays on the original device. Make an encrypted backup with credentials there, then restore it with Replace."

## Q6. Agreement with TASK-39116

TASK-39116 AC#1 asks for each package's cohort, "including whether Personal Context key material travels with its database". If approved, this spec answers that question for Personal Context. TASK-39116's decision records should then say:

- `personal_context.repository`: **capture** (this spec). Key material travels as an encrypted credential record (Q1), never in a readable archive.
- `personal_context.interview_drafts`: **exclude** (already installed, TASK-39110). It is also excluded in detached sources (Q5).
- The keyring writers of the profile key (`key_protector.py`): covered by the credential record. They are not a file owner.
- The link-key keyring writers (`link_key_custody.py`): **excluded device-bound credentials**, disclosed like skill-trust keys (`credentials.py:1052-1065`).
- The passphrase-bundle writer: no production caller (F3). It **stays `unsupported`** until a caller ships.
- `export_service` writers: user-chosen export destinations, not a profile store. The coordinator classifies them in TASK-39116's census vocabulary.

## Consistency note: the Plugins generation marker (TASK-40132)

TASK-40132 (owner 2026-11-10, "Capture all") restores Plugins byte-for-byte, then shows a documented missing-keyring-marker state with copy. That marker (`Plugins/authority_store.py:135-190`) is an **anti-rollback witness**, not a decryption key.

**What does not carry over:** the credential-record mechanism must not apply to it. A marker restored from an archive would bless the rolled-back store it exists to detect.

**What does carry over:**
- Restore never overwrites or deletes a keyring entry it did not create (ADR-126:181).
- A keyring account derived from a location does not follow a restore to a new path or into an isolated (null-keyring) profile.
- A missing entry yields a named locked or recovery-required state with copy, never a silent re-bootstrap. This mirrors ADR-102:210-211.

This spec does not design Plugins.

## Out of scope and proposed follow-ups

- **Re-key after restore** (Q2 option B): filed as TASK-40140 under decision 4.
- **`runtime.sync_state` device binding after restore.**
  - This is Notes/Chat Sync, not Personal Context.
  - Server Sync V2 calls are already gated after a restore. `_sync_call` holds `execution_scope(("config", "runtime.sync_state"), …)` and raises `sync_activation_required` (`Sync_Interop/server_sync_service.py:51-95`, `:85-88`), and `local_first_sync_service.py` uses the same decorator.
  - Not verified here:
    - what approves `config` and `runtime.sync_state` for a restored generation (no approver naming them was found);
    - whether that approval then lets Notes and Chat sync run with the restored shared `sync_profile_state.device_id` (F8), which carries the old device identity;
    - whether any Notes or Chat path bypasses `_sync_call`.
  - Whether general Sync can resume as the old device after a cross-machine restore with re-entered server credentials is **unverified** here. It is filed as TASK-40137, which also rules who may approve `config` and `runtime.sync_state` after a restore.
  - A fresh Personal Context link on a new machine will hit `personal_context_sync_profile_binding_conflict` (`link_service.py:1024-1028`) while that row holds the old device. TASK-40139 rebinds Personal Context and leaves Notes and Chat sync state unchanged.
- **Personal Context recovery-export import** (F17): still unshipped. It is unrelated to backup.
- **ADR records:** written on approval as dated notes on ADR-102 and ADR-126.
  - The ADR-102 note gives decision 8's reading of "never leave Chatbook", with ADR-203's qualifier. ADR-203 is Accepted. It sets the ceiling at Sync and automatic remote model egress, and it distinguishes qualified local model execution and separately authorized owner exports from that ceiling.
  - The ADR-126 note says restore never overwrites a keyring entry, and that the stored key reference keeps that rule workable.

## Decisions

**Status: Approved.** Owner decision 2026-11-10 (AskUserQuestion): "Approve all 11". The option text was: "Record approval of the 11 recommendations as written and file the implementation tasks: key reference (schema v9), capture, readable restore, restore without a key, no link resume, detached-layout treatment."

1. Carry the Personal Context profile key as a credential record inside the encrypted archive (only when a backup includes credentials, which already forces encryption), rather than a separate Personal Context passphrase or no key at all.
2. Give the repository a stored key reference (schema v9 `profile_meta.key_account`), so restore installs the key into a fresh keyring account and never overwrites an existing one (ADR-126 decision 8), rather than overwriting the path-keyed entry after escrowing it (which needs an ADR-126 amendment) or refusing replace-restore.
3. When a backup does not include credentials, capture the database without its key and say so in the preview and restore plan, rather than refusing Personal Context capture.
4. Install the restored key as it is and file re-encryption after restore as a separate follow-up, rather than re-encrypting every record during restore.
5. Never capture the device wrapping key, staged integrity keys or dataset staging keys, and never resume a restored link; Personal Context sync stays off until a fresh reviewed first link, rather than restoring or re-wrapping link keys.
6. Keep Personal Context locked in isolated restored profiles (they run with a null keyring), with its key retained in the encrypted restore copy, rather than giving isolated profiles keyring access.
7. Refuse Personal Context capture while a first-link freeze or rebaseline marker exists, rather than capturing mid-link state.
8. Include device-only records in the backup, as Export recovery copy already does, and record that ADR-102's "never leave Chatbook" means never synced or sent to a model, rather than filtering them out of the capture.
9. Put the repository in a new selectable "Personal Context" backup group, rather than in the existing "Settings" group.
10. Add the interview drafts to the detached-layout fixed owner list as excluded, and refuse the repository there by name, rather than refusing both as unknown content.
11. Leave `PassphraseProfileKeyProtector` unshipped and out of this work (it has no production caller), rather than adopting it as the backup key carrier.

## Filed implementation tasks

Filed 2026-11-10 after approval. Each delivers one testable outcome. Two follow-ups are listed after them. Dependencies, not list order, set the landing order; TASK-40139 waits for the TASK-40137 follow-up's ruling.

1. **TASK-40133, "Personal Context's repository names its key's keyring account."**
   - Outcome: the migration adds an empty reference, and the repository writes the path-derived account the first time that account's key authenticates the database; a locked one stays unstamped. The repository uses a key only after it authenticates the database, and opens from the stamped account even after the file moves. New and re-created profiles get a new, unused account. Link and Remove local profile act only on the authenticated account.
   - Includes the exact-SQL catalog for every reachable v9 layout, since in-place `ALTER TABLE` migrations make catalogs differ.
2. **TASK-40134, "A backup captures Personal Context's repository database."** (AC#3: capture.)
   - Outcome: a preview of a profile with Personal Context is complete.
   - The archive holds the lock-safe, schema-validated database in the "Personal Context" group.
   - An archive that includes credentials holds the key as a credential record. One that does not keeps the reference unchanged and says its key is not included.
   - Until readable restores ship, no restore publishes the repository. The live database is left unchanged, the archive keeps its copy, and the plan says Personal Context was not restored. This keeps the task safe on its own.
   - Capture refuses while a first link is in flight.
   - As an interim guard, Personal Context sync does not resume after any restore until the restored generation is reviewed.
3. **TASK-40135, "A replace restore makes restored Personal Context records readable."** (AC#3: a restore that reads its records.)
   - Outcome: replace restores now publish the repository. After one onto the same or a different machine or path, Personal Context opens and lists the restored records.
   - When no key authenticates, the database is published with the marker `recovery:setup_required:<reason>`, and every table except `profile_meta.key_account` is unchanged. The plan says before the restore runs that Personal Context will end locked.
   - When a key authenticates, it goes into a fresh account. The restored database names an account no other database names, and any pre-existing keyring entry is unchanged.
   - A same-device replace restore from a backup without credentials reads its records. That backup's preview says it is readable only on this device, and only until Personal Context is linked or removed.
   - A later rollback opens the original records, even after the restored profile has linked or been removed.
4. **TASK-40136, "A restore without a usable Personal Context key keeps the data and says what to do."** (AC#3: key unavailable.)
   - Covers four cases: a key-less archive on another machine, an isolated profile, no secure keyring, and a key mismatch.
   - Isolated restores now publish the repository, with the marker (decision 6).
   - Outcome: no keyring entry is created. Every table except `key_account` is unchanged, `key_account` carries the marker with its reason, and Personal Context is locked with that reason, including after a restart.
   - The restore plan and the Settings panel show the Q2 copy, and the User Guide is updated.
5. **TASK-40139, "A restored Personal Context does not resume its old link."** It depends on TASK-40137's ruling.
   - Outcome: after a restore that publishes the repository or `runtime.sync_state` (including an automation-only one), Personal Context sync stays off until a fresh reviewed first link. Its review approves only `personal_context.repository`, so Notes and Chat sync stay refused. The old link state for that server profile is cleared. The rebind leaves Notes and Chat sync state unchanged, and no link key is ever in an archive.
6. **TASK-40138, "A detached source holding Personal Context files is treated by name."** (AC#3: detached treatment of the drafts file.)
   - Outcome: drafts are an excluded row with a review line, instead of `detached_unknown_content`. The repository is refused as `detached_owner_unavailable:personal_context.repository`.

Follow-ups:
- **TASK-40137:** investigate whether a restored `runtime.sync_state` lets Notes and Chat sync resume as the old device after a cross-machine restore (security). It rules who may approve `config` and `runtime.sync_state` after a restore, which TASK-40139 needs.
- **TASK-40140:** re-encrypt the Personal Context profile after a restore (decision 4).

## Not verified

- **Server behaviour on a fresh link after restore.** This covers whether a new `device_id` is issued for a reused wrapping key, and whether the integrity key is re-issued. tldw_server was not read.
- **Inert extraction of `credential-recovery.json`.** It looks impossible, since credentials have no group (`group_selection.py:96`), but no test pins it. TASK-40134 must pin it.
- **The set of v8 catalog variants** produced by the repository's in-place migrations (`repository.py:636-733`). These were not enumerated.
- **General Sync impersonation after a restore** (TASK-40137).
