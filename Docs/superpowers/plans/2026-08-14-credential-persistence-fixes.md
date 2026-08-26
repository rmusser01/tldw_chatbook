# Credential/URL Persistence Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make user-entered server credentials/URLs actually persist — no false "saved" reports, eager OS-keyring writes with surfaced failures, no silent plaintext downgrade, and removal of stale artifacts that confuse diagnosis.

**Architecture:** Four independent fixes on the existing save path: (1) correct the boolean contract of `save_settings_to_cli_config`, (2) make encryption failures hard failures instead of plaintext fallback, (3) add an eager `store_static_server_credential` on `RuntimeServerContextProvider` and call it from the server-switch save flow, (4) delete the stale `Widgets/Media_Ingest/` pycache-only directory. The lazy legacy keyring import stays as a fallback but logs instead of swallowing.

**Tech Stack:** Python 3.11+, pytest with `monkeypatch`/`tmp_path`, stdlib `toml`/`tomllib`, keyring (via existing `runtime_policy/server_credentials.py`).

**Spec:** `backlog/tasks/task-16310 - Fix-credential-URL-persistence-silent-keyring-gap-false-success-config-saves-and-encryption-fallbacks.md`

ADR required: no
ADR path: N/A
Reason: Bug fixes within already-designed boundaries. The keyring credential store, atomic config mutation, and encryption-at-rest designs already exist (see `runtime_policy/server_credentials.py`, `config.py` mutation machinery); this plan corrects implementations and adds one public method on an existing class. No new storage medium, sync policy, or cross-module contract.

## Global Constraints

- Never log secret values — log exception categories and key names only (existing convention: `logger.warning("... (exception_category=%s).", type(exc).__name__)`).
- All config-file tests use the `TLDW_CONFIG_PATH` env var pointed at a `tmp_path` file via `monkeypatch.setenv` (pattern in `Tests/test_config_delete_settings.py`).
- Config writes must go through the existing atomic mutation machinery — no direct file writes in new code.
- Keyring writes go through `ServerCredentialStore.set_secret`; never call `keyring` directly.
- Commit message style: `fix(<area>): <summary>` (matches recent history, e.g. `fix(notes): ...`).
- Run the full relevant test dirs before each commit: `pytest Tests/test_config_save_settings_semantics.py Tests/test_config_encryption_save_behavior.py Tests/RuntimePolicy/ Tests/UI/test_settings_runtime_source_switch.py -x -q` (only the files that exist at that point in the sequence).

---

### Task 1: `save_settings_to_cli_config` must report refused mutations as failure

**Files:**
- Modify: `tldw_chatbook/config.py:5464-5476` (`save_settings_to_cli_config`)
- Test: `Tests/test_config_save_settings_semantics.py` (create)

**Interfaces:**
- Consumes: `ConfigMutationResult(file_replaced, caches_reloaded, failure_phase, conflict, conflict_reason)` from `config.py:5070`.
- Produces: unchanged public signature `save_settings_to_cli_config(section_values, *, delete_keys=None) -> bool`. New contract: `True` only for fully-applied **or** genuine no-op; `False` for conflict, `before_replace`, and `cache_reload` failures.

Context: today a conflict result (`ConfigMutationResult(False, False, None, conflict=True)`) falls into the `failure_phase is None and not file_replaced` branch and returns `True`, so `_perform_runtime_source_switch` (`UI/Screens/settings_screen.py:16348`) tells the user "saved" while nothing hit disk. The no-op return of `True` must be preserved: `UI/Dictation_Window_Improved.py:902` saves snapshots that can be empty.

- [ ] **Step 1: Write the failing tests**

Create `Tests/test_config_save_settings_semantics.py`:

```python
"""Boolean contract of config_module.save_settings_to_cli_config."""

from tldw_chatbook import config as config_module
from tldw_chatbook.config import ConfigMutationResult


def _patch_result(monkeypatch, result: ConfigMutationResult) -> None:
    monkeypatch.setattr(
        config_module,
        "apply_settings_mutation_to_cli_config",
        lambda *args, **kwargs: result,
    )


def test_identity_conflict_reports_failure(monkeypatch):
    _patch_result(
        monkeypatch,
        ConfigMutationResult(
            False, False, None, conflict=True, conflict_reason="identity_changed"
        ),
    )
    assert config_module.save_settings_to_cli_config({"tldw_api": {"base_url": "x"}}) is False


def test_fully_applied_reports_success(monkeypatch):
    _patch_result(monkeypatch, ConfigMutationResult(True, True, None))
    assert config_module.save_settings_to_cli_config({"tldw_api": {"base_url": "x"}}) is True


def test_noop_reports_success(monkeypatch):
    _patch_result(monkeypatch, ConfigMutationResult(False, False, None))
    assert config_module.save_settings_to_cli_config({}) is True


def test_before_replace_failure_reports_failure(monkeypatch):
    _patch_result(monkeypatch, ConfigMutationResult(False, False, "before_replace"))
    assert config_module.save_settings_to_cli_config({"tldw_api": {"base_url": "x"}}) is False


def test_cache_reload_failure_reports_failure(monkeypatch):
    _patch_result(monkeypatch, ConfigMutationResult(True, False, "cache_reload"))
    assert config_module.save_settings_to_cli_config({"tldw_api": {"base_url": "x"}}) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/test_config_save_settings_semantics.py -q`
Expected: `test_identity_conflict_reports_failure` FAILS (returns `True`), the others PASS.

- [ ] **Step 3: Implement the fix**

Replace the body of `save_settings_to_cli_config` in `tldw_chatbook/config.py:5464-5476`:

```python
def save_settings_to_cli_config(
    section_values: Mapping[str, Mapping[Any, Any]],
    *,
    delete_keys: Mapping[str, Collection[str]] | None = None,
) -> bool:
    """Persist multiple config values with one atomic mutation and cache reload."""
    result = apply_settings_mutation_to_cli_config(
        section_values,
        delete_keys=delete_keys,
    )
    if result.conflict:
        return False
    if result.failure_phase is None and not result.file_replaced:
        return True
    return result.fully_applied
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Tests/test_config_save_settings_semantics.py -q`
Expected: 5 passed.

- [ ] **Step 5: Run the callers' existing tests for regressions**

Run: `pytest Tests/test_config_delete_settings.py Tests/UI/test_dictation_settings_debounce.py Tests/UI/test_profile_owned_settings_paths.py -q`
Expected: all pass (no caller relied on conflict reporting `True`).

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/config.py Tests/test_config_save_settings_semantics.py
git commit -m "fix(config): report refused config mutations as save failures"
```

---

### Task 2: Encryption failures must not silently persist plaintext secrets

**Files:**
- Modify: `tldw_chatbook/config.py:4520-4545` (`_maybe_encrypt_setting_value`)
- Modify: `tldw_chatbook/config.py:5418-5429` (value-mutation loop inside `apply_settings_mutation_to_cli_config`)
- Test: `Tests/test_config_encryption_save_behavior.py` (create)

**Interfaces:**
- Consumes: `get_encryption_password()` and `get_encryption_module()` (module-level names in `config.py`), `is_sensitive_config_key` from `Utils/sensitive_config_keys.py` (`auth_token` is in `SENSITIVE_CONFIG_EXACT_KEYS`).
- Produces: no signature changes. New behavior: when `[encryption] enabled` and a sensitive value cannot be encrypted, the mutation fails (`before_replace`) instead of writing plaintext.

Context: `_maybe_encrypt_setting_value`'s `except` branch (`config.py:4543-4545`) returns the plaintext value on encryption failure, and the value-mutation loop (`config.py:5418-5426`) has no `try` — but the later `_config_data_for_persistence` call is already inside one. Making the helper raise and wrapping the loop turns the silent downgrade into a hard, logged failure. The "encryption enabled but no password" case already fails today via `_config_data_for_persistence` (`config.py:5026-5029`); we lock that in with a test.

- [ ] **Step 1: Write the failing tests**

Create `Tests/test_config_encryption_save_behavior.py`:

```python
"""Config saves must fail hard rather than persist plaintext secrets."""

import tomllib

import toml

from tldw_chatbook import config as config_module


class _BrokenEncryptionModule:
    def encrypt_value(self, value: str, password: str) -> str:
        raise RuntimeError("encryption backend exploded")


def _write_config(config_path, data: dict) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(toml.dumps(data), encoding="utf-8")


def test_encrypt_failure_blocks_save_and_leaves_file_unchanged(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    _write_config(config_path, {"chat_defaults": {"streaming": True}})
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(config_module, "get_encryption_password", lambda: "pw")
    monkeypatch.setattr(
        config_module, "get_encryption_module", lambda: _BrokenEncryptionModule()
    )

    saved = config_module.save_settings_to_cli_config(
        {"tldw_api": {"base_url": "https://s.example.com", "auth_token": "secret-1"}}
    )

    assert saved is False
    on_disk = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert "tldw_api" not in on_disk
    assert on_disk["chat_defaults"] == {"streaming": True}


def test_locked_encryption_blocks_plaintext_secret_save(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    _write_config(
        config_path,
        {"encryption": {"enabled": True}, "chat_defaults": {"streaming": True}},
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(config_module, "get_encryption_password", lambda: None)

    saved = config_module.save_settings_to_cli_config(
        {"tldw_api": {"base_url": "https://s.example.com", "auth_token": "secret-2"}}
    )

    assert saved is False
    on_disk = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert "secret-2" not in config_path.read_text(encoding="utf-8")
    assert on_disk["chat_defaults"] == {"streaming": True}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/test_config_encryption_save_behavior.py -q`
Expected: `test_encrypt_failure_blocks_save_and_leaves_file_unchanged` FAILS (returns `True`, plaintext written); `test_locked_encryption_blocks_plaintext_secret_save` may already PASS (locks in existing behavior).

- [ ] **Step 3: Implement the fix**

3a. In `tldw_chatbook/config.py:4520-4545`, change the `except` branch of `_maybe_encrypt_setting_value` from returning the plaintext value to raising:

```python
    try:
        enc_module = get_encryption_module()
        encrypted_value = enc_module.encrypt_value(value, password)
        logger.info(f"Encrypted {key} in config section")
        return encrypted_value
    except Exception as e:
        logger.error(f"Failed to encrypt value for key {key}: {e}")
        raise
```

3b. In `apply_settings_mutation_to_cli_config`, wrap the value-mutation loop (`config.py:5418-5429`) so the raise becomes a typed mutation failure:

```python
        try:
            deleted_any = _delete_config_keys(config_data, requested_deletes)
            for section, values in section_values.items():
                if not values:
                    continue
                current_level = _target_config_section(config_data, section)
                for key, value in values.items():
                    current_level[key] = _maybe_encrypt_setting_value(
                        config_data, key, value
                    )
        except Exception as error:
            logger.error(
                "Configuration mutation failed "
                "(phase=before_replace, config_path={}, error_type={}).",
                config_path,
                type(error).__name__,
            )
            return ConfigMutationResult(False, False, "before_replace")
        set_any = any(bool(values) for values in section_values.values())
        if not set_any and not deleted_any:
            return ConfigMutationResult(False, False, None)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Tests/test_config_encryption_save_behavior.py -q`
Expected: 2 passed.

- [ ] **Step 5: Run config test suite for regressions**

Run: `pytest Tests/test_config_delete_settings.py Tests/test_config_save_settings_semantics.py Tests/test_config_private_bootstrap.py Tests/test_config_console_defaults.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/config.py Tests/test_config_encryption_save_behavior.py
git commit -m "fix(config): fail config saves instead of silently storing plaintext secrets"
```

---

### Task 3: Eager credential-store write on `RuntimeServerContextProvider`

**Files:**
- Modify: `tldw_chatbook/runtime_policy/server_context.py` (add `store_static_server_credential` near `store_auth_tokens` at `:421`; fix `_import_legacy_token` at `:741-753`)
- Test: `Tests/RuntimePolicy/test_server_context_provider.py` (append)

**Interfaces:**
- Consumes: `self.credential_store.set_secret(server_id, purpose, secret)`; `self._purposes_for_auth_mode(auth_mode)` (`server_context.py:853` — returns `(SERVER_CREDENTIAL_BEARER_TOKEN, SERVER_CREDENTIAL_ACCESS_TOKEN)` for `"bearer"`/`"custom_token"`, `(SERVER_CREDENTIAL_API_KEY,)` for `"api_key"`); `self.target_store.get_target(server_id)`; `self._legacy_cleared_server_ids` (in-memory sign-out marker set); `self._invalidate_cached_client()`.
- Produces: `RuntimeServerContextProvider.store_static_server_credential(server_id: str, secret: str, *, auth_mode: str | None = None) -> str` — persists the modal-entered token under the first purpose for the server's auth mode, discards the sign-out marker for that server, invalidates the cached client, and returns the purpose written. Raises whatever the credential store raises (`CredentialStoreUnavailable` for unavailable stores) and `ValueError` for empty `server_id`/`secret`.

Context: the modal token currently only lands in plaintext config.toml; the keyring gets it lazily via `_import_legacy_token` (`server_context.py:741`), which swallows every exception. After a sign-out (`clear_server_credentials` → `_mark_legacy_server_id_cleared`), the cleared marker blocks the lazy import entirely, so a re-entered token never resolves — this task fixes both by making the save-time write authoritative.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/RuntimePolicy/test_server_context_provider.py` (the file's `_provider` helper at line 127 and `InMemoryServerCredentialStore` are already available there; also import `UnavailableServerCredentialStore` is already present at line 24, and `SERVER_CREDENTIAL_BEARER_TOKEN` — verify its import at the top of the file and add if missing):

```python
def test_store_static_server_credential_writes_first_purpose_and_clears_signout(
    tmp_path,
):
    store = InMemoryServerCredentialStore()
    provider = _provider(
        tmp_path,
        credential_store=store,
        targets=[
            ConfiguredServerTarget(
                server_id="https://server.example.com/api",
                label="Primary",
                base_url="https://server.example.com/api",
                auth_mode="bearer",
                is_default=True,
            )
        ],
        app_config={
            "tldw_api": {
                "base_url": "https://server.example.com/api",
                "bearer_token": "legacy-bearer",
                "auth_mode": "bearer",
            }
        },
    )
    provider.clear_server_credentials("https://server.example.com/api")

    purpose = provider.store_static_server_credential(
        "https://server.example.com/api", "re-entered-token"
    )

    assert purpose == SERVER_CREDENTIAL_BEARER_TOKEN
    assert (
        store.get_secret(
            "https://server.example.com/api", SERVER_CREDENTIAL_BEARER_TOKEN
        )
        == "re-entered-token"
    )


def test_re_entered_token_resolves_after_signout_without_legacy_config(tmp_path):
    store = InMemoryServerCredentialStore()
    provider = _provider(
        tmp_path,
        credential_store=store,
        targets=[
            ConfiguredServerTarget(
                server_id="https://server.example.com/api",
                label="Primary",
                base_url="https://server.example.com/api",
                auth_mode="bearer",
                is_default=True,
            )
        ],
        app_config={
            "tldw_api": {
                "base_url": "https://server.example.com/api",
                "bearer_token": "",
                "auth_mode": "bearer",
            }
        },
    )
    server_id = "https://server.example.com/api"
    provider.clear_server_credentials(server_id)

    provider.store_static_server_credential(server_id, "fresh-token")

    context = provider.get_active_context()
    assert context.auth_token == "fresh-token"
    assert context.credential_source == (
        f"credential_store:{SERVER_CREDENTIAL_BEARER_TOKEN}"
    )


def test_store_static_server_credential_raises_for_unavailable_store(tmp_path):
    provider = _provider(
        tmp_path,
        credential_store=UnavailableServerCredentialStore("no secure store"),
    )

    with pytest.raises(CredentialStoreUnavailable):
        provider.store_static_server_credential(
            "https://server.example.com/api", "token"
        )


def test_store_static_server_credential_rejects_empty_arguments(tmp_path):
    provider = _provider(tmp_path)

    with pytest.raises(ValueError):
        provider.store_static_server_credential("", "token")
    with pytest.raises(ValueError):
        provider.store_static_server_credential("https://server.example.com/api", "  ")
```

If `CredentialStoreUnavailable` is not already imported at the top of the test file, add it to the existing `from tldw_chatbook.runtime_policy.server_credentials import ...` block.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/RuntimePolicy/test_server_context_provider.py -q -k store_static_server_credential`
Expected: FAIL with `AttributeError: ... has no attribute 'store_static_server_credential'` (and the signout test fails with `ServerCredentialsUnavailable: profile_no_longer_authorized`).

- [ ] **Step 3: Implement the method**

In `tldw_chatbook/runtime_policy/server_context.py`, add after `store_auth_tokens` (after line 442):

```python
    def store_static_server_credential(
        self,
        server_id: str,
        secret: str,
        *,
        auth_mode: str | None = None,
    ) -> str:
        """Persist a user-entered static token to the credential store eagerly.

        Unlike the lazy legacy-config import, this write is authoritative for
        the server profile: it also clears any sign-out marker so a re-entered
        token resolves immediately without the legacy config fallback.

        Args:
            server_id: Server profile the secret belongs to.
            secret: User-entered token value; must be non-empty.
            auth_mode: Optional auth mode override; resolved from the
                configured target when omitted.

        Returns:
            The credential purpose the secret was stored under.

        Raises:
            ValueError: If ``server_id`` or ``secret`` is empty.
            CredentialStoreUnavailable: If no secure credential store exists.
        """
        normalized_server_id = str(server_id or "").strip()
        normalized_secret = str(secret or "").strip()
        if not normalized_server_id or not normalized_secret:
            raise ValueError("server_id and secret must be non-empty")

        resolved_auth_mode = auth_mode
        if resolved_auth_mode is None:
            target = self.target_store.get_target(normalized_server_id)
            resolved_auth_mode = str(getattr(target, "auth_mode", "") or "")
        purposes = self._purposes_for_auth_mode(resolved_auth_mode) or (
            SERVER_CREDENTIAL_BEARER_TOKEN,
            SERVER_CREDENTIAL_ACCESS_TOKEN,
        )
        purpose = purposes[0]
        self.credential_store.set_secret(normalized_server_id, purpose, normalized_secret)
        self._legacy_cleared_server_ids.discard(normalized_server_id)
        self._invalidate_cached_client()
        return purpose
```

- [ ] **Step 4: Fix the silent exception swallow in `_import_legacy_token`**

Replace `server_context.py:749-753` body:

```python
        purpose = purposes[0]
        try:
            self.credential_store.set_secret(server_id, purpose, token)
        except CredentialStoreUnavailable as exc:
            logger.warning(
                "Legacy token keyring import skipped; credential store "
                "unavailable (reason_code={}).",
                exc.reason_code,
            )
            return None
        except Exception as exc:
            logger.warning(
                "Legacy token keyring import failed "
                "(purpose={}, exception_category={}).",
                purpose,
                type(exc).__name__,
            )
            return None
        return purpose
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest Tests/RuntimePolicy/test_server_context_provider.py -q`
Expected: all pass, including pre-existing legacy-import tests.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/runtime_policy/server_context.py Tests/RuntimePolicy/test_server_context_provider.py
git commit -m "fix(runtime-policy): eager server credential store write with surfaced failures"
```

---

### Task 4: Wire the eager write into the server-switch save flow and surface unavailable keyrings

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py:16385-16391` (inside `_perform_runtime_source_switch`)
- Modify: `tldw_chatbook/app.py:5814-5828` (`_wire_server_context_provider`)
- Test: `Tests/UI/test_settings_runtime_source_switch.py` (create)

**Interfaces:**
- Consumes: `app.server_context_provider` (wired at `app.py:5823`), `provider.store_static_server_credential(server_id, token)` from Task 3, `app.server_credential_store_unavailable_reason` (new attribute set in this task).
- Produces: no new consumed interfaces; user-visible behavior only.

Context: `_perform_runtime_source_switch` saves the token to config.toml and activates the server, but never touches the keyring. `_wire_server_context_provider` runs during `basic_init` (before the UI can receive notifications), so startup surfacing is a warning log plus a stored attribute; the user-facing notification happens at save time.

- [ ] **Step 1: Write the failing test**

Create `Tests/UI/test_settings_runtime_source_switch.py`. The method is driven unbound with a fake `self` so no Textual app mount is needed:

```python
"""Server-switch save flow writes the token to the credential store eagerly."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen


def _fake_self(app, server_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        app_instance=app,
        app=app,
        _runtime_source_state=lambda: SimpleNamespace(active_server_id=server_id),
        _refresh_manual_sync_rows=lambda: None,
    )


def _fake_app(*, switched: bool = True) -> SimpleNamespace:
    app = SimpleNamespace()

    async def _switch(*args, **kwargs):
        return switched

    app.handle_runtime_backend_changed = MagicMock(side_effect=_switch)
    notified: list[tuple[str, str]] = []
    app.notify = lambda message, severity="information": notified.append(
        (severity, message)
    )
    provider = MagicMock()
    provider.store_static_server_credential = MagicMock(return_value="bearer_token")
    app.server_context_provider = provider
    app.sync_scope_service = None
    return app, notified, provider


def test_switch_persists_token_to_credential_store(monkeypatch, tmp_path):
    app, _notified, provider = _fake_app()
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "config.toml"))
    fake = _fake_self(app, "https://server.example.com/api")

    asyncio.run(
        SettingsScreen._perform_runtime_source_switch(
            fake,
            {
                "action": "activate",
                "base_url": "https://server.example.com/api",
                "auth_token": "tok-123",
            },
        )
    )

    provider.store_static_server_credential.assert_called_once_with(
        "https://server.example.com/api", "tok-123"
    )


def test_switch_notifies_when_keyring_write_fails(monkeypatch, tmp_path):
    app, notified, provider = _fake_app()
    provider.store_static_server_credential.side_effect = RuntimeError("boom")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "config.toml"))
    fake = _fake_self(app, "https://server.example.com/api")

    asyncio.run(
        SettingsScreen._perform_runtime_source_switch(
            fake,
            {
                "action": "activate",
                "base_url": "https://server.example.com/api",
                "auth_token": "tok-123",
            },
        )
    )

    assert any(severity == "warning" and "keyring" in message for severity, message in notified)


def test_switch_without_token_skips_credential_store(monkeypatch, tmp_path):
    app, _notified, provider = _fake_app()
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "config.toml"))
    fake = _fake_self(app, "https://server.example.com/api")

    asyncio.run(
        SettingsScreen._perform_runtime_source_switch(
            fake,
            {
                "action": "activate",
                "base_url": "https://server.example.com/api",
                "auth_token": "",
            },
        )
    )

    provider.store_static_server_credential.assert_not_called()
```

Note: `load_settings(force_reload=True)` inside the method runs against the temp config path — acceptable (it reads/creates the temp file). If `SettingsScreen`'s import side effects make this test slow or flaky in CI, mark it `@pytest.mark.ui` consistent with other `Tests/UI` files that import the settings screen; check how sibling files (e.g. `Tests/UI/test_settings_configuration_hub.py`) import and mirror their import guards.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest Tests/UI/test_settings_runtime_source_switch.py -q`
Expected: `test_switch_persists_token_to_credential_store` FAILS (`store_static_server_credential` never called); the notify test FAILS (no warning); the no-token test may already PASS.

- [ ] **Step 3: Implement the settings-screen wiring**

In `_perform_runtime_source_switch` (`settings_screen.py`), immediately after the `server_id` empty-guard block ending at line 16391 (`return`), insert:

```python
        if auth_token:
            provider = getattr(app, "server_context_provider", None)
            if provider is not None:
                try:
                    provider.store_static_server_credential(server_id, auth_token)
                except Exception as exc:
                    logger.warning(
                        "Server token could not be stored in the OS credential "
                        "store (exception_category=%s).",
                        type(exc).__name__,
                    )
                    self.app.notify(
                        "Server activated, but the token could not be saved to "
                        "the OS keyring; it remains saved in config.toml only.",
                        severity="warning",
                    )
```

- [ ] **Step 4: Implement the app-level unavailability surfacing**

In `app.py:_wire_server_context_provider` (lines 5819-5822), replace:

```python
        try:
            self.server_credential_store = build_default_server_credential_store()
        except CredentialStoreUnavailable as exc:
            self.server_credential_store = UnavailableServerCredentialStore(str(exc))
```

with:

```python
        try:
            self.server_credential_store = build_default_server_credential_store()
            self.server_credential_store_unavailable_reason = None
        except CredentialStoreUnavailable as exc:
            self.server_credential_store = UnavailableServerCredentialStore(str(exc))
            self.server_credential_store_unavailable_reason = str(exc)
            logger.warning(
                "No secure OS credential store available; server tokens will "
                "remain config-only (reason={}).",
                str(exc),
            )
```

Also add the attribute's default next to the other instance defaults: in `__init__` near where `server_credential_store` is initialized (search for `self.server_credential_store =` first assignment; if it is only assigned in `_wire_server_context_provider`, add `self.server_credential_store_unavailable_reason: str | None = None` in `__init__` alongside `self._startup_phases` initialization).

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest Tests/UI/test_settings_runtime_source_switch.py Tests/RuntimePolicy/ -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/app.py Tests/UI/test_settings_runtime_source_switch.py
git commit -m "fix(settings): write server token to OS keyring at save time and surface failures"
```

---

### Task 5: Remove stale `Media_Ingest` pycache directory and close out task-16310

**Files:**
- Delete: `tldw_chatbook/Widgets/Media_Ingest/` (entire directory — contains only `__pycache__` with `.pyc` files of modules deleted in commit `0d45bf802`)
- Modify: `backlog/tasks/task-16310 - ....md` (check ACs, add Implementation Notes)

**Interfaces:**
- Consumes: none.
- Produces: none.

Context: no `.py` file anywhere references `Widgets.Media_Ingest` (verified by grep excluding `__pycache__`). The directory is not a package (no `__init__.py`) so setuptools' `packages.find` never includes it; `.gitignore` already excludes `__pycache__/`, so this deletion produces no git-tracked change — it removes locally misleading stale bytecode only.

- [ ] **Step 1: Delete the directory**

```bash
rm -rf tldw_chatbook/Widgets/Media_Ingest
```

- [ ] **Step 2: Verify nothing references it and the suite still imports**

```bash
grep -rn "Media_Ingest" --include="*.py" tldw_chatbook Tests | grep -v __pycache__ ; ls tldw_chatbook/Widgets/ | head
```
Expected: grep returns nothing; `Media_Ingest` no longer listed under `tldw_chatbook/Widgets/`.

Run: `python3 -c "import tldw_chatbook.app"`
Expected: no ImportError.

- [ ] **Step 3: Run the affected test suites together**

Run: `pytest Tests/test_config_save_settings_semantics.py Tests/test_config_encryption_save_behavior.py Tests/RuntimePolicy/ Tests/UI/test_settings_runtime_source_switch.py -q`
Expected: all pass.

- [ ] **Step 4: Update the backlog task**

- Mark all ACs `- [x]` in `backlog/tasks/task-16310 - ....md`.
- Append an `## Implementation Notes` section: summary of the four fixes, files touched, test files added, the `Widgets/Media_Ingest` deletion (local-only, no git change), and the note that the LLM provider key path was audited and needed no changes.
- `backlog task edit 16310 -s Done --notes "Implemented: config save conflict semantics, encryption hard-fail, eager keyring write + surfacing, stale pycache removal"`

- [ ] **Step 5: Commit**

```bash
git add backlog/tasks/ docs/superpowers/plans/2026-08-14-credential-persistence-fixes.md
git commit -m "fix(backlog): complete task-16310 credential persistence fixes"
```

---

## Self-Review

**Spec coverage:** AC1 (conflict false-success) → Task 1. AC2 (eager keyring write + surfaced failures) → Tasks 3 & 4. AC3 (unavailable backend reported) → Task 4 (startup warning log + reason attribute + save-time notify). AC4 (no silent plaintext downgrade; clear failure) → Task 2. AC5 (stale pycache) → Task 5. AC6 (sign-out re-login regression test) → Task 3 `test_re_entered_token_resolves_after_signout_without_legacy_config`. All six ACs covered.

**Placeholder scan:** No TBDs; every code step includes the actual code; every run step includes the command and expected outcome.

**Type consistency:** `store_static_server_credential(server_id: str, secret: str, *, auth_mode: str | None = None) -> str` is defined in Task 3 and called identically in Task 4's screen wiring and tests. `ConfigMutationResult` constructor args match the dataclass at `config.py:5070-5078`. Purposes constants (`SERVER_CREDENTIAL_BEARER_TOKEN`, `SERVER_CREDENTIAL_ACCESS_TOKEN`) are imported in `server_context.py` today (used at lines 413-417, 855).
