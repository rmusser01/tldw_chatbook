"""Pure Trace v2 collaboration bundle contract (TASK-19913)."""

from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import json
import random
import string
import time
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tldw_chatbook.Chat import trajectory_export, trajectory_import
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.Chat.trajectory import (
    TrajectoryRecord,
    TrajectorySnapshot,
    TrajectoryTurn,
)

FIXED_EXPORTED_AT = "2026-08-22T19:00:00+00:00"
SECRET_VALUES = (
    "sk-adversarial-1234567890",
    "Bearer nested-auth-token",
    "correct-horse-battery-staple",
)
LOCAL_PATH = "/Users/alice/private/trace.txt"
LONG_CONTENT = "diagnostic-body-" * 30
PRIVATE_IDENTIFIER = "customer-999-secret-id"
URL_CREDENTIAL = "https://alice:hunter2@example.com/private"


def _record(**overrides: object) -> TrajectoryRecord:
    values: dict[str, object] = {
        "seq": 1,
        "kind": "tool_result",
        "turn_id": "turn-1",
        "message_id": "message-1",
        "content_preview": "A bounded diagnostic summary",
        "usage": ProviderUsage(
            uncached_input=11,
            cache_read=3,
            output=7,
            provider="provider-a",
            model="model-a",
        ),
        "step_started_at": 10.0,
        "first_token_at": 11.5,
        "completed_at": 13.0,
        "model": "model-a",
        "provider": "provider-a",
        "payload": {"result": "ordinary diagnostic payload"},
        "variants": ("superseded answer",),
        "depth": 1,
        "event_id": "event-1",
        "conversation_id": "conversation-1",
        "source_seq": 41,
        "label": "Tool result",
        "status": "succeeded",
        "actor_kind": "tool",
        "actor_id": "fs_read",
        "run_id": "run-1",
        "parent_event_id": None,
        "source_event_id": None,
        "replacement_event_id": None,
        "observed_at": 12.75,
        "field_states": {
            "content_preview": "observed",
            "payload": "observed",
        },
        "sensitivity": "diagnostic",
    }
    values.update(overrides)
    return TrajectoryRecord(**values)  # type: ignore[arg-type]


def _snapshot(*records: TrajectoryRecord) -> TrajectorySnapshot:
    return TrajectorySnapshot(
        turns=(TrajectoryTurn("turn-1", records or (_record(),)),)
    )


def _build(
    snapshot: TrajectorySnapshot,
    profile: object | None = None,
    *,
    confirm_full: bool = False,
) -> dict:
    selected = profile or trajectory_export.TraceExportProfile.REDACTED_DIAGNOSTIC
    preflight = trajectory_export.preflight_trace_export(snapshot, profile=selected)
    return trajectory_export.build_trace_export(
        snapshot,
        preflight=preflight,
        confirm_full=confirm_full,
        exported_at=FIXED_EXPORTED_AT,
    )


def _canonical_digest(payload: dict) -> str:
    unsigned = copy.deepcopy(payload)
    unsigned["integrity"].pop("digest", None)
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resign(payload: dict) -> dict:
    payload["integrity"]["digest"] = _canonical_digest(payload)
    return payload


def test_preflight_trace_export_has_google_style_public_api_docstring() -> None:
    docstring = trajectory_export.preflight_trace_export.__doc__ or ""

    assert "Args:" in docstring
    assert "Returns:" in docstring


def test_profile_enum_and_redacted_diagnostic_default_manifest() -> None:
    profile = trajectory_export.TraceExportProfile
    assert {member.value for member in profile} == {
        "safe_summary",
        "redacted_diagnostic",
        "full_trace",
    }

    preflight = trajectory_export.preflight_trace_export(_snapshot())
    payload = trajectory_export.build_trace_export(
        _snapshot(), preflight=preflight, exported_at=FIXED_EXPORTED_AT
    )

    assert preflight.profile is profile.REDACTED_DIAGNOSTIC
    assert payload["format"] == "tldw-trace"
    assert payload["version"] == 2
    assert payload["manifest"]["profile"] == "redacted_diagnostic"
    assert payload["manifest"]["schema_version"] == 2
    assert payload["manifest"]["event_count"] == 2
    assert payload["manifest"]["privacy_inventory"] == preflight.privacy_inventory
    assert payload["events"][-1]["kind"] == "trace_export"
    assert payload["events"][-1]["payload"]["profile"] == "redacted_diagnostic"


def test_safe_summary_omits_payload_bodies_and_records_provenance() -> None:
    payload = _build(
        _snapshot(_record(payload={"result": LONG_CONTENT})),
        trajectory_export.TraceExportProfile.SAFE_SUMMARY,
    )
    event = payload["events"][0]

    assert event["payload"] is None
    assert event["field_states"]["payload"] == "omitted"
    assert event["field_provenance"]["payload"]["reason"] == "safe_summary"
    assert LONG_CONTENT not in json.dumps(payload)


def test_redacted_diagnostic_previews_content_and_redacts_paths_and_ids() -> None:
    payload = _build(
        _snapshot(
            _record(
                payload={
                    "result": LONG_CONTENT,
                    "path": LOCAL_PATH,
                    "customer_id": PRIVATE_IDENTIFIER,
                }
            )
        )
    )
    event = payload["events"][0]
    serialized = json.dumps(payload)

    assert LONG_CONTENT not in serialized
    assert LOCAL_PATH not in serialized
    assert PRIVATE_IDENTIFIER not in serialized
    assert event["payload"]["result"].endswith("…")
    assert event["field_states"]["payload"] in {"redacted", "truncated"}
    assert event["redaction_provenance"]
    assert payload["manifest"]["redaction_provenance"]


def test_redacted_profiles_alias_envelope_identifiers_without_breaking_lineage() -> (
    None
):
    first = _record(
        event_id="event:customer@example.com:1",
        conversation_id="conversation:customer@example.com",
        turn_id="turn:account-4481",
        message_id="message:private-4481",
        run_id="run:private-4481",
        actor_id="customer@example.com",
    )
    second = _record(
        seq=2,
        source_seq=42,
        event_id="event:customer@example.com:2",
        conversation_id=first.conversation_id,
        turn_id=first.turn_id,
        message_id="message:private-4482",
        run_id=first.run_id,
        actor_id=first.actor_id,
        parent_event_id=first.event_id,
    )

    payload = _build(_snapshot(first, second))
    serialized = json.dumps(payload)
    for private in (
        first.event_id,
        first.conversation_id,
        first.turn_id,
        first.message_id,
        first.run_id,
        first.actor_id,
        second.event_id,
        second.message_id,
    ):
        assert private not in serialized
    assert payload["events"][1]["parent_event_id"] == payload["events"][0]["event_id"]
    assert payload["lineage"][0]["target"] == payload["events"][0]["event_id"]

    full = _build(
        _snapshot(first, second),
        trajectory_export.TraceExportProfile.FULL_TRACE,
        confirm_full=True,
    )
    assert full["events"][0]["event_id"] == first.event_id
    assert full["events"][1]["parent_event_id"] == first.event_id


def test_full_trace_requires_confirmation_and_still_forbids_credentials() -> None:
    profile = trajectory_export.TraceExportProfile.FULL_TRACE
    snapshot = _snapshot(
        _record(
            payload={
                "api_key": SECRET_VALUES[0],
                "nested": {
                    "authorization": SECRET_VALUES[1],
                    "password": SECRET_VALUES[2],
                    "body": LONG_CONTENT,
                    "path": LOCAL_PATH,
                    "customer_id": PRIVATE_IDENTIFIER,
                },
            }
        )
    )
    preflight = trajectory_export.preflight_trace_export(snapshot, profile=profile)

    with pytest.raises(trajectory_export.TrajectoryExportError, match="confirm"):
        trajectory_export.build_trace_export(
            snapshot, preflight=preflight, exported_at=FIXED_EXPORTED_AT
        )

    payload = trajectory_export.build_trace_export(
        snapshot,
        preflight=preflight,
        confirm_full=True,
        exported_at=FIXED_EXPORTED_AT,
    )
    serialized = json.dumps(payload)
    for secret in SECRET_VALUES:
        assert secret not in serialized
    assert LONG_CONTENT in serialized
    assert LOCAL_PATH in serialized
    assert PRIVATE_IDENTIFIER in serialized


def test_full_trace_aliases_credentials_embedded_in_envelope_identifiers() -> None:
    secret_event_id = f"event:{SECRET_VALUES[0]}"
    snapshot = _snapshot(
        _record(
            event_id=secret_event_id,
            actor_id=SECRET_VALUES[1],
            conversation_id=f"conversation:password={SECRET_VALUES[2]}",
        )
    )

    payload = _build(
        snapshot,
        trajectory_export.TraceExportProfile.FULL_TRACE,
        confirm_full=True,
    )
    serialized = json.dumps(payload)

    for secret in SECRET_VALUES:
        assert secret not in serialized
    assert payload["events"][0]["event_id"] != secret_event_id


@pytest.mark.parametrize(
    "profile",
    ["safe_summary", "redacted_diagnostic", "full_trace"],
)
def test_credentials_are_scrubbed_from_usage_envelope_and_pem_fields(
    profile: str,
) -> None:
    pem = (
        "-----BEGIN PRIVATE KEY-----\n"
        "TOP-SECRET-PRIVATE-MATERIAL\n"
        "-----END PRIVATE KEY-----"
    )
    snapshot = _snapshot(
        _record(
            label=pem,
            status=f"password={SECRET_VALUES[2]}",
            usage=ProviderUsage(
                uncached_input=1,
                output=1,
                provider=SECRET_VALUES[1],
                model="model-a",
            ),
            payload={"private_key": pem},
        )
    )

    payload = _build(snapshot, profile, confirm_full=profile == "full_trace")
    serialized = json.dumps(payload)

    assert "TOP-SECRET-PRIVATE-MATERIAL" not in serialized
    for secret in SECRET_VALUES:
        assert secret not in serialized
    assert any(
        item["reason"] == "credential"
        for item in payload["manifest"]["redaction_provenance"]
    )


@pytest.mark.parametrize(
    "profile",
    ["safe_summary", "redacted_diagnostic", "full_trace"],
)
def test_credentials_are_absent_from_serialized_bytes_in_every_profile(
    profile: str,
) -> None:
    snapshot = _snapshot(
        _record(
            content_preview=f"password={SECRET_VALUES[2]}",
            payload={
                "api_key": SECRET_VALUES[0],
                "innocent": {"authorization": SECRET_VALUES[1]},
                SECRET_VALUES[0]: "credential used as a nested key",
            },
        )
    )
    payload = _build(snapshot, profile, confirm_full=profile == "full_trace")
    serialized = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    for secret in SECRET_VALUES:
        assert secret.encode() not in serialized


@pytest.mark.parametrize(
    "profile",
    ["safe_summary", "redacted_diagnostic", "full_trace"],
)
def test_url_userinfo_credentials_are_absent_from_every_profile(profile: str) -> None:
    snapshot = _snapshot(
        _record(
            content_preview=f"Request failed at {URL_CREDENTIAL}",
            payload={"endpoint": URL_CREDENTIAL},
        )
    )

    payload = _build(snapshot, profile, confirm_full=profile == "full_trace")

    assert URL_CREDENTIAL not in json.dumps(payload)


#: TASK-39149: bare secret and restricted keys, live and test (synthetic).
UNDERSCORE_KEYS = (
    "sk_live_abcdefgh12345678",
    "sk_test_abcdefgh12345678",
    "rk_live_abcdefgh12345678",
    "rk_test_abcdefgh12345678",
)


def _export_bytes_with(secret: str, profile: str) -> tuple[bytes, dict]:
    snapshot = _snapshot(
        _record(
            content_preview=f"Request failed with {secret}",
            payload={
                "result": f"key {secret} in a tool result",
                secret: "credential used as a nested key",
            },
        )
    )
    payload = _build(snapshot, profile, confirm_full=profile == "full_trace")
    return json.dumps(payload, ensure_ascii=False).encode("utf-8"), payload


@pytest.mark.parametrize("profile", ["safe_summary", "redacted_diagnostic", "full_trace"])
@pytest.mark.parametrize("secret", UNDERSCORE_KEYS)
def test_bare_underscore_keys_are_masked_wherever_sk_dash_keys_are(
    secret: str, profile: str
) -> None:
    """TASK-39149 AC#1: the export's one key-prefix pattern was ``sk-``, so a
    bare ``sk_live_``/``sk_test_``/``rk_live_``/``rk_test_`` key was exported
    unchanged in every profile."""
    serialized, payload = _export_bytes_with(secret, profile)

    assert secret.encode() not in serialized
    assert any(
        item["reason"] in {"credential", "credential_key"}
        for item in payload["manifest"]["redaction_provenance"]
    )
    for value in ("sk-abcdefgh12345678", secret):
        assert trajectory_export._credential_text(value) == (
            "[credential redacted]",
            True,
        )


#: TASK-39149 AC#2: every class the export masked before, synthetic values.
_OLD_CREDENTIAL_CLASSES = {
    "sk-dash": "sk-abcdefgh12345678",
    "pem": (
        "-----BEGIN RSA PRIVATE KEY-----\nTOP-SECRET-PEM-BODY\n"
        "-----END RSA PRIVATE KEY-----"
    ),
    "bearer": "Bearer bearer-token-abc123",
    "basic": "Basic dXNlcjpwYXNzd29yZA==",
    "aws": "AKIAABCDEFGHIJKLMNOP",
    "userinfo": "https://alice:hunter2@example.com/private",
}


@pytest.mark.parametrize("profile", ["safe_summary", "redacted_diagnostic", "full_trace"])
@pytest.mark.parametrize("name", sorted(_OLD_CREDENTIAL_CLASSES))
def test_the_credential_classes_masked_before_are_still_masked(
    name: str, profile: str
) -> None:
    """TASK-39149 AC#2: ``sk-`` keys, private keys, bearer and basic tokens,
    AWS key ids and URL user info stay masked in every profile."""
    secret = _OLD_CREDENTIAL_CLASSES[name]
    body = {
        "pem": "TOP-SECRET-PEM-BODY",
        "bearer": "bearer-token-abc123",
        "basic": "dXNlcjpwYXNzd29yZA==",
        "userinfo": "hunter2",
    }.get(name, secret)

    serialized, _payload = _export_bytes_with(secret, profile)

    assert body.encode() not in serialized


_FAKE = "DONOTUSEEXAMPLEONLY"
#: TASK-40301: every credential shape the log sanitizer masks, as
#: (fake value, the part that must not survive). Each value is assembled from
#: fragments so no committed line holds a token-shaped literal (secret
#: scanners and GitHub push protection match the literal).
_LOG_SANITIZER_SHAPES = {
    "google": ("AIza" + "DO_NOT_USE_EXAMPLE_ONLY_" + "0" * 11, None),
    **{
        f"github-gh{kind}": ("gh" + kind + "_" + _FAKE + "a" * 22, None)
        for kind in "pousr"
    },
    "github-pat": ("github" + "_pat_" + _FAKE + "a" * 20, None),
    "huggingface": ("hf" + "_" + _FAKE + "a" * 16, None),
    **{
        f"slack-xox{kind}": ("xox" + kind + "-1234567890-" + _FAKE, None)
        for kind in "baprs"
    },
    **{
        f"aws-{prefix.lower()}": (prefix + "DONOTUSEEXAMPLE1", None)
        for prefix in ("AKIA", "ASIA", "AGPA", "AIDA", "AROA")
    },
    "jwt": (
        "eyJ" + "hbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0" + ".DONOTUSEEXAMPLE1",
        None,
    ),
    "openai-project": ("sk" + "-proj-" + _FAKE + "_123456", None),
    "anthropic": ("sk" + "-ant-api03-" + _FAKE + "_123456", None),
    "anthropic-oauth": ("sk" + "-ant-oat01-" + _FAKE + "_123456", None),
    "openrouter": ("sk" + "-or-v1-" + _FAKE + "aaaaaaaa", None),
    "stripe-underscore": ("sk" + "_live_" + _FAKE + "a1B2c3D4e5", None),
    "webhook-secret": ("whsec" + "_" + _FAKE + "a1B2" * 4, None),
    "groq": ("gsk" + "_" + _FAKE + "a1B2" * 6, None),
    # The secret part must vanish, not only the whole string (review
    # r14ep-rev2 C1: masking "Bearer tok" alone left the rest).
    "bearer-any-token": ("Bearer " + "tok!en$" + _FAKE, _FAKE),
    # Review r14ep-rev2 C1: Laravel Sanctum's ``<id>|<secret>`` token.
    "bearer-sanctum": ("Authorization: Bearer " + "5|" + _FAKE, _FAKE),
    "url-user-only": ("https://" + _FAKE + "@example.com/repo.git", _FAKE),
    "pem-unterminated": (
        "-----BEGIN PRIVATE KEY-----\n" + _FAKE + "-PEM-BODY",
        _FAKE + "-PEM-BODY",
    ),
    # Review r14ep-rev2 M-g: a truncated legacy encrypted key, headers first.
    "pem-encrypted-unterminated": (
        "-----BEGIN RSA PRIVATE KEY-----\nProc-Type: 4,ENCRYPTED\n"
        "DEK-Info: AES-128-CBC,0A\n\nMIIEv" + _FAKE,
        "MIIEv" + _FAKE,
    ),
}


@pytest.mark.parametrize("profile", ["safe_summary", "redacted_diagnostic", "full_trace"])
@pytest.mark.parametrize("name", sorted(_LOG_SANITIZER_SHAPES))
def test_every_shape_the_log_sanitizer_masks_is_masked_in_a_shared_bundle(
    name: str, profile: str
) -> None:
    """TASK-40301 AC#1/AC#2: the export kept its own credential list, so a
    GitHub, Hugging Face, Google, Slack, AWS session or JWT token (and
    ``whsec_``/``gsk_``, in neither list) shipped verbatim in a preview, a
    payload value and a payload key, while logs masked the same text."""
    from tldw_chatbook.Utils.log_sanitizer import sanitize_string

    secret, body = _LOG_SANITIZER_SHAPES[name]
    body = body or secret
    assert body not in sanitize_string(f"request failed: {secret}")

    serialized, payload = _export_bytes_with(secret, profile)

    assert body.encode() not in serialized
    assert any(
        item["reason"] in {"credential", "credential_key"}
        for item in payload["manifest"]["redaction_provenance"]
    )


def test_the_bundle_masks_compound_labels_with_the_log_sanitizers_own_labels() -> None:
    """TASK-41133: a compound label the logs name a credential is masked in a
    shared bundle, and a PII label they also mask is not, from one list."""
    from tldw_chatbook.Utils import log_sanitizer

    value = "DONOTUSE" + "123456789"
    for label in sorted(log_sanitizer._DIAGNOSTIC_CREDENTIAL_LABELS):
        if "_" not in label:
            continue
        for spelled in (label, label.upper().replace("_", "-")):
            masked, changed = trajectory_export._credential_text(f"{spelled}={value}")
            assert changed and value not in masked, spelled
    for label in sorted(log_sanitizer._DIAGNOSTIC_PII_LABELS):
        text = f"{label}={value}"
        assert trajectory_export._credential_text(text) == (text, False), label


def test_the_bundle_masks_with_the_log_sanitizers_own_patterns() -> None:
    """TASK-40301: one source of truth, so the two lists cannot drift again."""
    from tldw_chatbook.Utils import log_sanitizer

    shared = log_sanitizer.CREDENTIAL_VALUE_PATTERNS
    assert set(map(id, shared)) <= set(map(id, trajectory_export._CREDENTIAL_VALUE_RES))
    for pattern in (
        *log_sanitizer._STANDALONE_CREDENTIALS,
        *log_sanitizer._UNDERSCORE_KEY_CREDENTIALS,
    ):
        assert pattern in shared


#: TASK-40301 review r14ep-rev1 I2: code and docs that only look like a
#: credential (a template, a placeholder, a shell variable, the user-only
#: ``git@`` form, prose naming a PEM header).
_LOOK_ALIKE_TEXTS = (
    'headers = {"Authorization": f"Bearer {api_key}"}',
    "curl -H 'Authorization: Bearer <token>' https://api.example.com",
    'export AUTH="Bearer $OPENAI_API_KEY"',
    "git clone ssh://git@github.com/org/repo.git",
    # Review r14ep-rev2 M-f: ordinary user names.
    "git clone https://jdoe1234@bitbucket.org/team/repo.git",
    "scp file ssh://username@server.example.com/x",
    "A PEM file begins with -----BEGIN PRIVATE KEY----- then base64 lines.",
    # TASK-41133: compound credential labels over code, placeholders and
    # prose, and labels that are not credentials (user names among them).
    "creds = oauth.fetch_token(url, client_secret=client_secret)",
    'client_secret = os.environ["CLIENT_SECRET"]',
    "refresh_token: <your refresh token>",
    "export CLIENT_SECRET=${CLIENT_SECRET}",
    "Set client_secret: it comes from the provider console.",
    "user_name=alice1987 logged in, username: jdoe1234",
    "max_tokens=4096, next_page_token=CAESBggDEAEYAQ",
    "client_secret=[credential redacted]",
)


#: Review r15tr-rev3 (engineering ruling): a value under a credential label is
#: masked unless it is on a narrow look-alike allowlist. These two tables are
#: the contract. Every secret shape and every look-alike from the reviewers'
#: probes (rev1 to rev3) is in one of them. Labels are split at ``_L`` so no
#: committed line holds a whole assignment; the fakes are assembled likewise.
_L = "_"
_F = "DONOTUSE"
_RNG = random.Random(41133)
_B64 = string.ascii_letters + string.digits + "+/"
_B64URL = string.ascii_letters + string.digits + "-_"


def _fake(length: int, alphabet: str = _B64) -> str:
    return "".join(_RNG.choice(alphabet) for _ in range(length))


_LONG = {n: _fake(n) for n in (2000, 2048, 2049, 2100, 3200)}
_LONG_QUOTED = _fake(3200)
_LONG_SESSION = _fake(2300, _B64URL)
_LONG_YAML = _fake(2300, _B64URL)
_LONG_CONCAT = _fake(1500) + _fake(1500)
_AZURE = "abc8Q~" + _fake(30, string.ascii_letters + string.digits)
_LONG_SIGNATURE = _fake(3000, _B64URL)
_P8 = "Xk9mP2vL7wQ8"  # probe8's fake secret body
_G = "GOCSPX-" + _P8  # probe16's fake Google client secret

#: name -> (text, the secret that must not survive, in whole or in part).
_MUST_MASK: dict[str, tuple[str, str]] = {
    # rev1 (probe2, probe_labels)
    "pw-punct": ("db" + _L + "password=P@ssw0rd!2024", "P@ssw0rd!2024"),
    "pw-punct-json": ('{"db' + _L + 'password": "Tr0ub4dor&3xyz"}', "Tr0ub4dor&3xyz"),
    "pw-amp-tail": ("db" + _L + "password=Tr0ub4dor&3" + _F, "Tr0ub4dor&3" + _F),
    "pw-pct-tail": ("db" + _L + "password=abcdefgh1%2Frest" + _F, "abcdefgh1%2Frest" + _F),
    "pw-hyphen-label": ("db" + "-password=P@ssw0rd!2024", "P@ssw0rd!2024"),
    "pw-short": ("db" + _L + "password=hunt3r2", "hunt3r2"),
    "pw-letters": ("db" + _L + "password=correcthorsebatterystaple", "correcthorsebatterystaple"),
    "refresh-form-body": (
        "grant_type=refresh_token&client_id=x&refresh" + _L + "token=1%2F%2F0g" + _F + "123",
        "1%2F%2F0g" + _F + "123",
    ),
    "client-secret-urlencoded": ("client" + _L + "secret=ab%2Bcd" + _F + "123", "ab%2Bcd" + _F + "123"),
    "client-secret-kv": ("client" + _L + "secret=" + _F + "abc123xyz", _F + "abc123xyz"),
    "pydantic-env": ("APP__CLIENT" + _L + "SECRET=" + _F + "abc123def", _F + "abc123def"),
    "pydantic-env-password": ("DATABASE__PASS" + "WORD=" + _F + "abc123def", _F + "abc123def"),
    "url-key-query": ("https://api.example.com/v1/x?key=" + _F + "abc123def456", _F + "abc123def456"),
    "x-api-key-header": ("x-api" + "-key: " + _F + "0123456789abcdef", _F + "0123456789abcdef"),
    "x-api-key-header-capitals": ("X-API" + "-Key: " + _F + "0123456789abcdef", _F + "0123456789abcdef"),
    "authorization-basic": ("Authorization: Basic dXNlcjpwYXNz" + "d29yZA==", "dXNlcjpwYXNz" + "d29yZA=="),
    "proxy-authorization-basic": (
        "Proxy" + "-Authorization: Basic dXNlcjpwYXNz" + "d29yZA==",
        "dXNlcjpwYXNz" + "d29yZA==",
    ),
    "json-api-key": ('{"api' + _L + 'key": "' + _F + 'abc123def456"}', _F + "abc123def456"),
    "json-api-key-single-quotes": ("{'api" + _L + "key': '" + _F + "abc123def456'}", _F + "abc123def456"),
    "env-api-key": ("OPENAI" + _L + "API_KEY=" + _F + "abc123def456", _F + "abc123def456"),
    "unterminated-quote": ("client" + _L + 'secret="' + _F + "abc123def456", _F + "abc123def456"),
    "azure-subscription-key": (
        "Ocp-Apim" + "-Subscription-Key: 0123456789abcdef0123456789abcdef",
        "0123456789abcdef0123456789abcdef",
    ),
    "dotted-label": ("settings.api" + _L + "key=" + _F + "abc123", _F + "abc123"),
    "space-before-colon": ("client" + _L + "secret : " + _F + "abc123def", _F + "abc123def"),
    "aws-secret-yaml": (
        "aws" + _L + "secret_access_key: wJalrXUtnFEMI/K7MDENG/bPxRfiCY" + _F,
        "wJalrXUtnFEMI/K7MDENG/bPxRfiCY" + _F,
    ),
    "session-token-cookie": ("session" + _L + "token=" + _F + "abc123def456; Path=/", _F + "abc123def456"),
    # rev2 (probe3, probe4, probe5)
    "six-chars-digit": ("db" + _L + "password=hunt3r", "hunt3r"),
    "five-chars-digit": ("db" + _L + "password=hu3er", "hu3er"),
    "hex-no-digit": ("api" + _L + "token=abcdefabcdefabcdefabcdef", "abcdefabcdefabcdefabcdef"),
    "lowercase-no-digit": (
        "api" + _L + "token=qwertyuiopasdfghjklzxcvbnmqwerty",
        "qwertyuiopasdfghjklzxcvbnmqwerty",
    ),
    "mixed-case-15": ("client" + _L + "secret=AbCdEfGhIjKlMnO", "AbCdEfGhIjKlMnO"),
    "mixed-case-16": ("client" + _L + "secret=AbCdEfGhIjKlMnOp", "AbCdEfGhIjKlMnOp"),
    "base64-slash-start": (
        "aws" + _L + "secret_access_key=/K7MDENGbPxRfiCY" + _F + "Ab12+xyz",
        "/K7MDENGbPxRfiCY" + _F + "Ab12+xyz",
    ),
    "base64-plus-start": (
        "aws" + _L + "secret_access_key=+K7MDENGbPxRfiCY" + _F + "Ab12/xyz",
        "+K7MDENGbPxRfiCY" + _F + "Ab12/xyz",
    ),
    "azure-dot-start": ("client" + _L + "secret=.8Q~" + _F + "ab12cd34ef", ".8Q~" + _F + "ab12cd34ef"),
    "tilde-start": ("client" + _L + "secret=~8Q" + _F + "ab12cd34ef", "~8Q" + _F + "ab12cd34ef"),
    "telegram-token": (
        "telegram" + _L + "api_token=123456789:AAH" + _F + "dqTcvCH1vGWJxfSeof",
        "123456789:AAH" + _F + "dqTcvCH1vGWJxfSeof",
    ),
    "dotted-token": ("access" + _L + "token=ya29.a0AfB" + _L + "byC" + _F + "abc123def", "ya29.a0AfB_byC" + _F + "abc123def"),
    "dotted-token-dash": ("access" + _L + "token=ya29.a0AfB-byC" + _F + "abc123def", "ya29.a0AfB-byC" + _F + "abc123def"),
    "numeric-password": ("db" + _L + "password=12345678", "12345678"),
    "numeric-pin": ("api" + _L + "secret=9876543210123456", "9876543210123456"),
    "colon-password": ("db" + _L + "password=Tr0ub4dor:3x", "Tr0ub4dor:3x"),
    "parenthesis-password": ("db" + _L + "password=P4ssw0rd(1)", "P4ssw0rd(1)"),
    "bcrypt-hash": ("db" + _L + "password=$2b$12$" + _F + "abcdefghijklmnopqrstuv", "$2b$12$" + _F + "abcdefghijklmnopqrstuv"),
    "dollar-password": ("db" + _L + "password=$ecretP4ssw0rd", "$ecretP4ssw0rd"),
    "apostrophe-quoted": ("db" + _L + "password: \"Tr0ub4d0r's" + _F + '"', "Tr0ub4d0r's" + _F),
    "passphrase-quoted": ("db" + _L + 'password: "correct horse 42 battery"', "correct horse 42 battery"),
    "escaped-json": ('{\\"client' + _L + 'secret\\": \\"GOCSPX-' + _F + 'aB12\\"}', "GOCSPX-" + _F + "aB12"),
    "escaped-quote-inside": ("client" + _L + 'secret: "pa\\"ss' + _F + '12"', "pa\\\"ss" + _F + "12"),
    "nested-url-fragment": (
        "redirect" + _L + "url=https://app/cb#access" + _L + "token=ya29.a0Af-" + _F + "x12&token_type=Bearer",
        "ya29.a0Af-" + _F + "x12",
    ),
    "nested-json-url": (
        '{"next' + _L + 'page_url": "https://graph.example.com/v1/feed?access' + _L + "token=EAAG" + _F + 'x12&limit=25"}',
        "EAAG" + _F + "x12",
    ),
    "nested-quoted-query": (
        '"query' + _L + 'string": "client_id=x&client' + _L + "secret=GOCSPX-" + _F + 'aB12"',
        "GOCSPX-" + _F + "aB12",
    ),
    "nested-pipe": ("request" + _L + "id=abc123|client" + _L + "secret=GOCSPX-" + _F + "aB12", "GOCSPX-" + _F + "aB12"),
    "nested-unlabelled-url": (
        "see https://app/cb#access" + _L + "token=ya29.a0Af-" + _F + "x12&token_type=Bearer",
        "ya29.a0Af-" + _F + "x12",
    ),
    "nested-error-description": (
        '{"error' + _L + 'description": "client' + _L + "secret=GOCSPX-" + _F + 'aB12 is invalid"}',
        "GOCSPX-" + _F + "aB12",
    ),
    "nested-help-text": (
        '{"help' + _L + 'text": "Set refresh' + _L + "token: 1//0g" + _F + 'x12 in config"}',
        "1//0g" + _F + "x12",
    ),
    "unlabelled-description": (
        '{"description": "Set refresh' + _L + "token: 1//0g" + _F + 'x12 in config"}',
        "1//0g" + _F + "x12",
    ),
    "session-token-uuid": ("session" + _L + "token=550e8400-e29b-41d4-a716-446655440000", "550e8400-e29b-41d4-a716-446655440000"),
    "sentence-version-word": ("Set the refresh" + _L + "token: rotated2024.", "rotated2024"),
    # rev3 (probe6, probe7): long values, identifier-shaped passwords, words
    **{
        f"long-bare-{n}": ("private" + _L + "key=" + value + " end", value)
        for n, value in _LONG.items()
    },
    "long-quoted": ('{"private' + _L + 'key": "' + _LONG_QUOTED + '"}', _LONG_QUOTED),
    "long-session-token": ("aws" + _L + "session_token=" + _LONG_SESSION, _LONG_SESSION),
    "long-yaml": ("client" + _L + "secret: " + _LONG_YAML + "\nnext: 1", _LONG_YAML),
    "long-concatenated": ("refresh" + _L + "token: " + _LONG_CONCAT, _LONG_CONCAT),
    **{
        f"password-{value}": ("db" + _L + "password=" + value, value)
        for value in (
            "hunter22", "password123", "summer2024", "welcome1", "admin123", "letmein99",
            "abc_123", "mysecret_2024", "companyname_2024", "secret_42_prod", "qwerty_123456",
            "deadbeef1234", "a" * 10 + "123456",
        )
    },
    **{
        f"word-run-{value}": ("client" + _L + "secret=" + value, value)
        for value in ("CorrectHorseBatteryStaple", "MySuperSecretPassphrase", "TrustNoOneEverAgain")
    },
    "azure-trailing-dot": ("client" + _L + "secret=" + _AZURE + ".", _AZURE),
    # rev4: moved from the keep table by the exact rev4 rules (a strict label;
    # a call whose argument is a call; an index whose argument is a number).
    "strict-label-call-argument": (
        "secret" + _L + "key=base64.b64encode(os.urandom(32))",
        "base64.b64encode(os.urandom(32))",
    ),
    "strict-label-date": ("the secret" + _L + "key: 2024-01-01 rotated", "2024-01-01"),
    "numeric-index": ("refresh" + _L + "token=tokens[0]", "tokens[0]"),
    # rev4 (probe8)
    **{
        f"probe8-{label}-{value}": (label + "=" + value, value)
        for label, value in (
            ("db" + _L + "password", "password_123"),
            ("db" + _L + "password", "Password_123"),
            ("db" + _L + "password", "Password-2024"),
            ("db" + _L + "password", "Secret_2024"),
            ("db" + _L + "password", "secret_key_2024"),
            ("db" + _L + "password", "token_token"),
            ("db" + _L + "password", "SecretPassword1"),
            ("db" + _L + "password", "AccessKey2024"),
            ("db" + _L + "password", "AuthToken123"),
            ("client" + _L + "secret", "Secret_Key_99"),
            ("client" + _L + "secret", "ApiSecret2024"),
            ("admin" + _L + "password", "AdminPassword1"),
            ("admin" + _L + "password", "Admin_Password_2024"),
            ("root" + _L + "password", "RootPassword123"),
            ("db" + _L + "password", "DbPassword2024"),
            ("mysql_root" + _L + "password", "MysqlRoot2024Password"),
            ("client" + _L + "secret", "Client_Secret_2024"),
            ("db" + _L + "password", "Welcome(123)"),
            ("db" + _L + "password", "Admin[2024]"),
            ("db" + _L + "password", "Password(1)!"),
            ("db" + _L + "password", "letmein(99)"),
            ("db" + _L + "password", "Summer12:30!"),
            ("db" + _L + "password", "1234-5678"),
            ("db" + _L + "password", "12.34.56"),
            ("db" + _L + "password", "pass:/word123"),
            ("db" + _L + "password", "changeme"),
            ("db" + _L + "password", "Your_Pass_2024!"),
            ("db" + _L + "password", "$hunter22"),
            ("db" + _L + "password", "%Passw0rd%"),
            ("db" + _L + "password", "{Secret123}"),
            ("db" + _L + "password", "[Hunter Pass]"),
        )
    },
    "probe8-secret-call": ("client" + _L + "secret=secret(" + _P8 + "!)", "secret(" + _P8),
    "probe8-time-inside": ("client" + _L + "secret=ab12:34" + _P8, "ab12:34" + _P8),
    "probe8-drive-like": ("client" + _L + "secret=c:/" + _P8, "c:/" + _P8),
    "probe8-angle-token": ("client" + _L + "secret=<" + _P8 + ">", _P8),
    "probe8-home-token": ("client" + _L + "secret=~/" + _P8, _P8),
    "probe8-prose-word": ("client" + _L + "secret: letmein then rotate", "letmein"),
    "probe8-jwt-prose": ("jwt" + _L + "secret: supersecret for dev", "supersecret"),
    "probe8-app-prose": ("app" + _L + "secret=monkey and the id is 4", "monkey"),
    "probe8-webhook-prose": ("webhook" + _L + "secret: whsecret in staging", "whsecret"),
    "probe8-jwt-quoted": ('"jwt' + _L + 'secret": "this is my secret 123"', "this is my secret 123"),
    "probe8-quoted-token": ('"client' + _L + 'secret": "' + _P8 + ' zR4"', _P8),
    "probe8-quoted-tokens": ('"session' + _L + 'token": "abc123XYZdef ghi456"', "abc123XYZdef"),
    "probe8-jwt-env-passphrase": ("JWT" + _L + 'SECRET="correct horse battery staple"', "correct horse battery staple"),
    "probe8-yaml-block": ("client" + _L + "secret: |\n  " + _P8 + "abcdef", _P8),
    "probe8-cli-flag": ("--client-secret=" + _P8, _P8),
    "probe8-spaced-colon-tab": ("CLIENT" + _L + "SECRET :\t" + _P8, _P8),
    # rev4 (probe9): each pins a rule the reviewer loosened
    "quoted-letters-only": ('{"client' + _L + 'secret": "AbCdEfGhIjKlMnOpQr"}', "AbCdEfGhIjKlMnOpQr"),
    "snake-with-other-word": ("client" + _L + "secret=acme_prod_2024", "acme_prod_2024"),
    "long-signed-url": (
        "client" + _L + "secret=https://storage.example.com/o?sig=" + _LONG_SIGNATURE,
        _LONG_SIGNATURE,
    ),
    "scheme-inside-value": ("client" + _L + "secret=Pa55word://host", "Pa55word://host"),
    "date-prefixed-value": ("client" + _L + "secret=2024-01-01" + _P8, "2024-01-01" + _P8),
    "upper-snake-password": ("db" + _L + "password=SUMMER_2024", "SUMMER_2024"),
    "call-password": ("db" + _L + "password=Hunter2()", "Hunter2()"),
    "junk-dotted-prefix": ("client" + _L + "secret=hunter2!.client_secret", "hunter2!"),
    # rev5: moved from the keep table: a name with words besides the label's.
    "name-with-other-words": ("client" + _L + "secret=oauth-client-secret", "oauth-client-secret"),
    "camel-name-with-other-words": (
        "client" + _L + "secret=ClientSecretFromVaultConfig",
        "ClientSecretFromVaultConfig",
    ),
    # rev5 (probe10): labels ending in "secret" are strict, bar issued ones
    "probe10-nextauth": ("NEXTAUTH" + _L + 'SECRET="my super secret value"', "my super secret value"),
    "probe10-auth-quoted": ('"auth' + _L + 'secret": "correct horse battery staple"', "correct horse battery staple"),
    "probe10-hmac-changeme": ("hmac" + _L + "secret=changeme", "changeme"),
    "probe10-shared-prose": ("shared" + _L + "secret: the quick brown fox", "quick brown fox"),
    "probe10-admin-snake": ("admin" + _L + "secret=secret_key_2024", "secret_key_2024"),
    "probe10-encryption-time": ("encryption" + _L + "secret=12:30", "12:30"),
    "probe10-master-date": ("master" + _L + "secret=2024-01-01", "2024-01-01"),
    "probe10-session-quoted": ("session" + _L + 'secret="correct horse battery staple"', "correct horse battery staple"),
    "unquoted-passphrase": ("db" + _L + "password: correct horse battery staple", "horse battery staple"),
    # rev5 (probe10): code whose literal is not a look-alike masks whole
    "probe10-secretstr": ("client" + _L + 'secret=SecretStr("GOCSPX-' + _P8 + '")', "GOCSPX-" + _P8),
    "probe10-pydantic-secretstr": (
        "client" + _L + 'secret=pydantic.SecretStr("GOCSPX-' + _P8 + '")',
        "GOCSPX-" + _P8,
    ),
    "probe10-b64decode": ("db" + _L + 'password=base64.b64decode("cGFzc3dvcmQxMjM=")', "cGFzc3dvcmQxMjM="),
    "probe10-fromhex": ("api" + _L + 'token=bytes.fromhex("deadbeef00112233aabb")', "deadbeef00112233aabb"),
    "probe10-cfg-get": ("refresh" + _L + 'token=cfg.get("1//0g' + _P8 + '")', "1//0g" + _P8),
    "probe10-environ-get-default": (
        "client" + _L + 'secret=os.environ.get("CLIENT_SECRET", "GOCSPX-' + _P8 + '")',
        "GOCSPX-" + _P8,
    ),
    "probe10-getenv-default": ("db" + _L + 'password=os.getenv("DB_PASSWORD", "Tr0ub4dor&3")', "Tr0ub4dor&3"),
    "probe10-secret-call": ("client" + _L + "secret=Secret('" + _P8 + "')", _P8),
    # rev5 (probe10): names, templates and placeholders
    "probe10-my-name": ("client" + _L + "secret=myClientSecret99", "myClientSecret99"),
    "probe10-year-version": ("access" + _L + "token=accessToken2024", "accessToken2024"),
    "probe10-kebab-words": ("api" + _L + "token=token-token", "token-token"),
    "probe10-name-suffix": ("client" + _L + "secret=client_secret_prod", "client_secret_prod"),
    "probe10-brace-word": ("db" + _L + "password={hunter}", "hunter"),
    "probe10-mustache-word": ("db" + _L + "password={{ hunter }}", "hunter"),
    "probe10-percent-word": ("db" + _L + "password=%(hunter)s", "hunter"),
    "probe10-title-placeholder": ("db" + _L + "password=Your-Password", "Your-Password"),
    "probe10-reference-word": ("db" + _L + "password=$ADMIN123", "ADMIN123"),
    "probe10-function-word-token": ("client" + _L + "secret: the " + _P8, _P8),
    "probe10-url-key-snake": ("https://x.example/api?key=api" + _L + "key_2024", "api_key_2024"),
    "probe10-url-key-call": ("https://x.example/api?key=Secret(abc)", "Secret(abc)"),
    "probe10-url-key-token": ("https://x.example/api?key=" + _P8 + "abc", _P8),
    # rev5 (probe9 adapted): rules whose loosening needs its own row
    "lowercase-reference": ("db" + _L + "password=$db_password", "$db_password"),
    "title-snake-versioned": ("client" + _L + "secret=Client_Secret_2", "Client_Secret_2"),
    # rev6 (probe12): a default inside code is judged as the label's value
    "probe12-getenv-default": ("db" + _L + 'password=os.getenv("DB_PASSWORD", "postgres")', "postgres"),
    "probe12-environ-get-default": ("db" + _L + 'password=os.environ.get("DB_PASSWORD", "hunter")', "hunter"),
    "probe12-secret-key-default": ("SECRET" + _L + 'KEY=os.environ.get("SECRET_KEY", "devsecret")', "devsecret"),
    "probe12-jwt-default": ("jwt" + _L + 'secret=os.getenv("JWT_SECRET", "supersecret")', "supersecret"),
    "probe12-bare-getenv-default": ("client" + _L + 'secret=getenv("CLIENT_SECRET", "letmein_now")', "letmein_now"),
    "probe12-secretstr-word": ("client" + _L + 'secret=SecretStr("supersecretvalue")', "supersecretvalue"),
    # rev6 (probe9 adapted): a default that names a credential is still a value
    "default-names-a-credential": ("db" + _L + 'password=os.getenv("DB_PASSWORD", "admin_password")', "admin_password"),
    # rev6: a function word before a mixed-case token without digits
    "function-word-mixed-case": ("client" + _L + "secret: the XkmPvLwQabcd", "XkmPvLwQabcd"),
    "default-changeme": ("db" + _L + 'password=os.getenv("DB_PASSWORD", "changeme")', "changeme"),
    "angle-symbols": ("db" + _L + "password=<p@ss!word>", "p@ss!word"),
    # rev7 (probe14): a literal next to kept code is judged with it
    "fallback-or": ("SECRET" + _L + 'KEY=os.getenv("SECRET_KEY") or "dev-' + _P8 + '"', "dev-" + _P8),
    "fallback-js-or": (
        "client" + _L + 'secret: process.env.CLIENT_SECRET || "GOCSPX-' + _P8 + '"',
        "GOCSPX-" + _P8,
    ),
    "fallback-or-single-quotes": (
        "client" + _L + "secret=os.environ.get('CLIENT_SECRET') or 'GOCSPX-" + _P8 + "'",
        "GOCSPX-" + _P8,
    ),
    "fallback-ternary": (
        "client" + _L + 'secret=os.environ["CLIENT_SECRET"] if prod else "GOCSPX-' + _P8 + '"',
        "GOCSPX-" + _P8,
    ),
    "keyword-default": ("db" + _L + 'password=os.getenv("DB_PASSWORD", default="hunter2x")', "hunter2x"),
    "keyword-value": ("client" + _L + 'secret=SecretStr(value="GOCSPX-' + _P8 + '")', "GOCSPX-" + _P8),
    "concat-after-name": ("client" + _L + 'secret=prefix + "' + _P8 + '"', _P8),
    "concat-after-constant": ("client" + _L + 'secret=CLIENT_SECRET + "' + _P8 + '"', _P8),
    # rev7 (probe9 adapted): the rule's own parts
    "fallback-js-ternary": (
        "client" + _L + 'secret: process.env.CLIENT_SECRET ? fallback : "GOCSPX-' + _P8 + '"',
        "GOCSPX-" + _P8,
    ),
    "fallback-comparison": (
        "client" + _L + 'secret=os.environ["CLIENT_SECRET"] if env == "dev" else "GOCSPX-' + _P8 + '"',
        "GOCSPX-" + _P8,
    ),
    "call-word-argument": ("access" + _L + 'token=SecretStr("hunter")', "hunter"),
    "call-capitals-digit": ("client" + _L + 'secret=SecretStr("HUNTER2")', "HUNTER2"),
    "tail-random-word": ("client" + _L + "secret=CLIENT_SECRET or " + _P8, _P8),
    "nested-call": ("client" + _L + 'secret=f(g("GOCSPX-' + _P8 + '"))', "GOCSPX-" + _P8),
    # rev7 (probe14, probe12): a function word before random letters
    "function-word-capitals": ("client" + _L + "secret: the XKQZVBNMPLWRTDXK", "XKQZVBNMPLWRTDXK"),
    "function-word-lowercase-token": ("client" + _L + "secret: use gocspx-abcdefghijklmn", "gocspx-abcdefghijklmn"),
    "function-word-consonants": ("client" + _L + "secret: see xkqzvbnmplwrtd", "xkqzvbnmplwrtd"),
    "function-word-mixed-vowels": ("client" + _L + "secret: the QaZeRtYuIoPa", "QaZeRtYuIoPa"),
    # rev8 (probe16): every value is judged with its whole expression
    **{
        f"probe16-{name}": ("client" + _L + "secret" + text, secret)
        for name, text, secret in (
            ("nested-call", '=wrap(os.getenv("CLIENT_SECRET", "' + _G + '"))', _G),
            ("call-in-call", '=SecretStr(os.environ.get("X") or "' + _G + '")', _G),
            ("f-string", '=f"' + _G + '"', _G),
            ("f-string-field", '=f"{prefix}' + _P8 + '"', _P8),
            ("format", ' = "{}".format("' + _G + '")', _G),
            ("percent", ' = "%s" % "' + _G + '"', _G),
            ("plus", ' = "GOCSPX-" + "' + _P8 + '"', _P8),
            ("empty-or", ' = "" or "' + _G + '"', _G),
            ("empty-if-else", ' = "" if dev else "' + _G + '"', _G),
            ("implicit-concat", ' = "GOCSPX-" "' + _P8 + '"', _P8),
            ("black-call", ' = os.environ.get(\n    "CLIENT_SECRET", "' + _G + '"\n)', _G),
            ("black-call-exploded", ' = os.environ.get(\n    "CLIENT_SECRET",\n    "' + _G + '",\n)', _G),
            ("dict", ' = {"prod": "' + _G + '", "dev": "x"}', _G),
            ("list", ' = ["' + _G + '"]', _G),
            ("tuple", ' = ("' + _G + '",)', _G),
            ("lambda", ' = lambda: "' + _G + '"', _G),
            ("lambda-call", ' = lambda: os.getenv("X", "' + _G + '")', _G),
            ("random-capitals-argument", '=SecretStr("XKQZVBNMPLWRTDXK")', "XKQZVBNMPLWRTDXK"),
            ("nullish", '=os.getenv("CLIENT_SECRET") ?? "' + _G + '"', _G),
            ("js-ternary", ': process.env.CLIENT_SECRET ? process.env.CLIENT_SECRET : "' + _G + '"', _G),
        )
    },
    "probe16-or-changeme": ("db" + _L + 'password=os.getenv("DB_PASSWORD") or "changeme"', "changeme"),
    # rev9 (probe17): every hunt row
    **{
        f"probe17-{name}": (text, secret)
        for name, text, secret in (
            ("after-padding", "session" + _L + "token=dGVzdHRva2VuMTIz== client" + _L + "secret=" + _G, _G),
            ("after-short-padding", "sig=AbCdEf123== client" + _L + "secret=" + _G, _G),
            ("after-glued-comparison", "x>= client" + _L + "secret=" + _G, _G),
            ("after-spaced-comparison", "a == client" + _L + "secret: " + _G, _G),
            ("ternary-first", "client" + _L + 'secret: isProd ? "' + _G + '" : "dev"', _G),
            ("ternary-second", "client" + _L + 'secret: isProd ? "dev" : "' + _G + '"', _G),
            ("ternary-in-dict", '{"client' + _L + 'secret": prod ? "' + _G + '" : "x", "user": "bob"}', _G),
            ("quoted-key-after-code", "client" + _L + 'secret = cfg.get("x") "note": "' + _G + '"', _G),
            ("dict-over-lines", "client" + _L + 'secret = {\n    "prod": "' + _G + '",\n}', _G),
            ("parenthesised-concat", "client" + _L + 'secret = (\n    "GOCSPX-"\n    "' + _P8 + '"\n)', _P8),
            ("continued-line", "client" + _L + 'secret = "GOCSPX-" \\\n    "' + _P8 + '"', _P8),
            ("triple-quoted", "client" + _L + 'secret = """\n' + _G + '\n"""', _G),
            ("black-call-exploded", "client" + _L + 'secret = os.environ.get(\n    "CLIENT_SECRET",\n    "' + _G + '",\n)', _G),
            ("tuple", "client" + _L + 'secret = CLIENT_SECRET, "' + _G + '"', _G),
            ("dotted-or", "client" + _L + 'secret = cfg.client_secret or "' + _G + '"', _G),
        )
    },
    "annotation-secret-default": ("client" + _L + 'secret: str = "' + _G + '"', _G),
    # rev9 (probe9 adapted): the rules' own parts
    "quoted-key-after-name": ("client" + _L + 'secret = CLIENT_SECRET "note": "' + _G + '"', _G),
    "annotation-like-word": ("db" + _L + "password: Secretpassword", "Secretpassword"),
    "path-key-token": ("client" + _L + 'secret = vault.read("secret/k9mp2vl7wq8z")', "k9mp2vl7wq8z"),
    # rev10: a credential label inside a kept call is judged under its own label
    "inner-label-word": ("client" + _L + "secret=build(2, access" + _L + "token=hunter2x)", "hunter2x"),
    "inner-label-after-call": ("client" + _L + "secret=build(g(x), db" + _L + "password=letmein99)", "letmein99"),
    "inner-label-quoted-strict": ("client" + _L + 'secret=build(2, db' + _L + 'password="changeme")', "changeme"),
}

#: name -> text that must survive a shared bundle unchanged.
_MUST_KEEP: dict[str, str] = {
    # rev1 (probe2, probe_labels)
    "placeholder-your": "client" + _L + "secret=your-client-secret",
    "placeholder-your-capitals": "refresh" + _L + "token: YOUR-REFRESH-TOKEN",
    "placeholder-x": "client" + _L + "secret=xxxxxxxx-xxxx",
    "path-etc": "private" + _L + "key=/etc/ssl/private/server.key",
    "path-run": "client" + _L + "secret=/run/secrets/client_secret",
    "path-relative": "private" + _L + "key=./keys/id_rsa2",
    "code-hashlib": "db" + _L + "password=hashlib.sha256(raw).hexdigest()",
    "code-boto3": "aws" + _L + "secret_access_key=boto3.session.Session().get_credentials()",
    "date": "the refresh" + _L + "token: 2024-01-01 rotated",
    "iso-time": "refresh" + _L + "token: 2024-11-16T10:00:00Z expiry",
    "version-label": "api" + _L + "key_version=v2.10.3-rc1",
    "name-label": "client" + _L + "secret_name=my-secret-2024-prod",
    "ruby-symbol": "refresh" + _L + "token: :refresh_token_v2",
    "none-literal": "access" + _L + "token=None, max_retries=3",
    "expiry-label": "access" + _L + "token_expires_in=3600",
    "reference-dollar": "client" + _L + "secret=$CLIENT_SECRET_2",
    "url": "client" + _L + "secret: https://vault.example.com/v1/oauth",
    "placeholder-angle": "refresh" + _L + "token: <your refresh token>",
    "reference-braces": "export CLIENT" + _L + "SECRET=${CLIENT_SECRET}",
    "code-environ": "client" + _L + 'secret = os.environ["CLIENT_SECRET"]',
    "code-keyword-argument": "creds = oauth.fetch_token(url, client" + _L + "secret=client_secret)",
    "prose-bare": "Set client" + _L + "secret: it comes from the provider console.",
    # rev2 (probe3)
    "identifier-digit": "client" + _L + "secret=client_secret_2",
    "identifier-version": "access" + _L + "token=token_v2",
    "keyword-argument-paren": "fetch(refresh" + _L + "token=refresh_token2)",
    "semver-label": "api" + _L + "key_version=1.2.3",
    "prose-section": "client" + _L + "secret: see section 4.2 of the guide",
    "word-run-placeholder": "client" + _L + "secret=YourClientSecretHere",
    "mustache": "client" + _L + "secret={{ secrets.CLIENT_SECRET }}",
    "jinja": "client" + _L + "secret={{secrets.CLIENT_SECRET}}",
    "github-actions": "client" + _L + "secret=${{ secrets.CLIENT_SECRET }}",
    "percent-format": "client" + _L + "secret=%(secret)s",
    "stars": "client" + _L + "secret=****************",
    "redacted-word": "client" + _L + "secret=REDACTED",
    "file-uri": "client" + _L + "secret=file:/run/secrets/x",
    "go-struct": "ClientSecret: cfg.ClientSecret2,",
    "csv-header": "client" + _L + "secret,refresh" + _L + "token,user",
    "markdown-table": "| client" + _L + "secret | string | your OAuth secret, e.g. GOCSPX-xxx |",
    # rev3 (probe6, probe7)
    "path-users": "private" + _L + "key=/Users/alice/.ssh/id_rsa",
    "path-home-mixed-case": "private" + _L + "key=/home/Alice/keys/server.pem",
    "path-drive": "private" + _L + "key=C:\\Users\\alice\\key.pem",
    "prose-quoted": '{"client' + _L + 'secret": "rotate every 90 days"}',
    "prose-quoted-digit": "client" + _L + 'secret: "Issued in step 2 of the guide"',
    "code-getenv": "client" + _L + "secret=os.getenv('CLIENT_SECRET_2')",
    "code-attribute": "client" + _L + "secret=settings.oauth.client_secret_v2",
    "code-private-attribute": "client" + _L + "secret=self._client_secret2",
    "code-json-index": "access" + _L + "token=resp.json()['access_token']",
    "code-index-name": "refresh" + _L + "token=tokens[name]",
    "constant-name": "api" + _L + "secret=SECRET_KEY_V2",
    "constant-name-password": "db" + _L + "password=DB_PASSWORD_2",
    "pascal-name": "client" + _L + "secret=ClientSecret2",
    "code-call": "client" + _L + "secret=getClientSecret2()",
    # rev4 (probe8): a URL's ?key= value judged as an api_key value.
    "url-key-placeholder": "https://maps.googleapis.com/maps/api/js?key=YOUR_API_KEY&callback=init",
    "url-key-constant": "https://x.example/api?key=API" + _L + "KEY",
    "url-key-template": "https://x.example/api?key={api" + _L + "key}",
    # rev5 (probe10): a ?key= value with no token-shaped word.
    "url-key-word": "https://example.com/search?key=python",
    "url-key-path": "https://s3.example.com/bucket?key=photos/2024/cat.jpg",
    # rev6 (probe12): <name> placeholders from READMEs, kept under any label
    "angle-word": "db" + _L + "password=<letmein>",
    "angle-camel": "db" + _L + "password=<MyPassword>",
    "readme-angle-your-client-secret": "client" + _L + "secret=<your-client-secret>",
    "readme-angle-client-secret": "CLIENT" + _L + "SECRET=<client-secret>",
    "readme-angle-constant": "client" + _L + "secret: <CLIENT_SECRET>",
    "readme-angle-token": "refresh" + _L + "token=<token>",
    "readme-angle-access-token": "access" + _L + "token=<ACCESS_TOKEN>",
    "readme-angle-password": "db" + _L + "password=<password>",
    "readme-angle-your-db-password": "db" + _L + "password=<your-db-password>",
    # rev6: the label's own camelCase name
    "camel-label-name": "client" + _L + "secret = clientSecret",
    # rev6 (probe12): a function word before words is prose, fields after it kept
    "function-word-docs": "client" + _L + "secret: see the OAuth2 docs",
    "function-word-fields": "refresh" + _L + "token: the token expired at 10:30, retry_after=60 code=E401",
    "function-word-piped-fields": "client" + _L + "secret: see the OAuth2 docs | request_id=abc123 | status=401",
    # rev7: code whose literals are all look-alikes, fallbacks included
    "fallback-two-lookups": "client" + _L + 'secret=os.getenv("X") or os.getenv("Y")',
    "default-none": "client" + _L + 'secret=os.environ.get("CLIENT_SECRET", None)',
    "keyword-default-none": "client" + _L + 'secret=os.environ.get("CLIENT_SECRET", default=None)',
    "ternary-none": "client" + _L + 'secret=cfg["client_secret"] if cfg else None',
    "env-name-any": "client" + _L + "secret=os.getenv('MY_APP_CS')",
    "dict-entry": '{"client' + _L + 'secret": os.environ["CLIENT_SECRET"], "user": "bob"}',
    "keyword-argument-then-field": "Client(client" + _L + 'secret=os.getenv("CLIENT_SECRET"), timeout=30)',
    "function-word-acronym": "client" + _L + "secret: see HTTPS docs",
    # rev8 (probe16): prose with a short consonant run, and comparisons
    "function-word-strengths": "client" + _L + "secret: see strengths of each option",
    "comparison-operand": "if client" + _L + "secret == other_secret: return",
    "comparison-empty-literal": "if client" + _L + 'secret == "": raise ValueError("missing")',
    "comparison-assert-name": "assert client" + _L + 'secret == "client_secret"',
    "comparison-return": "return client" + _L + "secret == settings.CLIENT_SECRET",
    "comparison-not-equal": "if client" + _L + "secret != other_secret: return",
    "comparison-less-equal": "while client" + _L + "secret <= max_secret: pass",
    "comparison-greater-equal": "if client" + _L + "secret >= min_secret: break",
    # rev8/rev9: a quoted credential key ends the expression before it, comma
    # or not ('"note": "..."' does not: it belongs to the value)
    "quoted-keys-without-commas": '"client' + _L + 'secret": "YOUR_CLIENT_SECRET" "refresh' + _L + 'token": "YOUR_TOKEN"',
    "quoted-keys-after-code": '"client' + _L + 'secret": os.getenv("CLIENT_SECRET") "refresh' + _L + 'token": "YOUR_TOKEN"',
    # rev9 (probe17): type annotations, kept, with a look-alike default
    "annotation-str": "client" + _L + "secret: str",
    "annotation-secretstr": "client" + _L + "secret: SecretStr",
    "annotation-field-default": "client" + _L + 'secret: str = Field(..., description="OAuth client secret")',
    "annotation-union-none": "client" + _L + "secret: str | None = None",
    "annotation-optional-none": "client" + _L + "secret: Optional[str] = None",
    "annotation-indented": "    client" + _L + "secret: str | None = None",
    # rev9 (probe17): balanced calls judged by their literals
    "balanced-nested-call": "client" + _L + 'secret = base64.b64decode(os.environ["CLIENT_SECRET_B64"])',
    "balanced-call-then-index": "client" + _L + 'secret = json.loads(os.environ["SECRETS"])["client_secret"]',
    "balanced-numeric-keyword": "client" + _L + 'secret = vault.read("secret/data/app", version=2)',
    "balanced-call-over-lines": "client" + _L + 'secret = os.environ.get(\n    "CLIENT_SECRET"\n)',
    # rev9 (probe17): comparisons
    "comparison-none": "if access" + _L + "token == None:",
    "comparison-two-names": "ok = refresh" + _L + "token == expected_token",
}


def _survives(secret: str, original: str, output: str) -> bool:
    """Whether the secret, or its first or last six characters, is still in
    the output (more often than the rest of the original text holds them)."""
    rest = original.replace(secret, "")
    return any(
        output.count(part) > rest.count(part)
        for part in {secret, secret[:6], secret[-6:]}
        if part
    )


def _imported_result(text: str) -> str:
    """Import an older bundle holding ``text`` and return it as imported."""
    older = _build(_snapshot())
    older["events"][0]["payload"]["result"] = text
    imported = trajectory_import.load_imported_trace(_resign(older))
    return imported.snapshot.turns[0].records[0].payload["result"]


@pytest.mark.parametrize("name", sorted(_MUST_MASK))
def test_every_secret_in_the_must_mask_table_is_masked(name: str) -> None:
    """Review r15tr-rev3: masked whole in a shared bundle and a second pass
    changes nothing. An older bundle holding it imports, with the secret
    masked (owner decision 2026-11-17, "Re-mask on import")."""
    text, secret = _MUST_MASK[name]

    masked, changed = trajectory_export._credential_text(text)
    serialized, _payload = _export_bytes_with(text, "full_trace")
    imported = _imported_result(text)

    assert changed and not _survives(secret, text, masked), masked
    assert trajectory_export._credential_text(masked) == (masked, False)
    assert secret.encode() not in serialized
    assert imported == masked and not _survives(secret, text, imported), imported


@pytest.mark.parametrize("name", sorted(_MUST_KEEP))
def test_every_look_alike_in_the_must_keep_table_is_kept_and_imports(name: str) -> None:
    """Review r15tr-rev3: kept unchanged in every profile, and an older bundle
    holding it imports unchanged."""
    text = _MUST_KEEP[name]
    assert trajectory_export._credential_text(text) == (text, False)

    for profile in ("safe_summary", "redacted_diagnostic", "full_trace"):
        snapshot = _snapshot(
            _record(content_preview=f"See {text}", payload={"result": f"run {text}"})
        )
        payload = _build(snapshot, profile, confirm_full=profile == "full_trace")
        assert not any(
            item["reason"] in {"credential", "credential_key"}
            for item in payload["manifest"]["redaction_provenance"]
        ), profile
    assert _imported_result(text) == text


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "{client" + "_secret: GOCSPX-" + _FAKE + "aB12}",
            "{client" + "_secret: [credential redacted]}",
        ),
        (
            "(db" + "_password=P@ss" + _FAKE + "1)",
            "(db" + "_password=[credential redacted])",
        ),
        (
            "Use refresh" + "_token: 1//0g" + _FAKE + "x12.",
            "Use refresh" + "_token: [credential redacted].",
        ),
    ],
    ids=("yaml-flow-brace", "parenthesised", "full-stop"),
)
def test_a_masked_label_value_keeps_its_closing_bracket_and_full_stop(
    text: str, expected: str
) -> None:
    """Review r15tr-rev2: an unbalanced closing bracket or a sentence's full
    stop after a value is text, not part of the secret."""
    assert trajectory_export._credential_text(text) == (expected, True)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "client" + "_secret: the " + _P8 + " user=bob",
            "client" + "_secret: [credential redacted] user=bob",
        ),
        (
            "db" + "_password=hunter2 | user=bob | host=db.example.com",
            "db" + "_password=[credential redacted] | user=bob | host=db.example.com",
        ),
        (
            "db" + "_password: correct horse battery staple, user=bob",
            "db" + "_password: [credential redacted], user=bob",
        ),
    ],
    ids=("function-word-token", "piped-fields", "comma-field"),
)
def test_a_masked_run_stops_before_the_next_field(text: str, expected: str) -> None:
    """Review r15tr-rev6: a value that runs on past its first word (a token
    after a function word, an unquoted passphrase) stops before the next
    field, and the separator before that field stays visible."""
    assert trajectory_export._credential_text(text) == (expected, True)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "db" + '_password=os.getenv("DB_PASSWORD", default="hunter2x")',
            "db" + "_password=[credential redacted]",
        ),
        (
            "client" + '_secret=SecretStr(value="GOCSPX-' + _P8 + '")',
            "client" + "_secret=[credential redacted]",
        ),
        ("client" + '_secret=SecretStr("secret")', "client" + "_secret=[credential redacted]"),
        (
            "client" + '_secret=f(g("GOCSPX-' + _P8 + '")), user=bob',
            "client" + "_secret=[credential redacted], user=bob",
        ),
        (
            "SECRET" + '_KEY=os.getenv("SECRET_KEY") or "dev-' + _P8 + '"; DEBUG=1',
            "SECRET" + "_KEY=[credential redacted]; DEBUG=1",
        ),
        (
            "client" + '_secret = os.environ.get(\n    "CLIENT_SECRET", "' + _G + '"\n)',
            "client" + "_secret = [credential redacted]",
        ),
        (
            "client" + '_secret = os.environ.get(\n    "CLIENT_SECRET",\n    "' + _G + '",\n)\nnext = 1',
            "client" + "_secret = [credential redacted]\nnext = 1",
        ),
    ],
    ids=(
        "keyword-default",
        "keyword-value",
        "call-word",
        "nested-call",
        "fallback-then-statement",
        "black-call",
        "black-call-exploded",
    ),
)
def test_code_next_to_a_secret_is_masked_whole(text: str, expected: str) -> None:
    """Review r15tr-rev7: code and the literals next to it (a keyword
    argument, a fallback after ``or``) are masked as one span, never in part,
    and what follows the expression stays visible."""
    assert trajectory_export._credential_text(text) == (expected, True)


def test_masking_never_caches_value_text() -> None:
    """Review r15tr-rev7: a secret must not linger in an ``lru_cache`` after
    masking. Only label text is cached; new values leave every cache the size
    it was once the labels are warm."""
    from tldw_chatbook.Utils import log_sanitizer

    caches = [
        obj
        for module in (log_sanitizer, trajectory_export)
        for obj in vars(module).values()
        if hasattr(obj, "cache_info")
    ]
    assert caches
    assert not hasattr(log_sanitizer._words, "cache_info")

    def mask(index: int) -> None:
        for text in (
            "db" + f"_password=summer202{index}word",
            "client" + f'_secret=SecretStr("supersecret{index}value")',
            "client" + f"_secret=client_secret_{index}x",
            "client" + f"_secret=super_secret_{index}",
            "refresh" + f"_token: the token{index}",
            f"https://x.example/api?key=python{index}x",
            # rev10: a secret in a quoted-key position after a value
            "client" + f'_secret = cfg.x "GOCSPX-{_P8}{index}": 1',
            "client" + f'_secret = CLIENT_SECRET "GOCSPX-{_P8}{index}": 1',
        ):
            trajectory_export._credential_text(text)

    mask(0)  # the labels are cached here
    before = [cache.cache_info().currsize for cache in caches]
    for index in range(1, 6):
        mask(index)
    assert [cache.cache_info().currsize for cache in caches] == before


#: Review r15tr-rev2: adversarial texts of about 200 KB. The export's URL
#: user-info rule took 198 s on the dash-separated one (quadratic), so a shared
#: bundle could hang its import.
_ADVERSARIAL_TEXTS = {
    "dash-label": "a" + "-a" * 100_000 + "=x1y2z3w4",
    "chained-credential-labels": ("client" + "_secret=") * 15_000,
    "chained-labels": "a_b=" * 50_000,
    "commas": "a_b=x" + ",a" * 100_000,
    "unclosed-quote": "client" + '_secret="' + "x" * 200_000,
    "many-assignments": " a_b=c1d2e3f4" * 16_000,
    # Review r15tr-rev3 (probe6): the rescan and value-length costs.
    "chained-credential-labels-spaced": ("client" + "_secret= ") * 15_000,
    "credential-label-long-runs": ("client" + "_secret=" + "a" * 2040 + " ") * 100,
    "credential-label-then-long-field": "client" + "_secret=x&" + "a" * 200_000,
    "many-delimited-segments": "client" + "_secret=x" + "&aaaaaaaaaaaaaaaaaaaa" * 10_000,
    # Review r15tr-rev7: code expressions scanned to their end.
    "kept-code-quoted-keys": ("client" + '_secret=CLIENT_SECRET "client' + '_secret": CLIENT_SECRET ') * 4_000,
    "kept-code-chain": ("client" + '_secret=os.getenv("X") or os.getenv("Y") ') * 4_000,
    "unclosed-calls": ("client" + "_secret=f(") * 20_000,
    "label-at-end": "x " * 100_000 + "client" + "_secret=",
    "separator-run": "a" + "_" * 200_000 + "b=x1y2z3w4",
    "trailing-closers": ("db" + "_password=a1" + ")" * 2000 + " ") * 100,
    "trailing-dots": ("db" + "_password=a1" + "." * 2000 + " ") * 100,
    "quoted-runs": ('"client' + '_secret": "' + "x" * 2047 + '" ') * 100,
    "nested-labels": ("a_b=" + "client" + "_secret=") * 15_000,
    "quoted-prose-1mb": ('"client' + '_secret": "a b" ') * 40_000,
}


def test_escape_quoted_values_scan_in_linear_time() -> None:
    """Review r15tr-rev4: each escape-quoted value scanned to the end of its
    line, so one long line of them was quadratic (0.84 s at 800 KB). Four
    times the text must take well under 16 times the CPU time."""
    from tldw_chatbook.Utils.log_sanitizer import mask_credential_labels

    def cpu(repeats: int) -> float:
        text = ("client" + '_secret=\\"x\\" ') * repeats
        best = float("inf")
        for _ in range(3):
            started = time.thread_time()
            mask_credential_labels(text)
            best = min(best, time.thread_time() - started)
        return best

    small, large = cpu(25_000), cpu(100_000)
    assert large < 8 * small, (small, large)


@pytest.mark.parametrize(
    ("unit", "plain_unit"),
    [
        ("client" + "_secret: the ", "client" + "_secret: thx "),
        ("db" + "_password=a b c ", "db" + "_password=abc "),
    ],
    ids=("function-word-chain", "passphrase-chain"),
)
def test_runs_after_a_value_scan_in_linear_time(unit: str, plain_unit: str) -> None:
    """Review r15tr-rev6: each function word looked up to 2 KB ahead for a
    token, so a 1 MB chain took 2.2 s against 0.29 s before. Four times the
    text (2 MB at the large end) must take well under 16 times the CPU time,
    and the chain must cost about what a chain of one-word values costs (the
    2 KB look-ahead was linear but 7 times slower, which the ratio misses)."""
    from tldw_chatbook.Utils.log_sanitizer import mask_credential_labels

    def cpu(text_unit: str, repeats: int) -> float:
        text = text_unit * repeats
        best = float("inf")
        for _ in range(3):
            started = time.thread_time()
            mask_credential_labels(text)
            best = min(best, time.thread_time() - started)
        return best

    small, large = cpu(unit, 25_000), cpu(unit, 100_000)
    plain = cpu(plain_unit, 100_000)
    assert large < 8 * small, (small, large)
    assert large < 3 * plain, (large, plain)


@pytest.mark.parametrize("name", sorted(_ADVERSARIAL_TEXTS))
def test_adversarial_bundle_text_is_masked_in_bounded_time(name: str) -> None:
    """Measured as this thread's CPU time, so machine load does not count.
    The slowest 200 KB case measured 0.104 s and the 920 KB quoted-prose
    case 0.368 s (review r15tr-rev4, best of three at load 57), so 2 s
    leaves over 5x headroom."""
    text = _ADVERSARIAL_TEXTS[name]
    started = time.thread_time()

    trajectory_export._credential_text(text)

    assert time.thread_time() - started < 2.0, name


@pytest.mark.parametrize("profile", ["safe_summary", "redacted_diagnostic", "full_trace"])
@pytest.mark.parametrize("text", _LOOK_ALIKE_TEXTS)
def test_code_and_docs_that_only_look_like_credentials_survive_a_shared_bundle(
    text: str, profile: str
) -> None:
    """TASK-40301 review I2: the log rules for Bearer, URL user info and an
    unterminated private key also masked these in a new export and refused an
    older bundle holding them. The bundle uses their credential-shaped forms,
    so these survive and import, while real ones stay masked (the shapes
    test above), also when an older bundle holds them (the import test
    below)."""
    assert trajectory_export._credential_text(text) == (text, False)

    _serialized, payload = _export_bytes_with(text, profile)

    assert not any(
        item["reason"] in {"credential", "credential_key"}
        for item in payload["manifest"]["redaction_provenance"]
    )
    older = _build(_snapshot())
    older["events"][0]["payload"]["result"] = text
    trajectory_import.load_imported_trace(_resign(older))


def test_an_older_bundle_holding_a_newly_masked_shape_imports_masked() -> None:
    """TASK-40301 AC#3: a bundle exported before the export masked these
    shapes can still hold one verbatim. It was refused on import; since the
    owner decision of 2026-11-17 ("Re-mask on import", TASK-41133) it imports
    with the value masked by the current rules."""
    secret, _body = _LOG_SANITIZER_SHAPES["github-ghp"]

    imported = _imported_result(f"cloned with {secret}")

    assert secret not in imported
    assert imported == trajectory_export._credential_text(f"cloned with {secret}")[0]


def test_preflight_has_one_decision_per_material_field_without_count_drift() -> None:
    snapshot = _snapshot(
        _record(
            field_states={
                "content_preview": "redacted",
                "payload": "capture_failed",
                "variants": "truncated",
                "model": "omitted",
                "provider": "not_available",
            }
        )
    )
    preflight = trajectory_export.preflight_trace_export(snapshot)
    decisions = preflight.field_decisions

    keys = [(decision.event_id, decision.field) for decision in decisions]
    assert len(keys) == len(set(keys))
    assert {(decision.field, decision.state) for decision in decisions} >= {
        ("content_preview", "redacted"),
        ("payload", "capture_failed"),
        ("variants", "truncated"),
        ("model", "omitted"),
        ("provider", "not_available"),
    }
    states = [decision.state for decision in decisions]
    inventory = preflight.privacy_inventory
    assert inventory["redacted"] == states.count("redacted")
    assert inventory["omitted"] == states.count("omitted")
    assert inventory["truncated"] == states.count("truncated")
    assert inventory["capture_failed"] == states.count("capture_failed")
    assert inventory["missing"] == sum(
        state in {"not_available", "capture_failed"} for state in states
    )
    assert inventory["sensitive"] == sum(d.sensitive for d in decisions)


def test_preflight_counts_overlapping_redaction_and_truncation_once_each() -> None:
    preflight = trajectory_export.preflight_trace_export(
        _snapshot(
            _record(
                payload={
                    "api_key": SECRET_VALUES[0],
                    "result": LONG_CONTENT,
                }
            )
        )
    )

    redactions = {
        (item["event_id"], item["field"], item["state"])
        for item in preflight.redaction_provenance
        if item["state"] == "redacted"
    }
    assert preflight.privacy_inventory["redacted"] == len(redactions)
    assert any(
        item["reason"] == "credential" for item in preflight.redaction_provenance
    )
    assert preflight.privacy_inventory["truncated"] == 1
    assert preflight.privacy_inventory["observed"] == sum(
        decision.source_state == "observed" for decision in preflight.field_decisions
    )


def test_credential_scrub_keeps_one_decision_per_event_field() -> None:
    preflight = trajectory_export.preflight_trace_export(
        _snapshot(_record(payload={"api_key": SECRET_VALUES[0]}))
    )
    keys = [
        (decision.event_id, decision.field) for decision in preflight.field_decisions
    ]

    assert len(keys) == len(set(keys))


def test_safe_summary_coarsens_timing_and_records_provenance() -> None:
    payload = _build(
        _snapshot(
            _record(
                observed_at=12.875,
                step_started_at=10.625,
                first_token_at=11.875,
                completed_at=13.625,
            )
        ),
        trajectory_export.TraceExportProfile.SAFE_SUMMARY,
    )
    event = payload["events"][0]

    assert event["observed_at"] == 12.0
    assert event["step_started_at"] == 10.0
    assert event["first_token_at"] == 11.0
    assert event["completed_at"] == 13.0
    assert all(
        event["field_states"][field] == "truncated"
        for field in (
            "observed_at",
            "step_started_at",
            "first_token_at",
            "completed_at",
        )
    )
    assert {
        item["field"]
        for item in payload["manifest"]["redaction_provenance"]
        if item["reason"] == "coarse_timing_1s"
    } == {
        "observed_at",
        "step_started_at",
        "first_token_at",
        "completed_at",
    }


def test_builder_rejects_a_preflight_from_a_different_snapshot() -> None:
    first = _snapshot(_record(event_id="event-1"))
    second = _snapshot(_record(event_id="event-2"))
    preflight = trajectory_export.preflight_trace_export(first)

    with pytest.raises(trajectory_export.TrajectoryExportError, match="snapshot"):
        trajectory_export.build_trace_export(
            second, preflight=preflight, exported_at=FIXED_EXPORTED_AT
        )


def test_round_trip_preserves_order_identity_lineage_timing_usage_and_missing() -> None:
    first = _record(
        event_id="event-1",
        seq=7,
        source_seq=41,
        field_states={"payload": "not_available"},
        payload=None,
    )
    second = _record(
        event_id="event-2",
        seq=8,
        source_seq=42,
        parent_event_id="event-1",
        source_event_id="event-1",
        replacement_event_id="event-2",
        field_states={"payload": "observed"},
    )
    payload = _build(
        _snapshot(first, second),
        trajectory_export.TraceExportProfile.FULL_TRACE,
        confirm_full=True,
    )

    imported = trajectory_import.load_imported_trace(payload)
    records = [record for turn in imported.snapshot.turns for record in turn.records]

    assert [record.event_id for record in records] == [
        "event-1",
        "event-2",
        payload["events"][-1]["event_id"],
    ]
    restored = records[:2]
    assert [record.seq for record in restored] == [7, 8]
    assert [record.source_seq for record in restored] == [41, 42]
    assert restored[1].parent_event_id == "event-1"
    assert restored[1].source_event_id == "event-1"
    assert restored[1].replacement_event_id == "event-2"
    assert restored[1].observed_at == 12.75
    assert restored[1].usage == second.usage
    assert restored[0].field_states["payload"] == "not_available"
    assert records[-1].kind == "trace_export"
    assert imported.operation_event.kind == "trace_import"
    assert imported.operation_event.event_id not in {
        event["event_id"] for event in payload["events"]
    }
    assert imported.integrity["verified"] is True
    assert imported.privacy_inventory == payload["manifest"]["privacy_inventory"]
    with pytest.raises(FrozenInstanceError):
        imported.snapshot = _snapshot()  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "state"),
    [
        ("query", "not_available"),
        ("result", "capture_failed"),
        ("task", "redacted"),
        ("event_id", "redacted"),
        ("token_count", "not_available"),
        ("input_token_count", "capture_failed"),
    ],
)
def test_round_trip_accounts_for_every_source_non_observed_field_state(
    field: str, state: str
) -> None:
    """Projection-level field states must agree with inventory and provenance."""
    payload = _build(_snapshot(_record(field_states={field: state})))

    imported = trajectory_import.load_imported_trace(payload)
    event = payload["events"][0]
    assert event["field_provenance"][field] == {
        "state": state,
        "reason": f"source_{state}",
        "sensitivity": "diagnostic",
    }
    assert {
        "event_id": event["event_id"],
        "field": field,
        "state": state,
        "reason": f"source_{state}",
    } in payload["manifest"]["redaction_provenance"]
    if state in {"not_available", "capture_failed"}:
        assert payload["manifest"]["privacy_inventory"][state] == 1
        assert payload["manifest"]["privacy_inventory"]["missing"] == 1
    else:
        assert payload["manifest"]["privacy_inventory"][state] >= 1
    assert imported.privacy_inventory == payload["manifest"]["privacy_inventory"]


@pytest.mark.parametrize(
    "exported_at",
    ["password=EXPORT-TIMESTAMP-SECRET", "not-a-timestamp"],
)
def test_builder_rejects_non_iso_or_credential_bearing_export_timestamp(
    exported_at: str,
) -> None:
    snapshot = _snapshot()
    preflight = trajectory_export.preflight_trace_export(snapshot)

    with pytest.raises(trajectory_export.TrajectoryExportError, match="ISO 8601"):
        trajectory_export.build_trace_export(
            snapshot,
            preflight=preflight,
            exported_at=exported_at,
        )


def test_canonical_digest_is_deterministic_and_detects_tampering() -> None:
    snapshot = _snapshot()
    one = _build(snapshot)
    two = _build(snapshot)

    assert one == two
    assert one["integrity"] == {
        "algorithm": "sha256",
        "digest": _canonical_digest(one),
        "authenticity": False,
    }

    tampered = copy.deepcopy(one)
    tampered["events"][0]["status"] = "tampered"
    with pytest.raises(
        trajectory_import.TrajectoryImportError, match="digest mismatch"
    ):
        trajectory_import.load_imported_trace(tampered)


def test_malformed_digest_is_rejected_actionably() -> None:
    payload = _build(_snapshot())
    payload["integrity"]["digest"] = "not-a-sha256"

    with pytest.raises(trajectory_import.TrajectoryImportError, match="digest"):
        trajectory_import.load_imported_trace(payload)


def test_resigned_bundle_cannot_claim_digest_authenticity() -> None:
    payload = _build(_snapshot())
    payload["integrity"]["authenticity"] = True
    _resign(payload)

    with pytest.raises(
        trajectory_import.TrajectoryImportError, match="authenticity.*false"
    ):
        trajectory_import.load_imported_trace(payload)


def test_resigned_bundle_with_credentials_imports_with_them_masked() -> None:
    """Owner decision 2026-11-17 ("Re-mask on import", TASK-41133): a
    resigned bundle holding a credential imports, masked."""
    payload = _build(_snapshot())
    payload["events"][0]["payload"]["injected"] = "password=IMPORTED-SECRET"
    _resign(payload)

    imported = trajectory_import.load_imported_trace(payload)

    injected = imported.snapshot.turns[0].records[0].payload["injected"]
    assert "IMPORTED-SECRET" not in injected
    assert injected == trajectory_export._credential_text("password=IMPORTED-SECRET")[0]


def _strings(value: object) -> list[str]:
    """Every string in a nested structure: keys, values and fields."""
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, dict):
        return [s for key, item in value.items() for s in (*_strings(key), *_strings(item))]
    if isinstance(value, (list, tuple)):
        return [s for item in value for s in _strings(item)]
    return [value] if isinstance(value, str) else []


def test_an_import_of_a_must_mask_bundle_stores_only_masked_text() -> None:
    """Owner decision 2026-11-17 ("Re-mask on import", TASK-41133): an older
    bundle holding every must-mask secret, under ordinary keys, under a
    credential-named key and as a key, imports; what the import keeps holds
    none of them, and nothing in it is credential material any more."""
    payload = _build(_snapshot())
    texts = {f"r{index}": text for index, (text, _secret) in enumerate(_MUST_MASK.values())}
    payload["events"][0]["payload"].update(texts)
    payload["events"][0]["payload"]["api" + "_key"] = "sk-" + _P8 + _P8
    payload["events"][0]["payload"]["client" + _L + "secret=" + _G] = "x"
    _resign(payload)

    imported = trajectory_import.load_imported_trace(payload)

    stored = imported.snapshot.turns[0].records[0].payload
    kept = "\n".join(_strings(imported.snapshot))
    for index, (text, secret) in enumerate(_MUST_MASK.values()):
        assert not _survives(secret, text, stored[f"r{index}"]), secret
    # The token-shaped fakes appear nowhere in what the import keeps.
    assert _P8 not in kept and "DONOTUSE" + "abc123" not in kept
    records = [record for turn in imported.snapshot.turns for record in turn.records]
    assert not trajectory_export._has_credential_material([record.payload for record in records])


@pytest.mark.parametrize("field", ["sensitive", "included", "observed"])
def test_resigned_bundle_rejects_contradictory_privacy_inventory(field: str) -> None:
    payload = _build(_snapshot())
    payload["manifest"]["privacy_inventory"][field] += 100
    payload["events"][-1]["payload"]["privacy_inventory"][field] += 100
    _resign(payload)

    with pytest.raises(
        trajectory_import.TrajectoryImportError,
        match=rf"privacy_inventory\.{field}",
    ):
        trajectory_import.load_imported_trace(payload)


def test_resigned_bundle_rejects_field_state_provenance_contradiction() -> None:
    payload = _build(_snapshot(_record(field_states={"payload": "redacted"})))
    payload["events"][0]["field_provenance"]["payload"]["state"] = "observed"
    _resign(payload)

    with pytest.raises(
        trajectory_import.TrajectoryImportError,
        match="field_provenance.*payload.*field_states",
    ):
        trajectory_import.load_imported_trace(payload)


def test_resigned_bundle_cannot_strip_the_complete_privacy_basis() -> None:
    payload = _build(_snapshot())
    event = payload["events"][0]
    event["field_states"] = {}
    event["field_provenance"] = {}
    event["redaction_provenance"] = []
    payload["manifest"]["privacy_decisions"] = []
    payload["manifest"]["redaction_provenance"] = []
    payload["manifest"]["missing_metadata"] = []
    payload["manifest"]["privacy_inventory"] = {
        key: 0 for key in payload["manifest"]["privacy_inventory"]
    }
    payload["events"][-1]["payload"]["privacy_inventory"] = dict(
        payload["manifest"]["privacy_inventory"]
    )
    _resign(payload)

    with pytest.raises(
        trajectory_import.TrajectoryImportError,
        match="material field_states|privacy_decisions",
    ):
        trajectory_import.load_imported_trace(payload)


def test_resigned_bundle_requires_exact_decision_field_state_coverage() -> None:
    payload = _build(_snapshot())
    payload["manifest"]["privacy_decisions"].pop()
    _resign(payload)

    with pytest.raises(
        trajectory_import.TrajectoryImportError,
        match="privacy_decisions.*exactly match",
    ):
        trajectory_import.load_imported_trace(payload)


def test_resigned_bundle_cannot_mark_transformed_fields_insensitive() -> None:
    payload = _build(_snapshot())
    for decision in payload["manifest"]["privacy_decisions"]:
        decision["sensitive"] = False
    payload["manifest"]["privacy_inventory"]["sensitive"] = 0
    payload["events"][-1]["payload"]["privacy_inventory"]["sensitive"] = 0
    _resign(payload)

    with pytest.raises(
        trajectory_import.TrajectoryImportError,
        match="sensitive.*field classification|sensitive.*state",
    ):
        trajectory_import.load_imported_trace(payload)


def test_resigned_bundle_cannot_strip_populated_identity_privacy_coverage() -> None:
    payload = _build(_snapshot())
    event = payload["events"][0]
    identity_fields = {
        "event_id",
        "conversation_id",
        "turn_id",
        "message_id",
        "actor_id",
        "run_id",
    }
    for field in identity_fields:
        event["field_states"].pop(field)
        event["field_provenance"].pop(field)
    event["redaction_provenance"] = [
        item
        for item in event["redaction_provenance"]
        if item["field"] not in identity_fields
    ]
    payload["manifest"]["redaction_provenance"] = list(event["redaction_provenance"])
    payload["manifest"]["privacy_decisions"] = [
        decision
        for decision in payload["manifest"]["privacy_decisions"]
        if decision["field"] not in identity_fields
    ]
    redacted = {
        (item["event_id"], item["field"], item["state"])
        for item in event["redaction_provenance"]
        if item["state"] == "redacted"
    }
    inventory = payload["manifest"]["privacy_inventory"]
    inventory["redacted"] = len(redacted)
    inventory["sensitive"] = sum(
        decision["sensitive"] for decision in payload["manifest"]["privacy_decisions"]
    )
    inventory["included"] = sum(
        decision["state"] not in {"not_available", "capture_failed", "omitted"}
        for decision in payload["manifest"]["privacy_decisions"]
    )
    inventory["observed"] = sum(
        decision["source_state"] == "observed"
        for decision in payload["manifest"]["privacy_decisions"]
    )
    payload["events"][-1]["payload"]["privacy_inventory"] = dict(inventory)
    _resign(payload)

    with pytest.raises(
        trajectory_import.TrajectoryImportError,
        match="identity field_states",
    ):
        trajectory_import.load_imported_trace(payload)


def test_resigned_bundle_narrowly_identifies_export_operation_exception() -> None:
    payload = _build(_snapshot())
    payload["manifest"]["export_operation_event_id"] = payload["events"][0]["event_id"]
    _resign(payload)

    with pytest.raises(
        trajectory_import.TrajectoryImportError,
        match="export_operation_event_id",
    ):
        trajectory_import.load_imported_trace(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda p: p["events"][0].update(step_started_at=float("nan")), "finite"),
        (lambda p: p["events"][0].update(completed_at=float("inf")), "finite"),
        (lambda p: p["events"][0].update(seq=-1), "seq"),
        (lambda p: p["events"][0].update(depth=-1), "depth"),
        (
            lambda p: p["events"][0]["usage"].update(uncached_input="many"),
            "usage.uncached_input",
        ),
        (
            lambda p: p["events"][0]["usage"].update(cache_read=-1),
            "usage.cache_read",
        ),
        (
            lambda p: p["events"][0]["usage"].update(
                transcription_seconds=float("inf")
            ),
            "usage.transcription_seconds",
        ),
        (
            lambda p: p["events"][0]["usage"].update(provider=7),
            "usage.provider",
        ),
        (
            lambda p: p["events"][0]["usage"].update(partial="false"),
            "usage.partial",
        ),
    ],
)
def test_resigned_bundle_rejects_invalid_numeric_and_usage_data(
    mutation, message: str
) -> None:
    payload = _build(_snapshot())
    mutation(payload)
    _resign(payload)

    with pytest.raises(trajectory_import.TrajectoryImportError, match=message):
        trajectory_import.load_imported_trace(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda p: p["manifest"].update(exported_at="yesterday"),
            "exported_at.*ISO 8601",
        ),
        (
            lambda p: p["manifest"].update(
                exported_timestamp="2026-08-22T20:00:00+00:00"
            ),
            "timestamps.*match",
        ),
        (
            lambda p: p["manifest"].update(source={"type": "database"}),
            "source",
        ),
        (
            lambda p: p["manifest"]["source"].update(
                conversation_ids=["wrong-conversation"]
            ),
            "conversation_ids.*events",
        ),
    ],
)
def test_resigned_bundle_rejects_invalid_manifest_source_and_timestamps(
    mutation, message: str
) -> None:
    payload = _build(_snapshot())
    mutation(payload)
    _resign(payload)

    with pytest.raises(trajectory_import.TrajectoryImportError, match=message):
        trajectory_import.load_imported_trace(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda p: p.update(version=3), "version 3"),
        (lambda p: p["manifest"].update(profile="unsafe"), "profile"),
        (lambda p: p["manifest"].update(profile=[]), "profile"),
        (
            lambda p: p["manifest"]["privacy_inventory"].update(redacted=-1),
            "privacy_inventory",
        ),
        (
            lambda p: p["manifest"]["privacy_inventory"].pop("sensitive"),
            "privacy_inventory.*sensitive",
        ),
        (lambda p: p["manifest"].pop("missing_metadata"), "missing_metadata"),
        (lambda p: p["manifest"].pop("redaction_provenance"), "redaction_provenance"),
        (lambda p: p["events"].__setitem__(0, "bad-event"), r"events\[0\]"),
        (lambda p: p["events"][0].pop("observed_at"), r"events\[0\].observed_at"),
        (
            lambda p: p["events"][0].pop("field_provenance"),
            r"events\[0\].field_provenance",
        ),
        (
            lambda p: p["events"][0].pop("redaction_provenance"),
            r"events\[0\].redaction_provenance",
        ),
        (lambda p: p["events"][0].update(seq="not-an-integer"), r"events\[0\].seq"),
        (lambda p: p["events"][0].update(kind=[]), r"events\[0\].kind"),
        (lambda p: p["events"][0].update(turn_id={}), r"events\[0\].turn_id"),
        (
            lambda p: p["events"][0].update(step_started_at="yesterday"),
            r"events\[0\].step_started_at",
        ),
        (lambda p: p["events"][0].update(variants=[{}]), r"events\[0\].variants"),
        (
            lambda p: p["events"].append(copy.deepcopy(p["events"][0])),
            "duplicate.*event_id",
        ),
        (
            lambda p: p["events"][0].update(parent_event_id="missing-event"),
            "dangling.*parent_event_id",
        ),
        (
            lambda p: p["events"][0].update(source_event_id="missing-event"),
            "dangling.*source_event_id",
        ),
        (
            lambda p: p["events"][0].update(replacement_event_id="missing-event"),
            "dangling.*replacement_event_id",
        ),
        (
            lambda p: p["lineage"].append(
                {
                    "source": "event-1",
                    "target": "missing-event",
                    "relationship": "parent",
                }
            ),
            "lineage",
        ),
    ],
)
def test_malformed_v2_structures_fail_closed(mutation, message: str) -> None:
    payload = _build(_snapshot())
    mutation(payload)
    if isinstance(payload.get("integrity"), dict):
        _resign(payload)

    with pytest.raises(trajectory_import.TrajectoryImportError, match=message):
        trajectory_import.load_imported_trace(payload)


def test_resigned_lineage_must_exactly_match_event_reference_fields() -> None:
    first = _record(event_id="event-1")
    second = _record(event_id="event-2", seq=2, parent_event_id="event-1")
    payload = _build(_snapshot(first, second))
    payload["lineage"][0]["target"] = payload["events"][1]["event_id"]
    _resign(payload)

    with pytest.raises(
        trajectory_import.TrajectoryImportError, match="lineage does not match events"
    ):
        trajectory_import.load_imported_trace(payload)


def test_v2_import_module_has_no_database_or_textual_imports() -> None:
    source = Path(trajectory_import.__file__).read_text(encoding="utf-8")
    imported_names: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported_names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_names.append(node.module or "")

    assert not any(name.startswith("tldw_chatbook.DB") for name in imported_names)
    assert not any(name.startswith("textual") for name in imported_names)
    assert "ChaChaNotes_DB" not in source
    assert "sqlite3" not in imported_names


# TASK-33621.8: the exact orphan a stored staged-Library turn carries.
_UNRECORDED_RETRIEVAL = (
    "console-trace:3c8cc2a1-d7a3-4ca3-b775-f178ffe438bb:retrieval:retrieval_completed"
)


@pytest.mark.parametrize("profile", list(trajectory_export.TraceExportProfile))
def test_stored_unrecorded_references_are_cleared_disclosed_and_reimportable(
    profile: trajectory_export.TraceExportProfile,
) -> None:
    attached_id = "console-trace:3c8cc2a1:retrieval:context_attached"
    attached = _record(
        event_id=attached_id,
        kind="context_attached",
        parent_event_id=_UNRECORDED_RETRIEVAL,
        source_event_id=_UNRECORDED_RETRIEVAL,
    )
    injected = _record(
        event_id="console-trace:3c8cc2a1:retrieval:context_injected",
        seq=2,
        kind="context_injected",
        parent_event_id=attached_id,
        source_event_id=attached_id,
    )

    payload = _build(
        _snapshot(attached, injected),
        profile,
        confirm_full=profile is trajectory_export.TraceExportProfile.FULL_TRACE,
    )

    first, second = payload["events"][:2]
    assert first["parent_event_id"] is None
    assert first["source_event_id"] is None
    # The recorded link is kept; only the unrecorded one is cleared.
    assert second["parent_event_id"] == first["event_id"]
    assert second["source_event_id"] == first["event_id"]
    for field in ("parent_event_id", "source_event_id"):
        assert first["field_states"][field] == "not_available"
        assert {
            "event_id": first["event_id"],
            "field": field,
            "state": "not_available",
        } in payload["manifest"]["missing_metadata"]
        assert {
            "event_id": first["event_id"],
            "field": field,
            "state": "not_available",
            "reason": "reference_not_recorded",
        } in payload["manifest"]["redaction_provenance"]
    assert _UNRECORDED_RETRIEVAL not in json.dumps(payload)
    imported = trajectory_import.load_imported_trace(payload)
    assert (
        imported.manifest["missing_metadata"] == payload["manifest"]["missing_metadata"]
    )


def test_clearing_unrecorded_references_logs_metadata_only() -> None:
    """A future producer that leaves orphans must surface in the app log, not
    only inside shared bundles; the line carries field names and event kinds,
    never event ids or content (review follow-up, TASK-33621.8)."""
    from loguru import logger

    attached = _record(
        event_id="console-trace:3c8cc2a1:retrieval:context_attached",
        kind="context_attached",
        parent_event_id=_UNRECORDED_RETRIEVAL,
        source_event_id=_UNRECORDED_RETRIEVAL,
    )
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)), level="WARNING")
    try:
        trajectory_export.preflight_trace_export(_snapshot(attached))
        orphaned = "".join(records)
        records.clear()
        trajectory_export.preflight_trace_export(_snapshot(_record()))
        clean = "".join(records)
    finally:
        logger.remove(sink_id)

    assert "cleared 2 reference(s) to unrecorded events" in orphaned
    assert "parent_event_id" in orphaned and "source_event_id" in orphaned
    assert "context_attached" in orphaned
    assert "3c8cc2a1" not in orphaned
    assert "A bounded diagnostic summary" not in orphaned
    assert clean == ""


@pytest.mark.parametrize("profile", list(trajectory_export.TraceExportProfile))
def test_credential_block_raises_its_own_error_type(
    profile: trajectory_export.TraceExportProfile,
) -> None:
    """Review follow-up (TASK-33621.8): the final credential check has its own
    error type, so the export dialog can say what stopped the export instead
    of the generic "could not be checked" copy and its retry advice.

    A credential-looking field name survives every scrubber (field names are
    kept verbatim), so it is blocked under every profile alike.
    """
    blocked = _record(
        field_states={
            "content_preview": "observed",
            "payload": "observed",
            "password=hunter22": "observed",
        }
    )

    with pytest.raises(
        trajectory_export.TraceExportCredentialBlocked, match="credential material"
    ) as raised:
        _build(
            _snapshot(blocked),
            profile,
            confirm_full=profile is trajectory_export.TraceExportProfile.FULL_TRACE,
        )
    assert isinstance(raised.value, trajectory_export.TrajectoryExportError)
    # Every other validation failure keeps the base type.
    with pytest.raises(trajectory_export.TrajectoryExportError) as duplicate:
        _build(_snapshot(_record(), _record(seq=2)), profile, confirm_full=True)
    assert not isinstance(
        duplicate.value, trajectory_export.TraceExportCredentialBlocked
    )
