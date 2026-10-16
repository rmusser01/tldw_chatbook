# Tests/Web_Server/test_webauthn_service.py
"""In-process WebAuthn ceremonies for the served passkey service (ADR-235).

Both halves of the ceremony -- relying party and a minimal software
authenticator -- are built in-process.  The fake authenticator produces
``none``-format attestations and ES256 (P-256) assertions with UP|UV flags,
exactly what a browser-managed authenticator would send for this RP.
"""

from __future__ import annotations

import base64
import hashlib
import itertools
import json
import os
import stat
import struct
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from fido2 import cbor

from tldw_chatbook.Canvas.web_auth import AuthenticationError
from tldw_chatbook.Web_Server.webauthn_store import (
    StoredCredential,
    WebAuthnCredentialStore,
)
from tldw_chatbook.Web_Server.webauthn_service import (
    PasskeysUnavailable,
    WebAuthnService,
)

RP_ID = "chatbook.example"
ORIGIN = "https://chatbook.example"
RP_ID_HASH = hashlib.sha256(RP_ID.encode()).digest()

_FLAG_UP_UV = 0x05  # UP | UV
_FLAG_UP_UV_AT = 0x45  # UP | UV | AT


class _FakeClock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _challenge_factory():
    counter = itertools.count()
    return lambda: f"challenge-{next(counter):04d}-0123456789abcdef"


def _service(tmp_path: Path, clock: _FakeClock) -> WebAuthnService:
    return WebAuthnService(
        public_url=f"https://{RP_ID}",
        store=WebAuthnCredentialStore(tmp_path / "webauthn_credentials.json"),
        clock=clock,
        challenge_factory=_challenge_factory(),
    )


def _cose_key_dict(public_key: ec.EllipticCurvePublicKey) -> dict:
    numbers = public_key.public_numbers()
    return {
        1: 2,  # kty: EC2
        3: -7,  # alg: ES256
        -1: 1,  # crv: P-256
        -2: numbers.x.to_bytes(32, "big"),
        -3: numbers.y.to_bytes(32, "big"),
    }


def _auth_data_bytes(
    flags: int, counter: int, credential_id: bytes | None, cose_key: bytes | None
) -> bytes:
    data = RP_ID_HASH + struct.pack(">BI", flags, counter)
    if credential_id is not None:
        assert cose_key is not None
        data += (
            b"\x00" * 16
            + struct.pack(">H", len(credential_id))
            + credential_id
            + cose_key
        )
    return data


def _client_data_json(kind: str, challenge: str, origin: str) -> str:
    return json.dumps({"type": kind, "challenge": challenge, "origin": origin})


def _register(
    service: WebAuthnService,
    *,
    label: str = "YubiKey 5",
    origin: str = ORIGIN,
    credential_id: bytes = b"\x02" * 32,
):
    """Drive register_begin -> register_complete with a fresh P-256 key."""
    begin = service.register_begin(label=label)
    # Discoverable credentials are mandated (ADR-235): roaming authenticators
    # otherwise create server-side credentials that login (allowCredentials [])
    # can never offer.
    assert begin["publicKey"]["authenticatorSelection"] == {
        "residentKey": "required",
        "userVerification": "preferred",
        "requireResidentKey": True,
    }
    private_key = ec.generate_private_key(ec.SECP256R1())
    cose_key = cbor.encode(_cose_key_dict(private_key.public_key()))
    auth_data = _auth_data_bytes(_FLAG_UP_UV_AT, 0, credential_id, cose_key)
    attestation_object = cbor.encode(
        {"fmt": "none", "attStmt": {}, "authData": auth_data}
    )
    client_data_json = _client_data_json(
        "webauthn.create", begin["publicKey"]["challenge"], origin
    )
    result = service.register_complete(
        begin["challenge_id"],
        client_data_json,
        base64.b64encode(attestation_object).decode(),
    )
    return private_key, credential_id, result


def _assertion(
    service: WebAuthnService,
    private_key: ec.EllipticCurvePrivateKey,
    credential_id: bytes,
    *,
    counter: int,
    origin: str = ORIGIN,
):
    """Drive login_begin -> login_complete, returning the completion outcome."""
    begin = service.login_begin()
    assert begin["publicKey"]["allowCredentials"] == []
    client_data_json = _client_data_json(
        "webauthn.get", begin["publicKey"]["challenge"], origin
    )
    auth_data = _auth_data_bytes(_FLAG_UP_UV, counter, None, None)
    signature = private_key.sign(
        auth_data + hashlib.sha256(client_data_json.encode()).digest(),
        ec.ECDSA(hashes.SHA256()),
    )
    return service.login_complete(
        begin["challenge_id"],
        client_data_json,
        base64.b64encode(auth_data).decode(),
        base64.b64encode(signature).decode(),
    )


def test_registration_and_login_round_trip(tmp_path: Path) -> None:
    clock = _FakeClock()
    service = _service(tmp_path, clock)
    assert service.available is True

    private_key, credential_id, result = _register(service)
    assert result["label"] == "YubiKey 5"
    assert result["credential_id"] == credential_id.hex()[:8]

    listing = service.list_credentials()
    assert [entry["label"] for entry in listing] == ["YubiKey 5"]
    assert listing[0]["credential_id_prefix"] == credential_id.hex()[:8]

    credential = _assertion(service, private_key, credential_id, counter=1)
    assert isinstance(credential, StoredCredential)
    assert credential.credential_id_hex == credential_id.hex()
    assert credential.sign_count == 1
    assert credential.last_used_at == clock()

    store_path = tmp_path / "webauthn_credentials.json"
    assert store_path.exists()
    mode = stat.S_IMODE(os.stat(store_path).st_mode)
    assert mode == 0o600
    payload = json.loads(store_path.read_text())
    record = payload[credential_id.hex()]
    assert set(record) == {
        "credential_id_hex",
        "public_key_b64",
        "rp_id",
        "sign_count",
        "label",
        "aaguid",
        "created_at",
        "last_used_at",
    }
    assert record["rp_id"] == RP_ID
    # Public material only: the COSE key decodes to a P-256 public key map and
    # nothing private (no PEM/seed/priv) ever reaches disk.
    cose_map = cbor.decode(base64.b64decode(record["public_key_b64"]))
    assert set(cose_map) <= {1, 3, -1, -2, -3}  # public EC2 parts only, no -4/d
    assert cose_map[1] == 2 and cose_map[3] == -7


def test_login_grants_nothing_directly_but_service_validates_origin(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path, _FakeClock())
    private_key, credential_id, _ = _register(service)

    with pytest.raises(AuthenticationError):
        _assertion(
            service,
            private_key,
            credential_id,
            counter=1,
            origin="https://evil.example",
        )
    # A replay with the correct origin still works afterwards.
    credential = _assertion(service, private_key, credential_id, counter=1)
    assert credential.credential_id_hex == credential_id.hex()


def test_challenge_one_use_and_ttl(tmp_path: Path) -> None:
    clock = _FakeClock()
    service = _service(tmp_path, clock)
    begin = service.register_begin(label="ttl probe")
    challenge = begin["publicKey"]["challenge"]
    private_key = ec.generate_private_key(ec.SECP256R1())
    cose_key = cbor.encode(_cose_key_dict(private_key.public_key()))
    credential_id = b"\x07" * 32
    auth_data = _auth_data_bytes(_FLAG_UP_UV_AT, 0, credential_id, cose_key)
    attestation_object = cbor.encode(
        {"fmt": "none", "attStmt": {}, "authData": auth_data}
    )
    client_data_json = _client_data_json("webauthn.create", challenge, ORIGIN)

    # Expired challenges are refused before any credential work happens.
    clock.advance(60.0)
    with pytest.raises(AuthenticationError):
        service.register_complete(
            begin["challenge_id"],
            client_data_json,
            base64.b64encode(attestation_object).decode(),
        )

    # A fresh challenge works once, then the id is dead.  The reuse attempt
    # carries a FRESH credential id so the duplicate-credential guard cannot
    # mask a popped-state regression.
    clock.advance(1.0)
    begin = service.register_begin(label="one use")
    client_data_json = _client_data_json(
        "webauthn.create", begin["publicKey"]["challenge"], ORIGIN
    )
    service.register_complete(
        begin["challenge_id"],
        client_data_json,
        base64.b64encode(attestation_object).decode(),
    )
    reuse_key = ec.generate_private_key(ec.SECP256R1())
    reuse_attestation = cbor.encode(
        {
            "fmt": "none",
            "attStmt": {},
            "authData": _auth_data_bytes(
                _FLAG_UP_UV_AT,
                0,
                b"\x08" * 32,
                cbor.encode(_cose_key_dict(reuse_key.public_key())),
            ),
        }
    )
    with pytest.raises(AuthenticationError):
        service.register_complete(
            begin["challenge_id"],
            client_data_json,
            base64.b64encode(reuse_attestation).decode(),
        )


def test_sign_count_regression_rejects(tmp_path: Path) -> None:
    service = _service(tmp_path, _FakeClock())
    private_key, credential_id, _ = _register(service)

    first = _assertion(service, private_key, credential_id, counter=5)
    assert first.sign_count == 5

    with pytest.raises(AuthenticationError):
        _assertion(service, private_key, credential_id, counter=5)

    forward = _assertion(service, private_key, credential_id, counter=6)
    assert forward.sign_count == 6


def test_ip_or_missing_public_url_refuses(tmp_path: Path) -> None:
    store = WebAuthnCredentialStore(tmp_path / "webauthn_credentials.json")
    with pytest.raises(PasskeysUnavailable):
        WebAuthnService(public_url=None, store=store)
    with pytest.raises(PasskeysUnavailable):
        WebAuthnService(public_url="https://10.0.0.5:8443", store=store)
    with pytest.raises(PasskeysUnavailable):
        WebAuthnService(public_url="https://localhost:8443", store=store)
    with pytest.raises(PasskeysUnavailable):
        WebAuthnService(public_url="https://[2001:db8::1]:8443", store=store)
    with pytest.raises(PasskeysUnavailable):
        WebAuthnService(public_url="", store=store)


def test_remove_credential(tmp_path: Path) -> None:
    service = _service(tmp_path, _FakeClock())
    private_key, credential_id, _ = _register(service)

    assert service.remove_credential(credential_id.hex()) is True
    assert service.remove_credential(credential_id.hex()) is False
    assert service.list_credentials() == []

    with pytest.raises(AuthenticationError):
        _assertion(service, private_key, credential_id, counter=1)


def test_store_fail_safe(tmp_path: Path) -> None:
    path = tmp_path / "webauthn_credentials.json"
    store = WebAuthnCredentialStore(path)
    assert store.load() == {}

    path.write_text("{not json")
    assert store.load() == {}

    path.write_text('["a", 1]')
    assert store.load() == {}

    path.write_text(json.dumps({"deadbeef": {"label": "missing fields"}}))
    assert store.load() == {}

    good = StoredCredential(
        credential_id_hex="cafe",
        public_key_b64="AAA=",
        rp_id=RP_ID,
        sign_count=0,
        label="key",
        aaguid="00" * 16,
        created_at=1.0,
        last_used_at=1.0,
    )
    store.save({"cafe": good})
    assert store.load() == {"cafe": good}
