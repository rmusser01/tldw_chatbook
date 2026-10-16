# tldw_chatbook/Web_Server/webauthn_service.py
"""WebAuthn (passkey) registration and login ceremonies for served mode (ADR-235).

The service owns a ``fido2.server.Fido2Server`` for the derived RP and drives
both ceremonies against it.  Pending challenges live in an in-memory,
digest-keyed ``OrderedDict`` that mirrors ``WebAuthManager._bootstraps``:
one-use pop, 60 s TTL, prune-on-use, and a 64-entry cap, all on an injectable
clock (ADR-233 discipline).  Verified credentials persist through
``WebAuthnCredentialStore`` as public material only.

Every ceremony failure -- bad base64, challenge misuse, origin mismatch,
unknown credential, invalid signature, or a sign-count regression -- surfaces
as ``Canvas.web_auth.AuthenticationError`` with a content-free message.
Persistence I/O errors after successful verification are not ceremony
failures and propagate unchanged.  Logs carry the credential-id hex prefix
(first 8 chars), the RP id, and outcome words only; never challenges, keys,
or client data.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import secrets
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from fido2 import cbor
from fido2.cose import CoseKey
from fido2.server import Fido2Server
from fido2.utils import websafe_encode
from fido2.webauthn import (
    AttestationObject,
    AttestedCredentialData,
    AuthenticatorData,
    PublicKeyCredentialRpEntity,
    PublicKeyCredentialUserEntity,
)

from ..Canvas.web_auth import AuthenticationError
from .webauthn_store import StoredCredential, WebAuthnCredentialStore

__all__ = ["PasskeysUnavailable", "WebAuthnService"]

logger = logging.getLogger(__name__)

_CHALLENGE_TTL_SECONDS = 60.0
_MAX_CHALLENGE_STATES = 64
_CREDENTIAL_ID_PREFIX_CHARS = 8
_LOCALHOST = "localhost"


class PasskeysUnavailable(Exception):
    """Raised when RP derivation fails (no public_url / unusable host)."""


@dataclass
class _ChallengeState:
    """A pending ceremony challenge; never logged or persisted."""

    expiry: float
    kind: str  # "register" | "login"
    label: str = ""
    fido2_state: Mapping[str, Any] = field(repr=False, default_factory=dict)


def _challenge_key(challenge: str) -> str:
    return hashlib.sha256(challenge.encode("utf-8")).hexdigest()


def _derive_rp_host(public_url: str | None) -> str:
    """Return the registrable RP host, refusing unusable public URLs.

    WebAuthn RP ids must be registrable domains: an IP literal or localhost
    cannot anchor an origin, so ``PasskeysUnavailable`` is raised for them and
    callers can degrade to token-only login.
    """
    if not public_url:
        raise PasskeysUnavailable("passkeys require a public URL")
    try:
        host = urlsplit(public_url).hostname
    except ValueError:
        raise PasskeysUnavailable("passkeys require a public URL") from None
    if not host:
        raise PasskeysUnavailable("passkeys require a public URL")
    host = host.strip("[]").lower()
    if host == _LOCALHOST or ":" in host:
        # localhost or an IPv6 literal (brackets already stripped by hostname)
        raise PasskeysUnavailable("passkeys require a registrable domain")
    if all(character.isdigit() or character == "." for character in host):
        raise PasskeysUnavailable("passkeys require a registrable domain")  # IPv4
    return host


class WebAuthnService:
    """WebAuthn ceremonies over the on-disk credential store."""

    def __init__(
        self,
        *,
        public_url: str | None,
        store: WebAuthnCredentialStore,
        clock: Callable[[], float] = time.time,
        challenge_factory: Callable[[], str] = lambda: secrets.token_urlsafe(32),
    ) -> None:
        self._rp_id = _derive_rp_host(public_url)
        self._store = store
        self._clock = clock
        self._challenge_factory = challenge_factory
        self._states: OrderedDict[str, _ChallengeState] = OrderedDict()
        self._server = Fido2Server(
            PublicKeyCredentialRpEntity(name=self._rp_id, id=self._rp_id),
            attestation="none",
        )
        self._user = PublicKeyCredentialUserEntity(
            id=hashlib.sha256(b"tldw-chatbook-passkey-user").digest()[:16],
            name="chatbook",
            display_name="Chatbook",
        )

    # ------------------------------------------------------------------- RP

    @property
    def available(self) -> bool:
        """Whether RP derivation succeeded (always True past construction)."""
        return self._rp_id is not None

    @property
    def rp_id(self) -> str:
        return self._rp_id

    # -------------------------------------------------------- challenge book

    def _prune_states(self, now: float) -> None:
        expired = [key for key, state in self._states.items() if now >= state.expiry]
        for key in expired:
            self._states.pop(key, None)

    def _issue(
        self,
        kind: str,
        label: str,
        begin: Callable[[bytes], tuple[Any, Any]],
    ) -> dict:
        """Run one begin ceremony and file its challenge under a digest key."""
        now = self._clock()
        self._prune_states(now)
        while len(self._states) >= _MAX_CHALLENGE_STATES:
            self._states.popitem(last=False)
        challenge = self._challenge_factory()
        options, fido2_state = begin(challenge.encode("utf-8"))
        key = _challenge_key(challenge)
        self._states[key] = _ChallengeState(
            expiry=now + _CHALLENGE_TTL_SECONDS,
            kind=kind,
            label=label,
            fido2_state=dict(fido2_state),
        )
        return {"publicKey": dict(options)["publicKey"], "challenge_id": key}

    def _take_state(self, challenge_id: str, kind: str) -> _ChallengeState:
        """Pop and validate a pending challenge (one use, TTL, kind)."""
        now = self._clock()
        self._prune_states(now)
        state = self._states.pop(challenge_id, None)
        if state is None or now >= state.expiry or state.kind != kind:
            raise AuthenticationError("authentication denied")
        return state

    # ------------------------------------------------------------- registry

    def _attested_credentials(
        self, creds: Mapping[str, StoredCredential]
    ) -> list[tuple[str, AttestedCredentialData]]:
        """Rebuild fido2 AttestedCredentialData records from stored material."""
        rebuilt: list[tuple[str, AttestedCredentialData]] = []
        for credential_id_hex, record in creds.items():
            try:
                cose_map = cbor.decode(base64.b64decode(record.public_key_b64))
                rebuilt.append(
                    (
                        credential_id_hex,
                        AttestedCredentialData.create(
                            bytes.fromhex(record.aaguid),
                            bytes.fromhex(record.credential_id_hex),
                            CoseKey.parse(cose_map),
                        ),
                    )
                )
            except (ValueError, TypeError, KeyError):
                logger.warning(
                    "webauthn skip unparsable credential credential=%s rp=%s "
                    "outcome=drop",
                    record.credential_id_hex[:_CREDENTIAL_ID_PREFIX_CHARS],
                    self._rp_id,
                )
        return rebuilt

    @staticmethod
    def _prefix(credential_id_hex: str) -> str:
        return credential_id_hex[:_CREDENTIAL_ID_PREFIX_CHARS]

    # ----------------------------------------------------------- ceremonies

    def register_begin(self, *, label: str) -> dict:
        """Start registration; returns publicKey options + one-use challenge id."""
        existing = [
            attested
            for _, attested in self._attested_credentials(self._store.load())
        ]
        return self._issue(
            "register",
            label,
            lambda challenge: self._server.register_begin(
                user=self._user, credentials=existing, challenge=challenge
            ),
        )

    def register_complete(
        self, challenge_id: str, client_data_json: str, attestation_object_b64: str
    ) -> dict:
        """Verify a registration response and persist the new credential."""
        prefix = "-"
        try:
            attestation_bytes = base64.b64decode(attestation_object_b64, validate=True)
            credential_id = AttestationObject(
                attestation_bytes
            ).auth_data.credential_data.credential_id
            prefix = self._prefix(credential_id.hex())
            state = self._take_state(challenge_id, "register")
            response = {
                "id": websafe_encode(credential_id),
                "rawId": websafe_encode(credential_id),
                "type": "public-key",
                "response": {
                    "clientDataJSON": websafe_encode(
                        client_data_json.encode("utf-8")
                    ),
                    "attestationObject": websafe_encode(attestation_bytes),
                },
            }
            auth_data = self._server.register_complete(state.fido2_state, response)
            credential_data = auth_data.credential_data
            if credential_data.credential_id != credential_id:
                raise ValueError("credential id mismatch")
        except Exception:
            logger.info(
                "webauthn register outcome=denied credential=%s rp=%s",
                prefix,
                self._rp_id,
            )
            raise AuthenticationError("authentication denied") from None

        credential_id_hex = credential_data.credential_id.hex()
        now = self._clock()
        creds = self._store.load()
        if credential_id_hex in creds:
            logger.info(
                "webauthn register outcome=duplicate credential=%s rp=%s",
                prefix,
                self._rp_id,
            )
            raise AuthenticationError("authentication denied")
        creds[credential_id_hex] = StoredCredential(
            credential_id_hex=credential_id_hex,
            public_key_b64=base64.b64encode(
                cbor.encode(credential_data.public_key)
            ).decode(),
            rp_id=self._rp_id,
            sign_count=auth_data.counter,
            label=state.label,
            aaguid=credential_data.aaguid.hex(),
            created_at=now,
            last_used_at=now,
        )
        self._store.save(creds)
        logger.info(
            "webauthn register outcome=ok credential=%s rp=%s", prefix, self._rp_id
        )
        return {"credential_id": prefix, "label": state.label}

    def login_begin(self) -> dict:
        """Start a discoverable login (allow_credentials is empty)."""
        return self._issue(
            "login",
            "",
            lambda challenge: self._server.authenticate_begin(
                credentials=[], challenge=challenge
            ),
        )

    def login_complete(
        self,
        challenge_id: str,
        client_data_json: str,
        authenticator_data_b64: str,
        signature_b64: str,
    ) -> StoredCredential:
        """Verify an assertion; returns the updated stored credential."""
        try:
            auth_data_bytes = base64.b64decode(authenticator_data_b64, validate=True)
            signature = base64.b64decode(signature_b64, validate=True)
            auth_data = AuthenticatorData(auth_data_bytes)
        except Exception:
            logger.info("webauthn login outcome=denied rp=%s", self._rp_id)
            raise AuthenticationError("authentication denied") from None

        prefix = "-"
        try:
            state = self._take_state(challenge_id, "login")
            creds = self._store.load()
            matched_hex: str | None = None
            for credential_id_hex, attested in self._attested_credentials(creds):
                response = {
                    "id": websafe_encode(attested.credential_id),
                    "rawId": websafe_encode(attested.credential_id),
                    "type": "public-key",
                    "response": {
                        "clientDataJSON": websafe_encode(
                            client_data_json.encode("utf-8")
                        ),
                        "authenticatorData": websafe_encode(auth_data_bytes),
                        "signature": websafe_encode(signature),
                    },
                }
                try:
                    self._server.authenticate_complete(
                        state.fido2_state, [attested], response
                    )
                except ValueError:
                    continue  # not this credential (id or signature mismatch)
                matched_hex = credential_id_hex
                break
            if matched_hex is None:
                raise ValueError("unknown credential")
        except Exception:
            logger.info(
                "webauthn login outcome=denied credential=%s rp=%s",
                prefix,
                self._rp_id,
            )
            raise AuthenticationError("authentication denied") from None

        prefix = self._prefix(matched_hex)
        stored = creds[matched_hex]
        if stored.sign_count and auth_data.counter <= stored.sign_count:
            logger.info(
                "webauthn login outcome=denied credential=%s rp=%s reason=counter",
                prefix,
                self._rp_id,
            )
            raise AuthenticationError("authentication denied")
        now = self._clock()
        updated = StoredCredential(
            credential_id_hex=stored.credential_id_hex,
            public_key_b64=stored.public_key_b64,
            rp_id=stored.rp_id,
            sign_count=auth_data.counter,
            label=stored.label,
            aaguid=stored.aaguid,
            created_at=stored.created_at,
            last_used_at=now,
        )
        creds[matched_hex] = updated
        self._store.save(creds)
        logger.info(
            "webauthn login outcome=ok credential=%s rp=%s", prefix, self._rp_id
        )
        return updated

    # ------------------------------------------------------------- registry

    def list_credentials(self) -> list[dict]:
        """Public metadata for management UIs; no key material."""
        creds = self._store.load()
        return [
            {
                "credential_id_prefix": self._prefix(record.credential_id_hex),
                "label": record.label,
                "created_at": record.created_at,
                "last_used_at": record.last_used_at,
            }
            for record in sorted(creds.values(), key=lambda item: item.created_at)
        ]

    def remove_credential(self, credential_id_hex: str) -> bool:
        creds = self._store.load()
        if credential_id_hex not in creds:
            return False
        del creds[credential_id_hex]
        self._store.save(creds)
        logger.info(
            "webauthn remove outcome=ok credential=%s rp=%s",
            self._prefix(credential_id_hex),
            self._rp_id,
        )
        return True
