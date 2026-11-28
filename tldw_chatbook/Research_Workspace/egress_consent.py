"""Consent for an exact route that may send Local source text off-device."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING

from .chat_models import ResearchChatTurnRequest
from .contracts import WorkspaceDataSource

if TYPE_CHECKING:
    from .studio_models import ResearchStudioRequest

MAX_EGRESS_CONSENTS = 16
_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")


def validate_egress_consents(fingerprints: tuple[str, ...]) -> None:
    if len(fingerprints) > MAX_EGRESS_CONSENTS:
        raise ValueError("chat consent exceeds the retained route limit")
    if any(
        not isinstance(value, str) or not _FINGERPRINT.fullmatch(value)
        for value in fingerprints
    ):
        raise ValueError("chat consent fingerprint is invalid")
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("chat consent fingerprints must be unique")


@dataclass(frozen=True, slots=True)
class ResearchEgressPreflight:
    fingerprint: str
    provider: str
    endpoint_class: str
    redaction_policy: str
    source_body_mode: str
    source_count: int
    source_types: tuple[str, ...]


class ResearchEgressConsentService:
    def preflight(
        self,
        request: ResearchChatTurnRequest | ResearchStudioRequest,
        approved: tuple[str, ...] = (),
    ) -> ResearchEgressPreflight | None:
        """Describe a remote transfer without including any user or source text."""
        route = request.processing_route
        sources = request.retrieval_sources
        if (
            request.workspace.data_source is WorkspaceDataSource.SERVER
            or not route.remote
            or not sources
        ):
            return None
        if (
            route.endpoint_class not in {"public_network", "private_network"}
            or not route.endpoint_fingerprint
            or not route.provider
        ):
            raise ValueError("Remote processing destination could not be verified")
        payload = (
            request.workspace.workspace_id,
            route.provider,
            route.endpoint_class,
            route.endpoint_fingerprint,
            route.redaction_policy,
            route.source_body_mode,
        )
        fingerprint = sha256(json.dumps(payload).encode()).hexdigest()
        if fingerprint in approved:
            return None
        return ResearchEgressPreflight(
            fingerprint,
            route.provider,
            route.endpoint_class,
            route.redaction_policy,
            route.source_body_mode,
            len(sources),
            tuple(sorted({source.source_type for source in sources})),
        )

    def is_approved(
        self,
        request: ResearchChatTurnRequest | ResearchStudioRequest,
        approved: tuple[str, ...],
    ) -> bool:
        return self.preflight(request, approved) is None

    def approve(
        self, preflight: ResearchEgressPreflight, approved: tuple[str, ...]
    ) -> tuple[str, ...]:
        validate_egress_consents((preflight.fingerprint,))
        retained = tuple(value for value in approved if value != preflight.fingerprint)
        return (*retained, preflight.fingerprint)[-MAX_EGRESS_CONSENTS:]
