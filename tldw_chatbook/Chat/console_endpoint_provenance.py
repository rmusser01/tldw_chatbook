"""Endpoint lifetime markers shared by Console runtime and durable boundaries."""

from __future__ import annotations

from enum import StrEnum


class ConsoleEndpointProvenance(StrEnum):
    """Whether an effective endpoint may cross a process-lifetime boundary."""

    DURABLE_CONFIGURATION = "durable_configuration"
    EPHEMERAL_SESSION = "ephemeral_session"


# Deliberately not a URL: recovered checkpoints can identify omitted live
# authority and fail closed without retaining or reconstructing its target.
EPHEMERAL_SESSION_ENDPOINT_OMITTED = "ephemeral_session_endpoint_omitted"
