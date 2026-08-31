"""Strict external configuration boundary for Console trace privacy."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError


class TracePrivacyConfig(BaseModel):
    """Validated privacy fields read from the Console config section."""

    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    exchange_capture_pii_redaction: bool = False
    trace_viewer_profile: Literal["safe", "full"] = "safe"
    trace_viewer_profile_version: Literal[1] | None = None

    @property
    def effective_viewer_profile(self) -> Literal["safe", "full"]:
        """Return the version-gated viewer profile.

        Returns:
            The stored profile only for the explicit v1 format; otherwise Safe.
        """

        if self.trace_viewer_profile_version != 1:
            return "safe"
        return self.trace_viewer_profile


def validate_trace_privacy_config(value: object) -> TracePrivacyConfig:
    """Validate raw Console privacy config with conservative fallback.

    Args:
        value: Raw external Console config section.

    Returns:
        Strict validated fields, or the all-Safe defaults if any privacy field
        is malformed. Unknown Console settings are ignored.
    """

    try:
        return TracePrivacyConfig.model_validate(value)
    except ValidationError:
        return TracePrivacyConfig()
