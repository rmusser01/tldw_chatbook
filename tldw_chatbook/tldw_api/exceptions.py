# tldw_chatbook/tldw_api/exceptions.py
#
#
#######################################################################################################################
#
# Functions:

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sync_schemas import SyncPersonalContextBootstrapAttention


class TLDWAPIError(Exception):
    """Base exception for tldw_api errors."""

    pass


class APIConnectionError(TLDWAPIError):
    """Raised for network or connection issues."""

    pass


class APIRequestError(TLDWAPIError):
    """Raised for errors in constructing or sending the request (e.g., bad data)."""

    def __init__(self, message: str, response_data: dict | None = None):
        super().__init__(message)
        self.response_data = response_data or {}


class APIResponseError(TLDWAPIError):
    """Raised for non-2xx responses or issues parsing the response."""

    def __init__(self, status_code: int, message: str, response_data: dict = None):
        super().__init__(f"API Error {status_code}: {message}")
        self.status_code = status_code
        self.response_data = response_data or {}


class PersonalContextBootstrapAttentionError(TLDWAPIError):
    """Raise one strictly validated, content-free bootstrap attention result."""

    def __init__(self, attention: "SyncPersonalContextBootstrapAttention") -> None:
        super().__init__("personal_context_bootstrap_attention_required")
        self.attention = attention


class AuthenticationError(TLDWAPIError):
    """Raised for authentication failures."""

    def __init__(self, message: str, response_data: dict | None = None):
        super().__init__(message)
        self.response_data = response_data or {}


#
# End of tldw_chatbook/tldw_api/exceptions.py
########################################################################################################################
