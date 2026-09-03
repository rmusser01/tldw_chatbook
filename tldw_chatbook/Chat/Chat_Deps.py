# ---------------- Exceptions ----------------------------
class ChatAPIError(Exception):
    """Base exception for chat API call errors."""

    def __init__(
        self,
        message="An error occurred during the chat API call.",
        status_code=500,
        provider=None,
    ):
        self.message = message
        self.status_code = status_code  # Suggested HTTP status code for the endpoint
        self.provider = provider
        super().__init__(self.message)


class ChatAuthenticationError(ChatAPIError):
    """Exception for authentication issues (e.g., invalid API key)."""

    def __init__(
        self, message="Authentication failed with the chat provider.", provider=None
    ):
        super().__init__(message, status_code=401, provider=provider)  # Default to 401


class ChatConfigurationError(ChatAPIError):
    """Exception for configuration issues (e.g., missing key, invalid model)."""

    def __init__(self, message="Chat provider configuration error.", provider=None):
        super().__init__(message, status_code=500, provider=provider)  # Default to 500


class ChatBadRequestError(ChatAPIError):
    """Exception for bad requests sent to the chat provider (e.g., invalid params).

    ``status_code`` defaults to 400 but accepts the REAL 4xx: the dispatcher
    previously collapsed every 4xx into a hardcoded 400, which erased the
    difference between "our request is malformed" and "the account is out of
    money" (402/403) -- making the credit-terminal fallback trigger
    (`Agents/fallback_chain.is_credit_terminal`) unreachable with real traffic
    (TASK-25902 review C3c). The type is unchanged so existing catchers keep
    working; only the carried status got honest.
    """

    def __init__(
        self,
        message="Invalid request sent to the chat provider.",
        provider=None,
        status_code=400,
    ):
        super().__init__(message, status_code=status_code, provider=provider)


class ChatRateLimitError(ChatAPIError):
    """Exception for rate limit errors from the chat provider.

    ``retry_after`` carries the provider's own Retry-After header, in seconds,
    when one was sent and was numeric; None otherwise. Consumed by
    `Agents/model_retry.retry_delay_seconds`, which honours it over computed
    backoff (TASK-25901 AC#2 -- previously the classifier honoured the
    attribute but nothing in the stack ever set it, review I3).
    """

    def __init__(
        self,
        message="Rate limit exceeded with the chat provider.",
        provider=None,
        retry_after=None,
    ):
        super().__init__(message, status_code=429, provider=provider)
        self.retry_after = retry_after


class ChatProviderError(ChatAPIError):
    """Exception for general errors reported by the chat provider API."""

    def __init__(
        self,
        message="Error received from the chat provider API.",
        status_code=502,
        provider=None,
        details=None,
    ):
        # 502 Bad Gateway often suitable for upstream errors
        self.details = details  # Store original error if available
        super().__init__(message, status_code=status_code, provider=provider)


# ---------------- End of Exceptions ----------------------------
