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
    """Exception for configuration issues (e.g., missing key, invalid model).

    ``status_code`` defaults to 500 but accepts ``None`` for a failure that
    never reached the provider at all -- a client-side request-preparation
    error (task-32342). Carrying no status keeps such a failure out of every
    surface that reads one as "the provider answered".
    """

    def __init__(
        self,
        message: str = "Chat provider configuration error.",
        provider: str | None = None,
        status_code: int | None = 500,
        field: str | None = None,
    ) -> None:
        """Build a configuration failure, with or without a provider status.

        Args:
            message: Human-readable description of the misconfiguration.
            provider: Provider key the failure belongs to, when known.
            status_code: The HTTP status the provider returned, or ``None``
                when the failure never reached it -- a client-side local
                configuration or data error raised during request
                preparation. Callers that classify by status must treat
                ``None`` as "no provider verdict", not as a server error.
            field: For a request stopped by a local check, the request field
                that failed it (``"messages"``), so the user copy can name it
                without showing the message (TASK-32369).
        """
        super().__init__(message, status_code=status_code, provider=provider)
        self.field = field


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


class ChatModelUnavailableError(ChatProviderError):
    """Provider explicitly classified this model as unavailable."""

    def __init__(
        self,
        message="The requested model is unavailable.",
        provider=None,
        status_code=404,
    ):
        super().__init__(message, status_code=status_code, provider=provider)


def project_provider_error(exc: BaseException, provider: str) -> ChatAPIError | None:
    """Preserve known retry semantics with content-free diagnostic bodies."""
    status = getattr(exc, "status_code", None)
    if isinstance(exc, ChatModelUnavailableError):
        return ChatModelUnavailableError(provider=provider, status_code=status)
    if isinstance(exc, ChatAuthenticationError):
        return ChatAuthenticationError(provider=provider)
    if isinstance(exc, ChatBadRequestError):
        return ChatBadRequestError(provider=provider, status_code=status)
    if isinstance(exc, ChatConfigurationError):
        return ChatConfigurationError(provider=provider, status_code=status)
    if isinstance(exc, ChatRateLimitError):
        return ChatRateLimitError(provider=provider, retry_after=exc.retry_after)
    return None


def model_unavailable_error(provider: str, status: int, payload: object):
    """Map documented machine codes only; status and prose never suffice."""
    if status not in {400, 404} or not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if not isinstance(error, dict):
        return None
    # The OpenAI-compatible model_not_found code is explicit. Providers with
    # ambiguous not_found_error envelopes remain terminal until a distinct
    # machine-code contract is available.
    if (
        str(provider).lower()
        in {
            "openai",
            "groq",
            "deepseek",
            "moonshot",
            "custom-openai-api",
            "custom-openai-api-2",
            "custom-hosted",
        }
        and error.get("code") == "model_not_found"
    ):
        return ChatModelUnavailableError(provider=provider, status_code=status)
    return None


# ---------------- End of Exceptions ----------------------------
