"""SwarmUI backend adapter."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote, urlparse

from loguru import logger

from tldw_chatbook.Image_Generation.http_client import fetch_json
from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest, ImageGenResult
from tldw_chatbook.Image_Generation.adapters.image_format_utils import (
    decode_data_url as decode_shared_data_url,
    fetch_image_bytes as fetch_shared_image_bytes,
    validate_and_convert_image_output,
)
from tldw_chatbook.Image_Generation.config import (
    DEFAULT_SWARMUI_BASE_URL,
    DEFAULT_SWARMUI_TIMEOUT_SECONDS,
    get_image_generation_config,
)
from tldw_chatbook.Image_Generation.exceptions import ImageBackendUnavailableError, ImageGenerationError
from tldw_chatbook.Image_Generation.request_validation import effective_inline_max_bytes
from tldw_chatbook.Utils.egress import origin_set, same_origin


class SwarmUIAdapter:
    name = "swarmui"
    supported_formats = {"png", "jpg"}

    def __init__(self) -> None:
        self._config = get_image_generation_config()
        self._session_id: str | None = None

    def generate(self, request: ImageGenRequest) -> ImageGenResult:
        output_format = request.format.lower()
        if output_format not in self.supported_formats:
            raise ImageGenerationError(f"unsupported output format: {output_format}")

        base_url = self._resolve_base_url()
        # base_url is user-configured (config.toml/env), never API-returned data,
        # so its host is trusted to resolve to a private/local IP (e.g. 127.0.0.1).
        trusted_origins = origin_set(base_url)
        session_id = self._ensure_session(base_url, trusted_origins)

        # task-558: capture the model this request actually resolved to
        # (request override, else the configured default, else none) so the
        # Console generation card can show it instead of always blank --
        # this is client-side resolution of a value already fully known
        # before the call, not anything guessed from SwarmUI's response.
        resolved_model = self._resolve_model(request)
        payload = self._build_payload(request, session_id, resolved_model)
        generate_url = f"{base_url}/API/GenerateText2Image"
        data = self._post_generate(generate_url, payload, trusted_origins)

        image_ref = self._extract_first_image_ref(data)
        if not image_ref:
            raise ImageGenerationError("SwarmUI did not return any images")

        if image_ref.startswith("data:"):
            content, content_type = decode_shared_data_url(image_ref, max_bytes=self._max_output_bytes())
            content, content_type = validate_and_convert_image_output(
                content,
                content_type,
                output_format,
                max_bytes=self._max_output_bytes(),
            )
            return ImageGenResult(
                content=content,
                content_type=content_type,
                bytes_len=len(content),
                resolved_model=resolved_model,
            )

        image_url = self._resolve_image_url(base_url, image_ref)
        content, content_type = self._fetch_image_bytes(image_url, trusted_origins)
        content, content_type = validate_and_convert_image_output(
            content,
            content_type,
            output_format,
            max_bytes=self._max_output_bytes(),
        )
        return ImageGenResult(
            content=content,
            content_type=content_type,
            bytes_len=len(content),
            resolved_model=resolved_model,
        )

    def _resolve_base_url(self) -> str:
        raw = (self._config.swarmui_base_url or DEFAULT_SWARMUI_BASE_URL or "").strip()
        if not raw:
            raise ImageBackendUnavailableError("swarmui_base_url is not configured")
        if not raw.startswith("http://") and not raw.startswith("https://"):
            raw = f"http://{raw}"
        return raw.rstrip("/")

    def _cookies(self) -> dict[str, str] | None:
        token = (self._config.swarmui_swarm_token or "").strip()
        if not token:
            return None
        return {"swarm_token": token}

    def _max_output_bytes(self) -> int:
        return effective_inline_max_bytes(self._config)

    def _ensure_session(self, base_url: str, trusted_origins: frozenset) -> str:
        if self._session_id:
            return self._session_id
        self._session_id = self._request_session_id(base_url, trusted_origins)
        return self._session_id

    def _request_session_id(self, base_url: str, trusted_origins: frozenset) -> str:
        url = f"{base_url}/API/GetNewSession"
        data = self._post_json(url, {}, trusted_origins)
        session_id = None
        if isinstance(data, dict):
            session_id = data.get("session_id")
        if not session_id:
            raise ImageGenerationError("SwarmUI did not return a session_id")
        return str(session_id)

    def _resolve_model(self, request: ImageGenRequest) -> str | None:
        """Return the model this request will actually use (override or configured default)."""
        return request.model or (self._config.swarmui_default_model or None)

    def _build_payload(
        self, request: ImageGenRequest, session_id: str, resolved_model: str | None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "session_id": session_id,
            "images": 1,
            "prompt": request.prompt,
        }
        if request.negative_prompt:
            payload["negativeprompt"] = request.negative_prompt
        if request.width is not None:
            payload["width"] = request.width
        if request.height is not None:
            payload["height"] = request.height
        if request.steps is not None:
            payload["steps"] = request.steps
        if request.cfg_scale is not None:
            payload["cfgscale"] = request.cfg_scale
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.sampler:
            payload["sampler"] = request.sampler

        if resolved_model:
            payload["model"] = resolved_model

        extra_params = request.extra_params or {}
        if isinstance(extra_params, dict):
            for key, value in extra_params.items():
                if key in {"session_id", "images"}:
                    continue
                payload[key] = value
        return payload

    def _post_generate(self, url: str, payload: dict[str, Any], trusted_origins: frozenset) -> dict[str, Any]:
        data = self._post_json(url, payload, trusted_origins)
        error_id = data.get("error_id") if isinstance(data, dict) else None
        if error_id == "invalid_session_id":
            logger.info("SwarmUI session invalid; refreshing session_id")
            self._session_id = self._request_session_id(self._resolve_base_url(), trusted_origins)
            retry_payload = dict(payload)
            retry_payload["session_id"] = self._session_id
            data = self._post_json(url, retry_payload, trusted_origins)
            error_id = data.get("error_id") if isinstance(data, dict) else None

        if isinstance(data, dict):
            if error_id:
                raise ImageGenerationError(f"SwarmUI error_id: {error_id}")
            if data.get("error"):
                raise ImageGenerationError(str(data.get("error")))
        return data if isinstance(data, dict) else {}

    def _post_json(self, url: str, payload: dict[str, Any], trusted_origins: frozenset) -> dict[str, Any]:
        try:
            data = fetch_json(
                method="POST",
                url=url,
                json=payload,
                cookies=self._cookies(),
                timeout=self._config.swarmui_timeout_seconds or DEFAULT_SWARMUI_TIMEOUT_SECONDS,
                trusted_origins=trusted_origins,
            )
        except Exception as exc:
            raise ImageGenerationError(f"SwarmUI request failed: {exc}") from exc
        if not isinstance(data, dict):
            raise ImageGenerationError("SwarmUI response was not JSON")
        return data

    @staticmethod
    def _extract_first_image_ref(data: dict[str, Any]) -> str | None:
        if not isinstance(data, dict):
            return None
        images = data.get("images")
        if isinstance(images, list) and images:
            first = images[0]
            if isinstance(first, dict):
                image_ref = first.get("image")
                return str(image_ref) if image_ref else None
            if isinstance(first, str):
                return first
        return None

    @staticmethod
    def _resolve_image_url(base_url: str, image_ref: str) -> str:
        if image_ref.startswith("http://") or image_ref.startswith("https://"):
            if not same_origin(base_url, image_ref):
                raise ImageGenerationError("SwarmUI returned off-origin image URL")
            return image_ref
        parsed = urlparse(image_ref)
        if parsed.scheme in {"http", "https"}:
            if not same_origin(base_url, image_ref):
                raise ImageGenerationError("SwarmUI returned off-origin image URL")
            return image_ref
        path = image_ref.lstrip("/")
        encoded_path = "/".join(quote(part) for part in path.split("/"))
        return f"{base_url.rstrip('/')}/{encoded_path}"

    def _fetch_image_bytes(self, url: str, trusted_origins: frozenset) -> tuple[bytes, str]:
        try:
            return fetch_shared_image_bytes(
                url,
                cookies=self._cookies(),
                timeout=self._config.swarmui_timeout_seconds or DEFAULT_SWARMUI_TIMEOUT_SECONDS,
                max_bytes=self._max_output_bytes(),
                trusted_origins=trusted_origins,
            )
        except Exception as exc:
            raise ImageGenerationError(f"SwarmUI image fetch failed: {exc}") from exc
