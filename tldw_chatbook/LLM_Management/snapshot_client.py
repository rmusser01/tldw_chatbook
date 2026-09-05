"""Bounded loopback HTTP transport for llama.cpp prompt-cache snapshots."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Literal

import httpx
from pydantic import ValidationError

from .snapshot_models import (
    LaunchDescriptor,
    ReadinessObservation,
    SlotObservation,
    SlotReceipt,
    SnapshotError,
)

PROBE_SECONDS = 5.0
MUTATION_SECONDS = 600.0
MUTATION_TIMEOUT = httpx.Timeout(
    connect=5.0,
    pool=5.0,
    write=30.0,
    read=600.0,
)
MAX_RESPONSE_BYTES = 1024 * 1024

_UNSUPPORTED_STATUSES = frozenset({404, 405, 501})
_AUTH_STATUSES = frozenset({401, 403})


def _error(code: str, *, submission_possible: bool = False) -> SnapshotError:
    return SnapshotError(code, submission_possible=submission_possible)


def _check_status(status_code: int) -> None:
    if status_code in _AUTH_STATUSES:
        raise _error("authentication_failed")
    if status_code in _UNSUPPORTED_STATUSES:
        raise _error("unsupported_route")
    if 300 <= status_code < 400:
        raise _error("unexpected_redirect")
    if not 200 <= status_code < 300:
        raise _error("request_failed")


def _strict_int(value: object, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError("invalid integer")
    return value


def _optional_int(value: object, *, minimum: int) -> int | None:
    if value is None:
        return None
    return _strict_int(value, minimum=minimum)


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if type(value) is not bool:
        raise ValueError("invalid boolean")
    return value


def _is_basename(value: object) -> bool:
    if not isinstance(value, str) or not value or "\0" in value:
        return False
    return "/" not in value and "\\" not in value and value not in {".", ".."}


class SnapshotClient:
    """Own one proxy-free, redirect-free client for an admitted llama.cpp launch."""

    def __init__(
        self,
        descriptor: LaunchDescriptor,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = (
            {"Authorization": f"Bearer {descriptor.bearer_token}"}
            if descriptor.bearer_token
            else {}
        )
        self._client = httpx.AsyncClient(
            base_url=descriptor.base_url,
            headers=headers,
            trust_env=False,
            follow_redirects=False,
            timeout=MUTATION_TIMEOUT,
            transport=transport,
        )

    async def readiness(self) -> ReadinessObservation:
        """Return one aggregate-deadline health, properties, and slot observation."""

        failure: str | None = None
        try:
            async with asyncio.timeout(PROBE_SECONDS):
                health = await self._get_json("/health")
                if not isinstance(health, dict) or health.get("status") != "ok":
                    raise _error("not_ready")
                props = await self._get_json("/props")
                slots = await self._get_json("/slots")
                return self._readiness(props, slots)
        except TimeoutError:
            failure = "probe_timeout"
        except SnapshotError:
            raise
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout):
            failure = "connection_failed"
        except httpx.RemoteProtocolError:
            failure = "protocol_error"
        except httpx.TransportError:
            failure = "transport_error"
        assert failure is not None
        raise _error(failure)

    async def slots(self) -> tuple[SlotObservation, ...]:
        """Return a bounded projection of the current server slots."""

        failure: str | None = None
        try:
            async with asyncio.timeout(PROBE_SECONDS):
                payload = await self._get_json("/slots")
                return self._slots(payload)
        except TimeoutError:
            failure = "probe_timeout"
        except SnapshotError:
            raise
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout):
            failure = "connection_failed"
        except httpx.RemoteProtocolError:
            failure = "protocol_error"
        except httpx.TransportError:
            failure = "transport_error"
        assert failure is not None
        raise _error(failure)

    async def save(self, slot_id: int, filename: str) -> SlotReceipt:
        """Submit exactly one slot-save mutation and validate its receipt."""

        return await self._mutate("save", slot_id, filename)

    async def restore(self, slot_id: int, filename: str) -> SlotReceipt:
        """Submit exactly one slot-restore mutation and validate its receipt."""

        return await self._mutate("restore", slot_id, filename)

    async def aclose(self) -> None:
        """Close the owned HTTP client and transport."""

        await self._client.aclose()

    async def _get_json(self, path: str) -> Any:
        return await self._request_json("GET", path)

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        json_body: dict[str, str] | None = None,
    ) -> Any:
        async with self._client.stream(
            method,
            path,
            params=params,
            json=json_body,
        ) as response:
            _check_status(response.status_code)
            body = bytearray()
            async for chunk in response.aiter_bytes():
                if len(body) + len(chunk) > MAX_RESPONSE_BYTES:
                    raise _error("response_too_large")
                body.extend(chunk)
        try:
            return json.loads(body)
        except (ValueError, UnicodeError):
            pass
        raise _error("invalid_response")

    def _readiness(self, props: Any, slots: Any) -> ReadinessObservation:
        if not isinstance(props, dict):
            raise _error("invalid_response")
        build_info = props.get("build_info")
        model_path = props.get("model_path")
        if not isinstance(build_info, str) or not isinstance(model_path, str):
            raise _error("invalid_response")
        try:
            return ReadinessObservation(
                slots=self._slots(slots),
                build_info=build_info,
                model_path=model_path,
                # The pinned /props schema does not report effective auto-resolved
                # flash-attention or device settings. Do not manufacture evidence.
                runtime_values=(),
            )
        except ValidationError:
            pass
        raise _error("invalid_response")

    def _slots(self, payload: Any) -> tuple[SlotObservation, ...]:
        if not isinstance(payload, list):
            raise _error("invalid_response")
        if any(not isinstance(item, dict) for item in payload):
            raise _error("invalid_response")
        observed_at = time.monotonic()
        try:
            return tuple(
                SlotObservation(
                    slot_id=_strict_int(item["id"], minimum=0),
                    busy=_optional_bool(item.get("is_processing")),
                    # The pinned /slots response has no cache-token field.
                    tokens=None,
                    context_size=_optional_int(item.get("n_ctx"), minimum=1),
                    observed_at=observed_at,
                )
                for item in payload
            )
        except (KeyError, TypeError, ValueError, ValidationError):
            pass
        raise _error("invalid_response")

    async def _mutate(
        self,
        action: Literal["save", "restore"],
        slot_id: int,
        filename: str,
    ) -> SlotReceipt:
        if type(slot_id) is not int or slot_id < 0:
            raise _error("invalid_slot")
        if not _is_basename(filename):
            raise _error("invalid_filename")
        failure: tuple[str, bool] | None = None
        receipt: SlotReceipt | None = None
        try:
            async with asyncio.timeout(MUTATION_SECONDS):
                try:
                    payload = await self._request_json(
                        "POST",
                        f"/slots/{slot_id}",
                        params={"action": action},
                        json_body={"filename": filename},
                    )
                except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout):
                    failure = ("connection_failed", False)
                except SnapshotError as exc:
                    if exc.code in {"invalid_response", "response_too_large"}:
                        failure = ("outcome_unknown", True)
                    else:
                        raise
                except httpx.TransportError:
                    failure = ("outcome_unknown", True)
                if failure is None:
                    try:
                        receipt = self._receipt(payload, action, slot_id, filename)
                    except SnapshotError:
                        failure = ("outcome_unknown", True)
        except TimeoutError:
            failure = ("outcome_unknown", True)
        except SnapshotError:
            raise
        if failure is not None:
            raise _error(failure[0], submission_possible=failure[1])
        assert receipt is not None
        return receipt

    def _receipt(
        self,
        payload: Any,
        action: Literal["save", "restore"],
        slot_id: int,
        filename: str,
    ) -> SlotReceipt:
        if not isinstance(payload, dict):
            raise _error("invalid_receipt")
        token_field, byte_field = (
            ("n_saved", "n_written")
            if action == "save"
            else ("n_restored", "n_read")
        )
        try:
            receipt = SlotReceipt(
                slot_id=payload["id_slot"],
                filename=payload["filename"],
                tokens=payload[token_field],
                bytes=payload[byte_field],
            )
        except (KeyError, TypeError, ValueError, ValidationError):
            receipt = None
        if receipt is None:
            raise _error("invalid_receipt")
        if receipt.slot_id != slot_id or receipt.filename != filename:
            raise _error("invalid_receipt")
        return receipt
