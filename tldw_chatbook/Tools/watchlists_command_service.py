"""Bounded synchronous Console commands for local Watchlists authoring."""

from __future__ import annotations

import ipaddress
import json
import re
import unicodedata
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urlsplit


_SOURCE_ID = re.compile(r"^local:subscription:([1-9][0-9]*)$")
_WATCHLIST_ID = re.compile(r"^local:watchlist:([1-9][0-9]*)$")
_SOURCE_KEYS = frozenset(
    {"url", "name", "type", "tags", "active", "check_frequency"}
)
_TOP_SOURCE_KEYS = frozenset({"sources"})
_COLLECTION_KEYS = frozenset(
    {"name", "description", "tags", "source_ids", "if_exists"}
)
_UPDATE_KEYS = frozenset(
    {"collection_id", "add_source_ids", "remove_source_ids"}
)
_COLLISION_POLICIES = frozenset({"conflict", "return_existing", "auto_suffix"})
_HOST_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z", re.IGNORECASE)


class WatchlistsCommandService:
    """Validate and shape Console-only Watchlists mutations."""

    def __init__(
        self,
        *,
        runtime_source_loader: Callable[[], object],
        create_sources_batch: Callable[[list[Mapping[str, Any]]], Any],
        create_collection: Callable[..., Any],
        update_collection_sources: Callable[..., Any],
    ) -> None:
        self._runtime_source_loader = runtime_source_loader
        self._create_sources_batch = create_sources_batch
        self._create_collection = create_collection
        self._update_collection_sources = update_collection_sources

    @staticmethod
    def _json(payload: Mapping[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def _invalid(cls, message: str) -> str:
        return cls._json(
            {"status": "invalid_argument", "retryable": False, "message": message}
        )

    @classmethod
    def _unavailable(cls) -> str:
        return cls._json(
            {
                "status": "feature_unavailable",
                "retryable": True,
                "message": "Watchlists storage is temporarily unavailable. Try again.",
            }
        )

    def _local_or_refusal(self) -> str | None:
        try:
            loaded = self._runtime_source_loader()
            source = str(getattr(loaded, "active_source", loaded)).strip().casefold()
        except Exception:  # noqa: BLE001 - the boundary returns fixed copy
            return self._unavailable()
        if source != "local":
            return self._json(
                {
                    "status": "unsupported",
                    "retryable": False,
                    "message": "Watchlists authoring commands require local mode.",
                }
            )
        return None

    @staticmethod
    def _exact_object(
        value: object, *, allowed: frozenset[str], required: frozenset[str]
    ) -> dict[str, Any] | None:
        if type(value) is not dict:
            return None
        if not required <= set(value) or set(value) - allowed:
            return None
        return dict(value)

    @staticmethod
    def _string(
        value: object, *, maximum: int, allow_empty: bool = False
    ) -> str | None:
        if not isinstance(value, str) or len(value) > maximum:
            return None
        stripped = value.strip()
        if not allow_empty and not stripped:
            return None
        return stripped

    @classmethod
    def _tags(cls, value: object) -> list[str] | None:
        if type(value) is not list or len(value) > 20:
            return None
        tags: list[str] = []
        for raw_tag in value:
            tag = cls._string(raw_tag, maximum=64)
            if tag is None:
                return None
            tags.append(tag)
        return tags

    @staticmethod
    def _valid_hostname(hostname: str) -> bool:
        try:
            ipaddress.ip_address(hostname)
            return True
        except ValueError:
            pass
        try:
            ascii_hostname = hostname.encode("idna").decode("ascii")
        except UnicodeError:
            return False
        labels = ascii_hostname.split(".")
        if len(ascii_hostname) > 253 or any(not label for label in labels):
            return False
        for label in labels:
            if _HOST_LABEL.fullmatch(label) is None:
                return False
            if label.casefold().startswith("xn--"):
                try:
                    decoded = label.encode("ascii").decode("idna")
                    round_trip = decoded.encode("idna").decode("ascii")
                except UnicodeError:
                    return False
                if round_trip.casefold() != label.casefold():
                    return False
        return True

    @staticmethod
    def _source_url(value: object) -> tuple[str | None, str | None]:
        if (
            not isinstance(value, str)
            or not value.strip()
            or len(value) > 2_048
            or any(unicodedata.category(character) == "Cc" for character in value)
        ):
            return None, "Source URL must be an absolute HTTP(S) URL."
        source = value.strip()
        if "\\" in source or any(character.isspace() for character in source):
            return None, "Source URL must be an absolute HTTP(S) URL."
        try:
            parsed = urlsplit(source)
            hostname = parsed.hostname
            parsed.port
        except ValueError:
            return None, "Source URL must be an absolute HTTP(S) URL."
        if parsed.username is not None or parsed.password is not None:
            return None, "Source URL must not include credentials."
        if (
            parsed.scheme.casefold() not in {"http", "https"}
            or not hostname
            or not WatchlistsCommandService._valid_hostname(hostname)
        ):
            return None, "Source URL must be an absolute HTTP(S) URL."
        return source, None

    @staticmethod
    def _canonical_id(value: object, pattern: re.Pattern[str]) -> int | None:
        if not isinstance(value, str):
            return None
        match = pattern.fullmatch(value)
        if match is None:
            return None
        number = int(match.group(1))
        return number if number <= 2**63 - 1 else None

    @classmethod
    def _canonical_ids(
        cls, value: object, *, maximum: int
    ) -> list[int] | None:
        if type(value) is not list or len(value) > maximum:
            return None
        ids: list[int] = []
        for raw_id in value:
            source_id = cls._canonical_id(raw_id, _SOURCE_ID)
            if source_id is None or source_id in ids:
                return None
            ids.append(source_id)
        return ids

    @classmethod
    def approval_source_destinations(cls, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Return content-free source-creation approval scope."""
        sources = arguments.get("sources")
        if type(sources) is not list:
            return {"source_count": 0, "destination_hosts": []}
        hosts: list[str] = []
        for row in sources[:50]:
            if type(row) is not dict:
                continue
            source, error = cls._source_url(row.get("url"))
            if error is not None or source is None:
                continue
            try:
                host = urlsplit(source).hostname
            except ValueError:
                continue
            if host and host not in hosts:
                hosts.append(host)
        return {"source_count": len(sources), "destination_hosts": hosts}

    def create_sources(self, arguments: object) -> str:
        """Validate and create one bounded source batch."""
        refusal = self._local_or_refusal()
        if refusal is not None:
            return refusal
        values = self._exact_object(
            arguments, allowed=_TOP_SOURCE_KEYS, required=_TOP_SOURCE_KEYS
        )
        if values is None or type(values.get("sources")) is not list:
            return self._invalid("Expected exactly one sources array.")
        raw_sources = values["sources"]
        if not 1 <= len(raw_sources) <= 50:
            return self._invalid("Provide between 1 and 50 sources.")

        valid: list[dict[str, Any]] = []
        valid_indexes: list[int] = []
        results: dict[int, dict[str, Any]] = {}
        for index, raw_source in enumerate(raw_sources):
            row = self._exact_object(
                raw_source, allowed=_SOURCE_KEYS, required=frozenset({"url"})
            )
            message = "Source definition has unsupported or missing fields."
            if row is not None:
                url, url_error = self._source_url(row["url"])
                if url_error is not None:
                    message = url_error
                else:
                    source_type = row.get("type", "rss")
                    name = row.get("name")
                    tags = row.get("tags")
                    active = row.get("active", True)
                    frequency = row.get("check_frequency")
                    if not isinstance(source_type, str) or source_type not in {
                        "rss",
                        "atom",
                        "url",
                    }:
                        message = "Source type must be rss, atom, or url."
                    elif name is not None and self._string(name, maximum=512) is None:
                        message = "Source name must be a non-empty string of at most 512 characters."
                    elif tags is not None and self._tags(tags) is None:
                        message = "Source tags must contain at most 20 short strings."
                    elif type(active) is not bool:
                        message = "Source active must be a boolean."
                    elif frequency is not None and (
                        type(frequency) is not int
                        or not 60 <= frequency <= 2_678_400
                    ):
                        message = "Source check_frequency must be an integer from 60 to 2678400."
                    else:
                        valid.append(
                            {
                                "url": url,
                                "name": self._string(name, maximum=512)
                                if name is not None
                                else None,
                                "source_type": source_type,
                                "tags": self._tags(tags) if tags is not None else [],
                                "active": active,
                                **(
                                    {"check_frequency": frequency}
                                    if frequency is not None
                                    else {}
                                ),
                            }
                        )
                        valid_indexes.append(index)
                        continue
            results[index] = {
                "input_index": index,
                "outcome": "invalid",
                "message": message,
            }

        if not valid:
            return self._json(
                {
                    "status": "invalid_argument",
                    "retryable": False,
                    "message": "No valid sources were provided.",
                    "results": [results[index] for index in range(len(raw_sources))],
                }
            )
        try:
            outcomes = self._create_sources_batch(valid)
            for outcome in outcomes:
                valid_index = outcome["input_index"]
                if type(valid_index) is not int or not 0 <= valid_index < len(valid):
                    raise ValueError("invalid domain source index")
                input_index = valid_indexes[valid_index]
                source = outcome["source"]
                source_id = source["source_id"]
                result_outcome = str(outcome["outcome"])
                if (
                    result_outcome not in {"created", "existing"}
                    or type(source_id) is not int
                    or not 1 <= source_id <= 2**63 - 1
                    or input_index in results
                ):
                    raise ValueError("invalid domain source outcome")
                results[input_index] = {
                    "input_index": input_index,
                    "outcome": result_outcome,
                    "source_id": f"local:subscription:{source_id}",
                }
            ordered_results = [results[index] for index in range(len(raw_sources))]
            partial = len(valid) != len(raw_sources)
            return self._json(
                {
                    "status": "partial_success" if partial else "ok",
                    "retryable": False,
                    "follow_on_confirmation_required": partial,
                    "results": ordered_results,
                }
            )
        except Exception:  # noqa: BLE001 - fixed protocol-safe failure
            return self._unavailable()

    def create_collection(self, arguments: object) -> str:
        """Validate and atomically create or resolve one collection."""
        refusal = self._local_or_refusal()
        if refusal is not None:
            return refusal
        values = self._exact_object(
            arguments, allowed=_COLLECTION_KEYS, required=frozenset({"name"})
        )
        if values is None:
            return self._invalid("Collection arguments are invalid.")
        name = self._string(values["name"], maximum=256)
        description = values.get("description")
        tags = values.get("tags")
        source_ids = self._canonical_ids(values.get("source_ids", []), maximum=100)
        policy = values.get("if_exists", "conflict")
        if name is None:
            return self._invalid("Collection name is invalid.")
        if description is not None and self._string(
            description, maximum=2_048, allow_empty=True
        ) is None:
            return self._invalid("Collection description is invalid.")
        if tags is not None and self._tags(tags) is None:
            return self._invalid("Collection tags are invalid.")
        if source_ids is None:
            return self._invalid("Collection source IDs must be unique canonical IDs.")
        if not isinstance(policy, str) or policy not in _COLLISION_POLICIES:
            return self._invalid("Collection collision policy is invalid.")
        try:
            outcome = self._create_collection(
                name=name,
                description=description,
                tags=self._tags(tags) if tags is not None else None,
                source_ids=source_ids,
                if_exists=policy,
            )
        except ValueError as exc:
            if "already exists" in str(exc):
                return self._json(
                    {
                        "status": "conflict",
                        "retryable": False,
                        "message": "A collection with that name already exists.",
                    }
                )
            return self._unavailable()
        except KeyError:
            return self._json(
                {
                    "status": "not_found",
                    "retryable": False,
                    "message": "One or more source IDs were not found.",
                }
            )
        except Exception:  # noqa: BLE001 - fixed protocol-safe failure
            return self._unavailable()
        try:
            result_outcome = outcome["outcome"]
            collection_id = outcome["watchlist"]["id"]
            membership_count = outcome["membership_count"]
            if (
                result_outcome not in {"created", "existing"}
                or type(collection_id) is not int
                or not 1 <= collection_id <= 2**63 - 1
                or type(membership_count) is not int
                or membership_count < 0
            ):
                raise ValueError("invalid domain collection outcome")
            return self._json(
                {
                    "status": "ok",
                    "retryable": False,
                    "outcome": result_outcome,
                    "collection_id": f"local:watchlist:{collection_id}",
                    "collision_policy": policy,
                    "membership_count": membership_count,
                }
            )
        except Exception:  # noqa: BLE001 - fixed protocol-safe failure
            return self._unavailable()

    def update_collection_sources(self, arguments: object) -> str:
        """Validate and atomically replace requested collection memberships."""
        refusal = self._local_or_refusal()
        if refusal is not None:
            return refusal
        values = self._exact_object(
            arguments,
            allowed=_UPDATE_KEYS,
            required=frozenset({"collection_id"}),
        )
        if values is None:
            return self._invalid("Collection membership arguments are invalid.")
        watchlist_id = self._canonical_id(values["collection_id"], _WATCHLIST_ID)
        add_ids = self._canonical_ids(values.get("add_source_ids", []), maximum=100)
        remove_ids = self._canonical_ids(
            values.get("remove_source_ids", []), maximum=100
        )
        if watchlist_id is None or add_ids is None or remove_ids is None:
            return self._invalid("Use unique canonical collection and source IDs.")
        if not add_ids and not remove_ids:
            return self._invalid("Provide at least one source to add or remove.")
        if len(add_ids) + len(remove_ids) > 100:
            return self._invalid("Provide at most 100 membership changes.")
        if set(add_ids) & set(remove_ids):
            return self._invalid("A source cannot be both added and removed.")
        try:
            outcome = self._update_collection_sources(
                watchlist_id=watchlist_id,
                add_ids=add_ids,
                remove_ids=remove_ids,
            )
        except KeyError:
            return self._json(
                {
                    "status": "not_found",
                    "retryable": False,
                    "message": "The collection or one of its sources was not found.",
                }
            )
        except Exception:  # noqa: BLE001 - fixed protocol-safe failure
            return self._unavailable()
        try:
            added = outcome["added"]
            removed = outcome["removed"]
            membership_count = outcome["membership_count"]
            if (
                type(added) is not int
                or added < 0
                or type(removed) is not int
                or removed < 0
                or type(membership_count) is not int
                or membership_count < 0
            ):
                raise ValueError("invalid domain membership outcome")
            return self._json(
                {
                    "status": "ok",
                    "retryable": False,
                    "collection_id": f"local:watchlist:{watchlist_id}",
                    "added": added,
                    "removed": removed,
                    "membership_count": membership_count,
                }
            )
        except Exception:  # noqa: BLE001 - fixed protocol-safe failure
            return self._unavailable()
