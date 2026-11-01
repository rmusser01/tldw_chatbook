
====================================================================================================
# _coerce_bool: 13 defs, 7 distinct bodies, 7 distinct shapes

--- body 623c6552 (shape 316c283e): 3 copies (core 3, interop 0)
   files: Character_Chat/Character_Chat_Lib.py:158, Character_Chat/world_book_import.py:34, Character_Chat/world_info_processor.py:20
   | def _coerce_bool(value: Any, default: bool) -> bool:
   |     """Best-effort bool coercion for loosely-typed card fields.

   |     None -> ``default``; actual bools pass through; ints/floats map by
   |     ``!= 0``; recognized string booleans (true/false/1/0/yes/no/on/off,
   |     case-insensitive) map by value; anything else -> ``default``.
   |     """
   |     if value is None:
   |         return default
   |     if isinstance(value, bool):
   |         return value
   |     if isinstance(value, (int, float)):
   |         return value != 0
   |     if isinstance(value, str):
   |         token = value.strip().lower()
   |         if token in _TRUE_STRINGS:
   |             return True
   |         if token in _FALSE_STRINGS:
   |             return False
   |         return default
   |     return default

--- body e495e6f8 (shape c23df0ea): 3 copies (core 3, interop 0)
   files: Chat/console_rail_state.py:335, Home/home_rail_state.py:24, Library/library_rail_state.py:42
   | def _coerce_bool(value: Any, fallback: bool) -> bool:
   |     if isinstance(value, bool):
   |         return value
   |     if isinstance(value, int):
   |         return value != 0
   |     if isinstance(value, str):
   |         normalized = value.strip().lower()
   |         if normalized in _TRUE_STRINGS:
   |             return True
   |         if normalized in _FALSE_STRINGS:
   |             return False
   |     return fallback

--- body 4afd58bc (shape 5b87f961): 3 copies (core 3, interop 0)
   files: Image_Generation/config.py:466, Utils/console_background_effects.py:43, Video_Generation/config.py:333
   | def _coerce_bool(value: Any, default: bool) -> bool:
   |     if isinstance(value, bool):
   |         return value
   |     if isinstance(value, str):
   |         lowered = value.strip().lower()
   |         if lowered in {"true", "1", "yes", "on"}:
   |             return True
   |         if lowered in {"false", "0", "no", "off"}:
   |             return False
   |     return default

--- body 0e39f24b (shape ad77067b): 1 copies (core 1, interop 0)
   files: Character_Chat/Chat_Dictionary_Lib.py:95
   | def _coerce_bool(value: Any, default: bool) -> bool:
   |     """Best-effort bool coercion that treats quoted booleans honestly.

   |     Args:
   |         value: Raw value from a payload or persisted JSON.
   |         default: Fallback for None or unrecognized strings.

   |     Returns:
   |         ``value`` itself for real bools; a case-insensitive allowlist parse
   |         for strings ("false"/"0"/"no"/"off" are False); ``default`` for None
   |         or unrecognized strings; ``bool(value)`` otherwise.
   |     """
   |     if isinstance(value, bool):
   |         return value
   |     if value is None:
   |         return default
   |     if isinstance(value, str):
   |         lowered = value.strip().lower()
   |         if lowered in _TRUTHY_STRINGS:
   |             return True
   |         if lowered in _FALSY_STRINGS:
   |             return False
   |         return default
   |     return bool(value)

--- body c1c0fb3b (shape fae8bdef): 1 copies (core 1, interop 0)
   files: Character_Chat/world_book_manager.py:48
   | def _coerce_bool(value: Any, default: bool) -> bool:
   |     """Best-effort bool coercion for loosely-typed / imported embedded fields.

   |     Accepts real bools, and the strings ``"true"/"false"/"1"/"0"/"yes"/"no"``
   |     (case-insensitive). Anything else falls back to ``default``. Mirrors the
   |     processor's ``_coerce_bool`` so a hand-edited/imported ``enabled`` never
   |     misreads (e.g. the string ``"false"`` must not be truthy).
   |     """
   |     if isinstance(value, bool):
   |         return value
   |     if isinstance(value, (int, float)):
   |         return bool(value)
   |     if isinstance(value, str):
   |         s = value.strip().lower()
   |         if s in ("true", "1", "yes", "on"):
   |             return True
   |         if s in ("false", "0", "no", "off"):
   |             return False
   |     return default

--- body 288e488d (shape 82ceb56f): 1 copies (core 1, interop 0)
   files: UI/Screens/settings_appearance_defaults.py:91
   | def _coerce_bool(value: Any, default: bool) -> bool:
   |     """Coerce common config boolean values."""
   |     if isinstance(value, bool):
   |         return value
   |     if isinstance(value, (str, int)):
   |         normalized = str(value).strip().lower()
   |         if normalized in {"1", "true", "yes", "on", "enabled"}:
   |             return True
   |         if normalized in {"0", "false", "no", "off", "disabled"}:
   |             return False
   |     return default

--- body a0c0385c (shape c3908cea): 1 copies (core 1, interop 0)
   files: Utils/adaptive_reader_state.py:154
   | def _coerce_bool(value: Any, default: bool) -> bool:
   |     if type(value) is bool:
   |         return value
   |     if isinstance(value, str):
   |         normalized = value.strip().lower()
   |         if normalized in {"true", "1", "yes", "on"}:
   |             return True
   |         if normalized in {"false", "0", "no", "off"}:
   |             return False
   |     return default

====================================================================================================
# _coerce_int: 10 defs, 5 distinct bodies, 4 distinct shapes

--- body 1f418301 (shape 28f0d6dd): 5 copies (core 5, interop 0)
   files: Character_Chat/Chat_Dictionary_Lib.py:75, Character_Chat/world_book_import.py:22, Character_Chat/world_book_manager.py:27, Character_Chat/world_info_processor.py:44, Subscriptions/local_watchlists_service.py:2965
   | def _coerce_int(value: Any, default: int = 0) -> int:
   |     """Best-effort int coercion for loosely-typed entry fields.

   |     Args:
   |         value: Raw value from a payload or persisted JSON.
   |         default: Fallback when the value is missing or malformed.

   |     Returns:
   |         The coerced int, or ``default`` on None/non-numeric input.
   |     """
   |     try:
   |         return int(value)
   |     except (TypeError, ValueError):
   |         return default

--- body aeae325f (shape f1e3b2b8): 2 copies (core 2, interop 0)
   files: Image_Generation/config.py:376, Video_Generation/config.py:319
   | def _coerce_int(value: Any, default: int) -> int:
   |     try:
   |         return int(str(value).strip())
   |     except (TypeError, ValueError):
   |         return default

--- body 36b11ae6 (shape f1e3b2b8): 1 copies (core 1, interop 0)
   files: Library/server_ingest_request.py:329
   | def _coerce_int(value: Any, fallback: int) -> int:
   |     """Return ``value`` as an int, falling back rather than raising.

   |     Option values arrive from the canvas's form echo, where numbers are display
   |     *text*; a half-typed field must not be able to break a submission.
   |     """
   |     try:
   |         return int(str(value).strip())
   |     except (TypeError, ValueError):
   |         return fallback

--- body bbc58cc1 (shape b926b0fa): 1 copies (core 1, interop 0)
   files: Library/web_clip_request.py:73
   | def _coerce_int(value: Any, fallback: int) -> int:
   |     """Read an int from a form value, falling back rather than raising.

   |     Number fields round-trip through display text, so a partially typed or
   |     cleared input arrives as a string. A bad value must not abort a submission
   |     the user has already confirmed.
   |     """
   |     try:
   |         parsed = int(str(value).strip())
   |     except (TypeError, ValueError):
   |         return fallback
   |     return parsed if parsed > 0 else fallback

--- body 94dcff19 (shape a9b4a34d): 1 copies (core 1, interop 0)
   files: UI/Screens/settings_appearance_defaults.py:104
   | def _coerce_int(value: Any, default: int) -> int:
   |     """Coerce integral config values with a safe fallback."""
   |     if isinstance(value, bool):
   |         return default
   |     try:
   |         parsed = float(str(value).strip())
   |     except (TypeError, ValueError):
   |         return default
   |     if not parsed.is_integer():
   |         return default
   |     return int(parsed)

====================================================================================================
# _safe_text: 12 defs, 9 distinct bodies, 9 distinct shapes

--- body 7086af40 (shape 83586053): 4 copies (core 4, interop 0)
   files: UI/Library_Modules/library_ingest_controller.py:885, UI/Library_Modules/library_media_controller.py:1081, UI/Library_Modules/library_prompts_controller.py:726, UI/Library_Modules/library_rag_search_controller.py:630
   | def _safe_text(self) -> Any:
   |         return self._safe_text_fn

--- body 56f751cf (shape b5e97f36): 1 copies (core 1, interop 0)
   files: MCP/unified_control_plane_service.py:3828
   | def _safe_text(value: str, limit: int) -> str:
   |             rooted = str(redact_root_locator(str(value), root))
   |             try:
   |                 decoded = json.loads(rooted)
   |             except (TypeError, ValueError, json.JSONDecodeError):
   |                 decoded = None
   |             if isinstance(decoded, (Mapping, list, tuple)):
   |                 rooted = json.dumps(_sanitize_json(decoded), sort_keys=True)
   |             rooted = _scrub_secret_fragments(rooted)
   |             encoded = rooted.encode("utf-8")
   |             if len(encoded) <= limit:
   |                 return rooted
   |             return encoded[:limit].decode("utf-8", errors="ignore") + "\n… [truncated]"

--- body 76491f5d (shape 81f2b54c): 1 copies (core 0, interop 1)
   files: Sync_Interop/sync_profile_status_state.py:15
   | def _safe_text(value: Any, fallback: str = "", *, max_length: int = 200) -> str:
   |     text = sanitize_string(str(value or ""), max_length=max_length).strip()
   |     text = " ".join(text.split())
   |     if not text:
   |         return fallback
   |     if not validate_text_input(text, max_length=max_length, allow_html=False):
   |         return fallback
   |     return text

--- body 02f51f71 (shape 02937051): 1 copies (core 1, interop 0)
   files: Tools/watchlists_tool_service.py:2025
   | def _safe_text(value: object, maximum_bytes: int) -> str | None:
   |         """Return control-free bounded stored text without its truncation flag."""
   |         return WatchlistsToolService._bounded_text(value, maximum_bytes)[0]

--- body f5fceac2 (shape 5dc204f0): 1 copies (core 1, interop 0)
   files: UI/LLM_Management/vllm_profiles.py:191
   | def _safe_text(value: object, field: str, *, maximum: int, allow_empty: bool) -> str:
   |     if type(value) is not str:
   |         raise VllmProfileValidationError(f"{field} must be a string")
   |     if any(
   |         unicodedata.category(character) in _UNSAFE_TEXT_CATEGORIES
   |         for character in value
   |     ):
   |         raise VllmProfileValidationError(f"{field} contains unsafe characters")
   |     normalized = unicodedata.normalize("NFKC", value)
   |     if not allow_empty and not normalized:
   |         raise VllmProfileValidationError(f"{field} must not be empty")
   |     if len(normalized) > maximum:
   |         raise VllmProfileValidationError(f"{field} is too long")
   |     return normalized

--- body 5efd716a (shape 6fa7b4f3): 1 copies (core 1, interop 0)
   files: UI/Screens/artifacts_screen.py:610
   | def _safe_text(
   |         cls, value: Any, fallback: str = "", *, max_length: int = 1000
   |     ) -> str:
   |         text = sanitize_string(str(value or ""), max_length=max_length).strip()
   |         if not text:
   |             return fallback
   |         text = html_escape(text, quote=False)
   |         if validate_text_input(text, max_length=max_length, allow_html=False):
   |             return text
   |         for pattern in DANGEROUS_TEXT_PATTERNS:
   |             text = re.sub(
   |                 re.escape(pattern),
   |                 pattern.rstrip(":=").replace("=", ""),
   |                 text,
   |                 flags=re.IGNORECASE,
   |             )
   |         if validate_text_input(text, max_length=max_length, allow_html=False):
   |             return text
   |         return fallback

--- body 1e5fb84e (shape 3ed5eb95): 1 copies (core 1, interop 0)
   files: UI/Screens/chat_screen.py:17550
   | def _safe_text(value: Any, max_length: int = 500) -> str:
   |             return sanitize_string(str(value or ""), max_length=max_length).strip()

--- body df35bc27 (shape 2f7ca56c): 1 copies (core 1, interop 0)
   files: UI/Screens/library_screen.py:12965
   | def _safe_text(value: Any, fallback: str = "", *, max_length: int = 500) -> str:
   |         text = sanitize_string(str(value or ""), max_length=max_length).strip()
   |         if not text:
   |             return fallback
   |         text = text.replace("<", "").replace(">", "")
   |         for pattern in ("javascript:", "onclick=", "onerror="):
   |             text = text.replace(pattern, "")
   |         if validate_text_input(text, max_length=max_length, allow_html=False):
   |             return text
   |         return fallback

--- body b008ea0e (shape 79d89b85): 1 copies (core 1, interop 0)
   files: UI/Screens/watchlists_collections_screen.py:1954
   | def _safe_text(value: Any, fallback: str = "", *, max_length: int = 500) -> str:
   |         text = sanitize_string(str(value or ""), max_length=max_length).strip()
   |         if not text:
   |             return fallback
   |         if validate_text_input(text, max_length=max_length, allow_html=False):
   |             return text
   |         return fallback

====================================================================================================
# _clean_text: 7 defs, 5 distinct bodies, 5 distinct shapes

--- body 18723f2a (shape 2200827d): 2 copies (core 1, interop 1)
   files: ACP_Interop/runtime_process.py:32, Chat/console_live_work.py:25
   | def _clean_text(value: Any, fallback: str = "") -> str:
   |     text = str(value or "").strip()
   |     return text or fallback

--- body 6f2d7b8d (shape b6955f36): 2 copies (core 2, interop 0)
   files: Character_Chat/local_character_persona_service.py:55, Chat/chat_conversation_service.py:27
   | def _clean_text(value: Any) -> str | None:
   |     if value is None:
   |         return None
   |     text = str(value).strip()
   |     return text or None

--- body fb9b5174 (shape 57e9bd8f): 1 copies (core 1, interop 0)
   files: Agents/ask_user_questions.py:178
   | def _clean_text(value: object, *, required: bool) -> str:
   |     """Pre-validate one text field: type, UTF-8, control flattening, blank.

   |     Length is left to the model's ``Field`` constraint so the limit lives in
   |     exactly one place per field.

   |     Args:
   |         value: The raw value.
   |         required: Whether a blank value is an error.

   |     Returns:
   |         The flattened string.

   |     Raises:
   |         ValueError: Wrong type, invalid UTF-8, or blank when required.
   |     """
   |     if not isinstance(value, str):
   |         # ValueError on purpose: Pydantic wraps ValueError/AssertionError from a
   |         # validator into a ValidationError; a TypeError would escape as a crash.
   |         raise ValueError("must be a string")  # noqa: TRY004
   |     try:
   |         value.encode("utf-8")
   |     except UnicodeEncodeError as exc:
   |         raise ValueError("is not valid UTF-8") from exc
   |     cleaned = _flatten(value)
   |     if required and not cleaned:
   |         raise ValueError("must not be blank")
   |     return cleaned

--- body 6ab231c2 (shape 0ff5c158): 1 copies (core 1, interop 0)
   files: Chat/console_rail_state.py:555
   | def _clean_text(value: Any) -> str:
   |     if value is None:
   |         return ""
   |     return str(value).strip()

--- body 7e697318 (shape 4dda0124): 1 copies (core 1, interop 0)
   files: Library/library_rag_state.py:332
   | def _clean_text(value: Any, fallback: str = "") -> str:
   |     if value is None:
   |         return fallback
   |     text = " ".join(str(value).strip().split())
   |     return text or fallback

====================================================================================================
# _normalize_keywords: 13 defs, 10 distinct bodies, 10 distinct shapes

--- body 2861943c (shape 926b4d49): 4 copies (core 4, interop 0)
   files: tldw_api/media_reading_schemas.py:279, tldw_api/media_reading_schemas.py:301, tldw_api/media_reading_schemas.py:337, tldw_api/media_reading_schemas.py:937
   | def _normalize_keywords(cls, value: Any) -> Any:
   |         if value is None:
   |             return []
   |         if not isinstance(value, list):
   |             raise ValueError("keywords_must_be_list")
   |         return [
   |             _normalize_nonempty_string(entry, field_name="keyword") for entry in value
   |         ]

--- body 0e8f8f12 (shape bd10a235): 1 copies (core 1, interop 0)
   files: Chat/chat_conversation_service.py:96
   | def _normalize_keywords(keyword_rows: Any) -> list[str]:
   |     if not keyword_rows:
   |         return []

   |     normalized: list[str] = []
   |     seen: set[str] = set()
   |     for item in keyword_rows:
   |         keyword_text = item
   |         if isinstance(item, Mapping):
   |             keyword_text = item.get("keyword")
   |         text = _clean_text(keyword_text)
   |         if text is None:
   |             continue
   |         key = text.lower()
   |         if key in seen:
   |             continue
   |         seen.add(key)
   |         normalized.append(text)
   |     return normalized

--- body 97af8f88 (shape 7af1d0c5): 1 copies (core 1, interop 0)
   files: Notes/note_import_executor.py:784
   | def _normalize_keywords(keywords: Iterable[str]) -> dict[str, str]:
   |     if isinstance(keywords, (str, bytes)):
   |         raise _ImportTargetValidationError
   |     try:
   |         iterator = iter(keywords)
   |     except TypeError:
   |         raise _ImportTargetValidationError from None
   |     normalized: dict[str, str] = {}
   |     for count, value in enumerate(iterator, start=1):
   |         if count > MAX_IMPORT_KEYWORDS_PER_NOTE:
   |             raise _ImportTargetValidationError
   |         if not isinstance(value, str):
   |             raise _ImportTargetValidationError
   |         display = value.strip()
   |         if not display or len(display) > MAX_IMPORT_KEYWORD_LENGTH or "\x00" in display:
   |             raise _ImportTargetValidationError
   |         normalized.setdefault(_sqlite_nocase_key(display), display)
   |     return dict(sorted(normalized.items()))

--- body 945e7442 (shape 513963b5): 1 copies (core 1, interop 0)
   files: Notes/notes_scope_service.py:1024
   | def _normalize_keywords(keywords: Optional[Sequence[str]]) -> list[str]:
   |         if keywords is None:
   |             return []
   |         normalized: list[str] = []
   |         seen: set[str] = set()
   |         for item in keywords:
   |             text = str(item).strip()
   |             if not text:
   |                 continue
   |             key = text.lower()
   |             if key in seen:
   |                 continue
   |             seen.add(key)
   |             normalized.append(text)
   |         return normalized

--- body b255bb40 (shape 5c020467): 1 copies (core 1, interop 0)
   files: Notes/server_notes_workspace_service.py:115
   | def _normalize_keywords(self, keywords: Any) -> list[str]:
   |         if keywords is None:
   |             return []
   |         if isinstance(keywords, str):
   |             return [part.strip() for part in keywords.split(",") if part.strip()]

   |         normalized: list[str] = []
   |         seen: set[str] = set()
   |         for item in keywords:
   |             value: Any = item
   |             if isinstance(item, Mapping):
   |                 value = (
   |                     item.get("keyword")
   |                     or item.get("text")
   |                     or item.get("name")
   |                     or item.get("value")
   |                 )
   |             if not isinstance(value, str):
   |                 continue
   |             cleaned = value.strip()
   |             if not cleaned:
   |                 continue
   |             lowered = cleaned.lower()
   |             if lowered in seen:
   |                 continue
   |             seen.add(lowered)
   |             normalized.append(cleaned)
   |         return normalized

--- body bb754114 (shape 04e76878): 1 copies (core 1, interop 0)
   files: Prompt_Management/prompt_normalizers.py:75
   | def _normalize_keywords(value: Any) -> list[str]:
   |     if value is None:
   |         return []
   |     if isinstance(value, str):
   |         raw_items = value.split(",")
   |     else:
   |         raw_items = value

   |     normalized: list[str] = []
   |     seen: set[str] = set()
   |     for item in raw_items:
   |         text = str(item).strip()
   |         if not text:
   |             continue
   |         key = text.lower()
   |         if key in seen:
   |             continue
   |         seen.add(key)
   |         normalized.append(text)
   |     return normalized

--- body 7d6670ec (shape 443df459): 1 copies (core 1, interop 0)
   files: Prompt_Management/server_prompt_adapter.py:21
   | def _normalize_keywords(keywords: Any) -> List[str]:
   |     if keywords is None:
   |         return []
   |     if isinstance(keywords, list):
   |         return [str(keyword).strip() for keyword in keywords if str(keyword).strip()]
   |     if isinstance(keywords, str):
   |         return [keyword.strip() for keyword in keywords.split(",") if keyword.strip()]
   |     return [str(keyword).strip() for keyword in list(keywords) if str(keyword).strip()]

--- body 60266737 (shape 3a04a50a): 1 copies (core 1, interop 0)
   files: tldw_api/chat_conversation_schemas.py:199
   | def _normalize_keywords(cls, value: list[str] | None) -> list[str] | None:
   |         if value is None:
   |             return None

   |         cleaned: list[str] = []
   |         seen: set[str] = set()
   |         for item in value:
   |             if item is None:
   |                 continue
   |             normalized = str(item).strip()
   |             if not normalized:
   |                 continue
   |             key = normalized.lower()
   |             if key in seen:
   |                 continue
   |             seen.add(key)
   |             cleaned.append(normalized)
   |         return cleaned

--- body ef09b385 (shape 9a4f7333): 1 copies (core 1, interop 0)
   files: tldw_api/media_reading_schemas.py:261
   | def _normalize_keywords(cls, value: Any) -> Any:
   |         if value is None:
   |             return value
   |         if not isinstance(value, list):
   |             raise ValueError("keywords_must_be_list")
   |         return [
   |             _normalize_nonempty_string(entry, field_name="keyword") for entry in value
   |         ]

--- body 02638318 (shape 40bb2b91): 1 copies (core 1, interop 0)
   files: tldw_api/media_reading_schemas.py:516
   | def _normalize_keywords(cls, value: Any) -> Any:
   |         if not isinstance(value, list):
   |             raise ValueError("keywords_must_be_list")
   |         return [
   |             _normalize_nonempty_string(entry, field_name="keyword") for entry in value
   |         ]

====================================================================================================
# _truncate_text: 2 defs, 2 distinct bodies, 2 distinct shapes

--- body b68af25c (shape 71eb7cd4): 1 copies (core 1, interop 0)
   files: Event_Handlers/ingest_utils.py:30
   | def _truncate_text(text: Optional[str], max_len: int) -> str:
   |     """
   |     Truncates a string to a maximum length, adding ellipsis if truncated.
   |     Returns 'N/A' if the input text is None or empty.
   |     """
   |     if not text:  # Handles None or empty string
   |         return "N/A"
   |     if len(text) > max_len:
   |         return text[: max_len - 3] + "..."
   |     return text

--- body b31fd935 (shape 65d4a542): 1 copies (core 1, interop 0)
   files: Widgets/chunk_preview.py:104
   | def _truncate_text(self, text: str, max_length: int) -> str:
   |         """Truncate text to maximum length."""
   |         if len(text) <= max_length:
   |             return text
   |         return text[:max_length]

====================================================================================================
# _truncate: 1 defs, 1 distinct bodies, 1 distinct shapes

--- body 55b6cc30 (shape 234bb4db): 1 copies (core 1, interop 0)
   files: Agents/run_hooks.py:193
   | def _truncate(text: str) -> str:
   |     """Cap hook-produced text at HOOK_IO_BUDGET_CHARS *total* (marker included)."""
   |     if len(text) <= HOOK_IO_BUDGET_CHARS:
   |         return text
   |     return text[: HOOK_IO_BUDGET_CHARS - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER
