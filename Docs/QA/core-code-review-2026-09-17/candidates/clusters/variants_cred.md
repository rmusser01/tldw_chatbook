
====================================================================================================
# _resolve_api_key: 10 defs, 10 distinct bodies, 5 distinct shapes

--- body 6d954d6f (shape 741f63b9): 1 copies (core 1, interop 0)
   files: Image_Generation/adapters/fal_image_adapter.py:262
   | def _resolve_api_key(self) -> str:
   |         api_key = (self._config.fal_image_api_key or "").strip()
   |         if not api_key:
   |             api_key = (os.getenv("FAL_KEY") or "").strip()
   |         if not api_key:
   |             raise ImageBackendUnavailableError("fal image api key is not configured")
   |         return api_key

--- body 0a70de09 (shape bfb44561): 1 copies (core 1, interop 0)
   files: Image_Generation/adapters/gemini_image_adapter.py:176
   | def _resolve_api_key(self) -> str:
   |         api_key = (self._config.gemini_image_api_key or "").strip()
   |         if not api_key:
   |             api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
   |         if not api_key:
   |             raise ImageBackendUnavailableError("gemini image api key is not configured")
   |         return api_key

--- body 334151f8 (shape 7af3ac49): 1 copies (core 1, interop 0)
   files: Image_Generation/adapters/modelstudio_image_adapter.py:180
   | def _resolve_api_key(self) -> str:
   |         api_key = (self._config.modelstudio_image_api_key or "").strip()
   |         if not api_key:
   |             api_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
   |         if not api_key:
   |             api_key = (os.getenv("QWEN_API_KEY") or "").strip()
   |         if not api_key:
   |             raise ImageBackendUnavailableError("modelstudio image api key is not configured")
   |         return api_key

--- body 8b3f8257 (shape 741f63b9): 1 copies (core 1, interop 0)
   files: Image_Generation/adapters/novita_image_adapter.py:63
   | def _resolve_api_key(self) -> str:
   |         api_key = (self._config.novita_image_api_key or "").strip()
   |         if not api_key:
   |             api_key = (os.getenv("NOVITA_API_KEY") or "").strip()
   |         if not api_key:
   |             raise ImageBackendUnavailableError("novita image api key is not configured")
   |         return api_key

--- body ddc2f862 (shape 741f63b9): 1 copies (core 1, interop 0)
   files: Image_Generation/adapters/openrouter_image_adapter.py:89
   | def _resolve_api_key(self) -> str:
   |         api_key = (self._config.openrouter_image_api_key or "").strip()
   |         if not api_key:
   |             api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
   |         if not api_key:
   |             raise ImageBackendUnavailableError("openrouter image api key is not configured")
   |         return api_key

--- body 7fc622db (shape 741f63b9): 1 copies (core 1, interop 0)
   files: Image_Generation/adapters/together_image_adapter.py:71
   | def _resolve_api_key(self) -> str:
   |         api_key = (self._config.together_image_api_key or "").strip()
   |         if not api_key:
   |             api_key = (os.getenv("TOGETHER_API_KEY") or "").strip()
   |         if not api_key:
   |             raise ImageBackendUnavailableError("together image api key is not configured")
   |         return api_key

--- body df6e19fb (shape 08b15543): 1 copies (core 1, interop 0)
   files: LLM_Calls/moonshot.py:601
   | def _resolve_api_key(
   |     explicit: object,
   |     settings: Mapping[str, object],
   |     environ: Mapping[str, str],
   | ) -> str:
   |     if explicit is not None:
   |         resolved = resolve_provider_api_key(explicit)
   |         if resolved is None:
   |             raise _configuration_error("Moonshot explicit API key is invalid.")
   |         return resolved
   |     if "api_key" in settings:
   |         resolved = resolve_provider_api_key(settings.get("api_key"))
   |         if resolved is None:
   |             raise _configuration_error(
   |                 "Moonshot api_settings.moonshot.api_key is invalid."
   |             )
   |         return resolved
   |     env_name = settings.get("api_key_env_var", "MOONSHOT_API_KEY")
   |     if not isinstance(env_name, str) or not env_name.strip():
   |         raise _configuration_error(
   |             "Moonshot api_settings.moonshot.api_key_env_var is invalid."
   |         )
   |     for candidate_name in dict.fromkeys((env_name.strip(), "MOONSHOT_API_KEY")):
   |         resolved = resolve_provider_api_key(environ.get(candidate_name))
   |         if resolved is not None:
   |             return resolved
   |     raise _configuration_error("Moonshot API key is required.")

--- body 61c6f028 (shape 08b15543): 1 copies (core 1, interop 0)
   files: LLM_Calls/zai.py:569
   | def _resolve_api_key(
   |     explicit: object,
   |     settings: Mapping[str, object],
   |     environ: Mapping[str, str],
   | ) -> str:
   |     if explicit is not None:
   |         resolved = resolve_provider_api_key(explicit)
   |         if resolved is None:
   |             raise _configuration_error("Z.ai explicit API key is invalid.")
   |         return resolved
   |     if "api_key" in settings:
   |         resolved = resolve_provider_api_key(settings.get("api_key"))
   |         if resolved is None:
   |             raise _configuration_error("Z.ai api_settings.zai.api_key is invalid.")
   |         return resolved
   |     env_name = settings.get("api_key_env_var", "ZAI_API_KEY")
   |     if not isinstance(env_name, str) or not env_name.strip():
   |         raise _configuration_error("Z.ai api_settings.zai.api_key_env_var is invalid.")
   |     for candidate in dict.fromkeys((env_name.strip(), "ZAI_API_KEY")):
   |         resolved = resolve_provider_api_key(environ.get(candidate))
   |         if resolved is not None:
   |             return resolved
   |     raise _configuration_error("Z.ai API key is required.")

--- body f5c1bebf (shape 1568eed6): 1 copies (core 1, interop 0)
   files: LLM_Provider_Catalog/local_llm_provider_catalog_service.py:319
   | def _resolve_api_key(
   |         self,
   |         *,
   |         provider: str,
   |         provider_key: str,
   |         saved_settings: Mapping[str, Any],
   |         staged_settings: Mapping[str, Any] | None,
   |     ) -> str | None:
   |         try:
   |             staged_provider_settings = self._provider_settings_for_key(
   |                 staged_settings, provider_key
   |             )
   |             saved_provider_settings = self._provider_settings_for_key(
   |                 saved_settings, provider_key
   |             )
   |         except ProviderSettingsError:
   |             return None
   |         staged_source = configured_provider_credential_source(
   |             staged_provider_settings
   |         )
   |         if staged_source == "stored":
   |             pinned_saved_settings = dict(saved_provider_settings)
   |             pinned_saved_settings["credential_source"] = "stored"
   |             return self._api_key_from_provider_settings(
   |                 provider_key, pinned_saved_settings
   |             )
   |         staged_key = self._api_key_from_provider_settings(
   |             provider_key, staged_provider_settings
   |    ...

--- body b3d7d89a (shape 741f63b9): 1 copies (core 1, interop 0)
   files: Video_Generation/adapters/minimax_video_adapter.py:196
   | def _resolve_api_key(self) -> str:
   |         api_key = (self._config.minimax_video_api_key or "").strip()
   |         if not api_key:
   |             api_key = (os.getenv("MINIMAX_API_KEY") or "").strip()
   |         if not api_key:
   |             raise VideoBackendUnavailableError(
   |                 "MiniMax video api key is not configured -- set MINIMAX_API_KEY, "
   |                 "[video_generation.minimax] api_key, or the keyring entry"
   |             )
   |         return api_key

====================================================================================================
# _resolve_base_url: 10 defs, 10 distinct bodies, 6 distinct shapes

--- body 2616013a (shape a957b2e9): 1 copies (core 1, interop 0)
   files: Image_Generation/adapters/fal_image_adapter.py:270
   | def _resolve_base_url(self) -> str:
   |         raw = self._config.fal_image_base_url or DEFAULT_FAL_IMAGE_BASE_URL
   |         cleaned = str(raw).strip()
   |         if not cleaned:
   |             raise ImageBackendUnavailableError("fal image base URL is not configured")
   |         if not cleaned.startswith("http://") and not cleaned.startswith("https://"):
   |             cleaned = f"https://{cleaned}"
   |         return cleaned.rstrip("/")

--- body b4197a83 (shape a957b2e9): 1 copies (core 1, interop 0)
   files: Image_Generation/adapters/gemini_image_adapter.py:184
   | def _resolve_base_url(self) -> str:
   |         raw = self._config.gemini_image_base_url or DEFAULT_GEMINI_IMAGE_BASE_URL
   |         cleaned = str(raw).strip()
   |         if not cleaned:
   |             raise ImageBackendUnavailableError("gemini image base URL is not configured")
   |         if not cleaned.startswith("http://") and not cleaned.startswith("https://"):
   |             cleaned = f"https://{cleaned}"
   |         return cleaned.rstrip("/")

--- body 7e216d0e (shape 9fec0222): 1 copies (core 1, interop 0)
   files: Image_Generation/adapters/modelstudio_image_adapter.py:199
   | def _resolve_base_url(self) -> str:
   |         raw = os.getenv("MODELSTUDIO_IMAGE_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL")
   |         if not raw:
   |             raw = self._config.modelstudio_image_base_url
   |         if not raw:
   |             raw = self._REGION_BASE_URLS.get(self._resolve_region(), DEFAULT_MODELSTUDIO_IMAGE_BASE_URL)
   |         cleaned = str(raw).strip()
   |         if not cleaned:
   |             raise ImageBackendUnavailableError("modelstudio image base URL is not configured")
   |         if not cleaned.startswith("http://") and not cleaned.startswith("https://"):
   |             cleaned = f"https://{cleaned}"
   |         return cleaned.rstrip("/")

--- body 76e72fb5 (shape 31bfaae9): 1 copies (core 1, interop 0)
   files: Image_Generation/adapters/novita_image_adapter.py:71
   | def _resolve_base_url(self) -> str:
   |         raw = (
   |             os.getenv("NOVITA_IMAGE_BASE_URL")
   |             or self._config.novita_image_base_url
   |             or DEFAULT_NOVITA_IMAGE_BASE_URL
   |         )
   |         cleaned = str(raw).strip()
   |         if not cleaned:
   |             raise ImageBackendUnavailableError("novita image base URL is not configured")
   |         if not cleaned.startswith("http://") and not cleaned.startswith("https://"):
   |             cleaned = f"https://{cleaned}"
   |         return cleaned.rstrip("/")

--- body dfa5a33f (shape 31bfaae9): 1 copies (core 1, interop 0)
   files: Image_Generation/adapters/openrouter_image_adapter.py:97
   | def _resolve_base_url(self) -> str:
   |         raw = (
   |             os.getenv("OPENROUTER_BASE_URL")
   |             or self._config.openrouter_image_base_url
   |             or DEFAULT_OPENROUTER_IMAGE_BASE_URL
   |         )
   |         cleaned = str(raw).strip()
   |         if not cleaned:
   |             raise ImageBackendUnavailableError("openrouter image base URL is not configured")
   |         if not cleaned.startswith("http://") and not cleaned.startswith("https://"):
   |             cleaned = f"https://{cleaned}"
   |         return cleaned.rstrip("/")

--- body b241b00b (shape c0ead22a): 1 copies (core 1, interop 0)
   files: Image_Generation/adapters/swarmui_adapter.py:90
   | def _resolve_base_url(self) -> str:
   |         raw = (self._config.swarmui_base_url or DEFAULT_SWARMUI_BASE_URL or "").strip()
   |         if not raw:
   |             raise ImageBackendUnavailableError("swarmui_base_url is not configured")
   |         if not raw.startswith("http://") and not raw.startswith("https://"):
   |             raw = f"http://{raw}"
   |         return raw.rstrip("/")

--- body 2f851ad0 (shape 31bfaae9): 1 copies (core 1, interop 0)
   files: Image_Generation/adapters/together_image_adapter.py:79
   | def _resolve_base_url(self) -> str:
   |         raw = (
   |             os.getenv("TOGETHER_BASE_URL")
   |             or self._config.together_image_base_url
   |             or DEFAULT_TOGETHER_IMAGE_BASE_URL
   |         )
   |         cleaned = str(raw).strip()
   |         if not cleaned:
   |             raise ImageBackendUnavailableError("together image base URL is not configured")
   |         if not cleaned.startswith("http://") and not cleaned.startswith("https://"):
   |             cleaned = f"https://{cleaned}"
   |         return cleaned.rstrip("/")

--- body 1212f70e (shape d70f6785): 1 copies (core 1, interop 0)
   files: LLM_Calls/moonshot.py:643
   | def _resolve_base_url(
   |     explicit: object,
   |     settings: Mapping[str, object],
   | ) -> str:
   |     if explicit is not None:
   |         candidate = explicit
   |     elif "api_base_url" in settings:
   |         candidate = settings.get("api_base_url")
   |     else:
   |         region = settings.get("api_region", "international")
   |         if not isinstance(region, str) or region not in {"international", "china"}:
   |             raise _configuration_error("Moonshot API region is invalid.")
   |         candidate = _CHINA_BASE_URL if region == "china" else _DEFAULT_BASE_URL
   |     try:
   |         return normalize_hosted_chat_base_url(candidate, default=_DEFAULT_BASE_URL)
   |     except ValueError:
   |         raise _configuration_error("Moonshot API base URL is invalid.") from None

--- body 3d229283 (shape 05d9ca77): 1 copies (core 1, interop 0)
   files: LLM_Calls/zai.py:594
   | def _resolve_base_url(explicit: object, settings: Mapping[str, object]) -> str:
   |     candidate = (
   |         explicit
   |         if explicit is not None
   |         else settings.get("api_base_url", _DEFAULT_BASE_URL)
   |     )
   |     try:
   |         return normalize_hosted_chat_base_url(candidate, default=_DEFAULT_BASE_URL)
   |     except ValueError:
   |         raise _configuration_error("Z.ai API base URL is invalid.") from None

--- body c7de9f53 (shape a957b2e9): 1 copies (core 1, interop 0)
   files: Video_Generation/adapters/minimax_video_adapter.py:207
   | def _resolve_base_url(self) -> str:
   |         raw = self._config.minimax_video_base_url or DEFAULT_MINIMAX_VIDEO_BASE_URL
   |         cleaned = str(raw).strip()
   |         if not cleaned:
   |             raise VideoBackendUnavailableError("MiniMax video base URL is not configured")
   |         if not cleaned.startswith("http://") and not cleaned.startswith("https://"):
   |             cleaned = f"https://{cleaned}"
   |         return cleaned.rstrip("/")
