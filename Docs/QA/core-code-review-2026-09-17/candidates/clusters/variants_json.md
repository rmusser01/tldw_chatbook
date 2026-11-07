
====================================================================================================
# _strict_json_loads: 4 defs, 4 distinct bodies, 4 distinct shapes

--- body ce7f49b3 (shape 5c9fd90b): 1 copies (core 1, interop 0)
   files: Chat/provider_continuation.py:204
   | def _strict_json_loads(value: str) -> object:
   |     def reject_constant(_value: str) -> None:
   |         _fail()

   |     def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
   |         result: dict[str, object] = {}
   |         for key, item in pairs:
   |             if key in result:
   |                 _fail()
   |             result[key] = item
   |         return result

   |     return json.loads(
   |         value,
   |         parse_constant=reject_constant,
   |         object_pairs_hook=unique_object,
   |     )

--- body 24c2c6ec (shape 05e5994f): 1 copies (core 1, interop 0)
   files: Chat/thinking_blocks.py:221
   | def _strict_json_loads(value: str) -> object:
   |     def reject_constant(_value: str) -> None:
   |         _fail("JSON number")

   |     def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
   |         result: dict[str, object] = {}
   |         for key, item in pairs:
   |             if key in result:
   |                 _fail("duplicate JSON key")
   |             result[key] = item
   |         return result

   |     return json.loads(
   |         value, parse_constant=reject_constant, object_pairs_hook=unique_object
   |     )

--- body 304a6492 (shape 91ae7670): 1 copies (core 1, interop 0)
   files: LLM_Calls/hosted_chat.py:673
   | def _strict_json_loads(value: str) -> object:
   |     try:
   |         decoded = json.loads(value, parse_constant=_reject_json_constant)
   |     except (RecursionError, TypeError, ValueError):
   |         return _JSON_DECODE_FAILED
   |     return decoded if _json_shape_is_safe(decoded) else _JSON_DECODE_FAILED

--- body 3826b9db (shape b672d466): 1 copies (core 1, interop 0)
   files: LLM_Calls/qwencloud_streaming.py:65
   | def _strict_json_loads(value: str) -> Any:
   |     try:
   |         decoded = json.loads(value, parse_constant=_reject_json_constant)
   |     except (RecursionError, TypeError, ValueError):
   |         return _JSON_DECODE_FAILED
   |     if not _json_shape_is_safe(decoded):
   |         return _JSON_DECODE_FAILED
   |     return decoded

====================================================================================================
# _reject_json_constant: 9 defs, 3 distinct bodies, 2 distinct shapes

--- body 776907a6 (shape 8a9df0d0): 7 copies (core 7, interop 0)
   files: Actor_Packs/export.py:718, Character_Chat/visual_identity.py:1658, LLM_Calls/hosted_chat.py:633, LLM_Calls/qwencloud_streaming.py:61, Persona_Visual/importer.py:1090, Persona_Visual/publication.py:1306, Persona_Visual/repository.py:1368
   | def _reject_json_constant(_value: str) -> None:
   |     raise ValueError

--- body e5004ed6 (shape fa37e82d): 1 copies (core 1, interop 0)
   files: MCP/permission_store.py:398
   | def _reject_json_constant(value: str) -> None:
   |     raise PermissionStoreSnapshotError("invalid_json")

--- body f372beaa (shape 8a9df0d0): 1 copies (core 1, interop 0)
   files: TTS/audio_cpp_contract.py:87
   | def _reject_json_constant(_: str) -> NoReturn:
   |     raise _InvalidJsonTokenError

====================================================================================================
# _json_shape_is_safe: 2 defs, 2 distinct bodies, 1 distinct shapes

--- body 53166d94 (shape c964fb46): 1 copies (core 1, interop 0)
   files: LLM_Calls/hosted_chat.py:637
   | def _json_shape_is_safe(value: object) -> bool:
   |     stack: list[tuple[object, int]] = [(value, 1)]
   |     scheduled_nodes = 1
   |     try:
   |         while stack:
   |             node, depth = stack.pop()
   |             if depth > _MAX_JSON_DEPTH:
   |                 return False
   |             if type(node) is dict:
   |                 for key, child in cast(dict[object, object], node).items():
   |                     if not isinstance(key, str):
   |                         return False
   |                     scheduled_nodes += 1
   |                     if scheduled_nodes > _MAX_JSON_NODES:
   |                         return False
   |                     stack.append((child, depth + 1))
   |                 continue
   |             if type(node) is list:
   |                 for child in cast(list[object], node):
   |                     scheduled_nodes += 1
   |                     if scheduled_nodes > _MAX_JSON_NODES:
   |                         return False
   |                     stack.append((child, depth + 1))
   |                 continue
   |             if node is None or isinstance(node, (str, bool)):
   |                 continue
   |             if isinstance(node, int) and not isinstance(node, bool):
   |                 continue
   |    ...

--- body ea955b2d (shape c964fb46): 1 copies (core 1, interop 0)
   files: LLM_Calls/qwencloud_streaming.py:75
   | def _json_shape_is_safe(value: Any) -> bool:
   |     """Validate JSON types, depth, and nodes iteratively before recursive work."""
   |     stack: list[tuple[Any, int]] = [(value, 1)]
   |     scheduled_nodes = 1
   |     try:
   |         while stack:
   |             node, depth = stack.pop()
   |             if depth > _MAX_JSON_DEPTH:
   |                 return False
   |             if type(node) is dict:
   |                 for key, child in cast(dict[Any, Any], node).items():
   |                     if not isinstance(key, str):
   |                         return False
   |                     scheduled_nodes += 1
   |                     if scheduled_nodes > _MAX_JSON_NODES:
   |                         return False
   |                     stack.append((child, depth + 1))
   |                 continue
   |             if type(node) is list:
   |                 for child in cast(list[Any], node):
   |                     scheduled_nodes += 1
   |                     if scheduled_nodes > _MAX_JSON_NODES:
   |                         return False
   |                     stack.append((child, depth + 1))
   |                 continue
   |             if node is None or isinstance(node, (str, bool)):
   |                 continue
   |             if isinstance(node, int) and not isinstance(node, bool):
   |    ...

====================================================================================================
# _json_safe: 2 defs, 2 distinct bodies, 2 distinct shapes

--- body d0e44d0b (shape f5068647): 1 copies (core 1, interop 0)
   files: Library/local_library_tool_service.py:167
   | def _json_safe(value: Any) -> Any:
   |     """Coerce backend values (e.g. parsed DATETIME columns) to JSON-safe forms.

   |     Some stores decode timestamp columns into ``datetime`` objects; those break
   |     ``serialized_size`` and are not wire-serializable, so every value crossing
   |     into a response payload passes through here.
   |     """
   |     if value is None or isinstance(value, (str, int, float, bool)):
   |         return value
   |     if isinstance(value, (datetime, date)):
   |         return value.isoformat()
   |     return str(value)

--- body d0e23149 (shape 76de25a4): 1 copies (core 1, interop 0)
   files: Subscriptions/briefing_voices.py:131
   | def _json_safe(value: Any) -> Any:
   |     """Recursively convert a profile's frozen options into plain JSON types.

   |     `TTSGenerationProfile.options` freezes nested dicts/lists into
   |     `Mapping`/`tuple` (`TTS/profile_types.py`'s `FrozenJsonOptions`), which
   |     `json.dumps` cannot serialize directly (a `Mapping` that is not itself
   |     a `dict`, and a `tuple`, are both rejected). This mirrors that module's
   |     own private `_json_ready` helper (not imported -- it is private to its
   |     own module).
   |     """

   |     if isinstance(value, Mapping):
   |         return {key: _json_safe(item) for key, item in value.items()}
   |     if isinstance(value, (list, tuple)):
   |         return [_json_safe(item) for item in value]
   |     return value

====================================================================================================
# _load_json: 1 defs, 1 distinct bodies, 1 distinct shapes

--- body 4b532dc9 (shape 6a928f71): 1 copies (core 0, interop 1)
   files: Research_Interop/local_research_service.py:620
   | def _load_json(value: str | None) -> Any:
   |         if not value:
   |             return {}
   |         return json.loads(value)
