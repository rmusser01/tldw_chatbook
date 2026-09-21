
====================================================================================================
# _format_size: 5 defs, 5 distinct bodies, 5 distinct shapes

--- body f9bd6eb5 (shape 0752fbff): 1 copies (core 1, interop 0)
   files: Chat/attachment_core.py:64
   | def _format_size(size: int) -> str:
   |     if size >= 1024 * 1024:
   |         return f"{size / 1024 / 1024:.1f} MB"
   |     if size >= 1024:
   |         return f"{size / 1024:.0f} KB"
   |     return f"{size} B"

--- body 166ac544 (shape d37464d8): 1 copies (core 1, interop 0)
   files: Tools/web_tool_impls.py:463
   | def _format_size(num_bytes: int) -> str:
   |     """Human-readable byte count for binary-fetch metadata lines."""
   |     size = float(num_bytes)
   |     for unit in ("B", "KB", "MB", "GB"):
   |         if size < 1024.0:
   |             return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
   |         size /= 1024.0
   |     return f"{size:.1f} TB"

--- body 88387c89 (shape 951426bb): 1 copies (core 1, interop 0)
   files: UI/CodeRepoCopyPasteWindow.py:1209
   | def _format_size(self, size: int) -> str:
   |         """Format file size in human-readable format."""
   |         for unit in ["B", "KB", "MB", "GB"]:
   |             if size < 1024.0:
   |                 return f"{size:.1f} {unit}"
   |             size /= 1024.0
   |         return f"{size:.1f} TB"

--- body 0c54d127 (shape 6acee373): 1 copies (core 1, interop 0)
   files: Utils/file_handlers.py:574
   | def _format_size(self, size: int) -> str:
   |         """Format file size in human-readable format."""
   |         for unit in ["B", "KB", "MB", "GB"]:
   |             if size < 1024.0:
   |                 return f"{size:.1f}{unit}"
   |             size /= 1024.0
   |         return f"{size:.1f}TB"

--- body dd291a93 (shape e3619188): 1 copies (core 1, interop 0)
   files: Widgets/Coding_Widgets/repo_tree_widgets.py:201
   | def _format_size(self, size: Optional[int]) -> str:
   |         """Format file size in human-readable format."""
   |         if size is None:
   |             return ""

   |         for unit in ["B", "KB", "MB", "GB"]:
   |             if size < 1024.0:
   |                 return f"{size:.1f} {unit}"
   |             size /= 1024.0
   |         return f"{size:.1f} TB"

====================================================================================================
# _format_file_size: 4 defs, 3 distinct bodies, 3 distinct shapes

--- body 750e3a85 (shape 9290eaac): 2 copies (core 2, interop 0)
   files: Widgets/file_list_item_enhanced.py:72, Widgets/file_list_item_enhanced.py:233
   | def _format_file_size(self, size_bytes: int) -> str:
   |         """Format file size in human readable format."""
   |         for unit in ["B", "KB", "MB", "GB", "TB"]:
   |             if size_bytes < 1024.0:
   |                 return f"{size_bytes:.1f} {unit}"
   |             size_bytes /= 1024.0
   |         return f"{size_bytes:.1f} PB"

--- body 72ce7e66 (shape 951426bb): 1 copies (core 1, interop 0)
   files: UI/Tools_Settings_Window.py:6421
   | def _format_file_size(self, size_bytes: int) -> str:
   |         """Format file size in human-readable format."""
   |         for unit in ["B", "KB", "MB", "GB"]:
   |             if size_bytes < 1024.0:
   |                 return f"{size_bytes:.1f} {unit}"
   |             size_bytes /= 1024.0
   |         return f"{size_bytes:.1f} TB"

--- body 772f96ff (shape d89acf04): 1 copies (core 1, interop 0)
   files: Widgets/NewIngest/SmartFileDropZone.py:97
   | def _format_file_size(size: int) -> str:
   |         if size >= 1_000_000_000:
   |             return f"{size / (1024 * 1024 * 1024):.1f} GB"
   |         if size >= 1_000_000:
   |             return f"{size / (1024 * 1024):.1f} MB"
   |         if size >= 1024:
   |             return f"{size / 1024:.1f} KB"
   |         return f"{float(size):.1f} B"

====================================================================================================
# _human_size: 2 defs, 2 distinct bodies, 2 distinct shapes

--- body 33181035 (shape 295887c0): 1 copies (core 1, interop 0)
   files: Library/library_ingest_state.py:609
   | def _human_size(size_bytes: int) -> str:
   |     """Return a compact human-readable size string."""
   |     if size_bytes < 1024:
   |         return f"{size_bytes} B"
   |     value = float(size_bytes)
   |     for unit in ("KB", "MB", "GB", "TB"):
   |         value /= 1024
   |         if value < 1024:
   |             return f"{value:.1f} {unit}"
   |     # Sizes above ~1 PB: divide once more so the value is actually in petabytes.
   |     value /= 1024
   |     return f"{value:.1f} PB"

--- body f9bd6eb5 (shape 0752fbff): 1 copies (core 1, interop 0)
   files: Widgets/Console/console_transcript.py:639
   | def _human_size(size: int) -> str:
   |     """Format a byte count for display, matching ``attachment_core._format_size``.

   |     Kept as a small local helper (rather than importing ``attachment_core``)
   |     to keep this widget free of that dependency.
   |     """
   |     if size >= 1024 * 1024:
   |         return f"{size / 1024 / 1024:.1f} MB"
   |     if size >= 1024:
   |         return f"{size / 1024:.0f} KB"
   |     return f"{size} B"

====================================================================================================
# _format_bytes: 2 defs, 2 distinct bodies, 2 distinct shapes

--- body baa1d5c1 (shape 342dd9b0): 1 copies (core 1, interop 0)
   files: UI/Screens/settings_privacy_security.py:364
   | def _format_bytes(value: int) -> str:
   |     amount = float(value)
   |     for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
   |         if amount < 1024 or unit == "TiB":
   |             return f"{int(amount)} {unit}" if unit == "B" else f"{amount:.1f} {unit}"
   |         amount /= 1024
   |     raise AssertionError("unreachable")

--- body ffafd618 (shape c8de9e7a): 1 copies (core 1, interop 0)
   files: Widgets/Console/console_video_capacity_modal.py:23
   | def _format_bytes(size_bytes: int) -> str:
   |     """Return a compact binary-unit size for the modal copy."""
   |     mib = size_bytes / (1024 * 1024)
   |     return f"{mib:.1f} MiB"
