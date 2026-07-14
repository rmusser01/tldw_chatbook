# tldw_chatbook/Event_Handlers/Chat_Events/chat_image_events.py
# Description: Handle image operations for chat
#
# Imports
#
# Standard Library
import logging
import mimetypes
import re
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

# 3rd-party Libraries
from PIL import Image as PILImage

#
# Local Imports
#
#######################################################################################################################
#
# Functions:

# Formats vision providers accept as payloads; anything else transcodes to PNG.
PAYLOAD_SAFE_FORMATS = {"PNG", "JPEG", "WEBP", "GIF"}
PAYLOAD_FORMAT_MIME = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
    "GIF": "image/gif",
}


class ChatImageHandler:
    """Handle image operations for chat."""

    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}

    @staticmethod
    async def process_image_file(file_path: str) -> Tuple[bytes, str]:
        """
        Process an image file for chat attachment.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Tuple of (image_data, mime_type)
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format unsupported or too large
        """
        path = Path(file_path).expanduser().resolve()

        # Validate file exists
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")

        from tldw_chatbook.Chat.attachment_core import (
            max_image_bytes,
            supported_image_formats,
        )

        # Check file extension against the effective (config-driven) allowlist
        effective_formats = supported_image_formats()
        if path.suffix.lower() not in effective_formats:
            raise ValueError(
                f"Unsupported image format: {path.suffix}. "
                f"Supported formats: {', '.join(effective_formats)}"
            )

        # Check file size against the config-driven cap
        file_size = path.stat().st_size
        size_cap = max_image_bytes()
        if file_size > size_cap:
            raise ValueError(
                f"Image file too large ({file_size / 1024 / 1024:.1f}MB). "
                f"Maximum size: {size_cap / 1024 / 1024}MB"
            )

        image_data = path.read_bytes()
        extension = path.suffix.lower()

        if extension == '.svg':
            # No usable fallback for un-rasterized SVG bytes — errors reject.
            final_bytes, final_mime = await ChatImageHandler.prepare_image_payload(
                image_data, extension
            )
        else:
            mime_type = mimetypes.guess_type(str(path))[0] or 'image/png'
            try:
                final_bytes, final_mime = await ChatImageHandler.prepare_image_payload(
                    image_data, extension
                )
            except Exception as e:
                logging.warning(f"Failed to process image, using original: {e}")
                # If processing fails, use original data
                final_bytes, final_mime = image_data, mime_type

        # Processing (SVG rasterization, transcodes) can GROW bytes past the
        # source cap checked above — enforce the cap on the final payload too.
        # This must stay outside the try/except above: a policy rejection is
        # not a processing failure and must not trigger the original-bytes
        # fallback.
        if len(final_bytes) > size_cap:
            raise ValueError(
                f"Processed image too large ({len(final_bytes) / 1024 / 1024:.1f}MB). "
                f"Maximum size: {size_cap / 1024 / 1024}MB"
            )
        return final_bytes, final_mime

    @staticmethod
    def _svg_raster_kwargs(svg_bytes: bytes, cap: int) -> dict:
        """Bounded svg2png output kwargs: longer rendered side ≤ cap.

        The byte-size cap guards the SVG *source* only; a small file can
        declare a huge canvas, and cairo allocates that surface during
        render — before any PIL bomb guard runs. Aspect comes from viewBox,
        else numeric width/height; an unparseable aspect falls back to a
        both-dims hard bound (may distort that degenerate case; logged).
        """
        intrinsic = None  # (width, height) in user units
        try:
            from defusedxml import ElementTree as SafeET  # ships with cairosvg

            root = SafeET.fromstring(svg_bytes)
            view_box = root.get("viewBox")
            if view_box:
                parts = view_box.replace(",", " ").split()
                if len(parts) == 4:
                    vb_w, vb_h = float(parts[2]), float(parts[3])
                    if vb_w > 0 and vb_h > 0:
                        intrinsic = (vb_w, vb_h)
            if intrinsic is None:
                w_attr, h_attr = root.get("width"), root.get("height")
                if w_attr and h_attr:
                    w = float(re.sub(r"px\s*$", "", w_attr.strip(), count=1))
                    h = float(re.sub(r"px\s*$", "", h_attr.strip(), count=1))
                    if w > 0 and h > 0:
                        intrinsic = (w, h)
        except Exception:
            intrinsic = None
        if intrinsic is None:
            logging.warning(
                "SVG has no parseable aspect; rasterizing with a hard "
                f"{cap}x{cap} bound"
            )
            return {"output_width": cap, "output_height": cap}
        w, h = intrinsic
        target = max(1, int(round(min(max(w, h), cap))))
        if w >= h:
            return {"output_width": target}
        return {"output_height": target}

    @staticmethod
    async def prepare_image_payload(image_data: bytes, extension: str) -> Tuple[bytes, str]:
        """Normalize image bytes for provider payloads.

        SVG rasterizes to PNG first (bounded — see _svg_raster_kwargs);
        rasters larger than [chat.images].resize_max_dimension shrink;
        anything outside PAYLOAD_SAFE_FORMATS transcodes to PNG. The
        returned mime always matches the returned bytes.

        Args:
            image_data: Raw image bytes.
            extension: Lowercased dotted extension ('' when unknown, e.g.
                clipboard bytes — the SVG branch then never triggers).

        Returns:
            Tuple of (payload_bytes, mime_type).

        Raises:
            ValueError: If SVG rendering is required but unavailable, or the
                SVG cannot be rendered.
        """
        from tldw_chatbook.Chat.attachment_core import (
            image_resize_max_dimension,
            svg_rendering_available,
        )

        if extension == '.svg':
            if not svg_rendering_available():
                raise ValueError(
                    "SVG attachments require the optional cairosvg dependency "
                    "(pip install tldw_chatbook[svg])."
                )
            import cairosvg

            kwargs = ChatImageHandler._svg_raster_kwargs(
                image_data, image_resize_max_dimension()
            )
            try:
                # NOTE: cairosvg's `unsafe` parameter stays at its default
                # (False): XML entities are hard-blocked and external file
                # references are not read. Never pass unsafe=True.
                image_data = cairosvg.svg2png(bytestring=image_data, **kwargs)
            except Exception as exc:
                raise ValueError(f"Could not render SVG: {exc}") from exc

        pil_image = PILImage.open(BytesIO(image_data))
        actual_format = (pil_image.format or "").upper()
        max_dimension = image_resize_max_dimension()
        needs_resize = (
            pil_image.width > max_dimension or pil_image.height > max_dimension
        )
        needs_transcode = actual_format not in PAYLOAD_SAFE_FORMATS
        if not needs_resize and not needs_transcode:
            return image_data, PAYLOAD_FORMAT_MIME[actual_format]

        if needs_resize:
            pil_image.thumbnail(
                (max_dimension, max_dimension), PILImage.Resampling.LANCZOS
            )
        if actual_format == 'JPEG':
            save_format, save_kwargs = 'JPEG', {'optimize': True, 'quality': 85}
        elif actual_format == 'WEBP':
            save_format, save_kwargs = 'WEBP', {'quality': 85}
        else:
            # PNG stays PNG; GIF and every non-payload-safe format transcode
            # to PNG (the legacy else-branch, now with a truthful mime).
            save_format, save_kwargs = 'PNG', {'optimize': True}
        if save_format == 'PNG' and pil_image.mode not in (
            '1', 'L', 'LA', 'I', 'P', 'RGB', 'RGBA'
        ):
            pil_image = pil_image.convert('RGB')  # e.g. CMYK — PNG can't encode it
        buffer = BytesIO()
        pil_image.save(buffer, format=save_format, **save_kwargs)
        return buffer.getvalue(), PAYLOAD_FORMAT_MIME[save_format]

    @staticmethod
    async def _process_image_data(
        image_data: bytes,
        extension: str,
        mime_type: str
    ) -> bytes:
        """Legacy adapter: bytes-only view of prepare_image_payload.

        Signature and return shape are pinned by existing callers and tests;
        mime-aware callers use prepare_image_payload directly.
        """
        processed_data, _mime = await ChatImageHandler.prepare_image_payload(
            image_data, extension
        )
        return processed_data

    @staticmethod
    def validate_image_data(image_data: bytes) -> bool:
        """
        Validate that the bytes represent a valid image.
        
        Args:
            image_data: Image bytes to validate
            
        Returns:
            True if valid image, False otherwise
        """
        try:
            # Try to open the image
            PILImage.open(BytesIO(image_data))
            return True
        except Exception:
            return False
    
    @staticmethod
    def get_image_info(image_data: bytes) -> dict:
        """
        Get information about an image.
        
        Args:
            image_data: Image bytes
            
        Returns:
            Dictionary with image information
        """
        try:
            pil_image = PILImage.open(BytesIO(image_data))
            return {
                'width': pil_image.width,
                'height': pil_image.height,
                'format': pil_image.format,
                'mode': pil_image.mode,
                'size_kb': len(image_data) / 1024
            }
        except Exception as e:
            logging.error(f"Error getting image info: {e}")
            return {
                'error': str(e),
                'size_kb': len(image_data) / 1024
            }

#
#
#######################################################################################################################