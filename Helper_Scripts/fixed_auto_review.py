#!/usr/bin/env python3
"""auto_review.py
PreToolUse hook for Claude Code's Edit, Write, and MultiEdit tools.

Reads the JSON event object from STDIN, generates a unified diff for the
current patch, feeds that plus the user's most recent prompt to a selected
LLM (DeepSeek by default) for automatic review, then exits:

* **0** – Review passed, Claude continues.
* **2** – Review failed; JSON issues are emitted on STDERR. Claude will try to fix.

Set ENV:
    DEEPSEEK_API_KEY      – required for provider "deepseek"
    GOOGLE_API_KEY        – optional, if provider "google" chosen
    REVIEW_PROVIDER       – "deepseek" (default) or "google"
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Import requests - you'll need to install this: pip install requests
try:
    import requests
except ImportError:
    logging.error("requests module not found. Please install with: pip install requests")
    sys.exit(1)

# --------------------------------------------------------------------------- #
#  Logging
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] auto_review: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)

# --------------------------------------------------------------------------- #
#  LLM client selection
# --------------------------------------------------------------------------- #

PROVIDER = os.getenv("REVIEW_PROVIDER", "deepseek").lower()


def chat_with_deepseek(
        input_data: List[Dict[str, Any]],
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        system_message: Optional[str] = None,
        temp: Optional[float] = None,
        streaming: Optional[bool] = False,
        max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Simplified DeepSeek chat function."""
    final_api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not final_api_key:
        raise RuntimeError("DeepSeek API Key required. Set DEEPSEEK_API_KEY environment variable.")

    current_model = model or "deepseek-chat"
    current_temp = temp if temp is not None else 0.1
    current_max_tokens = max_tokens or 1024

    api_messages = []
    if system_message:
        api_messages.append({"role": "system", "content": system_message})
    api_messages.extend(input_data)

    headers = {'Authorization': f'Bearer {final_api_key}', 'Content-Type': 'application/json'}
    data = {
        "model": current_model,
        "messages": api_messages,
        "stream": False,  # Always non-streaming for simplicity
        "temperature": current_temp,
        "max_tokens": current_max_tokens,
    }

    api_url = 'https://api.deepseek.com/chat/completions'

    try:
        response = requests.post(api_url, headers=headers, json=data, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        logging.error(f"DeepSeek API HTTP error: {e}")
        raise
    except Exception as e:
        logging.error(f"DeepSeek API error: {e}")
        raise


def chat_with_google(
        input_data: List[Dict[str, Any]],
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        system_message: Optional[str] = None,
        temp: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Simplified Google Gemini chat function."""
    final_api_key = api_key or os.getenv("GOOGLE_API_KEY")
    if not final_api_key:
        raise RuntimeError("Google API Key required. Set GOOGLE_API_KEY environment variable.")

    current_model = model or "gemini-1.5-flash-latest"

    # Convert messages to Gemini format
    gemini_contents = []
    for msg in input_data:
        role = msg.get("role")
        content = msg.get("content")
        gemini_role = "user" if role == "user" else "model" if role == "assistant" else None
        if gemini_role and content:
            gemini_contents.append({
                "role": gemini_role,
                "parts": [{"text": content}]
            })

    generation_config = {}
    if temp is not None:
        generation_config["temperature"] = temp
    if max_output_tokens is not None:
        generation_config["maxOutputTokens"] = max_output_tokens

    payload = {"contents": gemini_contents}
    if generation_config:
        payload["generationConfig"] = generation_config
    if system_message:
        payload["system_instruction"] = {"parts": [{"text": system_message}]}

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{current_model}:generateContent"
    headers = {'x-goog-api-key': final_api_key, 'Content-Type': 'application/json'}

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()

        # Convert Gemini response to OpenAI format
        response_data = response.json()
        assistant_content = ""

        if response_data.get("candidates"):
            candidate = response_data["candidates"][0]
            if candidate.get("content", {}).get("parts"):
                for part in candidate["content"]["parts"]:
                    if "text" in part:
                        assistant_content += part.get("text", "")

        return {
            "id": f"gemini-{time.time_ns()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": current_model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": assistant_content.strip()},
                "finish_reason": "stop"
            }]
        }
    except requests.exceptions.HTTPError as e:
        logging.error(f"Google API HTTP error: {e}")
        raise
    except Exception as e:
        logging.error(f"Google API error: {e}")
        raise


# -----------------------------------------------------------------------------
# ------------------------------ LLM Call -------------------------------------
# -----------------------------------------------------------------------------
def _call_llm(prompt: str, stream: bool = False) -> Dict[str, Any]:
    """Call the configured provider and return the response object."""
    if PROVIDER == "google":
        return chat_with_google(
            input_data=[{"role": "user", "content": prompt}],
            model=os.getenv("GOOGLE_MODEL", "gemini-1.5-flash-latest"),
            max_output_tokens=int(os.getenv("GOOGLE_MAX_TOKENS", "1024")),
            temp=float(os.getenv("GOOGLE_TEMPERATURE", "0.1")),
            api_key=os.getenv("GOOGLE_API_KEY"),
            system_message="You are a strict senior code reviewer.",
        )
    # default to deepseek
    return chat_with_deepseek(
        input_data=[{"role": "user", "content": prompt}],
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        max_tokens=int(os.getenv("DEEPSEEK_MAX_TOKENS", "1024")),
        temp=float(os.getenv("DEEPSEEK_TEMPERATURE", "0.1")),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        system_message="You are a strict senior code reviewer.",
    )


# --------------------------------------------------------------------------- #
#  Helper functions
# --------------------------------------------------------------------------- #


def _last_user_message(transcript_path: str) -> str:
    """Return the most recent user message text from a Claude transcript file."""
    try:
        last_msg = ""
        with open(transcript_path, "r", encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                if record.get("role") == "user":
                    last_msg = record.get("text", "")
        return last_msg.strip()
    except Exception as exc:
        logging.warning("Could not parse transcript %s: %s", transcript_path, exc)
        return ""


def _unified_diff(orig_path: Path, new_content: str) -> str:
    """Write *new_content* to a temp file and return `diff -u` vs *orig_path*."""
    with tempfile.NamedTemporaryFile("w+", delete=False) as tmp:
        tmp.write(new_content)
        tmp.flush()
        cmd = ["diff", "-u", str(orig_path), tmp.name]
        try:
            diff_bytes = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as cpe:
            # diff returns exit-status 1 when files differ – this is expected
            diff_bytes = cpe.output
        return diff_bytes or "(no diff – empty change)"


def _read_file_content(file_path: Path) -> str:
    """Read the current content of a file."""
    try:
        if file_path.exists():
            return file_path.read_text(encoding='utf-8')
    except Exception as e:
        logging.warning(f"Could not read file {file_path}: {e}")
    return ""


def _apply_multi_edits(original_content: str, edits: List[Dict[str, Any]]) -> str:
    """Apply MultiEdit edits sequentially to produce the final content."""
    content = original_content
    for edit in edits:
        old_string = edit.get("old_string", "")
        new_string = edit.get("new_string", "")
        replace_all = edit.get("replace_all", False)
        
        if old_string in content:
            if replace_all:
                content = content.replace(old_string, new_string)
            else:
                # Replace only first occurrence
                content = content.replace(old_string, new_string, 1)
    
    return content


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #


def main() -> None:
    # 1️⃣ Read JSON event from Claude
    try:
        event: Dict[str, Any] = json.load(sys.stdin)
    except json.JSONDecodeError:
        logging.error("STDIN did not contain valid JSON event.")
        sys.exit(1)

    tool_input = event.get("tool_input", {})
    tool_name = event.get("tool_name", "")
    file_path_str = tool_input.get("file_path", "")
    file_path = Path(file_path_str) if file_path_str else None

    # Handle different tool types
    if tool_name == "MultiEdit":
        # For MultiEdit, we need to read the current file and apply edits
        if not file_path:
            logging.error("MultiEdit requires file_path")
            sys.exit(1)
        
        edits = tool_input.get("edits", [])
        if not edits:
            logging.error("MultiEdit requires edits array")
            sys.exit(1)
        
        # Read current file content
        original_content = _read_file_content(file_path)
        
        # Apply all edits to get the final content
        new_content = _apply_multi_edits(original_content, edits)
        
    elif tool_name == "Edit":
        new_content = tool_input.get("new_string", "")
    elif tool_name == "Write":
        new_content = tool_input.get("content", "")
    else:
        new_content = tool_input.get("new_string") or tool_input.get("content", "")

    transcript_path = event.get("transcript_path")

    if not file_path or not new_content:
        logging.error(f"Missing file_path or content. file_path='{file_path}', content_length={len(new_content) if new_content else 0}, tool_name='{tool_name}'")
        sys.exit(1)

    # 2️⃣ Collect context
    last_user_prompt = _last_user_message(transcript_path) if transcript_path else ""
    patch = _unified_diff(file_path, new_content)

    # 3️⃣ Build review prompt
    prompt = textwrap.dedent(
        f"""\
        Act as an uncompromising senior code reviewer.

        ### User request
        {last_user_prompt or "(prompt unavailable)"}

        ### Patch
        {patch}

        Analyse whether the patch:
        1. Implements exactly what the user asked – no scope creep.
        2. Introduces no hard-coded secrets, credentials, magic numbers.
        3. Adds no unnecessary fallbacks / dead code.
        4. Follows project style conventions.

        Respond in *single-line* JSON:
        {{
          "pass": <true|false>,
          "issues": ["bullet 1", "bullet 2", ...],
          "suggest": "concise overall advice"
        }}
        """
    )

    # 4️⃣ Call LLM
    try:
        llm_resp = _call_llm(prompt, stream=False)
    except Exception as exc:
        logging.error("LLM call failed: %s", exc, exc_info=True)
        sys.exit(1)

    # 5️⃣ Extract assistant JSON
    try:
        # Normalized response shape for DeepSeek/Google
        content = (
            llm_resp["choices"][0]["message"].get("content")
            if "choices" in llm_resp
            else None
        )

        # Log the raw content for debugging
        logging.info(f"LLM response content: {content[:500] if content else 'None'}")

        if not content:
            logging.error("LLM returned empty content")
            sys.exit(1)

        # Try to extract JSON from the content
        # Sometimes LLMs wrap JSON in markdown code blocks
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = content[json_start:json_end]
        else:
            json_str = content

        result = json.loads(json_str)
    except json.JSONDecodeError as exc:
        logging.error(f"Could not parse LLM JSON: {exc}")
        logging.error(f"Raw content: {content[:1000] if content else 'None'}")
        # Fallback: assume review passed if we can't parse
        result = {"pass": True, "issues": [], "suggest": "JSON parse error - assuming pass"}
    except Exception as exc:
        logging.error(f"Unexpected error parsing LLM response: {exc}")
        sys.exit(1)

    passed = bool(result.get("pass"))
    if passed:
        logging.info("Review passed.")
        sys.exit(0)

    # 6️⃣ Emit issues on stderr and exit 2 to block Claude
    sys.stderr.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    logging.info("Review failed with %d issue(s).", len(result.get("issues", [])))
    sys.exit(2)


if __name__ == "__main__":
    main()