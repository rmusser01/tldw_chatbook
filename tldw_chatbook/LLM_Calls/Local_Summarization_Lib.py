# Local_Summarization_Lib.py
#########################################
# Local Summarization Library
# This library is used to perform summarization with a 'local' inference engine.
#
####
#
####################
# Function List
# FIXME - UPDATE Function Arguments
# 1. summarize_with_local_llm(text, custom_prompt_arg)
# 2. summarize_with_llama(api_url, text, token, custom_prompt)
# 3. summarize_with_kobold(api_url, text, kobold_api_token, custom_prompt)
# 4. summarize_with_oobabooga(api_url, text, ooba_api_token, custom_prompt)
# 5. summarize_with_vllm(vllm_api_url, vllm_api_key_function_arg, llm_model, text, vllm_custom_prompt_function_arg)
# 6. summarize_with_tabbyapi(tabby_api_key, tabby_api_IP, text, tabby_model, custom_prompt)
# 7. save_summary_to_file(summary, file_path)
#
###############################
# Import necessary libraries
import json
import os

# Import 3rd-party Libraries
import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry

#
# Import Local Libraries
from tldw_chatbook.Utils.Utils import extract_text_from_segments, logging
from tldw_chatbook.Utils.persistent_diagnostics import safe_metadata_token
from tldw_chatbook.config import load_settings
from tldw_chatbook.Internal_Prompts import get_internal_prompt

#
#######################################################################################################################
# Function Definitions
#


def summarize_with_local_llm(
    input_data, custom_prompt_arg, temp, system_message=None, streaming=False
):
    try:
        logging.debug("openai: Using provided string data for summarization")
        data = input_data

        logging.debug(f"Local LLM: Type of data: {type(data)}")

        if isinstance(data, dict) and "summary" in data:
            # If the loaded data is a dictionary and already contains a summary, return it
            logging.debug("Local LLM: Summary already exists in the loaded data")
            return data["summary"]

        temp = temp or 0.7
        # If the loaded data is a list of segment dictionaries or a string, proceed with summarization
        if isinstance(data, list):
            segments = data
            text = extract_text_from_segments(segments)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError("Invalid input data format")

        if system_message is None:
            system_message = "You are a helpful AI assistant."

        headers = {"Content-Type": "application/json"}

        logging.debug("Local LLM: Preparing data + prompt for submittal")
        local_llm_prompt = f"{text} \n\n\n\n{custom_prompt_arg}"
        data = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": local_llm_prompt},
            ],
            "max_tokens": 4096,
            "temperature": temp,
            "stream": streaming,
        }

        logging.debug("Local LLM: Posting request")
        response = requests.post(
            "http://127.0.0.1:8080/v1/chat/completions",
            headers=headers,
            json=data,
        )

        if response.status_code == 200:
            if streaming:
                logging.debug("Local LLM: Processing streaming response")

                def stream_generator():
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode("utf-8").strip()
                            if decoded_line.startswith("data:"):
                                data_str = decoded_line[len("data:") :].strip()
                                if data_str == "[DONE]":
                                    break
                                try:
                                    data_json = json.loads(data_str)
                                    if (
                                        "choices" in data_json
                                        and len(data_json["choices"]) > 0
                                    ):
                                        delta = data_json["choices"][0].get("delta", {})
                                        if "content" in delta:
                                            content = delta["content"]
                                            yield content
                                except json.JSONDecodeError:
                                    logging.error(
                                        "Local LLM: Failed to decode streamed JSON; "
                                        "line_length=%s",
                                        len(decoded_line),
                                    )
                                    continue

                return stream_generator()
            else:
                logging.debug("Local LLM: Processing non-streaming response")
                response_data = response.json()
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    summary = response_data["choices"][0]["message"]["content"].strip()
                    logging.debug("Local LLM: Summarization successful")
                    logging.info("Local LLM: Summarization successful.")
                    return summary
                else:
                    logging.warning("Local LLM: Summary not found in the response data")
                    return "Local LLM: Summary not available"
        else:
            logging.error(
                f"Local LLM: Request failed with status code {response.status_code}"
            )
            return f"Local LLM: Failed to process summary, status code {response.status_code}"
    except Exception as e:
        logging.error(
            "Local LLM: Processing failed; exception_type=%s",
            safe_metadata_token(type(e).__name__),
        )
        return f"Local LLM: Error occurred while processing summary: {str(e)}"


#: Completion budget used when neither the modern api_settings entry nor the
#: legacy section names one (Qodo, PR 1774: the literal was repeated at each
#: fallback). Measured against a thinking model, this is TIGHT: a real 6000-char
#: chunk spent 4028 of these on reasoning before emitting content (task-17384).
DEFAULT_SUMMARY_MAX_TOKENS = 4096


def _resolve_local_provider_config(
    loaded_config_data, modern_key: str, legacy_key: str
) -> tuple[dict, dict]:
    """Return ``(modern, legacy)`` config tables for a local provider.

    task-17383: several summarizers indexed sections the loader has never built
    (`api_keys`, `local_api_ip`, `models`), so they raised before contacting a
    server and reported failure by RETURNING an error string -- which the
    deep-search caller could store as a result's evidence. Resolved by name and
    defensively here, never by index.

    Args:
        loaded_config_data: The loaded settings mapping (may be None).
        modern_key: Provider key under ``api_settings`` (e.g. "koboldcpp").
        legacy_key: Historical top-level section (e.g. "kobold_api").

    Returns:
        The two tables, each an empty dict when absent.
    """
    if not isinstance(loaded_config_data, dict):
        return {}, {}
    modern = (loaded_config_data.get("api_settings") or {}).get(modern_key) or {}
    legacy = loaded_config_data.get(legacy_key) or {}
    return (
        modern if isinstance(modern, dict) else {},
        legacy if isinstance(legacy, dict) else {},
    )


def _resolve_provider_credential(parameter_key, modern: dict, legacy: dict):
    """Credential for a local provider: the caller's parameter, then the modern
    table, then the legacy section, then the environment variable the config
    NAMES (``api_key_env_var`` -- this repo's existing convention, honoured by
    the media and settings windows; tabbyapi's modern table carries only that).

    Args:
        parameter_key: Key passed by the caller, if any.
        modern: Modern per-provider table.
        legacy: Legacy section.

    Returns:
        The resolved key, or None when nothing supplies one.
    """
    if parameter_key and str(parameter_key).strip():
        return str(parameter_key).strip()
    declared = False
    for table in (modern, legacy):
        if "api_key" not in table:
            continue
        candidate = table.get("api_key")
        if candidate is None:
            # Declared as null reads as ABSENT, not blank: these functions
            # refuse a run with no credential at all while proceeding without
            # an Authorization header for a configured empty string.
            continue
        declared = True
        if str(candidate).strip():
            return str(candidate).strip()
    env_name = str(modern.get("api_key_env_var") or "").strip()
    if env_name:
        env_value = os.environ.get(env_name, "").strip()
        if env_value:
            return env_value
    # "Configured but blank" is NOT the same as "absent": these summarizers
    # proceed without an Authorization header for the former and refuse the
    # latter, and their tests pin that distinction. Collapsing both to None
    # turned a working blank-credential call into a failure.
    return "" if declared else None


def summarize_with_llama(
    input_data,
    custom_prompt,
    api_key=None,
    temp=None,
    system_message=None,
    streaming=False,
):
    try:
        logging.debug("Llama.cpp: Loading and validating configurations")
        loaded_config_data = load_settings()
        # task-17382: this function indexed a `llama_api` section in ten
        # places. No such section has ever existed -- the loader builds
        # `llama_cpp_api` -- so the FIRST read raised KeyError, the except at
        # the bottom turned it into an error STRING, and the deep-search
        # caller stored that string as a result's evidence content. Resolved
        # once here, defensively, the way the chat handler does it.
        llama_config = {}
        if isinstance(loaded_config_data, dict):
            llama_config = loaded_config_data.get("llama_cpp_api") or {}
        if loaded_config_data is None:
            logging.error("Failed to load configuration data")
            llama_api_key = None
        else:
            # Prioritize the API key passed as a parameter
            if api_key and api_key.strip():
                llama_api_key = api_key
                logging.info("Llama.cpp: Using API key provided as parameter")
            else:
                # If no parameter is provided, use the key from the config
                llama_api_key = llama_config.get("api_key")
                if llama_api_key:
                    logging.info("Llama.cpp: Using API key from config file")
                else:
                    logging.warning("Llama.cpp: No API key found in config file")

        logging.info("llama.cpp: Attempting to use API URL from config file")
        # task-17382: prefer the modern api_settings entry -- what routes the
        # chat handler, and what a run priming a local endpoint sets -- over
        # the legacy section's api_ip, which otherwise sends every summary to
        # the default port regardless of where the run's model actually is.
        api_settings_llama = {}
        if isinstance(loaded_config_data, dict):
            api_settings_llama = (
                (loaded_config_data.get("api_settings") or {}).get("llama_cpp") or {}
            )
        configured_url = (
            api_settings_llama.get("api_url")
            or api_settings_llama.get("api_ip")
            or llama_config.get("api_ip")
        )
        if not configured_url:
            raise ValueError(
                "Llama.cpp Summarize: no API URL configured "
                "(api_settings.llama_cpp.api_url or llama_cpp_api.api_ip)"
            )
        # These keys legitimately hold any of a server root, a base ending in
        # /v1, a full chat-completions endpoint, or a bare host:port. This
        # function POSTs directly rather than going through the shared caller,
        # so it must land on the endpoint exactly once itself: normalize to the
        # origin with the same helper the chat handler uses, then append the
        # path. Posting a base URL raw returned llama-server's 404 "File Not
        # Found" (observed live during the task-17370 measurement).
        from ..Chat.console_provider_gateway import normalize_llamacpp_base_url

        api_url = f"{normalize_llamacpp_base_url(configured_url)}/v1/chat/completions"
        logging.debug("Llama: API endpoint configured")

        # Load transcript
        logging.debug("Llama.cpp: Using provided string data for summarization")
        data = input_data

        logging.debug(f"Llama.cpp Summarize: Type of data: {type(data)}")

        if isinstance(data, dict) and "summary" in data:
            # If the loaded data is a dictionary and already contains a summary, return it
            logging.debug(
                "Llama.cpp Summarize: Summary already exists in the loaded data"
            )
            return data["summary"]

        # If the loaded data is a list of segment dictionaries or a string, proceed with summarization
        if isinstance(data, list):
            segments = data
            text = extract_text_from_segments(segments)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError("Llama.cpp Summarize: Invalid input data format")

        # Prepare headers
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
        }
        if llama_api_key and len(llama_api_key) > 5:
            headers["Authorization"] = f"Bearer {llama_api_key}"

        # Prepare system message and prompt
        if system_message is None:
            system_message = "You are a helpful AI assistant."
        logging.debug("Llama Summarize: System prompt prepared")

        if custom_prompt is None:
            llama_prompt = f"{get_internal_prompt('summarization.local_summarizer_template')}\n\n{text}"
        else:
            llama_prompt = f"{custom_prompt}\n\n{text}"

        logging.debug(
            "Llama Summarize: Prompt prepared; character_count=%s",
            len(llama_prompt),
        )

        # Temperature handling
        if temp is None:
            # Check config
            if "temperature" in llama_config:
                temp = llama_config["temperature"]
                temp = float(temp)
            else:
                temp = 0.7
        logging.debug(f"Llama: Using temperature: {temp}")

        # Check for max tokens. task-17384: prefer the modern api_settings entry
        # for the same reason as the URL above -- that is what a run priming a
        # local endpoint sets, and reading only the legacy section left this at
        # 4096 while the chat path ran on 16384. Captured live on a real
        # 6000-char chunk: the model spent 4028 of 4096 completion tokens on
        # reasoning_content and emitted 465 characters of content, so a chunk
        # that reasons slightly longer returns EMPTY content -- which is exactly
        # how map-reduce chunk summarization was failing.
        raw_max_tokens = api_settings_llama.get("max_tokens")
        if raw_max_tokens is None:
            raw_max_tokens = llama_config.get("max_tokens")
        try:
            max_tokens = (
                int(raw_max_tokens)
                if raw_max_tokens is not None
                else DEFAULT_SUMMARY_MAX_TOKENS
            )
        except (TypeError, ValueError):
            max_tokens = DEFAULT_SUMMARY_MAX_TOKENS
        if max_tokens < 1:
            max_tokens = DEFAULT_SUMMARY_MAX_TOKENS
        logging.debug(f"Llama: Using max tokens: {max_tokens}")

        # Check for streaming
        if not isinstance(streaming, bool):
            if "streaming" in llama_config:
                streaming = llama_config["streaming"]
                streaming = bool(streaming)
        logging.debug(f"Llama: Streaming mode: {streaming}")

        # Prepare data payload
        data = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": llama_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temp,
            "stream": streaming,
        }

        # Create a session
        session = requests.Session()

        # Load config values
        retry_count = int(llama_config.get("api_retries", 3))
        retry_delay = int(llama_config.get("api_retry_delay", 5))

        # Configure the retry strategy
        retry_strategy = Retry(
            total=retry_count,  # Total number of retries
            backoff_factor=retry_delay,  # A delay factor (exponential backoff)
            status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
        )

        # Create the adapter
        adapter = HTTPAdapter(max_retries=retry_strategy)

        # Mount adapters for both HTTP and HTTPS
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        logging.debug("Llama: Submitting request to API endpoint")
        response = session.post(api_url, headers=headers, json=data, stream=streaming)

        if response.status_code == 200:
            if streaming:
                logging.debug("Llama: Processing streaming response")

                def stream_generator():
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode("utf-8").strip()
                            if decoded_line.startswith("data:"):
                                data_str = decoded_line[len("data:") :].strip()
                                if data_str == "[DONE]":
                                    break
                                try:
                                    data_json = json.loads(data_str)
                                    if (
                                        "choices" in data_json
                                        and len(data_json["choices"]) > 0
                                    ):
                                        delta = data_json["choices"][0].get("delta", {})
                                        if "content" in delta:
                                            content = delta["content"]
                                            yield content
                                except json.JSONDecodeError:
                                    logging.error(
                                        "Llama: Failed to decode streamed JSON; "
                                        "line_length=%s",
                                        len(decoded_line),
                                    )
                                    continue

                return stream_generator()
            else:
                logging.debug("Llama.cpp Summarizer: Processing non-streaming response")
                response_data = response.json()
                # task-17382: this parsed ONLY llama.cpp's native
                # `{"content": ...}` shape while posting to
                # /v1/chat/completions, whose payload puts the text under
                # choices[0].message.content -- so every real chunk
                # summarization came back "No choices in response data" once
                # the endpoint was reached. Accept the OpenAI shape first,
                # then the native one, so either endpoint works.
                summary = ""
                if isinstance(response_data, dict):
                    choices = response_data.get("choices")
                    if isinstance(choices, list) and choices:
                        first = choices[0] if isinstance(choices[0], dict) else {}
                        message = first.get("message")
                        if isinstance(message, dict):
                            summary = str(message.get("content") or "")
                        if not summary:
                            summary = str(first.get("text") or "")
                    if not summary:
                        summary = str(response_data.get("content") or "")
                summary = summary.strip()
                if summary:
                    logging.debug("llama: Summarization successful")
                    logging.info("Summarization successful.")
                    return summary
                # The log line stays verbatim -- it is tracked in the reviewed
                # diagnostic inventory (task-492/3750) and this change is about
                # the RETURNED value, which is what a caller surfaces in a run's
                # warnings. task-17384: "no choices" was a guess at the cause;
                # the real one is a completion that spent its token budget on
                # reasoning and emitted no content. The "no choices in response"
                # prefix is preserved so the deep-search failure detector still
                # recognizes it.
                logging.error("Llama: No choices in response data")
                detail = ""
                if isinstance(response_data, dict):
                    first = (response_data.get("choices") or [{}])[0]
                    if isinstance(first, dict):
                        finish = first.get("finish_reason")
                        message = first.get("message")
                        reasoning = ""
                        if isinstance(message, dict):
                            reasoning = str(message.get("reasoning_content") or "")
                        spent = (response_data.get("usage") or {}).get(
                            "completion_tokens"
                        )
                        parts = []
                        if finish:
                            parts.append(f"finish_reason={finish}")
                        if spent is not None:
                            parts.append(f"completion_tokens={spent}/{max_tokens}")
                        if reasoning:
                            parts.append(
                                f"reasoning-only completion ({len(reasoning)} chars "
                                "of reasoning, no content)"
                            )
                        if parts:
                            detail = " (" + "; ".join(parts) + ")"
                return f"Llama: No choices in response data{detail}"
        else:
            logging.error(
                "Llama: API request failed; status_code=%s",
                response.status_code,
            )
            return f"Llama: API request failed: {response.text}"

    except Exception as e:
        logging.error(
            "Llama: Processing failed; exception_type=%s",
            safe_metadata_token(type(e).__name__),
        )
        return f"Llama: Error occurred while processing summary with Llama: {str(e)}"


# https://lite.koboldai.net/koboldcpp_api#/api%2Fv1/post_api_v1_generate
def summarize_with_kobold(
    input_data,
    api_key,
    custom_prompt_input,
    system_message=None,
    temp=None,
    kobold_api_ip="http://127.0.0.1:5001/api/v1/generate",
    streaming=False,
):
    logging.debug("Kobold: Summarization process starting...")
    try:
        logging.debug("Kobold: Loading and validating configurations")
        loaded_config_data = load_settings()
        # task-17383: this function indexed `api_keys` and `local_api_ip`,
        # names the loader has never built, so it raised before reaching a
        # server. Both a modern api_settings entry and a legacy section exist.
        kobold_modern, kobold_legacy = _resolve_local_provider_config(
            loaded_config_data, "koboldcpp", "kobold_api"
        )
        if loaded_config_data is None:
            logging.error("Failed to load configuration data")
            kobold_api_key = None
        else:
            # Prioritize the API key passed as a parameter
            if api_key and api_key.strip():
                kobold_api_key = api_key
                logging.info("Kobold: Using API key provided as parameter")
            else:
                # If no parameter is provided, use the key from the config
                kobold_api_key = _resolve_provider_credential(
                    None, kobold_modern, kobold_legacy
                )
                if kobold_api_key:
                    logging.info("Kobold: Using API key from config file")
                else:
                    logging.warning("Kobold: No API key found in config file")
            # Get the Streaming API IP from the config
            kobold_openai_api_IP = (
                kobold_modern.get("api_url")
                or kobold_legacy.get("api_streaming_ip")
                or kobold_legacy.get("api_ip")
            )

        if kobold_api_key is None:
            raise TypeError("'NoneType' object is not subscriptable")
        logging.debug("Kobold: Credential state resolved")

        logging.debug("Kobold.cpp: Using provided string data for summarization")
        data = input_data

        logging.debug(f"Kobold.cpp: Type of data: {type(data)}")

        if isinstance(data, dict) and "summary" in data:
            # If the loaded data is a dictionary and already contains a summary, return it
            logging.debug("Kobold.cpp: Summary already exists in the loaded data")
            return data["summary"]

        # If the loaded data is a list of segment dictionaries or a string, proceed with summarization
        if isinstance(data, list):
            segments = data
            text = extract_text_from_segments(segments)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError("Kobold.cpp: Invalid input data format")

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
        }
        if custom_prompt_input is None:
            kobold_prompt = f"{get_internal_prompt('summarization.local_summarizer_template')}\n\n\n\n{text}"
        else:
            kobold_prompt = f"{custom_prompt_input}\n\n\n\n{text}"

        logging.debug(
            "Kobold summarization: Prompt prepared; character_count=%s",
            len(kobold_prompt),
        )

        # Construct the data payload
        data_payload = {
            "max_context_length": 8096,
            "max_length": 4096,
            "prompt": kobold_prompt,
            "temperature": 0.7,
            "stream": streaming,
            # Include other parameters if needed
            # "top_p": 0.9,
            # "top_k": 100,
            # "rep_penalty": 1.0,
        }

        logging.debug("Kobold Summarization: Submitting request to API endpoint")
        logging.info("Kobold Summarization: Submitting request to API endpoint")
        kobold_api_ip = kobold_modern.get("api_url") or kobold_legacy.get("api_ip")
        if not kobold_api_ip:
            raise ValueError(
                "Kobold Summarize: no API URL configured "
                "(api_settings.koboldcpp.api_url or kobold_api.api_ip)"
            )

        if streaming:
            logging.debug("Kobold Summarization: Streaming mode enabled")
            try:
                # Create a session
                session = requests.Session()

                # Load config values
                retry_count = kobold_legacy["api_retries"]
                retry_delay = kobold_legacy["api_retry_delay"]

                # Configure the retry strategy
                retry_strategy = Retry(
                    total=retry_count,  # Total number of retries
                    backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                    status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
                )

                # Create the adapter
                adapter = HTTPAdapter(max_retries=retry_strategy)

                # Mount adapters for both HTTP and HTTPS
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                # Send the request with streaming enabled
                response = session.post(
                    kobold_openai_api_IP,
                    headers=headers,
                    json=data_payload,
                    stream=True,
                )
                logging.debug(
                    "Kobold Summarization: API Response Status Code: %d",
                    response.status_code,
                )

                if response.status_code == 200:
                    # Process the streamed response
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode("utf-8")
                            # OpenAI API streams data prefixed with 'data: '
                            if decoded_line.startswith("data: "):
                                content = decoded_line[len("data: ") :].strip()
                                if content == "[DONE]":
                                    break
                                try:
                                    data_chunk = json.loads(content)
                                    if (
                                        "choices" in data_chunk
                                        and len(data_chunk["choices"]) > 0
                                    ):
                                        delta = data_chunk["choices"][0].get(
                                            "delta", {}
                                        )
                                        text = delta.get("content", "")
                                        if text:
                                            yield text
                                    else:
                                        logging.error(
                                            "Kobold: Expected data not found in streamed response."
                                        )
                                except json.JSONDecodeError as e:
                                    logging.error(
                                        "Kobold: Failed to decode streamed JSON; exception_type=%s",
                                        safe_metadata_token(type(e).__name__),
                                    )
                else:
                    logging.error(
                        "Kobold: API request failed; status_code=%s",
                        response.status_code,
                    )
                    yield f"Kobold: API request failed: {response.text}"
            except Exception as e:
                logging.error(
                    "Kobold: Processing failed; exception_type=%s",
                    safe_metadata_token(type(e).__name__),
                )
                yield f"Kobold: Error occurred while processing summary with Kobold: {str(e)}"
        else:
            try:
                # Create a session
                session = requests.Session()

                # Load config values
                retry_count = kobold_legacy["api_retries"]
                retry_delay = kobold_legacy["api_retry_delay"]

                # Configure the retry strategy
                retry_strategy = Retry(
                    total=retry_count,  # Total number of retries
                    backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                    status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
                )

                # Create the adapter
                adapter = HTTPAdapter(max_retries=retry_strategy)

                # Mount adapters for both HTTP and HTTPS
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                response = session.post(
                    kobold_api_ip, headers=headers, json=data_payload
                )
                logging.debug(
                    "Kobold Summarization: API Response Status Code: %d",
                    response.status_code,
                )

                if response.status_code == 200:
                    try:
                        response_data = response.json()

                        if (
                            response_data
                            and "results" in response_data
                            and len(response_data["results"]) > 0
                        ):
                            summary = response_data["results"][0]["text"].strip()
                            logging.debug("Kobold: Summarization successful")
                            return summary
                        else:
                            logging.error("Expected data not found in API response.")
                            return "Expected data not found in API response."
                    except ValueError as e:
                        logging.error(
                            "Kobold: Failed to parse JSON response; exception_type=%s",
                            safe_metadata_token(type(e).__name__),
                        )
                        return f"Error parsing JSON response: {str(e)}"
                else:
                    logging.error(
                        "Kobold: API request failed; status_code=%s",
                        response.status_code,
                    )
                    return f"Kobold: API request failed: {response.text}"
            except Exception as e:
                logging.error(
                    "Kobold: Processing failed; exception_type=%s",
                    safe_metadata_token(type(e).__name__),
                )
                return f"Kobold: Error occurred while processing summary with Kobold: {str(e)}"
    except Exception as e:
        logging.error(
            "Kobold: Processing failed; exception_type=%s",
            safe_metadata_token(type(e).__name__),
        )
        return f"Kobold: Error occurred while processing summary with Kobold: {str(e)}"


# https://github.com/oobabooga/text-generation-webui/wiki/12-%E2%80%90-OpenAI-API
def summarize_with_oobabooga(
    input_data,
    api_key,
    custom_prompt,
    system_message=None,
    temp=None,
    api_url=None,
    streaming=False,
):
    logging.debug("Oobabooga: Summarization process starting...")
    try:
        logging.debug("Oobabooga: Loading and validating configurations")
        loaded_config_data = load_settings()
        ooba_api_key = None

        if loaded_config_data is None:
            logging.error("Failed to load configuration data")
        else:
            # Prioritize the API key passed as a parameter
            if api_key and api_key.strip():
                ooba_api_key = api_key
                logging.info("Oobabooga: Using API key provided as parameter")
            else:
                # If no parameter is provided, use the key from the config
                ooba_api_key = loaded_config_data["ooba_api"]["api_key"]
                if ooba_api_key:
                    logging.info("Oobabooga: Using API key from config file")
                else:
                    logging.warning("Oobabooga: No API key found in config file")

        if not api_url:
            api_url = loaded_config_data["ooba_api"]["api_ip"]
            logging.debug("Oobabooga: API endpoint configured")

        if not isinstance(api_url, str) or not api_url.startswith(
            ("http://", "https://")
        ):
            logging.error("Oobabooga: Invalid API URL configured")
            return "Oobabooga: Invalid API URL configured"
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
        }
        if ooba_api_key:
            headers["Authorization"] = f"Bearer {ooba_api_key}"
            logging.debug("Oobabooga: Credential configured")
        else:
            logging.debug("Oobabooga: No API key provided")

        # Input data handling
        if isinstance(input_data, str):
            if input_data.strip().startswith("{"):
                try:
                    data = json.loads(input_data)
                    logging.debug("Oobabooga: Parsed JSON string input")
                except json.JSONDecodeError as e:
                    logging.error(
                        "Oobabooga: Failed to parse JSON input; exception_type=%s",
                        safe_metadata_token(type(e).__name__),
                    )
                    return f"Oobabooga: Error parsing JSON input: {str(e)}"
            else:
                data = input_data
                logging.debug("Oobabooga: Using provided string data")
        else:
            data = input_data

        logging.debug(f"Oobabooga: Processed data type: {type(data)}")

        # Check for existing summary
        if isinstance(data, dict) and "summary" in data:
            logging.debug("Oobabooga: Summary already exists")
            return data["summary"]

        # Text extraction
        if isinstance(data, dict):
            if "segments" in data:
                text = extract_text_from_segments(data["segments"])
            else:
                text = json.dumps(data)
        elif isinstance(data, list):
            text = extract_text_from_segments(data)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError("Oobabooga: Invalid input data format")

        # Construct prompt
        summarizer_prompt = (
            "Please summarize the following text:"  # Define this if not already
        )
        if custom_prompt is None:
            custom_prompt = summarizer_prompt
        ooba_prompt = f"{text}\n\n\n\n{custom_prompt}"
        logging.debug(
            "Oobabooga: Prompt prepared; character_count=%s", len(ooba_prompt)
        )

        # System message handling
        if system_message is None:
            system_message = "You are a helpful AI assistant."

        # Temperature handling
        if temp is None:
            # Check config
            if "temperature" in loaded_config_data["ooba_api"]:
                temp = loaded_config_data["ooba_api"]["temperature"]
            else:
                temp = 0.7

        # Prepare API payload
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": ooba_prompt},
        ]
        data = {
            "mode": "chat",
            "messages": messages,
            "stream": streaming,
            "temperature": temp,
        }

        if streaming:
            logging.debug("Oobabooga: Streaming mode enabled")
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data["ooba_api"]["api_retries"]
            retry_delay = loaded_config_data["ooba_api"]["api_retry_delay"]

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            response = session.post(api_url, headers=headers, json=data, stream=True)
            response.raise_for_status()
            try:

                def stream_generator():
                    collected_messages = ""
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode("utf-8").strip()
                            if decoded_line.startswith("data: "):
                                content = decoded_line[len("data: ") :]
                                if content == "[DONE]":
                                    break
                                try:
                                    data_chunk = json.loads(content)
                                    if (
                                        "choices" in data_chunk
                                        and data_chunk["choices"]
                                    ):
                                        delta = data_chunk["choices"][0].get(
                                            "delta", {}
                                        )
                                        if "content" in delta:
                                            chunk = delta["content"]
                                            collected_messages += chunk
                                            yield chunk
                                except json.JSONDecodeError as e:
                                    logging.error(
                                        "Oobabooga: Failed to decode streamed JSON; "
                                        "exception_type=%s; line_length=%s",
                                        safe_metadata_token(type(e).__name__),
                                        len(content),
                                    )
                                    continue

                return stream_generator()
            except requests.RequestException as e:
                logging.error(
                    "Oobabooga: Streaming request failed; exception_type=%s",
                    safe_metadata_token(type(e).__name__),
                )
                return f"Error summarizing with Oobabooga: {str(e)}"
        else:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data["ooba_api"]["api_retries"]
            retry_delay = loaded_config_data["ooba_api"]["api_retry_delay"]

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            logging.debug("Oobabooga: Posting request")
            response = session.post(api_url, headers=headers, json=data)

            if response.status_code == 200:
                response_data = response.json()
                logging.debug("Ooba API request successful")
                if "choices" in response_data and response_data["choices"]:
                    logging.debug("Ooba API: Summarization successful")
                    summary = response_data["choices"][0]["message"]["content"].strip()
                    return summary
                else:
                    error_msg = f"Ooba API request failed: {response.status_code} - {response.text}"
                    logging.error(
                        "Ooba API: Response missing choices; status_code=%s",
                        response.status_code,
                    )
                    return error_msg
            else:
                logging.error(
                    f"Ooba API: Summarization failed with status code {response.status_code}"
                )
                logging.error(
                    "Ooba API: Error response received; status_code=%s",
                    response.status_code,
                )
                return f"Ooba API: Failed to process summary. Status code: {response.status_code}"
    except json.JSONDecodeError as e:
        logging.error(
            "Ooba API: Failed to decode JSON; exception_type=%s",
            safe_metadata_token(type(e).__name__),
        )
        return f"Ooba API: Error decoding JSON input: {str(e)}"
    except requests.RequestException as e:
        logging.error(
            "Ooba API: API request failed; exception_type=%s",
            safe_metadata_token(type(e).__name__),
        )
        return f"Ooba API: Error making API request: {str(e)}"
    except Exception as e:
        logging.error(
            "Ooba API: Unexpected failure; exception_type=%s",
            safe_metadata_token(type(e).__name__),
        )
        return f"Ooba API: Unexpected error occurred: {str(e)}"


def summarize_with_tabbyapi(
    input_data,
    custom_prompt_input,
    system_message=None,
    api_key=None,
    temp=None,
    api_IP="http://127.0.0.1:5000/v1/chat/completions",
    streaming=False,
):
    logging.debug("TabbyAPI: Summarization process starting...")
    try:
        logging.debug("TabbyAPI: Loading and validating configurations")
        loaded_config_data = load_settings()
        # task-17383: same defect -- `api_keys`, `local_api_ip` and `models`
        # are names nothing produces.
        tabby_modern, tabby_legacy = _resolve_local_provider_config(
            loaded_config_data, "tabbyapi", "tabby_api"
        )
        if loaded_config_data is None:
            logging.error("Failed to load configuration data")
            tabby_api_key = None
        else:
            # Prioritize the API key passed as a parameter
            if api_key and api_key.strip():
                tabby_api_key = api_key
                logging.info("TabbyAPI: Using API key provided as parameter")
            else:
                # If no parameter is provided, use the key from the config
                tabby_api_key = _resolve_provider_credential(
                    None, tabby_modern, tabby_legacy
                )
                if tabby_api_key:
                    logging.info("TabbyAPI: Using API key from config file")
                else:
                    logging.warning("TabbyAPI: No API key found in config file")

        # Set API IP and model from config.txt
        tabby_api_ip = tabby_modern.get("api_url") or tabby_legacy.get("api_ip")
        if not tabby_api_ip:
            raise ValueError(
                "TabbyAPI Summarize: no API URL configured "
                "(api_settings.tabbyapi.api_url or tabby_api.api_ip)"
            )
        tabby_model = tabby_modern.get("model") or tabby_legacy.get("model")
        if temp is None:
            temp = 0.7

        if tabby_api_key is None:
            raise TypeError("'NoneType' object is not subscriptable")
        logging.debug("TabbyAPI: Credential state resolved")

        # Process input data
        logging.debug("TabbyAPI: Using provided data for summarization")
        data = input_data

        logging.debug("TabbyAPI: Input received")
        logging.debug(f"TabbyAPI: Type of data: {type(data)}")

        if isinstance(data, dict) and "summary" in data:
            logging.debug("TabbyAPI: Summary already exists in the loaded data")
            return data["summary"]

        # Extract text for summarization
        if isinstance(data, list):
            segments = data
            text = extract_text_from_segments(segments)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError("Invalid input data format")

        if system_message is None:
            system_message = "You are a helpful AI assistant."

        if custom_prompt_input is None:
            custom_prompt_input = f"{get_internal_prompt('summarization.local_summarizer_template')}\n\n\n\n{text}"
        else:
            custom_prompt_input = f"{custom_prompt_input}\n\n\n\n{text}"

        headers = {"Content-Type": "application/json"}
        if tabby_api_key:
            headers["Authorization"] = f"Bearer {tabby_api_key}"

        data2 = {
            "model": tabby_model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": custom_prompt_input},
            ],
            "temperature": temp,
            "max_tokens": 4096,
            "min_tokens": 0,
            #'top_p': 1.0,
            #'top_k': 0,
            #'frequency_penalty': 0,
            #'presence_penalty': 0.0,
            # "repetition_penalty": 1.0,
            "stream": streaming,
        }

        if streaming:
            logging.debug("TabbyAPI: Streaming mode enabled")
            try:
                # Create a session
                session = requests.Session()

                # Load config values
                retry_count = tabby_legacy["api_retries"]
                retry_delay = tabby_legacy["api_retry_delay"]

                # Configure the retry strategy
                retry_strategy = Retry(
                    total=retry_count,  # Total number of retries
                    backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                    status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
                )

                # Create the adapter
                adapter = HTTPAdapter(max_retries=retry_strategy)

                # Mount adapters for both HTTP and HTTPS
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                response = session.post(
                    tabby_api_ip, headers=headers, json=data2, stream=True
                )
                response.raise_for_status()
                # Process the streamed response
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode("utf-8").strip()
                        if decoded_line.startswith("data: "):
                            data_line = decoded_line[len("data: ") :]
                            if data_line == "[DONE]":
                                break
                            try:
                                data_json = json.loads(data_line)
                                if (
                                    "choices" in data_json
                                    and len(data_json["choices"]) > 0
                                ):
                                    delta = data_json["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield content
                            except json.JSONDecodeError as e:
                                logging.error(
                                    "TabbyAPI: Failed to parse streamed JSON; "
                                    "exception_type=%s; line_length=%s",
                                    safe_metadata_token(type(e).__name__),
                                    len(data_line),
                                )
                        else:
                            logging.debug(
                                "TabbyAPI: Ignored non-data stream line; line_length=%s",
                                len(decoded_line),
                            )
            except requests.exceptions.RequestException as e:
                logging.error(
                    "TabbyAPI: Streaming request failed; exception_type=%s",
                    safe_metadata_token(type(e).__name__),
                )
                yield f"Error summarizing with TabbyAPI: {str(e)}"
            except Exception as e:
                logging.error(
                    "TabbyAPI: Streaming failed; exception_type=%s",
                    safe_metadata_token(type(e).__name__),
                )
                yield f"Unexpected error in summarization process: {str(e)}"
        else:
            try:
                # Create a session
                session = requests.Session()

                # Load config values
                retry_count = tabby_legacy["api_retries"]
                retry_delay = tabby_legacy["api_retry_delay"]

                # Configure the retry strategy
                retry_strategy = Retry(
                    total=retry_count,  # Total number of retries
                    backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                    status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
                )

                # Create the adapter
                adapter = HTTPAdapter(max_retries=retry_strategy)

                # Mount adapters for both HTTP and HTTPS
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                response = session.post(tabby_api_ip, headers=headers, json=data2)
                response.raise_for_status()
                response_json = response.json()

                # Validate the response structure
                if all(
                    key in response_json
                    for key in ["id", "choices", "created", "model", "object", "usage"]
                ):
                    logging.info("TabbyAPI: Received a valid 200 response")
                    summary = (
                        response_json["choices"][0]
                        .get("message", {})
                        .get("content", "")
                    )
                    return summary
                else:
                    logging.error(
                        "TabbyAPI: Received a 200 response, but the structure is invalid"
                    )
                    return (
                        "Error: Received an invalid response structure from TabbyAPI."
                    )

            except requests.exceptions.RequestException as e:
                logging.error(
                    "TabbyAPI: Request failed; exception_type=%s",
                    safe_metadata_token(type(e).__name__),
                )
                return f"Error summarizing with TabbyAPI: {str(e)}"
            except json.JSONDecodeError:
                logging.error("TabbyAPI: Received an invalid JSON response")
                return (
                    "TabbyAPI: Error: Received an invalid JSON response from TabbyAPI."
                )
            except Exception as e:
                logging.error(
                    "TabbyAPI: Summarization failed; exception_type=%s",
                    safe_metadata_token(type(e).__name__),
                )
                return f"TabbyAPI: Unexpected error in summarization process: {str(e)}"

    except Exception as e:
        logging.error(
            "TabbyAPI: Unexpected failure; exception_type=%s",
            safe_metadata_token(type(e).__name__),
        )
        if streaming:
            yield f"TabbyAPI: Unexpected error in summarization process: {str(e)}"
        else:
            return f"TabbyAPI: Unexpected error in summarization process: {str(e)}"


def summarize_with_vllm(
    api_key,
    input_data,
    custom_prompt_arg,
    temp=None,
    system_message=None,
    streaming=False,
):
    try:
        # API key validation
        if not api_key or api_key.strip() == "":
            logging.info("vLLM Summarize: API key not provided as parameter")
            logging.info("vLLM Summarize: Attempting to use API key from config file")
            loaded_config_data = load_settings()
            api_key = loaded_config_data.get("vllm_api", {}).get("api_key", "")
            logging.debug("vLLM Summarize: Credential config lookup completed")

        if not api_key or api_key.strip() == "":
            logging.error("vLLM Summarize: API key not found or is empty")
            logging.debug(
                "vLLM Summarize: API Key Not Provided/Found in Config file or is empty"
            )

        logging.debug("vLLM Summarize: Credential state resolved")

        # Input data handling
        logging.debug(f"vLLM Summarize: Raw input data type: {type(input_data)}")
        logging.debug("vLLM Summarize: Raw input received")

        if isinstance(input_data, str):
            if input_data.strip().startswith("{"):
                # It's likely a JSON string
                logging.debug(
                    "vLLM Summarize: Parsing provided JSON string data for summarization"
                )
                try:
                    data = json.loads(input_data)
                except json.JSONDecodeError as e:
                    logging.error(
                        "vLLM Summarize: JSON input parsing failed; exception_type=%s",
                        safe_metadata_token(type(e).__name__),
                    )
                    return f"vLLM Summarize: Error parsing JSON input: {str(e)}"
            else:
                logging.debug(
                    "vLLM Summarize: Using provided string data for summarization"
                )
                data = input_data
        else:
            data = input_data

        logging.debug(f"vLLM Summarize: Processed data type: {type(data)}")
        logging.debug("vLLM Summarize: Input processing completed")

        # Text extraction
        if isinstance(data, dict):
            if "summary" in data:
                logging.debug(
                    "vLLM Summarize: Summary already exists in the loaded data"
                )
                return data["summary"]
            elif "segments" in data:
                text = extract_text_from_segments(data["segments"])
            else:
                text = json.dumps(data)  # Convert dict to string if no specific format
        elif isinstance(data, list):
            text = extract_text_from_segments(data)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError(f"vLLM Summarize: Invalid input data format: {type(data)}")

        logging.debug("vLLM Summarize: Text extraction completed")
        logging.debug("vLLM Summarize: Custom prompt received")

        config_settings = load_settings()
        vllm_model = config_settings["vllm_api"]["model"]
        logging.debug(f"vLLM Summarize: Using model: {vllm_model}")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        logging.debug("vLLM Summarize: Authorization header prepared")
        logging.debug("vLLM Summarize: Preparing data + prompt for submittal")
        user_prompt = f"{text} \n\n\n\n{custom_prompt_arg}"
        if temp is None:
            temp = load_settings()["vllm_api"]["temperature"]
        if system_message is None:
            system_message = (
                "You are a helpful AI assistant who does whatever the user requests."
            )
        temp = float(temp)

        # Set max tokens
        max_tokens = load_settings()["vllm_api"]["max_tokens"]
        max_tokens = int(max_tokens)
        logging.debug(f"vLLM Summarize: Using max tokens: {max_tokens}")

        # Prepare data payload
        data = {
            "model": vllm_model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temp,
            "stream": streaming,
        }

        # Setup URL
        url = load_settings()["vllm_api"]["api_ip"]

        # Handle streaming
        if streaming:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data["vllm_api"]["api_retries"]
            retry_delay = loaded_config_data["vllm_api"]["api_retry_delay"]

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            response = session.post(url, headers=headers, json=data, stream=True)
            response.raise_for_status()

            def stream_generator():
                collected_messages = ""
                for line in response.iter_lines():
                    line = line.decode("utf-8").strip()

                    if line == "":
                        continue

                    if line.startswith("data: "):
                        data_str = line[len("data: ") :]
                        if data_str == "[DONE]":
                            break
                        try:
                            data_json = json.loads(data_str)
                            chunk = data_json["choices"][0]["delta"].get("content", "")
                            collected_messages += chunk
                            yield chunk
                        except json.JSONDecodeError:
                            logging.error(
                                "vLLM Summarize: Failed to decode streamed JSON; "
                                "line_length=%s",
                                len(line),
                            )
                            continue

            return stream_generator()
        # Handle non-streaming
        else:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data["vllm_api"]["api_retries"]
            retry_delay = loaded_config_data["vllm_api"]["api_retry_delay"]

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            logging.debug("vLLM Summarization: Posting request")
            response = session.post(url, headers=headers, json=data)

            if response.status_code == 200:
                response_data = response.json()
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    summary = response_data["choices"][0]["message"]["content"].strip()
                    logging.debug("vLLM Summarization: Summarization successful")
                    logging.debug(
                        "vLLM Summarization: Summary produced; character_count=%s",
                        len(summary),
                    )
                    return summary
                else:
                    logging.warning(
                        "vLLM Summarization: Summary not found in the response data"
                    )
                    return "vLLM Summarization: Summary not available"
            else:
                logging.error(
                    f"vLLM Summarization: Summarization failed with status code {response.status_code}"
                )
                logging.error("vLLM Summarization: Error response received")
                return f"vLLM Summarization: Failed to process summary. Status code: {response.status_code}"
    except json.JSONDecodeError as e:
        logging.error(
            "vLLM Summarization: JSON decoding failed; exception_type=%s",
            safe_metadata_token(type(e).__name__),
        )
        return f"vLLM Summarization: Error decoding JSON input: {str(e)}"
    except requests.RequestException as e:
        logging.error(
            "vLLM Summarization: API request failed; exception_type=%s",
            safe_metadata_token(type(e).__name__),
        )
        return f"vLLM Summarization: Error making API request: {str(e)}"
    except Exception as e:
        logging.error(
            "vLLM Summarization: Unexpected failure; exception_type=%s",
            safe_metadata_token(type(e).__name__),
        )
        return f"vLLM Summarization: Unexpected error occurred: {str(e)}"


def summarize_with_ollama(
    input_data,
    custom_prompt,
    api_url=None,
    api_key=None,
    temp=None,
    system_message=None,
    model=None,
    max_retries=5,
    retry_delay=20,
    streaming=False,
    top_p=None,
):
    """
    Summarizes text via the Ollama API, returning:
      - a generator (if streaming=True)
      - a single string (if streaming=False)
    Follows the same style as chat_with_llama / chat_with_ollama.
    """
    # 1) Load config
    try:
        logging.debug("Ollama: Loading and validating configurations")
        loaded_config_data = load_settings()
        if not loaded_config_data:
            logging.error("summarize_with_ollama: Could not load config data.")
            return "Ollama: No config data found."

        ollama_config = loaded_config_data.get("ollama_api", {})
    except Exception as e:
        logging.error(
            "summarize_with_ollama: Config loading failed; exception_type=%s",
            safe_metadata_token(type(e).__name__),
        )
        return f"Ollama: Error loading config: {str(e)}"

    # 2) Determine API Key
    try:
        if not api_key or not api_key.strip():
            # Use config if parameter not given
            api_key = ollama_config.get("api_key", "")
            if not api_key:
                logging.warning("Ollama: No API key found in config or param.")
        else:
            logging.info("Ollama: Using API key provided as parameter.")

        # 3) Determine API URL
        if not api_url or not api_url.strip():
            api_url = ollama_config.get("api_url", "")
        if not api_url:
            logging.error("Ollama: API URL not found or is empty.")
            return "Ollama: API URL not found in config or parameter."

        # 4) Determine Model
        if not model:
            model = ollama_config.get("model", None)
        if not model:
            logging.error("Ollama: Model not provided or found in config.")
            return "Ollama: Model not found in config or parameter."

        # 5) Determine Temperature
        if temp is None:
            temp = ollama_config.get("temperature", 0.7)
        if isinstance(temp, str):
            temp = float(temp)
        temp = float(temp)
        logging.info(f"Ollama: Temp is {temp}")

        # 6) Determine top_p
        if top_p is None:
            top_p = ollama_config.get("top_p", 0.95)
        if isinstance(top_p, str):
            top_p = float(top_p)
        top_p = float(top_p)

        # 7) Determine streaming
        if isinstance(streaming, str):
            streaming = streaming.lower() == "true"
        elif isinstance(streaming, int):
            streaming = bool(streaming)
        elif streaming is None:
            streaming = ollama_config.get("streaming", False)
        if not isinstance(streaming, bool):
            raise ValueError(
                f"Invalid type for 'streaming': expected bool, got {type(streaming).__name__}"
            )
        logging.debug(f"Ollama: streaming = {streaming}")

        # 8) Determine system_message
        if not system_message:
            system_message = "You are a helpful AI assistant."

        # 9) Parse/prepare input_data
        logging.debug("Ollama: Attempting to parse the input data for summarization.")
        text_content = None
        # Input data is not a file path, treat as raw text or list
        if isinstance(input_data, dict) and "summary" in input_data:
            logging.debug("Ollama: 'summary' already present in input_data dict.")
            return input_data["summary"]
        elif isinstance(input_data, list):
            text_content = extract_text_from_segments(input_data)
        elif isinstance(input_data, str):
            text_content = input_data
        else:
            raise ValueError(
                "Ollama: Invalid input_data format; must be str or list/dict with summary."
            )

        if not text_content:
            logging.error("Ollama: No valid text content found to summarize.")
            return "Ollama: Could not extract text for summarization."

        # 10) Build final prompt
        ollama_prompt = f"{custom_prompt}\n\n{text_content}"
        logging.debug(
            "Ollama: Summarization prompt prepared; character_count=%s",
            len(ollama_prompt),
        )

        # 11) Prepare request
        max_tokens = int(ollama_config.get("max_tokens", 500))
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
        }
        # Optionally set Authorization
        if api_key and len(api_key) > 3:
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            logging.debug("Ollama: No valid API key to set in headers.")

        data_payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": ollama_prompt},
            ],
            "temperature": temp,
            "stream": streaming,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        local_api_timeout = int(loaded_config_data["ollama_api"]["api_timeout"])
        logging.info(f"Ollama: Using local API timeout: {local_api_timeout} seconds")

        # 12) Attempt request with retries

        try:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data["ollama_api"]["api_retries"]
            retry_delay = loaded_config_data["ollama_api"]["api_retry_delay"]

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            logging.debug("Ollama Summarize request being sent")
            response = session.post(
                api_url,
                headers=headers,
                json=data_payload,
                stream=streaming,
                timeout=local_api_timeout,
            )
            response.raise_for_status()  # Raise HTTPError if not 2xx
        except requests.exceptions.Timeout:
            logging.error("Ollama: Request timed out.")
            return "Ollama: Request timed out."
        except requests.exceptions.HTTPError as http_err:
            logging.error(
                "Ollama: HTTP request failed; exception_type=%s",
                safe_metadata_token(type(http_err).__name__),
            )
            return f"Ollama: HTTP error: {http_err}"
        except requests.exceptions.RequestException as req_err:
            logging.error(
                "Ollama: Request failed; exception_type=%s",
                safe_metadata_token(type(req_err).__name__),
            )
            return f"Ollama: Request exception: {req_err}"
        except Exception as e:
            logging.error(
                "Ollama: Request setup failed; exception_type=%s",
                safe_metadata_token(type(e).__name__),
            )
            return f"Ollama: Unexpected error: {str(e)}"

        # 13) Handle streaming or non-streaming
        if streaming:
            # Return a generator that yields partial text chunks
            logging.debug("Ollama: Processing streaming response.")

            def stream_generator():
                for line in response.iter_lines():
                    if not line:
                        continue
                    decoded_line = line.decode("utf-8").strip()
                    try:
                        # Some Ollama builds prefix with "data: ...".
                        # If that's the case, parse accordingly:
                        if decoded_line.startswith("data:"):
                            decoded_line = decoded_line[len("data:") :].strip()
                            if decoded_line == "[DONE]":
                                break
                        json_chunk = json.loads(decoded_line)
                        if "response" in json_chunk:
                            yield json_chunk["response"]
                        if json_chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        logging.error(
                            "Ollama: Failed to decode streamed JSON; line_length=%s",
                            len(decoded_line),
                        )
                        continue

            return stream_generator()
        else:
            # Non-streaming => parse entire JSON once and return the text
            try:
                # Create a session
                session = requests.Session()

                # Load config values
                retry_count = loaded_config_data["ollama_api"]["api_retries"]
                retry_delay = loaded_config_data["ollama_api"]["api_retry_delay"]

                # Configure the retry strategy
                retry_strategy = Retry(
                    total=retry_count,  # Total number of retries
                    backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                    status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
                )

                # Create the adapter
                adapter = HTTPAdapter(max_retries=retry_strategy)

                # Mount adapters for both HTTP and HTTPS
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                response_data = response.json()  # corrected object to get an json method(not avaliable in the session object)
            except json.JSONDecodeError:
                logging.error("Ollama: Failed to parse JSON response.")
                return "Ollama: JSON parse error from summarization API."

            logging.debug("Ollama: Response parsed")
            # Attempt to retrieve final summary
            summary = None
            if "response" in response_data and response_data["response"]:
                summary = response_data["response"].strip()
            elif "choices" in response_data and len(response_data["choices"]) > 0:
                choice = response_data["choices"][0]
                content = choice.get("message", {}).get("content", "").strip()
                if content:
                    summary = content

            if summary:
                logging.debug("Ollama: Summarization request successful.")
                return summary
            else:
                logging.error("Ollama: 'response' or 'choices' not found in JSON.")
                return "Ollama: Summarization API response missing text."

    except Exception as e:
        logging.error(
            "Ollama Summarize: Summarization failed; exception_type=%s",
            safe_metadata_token(type(e).__name__),
        )
        return f"Ollama: Error occurred while summarizing: {str(e)}"


def summarize_with_custom_openai(
    api_key,
    input_data,
    custom_prompt_arg,
    temp=None,
    system_message=None,
    streaming=False,
):
    loaded_config_data = load_settings()
    custom_openai_api_key = api_key
    try:
        # API key validation
        if not custom_openai_api_key:
            logging.info("Custom OpenAI API: API key not provided as parameter")
            logging.info(
                "Custom OpenAI API: Attempting to use API key from config file"
            )
            custom_openai_api_key = loaded_config_data["custom_openai_api"]["api_key"]

        if not custom_openai_api_key:
            logging.error("Custom OpenAI API: API key not found or is empty")
            return "Custom OpenAI API: API Key Not Provided/Found in Config file or is empty"

        logging.debug("Custom OpenAI API: Credential configured")

        # Input data handling
        logging.debug(f"Custom OpenAI API: Raw input data type: {type(input_data)}")
        logging.debug("Custom OpenAI API: Input received")

        if isinstance(input_data, str):
            if input_data.strip().startswith("{"):
                # It's likely a JSON string
                logging.debug(
                    "Custom OpenAI API: Parsing provided JSON string data for summarization"
                )
                try:
                    data = json.loads(input_data)
                except json.JSONDecodeError as e:
                    logging.error(
                        "Custom OpenAI API: Input JSON parse failed; exception_type=%s",
                        safe_metadata_token(type(e).__name__),
                    )
                    data = input_data
                    pass
            else:
                logging.debug(
                    "Custom OpenAI API: Using provided string data for summarization"
                )
                data = input_data
        else:
            data = input_data

        logging.debug(f"Custom OpenAI API: Processed data type: {type(data)}")
        logging.debug("Custom OpenAI API: Input processing completed")

        # Text extraction
        if isinstance(data, dict):
            if "summary" in data:
                logging.debug(
                    "Custom OpenAI API: Summary already exists in the loaded data"
                )
                return data["summary"]
            elif "segments" in data:
                text = extract_text_from_segments(data["segments"])
            else:
                text = json.dumps(data)  # Convert dict to string if no specific format
        elif isinstance(data, list):
            text = extract_text_from_segments(data)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError(
                f"Custom OpenAI API: Invalid input data format: {type(data)}"
            )

        logging.debug("Custom OpenAI API: Text extraction completed")
        logging.debug(
            "Custom OpenAI API: Prompt prepared; character_count=%s",
            len(f"{custom_prompt_arg}"),
        )

        if input_data is None:
            input_data = f"{get_internal_prompt('summarization.local_summarizer_template')}\n\n\n\n{text}"
        else:
            input_data = f"{input_data}\n\n\n\n{text}"

        # Model Selection
        custom_openai_model = loaded_config_data["custom_openai_api"]["model"]
        logging.debug(f"Custom OpenAI API: Using model: {custom_openai_model}")

        # Set max tokens
        max_tokens = loaded_config_data["custom_openai_api"]["max_tokens"]
        max_tokens = int(max_tokens)
        logging.debug(f"Custom OpenAI API: Using max tokens: {max_tokens}")

        # Set temperature
        if temp is None:
            temp = load_settings()["custom_openai_api"]["temperature"]
        temp = float(temp)

        # Set system message
        if system_message is None:
            system_message = (
                "You are a helpful AI assistant who does whatever the user requests."
            )

        # Set Streaming
        if streaming is None:
            streaming = load_settings()["custom_openai_api"]["streaming"]

        # Set API URL
        custom_openai_api_url = loaded_config_data["custom_openai_api"]["api_ip"]
        logging.debug("Custom OpenAI API: API endpoint configured")

        logging.debug("Custom OpenAI API: Preparing data + prompt for submittal")
        openai_prompt = f"{text} \n\n\n\n{custom_prompt_arg}"

        # Set headers
        headers = {
            "Authorization": f"Bearer {custom_openai_api_key}",
            "Content-Type": "application/json",
        }

        # Payload setup
        data = {
            "model": custom_openai_model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": openai_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temp,
            "stream": streaming,
        }

        if streaming:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data["custom_openai_api"]["api_retries"]
            retry_delay = loaded_config_data["custom_openai_api"]["api_retry_delay"]

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            response = session.post(
                custom_openai_api_url, headers=headers, json=data, stream=True
            )
            response.raise_for_status()

            def stream_generator():
                collected_messages = ""
                for line in response.iter_lines():
                    line = line.decode("utf-8").strip()

                    if line == "":
                        continue

                    if line.startswith("data: "):
                        data_str = line[len("data: ") :]
                        if data_str == "[DONE]":
                            break
                        try:
                            data_json = json.loads(data_str)
                            chunk = data_json["choices"][0]["delta"].get("content", "")
                            collected_messages += chunk
                            yield chunk
                        except json.JSONDecodeError:
                            logging.error(
                                "Custom OpenAI API: Failed to decode streamed JSON; "
                                "line_length=%s",
                                len(data_str),
                            )
                            continue
                yield collected_messages

            return stream_generator()
        else:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data["custom_openai_api"]["api_retries"]
            retry_delay = loaded_config_data["custom_openai_api"]["api_retry_delay"]

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            logging.debug("Custom OpenAI API: Posting request")
            response = session.post(custom_openai_api_url, headers=headers, json=data)
            logging.debug(
                "Custom OpenAI API: Response received; status_code=%s",
                response.status_code,
            )
            if response.status_code == 200:
                response_data = response.json()
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    chat_response = response_data["choices"][0]["message"][
                        "content"
                    ].strip()
                    logging.debug("Custom OpenAI API: Chat Sent successfully")
                    logging.debug(
                        "Custom OpenAI API: Chat response received; character_count=%s",
                        len(chat_response),
                    )
                    return chat_response
                else:
                    logging.warning(
                        "Custom OpenAI API: Chat response not found in the response data"
                    )
                    return "Custom OpenAI API: Chat not available"
            else:
                logging.error(
                    f"Custom OpenAI API: Chat request failed with status code {response.status_code}"
                )
                return f"OpenAI: Failed to process chat response. Status code: {response.status_code}"
    except json.JSONDecodeError as e:
        logging.error(
            "Custom OpenAI API: Response JSON decode failed; exception_type=%s",
            safe_metadata_token(type(e).__name__),
        )
        return f"Custom OpenAI API: Error decoding JSON input: {str(e)}"
    except requests.RequestException as e:
        logging.error(
            "Custom OpenAI API: API request failed; exception_type=%s",
            safe_metadata_token(type(e).__name__),
        )
        return f"Custom OpenAI API: Error making API request: {str(e)}"
    except Exception as e:
        logging.error(
            "Custom OpenAI API: Unexpected failure; exception_type=%s",
            safe_metadata_token(type(e).__name__),
        )
        return f"Custom OpenAI API: Unexpected error occurred: {str(e)}"


def summarize_with_custom_openai_2(
    api_key,
    input_data,
    custom_prompt_arg,
    temp=None,
    system_message=None,
    streaming=False,
):
    loaded_config_data = load_settings()
    custom_openai_api_key = api_key
    try:
        # API key validation
        if not custom_openai_api_key:
            logging.info("Custom OpenAI API-2: API key not provided as parameter")
            logging.info(
                "Custom OpenAI API-2: Attempting to use API key from config file"
            )
            custom_openai_api_key = loaded_config_data["custom_openai_api_2"]["api_key"]

        if not custom_openai_api_key:
            logging.error("Custom OpenAI API-2: API key not found or is empty")
            return "Custom OpenAI API-2: API Key Not Provided/Found in Config file or is empty"

        logging.debug("Custom OpenAI API-2: Credential configured")

        # Input data handling
        logging.debug(f"Custom OpenAI API-2: Raw input data type: {type(input_data)}")
        logging.debug("Custom OpenAI API-2: Input received")

        if isinstance(input_data, str):
            if input_data.strip().startswith("{"):
                # It's likely a JSON string
                logging.debug(
                    "Custom OpenAI API-2: Parsing provided JSON string data for summarization"
                )
                try:
                    data = json.loads(input_data)
                except json.JSONDecodeError as e:
                    logging.error(
                        "Custom OpenAI API-2: Input JSON parse failed; "
                        "exception_type=%s",
                        safe_metadata_token(type(e).__name__),
                    )
                    data = input_data
                    pass
            else:
                logging.debug(
                    "Custom OpenAI API-2: Using provided string data for summarization"
                )
                data = input_data
        else:
            data = input_data

        logging.debug(f"Custom OpenAI API-2: Processed data type: {type(data)}")
        logging.debug("Custom OpenAI API-2: Input processing completed")

        # Text extraction
        if isinstance(data, dict):
            if "summary" in data:
                logging.debug(
                    "Custom OpenAI API-2: Summary already exists in the loaded data"
                )
                return data["summary"]
            elif "segments" in data:
                text = extract_text_from_segments(data["segments"])
            else:
                text = json.dumps(data)  # Convert dict to string if no specific format
        elif isinstance(data, list):
            text = extract_text_from_segments(data)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError(
                f"Custom OpenAI API-2: Invalid input data format: {type(data)}"
            )

        logging.debug("Custom OpenAI API-2: Text extraction completed")
        logging.debug(
            "Custom OpenAI API-2: Prompt prepared; character_count=%s",
            len(f"{custom_prompt_arg}"),
        )

        if input_data is None:
            input_data = f"{get_internal_prompt('summarization.local_summarizer_template')}\n\n\n\n{text}"
        else:
            input_data = f"{input_data}\n\n\n\n{text}"

        # Model Selection
        custom_openai_model = loaded_config_data["custom_openai_api_2"]["model"]
        logging.debug(f"Custom OpenAI API-2: Using model: {custom_openai_model}")

        # Set max tokens
        max_tokens = loaded_config_data["custom_openai_api_2"]["max_tokens"]
        max_tokens = int(max_tokens)
        logging.debug(f"Custom OpenAI API: Using max tokens: {max_tokens}")

        # Set temperature
        if temp is None:
            temp = load_settings()["custom_openai_api_2"]["temperature"]
        temp = float(temp)

        # Set system message
        if system_message is None:
            system_message = (
                "You are a helpful AI assistant who does whatever the user requests."
            )

        # Set Streaming
        if streaming is None:
            streaming = load_settings()["custom_openai_api_2"]["streaming"]

        # Set API URL
        custom_openai_api_url = loaded_config_data["custom_openai_api_2"]["api_ip"]
        logging.debug("Custom OpenAI API-2: API endpoint configured")

        logging.debug("Custom OpenAI API-2: Preparing data + prompt for submittal")
        openai_prompt = f"{text} \n\n\n\n{custom_prompt_arg}"

        # Set headers
        headers = {
            "Authorization": f"Bearer {custom_openai_api_key}",
            "Content-Type": "application/json",
        }

        # Payload setup
        data = {
            "model": custom_openai_model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": openai_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temp,
            "stream": streaming,
        }

        if streaming:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data["custom_openai_api_2"]["api_retries"]
            retry_delay = loaded_config_data["custom_openai_api_2"]["api_retry_delay"]

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            response = session.post(
                custom_openai_api_url, headers=headers, json=data, stream=True
            )
            response.raise_for_status()

            def stream_generator():
                collected_messages = ""
                for line in response.iter_lines():
                    line = line.decode("utf-8").strip()

                    if line == "":
                        continue

                    if line.startswith("data: "):
                        data_str = line[len("data: ") :]
                        if data_str == "[DONE]":
                            break
                        try:
                            data_json = json.loads(data_str)
                            chunk = data_json["choices"][0]["delta"].get("content", "")
                            collected_messages += chunk
                            yield chunk
                        except json.JSONDecodeError:
                            logging.error(
                                "Custom OpenAI API-2: Failed to decode streamed JSON; "
                                "line_length=%s",
                                len(data_str),
                            )
                            continue
                yield collected_messages

            return stream_generator()
        else:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data["custom_openai_api_2"]["api_retries"]
            retry_delay = loaded_config_data["custom_openai_api_2"]["api_retry_delay"]

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            logging.debug("Custom OpenAI API-2: Posting request")
            response = session.post(custom_openai_api_url, headers=headers, json=data)
            logging.debug(
                "Custom OpenAI API-2: Response received; status_code=%s",
                response.status_code,
            )
            if response.status_code == 200:
                response_data = response.json()
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    chat_response = response_data["choices"][0]["message"][
                        "content"
                    ].strip()
                    logging.debug("Custom OpenAI API-2: Chat Sent successfully")
                    logging.debug(
                        "Custom OpenAI API-2: Chat response received; "
                        "character_count=%s",
                        len(chat_response),
                    )
                    return chat_response
                else:
                    logging.warning(
                        "Custom OpenAI API-2: Chat response not found in the response data"
                    )
                    return "Custom OpenAI API-2: Chat not available"
            else:
                logging.error(
                    f"Custom OpenAI API-2: Chat request failed with status code {response.status_code}"
                )
                return f"OpenAI: Failed to process chat response. Status code: {response.status_code}"
    except json.JSONDecodeError as e:
        logging.error(
            "Custom OpenAI API-2: Response JSON decode failed; exception_type=%s",
            safe_metadata_token(type(e).__name__),
        )
        return f"Custom OpenAI API-2: Error decoding JSON input: {str(e)}"
    except requests.RequestException as e:
        logging.error(
            "Custom OpenAI API-2: API request failed; exception_type=%s",
            safe_metadata_token(type(e).__name__),
        )
        return f"Custom OpenAI API-2: Error making API request: {str(e)}"
    except Exception as e:
        logging.error(
            "Custom OpenAI API-2: Unexpected failure; exception_type=%s",
            safe_metadata_token(type(e).__name__),
        )
        return f"Custom OpenAI API-2: Unexpected error occurred: {str(e)}"


def save_summary_to_file(summary, file_path):
    logging.debug("Now saving summary to file...")
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    summary_file_path = os.path.join(
        os.path.dirname(file_path), base_name + "_summary.txt"
    )
    os.makedirs(os.path.dirname(summary_file_path), exist_ok=True)
    logging.debug("Opening summary file for writing, *segments.json with *_summary.txt")
    with open(summary_file_path, "w") as file:
        file.write(summary)
    logging.info("Summary saved to file")


#
#
#######################################################################################################################
