# Summarization_General_Lib.py
#########################################
# General Summarization Library
# This library is used to perform summarization.
#
####
####################
# Function List
#
# 1. extract_text_from_segments(segments: List[Dict]) -> str
# 2. analyze(api_name, input_data, custom_prompt_arg, ...) -- the public
#    entry; dispatches every provider through _summarize_via_chat_api
#    (per-provider spec table, TASK-34895) over the unified chat dispatch
# 3. summarize_with_mock_llm(...) -- the test LLM
#
#
####################
# Import necessary libraries
from tldw_chatbook.LLM_Calls import recovery_review as _provider_recovery

import inspect
import json
import os
import time
from dataclasses import dataclass
from typing import Optional, Union, Generator, Any, Dict, List, Callable, Iterable

#
# 3rd-Party Imports
from loguru import logger

#
# Import Local
from tldw_chatbook.Logging_Config import logging
from tldw_chatbook.config import (
    ProviderSettingsError,
    get_cli_setting,
    load_settings,
    provider_settings_for_key,
    without_ciphertext,
)
from tldw_chatbook.Internal_Prompts import get_internal_prompt
from tldw_chatbook.Utils.persistent_diagnostics import safe_metadata_token

try:
    from tldw_chatbook.Chunking.Chunk_Lib import improved_chunking_process

    CHUNKER_AVAILABLE = True
except ImportError:
    logger.warning("Failed to import chunking library. Will not be available.")
    CHUNKER_AVAILABLE = False


# FIXME
def load_and_log_configs():
    pass


#
#######################################################################################################################
# Function Definitions
# (TASK-32854: the module-level `api_key = get_cli_setting(...)` read here
# was dead -- no importer of that name exists -- and its import-time config
# read (a secret into a module global, plus one admission handshake per
# import) broke the local summarizers' lazy import of this module's shared
# transport. Deleted per the 2026-09-17 review's finding.)
#
#######################################################################################################################
# Helper Function Definitions
#


def log_debug_data(data: Any, source_name: str) -> None:
    """
    Helper function to log debug information about loaded data.

    Args:
        data: The data to log information about
        source_name: The name of the API/source (e.g., "AnthropicAI", "Cohere", etc.)
    """
    logging.debug(
        "Summarization input loaded; source=%s",
        safe_metadata_token(source_name),
    )
    logging.debug(f"{source_name}: Type of data: {type(data)}")


# --- Keep existing helper functions ---
def extract_text_from_segments(segments: List[Dict]) -> str:
    # (Keep existing implementation)
    logging.debug("Segments received")
    logging.debug(f"Type of segments: {type(segments)}")
    text = ""
    if isinstance(segments, list):
        for segment in segments:
            # logging.debug(f"Current segment: {segment}") # Can be verbose
            # logging.debug(f"Type of segment: {type(segment)}")
            if isinstance(segment, dict) and "Text" in segment:
                text += segment["Text"] + " "
            elif (
                isinstance(segment, dict) and "text" in segment
            ):  # Adding flexibility for key case
                text += segment["text"] + " "
            else:
                logging.warning(
                    "Skipping segment due to missing text key or wrong type"
                )
    elif isinstance(segments, str):  # Allow passing a pre-joined string
        logging.debug("Segments received as a single string.")
        text = segments
    else:
        logging.warning(
            f"Unexpected type of 'segments': {type(segments)}. Trying to convert to string."
        )
        text = str(segments)  # Attempt conversion

    return text.strip()


def recursive_summarize_chunks(
    chunks: List[str],
    summarize_func: Callable[[str], str],  # Function now only needs to accept the text
) -> str:
    """
    Recursively processes chunks by combining the result of the previous step
    with the next chunk and applying the summarize_func.

    This is suitable for tasks like recursive summarization where context
    from the previous summary is needed for the next chunk.

    Args:
        chunks: A list of text chunks to process.
        summarize_func: A function that takes a single string argument (the text
                        to process) and returns a single string result (the summary
                        or analysis). This function should handle its own configuration
                        (like API keys, prompts, temperature) internally or via closure.
                        It should also handle potential errors and return an error string
                        (e.g., starting with "Error:") if processing fails.

    Returns:
        A single string representing the final result after processing all chunks,
        or an error string if any step failed. Returns an empty string if
        the input chunks list is empty.
    """
    if not chunks:
        logging.warning("recursive_summarize_chunks called with empty chunk list.")
        return ""

    logging.info(f"Starting recursive processing of {len(chunks)} chunks.")
    current_summary = ""

    for i, chunk in enumerate(chunks):
        logging.debug(f"Processing chunk {i + 1}/{len(chunks)} recursively.")
        text_to_process: str

        if i == 0:
            # Process the first chunk directly
            text_to_process = chunk
            logging.debug(f"Processing first chunk (length {len(text_to_process)}).")
        else:
            # Combine the previous summary with the current chunk
            # Add a separator for clarity for the LLM
            combined_text = f"{current_summary}\n\n---\n\n{chunk}"
            text_to_process = combined_text
            logging.debug(
                f"Processing combination of previous summary and chunk {i + 1} (total length {len(text_to_process)})."
            )

        # Apply the processing function
        try:
            step_result = summarize_func(text_to_process)

            # Check if the processing function indicated an error
            if isinstance(step_result, str) and step_result.startswith("Error:"):
                logging.error("Error during recursive step; step=%s", i + 1)
                return step_result  # Propagate the error immediately

            if not isinstance(step_result, str):
                # This shouldn't happen if summarize_func adheres to the contract, but good to check
                logging.error(
                    f"Recursive step {i + 1} did not return a string. Got: {type(step_result)}"
                )
                return f"Error: Processing step {i + 1} returned unexpected type {type(step_result)}"

            current_summary = step_result
            logging.debug(
                f"Chunk {i + 1} processed. Current summary length: {len(current_summary)}"
            )

        except Exception as e:
            logging.error(
                "Unexpected error calling summarize_func; step=%s exception_type=%s",
                i + 1,
                safe_metadata_token(type(e).__name__),
            )
            return f"Error: Unexpected failure during recursive step {i + 1}: {e}"

    logging.info("Recursive processing completed successfully.")
    return current_summary


def extract_text_from_input(input_data: Any) -> str:
    """Extracts usable text content from various input types."""
    logging.debug(f"Extracting text from input of type: {type(input_data)}")
    if isinstance(input_data, str):
        # Check if it's a file path
        if os.path.isfile(input_data):
            logging.debug("Input resolved as file path")
            try:
                with open(input_data, "r", encoding="utf-8") as f:
                    content = f.read()
                # Attempt to parse as JSON, otherwise return raw content
                try:
                    data = json.loads(content)
                    logging.debug("File content parsed as JSON.")
                    return extract_text_from_input(data)  # Recurse with parsed data
                except json.JSONDecodeError:
                    logging.debug("File content is not JSON, returning raw text.")
                    return content.strip()
            except Exception as e:
                logging.error(
                    "Error reading input file; exception_type=%s",
                    safe_metadata_token(type(e).__name__),
                )
                return ""
        # Check if it's a JSON string
        elif input_data.strip().startswith("{") or input_data.strip().startswith("["):
            logging.debug("Input is potentially a JSON string.")
            try:
                data = json.loads(input_data)
                logging.debug("Input string parsed as JSON.")
                return extract_text_from_input(data)  # Recurse with parsed data
            except json.JSONDecodeError:
                logging.debug("Input string is not JSON, treating as plain text.")
                return input_data.strip()
        # Otherwise, treat as plain text
        else:
            logging.debug("Input is a plain text string.")
            return input_data.strip()

    elif isinstance(input_data, dict):
        logging.debug("Input is a dictionary.")
        # Prioritize known structures
        if "transcription" in input_data:
            logging.debug("Extracting text from 'transcription' field.")
            return extract_text_from_segments(input_data["transcription"])
        elif "segments" in input_data:
            logging.debug("Extracting text from 'segments' field.")
            return extract_text_from_segments(input_data["segments"])
        elif "text" in input_data:
            logging.debug("Extracting text from 'text' field.")
            return str(input_data["text"]).strip()
        elif "content" in input_data:
            logging.debug("Extracting text from 'content' field.")
            return str(input_data["content"]).strip()
        else:
            # Fallback: try to convert the whole dict to string (might be noisy)
            logging.warning(
                "No specific text field found in dict, converting entire dict to string."
            )
            try:
                return json.dumps(input_data, indent=2)
            except Exception:
                return str(input_data)  # Final fallback

    elif isinstance(input_data, list):
        logging.debug("Input is a list, assuming list of segments.")
        # Assume it's a list of segments like {'Text': '...'} or {'text': '...'}
        return extract_text_from_segments(input_data)

    else:
        logging.warning(
            f"Unhandled input type: {type(input_data)}. Attempting string conversion."
        )
        return str(input_data).strip()


# --- Internal API Dispatcher ---
# (task-3301 xhigh review round) Spellings used by `Chat/Chat_Functions.py`'s
# `API_CALL_HANDLERS` dispatch table mapped onto this module's own provider
# names. The Library ingest seam normalizes `[analysis_defaults] provider`
# to the chat dispatcher's names; the pdf/ebook/audio processors then hand
# that same name to `analyze()`, which previously rejected e.g.
# "koboldcpp"/"oobabooga"/"mistralai" with "Error: Invalid API Name" even
# though an implementation exists here under another spelling.
_CHAT_DISPATCH_NAME_ALIASES = {
    "koboldcpp": "kobold",
    "oobabooga": "ooba",
    "mistralai": "mistral",
    "llama_cpp": "llama.cpp",
    "local_llamacpp": "llama.cpp",
    "local_llamafile": "llama.cpp",
    "local_ollama": "ollama",
    "local_vllm": "vllm",
    "local_llm": "local-llm",
}
# aphrodite, mlx_lm/local_mlx_lm, moonshot and zai are covered by the
# TASK-34895 engine-preset specs below (they had no legacy per-provider
# function; the chat handler resolves their modern api_settings keys).
# A summary decides its key under the provider the caller named
# (TASK-33001.14): Settings keeps MistralAI in [api_settings.mistralai],
# not the "mistral" spec's table. The local_* aliases keep their target's
# table, because the summary is sent to the target handler's URL.
_ALIASES_WITH_OWN_CREDENTIAL = frozenset({"mistralai"})


def _dispatch_to_api(
    text_to_summarize: str,
    custom_prompt_arg: Optional[str],
    api_name: str,
    api_key: Optional[str],
    temp: Optional[float],
    system_message: Optional[str],
    streaming: bool,
) -> Union[str, Generator[str, None, None], None]:
    """
    Internal dispatch: map api_name to the provider's chat-path handler.
    """
    # TASK-34100.4: still-encrypted `enc:` ciphertext is never a credential;
    # each summarizer then reports its key as missing (mirrors chat_api_call).
    api_key = without_ciphertext(api_key)
    try:
        requested = api_name.lower()
        api_name_lower = _CHAT_DISPATCH_NAME_ALIASES.get(requested, requested)
        logging.debug(f"Dispatching to API: {api_name_lower}")

        # TASK-33641.10: every argument by keyword, so none can land on a
        # parameter of another name. Names resolve at call time (the analyze
        # and privacy suites monkeypatch them).
        if api_name_lower == "mock-llm":
            return summarize_with_mock_llm(
                text_to_summarize=text_to_summarize,
                custom_prompt_arg=custom_prompt_arg,
                api_key=api_key,
                temp=temp,
                system_message=system_message,
                streaming=streaming,
            )
        if api_name_lower not in _SUMMARIZER_CHAT_SPECS:
            error_msg = f"Error: Invalid API Name '{api_name}'"
            logging.error(error_msg)
            return error_msg
        # The core loads API keys, models, etc. from config itself when the
        # provided arguments are None.
        return _summarize_via_chat_api(
            api_name_lower,
            api_key=api_key,
            input_data=text_to_summarize,
            custom_prompt_arg=custom_prompt_arg,
            temp=temp,
            system_message=system_message,
            streaming=streaming,
            credential_provider=(
                requested if requested in _ALIASES_WITH_OWN_CREDENTIAL else None
            ),
        )

    except Exception as e:
        logging.error(
            "Error during dispatch to API; provider=%s exception_type=%s",
            safe_metadata_token(api_name),
            safe_metadata_token(type(e).__name__),
        )
        return f"Error calling API {api_name}: {str(e)}"


# --- Main Summarization Function ---
def analyze(
    api_name: str,
    input_data: Any,
    custom_prompt_arg: Optional[str],
    api_key: Optional[str] = None,
    system_message: Optional[str] = None,
    temp: Optional[float] = None,
    streaming: bool = False,
    recursive_summarization: bool = False,
    chunked_summarization: bool = False,  # Summarize chunks separately & combine
    chunk_options: Optional[dict] = None,
) -> Union[str, Generator[str, None, None]]:
    """
    Performs analysis(summarization by default) using a specified API, with optional chunking strategies. Provide a system prompt to avoid summarization.

    Args:
        input_data: Data to analyze(Default is summarization) (text string, file path to JSON, dict, list of dicts).
        custom_prompt_arg: Custom prompt instructions for the LLM.
        api_name: Name of the API service to use (e.g., 'openai', 'anthropic', 'ollama').
        api_key: Optional API key. If None, the specific API function will attempt to load from config.
        temp: Optional temperature setting for the LLM (default varies by API).
        system_message: Optional system message/persona for the LLM. If None, a default is used.
        streaming: If True, attempts to return a generator for streaming output.
                   NOTE: Streaming output is only supported when NO chunking strategy is used.
                   If chunking is enabled, the function will process internally and return a final string.
        recursive_summarization: If True, uses a recursive summarization strategy:
                                 Summarize chunk 1 -> Combine summary 1 + chunk 2 -> Summarize -> ...
        chunked_summarization: If True, summarizes each chunk individually and concatenates the results.
                               Mutually exclusive with recursive_summarization.
        chunk_options: Dictionary of options for the chunking process (passed to improved_chunking_process).
                       Defaults: {'method': 'words', 'max_size': 1000, 'overlap': 100}.

    Returns:
        - A string containing the final summary.
        - A generator yielding summary tokens if streaming=True AND no chunking is used.
        - An error string (starting with "Error:") if summarization fails.
    """
    # Load config here if needed for top-level decisions, otherwise let specific funcs handle it
    # loaded_config_data = load_and_log_configs() # Load once if needed globally
    logging.info(
        f"Starting summarization process. API: {api_name}, Recursive: {recursive_summarization}, Chunked: {chunked_summarization}, Streaming: {streaming}"
    )

    if recursive_summarization and chunked_summarization:
        error_msg = "Error: Cannot perform both recursive and chunked summarization simultaneously."
        logging.error(error_msg)
        return error_msg

    # Set default system message if not provided
    if system_message is None:
        logging.debug("Using default system message.")
        system_message = get_internal_prompt("summarization.analyze_default_system")

    try:
        # 1. Extract text content from input_data
        text_content = extract_text_from_input(input_data)
        if not text_content:
            logging.error("Could not extract text content from input data.")
            return "Error: Could not extract text content."
        logging.info(f"Extracted text content length: {len(text_content)} characters.")

        # --- Define helper to consume potential generators ---
        def consume_generator(gen):
            # TASK-32628's recovery admission retains every generator a
            # decorated provider call returns behind recovery_review's
            # _OpenAIStream wrapper, so the isgenerator-only check handed
            # the wrapper through unconsumed -- the generator body never
            # ran, no request was made, and kobold/tabby callers got an
            # opaque object instead of the (task-17387-pinned) empty join.
            # Unwrap and iterate the wrapper like a plain generator.
            if isinstance(gen, _provider_recovery._OpenAIStream):
                gen = gen.iterator
            if inspect.isgenerator(gen):
                logging.debug("Consuming generator stream...")
                result_list = []
                try:
                    for chunk in gen:
                        if isinstance(chunk, str):
                            result_list.append(chunk)
                        else:
                            logging.warning(
                                f"Generator yielded non-string chunk: {type(chunk)}"
                            )
                    final_string = "".join(result_list)
                    logging.debug("Generator consumed.")
                    return final_string
                except Exception as e:
                    logging.error(
                        "Error consuming generator; exception_type=%s",
                        safe_metadata_token(type(e).__name__),
                    )
                    return f"Error consuming stream: {e}"
            return gen  # Return as is if not a generator

        # --- Chunking and Summarization Logic ---
        final_result: Union[str, Generator[str, None, None], None] = None
        effective_streaming_for_api_call = False  # Default for chunking modes

        # Default chunk options
        default_chunk_opts = {"method": "sentences", "max_size": 500, "overlap": 200}
        current_chunk_options = (
            chunk_options if isinstance(chunk_options, dict) else default_chunk_opts
        )

        # (task-3301 xhigh review round, F1) The direct-dispatch path below
        # used to be the ``else`` of ``if CHUNKER_AVAILABLE:`` -- so with the
        # chunk lib importable (every normal install) and NO chunking
        # strategy requested, ``final_result`` was never assigned and this
        # function returned 'Error: Summarization failed unexpectedly.'
        # WITHOUT making any API call. Chunking strategies now gate on
        # BOTH the request flag and chunker availability; everything else
        # dispatches directly.
        if recursive_summarization and not CHUNKER_AVAILABLE:
            logging.warning(
                "Recursive summarization requested but the chunk lib is "
                "unavailable; falling back to direct summarization."
            )
        if chunked_summarization and not CHUNKER_AVAILABLE:
            logging.warning(
                "Chunked summarization requested but the chunk lib is "
                "unavailable; falling back to direct summarization."
            )

        if CHUNKER_AVAILABLE and (recursive_summarization or chunked_summarization):
            if recursive_summarization:
                logging.info("Performing recursive summarization.")
                chunks_data = improved_chunking_process(
                    text_content, current_chunk_options
                )  # Renamed variable for clarity
                if not chunks_data:
                    logging.warning(
                        "Recursive summarization: Chunking produced no chunks."
                    )
                    return (
                        "Error: Recursive summarization failed - no chunks generated."
                    )

                # Extract just the text from the chunk data
                text_chunks = [chunk["text"] for chunk in chunks_data]
                logging.debug(
                    f"Generated {len(text_chunks)} text chunks for recursive summarization."
                )

                # Define the summarizer function for recursive_summarize_chunks
                # It must accept ONE argument (the text) and return the summary string.
                # It captures necessary variables (api_name, key, temp, prompts, etc.) from the outer scope (closure).
                # It must handle potential errors from the API call and return an error string if needed.
                def recursive_step_processor(text_to_summarize: str) -> str:
                    logging.debug(
                        f"recursive_step_processor called with text length: {len(text_to_summarize)}"
                    )
                    # Force non-streaming for internal steps and consume immediately
                    api_result = _dispatch_to_api(
                        text_to_summarize,
                        custom_prompt_arg,  # Custom prompt is handled by _dispatch_to_api
                        api_name,
                        api_key,
                        temp,
                        system_message,  # System message is handled by _dispatch_to_api
                        streaming=False,  # IMPORTANT: Force non-streaming for internal recursive steps
                    )
                    # consume_generator handles both strings and generators, returning a string
                    processed_result = consume_generator(api_result)

                    # Ensure the result is a string (consume_generator should do this)
                    if not isinstance(processed_result, str):
                        logging.error(
                            f"API dispatch/consumption did not return a string. Got: {type(processed_result)}"
                        )
                        # Return an error string that recursive_summarize_chunks can detect
                        return f"Error: Internal summarization step failed to produce string output (got {type(processed_result)})"

                    logging.debug(
                        f"recursive_step_processor finished. Result length: {len(processed_result)}"
                    )
                    # Return the result string (which could be a summary or an error message from consume_generator)
                    return processed_result

                # Call the simplified recursive_summarize_chunks utility
                # It now only needs the list of text chunks and the processing function
                final_result = recursive_summarize_chunks(
                    chunks=text_chunks, summarize_func=recursive_step_processor
                )
                # The result of recursive_summarize_chunks is now the final string summary or an error string

            elif chunked_summarization:
                logging.info(
                    "Performing chunked summarization (summarize each, then combine)."
                )
                chunks = improved_chunking_process(text_content, current_chunk_options)
                if not chunks:
                    logging.warning(
                        "Chunked summarization: Chunking produced no chunks."
                    )
                    return "Error: Chunked summarization failed - no chunks generated."
                logging.debug(
                    f"Generated {len(chunks)} chunks for chunked summarization."
                )

                chunk_summaries = []
                for i, chunk in enumerate(chunks):
                    logging.debug(f"Summarizing chunk {i + 1}/{len(chunks)}")
                    # Summarize each chunk - force non-streaming for API call
                    chunk_summary_result = _dispatch_to_api(
                        chunk["text"],
                        custom_prompt_arg,
                        api_name,
                        api_key,
                        temp,
                        system_message,
                        streaming=False,  # Force non-streaming
                    )
                    # Consume generator immediately
                    processed_chunk_summary = consume_generator(chunk_summary_result)

                    if isinstance(
                        processed_chunk_summary, str
                    ) and not processed_chunk_summary.startswith("Error:"):
                        chunk_summaries.append(processed_chunk_summary)
                    else:
                        error_detail = (
                            processed_chunk_summary
                            if isinstance(processed_chunk_summary, str)
                            else "Unknown error"
                        )
                        logging.warning(
                            "Failed to summarize chunk; chunk=%s",
                            i + 1,
                        )
                        chunk_summaries.append(
                            f"[Error summarizing chunk {i + 1}: {error_detail}]"
                        )  # Add error placeholder

                # Combine the summaries
                final_result = "\n\n---\n\n".join(
                    chunk_summaries
                )  # Join with a separator

        else:
            # No chunking - direct summarization
            logging.info("Performing direct summarization (no chunking).")
            # Use the user's requested streaming setting for the API call
            effective_streaming_for_api_call = streaming
            final_result = _dispatch_to_api(
                text_content,
                custom_prompt_arg,
                api_name,
                api_key,
                temp,
                system_message,
                streaming=effective_streaming_for_api_call,
            )

        # --- Post-processing and Return ---

        # If streaming was requested AND no chunking was done AND result is a generator, return it directly
        if (
            streaming
            and not recursive_summarization
            and not chunked_summarization
            and inspect.isgenerator(final_result)
        ):
            logging.info("Returning generator for streaming output.")
            return final_result
        else:
            # Otherwise, consume any potential generator to get the final string
            logging.debug(
                "Consuming final result (if generator) as streaming=False or chunking was used."
            )
            final_string_summary = consume_generator(final_result)

            # Final check and return
            if final_string_summary is None:
                logging.error("Summarization resulted in None after processing.")
                return "Error: Summarization failed unexpectedly."
            elif isinstance(
                final_string_summary, str
            ) and final_string_summary.startswith("Error:"):
                logging.error(
                    "Summarization failed; provider=%s",
                    safe_metadata_token(api_name),
                )
                return final_string_summary
            elif isinstance(final_string_summary, str):
                logging.info(
                    f"Summarization completed successfully. Final Length: {len(final_string_summary)}"
                )
                return final_string_summary
            else:
                # This case should ideally not be reached if consume_generator works correctly
                logging.error(
                    f"Unexpected final result type after processing: {type(final_string_summary)}"
                )
                return f"Error: Unexpected result type {type(final_string_summary)}"

    except Exception as e:
        logging.error(
            "Critical error in summarize function; exception_type=%s",
            safe_metadata_token(type(e).__name__),
        )
        return f"Error: An unexpected error occurred during summarization: {str(e)}"


#
# End of Analysis Function
###################################################################################


###################################################################################
#
# API Calls


def summarize_with_mock_llm(
    text_to_summarize: str,
    custom_prompt_arg: Optional[str],
    api_key: Optional[str] = None,
    temp: Optional[float] = None,
    system_message: Optional[str] = None,
    streaming: bool = False,
):
    """
    Mock implementation of OpenAI summarization function that mimics the behavior
    without making actual API calls.

    Returns either a string summary or a generator for streaming responses,
    matching the behavior of the real function.
    """
    try:
        # Log the same debug information as the real function
        logging.debug(
            f"MOCK-LLM (MOCK): Received text length: {len(str(text_to_summarize))}"
        )
        logging.debug("MOCK-LLM (MOCK): Custom prompt received")
        logging.debug("MOCK-LLM (MOCK): Request options prepared")

        # Extract a sample of text to include in mock response
        sample_text = (
            str(text_to_summarize)[:50] + "..."
            if len(str(text_to_summarize)) > 50
            else str(text_to_summarize)
        )

        # Create mock summary
        mock_summary = (
            f"[MOCK OPENAI RESPONSE]\n"
            f"This is a mock summary generated for testing purposes.\n\n"
            f"Sample of input text: '{sample_text}'\n"
            f"Custom prompt: '{custom_prompt_arg}'\n"
            f"Temperature: {temp or 0.7}\n"
            f"System message: '{system_message or 'You are a helpful AI assistant.'}'\n"
            f"Time generated: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # Add some simulated delay to mimic API latency
        time.sleep(0.5)

        mock_summary_text = f"Mocked summary for: {text_to_summarize[:30]}..."
        if streaming:

            def _stream():
                yield mock_summary_text

            return _stream()
        else:
            logging.debug("OpenAI (MOCK): Returning non-streaming mock response")
            return mock_summary

    except Exception as e:
        logging.error(
            "OpenAI (MOCK): Processing failed; exception_type=%s",
            safe_metadata_token(type(e).__name__),
        )
        return f"Error: OpenAI mock function unexpected error: {str(e)}"


#
#
############################################################################################################################################


# --- TASK-34895: chat-dispatch core -----------------------------------
# The OpenAI-wire cloud summarizers ride the SAME provider layer as chat:
# analyze()'s dispatch routes them through chat_api_call instead of the
# per-provider HTTP copies. Each spec preserves the legacy function's
# model sourcing (legacy `<provider>_api` section incl. its fallback),
# prompt assembly order, default temperature, and in-band error strings,
# so ingest callers and the Web_Scraping failure-marker contract see
# identical behavior. Deliberately NO diagnostic statements in this
# section: the diagnostic ledger pins this module's statement counts
# (Tests/LLM_Calls/test_summarization_diagnostic_privacy.py).
#
# Key resolution: parameter key, then the key readiness resolves
# (TASK-33001.14), then the legacy `<provider>_api` section while no
# credential_source is recorded; ciphertext is never a key, and keyless
# users see the same in-band refusal the per-provider functions produced.

_CHAT_SUMMARIZER_DEFAULT_SYSTEM = (
    "You are a helpful AI assistant who does whatever the user requests."
)


@dataclass(frozen=True)
class _SummarizerChatSpec:
    endpoint: str
    legacy_section: str
    default_model: Optional[str]
    default_temp: Optional[float]
    missing_key_message: str
    empty_message: str
    error_template: str
    # Local endpoints have no key requirement (auth optional or absent);
    # the keyless in-band refusal applies only where the legacy function
    # had one.
    require_key: bool = True
    # Read `<section>.temperature` when the caller passes no temp, the
    # way the legacy local functions did.
    config_temp: bool = False
    # "text" -> f"{text} \n\n\n\n{cp}" (the openai-family order);
    # "mistral" -> f"{cp}\n\n\n\n{text} " (legacy summarize_with_mistral);
    # "huggingface" -> f"{cp}\n\n\n{text}" (legacy summarize_with_huggingface).
    prompt_style: str = "text"
    # The chat path already applies the model_capabilities sampling gates
    # (openai max_completion_tokens swap, anthropic temperature/top_k/top_p
    # predicates) when the model is forwarded, so specs only carry the
    # per-provider system-message default the legacy function used.
    system_default: str = _CHAT_SUMMARIZER_DEFAULT_SYSTEM


_SUMMARIZER_CHAT_SPECS: Dict[str, _SummarizerChatSpec] = {
    "groq": _SummarizerChatSpec(
        endpoint="groq",
        legacy_section="groq_api",
        default_model="llama3-70b-8192",
        default_temp=0.2,
        missing_key_message=(
            "Groq: API Key Not Provided/Found in Config file or is empty"
        ),
        empty_message="Groq: Expected data not found in API response.",
        error_template=(
            "Groq: Error occurred while processing summary with Groq: {detail}"
        ),
    ),
    "deepseek": _SummarizerChatSpec(
        endpoint="deepseek",
        legacy_section="deepseek_api",
        default_model="deepseek-chat",
        default_temp=0.1,
        missing_key_message=(
            "DeepSeek: API Key Not Provided/Found in Config file or is empty"
        ),
        empty_message="DeepSeek: Summary not available",
        error_template="DeepSeek: Error occurred while processing summary: {detail}",
    ),
    "mistral": _SummarizerChatSpec(
        endpoint="mistral",
        legacy_section="mistral_api",
        default_model="mistral-large-latest",
        default_temp=0.2,
        missing_key_message=(
            "Mistral: API Key Not Provided/Found in Config file or is empty"
        ),
        empty_message="Mistral: Summary not available",
        error_template="Mistral: Error occurred while processing summary: {detail}",
        prompt_style="mistral",
    ),
    "openrouter": _SummarizerChatSpec(
        endpoint="openrouter",
        legacy_section="openrouter_api",
        default_model="mistralai/mistral-7b-instruct",
        default_temp=0.1,
        # Marker-matching on purpose: Web_Scraping's failure detector
        # recognizes "error occurred", not the legacy "not provided" shape.
        missing_key_message=(
            "OpenRouter: Error occurred while processing summary with "
            "OpenRouter: API key not provided/found or is empty"
        ),
        empty_message="openrouter: Expected data not found in API response.",
        error_template=(
            "openrouter: Error occurred while processing summary with "
            "openrouter: {detail}"
        ),
    ),
    "openai": _SummarizerChatSpec(
        endpoint="openai",
        legacy_section="openai_api",
        default_model="gpt-4o",
        default_temp=0.7,
        missing_key_message=("Error: OpenAI API Key Not Provided/Found or is empty."),
        empty_message="Error: OpenAI Summary not found in response.",
        error_template="Error: OpenAI API request failed: {detail}",
        system_default="You are a helpful AI assistant.",
    ),
    "anthropic": _SummarizerChatSpec(
        endpoint="anthropic",
        legacy_section="anthropic_api",
        default_model="claude-haiku-4-5",
        default_temp=0.1,
        missing_key_message=(
            "Anthropic: API Key Not Provided/Found in Config file or is empty"
        ),
        empty_message="Anthropic: Summary not available",
        error_template=(
            "Anthropic: Error occurred while processing summary with "
            "Anthropic: {detail}"
        ),
    ),
    "cohere": _SummarizerChatSpec(
        endpoint="cohere",
        legacy_section="cohere_api",
        default_model="command-a-03-2025",
        default_temp=0.3,
        missing_key_message=(
            "Cohere: API Key Not Provided/Found in Config file or is empty"
        ),
        empty_message="Cohere: Expected data not found in API response.",
        error_template=(
            "Cohere: Error occurred while processing summary with Cohere: {detail}"
        ),
    ),
    "google": _SummarizerChatSpec(
        endpoint="google",
        legacy_section="google_api",
        default_model="gemini-1.5-pro",
        default_temp=0.2,
        missing_key_message=(
            "Google: API Key Not Provided/Found in Config file or is empty"
        ),
        empty_message="Google: Summary not available",
        error_template="Google: Unexpected error occurred: {detail}",
    ),
    "huggingface": _SummarizerChatSpec(
        endpoint="huggingface",
        legacy_section="huggingface_api",
        default_model="mistralai/Mistral-7B-Instruct-v0.2",
        default_temp=0.1,
        missing_key_message=(
            "HuggingFace: Error occurred while processing summary with "
            "HuggingFace: API key not provided/found or is empty"
        ),
        empty_message="HuggingFace: Expected 'generated_text' in API response.",
        error_template=(
            "HuggingFace: Error occurred while processing summary with "
            "HuggingFace: {detail}"
        ),
        prompt_style="huggingface",
    ),
    "llama.cpp": _SummarizerChatSpec(
        endpoint="llama_cpp",
        legacy_section="llama_cpp_api",
        default_model=None,
        default_temp=0.7,
        missing_key_message="Llama: API Key Not Provided/Found in Config file or is empty",
        empty_message="Llama: No choices in response data",
        error_template="Llama: Error occurred while processing summary with Llama: {detail}",
        require_key=False,
        prompt_style="template_nl",
        config_temp=True,
    ),
    "kobold": _SummarizerChatSpec(
        endpoint="koboldcpp",
        legacy_section="kobold_api",
        default_model=None,
        default_temp=0.7,
        missing_key_message="Kobold: API Key Not Provided/Found in Config file or is empty",
        empty_message="Kobold: Expected data not found in API response.",
        error_template="Kobold: Error occurred while processing summary with Kobold: {detail}",
        require_key=False,
        prompt_style="template_nl4",
    ),
    "ooba": _SummarizerChatSpec(
        endpoint="oobabooga",
        legacy_section="ooba_api",
        default_model=None,
        default_temp=0.7,
        missing_key_message="Ooba API: API Key Not Provided/Found in Config file or is empty",
        empty_message="Ooba API: Failed to process summary.",
        error_template="Ooba API: Unexpected error occurred: {detail}",
        require_key=False,
        config_temp=True,
    ),
    "tabbyapi": _SummarizerChatSpec(
        endpoint="tabbyapi",
        legacy_section="tabby_api",
        default_model=None,
        default_temp=None,
        missing_key_message="TabbyAPI: API Key Not Provided/Found in Config file or is empty",
        empty_message="Error: Received an invalid response structure from TabbyAPI.",
        error_template="TabbyAPI: Unexpected error in summarization process: {detail}",
        require_key=False,
        prompt_style="template_nl4",
    ),
    "vllm": _SummarizerChatSpec(
        endpoint="vllm",
        legacy_section="vllm_api",
        default_model=None,
        default_temp=None,
        missing_key_message="vLLM Summarization: API Key Not Provided/Found in Config file or is empty",
        empty_message="vLLM Summarization: Summary not available",
        error_template="vLLM Summarization: Unexpected error occurred: {detail}",
        require_key=False,
        config_temp=True,
    ),
    "local-llm": _SummarizerChatSpec(
        endpoint="local-llm",
        legacy_section="local_llm",
        default_model=None,
        default_temp=0.7,
        missing_key_message="Local LLM: API Key Not Provided/Found in Config file or is empty",
        empty_message="Local LLM: Summary not available",
        error_template="Local LLM: Error occurred while processing summary: {detail}",
        require_key=False,
    ),
    "ollama": _SummarizerChatSpec(
        endpoint="ollama",
        legacy_section="ollama_api",
        default_model=None,
        default_temp=0.7,
        missing_key_message="Ollama: API Key Not Provided/Found in Config file or is empty",
        empty_message="Ollama: Summarization API response missing text.",
        error_template="Ollama: Error occurred while summarizing: {detail}",
        require_key=False,
        prompt_style="ollama",
        config_temp=True,
    ),
    "custom-openai-api": _SummarizerChatSpec(
        endpoint="custom-openai-api",
        legacy_section="custom_openai_api",
        default_model=None,
        default_temp=None,
        missing_key_message=(
            "Custom OpenAI API: API Key Not Provided/Found in Config file or is empty"
        ),
        empty_message="Custom OpenAI API: Chat not available",
        error_template="OpenAI: Failed to process chat response. Status code: {detail}",
        config_temp=True,
    ),
    "custom-openai-api-2": _SummarizerChatSpec(
        endpoint="custom-openai-api-2",
        legacy_section="custom_openai_api_2",
        default_model=None,
        default_temp=None,
        missing_key_message=(
            "Custom OpenAI API API-2: API Key Not Provided/Found in Config file or is empty"
        ),
        empty_message="Custom OpenAI API API-2: Chat not available",
        error_template="OpenAI: Failed to process chat response. Status code: {detail}",
        config_temp=True,
    ),
    # --- engine presets (task-34895): summarization can now select any
    # chat-dispatch endpoint; these had NO legacy per-provider function,
    # so there is no legacy behavior to preserve -- the chat handler's own
    # defaults govern model/temperature. Moonshot and Z.ai require a key:
    # their handlers search the environment themselves when handed none,
    # so a summary refuses where readiness resolves none (TASK-33001.14).
    "moonshot": _SummarizerChatSpec(
        endpoint="moonshot",
        legacy_section="moonshot_api",
        default_model=None,
        default_temp=None,
        missing_key_message="Moonshot: API Key Not Provided/Found in Config file or is empty",
        empty_message="Moonshot: Summary not available",
        error_template="Moonshot: Error occurred while processing summary: {detail}",
    ),
    "zai": _SummarizerChatSpec(
        endpoint="zai",
        legacy_section="zai_api",
        default_model=None,
        default_temp=None,
        missing_key_message="ZAI: API Key Not Provided/Found in Config file or is empty",
        empty_message="ZAI: Summary not available",
        error_template="ZAI: Error occurred while processing summary: {detail}",
    ),
    "aphrodite": _SummarizerChatSpec(
        endpoint="aphrodite",
        legacy_section="aphrodite_api",
        default_model=None,
        default_temp=None,
        missing_key_message="Aphrodite: API Key Not Provided/Found in Config file or is empty",
        empty_message="Aphrodite: Summary not available",
        error_template="Aphrodite: Error occurred while processing summary: {detail}",
        require_key=False,
    ),
    "mlx_lm": _SummarizerChatSpec(
        endpoint="mlx_lm",
        legacy_section="mlx_lm_api",
        default_model=None,
        default_temp=None,
        missing_key_message="MLX: API Key Not Provided/Found in Config file or is empty",
        empty_message="MLX: Summary not available",
        error_template="MLX: Error occurred while processing summary: {detail}",
        require_key=False,
    ),
}


@_provider_recovery.unqualified
def _chat_stream_text_chunks(stream: Iterable[Any]) -> Generator[str, None, None]:
    """Translate one chat streaming reply into plain text chunks.

    Chat handlers stream OpenAI-style SSE wire frames (``data: {...}``
    lines with a terminal ``data: [DONE]``); the summarization contract
    yields plain text chunks. Unparseable frames are skipped, matching the
    legacy per-provider stream generators; an in-stream provider error
    frame becomes one ``Error during streaming: ...`` chunk, matching the
    legacy OpenAI summarizer's behavior.
    """
    for frame in stream:
        for line in frame.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if not payload:
                continue
            if payload == "[DONE]":
                return
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            failure = event.get("error")
            if isinstance(failure, dict):
                failure = failure.get("message") or failure.get("code")
            if failure:
                yield f"Error during streaming: {failure}"
                return
            choices = event.get("choices") or []
            if not choices or not isinstance(choices[0], dict):
                continue
            delta = choices[0].get("delta") or {}
            piece = delta.get("content") or choices[0].get("text") or ""
            if piece:
                yield piece


def _resolve_summarizer_temp(spec: "_SummarizerChatSpec", temp: Any) -> Any:
    if temp is not None:
        return float(temp)
    if spec.config_temp:
        return get_cli_setting(spec.legacy_section, "temperature", spec.default_temp)
    return spec.default_temp


def _readiness_credential(endpoint: str) -> tuple[str | None, str | None]:
    """The key readiness resolves for ``endpoint``, and the recorded source."""
    from ..Chat.provider_readiness import (
        configured_provider_credential_source,
        provider_config_key,
        resolve_provider_credential,
    )

    provider_key = provider_config_key(endpoint)
    try:
        table = provider_settings_for_key(
            load_settings().get("api_settings"), provider_key
        )
    except ProviderSettingsError:
        return None, None
    key = resolve_provider_credential(provider_key, table, environ=os.environ)[0]
    return key, configured_provider_credential_source(table)


def _summarize_via_chat_api(
    spec_key: str,
    api_key,
    input_data,
    custom_prompt_arg,
    temp,
    system_message,
    streaming,
    credential_provider: str | None = None,
):
    """One summarization call over the unified chat provider dispatch.

    ``credential_provider`` names the provider whose ``[api_settings]``
    table decides the key; the spec's endpoint when None.
    """
    spec = _SUMMARIZER_CHAT_SPECS[spec_key]
    if isinstance(input_data, dict) and "summary" in input_data:
        return input_data["summary"]
    if isinstance(input_data, list):
        text = extract_text_from_segments(input_data)
    elif isinstance(input_data, str):
        text = input_data
    else:
        raise ValueError(f"{spec.endpoint}: Invalid input data format")

    if spec.prompt_style == "mistral":
        user_content = f"{custom_prompt_arg}\n\n\n\n{text} "
    elif spec.prompt_style == "huggingface":
        user_content = f"{custom_prompt_arg}\n\n\n{text}"
    elif spec.prompt_style in ("template_nl", "template_nl4", "ollama"):
        # The local summarizers fall back to the registry's local template
        # when the caller brings no custom prompt (the five LS call sites
        # the internal-prompts migration pinned).
        effective_cp = custom_prompt_arg or get_internal_prompt(
            "summarization.local_summarizer_template"
        )
        if spec.prompt_style == "ollama":
            user_content = f"{effective_cp}\n\n{text}"
        elif spec.prompt_style == "template_nl":
            user_content = f"{effective_cp}\n\n{text}"
        else:
            user_content = f"{effective_cp}\n\n\n\n{text}"
    else:
        user_content = f"{text} \n\n\n\n{custom_prompt_arg}"
    messages = [{"role": "user", "content": user_content}]

    # Deferred: chat-side imports stay off this module's import time (the
    # ingest import-weight tests pin it).
    from ..Chat.Chat_Functions import chat_api_call, chat_reply_text
    from ..Chat.Chat_Deps import (
        ChatAPIError,
    )

    # Key resolution: the parameter key first; then (TASK-33001.14, ADR-012)
    # the key readiness resolves for the provider's [api_settings] table, so
    # credential_source decides here as it does for the Console; the legacy
    # `<provider>_api` section only while no source is recorded. `enc:`
    # ciphertext is never a credential. Refusing HERE (in-band, before the
    # chat dispatch) keeps the legacy refusal shape.
    raw_config_key = None
    if api_key and api_key.strip():
        resolved_key = api_key
    else:
        resolved_key, source = _readiness_credential(
            credential_provider or spec.endpoint
        )
        if resolved_key is None and source is None:
            raw_config_key = get_cli_setting(spec.legacy_section, "api_key")
            resolved_key = without_ciphertext(raw_config_key)
    if not (resolved_key and resolved_key.strip()):
        # Refuse where the provider requires a key, and wherever a key WAS
        # declared but is ciphertext -- `enc:` text is never a credential,
        # even for providers where auth is otherwise optional.
        if spec.require_key or raw_config_key:
            return spec.missing_key_message

    try:
        result = chat_api_call(
            spec.endpoint,
            messages,
            api_key=resolved_key,
            temp=_resolve_summarizer_temp(spec, temp),
            system_message=(
                spec.system_default if system_message is None else system_message
            ),
            model=get_cli_setting(spec.legacy_section, "model", spec.default_model),
            streaming=streaming,
        )
    except ChatAPIError as exc:
        # Handler-side failures (auth, missing URL/model, provider status)
        # all use the generic template: the pre-dispatch key resolution
        # above is the only source of the missing-key literal, so a
        # valid-but-rejected key or a missing local URL reports what
        # actually happened instead of masquerading as "no key".
        return spec.error_template.format(detail=str(exc))

    if streaming:
        return _chat_stream_text_chunks(result)
    summary = chat_reply_text(result).strip()
    if not summary:
        # An empty completion with reasoning-content evidence gets the
        # spec's diagnostic; a plain empty completion returns "" (the
        # openai-family legacy behavior) so downstream falsy-summary
        # fallbacks fire instead of storing a pseudo-error string.
        detail = _reasoning_only_completion_detail(result)
        if detail:
            return spec.empty_message + detail
        return ""
    return summary


def _reasoning_only_completion_detail(result: Any) -> str:
    """Explain an empty completion that spent its budget on reasoning.

    Port of the legacy llama summarizer's diagnostic (task-17384): a
    response with choices but no message content usually spent its token
    budget on ``reasoning_content``. The empty-message prefix stays
    spec-owned so the Web_Scraping failure detector keeps recognizing it.
    """
    if not isinstance(result, dict):
        return ""
    first = (result.get("choices") or [{}])[0]
    if not isinstance(first, dict):
        return ""
    finish = first.get("finish_reason")
    message = first.get("message")
    reasoning = ""
    if isinstance(message, dict):
        reasoning = str(message.get("reasoning_content") or "")
    spent = (result.get("usage") or {}).get("completion_tokens")
    parts = []
    if finish:
        parts.append(f"finish_reason={finish}")
    if spent is not None:
        parts.append(f"completion_tokens={spent}")
    if reasoning:
        parts.append(
            f"reasoning-only completion ({len(reasoning)} chars "
            "of reasoning, no content)"
        )
    if parts:
        return " (" + "; ".join(parts) + ")"
    return ""
