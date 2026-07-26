# worker_events.py
# Description:
#
# Imports
import json  # Added for SSE JSON parsing
import time
from typing import TYPE_CHECKING, Generator, Any, Union

#
# 3rd-Party Imports
from loguru import logger as _loguru_fallback_logger
from textual.message import Message

#
# Local Imports
# Import the actual chat function if it's to be called from chat_wrapper_function
from ..Chat.Chat_Functions import chat as core_chat_function
from ..Metrics.metrics_logger import log_counter, log_histogram

#
if TYPE_CHECKING:
    from ..app import TldwCli


# Custom Messages for Streaming
class StreamingChunk(Message):
    """Custom message to send a piece of streamed text."""

    def __init__(self, text_chunk: str) -> None:
        super().__init__()
        self.text_chunk = text_chunk


class StreamingChunkWithLogits(StreamingChunk):
    """Extended streaming chunk that includes logprobs data."""

    def __init__(self, text_chunk: str, logprobs: Union[dict, None] = None) -> None:
        super().__init__(text_chunk)
        self.logprobs = logprobs


class StreamDone(Message):
    """Custom message to signal the end of a stream."""

    def __init__(
        self,
        full_text: str,
        error: Union[str, None] = None,
        response_data: Union[dict, None] = None,
    ) -> None:
        super().__init__()
        self.full_text = full_text
        self.error = error  # Store error
        self.response_data = (
            response_data  # Store the raw response data for tool parsing
        )


########################################################################################################################
#
# Worker Target Function
#
########################################################################################################################


def chat_wrapper_function(
    app_instance: "TldwCli", strip_thinking_tags: bool = True, **kwargs: Any
) -> Any:
    """
    This function is the target for the worker. It calls the core_chat_function.
    If core_chat_function returns a generator (for streaming), this function consumes it,
    posts StreamingChunk and StreamDone messages, and returns a specific string value.
    If core_chat_function returns a direct result (non-streaming), this function returns it as is.
    """
    logger = getattr(app_instance, "loguru_logger", _loguru_fallback_logger)
    start_time = time.time()

    # Extract relevant parameters for metrics
    api_endpoint = kwargs.get("api_endpoint", "unknown")
    streaming = kwargs.get("streaming", False)
    model = kwargs.get("model", "unknown")

    # Log worker execution start
    log_counter(
        "chat_worker_execution_start",
        labels={"provider": api_endpoint, "streaming": str(streaming), "model": model},
    )
    api_endpoint = kwargs.get("api_endpoint")
    model_name = kwargs.get("model")
    logprobs_enabled = kwargs.get("llm_logprobs", False)
    top_logprobs = kwargs.get("llm_top_logprobs", 0)
    # streaming_requested flag from kwargs is implicitly handled by core_chat_function's return type.
    logger.debug(
        f"chat_wrapper_function executing for endpoint '{api_endpoint}', model '{model_name}', logprobs={logprobs_enabled}, top_logprobs={top_logprobs}"
    )

    try:
        # core_chat_function is your synchronous `Chat.Chat_Functions.chat`
        result = core_chat_function(strip_thinking_tags=strip_thinking_tags, **kwargs)

        if isinstance(result, Generator):  # Streaming case
            logger.info(
                f"Core chat function returned a generator for '{api_endpoint}' (model '{model_name}', strip_tags={strip_thinking_tags}). Processing stream in worker."
            )
            accumulated_full_text = ""
            accumulated_reasoning_text = (
                ""  # Separate accumulator for reasoning content
            )
            accumulated_tool_calls = []  # Accumulator for tool calls in streaming
            error_message_if_any = None
            chunk_count = 0
            stream_start_time = time.time()

            log_counter(
                "chat_worker_streaming_started",
                labels={"provider": api_endpoint, "model": model_name},
            )

            try:
                for chunk_raw in result:
                    # Check for worker cancellation at the beginning of each iteration
                    if (
                        app_instance.current_chat_worker
                        and app_instance.current_chat_worker.is_cancelled
                    ):
                        app_instance.loguru_logger.info(
                            "Chat worker cancelled by user during stream processing in chat_wrapper_function."
                        )
                        if hasattr(result, "close"):
                            result.close()
                            app_instance.loguru_logger.debug(
                                "Closed response_gen (result)."
                            )
                        # Post a StreamDone event indicating cancellation
                        # Use accumulated_full_text if it contains partial data, or a specific message
                        cancellation_message = "Streaming cancelled by user."
                        # If accumulated_full_text is meaningful, you could prepend/append the cancellation reason.
                        # For now, we assume the UI will primarily use the 'error' field from StreamDone.
                        app_instance.post_message(
                            StreamDone(
                                full_text=accumulated_full_text,
                                error=cancellation_message,
                                response_data=None,
                            )
                        )
                        return "STREAMING_HANDLED_BY_EVENTS"  # Exit the worker function

                    # Process each raw chunk from the generator (expected to be SSE lines)
                    line = str(chunk_raw).strip()
                    if not line:
                        continue

                    if line.startswith("data:"):
                        json_str = line[len("data:") :].strip()
                        if json_str == "[DONE]":
                            logger.info(
                                f"SSE Stream: Received [DONE] for '{api_endpoint}', model '{model_name}'."
                            )
                            break  # End of stream
                        if not json_str:
                            continue
                        try:
                            json_data = json.loads(json_str)
                            actual_text_chunk = ""
                            reasoning_chunk = ""
                            logprobs_data = None
                            # Standard OpenAI SSE structure, adapt if providers differ or if pre-parsed objects are yielded
                            # Log the entire json_data structure when logprobs is requested
                            if logprobs_enabled and api_endpoint in [
                                "openai",
                                "llama_cpp",
                                "vllm",
                            ]:
                                if "choices" in json_data and json_data.get("choices"):
                                    # Only log if we haven't logged the structure yet
                                    if not hasattr(
                                        chat_wrapper_function, "_logged_structure"
                                    ):
                                        logger.info(
                                            f"First streaming chunk structure for {api_endpoint} with logprobs enabled: {json.dumps(json_data, indent=2)[:1000]}..."
                                        )
                                        chat_wrapper_function._logged_structure = True

                            choices = json_data.get("choices")
                            if (
                                choices
                                and isinstance(choices, list)
                                and len(choices) > 0
                            ):
                                delta = choices[0].get("delta", {})
                                if "content" in delta and delta["content"] is not None:
                                    actual_text_chunk = delta["content"]
                                # Check for DeepSeek reasoning content in delta
                                if (
                                    "reasoning_content" in delta
                                    and delta["reasoning_content"] is not None
                                ):
                                    reasoning_chunk = delta["reasoning_content"]
                                    logger.debug(
                                        f"Found reasoning_content in streaming delta: {reasoning_chunk[:100]}..."
                                    )
                                    # Accumulate reasoning content separately
                                    accumulated_reasoning_text += reasoning_chunk

                                # Check for logprobs in the choice (OpenAI format)
                                if (
                                    "logprobs" in choices[0]
                                    and choices[0]["logprobs"] is not None
                                ):
                                    logprobs_data = choices[0]["logprobs"]
                                    logger.info(
                                        f"Found logprobs in streaming choice: {logprobs_data}"
                                    )
                                # Also check for logprobs in delta (some providers put it here)
                                elif (
                                    "logprobs" in delta
                                    and delta["logprobs"] is not None
                                ):
                                    logprobs_data = delta["logprobs"]
                                    logger.info(
                                        f"Found logprobs in streaming delta: {logprobs_data}"
                                    )
                                # Log if we're expecting logprobs but not finding them
                                elif logprobs_enabled:
                                    logger.debug(
                                        f"Logprobs enabled but not found in chunk. Choice keys: {list(choices[0].keys())}, Delta keys: {list(delta.keys())}"
                                    )

                                # Check for tool calls in the delta or choice
                                if "tool_calls" in delta:
                                    logger.debug(
                                        f"Found tool_calls in streaming delta: {delta['tool_calls']}"
                                    )
                                    accumulated_tool_calls.extend(delta["tool_calls"])
                                elif "tool_calls" in choices[0]:
                                    logger.debug(
                                        f"Found tool_calls in streaming choice: {choices[0]['tool_calls']}"
                                    )
                                    accumulated_tool_calls = choices[0][
                                        "tool_calls"
                                    ]  # Replace, don't extend

                            if actual_text_chunk:
                                # Use StreamingChunkWithLogits if logprobs are available
                                if logprobs_data:
                                    logger.debug(
                                        f"Posting StreamingChunkWithLogits with text: '{actual_text_chunk[:20]}...' and logprobs"
                                    )
                                    app_instance.post_message(
                                        StreamingChunkWithLogits(
                                            actual_text_chunk, logprobs_data
                                        )
                                    )
                                else:
                                    if logprobs_enabled:
                                        logger.debug(
                                            f"Posting regular StreamingChunk (no logprobs found) with text: '{actual_text_chunk[:20]}...'"
                                        )
                                    app_instance.post_message(
                                        StreamingChunk(actual_text_chunk)
                                    )
                                accumulated_full_text += actual_text_chunk
                                chunk_count += 1
                            # else:
                            #     logger.trace(f"SSE Stream: No text content in data chunk: {json_str[:100]}")

                        except json.JSONDecodeError as e_json:
                            logger.warning(
                                f"SSE Stream: JSON parsing error for chunk in '{api_endpoint}', model '{model_name}': {e_json}. Chunk: >>{json_str[:100]}<<"
                            )
                        except Exception as e_parse:
                            logger.opt(exception=True).error(
                                f"SSE Stream: Error processing JSON data in '{api_endpoint}', model '{model_name}': {e_parse}. Data: >>{json_str[:100]}<<"
                            )

                    elif line.startswith("event:"):
                        event_type = line[len("event:") :].strip()
                        logger.debug(
                            f"SSE Stream: Received event type '{event_type}' for '{api_endpoint}', model '{model_name}'."
                        )
                        # Handle specific events if necessary (e.g., error events from provider)
                        if (
                            event_type == "error"
                        ):  # Example for a hypothetical provider error event
                            logger.error(
                                f"SSE Stream: Provider indicated an error event for '{api_endpoint}', model '{model_name}'. Line: {line}"
                            )
                            error_message_if_any = f"Provider error event: {event_type}"  # Potentially parse more details
                            # Depending on severity, might want to break here.
                    # else:
                    # logger.trace(f"SSE Stream: Ignoring non-data/non-event line: {line[:100]}")

            except Exception as e_stream_loop:
                logger.exception(
                    f"Error during stream processing loop for '{api_endpoint}', model '{model_name}': {e_stream_loop}"
                )
                error_message_if_any = f"Error during streaming: {str(e_stream_loop)}"
            finally:
                stream_duration = time.time() - stream_start_time
                logger.info(
                    f"SSE Stream: Loop finished for '{api_endpoint}', model '{model_name}'. Posting StreamDone. Total length: {len(accumulated_full_text)}. Error: {error_message_if_any}"
                )

                # Log streaming metrics
                log_histogram(
                    "chat_worker_streaming_duration",
                    stream_duration,
                    labels={
                        "provider": api_endpoint,
                        "model": model_name,
                        "success": str(error_message_if_any is None),
                    },
                )
                log_histogram(
                    "chat_worker_streaming_chunks",
                    chunk_count,
                    labels={"provider": api_endpoint, "model": model_name},
                )
                log_histogram(
                    "chat_worker_streaming_response_length",
                    len(accumulated_full_text),
                    labels={"provider": api_endpoint, "model": model_name},
                )
                if chunk_count > 0:
                    log_histogram(
                        "chat_worker_streaming_chunk_rate",
                        chunk_count / stream_duration,
                        labels={"provider": api_endpoint, "model": model_name},
                    )

                # Combine reasoning and regular content if needed
                final_text = accumulated_full_text
                if accumulated_reasoning_text and hasattr(app_instance, "app_config"):
                    strip_tags_setting = app_instance.app_config.get(
                        "chat_defaults", {}
                    ).get("strip_thinking_tags", True)
                    if not strip_tags_setting:
                        # Prepend all accumulated reasoning wrapped in a single pair of thinking tags
                        final_text = f"<thinking>\n{accumulated_reasoning_text}\n</thinking>\n\n{accumulated_full_text}"
                        logger.info(
                            f"Combined reasoning content with main content (reasoning length: {len(accumulated_reasoning_text)})"
                        )

                # Create a response data structure if we have tool calls
                response_data = None
                if accumulated_tool_calls:
                    response_data = {
                        "choices": [
                            {
                                "message": {
                                    "content": accumulated_full_text,
                                    "tool_calls": accumulated_tool_calls,
                                }
                            }
                        ]
                    }
                    logger.info(
                        f"Created response_data with {len(accumulated_tool_calls)} tool calls for StreamDone"
                    )

                app_instance.post_message(
                    StreamDone(
                        full_text=final_text,
                        error=error_message_if_any,
                        response_data=response_data,
                    )
                )

            return "STREAMING_HANDLED_BY_EVENTS"  # Signal that streaming was handled via events

        else:  # Non-streaming case
            logger.debug(
                f"chat_wrapper_function for '{api_endpoint}' (model '{model_name}', strip_tags={strip_thinking_tags}) is returning a direct result (type: {type(result)})."
            )

            # Log non-streaming metrics
            worker_duration = time.time() - start_time
            log_histogram(
                "chat_worker_execution_duration",
                worker_duration,
                labels={
                    "provider": api_endpoint,
                    "model": model_name,
                    "streaming": "false",
                    "success": "true",
                },
            )

            return result  # Return the complete response directly

    except Exception as e:
        # This catches errors from core_chat_function if it fails *before* returning a generator,
        # or other unexpected errors within this wrapper function itself (outside the stream loop).
        logger.exception(
            f"Error in chat_wrapper_function for '{api_endpoint}', model '{model_name}' (potentially pre-stream or non-stream path): {e}"
        )

        # Log error metrics
        worker_duration = time.time() - start_time
        log_counter(
            "chat_worker_error",
            labels={
                "provider": api_endpoint,
                "model": model_name,
                "error_type": type(e).__name__,
            },
        )
        log_histogram(
            "chat_worker_execution_duration",
            worker_duration,
            labels={
                "provider": api_endpoint,
                "model": model_name,
                "streaming": str(streaming),
                "success": "false",
            },
        )

        if streaming:
            # Legacy streaming contract: report the failure through the (now unhandled)
            # StreamDone event and return the sentinel.
            app_instance.post_message(
                StreamDone(
                    full_text="", error=f"Chat wrapper error: {str(e)}", response_data=None
                )
            )
            return "STREAMING_HANDLED_BY_EVENTS"  # Exit the worker function
        # task-634: non-streaming callers consume the return value directly -- a swallowed
        # exception rendered the sentinel as the response. Propagate instead.
        raise


#
# End of worker_events.py
########################################################################################################################
