# mlx_lm_inference_local.py
#
# Imports
import logging
import os
import subprocess
import time
from typing import Optional
#
# Third-party Libraries
#
# Local Imports
from ..Metrics.metrics_logger import log_counter, log_histogram
#
#
########################################################################################################################
#
# Functions:

def start_mlx_lm_server(
    model_path: str,
    host: str,
    port: int,
    additional_args: Optional[str] = None
) -> Optional[subprocess.Popen]:
    """
    Starts the MLX LM server using subprocess.Popen.

    Args:
        model_path: Path to the MLX model (HuggingFace ID or local path).
        host: Host address for the server.
        port: Port for the server.
        additional_args: Optional string of additional arguments for the server command.

    Returns:
        A subprocess.Popen object if successful, None otherwise.
    """
    start_time = time.time()
    log_counter("mlx_lm_server_start_attempt", labels={
        "model": model_path.split('/')[-1] if '/' in model_path else model_path,
        "port": str(port)
    })
    
    command = [
        "python", "-m", "mlx_lm.server",
        "--model", model_path,
        "--host", host,
        "--port", str(port)
    ]
    if additional_args:
        command.extend(additional_args.split())

    logging.info(f"Starting MLX-LM server with command: {' '.join(command)}")
    try:
        # Set environment variable to disable output buffering for Python
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True, # Ensure text mode for stdout/stderr
            env=env
        )
        logging.info(f"MLX-LM server started with PID: {process.pid}")
        
        # Log success metrics
        duration = time.time() - start_time
        log_histogram("mlx_lm_server_start_duration", duration, labels={"status": "success"})
        log_counter("mlx_lm_server_start_success", labels={
            "model": model_path.split('/')[-1] if '/' in model_path else model_path,
            "pid": str(process.pid)
        })
        
        return process
    except FileNotFoundError:
        logging.error(
            "Error starting MLX-LM server: 'python' or 'mlx_lm.server' not found. "
            "Ensure Python is installed and mlx-lm is in your Python path (pip install mlx-lm)."
        )
        log_counter("mlx_lm_server_start_error", labels={"error_type": "file_not_found"})
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while starting MLX-LM server: {e}", exc_info=True)
        log_counter("mlx_lm_server_start_error", labels={"error_type": type(e).__name__})
        return None

def stop_mlx_lm_server(process: subprocess.Popen) -> None:
    """
    Stops the MLX LM server process.

    Args:
        process: The subprocess.Popen object representing the server process.
    """
    if not process:
        logging.warning("Stop MLX-LM server: No process provided.")
        log_counter("mlx_lm_server_stop_error", labels={"error": "no_process"})
        return

    if process.poll() is not None:
        logging.info(f"MLX-LM server (PID: {process.pid}) already terminated with code {process.returncode}.")
        log_counter("mlx_lm_server_stop_already_terminated", labels={
            "return_code": str(process.returncode)
        })
        return

    start_time = time.time()
    log_counter("mlx_lm_server_stop_attempt", labels={"pid": str(process.pid)})
    
    logging.info(f"Stopping MLX-LM server with PID: {process.pid}...")
    try:
        process.terminate()
        try:
            process.wait(timeout=10) # Increased timeout for graceful shutdown
            logging.info(f"MLX-LM server (PID: {process.pid}) terminated gracefully with code {process.returncode}.")
            
            # Log successful graceful shutdown
            duration = time.time() - start_time
            log_histogram("mlx_lm_server_stop_duration", duration, labels={"method": "terminate"})
            log_counter("mlx_lm_server_stop_success", labels={
                "method": "terminate",
                "return_code": str(process.returncode)
            })
            
        except subprocess.TimeoutExpired:
            logging.warning(
                f"MLX-LM server (PID: {process.pid}) did not terminate gracefully within timeout. Killing..."
            )
            log_counter("mlx_lm_server_stop_timeout", labels={"phase": "terminate"})
            
            process.kill()
            try:
                process.wait(timeout=5) # Wait for kill
                logging.info(f"MLX-LM server (PID: {process.pid}) killed, return code {process.returncode}.")
                
                # Log successful kill
                duration = time.time() - start_time
                log_histogram("mlx_lm_server_stop_duration", duration, labels={"method": "kill"})
                log_counter("mlx_lm_server_stop_success", labels={
                    "method": "kill",
                    "return_code": str(process.returncode)
                })
                
            except subprocess.TimeoutExpired:
                logging.error(f"MLX-LM server (PID: {process.pid}) did not die even after kill. Manual intervention may be needed.")
                log_counter("mlx_lm_server_stop_error", labels={"error": "kill_timeout"})
        except Exception as e_wait: # Catch other errors during wait (e.g. InterruptedError)
            logging.error(f"Error waiting for MLX-LM server (PID: {process.pid}) to terminate: {e_wait}", exc_info=True)


    except ProcessLookupError: # If the process was already gone
        logging.info(f"MLX-LM server (PID: {process.pid}) was already gone before explicit stop.")
        log_counter("mlx_lm_server_stop_process_not_found")
    except Exception as e_term:
        logging.error(f"Error during initial termination of MLX-LM server (PID: {process.pid}): {e_term}", exc_info=True)
        log_counter("mlx_lm_server_stop_error", labels={"error_type": type(e_term).__name__})
        
        # If terminate fails, try to kill as a fallback if it's still running
        if process.poll() is None:
            logging.info(f"Attempting to kill MLX-LM server (PID: {process.pid}) as terminate failed.")
            process.kill()
            try:
                process.wait(timeout=5)
                logging.info(f"MLX-LM server (PID: {process.pid}) killed after terminate failed, return code {process.returncode}.")
                log_counter("mlx_lm_server_stop_fallback_kill_success")
            except Exception as e_kill_wait:
                logging.error(f"Error waiting for MLX-LM server (PID: {process.pid}) to die after kill: {e_kill_wait}", exc_info=True)
                log_counter("mlx_lm_server_stop_fallback_kill_error", labels={"error_type": type(e_kill_wait).__name__})

#
# End of mlx_lm_inference_local.py
########################################################################################################################
