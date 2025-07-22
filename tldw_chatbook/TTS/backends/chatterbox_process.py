#!/usr/bin/env python3
"""
Chatterbox Process Wrapper

This script runs ChatterboxTTS in a completely isolated subprocess to prevent:
1. Terminal output corruption in TUI applications
2. File descriptor conflicts
3. UI blocking during model operations

Communication is done via JSON over stdin/stdout.
"""

import sys
import os

# CRITICAL: Redirect stdout/stderr BEFORE any other imports
# This prevents any library initialization output from corrupting our communication
original_stdout_fd = os.dup(1)  # Save original stdout
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

# Now safe to import other modules
import json
import base64
import io
import traceback
from pathlib import Path

# Create a writer using the saved stdout file descriptor
# This bypasses the Python-level redirection
comm_fd = os.fdopen(original_stdout_fd, 'w', buffering=1)  # Line buffering

def send_response(data):
    """Send response back to parent process via the original stdout"""
    comm_fd.write(json.dumps(data) + '\n')
    comm_fd.flush()

def send_chunked_audio(audio_bytes, format="wav", sample_rate=22050):
    """Send large audio data in chunks to avoid buffer limits"""
    # Check if chunking is needed (threshold: 1MB of base64 data)
    base64_data = base64.b64encode(audio_bytes).decode('utf-8')
    chunk_size = 512 * 1024  # 512KB chunks (will be ~680KB after base64)
    
    if len(base64_data) <= chunk_size:
        # Small enough to send in one message
        send_response({
            "type": "audio",
            "data": base64_data,
            "format": format,
            "sample_rate": sample_rate
        })
    else:
        # Need to chunk the data
        total_chunks = (len(base64_data) + chunk_size - 1) // chunk_size
        
        for i in range(total_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(base64_data))
            chunk_data = base64_data[start:end]
            
            send_response({
                "type": "audio_chunk",
                "chunk_id": i,
                "total_chunks": total_chunks,
                "data": chunk_data,
                "format": format if i == 0 else None,  # Only send metadata with first chunk
                "sample_rate": sample_rate if i == 0 else None
            })
        
        # Send completion message
        send_response({
            "type": "audio_complete",
            "total_chunks": total_chunks
        })

def main():
    """Main process loop"""
    # Remove the current directory from sys.path to avoid conflicts
    # with the local chatterbox.py file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir in sys.path:
        sys.path.remove(current_dir)
    
    # Also remove parent directories that might contain chatterbox.py
    parent_dir = os.path.dirname(current_dir)
    if parent_dir in sys.path:
        sys.path.remove(parent_dir)
    
    # Import Chatterbox after silencing output and fixing path
    try:
        # Extra safety: ensure file descriptors are still redirected
        if sys.stdout.fileno() != os.open(os.devnull, os.O_WRONLY):
            sys.stdout = open(os.devnull, 'w')
        if sys.stderr.fileno() != os.open(os.devnull, os.O_WRONLY):
            sys.stderr = open(os.devnull, 'w')
            
        from chatterbox.tts import ChatterboxTTS
        import torch
        import torchaudio
        send_response({"type": "status", "message": "Imports successful"})
    except Exception as e:
        send_response({"type": "error", "message": f"Import failed: {str(e)}", "traceback": traceback.format_exc()})
        return

    model = None
    device = "cpu"
    
    # Read commands from stdin (which is not redirected)
    stdin = os.fdopen(0, 'r')
    
    while True:
        try:
            line = stdin.readline()
            if not line:
                break
                
            request = json.loads(line.strip())
            command = request.get("command")
            
            if command == "initialize":
                # Initialize model
                device = request.get("device", "cpu")
                if device == "cuda" and not torch.cuda.is_available():
                    device = "cpu"
                    send_response({"type": "warning", "message": "CUDA not available, using CPU"})
                
                try:
                    send_response({"type": "status", "message": f"Loading model on device: {device}"})
                    
                    # Extra protection during model loading
                    # Save current FDs
                    saved_stdout = os.dup(1)
                    saved_stderr = os.dup(2)
                    
                    try:
                        # Redirect at FD level
                        devnull = os.open(os.devnull, os.O_WRONLY)
                        os.dup2(devnull, 1)
                        os.dup2(devnull, 2)
                        os.close(devnull)
                        
                        # Load model
                        model = ChatterboxTTS.from_pretrained(device=device)
                        
                    finally:
                        # Restore FDs (but not to our comm channel)
                        os.dup2(saved_stdout, 1)
                        os.dup2(saved_stderr, 2)
                        os.close(saved_stdout)
                        os.close(saved_stderr)
                    
                    send_response({"type": "success", "message": "Model loaded successfully"})
                except Exception as e:
                    import traceback
                    send_response({
                        "type": "error", 
                        "message": f"Model loading failed: {str(e)}",
                        "traceback": traceback.format_exc()
                    })
                    
            elif command == "generate":
                if model is None:
                    send_response({"type": "error", "message": "Model not initialized"})
                    continue
                
                # Extract parameters
                text = request.get("text", "")
                audio_prompt_path = request.get("audio_prompt_path")
                exaggeration = request.get("exaggeration", 0.5)
                cfg_weight = request.get("cfg_weight", 0.5)
                
                try:
                    # Generate audio
                    if audio_prompt_path:
                        wav = model.generate(
                            text,
                            audio_prompt_path=audio_prompt_path,
                            exaggeration=exaggeration,
                            cfg_weight=cfg_weight
                        )
                    else:
                        wav = model.generate(
                            text,
                            exaggeration=exaggeration,
                            cfg_weight=cfg_weight
                        )
                    
                    # Use in-memory buffer instead of temporary file
                    buffer = io.BytesIO()
                    
                    # Ensure wav is a tensor
                    if hasattr(wav, 'cpu'):
                        wav = wav.cpu()
                    
                    # Determine sample rate (default to 22050 if not specified)
                    sample_rate = 22050
                    if hasattr(model, 'sample_rate'):
                        sample_rate = model.sample_rate
                    elif hasattr(model, 'sr'):
                        sample_rate = model.sr
                    
                    # Save audio to buffer
                    torchaudio.save(buffer, wav.unsqueeze(0) if wav.dim() == 1 else wav, sample_rate, format="wav")
                    
                    # Get bytes from buffer
                    buffer.seek(0)
                    audio_bytes = buffer.read()
                    
                    # Send audio (will chunk if needed)
                    send_chunked_audio(audio_bytes, format="wav", sample_rate=sample_rate)
                        
                except Exception as e:
                    send_response({"type": "error", "message": f"Generation failed: {str(e)}", "traceback": traceback.format_exc()})
                    
            elif command == "shutdown":
                send_response({"type": "success", "message": "Shutting down"})
                break
                
            else:
                send_response({"type": "error", "message": f"Unknown command: {command}"})
                
        except json.JSONDecodeError as e:
            send_response({"type": "error", "message": f"Invalid JSON: {str(e)}"})
        except Exception as e:
            send_response({"type": "error", "message": f"Unexpected error: {str(e)}", "traceback": traceback.format_exc()})

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        send_response({"type": "fatal_error", "message": str(e), "traceback": traceback.format_exc()})
    finally:
        comm_fd.close()