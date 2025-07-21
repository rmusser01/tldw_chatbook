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
import json
import base64
import tempfile
import traceback
from pathlib import Path

# Silence all output before importing anything
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

# Create a separate file for communication
# Use environment variable or fallback
comm_file_path = os.environ.get('COMM_FILE', '/tmp/chatterbox_comm.log')
comm_file = open(comm_file_path, 'a')

def send_response(data):
    """Send response back to parent process"""
    comm_file.write(json.dumps(data) + '\n')
    comm_file.flush()

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
        from chatterbox.tts import ChatterboxTTS
        import torch
        import torchaudio
        send_response({"type": "status", "message": "Imports successful"})
    except Exception as e:
        send_response({"type": "error", "message": f"Import failed: {str(e)}"})
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
                    model = ChatterboxTTS.from_pretrained(device=device)
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
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                        # Ensure wav is a tensor
                        if hasattr(wav, 'cpu'):
                            wav = wav.cpu()
                        
                        # Determine sample rate (default to 22050 if not specified)
                        sample_rate = 22050
                        if hasattr(model, 'sample_rate'):
                            sample_rate = model.sample_rate
                        
                        # Save audio
                        torchaudio.save(tmp.name, wav.unsqueeze(0) if wav.dim() == 1 else wav, sample_rate)
                        
                        # Read back as bytes
                        tmp.flush()
                        with open(tmp.name, 'rb') as f:
                            audio_bytes = f.read()
                        
                        # Send response with base64 encoded audio
                        send_response({
                            "type": "audio",
                            "data": base64.b64encode(audio_bytes).decode('utf-8'),
                            "format": "wav",
                            "sample_rate": sample_rate
                        })
                        
                        # Clean up
                        os.unlink(tmp.name)
                        
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
        comm_file.close()