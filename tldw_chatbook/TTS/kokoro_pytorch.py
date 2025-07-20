# kokoro_pytorch.py
# Description: PyTorch implementation of Kokoro TTS based on remsky/Kokoro-FastAPI design
#
# This module provides the core PyTorch functionality for Kokoro TTS in a single-user context
# Implements model loading, voice handling, phonemization, and audio generation
#
# Imports
import os
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Union, Any
import re

import torch
import torch.nn.functional as F
import numpy as np

# Optional imports with fallbacks
try:
    import phonemizer
    PHONEMIZER_AVAILABLE = True
except ImportError:
    PHONEMIZER_AVAILABLE = False

logger = logging.getLogger(__name__)

#######################################################################################################################
#
# Constants and Configuration

# Default model paths
DEFAULT_MODEL_PATH = Path.home() / ".config" / "tldw_cli" / "models" / "kokoro"
SAMPLE_RATE = 24000  # Kokoro uses 24kHz audio

# Voice configuration
VOICE_PATTERN = re.compile(r'([a-z]+_[a-z]+)(?:\((\d+)\))?')  # Matches voice names like "af_bella(2)"

#######################################################################################################################
#
# Model Building and Loading

class KokoroModel:
    """Wrapper for Kokoro PyTorch model with lazy loading"""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.config = None
        
    def load(self):
        """Load the model from disk"""
        if self.model is not None:
            return
            
        logger.info(f"Loading Kokoro model from {self.model_path}")
        
        # Load model checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract model configuration
        if 'config' in checkpoint:
            self.config = checkpoint['config']
        else:
            # Default config for v0.19 models
            self.config = {
                'n_vocab': 256,
                'n_tone': 7,
                'd_model': 512,
                'n_heads': 8,
                'n_layers': 12,
                'max_len': 4096,
            }
        
        # Build model architecture
        from .kokoro_model_arch import build_kokoro_model  # We'll need to implement this
        self.model = build_kokoro_model(self.config)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded successfully on {self.device}")


def build_model(model_path: str, device: str = "cpu") -> KokoroModel:
    """
    Build and return a Kokoro model instance.
    
    Args:
        model_path: Path to the .pth model file
        device: Device to load model on ('cpu', 'cuda', etc.)
        
    Returns:
        KokoroModel instance
    """
    model = KokoroModel(model_path, device)
    model.load()
    return model


#######################################################################################################################
#
# Voice Loading and Mixing

def load_voice(voice_path: str, device: str = "cpu") -> torch.Tensor:
    """
    Load a voice tensor from a .pt file.
    
    Args:
        voice_path: Path to the voice .pt file
        device: Device to load tensor on
        
    Returns:
        Voice embedding tensor
    """
    if not os.path.exists(voice_path):
        raise FileNotFoundError(f"Voice file not found: {voice_path}")
        
    voice_tensor = torch.load(voice_path, map_location=device)
    
    # Ensure it's a tensor
    if isinstance(voice_tensor, dict):
        # Some voice files might store additional metadata
        if 'voice' in voice_tensor:
            voice_tensor = voice_tensor['voice']
        elif 'embedding' in voice_tensor:
            voice_tensor = voice_tensor['embedding']
        else:
            # Try to find the tensor in the dict
            for key, value in voice_tensor.items():
                if isinstance(value, torch.Tensor):
                    voice_tensor = value
                    break
    
    if not isinstance(voice_tensor, torch.Tensor):
        raise ValueError(f"Invalid voice file format: {voice_path}")
        
    return voice_tensor.to(device)


def parse_voice_mix(voice_string: str) -> List[Tuple[str, float]]:
    """
    Parse a voice mixing string like "af_bella(2)+af_sky(1)" into components.
    
    Args:
        voice_string: Voice specification string
        
    Returns:
        List of (voice_name, weight) tuples
    """
    voices = []
    
    # Split by + to get individual voices
    parts = voice_string.split('+')
    
    for part in parts:
        part = part.strip()
        match = VOICE_PATTERN.match(part)
        
        if match:
            voice_name = match.group(1)
            weight = float(match.group(2)) if match.group(2) else 1.0
            voices.append((voice_name, weight))
        else:
            # Simple voice name without weight
            voices.append((part, 1.0))
    
    return voices


def mix_voices(voice_tensors: List[Tuple[torch.Tensor, float]], normalize: bool = True) -> torch.Tensor:
    """
    Mix multiple voice tensors with weights.
    
    Args:
        voice_tensors: List of (tensor, weight) tuples
        normalize: Whether to normalize weights to sum to 1
        
    Returns:
        Mixed voice tensor
    """
    if not voice_tensors:
        raise ValueError("No voice tensors provided")
        
    if len(voice_tensors) == 1:
        return voice_tensors[0][0]
    
    # Extract tensors and weights
    tensors = [t for t, _ in voice_tensors]
    weights = [w for _, w in voice_tensors]
    
    # Normalize weights if requested
    if normalize:
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
    
    # Stack tensors and apply weighted sum
    stacked = torch.stack(tensors)
    weights_tensor = torch.tensor(weights, device=stacked.device).view(-1, 1)
    
    mixed = (stacked * weights_tensor).sum(dim=0)
    
    return mixed


#######################################################################################################################
#
# Text Processing and Phonemization

def phonemize_text(text: str, language: str = "en-us") -> Tuple[str, List[str]]:
    """
    Convert text to phonemes using phonemizer library.
    
    Args:
        text: Input text
        language: Language code (default: en-us)
        
    Returns:
        Tuple of (phoneme_string, phoneme_list)
    """
    if not PHONEMIZER_AVAILABLE:
        # Fallback: return original text
        logger.warning("Phonemizer not available, returning original text")
        return text, text.split()
    
    try:
        from phonemizer import phonemize
        from phonemizer.separator import Separator
        
        # Configure separator
        separator = Separator(phone=' ', word=' | ', syllable='')
        
        # Phonemize
        phonemes = phonemize(
            text,
            language=language,
            backend='espeak',
            separator=separator,
            strip=True,
            preserve_punctuation=True,
        )
        
        # Split into list
        phoneme_list = phonemes.split()
        
        return phonemes, phoneme_list
        
    except Exception as e:
        logger.error(f"Phonemization failed: {e}")
        return text, text.split()


def prepare_text_for_generation(text: str, language: str = "en") -> Dict[str, Any]:
    """
    Prepare text for generation including normalization and phonemization.
    
    Args:
        text: Input text
        language: Language code
        
    Returns:
        Dictionary with processed text data
    """
    # Basic text normalization
    text = text.strip()
    
    # Convert common abbreviations
    text = text.replace("Mr.", "Mister")
    text = text.replace("Mrs.", "Missus")
    text = text.replace("Dr.", "Doctor")
    
    # Get phonemes
    phoneme_string, phoneme_list = phonemize_text(text, f"{language}-us")
    
    return {
        'original_text': text,
        'phonemes': phoneme_string,
        'phoneme_list': phoneme_list,
        'language': language,
    }


#######################################################################################################################
#
# Audio Generation

@torch.no_grad()
def generate(
    model: KokoroModel,
    text: str,
    voice: Union[torch.Tensor, str, Dict[str, float]],
    lang: str = 'a',
    speed: float = 1.0,
    voice_dir: Optional[str] = None,
    use_phonemes: bool = True,
    **kwargs
) -> Tuple[np.ndarray, str]:
    """
    Generate audio from text using Kokoro model.
    
    This is the main generation function following the remsky/Kokoro-FastAPI design.
    
    Args:
        model: KokoroModel instance
        text: Input text to synthesize
        voice: Voice specification (tensor, name, or mix dict)
        lang: Language code ('a' for American, 'b' for British, etc.)
        speed: Speed factor (1.0 = normal)
        voice_dir: Directory containing voice files
        use_phonemes: Whether to use phonemization
        **kwargs: Additional generation parameters
        
    Returns:
        Tuple of (audio_array, phonemes_used)
    """
    # Ensure model is loaded
    if model.model is None:
        model.load()
    
    # Process voice specification
    if isinstance(voice, str):
        # Load voice from file or parse mix
        if '+' in voice:
            # Voice mixing
            voice_specs = parse_voice_mix(voice)
            voice_tensors = []
            
            for voice_name, weight in voice_specs:
                voice_path = _find_voice_file(voice_name, voice_dir)
                v_tensor = load_voice(voice_path, model.device)
                voice_tensors.append((v_tensor, weight))
            
            voice_tensor = mix_voices(voice_tensors)
        else:
            # Single voice
            voice_path = _find_voice_file(voice, voice_dir)
            voice_tensor = load_voice(voice_path, model.device)
    elif isinstance(voice, dict):
        # Dictionary of voice_name: weight
        voice_tensors = []
        for voice_name, weight in voice.items():
            voice_path = _find_voice_file(voice_name, voice_dir)
            v_tensor = load_voice(voice_path, model.device)
            voice_tensors.append((v_tensor, weight))
        voice_tensor = mix_voices(voice_tensors)
    else:
        # Assume it's already a tensor
        voice_tensor = voice
    
    # Prepare text
    text_data = prepare_text_for_generation(text, language=lang)
    
    # Use phonemes if available and requested
    if use_phonemes and text_data['phonemes']:
        input_text = text_data['phonemes']
    else:
        input_text = text_data['original_text']
    
    # Tokenize text (simplified - real implementation would use proper tokenizer)
    # For now, we'll create a dummy implementation
    tokens = _tokenize_text(input_text)
    
    # Convert to tensor
    token_tensor = torch.tensor(tokens, device=model.device).unsqueeze(0)
    
    # Generate audio
    with torch.cuda.amp.autocast(enabled=(model.device != 'cpu')):
        # This is a placeholder for the actual model forward pass
        # Real implementation would call model.generate() or similar
        audio_features = _generate_audio_features(
            model.model, 
            token_tensor, 
            voice_tensor,
            speed=speed
        )
    
    # Convert to audio waveform
    audio_array = _features_to_audio(audio_features, sample_rate=SAMPLE_RATE)
    
    # Apply speed adjustment if needed
    if speed != 1.0:
        audio_array = _adjust_speed(audio_array, speed)
    
    return audio_array, text_data['phonemes']


def _find_voice_file(voice_name: str, voice_dir: Optional[str] = None) -> str:
    """Find voice file path"""
    if voice_dir is None:
        voice_dir = DEFAULT_MODEL_PATH / "voices"
    else:
        voice_dir = Path(voice_dir)
    
    # Try different extensions
    for ext in ['.pt', '.pth', '.ckpt']:
        voice_path = voice_dir / f"{voice_name}{ext}"
        if voice_path.exists():
            return str(voice_path)
    
    # Check if full path was provided
    if os.path.exists(voice_name):
        return voice_name
    
    raise FileNotFoundError(f"Voice file not found: {voice_name}")


def _tokenize_text(text: str) -> List[int]:
    """Simplified tokenization - real implementation would use proper tokenizer"""
    # This is a placeholder implementation
    # Real implementation would use the model's tokenizer
    tokens = []
    for char in text.lower():
        if char.isalpha():
            tokens.append(ord(char) - ord('a') + 1)
        elif char == ' ':
            tokens.append(0)
        else:
            tokens.append(100)  # Special token for punctuation
    return tokens


def _generate_audio_features(model, tokens, voice, speed=1.0):
    """Placeholder for actual model generation"""
    # This would be replaced with actual model.generate() call
    # For now, return dummy features
    seq_len = tokens.shape[1]
    feature_dim = 80  # Mel spectrogram bins
    
    # Dummy implementation - real would use the model
    features = torch.randn(1, seq_len * 10, feature_dim, device=tokens.device)
    
    return features


def _features_to_audio(features, sample_rate=24000):
    """Convert model output features to audio waveform"""
    # This is a placeholder - real implementation would use vocoder
    # For now, return dummy audio
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    
    # Dummy audio generation
    duration = features.shape[1] * 256 / sample_rate  # Assuming hop size of 256
    samples = int(duration * sample_rate)
    audio = np.random.randn(samples) * 0.1  # Low volume noise as placeholder
    
    return audio


def _adjust_speed(audio: np.ndarray, speed: float) -> np.ndarray:
    """Adjust audio playback speed"""
    if speed == 1.0:
        return audio
    
    # Simple speed adjustment by resampling
    # Real implementation would use proper time-stretching
    new_length = int(len(audio) / speed)
    indices = np.linspace(0, len(audio) - 1, new_length)
    return np.interp(indices, np.arange(len(audio)), audio)


#######################################################################################################################
#
# Utility Functions

def get_available_voices(voice_dir: Optional[str] = None) -> List[str]:
    """Get list of available voice names"""
    if voice_dir is None:
        voice_dir = DEFAULT_MODEL_PATH / "voices"
    else:
        voice_dir = Path(voice_dir)
    
    if not voice_dir.exists():
        return []
    
    voices = []
    for file in voice_dir.glob("*.pt"):
        voice_name = file.stem
        voices.append(voice_name)
    
    return sorted(voices)


def validate_voice_mix(voice_string: str, available_voices: Optional[List[str]] = None) -> bool:
    """Validate a voice mix string"""
    try:
        voice_specs = parse_voice_mix(voice_string)
        
        if available_voices:
            for voice_name, _ in voice_specs:
                if voice_name not in available_voices:
                    return False
        
        return len(voice_specs) > 0
    except Exception:
        return False


#
# End of kokoro_pytorch.py
#######################################################################################################################