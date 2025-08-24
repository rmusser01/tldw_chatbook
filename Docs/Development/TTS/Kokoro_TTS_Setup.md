# Kokoro TTS Setup Guide

Kokoro TTS supports two backends: ONNX and PyTorch. You need to install at least one to use Kokoro.

## Option 1: ONNX Backend (Recommended)

The ONNX backend is lighter and faster. Install it with:

```bash
pip install -e ".[local_tts]"
```

Or install directly:

```bash
pip install kokoro-onnx
```

## Option 2: PyTorch Backend

For the PyTorch backend, you need:

1. Install PyTorch and dependencies:
```bash
pip install torch transformers nltk scipy
```

2. Download the PyTorch model files:
   - Model: `kokoro-v0_19.pth` from [HuggingFace](https://huggingface.co/hexgrad/kLegacy/tree/main/v0.19)
   - Voice files: `.pt` files for each voice

3. Place the files in:
   - Model: `~/.config/tldw_cli/models/kokoro/kokoro-v0_19.pth`
   - Voices: `~/.config/tldw_cli/models/kokoro/voices/*.pt`

## Switching Between Backends

You can switch between ONNX and PyTorch backends in two places:

1. **Settings Window**: S/TT/S tab → Kokoro Settings → Use ONNX toggle
2. **TTS Playground**: When Kokoro is selected → Use ONNX toggle

## Troubleshooting

If you see errors about missing models:
- For ONNX: The models will be downloaded automatically on first use
- For PyTorch: You need to manually download the model files as described above

If you see import errors:
- Make sure you've installed the required dependencies for your chosen backend
- Try running `pip install -e ".[local_tts]"` to install all TTS dependencies