#!/usr/bin/env python3
"""
Higgs Audio Installation Verification Script

This script checks if Higgs Audio TTS is properly installed and configured
for use with tldw_chatbook.
"""

import sys
import importlib
import subprocess
from pathlib import Path

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header():
    """Print script header"""
    print("\n" + "="*50)
    print("Higgs Audio Installation Verification")
    print("="*50 + "\n")

def check_python_version():
    """Check if Python version meets requirements"""
    print(f"{BLUE}Checking Python version...{RESET}")
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"{GREEN}✓ Python {version.major}.{version.minor}.{version.micro} - OK{RESET}")
        return True
    else:
        print(f"{RED}✗ Python {version.major}.{version.minor}.{version.micro} - Python 3.11+ required{RESET}")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed and can be imported"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"{GREEN}✓ {package_name} - OK{RESET}")
        return True
    except ImportError:
        print(f"{RED}✗ {package_name} - Not installed{RESET}")
        return False

def check_higgs_audio():
    """Check if Higgs Audio is properly installed"""
    print(f"\n{BLUE}Checking Higgs Audio core installation...{RESET}")
    
    # Check for boson_multimodal
    try:
        import boson_multimodal
        print(f"{GREEN}✓ boson_multimodal - OK{RESET}")
        
        # Try to import the serve engine
        try:
            from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
            print(f"{GREEN}✓ HiggsAudioServeEngine - OK{RESET}")
            return True
        except ImportError as e:
            print(f"{RED}✗ HiggsAudioServeEngine import failed: {e}{RESET}")
            return False
            
    except ImportError:
        print(f"{RED}✗ boson_multimodal - Not installed{RESET}")
        print(f"\n{YELLOW}⚠️  Higgs Audio is not installed!{RESET}")
        print("\nTo install Higgs Audio:")
        print("1. git clone https://github.com/boson-ai/higgs-audio.git")
        print("2. cd higgs-audio")
        print("3. pip install -r requirements.txt")
        print("4. pip install -e .")
        print("5. cd ..")
        print("\nOr run: ./scripts/install_higgs.sh")
        return False

def check_dependencies():
    """Check required dependencies"""
    print(f"\n{BLUE}Checking required dependencies...{RESET}")
    
    deps = [
        ("torch", "PyTorch"),
        ("torchaudio", "torchaudio"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("librosa", "librosa"),
        ("soundfile", "soundfile"),
        ("transformers", "transformers"),
    ]
    
    all_ok = True
    for import_name, display_name in deps:
        if not check_package(display_name, import_name):
            all_ok = False
    
    return all_ok

def check_cuda():
    """Check CUDA availability"""
    print(f"\n{BLUE}Checking GPU support...{RESET}")
    
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"{GREEN}✓ CUDA available - {device_name}{RESET}")
            return True
        else:
            print(f"{YELLOW}⚠️  CUDA not available - Will use CPU (slower){RESET}")
            return False
    except:
        print(f"{YELLOW}⚠️  Could not check CUDA availability{RESET}")
        return False

def check_model_download():
    """Check if the model is downloaded"""
    print(f"\n{BLUE}Checking model availability...{RESET}")
    
    try:
        from huggingface_hub import snapshot_download, repo_exists
        
        model_id = "bosonai/higgs-audio-v2-generation-3B-base"
        
        # Check if repo exists
        if repo_exists(model_id):
            print(f"{GREEN}✓ Model repository accessible{RESET}")
            
            # Check if model is cached
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            model_cache = list(cache_dir.glob(f"models--{model_id.replace('/', '--')}*"))
            
            if model_cache:
                print(f"{GREEN}✓ Model appears to be cached{RESET}")
            else:
                print(f"{YELLOW}⚠️  Model not cached - Will download on first use (~6GB){RESET}")
        else:
            print(f"{RED}✗ Model repository not found{RESET}")
            
    except ImportError:
        print(f"{YELLOW}⚠️  huggingface_hub not installed - Cannot check model{RESET}")

def check_tldw_integration():
    """Check if tldw_chatbook can use Higgs"""
    print(f"\n{BLUE}Checking tldw_chatbook integration...{RESET}")
    
    try:
        from tldw_chatbook.TTS.backends.higgs import HiggsAudioTTSBackend
        print(f"{GREEN}✓ HiggsAudioTTSBackend can be imported{RESET}")
        
        # Try to instantiate (won't load model)
        backend = HiggsAudioTTSBackend()
        print(f"{GREEN}✓ Backend can be instantiated{RESET}")
        return True
        
    except ImportError as e:
        print(f"{RED}✗ Cannot import HiggsAudioTTSBackend: {e}{RESET}")
        return False
    except Exception as e:
        print(f"{YELLOW}⚠️  Backend instantiation warning: {e}{RESET}")
        return True  # This is OK, might just be missing config

def main():
    """Run all verification checks"""
    print_header()
    
    # Track overall status
    all_checks_passed = True
    
    # Run checks
    if not check_python_version():
        all_checks_passed = False
    
    if not check_higgs_audio():
        all_checks_passed = False
        # If Higgs isn't installed, no point checking further
        print(f"\n{RED}❌ Installation verification FAILED{RESET}")
        print("\nPlease install Higgs Audio first using the instructions above.")
        sys.exit(1)
    
    if not check_dependencies():
        all_checks_passed = False
    
    check_cuda()  # Just informational
    check_model_download()  # Just informational
    
    if not check_tldw_integration():
        all_checks_passed = False
    
    # Summary
    print("\n" + "="*50)
    if all_checks_passed:
        print(f"{GREEN}✅ Installation verification PASSED!{RESET}")
        print("\nHiggs Audio TTS is ready to use with tldw_chatbook.")
        print("\nNext steps:")
        print("1. Start tldw_chatbook: python -m tldw_chatbook.app")
        print("2. Press Ctrl+T in Chat tab to configure TTS")
        print("3. Select 'higgs' as the TTS provider")
    else:
        print(f"{RED}❌ Installation verification FAILED{RESET}")
        print("\nPlease fix the issues above before using Higgs Audio TTS.")
        print("\nFor help, see: Docs/Higgs-Audio-TTS-Guide.md")
    
    print("="*50 + "\n")

if __name__ == "__main__":
    main()