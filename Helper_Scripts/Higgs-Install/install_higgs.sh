#!/bin/bash
# install_higgs.sh - Automated Higgs Audio installation script for tldw_chatbook

set -e  # Exit on error

echo "========================================"
echo "Higgs Audio TTS Installation Script"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check Python version
print_status "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
required_version="3.11"

if [[ $(echo "$python_version < $required_version" | bc) -eq 1 ]]; then
    print_error "Python $required_version or higher is required. Found: $python_version"
    exit 1
fi
print_status "Python $python_version detected ✓"

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    print_warning "Not in a virtual environment. It's recommended to use a virtual environment."
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check available memory
print_status "Checking system resources..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    total_mem=$(sysctl -n hw.memsize | awk '{print $1/1024/1024/1024}')
else
    # Linux
    total_mem=$(free -g | awk '/^Mem:/{print $2}')
fi

if (( $(echo "$total_mem < 8" | bc -l) )); then
    print_warning "Less than 8GB RAM detected. Higgs Audio may run slowly."
fi

# Step 1: Clone Higgs Audio
print_status "Cloning Higgs Audio repository..."
if [ -d "higgs-audio" ]; then
    print_warning "higgs-audio directory already exists."
    read -p "Remove and re-clone? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf higgs-audio
        git clone https://github.com/boson-ai/higgs-audio.git
    else
        print_status "Using existing higgs-audio directory"
    fi
else
    git clone https://github.com/boson-ai/higgs-audio.git
fi

# Step 2: Install Higgs Audio
print_status "Installing Higgs Audio dependencies..."
cd higgs-audio
pip install -r requirements.txt

print_status "Installing Higgs Audio..."
pip install -e .
cd ..

# Step 3: Install tldw_chatbook with Higgs support
print_status "Installing tldw_chatbook with Higgs support..."
if [ -f "pyproject.toml" ]; then
    pip install -e ".[higgs_tts]"
else
    print_error "pyproject.toml not found. Please run this script from the tldw_chatbook directory."
    exit 1
fi

# Step 4: Verify installation
print_status "Verifying installation..."
echo ""

# Test boson_multimodal import
if python3 -c "import boson_multimodal" 2>/dev/null; then
    print_status "✅ boson_multimodal imported successfully"
else
    print_error "❌ Failed to import boson_multimodal"
    exit 1
fi

# Test other dependencies
if python3 -c "import torch, torchaudio, librosa, soundfile" 2>/dev/null; then
    print_status "✅ All dependencies imported successfully"
else
    print_warning "⚠️  Some optional dependencies may be missing"
fi

# Check CUDA availability
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    print_status "✅ CUDA is available for GPU acceleration"
else
    print_status "ℹ️  CUDA not available, will use CPU (slower)"
fi

echo ""
print_status "Higgs Audio installation completed successfully! 🎉"
echo ""
echo "Next steps:"
echo "1. Start tldw_chatbook: python -m tldw_chatbook.app"
echo "2. Press Ctrl+T in Chat tab to configure TTS"
echo "3. Select 'higgs' as the TTS provider"
echo ""
echo "For more information, see: Docs/Higgs-Audio-TTS-Guide.md"