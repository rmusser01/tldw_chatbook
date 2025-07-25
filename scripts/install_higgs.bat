@echo off
REM install_higgs.bat - Automated Higgs Audio installation script for Windows

echo ========================================
echo Higgs Audio TTS Installation Script
echo ========================================
echo.

REM Check Python version
echo [INFO] Checking Python version...
python --version 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if we're in the tldw_chatbook directory
if not exist "pyproject.toml" (
    echo [ERROR] pyproject.toml not found. Please run this script from the tldw_chatbook directory.
    pause
    exit /b 1
)

REM Step 1: Clone Higgs Audio
echo [INFO] Cloning Higgs Audio repository...
if exist "higgs-audio" (
    echo [WARNING] higgs-audio directory already exists.
    set /p response="Remove and re-clone? (y/N): "
    if /i "%response%"=="y" (
        rmdir /s /q higgs-audio
        git clone https://github.com/boson-ai/higgs-audio.git
    ) else (
        echo [INFO] Using existing higgs-audio directory
    )
) else (
    git clone https://github.com/boson-ai/higgs-audio.git
)

if %errorlevel% neq 0 (
    echo [ERROR] Failed to clone Higgs Audio repository
    pause
    exit /b 1
)

REM Step 2: Install Higgs Audio
echo [INFO] Installing Higgs Audio dependencies...
cd higgs-audio
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install Higgs Audio requirements
    cd ..
    pause
    exit /b 1
)

echo [INFO] Installing Higgs Audio...
pip install -e .
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install Higgs Audio
    cd ..
    pause
    exit /b 1
)
cd ..

REM Step 3: Install tldw_chatbook with Higgs support
echo [INFO] Installing tldw_chatbook with Higgs support...
pip install -e ".[higgs_tts]"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install tldw_chatbook with Higgs support
    pause
    exit /b 1
)

REM Step 4: Verify installation
echo [INFO] Verifying installation...
echo.

REM Test boson_multimodal import
python -c "import boson_multimodal; print('✅ boson_multimodal imported successfully')" 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Failed to import boson_multimodal
    pause
    exit /b 1
)

REM Test other dependencies
python -c "import torch, torchaudio, librosa, soundfile; print('✅ All dependencies imported successfully')" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] Some optional dependencies may be missing
)

REM Check CUDA availability
python -c "import torch; print('✅ CUDA is available' if torch.cuda.is_available() else 'ℹ️  CUDA not available, will use CPU')"

echo.
echo [INFO] Higgs Audio installation completed successfully!
echo.
echo Next steps:
echo 1. Start tldw_chatbook: python -m tldw_chatbook.app
echo 2. Press Ctrl+T in Chat tab to configure TTS
echo 3. Select 'higgs' as the TTS provider
echo.
echo For more information, see: Docs\Higgs-Audio-TTS-Guide.md
echo.
pause