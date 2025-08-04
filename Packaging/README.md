# Packaging Guide for tldw_chatbook

This guide covers building distributable packages for tldw_chatbook on Windows and macOS.

## Overview

The packaging system supports creating native installers and portable distributions for:
- **Windows**: Using Nuitka + NSIS (installer) or ZIP (portable)
- **macOS**: Using py2app or Nuitka + DMG installer

## Prerequisites

### Common Requirements
- Python 3.11 or later
- The main tldw_chatbook dependencies installed
- Git (for version info)

### Windows Requirements
- **Nuitka**: `pip install nuitka`
- **NSIS**: Download from https://nsis.sourceforge.io/Download
- **Visual Studio Build Tools** (for Nuitka compilation)

### macOS Requirements
- **py2app**: `pip install py2app` (recommended)
- **Nuitka**: `pip install nuitka` (alternative)
- **Xcode Command Line Tools**: `xcode-select --install`

## Build Modes

Three build modes are available:

1. **minimal**: Core features only (smallest size)
2. **standard**: Core + web server + common features (recommended)
3. **full**: All features including ML models (largest size)

## Windows Packaging

### Quick Build

```bash
cd packaging/windows
python build_windows.py --mode standard
```

This will:
1. Compile the Python code with Nuitka
2. Create an NSIS installer
3. Output: `tldw-chatbook-{version}-setup.exe`

### Build Options

```bash
# Create portable ZIP instead of installer
python build_windows.py --mode standard --portable

# Clean build artifacts
python build_windows.py --clean

# Specify NSIS path if not in PATH
python build_windows.py --nsis-path "C:\Program Files\NSIS\makensis.exe"
```

### Manual Build Steps

1. **Build executables**:
   ```bash
   python build_exe.py --mode standard
   ```

2. **Create installer** (requires NSIS):
   ```bash
   makensis installer.nsi
   ```

### Customization

- **Icon**: Place `icon.ico` in `packaging/windows/assets/`
- **Banner**: Place `banner.bmp` (164x314 pixels) in `packaging/windows/assets/`
- **License**: Automatically copied from project root

## macOS Packaging

### Quick Build

```bash
cd packaging/macos
python build_app.py --mode standard
./scripts/package_dmg.sh
```

This will:
1. Create a .app bundle
2. Package it into a DMG installer
3. Output: `tldw-chatbook-{version}.dmg`

### Build Options

```bash
# Use Nuitka instead of py2app
python build_app.py --mode standard --builder nuitka

# Different build modes
python build_app.py --mode minimal  # Smallest size
python build_app.py --mode full     # All features
```

### Code Signing

To sign the app for distribution:

1. **Get a Developer ID Certificate** from Apple Developer Program
2. **Sign the app**:
   ```bash
   codesign --deep --force --sign "Developer ID Application: Your Name" "dist/tldw chatbook.app"
   ```
3. **Notarize** (required for macOS 10.15+):
   ```bash
   xcrun altool --notarize-app \
     --primary-bundle-id "com.tldwproject.tldw-chatbook" \
     --username "your-apple-id" \
     --password "your-app-specific-password" \
     --file "tldw-chatbook-{version}.dmg"
   ```

### Customization

- **Icon**: Create `icon.icns` in `packaging/macos/assets/`
  - Convert PNG: `iconutil -c icns icon.iconset`
- **DMG Background**: Place `dmg-background.png` in `packaging/macos/assets/`

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build Packages

on:
  push:
    tags:
      - 'v*'

jobs:
  build-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -e .
          pip install nuitka
      - name: Build
        run: |
          cd packaging/windows
          python build_windows.py --mode standard
      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: windows-installer
          path: packaging/windows/*.exe

  build-macos:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -e .
          pip install py2app
      - name: Build
        run: |
          cd packaging/macos
          python build_app.py --mode standard
          ./scripts/package_dmg.sh
      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: macos-dmg
          path: packaging/macos/dist/*.dmg
```

## Version Management

Version information is centralized in `packaging/common/version.py`. Update this file to change version numbers across all packages.

## Troubleshooting

### Windows Issues

**"NSIS not found"**
- Install NSIS from https://nsis.sourceforge.io/
- Or specify path: `--nsis-path "C:\path\to\makensis.exe"`

**"Nuitka compilation failed"**
- Ensure Visual Studio Build Tools are installed
- Check Python version (3.11+ required)

**Large file size**
- Use `--mode minimal` for smaller builds
- Consider UPX compression (add to Nuitka args)

### macOS Issues

**"py2app not found"**
- Install with: `pip install py2app`
- Alternative: use `--builder nuitka`

**"App can't be opened because Apple cannot check it for malicious software"**
- Right-click the app and select "Open"
- Or sign and notarize the app

**DMG creation fails**
- Ensure you have sufficient disk space
- Check that app bundle was created successfully

## Future Enhancements

### Planned Features
- Auto-update mechanism
- Delta updates
- Linux packaging (AppImage/Snap)
- Windows MSI packages (using WiX)
- Chocolatey/Homebrew packages

### WiX Integration (Future)
For enterprise deployments, WiX will provide:
- MSI packages
- Group Policy support
- Per-machine installations
- Custom actions for complex setups

To prepare for WiX:
- Keep installer logic modular
- Document registry keys and file locations
- Plan for upgrade/migration paths

## Support

For packaging issues:
- Check the GitHub Issues
- Review build logs carefully
- Ensure all prerequisites are installed
- Try a minimal build first