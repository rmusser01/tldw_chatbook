#!/bin/bash
# Script to create a DMG installer for tldw chatbook

set -e

# Configuration
APP_NAME="tldw chatbook"
DMG_NAME="tldw-chatbook"

# Get version from command line argument or read from version.py
if [ -n "$1" ]; then
    VERSION="$1"
else
    # Try to read from version.py
    VERSION_PY="${BASH_SOURCE%/*}/../../common/version.py"
    if [ -f "$VERSION_PY" ]; then
        VERSION=$(python3 -c "exec(open('$VERSION_PY').read()); print(VERSION)")
    else
        echo "Error: No version specified and version.py not found"
        echo "Usage: $0 [VERSION]"
        exit 1
    fi
fi

VOLUME_NAME="tldw chatbook ${VERSION}"

echo "Building DMG for version: ${VERSION}"

# Paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR="${SCRIPT_DIR}/../dist"
APP_PATH="${BUILD_DIR}/${APP_NAME}.app"
DMG_DIR="${BUILD_DIR}/dmg"
DMG_PATH="${BUILD_DIR}/${DMG_NAME}-${VERSION}.dmg"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Check if app bundle exists
if [ ! -d "$APP_PATH" ]; then
    echo_error "App bundle not found at: $APP_PATH"
    echo_error "Please run build_app.py first"
    exit 1
fi

echo_info "Creating DMG for ${APP_NAME}..."

# Clean up any existing DMG directory
if [ -d "$DMG_DIR" ]; then
    echo_info "Cleaning up existing DMG directory..."
    rm -rf "$DMG_DIR"
fi

# Create DMG directory structure
echo_info "Creating DMG directory structure..."
mkdir -p "$DMG_DIR"

# Copy app bundle
echo_info "Copying app bundle..."
cp -R "$APP_PATH" "$DMG_DIR/"

# Create Applications symlink
echo_info "Creating Applications symlink..."
ln -s /Applications "$DMG_DIR/Applications"

# Create README file
echo_info "Creating README..."
cat > "$DMG_DIR/README.txt" << EOF
${APP_NAME} v${VERSION}
=======================

Installation:
1. Drag the "${APP_NAME}" icon to the Applications folder
2. Double-click the app in Applications to run

First Run:
- The app will open in Terminal.app
- On first run, you may need to right-click and select "Open" due to Gatekeeper

Web Server Mode:
To run in web browser mode, open Terminal and run:
/Applications/${APP_NAME}.app/Contents/MacOS/tldw-chatbook --serve

For more information:
https://github.com/rmusser01/tldw_chatbook

EOF

# Create a simple background image (optional)
# In production, you'd have a proper background.png
if [ -f "${SCRIPT_DIR}/../assets/dmg-background.png" ]; then
    echo_info "Background image found"
    DMG_BACKGROUND="${SCRIPT_DIR}/../assets/dmg-background.png"
else
    echo_warn "No background image found"
    DMG_BACKGROUND=""
fi

# Remove any existing DMG
if [ -f "$DMG_PATH" ]; then
    echo_info "Removing existing DMG..."
    rm -f "$DMG_PATH"
fi

# Create temporary DMG
echo_info "Creating temporary DMG..."
TEMP_DMG="${BUILD_DIR}/temp.dmg"
hdiutil create -volname "${VOLUME_NAME}" -srcfolder "$DMG_DIR" -ov -format UDRW "$TEMP_DMG"

# Mount the temporary DMG
echo_info "Mounting temporary DMG..."
MOUNT_DIR="/Volumes/${VOLUME_NAME}"
hdiutil attach "$TEMP_DMG"

# Wait for mount
sleep 2

# Set up the DMG window properties using AppleScript
echo_info "Configuring DMG window..."
osascript << EOF
tell application "Finder"
    tell disk "${VOLUME_NAME}"
        open
        set current view of container window to icon view
        set toolbar visible of container window to false
        set statusbar visible of container window to false
        set the bounds of container window to {100, 100, 650, 400}
        set viewOptions to the icon view options of container window
        set arrangement of viewOptions to not arranged
        set icon size of viewOptions to 128
        
        -- Position items
        set position of item "${APP_NAME}.app" of container window to {150, 150}
        set position of item "Applications" of container window to {400, 150}
        set position of item "README.txt" of container window to {275, 250}
        
        -- Set background if available
        -- set background picture of viewOptions to file ".background:background.png"
        
        close
        open
        update without registering applications
        delay 2
    end tell
end tell
EOF

# Unmount the temporary DMG
echo_info "Unmounting temporary DMG..."
hdiutil detach "$MOUNT_DIR"

# Convert to compressed DMG
echo_info "Creating final DMG..."
hdiutil convert "$TEMP_DMG" -format UDZO -o "$DMG_PATH"

# Clean up
echo_info "Cleaning up..."
rm -f "$TEMP_DMG"
rm -rf "$DMG_DIR"

# Sign the DMG if certificate is available
if command -v codesign &> /dev/null; then
    if security find-identity -p basic -v | grep -q "Developer ID"; then
        echo_info "Signing DMG..."
        codesign --force --sign "Developer ID Application" "$DMG_PATH"
    else
        echo_warn "No Developer ID certificate found, DMG will not be signed"
    fi
else
    echo_warn "codesign not found, DMG will not be signed"
fi

# Verify the DMG
echo_info "Verifying DMG..."
hdiutil verify "$DMG_PATH"

# Calculate size
DMG_SIZE=$(du -h "$DMG_PATH" | cut -f1)

echo_info "✅ DMG created successfully!"
echo_info "📦 Output: $DMG_PATH"
echo_info "📊 Size: $DMG_SIZE"

# Optional: Notarize the DMG
echo ""
echo_info "To notarize this DMG for distribution:"
echo ""
echo "1. First, store your credentials in a keychain profile (one-time setup):"
echo "   xcrun notarytool store-credentials \"YOUR_PROFILE_NAME\" --apple-id \"your@email.com\" --team-id \"TEAM_ID\""
echo ""
echo "2. Then submit for notarization:"
echo "   xcrun notarytool submit \"$DMG_PATH\" --keychain-profile \"YOUR_PROFILE_NAME\" --wait"
echo ""
echo "3. Once notarized, staple the ticket to the DMG:"
echo "   xcrun stapler staple \"$DMG_PATH\""
echo ""
echo "Note: altool is deprecated. Use notarytool for all new notarization workflows."