#!/bin/bash

# =============================================================================
# DermOS Launcher - Jetson Nano Fullscreen Touch Interface Startup Script
# =============================================================================

# Colors for log statements
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=====================================================${NC}"
echo -e "${BLUE}      DermOS: Launching Skin Diagnostics Terminal    ${NC}"
echo -e "${BLUE}=====================================================${NC}"

# Navigate to project directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# 1. Activate Python virtual environment if it exists
if [ -d "venv" ]; then
    echo -e "${GREEN}[1/3] Activating virtual environment (venv)...${NC}"
    source venv/bin/activate
elif [ -d "../venv" ]; then
    echo -e "${GREEN}[1/3] Activating virtual environment (venv) from parent...${NC}"
    source ../venv/bin/activate
else
    echo -e "${YELLOW}[1/3] Warning: Python venv not detected. Running with system python.${NC}"
fi

# 2. Spin up Flask server in the background
echo -e "${GREEN}[2/3] Launching Flask Edge Server (inference/app.py)...${NC}"
python3 inference/app.py &
SERVER_PID=$!

# Register exit trap to clean up the Flask server when script is terminated
cleanup() {
    echo -e "\n${YELLOW}Shutting down DermOS Edge Server (PID: $SERVER_PID)...${NC}"
    kill $SERVER_PID 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Wait for server to initialize
echo -e "${YELLOW}Waiting for local webserver to start on port 5000...${NC}"
for i in {1..10}; do
    if curl -s http://localhost:5000 > /dev/null; then
        echo -e "${GREEN}✓ Local Web Server is responsive.${NC}"
        break
    fi
    sleep 1
done

# 3. Boot Chromium in Kiosk mode for Touchscreen Display
echo -e "${GREEN}[3/3] Displaying Touchscreen Dashboard in Chromium Kiosk Mode...${NC}"
echo -e "${YELLOW}Press Ctrl+C in this terminal window to exit and shut down server.${NC}"

# Disable screen saver / power management on the touchscreen display
if command -v xset &> /dev/null; then
    xset s off      # disable screen saver
    xset -dpms      # disable DPMS (Energy Star) features
    xset s noblank  # don't blank the video device
fi

# Start Chromium in fullscreen app/kiosk mode
if command -v chromium-browser &> /dev/null; then
    chromium-browser --kiosk \
                     --no-first-run \
                     --no-default-browser-check \
                     --disable-infobars \
                     --app=http://localhost:5000
elif command -v chromium &> /dev/null; then
    chromium --kiosk \
             --no-first-run \
             --no-default-browser-check \
             --disable-infobars \
             --app=http://localhost:5000
else
    echo -e "${YELLOW}Warning: Chromium browser not found. Opening default browser...${NC}"
    if command -v xdg-open &> /dev/null; then
        xdg-open http://localhost:5000
    else
        echo -e "${YELLOW}Please open browser and navigate manually to: http://localhost:5000${NC}"
    fi
    # Keep script alive to hold the background server
    wait $SERVER_PID
fi
