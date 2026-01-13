#!/bin/bash
# Claude Code Remote Control - Uninstall Script
#
# Usage:
#   ./uninstall.sh          # Stop processes, remove venv (keeps config)
#   ./uninstall.sh --full   # Remove everything including config

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

FULL_UNINSTALL=false

# Parse arguments
if [[ "$1" == "--full" ]] || [[ "$1" == "-f" ]]; then
    FULL_UNINSTALL=true
fi

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║   Claude Code Remote Control - Uninstall                  ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

if $FULL_UNINSTALL; then
    echo -e "${RED}Full uninstall: Will remove ALL files including config${NC}"
else
    echo -e "${YELLOW}Standard uninstall: Will keep config.yaml${NC}"
fi
echo ""

# Confirm
read -p "Continue? [y/N] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""

# Stop running processes
echo -e "${YELLOW}Stopping processes...${NC}"

# Kill server
if pgrep -f "python3 server.py" >/dev/null 2>&1; then
    pkill -f "python3 server.py" 2>/dev/null || true
    echo "  Stopped server"
else
    echo "  Server not running"
fi

echo -e "${GREEN}Processes stopped${NC}"
echo ""

# Remove Python virtual environment
if [ -d "venv" ]; then
    echo -e "${YELLOW}Removing Python virtual environment...${NC}"
    rm -rf venv
    echo -e "${GREEN}Removed venv/${NC}"
else
    echo "  venv/ not found"
fi
echo ""

# Remove __pycache__
if [ -d "__pycache__" ]; then
    rm -rf __pycache__
    echo "  Removed __pycache__/"
fi

# Full uninstall - remove all project files
if $FULL_UNINSTALL; then
    echo -e "${YELLOW}Removing project files...${NC}"

    # List of files to remove
    FILES=(
        "server.py"
        "start.sh"
        "install.sh"
        "config.yaml"
        "README.md"
        ".gitignore"
    )

    for f in "${FILES[@]}"; do
        if [ -f "$f" ]; then
            rm "$f"
            echo "  Removed $f"
        fi
    done

    echo ""
    echo -e "${GREEN}Full uninstall complete.${NC}"
    echo ""
    echo -e "${YELLOW}This script (uninstall.sh) remains. Remove manually:${NC}"
    echo "  rm $SCRIPT_DIR/uninstall.sh"
    echo "  rmdir $SCRIPT_DIR  # if empty"
else
    echo -e "${GREEN}Standard uninstall complete.${NC}"
    echo ""
    echo "Kept:"
    echo "  - config.yaml (your configuration)"
    echo "  - All scripts (reinstall with ./install.sh)"
    echo ""
    echo -e "${YELLOW}To fully remove everything:${NC}"
    echo "  ./uninstall.sh --full"
fi

echo ""

# Note about tmux session
echo -e "${BLUE}Note:${NC} tmux session 'claude' was NOT removed."
echo "  To kill it: tmux kill-session -t claude"
echo ""

# Note about brew packages
echo -e "${BLUE}Note:${NC} System packages (tmux) were NOT removed."
echo "  To remove: brew uninstall tmux"
echo ""
