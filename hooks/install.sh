#!/bin/bash
# Install Claude Code hooks for remotecontrol
#
# This script helps configure Claude Code to send hook events to the remotecontrol server.
# It can install hooks either globally (~/.claude/settings.json) or locally (.claude/settings.json).
#
# Usage:
#   ./install.sh           # Interactive mode - prompts for choice
#   ./install.sh --global  # Install to ~/.claude/settings.json
#   ./install.sh --local   # Install to ./.claude/settings.json (current directory)
#   ./install.sh --help    # Show help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
NOTIFY_SCRIPT="$SCRIPT_DIR/notify.sh"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

show_help() {
    echo "Claude Code Hooks Installer"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --global    Install hooks globally to ~/.claude/settings.json"
    echo "              Affects all Claude Code sessions for this user"
    echo
    echo "  --local     Install hooks locally to ./.claude/settings.json"
    echo "              Only affects Claude Code sessions in this directory"
    echo
    echo "  --help      Show this help message"
    echo
    echo "If no option is provided, you will be prompted to choose."
    exit 0
}

# Parse command line arguments
INSTALL_MODE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --global)
            INSTALL_MODE="global"
            shift
            ;;
        --local)
            INSTALL_MODE="local"
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}Claude Code Hooks Installer${NC}"
echo "=============================="
echo

# Check if notify.sh exists and is executable
if [[ ! -x "$NOTIFY_SCRIPT" ]]; then
    echo -e "${RED}Error: $NOTIFY_SCRIPT not found or not executable${NC}"
    exit 1
fi

# Check if jq is available
if ! command -v jq &> /dev/null; then
    echo -e "${RED}Error: jq is required but not installed${NC}"
    echo "Install with: brew install jq"
    exit 1
fi

# If no mode specified, prompt user
if [[ -z "$INSTALL_MODE" ]]; then
    echo "Where would you like to install the hooks?"
    echo
    echo -e "  ${CYAN}1) Global${NC} (~/.claude/settings.json)"
    echo "     Hooks will run for ALL Claude Code sessions"
    echo
    echo -e "  ${CYAN}2) Local${NC} (./.claude/settings.json)"
    echo "     Hooks will only run in the current directory: $(pwd)"
    echo
    read -p "Enter choice [1/2]: " choice
    case $choice in
        1) INSTALL_MODE="global" ;;
        2) INSTALL_MODE="local" ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            exit 1
            ;;
    esac
    echo
fi

# Set the settings file path based on mode
if [[ "$INSTALL_MODE" == "global" ]]; then
    SETTINGS_DIR="$HOME/.claude"
    SETTINGS_FILE="$SETTINGS_DIR/settings.json"
    echo -e "Installing ${CYAN}globally${NC} to: $SETTINGS_FILE"
else
    SETTINGS_DIR="./.claude"
    SETTINGS_FILE="$SETTINGS_DIR/settings.json"
    echo -e "Installing ${CYAN}locally${NC} to: $SETTINGS_FILE"
fi
echo

# Create .claude directory if it doesn't exist
mkdir -p "$SETTINGS_DIR"

# Initialize settings file if it doesn't exist
if [[ ! -f "$SETTINGS_FILE" ]]; then
    echo "{}" > "$SETTINGS_FILE"
    echo -e "${YELLOW}Created new settings file: $SETTINGS_FILE${NC}"
fi

# Backup existing settings
BACKUP_FILE="$SETTINGS_FILE.backup.$(date +%Y%m%d_%H%M%S)"
cp "$SETTINGS_FILE" "$BACKUP_FILE"
echo -e "${GREEN}Backed up settings to: $BACKUP_FILE${NC}"

# Build the hooks configuration
HOOKS_CONFIG=$(cat <<EOF
{
  "hooks": {
    "Stop": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "$NOTIFY_SCRIPT",
            "timeout": 10
          }
        ]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "$NOTIFY_SCRIPT",
            "timeout": 10
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "$NOTIFY_SCRIPT",
            "timeout": 10
          }
        ]
      }
    ],
    "Notification": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "$NOTIFY_SCRIPT",
            "timeout": 10
          }
        ]
      }
    ],
    "SessionStart": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "$NOTIFY_SCRIPT",
            "timeout": 10
          }
        ]
      }
    ],
    "SessionEnd": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "$NOTIFY_SCRIPT",
            "timeout": 10
          }
        ]
      }
    ]
  }
}
EOF
)

# Merge hooks into existing settings
# This preserves other settings while adding/replacing hooks
MERGED=$(jq -s '.[0] * .[1]' "$SETTINGS_FILE" <(echo "$HOOKS_CONFIG"))

# Write merged settings
echo "$MERGED" > "$SETTINGS_FILE"

echo
echo -e "${GREEN}Hooks installed successfully!${NC}"
echo
echo "The following hooks are now configured:"
echo "  - Stop: Notifies when Claude finishes a response"
echo "  - PreToolUse: Notifies before tool execution (for permissions)"
echo "  - PostToolUse: Notifies after tool execution"
echo "  - Notification: Forwards Claude Code notifications"
echo "  - SessionStart: Notifies when a session starts"
echo "  - SessionEnd: Notifies when a session ends"
echo
echo "Events will be sent to: http://localhost:8765/api/hook"
echo
if [[ "$INSTALL_MODE" == "local" ]]; then
    echo -e "${YELLOW}Note: These hooks only apply when running Claude Code in:${NC}"
    echo "      $(pwd)"
    echo
fi
echo -e "${YELLOW}Make sure the remotecontrol server is running!${NC}"
echo "Start it with: python server.py"
echo
echo "View hook events at: http://localhost:8765/hooks"
