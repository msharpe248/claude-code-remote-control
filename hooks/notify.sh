#!/bin/bash
# Claude Code Hook Script
# This script receives hook events from Claude Code and forwards them to the remotecontrol server.
#
# Hook input is received via stdin as JSON containing:
#   - session_id: The current session ID
#   - hook_event_name: The event type (PreToolUse, PostToolUse, Stop, etc.)
#   - tool_name: The tool being used (for PreToolUse/PostToolUse)
#   - tool_input: The tool's input parameters
#   - tool_result: The tool's output (for PostToolUse)
#   - And other event-specific fields
#
# Usage: Configure in ~/.claude/settings.json or project .claude/settings.json

set -euo pipefail

# Configuration - adjust these as needed
REMOTECONTROL_HOST="${REMOTECONTROL_HOST:-localhost}"
REMOTECONTROL_PORT="${REMOTECONTROL_PORT:-8765}"
REMOTECONTROL_URL="http://${REMOTECONTROL_HOST}:${REMOTECONTROL_PORT}/api/hook"

# Read JSON from stdin
input=$(cat)

# Forward to remotecontrol server
# Use timeout to prevent hanging, and ignore errors (hook shouldn't block Claude)
curl -s -X POST "$REMOTECONTROL_URL" \
    -H "Content-Type: application/json" \
    -d "$input" \
    --connect-timeout 2 \
    --max-time 5 \
    2>/dev/null || true

# Always exit 0 so we don't block Claude Code
exit 0
