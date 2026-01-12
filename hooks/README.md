# Claude Code Hooks for Remote Control

This directory contains hook scripts that integrate Claude Code with the remote control server.

## Overview

Claude Code hooks allow external scripts to be notified of events during Claude's operation. These scripts forward those events to the remotecontrol server, enabling:

- **Rich Question UI** - AskUserQuestion prompts show as interactive panels with clickable options
- **Permission Dialogs** - Approve or deny tool usage (Edit, Bash, Write) from your phone
- **Idle Detection** - Shows input panel when Claude finishes and is ready for new tasks
- **Push Notifications** - Get notified when Claude needs input or finishes a task
- **Hook Events Viewer** - Real-time log at `/hooks` shows all Claude activity

## Installation

### Automatic Installation

Run the install script:

```bash
./install.sh
```

You'll be prompted to choose between:
- **Global install** (`~/.claude/settings.json`) - Hooks run for all Claude Code sessions
- **Local install** (`./.claude/settings.json`) - Hooks only run in the current directory

You can also specify the mode directly:

```bash
# Install globally (all sessions)
./install.sh --global

# Install locally (current directory only)
./install.sh --local

# Show help
./install.sh --help
```

The script will:
1. Backup your existing settings file
2. Add hook configuration for all relevant events
3. Point hooks to the `notify.sh` script

### Manual Installation

Add the following to `~/.claude/settings.json` (global) or `.claude/settings.json` (local):

```json
{
  "hooks": {
    "Stop": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/remotecontrol/hooks/notify.sh",
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
            "command": "/path/to/remotecontrol/hooks/notify.sh",
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
            "command": "/path/to/remotecontrol/hooks/notify.sh",
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
            "command": "/path/to/remotecontrol/hooks/notify.sh",
            "timeout": 10
          }
        ]
      }
    ]
  }
}
```

## Hook Events

| Event | Description | UI Effect | Notification |
|-------|-------------|-----------|--------------|
| `Stop` | Claude finished responding | Shows idle input panel | Yes |
| `PreToolUse` | Before tool execution | Shows permission dialog or question panel | Yes (if permission needed) |
| `PostToolUse` | After tool execution | Hides dialogs | No |
| `Notification` | Claude Code notifications (AskUserQuestion) | Shows question panel with options | Yes |
| `SessionStart` | Session started | Logged | No |
| `SessionEnd` | Session ended | Logged | No |

## Configuration

The `notify.sh` script can be configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `REMOTECONTROL_HOST` | `localhost` | Server hostname |
| `REMOTECONTROL_PORT` | `8765` | Server port |

Example:
```bash
REMOTECONTROL_HOST=192.168.1.100 REMOTECONTROL_PORT=8765 claude
```

## Files

- `notify.sh` - Main hook script that forwards events to the server
- `install.sh` - Installation helper script
- `README.md` - This documentation

## Troubleshooting

### Hooks not firing

1. Check that hooks are configured in `~/.claude/settings.json`
2. Verify `notify.sh` is executable: `chmod +x notify.sh`
3. Test the script manually: `echo '{"hook_event_name":"test"}' | ./notify.sh`

### Events not appearing in web UI

1. Check that the remotecontrol server is running
2. Verify the server URL is correct (default: `http://localhost:8765`)
3. Check server logs for incoming requests

### Permission issues

The hook script runs with the same permissions as Claude Code. Ensure:
- `notify.sh` is executable
- Network access is available to the remotecontrol server
