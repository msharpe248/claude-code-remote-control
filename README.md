# Claude Code Remote Control

Control Claude Code from your phone. Get push notifications with action buttons when Claude needs input, respond quickly without being at your desk.

> **Platform:** macOS only (Linux support planned)

```
┌─────────────────────────────────────────────────────────┐
│  Desktop                                                │
│  ┌─────────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │tmux/Ghostty │─▶│ Flask +  │─▶│ xterm.js         │   │
│  │  terminal   │  │ WebSocket│  │ (phone browser)  │   │
│  └──────┬──────┘  └──────────┘  └──────────────────┘   │
│         │                                               │
│         ▼                                               │
│  ┌─────────────┐       ┌─────────────┐                 │
│  │   watcher   │──────▶│ ntfy.sh     │──▶ 📱 Push     │
│  │  + parser   │       │ + actions   │  [Open][Ignore]│
│  └─────────────┘       └─────────────┘                 │
└─────────────────────────────────────────────────────────┘
```

## Features

- **Multiple Backends** - Works with tmux sessions OR Ghostty terminal (macOS)
- **Multi-Session Support** - Automatically watches ALL sessions, separate notifications per session
- **Push Notifications** - Get notified when any Claude session needs input (ntfy.sh or Pushover)
- **Notification Modes** - Active (immediate), Standby (start paused), or Log-only (testing)
- **Smart Prompt Detection** - Detects questions, confirmations, and idle states
- **Idle Prompt Actions** - Shows default text with "Send" button when Claude suggests a command
- **Session Tabs** - Switch between sessions in the web UI, badges show pending counts
- **Embedded Terminal** - Full terminal in your browser (xterm.js for tmux, polling for Ghostty)
- **Mobile-Friendly Control Panel** - Quick buttons for common responses
- **Pause Toggle** - Pause notifications from the web UI when actively working
- **Single Port** - Everything runs on one port (8765), no ttyd needed

## Quick Start

```bash
# Clone and setup
git clone https://github.com/msharpe248/claude-code-remote-control.git
cd claude-code-remote-control
cp config.yaml.example config.yaml

# Install dependencies
./install.sh

# Start the server
./start.sh

# In another terminal, run Claude
./claude-remote
```

Then open `http://<your-lan-ip>:8765/` on your phone.

## How It Works

1. **Claude Code runs in terminal** - tmux sessions OR Ghostty tabs/panes
2. **Auto-detect backend** - Server finds tmux or Ghostty automatically
3. **Watcher monitors ALL sessions** - Polls every 2 seconds, detects prompts
4. **Smart detection** - Recognizes questions ("Esc to cancel"), idle state ("? for shortcuts"), and busy state ("esc to interrupt")
5. **Per-session notifications** - Each session gets its own ntfy topic
6. **Tap to respond** - Open notification to access web UI, or tap Ignore to dismiss

## Terminal Backends

### tmux (default)
Traditional approach using tmux sessions.

```bash
# Create a tmux session
tmux new-session -d -s claude
tmux send-keys -t claude 'claude' Enter
```

### Ghostty (macOS only, no tmux needed!)
Direct access to Ghostty terminal via macOS Accessibility API.

**Requirements:**
1. Ghostty 1.2.0+ (has accessibility API support)
2. Grant accessibility permissions to Python/terminal

**Setup:**
1. Open System Settings > Privacy & Security > Accessibility
2. Add Terminal.app (or your terminal) to the allowed list
3. Run Claude Code in any Ghostty tab/pane - no tmux needed!

## Notification Actions

Notifications include two buttons:
- **[Open]** - Opens the web control panel for that session
- **[Ignore]** - Dismisses the prompt without responding

Tap the notification itself to open the web UI where you can type your response or use quick buttons.

## claude-remote wrapper

```bash
./claude-remote                      # Default "claude" session
./claude-remote -s work              # Use "work" session
./claude-remote --resume             # Pass --resume to claude
./claude-remote -s dev "fix bug"     # Session + prompt
```

## Configuration

Edit `config.yaml`:

### Terminal Backend

```yaml
terminal_backend:
  # auto: prefer Ghostty if available, fall back to tmux
  # tmux: always use tmux
  # ghostty: always use Ghostty (macOS only)
  prefer: "auto"
```

### Notification Mode

```yaml
notifications:
  # active:   Send notifications immediately when prompts detected
  # standby:  Start paused, toggle on in web UI when needed
  # log_only: Log notifications but don't send (for testing)
  mode: "active"
```

### Notifications (ntfy)

```yaml
notifications:
  ntfy:
    enabled: true
    server: "https://ntfy.sh"
    topic_prefix: "claude-yourname"  # CHANGE THIS - must be unique!
    priority: "high"
```

Topic = `{prefix}-{session}`, e.g., `claude-yourname-claude`

### Authentication

```yaml
auth:
  method: "basic"  # none | basic | token
  username: "claude"
  password: "auto-generated"
```

### Terminal Settings

```yaml
terminal:
  cols: 120       # Initial width (auto-resizes to browser)
  rows: 30        # Initial height (auto-resizes to browser)
  scrollback: 1000  # Lines of history in terminal buffer
```

## iOS Setup

1. **Install ntfy app**: [App Store Link](https://apps.apple.com/app/ntfy/id1625396347)

2. **Subscribe to topics for each session**:
   - Open ntfy → tap + → enter `{prefix}-{session-name}`
   - Example: If your prefix is `claude-yourname` and you have sessions `work` and `personal`:
     - Subscribe to `claude-yourname-work`
     - Subscribe to `claude-yourname-personal`

3. **Configure server**:
   ```yaml
   ntfy:
     topic_prefix: "claude-yourname"  # Topics will be: claude-yourname-<session>
   ```

4. **Test**: Start server → visit `/test-notify` → check phone

5. **Add to Home Screen**: Safari → Share → Add to Home Screen

## Web Interface

### Control Panel (`/`)
- **Session Tabs** - Switch between tmux sessions (appears when 2+ sessions exist)
- **Pending Badges** - Red badges show how many prompts are waiting per session
- Status indicator (waiting/working)
- Current prompt display
- **Detected Options** - Parsed numbered choices with labels
- Quick actions: 1-4, y, n, Enter, q
- Custom text input
- Link to full terminal

### Terminal (`/terminal`)
- **tmux**: Full xterm.js terminal with WebSocket, resize support
- **Ghostty**: Polling-based view with command input bar
- Auto-reconnect

### API
All endpoints accept `?session=<name>` to target a specific session.

- `GET /` - Control panel (defaults to session with pending prompts)
- `GET /terminal` - Full terminal for selected session
- `GET /history` - Prompt history
- `GET /api/status` - JSON status (includes all sessions info)
- `POST /send` - Send input `{input: "text", session: "name"}`
- `GET/POST /quick/<n>` - Quick send
- `POST /api/clear-prompts` - Clear pending prompts (used by Ignore button)
- `GET /test-notify` - Test notification preview
- `GET /config` - Current config

## Workflow

1. `./start.sh` - Start the server (watches all tmux sessions)
2. Start Claude in one or more tmux sessions
3. Step away
4. Get notification: `Claude [session-name]: needs input`
5. Tap notification to open web UI, or tap Ignore to dismiss
6. Type your response or use quick buttons
7. Claude continues

## Security

- **LAN only by default** - Bind to 0.0.0.0 for local network
- **Random credentials** - Install script generates password/token
- **ntfy topics are public** - Use unique, unguessable prefix
- For internet: add TLS (nginx/caddy) + stronger auth

## Support Tools

Diagnostic and support tools are included to help troubleshoot issues.

### cc-status
Show current status of Claude sessions, backends, and server.
```bash
./cc-status
```

### cc-diagnose
Run comprehensive diagnostics to check dependencies, permissions, config, and connectivity.
```bash
./cc-diagnose
```

### cc-test-notify
Send a test notification without starting the server.
```bash
./cc-test-notify                    # Send to default "test" session
./cc-test-notify -s work            # Send to specific session topic
./cc-test-notify -m "Hello"         # Custom message
./cc-test-notify --dry-run          # Show what would be sent
```

### cc-logs
View and filter server logs.
```bash
./cc-logs                           # Show last 100 lines
./cc-logs -f                        # Follow logs in real-time
./cc-logs -e                        # Show errors only
./cc-logs -s work                   # Filter by session
./cc-logs --prompts                 # Show prompt detections only
./cc-logs --notify                  # Show notification events only
./cc-logs -g 'pattern'              # Grep for pattern
```

## Troubleshooting

### WebSocket not connecting
- Check browser console for errors
- Ensure no firewall blocking port 8765
- Try `http://` not `https://` on local network

### Notifications not working
1. Check ntfy app is subscribed to correct topic
2. Visit `/test-notify` in browser
3. Check server logs for `[ntfy]` messages

### No tmux sessions found
The server watches all tmux sessions. Create one or more:
```bash
tmux new-session -d -s work
tmux new-session -d -s personal
```

### Can't connect from phone
1. Same WiFi network?
2. Try IP address: `http://192.168.x.x:8765/`
3. Check macOS firewall settings

## Uninstall

```bash
./uninstall.sh        # Keep config
./uninstall.sh --full # Remove everything
```

## Files

```
remotecontrol/
├── config.yaml.example  # Configuration template (copy to config.yaml)
├── config.yaml          # Your config (git-ignored, contains secrets)
├── server.py            # Flask server + WebSocket terminal + watcher
├── ghostty_reader.py    # Ghostty accessibility API helper
├── claude-remote        # Wrapper script for session naming
├── start.sh             # Startup script
├── install.sh           # Installation
├── uninstall.sh         # Cleanup
├── cc-status            # Show session/server status
├── cc-diagnose          # Run diagnostics
├── cc-test-notify       # Test notifications
├── cc-logs              # View/filter logs
├── README.md
└── venv/                # Python virtual environment
```

## License

MIT
