#!/usr/bin/env python3
"""
Remote Control Server for Claude Code
Provides web interface, embedded terminal, and notifications for remote prompt answering.

Supports multiple terminal backends:
- tmux: Traditional tmux sessions (default, cross-platform)
- ghostty: Direct Ghostty access via macOS Accessibility API (macOS only)
"""

import os
import sys
import pty
import select
import time
import json
import yaml
import subprocess
import threading
import struct
import fcntl
import termios
import signal
import re
import hashlib
import logging
import requests
from abc import ABC, abstractmethod
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Optional, List, Dict, Any

from flask import Flask, request, Response, render_template_string, jsonify, redirect, url_for
from flask_sock import Sock

# Optional: Ghostty backend (macOS only)
try:
    from ApplicationServices import (
        AXUIElementCopyAttributeValue,
        AXUIElementCopyAttributeNames,
        AXUIElementCreateApplication,
        kAXErrorSuccess,
    )
    from Quartz import (
        CGWindowListCopyWindowInfo,
        kCGWindowListOptionOnScreenOnly,
        kCGNullWindowID,
    )
    GHOSTTY_AVAILABLE = True
except ImportError:
    GHOSTTY_AVAILABLE = False

# Load configuration
CONFIG_PATH = Path(__file__).parent / "config.yaml"
LOG_PATH = Path(__file__).parent / "server.log"

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

config = load_config()

# Setup logging
def setup_logging():
    """Setup logging to both console and file."""
    log_cfg = config.get('logging', {})
    log_to_file = log_cfg.get('file_enabled', False)
    log_level = getattr(logging, log_cfg.get('level', 'INFO').upper(), logging.INFO)

    # Create logger
    logger = logging.getLogger('remotecontrol')
    logger.setLevel(log_level)
    logger.handlers = []  # Clear existing handlers

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter('[%(name)s] %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (if enabled)
    if log_to_file:
        file_handler = logging.FileHandler(LOG_PATH)
        file_handler.setLevel(log_level)
        file_format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        print(f"[logging] Writing to {LOG_PATH}")

    return logger

log = setup_logging()

# Allow environment variable to override tmux session name
if os.environ.get('TMUX_SESSION'):
    config['tmux']['session_name'] = os.environ['TMUX_SESSION']

# ============================================================================
# Terminal Backend Abstraction
# ============================================================================

class TerminalBackend(ABC):
    """Abstract base class for terminal backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name for display."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available and usable."""
        pass

    @abstractmethod
    def get_sessions(self) -> List[str]:
        """Get list of available session/terminal names."""
        pass

    @abstractmethod
    def get_content(self, session: str) -> str:
        """Get terminal content for a session."""
        pass

    @abstractmethod
    def send_keys(self, session: str, keys: str) -> bool:
        """Send keystrokes to a session. Returns success."""
        pass

    def send_esc(self, session: str) -> bool:
        """Send Escape key to a session. Returns success."""
        return False  # Default implementation

    def supports_terminal_attach(self) -> bool:
        """Whether this backend supports interactive terminal attachment."""
        return False

    def get_attach_command(self, session: str) -> Optional[List[str]]:
        """Get command to attach to session (for WebSocket terminal)."""
        return None


class TmuxBackend(TerminalBackend):
    """tmux-based terminal backend."""

    @property
    def name(self) -> str:
        return "tmux"

    def is_available(self) -> bool:
        """Check if tmux is installed and running."""
        try:
            result = subprocess.run(
                ['tmux', 'list-sessions'],
                capture_output=True,
                timeout=5
            )
            return True  # tmux is installed, even if no sessions
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def get_sessions(self) -> List[str]:
        """Get list of all tmux session names."""
        try:
            result = subprocess.run(
                ['tmux', 'list-sessions', '-F', '#{session_name}'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return [s.strip() for s in result.stdout.strip().split('\n') if s.strip()]
            return []
        except Exception as e:
            log.error(f"[tmux] Error listing sessions: {e}")
            return []

    def get_content(self, session: str) -> str:
        """Capture current tmux pane content for a session."""
        try:
            result = subprocess.run(
                ['tmux', 'capture-pane', '-t', session, '-p', '-S', '-50'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            return ""
        except Exception as e:
            log.error(f"[tmux] Error capturing pane for {session}: {e}")
            return ""

    def send_keys(self, session: str, keys: str) -> bool:
        """Send keystrokes to tmux session."""
        try:
            if keys:
                subprocess.run(['tmux', 'send-keys', '-t', session, keys], check=True, timeout=5)
            subprocess.run(['tmux', 'send-keys', '-t', session, 'Enter'], check=True, timeout=5)
            return True
        except Exception as e:
            log.error(f"[tmux] Error sending keys to {session}: {e}")
            return False

    def send_esc(self, session: str) -> bool:
        """Send Escape key to tmux session."""
        try:
            subprocess.run(['tmux', 'send-keys', '-t', session, 'Escape'], check=True, timeout=5)
            return True
        except Exception as e:
            log.error(f"[tmux] Error sending Esc to {session}: {e}")
            return False

    def supports_terminal_attach(self) -> bool:
        return True

    def get_attach_command(self, session: str) -> Optional[List[str]]:
        return ['tmux', 'attach-session', '-t', session]


class GhosttyBackend(TerminalBackend):
    """Ghostty terminal backend using macOS Accessibility API."""

    def __init__(self):
        self._pid: Optional[int] = None
        self._app = None
        self._terminals_cache: Dict[str, str] = {}
        self._last_refresh = 0
        self._refresh_interval = 1.0  # Seconds between tree refreshes

    @property
    def name(self) -> str:
        return "ghostty"

    def _get_ax_attribute(self, element, attribute):
        """Get an accessibility attribute value."""
        err, value = AXUIElementCopyAttributeValue(element, attribute, None)
        if err == kAXErrorSuccess:
            return value
        return None

    def _get_ax_attributes(self, element):
        """Get list of available attributes on an element."""
        err, attrs = AXUIElementCopyAttributeNames(element, None)
        if err == kAXErrorSuccess:
            return list(attrs) if attrs else []
        return []

    def _find_ghostty_pid(self) -> Optional[int]:
        """Find Ghostty's process ID."""
        windows = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
        for window in windows:
            owner = window.get('kCGWindowOwnerName', '')
            pid = window.get('kCGWindowOwnerPID', 0)
            if owner == "Ghostty" and pid:
                return pid
        return None

    def _get_ghostty_app(self):
        """Get or create accessibility element for Ghostty."""
        # Check if we need to refresh
        if self._pid is None:
            self._pid = self._find_ghostty_pid()
            if self._pid is None:
                return None

        if self._app is None:
            self._app = AXUIElementCreateApplication(self._pid)
            # Verify access
            if not self._get_ax_attributes(self._app):
                self._app = None
                return None

        return self._app

    def _get_claude_processes(self) -> List[Dict[str, str]]:
        """Find running Claude processes via ps.

        Session names are determined by finding claude processes whose parent
        is claude-remote, then extracting the -s argument from claude-remote.
        """
        try:
            # Get all processes with PID, PPID, and command
            result = subprocess.run(
                ['ps', '-axo', 'pid,ppid,command'],
                capture_output=True,
                text=True,
                timeout=5
            )

            # Build maps: pid -> command, pid -> ppid
            pid_to_cmd = {}
            pid_to_ppid = {}
            claude_pids = []

            for line in result.stdout.split('\n')[1:]:  # Skip header
                parts = line.split(None, 2)  # Split into at most 3 parts
                if len(parts) >= 3:
                    pid, ppid, cmd = parts[0], parts[1], parts[2]
                    pid_to_cmd[pid] = cmd
                    pid_to_ppid[pid] = ppid

                    # Identify claude processes (not claude-remote wrapper)
                    if '/claude' in cmd or ' claude ' in cmd or cmd.endswith(' claude') or cmd.strip() == 'claude':
                        # Skip if it's the wrapper script itself
                        if 'claude-remote' in cmd:
                            continue
                        # Skip tmux commands that happen to contain 'claude'
                        if cmd.startswith('tmux '):
                            continue
                        claude_pids.append(pid)

            processes = []
            for pid in claude_pids:
                session_name = None
                ppid = pid_to_ppid.get(pid)

                # Check if parent is claude-remote and extract -s from it
                if ppid and ppid in pid_to_cmd:
                    parent_cmd = pid_to_cmd[ppid]
                    if 'claude-remote' in parent_cmd:
                        # Extract -s argument from parent command
                        match = re.search(r'\s-s\s+([a-zA-Z0-9_-]+)', parent_cmd)
                        if match:
                            session_name = match.group(1)

                processes.append({
                    'pid': pid,
                    'session': session_name,
                    'ppid': ppid,
                    'line': pid_to_cmd.get(pid, '')
                })

            return processes
        except Exception as e:
            log.error(f"[ghostty] Error getting claude processes: {e}")
            return []

    def _refresh_terminals(self):
        """Refresh the terminal content cache from Ghostty."""
        now = time.time()
        if now - self._last_refresh < self._refresh_interval:
            return

        self._last_refresh = now

        # Get running Claude processes
        claude_procs = self._get_claude_processes()
        if not claude_procs:
            self._terminals_cache = {}
            return

        # Get Ghostty terminal contents
        app = self._get_ghostty_app()
        if not app:
            self._terminals_cache = {}
            return

        terminal_contents = []
        def find_terminals(element, depth=0):
            if depth > 10:
                return
            role = self._get_ax_attribute(element, "AXRole")
            if role == "AXTextArea":
                value = self._get_ax_attribute(element, "AXValue")
                if value and len(str(value)) > 10:
                    terminal_contents.append(str(value))
                return
            children = self._get_ax_attribute(element, "AXChildren")
            if children:
                for child in children:
                    find_terminals(child, depth + 1)
        find_terminals(app)

        # Create sessions for each Claude process
        # Match to terminal content if possible, otherwise use empty content
        terminals = {}
        used_contents = set()

        log.debug(f"[ghostty] Found {len(terminal_contents)} terminal panes, {len(claude_procs)} Claude processes")

        # Sort processes so named sessions come first (they get priority for matching)
        # This ensures sessions started with `claude-remote -s <name>` get first pick
        sorted_procs = sorted(claude_procs, key=lambda p: (p['session'] is None, p['pid']))

        for proc in sorted_procs:
            session_name = proc['session'] or f"pid-{proc['pid']}"

            # Try to find matching terminal content
            matched_content = ""
            matched_idx = None
            best_score = -1

            for i, content in enumerate(terminal_contents):
                if i in used_contents:
                    continue

                recent = content[-3000:]
                deep_content = content[-15000:]
                score = 0

                # Positive signals: Claude UI patterns (high scores)
                if "? for shortcuts" in recent:
                    score += 50
                if "Esc to cancel" in recent:
                    score += 50
                if "esc to interrupt" in recent.lower():
                    score += 40
                if "↵send" in recent or "↵ send" in recent:
                    score += 50
                # Claude Code specific patterns
                if "╭─" in recent:
                    score += 20
                if "│" in recent and ("Claude" in recent or ">" in recent):
                    score += 15
                # Look deeper for Claude indicators
                if "Claude Code" in deep_content:
                    score += 30
                if "Anthropic" in deep_content:
                    score += 20
                if "claude-remote" in deep_content or "claude-code" in deep_content.lower():
                    score += 25
                if "TodoWrite" in deep_content or "AskUserQuestion" in deep_content:
                    score += 30
                # General Claude mention (lower score as it could be in any terminal)
                if "Claude" in deep_content:
                    score += 10

                # Negative signals: NOT a Claude terminal (server logs, shells, etc.)
                if "HTTP/1.1" in recent:
                    score -= 20
                if "Running on http://" in recent:
                    score -= 20
                if "[remotecontrol]" in recent:
                    score -= 20
                if "GET /" in recent or "POST /" in recent:
                    score -= 15
                if "pip install" in recent:
                    score -= 10
                if "$ " in recent[-500:] and "? for shortcuts" not in recent:
                    # Looks like a plain shell prompt, not Claude
                    score -= 5
                # Log file patterns (timestamps, log levels)
                if "[INFO]" in recent or "[DEBUG]" in recent or "[ERROR]" in recent:
                    score -= 20
                if "NEW prompt detected!" in recent:
                    score -= 20
                if "Sending notification:" in recent:
                    score -= 20
                if "[ghostty]" in recent or "[rc]" in recent:
                    score -= 15
                # Timestamp patterns like "2026-01-01 01:24:24,136"
                if re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}', recent):
                    score -= 20

                log.debug(f"[ghostty] Terminal {i} score: {score} (len={len(content)})")
                if score > best_score:
                    best_score = score
                    matched_content = content
                    matched_idx = i

            # Only use the match if it has a reasonable score
            if matched_content and matched_idx is not None:
                if best_score >= 0:
                    used_contents.add(matched_idx)
                    terminals[session_name] = matched_content
                    log.debug(f"[ghostty] Session '{session_name}' matched to Terminal {matched_idx} (score: {best_score})")
                else:
                    # Negative score means it's likely NOT a Claude terminal
                    # Don't use it - better to have no content than wrong content
                    log.debug(f"[ghostty] Rejecting match for '{session_name}' - Terminal {matched_idx} (score: {best_score})")
                    terminals[session_name] = ""

        self._terminals_cache = terminals

    def is_available(self) -> bool:
        """Check if Ghostty is available."""
        if not GHOSTTY_AVAILABLE:
            return False
        pid = self._find_ghostty_pid()
        if not pid:
            return False
        # Try to access it
        app = AXUIElementCreateApplication(pid)
        attrs = self._get_ax_attributes(app)
        return len(attrs) > 0

    def get_sessions(self) -> List[str]:
        """Get list of terminal pane names."""
        self._refresh_terminals()
        return list(self._terminals_cache.keys())

    def get_content(self, session: str) -> str:
        """Get terminal content for a session."""
        self._refresh_terminals()
        content = self._terminals_cache.get(session, "")
        log.debug(f"[ghostty] get_content('{session}'): cache keys={list(self._terminals_cache.keys())}, content_len={len(content)}")
        return content

    def send_keys(self, session: str, keys: str) -> bool:
        """
        Send keystrokes to Ghostty.

        Note: The macOS Accessibility API is read-only for Ghostty.
        We use clipboard paste for reliability (keystroke can drop characters).
        """
        try:
            # First, try to activate Ghostty with a short timeout
            activate_script = 'tell application "Ghostty" to activate'
            try:
                subprocess.run(
                    ['osascript', '-e', activate_script],
                    capture_output=True,
                    text=True,
                    timeout=3
                )
            except subprocess.TimeoutExpired:
                log.warning("[ghostty] Activate timed out, trying to send keys anyway")

            if keys:
                # Escape special characters for AppleScript string
                escaped_keys = keys.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')

                script = f'''
                tell application "Ghostty" to activate
                delay 0.2
                tell application "System Events"
                    keystroke "{escaped_keys}"
                    delay 0.1
                    key code 36
                end tell
                '''
            else:
                # Just Enter
                script = '''
                tell application "System Events"
                    tell process "ghostty"
                        key code 36
                    end tell
                end tell
                '''

            result = subprocess.run(
                ['osascript', '-e', script],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                log.error(f"[ghostty] AppleScript error: {result.stderr}")
                return False
            return True
        except subprocess.TimeoutExpired:
            log.error("[ghostty] AppleScript timed out sending keys")
            return False
        except Exception as e:
            log.error(f"[ghostty] Error sending keys: {e}")
            return False

    def send_esc(self, session: str) -> bool:
        """Send Escape key to Ghostty."""
        try:
            script = '''
            tell application "Ghostty" to activate
            delay 0.2
            tell application "System Events"
                key code 53
            end tell
            '''
            result = subprocess.run(
                ['osascript', '-e', script],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                log.error(f"[ghostty] AppleScript error: {result.stderr}")
                return False
            return True
        except subprocess.TimeoutExpired:
            log.error("[ghostty] AppleScript timed out sending Esc")
            return False
        except Exception as e:
            log.error(f"[ghostty] Error sending Esc: {e}")
            return False

    def supports_terminal_attach(self) -> bool:
        # Ghostty doesn't support attaching like tmux
        return False


# Initialize backends
_tmux_backend = TmuxBackend()
_ghostty_backend = GhosttyBackend() if GHOSTTY_AVAILABLE else None

def get_active_backend() -> TerminalBackend:
    """Get the currently active terminal backend based on config and availability."""
    backend_pref = config.get('terminal_backend', {}).get('prefer', 'auto')

    if backend_pref == 'tmux':
        if _tmux_backend.is_available():
            return _tmux_backend
        log.warning("[backend] tmux preferred but not available")

    elif backend_pref == 'ghostty':
        if _ghostty_backend and _ghostty_backend.is_available():
            return _ghostty_backend
        log.warning("[backend] ghostty preferred but not available")

    # Auto-detect: prefer Ghostty if available (no tmux needed), fall back to tmux
    if backend_pref == 'auto':
        # Check if Ghostty has Claude sessions
        if _ghostty_backend and _ghostty_backend.is_available():
            sessions = _ghostty_backend.get_sessions()
            if sessions:
                # Check if any session has Claude content
                for s in sessions:
                    content = _ghostty_backend.get_content(s)
                    if content and ("claude" in content.lower()[-2000:] or
                                   "? for shortcuts" in content[-1000:]):
                        return _ghostty_backend

        # Fall back to tmux
        if _tmux_backend.is_available():
            return _tmux_backend

    # Last resort
    if _tmux_backend.is_available():
        return _tmux_backend
    if _ghostty_backend and _ghostty_backend.is_available():
        return _ghostty_backend

    log.error("[backend] No terminal backend available!")
    return _tmux_backend  # Return tmux as fallback even if not available


# Global backend reference (updated in main)
backend: TerminalBackend = _tmux_backend


app = Flask(__name__)
sock = Sock(app)

# Per-session state
class SessionState:
    def __init__(self, session_name):
        self.session_name = session_name
        self.last_prompt = None
        self.last_prompt_signature = None  # Hash of key prompt elements
        self.last_prompt_time = None
        self.last_notification_time = 0
        self.notified_for_current = False  # Track if we've notified for current prompt
        self.pending_prompts = []
        self.prompt_history = []
        self.parsed_options = []

class GlobalState:
    def __init__(self):
        self.sessions = {}  # session_name -> SessionState
        self.active_session = None  # Currently selected session in UI

        # Notification mode from config: active, standby, log_only
        self.notification_mode = config.get('notifications', {}).get('mode', 'active')
        # Start paused if mode is "standby"
        self.notifications_paused = (self.notification_mode == 'standby')

    def get_session(self, session_name):
        """Get or create session state."""
        if session_name not in self.sessions:
            self.sessions[session_name] = SessionState(session_name)
        return self.sessions[session_name]

    def get_all_pending(self):
        """Get all pending prompts across all sessions."""
        all_pending = []
        for session_name, session_state in self.sessions.items():
            for prompt in session_state.pending_prompts:
                all_pending.append({**prompt, 'session': session_name})
        return all_pending

state = GlobalState()

# ============================================================================
# Authentication Middleware
# ============================================================================

def check_auth_basic(username, password):
    """Check if username/password is valid for basic auth."""
    return (username == config['auth']['username'] and
            password == config['auth']['password'])

def check_auth_token(token):
    """Check if token is valid."""
    return token == config['auth']['token']

def authenticate():
    """Send 401 response for basic auth."""
    return Response(
        'Authentication required', 401,
        {'WWW-Authenticate': 'Basic realm="Claude Remote Control"'}
    )

def requires_auth(f):
    """Decorator for routes that require authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_method = config['auth']['method']

        if auth_method == 'none':
            return f(*args, **kwargs)

        elif auth_method == 'basic':
            auth = request.authorization
            if not auth or not check_auth_basic(auth.username, auth.password):
                return authenticate()
            return f(*args, **kwargs)

        elif auth_method == 'token':
            token = request.args.get('token') or request.headers.get('X-Auth-Token')
            if not token or not check_auth_token(token):
                return Response('Invalid or missing token', 403)
            return f(*args, **kwargs)

        return f(*args, **kwargs)

    return decorated

def check_ws_auth():
    """Check authentication for websocket connections."""
    auth_method = config['auth']['method']

    if auth_method == 'none':
        return True
    elif auth_method == 'token':
        token = request.args.get('token')
        return token and check_auth_token(token)
    elif auth_method == 'basic':
        auth = request.authorization
        return auth and check_auth_basic(auth.username, auth.password)
    return False

# ============================================================================
# Option Parsing
# ============================================================================

def parse_numbered_options(content):
    """Parse numbered options from prompt content."""
    options = []

    # Pattern: "1. Option text" or "1) Option text" or "[1] Option text"
    # Allow for selection markers like › ❯ > * at the start
    patterns = [
        r'^[›❯>\*\s]*(\d+)\.\s+(.+?)$',      # 1. Option (with optional selection marker)
        r'^[›❯>\*\s]*(\d+)\)\s+(.+?)$',      # 1) Option
        r'^[›❯>\*\s]*\[(\d+)\]\s+(.+?)$',    # [1] Option
    ]

    lines = content.split('\n')
    for line in lines:
        for pattern in patterns:
            match = re.match(pattern, line.strip())
            if match:
                num = match.group(1)
                text = match.group(2).strip()
                # Truncate long options for display
                if len(text) > 50:
                    text = text[:47] + "..."
                options.append({'num': num, 'text': text})
                break

    return options


def parse_idle_default_text(content):
    """Parse the default text from an idle prompt input line.

    The idle prompt looks like:
        > some default text           ↵send

    Returns the text between > and ↵send (or end of line if no ↵send).
    """
    if not content:
        return None

    lines = content.strip().split('\n')

    # Look for the input line with > prompt, searching from bottom
    for line in reversed(lines):
        stripped = line.strip()
        # Match line starting with > (the input prompt)
        if stripped.startswith('>'):
            # Get everything after the >
            text = stripped[1:].strip()
            # Remove ↵send or similar send indicators if present
            text = re.sub(r'\s*↵\s*send\s*$', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\s*↵send\s*$', '', text)
            # Also remove any trailing whitespace
            text = text.strip()
            if text:
                return text
            # Found the > prompt but no text - return None
            return None

    return None

# ============================================================================
# Notification Backends
# ============================================================================

def get_local_ip():
    """Get local IP address for LAN access."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

def get_ntfy_topic(session=None):
    """Build ntfy topic from prefix + session name."""
    cfg = config['notifications']['ntfy']
    prefix = cfg.get('topic_prefix', 'claude-remote')
    # Use provided session, or fall back to config default
    if session is None:
        session = config['tmux']['session_name']
    return f"{prefix}-{session}"

def build_ntfy_actions(options, session=None):
    """Build ntfy action buttons. Max 3 allowed by ntfy."""
    base_url = f"http://{get_local_ip()}:{config['server']['port']}"
    session_param = f"?session={session}" if session else ""

    return [
        {
            "action": "view",
            "label": "Open",
            "url": f"{base_url}/{session_param}",
            "clear": True
        },
        {
            "action": "http",
            "label": "Ignore",
            "url": f"{base_url}/api/clear-prompts{session_param}",
            "method": "POST",
            "clear": True
        }
    ]

def send_ntfy(message, title=None, options=None, session=None):
    """Send notification via ntfy.sh with action buttons."""
    cfg = config['notifications']['ntfy']
    if not cfg['enabled']:
        return False

    topic = get_ntfy_topic(session)

    if title is None:
        title = f"Claude [{session}]" if session else "Claude"

    # Build actions
    actions = build_ntfy_actions(options or [], session)

    try:
        # Use JSON format to include actions
        payload = {
            "topic": topic,
            "title": title,
            "message": message,
            "priority": 4 if cfg.get('priority') == 'high' else 3,
            "tags": ["robot", "question"],
            "click": f"http://{get_local_ip()}:{config['server']['port']}/?session={session}" if session else f"http://{get_local_ip()}:{config['server']['port']}/",
            "actions": actions
        }

        resp = requests.post(
            cfg['server'],
            json=payload,
            timeout=10
        )
        if resp.status_code == 200:
            log.debug(f"[ntfy] Sent to {topic}")
            return True
        else:
            log.error(f"[ntfy] Server returned {resp.status_code}: {resp.text}")
            return False
    except Exception as e:
        log.error(f"[ntfy] Error: {e}")
        return False

def send_pushover(message, title="Claude Code Prompt"):
    """Send notification via Pushover"""
    cfg = config['notifications']['pushover']
    if not cfg['enabled']:
        return False

    if not cfg.get('user_key') or not cfg.get('api_token'):
        log.warning("[pushover] Missing credentials")
        return False

    session = config['tmux']['session_name']
    if title == "Claude Code Prompt":
        title = f"Claude [{session}]"

    try:
        requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": cfg['api_token'],
                "user": cfg['user_key'],
                "title": title,
                "message": message,
                "priority": 1,
                "url": f"http://{get_local_ip()}:{config['server']['port']}/",
                "url_title": "Open Control Panel"
            },
            timeout=10
        )
        print(f"[pushover] Notification sent")
        return True
    except Exception as e:
        log.error(f"[pushover] Error: {e}")
        return False

def send_notification(message, title=None, options=None, session=None):
    """Send notification via all enabled backends."""
    # Check if notifications are paused (user is actively using web UI)
    if state.notifications_paused:
        log.debug(f"[{session}] Notification skipped (paused)")
        return

    now = time.time()
    cooldown = config['detection']['notification_cooldown']

    # Per-session cooldown as safety net (main dedup is via signature)
    if session:
        session_state = state.get_session(session)
        if now - session_state.last_notification_time < cooldown:
            remaining = cooldown - (now - session_state.last_notification_time)
            log.debug(f"[{session}] Notification skipped (cooldown: {remaining:.0f}s)")
            return
        session_state.last_notification_time = now

    if title is None:
        title = f"Claude [{session}]" if session else "Claude"

    # log_only mode: log but don't send notifications
    if state.notification_mode == 'log_only':
        log.info(f"[{session}] LOG ONLY - Would send: {title}")
        log.info(f"[{session}] LOG ONLY - Message:\n{message}")
        return

    log.info(f"[{session}] Sending notification: {title}")
    send_ntfy(message, title, options, session)
    send_pushover(message, title)

# ============================================================================
# Terminal Watcher
# ============================================================================

def extract_prompt_signature(lines, prompt_type):
    """Extract a stable signature from prompt content.

    The signature should identify THIS specific prompt, ignoring
    things that change like cursor position or timestamps.
    """
    # For idle prompts, use a fixed signature since all idle states are equivalent
    # (Claude is ready for input - the surrounding terminal content doesn't matter)
    if prompt_type == 'idle':
        return "idle_ready"

    # For question prompts, extract the actual question content
    # Find the key identifying lines - the question/prompt text
    key_lines = []
    for line in lines:
        line_stripped = line.strip()
        # Skip empty lines and navigation hints
        if not line_stripped:
            continue
        if line_stripped in ('Esc to cancel', '? for shortcuts'):
            continue
        if 'Enter to select' in line_stripped:
            continue
        if 'Tab/Arrow keys' in line_stripped:
            continue
        # Skip lines that are just UI chrome
        if line_stripped.startswith('╭') or line_stripped.startswith('╰'):
            continue
        if line_stripped.startswith('│') and len(line_stripped) < 5:
            continue
        # Include numbered options and question text
        key_lines.append(line_stripped)

    # Create signature from type + key content
    sig_content = f"{prompt_type}:" + "|".join(key_lines[-6:])  # Last 6 key lines
    return hashlib.md5(sig_content.encode()).hexdigest()[:12]


def detect_prompt(content):
    """Check if content indicates Claude is waiting for input.

    Returns: (is_prompt, context, prompt_type, signature)
        prompt_type: 'question' (has options), 'idle' (main input), or None
        signature: stable hash identifying this specific prompt
    """
    if not content:
        return False, None, None, None

    lines = content.strip().split('\n')
    if not lines:
        return False, None, None, None

    recent_lines = [l for l in lines[-15:] if l.strip()]
    if not recent_lines:
        return False, None, None, None

    # Check if Claude is busy (spinner active) - if so, ignore everything
    # The spinner line contains "esc to interrupt" or similar
    for line in recent_lines[-3:]:  # Check last few lines for spinner
        line_lower = line.lower()
        if "esc to interrupt" in line_lower or "to interrupt" in line_lower:
            return False, None, None, None

    recent_text = '\n'.join(recent_lines[-8:])

    # Check for any prompt with "Esc to cancel" - covers all question types
    # (AskUserQuestion, edit confirmations, etc.)
    if "Esc to cancel" in recent_text:
        context = recent_text
        signature = extract_prompt_signature(recent_lines[-8:], 'question')
        return True, context, 'question', signature

    # Check for main input prompt (idle, waiting for next instruction)
    # Has "↵send" or "? for shortcuts" visible
    if "send" in recent_text and "? for shortcuts" in recent_text:
        context = recent_text
        signature = extract_prompt_signature(recent_lines[-8:], 'idle')
        return True, context, 'idle', signature

    return False, None, None, None

def watcher_loop():
    """Background thread that watches all terminal sessions for prompts."""
    global backend
    backend = get_active_backend()
    log.info(f"Starting watcher using {backend.name} backend...")
    mode_desc = {
        'active': 'active (sending immediately)',
        'standby': 'standby (paused, toggle in UI)',
        'log_only': 'log only (not sending)'
    }
    log.info(f"Notification mode: {mode_desc.get(state.notification_mode, state.notification_mode)}")
    interval = config['detection']['poll_interval']
    known_sessions = set()

    while True:
        try:
            # Re-check backend periodically (in case Ghostty starts/stops)
            backend = get_active_backend()

            # Get all active sessions from current backend
            sessions = backend.get_sessions()

            # Log new sessions
            for s in sessions:
                if s not in known_sessions:
                    log.info(f"[{backend.name}] Found session: {s}")
                    known_sessions.add(s)

            # Log removed sessions
            for s in list(known_sessions):
                if s not in sessions:
                    log.info(f"[{backend.name}] Session ended: {s}")
                    known_sessions.discard(s)

            for session_name in sessions:
                content = backend.get_content(session_name)
                is_prompt, context, prompt_type, signature = detect_prompt(content)

                session_state = state.get_session(session_name)

                if is_prompt:
                    # Check if this is a NEW prompt (different signature)
                    # For question prompts: only treat as new if there was no active prompt
                    # This prevents re-detection when user is typing their answer
                    if prompt_type == 'question' and session_state.last_prompt_signature is not None:
                        # Already have an active question prompt - user is probably typing
                        # Just update context but don't treat as new
                        session_state.last_prompt = context
                        is_new_prompt = False
                    else:
                        is_new_prompt = (signature != session_state.last_prompt_signature)

                    if is_new_prompt:
                        # New prompt detected
                        session_state.last_prompt = context
                        session_state.last_prompt_signature = signature
                        session_state.last_prompt_time = datetime.now()
                        session_state.notified_for_current = False  # Reset notification flag

                        # Parse numbered options (only relevant for question prompts)
                        # Use context (recent prompt area) not full content (terminal history)
                        idle_default_text = None
                        if prompt_type == 'question':
                            session_state.parsed_options = parse_numbered_options(context)
                        elif prompt_type == 'idle':
                            session_state.parsed_options = []
                            # Extract default text from idle prompt line (e.g., "> ok, restarted")
                            idle_default_text = parse_idle_default_text(context)
                        else:
                            session_state.parsed_options = []

                        prompt_info = {
                            'context': context,
                            'time': session_state.last_prompt_time.isoformat(),
                            'id': len(session_state.prompt_history),
                            'options': session_state.parsed_options,
                            'type': prompt_type,
                            'session': session_name,
                            'signature': signature,
                            'idle_default': idle_default_text
                        }

                        session_state.pending_prompts.append(prompt_info)
                        session_state.prompt_history.append(prompt_info)

                        if len(session_state.prompt_history) > 100:
                            session_state.prompt_history = session_state.prompt_history[-100:]

                        log.info(f"[{session_name}] NEW prompt detected! Type: {prompt_type}, Sig: {signature}")
                        log.debug(f"[{session_name}] Context:\n{context}")

                    # Only notify if we haven't already for this prompt
                    if not session_state.notified_for_current:
                        session_state.notified_for_current = True

                        if prompt_type == 'idle':
                            send_notification(
                                "Claude is ready for your next instruction.",
                                title=f"Claude Ready [{session_name}]",
                                options=[],
                                session=session_name
                            )
                        else:
                            send_notification(
                                f"Claude needs your input:\n\n{context[:200]}",
                                options=session_state.parsed_options,
                                session=session_name
                            )
                else:
                    # No prompt detected - reset state so we can detect next prompt
                    if session_state.last_prompt_signature is not None:
                        log.info(f"[{session_name}] Prompt answered/cleared (was: {session_state.last_prompt_signature})")
                        session_state.last_prompt_signature = None
                        session_state.notified_for_current = False

            time.sleep(interval)

        except Exception as e:
            log.error(f"Watcher error: {e}")
            time.sleep(interval)

# ============================================================================
# Terminal WebSocket
# ============================================================================

@sock.route('/ws/terminal')
def terminal_websocket(ws):
    """WebSocket endpoint for terminal access."""
    if not check_ws_auth():
        ws.close(1008, "Unauthorized")
        return

    session = request.args.get('session') or state.active_session or config['tmux']['session_name']

    # For backends that don't support PTY attachment (like Ghostty),
    # use a polling approach: read content periodically and send input via send_keys
    if not backend.supports_terminal_attach():
        ws.send(f"\r\n  [{backend.name}] Polling mode - content updates every 1s\r\n")
        ws.send(f"  Type commands below and press Enter to send.\r\n\r\n")

        last_content = ""
        try:
            while True:
                # Check for user input (non-blocking with short timeout)
                try:
                    msg = ws.receive(timeout=0.5)
                    if msg is None:
                        break
                    # Handle user input
                    try:
                        data = json.loads(msg)
                        if data.get('type') == 'input':
                            # Send the input to the terminal
                            backend.send_keys(session, data.get('data', ''))
                    except json.JSONDecodeError:
                        # Plain text input - send directly
                        backend.send_keys(session, msg)
                except:
                    pass  # Timeout, continue to poll content

                # Poll terminal content
                content = backend.get_content(session)
                if content and content != last_content:
                    # Clear screen and write new content
                    ws.send("\x1b[2J\x1b[H")  # ANSI clear screen + cursor home
                    ws.send(content)
                    last_content = content

                time.sleep(0.5)
        except Exception as e:
            log.debug(f"[terminal-ws] Polling loop ended: {e}")
        return

    # For tmux: use PTY attachment
    attach_cmd = backend.get_attach_command(session)

    if not attach_cmd:
        ws.send(f"\r\n  Cannot attach to session: {session}\r\n")
        return

    # Create a PTY and run the attach command
    master_fd, slave_fd = pty.openpty()

    # Set terminal size
    cols = config.get('terminal', {}).get('cols', 120)
    rows = config.get('terminal', {}).get('rows', 30)
    winsize = struct.pack('HHHH', rows, cols, 0, 0)
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)

    # Fork process
    pid = os.fork()

    if pid == 0:
        # Child process
        os.close(master_fd)
        os.setsid()
        os.dup2(slave_fd, 0)
        os.dup2(slave_fd, 1)
        os.dup2(slave_fd, 2)
        os.close(slave_fd)
        os.execvp(attach_cmd[0], attach_cmd)
    else:
        # Parent process
        os.close(slave_fd)

        # Set non-blocking
        flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
        fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        try:
            while True:
                # Check for data from terminal
                r, _, _ = select.select([master_fd], [], [], 0.1)
                if master_fd in r:
                    try:
                        data = os.read(master_fd, 1024)
                        if data:
                            ws.send(data)
                    except OSError:
                        break

                # Check for data from websocket
                try:
                    message = ws.receive(timeout=0.01)
                    if message is None:
                        break

                    if isinstance(message, str):
                        # Check for resize message
                        if message.startswith('{"type":"resize"'):
                            try:
                                msg = json.loads(message)
                                if msg.get('type') == 'resize':
                                    new_cols = msg.get('cols', cols)
                                    new_rows = msg.get('rows', rows)
                                    winsize = struct.pack('HHHH', new_rows, new_cols, 0, 0)
                                    fcntl.ioctl(master_fd, termios.TIOCSWINSZ, winsize)
                            except:
                                pass
                        else:
                            os.write(master_fd, message.encode())
                    else:
                        os.write(master_fd, message)
                except:
                    pass

        finally:
            os.close(master_fd)
            os.kill(pid, signal.SIGTERM)
            os.waitpid(pid, 0)

# ============================================================================
# Web Interface
# ============================================================================

TEST_NOTIFY_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Test Notification - Claude Remote Control</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
            padding: 1rem;
        }
        .container { max-width: 600px; margin: 0 auto; }
        h1 { font-size: 1.5rem; margin-bottom: 1rem; color: #ff6b6b; }
        h3 { margin: 1rem 0 0.5rem; color: #ccc; }

        .info-box {
            background: #0d0d1a;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            border: 1px solid #333;
        }
        .info-row {
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px solid #333;
        }
        .info-row:last-child { border-bottom: none; }
        .info-label { color: #888; }
        .info-value { color: #60a5fa; font-family: monospace; }

        .preview-box {
            background: #2d3a4a;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            border-left: 4px solid #6366f1;
        }
        .preview-title {
            font-weight: bold;
            margin-bottom: 0.5rem;
            color: #fff;
        }
        .preview-message {
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
            font-size: 0.85rem;
            white-space: pre-wrap;
            color: #ccc;
        }
        .preview-options {
            margin-top: 1rem;
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }
        .preview-option {
            background: #4b5563;
            padding: 0.25rem 0.75rem;
            border-radius: 4px;
            font-size: 0.8rem;
        }

        .actions { display: flex; gap: 0.5rem; margin-top: 1.5rem; }
        .btn {
            flex: 1;
            padding: 1rem;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            cursor: pointer;
            text-decoration: none;
            text-align: center;
            transition: transform 0.1s, opacity 0.1s;
        }
        .btn:active { transform: scale(0.97); opacity: 0.9; }
        .btn-primary { background: #6366f1; color: white; }
        .btn-secondary { background: #4b5563; color: white; }

        .hint {
            margin-top: 1rem;
            padding: 1rem;
            background: #2d4a3e;
            border-radius: 8px;
            font-size: 0.85rem;
            color: #4ade80;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Test Notification Preview</h1>

        <h3>Destination</h3>
        <div class="info-box">
            <div class="info-row">
                <span class="info-label">Server</span>
                <span class="info-value">{{ server }}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Topic</span>
                <span class="info-value">{{ topic }}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Full URL</span>
                <span class="info-value">{{ server }}/{{ topic }}</span>
            </div>
        </div>

        <h3>Notification Preview</h3>
        <div class="preview-box">
            <div class="preview-title">{{ title }}</div>
            <div class="preview-message">{{ message }}</div>
            <div class="preview-options">
                {% for opt in options %}
                <span class="preview-option">{{ opt.num }}: {{ opt.text }}</span>
                {% endfor %}
            </div>
        </div>

        <div class="hint">
            <strong>Tip:</strong> Open the ntfy app on your phone and subscribe to:<br>
            <code>{{ topic }}</code><br>
            Then click "Send Test Notification" below.
        </div>

        <div class="actions">
            <a href="/" class="btn btn-secondary">Cancel</a>
            <form method="POST" style="flex: 1; display: flex;">
                <button type="submit" class="btn btn-primary" style="flex: 1;">Send Test Notification</button>
            </form>
        </div>
    </div>
</body>
</html>
"""

MAIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Claude Remote Control</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
            padding: 1rem;
        }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { font-size: 1.5rem; margin-bottom: 1rem; color: #ff6b6b; }
        h3 { margin: 1rem 0 0.5rem; color: #ccc; }

        .status { padding: 1rem; border-radius: 8px; margin-bottom: 1rem; }
        .status.waiting { background: #2d4a3e; border-left: 4px solid #4ade80; }
        .status.idle { background: #2d3a4a; border-left: 4px solid #60a5fa; }

        .prompt-box {
            background: #0d0d1a;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
            font-size: 0.85rem;
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 250px;
            overflow-y: auto;
            border: 1px solid #333;
        }

        .actions { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; margin-bottom: 1rem; }
        .actions.two-col { grid-template-columns: repeat(2, 1fr); }

        .btn {
            padding: 0.875rem 0.5rem;
            border: none;
            border-radius: 8px;
            font-size: 0.95rem;
            cursor: pointer;
            text-decoration: none;
            text-align: center;
            transition: transform 0.1s, opacity 0.1s;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .btn:active { transform: scale(0.97); opacity: 0.9; }
        .btn-primary { background: #6366f1; color: white; }
        .btn-success { background: #22c55e; color: white; }
        .btn-warning { background: #f59e0b; color: black; }
        .btn-danger { background: #ef4444; color: white; }
        .btn-secondary { background: #4b5563; color: white; }
        .btn-full { grid-column: 1 / -1; }
        .btn-num { font-size: 1.25rem; font-weight: bold; }
        .btn-label { font-size: 0.7rem; opacity: 0.8; margin-top: 2px; }

        .input-group { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
        .input-group input {
            flex: 1;
            padding: 0.875rem;
            border: 1px solid #333;
            border-radius: 8px;
            background: #0d0d1a;
            color: #eee;
            font-size: 1rem;
        }

        .links { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem; }
        .links a { color: #60a5fa; padding: 0.5rem; font-size: 0.9rem; }

        .meta { font-size: 0.75rem; color: #666; }

        #refresh-indicator {
            position: fixed; top: 1rem; right: 1rem;
            width: 10px; height: 10px; border-radius: 50%;
            background: #22c55e; animation: pulse 2s infinite;
        }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

        .option-btn {
            background: #374151;
            color: white;
            flex-direction: row;
            gap: 0.5rem;
            padding: 0.75rem;
            text-align: left;
            justify-content: flex-start;
        }
        .option-btn .btn-num {
            background: #6366f1;
            border-radius: 4px;
            padding: 0.25rem 0.5rem;
            font-size: 0.9rem;
        }
        .option-btn .btn-label {
            font-size: 0.85rem;
            opacity: 1;
            margin: 0;
        }

        .session-tabs {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }
        .session-tab {
            padding: 0.5rem 1rem;
            border-radius: 8px;
            background: #374151;
            color: #ccc;
            text-decoration: none;
            font-size: 0.85rem;
            position: relative;
        }
        .session-tab.active {
            background: #6366f1;
            color: white;
        }
        .session-tab .badge {
            position: absolute;
            top: -5px;
            right: -5px;
            background: #ef4444;
            color: white;
            border-radius: 50%;
            width: 18px;
            height: 18px;
            font-size: 0.7rem;
            display: flex;
            align-items: center;
            justify-content: center;
        }
    </style>
</head>
<body>
    <div id="refresh-indicator" title="Auto-refreshing"></div>
    <div class="container">
        <h1>Claude Remote Control</h1>

        {% if sessions|length > 1 %}
        <div class="session-tabs">
            {% for s in sessions %}
            <a href="/?session={{ s.name }}" class="session-tab {{ 'active' if s.active else '' }}">
                {{ s.name }}
                {% if s.waiting %}<span class="badge">!</span>{% endif %}
            </a>
            {% endfor %}
        </div>
        {% endif %}

        <div id="status-area">
            {% if pending_prompts %}
            <div class="status waiting">
                <strong>Waiting for input!</strong>
            </div>
            {% else %}
            <div class="status idle">
                <strong>Claude is working...</strong>
            </div>
            {% endif %}
        </div>

        {% if pending_prompts %}
        <div class="prompt-box">{{ pending_prompts[-1].context }}</div>
        {% else %}
        <div class="prompt-box">{{ last_content or 'No recent activity' }}</div>
        {% endif %}

        {% if options %}
        <h3>Detected Options</h3>
        <div class="actions two-col">
            {% for opt in options %}
            <button class="btn option-btn" onclick="handleOption('{{ opt.num }}', '{{ opt.text | e }}')">
                <span class="btn-num">{{ opt.num }}</span>
                <span class="btn-label">{{ opt.text }}</span>
            </button>
            {% endfor %}
        </div>
        {% endif %}

        {% if pending_prompts and pending_prompts[-1].type == 'idle' %}
        <h3>Quick Actions</h3>
        <div class="actions">
            {% if pending_prompts[-1].idle_default %}
            <button class="btn btn-primary" onclick="sendInput('')" style="flex: 2;">
                <span class="btn-num">↵</span>
                <span class="btn-label">Send: {{ pending_prompts[-1].idle_default[:30] }}{% if pending_prompts[-1].idle_default|length > 30 %}...{% endif %}</span>
            </button>
            {% endif %}
            <button class="btn btn-secondary" onclick="promptAndSendIdle()">
                <span class="btn-num">✎</span>
                <span class="btn-label">Type something</span>
            </button>
        </div>
        {% else %}
        <h3>Quick Actions</h3>
        {% endif %}
        <div class="actions">
            <button class="btn btn-danger" onclick="sendEsc()">
                <span class="btn-num">⎋</span>
                <span class="btn-label">Cancel</span>
            </button>
        </div>

        <div class="input-group">
            <input type="text" id="custom-input" placeholder="Custom response...">
            <button class="btn btn-primary" onclick="sendCustom()">Send</button>
        </div>

        <div class="actions two-col">
            <a href="/terminal?session={{ session }}" class="btn btn-primary btn-full">Open Full Terminal</a>
        </div>

        <div class="links">
            {% if notification_mode == 'log_only' %}
            <span style="color: #666;">📝 Log Only Mode</span>
            {% else %}
            <a href="#" id="notify-toggle" onclick="toggleNotifications(); return false;">
                {% if notifications_paused %}🔕 Notifications Paused{% else %}🔔 Notifications On{% endif %}
            </a>
            {% endif %}
            <a href="/history">History</a>
            <a href="/test-notify?session={{ session }}">Test</a>
            <a href="/config">Config</a>
        </div>

        <div class="meta">
            Session: {{ session }} | Last: {{ now }} | {{ local_ip }}:{{ port }}
        </div>
    </div>

    <script>
        const currentSession = '{{ session }}';
        let userInteracting = false;  // Track if user is interacting (pauses auto-refresh)

        function sendInput(text) {
            fetch('/send', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({input: text, session: currentSession})
            }).then(r => r.json()).then(d => {
                if(d.success) location.reload();
                else alert('Error: ' + d.error);
            });
        }

        function handleOption(num, text) {
            // Check if this is a "type something" / "other" option
            const lowerText = text.toLowerCase();
            if (lowerText.includes('type something') || lowerText.includes('other')) {
                userInteracting = true;  // Pause auto-refresh
                const customText = prompt('Enter your custom response:');
                if (customText !== null && customText.trim() !== '') {
                    // Prepend option number to custom text (e.g., "3hello world")
                    const fullText = String(num) + customText;
                    fetch('/send', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({input: fullText, session: currentSession})
                    }).then(r => r.json()).then(d => {
                        userInteracting = false;
                        if(d.success) location.reload();
                        else alert('Error: ' + d.error);
                    }).catch(e => {
                        userInteracting = false;
                        alert('Network error: ' + e);
                    });
                } else {
                    userInteracting = false;
                }
            } else {
                // Normal option - just send the number
                sendInput(num);
            }
        }

        function sendCustom() {
            const input = document.getElementById('custom-input').value;
            sendInput(input);
        }

        function sendEsc() {
            fetch('/send-esc', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({session: currentSession})
            }).then(r => r.json()).then(d => {
                if(d.success) location.reload();
                else alert('Error: ' + d.error);
            });
        }

        function promptAndSendIdle() {
            userInteracting = true;
            const customText = prompt('Enter your message to Claude:');
            if (customText !== null && customText.trim() !== '') {
                fetch('/send', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({input: customText, session: currentSession})
                }).then(r => r.json()).then(d => {
                    userInteracting = false;
                    if(d.success) location.reload();
                    else alert('Error: ' + d.error);
                }).catch(e => {
                    userInteracting = false;
                    alert('Network error: ' + e);
                });
            } else {
                userInteracting = false;
            }
        }

        document.getElementById('custom-input').addEventListener('keypress', e => {
            if(e.key === 'Enter') sendCustom();
        });

        // Smart auto-refresh: pause while user is typing or interacting
        const customInput = document.getElementById('custom-input');

        customInput.addEventListener('focus', () => { userInteracting = true; });
        customInput.addEventListener('blur', () => { userInteracting = false; });

        function toggleNotifications() {
            fetch('/toggle-notifications', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            }).then(r => r.json()).then(d => {
                if(d.success) {
                    const toggle = document.getElementById('notify-toggle');
                    if(d.paused) {
                        toggle.innerHTML = '🔕 Notifications Paused';
                    } else {
                        toggle.innerHTML = '🔔 Notifications On';
                    }
                } else {
                    alert('Error toggling notifications');
                }
            });
        }

        function scheduleRefresh() {
            setTimeout(() => {
                if (!userInteracting && document.visibilityState === 'visible') {
                    location.reload();
                } else {
                    // User is typing or page not visible, check again later
                    scheduleRefresh();
                }
            }, 5000);
        }
        scheduleRefresh();
    </script>
</body>
</html>
"""

TERMINAL_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Terminal - {{ session }}</title>
    {% if not polling_mode %}<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/xterm@5.3.0/css/xterm.css">{% endif %}
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #0d0d1a;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .header {
            background: #1a1a2e;
            padding: 0.5rem 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #333;
        }
        .header h1 { font-size: 1rem; color: #ff6b6b; }
        .header a { color: #60a5fa; text-decoration: none; font-size: 0.9rem; }
        #terminal-container {
            flex: 1;
            padding: 0.5rem;
            overflow: hidden;
        }
        #terminal { height: 100%; }
        #terminal-output {
            height: 100%;
            overflow-y: auto;
            background: #0d0d1a;
            color: #eee;
            font-family: SF Mono, Monaco, Inconsolata, monospace;
            font-size: 12px;
            line-height: 1.4;
            white-space: pre-wrap;
            word-wrap: break-word;
            padding: 0.5rem;
        }
        .status-bar {
            background: #1a1a2e;
            padding: 0.25rem 1rem;
            font-size: 0.75rem;
            color: #666;
            border-top: 1px solid #333;
        }
        .connected { color: #4ade80; }
        .disconnected { color: #ef4444; }
        .command-bar {
            display: flex;
            gap: 0.5rem;
            padding: 0.5rem;
            background: #1a1a2e;
            border-top: 1px solid #333;
        }
        .command-bar input {
            flex: 1;
            padding: 0.5rem;
            background: #0d0d1a;
            border: 1px solid #444;
            border-radius: 4px;
            color: #eee;
            font-family: SF Mono, Monaco, monospace;
            font-size: 14px;
        }
        .command-bar button {
            padding: 0.5rem 1rem;
            background: #6366f1;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Terminal: {{ session }}{% if polling_mode %} ({{ backend_name }}){% endif %}</h1>
        <a href="/">← Back</a>
    </div>
    <div id="terminal-container">
        {% if polling_mode %}
        <pre id="terminal-output">Loading...</pre>
        {% else %}
        <div id="terminal"></div>
        {% endif %}
    </div>
    {% if polling_mode %}
    <div class="command-bar">
        <input type="text" id="cmd-input" placeholder="Type command and press Enter...">
        <button id="cmd-send">Send</button>
    </div>
    {% endif %}
    <div class="status-bar">
        Status: <span id="ws-status" class="disconnected">Connecting...</span>
        {% if polling_mode %} | Polling mode (read-only view + command input){% endif %}
    </div>

    {% if polling_mode %}
    <script>
        // Polling mode for Ghostty - simple fetch-based content updates
        const output = document.getElementById('terminal-output');
        const statusEl = document.getElementById('ws-status');

        function stripAnsi(str) {
            // Remove ANSI escape codes for cleaner display
            return str.replace(/\x1b\[[0-9;]*[a-zA-Z]/g, '');
        }

        function pollContent() {
            fetch('/api/content?session={{ session }}')
                .then(r => r.json())
                .then(d => {
                    if (d.content) {
                        output.textContent = stripAnsi(d.content);
                        // Auto-scroll to bottom
                        output.scrollTop = output.scrollHeight;
                        statusEl.textContent = 'Connected';
                        statusEl.className = 'connected';
                    }
                })
                .catch(e => {
                    statusEl.textContent = 'Error polling';
                    statusEl.className = 'disconnected';
                });
        }

        // Poll every second
        pollContent();
        setInterval(pollContent, 1000);

        // Command bar
        const cmdInput = document.getElementById('cmd-input');
        const cmdSend = document.getElementById('cmd-send');

        function sendCommand() {
            const cmd = cmdInput.value;
            if (!cmd) return;
            console.log('Sending command:', cmd);
            fetch('/send', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({input: cmd, session: '{{ session }}'})
            }).then(r => {
                console.log('Response status:', r.status);
                return r.json();
            }).then(d => {
                console.log('Response data:', d);
                if (d.success) {
                    cmdInput.value = '';
                    setTimeout(pollContent, 500);
                } else {
                    alert('Error: ' + d.error);
                }
            }).catch(e => {
                console.error('Fetch error:', e);
                alert('Network error: ' + e);
            });
        }

        if (cmdSend) {
            cmdSend.addEventListener('click', sendCommand);
        }
        if (cmdInput) {
            cmdInput.addEventListener('keypress', e => {
                if (e.key === 'Enter') sendCommand();
            });
        }
    </script>
    {% else %}
    <script src="https://cdn.jsdelivr.net/npm/xterm@5.3.0/lib/xterm.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/xterm-addon-fit@0.8.0/lib/xterm-addon-fit.min.js"></script>
    <script>
        const term = new Terminal({
            cursorBlink: true,
            fontSize: 14,
            fontFamily: 'SF Mono, Monaco, Inconsolata, monospace',
            theme: {
                background: '#0d0d1a',
                foreground: '#eee',
                cursor: '#ff6b6b',
                selection: 'rgba(255, 107, 107, 0.3)'
            },
            scrollback: {{ scrollback }}
        });

        const fitAddon = new FitAddon.FitAddon();
        term.loadAddon(fitAddon);
        term.open(document.getElementById('terminal'));
        fitAddon.fit();

        // WebSocket URL
        const wsProtocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = wsProtocol + '//' + location.host + '/ws/terminal' + location.search;

        let ws = null;
        let reconnectTimer = null;

        function connect() {
            ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                document.getElementById('ws-status').textContent = 'Connected';
                document.getElementById('ws-status').className = 'connected';

                // Send initial size
                const dims = fitAddon.proposeDimensions();
                if (dims) {
                    ws.send(JSON.stringify({type: 'resize', cols: dims.cols, rows: dims.rows}));
                }
            };

            ws.onmessage = (event) => {
                if (event.data instanceof Blob) {
                    event.data.text().then(text => term.write(text));
                } else {
                    term.write(event.data);
                }
            };

            ws.onclose = () => {
                document.getElementById('ws-status').textContent = 'Disconnected - Reconnecting...';
                document.getElementById('ws-status').className = 'disconnected';
                reconnectTimer = setTimeout(connect, 2000);
            };

            ws.onerror = (err) => {
                console.error('WebSocket error:', err);
            };
        }

        term.onData(data => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(data);
            }
        });

        window.addEventListener('resize', () => {
            fitAddon.fit();
            if (ws && ws.readyState === WebSocket.OPEN) {
                const dims = fitAddon.proposeDimensions();
                if (dims) {
                    ws.send(JSON.stringify({type: 'resize', cols: dims.cols, rows: dims.rows}));
                }
            }
        });

        connect();
    </script>
    {% endif %}
</body>
</html>
"""

HISTORY_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prompt History</title>
    <style>
        body {
            font-family: -apple-system, sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 1rem;
        }
        h1 { color: #ff6b6b; margin-bottom: 1rem; }
        .prompt-item {
            background: #0d0d1a;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            border-left: 3px solid #6366f1;
        }
        .prompt-time { color: #888; font-size: 0.8rem; }
        .prompt-content {
            font-family: monospace;
            white-space: pre-wrap;
            margin-top: 0.5rem;
            font-size: 0.85rem;
        }
        a { color: #60a5fa; }
    </style>
</head>
<body>
    <h1>Prompt History</h1>
    <p><a href="/">← Back</a></p>
    {% for p in history|reverse %}
    <div class="prompt-item">
        <div class="prompt-time">{{ p.time }}</div>
        <div class="prompt-content">{{ p.context }}</div>
    </div>
    {% endfor %}
    {% if not history %}
    <p>No prompts recorded yet.</p>
    {% endif %}
</body>
</html>
"""

# ============================================================================
# Routes
# ============================================================================

@app.route('/')
@requires_auth
def index():
    # Get session from query param, or use first available
    sessions = backend.get_sessions()
    selected_session = request.args.get('session')

    if not selected_session or selected_session not in sessions:
        # Default to first session with pending prompts, or first session overall
        for s in sessions:
            if state.get_session(s).pending_prompts:
                selected_session = s
                break
        if not selected_session and sessions:
            selected_session = sessions[0]

    state.active_session = selected_session

    if selected_session:
        content = backend.get_content(selected_session)
        session_state = state.get_session(selected_session)
        pending_prompts = session_state.pending_prompts
        parsed_options = session_state.parsed_options
    else:
        content = ""
        pending_prompts = []
        parsed_options = []

    lines = content.strip().split('\n')[-10:] if content else []
    last_content = '\n'.join(lines)

    # Build session info for UI
    sessions_info = []
    for s in sessions:
        ss = state.get_session(s)
        sessions_info.append({
            'name': s,
            'waiting': len(ss.pending_prompts) > 0,
            'active': s == selected_session
        })

    return render_template_string(
        MAIN_TEMPLATE,
        pending_prompts=pending_prompts,
        options=parsed_options,
        last_content=last_content,
        now=datetime.now().strftime('%H:%M:%S'),
        local_ip=get_local_ip(),
        port=config['server']['port'],
        session=selected_session or 'No sessions',
        sessions=sessions_info,
        all_pending=state.get_all_pending(),
        notifications_paused=state.notifications_paused,
        notification_mode=state.notification_mode
    )

@app.route('/terminal')
@requires_auth
def terminal():
    session = request.args.get('session') or state.active_session or config['tmux']['session_name']
    polling_mode = not backend.supports_terminal_attach()
    return render_template_string(
        TERMINAL_TEMPLATE,
        session=session,
        scrollback=config.get('terminal', {}).get('scrollback', 1000),
        polling_mode=polling_mode,
        backend_name=backend.name
    )

@app.route('/history')
@requires_auth
def history():
    session = request.args.get('session') or state.active_session
    if session:
        session_state = state.get_session(session)
        hist = session_state.prompt_history
    else:
        # Combine all histories
        hist = []
        for ss in state.sessions.values():
            hist.extend(ss.prompt_history)
        hist.sort(key=lambda x: x.get('time', ''), reverse=True)
    return render_template_string(HISTORY_TEMPLATE, history=hist)

@app.route('/send', methods=['POST'])
@requires_auth
def send_input():
    """Send input to the terminal session."""
    data = request.get_json()
    text = data.get('input', '')
    session = data.get('session') or state.active_session

    if not session:
        return jsonify({'success': False, 'error': 'No session selected'})

    success = backend.send_keys(session, text)

    if success:
        # Clear session-specific state
        session_state = state.get_session(session)
        session_state.pending_prompts = []
        session_state.last_prompt = None
        session_state.parsed_options = []
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': f'Failed to send keys via {backend.name}'})

@app.route('/send-esc', methods=['POST'])
@requires_auth
def send_esc():
    """Send Escape key to cancel current prompt."""
    data = request.get_json()
    session = data.get('session') or state.active_session

    if not session:
        return jsonify({'success': False, 'error': 'No session selected'})

    success = backend.send_esc(session)

    if success:
        session_state = state.get_session(session)
        session_state.pending_prompts = []
        session_state.last_prompt = None
        session_state.parsed_options = []
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': f'Failed to send Esc via {backend.name}'})

@app.route('/toggle-notifications', methods=['POST'])
@requires_auth
def toggle_notifications():
    """Toggle notification pause state."""
    state.notifications_paused = not state.notifications_paused
    status = "paused" if state.notifications_paused else "resumed"
    log.info(f"Notifications {status}")
    return jsonify({'success': True, 'paused': state.notifications_paused})

@app.route('/quick/<input_val>', methods=['GET', 'POST'])
@requires_auth
def quick_send(input_val):
    """Quick send endpoint for ntfy action buttons."""
    session = request.args.get('session') or state.active_session

    if not session:
        if request.method == 'POST':
            return jsonify({'success': False, 'error': 'No session'})
        return "No session selected", 400

    success = backend.send_keys(session, input_val)

    if success:
        # Clear session-specific state
        session_state = state.get_session(session)
        session_state.pending_prompts = []
        session_state.last_prompt = None
        session_state.parsed_options = []

        # Return JSON for POST (ntfy), redirect for GET (browser)
        if request.method == 'POST':
            return jsonify({'success': True})
        return redirect(url_for('index', session=session))
    else:
        error = f'Failed to send keys via {backend.name}'
        if request.method == 'POST':
            return jsonify({'success': False, 'error': error})
        return f"Error: {error}", 500

@app.route('/api/content')
@requires_auth
def api_content():
    """Get terminal content for polling mode."""
    session = request.args.get('session') or state.active_session
    if not session:
        return jsonify({'content': '', 'error': 'No session'})

    content = backend.get_content(session)
    return jsonify({'content': content or ''})

@app.route('/api/status')
@requires_auth
def api_status():
    session = request.args.get('session') or state.active_session
    sessions = backend.get_sessions()

    if session:
        content = backend.get_content(session)
        is_prompt, context, prompt_type, signature = detect_prompt(content)
        session_state = state.get_session(session)
        pending_count = len(session_state.pending_prompts)
        options = session_state.parsed_options
        last_time = session_state.last_prompt_time.isoformat() if session_state.last_prompt_time else None
    else:
        is_prompt, context, prompt_type, signature = False, None, None, None
        pending_count = 0
        options = []
        last_time = None

    return jsonify({
        'waiting': is_prompt,
        'prompt_type': prompt_type,
        'pending_count': pending_count,
        'context': context,
        'options': options,
        'last_prompt_time': last_time,
        'session': session,
        'sessions': sessions,
        'backend': backend.name,
        'all_pending': state.get_all_pending()
    })

@app.route('/test-notify', methods=['GET', 'POST'])
@requires_auth
def test_notify():
    session = request.args.get('session') or state.active_session or config['tmux']['session_name']
    topic = get_ntfy_topic(session)
    server = config['notifications']['ntfy']['server']
    test_options = [
        {'num': '1', 'text': 'Yes'},
        {'num': '2', 'text': 'Yes, and remember'},
        {'num': '3', 'text': 'No'},
        {'num': '4', 'text': 'No, and never ask again'}
    ]
    title = f"Test [{session}]"
    message = f"Test notification!\nSession: {session}\nTopic: {topic}"

    if request.method == 'POST':
        # Actually send the notification - reset cooldown for this session
        session_state = state.get_session(session)
        session_state.last_notification_time = 0
        result = send_ntfy(message, title, options=test_options, session=session)
        if result:
            print(f"[test-notify] Successfully sent to {server}/{topic}")
        else:
            print(f"[test-notify] FAILED to send to {server}/{topic}")
        return redirect(url_for('index', session=session))

    # GET: Show preview page
    return render_template_string(
        TEST_NOTIFY_TEMPLATE,
        server=server,
        topic=topic,
        title=title,
        message=message,
        options=test_options
    )

@app.route('/config')
@requires_auth
def show_config():
    safe_config = {
        'server': config['server'],
        'auth': {'method': config['auth']['method']},
        'notifications': {
            'ntfy': {
                'enabled': config['notifications']['ntfy']['enabled'],
                'topic': get_ntfy_topic()
            },
            'pushover': {'enabled': config['notifications']['pushover']['enabled']},
            'polling': {'enabled': config['notifications']['polling']['enabled']}
        },
        'tmux': config['tmux'],
        'terminal': config.get('terminal', {}),
        'detection': config['detection']
    }
    return jsonify(safe_config)

@app.route('/api/clear-prompts', methods=['POST'])
@requires_auth
def api_clear_prompts():
    """Clear pending prompts (used by Ignore button in notifications)."""
    session = request.args.get('session')

    if session:
        # Clear specific session
        session_state = state.get_session(session)
        count = len(session_state.pending_prompts)
        session_state.pending_prompts.clear()
        print(f"[api] Cleared {count} pending prompts for {session}")
    else:
        # Clear all sessions
        count = 0
        for session_state in state.sessions.values():
            count += len(session_state.pending_prompts)
            session_state.pending_prompts.clear()
        print(f"[api] Cleared {count} pending prompts (all sessions)")

    return jsonify({'success': True, 'cleared': count})

# ============================================================================
# Main
# ============================================================================

def main():
    global backend
    backend = get_active_backend()

    local_ip = get_local_ip()
    port = config['server']['port']
    sessions = backend.get_sessions()
    ntfy_prefix = config['notifications']['ntfy'].get('topic_prefix', 'claude-remote')

    # Check backend availability
    ghostty_status = "available" if (_ghostty_backend and _ghostty_backend.is_available()) else "not available"
    tmux_status = "available" if _tmux_backend.is_available() else "not available"

    print(f"""
╔═══════════════════════════════════════════════════════════╗
║         Claude Code Remote Control Server                  ║
╠═══════════════════════════════════════════════════════════╣
║  Control Panel: http://{local_ip}:{port}/
║  Terminal:      http://{local_ip}:{port}/terminal
║  Auth Method:   {config['auth']['method']}
║  Backend:       {backend.name}
║  ntfy prefix:   {ntfy_prefix}-<session>
╠═══════════════════════════════════════════════════════════╣
║  Backends:  tmux: {tmux_status:<12}  ghostty: {ghostty_status:<12}  ║
╚═══════════════════════════════════════════════════════════╝
    """)

    if sessions:
        print(f"[{backend.name}] Found {len(sessions)} session(s): {', '.join(sessions)}")
    else:
        print(f"[{backend.name}] No sessions found!")
        if backend.name == 'tmux':
            print("          Create with: tmux new-session -d -s <name>")
        else:
            print("          Open Ghostty and run Claude Code")

    watcher_thread = threading.Thread(target=watcher_loop, daemon=True)
    watcher_thread.start()

    app.run(
        host=config['server']['host'],
        port=config['server']['port'],
        debug=False,
        threaded=True
    )

if __name__ == '__main__':
    main()
