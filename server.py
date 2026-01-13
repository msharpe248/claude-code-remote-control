#!/usr/bin/env python3
"""
Remote Control Server for Claude Code
Provides web interface, embedded terminal, and notifications for remote prompt answering.

Supports multiple terminal backends:
- tmux: Traditional tmux sessions (default, cross-platform)
- accessibility: macOS Accessibility API (supports Ghostty, iTerm2, Terminal.app, Kitty, Alacritty, WezTerm)
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
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Optional, List, Dict, Any

from flask import Flask, request, Response, render_template_string, jsonify, redirect, url_for
from flask_sock import Sock

# Optional: Accessibility backend (macOS only)
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
    ACCESSIBILITY_AVAILABLE = True
except ImportError:
    ACCESSIBILITY_AVAILABLE = False

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

    def send_shift_tab(self, session: str) -> bool:
        """Send Shift+Tab to a session (cycles Claude modes). Returns success."""
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

    def send_shift_tab(self, session: str) -> bool:
        """Send Shift+Tab to tmux session (cycles Claude modes)."""
        try:
            subprocess.run(['tmux', 'send-keys', '-t', session, 'BTab'], check=True, timeout=5)
            return True
        except Exception as e:
            log.error(f"[tmux] Error sending Shift+Tab to {session}: {e}")
            return False

    def supports_terminal_attach(self) -> bool:
        return True

    def get_attach_command(self, session: str) -> Optional[List[str]]:
        return ['tmux', 'attach-session', '-t', session]


# Supported terminals for macOS Accessibility backend
# Maps display name -> (window owner names, app name for activation, process name for System Events)
# Note: System Events process names are case-sensitive and often lowercase
SUPPORTED_TERMINALS = {
    'Ghostty': (['Ghostty'], 'Ghostty', 'ghostty'),
    'iTerm2': (['iTerm2', 'iTerm'], 'iTerm', 'iTerm2'),
    'Terminal': (['Terminal'], 'Terminal', 'Terminal'),
    'Kitty': (['kitty'], 'kitty', 'kitty'),
    'Alacritty': (['Alacritty', 'alacritty'], 'Alacritty', 'Alacritty'),
    'WezTerm': (['WezTerm', 'wezterm-gui'], 'WezTerm', 'WezTerm'),
}


class AccessibilityBackend(TerminalBackend):
    """macOS Accessibility API backend - works with multiple terminal apps."""

    def __init__(self):
        self._pid: Optional[int] = None
        self._app = None
        self._terminals_cache: Dict[str, str] = {}
        self._last_refresh = 0
        self._refresh_interval = 1.0  # Seconds between tree refreshes
        self._detected_terminal: Optional[str] = None  # e.g., 'Ghostty', 'iTerm2'
        self._applescript_name: Optional[str] = None   # e.g., 'Ghostty', 'iTerm' (for tell application)
        self._process_name: Optional[str] = None       # e.g., 'ghostty' (for System Events tell process)

    @property
    def name(self) -> str:
        if self._detected_terminal:
            return f"accessibility ({self._detected_terminal})"
        return "accessibility"

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

    def _find_terminal_pid(self) -> Optional[int]:
        """Find a supported terminal's process ID."""
        windows = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)

        # Try each supported terminal
        for terminal_name, (window_names, applescript_name, process_name) in SUPPORTED_TERMINALS.items():
            for window in windows:
                owner = window.get('kCGWindowOwnerName', '')
                pid = window.get('kCGWindowOwnerPID', 0)
                if owner in window_names and pid:
                    self._detected_terminal = terminal_name
                    self._applescript_name = applescript_name
                    self._process_name = process_name
                    log.info(f"[accessibility] Detected terminal: {terminal_name} (pid={pid})")
                    return pid

        return None

    def _get_terminal_app(self):
        """Get or create accessibility element for the detected terminal."""
        # Check if we need to find a terminal
        if self._pid is None:
            self._pid = self._find_terminal_pid()
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

        Session names are derived from TTY (e.g., "s006").
        """
        try:
            # Get all processes with PID, TTY, and command
            result = subprocess.run(
                ['ps', '-axo', 'pid,tty,command'],
                capture_output=True,
                text=True,
                timeout=5
            )

            processes = []
            for line in result.stdout.split('\n')[1:]:  # Skip header
                parts = line.split(None, 2)  # Split into at most 3 parts
                if len(parts) >= 3:
                    pid, tty, cmd = parts[0], parts[1], parts[2]

                    # Identify claude processes
                    if '/claude' in cmd or ' claude ' in cmd or cmd.endswith(' claude') or cmd.strip() == 'claude':
                        # Skip tmux commands that happen to contain 'claude'
                        if cmd.startswith('tmux '):
                            continue

                        # Use TTY as session name
                        session_name = tty if tty and tty != '??' else None

                        processes.append({
                            'pid': pid,
                            'session': session_name,
                            'tty': tty,
                            'line': cmd
                        })

            return processes
        except Exception as e:
            log.error(f"[accessibility] Error getting claude processes: {e}")
            return []

    def _refresh_terminals(self):
        """Refresh the terminal content cache from the detected terminal."""
        now = time.time()
        if now - self._last_refresh < self._refresh_interval:
            return

        self._last_refresh = now

        # Get running Claude processes
        claude_procs = self._get_claude_processes()
        if not claude_procs:
            self._terminals_cache = {}
            return

        # Get terminal contents via accessibility API
        app = self._get_terminal_app()
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

        log.debug(f"[accessibility] Found {len(terminal_contents)} terminal panes, {len(claude_procs)} Claude processes")

        # Sort processes so explicitly named sessions come first (they get priority for matching)
        # Sessions without explicit names will use TTY as session name
        sorted_procs = sorted(claude_procs, key=lambda p: (p.get('tty') == p.get('session'), p['pid']))

        for proc in sorted_procs:
            session_name = proc['session']
            if not session_name:
                continue  # Skip if no session name (shouldn't happen now with TTY fallback)

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
                if "claude-code" in deep_content.lower():
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
                if "[accessibility]" in recent or "[rc]" in recent:
                    score -= 15
                # Timestamp patterns like "2026-01-01 01:24:24,136"
                if re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}', recent):
                    score -= 20

                log.debug(f"[accessibility] Terminal {i} score: {score} (len={len(content)})")
                if score > best_score:
                    best_score = score
                    matched_content = content
                    matched_idx = i

            # Only use the match if it has a reasonable score
            if matched_content and matched_idx is not None:
                if best_score >= 0:
                    used_contents.add(matched_idx)
                    terminals[session_name] = matched_content
                    log.debug(f"[accessibility] Session '{session_name}' matched to Terminal {matched_idx} (score: {best_score})")
                else:
                    # Negative score means it's likely NOT a Claude terminal
                    # Don't use it - better to have no content than wrong content
                    log.debug(f"[accessibility] Rejecting match for '{session_name}' - Terminal {matched_idx} (score: {best_score})")
                    terminals[session_name] = ""

        self._terminals_cache = terminals

    def is_available(self) -> bool:
        """Check if a supported terminal with accessibility access is available."""
        if not ACCESSIBILITY_AVAILABLE:
            return False
        pid = self._find_terminal_pid()
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
        # Try exact match first
        content = self._terminals_cache.get(session, "")
        if not content:
            # Try normalized TTY (ttys006 -> s006)
            normalized = session
            if session.startswith('tty'):
                normalized = session[3:]
            content = self._terminals_cache.get(normalized, "")
        log.debug(f"[accessibility] get_content('{session}'): cache keys={list(self._terminals_cache.keys())}, content_len={len(content)}")
        return content

    def send_keys(self, session: str, keys: str) -> bool:
        """
        Send keystrokes to the detected terminal.

        Note: The macOS Accessibility API is read-only.
        We use AppleScript to send keystrokes to the terminal.
        """
        if not self._applescript_name:
            log.error("[accessibility] No terminal detected, cannot send keys")
            return False

        app_name = self._applescript_name
        try:
            # First, try to activate the terminal with a short timeout
            activate_script = f'tell application "{app_name}" to activate'
            try:
                subprocess.run(
                    ['osascript', '-e', activate_script],
                    capture_output=True,
                    text=True,
                    timeout=3
                )
            except subprocess.TimeoutExpired:
                log.warning("[accessibility] Activate timed out, trying to send keys anyway")

            if keys:
                # Escape special characters for AppleScript string
                escaped_keys = keys.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')

                script = f'''
                tell application "{app_name}" to activate
                delay 0.2
                tell application "System Events"
                    keystroke "{escaped_keys}"
                    delay 0.1
                    key code 36
                end tell
                '''
            else:
                # Just Enter
                process_name = app_name.lower()
                script = f'''
                tell application "System Events"
                    tell process "{process_name}"
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
                log.error(f"[accessibility] AppleScript error: {result.stderr}")
                return False
            return True
        except subprocess.TimeoutExpired:
            log.error("[accessibility] AppleScript timed out sending keys")
            return False
        except Exception as e:
            log.error(f"[accessibility] Error sending keys: {e}")
            return False

    def send_esc(self, session: str) -> bool:
        """Send Escape key to the detected terminal."""
        if not self._applescript_name:
            log.error("[accessibility] No terminal detected, cannot send Esc")
            return False

        app_name = self._applescript_name
        try:
            script = f'''
            tell application "{app_name}" to activate
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
                log.error(f"[accessibility] AppleScript error: {result.stderr}")
                return False
            return True
        except subprocess.TimeoutExpired:
            log.error("[accessibility] AppleScript timed out sending Esc")
            return False
        except Exception as e:
            log.error(f"[accessibility] Error sending Esc: {e}")
            return False

    def send_shift_tab(self, session: str) -> bool:
        """Send Shift+Tab to the detected terminal (cycles Claude modes)."""
        if not self._applescript_name or not self._process_name:
            log.error("[accessibility] No terminal detected, cannot send Shift+Tab")
            return False

        app_name = self._applescript_name
        process_name = self._process_name
        try:
            # Use key code 48 (Tab) with shift modifier
            # Note: app_name is for activation, process_name is for System Events (case-sensitive)
            script = f'''
            tell application "{app_name}" to activate
            delay 0.2
            tell application "System Events"
                tell process "{process_name}"
                    key code 48 using {{shift down}}
                end tell
            end tell
            '''
            log.info(f"[accessibility] Sending Shift+Tab to {app_name} (process: {process_name})")
            result = subprocess.run(
                ['osascript', '-e', script],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                log.error(f"[accessibility] AppleScript error: {result.stderr}")
                return False
            log.info(f"[accessibility] Shift+Tab sent successfully")
            return True
        except subprocess.TimeoutExpired:
            log.error("[accessibility] AppleScript timed out sending Shift+Tab")
            return False
        except Exception as e:
            log.error(f"[accessibility] Error sending Shift+Tab: {e}")
            return False

    def supports_terminal_attach(self) -> bool:
        # Accessibility backend doesn't support attaching like tmux
        return False


# Initialize backends
_tmux_backend = TmuxBackend()
_accessibility_backend = AccessibilityBackend() if ACCESSIBILITY_AVAILABLE else None

def get_active_backend() -> TerminalBackend:
    """Get the currently active terminal backend based on config and availability."""
    backend_pref = config.get('terminal_backend', {}).get('prefer', 'auto')

    if backend_pref == 'tmux':
        if _tmux_backend.is_available():
            return _tmux_backend
        log.warning("[backend] tmux preferred but not available")

    elif backend_pref in ('ghostty', 'accessibility'):
        if _accessibility_backend and _accessibility_backend.is_available():
            return _accessibility_backend
        log.warning("[backend] accessibility preferred but not available")

    # Auto-detect: prefer accessibility if available (no tmux needed), fall back to tmux
    if backend_pref == 'auto':
        # Check if accessibility backend has Claude sessions
        if _accessibility_backend and _accessibility_backend.is_available():
            sessions = _accessibility_backend.get_sessions()
            if sessions:
                # Check if any session has Claude content
                for s in sessions:
                    content = _accessibility_backend.get_content(s)
                    if content and ("claude" in content.lower()[-2000:] or
                                   "? for shortcuts" in content[-1000:]):
                        return _accessibility_backend

        # Fall back to tmux
        if _tmux_backend.is_available():
            return _tmux_backend

    # Last resort
    if _tmux_backend.is_available():
        return _tmux_backend
    if _accessibility_backend and _accessibility_backend.is_available():
        return _accessibility_backend

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
        self.last_notification_time = 0
        self.is_compacting = False  # Claude is compacting conversation
        self.compacting_started_at = 0  # When compacting was detected (for minimum display time)
        # Note: prompt detection fields removed - using hooks instead

@dataclass
class HookEvent:
    """Represents a Claude Code hook event."""
    timestamp: datetime
    event_type: str        # Stop, PreToolUse, PostToolUse, Notification, etc.
    session_id: str
    tool_name: Optional[str] = None
    tool_input: Optional[Dict] = None
    tool_result: Optional[str] = None
    reason: Optional[str] = None
    raw_data: Optional[Dict] = None
    permission_mode: Optional[str] = None  # 'ask' if tool requires permission

    def to_dict(self):
        """Convert to JSON-serializable dict."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'session_id': self.session_id,
            'tool_name': self.tool_name,
            'tool_input': self.tool_input,
            'tool_result': self.tool_result[:500] if self.tool_result and len(self.tool_result) > 500 else self.tool_result,
            'reason': self.reason,
            'permission_mode': self.permission_mode,
            'raw_data': self.raw_data,
        }


@dataclass
class RegisteredSession:
    """A Claude session registered via hooks (auto-discovered or manually named)."""
    session_id: str
    name: str              # Friendly name (e.g., "myapp@ttys001")
    cwd: str               # Working directory
    tty: str               # Terminal (e.g., "ttys001")
    custom_name: Optional[str] = None  # User-set override via API
    registered_at: datetime = field(default_factory=datetime.now)

    def display_name(self) -> str:
        """Get the name to display (custom name takes priority)."""
        return self.custom_name or self.name

    def to_dict(self):
        """Convert to JSON-serializable dict."""
        return {
            'session_id': self.session_id,
            'name': self.name,
            'display_name': self.display_name(),
            'cwd': self.cwd,
            'tty': self.tty,
            'custom_name': self.custom_name,
            'registered_at': self.registered_at.isoformat(),
        }


class GlobalState:
    def __init__(self):
        self.sessions = {}  # session_name -> SessionState
        self.active_session = None  # Currently selected session in UI

        # Notification mode from config: active, standby, log_only
        self.notification_mode = config.get('notifications', {}).get('mode', 'active')
        # Start paused if mode is "standby"
        self.notifications_paused = (self.notification_mode == 'standby')

        # Hook events storage
        self.hook_events: List[HookEvent] = []  # Recent hook events (max 500)
        self.hook_clients: List = []  # WebSocket clients for hook updates

        # Registered sessions (discovered via hooks)
        self.registered_sessions: Dict[str, RegisteredSession] = {}  # session_id -> RegisteredSession

    def get_session(self, session_name):
        """Get or create session state."""
        if session_name not in self.sessions:
            self.sessions[session_name] = SessionState(session_name)
        return self.sessions[session_name]

    # get_all_pending removed - using hooks instead

    def add_hook_event(self, event: HookEvent):
        """Add a hook event and broadcast to WebSocket clients."""
        self.hook_events.append(event)
        # Keep only last 500 events
        if len(self.hook_events) > 500:
            self.hook_events = self.hook_events[-500:]
        # Broadcast to all connected clients
        self.broadcast_hook_event(event)

    def broadcast_hook_event(self, event: HookEvent):
        """Send hook event to all connected WebSocket clients."""
        event_json = json.dumps(event.to_dict())
        log.info(f"[hooks-ws] Broadcasting to {len(self.hook_clients)} clients: {event.event_type} / {event.tool_name}")
        dead_clients = []
        for client in self.hook_clients:
            try:
                client.send(event_json)
                log.info(f"[hooks-ws] Sent to client successfully")
            except Exception as e:
                log.info(f"[hooks-ws] Failed to send to client: {e}")
                dead_clients.append(client)
        # Remove dead clients
        for client in dead_clients:
            if client in self.hook_clients:
                self.hook_clients.remove(client)

    def get_hook_events(self, session_id: Optional[str] = None, event_type: Optional[str] = None, limit: int = 100):
        """Get hook events with optional filtering."""
        events = self.hook_events
        if session_id:
            events = [e for e in events if e.session_id == session_id]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]

    # ---- Session Registry Methods ----

    @staticmethod
    def normalize_tty(tty: str) -> str:
        """Normalize TTY name for comparison (ttys006 -> s006, s006 -> s006)."""
        if not tty:
            return tty
        # Strip /dev/ prefix if present
        if tty.startswith('/dev/'):
            tty = tty[5:]
        # Strip 'tty' prefix if present (ttys006 -> s006)
        if tty.startswith('tty'):
            tty = tty[3:]
        return tty

    def register_session(self, session_id: str, name: str, cwd: str, tty: str):
        """Register a session discovered via hooks."""
        self.registered_sessions[session_id] = RegisteredSession(
            session_id=session_id,
            name=name,
            cwd=cwd,
            tty=tty,
        )
        log.info(f"[session] Registered session: {name} (id={session_id[:8]}..., tty={tty})")

    def unregister_session(self, session_id: str):
        """Remove a session from the registry."""
        if session_id in self.registered_sessions:
            sess = self.registered_sessions.pop(session_id)
            log.info(f"[session] Unregistered session: {sess.display_name()} (id={session_id[:8]}...)")

    def get_session_display_name(self, session_id: str) -> str:
        """Get friendly display name for a session ID."""
        sess = self.registered_sessions.get(session_id)
        if sess:
            return sess.display_name()
        return session_id[:8]  # Fallback to short UUID

    def set_session_custom_name(self, session_id: str, custom_name: str) -> bool:
        """Set a custom name for a session."""
        sess = self.registered_sessions.get(session_id)
        if sess:
            sess.custom_name = custom_name
            log.info(f"[session] Set custom name: {custom_name} for session {session_id[:8]}...")
            return True
        return False

    def get_registered_sessions(self) -> List[RegisteredSession]:
        """Get all registered sessions."""
        return list(self.registered_sessions.values())

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
# Notification Backends
# (parse_numbered_options and parse_idle_default_text removed - using hooks)
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
    prefix = cfg.get('topic_prefix', 'claude')
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
# Terminal Watcher (simplified - only compacting detection, prompts use hooks)
# ============================================================================

def detect_compacting(content):
    """Check if Claude is compacting the conversation."""
    if not content:
        return False
    # Check recent lines for compacting message
    lines = content.strip().split('\n')
    recent_text = '\n'.join(lines[-10:]).lower()
    return "compacting conversation" in recent_text

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
            # Re-check backend periodically (in case terminals start/stop)
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
                session_state = state.get_session(session_name)

                # Check for compacting state (with minimum 3s display time)
                # This is still terminal-based since there's no hook for compacting
                is_compacting_now = detect_compacting(content)
                now = time.time()

                if is_compacting_now and not session_state.is_compacting:
                    # Just started compacting
                    session_state.is_compacting = True
                    session_state.compacting_started_at = now
                    log.info(f"[{session_name}] Compacting conversation...")
                elif not is_compacting_now and session_state.is_compacting:
                    # Compacting might be done - but keep visible for minimum 3 seconds
                    elapsed = now - session_state.compacting_started_at
                    if elapsed >= 3.0:
                        session_state.is_compacting = False
                        log.info(f"[{session_name}] Compacting done")

                # Note: Prompt detection removed - using hooks instead

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

    # For backends that don't support PTY attachment (like the accessibility backend),
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


@sock.route('/ws/hooks')
def hooks_websocket(ws):
    """WebSocket endpoint for real-time hook events."""
    if not check_ws_auth():
        ws.close(1008, "Unauthorized")
        return

    # Register this client
    state.hook_clients.append(ws)
    log.info(f"[hooks-ws] Client connected ({len(state.hook_clients)} total)")

    # Send recent events to catch up
    recent_events = state.get_hook_events(limit=50)
    for event in recent_events:
        try:
            ws.send(json.dumps(event.to_dict()))
        except Exception:
            break

    try:
        # Keep connection alive and wait for close
        while True:
            try:
                msg = ws.receive(timeout=30)
                if msg is None:
                    break
                # Handle ping/pong or other messages if needed
                if msg == 'ping':
                    ws.send('pong')
            except Exception:
                break
    finally:
        if ws in state.hook_clients:
            state.hook_clients.remove(ws)
        log.info(f"[hooks-ws] Client disconnected ({len(state.hook_clients)} total)")


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
        /* Theme CSS Variables */
        :root {
            --bg-primary: #1a1a2e;
            --bg-secondary: #0d0d1a;
            --bg-tertiary: #2d2d4a;
            --bg-card: #1e1e3f;
            --text-primary: #eee;
            --text-secondary: #ccc;
            --text-muted: #888;
            --text-heading: #ff6b6b;
            --border-color: #333;
            --accent-primary: #6366f1;
            --accent-hover: #4f46e5;
            --accent-success: #22c55e;
            --accent-info: #60a5fa;
            --overlay-bg: rgba(13, 13, 26, 0.95);
        }
        [data-theme="light"] {
            --bg-primary: #f5f5f7;
            --bg-secondary: #ffffff;
            --bg-tertiary: #e5e5ea;
            --bg-card: #ffffff;
            --text-primary: #1d1d1f;
            --text-secondary: #424245;
            --text-muted: #6e6e73;
            --text-heading: #dc2626;
            --border-color: #d1d1d6;
            --accent-primary: #6366f1;
            --accent-hover: #4f46e5;
            --accent-success: #16a34a;
            --accent-info: #2563eb;
            --overlay-bg: rgba(245, 245, 247, 0.95);
        }
        /* Light mode component overrides */
        [data-theme="light"] .global-mode-bar {
            background: #e5e7eb;
        }
        [data-theme="light"] .global-mode-bar:hover {
            background: #d1d5db;
        }
        [data-theme="light"] .global-mode-bar:active {
            background: #c4c8cf;
        }
        [data-theme="light"] .global-mode-text {
            color: #374151;
        }
        [data-theme="light"] .global-mode-hint {
            color: #6b7280;
        }
        [data-theme="light"] .mode-toggle {
            background: rgba(0,0,0,0.05);
            border-color: rgba(0,0,0,0.15);
        }
        [data-theme="light"] .mode-toggle:hover {
            background: rgba(0,0,0,0.1);
            border-color: rgba(0,0,0,0.2);
        }
        [data-theme="light"] .mode-label {
            color: #374151;
        }
        [data-theme="light"] .mode-hint {
            color: #6b7280;
        }
        [data-theme="light"] .context-indicator {
            background: #e5e7eb;
        }
        [data-theme="light"] .context-bar {
            background: #d1d5db;
        }
        [data-theme="light"] .context-text {
            color: #4b5563;
        }
        [data-theme="light"] .idle-panel {
            background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        }
        [data-theme="light"] .idle-title {
            color: #1e40af;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 1rem;
            transition: background 0.3s, color 0.3s;
        }
        .container { max-width: 800px; margin: 0 auto; }
        .page-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }
        h1 { font-size: 1.5rem; color: var(--text-heading); }
        h3 { margin: 1rem 0 0.5rem; color: var(--text-secondary); }
        .theme-toggle {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 8px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1rem;
        }
        .theme-toggle:hover { background: var(--bg-tertiary); }

        .status { padding: 1rem; border-radius: 8px; margin-bottom: 1rem; }
        .status.waiting { background: #2d4a3e; border-left: 4px solid #4ade80; }
        .status.idle { background: #2d3a4a; border-left: 4px solid #60a5fa; }
        .status.compacting { background: #3d2d4a; border-left: 4px solid #c084fc; }

        /* Brain crunching animation */
        @keyframes brain-crunch {
            0%, 100% { transform: scale(1) rotate(0deg); }
            25% { transform: scale(1.1) rotate(-5deg); }
            50% { transform: scale(0.9) rotate(5deg); }
            75% { transform: scale(1.05) rotate(-3deg); }
        }
        @keyframes brain-pulse {
            0%, 100% { opacity: 1; filter: brightness(1); }
            50% { opacity: 0.8; filter: brightness(1.3); }
        }
        .brain-icon {
            display: inline-block;
            font-size: 1.5em;
            animation: brain-crunch 0.6s ease-in-out infinite, brain-pulse 1s ease-in-out infinite;
        }

        /* Full-screen compacting overlay */
        .compacting-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: var(--overlay-bg);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 9999;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.3s ease, visibility 0.3s ease;
        }
        .compacting-overlay.active {
            opacity: 1;
            visibility: visible;
        }
        .compacting-overlay .brain-large {
            font-size: 5rem;
            animation: brain-crunch 0.5s ease-in-out infinite, brain-glow 1.5s ease-in-out infinite;
            margin-bottom: 1.5rem;
        }
        @keyframes brain-glow {
            0%, 100% {
                filter: drop-shadow(0 0 10px rgba(192, 132, 252, 0.5));
            }
            50% {
                filter: drop-shadow(0 0 30px rgba(192, 132, 252, 0.9)) drop-shadow(0 0 60px rgba(192, 132, 252, 0.4));
            }
        }
        .compacting-overlay .title {
            font-size: 1.5rem;
            font-weight: bold;
            color: #c084fc;
            margin-bottom: 0.5rem;
        }
        .compacting-overlay .subtitle {
            font-size: 0.9rem;
            color: #888;
        }

        /* .prompt-box removed - using hook-based panels instead */

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
        .btn-mode { background: #0891b2; color: white; }
        .btn-full { grid-column: 1 / -1; }
        .btn-num { font-size: 1.25rem; font-weight: bold; }
        .btn-label { font-size: 0.7rem; opacity: 0.8; margin-top: 2px; }

        .input-group { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
        .input-group input {
            flex: 1;
            padding: 0.875rem;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            background: var(--bg-secondary);
            color: var(--text-primary);
            font-size: 1rem;
        }

        .links { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem; }
        .links a { color: var(--accent-info); padding: 0.5rem; font-size: 0.9rem; }

        .meta { font-size: 0.75rem; color: var(--text-muted); }

        #refresh-indicator {
            position: fixed; top: 1rem; right: 1rem;
            width: 10px; height: 10px; border-radius: 50%;
            background: #22c55e; animation: pulse 2s infinite;
        }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

        /* .option-btn removed - using hook-based panels instead */

        .session-tabs {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }
        .session-tab {
            padding: 0.5rem 1rem;
            border-radius: 8px;
            background: var(--bg-tertiary);
            color: var(--text-secondary);
            text-decoration: none;
            font-size: 0.85rem;
            position: relative;
        }
        .session-tab.active {
            background: var(--accent-primary);
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

        .nav-links {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }
        .nav-links a {
            color: var(--accent-info);
            text-decoration: none;
            padding: 0.5rem 1rem;
            background: var(--bg-secondary);
            border-radius: 6px;
            font-size: 0.9rem;
        }
        .nav-links a:hover { background: var(--bg-tertiary); }
        .nav-links a.active {
            background: var(--accent-primary);
            color: white;
        }

        /* Rich Question UI from Hook Events */
        .hook-question-panel {
            display: none;
            background: var(--bg-card);
            border: 1px solid var(--accent-primary);
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 20px rgba(99, 102, 241, 0.2);
        }
        .hook-question-panel.active {
            display: block;
            animation: slideIn 0.3s ease;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .hook-question-header {
            display: inline-block;
            background: var(--accent-primary);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.75rem;
        }
        .hook-question-text {
            font-size: 1.1rem;
            color: var(--text-primary);
            margin-bottom: 1rem;
            line-height: 1.4;
        }
        .hook-question-options {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        .hook-option-btn {
            display: flex;
            align-items: flex-start;
            gap: 0.75rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.875rem 1rem;
            cursor: pointer;
            transition: all 0.2s ease;
            text-align: left;
        }
        .hook-option-btn:hover {
            background: var(--bg-tertiary);
            border-color: var(--accent-primary);
            transform: translateX(4px);
        }
        .hook-option-num {
            background: var(--accent-primary);
            color: white;
            min-width: 28px;
            height: 28px;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 0.9rem;
            flex-shrink: 0;
        }
        .hook-option-content {
            flex: 1;
        }
        .hook-option-label {
            color: var(--text-primary);
            font-weight: 500;
            font-size: 0.95rem;
            margin-bottom: 0.25rem;
        }
        .hook-option-desc {
            color: var(--text-muted);
            font-size: 0.8rem;
            line-height: 1.3;
        }
        .hook-question-meta {
            margin-top: 0.75rem;
            font-size: 0.7rem;
            color: var(--text-muted);
        }
        .hook-question-source {
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            background: rgba(34, 197, 94, 0.2);
            color: #4ade80;
            padding: 0.15rem 0.5rem;
            border-radius: 4px;
            font-size: 0.7rem;
        }
        .hook-question-other {
            display: flex;
            gap: 0.5rem;
            margin-top: 0.75rem;
            padding-top: 0.75rem;
            border-top: 1px solid var(--border-color);
        }
        .hook-question-other input {
            flex: 1;
            padding: 0.75rem 1rem;
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 8px;
            background: rgba(255,255,255,0.05);
            color: #fff;
            font-size: 0.95rem;
        }
        [data-theme="light"] .hook-question-other input {
            border-color: var(--border-color);
            background: var(--bg-secondary);
            color: var(--text-primary);
        }
        .hook-question-other input:focus {
            outline: none;
            border-color: #6366f1;
            background: rgba(99, 102, 241, 0.1);
        }
        [data-theme="light"] .hook-question-other input:focus {
            border-color: var(--accent-primary);
            background: var(--bg-tertiary);
        }
        .hook-question-other input::placeholder {
            color: #666;
        }
        [data-theme="light"] .hook-question-other input::placeholder {
            color: var(--text-muted);
        }
        .hook-question-other button {
            padding: 0.75rem 1.25rem;
            border: none;
            border-radius: 8px;
            background: #6366f1;
            color: white;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.2s;
        }
        .hook-question-other button:hover {
            background: #4f46e5;
        }
        .hook-question-actions {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 1rem;
            padding-top: 0.75rem;
            border-top: 1px solid rgba(255,255,255,0.1);
        }
        .hook-question-escape {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            border: 1px solid #ef4444;
            border-radius: 6px;
            background: transparent;
            color: #ef4444;
            font-size: 0.85rem;
            cursor: pointer;
            transition: background 0.2s;
        }
        .hook-question-escape:hover {
            background: rgba(239, 68, 68, 0.2);
        }

        /* Tool Permission Dialog */
        .permission-panel {
            display: none;
            background: linear-gradient(135deg, #1e2a1e 0%, #2d3a2d 100%);
            border: 1px solid #22c55e;
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 20px rgba(34, 197, 94, 0.2);
        }
        [data-theme="light"] .permission-panel {
            background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        }
        .permission-panel.active {
            display: block;
            animation: slideIn 0.3s ease;
        }
        .permission-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1rem;
        }
        .permission-icon {
            font-size: 1.5rem;
        }
        .permission-title {
            font-size: 1rem;
            color: #fff;
            font-weight: 500;
        }
        [data-theme="light"] .permission-title {
            color: #1d1d1f;
        }
        .permission-tool {
            color: #4ade80;
            font-family: 'SF Mono', Monaco, monospace;
        }
        [data-theme="light"] .permission-tool {
            color: #16a34a;
        }
        .permission-detail {
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            padding: 0.75rem;
            margin-bottom: 1rem;
            max-height: 200px;
            overflow-y: auto;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 0.8rem;
            color: #ccc;
            white-space: pre-wrap;
            word-break: break-all;
        }
        [data-theme="light"] .permission-detail {
            background: rgba(34, 197, 94, 0.1);
            color: #424245;
        }
        .permission-options {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        .permission-btn {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
            text-align: left;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .permission-btn.yes {
            background: rgba(34, 197, 94, 0.2);
            border-color: rgba(34, 197, 94, 0.3);
            color: #4ade80;
        }
        [data-theme="light"] .permission-btn.yes {
            color: #16a34a;
        }
        .permission-btn.yes:hover {
            background: rgba(34, 197, 94, 0.3);
            border-color: #22c55e;
        }
        .permission-btn.yes-all {
            background: rgba(59, 130, 246, 0.2);
            border-color: rgba(59, 130, 246, 0.3);
            color: #60a5fa;
        }
        [data-theme="light"] .permission-btn.yes-all {
            color: #2563eb;
        }
        .permission-btn.yes-all:hover {
            background: rgba(59, 130, 246, 0.3);
            border-color: #3b82f6;
        }
        .permission-btn.no {
            background: rgba(239, 68, 68, 0.15);
            border-color: rgba(239, 68, 68, 0.3);
            color: #f87171;
        }
        [data-theme="light"] .permission-btn.no {
            color: #dc2626;
        }
        .permission-btn.no:hover {
            background: rgba(239, 68, 68, 0.25);
            border-color: #ef4444;
        }
        .permission-btn-num {
            min-width: 24px;
            height: 24px;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 0.85rem;
            background: rgba(255,255,255,0.1);
        }
        .permission-btn-text {
            flex: 1;
        }
        .permission-btn-hint {
            font-size: 0.75rem;
            color: #888;
        }
        .permission-extra-input {
            display: none;
            margin-top: 0.5rem;
            padding-left: 2.5rem;
        }
        .permission-extra-input.active {
            display: flex;
            gap: 0.5rem;
        }
        .permission-extra-input input {
            flex: 1;
            padding: 0.6rem 0.75rem;
            border: 1px solid rgba(34, 197, 94, 0.3);
            border-radius: 6px;
            background: rgba(0,0,0,0.3);
            color: #fff;
            font-size: 0.9rem;
        }
        [data-theme="light"] .permission-extra-input input {
            border-color: rgba(34, 197, 94, 0.5);
            background: rgba(34, 197, 94, 0.1);
            color: #1d1d1f;
        }
        .permission-extra-input input:focus {
            outline: none;
            border-color: #22c55e;
        }
        .permission-extra-input button {
            padding: 0.6rem 1rem;
            border: none;
            border-radius: 6px;
            background: #22c55e;
            color: #000;
            font-weight: 500;
            cursor: pointer;
        }
        .permission-extra-input button:hover {
            background: #16a34a;
        }
        .permission-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.5rem;
            margin-top: 0.75rem;
            padding-top: 0.5rem;
            border-top: 1px solid rgba(255,255,255,0.1);
            font-size: 0.7rem;
            color: #666;
        }
        .permission-meta span:first-child {
            flex: 1;
        }
        .permission-meta-btn {
            padding: 0.3rem 0.6rem;
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 4px;
            background: rgba(255,255,255,0.05);
            color: #888;
            font-size: 0.7rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        .permission-meta-btn:hover {
            background: rgba(255,255,255,0.1);
            color: #fff;
            border-color: rgba(255,255,255,0.3);
        }
        .permission-meta-btn.cancel:hover {
            background: rgba(239, 68, 68, 0.2);
            border-color: rgba(239, 68, 68, 0.4);
            color: #f87171;
        }
        .permission-add-link {
            font-size: 0.7rem;
            color: #888;
            text-decoration: none;
            margin-left: auto;
        }
        .permission-add-link:hover {
            color: #22c55e;
            text-decoration: underline;
        }

        /* Idle/Ready Panel - shown when Claude is waiting for new instructions */
        .idle-panel {
            display: none;
            background: var(--bg-card);
            border: 1px solid var(--accent-info);
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1rem;
        }
        .idle-panel.active {
            display: block;
        }
        .idle-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.75rem;
        }
        .idle-icon {
            font-size: 1.5rem;
        }
        .idle-title {
            font-weight: 600;
            color: var(--accent-info);
            font-size: 1.1rem;
        }
        .idle-reason {
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 1rem;
            padding: 0.5rem;
            background: var(--bg-secondary);
            border-radius: 6px;
        }
        .idle-input-group {
            display: flex;
            gap: 0.5rem;
        }
        .idle-input-group input {
            flex: 1;
            padding: 0.75rem 1rem;
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 8px;
            background: rgba(0,0,0,0.3);
            color: #fff;
            font-size: 16px;
        }
        [data-theme="light"] .idle-input-group input {
            border-color: rgba(59, 130, 246, 0.5);
            background: rgba(59, 130, 246, 0.1);
            color: #1d1d1f;
        }
        .idle-input-group input:focus {
            outline: none;
            border-color: #3b82f6;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
        }
        .idle-input-group button {
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 8px;
            background: #3b82f6;
            color: #fff;
            font-weight: 600;
            cursor: pointer;
            font-size: 1rem;
        }
        .idle-input-group button:hover {
            background: #2563eb;
        }
        .idle-meta {
            display: flex;
            justify-content: flex-end;
            margin-top: 0.75rem;
            font-size: 0.7rem;
            color: #666;
        }

        /* Mode toggle button */
        .mode-toggle {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            padding: 0.6rem 1rem;
            margin-bottom: 1rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        .mode-toggle:hover {
            background: rgba(0,0,0,0.4);
            border-color: rgba(255,255,255,0.2);
        }
        .mode-toggle:active {
            transform: scale(0.98);
        }
        .mode-info {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .mode-icon {
            font-size: 1.1rem;
            width: 1.5rem;
            text-align: center;
        }
        .mode-icon.plan { color: #22d3ee; }
        .mode-icon.edits { color: #a78bfa; }
        .mode-icon.normal { color: #9ca3af; }
        .mode-label {
            font-size: 0.9rem;
            color: #e5e7eb;
        }
        .mode-hint {
            font-size: 0.75rem;
            color: #6b7280;
        }
        /* Global mode indicator bar */
        .global-mode-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.5rem 1rem;
            background: #1f2937;
            border-radius: 8px;
            margin-bottom: 1rem;
            cursor: pointer;
            transition: background 0.2s;
        }
        .global-mode-bar:hover {
            background: #374151;
        }
        .global-mode-bar:active {
            background: #4b5563;
        }
        .global-mode-info {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .global-mode-icon {
            font-size: 1.2rem;
        }
        .global-mode-icon.plan { color: #22d3ee; }
        .global-mode-icon.edits { color: #a78bfa; }
        .global-mode-icon.normal { color: #6b7280; }
        .global-mode-text {
            font-size: 0.9rem;
            color: #e5e7eb;
        }
        .global-mode-hint {
            font-size: 0.75rem;
            color: #6b7280;
        }
        /* Context percentage indicator */
        .context-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.25rem 0.5rem;
            background: #374151;
            border-radius: 4px;
            font-size: 0.8rem;
        }
        .context-indicator.hidden { display: none; }
        .context-bar {
            width: 60px;
            height: 6px;
            background: #1f2937;
            border-radius: 3px;
            overflow: hidden;
        }
        .context-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.3s, background 0.3s;
        }
        .context-fill.high { background: #22c55e; }
        .context-fill.medium { background: #eab308; }
        .context-fill.low { background: #ef4444; }
        .context-text {
            color: #9ca3af;
            min-width: 2rem;
        }
    </style>
    <script>
        // Theme management - runs before body renders to prevent flash
        (function() {
            const savedTheme = localStorage.getItem('claude-remote-theme') || 'dark';
            if (savedTheme === 'light') {
                document.documentElement.setAttribute('data-theme', 'light');
            }
        })();
    </script>
</head>
<body>
    <!-- Compacting overlay - takes over the whole screen -->
    <div id="compacting-overlay" class="compacting-overlay">
        <div class="brain-large">🧠</div>
        <div class="title">Compacting Conversation</div>
        <div class="subtitle">Crunching context to fit more...</div>
    </div>

    <div id="refresh-indicator" title="Auto-refreshing"></div>
    <div class="container">
        <div class="page-header">
            <h1><a href="/" style="color: inherit; text-decoration: none;">Claude Remote Control</a></h1>
            <button class="theme-toggle" onclick="toggleTheme()" title="Toggle light/dark mode">
                <span id="theme-icon">🌙</span>
            </button>
        </div>

        <nav class="nav-links">
            <a href="/control?session={{ session }}" class="active">Control</a>
            <a href="/terminal?session={{ session }}">Terminal</a>
            <a href="/hooks?session={{ session }}">Hooks</a>
        </nav>

        <!-- Global Mode Indicator (always visible) -->
        <div class="global-mode-bar" onclick="cycleMode()" title="Click to cycle modes (shift+tab)">
            <div class="global-mode-info">
                <span class="global-mode-icon {{ 'plan' if initial_mode == 'plan' else ('edits' if initial_mode == 'edits' else 'normal') }}" id="global-mode-icon">
                    {{ '⏸' if initial_mode == 'plan' else ('▶▶' if initial_mode == 'edits' else '●') }}
                </span>
                <span class="global-mode-text" id="global-mode-text">
                    {{ 'Plan mode' if initial_mode == 'plan' else ('Auto-accept edits' if initial_mode == 'edits' else 'Normal mode') }}
                </span>
            </div>
            <span class="global-mode-hint">tap to cycle</span>
        </div>

        <!-- Context percentage indicator (only visible when context is tracked) -->
        <div id="context-indicator" class="context-indicator{% if not initial_context_left %} hidden{% endif %}">
            <span class="context-text" id="context-text">{{ initial_context_left if initial_context_left else '' }}%</span>
            <div class="context-bar">
                <div class="context-fill {% if initial_context_left and initial_context_left > 50 %}high{% elif initial_context_left and initial_context_left > 20 %}medium{% else %}low{% endif %}"
                     id="context-fill"
                     style="width: {{ initial_context_left if initial_context_left else 0 }}%"></div>
            </div>
            <span style="color: #6b7280; font-size: 0.7rem;">context left</span>
        </div>

        <div id="status-area">
            {% if is_compacting %}
            <div class="status compacting">
                <span class="brain-icon">🧠</span>
                <strong>Compacting conversation...</strong>
            </div>
            {% else %}
            <div id="status-working" class="status idle"{% if is_idle %} style="display: none;"{% endif %}>
                <strong>Watching for Claude activity...</strong>
            </div>
            {% endif %}
        </div>

        <!-- Rich Question Panel (populated via WebSocket from hook events) -->
        <div id="hook-question-panel" class="hook-question-panel">
            <div id="hook-question-header" class="hook-question-header"></div>
            <div id="hook-question-text" class="hook-question-text"></div>
            <div id="hook-question-options" class="hook-question-options"></div>
            <div class="hook-question-other">
                <input type="text" id="hook-other-input" placeholder="Or type your own response...">
                <button onclick="sendHookOtherInput()">Send</button>
            </div>
            <div class="hook-question-actions">
                <span class="hook-question-source">Via Hook Event</span>
                <button class="hook-question-escape" onclick="cancelHookQuestion()">
                    <span>⎋</span> Cancel
                </button>
            </div>
        </div>

        <!-- Tool Permission Panel -->
        <div id="permission-panel" class="permission-panel">
            <div class="permission-header">
                <span class="permission-icon">🔐</span>
                <span class="permission-title">Permission required for <span id="permission-tool" class="permission-tool"></span></span>
            </div>
            <div id="permission-detail" class="permission-detail"></div>
            <div id="permission-options" class="permission-options">
                <!-- Dynamically populated based on terminal content -->
            </div>
            <div id="permission-extra-input" class="permission-extra-input">
                <input type="text" id="permission-extra-text" placeholder="Additional instructions for Claude...">
                <button onclick="sendPermissionWithText()">Send</button>
            </div>
            <div class="permission-meta">
                <span>Via Hook Event</span>
                <button class="permission-meta-btn cancel" onclick="cancelPermission()">
                    <span>⎋</span> Cancel
                </button>
            </div>
        </div>

        <!-- Debug Panel - Hidden by default, enable via browser console: document.getElementById('debug-panel').style.display='block' -->
        <div id="debug-panel" style="display: none; background: #2a2a4a; border: 1px solid #666; border-radius: 8px; padding: 10px; margin: 10px 0; font-size: 12px; max-height: 150px; overflow-y: auto;">
            <strong style="color: #aaa;">Hook Events Debug:</strong>
            <div id="debug-events" style="color: #8f8; font-family: monospace; white-space: pre-wrap;"></div>
        </div>

        <!-- Idle/Ready Panel (populated via WebSocket from Stop events) -->
        <div id="idle-panel" class="idle-panel{% if is_idle %} active{% endif %}">
            <div class="idle-header">
                <span class="idle-icon">✨</span>
                <span class="idle-title">Claude is ready for a new task</span>
            </div>
            <div id="idle-reason" class="idle-reason" style="display: none;"></div>
            <div class="mode-toggle" onclick="cycleMode()" id="mode-toggle">
                <div class="mode-info">
                    <span class="mode-icon normal" id="mode-icon">●</span>
                    <span class="mode-label" id="mode-label">Normal mode</span>
                </div>
                <span class="mode-hint">⇧⇥ cycle</span>
            </div>
            <div class="idle-input-group">
                <input type="text" id="idle-input" placeholder="What would you like Claude to do?">
                <button onclick="sendIdleInput()">Send</button>
            </div>
            <div class="idle-meta">
                <span id="idle-meta-source">{% if is_idle %}Via Terminal Detection{% else %}Via Hook Event{% endif %}</span>
            </div>
        </div>

        <!-- Fallback actions -->
        <div class="actions two-col">
            <button class="btn btn-danger" onclick="sendEsc()">
                <span class="btn-num">⎋</span>
                <span class="btn-label">Cancel/Interrupt</span>
            </button>
            <button class="btn btn-mode" onclick="cycleMode()">
                <span class="btn-num" id="mode-btn-icon">⇧⇥</span>
                <span class="btn-label" id="mode-btn-label">Cycle Mode</span>
            </button>
        </div>

        <div class="links">
            {% if notification_mode == 'log_only' %}
            <span style="color: #666;">📝 Log Only Mode</span>
            {% else %}
            <a href="#" id="notify-toggle" onclick="toggleNotifications(); return false;">
                {% if notifications_paused %}🔕 Notifications Paused{% else %}🔔 Notifications On{% endif %}
            </a>
            {% endif %}
            <a href="/test-notify?session={{ session }}">Test Notify</a>
            <a href="/config">Config</a>
        </div>

        <div class="meta">
            Session: {{ session }} | Last: {{ now }} | {{ local_ip }}:{{ port }}
        </div>
    </div>

    <script>
        const currentSession = '{{ session }}';
        let userInteracting = false;  // Track if user is interacting (pauses auto-refresh)
        let currentHookQuestion = null;  // Current AskUserQuestion being displayed
        let currentPermissionEvent = null;  // Current tool permission prompt being displayed
        let isClaudeIdle = {{ 'true' if is_idle else 'false' }};  // Track if Claude is idle

        // Debug logging to visible panel
        function debugLog(msg) {
            const el = document.getElementById('debug-events');
            if (el) {
                const time = new Date().toLocaleTimeString();
                el.textContent = '[' + time + '] ' + msg + '\\n' + el.textContent;
                // Keep only last 20 lines
                const lines = el.textContent.split('\\n').slice(0, 20);
                el.textContent = lines.join('\\n');
            }
            console.log('[DEBUG]', msg);
        }

        // Immediate test - script is running
        debugLog('Script started');

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

        // handleOption removed - using hook-based handleHookOption instead

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

        // Current mode tracking - initialize from server-detected mode
        let currentMode = '{{ initial_mode }}';  // 'normal', 'plan', 'edits'

        function cycleMode() {
            // Just send shift+tab - UI will update from terminal content detection
            fetch('/send-shift-tab', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({session: currentSession})
            }).catch(err => console.error('cycleMode error:', err));
        }

        function updateModeDisplay(mode) {
            currentMode = mode;
            // Update idle panel display
            const iconEl = document.getElementById('mode-icon');
            const labelEl = document.getElementById('mode-label');
            // Update button display
            const btnIconEl = document.getElementById('mode-btn-icon');
            const btnLabelEl = document.getElementById('mode-btn-label');
            // Update global mode bar
            const globalIconEl = document.getElementById('global-mode-icon');
            const globalTextEl = document.getElementById('global-mode-text');

            if (mode === 'plan') {
                if (iconEl) { iconEl.textContent = '⏸'; iconEl.className = 'mode-icon plan'; }
                if (labelEl) labelEl.textContent = 'Plan mode on';
                if (btnIconEl) btnIconEl.textContent = '⏸';
                if (btnLabelEl) btnLabelEl.textContent = 'Plan Mode';
                if (globalIconEl) { globalIconEl.textContent = '⏸'; globalIconEl.className = 'global-mode-icon plan'; }
                if (globalTextEl) globalTextEl.textContent = 'Plan mode';
            } else if (mode === 'edits') {
                if (iconEl) { iconEl.textContent = '▶▶'; iconEl.className = 'mode-icon edits'; }
                if (labelEl) labelEl.textContent = 'Accept edits on';
                if (btnIconEl) btnIconEl.textContent = '▶▶';
                if (btnLabelEl) btnLabelEl.textContent = 'Auto-Accept';
                if (globalIconEl) { globalIconEl.textContent = '▶▶'; globalIconEl.className = 'global-mode-icon edits'; }
                if (globalTextEl) globalTextEl.textContent = 'Auto-accept edits';
            } else {
                if (iconEl) { iconEl.textContent = '●'; iconEl.className = 'mode-icon normal'; }
                if (labelEl) labelEl.textContent = 'Normal mode';
                if (btnIconEl) btnIconEl.textContent = '⇧⇥';
                if (btnLabelEl) btnLabelEl.textContent = 'Cycle Mode';
                if (globalIconEl) { globalIconEl.textContent = '●'; globalIconEl.className = 'global-mode-icon normal'; }
                if (globalTextEl) globalTextEl.textContent = 'Normal mode';
            }
        }

        function detectModeFromContent(content) {
            if (!content) return 'normal';
            // Strip trailing whitespace, then check last 500 chars for mode indicators
            const tail = content.trimEnd().slice(-500);
            if (tail.includes('accept edits on')) return 'edits';
            if (tail.includes('plan mode on')) return 'plan';
            return 'normal';
        }

        function updateContextIndicator(contextLeft) {
            const indicator = document.getElementById('context-indicator');
            const textEl = document.getElementById('context-text');
            const fillEl = document.getElementById('context-fill');

            if (!indicator) return;

            if (contextLeft === null || contextLeft === undefined) {
                indicator.classList.add('hidden');
                return;
            }

            indicator.classList.remove('hidden');
            if (textEl) textEl.textContent = contextLeft + '%';
            if (fillEl) {
                fillEl.style.width = contextLeft + '%';
                fillEl.className = 'context-fill';
                if (contextLeft > 50) {
                    fillEl.classList.add('high');
                } else if (contextLeft > 20) {
                    fillEl.classList.add('medium');
                } else {
                    fillEl.classList.add('low');
                }
            }
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

        // Smart auto-refresh: pause while user is typing or interacting
        const customInput = document.getElementById('custom-input');
        if (customInput) {
            customInput.addEventListener('keypress', e => {
                if(e.key === 'Enter') sendCustom();
            });
            customInput.addEventListener('focus', () => { userInteracting = true; });
            customInput.addEventListener('blur', () => { userInteracting = false; });
        }

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
                // Don't refresh if user is interacting, page not visible, or a dialog is shown
                if (!userInteracting && !currentHookQuestion && !currentPermissionEvent && !isClaudeIdle && document.visibilityState === 'visible') {
                    location.reload();
                } else {
                    // Check again later
                    scheduleRefresh();
                }
            }, 5000);
        }
        scheduleRefresh();

        // Fast polling for compacting state to show/hide overlay immediately
        const compactingOverlay = document.getElementById('compacting-overlay');
        let lastCompactingState = {{ 'true' if is_compacting else 'false' }};

        // Show overlay immediately if already compacting on page load
        if (lastCompactingState) {
            compactingOverlay.classList.add('active');
        }

        function checkCompactingState() {
            fetch('/api/status?session=' + encodeURIComponent(currentSession))
                .then(r => r.json())
                .then(data => {
                    const isCompacting = data.is_compacting || false;

                    if (isCompacting !== lastCompactingState) {
                        lastCompactingState = isCompacting;
                        if (isCompacting) {
                            compactingOverlay.classList.add('active');
                        } else {
                            compactingOverlay.classList.remove('active');
                        }
                    }

                    // Update mode display from terminal content
                    if (data.mode && data.mode !== currentMode) {
                        updateModeDisplay(data.mode);
                    }

                    // Update context percentage indicator
                    updateContextIndicator(data.context_left);
                })
                .catch(e => console.log('Status check failed:', e))
                .finally(() => {
                    // Poll every 1 second for responsive overlay
                    setTimeout(checkCompactingState, 1000);
                });
        }
        checkCompactingState();

        // Set initial mode display on page load
        updateModeDisplay(currentMode);

        // ============================================================
        // Hook-based Question UI
        // ============================================================
        let hookWs = null;

        function escapeHtml(str) {
            if (!str) return '';
            const div = document.createElement('div');
            div.textContent = str;
            return div.innerHTML;
        }

        // Escape string for use inside JavaScript single-quoted strings
        function escapeJs(str) {
            if (!str) return '';
            let result = '';
            for (let i = 0; i < str.length; i++) {
                const c = str[i];
                if (c === String.fromCharCode(92)) result += String.fromCharCode(92, 92); // backslash
                else if (c === String.fromCharCode(39)) result += String.fromCharCode(92, 39); // single quote
                else if (c === String.fromCharCode(34)) result += String.fromCharCode(92, 34); // double quote
                else if (c === String.fromCharCode(10)) result += String.fromCharCode(92, 110); // newline -> \n
                else if (c === String.fromCharCode(13)) result += String.fromCharCode(92, 114); // carriage return -> \r
                else result += c;
            }
            return result;
        }

        function showHookQuestion(questions) {
            if (!questions || questions.length === 0) return;

            // For now, handle the first question (multi-question support could be added)
            const q = questions[0];
            currentHookQuestion = q;

            const panel = document.getElementById('hook-question-panel');
            const headerEl = document.getElementById('hook-question-header');
            const textEl = document.getElementById('hook-question-text');
            const optionsEl = document.getElementById('hook-question-options');

            headerEl.textContent = q.header || 'Question';
            textEl.textContent = q.question || '';

            // Build options HTML
            let optionsHtml = '';
            if (q.options && q.options.length > 0) {
                q.options.forEach((opt, idx) => {
                    const num = idx + 1;
                    optionsHtml += `
                        <button class="hook-option-btn" onclick="handleHookOption(${num}, '${escapeJs(opt.label)}')">
                            <span class="hook-option-num">${num}</span>
                            <div class="hook-option-content">
                                <div class="hook-option-label">${escapeHtml(opt.label)}</div>
                                ${opt.description ? `<div class="hook-option-desc">${escapeHtml(opt.description)}</div>` : ''}
                            </div>
                        </button>
                    `;
                });
            }

            optionsEl.innerHTML = optionsHtml;

            // Clear the "Other" input field
            document.getElementById('hook-other-input').value = '';

            panel.classList.add('active');
        }

        function hideHookQuestion() {
            const panel = document.getElementById('hook-question-panel');
            panel.classList.remove('active');
            currentHookQuestion = null;
        }

        function handleHookOption(num, label) {
            // Send the option number to the terminal
            sendInput(String(num));
            hideHookQuestion();
        }

        function sendHookOtherInput() {
            const input = document.getElementById('hook-other-input');
            const customText = input.value.trim();
            if (customText) {
                // "Other" is the last option, so its number is options.length + 1
                const otherNum = (currentHookQuestion?.options?.length || 0) + 1;
                const fullText = String(otherNum) + customText;
                fetch('/send', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({input: fullText, session: currentSession})
                }).then(r => r.json()).then(d => {
                    hideHookQuestion();
                    if(d.success) location.reload();
                    else alert('Error: ' + d.error);
                }).catch(e => {
                    alert('Network error: ' + e);
                });
            }
        }

        function cancelHookQuestion() {
            // Send escape to cancel the question
            fetch('/send-esc', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({session: currentSession})
            }).then(r => r.json()).then(d => {
                hideHookQuestion();
                if(d.success) location.reload();
                else alert('Error: ' + d.error);
            });
        }

        // Handle Enter key in the "Other" input
        const hookOtherEl = document.getElementById('hook-other-input');
        if (hookOtherEl) {
            hookOtherEl.addEventListener('keypress', e => {
                if (e.key === 'Enter') sendHookOtherInput();
            });
            hookOtherEl.addEventListener('focus', () => { userInteracting = true; });
            hookOtherEl.addEventListener('blur', () => { userInteracting = false; });
        }

        // ============================================================
        // Tool Permission UI
        // ============================================================

        function buildPermissionButtons(options) {
            const container = document.getElementById('permission-options');
            if (!container) return;

            // Clear existing buttons
            container.innerHTML = '';

            if (options.length === 0) {
                // Fallback to default options if we couldn't parse any
                options = [
                    { num: '1', text: 'Yes' },
                    { num: '2', text: 'No' }
                ];
            }

            for (const opt of options) {
                const btn = document.createElement('button');
                // Determine button style based on text content
                let btnClass = 'permission-btn';
                const textLower = opt.text.toLowerCase();
                if (textLower.includes('no') || textLower.includes('deny') || textLower.includes('reject')) {
                    btnClass += ' no';
                } else if (textLower.includes('all') || textLower.includes('session') || textLower.includes('always')) {
                    btnClass += ' yes-all';
                } else {
                    btnClass += ' yes';
                }
                btn.className = btnClass;
                btn.onclick = () => handlePermissionOption(opt.num);

                btn.innerHTML = '<span class="permission-btn-num">' + opt.num + '</span>' +
                    '<span class="permission-btn-text">' + escapeHtml(opt.text) + '</span>' +
                    '<a href="#" class="permission-add-link" onclick="event.stopPropagation(); showPermissionInput(' + String.fromCharCode(39) + 'opt' + opt.num + String.fromCharCode(39) + '); return false;">+ add instructions</a>';

                container.appendChild(btn);
            }
        }

        function handlePermissionOption(num) {
            // Send the number key to select that option
            sendInput(num);
            hidePermissionPanel();
        }

        function showPermissionPanel(event) {
            debugLog('>>> SHOWING PERMISSION PANEL for ' + event.tool_name);
            currentPermissionEvent = event;
            const panel = document.getElementById('permission-panel');
            const toolEl = document.getElementById('permission-tool');
            const detailEl = document.getElementById('permission-detail');

            toolEl.textContent = event.tool_name || 'Unknown Tool';

            // Format the tool input for display
            let detailText = '';
            if (event.tool_input) {
                if (typeof event.tool_input === 'string') {
                    detailText = event.tool_input;
                } else {
                    detailText = JSON.stringify(event.tool_input, null, 2);
                }
            }
            detailEl.textContent = detailText || 'No details available';

            // Reset the extra input state
            document.getElementById('permission-extra-input').classList.remove('active');
            document.getElementById('permission-extra-text').value = '';

            // Use standard permission options - Claude always shows these same 3 options
            // Don't parse terminal content as it contains conversation history with numbered lists
            const standardOptions = [
                { num: '1', text: 'Yes' },
                { num: '2', text: "Yes, don't ask again" },
                { num: '3', text: 'No' }
            ];
            buildPermissionButtons(standardOptions);

            panel.classList.add('active');
        }

        function hidePermissionPanel() {
            document.getElementById('permission-panel').classList.remove('active');
            document.getElementById('permission-extra-input').classList.remove('active');
            currentPermissionEvent = null;
        }

        let pendingPermissionOption = '1';  // Track which option to send with instructions

        function showPermissionInput(action) {
            // action is like 'opt1', 'opt2', etc.
            pendingPermissionOption = action.replace('opt', '');
            const extraInput = document.getElementById('permission-extra-input');
            extraInput.classList.add('active');
            document.getElementById('permission-extra-text').focus();
        }

        function sendPermissionWithText() {
            const extraText = document.getElementById('permission-extra-text').value.trim();
            const optNum = pendingPermissionOption;

            if (extraText) {
                // Send Tab first to enable text input mode, then the text
                fetch('/send', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({input: '\\t', session: currentSession})
                }).then(() => {
                    // Small delay then send the actual text
                    setTimeout(() => {
                        fetch('/send', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({input: extraText + '\\n', session: currentSession})
                        }).then(r => r.json()).then(d => {
                            hidePermissionPanel();
                            if(d.success) location.reload();
                            else alert('Error: ' + d.error);
                        });
                    }, 100);
                });
            } else {
                // No text, just send the selected option number
                handlePermissionOption(optNum);
            }
        }

        function cancelPermission() {
            fetch('/send-esc', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({session: currentSession})
            }).then(r => r.json()).then(d => {
                hidePermissionPanel();
                if(d.success) location.reload();
                else alert('Error: ' + d.error);
            });
        }

        // Handle Enter key in permission extra input
        const permExtraEl = document.getElementById('permission-extra-text');
        if (permExtraEl) {
            permExtraEl.addEventListener('keypress', e => {
                if (e.key === 'Enter') sendPermissionWithText();
            });
            permExtraEl.addEventListener('focus', () => { userInteracting = true; });
            permExtraEl.addEventListener('blur', () => { userInteracting = false; });
        }

        // ============================================================
        // Idle/Ready Panel UI (for Stop events)
        // ============================================================

        function showIdlePanel(reason) {
            debugLog('Showing idle panel' + (reason ? ': ' + reason : ''));
            isClaudeIdle = true;
            const panel = document.getElementById('idle-panel');
            const reasonEl = document.getElementById('idle-reason');
            const statusEl = document.getElementById('status-working');
            const metaEl = document.getElementById('idle-meta-source');

            if (reason) {
                reasonEl.textContent = reason;
                reasonEl.style.display = 'block';
            } else {
                reasonEl.style.display = 'none';
            }

            panel.classList.add('active');
            if (statusEl) statusEl.style.display = 'none';
            if (metaEl) metaEl.textContent = 'Via Hook Event';
            document.getElementById('idle-input').value = '';
            document.getElementById('idle-input').focus();
        }

        function hideIdlePanel() {
            document.getElementById('idle-panel').classList.remove('active');
            const statusEl = document.getElementById('status-working');
            if (statusEl) {
                statusEl.style.display = 'block';
                statusEl.innerHTML = '<strong>Claude is working...</strong>';
            }
            isClaudeIdle = false;
        }

        function sendIdleInput() {
            const input = document.getElementById('idle-input').value.trim();
            if (!input) return;

            fetch('/send', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({input: input, session: currentSession})
            }).then(r => r.json()).then(d => {
                hideIdlePanel();
                if(d.success) {
                    // Don't reload - let hooks show new state
                } else {
                    alert('Error: ' + d.error);
                }
            });
        }

        // Handle Enter key in idle input
        const idleInputEl = document.getElementById('idle-input');
        if (idleInputEl) {
            idleInputEl.addEventListener('keypress', e => {
                if (e.key === 'Enter') sendIdleInput();
            });
            idleInputEl.addEventListener('focus', () => { userInteracting = true; });
            idleInputEl.addEventListener('blur', () => { userInteracting = false; });
        }

        function connectHookWebSocket() {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            hookWs = new WebSocket(`${protocol}//${location.host}/ws/hooks?session=${encodeURIComponent(currentSession)}`);

            // Track state for detecting unanswered questions
            let initialBatchEvents = [];
            let initialBatchTimeout = null;
            let isRealTime = false;

            hookWs.onopen = () => {
                console.log('Hook WebSocket connected');
                debugLog('WS CONNECTED - waiting for initial batch...');
                // After initial batch arrives, process it to find unanswered questions
                initialBatchTimeout = setTimeout(() => {
                    debugLog('Processing batch of ' + initialBatchEvents.length + ' events');
                    processInitialBatch();
                    isRealTime = true;
                    debugLog('NOW IN REAL-TIME MODE');
                    console.log('Hook WebSocket now in real-time mode');
                }, 300);
            };

            function processInitialBatch() {
                // Find if there's an unanswered AskUserQuestion
                // Look for PreToolUse without a matching PostToolUse
                const askEvents = initialBatchEvents.filter(e =>
                    e.tool_name === 'AskUserQuestion'
                );

                // Get the last PreToolUse and last PostToolUse for AskUserQuestion
                let lastAskPre = null;
                let lastAskPost = null;
                for (const e of askEvents) {
                    if (e.event_type === 'PreToolUse') lastAskPre = e;
                    if (e.event_type === 'PostToolUse') lastAskPost = e;
                }

                // If there's a PreToolUse that's newer than the last PostToolUse, show it
                // But only if there's no Stop event after the question (which means it was abandoned)
                const stopEvents = initialBatchEvents.filter(e => e.event_type === 'Stop');
                const lastStopEvent = stopEvents.length > 0 ? stopEvents[stopEvents.length - 1] : null;

                if (lastAskPre && (!lastAskPost || lastAskPre.timestamp > lastAskPost.timestamp)) {
                    // Don't show if there's a Stop event after the question
                    if (!lastStopEvent || lastAskPre.timestamp > lastStopEvent.timestamp) {
                        const toolInput = lastAskPre.tool_input;
                        if (toolInput && toolInput.questions) {
                            console.log('Found unanswered question from initial batch');
                            showHookQuestion(toolInput.questions);
                        }
                    }
                }

                // Also check for unanswered permission prompts
                // These are PreToolUse events with any permission_mode and NOT AskUserQuestion
                const permissionEvents = initialBatchEvents.filter(e =>
                    e.tool_name !== 'AskUserQuestion' && e.permission_mode
                );

                // Find the most recent PreToolUse with permission_mode that doesn't have a PostToolUse
                let lastPermPre = null;
                let lastPermPost = null;
                for (const e of permissionEvents) {
                    if (e.event_type === 'PreToolUse') lastPermPre = e;
                }
                // Check for PostToolUse of the same tool
                for (const e of initialBatchEvents) {
                    if (e.event_type === 'PostToolUse' && lastPermPre && e.tool_name === lastPermPre.tool_name) {
                        if (!lastPermPost || e.timestamp > lastPermPost.timestamp) {
                            lastPermPost = e;
                        }
                    }
                }

                // If there's a permission PreToolUse newer than its PostToolUse, show it
                debugLog('Batch check: lastPermPre=' + (lastPermPre ? lastPermPre.tool_name : 'null') + ', lastPermPost=' + (lastPermPost ? lastPermPost.tool_name : 'null'));
                if (lastPermPre && (!lastPermPost || lastPermPre.timestamp > lastPermPost.timestamp)) {
                    debugLog('Found unanswered permission prompt from batch: ' + lastPermPre.tool_name);
                    showPermissionPanel(lastPermPre);
                }

                // Check for Stop event (Claude is idle)
                // If the last event is Stop and there's no subsequent PreToolUse, show idle panel
                // (lastStopEvent is already defined above)
                if (lastStopEvent) {
                    // Check if there's a PreToolUse after this Stop
                    const hasLaterPreToolUse = initialBatchEvents.some(e =>
                        e.event_type === 'PreToolUse' && e.timestamp > lastStopEvent.timestamp
                    );
                    if (!hasLaterPreToolUse) {
                        debugLog('Found idle state from batch: ' + (lastStopEvent.reason || 'no reason'));
                        showIdlePanel(lastStopEvent.reason);
                    }
                }

                // Clear the batch
                initialBatchEvents = [];
            }

            hookWs.onclose = () => {
                console.log('Hook WebSocket disconnected, reconnecting...');
                if (initialBatchTimeout) clearTimeout(initialBatchTimeout);
                setTimeout(connectHookWebSocket, 3000);
            };

            hookWs.onerror = (e) => {
                console.log('Hook WebSocket error:', e);
            };

            hookWs.onmessage = (e) => {
                // Skip non-JSON messages (like 'pong' responses)
                if (!e.data || !e.data.startsWith('{')) {
                    debugLog('Non-JSON: ' + (e.data ? e.data.substring(0,20) : 'null'));
                    return;
                }

                try {
                    const event = JSON.parse(e.data);
                    // Visible debug logging
                    debugLog(`${event.event_type} | ${event.tool_name} | perm=${event.permission_mode} | rt=${isRealTime}`);
                    console.log('Hook event:', event.event_type, event.tool_name, 'isRealTime:', isRealTime, 'permission_mode:', event.permission_mode);

                    if (!isRealTime) {
                        // Collecting initial batch
                        initialBatchEvents.push(event);
                        return;
                    }

                    // Real-time mode: process events immediately

                    // Check for AskUserQuestion PreToolUse event
                    if (event.event_type === 'PreToolUse' && event.tool_name === 'AskUserQuestion') {
                        const toolInput = event.tool_input;
                        if (toolInput && toolInput.questions) {
                            showHookQuestion(toolInput.questions);
                        }
                    }

                    // Check for permission-requiring PreToolUse events (not AskUserQuestion)
                    // permission_mode can be 'ask', 'default', or other values - show panel for any non-null
                    if (event.event_type === 'PreToolUse' &&
                        event.tool_name !== 'AskUserQuestion' &&
                        event.permission_mode) {
                        debugLog('Showing permission panel for: ' + event.tool_name);
                        showPermissionPanel(event);
                    }

                    // Hide question panel when we get a PostToolUse for AskUserQuestion
                    if (event.event_type === 'PostToolUse' && event.tool_name === 'AskUserQuestion') {
                        hideHookQuestion();
                    }

                    // Hide permission panel when we get a PostToolUse for the permission tool
                    if (event.event_type === 'PostToolUse' && currentPermissionEvent &&
                        event.tool_name === currentPermissionEvent.tool_name) {
                        hidePermissionPanel();
                    }

                    // Show idle panel when Claude stops (ready for new input)
                    // Also hide any pending question/permission panels since they're no longer active
                    if (event.event_type === 'Stop') {
                        hideHookQuestion();
                        hidePermissionPanel();
                        showIdlePanel(event.reason);
                    }

                    // Hide idle panel when Claude starts working (PreToolUse)
                    if (event.event_type === 'PreToolUse' && isClaudeIdle) {
                        hideIdlePanel();
                    }

                } catch (err) {
                    console.error('Failed to parse hook event:', err);
                }
            };
        }

        // Start hook WebSocket connection
        debugLog('About to connect WebSocket...');
        connectHookWebSocket();
        debugLog('connectHookWebSocket() called');

        // Keep hook WebSocket alive
        setInterval(() => {
            if (hookWs && hookWs.readyState === WebSocket.OPEN) {
                hookWs.send('ping');
            }
        }, 25000);

        // Theme toggle
        function toggleTheme() {
            const html = document.documentElement;
            const currentTheme = html.getAttribute('data-theme');
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';
            if (newTheme === 'dark') {
                html.removeAttribute('data-theme');
            } else {
                html.setAttribute('data-theme', 'light');
            }
            localStorage.setItem('claude-remote-theme', newTheme);
            updateThemeIcon();
        }

        function updateThemeIcon() {
            const isDark = !document.documentElement.getAttribute('data-theme');
            document.getElementById('theme-icon').textContent = isDark ? '🌙' : '☀️';
        }

        // Initialize theme icon
        updateThemeIcon();
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
        /* Theme CSS Variables */
        :root {
            --bg-primary: #0d0d1a;
            --bg-secondary: #1a1a2e;
            --bg-tertiary: #1a1a3e;
            --text-primary: #eee;
            --text-secondary: #ccc;
            --text-muted: #666;
            --text-heading: #ff6b6b;
            --border-color: #333;
            --accent-primary: #6366f1;
            --accent-info: #60a5fa;
        }
        [data-theme="light"] {
            --bg-primary: #ffffff;
            --bg-secondary: #f5f5f7;
            --bg-tertiary: #e5e5ea;
            --text-primary: #1d1d1f;
            --text-secondary: #424245;
            --text-muted: #6e6e73;
            --text-heading: #dc2626;
            --border-color: #d1d1d6;
            --accent-primary: #6366f1;
            --accent-info: #2563eb;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: var(--bg-primary);
            height: 100vh;
            display: flex;
            flex-direction: column;
            transition: background 0.3s;
        }
        .header {
            background: var(--bg-secondary);
            padding: 0.5rem 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
        }
        .header h1 { font-size: 1rem; color: var(--text-heading); margin: 0; }
        .header-left { display: flex; align-items: center; gap: 1rem; }
        .theme-toggle {
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 4px 8px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9rem;
        }
        .nav-links {
            display: flex;
            gap: 0.5rem;
        }
        .nav-links a {
            color: var(--accent-info);
            text-decoration: none;
            padding: 0.35rem 0.75rem;
            background: var(--bg-primary);
            border-radius: 4px;
            font-size: 0.8rem;
        }
        .nav-links a:hover { background: var(--bg-tertiary); }
        .nav-links a.active {
            background: var(--accent-primary);
            color: white;
        }
        #terminal-container {
            flex: 1;
            padding: 0.5rem;
            overflow: hidden;
        }
        #terminal { height: 100%; }
        #terminal-output {
            height: 100%;
            overflow-y: auto;
            background: var(--bg-primary);
            color: var(--text-primary);
            font-family: SF Mono, Monaco, Inconsolata, monospace;
            font-size: 12px;
            line-height: 1.4;
            white-space: pre-wrap;
            word-wrap: break-word;
            padding: 0.5rem;
        }
        .status-bar {
            background: var(--bg-secondary);
            padding: 0.25rem 1rem;
            font-size: 0.75rem;
            color: var(--text-muted);
            border-top: 1px solid var(--border-color);
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
    <script>
        // Theme management - runs before body renders to prevent flash
        (function() {
            const savedTheme = localStorage.getItem('claude-remote-theme') || 'dark';
            if (savedTheme === 'light') {
                document.documentElement.setAttribute('data-theme', 'light');
            }
        })();
    </script>
</head>
<body>
    <div class="header">
        <div class="header-left">
            <h1><a href="/" style="color: inherit; text-decoration: none;">Claude Remote Control</a></h1>
            <button class="theme-toggle" onclick="toggleTheme()" title="Toggle light/dark mode">
                <span id="theme-icon">🌙</span>
            </button>
        </div>
        <nav class="nav-links">
            <a href="/control?session={{ session }}">Control</a>
            <a href="/terminal?session={{ session }}" class="active">Terminal</a>
            <a href="/hooks?session={{ session }}">Hooks</a>
        </nav>
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
        // Polling mode for accessibility backend - simple fetch-based content updates
        const output = document.getElementById('terminal-output');
        const statusEl = document.getElementById('ws-status');

        function stripAnsi(str) {
            // Remove ANSI escape codes for cleaner display
            return str.replace(/\\x1b\\[[0-9;]*[a-zA-Z]/g, '');
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

        // Theme toggle
        function toggleTheme() {
            const html = document.documentElement;
            const currentTheme = html.getAttribute('data-theme');
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';
            if (newTheme === 'dark') {
                html.removeAttribute('data-theme');
            } else {
                html.setAttribute('data-theme', 'light');
            }
            localStorage.setItem('claude-remote-theme', newTheme);
            updateThemeIcon();
        }

        function updateThemeIcon() {
            const isDark = !document.documentElement.getAttribute('data-theme');
            document.getElementById('theme-icon').textContent = isDark ? '🌙' : '☀️';
        }
        updateThemeIcon();
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

        // Theme toggle
        function toggleTheme() {
            const html = document.documentElement;
            const currentTheme = html.getAttribute('data-theme');
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';
            if (newTheme === 'dark') {
                html.removeAttribute('data-theme');
            } else {
                html.setAttribute('data-theme', 'light');
            }
            localStorage.setItem('claude-remote-theme', newTheme);
            updateThemeIcon();
        }

        function updateThemeIcon() {
            const isDark = !document.documentElement.getAttribute('data-theme');
            document.getElementById('theme-icon').textContent = isDark ? '🌙' : '☀️';
        }
        updateThemeIcon();

        connect();
    </script>
    {% endif %}
</body>
</html>
"""

HOOKS_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hook Events - Claude Remote Control</title>
    <style>
        /* Theme CSS Variables */
        :root {
            --bg-primary: #1a1a2e;
            --bg-secondary: #0d0d1a;
            --bg-tertiary: #1a1a3e;
            --text-primary: #eee;
            --text-secondary: #ccc;
            --text-muted: #888;
            --text-heading: #ff6b6b;
            --border-color: #333;
            --accent-primary: #6366f1;
            --accent-info: #60a5fa;
        }
        [data-theme="light"] {
            --bg-primary: #f5f5f7;
            --bg-secondary: #ffffff;
            --bg-tertiary: #e5e5ea;
            --text-primary: #1d1d1f;
            --text-secondary: #424245;
            --text-muted: #6e6e73;
            --text-heading: #dc2626;
            --border-color: #d1d1d6;
            --accent-primary: #6366f1;
            --accent-info: #2563eb;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            transition: background 0.3s, color 0.3s;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 1rem;
        }
        .page-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }
        h1 { color: var(--text-heading); }
        .theme-toggle {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 8px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1rem;
        }
        .theme-toggle:hover { background: var(--bg-tertiary); }
        .nav-links {
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }
        .nav-links a {
            color: var(--accent-info);
            text-decoration: none;
            padding: 0.5rem 1rem;
            background: var(--bg-secondary);
            border-radius: 6px;
        }
        .nav-links a:hover { background: var(--bg-tertiary); }
        .nav-links a.active {
            background: var(--accent-primary);
            color: white;
        }
        .status-bar {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 0.75rem 1rem;
            background: var(--bg-secondary);
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #4ade80;
            animation: pulse 2s infinite;
        }
        .status-dot.disconnected { background: #f87171; animation: none; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .filter-bar {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin-bottom: 1rem;
        }
        .filter-btn {
            padding: 0.4rem 0.8rem;
            border: 1px solid var(--border-color);
            background: var(--bg-secondary);
            color: var(--text-secondary);
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85rem;
        }
        .filter-btn:hover { border-color: #6366f1; }
        .filter-btn.active {
            background: #6366f1;
            border-color: #6366f1;
            color: white;
        }
        .events-table {
            width: 100%;
            background: #0d0d1a;
            border-radius: 8px;
            overflow: hidden;
        }
        .events-table table {
            width: 100%;
            border-collapse: collapse;
        }
        .events-table th {
            text-align: left;
            padding: 0.75rem 1rem;
            background: #16162a;
            color: #888;
            font-weight: 500;
            font-size: 0.85rem;
            text-transform: uppercase;
        }
        .events-table td {
            padding: 0.6rem 1rem;
            border-top: 1px solid #222;
            font-size: 0.9rem;
        }
        .events-table tr:hover { background: #16162a; }
        .event-type {
            display: inline-block;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 500;
        }
        .event-type.Stop { background: #22c55e22; color: #4ade80; }
        .event-type.PreToolUse { background: #3b82f622; color: #60a5fa; }
        .event-type.PostToolUse { background: #6b728022; color: #9ca3af; }
        .event-type.Notification { background: #f59e0b22; color: #fbbf24; }
        .event-type.SessionStart { background: #8b5cf622; color: #a78bfa; }
        .event-type.SessionEnd { background: #ef444422; color: #f87171; }
        .event-type.UserPromptSubmit { background: #06b6d422; color: #22d3ee; }
        .event-type.PreCompact { background: #d946ef22; color: #e879f9; }
        .tool-name {
            font-family: 'SF Mono', Monaco, monospace;
            color: #fbbf24;
        }
        .perm-badge {
            margin-left: 0.4rem;
            font-size: 0.75rem;
        }
        .session-id {
            font-family: 'SF Mono', Monaco, monospace;
            color: #888;
            font-size: 0.8rem;
        }
        .timestamp {
            color: #666;
            font-size: 0.85rem;
            font-family: 'SF Mono', Monaco, monospace;
        }
        .detail-text {
            color: #aaa;
            max-width: 400px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .empty-state {
            text-align: center;
            padding: 3rem;
            color: #666;
        }
        .event-count {
            color: #888;
            font-size: 0.9rem;
        }
        .clear-btn {
            padding: 0.4rem 0.8rem;
            border: 1px solid #ef4444;
            background: transparent;
            color: #ef4444;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85rem;
            margin-left: auto;
        }
        .clear-btn:hover { background: #ef444422; }
        .events-table tr { cursor: pointer; }
        .events-table tr.selected { background: #1a1a3e !important; }

        /* Inline detail expansion */
        .event-detail-row td {
            padding: 0 !important;
            border-top: none !important;
        }
        .event-detail-content {
            background: #16162a;
            padding: 1rem;
            border-top: 1px solid #333;
            animation: expandIn 0.2s ease;
        }
        @keyframes expandIn {
            from { opacity: 0; max-height: 0; }
            to { opacity: 1; max-height: 500px; }
        }
        .detail-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }
        .detail-item {
            min-width: 0;
        }
        .detail-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            color: #666;
            margin-bottom: 0.25rem;
            font-weight: 600;
        }
        .detail-value {
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 0.85rem;
            color: #eee;
            word-break: break-word;
        }
        .detail-value.json {
            background: #0d0d1a;
            padding: 0.75rem;
            border-radius: 6px;
            white-space: pre-wrap;
            max-height: 200px;
            overflow-y: auto;
            font-size: 0.8rem;
        }
        .detail-value.reason {
            color: #fbbf24;
        }
        .detail-full-width {
            grid-column: 1 / -1;
        }
    </style>
    <script>
        // Theme management - runs before body renders to prevent flash
        (function() {
            const savedTheme = localStorage.getItem('claude-remote-theme') || 'dark';
            if (savedTheme === 'light') {
                document.documentElement.setAttribute('data-theme', 'light');
            }
        })();
    </script>
</head>
<body>
    <div class="container">
        <div class="page-header">
            <h1><a href="/" style="color: inherit; text-decoration: none;">Claude Remote Control</a></h1>
            <button class="theme-toggle" onclick="toggleTheme()" title="Toggle light/dark mode">
                <span id="theme-icon">🌙</span>
            </button>
        </div>
        <nav class="nav-links">
            <a href="/control{% if session %}?session={{ session }}{% endif %}">Control</a>
            <a href="/terminal{% if session %}?session={{ session }}{% endif %}">Terminal</a>
            <a href="/hooks{% if session %}?session={{ session }}{% endif %}" class="active">Hooks</a>
        </nav>

        <div class="status-bar">
            <div class="status-indicator">
                <div class="status-dot" id="wsStatus"></div>
                <span id="wsStatusText">Connecting...</span>
            </div>
            <span class="event-count"><span id="eventCount">0</span> events</span>
            <button class="clear-btn" onclick="clearEvents()">Clear</button>
        </div>

        <div class="filter-bar">
            <button class="filter-btn active" data-filter="all">All</button>
            <button class="filter-btn" data-filter="Stop">Stop</button>
            <button class="filter-btn" data-filter="PreToolUse">PreToolUse</button>
            <button class="filter-btn" data-filter="PostToolUse">PostToolUse</button>
            <button class="filter-btn" data-filter="Notification">Notification</button>
        </div>

        <div class="events-table">
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Event</th>
                        <th>Tool</th>
                        <th>Session</th>
                        <th>Details</th>
                    </tr>
                </thead>
                <tbody id="eventsBody">
                    <tr class="empty-state-row">
                        <td colspan="5" class="empty-state">
                            Waiting for hook events...<br>
                            <small>Make sure Claude Code hooks are configured to POST to this server.</small>
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>

    <script>
        let events = [];
        let activeFilter = 'all';
        let ws = null;
        let expandedEventId = null;
        // Get session filter from URL if present
        const urlParams = new URLSearchParams(window.location.search);
        const sessionFilter = urlParams.get('session');

        function formatTime(isoString) {
            const d = new Date(isoString);
            return d.toLocaleTimeString();
        }

        function formatFullTime(isoString) {
            const d = new Date(isoString);
            return d.toLocaleString();
        }

        function truncate(str, len) {
            if (!str) return '';
            return str.length > len ? str.substring(0, len) + '...' : str;
        }

        function escapeHtml(str) {
            if (!str) return '';
            const div = document.createElement('div');
            div.textContent = str;
            return div.innerHTML;
        }

        function formatJson(obj) {
            if (!obj) return '';
            try {
                const str = typeof obj === 'string' ? obj : JSON.stringify(obj, null, 2);
                return escapeHtml(str);
            } catch (e) {
                return escapeHtml(String(obj));
            }
        }

        function getEventDetail(event) {
            if (event.reason) return event.reason;
            if (event.tool_input) {
                const input = typeof event.tool_input === 'string'
                    ? event.tool_input
                    : JSON.stringify(event.tool_input);
                return truncate(input, 60);
            }
            if (event.tool_result) return truncate(event.tool_result, 60);
            return '';
        }

        function buildDetailHtml(event) {
            let html = '<div class="detail-grid">';

            // Row 1: Timestamp, Event Type, Session
            html += `<div class="detail-item">
                <div class="detail-label">Timestamp</div>
                <div class="detail-value">${formatFullTime(event.timestamp)}</div>
            </div>`;
            html += `<div class="detail-item">
                <div class="detail-label">Event Type</div>
                <div class="detail-value"><span class="event-type ${event.event_type}">${event.event_type}</span></div>
            </div>`;
            html += `<div class="detail-item">
                <div class="detail-label">Session ID</div>
                <div class="detail-value">${escapeHtml(event.session_id)}</div>
            </div>`;

            // Tool Name (if present)
            if (event.tool_name) {
                html += `<div class="detail-item">
                    <div class="detail-label">Tool Name</div>
                    <div class="detail-value" style="color: #fbbf24;">${escapeHtml(event.tool_name)}</div>
                </div>`;
            }

            // Permission Mode (if present)
            if (event.permission_mode) {
                const permColor = event.permission_mode === 'ask' ? '#f87171' : '#4ade80';
                const permLabel = event.permission_mode === 'ask' ? 'Permission Required' : event.permission_mode;
                html += `<div class="detail-item">
                    <div class="detail-label">Permission</div>
                    <div class="detail-value" style="color: ${permColor};">${escapeHtml(permLabel)}</div>
                </div>`;
            }

            // Reason (if present)
            if (event.reason) {
                html += `<div class="detail-item detail-full-width">
                    <div class="detail-label">Reason</div>
                    <div class="detail-value reason">${escapeHtml(event.reason)}</div>
                </div>`;
            }

            // Tool Input (if present)
            if (event.tool_input) {
                html += `<div class="detail-item detail-full-width">
                    <div class="detail-label">Tool Input</div>
                    <div class="detail-value json">${formatJson(event.tool_input)}</div>
                </div>`;
            }

            // Tool Result (if present)
            if (event.tool_result) {
                html += `<div class="detail-item detail-full-width">
                    <div class="detail-label">Tool Result</div>
                    <div class="detail-value json">${escapeHtml(event.tool_result)}</div>
                </div>`;
            }

            html += '</div>';
            return html;
        }

        function toggleEventDetail(eventId, rowElement) {
            const existingDetail = document.getElementById('detail-' + eventId);

            // If clicking the same row, collapse it
            if (existingDetail) {
                existingDetail.remove();
                rowElement.classList.remove('selected');
                expandedEventId = null;
                return;
            }

            // Collapse any previously expanded row
            const prevDetail = document.querySelector('.event-detail-row');
            if (prevDetail) {
                prevDetail.remove();
            }
            document.querySelectorAll('.events-table tbody tr.selected').forEach(tr => {
                tr.classList.remove('selected');
            });

            // Find the event
            const filtered = activeFilter === 'all'
                ? events
                : events.filter(e => e.event_type === activeFilter);
            const event = filtered.find(e => e.id === eventId);
            if (!event) return;

            // Create detail row
            const detailRow = document.createElement('tr');
            detailRow.className = 'event-detail-row';
            detailRow.id = 'detail-' + eventId;
            detailRow.innerHTML = `<td colspan="5"><div class="event-detail-content">${buildDetailHtml(event)}</div></td>`;

            // Insert after the clicked row
            rowElement.classList.add('selected');
            rowElement.after(detailRow);
            expandedEventId = eventId;
        }

        // Close expanded detail with Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && expandedEventId) {
                const detail = document.querySelector('.event-detail-row');
                if (detail) detail.remove();
                document.querySelectorAll('.events-table tbody tr.selected').forEach(tr => {
                    tr.classList.remove('selected');
                });
                expandedEventId = null;
            }
        });

        function renderEvents() {
            const tbody = document.getElementById('eventsBody');
            // First filter by session if sessionFilter is set
            // Note: sessionFilter from URL could be TTY (ttys003) or session_id (UUID)
            // We check both the tty field in raw_data and session_id
            let sessionFiltered = sessionFilter
                ? events.filter(e => {
                    // Check if it matches TTY (normalize both for comparison)
                    const eventTty = e.raw_data?.tty || '';
                    const normalizedFilter = sessionFilter.startsWith('tty') ? sessionFilter.slice(3) : sessionFilter;
                    const normalizedTty = eventTty.startsWith('tty') ? eventTty.slice(3) : eventTty;
                    if (normalizedTty && normalizedTty === normalizedFilter) return true;
                    // Also check session_id for backward compatibility
                    if (e.session_id && e.session_id.includes(sessionFilter)) return true;
                    return false;
                })
                : events;
            // Then filter by event type
            const filtered = activeFilter === 'all'
                ? sessionFiltered
                : sessionFiltered.filter(e => e.event_type === activeFilter);

            if (filtered.length === 0) {
                tbody.innerHTML = `
                    <tr class="empty-state-row">
                        <td colspan="5" class="empty-state">
                            No ${activeFilter === 'all' ? '' : activeFilter + ' '}events yet
                        </td>
                    </tr>`;
                expandedEventId = null;
                return;
            }

            // Assign unique IDs to events if not present
            filtered.forEach((e, i) => {
                if (!e.id) e.id = 'evt-' + i + '-' + Date.now();
            });

            tbody.innerHTML = filtered.slice().reverse().map((e) => {
                const permBadge = e.permission_mode === 'ask' ? '<span class="perm-badge">🔐</span>' : '';
                return `
                <tr onclick="toggleEventDetail('${e.id}', this)" class="${e.id === expandedEventId ? 'selected' : ''}">
                    <td class="timestamp">${formatTime(e.timestamp)}</td>
                    <td><span class="event-type ${e.event_type}">${e.event_type}</span></td>
                    <td class="tool-name">${e.tool_name || '-'}${permBadge}</td>
                    <td class="session-id">${truncate(e.session_id, 12)}</td>
                    <td class="detail-text">${escapeHtml(getEventDetail(e))}</td>
                </tr>`;
            }).join('');

            document.getElementById('eventCount').textContent = events.length;

            // Re-expand the previously expanded event if it still exists
            if (expandedEventId) {
                const row = document.querySelector(`tr[onclick*="${expandedEventId}"]`);
                if (row) {
                    const event = filtered.find(e => e.id === expandedEventId);
                    if (event) {
                        const detailRow = document.createElement('tr');
                        detailRow.className = 'event-detail-row';
                        detailRow.id = 'detail-' + expandedEventId;
                        detailRow.innerHTML = `<td colspan="5"><div class="event-detail-content">${buildDetailHtml(event)}</div></td>`;
                        row.classList.add('selected');
                        row.after(detailRow);
                    }
                } else {
                    expandedEventId = null;
                }
            }
        }

        function addEvent(event) {
            events.push(event);
            // Keep last 500 events in UI
            if (events.length > 500) {
                events = events.slice(-500);
            }
            renderEvents();
        }

        function clearEvents() {
            events = [];
            renderEvents();
        }

        function connectWebSocket() {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${location.host}/ws/hooks${location.search}`);

            ws.onopen = () => {
                document.getElementById('wsStatus').classList.remove('disconnected');
                document.getElementById('wsStatusText').textContent = 'Connected';
            };

            ws.onclose = () => {
                document.getElementById('wsStatus').classList.add('disconnected');
                document.getElementById('wsStatusText').textContent = 'Disconnected - reconnecting...';
                setTimeout(connectWebSocket, 3000);
            };

            ws.onerror = () => {
                document.getElementById('wsStatus').classList.add('disconnected');
                document.getElementById('wsStatusText').textContent = 'Connection error';
            };

            ws.onmessage = (e) => {
                // Skip non-JSON messages (like 'pong' responses)
                if (!e.data || !e.data.startsWith('{')) {
                    return;
                }
                try {
                    const event = JSON.parse(e.data);
                    addEvent(event);
                } catch (err) {
                    console.error('Failed to parse event:', err);
                }
            };
        }

        // Filter buttons
        document.querySelectorAll('.filter-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                activeFilter = btn.dataset.filter;
                renderEvents();
            });
        });

        // Start WebSocket connection
        connectWebSocket();

        // Ping to keep connection alive
        setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send('ping');
            }
        }, 25000);

        // Theme toggle
        function toggleTheme() {
            const html = document.documentElement;
            const currentTheme = html.getAttribute('data-theme');
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';
            if (newTheme === 'dark') {
                html.removeAttribute('data-theme');
            } else {
                html.setAttribute('data-theme', 'light');
            }
            localStorage.setItem('claude-remote-theme', newTheme);
            updateThemeIcon();
        }

        function updateThemeIcon() {
            const isDark = !document.documentElement.getAttribute('data-theme');
            document.getElementById('theme-icon').textContent = isDark ? '🌙' : '☀️';
        }
        updateThemeIcon();
    </script>
</body>
</html>
"""

SESSIONS_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Sessions - Claude Remote Control</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        /* Theme CSS Variables */
        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --bg-input: #0f172a;
            --text-primary: #e2e8f0;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --border-color: #334155;
            --border-hover: #475569;
            --accent-primary: #6366f1;
            --accent-hover: #4f46e5;
            --accent-success: #22c55e;
            --accent-warning: #f59e0b;
            --accent-danger: #ef4444;
            --shadow-color: rgba(0, 0, 0, 0.3);
            --card-bg: #1e293b;
            --status-success-bg: #065f46;
            --status-success-text: #a7f3d0;
            --status-error-bg: #7f1d1d;
            --status-error-text: #fecaca;
        }
        [data-theme="light"] {
            --bg-primary: #f8fafc;
            --bg-secondary: #ffffff;
            --bg-tertiary: #e2e8f0;
            --bg-input: #ffffff;
            --text-primary: #1e293b;
            --text-secondary: #475569;
            --text-muted: #64748b;
            --border-color: #cbd5e1;
            --border-hover: #94a3b8;
            --accent-primary: #6366f1;
            --accent-hover: #4f46e5;
            --accent-success: #16a34a;
            --accent-warning: #d97706;
            --accent-danger: #dc2626;
            --shadow-color: rgba(0, 0, 0, 0.1);
            --card-bg: #ffffff;
            --status-success-bg: #dcfce7;
            --status-success-text: #166534;
            --status-error-bg: #fee2e2;
            --status-error-text: #991b1b;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 20px;
            transition: background 0.3s, color 0.3s;
        }
        .container { max-width: 900px; margin: 0 auto; }
        .page-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        h1 { font-size: 1.5rem; }
        .theme-toggle {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 8px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1rem;
            transition: all 0.2s;
        }
        .theme-toggle:hover { background: var(--bg-tertiary); }
        .nav-links {
            display: flex;
            gap: 15px;
            margin-bottom: 30px;
            padding-bottom: 15px;
            border-bottom: 1px solid var(--border-color);
        }
        .nav-links a {
            color: var(--text-secondary);
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 6px;
            transition: all 0.2s;
        }
        .nav-links a:hover { background: var(--bg-secondary); color: var(--text-primary); }
        .nav-links a.active { background: var(--accent-primary); color: white; }

        .sessions-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .session-count {
            color: var(--text-muted);
            font-size: 0.875rem;
        }
        .refresh-btn {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.875rem;
        }
        .refresh-btn:hover { background: var(--bg-tertiary); }

        .sessions-list {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .session-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        .session-card:hover { border-color: var(--border-hover); box-shadow: 0 2px 8px var(--shadow-color); }

        .session-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 12px;
        }
        .session-name-section { flex: 1; }
        .session-display-name {
            font-size: 1.125rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 4px;
        }
        .session-auto-name {
            font-size: 0.75rem;
            color: var(--text-muted);
        }
        .session-id {
            font-family: monospace;
            font-size: 0.75rem;
            color: var(--text-muted);
            background: var(--bg-primary);
            padding: 2px 6px;
            border-radius: 4px;
        }
        .source-badge {
            font-size: 0.625rem;
            font-weight: 600;
            text-transform: uppercase;
            padding: 2px 6px;
            border-radius: 4px;
            margin-left: 8px;
        }
        .source-badge.hooks {
            background: #065f46;
            color: #a7f3d0;
        }
        .source-badge.process {
            background: #1e40af;
            color: #bfdbfe;
        }
        .hook-badge {
            font-size: 0.625rem;
            font-weight: 600;
            text-transform: uppercase;
            padding: 2px 6px;
            border-radius: 4px;
            margin-left: 6px;
        }
        .hook-badge.global {
            background: #854d0e;
            color: #fef3c7;
        }
        .hook-badge.local {
            background: #166534;
            color: #bbf7d0;
        }
        .hook-badge.none {
            background: #374151;
            color: #9ca3af;
        }
        .install-hooks-btn {
            font-size: 0.625rem;
            padding: 2px 6px;
            background: #374151;
            color: #d1d5db;
            border: 1px solid #4b5563;
            border-radius: 4px;
            cursor: pointer;
            margin-left: 6px;
        }
        .install-hooks-btn:hover {
            background: #4b5563;
            color: white;
        }

        .session-details {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            margin-bottom: 12px;
        }
        .detail-item {
            font-size: 0.875rem;
        }
        .detail-label {
            color: var(--text-muted);
            margin-right: 8px;
        }
        .detail-value {
            color: var(--text-secondary);
            font-family: monospace;
            word-break: break-all;
        }

        .session-actions {
            display: flex;
            gap: 8px;
            padding-top: 12px;
            border-top: 1px solid var(--border-color);
            flex-wrap: wrap;
        }
        .action-btn {
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 0.875rem;
            cursor: pointer;
            border: 1px solid transparent;
            transition: all 0.2s;
        }
        .terminal-btn {
            background: #7c3aed;
            color: white;
        }
        .terminal-btn:hover { background: #8b5cf6; }
        .hooks-btn {
            background: #0891b2;
            color: white;
        }
        .hooks-btn:hover { background: #06b6d4; }
        .rename-btn {
            background: #1e40af;
            color: white;
        }
        .rename-btn:hover { background: #1d4ed8; }
        .delete-btn {
            background: transparent;
            border-color: #dc2626;
            color: #dc2626;
        }
        .delete-btn:hover { background: #dc2626; color: white; }
        .control-btn {
            background: #065f46;
            color: white;
        }
        .control-btn:hover { background: #047857; }

        .rename-form {
            display: none;
            gap: 8px;
            align-items: center;
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid var(--border-color);
        }
        .rename-form.active { display: flex; }
        .rename-input {
            flex: 1;
            padding: 8px 12px;
            border: 1px solid var(--border-color);
            border-radius: 4px;
            background: var(--bg-input);
            color: var(--text-primary);
            font-size: 0.875rem;
        }
        .rename-input:focus {
            outline: none;
            border-color: var(--accent-primary);
        }
        .save-btn {
            background: #059669;
            color: white;
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .save-btn:hover { background: #10b981; }
        .cancel-btn {
            background: #475569;
            color: white;
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .cancel-btn:hover { background: #64748b; }

        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: var(--text-muted);
        }
        .empty-state h2 {
            font-size: 1.25rem;
            margin-bottom: 12px;
            color: var(--text-secondary);
        }
        .empty-state p { margin-bottom: 8px; }
        .empty-state code {
            background: var(--bg-secondary);
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.875rem;
        }

        .status-message {
            padding: 12px 16px;
            border-radius: 6px;
            margin-bottom: 20px;
            display: none;
        }
        .status-message.success {
            display: block;
            background: var(--status-success-bg);
            color: var(--status-success-text);
        }
        .status-message.error {
            display: block;
            background: var(--status-error-bg);
            color: var(--status-error-text);
        }
    </style>
    <script>
        // Theme management - runs before body renders to prevent flash
        (function() {
            const savedTheme = localStorage.getItem('claude-remote-theme') || 'dark';
            if (savedTheme === 'light') {
                document.documentElement.setAttribute('data-theme', 'light');
            }
        })();
    </script>
</head>
<body>
    <div class="container">
        <div class="page-header">
            <h1><a href="/" style="color: inherit; text-decoration: none;">Claude Remote Control</a></h1>
            <button class="theme-toggle" onclick="toggleTheme()" title="Toggle light/dark mode">
                <span id="theme-icon">🌙</span>
            </button>
        </div>

        <div id="status-message" class="status-message"></div>

        <div class="sessions-header">
            <div>
                <h2 style="font-size: 1.25rem; margin-bottom: 4px;">Registered Sessions</h2>
                <span class="session-count" id="session-count">Loading...</span>
            </div>
            <button class="refresh-btn" onclick="loadSessions()">Refresh</button>
        </div>

        <div id="sessions-list" class="sessions-list">
            <div class="empty-state">
                <p>Loading sessions...</p>
            </div>
        </div>
    </div>

    <script>
        function showStatus(message, isError) {
            const el = document.getElementById('status-message');
            el.textContent = message;
            el.className = 'status-message ' + (isError ? 'error' : 'success');
            setTimeout(() => { el.className = 'status-message'; }, 3000);
        }

        function loadSessions() {
            fetch('/api/sessions')
                .then(r => r.json())
                .then(data => {
                    const container = document.getElementById('sessions-list');
                    const countEl = document.getElementById('session-count');

                    const hooked = data.registered_count || 0;
                    const discovered = data.discovered_count || 0;
                    countEl.textContent = `${data.count} session${data.count !== 1 ? 's' : ''} (${hooked} via hooks, ${discovered} discovered)`;

                    if (data.sessions.length === 0) {
                        container.innerHTML = `
                            <div class="empty-state">
                                <h2>No Sessions Detected</h2>
                                <p>No running Claude Code CLI instances found.</p>
                                <p>Start a Claude Code session with: <code>claude</code></p>
                            </div>
                        `;
                        return;
                    }

                    container.innerHTML = data.sessions.map(s => {
                        const isHooked = s.source === 'hooks';
                        const idDisplay = s.pid ? `PID ${s.pid}` : s.session_id.substring(0, 12) + '...';
                        const hookStatus = s.hook_status || 'none';
                        const hookBadge = hookStatus === 'both' ? '<span class="hook-badge global">Global</span><span class="hook-badge local">Local</span>'
                            : hookStatus === 'global' ? '<span class="hook-badge global">Global Hooks</span>'
                            : hookStatus === 'local' ? '<span class="hook-badge local">Local Hooks</span>'
                            : `<span class="hook-badge none">No Hooks</span><button class="install-hooks-btn" onclick="showInstallHooks('${escapeHtml(s.cwd)}')">Install</button>`;
                        return `
                        <div class="session-card" id="session-${s.session_id}">
                            <div class="session-header">
                                <div class="session-name-section">
                                    <div class="session-display-name">
                                        ${escapeHtml(s.display_name)}
                                        ${hookBadge}
                                    </div>
                                    ${s.custom_name ? `<div class="session-auto-name">Auto: ${escapeHtml(s.name)}</div>` : ''}
                                </div>
                                <span class="session-id">${escapeHtml(idDisplay)}</span>
                            </div>
                            <div class="session-details">
                                <div class="detail-item">
                                    <span class="detail-label">TTY:</span>
                                    <span class="detail-value">${escapeHtml(s.tty)}</span>
                                </div>
                                <div class="detail-item">
                                    <span class="detail-label">Directory:</span>
                                    <span class="detail-value">${escapeHtml(s.cwd)}</span>
                                </div>
                            </div>
                            <div class="session-actions">
                                ${hookStatus !== 'none' ? `<button class="action-btn control-btn" onclick="goToControl('${escapeHtml(s.tty)}')">Control</button>` : ''}
                                <button class="action-btn terminal-btn" onclick="goToTerminal('${escapeHtml(s.tty)}')">Terminal</button>
                                ${hookStatus !== 'none' ? `<button class="action-btn hooks-btn" onclick="goToHooks('${escapeHtml(s.tty)}')">Hooks</button>` : ''}
                                ${isHooked ? `<button class="action-btn rename-btn" onclick="showRenameForm('${escapeHtml(s.session_id)}')">Rename</button>` : ''}
                                ${isHooked ? `<button class="action-btn delete-btn" onclick="deleteSession('${escapeHtml(s.session_id)}')">Remove</button>` : ''}
                            </div>
                            ${isHooked ? `
                            <div class="rename-form" id="rename-form-${s.session_id}">
                                <input type="text" class="rename-input" id="rename-input-${s.session_id}"
                                       placeholder="Enter custom name" value="${escapeHtml(s.custom_name || '')}">
                                <button class="save-btn" onclick="saveRename('${escapeHtml(s.session_id)}')">Save</button>
                                <button class="cancel-btn" onclick="hideRenameForm('${escapeHtml(s.session_id)}')">Cancel</button>
                            </div>
                            ` : ''}
                        </div>
                    `}).join('');
                })
                .catch(err => {
                    console.error('Failed to load sessions:', err);
                    showStatus('Failed to load sessions', true);
                });
        }

        function escapeHtml(str) {
            if (!str) return '';
            return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
        }

        function formatTime(isoStr) {
            if (!isoStr) return 'Unknown';
            const d = new Date(isoStr);
            return d.toLocaleString();
        }

        function goToControl(tty) {
            // Route using TTY which works for both registered and discovered sessions
            window.location.href = '/control?session=' + encodeURIComponent(tty);
        }

        function goToTerminal(tty) {
            window.location.href = '/terminal?session=' + encodeURIComponent(tty);
        }

        function goToHooks(tty) {
            window.location.href = '/hooks?session=' + encodeURIComponent(tty);
        }

        function showInstallHooks(cwd) {
            const hooksDir = '/Users/msharpe/projects/remotecontrol/hooks';
            const instructions = `To install hooks for this session, add the following to your Claude settings:

Global (~/.claude/settings.json):
{
  "hooks": {
    "PreToolUse": ["${hooksDir}/notify.sh"],
    "PostToolUse": ["${hooksDir}/notify.sh"],
    "Stop": ["${hooksDir}/notify.sh"],
    "Notification": ["${hooksDir}/notify.sh"]
  }
}

Or local (${cwd}/.claude/settings.json) for project-specific hooks.

After adding hooks, restart Claude Code for changes to take effect.`;
            alert(instructions);
        }

        function showRenameForm(sessionId) {
            document.getElementById('rename-form-' + sessionId).classList.add('active');
            document.getElementById('rename-input-' + sessionId).focus();
        }

        function hideRenameForm(sessionId) {
            document.getElementById('rename-form-' + sessionId).classList.remove('active');
        }

        function saveRename(sessionId) {
            const input = document.getElementById('rename-input-' + sessionId);
            const name = input.value.trim();

            fetch('/api/session/' + encodeURIComponent(sessionId) + '/name', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name || null })
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    showStatus(name ? 'Session renamed to "' + name + '"' : 'Custom name removed', false);
                    loadSessions();
                } else {
                    showStatus(data.error || 'Failed to rename session', true);
                }
            })
            .catch(err => {
                console.error('Rename failed:', err);
                showStatus('Failed to rename session', true);
            });
        }

        function deleteSession(sessionId) {
            if (!confirm('Remove this session from tracking? (This does not stop Claude Code)')) {
                return;
            }

            fetch('/api/session/' + encodeURIComponent(sessionId), {
                method: 'DELETE'
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    showStatus('Session removed', false);
                    loadSessions();
                } else {
                    showStatus(data.error || 'Failed to remove session', true);
                }
            })
            .catch(err => {
                console.error('Delete failed:', err);
                showStatus('Failed to remove session', true);
            });
        }

        // Theme toggle
        function toggleTheme() {
            const html = document.documentElement;
            const currentTheme = html.getAttribute('data-theme');
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';
            html.setAttribute('data-theme', newTheme === 'light' ? 'light' : '');
            if (newTheme === 'dark') {
                html.removeAttribute('data-theme');
            }
            localStorage.setItem('claude-remote-theme', newTheme);
            updateThemeIcon();
        }

        function updateThemeIcon() {
            const isDark = !document.documentElement.getAttribute('data-theme');
            document.getElementById('theme-icon').textContent = isDark ? '🌙' : '☀️';
        }

        // Initial load
        updateThemeIcon();
        loadSessions();

        // Auto-refresh every 10 seconds
        setInterval(loadSessions, 10000);
    </script>
</body>
</html>
"""

# ============================================================================
# Routes
# ============================================================================

def detect_idle_from_terminal(content: str) -> bool:
    """Check if terminal content indicates Claude is idle (waiting for input)."""
    if not content:
        return False
    # Check the last portion of content (last ~500 chars) for idle indicators
    # The "↵send" hint may not be on the absolute last line due to terminal formatting
    tail = content[-500:] if len(content) > 500 else content

    # Check for common idle indicators anywhere in the tail:
    # - "↵" with "send" nearby - enter to send hint
    # - "❯" prompt character on a line
    if '↵' in tail and 'send' in tail.lower():
        return True

    # Also check the last few lines for prompt characters
    lines = content.strip().split('\n')
    for line in lines[-5:]:  # Check last 5 lines
        line = line.strip()
        # Prompt at start of line
        if line.startswith('❯') or line.startswith('›'):
            return True
        # Just the prompt character
        if line in ('❯', '>', '›'):
            return True

    return False


@app.route('/')
@requires_auth
def index():
    """Sessions overview - the landing page."""
    return render_template_string(SESSIONS_TEMPLATE)


@app.route('/control')
@requires_auth
def control():
    """Control panel for a specific session."""
    # Get session from query param, or use first available
    sessions = backend.get_sessions()
    selected_session = request.args.get('session')

    # Redirect to sessions page if no session specified
    if not selected_session:
        return redirect('/')

    if selected_session not in sessions:
        # Default to first session
        if sessions:
            selected_session = sessions[0]

    state.active_session = selected_session

    is_compacting = False
    is_idle = False
    initial_mode = 'normal'
    initial_context_left = None
    if selected_session:
        session_state = state.get_session(selected_session)
        is_compacting = session_state.is_compacting

        # Detect initial mode and context % from terminal content (strip trailing whitespace)
        content = backend.get_content(selected_session)
        if content:
            tail = content.rstrip()[-500:]
            if 'accept edits on' in tail:
                initial_mode = 'edits'
            elif 'plan mode on' in tail:
                initial_mode = 'plan'
            # Detect context percentage
            match = re.search(r'Context left until auto-compact: (\d+)%', tail)
            if match:
                initial_context_left = int(match.group(1))

        # Determine idle state from hook events first, terminal detection as fallback
        hook_events = state.hook_events
        if hook_events:
            # Find the last Stop and last PreToolUse events
            last_stop = None
            last_pre_tool = None
            for event in reversed(hook_events):
                if event.event_type == 'Stop' and last_stop is None:
                    last_stop = event
                if event.event_type == 'PreToolUse' and last_pre_tool is None:
                    last_pre_tool = event
                if last_stop and last_pre_tool:
                    break

            # Idle if last Stop is more recent than last PreToolUse
            if last_stop:
                if not last_pre_tool or last_stop.timestamp > last_pre_tool.timestamp:
                    is_idle = True
        else:
            # No hook events yet - fall back to terminal detection
            content = backend.get_content(selected_session)
            is_idle = detect_idle_from_terminal(content)

    # Build session info for UI - combine backend sessions with registered sessions
    sessions_info = []
    seen_session_ids = set()

    # First, add registered sessions (from hooks) with friendly names
    for reg_sess in state.get_registered_sessions():
        sessions_info.append({
            'id': reg_sess.session_id,
            'name': reg_sess.display_name(),
            'active': reg_sess.session_id == selected_session or reg_sess.name == selected_session,
            'tty': reg_sess.tty,
            'cwd': reg_sess.cwd,
        })
        seen_session_ids.add(reg_sess.session_id)

    # Then add any backend sessions not already in registered (backwards compatibility)
    for s in sessions:
        if s not in seen_session_ids:
            sessions_info.append({
                'id': s,
                'name': s,
                'active': s == selected_session
            })

    return render_template_string(
        MAIN_TEMPLATE,
        now=datetime.now().strftime('%H:%M:%S'),
        local_ip=get_local_ip(),
        port=config['server']['port'],
        session=selected_session or 'No sessions',
        sessions=sessions_info,
        notifications_paused=state.notifications_paused,
        notification_mode=state.notification_mode,
        is_compacting=is_compacting,
        is_idle=is_idle,
        initial_mode=initial_mode,
        initial_context_left=initial_context_left
    )

@app.route('/terminal')
@requires_auth
def terminal():
    session = request.args.get('session') or state.active_session or config['tmux']['session_name']

    # Normalize session (TTY) if needed - backend uses s006 format, hooks use ttys006
    sessions = backend.get_sessions()
    if session and session not in sessions:
        # Try normalized form (ttys006 -> s006)
        normalized = session[3:] if session.startswith('tty') else session
        if normalized in sessions:
            session = normalized
        elif sessions:
            # Fallback to first available session
            session = sessions[0]

    polling_mode = not backend.supports_terminal_attach()
    return render_template_string(
        TERMINAL_TEMPLATE,
        session=session,
        scrollback=config.get('terminal', {}).get('scrollback', 1000),
        polling_mode=polling_mode,
        backend_name=backend.name
    )

@app.route('/hooks')
@requires_auth
def hooks():
    """Hook events visualization page."""
    session = request.args.get('session')
    return render_template_string(HOOKS_TEMPLATE, session=session)


@app.route('/sessions')
@requires_auth
def sessions_page_redirect():
    """Redirect old /sessions URL to home."""
    return redirect('/')


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
        # State is now managed via hooks, no need to clear prompt state
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
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': f'Failed to send Esc via {backend.name}'})

@app.route('/send-shift-tab', methods=['POST'])
@requires_auth
def send_shift_tab():
    """Send Shift+Tab to cycle Claude modes (plan/accept edits)."""
    data = request.get_json()
    session = data.get('session') or state.active_session

    if not session:
        return jsonify({'success': False, 'error': 'No session selected'})

    success = backend.send_shift_tab(session)

    if success:
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': f'Failed to send Shift+Tab via {backend.name}'})

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
    """Get session status. Note: prompt detection now uses hooks instead of terminal parsing."""
    session = request.args.get('session') or state.active_session
    sessions = backend.get_sessions()

    is_compacting = False
    mode = 'normal'
    context_left = None
    if session:
        session_state = state.get_session(session)
        is_compacting = session_state.is_compacting
        # Detect mode and context % from terminal content (strip trailing whitespace first)
        content = backend.get_content(session)
        if content:
            # Terminal content often has trailing whitespace padding
            tail = content.rstrip()[-500:]
            if 'accept edits on' in tail:
                mode = 'edits'
            elif 'plan mode on' in tail:
                mode = 'plan'
            # Detect context percentage
            import re
            match = re.search(r'Context left until auto-compact: (\d+)%', tail)
            if match:
                context_left = int(match.group(1))

    return jsonify({
        'is_compacting': is_compacting,
        'session': session,
        'sessions': sessions,
        'backend': backend.name,
        'mode': mode,
        'context_left': context_left
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
    """Legacy endpoint - prompts now handled via hooks. Kept for backward compatibility."""
    return jsonify({'success': True, 'cleared': 0})


@app.route('/api/hook', methods=['POST'])
def api_hook():
    """Receive hook events from Claude Code hooks."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data'}), 400

        # Extract common fields from hook input
        event_type = data.get('hook_event_name', 'unknown')
        session_id = data.get('session_id', 'unknown')
        tool_name = data.get('tool_name')
        tool_input = data.get('tool_input')
        tool_result = data.get('tool_result')
        reason = data.get('reason')

        # Create and store the hook event
        # Default permission_mode to 'ask' for PreToolUse events (matches notification logic)
        permission_mode = data.get('permission_mode', 'ask') if event_type == 'PreToolUse' else None
        event = HookEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            session_id=session_id,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_result=str(tool_result) if tool_result else None,
            reason=reason,
            raw_data=data,
            permission_mode=permission_mode
        )
        state.add_hook_event(event)

        log.info(f"[hook] {event_type}: {tool_name or reason or 'no details'} (session: {session_id})")

        # Auto-register session on first event (Claude Code doesn't send SessionStart)
        if session_id not in state.registered_sessions:
            cwd = data.get('cwd', '')
            tty = data.get('tty', 'unknown')  # Injected by notify.sh

            # Derive friendly name: basename@tty
            dir_name = os.path.basename(cwd) if cwd else 'claude'
            friendly_name = f"{dir_name}@{tty}"

            state.register_session(session_id, friendly_name, cwd, tty)

        # Handle session end event
        if event_type == 'SessionEnd':
            # Remove the session from the registry
            state.unregister_session(session_id)

        # Determine if this event should trigger a notification
        should_notify = False
        notification_message = None
        notification_title = None

        if event_type == 'Stop':
            should_notify = True
            notification_message = f"Claude has finished"
            if reason:
                notification_message += f": {reason[:100]}"
            notification_title = f"Claude Done [{session_id[:8]}]"

        elif event_type == 'Notification':
            should_notify = True
            notification_message = data.get('message', 'Notification from Claude')
            notification_title = f"Claude [{session_id[:8]}]"

        elif event_type == 'PreToolUse':
            # Check if this is a permission-requiring tool
            permission_mode = data.get('permission_mode', 'ask')
            if permission_mode == 'ask':
                should_notify = True
                notification_message = f"Permission needed for: {tool_name}"
                if tool_input:
                    # Add brief context from tool input
                    input_str = str(tool_input)[:100]
                    notification_message += f"\n{input_str}"
                notification_title = f"Claude Permission [{session_id[:8]}]"

        # Send notification if warranted
        if should_notify:
            send_notification(
                message=notification_message,
                title=notification_title,
                session=session_id
            )

        return jsonify({'success': True, 'event_type': event_type})

    except Exception as e:
        log.error(f"[hook] Error processing hook: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/hooks')
@requires_auth
def api_hooks():
    """Get recent hook events."""
    session_id = request.args.get('session')
    event_type = request.args.get('type')
    limit = int(request.args.get('limit', 100))

    events = state.get_hook_events(session_id=session_id, event_type=event_type, limit=limit)
    return jsonify({
        'events': [e.to_dict() for e in events],
        'total': len(state.hook_events)
    })


# ============================================================================
# Session Management API
# ============================================================================

def discover_claude_processes():
    """Discover running Claude Code CLI processes using ps and lsof."""
    discovered = []
    try:
        # Find claude CLI processes (not Claude.app)
        ps_result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True, text=True, timeout=5
        )

        claude_pids = []
        for line in ps_result.stdout.splitlines():
            # Match "claude " at the end (CLI) but not Claude.app paths
            if 'claude ' in line.lower() and '/Applications/Claude.app' not in line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        pid = int(parts[1])
                        # Extract TTY (column 6 in ps aux)
                        tty = parts[6] if len(parts) > 6 else '??'
                        claude_pids.append((pid, tty))
                    except ValueError:
                        continue

        if not claude_pids:
            return discovered

        # Get working directories using lsof
        pid_list = ','.join(str(p[0]) for p in claude_pids)
        lsof_result = subprocess.run(
            ['lsof', '-p', pid_list],
            capture_output=True, text=True, timeout=5
        )

        # Parse lsof output for cwd entries
        pid_to_cwd = {}
        for line in lsof_result.stdout.splitlines():
            if ' cwd ' in line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        pid = int(parts[1])
                        # cwd path is the last element
                        cwd = parts[-1]
                        pid_to_cwd[pid] = cwd
                    except ValueError:
                        continue

        # Build discovered sessions
        for pid, tty in claude_pids:
            cwd = pid_to_cwd.get(pid, '')
            dir_name = os.path.basename(cwd) if cwd else 'claude'
            discovered.append({
                'pid': pid,
                'tty': tty,
                'cwd': cwd,
                'name': f"{dir_name}@{tty}",
                'source': 'process'
            })

    except Exception as e:
        log.warning(f"[sessions] Failed to discover processes: {e}")

    return discovered


def detect_hooks_for_session(cwd: str) -> dict:
    """Detect if Claude hooks are configured for a session.

    Checks:
    1. Global hooks in ~/.claude/settings.json
    2. Local hooks in <cwd>/.claude/settings.json

    Returns dict with:
        - global_hooks: bool - True if hooks configured globally
        - local_hooks: bool - True if hooks configured locally
        - hook_status: str - 'global', 'local', 'both', or 'none'
    """
    result = {
        'global_hooks': False,
        'local_hooks': False,
        'hook_status': 'none'
    }

    def has_hooks_configured(settings_path: str) -> bool:
        """Check if a settings.json file has hooks configured."""
        try:
            if not os.path.exists(settings_path):
                return False
            with open(settings_path, 'r') as f:
                settings = json.load(f)
            # Check for hooks configuration
            hooks = settings.get('hooks', {})
            if hooks:
                # Check if any hook type has commands configured
                for hook_type in ['PreToolUse', 'PostToolUse', 'Stop', 'Notification']:
                    if hook_type in hooks and hooks[hook_type]:
                        return True
            return False
        except Exception:
            return False

    # Check global hooks
    global_settings = os.path.expanduser('~/.claude/settings.json')
    result['global_hooks'] = has_hooks_configured(global_settings)

    # Check local hooks
    if cwd:
        local_settings = os.path.join(cwd, '.claude', 'settings.json')
        result['local_hooks'] = has_hooks_configured(local_settings)

    # Determine overall status
    if result['global_hooks'] and result['local_hooks']:
        result['hook_status'] = 'both'
    elif result['global_hooks']:
        result['hook_status'] = 'global'
    elif result['local_hooks']:
        result['hook_status'] = 'local'
    else:
        result['hook_status'] = 'none'

    return result


@app.route('/api/sessions')
@requires_auth
def api_sessions():
    """Get all sessions: both registered (via hooks) and discovered (via ps)."""
    # Get hook-registered sessions
    registered = state.get_registered_sessions()

    # Discover running Claude processes
    discovered = discover_claude_processes()

    # Build a map of cwd -> discovered process for matching
    cwd_to_discovered = {}
    for proc in discovered:
        cwd = proc.get('cwd', '')
        if cwd and cwd not in cwd_to_discovered:
            cwd_to_discovered[cwd] = proc

    # Track which discovered processes have been matched to registered sessions
    # Use normalized TTYs for comparison (ttys006 and s006 should match)
    matched_discovered_ttys = set()

    # Build combined session list
    sessions = []

    # Add registered sessions first, matching with discovered processes when possible
    for s in registered:
        d = s.to_dict()
        d['source'] = 'hooks'

        # If TTY is unknown/invalid, try to match by cwd to get real TTY/PID
        tty = s.tty
        if not tty or tty == 'unknown' or 'not a tty' in tty.lower():
            matched_proc = cwd_to_discovered.get(s.cwd)
            if matched_proc:
                # Update with discovered process info
                d['tty'] = matched_proc['tty']
                d['pid'] = matched_proc['pid']
                # Update the name to use the real TTY
                dir_name = os.path.basename(s.cwd) if s.cwd else 'claude'
                d['name'] = f"{dir_name}@{matched_proc['tty']}"
                if not s.custom_name:
                    d['display_name'] = d['name']
                matched_discovered_ttys.add(state.normalize_tty(matched_proc['tty']))

        # Track valid TTYs for deduplication (using normalized form)
        if d.get('tty') and d['tty'] != 'unknown' and 'not a tty' not in d.get('tty', '').lower():
            matched_discovered_ttys.add(state.normalize_tty(d['tty']))

        # Add hook detection info
        hook_info = detect_hooks_for_session(s.cwd)
        d.update(hook_info)
        sessions.append(d)

    # Add discovered processes that haven't been matched to registered sessions
    for proc in discovered:
        if state.normalize_tty(proc['tty']) not in matched_discovered_ttys:
            # Add hook detection info
            hook_info = detect_hooks_for_session(proc['cwd'])
            sessions.append({
                'session_id': f"pid-{proc['pid']}",
                'name': proc['name'],
                'display_name': proc['name'],
                'cwd': proc['cwd'],
                'tty': proc['tty'],
                'pid': proc['pid'],
                'custom_name': None,
                'registered_at': None,
                'source': 'process',
                **hook_info
            })

    return jsonify({
        'sessions': sessions,
        'count': len(sessions),
        'registered_count': len(registered),
        'discovered_count': len(discovered)
    })


@app.route('/api/session/<session_id>/name', methods=['POST'])
@requires_auth
def api_session_set_name(session_id):
    """Set a custom name for a session."""
    try:
        data = request.get_json()
        if not data or 'name' not in data:
            return jsonify({'error': 'Missing "name" in request body'}), 400

        custom_name = data['name'].strip()
        if not custom_name:
            return jsonify({'error': 'Name cannot be empty'}), 400

        success = state.set_session_custom_name(session_id, custom_name)
        if success:
            return jsonify({'success': True, 'name': custom_name})
        else:
            return jsonify({'error': 'Session not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/session/<session_id>', methods=['DELETE'])
@requires_auth
def api_session_delete(session_id):
    """Manually remove a session from tracking."""
    if session_id in state.registered_sessions:
        state.unregister_session(session_id)
        return jsonify({'success': True})
    else:
        return jsonify({'error': 'Session not found'}), 404


# ============================================================================
# Main
# ============================================================================

def main():
    global backend
    backend = get_active_backend()

    local_ip = get_local_ip()
    port = config['server']['port']
    sessions = backend.get_sessions()
    ntfy_prefix = config['notifications']['ntfy'].get('topic_prefix', 'claude')

    # Check backend availability
    accessibility_status = "available" if (_accessibility_backend and _accessibility_backend.is_available()) else "not available"
    tmux_status = "available" if _tmux_backend.is_available() else "not available"

    # Build banner with proper alignment
    url_base = f"http://{local_ip}:{port}"

    lines = [
        f"Control Panel: {url_base}/",
        f"Terminal:      {url_base}/terminal",
        f"Hooks:         {url_base}/hooks",
        f"Auth Method:   {config['auth']['method']}",
        f"Backend:       {backend.name}",
        f"ntfy prefix:   {ntfy_prefix}-<session>",
    ]
    bottom_lines = [
        f"Backends:  tmux: {tmux_status:<12}  a11y: {accessibility_status}",
        "Hooks:     Run ./hooks/install.sh to configure",
    ]

    # Calculate box width based on longest line
    all_lines = lines + bottom_lines + ["Claude Code Remote Control Server"]
    box_width = max(len(line) for line in all_lines) + 4  # +4 for padding

    print("╔" + "═" * box_width + "╗")
    print("║" + "Claude Code Remote Control Server".center(box_width) + "║")
    print("╠" + "═" * box_width + "╣")
    for line in lines:
        print(f"║  {line:<{box_width - 2}}║")
    print("╠" + "═" * box_width + "╣")
    for line in bottom_lines:
        print(f"║  {line:<{box_width - 2}}║")
    print("╚" + "═" * box_width + "╝")

    if sessions:
        print(f"[{backend.name}] Found {len(sessions)} session(s): {', '.join(sessions)}")
    else:
        print(f"[{backend.name}] No sessions found!")
        if backend.name == 'tmux':
            print("          Create with: tmux new-session -d -s <name>")
        elif 'accessibility' in backend.name:
            detected = _accessibility_backend._detected_terminal if _accessibility_backend else 'a terminal'
            print(f"          Open {detected} and run Claude Code")

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
