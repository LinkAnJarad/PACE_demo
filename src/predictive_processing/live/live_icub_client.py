"""Client module for communicating with the pyCub iCub simulation server.

Provides non-blocking async/thread calls to trigger robot speech animation when
TTS audio is generated in Streamlit or live interactive sessions.
"""

from __future__ import annotations

import threading
import urllib.request
import urllib.parse
import json


ICUB_SERVER_URL = "http://localhost:8000"


def check_icub_server_health(server_url: str = ICUB_SERVER_URL) -> bool:
    """Check if the pyCub FastAPI server is running and responsive."""
    try:
        req = urllib.request.Request(f"{server_url}/", method="GET")
        with urllib.request.urlopen(req, timeout=0.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("status") == "running"
    except Exception:
        return False


def _send_talk_request_bg(
    duration: float,
    server_url: str,
    valence: float = 0.0,
    arousal: float = 0.0,
    prediction_error: float = 0.0,
    start_delay: float = 0.5,
):
    """Background worker function for HTTP post."""
    try:
        payload = json.dumps({
            "duration": duration,
            "valence": valence,
            "arousal": arousal,
            "prediction_error": prediction_error,
            "start_delay": start_delay,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{server_url}/talk",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            resp.read()
    except Exception:
        pass


def notify_icub_talk(
    duration: float,
    server_url: str = ICUB_SERVER_URL,
    valence: float = 0.0,
    arousal: float = 0.0,
    prediction_error: float = 0.0,
    start_delay: float = 0.5,
):
    """Trigger iCub body & head talk animation in a non-blocking background thread."""
    if duration <= 0:
        return
    thread = threading.Thread(
        target=_send_talk_request_bg,
        args=(duration, server_url, valence, arousal, prediction_error, start_delay),
        daemon=True,
    )
    thread.start()


def _send_gesture_request_bg(
    gesture: str,
    duration: float,
    server_url: str,
    start_delay: float,
):
    """Background worker function for sending gesture HTTP post."""
    try:
        payload = json.dumps({
            "gesture": gesture,
            "duration": duration,
            "start_delay": start_delay,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{server_url}/gesture",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            resp.read()
    except Exception:
        pass


def notify_icub_gesture(
    gesture: str,
    duration: float = 2.5,
    server_url: str = ICUB_SERVER_URL,
    start_delay: float = 0.5,
):
    """Trigger an LLM-selected gesture on pyCub in a non-blocking background thread."""
    if not gesture or gesture.strip().lower() == "none":
        return
    thread = threading.Thread(
        target=_send_gesture_request_bg,
        args=(gesture.strip().lower(), duration, server_url, start_delay),
        daemon=True,
    )
    thread.start()


