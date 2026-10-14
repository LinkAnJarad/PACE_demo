"""FastAPI Server hosting pyCub iCub Robot Simulator.

Provides continuous background physics updates and HTTP endpoints for controlling
robot joints and triggering speech-driven head micro-movements during live demos.

Usage:
    uv run uvicorn icub_server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import io
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

# Ensure local pyCub module is on Python path
_project_root = Path(__file__).resolve().parent
_pycub_path = _project_root / "pyCub"
if str(_pycub_path) not in sys.path:
    sys.path.insert(0, str(_pycub_path))

import traceback

try:
    from icub_pybullet.pycub import pyCub
except Exception as e:
    print(f"Warning: Failed to import pyCub: {e}")
    traceback.print_exc()
    pyCub = None


# ── Global State & Kinematic Defaults ────────────────────────────────────────

client: Any | None = None
sim_task: asyncio.Task | None = None
gesture_task: asyncio.Task | None = None
talk_task: asyncio.Task | None = None

# Default resting angles in radians from Demo repository
DEFAULT_SH_PITCH = math.radians(-20.0)  # ~ -0.349 rad (slight natural forward rest)
DEFAULT_SH_ROLL = math.radians(22.0)    # ~ 0.384 rad (natural lateral clearance)
DEFAULT_SH_YAW = 0.0
DEFAULT_ELBOW = math.radians(35.0)      # ~ 0.610 rad (relaxed elbow bend)
DEFAULT_WRIST_PROSUP = 0.0
DEFAULT_WRIST_PITCH = 0.0
DEFAULT_WRIST_YAW = 0.0

R_HAND_FINGERS = (
    [f"r_hand_thumb_{i}_joint" for i in range(4)]
    + [f"r_hand_index_{i}_joint" for i in range(4)]
    + [f"r_hand_middle_{i}_joint" for i in range(4)]
    + [f"r_hand_ring_{i}_joint" for i in range(4)]
    + [f"r_hand_little_{i}_joint" for i in range(4)]
)
L_HAND_FINGERS = (
    [f"l_hand_thumb_{i}_joint" for i in range(4)]
    + [f"l_hand_index_{i}_joint" for i in range(4)]
    + [f"l_hand_middle_{i}_joint" for i in range(4)]
    + [f"l_hand_ring_{i}_joint" for i in range(4)]
    + [f"l_hand_little_{i}_joint" for i in range(4)]
)

JOINT_LIMITS = {
    "neck_pitch": (-0.65, 0.35),
    "neck_roll": (-0.32, 0.32),
    "neck_yaw": (-0.80, 0.80),
    "torso_pitch": (-0.30, 1.10),
    "torso_roll": (-0.45, 0.45),
    "torso_yaw": (-0.80, 0.80),
    "r_shoulder_pitch": (-1.60, 0.15),
    "l_shoulder_pitch": (-1.60, 0.15),
    "r_shoulder_roll": (0.05, 2.70),
    "l_shoulder_roll": (0.05, 2.70),
    "r_shoulder_yaw": (-0.60, 1.35),
    "l_shoulder_yaw": (-0.60, 1.35),
    "r_elbow": (0.28, 1.80),
    "l_elbow": (0.28, 1.80),
    "r_wrist_prosup": (-1.00, 1.00),
    "l_wrist_prosup": (-1.00, 1.00),
    "r_wrist_pitch": (-1.30, 0.40),
    "l_wrist_pitch": (-1.30, 0.40),
    "r_wrist_yaw": (-0.30, 0.40),
    "l_wrist_yaw": (-0.30, 0.40),
}


def set_joint_positions(joint_dict: dict[str, float], velocity: float = 6.0):
    """Safely clip and dispatch joint positions to pyCub in a single batched call."""
    if client is None:
        return
    joints = []
    positions = []
    for j, val in joint_dict.items():
        if j in JOINT_LIMITS:
            low, high = JOINT_LIMITS[j]
            val = float(np.clip(val, low, high))
        joints.append(j)
        positions.append(val)
    try:
        client.move_position(joints, positions, wait=False, velocity=velocity)
    except Exception as e:
        print(f"[Joints Error] Failed setting joint batch: {e}")


def reset_robot_posture(velocity: float = 5.0):
    """Smoothly return pyCub to default standing neutral posture."""
    if client is None:
        return
    targets = {
        "neck_pitch": 0.0,
        "neck_yaw": 0.0,
        "neck_roll": 0.0,
        "torso_pitch": 0.0,
        "torso_roll": 0.0,
        "torso_yaw": 0.0,
        "r_shoulder_pitch": DEFAULT_SH_PITCH,
        "l_shoulder_pitch": DEFAULT_SH_PITCH,
        "r_shoulder_roll": DEFAULT_SH_ROLL,
        "l_shoulder_roll": DEFAULT_SH_ROLL,
        "r_shoulder_yaw": DEFAULT_SH_YAW,
        "l_shoulder_yaw": DEFAULT_SH_YAW,
        "r_elbow": DEFAULT_ELBOW,
        "l_elbow": DEFAULT_ELBOW,
        "r_wrist_prosup": DEFAULT_WRIST_PROSUP,
        "l_wrist_prosup": DEFAULT_WRIST_PROSUP,
        "r_wrist_pitch": DEFAULT_WRIST_PITCH,
        "l_wrist_pitch": DEFAULT_WRIST_PITCH,
        "r_wrist_yaw": DEFAULT_WRIST_YAW,
        "l_wrist_yaw": DEFAULT_WRIST_YAW,
    }
    for fj in R_HAND_FINGERS + L_HAND_FINGERS:
        targets[fj] = 0.0
    set_joint_positions(targets, velocity=velocity)


async def run_simulation_loop():
    """Background task to continuously step pyCub physics and render updates."""
    global client
    print("[Simulation] pyCub background physics loop started.")

    while client is not None and hasattr(client, "is_alive") and client.is_alive():
        try:
            if hasattr(client, "_client") and client._client < 0:
                print("[Simulation] PyBullet physics engine disconnected. Exiting loop.")
                break
            client.update_simulation()
        except Exception as e:
            print(f"[Simulation Warning] Error in update_simulation: {e}")
            await asyncio.sleep(0.05)
            continue

        await asyncio.sleep(0.01)

    print("[Simulation] pyCub background simulation loop exited.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, sim_task
    config_path = str(_project_root / "icub_config.yaml")

    if pyCub is not None:
        try:
            print(f"Initializing pyCub simulator with config: {config_path}...")
            client = pyCub(config=config_path)
            reset_robot_posture(velocity=5.0)
            sim_task = asyncio.create_task(run_simulation_loop())
            print("pyCub simulation initialized successfully.")
        except Exception as e:
            print(f"Warning: Failed to initialize pyCub simulator: {e}")
            traceback.print_exc()
            client = None
    else:
        print("Warning: pyCub library not imported.")

    yield

    if sim_task:
        sim_task.cancel()


app = FastAPI(
    title="iCub Robot Simulation API",
    description="FastAPI server for driving pyCub robot head motion in sync with speech",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# ── Schemas ─────────────────────────────────────────────────────────────────

class MoveRequest(BaseModel):
    joint: str
    position: float


class TalkRequest(BaseModel):
    duration: float = 3.0  # Duration of speech in seconds
    frequency: float = 3.0  # Cadence of speech cadence (Hz)
    pitch_amplitude: float = 0.40  # Max pitch angle oscillation (radians ~23 deg)
    yaw_amplitude: float = 0.38  # Max yaw angle oscillation (radians ~21.7 deg)
    roll_amplitude: float = 0.20  # Max roll angle oscillation (radians ~11.5 deg)
    valence: float = 0.0  # Perceived Valence [-1.0 to 1.0]
    arousal: float = 0.0  # Perceived Arousal [-1.0 to 1.0]
    prediction_error: float = 0.0  # Prediction Error magnitude [0.0 to 2.0]
    start_delay: float = 1.0  # Delay in seconds before starting movement to sync with browser audio


class GestureRequest(BaseModel):
    gesture: str = "neutral"
    duration: float = 2.5
    start_delay: float = 0.5


# ── Full-Body Affective Speech & Gesture Logic ──────────────────────────────

async def animate_gesture(
    gesture: str,
    duration: float = 2.5,
    start_delay: float = 0.5,
):
    """Animate parametric coordinated body and head gestures on pyCub for the 8 core gestures."""
    global client
    if client is None:
        print("[Gesture] Skipped: pyCub client is None")
        return

    if start_delay > 0:
        await asyncio.sleep(start_delay)

    name = gesture.lower().strip()
    if name in ("neutral", "reset"):
        reset_robot_posture(velocity=5.0)
        return

    print(f"[Gesture] Executing gesture '{name}' (duration={duration:.2f}s)")
    start_time = asyncio.get_event_loop().time()
    dt = 0.03  # ~33 FPS

    try:
        while (asyncio.get_event_loop().time() - start_time) < duration:
            elapsed = asyncio.get_event_loop().time() - start_time
            progress = min(1.0, max(0.0, elapsed / max(0.1, duration)))

            # Sinusoidal bell envelope for smooth ease-in / ease-out (clamped non-negative)
            envelope = max(0.0, math.sin(math.pi * progress)) ** 1.2

            # Start with default neutral values
            targets: dict[str, float] = {
                "neck_pitch": 0.0,
                "neck_yaw": 0.0,
                "neck_roll": 0.0,
                "torso_pitch": 0.0,
                "torso_roll": 0.0,
                "torso_yaw": 0.0,
                "r_shoulder_pitch": DEFAULT_SH_PITCH,
                "l_shoulder_pitch": DEFAULT_SH_PITCH,
                "r_shoulder_roll": DEFAULT_SH_ROLL,
                "l_shoulder_roll": DEFAULT_SH_ROLL,
                "r_shoulder_yaw": DEFAULT_SH_YAW,
                "l_shoulder_yaw": DEFAULT_SH_YAW,
                "r_elbow": DEFAULT_ELBOW,
                "l_elbow": DEFAULT_ELBOW,
                "r_wrist_prosup": DEFAULT_WRIST_PROSUP,
                "l_wrist_prosup": DEFAULT_WRIST_PROSUP,
                "r_wrist_pitch": DEFAULT_WRIST_PITCH,
                "l_wrist_pitch": DEFAULT_WRIST_PITCH,
                "r_wrist_yaw": DEFAULT_WRIST_YAW,
                "l_wrist_yaw": DEFAULT_WRIST_YAW,
            }

            # Hand fingers defaulted to relaxed/open
            for fj in R_HAND_FINGERS + L_HAND_FINGERS:
                targets[fj] = 0.0

            if name == "nod":
                # Affirmative double-dip nod with torso bow (pure head & torso motion, arms at rest)
                dip = 0.5 * (1.0 - math.cos(2 * math.pi * 1.6 * elapsed))
                targets["neck_pitch"] = (-0.50 * dip - 0.08) * envelope
                targets["neck_roll"] = 0.06 * envelope
                targets["torso_pitch"] = (0.16 * dip + 0.06) * envelope

            elif name in ("shake_head", "shake"):
                # Wide emphatic head-only shake (~1.5 Hz) with full neck yaw range of motion (arms & torso at rest)
                shake_cycle = math.sin(2 * math.pi * 1.5 * elapsed)
                targets["neck_yaw"] = 0.78 * shake_cycle * envelope
                targets["neck_roll"] = 0.12 * shake_cycle * envelope
                targets["neck_pitch"] = -0.10 * envelope

            elif name in ("tilt_head", "tilt"):
                # Deep empathetic curious tilt: head rolls & yaws, torso leans in (pure head & torso motion, arms at rest)
                targets["neck_roll"] = 0.32 * envelope
                targets["neck_pitch"] = -0.26 * envelope
                targets["neck_yaw"] = -0.32 * envelope
                targets["torso_roll"] = 0.16 * envelope
                targets["torso_pitch"] = 0.14 * envelope
                targets["torso_yaw"] = -0.12 * envelope

            elif name == "shrug":
                # Dramatic high shoulder elevation, tight elbow bend, palms supinated flat up & fingers spread
                targets["torso_pitch"] = -0.10 * envelope
                targets["neck_roll"] = 0.26 * envelope
                targets["neck_pitch"] = -0.16 * envelope
                targets["r_shoulder_pitch"] = -0.10 * envelope + DEFAULT_SH_PITCH * (1 - envelope)
                targets["l_shoulder_pitch"] = -0.10 * envelope + DEFAULT_SH_PITCH * (1 - envelope)
                targets["r_shoulder_roll"] = 1.25 * envelope + DEFAULT_SH_ROLL * (1 - envelope)
                targets["l_shoulder_roll"] = 1.25 * envelope + DEFAULT_SH_ROLL * (1 - envelope)
                targets["r_elbow"] = 1.70 * envelope + DEFAULT_ELBOW * (1 - envelope)
                targets["l_elbow"] = 1.70 * envelope + DEFAULT_ELBOW * (1 - envelope)
                targets["r_wrist_prosup"] = 0.95 * envelope
                targets["l_wrist_prosup"] = -0.95 * envelope
                targets["r_wrist_pitch"] = -0.50 * envelope
                targets["l_wrist_pitch"] = -0.50 * envelope
                # Splayed open palms
                targets["r_hand_index_0_joint"] = -0.25 * envelope
                targets["l_hand_index_0_joint"] = -0.25 * envelope
                targets["r_hand_little_0_joint"] = 0.25 * envelope
                targets["l_hand_little_0_joint"] = 0.25 * envelope

            elif name == "think":
                # Hand brought up near chin with closed fist hand posture, head tilted down
                targets["neck_pitch"] = -0.22 * envelope
                targets["neck_roll"] = 0.18 * envelope
                targets["neck_yaw"] = -0.15 * envelope
                targets["r_shoulder_pitch"] = -0.95 * envelope + DEFAULT_SH_PITCH * (1 - envelope)
                targets["r_shoulder_roll"] = 0.42 * envelope + DEFAULT_SH_ROLL * (1 - envelope)
                targets["r_shoulder_yaw"] = 0.58 * envelope
                targets["r_elbow"] = 1.65 * envelope + DEFAULT_ELBOW * (1 - envelope)
                targets["r_wrist_pitch"] = -0.35 * envelope
                targets["r_wrist_prosup"] = -0.30 * envelope
                targets["l_shoulder_pitch"] = DEFAULT_SH_PITCH + 0.08 * envelope
                targets["l_elbow"] = DEFAULT_ELBOW + 0.15 * envelope

                # Closed hand / fist for thinking pose
                for i in range(4):
                    targets[f"r_hand_thumb_{i}_joint"] = 1.30 * envelope
                targets["r_hand_index_0_joint"] = -0.20 * envelope
                for i in range(1, 4):
                    targets[f"r_hand_index_{i}_joint"] = 1.45 * envelope
                targets["r_hand_middle_0_joint"] = 0.15 * envelope
                for i in range(1, 4):
                    targets[f"r_hand_middle_{i}_joint"] = 1.45 * envelope
                targets["r_hand_ring_0_joint"] = 0.15 * envelope
                for i in range(1, 4):
                    targets[f"r_hand_ring_{i}_joint"] = 1.45 * envelope
                targets["r_hand_little_0_joint"] = 0.15 * envelope
                for i in range(1, 4):
                    targets[f"r_hand_little_{i}_joint"] = 1.45 * envelope

            elif name == "wave":
                # Natural friendly greeting wave: upper arm elevated laterally to shoulder height,
                # forearm held vertically upright beside head, swinging prominently side-to-side (~1.5 Hz)
                wave_cycle = math.sin(2 * math.pi * 1.5 * elapsed)
                targets["r_shoulder_pitch"] = -1.05 * envelope + DEFAULT_SH_PITCH * (1 - envelope)
                targets["r_shoulder_roll"] = (1.45 + 0.10 * wave_cycle) * envelope + DEFAULT_SH_ROLL * (1 - envelope)
                targets["r_shoulder_yaw"] = (-0.40 + 0.15 * wave_cycle) * envelope
                targets["r_elbow"] = (1.60 - 0.12 * wave_cycle) * envelope + DEFAULT_ELBOW * (1 - envelope)
                targets["r_wrist_prosup"] = 0.40 * envelope
                targets["r_wrist_yaw"] = (0.25 * wave_cycle) * envelope
                # Head maintains direct forward eye contact with user (neck joints remain neutral 0.0)
                # Naturally open hand
                targets["r_hand_index_0_joint"] = -0.20 * envelope
                targets["r_hand_little_0_joint"] = 0.20 * envelope

            elif name == "bow":
                # Deep, respectful Japanese bow (ojigi ~40 deg) with straight arms along thighs
                targets["torso_pitch"] = 0.72 * envelope
                targets["neck_pitch"] = -0.35 * envelope
                targets["r_shoulder_pitch"] = 0.08 * envelope + DEFAULT_SH_PITCH * (1 - envelope)
                targets["l_shoulder_pitch"] = 0.08 * envelope + DEFAULT_SH_PITCH * (1 - envelope)
                targets["r_shoulder_roll"] = 0.08 * envelope + DEFAULT_SH_ROLL * (1 - envelope)
                targets["l_shoulder_roll"] = 0.08 * envelope + DEFAULT_SH_ROLL * (1 - envelope)
                targets["r_elbow"] = 0.28 * envelope + DEFAULT_ELBOW * (1 - envelope)
                targets["l_elbow"] = 0.28 * envelope + DEFAULT_ELBOW * (1 - envelope)
                targets["r_wrist_pitch"] = 0.20 * envelope
                targets["l_wrist_pitch"] = 0.20 * envelope
                targets["r_wrist_prosup"] = -0.25 * envelope
                targets["l_wrist_prosup"] = 0.25 * envelope

            set_joint_positions(targets, velocity=3.8)
            await asyncio.sleep(dt)

        # Smooth return to neutral
        reset_robot_posture(velocity=2.0)
        print(f"[Gesture] Finished gesture '{name}' -> returned to neutral.")
    except Exception as e:
        print(f"[Gesture Error] Error animating gesture '{name}': {e}")


async def animate_speech_body_movement(
    duration: float,
    frequency: float = 3.0,
    pitch_amp: float = 0.40,
    yaw_amp: float = 0.38,
    roll_amp: float = 0.20,
    valence: float = 0.0,
    arousal: float = 0.0,
    prediction_error: float = 0.0,
    start_delay: float = 0.5,
):
    """Animate dynamic iCub head, torso, and arm gestures based on speech & Bayesian state."""
    global client
    if client is None:
        return

    if start_delay > 0:
        await asyncio.sleep(start_delay)

    start_time = asyncio.get_event_loop().time()
    dt = 0.03  # ~33 FPS motion steps
    seed_offset = (start_time * 100) % 100

    # Base default shoulder/elbow angles (in radians: deg * pi / 180)
    default_sh_pitch = DEFAULT_SH_PITCH
    default_sh_roll = DEFAULT_SH_ROLL
    default_elbow = DEFAULT_ELBOW

    while (asyncio.get_event_loop().time() - start_time) < duration:
        elapsed = asyncio.get_event_loop().time() - start_time
        
        # 1. Natural speech cadence (~3.0 Hz syllable rhythm)
        rhythm_pitch = math.sin(2 * math.pi * frequency * elapsed)
        rhythm_yaw = math.sin(2 * math.pi * (frequency * 0.6) * elapsed + 0.5)
        rhythm_roll = math.sin(2 * math.pi * (frequency * 0.4) * elapsed + 1.0)

        # 2. Head macro posture shifts (gaze dynamics)
        macro_yaw_shift = 0.20 * math.sin(2 * math.pi * 0.3 * elapsed + seed_offset)
        macro_pitch_shift = 0.10 * math.cos(2 * math.pi * 0.4 * elapsed)
        macro_roll_shift = 0.08 * math.sin(2 * math.pi * 0.25 * elapsed)

        neck_pitch_target = pitch_amp * rhythm_pitch + macro_pitch_shift
        neck_yaw_target = yaw_amp * rhythm_yaw + macro_yaw_shift
        neck_roll_target = roll_amp * rhythm_roll + macro_roll_shift

        # 3. Torso Affective Posture (Valence / Arousal driven)
        # High Arousal -> lean forward; Low Arousal / Negative Valence -> slump
        torso_pitch_base = 0.12 * arousal - 0.08 * valence
        torso_roll_base = 0.04 * valence
        torso_yaw_base = 0.06 * math.sin(2 * math.pi * 0.2 * elapsed)

        # 4. Arm Gesticulation & Rhythm Beats
        arousal_factor = max(0.2, (arousal + 1.0) / 2.0)  # [0.2 to 1.0]
        arm_rhythm = math.sin(2 * math.pi * (frequency * 0.8) * elapsed)
        
        r_sh_pitch_target = default_sh_pitch + 0.16 * arm_rhythm * arousal_factor
        l_sh_pitch_target = default_sh_pitch - 0.16 * arm_rhythm * arousal_factor
        r_sh_roll_target = default_sh_roll + 0.08 * math.cos(2 * math.pi * 0.4 * elapsed)
        l_sh_roll_target = default_sh_roll + 0.08 * math.sin(2 * math.pi * 0.4 * elapsed)
        r_elbow_target = default_elbow + 0.12 * arm_rhythm
        l_elbow_target = default_elbow - 0.12 * arm_rhythm

        # 5. Cognitive Surprise Reaction (High Prediction Error PE > 0.5)
        if prediction_error >= 0.5:
            pe_intensity = min(1.0, (prediction_error - 0.4) / 1.0)
            # Torso reclines backward in cognitive surprise
            torso_pitch_base -= 0.10 * pe_intensity
            # Head tilts to side in curiosity
            neck_roll_target += 0.12 * pe_intensity
            # Shrug arms open (palm-up elevation)
            r_sh_roll_target += 0.15 * pe_intensity
            l_sh_roll_target += 0.15 * pe_intensity
            r_elbow_target += 0.15 * pe_intensity
            l_elbow_target += 0.15 * pe_intensity
            r_sh_pitch_target += 0.10 * pe_intensity
            l_sh_pitch_target += 0.10 * pe_intensity

        # Clip joint targets safely within pyCub physical limits (all in radians)
        r_sh_pitch_target = float(np.clip(r_sh_pitch_target, -0.60, 1.80))
        l_sh_pitch_target = float(np.clip(l_sh_pitch_target, -0.60, 1.80))
        r_sh_roll_target = float(np.clip(r_sh_roll_target, 0.05, 2.50))
        l_sh_roll_target = float(np.clip(l_sh_roll_target, 0.05, 2.50))
        r_elbow_target = float(np.clip(r_elbow_target, 0.30, 1.80))
        l_elbow_target = float(np.clip(l_elbow_target, 0.30, 1.80))
        torso_pitch_base = float(np.clip(torso_pitch_base, -0.15, 0.80))
        torso_roll_base = float(np.clip(torso_roll_base, -0.40, 0.40))
        torso_yaw_base = float(np.clip(torso_yaw_base, -0.90, 0.90))
        neck_pitch_target = float(np.clip(neck_pitch_target, -0.65, 0.35))
        neck_roll_target = float(np.clip(neck_roll_target, -0.30, 0.30))
        neck_yaw_target = float(np.clip(neck_yaw_target, -0.65, 0.65))

        # Command pyCub joints with increased motor velocity (velocity=5.0) for responsive speech motion
        try:
            client.move_position("neck_pitch", neck_pitch_target, wait=False, velocity=5.0)
            client.move_position("neck_yaw", neck_yaw_target, wait=False, velocity=5.0)
            client.move_position("neck_roll", neck_roll_target, wait=False, velocity=5.0)
            client.move_position("eyes_tilt", 0.08 * math.sin(2 * math.pi * 0.5 * elapsed), wait=False, velocity=5.0)
            client.move_position("l_eye_pan_joint", 0.10 * math.cos(2 * math.pi * 0.4 * elapsed), wait=False, velocity=5.0)
            client.move_position("r_eye_pan_joint", 0.10 * math.cos(2 * math.pi * 0.4 * elapsed), wait=False, velocity=5.0)
            client.move_position("torso_pitch", torso_pitch_base, wait=False, velocity=4.0)
            client.move_position("torso_roll", torso_roll_base, wait=False, velocity=4.0)
            client.move_position("torso_yaw", torso_yaw_base, wait=False, velocity=4.0)
            client.move_position("r_shoulder_pitch", r_sh_pitch_target, wait=False, velocity=4.0)
            client.move_position("l_shoulder_pitch", l_sh_pitch_target, wait=False, velocity=4.0)
            client.move_position("r_shoulder_roll", r_sh_roll_target, wait=False, velocity=4.0)
            client.move_position("l_shoulder_roll", l_sh_roll_target, wait=False, velocity=4.0)
            client.move_position("r_elbow", r_elbow_target, wait=False, velocity=4.0)
            client.move_position("l_elbow", l_elbow_target, wait=False, velocity=4.0)
        except Exception:
            pass

        await asyncio.sleep(dt)

    # Smooth reset back to home posture
    reset_robot_posture(velocity=2.0)


# ── API Endpoints ───────────────────────────────────────────────────────────

@app.get("/status")
def robot_status():
    return {
        "status": "running",
        "pycub_loaded": client is not None,
        "pycub_alive": bool(client and client.is_alive()) if client else False,
    }


@app.post("/move")
def move_joint(request: MoveRequest):
    """Directly set position of an iCub joint."""
    if not client:
        raise HTTPException(status_code=503, detail="pyCub client is not initialized")
    try:
        client.move_position(request.joint, request.position, wait=False)
        return {"status": "success", "joint": request.joint, "position": request.position}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/gesture")
async def trigger_gesture_animation(request: GestureRequest):
    """Trigger a discrete LLM-selected gesture animation on the pyCub robot."""
    global gesture_task, talk_task, client
    if not client:
        return {"status": "mock", "message": f"pyCub not loaded; gesture '{request.gesture}' mocked"}

    # Cancel any ongoing speech micro-movement task so gesture has clean exclusive control
    if talk_task and not talk_task.done():
        talk_task.cancel()

    if gesture_task and not gesture_task.done():
        gesture_task.cancel()

    gesture_task = asyncio.create_task(
        animate_gesture(
            gesture=request.gesture,
            duration=request.duration,
            start_delay=request.start_delay,
        )
    )
    return {
        "status": "success",
        "gesture": request.gesture,
        "duration": request.duration,
        "start_delay": request.start_delay,
    }


@app.post("/talk")
async def trigger_talk_animation(request: TalkRequest):
    """Trigger speech-driven body & head micro-movements for a specified duration."""
    # Temporarily disabled to isolate and directly observe discrete LLM gestures
    return {"status": "disabled", "message": "Speech micro-movements temporarily disabled for gesture isolation"}


@app.post("/talk_audio")
async def trigger_talk_from_audio(file: UploadFile = File(...)):
    """Trigger speech head animation from uploaded audio bytes."""
    content = await file.read()
    
    # Estimate audio duration from standard WAV header or raw bytes length
    # Defaulting to 16kHz 16-bit mono -> 32000 bytes per second
    duration = max(0.5, len(content) / 32000.0)

    try:
        import soundfile as sf
        audio_data, sr = sf.read(io.BytesIO(content))
        duration = len(audio_data) / float(sr)
    except Exception:
        pass

    return await trigger_talk_animation(TalkRequest(duration=duration))


@app.post("/reset")
def reset_robot():
    """Reset head, torso, arm, and finger joints to default neutral position."""
    if not client:
        return {"status": "mock"}
    try:
        reset_robot_posture(velocity=5.0)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Mount Predictive Processing API Endpoints ──────────────────────────────

try:
    if str(_project_root / "src") not in sys.path:
        sys.path.insert(0, str(_project_root / "src"))
    from predictive_processing.api.server import router as pp_router
    app.include_router(pp_router)
except Exception as exc:
    print(f"Warning: Could not mount predictive processing API routes: {exc}")

# ── Mount Pre-built React Frontend (No Node.js / NPM needed on Mac) ────────
frontend_dist = _project_root / "frontend" / "dist"
if frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
    print(f"[Static Web UI] Successfully mounted React app from {frontend_dist}")
