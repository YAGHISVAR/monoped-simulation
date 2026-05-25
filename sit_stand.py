"""
Monoped — Height Control
=========================
Type a target height (m) at the prompt and press Enter.
The angle block at the top updates in place — typing is never interrupted.

Mechanism: equal links L = 0.20 m
  pelvis_z = 2 × L × cos(hip)
  Range: 0.06 m (crouched)  →  0.40 m (extended)
"""

import mujoco
import mujoco.viewer
import numpy as np
import sys
import threading

# ── Mechanism constants ────────────────────────────────────────
L            = 0.20
HEIGHT_MIN   = 0.06
HEIGHT_MAX   = 0.40
HEIGHT_STAND = 0.36
DURATION     = 1.5


def fk_pelvis_z(hip: float) -> float:
    return 2.0 * L * np.cos(hip)


def ik_hip_angle(target_z: float) -> float:
    target_z = np.clip(target_z, HEIGHT_MIN + 1e-4, HEIGHT_MAX - 1e-4)
    return float(np.arccos(np.clip(target_z / (2.0 * L), 0.0, 1.0)))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-10.0 * (x - 0.5)))


# ── Shared state ───────────────────────────────────────────────
state = {"src": HEIGHT_STAND, "target": HEIGHT_STAND, "t_start": 0.0}
active       = [True]
sim_time_ref = [0.0]
current_pz   = [HEIGHT_STAND]


def commanded_height(t: float) -> float:
    elapsed = t - state["t_start"]
    alpha   = _sigmoid(np.clip(elapsed / DURATION, 0.0, 1.0))
    return state["src"] + alpha * (state["target"] - state["src"])


# ── PID ───────────────────────────────────────────────────────
class PID:
    def __init__(self, Kp, Ki, Kd, dt, limit=1.0):
        self.Kp = Kp; self.Ki = Ki; self.Kd = Kd
        self.dt = dt; self.limit = limit
        self.integral = 0.0; self.prev_err = 0.0

    def compute(self, target, current):
        err           = target - current
        self.integral = np.clip(self.integral + err * self.dt, -2.0, 2.0)
        deriv         = (err - self.prev_err) / self.dt
        self.prev_err = err
        return np.clip(self.Kp * err + self.Ki * self.integral + self.Kd * deriv,
                       -self.limit, self.limit)


# ── Terminal helpers (ANSI — no external libs) ─────────────────
#
# Layout (fixed rows at top, input scrolls below):
#   row 1 │ ════ header ════
#   row 2 │  Monoped — Height Control
#   row 3 │ ════════════════
#   row 4 │  [angle line 1]          ← updated in place
#   row 5 │  [angle line 2]          ← updated in place
#   row 6 │ ────────────────
#   row 7+│  (input prompts scroll here naturally)

ANGLE_ROW_1 = 4   # 1-indexed ANSI rows
ANGLE_ROW_2 = 5


def setup_display():
    W = 66
    sys.stdout.write("\033[2J\033[H")          # clear screen + home
    sys.stdout.write("═" * W + "\n")
    sys.stdout.write(f"  Monoped — Height Control"
                     f"{'':>15}L = {L:.2f} m\n")
    sys.stdout.write("═" * W + "\n")
    sys.stdout.write("\n")                     # row 4 placeholder
    sys.stdout.write("\n")                     # row 5 placeholder
    sys.stdout.write("─" * W + "\n")          # row 6 separator
    sys.stdout.write(f"  Range : {HEIGHT_MIN:.2f} m  →  {HEIGHT_MAX:.2f} m\n")
    sys.stdout.write(f"  Stand : {HEIGHT_STAND:.2f} m   "
                     f"Crouch : 0.14 m   q = quit\n")
    sys.stdout.write("\n")
    sys.stdout.flush()


def update_angles(line1: str, line2: str):
    """Overwrite the two fixed angle rows without moving the input cursor."""
    sys.stdout.write(
        f"\033[s"                               # save cursor
        f"\033[{ANGLE_ROW_1};1H\033[2K{line1}" # row 4: clear + write
        f"\033[{ANGLE_ROW_2};1H\033[2K{line2}" # row 5: clear + write
        f"\033[u"                               # restore cursor
    )
    sys.stdout.flush()


# ── Input thread ───────────────────────────────────────────────
def input_loop():
    while active[0]:
        try:
            raw = input("  height (m) > ").strip()
            if raw.lower() in ("q", "quit", "exit"):
                active[0] = False
                break
            if not raw:
                continue
            h = float(raw)
            h = np.clip(h, HEIGHT_MIN, HEIGHT_MAX)
            hip_req = ik_hip_angle(h)
            state["src"]     = current_pz[0]
            state["target"]  = h
            state["t_start"] = sim_time_ref[0]
            print(f"  → {h:.3f} m  |  hip_target = "
                  f"{np.degrees(hip_req):.1f}°  ({hip_req:.3f} rad)")
        except ValueError:
            print(f"  ! enter a number between "
                  f"{HEIGHT_MIN:.2f} and {HEIGHT_MAX:.2f}")
        except EOFError:
            break


# ── Load model ─────────────────────────────────────────────────
model = mujoco.MjModel.from_xml_path("monoped.xml")
data  = mujoco.MjData(model)
dt    = model.opt.timestep

# qpos: [0]=slide_z  [1]=hip  [2]=connect_hinge
data.qpos[1] = ik_hip_angle(HEIGHT_STAND)
mujoco.mj_forward(model, data)

pid = PID(Kp=9.0, Ki=0.05, Kd=0.80, dt=dt)

setup_display()

t_input = threading.Thread(target=input_loop, daemon=True)
t_input.start()

# Sensor indices
S_HIP, S_CONN, S_SLIDE_Z = 3, 4, 5

# ── Simulation loop ────────────────────────────────────────────
last_print = -1.0
PRINT_DT   = 0.30
SETTLE     = 0.5

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth   = 90
    viewer.cam.elevation = -12
    viewer.cam.distance  = 1.6
    viewer.cam.lookat[:] = [0, 0, 0.25]

    while viewer.is_running() and active[0]:
        t               = data.time
        hip_now         = float(data.sensordata[S_HIP])
        current_pz[0]   = fk_pelvis_z(hip_now)
        sim_time_ref[0] = t

        hip_tgt      = ik_hip_angle(
            HEIGHT_STAND if t < SETTLE else commanded_height(t))
        data.ctrl[0] = pid.compute(hip_tgt, hip_now)

        mujoco.mj_step(model, data)
        viewer.sync()

        if t - last_print >= PRINT_DT:
            last_print = t
            conn = float(data.sensordata[S_CONN])
            sz   = float(data.sensordata[S_SLIDE_Z])
            pz   = current_pz[0]
            update_angles(
                f"  t = {t:6.2f} s  |  "
                f"pelvis = {pz:.3f} m   "
                f"target = {state['target']:.3f} m   "
                f"error = {state['target'] - pz:+.3f} m",

                f"  hip = {np.degrees(hip_now):+6.1f} deg  |  "
                f"conn = {np.degrees(conn):+6.1f} deg  |  "
                f"slide_z = {sz:+.4f} m"
            )
