"""
Monoped — Height Control (2-BLDC-motor)
========================================
Type a target height (m) at the prompt and press Enter.
The angle block at the top updates in place — typing is never interrupted.
A live plot shows PID height error since the last setpoint was set.

Mechanism: equal links L = 0.20 m
  Motor 1 (hip)  — direct drive on upper_link
  Motor 2 (knee) — chain drive on lower_link
  IK: knee = -2 * hip,  pelvis_z = 2L·cos(hip)
  Range: 0.06 m (crouched)  →  0.40 m (extended)
"""

import mujoco
import mujoco.viewer
import numpy as np
import sys
import threading
import matplotlib.pyplot as plt

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

# Plot data: filled after each setpoint command
plot_data = {"times": [], "errors": [], "t0": None, "target": HEIGHT_STAND}


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


# ── Terminal helpers (ANSI) ────────────────────────────────────
ANGLE_ROW_1 = 4
ANGLE_ROW_2 = 5


def setup_display():
    W = 66
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.write("═" * W + "\n")
    sys.stdout.write(f"  Monoped — Height Control (2-BLDC)"
                     f"{'':>10}L = {L:.2f} m\n")
    sys.stdout.write("═" * W + "\n")
    sys.stdout.write("\n")
    sys.stdout.write("\n")
    sys.stdout.write("─" * W + "\n")
    sys.stdout.write(f"  Range : {HEIGHT_MIN:.2f} m  →  {HEIGHT_MAX:.2f} m\n")
    sys.stdout.write(f"  Stand : {HEIGHT_STAND:.2f} m   "
                     f"Crouch : 0.14 m   q = quit\n")
    sys.stdout.write("\n")
    sys.stdout.flush()


def update_angles(line1: str, line2: str):
    sys.stdout.write(
        f"\033[s"
        f"\033[{ANGLE_ROW_1};1H\033[2K{line1}"
        f"\033[{ANGLE_ROW_2};1H\033[2K{line2}"
        f"\033[u"
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
            hip_req  = ik_hip_angle(h)
            knee_req = -2.0 * hip_req
            state["src"]     = current_pz[0]
            state["target"]  = h
            state["t_start"] = sim_time_ref[0]
            # Reset error plot for this new setpoint
            plot_data["times"]  = []
            plot_data["errors"] = []
            plot_data["t0"]     = sim_time_ref[0]
            plot_data["target"] = h
            print(f"  → {h:.3f} m  |  hip = {np.degrees(hip_req):.1f}°  "
                  f"knee = {np.degrees(knee_req):.1f}°")
        except ValueError:
            print(f"  ! enter a number between "
                  f"{HEIGHT_MIN:.2f} and {HEIGHT_MAX:.2f}")
        except EOFError:
            break


# ── Live error plot ────────────────────────────────────────────
plt.ion()
fig, ax = plt.subplots(figsize=(7, 3))
ax.set_xlabel("time since setpoint (s)")
ax.set_ylabel("height error (m)")
ax.set_title("PID height error")
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
(err_line,) = ax.plot([], [], color="#f38ba8", linewidth=1.8, label="error")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
plt.show(block=False)


def refresh_plot():
    ts = plot_data["times"]
    es = plot_data["errors"]
    if len(ts) < 2:
        return
    err_line.set_data(ts, es)
    ax.set_title(f"PID height error  (target = {plot_data['target']:.3f} m)")
    ax.relim()
    ax.autoscale_view()
    # keep y-axis symmetric around zero for readability
    ylim = ax.get_ylim()
    bound = max(abs(ylim[0]), abs(ylim[1]), 0.01)
    ax.set_ylim(-bound, bound)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


# ── Load model ─────────────────────────────────────────────────
model = mujoco.MjModel.from_xml_path("monoped.xml")
data  = mujoco.MjData(model)
dt    = model.opt.timestep

hip_init     = ik_hip_angle(HEIGHT_STAND)
data.qpos[1] = hip_init
data.qpos[2] = -2.0 * hip_init
mujoco.mj_forward(model, data)

pid_hip  = PID(Kp=9.0,  Ki=0.05, Kd=0.80, dt=dt)
pid_knee = PID(Kp=7.0,  Ki=0.05, Kd=0.60, dt=dt)

setup_display()

t_input = threading.Thread(target=input_loop, daemon=True)
t_input.start()

S_HIP, S_KNEE, S_SLIDE_Z = 3, 4, 5

# ── Simulation loop ────────────────────────────────────────────
last_print = -1.0
PRINT_DT   = 0.10   # collect error data at 10 Hz
SETTLE     = 0.5

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth   = 90
    viewer.cam.elevation = -12
    viewer.cam.distance  = 1.6
    viewer.cam.lookat[:] = [0, 0, 0.25]

    while viewer.is_running() and active[0]:
        t               = data.time
        hip_now         = float(data.sensordata[S_HIP])
        knee_now        = float(data.sensordata[S_KNEE])
        current_pz[0]   = fk_pelvis_z(hip_now)
        sim_time_ref[0] = t

        h_cmd    = HEIGHT_STAND if t < SETTLE else commanded_height(t)
        hip_tgt  = ik_hip_angle(h_cmd)
        knee_tgt = -2.0 * hip_tgt

        data.ctrl[0] = pid_hip.compute(hip_tgt, hip_now)
        data.ctrl[1] = pid_knee.compute(knee_tgt, knee_now)

        mujoco.mj_step(model, data)
        viewer.sync()

        if t - last_print >= PRINT_DT:
            last_print = t
            sz  = float(data.sensordata[S_SLIDE_Z])
            pz  = current_pz[0]
            err = state["target"] - pz

            # Accumulate error data after a setpoint has been set
            if plot_data["t0"] is not None:
                elapsed = t - plot_data["t0"]
                plot_data["times"].append(elapsed)
                plot_data["errors"].append(err)

            update_angles(
                f"  t = {t:6.2f} s  |  "
                f"pelvis = {pz:.3f} m   "
                f"target = {state['target']:.3f} m   "
                f"error = {err:+.4f} m",

                f"  hip = {np.degrees(hip_now):+6.1f}°  "
                f"(tgt {np.degrees(hip_tgt):+5.1f}°)  |  "
                f"knee = {np.degrees(knee_now):+6.1f}°  "
                f"(tgt {np.degrees(knee_tgt):+5.1f}°)  |  "
                f"slide_z = {sz:+.4f} m"
            )

            # Refresh the matplotlib window (non-blocking)
            refresh_plot()
