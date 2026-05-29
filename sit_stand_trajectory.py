"""
Monoped — Autonomous Sit-Stand Trajectory
==========================================
Automatically cycles between sit (0.12 m) and stand (0.36 m) using a
trapezoidal velocity profile.  No user input needed — just watch it go.

Profile parameters (slowed down):
  V_MAX = 0.06 m/s  (cruise speed)
  A_MAX = 0.06 m/s² (acceleration / deceleration)

Live matplotlib plot shows commanded vs actual pelvis height.
"""

import mujoco
import mujoco.viewer
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

matplotlib.use("TkAgg")   # works alongside MuJoCo's passive viewer

# ── Mechanism constants ────────────────────────────────────────
L            = 0.20
HEIGHT_SIT   = 0.12   # m — crouched
HEIGHT_STAND = 0.36   # m — upright
HOLD_TIME    = 1.5    # s — pause at each end before reversing

# ── Slowed-down trapezoidal profile ───────────────────────────
TRAP_V_MAX = 0.06     # m/s  (was 0.15)
TRAP_A_MAX = 0.06     # m/s² (was 0.20)

# ── Plot update interval (sim-seconds between redraws) ────────
PLOT_INTERVAL = 0.05  # s  → ~20 fps refresh


def fk_pelvis_z(hip: float) -> float:
    return 2.0 * L * np.cos(hip)


def ik_hip_angle(z: float) -> float:
    z = np.clip(z, 0.06 + 1e-4, 0.40 - 1e-4)
    return float(np.arccos(np.clip(z / (2.0 * L), 0.0, 1.0)))


# ── Trapezoidal trajectory ─────────────────────────────────────
class TrapezoidalTrajectory:
    def __init__(self, x0: float, x1: float):
        self.x0   = x0
        dist      = abs(x1 - x0)
        self.sign = np.sign(x1 - x0) if x1 != x0 else 0.0

        t_acc_full = TRAP_V_MAX / TRAP_A_MAX
        d_acc_full = 0.5 * TRAP_A_MAX * t_acc_full ** 2

        if 2.0 * d_acc_full >= dist:
            # Triangular — too short to reach V_MAX
            self.t_acc    = np.sqrt(dist / TRAP_A_MAX)
            self.t_cruise = 0.0
            self.v_peak   = TRAP_A_MAX * self.t_acc
        else:
            self.t_acc    = t_acc_full
            self.v_peak   = TRAP_V_MAX
            self.t_cruise = (dist - 2.0 * d_acc_full) / TRAP_V_MAX

        self.t_total = 2.0 * self.t_acc + self.t_cruise
        self._d_acc  = 0.5 * TRAP_A_MAX * self.t_acc ** 2

    def position(self, elapsed: float) -> float:
        t = np.clip(elapsed, 0.0, self.t_total)
        if t <= self.t_acc:
            s = 0.5 * TRAP_A_MAX * t ** 2
        elif t <= self.t_acc + self.t_cruise:
            s = self._d_acc + self.v_peak * (t - self.t_acc)
        else:
            t_dec = t - self.t_acc - self.t_cruise
            s = self._d_acc + self.v_peak * self.t_cruise + \
                self.v_peak * t_dec - 0.5 * TRAP_A_MAX * t_dec ** 2
        return self.x0 + self.sign * s


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


# ── Live matplotlib figure ─────────────────────────────────────
plt.ion()
fig = plt.figure(figsize=(8, 4), facecolor="#1e1e2e")
fig.canvas.manager.set_window_title("Sit-Stand — Live Position")

gs  = gridspec.GridSpec(1, 1, figure=fig)
ax  = fig.add_subplot(gs[0])
ax.set_facecolor("#1e1e2e")
for spine in ax.spines.values():
    spine.set_edgecolor("#6c7086")

ax.set_title("Pelvis Height (live)", color="#cdd6f4", fontsize=13, pad=8)
ax.set_xlabel("Simulation time  (s)", color="#a6adc8")
ax.set_ylabel("Height  (m)",          color="#a6adc8")
ax.tick_params(colors="#a6adc8")
ax.set_ylim(0.05, 0.42)
ax.set_xlim(0, 20)

# Reference lines
ax.axhline(HEIGHT_STAND, color="#a6e3a1", linestyle="--",
           linewidth=1, alpha=0.5, label=f"Stand ref  {HEIGHT_STAND} m")
ax.axhline(HEIGHT_SIT,   color="#f38ba8", linestyle="--",
           linewidth=1, alpha=0.5, label=f"Sit ref  {HEIGHT_SIT} m")

# Live data lines
line_cmd, = ax.plot([], [], color="#89b4fa", linewidth=1.8, label="Commanded")
line_act, = ax.plot([], [], color="#fab387", linewidth=1.5,
                    linestyle=":", label="Actual")

ax.legend(facecolor="#313244", edgecolor="#6c7086",
          labelcolor="#cdd6f4", fontsize=9, loc="upper right")

fig.tight_layout(pad=1.5)
fig.canvas.draw()
plt.pause(0.001)

# Scrolling window width (seconds shown at once)
WINDOW = 20.0

# Data buffers
buf_t   = []
buf_cmd = []
buf_act = []


def update_plot(t_sim: float, h_cmd: float, h_act: float) -> None:
    """Append new sample and redraw the live plot."""
    buf_t.append(t_sim)
    buf_cmd.append(h_cmd)
    buf_act.append(h_act)

    arr_t   = np.asarray(buf_t)
    arr_cmd = np.asarray(buf_cmd)
    arr_act = np.asarray(buf_act)

    # Scrolling x-axis
    x_max = max(t_sim, WINDOW)
    x_min = x_max - WINDOW
    ax.set_xlim(x_min, x_max)

    # Only draw the visible slice for performance
    mask = arr_t >= x_min
    line_cmd.set_data(arr_t[mask], arr_cmd[mask])
    line_act.set_data(arr_t[mask], arr_act[mask])

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

pid_hip  = PID(Kp=9.0, Ki=0.05, Kd=0.80, dt=dt)
pid_knee = PID(Kp=7.0, Ki=0.05, Kd=0.60, dt=dt)

S_HIP, S_KNEE = 3, 4

# ── Trajectory state machine ───────────────────────────────────
targets  = [HEIGHT_STAND, HEIGHT_SIT]
phase    = 0
traj     = TrapezoidalTrajectory(HEIGHT_STAND, HEIGHT_SIT)
t_start  = 0.5
holding  = False
hold_end = 0.0

print("=" * 60)
print("  Monoped — Sit-Stand Trajectory  (SLOW)")
print(f"  stand={HEIGHT_STAND} m   sit={HEIGHT_SIT} m")
print(f"  V_MAX={TRAP_V_MAX} m/s   A_MAX={TRAP_A_MAX} m/s²")
print(f"  move time (full stroke): {traj.t_total:.2f} s")
print("=" * 60)

log_t    = -1.0
plot_t   = -1.0

# ── Simulation loop ────────────────────────────────────────────
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth   = 90
    viewer.cam.elevation = -12
    viewer.cam.distance  = 1.6
    viewer.cam.lookat[:] = [0, 0, 0.25]

    while viewer.is_running():
        t = data.time

        # ── Settle ─────────────────────────────────────────────
        if t < t_start:
            h_cmd = HEIGHT_STAND

        elif holding:
            h_cmd = targets[phase]
            if t >= hold_end:
                phase    = 1 - phase
                src      = targets[1 - phase]
                dst      = targets[phase]
                traj     = TrapezoidalTrajectory(src, dst)
                t_start  = t
                holding  = False
                label    = "STAND" if dst == HEIGHT_STAND else "SIT"
                print(f"  t={t:.2f}s → {label} ({dst:.2f} m)  "
                      f"t_total={traj.t_total:.2f}s")

        else:
            elapsed = t - t_start
            h_cmd   = traj.position(elapsed)
            if elapsed >= traj.t_total:
                holding  = True
                hold_end = t + HOLD_TIME

        # ── IK → PID → actuators ───────────────────────────────
        hip_tgt  = ik_hip_angle(h_cmd)
        knee_tgt = -2.0 * hip_tgt
        hip_now  = float(data.sensordata[S_HIP])
        knee_now = float(data.sensordata[S_KNEE])

        data.ctrl[0] = pid_hip.compute(hip_tgt, hip_now)
        data.ctrl[1] = pid_knee.compute(knee_tgt, knee_now)

        mujoco.mj_step(model, data)
        viewer.sync()

        # ── Live plot update ────────────────────────────────────
        if t - plot_t >= PLOT_INTERVAL:
            plot_t = t
            h_act  = fk_pelvis_z(hip_now)
            update_plot(t, h_cmd, h_act)

        # ── Console log ────────────────────────────────────────
        if t - log_t >= 0.5:
            log_t = t
            h_act = fk_pelvis_z(hip_now)
            state = "HOLD" if holding else "MOVE"
            print(f"  [{state}] t={t:6.2f}s | "
                  f"pelvis={h_act:.3f}m  ref={h_cmd:.3f}m  "
                  f"err={h_cmd-h_act:+.4f}m")

plt.ioff()
plt.show()
