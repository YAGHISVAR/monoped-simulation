"""
Monoped — Autonomous Sit-Stand Trajectory
==========================================
Automatically cycles between sit (0.12 m) and stand (0.36 m) using a
trapezoidal velocity profile.  No user input needed — just watch it go.

Profile parameters:
  V_MAX = 0.15 m/s  (cruise speed)
  A_MAX = 0.20 m/s² (acceleration / deceleration)
"""

import mujoco
import mujoco.viewer
import numpy as np

# ── Mechanism constants ────────────────────────────────────────
L            = 0.20
HEIGHT_SIT   = 0.12   # m — crouched
HEIGHT_STAND = 0.36   # m — upright
HOLD_TIME    = 1.0    # s — pause at each end before reversing

TRAP_V_MAX = 0.15     # m/s
TRAP_A_MAX = 0.20     # m/s²


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
# Alternate targets: STAND → SIT → STAND → ...
targets  = [HEIGHT_STAND, HEIGHT_SIT]
phase    = 0                                     # index into targets[]
traj     = TrapezoidalTrajectory(HEIGHT_STAND, HEIGHT_SIT)
t_start  = 0.5                                   # let physics settle first
holding  = False
hold_end = 0.0

print("=" * 52)
print("  Monoped — Sit-Stand Trajectory")
print(f"  stand={HEIGHT_STAND} m  sit={HEIGHT_SIT} m")
print(f"  V_MAX={TRAP_V_MAX} m/s  A_MAX={TRAP_A_MAX} m/s²")
print(f"  move time (full stroke): {traj.t_total:.2f} s")
print("=" * 52)

# ── Simulation loop ────────────────────────────────────────────
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth   = 90
    viewer.cam.elevation = -12
    viewer.cam.distance  = 1.6
    viewer.cam.lookat[:] = [0, 0, 0.25]

    log_t = -1.0

    while viewer.is_running():
        t = data.time

        # ── Settle: hold standing for the first 0.5 s ──────────
        if t < t_start:
            h_cmd = HEIGHT_STAND

        elif holding:
            h_cmd = targets[phase]
            if t >= hold_end:
                # Switch to next target and build new trajectory
                phase    = 1 - phase
                src      = targets[1 - phase]
                dst      = targets[phase]
                traj     = TrapezoidalTrajectory(src, dst)
                t_start  = t
                holding  = False
                label    = "STAND" if dst == HEIGHT_STAND else "SIT"
                print(f"  t={t:.2f}s → {label} ({dst:.2f} m)  "
                      f"t_acc={traj.t_acc:.2f}s  "
                      f"t_cruise={traj.t_cruise:.2f}s  "
                      f"t_total={traj.t_total:.2f}s")

        else:
            elapsed = t - t_start
            h_cmd   = traj.position(elapsed)

            # Trajectory complete — enter hold
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

        if t - log_t >= 0.5:
            log_t = t
            pz    = fk_pelvis_z(hip_now)
            state = "HOLD" if holding else "MOVE"
            print(f"  [{state}] t={t:6.2f}s | "
                  f"pelvis={pz:.3f}m  ref={h_cmd:.3f}m  "
                  f"err={targets[phase]-pz:+.4f}m")
