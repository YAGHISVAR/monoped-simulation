"""
Monoped — 2-BLDC-Motor Leg (Sinusoidal Gait)
==============================================
Motor 1 (hip)  — direct drive on upper_link
Motor 2 (knee) — chain drive on lower_link

IK: knee = -2 * hip,  pelvis_z = 2·L·cos(hip)
  At hip=0:      pelvis_z = 0.40 m (fully extended)
  At hip=~74°:   pelvis_z = 0.11 m (crouched)

Sensor indices:
  [0:3]  imu_accel
  [3]    hip_pos
  [4]    knee_pos
  [5]    slide_z_pos
  [6]    hip_vel
  [7]    knee_vel
  [8]    slide_z_vel
  [9:12] rod_tip_pos
  [12:15]pelvis_pos
"""

import mujoco
import mujoco.viewer
import numpy as np

# ── Mechanism constants ────────────────────────────────────────
L = 0.20

def fk_pelvis_z(hip: float) -> float:
    return 2.0 * L * np.cos(hip)

def ik_hip_angle(target_z: float) -> float:
    target_z = np.clip(target_z, 0.06 + 1e-4, 0.40 - 1e-4)
    return float(np.arccos(np.clip(target_z / (2.0 * L), 0.0, 1.0)))


# ── PID ───────────────────────────────────────────────────────
class PID:
    def __init__(self, Kp, Ki, Kd, dt, limit=1.0):
        self.Kp = Kp; self.Ki = Ki; self.Kd = Kd
        self.dt = dt; self.limit = limit
        self.integral = 0.0; self.prev_err = 0.0

    def reset(self):
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

print("=" * 55)
print("  Monoped — 2-BLDC-Motor Leg")
print("=" * 55)
print(f"  nq={model.nq}  nu={model.nu}  nsensordata={model.nsensordata}")
print(f"  L={L}m  (equal links)")
print(f"  Pelvis Z range: {fk_pelvis_z(np.radians(74)):.3f}m → {fk_pelvis_z(0.0):.3f}m")
print("=" * 55)

# ── Sensor / actuator indices ──────────────────────────────────
S_HIP      = 3
S_KNEE     = 4
S_SLIDE_Z  = 5

C_HIP  = 0
C_KNEE = 1

# Initial pose — standing
hip_init     = ik_hip_angle(0.36)
data.qpos[1] = hip_init
data.qpos[2] = -2.0 * hip_init
mujoco.mj_forward(model, data)

pid_hip  = PID(Kp=9.0, Ki=0.05, Kd=0.80, dt=dt)
pid_knee = PID(Kp=7.0, Ki=0.05, Kd=0.60, dt=dt)

# ── Gait target ────────────────────────────────────────────────
SETTLE = 0.5

def hip_target(t: float) -> float:
    if t < SETTLE:
        return ik_hip_angle(0.36)
    ts   = t - SETTLE
    freq = 0.8
    # Oscillates between ~hip(0.36) and ~hip(0.14)
    h_cmd = 0.25 + 0.11 * np.cos(2 * np.pi * freq * ts)
    return ik_hip_angle(h_cmd)

# ── Main loop ──────────────────────────────────────────────────
print("\nLaunching viewer — pelvis bobs up/down, both motors active.\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth   = 90
    viewer.cam.elevation = -12
    viewer.cam.distance  = 1.6
    viewer.cam.lookat[:] = [0, 0, 0.25]

    step = 0
    while viewer.is_running():
        t = data.time

        hip_tgt  = hip_target(t)
        knee_tgt = -2.0 * hip_tgt

        hip_now  = data.sensordata[S_HIP]
        knee_now = data.sensordata[S_KNEE]

        data.ctrl[C_HIP]  = pid_hip.compute(hip_tgt, hip_now)
        data.ctrl[C_KNEE] = pid_knee.compute(knee_tgt, knee_now)

        mujoco.mj_step(model, data)
        viewer.sync()
        step += 1

        if step % 500 == 0:
            sz  = data.sensordata[S_SLIDE_Z]
            pz  = fk_pelvis_z(hip_now)
            phase = "SETTLE" if t < SETTLE else "RUN"
            print(f"[{phase}] t={t:5.2f}s | "
                  f"hip={np.degrees(hip_now):+6.1f}° (tgt={np.degrees(hip_tgt):+5.1f}°) | "
                  f"knee={np.degrees(knee_now):+6.1f}° (tgt={np.degrees(knee_tgt):+5.1f}°) | "
                  f"pelvis_z={pz:.3f}m | slide_z={sz:+.3f}m")
