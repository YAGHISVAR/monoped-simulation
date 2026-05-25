"""
Monoped — Crank-Slider Mechanism (Inverted)
=============================================
Motor (hip) is in the pelvis and rotates the crank (upper_link).
A passive connecting rod (lower_link) is pinned at the crank tip.
An equality constraint pins the rod tip to a fixed foot on the ground.
The pelvis slides only vertically — it is the output (moving body).

Mechanism geometry:
  L_crank  = 0.25 m  (upper_link)
  L_rod    = 0.45 m  (lower_link)
  foot_z   = 0.00 m  (fixed ground anchor)

Pelvis Z (forward kinematics):
  pelvis_z = L_crank * cos(hip) + sqrt(L_rod² − L_crank² * sin²(hip))
  → At hip=0:    pelvis_z = 0.70 m  (highest — fully extended)
  → At hip=±90°: pelvis_z ≈ 0.37 m  (lowest — fully compressed)

Sensor indices:
  [0:3]  imu_accel
  [3]    hip_pos
  [4]    conn_pos
  [5]    slide_z_pos
  [6]    hip_vel
  [7]    conn_vel
  [8]    slide_z_vel
  [9:12] rod_tip_pos
  [12:15]pelvis_pos
"""

import mujoco
import mujoco.viewer
import numpy as np

# ── Mechanism constants ────────────────────────────────────────
L_CRANK  = 0.20
L_ROD    = 0.20

def fk_pelvis_z(hip_angle: float) -> float:
    """World Z of pelvis given hip angle (inverted crank-slider)."""
    sin_h = np.sin(hip_angle)
    d = max(L_ROD**2 - L_CRANK**2 * sin_h**2, 0.0)
    return L_CRANK * np.cos(hip_angle) + np.sqrt(d)


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
print("  Monoped — Crank-Slider (Inverted)")
print("=" * 55)
print(f"  nq={model.nq}  nu={model.nu}  nsensordata={model.nsensordata}")
print(f"  L_crank={L_CRANK}m  L_rod={L_ROD}m")
print(f"  Pelvis Z: {fk_pelvis_z(np.pi/2):.3f}m (hip=90°, lowest) "
      f"→ {fk_pelvis_z(0.0):.3f}m (hip=0°, highest)")
print("=" * 55)

# ── Sensor indices ─────────────────────────────────────────────
S_ACCEL    = slice(0, 3)
S_HIP      = 3
S_CONN     = 4
S_SLIDE_Z  = 5
S_HIP_VEL  = 6
S_SLIDE_Z_VEL = 8

C_HIP = 0   # only one actuator

pid_hip = PID(Kp=9.0, Ki=0.05, Kd=0.80, dt=dt)

# ── Gait target ────────────────────────────────────────────────
SETTLE = 0.5   # hold at hip=0 (pelvis highest) to let constraint settle

def hip_target(t: float) -> float:
    if t < SETTLE:
        return 0.0
    ts   = t - SETTLE
    freq = 0.8                    # Hz
    amp  = 1.10                   # rad — pelvis travels ~0.28 m vertically
    # Offset so oscillation stays on the positive side (compressing downward)
    return amp * 0.5 * (1.0 - np.cos(2 * np.pi * freq * ts))

# ── Main loop ──────────────────────────────────────────────────
print("\nLaunching viewer — red pelvis bobs up/down, foot stays on ground.\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth   = 90
    viewer.cam.elevation = -12
    viewer.cam.distance  = 2.0
    viewer.cam.lookat[:] = [0, 0, 0.4]

    step = 0
    while viewer.is_running():
        t = data.time

        hip_tgt          = hip_target(t)
        hip_now          = data.sensordata[S_HIP]
        data.ctrl[C_HIP] = pid_hip.compute(hip_tgt, hip_now)

        mujoco.mj_step(model, data)
        viewer.sync()
        step += 1

        if step % 500 == 0:
            sz   = data.sensordata[S_SLIDE_Z]
            conn = data.sensordata[S_CONN]
            pz   = fk_pelvis_z(hip_now)
            phase = "SETTLE" if t < SETTLE else "RUN"
            print(f"[{phase}] t={t:5.2f}s | "
                  f"hip={hip_now:+.3f}rad (tgt={hip_tgt:+.3f}) | "
                  f"conn={conn:+.3f}rad | "
                  f"slide_z={sz:+.3f}m | pelvis_z={pz:.3f}m")
