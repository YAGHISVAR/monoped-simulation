"""
Monoped — Height Control
=========================
Type a target pelvis height (m) and press Enter.
The hip motor drives the crank to reach that height.
Joint angles are printed to the terminal in real time.

Mechanism (equal links, L = 0.20 m):
  pelvis_z = 2 * L * cos(hip)
  Range: 0.06 m (crouched) → 0.40 m (extended)
"""

import mujoco
import mujoco.viewer
import numpy as np
import threading

# ── Mechanism constants ────────────────────────────────────────
L       = 0.20
HEIGHT_MIN   = 0.06
HEIGHT_MAX   = 0.40
HEIGHT_STAND = 0.36
DURATION     = 1.5   # transition time (s)


def fk_pelvis_z(hip: float) -> float:
    sin_h = np.sin(hip)
    return L * np.cos(hip) + np.sqrt(max((L**2) - (L**2) * sin_h**2, 0.0))


def ik_hip_angle(target_z: float) -> float:
    """Bisection IK: pelvis_z = 2*L*cos(hip)  →  hip = arccos(z / 2L)."""
    target_z = np.clip(target_z, HEIGHT_MIN + 1e-4, HEIGHT_MAX - 1e-4)
    return np.arccos(np.clip(target_z / (2 * L), 0.0, 1.0))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-10.0 * (x - 0.5)))


# ── Shared state ───────────────────────────────────────────────
state = {
    "src"    : HEIGHT_STAND,
    "target" : HEIGHT_STAND,
    "t_start": 0.0,
}
active         = [True]
sim_time_ref   = [0.0]
current_pz     = [HEIGHT_STAND]


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


# ── Terminal input thread ──────────────────────────────────────
def input_loop():
    print(f"\n  Type target height ({HEIGHT_MIN:.2f} – {HEIGHT_MAX:.2f} m) + Enter")
    print(f"  Presets: stand={HEIGHT_STAND:.2f}  crouch=0.14  full-extend=0.39")
    print(f"  Type 'q' to quit\n")
    while active[0]:
        try:
            raw = input("  height (m) > ").strip()
            if raw.lower() in ("q", "quit", "exit"):
                active[0] = False
                break
            h = float(raw)
            h = np.clip(h, HEIGHT_MIN, HEIGHT_MAX)
            hip_req = ik_hip_angle(h)
            state["src"]     = current_pz[0]
            state["target"]  = h
            state["t_start"] = sim_time_ref[0]
            print(f"  → target={h:.3f} m  |  hip_target={np.degrees(hip_req):.1f}°  "
                  f"({hip_req:.3f} rad)\n")
        except ValueError:
            print(f"  ! invalid — enter a number between {HEIGHT_MIN:.2f} and {HEIGHT_MAX:.2f}\n")
        except EOFError:
            break


# ── Load model ─────────────────────────────────────────────────
model = mujoco.MjModel.from_xml_path("monoped.xml")
data  = mujoco.MjData(model)
dt    = model.opt.timestep

print("=" * 52)
print("  Monoped — Height Control")
print("=" * 52)
print(f"  L_crank = L_rod = {L:.2f} m")
print(f"  pelvis_z = 2 × {L} × cos(hip)")
print(f"  Range  : {HEIGHT_MIN:.2f} m → {HEIGHT_MAX:.2f} m")
print(f"  STAND  : {HEIGHT_STAND:.2f} m  (hip = {np.degrees(ik_hip_angle(HEIGHT_STAND)):.1f}°)")
print(f"  CROUCH : 0.14 m   (hip = {np.degrees(ik_hip_angle(0.14)):.1f}°)")
print("=" * 52)

# qpos order: [0]=slide_z, [1]=hip, [2]=connect_hinge
data.qpos[1] = ik_hip_angle(HEIGHT_STAND)
mujoco.mj_forward(model, data)

# Sensor indices
S_HIP     = 3
S_CONN    = 4
S_SLIDE_Z = 5

pid = PID(Kp=9.0, Ki=0.05, Kd=0.80, dt=dt)

# Start input thread
t_input = threading.Thread(target=input_loop, daemon=True)
t_input.start()

# ── Simulation loop ────────────────────────────────────────────
print("\n  [angles printed every 0.3 s]\n")
last_print = -1.0
PRINT_DT   = 0.30

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth   = 90
    viewer.cam.elevation = -12
    viewer.cam.distance  = 1.6
    viewer.cam.lookat[:] = [0, 0, 0.25]

    SETTLE = 0.5
    while viewer.is_running() and active[0]:
        t = data.time

        hip_now = float(data.sensordata[S_HIP])
        current_pz[0]  = fk_pelvis_z(hip_now)
        sim_time_ref[0] = t

        if t < SETTLE:
            hip_tgt = ik_hip_angle(HEIGHT_STAND)
        else:
            hip_tgt = ik_hip_angle(commanded_height(t))

        data.ctrl[0] = pid.compute(hip_tgt, hip_now)

        mujoco.mj_step(model, data)
        viewer.sync()

        # ── Terminal angle print ───────────────────────────────
        if t - last_print >= PRINT_DT:
            last_print = t
            conn = float(data.sensordata[S_CONN])
            sz   = float(data.sensordata[S_SLIDE_Z])
            pz   = current_pz[0]
            err  = state["target"] - pz
            print(f"  t={t:6.2f}s  |  "
                  f"pelvis={pz:.3f}m (tgt={state['target']:.3f}m, err={err:+.3f}m)  |  "
                  f"hip={np.degrees(hip_now):+6.1f}°  "
                  f"conn={np.degrees(conn):+6.1f}°  "
                  f"slide_z={sz:+.4f}m")
