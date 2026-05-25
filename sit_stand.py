"""
Monoped — Height Control (Sit / Stand)   [Inverted Crank-Slider]
==================================================================
Set a target pelvis height; IK computes the required hip angle;
PID drives the single hip motor to that angle.

Mechanism geometry:
  L_crank = 0.25 m    L_rod = 0.45 m    foot fixed at z = 0.00 m

Forward kinematics (hip angle → pelvis Z):
  pelvis_z = L_crank * cos(hip) + sqrt(L_rod² − L_crank² * sin²(hip))
  → hip=0°:    pelvis_z = 0.70 m  (standing tall)
  → hip=±90°:  pelvis_z ≈ 0.37 m  (fully crouched)

Inverse kinematics (target pelvis Z → hip angle):
  Solved via bisection. fk_pelvis_z is monotonically decreasing on [0, π/2].

Height presets:
  STAND = 0.68 m  (hip ≈ 0.24 rad — nearly straight)
  SIT   = 0.42 m  (hip ≈ 1.20 rad — deeply crouched)
"""

import mujoco
import mujoco.viewer
import numpy as np
import threading
import tkinter as tk
import time

# ── Mechanism constants ────────────────────────────────────────
L_CRANK  = 0.20
L_ROD    = 0.20

HEIGHT_MIN   = 0.06   # practical floor limit (pelvis geom clears floor)
HEIGHT_MAX   = 0.40   # hip=0 (fully extended: 0.20 + 0.20)
HEIGHT_STAND = 0.36
HEIGHT_SIT   = 0.14
DURATION     = 1.8    # transition time (s)


def fk_pelvis_z(hip: float) -> float:
    """World Z of pelvis (inverted crank-slider, foot fixed at z=0)."""
    sin_h = np.sin(hip)
    d = max(L_ROD**2 - L_CRANK**2 * sin_h**2, 0.0)
    return L_CRANK * np.cos(hip) + np.sqrt(d)


def ik_hip_angle(target_z: float) -> float:
    """
    Bisection IK: returns hip angle in [0, π/2] for target pelvis Z.
    fk_pelvis_z decreases monotonically as hip goes from 0 → π/2.
    """
    target_z = np.clip(target_z, HEIGHT_MIN + 1e-4, HEIGHT_MAX - 1e-4)
    lo, hi   = 0.0, np.pi / 2
    for _ in range(40):
        mid = (lo + hi) / 2
        if fk_pelvis_z(mid) > target_z:
            lo = mid   # fk too high → need more hip angle
        else:
            hi = mid   # fk too low  → need less hip angle
    return (lo + hi) / 2


# ── Shared state ───────────────────────────────────────────────
state = {
    "target"  : HEIGHT_STAND,
    "src"     : HEIGHT_STAND,
    "t_start" : 0.0,
    "active"  : True,
}
sim_time_ref    = [0.0]
current_pelvis_z = [HEIGHT_STAND]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-10.0 * (x - 0.5)))


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

    def reset(self):
        self.integral = 0.0; self.prev_err = 0.0

    def compute(self, target, current):
        err           = target - current
        self.integral = np.clip(self.integral + err * self.dt, -2.0, 2.0)
        deriv         = (err - self.prev_err) / self.dt
        self.prev_err = err
        return np.clip(self.Kp * err + self.Ki * self.integral + self.Kd * deriv,
                       -self.limit, self.limit)


# ── GUI ───────────────────────────────────────────────────────
def run_gui():
    root = tk.Tk()
    root.title("Monoped — Height Control")
    root.geometry("380x480")
    root.resizable(False, False)
    root.configure(bg="#1e1e2e")

    def lbl(text, fg="#cdd6f4", size=10, pady=4):
        tk.Label(root, text=text, bg="#1e1e2e", fg=fg,
                 font=("Courier", size, "bold")).pack(pady=pady)

    lbl("MONOPED — HEIGHT CONTROL", "#89b4fa", 13, pady=14)
    lbl(f"Inverted crank-slider  |  range {HEIGHT_MIN:.2f}–{HEIGHT_MAX:.2f} m", "#6c7086", 8)

    target_var = tk.StringVar(value=f"Target :  {HEIGHT_STAND:.3f} m")
    actual_var = tk.StringVar(value="Actual :  ---    m")
    hip_var    = tk.StringVar(value="Hip    :  ---   rad")

    tk.Label(root, textvariable=target_var, bg="#1e1e2e", fg="#f9e2af",
             font=("Courier", 12, "bold")).pack(pady=(12, 2))
    tk.Label(root, textvariable=actual_var, bg="#1e1e2e", fg="#a6e3a1",
             font=("Courier", 11)).pack(pady=2)
    tk.Label(root, textvariable=hip_var,   bg="#1e1e2e", fg="#cba6f7",
             font=("Courier", 11)).pack(pady=2)

    slider_var = tk.DoubleVar(value=HEIGHT_STAND)

    def on_slider(val):
        h = float(val)
        state["src"]     = current_pelvis_z[0]
        state["target"]  = np.clip(h, HEIGHT_MIN, HEIGHT_MAX)
        state["t_start"] = sim_time_ref[0]
        target_var.set(f"Target :  {h:.3f} m")

    tk.Scale(root,
             from_=HEIGHT_MAX, to=HEIGHT_MIN,
             resolution=0.005,
             orient=tk.VERTICAL,
             length=200,
             variable=slider_var,
             bg="#313244", fg="#cdd6f4",
             troughcolor="#45475a",
             highlightthickness=0,
             label="Height (m)",
             font=("Courier", 9),
             command=on_slider).pack(pady=8)

    btn_frame = tk.Frame(root, bg="#1e1e2e")
    btn_frame.pack(pady=6)

    def preset(h):
        slider_var.set(h)
        on_slider(h)

    for text, h, bg in [
        ("STAND", HEIGHT_STAND, "#a6e3a1"),
        ("SIT",   HEIGHT_SIT,   "#89b4fa"),
    ]:
        tk.Button(btn_frame, text=text,
                  command=lambda v=h: preset(v),
                  bg=bg, fg="#1e1e2e",
                  font=("Courier", 12, "bold"),
                  relief=tk.FLAT, padx=14, pady=8, width=7
                  ).pack(side=tk.LEFT, padx=10)

    def refresh():
        if state["active"]:
            actual_var.set(f"Actual :  {current_pelvis_z[0]:.3f} m")
            hip_var.set(   f"Hip    :  {ik_hip_angle(current_pelvis_z[0]):.3f} rad")
            root.after(60, refresh)

    root.after(200, refresh)

    def on_close():
        state["active"] = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


# ── Load model ─────────────────────────────────────────────────
model = mujoco.MjModel.from_xml_path("monoped.xml")
data  = mujoco.MjData(model)
dt    = model.opt.timestep

print("=" * 55)
print("  Monoped — Height-Based Sit/Stand (Inverted)")
print("=" * 55)
print(f"  L_crank={L_CRANK}m  L_rod={L_ROD}m  foot at z=0")
print(f"  Height range: {HEIGHT_MIN:.2f}m (sit) → {HEIGHT_MAX:.2f}m (stand)")
print(f"  STAND hip ≈ {ik_hip_angle(HEIGHT_STAND):.3f} rad")
print(f"  SIT   hip ≈ {ik_hip_angle(HEIGHT_SIT):.3f} rad")
print("=" * 55 + "\n")

# qpos order: [0]=slide_z, [1]=hip, [2]=connect_hinge
# Set initial hip to STAND position
data.qpos[1] = ik_hip_angle(HEIGHT_STAND)
mujoco.mj_forward(model, data)

S_HIP     = 3
S_SLIDE_Z = 5

pid_hip = PID(Kp=9.0, Ki=0.05, Kd=0.80, dt=dt)

SETTLE = 0.8

gui_thread = threading.Thread(target=run_gui, daemon=True)
gui_thread.start()
time.sleep(0.3)

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth   = 90
    viewer.cam.elevation = -12
    viewer.cam.distance  = 2.0
    viewer.cam.lookat[:] = [0, 0, 0.4]

    step = 0
    while viewer.is_running() and state["active"]:
        t = data.time

        hip_now = float(data.sensordata[S_HIP])
        sz      = float(data.sensordata[S_SLIDE_Z])

        current_pelvis_z[0] = fk_pelvis_z(hip_now)
        sim_time_ref[0]     = t

        if t < SETTLE:
            hip_tgt = ik_hip_angle(HEIGHT_STAND)
        else:
            hip_tgt = ik_hip_angle(commanded_height(t))

        data.ctrl[0] = pid_hip.compute(hip_tgt, hip_now)

        mujoco.mj_step(model, data)
        viewer.sync()
        step += 1

        if step % 1000 == 0:
            print(f"t={t:5.2f}s | "
                  f"target={state['target']:.3f}m | "
                  f"actual={current_pelvis_z[0]:.3f}m | "
                  f"hip={hip_now:.4f}rad (tgt={hip_tgt:.4f}) | "
                  f"slide_z={sz:+.4f}m")
